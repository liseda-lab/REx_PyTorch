from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import random
import json
import time
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from code.model.baseline import ReactiveBaseline
from scipy.special import logsumexp as lse
import shutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 

from torch.utils.tensorboard import SummaryWriter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from code.model.environment import env, Episode, threshold_for_step # -- NEW 


def trim_and_rank_batch(entity_traj, relation_traj, log_probs, end_entities, batch_size, K):
    """
    Trim each rollout at its first hit, sum only prefix log‐probs, and
    return per‐example sorted lists of dicts {entities, relations, score}.
    """
    ent_arr = np.stack(entity_traj, axis=0)  # [T, B*K]
    rel_arr = np.stack(relation_traj, axis=0)
    trimmed = []
    T, _ = ent_arr.shape
    for b in range(batch_size):
        start, end = b*K, (b+1)*K
        ents_b = ent_arr[:, start:end]      # [T, K]
        rels_b = rel_arr[:, start:end]
        scores = log_probs[b]               # [K]
        target = end_entities[start]

        paths = []
        for r in range(K):
            hits = (ents_b[:, r] == target)
            first_hit = np.argmax(hits) if hits.any() else T-1
            paths.append({
                'entities':    ents_b[:first_hit+1, r].tolist(),
                'relations':   rels_b[:first_hit+1, r].tolist(),
                'score':       float(scores[r]),
                'rollout_idx': r
            })
        trimmed.append(sorted(paths, key=lambda x: x['score'], reverse=True))
    return trimmed

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def configure_logger(log_file_path):
    """
    Configure the logger to output to both the console and a log file.
    :param log_file_path: Path to the log file.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(__name__)  
    logger.setLevel(logging.INFO)

    fmt = '%(asctime)s: [ %(message)s ]'
    datefmt = '%m/%d/%Y %I:%M:%S %p'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, 'w')  
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    return logger



class Trainer(object):
    def __init__(self, params, tensorboard_dir):
        for key, val in params.items(): setattr(self, key, val); 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.agent = Agent(params) 
        self.set_random_seed(self.seed) # set random seed for reproducibility
        self.save_path = None
        self.train_environment = env(params, 'train') # train environment
        self.dev_test_environment = env(params, 'dev') # dev environment
        self.test_test_environment = env(params, 'test') # test environment
        self.test_environment = self.dev_test_environment # default test environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0 # Track max hits at 10
        
        # NEW Add early stopping variables 
        self.best_metric = -1  # Track best MRR
        self.early_stopping = False
        self.waiting_period = 3  # Stop if no improvement for 3 evaluations
        self.current_waiting_period = self.waiting_period
        
        
        self.ePAD = self.entity_vocab['PAD'] # entity padding index
        self.rPAD = self.relation_vocab['PAD'] # relation padding index

        # Initialize baseline for reward adjustment and optimizer
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate) # //NOTE: before we didnt pass parameters to the optimizer

        self.global_step_tensor = 0  # Track global step for learning rate decay

        self.tensorboard_dir = tensorboard_dir

        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def set_random_seed(self, seed):
        """
        Set random seeds for reproducibility.
        :param seed: The seed value to use. If None, no seed is set.
        """
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)  # Fix hash-based randomness
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)  # Python's random seed
            logger.info(f"Random seeds set to {seed}")


    def _keep_correct_entry(self, groups, episode, b, r, p, rewards):
        """
        If rollout (b,r) is correct, append a compact dict into groups[start][end].
        Stores only path-level IC (mean), no per-step IC.
        """
        indx = b * self.test_rollouts + r
        if rewards[indx] <= 0:
            return  # keep only correct paths

        start_name = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
        end_name   = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]

        # Path-level IC (mean) – computed in Episode._cache_ic_summaries()
        ic_mean = float(episode.ic_mean[indx]) if hasattr(episode, "ic_mean") else None

        agentic_score = float(episode.agentic_scores[indx]) if episode.agentic_scores is not None else 0.0
        entry = {
            "beam_logprob": float(self.log_probs[b, r]),
            "agentic_score": agentic_score,
            "reward_kind": (episode.reward_kind[indx] if hasattr(episode, "reward_kind") else "n/a"),
            "path": {
                "entities":  [self.rev_entity_vocab[e]   for e  in p['entities']],
                "relations": [self.rev_relation_vocab[rr] for rr in p['relations']],
            },
            "ic_mean": ic_mean,
        }

        if indx in episode.llm_dimensions:
            dims = episode.llm_dimensions[indx]
            entry["llm_dimensions"] = {
                "validity": float(dims["validity"]),
                "completeness_conv": float(dims["completeness_conv"]),
                "relevance": float(dims["relevance"]),
            }

        groups.setdefault(start_name, {}).setdefault(end_name, []).append(entry)

    def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
        """
        Compute the REINFORCE loss with baseline and entropy regularization
        """

        # Stack losses across all time steps
        loss = torch.stack(per_example_loss, axis=1) # [B, T]

        # Compute the baseline value for reward adjustment
        baseline = self.baseline.get_baseline_value()
        baseline = torch.tensor(baseline, dtype=torch.float32, device=self.device)

        # Compute reward difference from baseline and normalize
        final_reward = cum_discounted_reward - baseline
        reward_mean = torch.mean(final_reward) # NOTE: use just python functions?
        #reward_mean = final_reward.mean()
        #reward_std = final_reward.std() + 1e-6 # stability adjustment
        # Constant added for numerical stability
        reward_std = torch.std(final_reward) + 1e-6 # stability adjustment
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        # Adjust loss using the normalized reward
        #loss = torch.mul(loss, final_reward)  # [B, T]
        loss = loss * final_reward  # [B, T]
        self.loss_before_reg = loss # NOTE: check if this is necessary

        decaying_beta = self.beta * (0.90 ** (self.global_step_tensor / 200))

        # Add entropy regularization to encourage exploration
        #total_loss = torch.mean(loss) - decaying_beta * self.entropy_reg_loss(per_example_logits)
        total_loss = loss.mean() - decaying_beta * self.entropy_reg_loss(per_example_logits)

        return total_loss

    def entropy_reg_loss(self, all_logits):
        """
        Compute the entropy regularization loss to encourage exploration
        """
        all_logits = torch.stack(all_logits, axis=2)  
        entropy_policy = - torch.mean(
            torch.sum(torch.exp(all_logits) * all_logits, dim=1)
        )  # Negative entropy
        return entropy_policy

    # def initialize(self, restore=None, sess=None):
    #     """
    #     Initialize the TensorFlow computation graph and initializes variables. 
    #     """

    #     logger.info("Creating TF graph...")

    #     # Create placeholders for inputs at each time step
    #     self.candidate_relation_sequence = []
    #     self.candidate_entity_sequence = []
    #     self.next_weights_sequence = [] # Placeholder for edge weights
    #     self.input_path = []
    #     self.first_state_of_test = tf.placeholder(tf.bool, name="is_first_state_of_test")
    #     self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
    #     self.range_arr = tf.placeholder(tf.int32, shape=[None, ])
    #     self.global_step = tf.Variable(0, trainable=False)
    #     self.decaying_beta = tf.train.exponential_decay(self.beta, self.global_step,
    #                                                200, 0.90, staircase=False)
    #     self.entity_sequence = []
    #     self.cum_discounted_reward = tf.placeholder(tf.float32, [None, self.path_length],
    #                                                 name="cumulative_discounted_reward")

    #     # Placeholder definitions for each step in the path
    #     for t in range(self.path_length):
    #         next_possible_relations = tf.placeholder(tf.int32, [None, self.max_num_actions],
    #                                                name="next_relations_{}".format(t))
    #         next_possible_entities = tf.placeholder(tf.int32, [None, self.max_num_actions],
    #                                                  name="next_entities_{}".format(t))

    #         next_weights = tf.placeholder(tf.float32, [None, self.max_num_actions], # Placeholder for edge weights
    #                                                     name="next_weights_{}".format(t))

    #         input_label_relation = tf.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
    #         start_entities = tf.placeholder(tf.int32, [None, ])
    #         self.input_path.append(input_label_relation)
    #         self.candidate_relation_sequence.append(next_possible_relations)
    #         self.candidate_entity_sequence.append(next_possible_entities)
    #         self.entity_sequence.append(start_entities)
    #         self.next_weights_sequence.append(next_weights) # Append edge weights
    #         self.loss_before_reg = tf.constant(0.0)


    #     # Compute losses and logits for all steps
    #     self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
    #         self.candidate_relation_sequence,
    #         self.candidate_entity_sequence, 
    #         self.entity_sequence,
    #         self.next_weights_sequence, # Add edge weights
    #         self.input_path,
    #         self.query_relation, 
    #         self.range_arr,
    #         self.first_state_of_test, 
    #         self.path_length
    #         )


    #     # Compute the REINFORCE loss
    #     self.loss_op = self.calc_reinforce_loss()

    #     # Define brackpropagation operation
    #     self.train_op = self.bp(self.loss_op)

    #     # Build the test graph
    #     self.prev_state = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
    #     self.prev_relation = tf.placeholder(tf.int32, [None, ], name="previous_relation")
    #     self.query_embedding = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.query_relation)  # [B, 2D]
    #     layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
    #     formated_state = [tf.unstack(s, 2) for s in layer_state]
    #     self.next_relations = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
    #     self.next_entities = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])

    #     self.current_entities = tf.placeholder(tf.int32, shape=[None,])

    #     with tf.variable_scope("policy_steps_unroll") as scope:
    #         scope.reuse_variables()
    #         self.test_loss, test_state, self.test_logits, self.test_action_idx, self.chosen_relation = self.agent.step(
    #             self.next_relations, self.next_entities, self.next_weights_sequence[0], formated_state, self.prev_relation, self.query_embedding,
    #             self.current_entities, self.input_path[0], self.range_arr, self.first_state_of_test)
    #         self.test_state = tf.stack(test_state)

    #     logger.info('TF Graph creation done..')
    #     self.model_saver = tf.train.Saver(max_to_keep=2)

    #     # Return the variable initializer or restore the model
    #     if not restore:
    #         return tf.global_variables_initializer()
    #     else:
    #         return  self.model_saver.restore(sess, restore)

    # def initialize_pretrained_embeddings(self, sess):
    #     """
    #     Initialize the agent's embeddings with pretrained embeddings
    #     """
    #     if self.pretrained_embeddings_action != '':
    #         embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
    #         _ = sess.run((self.agent.relation_embedding_init),
    #                      feed_dict={self.agent.action_embedding_placeholder: embeddings})
    #     if self.pretrained_embeddings_entity != '':
    #         embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
    #         _ = sess.run((self.agent.entity_embedding_init),
    #                      feed_dict={self.agent.entity_embedding_placeholder: embeddings})


    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])  # [B]
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  # [B, T]
        cum_disc_reward[:, self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward
    
    def train(self):
        """
        Trains the agent using Reinforce on the training environment
        """
        print('ENTERING TRAINING LOOP')
        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0
        cumulative_reward = 0

        # Set our agent to training mode, as it extends nn.Module
        self.agent.train()

        # Iterate through episodes from the training environment
        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            Episode.set_training_step(self.batch_counter)
            self.global_step_tensor = self.batch_counter

            query_relation = torch.tensor(episode.get_query_relation(), 
                                        dtype=torch.long, device=self.device)
            state = episode.get_state()

            range_arr = torch.arange(self.batch_size * self.num_rollouts, 
                                    dtype=torch.long, device=self.device)

            prev_relation = torch.full((self.batch_size * self.num_rollouts,), fill_value=self.relation_vocab['DUMMY_START_RELATION'], dtype=torch.long, device=self.device)

            agent_mem = None

            per_example_loss = []
            per_example_logits = []
            action_idx_list = []
            
            for i in range(self.path_length):
                print(f"Processing step {i}/{self.path_length} of episode {self.batch_counter}")
                next_relations = torch.tensor(state['next_relations'], 
                                            dtype=torch.long, device=self.device)
                next_entities = torch.tensor(state['next_entities'], 
                                            dtype=torch.long, device=self.device)
                current_entities = torch.tensor(state['current_entities'], 
                                               dtype=torch.long, device=self.device)
                next_weights = torch.tensor(state['weights'], 
                                           dtype=torch.float32, device=self.device)
                
                # Placebo
                label_action = torch.zeros(self.batch_size * self.num_rollouts,
                                          dtype=torch.long, device=self.device)
                
                loss, agent_mem, logits, action_idx, chosen_relation = self.agent(
                    next_relations,
                    next_entities,
                    current_entities,
                    next_weights,
                    label_action,
                    query_relation,
                    range_arr,
                    first_step_of_test=False,
                    prev_relation=prev_relation,
                    prev_state=agent_mem
                )

                per_example_loss.append(loss)
                per_example_logits.append(logits)
                action_idx_list.append(action_idx)

                prev_relation = chosen_relation

                # Update the state based on chosen actions
                idx = action_idx.cpu().numpy()
                state = episode(idx)
            
            # Get rewards and compute cumulative discounted reward
            rewards = episode.get_reward_agenticAI()
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            cum_discounted_reward = torch.tensor(cum_discounted_reward, 
                                                dtype=torch.float32, device=self.device)
            
            # Perform backpropagation
            self.optimizer.zero_grad()
            total_loss = self.calc_reinforce_loss(per_example_loss, per_example_logits, 
                                                 cum_discounted_reward)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            
            self.baseline.update(cum_discounted_reward.mean().item())
            
            batch_total_loss = total_loss.item()
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss

            # Log statistics
            avg_reward = np.mean(rewards)
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))
            reward_reshape = np.sum(reward_reshape, axis=1)
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            cumulative_reward += np.sum(rewards)
            mean_total_reward = cumulative_reward / self.batch_counter
            avg_positive_reward = np.mean(rewards[rewards > 0]) if np.any(rewards > 0) else 0.0
            
            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                       "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                       format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                             (num_ep_correct / self.batch_size), train_loss))

            self.summary_writer.add_scalar('avg_reward_per_batch', avg_reward, self.batch_counter)
            self.summary_writer.add_scalar('avg_positive_reward', avg_positive_reward, self.batch_counter)
            self.summary_writer.add_scalar('mean_total_reward', mean_total_reward, self.batch_counter)
            self.summary_writer.add_scalar('train_loss', train_loss, self.batch_counter)
            self.summary_writer.add_scalar('avg_ep_correct', (num_ep_correct / self.batch_size), 
                                          self.batch_counter)
            
            # Evaluate model periodically (ONLY ONCE!)
            if self.batch_counter % self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                
                os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"
                
                current_mrr = self.test(beam=True, print_paths=False)
            
            # Check early stopping
            if self.early_stopping:
                logger.info(f"[TRAINING STOPPED] Early stopping triggered at iteration {self.batch_counter}")
                break
            
            # Check if reached max iterations
            if self.batch_counter >= self.total_iterations:
                logger.info(f"[TRAINING COMPLETE] Reached maximum iterations: {self.total_iterations}")
                break
            
            gc.collect()
        
        # Close summary writer AFTER the loop ends
        self.summary_writer.close()
    
    
    def test(self, beam = True, print_paths = True, save_model = True, mrr = True):
        """
        Tests the trained agent on the test environment.

        Args:
            sess: The current TensorFlow session.
            beam: Whether to use beam search for test evaluation.
            print_paths: Whether to log the paths taken by the agent.
            save_model: Whether to save the model if it achieves the best performance.
            mrr: Whether to compute the Mean Reciprocal Rank (MRR).

        Returns:
            None. Logs results and saves paths/rewards.
        """
        batch_counter = 0
        paths = defaultdict(list) # store paths taken by the agent
        answers = [] # store answers for analysis
        #feed_dict = {}
        final_rewards = {
        "Hits@1": 0,
        "Hits@3": 0,
        "Hits@5": 0,
        "Hits@10": 0,
        "Hits@20": 0,
        "MRR": 0  
        }

        correct_groups = {}

        # Set our agent to evaluation mode
        self.agent.eval()


        print("ENTERING TEST LOOP")

        # Total number of examples to evaluate
        total_examples = self.test_environment.total_no_examples

        # Iterate through episodes from the test environment
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1
            temp_batch_size = episode.no_examples

            # Initialize query relation and state
            self.qr = episode.get_query_relation()
            #self.qr = torch.tensor(episode.get_query_relation(), dtype=torch.long, device=self.device)
            qr_emb = self.agent.relation_lookup_table(torch.tensor(episode.get_query_relation(), dtype=torch.long, device=self.device))  # [B, 2D]
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1)) # initial beam probs
            state = episode.get_state() # initial state

            # Initialize agent memory and previous relation
            #mem = self.agent.get_mem_shape()
            #agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            agent_mem = self.agent.zero_state(temp_batch_size * self.test_rollouts)
            #previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
            #    'DUMMY_START_RELATION']
            previous_relation = torch.full((temp_batch_size * self.test_rollouts,), 
                                         fill_value=self.relation_vocab['DUMMY_START_RELATION'], 
                                         dtype=torch.long, device=self.device)
            range_arr = torch.arange(temp_batch_size * self.test_rollouts, dtype=torch.long, device=self.device)

            # Initialize tracking of visited entities
            visited_entities = np.zeros((temp_batch_size * self.test_rollouts, 1), dtype=np.int32)
            visited_entities[:, 0] = state['current_entities']  # begin with current entities

            if print_paths:
                self.entity_trajectory = [] # track entities visited
                self.relation_trajectory = [] # track relations traversed

            # Initialize log probabilities
            #self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0
            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,), dtype=np.float32)


            # Process each time step in the path
            for i in range(self.path_length):
                if i == 0: # first state
                    first_state_of_test = True

                next_relations = torch.tensor(state['next_relations'], dtype=torch.long, device=self.device)
                next_entities = torch.tensor(state['next_entities'], dtype=torch.long, device=self.device)
                current_entities = torch.tensor(state['current_entities'], dtype=torch.long, device=self.device)
                next_weights = torch.tensor(state['weights'], dtype=torch.float32, device=self.device)
                prev_state = torch.tensor(agent_mem, dtype=torch.float32, device=self.device) if isinstance(agent_mem, np.ndarray) else agent_mem
                prev_relation = previous_relation
                label_action = torch.zeros(temp_batch_size * self.test_rollouts, dtype=torch.long, device=self.device)

                # Run the agent step
                loss, agent_mem, test_scores, test_action_idx, chosen_relation = self.agent.step(
                    next_relations, next_entities, next_weights, prev_state, prev_relation, qr_emb, current_entities,
                    label_action, range_arr, first_state_of_test)
                
                test_scores = test_scores.detach().cpu().numpy()

                # Perform beam search
                # if active will prioritize actions with higher acummulated scores
                if beam:
                    k = self.test_rollouts # Number of beams
                    new_scores = test_scores + beam_probs # Update scores with beam probabilities
                    
                    if i == 0: # first step
                        idx = np.argsort(new_scores) 
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k) # Get top k indices

                    # Update beam information
                    y = idx // self.max_num_actions
                    x = idx % self.max_num_actions

                    # SHIFT the environment’s arrays to maintain correct alignment
                    episode.visited_entities = episode.visited_entities[y, :]
                    episode.done_mask        = episode.done_mask[y]
                    episode.current_entities = episode.current_entities[y]

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]
                    #agent_mem = agent_mem[:, :, y, :]
                    # TODO: check if this is correct
                    h, c = agent_mem
                    h = h[:, y, :].contiguous()
                    c = c[:, y, :].contiguous()
                    agent_mem = (h, c)
                    
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    chosen_relation = torch.tensor(chosen_relation, dtype=torch.long, device=self.device)
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))

                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]

                previous_relation = chosen_relation

                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)

                # Update the state with the chosen actions
                state = episode(test_action_idx)
                step_scores = test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
                step_scores[episode.done_mask] = 0.0
                self.log_probs += step_scores
            if beam:
                self.log_probs = beam_probs

            if print_paths:
                self.entity_trajectory.append(
                    state['current_entities'])

            # Process final rewards from environment
           # rewards = episode.get_reward_weights_sigmoid()  
            rewards = episode.get_reward_agenticAI() # -- NEW 
            #print(rewards)
            reward_reshape = rewards.reshape((temp_batch_size, self.test_rollouts))
            # reshape and sort on the *frozen* full‐log_probs
            self.log_probs = self.log_probs.reshape((temp_batch_size, self.test_rollouts))
            sorted_indx  = np.argsort(-self.log_probs, axis=1)
            AP = 0.0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))

            # Compute HITS@k
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        #if reward_reshape[b,r] == self.positive_reward:
                        if reward_reshape[b,r] > 0: # positive reward
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        #if reward_reshape[b,r] == self.positive_reward:
                        if reward_reshape[b,r] > 0: # positive reward
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None

                # HITS@k update
                if answer_pos is not None:
                    if answer_pos < 20:
                        final_rewards["Hits@20"] += 1
                        if answer_pos < 10:
                            final_rewards["Hits@10"] += 1
                            if answer_pos < 5:
                                final_rewards["Hits@5"] += 1
                                if answer_pos < 3:
                                    final_rewards["Hits@3"] += 1
                                    if answer_pos < 1:
                                        final_rewards["Hits@1"] += 1
              
                    AP += 1.0/((answer_pos+1))

                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    trimmed = trim_and_rank_batch(
                    entity_traj   = self.entity_trajectory,
                    relation_traj = self.relation_trajectory,
                    log_probs     = self.log_probs,
                    end_entities  = episode.end_entities,
                    batch_size    = temp_batch_size,
                    K              = self.test_rollouts
                )
                    # for p in trimmed[b]:
                    #     # use the rollout index to recover reward & beam-score
                    #     r = p['rollout_idx']
                    #     indx = b * self.test_rollouts + r

                    #     # Get the agentic score (will be 0 if path failed)
                    #     agentic_score = episode.agentic_scores[indx] if episode.agentic_scores is not None else 0.0
                    #     score_line = f"{agentic_score:.4f}"
                       
                    #    # Binary reward indicator (1 if reached target with positive reward, -1 otherwise)
                    #     rev = 1 if rewards[indx] > 0 else -1

                    #     rk = episode.reward_kind[indx] if hasattr(episode, "reward_kind") else "n/a"

                    #     # Build the score line
                    #     score_line = f"{agentic_score:.4f} ({rk})"

                    #     # If this path was scored by LLM, add dimensions
                    #     if indx in episode.llm_dimensions:
                    #         dims = episode.llm_dimensions[indx]
                    #         score_line += f" v:{dims['validity']:.0f} c_conv:{dims['completeness_conv']:.0f} r:{dims['relevance']:.0f}"
                        
                    #     # elif hasattr(episode, 'reward_kind'):
                    #     #     score_line += f" ({episode.reward_kind[indx]})"

                    #     # OPTIONAL: still record args for answers[]
                    #     answers.append(
                    #         f"{self.rev_entity_vocab[se[b,r]]}\t"
                    #         f"{self.rev_entity_vocab[ce[b,r]]}\t"
                    #         f"{self.log_probs[b,r]:.4f}\n"
                    #     )
                    #     # now print the *trimmed* path
                    #     paths[str(qr)].append(
                    #         '\t'.join(self.rev_entity_vocab[e] for e in p['entities']) + '\n' +
                    #         '\t'.join(self.rev_relation_vocab[r] for r in p['relations']) + '\n' +
                    #         #f"{rev}\n{p['score']:.4f}\n___\n"
                    #         f"{rev}\n"
                    #         f"{p['score']:.4f}\n"  # beam score (log prob)
                    #         f"{score_line}\n___\n"  # agentic score + optional dimensions
                    #      )

                    #     self._keep_correct_entry(correct_groups, episode, b, r, p, rewards)

                                            
                    # paths[str(qr)].append("#####################\n")
                    for p in trimmed[b]:
                        # use the rollout index to recover reward & beam-score
                        r = p['rollout_idx']
                        indx = b * self.test_rollouts + r

                        # Get the agentic score (will be 0 if path failed)
                        agentic_score = episode.agentic_scores[indx] if episode.agentic_scores is not None else 0.0

                        # Binary reward indicator (1 if reached target with positive reward, -1 otherwise)
                        rev = 1 if rewards[indx] > 0 else -1

                        rk = episode.reward_kind[indx] if hasattr(episode, "reward_kind") else "n/a"

                        # Average IC for this rollout (computed in get_reward_agenticAI via _cache_ic_summaries)
                        ic_avg = None
                        if hasattr(episode, "ic_mean"):
                            try:
                                ic_avg = float(episode.ic_mean[indx])
                            except Exception:
                                ic_avg = None

                        # Build the score line
                        score_line = f"{agentic_score:.4f} ({rk})"
                        # Add IC average for all cases except "(none)"
                        if rk and rk.strip().lower() != "none" and ic_avg is not None:
                            score_line += f" ic_avg:{ic_avg:.3f}"

                        # If this path was scored by LLM, add dimensions
                        if indx in episode.llm_dimensions:
                            dims = episode.llm_dimensions[indx]
                            score_line += (
                                f" v:{dims['validity']:.0f}"
                                f" c_conv:{dims['completeness_conv']:.0f}"
                                f" r:{dims['relevance']:.0f}"
                            )

                        # OPTIONAL: still record args for answers[]
                        answers.append(
                            f"{self.rev_entity_vocab[se[b,r]]}\t"
                            f"{self.rev_entity_vocab[ce[b,r]]}\t"
                            f"{self.log_probs[b,r]:.4f}\n"
                        )

                        # now print the *trimmed* path
                        paths[str(qr)].append(
                            '\t'.join(self.rev_entity_vocab[e] for e in p['entities']) + '\n' +
                            '\t'.join(self.rev_relation_vocab[r] for r in p['relations']) + '\n' +
                            f"{rev}\n"
                            f"{p['score']:.4f}\n"  # beam score (log prob)
                            f"{score_line}\n___\n"  # agentic score + (kind) + ic_avg + optional dimensions
                        )

                        self._keep_correct_entry(correct_groups, episode, b, r, p, rewards)

                    paths[str(qr)].append("#####################\n")


                    

            final_rewards["MRR"] += AP

        # Normalize the final rewards
        for key in final_rewards:
            final_rewards[key] /= total_examples

        # # Save the model if it achieves the best performance
        # if save_model and final_rewards["Hits@10"] >= self.max_hits_at_10:
        #     self.max_hits_at_10 = final_rewards["Hits@10"]
        #     self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        #NEW WITH EARLY STOP EAVALUATION
        if save_model:
                current_metric = final_rewards["MRR"]  # Use MRR as the metric
                
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.current_waiting_period = self.waiting_period  # Reset WAITING 
                    #self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')
                    # FIXME: we aren't yet saving models, or loading them later on
                    self.save_path = self.model_dir
                    self.model_path = os.path.join(self.model_dir, "model.ckpt")
                    torch.save(self.agent.state_dict(), self.model_path)
                    self.max_hits_at_10 = final_rewards["Hits@10"]  # Keep existing logic
                    logger.info(f"[IMPROVE] New best MRR: {current_metric:.4f} at iteration {self.batch_counter}")
                    
                    # >>> NEW: write sidecar next to the checkpoint <<<
                    try:
                        meta = {
                            "best_step": int(self.batch_counter),
                            "threshold": float(threshold_for_step(self.batch_counter)),
                            "timestamp": time.time(),
                            "model_path": self.model_dir + "model" + ".ckpt",
                        }
                        sidecar_path = os.path.join(self.model_dir, "best_ckpt.json")
                        with open(sidecar_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                        logger.info(f"[META] Wrote {sidecar_path}: step={meta['best_step']} threshold={meta['threshold']:.2f}")
                    except Exception as e:
                        logger.warning(f"[META] Could not write best_ckpt.json: {e}")
               
                else:
                    self.current_waiting_period -= 1
                    logger.info(f"[NO IMPROVE] MRR: {current_metric:.4f}, Waiting Period: {self.current_waiting_period}/{self.waiting_period}")
                    
                    if self.current_waiting_period == 0:
                        self.early_stopping = True
                        logger.info(f"[EARLY STOP] Best MRR was {self.best_metric:.4f}")
        # Log paths and answers
        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        self.write_results_to_file(final_rewards)
        self.log_results(final_rewards)

        # Sort by agentic score (desc) and write JSON
        # Grouped JSON output: sort each [start][end] list by agentic_score desc
        if correct_groups:
            for s, ends in correct_groups.items():
                for e, lst in ends.items():
                    lst.sort(key=lambda x: x["agentic_score"], reverse=True)

            json_path = self.path_logger_file_ + "_correct_grouped.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(correct_groups, f, ensure_ascii=False, indent=2)
            logger.info(f"[SAVED] Correct grouped JSON → {json_path}")


        return final_rewards["MRR"]

    def write_results_to_file(self, final_rewards):
        """
        Write the results to a file
        """
        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(final_rewards["Hits@1"]))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(final_rewards["Hits@3"]))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(final_rewards["Hits@5"]))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(final_rewards["Hits@10"]))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(final_rewards["Hits@20"]))
            score_file.write("\n")
            score_file.write("MRR: {0:7.4f}".format(final_rewards["MRR"]))
            score_file.write("\n")
            score_file.write("\n")


    def log_results(self, final_rewards):
        """
        Log the results
        """
        logger.info("Hits@1: {0:7.4f}".format(final_rewards["Hits@1"]))
        logger.info("Hits@3: {0:7.4f}".format(final_rewards["Hits@3"]))
        logger.info("Hits@5: {0:7.4f}".format(final_rewards["Hits@5"]))
        logger.info("Hits@10: {0:7.4f}".format(final_rewards["Hits@10"]))
        logger.info("Hits@20: {0:7.4f}".format(final_rewards["Hits@20"]))
        logger.info("MRR: {0:7.4f}".format(final_rewards["MRR"]))


    def top_k(self, scores, k):
        """
        Get the top k indices
        """
        scores = scores.reshape(-1, k * self.max_num_actions) 
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:] # top k indices
        return idx.reshape((-1))

if __name__ == '__main__':

    options = read_options()
    # Default env log file if not provided via CLI/opts
    options.setdefault(
        'env_log_file',
        os.path.join(options['output_dir'], 'env_metrics.log')
    )

    # Configure the logger
    logger = configure_logger(options['log_file_name'])

    # read the vocab files
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    options['edges_weight'] = json.load(open(options['data_input_dir'] + 'clustered_IC_classes_edgeType.json' ))
    #options['edges_weight'] = json.load(open(options['data_input_dir'] + 'clustered_IC_classes.json' ))
    #options['edges_weight'] = json.load(open(options['data_input_dir'] + 'icNodeDegree_classes.json' ))
    ######## 
  
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    save_path = ''

    # Code to allow for loading of existing model, uncomment this and comment the next two code blocks titled #TRAINING and #TESTING
    #trainer = Trainer(options, tensorboard_dir=options['tensorboard_dir'])
    #if not options['load_model']:
        #TRAINING
        #trainer = Trainer(options, tensorboard_dir=options['tensorboard_dir'])
        #trainer.train()
        #save_path = trainer.save_path
        #path_logger_file = trainer.path_logger_file
        #output_dir = trainer.output_dir

    #else:
        #logger.info("Skipping training")
        #logger.info("Loading model from {}".format(options["model_load_dir"]))
        #save_path = options["model_load_dir"]
        #path_logger_file = trainer.path_logger_file
        #output_dir = trainer.output_dir
    
    #TRAINING
    trainer = Trainer(options, tensorboard_dir=options['tensorboard_dir'])
    trainer.train()
    save_path = trainer.save_path
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir
    
    #TESTING
    trainer.test_rollouts = 100

    os.mkdir(path_logger_file + "/" + "test_beam")
    trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
    with open(output_dir + '/scores.txt', 'a') as score_file:
        score_file.write("Test (beam) scores with best model from " + save_path + "\n")
    trainer.test_environment = trainer.test_test_environment
    trainer.test_environment.test_rollouts = 100

    ##-- NEW 
    # Infer directory and read sidecar
    ckpt_dir = os.path.dirname(save_path)
    sidecar = os.path.join(ckpt_dir, "best_ckpt.json")

    best_step = 0
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                meta = json.load(f)
            best_step = int(meta.get("best_step", 0))
            model = meta.get("model_path")
            th = float(meta.get("threshold", threshold_for_step(best_step)))
            logger.info(f"[TEST] Using best_step={best_step} → threshold={th:.2f} (from {sidecar})")
        except Exception as e:
            logger.warning(f"[TEST] Failed to read {sidecar}: {e}; defaulting step=0, threshold={threshold_for_step(0):.2f}")
    else:
        logger.warning(f"[TEST] {sidecar} not found; defaulting step=0, threshold={threshold_for_step(0):.2f}")

    # Load best model weights and parameters
    trainer.agent.load_state_dict(torch.load(model, map_location=trainer.device))

    # Make the threshold logic in Episode use this step
    Episode.set_training_step(best_step)

    trainer.test(beam = True, print_paths = True, save_model = False)

    # Erase empty folders in path_logger_file
    if os.path.isdir(path_logger_file):
        for subfolder in os.listdir(path_logger_file):
            subfolder_path = os.path.join(path_logger_file, subfolder)
            if os.path.isdir(subfolder_path) and not os.listdir(subfolder_path):
                shutil.rmtree(subfolder_path)  # Remove empty folder if it exists

