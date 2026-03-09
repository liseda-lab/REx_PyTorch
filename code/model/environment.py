from __future__ import absolute_import
from __future__ import division
import numpy as np
from collections import defaultdict
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import torch


import logging
logger = logging.getLogger()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#-----NEW 
import json 
from dotenv import load_dotenv
load_dotenv()  # ADD THIS
# Optional OpenAI client (kept safe if lib/key are missing)
"""try:
    from openai import OpenAI  
    _agentic_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY" 
    ) 
except Exception as _e:
    _agentic_client = None
    logger = logging.getLogger(__name__)
    logger.warning("OpenAI client not available: %s", _e)"""
#--------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    _llm_model = None
    _llm_tokenizer = None
    _llm_device = None
    
    def _init_llm():
        global _llm_model, _llm_tokenizer, _llm_device
        if _llm_model is not None:
            return
        
        MODEL_NAME = "Qwen/Qwen3.5-9B"
        print(f"[LLM] Loading {MODEL_NAME}...")
        
        _llm_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        _llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto"
            )
        
        print(f"[LLM] Ready on {_llm_device}")
    
    def _call_llm(messages, temperature=0):
        global _llm_model, _llm_tokenizer, _llm_device
        
        if _llm_model is None:
            _init_llm()
        
        try:
            prompt = _llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = _llm_tokenizer(prompt, return_tensors="pt").to(_llm_device)
            
            with torch.no_grad():
                outputs = _llm_model.generate(
                    **inputs,
                    max_new_tokens=3072,
                    temperature=temperature,
                    do_sample=True, # Check if these parameters are necessary
                    top_p=0.9,
                    pad_token_id=_llm_tokenizer.pad_token_id,
                )
            
            #response = _llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            #raw = response[len(prompt):].strip()

            #if "<think>" in raw:
            #    print("[LLM] Detected <think> in response; using content after it as final output.")
            #    raw = raw.split("<think>")[-1].strip()

            #return raw
            #return response

            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = _llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Keeping raw response for later debugging if no JSON is found
            responseSplit = response
            
            if "<think>" in response and "</think>" in response:
                responseSplit = response.split("</think>", 1)[1].strip()
            
            json_start = responseSplit.find("[")
            json_end = responseSplit.rfind("]")
            
            if json_start != -1 and json_end > json_start:
                return responseSplit[json_start:json_end+1], response
            
            return response, response
            
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""
    
except Exception as _e:
    _call_llm = None
    logger.warning("LLM not available: %s", _e)


# --- dedicated env logger (separate file, no console) ---
_env_logger = logging.getLogger("agentic.env")
_env_logger.propagate = False  # don't bubble up into root

def configure_env_logger(path):
    """Attach a file handler once; safe to call multiple times."""
    if not path:
        return
    if not _env_logger.handlers:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
        fh.setFormatter(fmt)
        _env_logger.addHandler(fh)
        _env_logger.setLevel(logging.INFO)
# --------------------------------------------------------


# Completeness is best at 3; triangular mapping back to a 1–5 scale
COMPLETENESS_MAP = {1: 1.0, 2: 3.0, 3: 5.0, 4: 3.0, 5: 1.0}

def threshold_for_step(step: int) -> float:
    if step < 60:   return 0.50
    elif step < 80: return 0.55
    elif step < 120:return 0.60
    elif step < 150:return 0.65
    else:           return 0.70


class Episode(object):
    _training_step = 0  # Current training step/iteration

    @classmethod
    def set_training_step(cls, step):
        cls._training_step = step

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher, weighted_reward, adjust_factor, sigmoid, size_flexibility, prevent_cycles, persona_path, agentic_ai_enabled  = params  # NEW - ADITION OF PERSONA PATH
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_answers, batch_weights = data 
        self.no_examples = start_entities.shape[0]
        self.batch_weights = batch_weights
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.weighted_reward = weighted_reward
        self.sigmoid = sigmoid
        self.size_flexibility = size_flexibility
        self.prevent_cycles = prevent_cycles
        self.adjust_factor = adjust_factor
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        # Load persona text (empty string if file missing) #  -- NEW 
        self.agentic_ai_enabled = bool(agentic_ai_enabled)
        self.persona_path = persona_path
        self.persona_text = ""
        if self.agentic_ai_enabled and self.persona_path and os.path.isfile(self.persona_path):
            with open(self.persona_path, "r", encoding="utf-8") as f:
                self.persona_text = f.read().strip()

        self.agentic_scores = None  # Final agentic scores for all rollouts
        self.llm_dimensions = {}    # Store v, c_conv, r for LLM-scored paths only

        #---

        # CREATE A DONE MASK (ONE ENTRY PER ROLLOUT IN THE BATCH)
        self.done_mask = np.zeros(self.no_examples * self.num_rollouts, dtype=bool)


        # Initialize visited entities tracking - each rollout will have its own list
        # Initially, just the starting entities
        self.visited_entities = np.zeros((self.no_examples * self.num_rollouts, 1), dtype=np.int32)
        self.visited_entities[:, 0] = self.start_entities

        self.weight_history = []
        next_actions, next_weights = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.visited_entities, self.prevent_cycles)
        
        self.relation_history = []  # NEW - track chosen relations at each step
    
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1] # Relations
        self.state['next_entities'] = next_actions[:, :, 0] # Target Entities
        self.state['current_entities'] = self.current_entities # Current Entities
        self.state['weights'] = next_weights # EDGE WEIGHTS
        self.state['visited_entities'] = self.visited_entities 

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward


    def _cache_ic_summaries(self):
        """
        Cache path-level IC mean per rollout (ignore sentinel=2.0 via NaN).
        Sets:
        - self.ic_mean: np.ndarray shape [B]
        """
        if not self.weight_history:
            B = self.start_entities.shape[0]
            self.ic_mean = np.zeros((B,), dtype=np.float32)
            return

        weights = np.array(self.weight_history, dtype=np.float32)  # [T, B]
        mask = (weights == 2.0)
        w = weights.copy()
        w[mask] = np.nan

        ic_mean = np.nanmean(w, axis=0)  # [B]
        self.ic_mean = np.nan_to_num(ic_mean, nan=0.0)


    #-- helper that will turn relation_history + visited_entities into a human-readable path string for each rollout, ready to send into get_reward_agenticAI() for persona scoring.
    def _build_paths_text(self, keep_idxs=None):
        
        """
        Build a readable string representation of one or more rollouts (paths).

        Parameters
        ----------
        keep_idxs : list[int] or None
            If provided, only these rollout indices will be included in the output.
            If None, all rollouts in the batch are included.

        Returns
        -------
        str : Human-readable multi-line string with query context and paths
        """
        
        # Try to load reverse vocabularies from the grapher (map IDs -> labels)
        rev_e = getattr(self.grapher, "rev_entity_vocab", None)
        rev_r = getattr(self.grapher, "rev_relation_vocab", None)

        # Stack relation history into an array of shape [T, B]
        rels = (np.array(self.relation_history, dtype=np.float32)
                if self.relation_history
                else np.zeros((0, self.visited_entities.shape[0]), dtype=np.float32))

        #NEW CODE - Helper to map entity numeric ID -> human-readable label
        # Flow: numeric ID -> rev_vocab -> vocab string (e.g. "Gene::134391") -> label (e.g. "SERPINC1")
        def name_e(eid):
            vocab_str = rev_e.get(int(eid), str(int(eid))) if rev_e else str(int(eid))
            return self.grapher.get_entity_label(vocab_str)

        # Helper to map relation numeric ID -> human-readable label
        # Flow: numeric ID -> rev_vocab -> vocab string (e.g. "CtD") -> label (e.g. "treats")
        def name_r(rid):
            vocab_str = rev_r.get(int(rid), str(int(rid))) if rev_r else str(int(rid))
            return self.grapher.get_relation_label(vocab_str)
        #END NEW CODE

        B = self.visited_entities.shape[0]  # Number of rollouts in the batch

        # If keep_idxs is provided, we only process those rollouts; else all
        idxs = range(B) if keep_idxs is None else keep_idxs

        lines = []
        
        # Add query context at the beginning
        # Get the query relation name (same for all rollouts in this batch)
        if len(idxs) > 0:
            first_idx = idxs[0]
            query_rel_id = self.query_relation[first_idx]
            query_rel_name = name_r(query_rel_id)
            
            # Get start and end entity names for context
            start_ent_name = name_e(self.start_entities[first_idx])
            end_ent_name = name_e(self.end_entities[first_idx])
            
            # Add query header
            lines.append(f"Query: Explaining why {start_ent_name} --[{query_rel_name}]--> {end_ent_name}")
            lines.append(f"Finding paths that explain the '{query_rel_name}' relationship.\n")
        
        # Build each path
        for out_i, b in enumerate(idxs, 1):
            # Start the path string with the starting entity
            path_segments = [name_e(self.visited_entities[b, 0])]

            # Steps = number of recorded relation hops (cannot exceed path_len-1)
            steps = min(rels.shape[0], self.visited_entities.shape[1] - 1)

            # Go through each step in this rollout
            for t in range(steps):
                rid = rels[t, b] if rels.size else 2.0
                # Stop if this step is marked as padding / rollout finished
                if rid == 2.0:
                    break
                ent_id = self.visited_entities[b, t + 1]
                # Append the relation and target entity to the path string
                path_segments.append(f"--[{name_r(rid)}]--> {name_e(ent_id)}")

            # Join all segments into one line for this rollout
            #lines.append(f"Path {out_i}: " + " ".join(path_segments))
            lines.append(f"Path {out_i} [id={b}]: " + " ".join(path_segments))


        # Join all path lines into one string separated by newlines
        return "\n".join(lines)

    # def get_scores_AgenticAI_OG(self, keep_idxs):
    #     """
    #     Sync micro-batching: score ALL eligible paths by splitting into batches,
    #     sleeping briefly between requests, and retrying on failures.
    #     Returns a list of dicts (len == len(keep_idxs)) in the same order.
    #     """
    #     if not self.agentic_ai_enabled or _agentic_client is None:
    #         return None
    #     if not keep_idxs:
    #         return None

    #     # ---- knobs (tune without code changes) ----
    #     BATCH_SIZE = int(os.getenv("AGENTIC_BATCH_SIZE", "20"))       # 20–40 recommended
    #     SLEEP_BETWEEN = float(os.getenv("AGENTIC_SLEEP_BETWEEN", "0.4"))  # 0.3–0.5s
    #     MAX_RETRIES = int(os.getenv("AGENTIC_MAX_RETRIES", "4"))
    #     # ------------------------------------------

    #     import time, random, json
    #     def _score_batch(batch_idxs, start_num):
    #         """One sync call with small retries + robust JSON extraction."""
    #         paths_text = self._build_paths_text(keep_idxs=batch_idxs)

    #         # # Debug: Check if paths_text looks correct
    #         # path_lines = [line for line in paths_text.split('\n') if line.startswith('Path ')]
    #         # print(f"[DEBUG] Generated text has {len(path_lines)} path lines for {len(batch_idxs)} indices")
            

    #         prompt = f"""
    #         You are evaluating drug–disease explanation paths from the perspective of the following persona:

    #         {self.persona_text}

    #         Score EACH path individually on three criteria:
    #         1. Scientific Validity (V): 1–5. Scientific correctness, plausibility, and coherence based on biomedical knowledge.
    #         2. Completeness (C): 1–5 where 3 is ideal. 1 = too simple, 5 = too complex. Reward paths that are sufficiently detailed without overload.
    #         3. Relevance (R): 1–5. Usefulness for understanding why the prediction matters and how it connects to the task.

    #         Paths to evaluate ({len(batch_idxs)} paths total):
    #             {paths_text}

    #         Return a JSON array with {len(batch_idxs)} objects (one per path above):
    #         [{{"validity": 4, "completeness": 3, "relevance": 5}}, ...]

    #         IMPORTANT: Your array must have EXACTLY {len(batch_idxs)} scores. DOUBLE CHECK BEFORE RETURNING RESULTS. 
    #         """.strip()

    #         last_err = None
    #         for attempt in range(1, MAX_RETRIES + 1):
    #             try:
    #                 resp = _agentic_client.chat.completions.create(
    #                     model="gpt-4o-mini",
    #                     messages=[{"role": "user", "content": prompt}],
    #                     temperature=0,
    #                 )
    #                 raw = resp.choices[0].message.content.strip()

    #                 # tolerate fenced code blocks
    #                 if "```" in raw:
    #                     parts = raw.split("```")
    #                     raw = "".join(p for p in parts if "[" in p and "]" in p)
    #                 # extract JSON slice
    #                 a, b = raw.find("["), raw.rfind("]")
    #                 if a != -1 and b != -1 and b > a:
    #                     raw = raw[a:b+1]

    #                 data = json.loads(raw)
    #                 if not isinstance(data, list):
    #                     raise ValueError("response is not a list")

    #                 # normalize and clamp
    #                 out = []
    #                 for item in data:
    #                     def _num(k, d=3.0):
    #                         try: return float(item.get(k, d))
    #                         except: return d
    #                     out.append({
    #                         "validity": max(1.0, min(5.0, _num("validity"))),
    #                         "completeness": max(1.0, min(5.0, _num("completeness"))),
    #                         "relevance": max(1.0, min(5.0, _num("relevance"))),
    #                     })

    #                 # Handle length mismatch gracefully
    #                 if len(out) != len(batch_idxs):
    #                     print(f"[WARN] Expected {len(batch_idxs)} scores, got {len(out)}")

    #                     # # Log the raw response for debugging
    #                     # if len(raw) < 2000:  # Only if not too long
    #                     #     print(f"[DEBUG] Raw response: {raw[:500]}...")
                        
    #                     # # Check if the LLM numbered the scores
    #                     # if out and isinstance(out[0], dict) and any('path' in str(k).lower() for k in out[0].keys()):
    #                     #     print("[DEBUG] LLM may have added path numbers/labels")
                        
    #                     # Pad with defaults if too few
    #                     while len(out) < len(batch_idxs):
    #                         print(f"  Adding default score for missing path {len(out)+1}")
    #                         out.append({"validity": 3.0, "completeness": 2.0, "relevance": 3.0})
                        
    #                     # Trim if too many
    #                     if len(out) > len(batch_idxs):
    #                         print(f"  Trimming extra scores")
    #                         out = out[:len(batch_idxs)]
                    
    #                 return out  # Return the padded/trimmed scores
                    
    #             except Exception as e:
    #                 last_err = e
    #                 if attempt < MAX_RETRIES:
    #                     delay = SLEEP_BETWEEN * (1.25 ** (attempt - 1)) + random.uniform(0, 0.2)
    #                     time.sleep(delay)
            
    #         # Only reached if all retries failed
    #         print(f"[FAIL] Batch completely failed after {MAX_RETRIES} attempts: {last_err}")
    #         return [{"validity": 2.5, "completeness": 3.0, "relevance": 2.5}
    #                 for _ in batch_idxs]
        
    #     # ---- NOW the main loop to process all batches ----
    #     results = []
    #     start_num = 1
    #     for i in range(0, len(keep_idxs), BATCH_SIZE):
    #         batch_idxs = keep_idxs[i:i + BATCH_SIZE]
    #         batch_scores = _score_batch(batch_idxs, start_num=start_num)
    #         results.extend(batch_scores)
    #         start_num += len(batch_idxs)
    #         if i + BATCH_SIZE < len(keep_idxs):  # Don't sleep after last batch
    #             time.sleep(SLEEP_BETWEEN)

    #     return results
    
    def get_scores_AgenticAI(self, keep_idxs):
        """
        Sync micro-batching: score ALL eligible paths by splitting into batches,
        sleeping briefly between requests, and retrying on failures.
        Returns a list of dicts (len == len(keep_idxs)) in the same order.
        """
        #if not self.agentic_ai_enabled or _agentic_client is None or not keep_idxs:
        if not self.agentic_ai_enabled or _call_llm is None or not keep_idxs:
            print("[INFO] Agentic AI disabled or client/persona missing; skipping LLM scoring.")
            return None

        # ---- knobs (tune without code changes) ----
        BATCH_SIZE = int(os.getenv("AGENTIC_BATCH_SIZE", "20"))       # 20–40 recommended
        SLEEP_BETWEEN = float(os.getenv("AGENTIC_SLEEP_BETWEEN", "0.4"))  # 0.3–0.5s
        MAX_RETRIES = int(os.getenv("AGENTIC_MAX_RETRIES", "4"))
        # ------------------------------------------

        import time, random, json

        def _score_batch(batch_idxs, start_num):
            """One sync call with small retries + robust JSON extraction."""
            paths_text = self._build_paths_text(keep_idxs=batch_idxs)

            # # Debug: Check if paths_text looks correct
            # path_lines = [line for line in paths_text.split('\n') if line.startswith('Path ')]
            # print(f"[DEBUG] Generated text has {len(path_lines)} path lines for {len(batch_idxs)} indices")
            n_lines = sum(1 for line in paths_text.splitlines() if line.startswith("Path "))
            if n_lines != len(batch_idxs):
                print(f"[WARN] Prompt contains {n_lines} Path-lines but batch size is {len(batch_idxs)}")

            id_list = list(map(int, batch_idxs))

            prompt = f"""
            You are evaluating drug–disease explanation paths from the perspective of the following persona:

            {self.persona_text}

            Score EACH path individually on three criteria:
            1. Scientific Validity (V): 1–5. Scientific correctness, plausibility, and coherence based on biomedical knowledge.
            2. Completeness (C): 1–5 where 3 is ideal. 1 = too simple, 5 = too complex. Reward paths that are sufficiently detailed without overload.
            3. Relevance (R): 1–5. Usefulness for understanding why the prediction matters and how it connects to the task.

            Paths to evaluate ({len(batch_idxs)} total). Each line has an [id=...]:
                {paths_text}

            Return ONLY valid JSON with an array of exactly {len(batch_idxs)} objects.
            Each object MUST be: {{"id": <int from {id_list}>, "validity": <int>, "completeness": <int>, "relevance": <int>}}.
            Use ONLY the ids from this set: {id_list}. Do not invent or omit ids. DOUBLE CHECK BEFORE RETURNING RESULTS.
            Do NOT return any text outside the JSON array. Do not return thinking traces or internal monologue. The output MUST be a JSON array of objects with id, validity, completeness, and relevance scores as described above. 
            """.strip()


            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"Attempted LLM call at time {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    """resp = _agentic_client.chat.completions.create(
                        model="Qwen/Qwen3.5-9B",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=81920,
                        temperature=1.0,
                        top_p=0.95,
                        presence_penalty=1.5, 
                        extra_body={
                            "top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    raw = resp.choices[0].message.content.strip()"""
                    #messages = [{"role": "system", "content": "You are a JSON generator. Respond ONLY with valid JSON. Do not explain or output anything except JSON. Think before producing a final response."}, {"role": "user", "content": prompt}]
                    messages = [{"role": "user", "content": prompt}]
                    resp, raw = _call_llm(messages, temperature=1.0)
                    print(f"[DEBUG] Parsed response: {resp}")

                    # tolerate fenced code blocks
                    if "```" in resp:
                        parts = resp.split("```")
                        resp = "".join(p for p in parts if "[" in p and "]" in p)
                    # extract JSON slice
                    a, b = resp.find("["), resp.rfind("]")
                    if a != -1 and b != -1 and b > a:
                        resp = resp[a:b+1]

                    data = json.loads(resp)  # after strip code fences

                    # Allow either a plain array or {"scores":[...]} wrapper
                    if isinstance(data, dict) and "scores" in data:
                        data = data["scores"]
                    if not isinstance(data, list):
                        raise ValueError("response is not a list")

                    def clamp1to5(x, d=3.0):
                        try:
                            v = float(x)
                        except Exception:
                            v = d
                        return max(1.0, min(5.0, v))

                    # keep only allowed ids (drop extras), prefer first occurrence
                    allowed = set(id_list)
                    by_id = {}
                    for item in data:
                        if not isinstance(item, dict) or "id" not in item: 
                            continue
                        try: pid = int(item["id"])
                        except: continue
                        if pid in allowed and pid not in by_id:
                            by_id[pid] = {
                                "validity":     clamp1to5(item.get("validity", 3)),
                                "completeness": clamp1to5(item.get("completeness", 2)),
                                "relevance":    clamp1to5(item.get("relevance", 3)),
                            }

                    # Now assemble output strictly in batch order, filling defaults when missing
                    out, missing = [], 0
                    for pid in id_list:
                        if pid in by_id:
                            out.append(by_id[pid])
                        else:
                            missing += 1
                            out.append({"validity":3.0, "completeness":2.0, "relevance":3.0})

                    if missing or len(by_id) != len(id_list):
                        print(f"[WARN] ID-mismatch: expected {len(id_list)}, got {len(by_id)}, "
                            f"filled defaults for {missing} ids")
                        print(f"[DEBUG] Prompt was:\n{prompt}\n")
                        print(f"[DEBUG] Raw response was:\n{raw}\n")
                    return out
                    
                except Exception as e:
                    last_err = e
                    if attempt < MAX_RETRIES:
                        delay = SLEEP_BETWEEN * (1.25 ** (attempt - 1)) + random.uniform(0, 0.2)
                        time.sleep(delay)
            
            # Only reached if all retries failed
            print(f"[FAIL] Batch completely failed after {MAX_RETRIES} attempts: {last_err}")
            return [{"validity": 3.0, "completeness": 2.0, "relevance": 3.0}
                    for _ in batch_idxs]
        
        # ---- group by example id so each prompt has a single (start,end) context ----
        groups = defaultdict(list)
        nr = int(self.num_rollouts)
        for idx in keep_idxs:
            groups[int(idx)//nr].append(int(idx))

        results = []
        for ex_id, g in groups.items():
            for i in range(0, len(g), BATCH_SIZE):
                batch_idxs = g[i:i+BATCH_SIZE]
                results.extend(_score_batch(batch_idxs, start_num=1))
                if i + BATCH_SIZE < len(g):
                    time.sleep(SLEEP_BETWEEN)

        return results
    
    def get_reward_agenticAI(self, wV=1/3, wC=1/3, wR=1/3):
        training_step = getattr(Episode, '_training_step', 0)
        
        base = self.get_reward_weights_sigmoid()
        self._cache_ic_summaries()
        B = base.shape[0]

        # >>> NEW: if agentic mode is disabled (or no client/persona), just use base
        #if (not self.agentic_ai_enabled) or (_agentic_client is None) or (not self.persona_text):
        if (not self.agentic_ai_enabled) or (_call_llm is None) or (not self.persona_text):
            # keep logging arrays coherent so the rest of the pipeline/test logger works
            print("[INFO] Agentic AI disabled or client/persona missing; using IC-based rewards only.")
            #print(self.agentic_ai_enabled, _call_llm is None, not self.persona_text)
            self.agentic_scores = base.astype(np.float32)
            self.llm_dimensions = {}
            self.reward_kind = np.array(['ic_only'] * B, dtype=object)
            return base
        # <

        # if training_step < 50:
        #     # Just return the IC-based rewards directly
        #     return base  # Simple and effective
        # PHASE 2: Persona-based refinement
        # Now we have a model that knows how to find paths
        # Time to shape those paths according to persona preferences

        # # Progressive threshold
        # if training_step < 60:
        #     threshold = 0.5
        # elif training_step < 80:
        #     threshold = 0.55
        # elif training_step < 120:
        #     threshold = 0.6
        # elif training_step < 150:
        #     threshold = 0.65
        # else:
        #     threshold = 0.7
        
        threshold = threshold_for_step(training_step)
        
        # Three tiers of paths
        high_ic_idxs = np.where(base > threshold)[0].tolist()
        medium_ic_idxs = np.where((base > 0.5) & (base <= threshold))[0].tolist() if threshold > 0.5 else []
        low_ic_idxs = np.where((base > 0.3) & (base <= 0.5))[0].tolist()  # NEW tier
        

        #Log distribution
        # print(f"[STEP {training_step}] Threshold={threshold:.2f}")
        # print(f"  High IC (>{threshold:.2f}): {len(high_ic_idxs)} paths for LLM")
        # print(f"  Medium IC (0.5-{threshold:.2f}): {len(medium_ic_idxs)} paths get 0.25")
        # print(f"  Low IC (0.3-0.5): {len(low_ic_idxs)} paths get 0.1")
        
        lines = [
            f"[STEP {training_step}] Threshold={threshold:.2f}",
            f"  High IC (>{threshold:.2f}): {len(high_ic_idxs)} paths for LLM",
            f"  Medium IC (0.5-{threshold:.2f}): {len(medium_ic_idxs)} paths get 0.25",
            f"  Low IC (0.3-0.5): {len(low_ic_idxs)} paths get 0.1",
        ]
        for s in lines:
            print(s)
            _env_logger.info(s)

        out = np.zeros((B,), dtype=np.float32)

        # Initialize storage for agentic scores and clear previous dimensions
        self.agentic_scores = np.zeros((B,), dtype=np.float32)
        self.llm_dimensions = {}  # Clear previous dimensions
        self.reward_kind = np.array(['none']*B, dtype=object)  #  track source


        # Give ALL tiers some reward
        for idx in low_ic_idxs:
            out[idx] = 0.1   # Small reward for weak paths
            self.agentic_scores[idx] = 0.10
            self.reward_kind[idx] = 'low_fixed'
        for idx in medium_ic_idxs:
            out[idx] = 0.25  # Decent reward for medium paths (increased from 0.20)
            self.agentic_scores[idx] = 0.25
            self.reward_kind[idx] = 'medium_fixed'
        
     
        # Score high-IC paths with LLM
        llm_rewards=[]
        if high_ic_idxs:
            scores_list = self.get_scores_AgenticAI(high_ic_idxs)
            if scores_list is None:
                # If API fails, give high-IC paths a default
                for idx in high_ic_idxs:
                    out[idx] = 0.30               
                    self.agentic_scores[idx] = 0.30  
                    self.reward_kind[idx] = 'high_default'
                    
            else:
                # Process LLM scores
                for idx, scores in zip(high_ic_idxs, scores_list):
                    sv = float(scores["validity"])
                    comp_raw = float(scores["completeness"])
                    rel = float(scores["relevance"])
                    
                    # Print raw LLM scores
                    #print(f"  Path {idx}: V={sv:.1f}, C={comp_raw:.1f}, R={rel:.1f}", end="")
                    
                    comp_conv = COMPLETENESS_MAP[int(comp_raw)]
                    raw = wV * sv + wC * comp_conv + wR * rel
                    
                    final_norm = (raw - 1.0) / 4.0
                    final_norm = float(max(0.0, min(1.0, final_norm)))
                    
                    #print(f" → reward={final_norm:.3f}")
                    
                    out[idx] = final_norm
                    self.agentic_scores[idx] = final_norm     
                    self.reward_kind[idx] = 'llm'   
                
                    # Store the individual dimensions for LLM-scored paths
                    self.llm_dimensions[idx] = {
                        'validity': sv,
                        'completeness_conv': comp_conv,
                        'relevance': rel,
                        'ic_mean': float(self.ic_mean[idx])   

                    }

                    llm_rewards.append(final_norm)

                # Summary statistics
                if llm_rewards:
                    # print(f"  LLM rewards: min={min(llm_rewards):.3f}, "
                    #     f"avg={np.mean(llm_rewards):.3f}, max={max(llm_rewards):.3f}")
                    s = (f"  LLM rewards: min={min(llm_rewards):.3f}, "
                        f"avg={np.mean(llm_rewards):.3f}, max={max(llm_rewards):.3f}")
                    print(s)
                    _env_logger.info(s)
        
        # Summary of all rewards
        positive_rewards = out[out > 0]
        if len(positive_rewards) > 0:
            # print(f"[TOTAL] {len(positive_rewards)} paths rewarded "
            #     f"(avg={np.mean(positive_rewards):.3f})")
            s = (f"[TOTAL] {len(positive_rewards)} paths rewarded "
            f"(avg={np.mean(positive_rewards):.3f})")
            print(s)
            _env_logger.info(s)
        
        # --- bonus: machine-friendly JSON line for analysis (JSONL) ---
        try:
            _env_logger.info("METRICS " + json.dumps({
                "step":               int(training_step),
                "threshold":          float(threshold),
                "counts": {
                    "high_ic":        int(len(high_ic_idxs)),
                    "medium_ic":      int(len(medium_ic_idxs)),
                    "low_ic":         int(len(low_ic_idxs)),
                    "rewarded_total": int(len(positive_rewards)),
                },
                "rewarded_avg":       (float(np.mean(positive_rewards))
                                    if len(positive_rewards) else 0.0),
                "llm_rewards": (
                    {"min": float(min(llm_rewards)),
                    "avg": float(np.mean(llm_rewards)),
                    "max": float(max(llm_rewards))}
                    if llm_rewards else None
                ),
            }, ensure_ascii=False))
        except Exception:
            pass
        return out





    def get_reward_weights_sigmoid(self):
        """
        CALCULATE REWARD BASED ON THE POSITIVE REWARD AND THE AVERAGE WEIGHT (IC).
        USE '2.0' AS A SENTINEL FOR PADDING AND IGNORE IT IN THE MEAN.
        """
        # 1) CONVERT THE LIST OF WEIGHT VECTORS INTO A 2D ARRAY: [TIME, BATCH]
        weights_array = np.array(self.weight_history)  # SHAPE (T, B), WHERE T = # STEPS

        # 2) MAKE A MASK FOR THE PADDING (WHERE WE HAVE 2.0) 
        #    'True' => IT IS PADDING, 'False' => REAL WEIGHT
        mask_2 = (weights_array == 2.0)

        # 3) REPLACE 2.0 BY np.nan SO WE CAN IGNORE IT IN THE AVERAGE
        weights_array[mask_2] = np.nan

        # 4) CALCULATE THE MEAN ALONG AXIS=0 (ROLLOUT DIM), IGNORING NaN
        average_ic = np.nanmean(weights_array, axis=0)  # SHAPE [B,]

        # 5) CALCULATE SIZE OF THE PATHS
        size = np.sum(~mask_2, axis=0)  # shape (B,)

       # 6) GIVE A PENALTY TO THE SIZE OF THE PATH:
        #    IF SIZE >= 3 => 0.5 (PENALIZE)
        #    ELSE         => 1 (KEEP REWARD)
        punish_size = np.where(size >= 3, 0.5, 1)
    
        # 7) BUILD THE REWARD BASED ON SUCCESS
        success_mask = (self.current_entities == self.end_entities)

        # 8) CALCULATE REWARD

        if self.weighted_reward==True and self.sigmoid==False:
            positive_part = self.positive_reward * average_ic
            
        else:
            positive_part = self.positive_reward

        # 9) BUILD THE FINAL REWARD
        condlist   = [success_mask, ~success_mask]
        choicelist = [positive_part, self.negative_reward]
        final_reward = np.select(condlist, choicelist)
        return final_reward


    def get_reward_weights(self):
        """
        Calculate reward based on the positive reward and the weights of the edges.
         - If the current entities match the end entities, reward is calculated as:
        average of edge weights multiplied by the positive reward.
        - Otherwise, the negative reward is applied.
        """
        reward = (self.current_entities == self.end_entities)

        #calculate the average of the edge weights
        average_ic = np.mean(self.weight_history, axis=0) 

        if self.weighted_reward:
            #simple multiplication of the positive reward with the average of the edge weights
            positive_reward = self.positive_reward * average_ic

        else:
            positive_reward = self.positive_reward

        
        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  

        # print if positive reward is given 
        if np.any(reward == self.positive_reward):
            print("Positive reward was given")

        return reward


    def __call__(self, action):
        if self.size_flexibility:
            self.current_hop += 1
            bsz = self.no_examples * self.num_rollouts

            # GET CHOSEN ENTITIES
            chosen_ents = self.state['next_entities'][np.arange(bsz), action]
            self.current_entities = chosen_ents

            # --- NEW
            #track chosen relations for each step
            chosen_rels = self.state['next_relations'][np.arange(bsz), action]
            
            # GET THE CHOSEN WEIGHTS
            chosen_weights = self.state['weights'][np.arange(bsz), action]

            # ANY ROLLOUT THAT REACHES THE END_ENTITY => done_mask = TRUE
            newly_done = (chosen_ents == self.end_entities)
            prev_done  = self.done_mask.copy()
        
            # PAD ONLY ROLLOUTS THAT WERE ALREADY DONE BEFORE THIS STEP
            chosen_weights[prev_done] = 2.0
            chosen_rels[prev_done]    = 2.0 #-- NEW 

            # APPEND REAL WEIGHT FOR THE LAST HOP OF NEWLY COMPLETED ROLLOUTS
            self.weight_history.append(chosen_weights)
            self.relation_history.append(chosen_rels) #-- NEW 

            # MARK NEW COMPLETION AS DONE FOR FUTURE STEPS
            self.done_mask = np.logical_or(self.done_mask, newly_done)
     
            # UPDATE VISITED ENTITIES
            new_visited = np.zeros((bsz, self.visited_entities.shape[1] + 1), dtype=np.int32)
            for i in range(bsz):
                new_visited[i, :self.visited_entities.shape[1]] = self.visited_entities[i]
                new_visited[i, -1] = chosen_ents[i]
            self.visited_entities = new_visited

            # GET NEXT ACTIONS/WEIGHTS (WE STILL NEED THIS FOR THE ROLLOUTS NOT DONE)
            next_actions, next_weights = self.grapher.return_next_actions(
                self.current_entities,
                self.start_entities,
                self.query_relation,
                self.end_entities,
                self.all_answers,
                (self.current_hop == self.path_len - 1),
                self.num_rollouts, 
                self.visited_entities, # Pass the visited entities list
                self.prevent_cycles

            )
     
            # UPDATE STATE
            self.state['next_relations']  = next_actions[:, :, 1]
            self.state['next_entities']   = next_actions[:, :, 0]
            self.state['current_entities'] = self.current_entities
            self.state['weights'] = next_weights

            return self.state
            

        else:
            self.current_hop += 1
            bsz = self.no_examples * self.num_rollouts

            self.current_entities = self.state['next_entities'][np.arange(bsz), action]

            # --- NEW
            chosen_rels = self.state['next_relations'][np.arange(bsz), action]
            self.relation_history.append(chosen_rels)

            # Append weights and update next actions as before
            self.weight_history.append(self.state['weights'][np.arange(bsz), action])

            # Update visited entities
            new_visited = np.zeros((bsz, self.visited_entities.shape[1] + 1), dtype=np.int32)
            for i in range(bsz):
                new_visited[i, :self.visited_entities.shape[1]] = self.visited_entities[i]
                new_visited[i, self.visited_entities.shape[1]] = self.current_entities[i]
                    
            self.visited_entities = new_visited

            next_actions, next_weights = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                            self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                            self.num_rollouts, self.visited_entities,  self.prevent_cycles )

            self.state['next_relations'] = next_actions[:, :, 1]
            self.state['next_entities'] = next_actions[:, :, 0]
            self.state['current_entities'] = self.current_entities
            self.state['weights'] = next_weights # EDGE WEIGHTS
            self.state['visited_entities'] = self.visited_entities  # Add visited entities to state

            return self.state


class env(object):
    def __init__(self, params, mode='train'):
        self.persona_path = params['persona_path'] # NEW 
        self.agentic_ai_enabled = params['agentic_ai_enabled'] # NEW 
        self.weighted_reward = params['weighted_reward']
        self.adjust_factor = params['IC_importance']
        self.sigmoid = params['sigmoid']
        self.size_flexibility = params['size_flexibility']
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.prevent_cycles = params['prevent_cycles']
        self.mode = mode
         # NEW: optional separate env log file path
        self.env_log_file = params.get('env_log_file')  # e.g. ".../env_metrics.log"
        configure_env_logger(self.env_log_file)
        # ...
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 edges_weight=params['edges_weight']
                                                 )
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode =mode,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 edges_weight=params['edges_weight'])

            self.total_no_examples = self.batcher.store.shape[0]
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'],
                                             edges_weight=params['edges_weight'],
                                             labels_dir=params['data_input_dir']
                                             )

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher, self.weighted_reward, self.adjust_factor, self.sigmoid, self.size_flexibility, self.prevent_cycles, self.persona_path, self.agentic_ai_enabled #NEW ADDITION OF PERSONA
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
