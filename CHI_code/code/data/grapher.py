from collections import defaultdict
import logging
import numpy as np
import csv
logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    def __init__(self, triple_store, relation_vocab, entity_vocab, edges_weight, max_num_actions, labels_dir=None):

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.edges_weight = edges_weight 
        self.store = defaultdict(list)
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = None

        # EDGES WEIGHTS WITH STANDARD VALUES
        self.weights_store = np.zeros((len(entity_vocab), max_num_actions), dtype=np.float32)

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])

        #NEW CODE - Load human-readable labels for LLM prompts (entity ID -> name, relation code -> name)
        self.entity_labels = {}   # e.g. "Compound::DB00808" -> "Loperamide"
        self.relation_labels = {} # e.g. "CtD" -> "treats"
        if labels_dir:
            self._load_labels(labels_dir)
        #END NEW CODE

        self.create_graph()
        print("KG constructed")

    #NEW CODE - Load and lookup human-readable labels
    def _load_labels(self, labels_dir):
        """Load human-readable labels from TSV files in the dataset folder."""
        import os
        # Load entity labels: graph_labels.tsv (entity_id \t label \t type)
        entity_labels_path = os.path.join(labels_dir, "graph_labels.tsv")
        if os.path.isfile(entity_labels_path):
            with open(entity_labels_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        self.entity_labels[row[0]] = row[1]
            logger.info(f"Loaded {len(self.entity_labels)} entity labels from {entity_labels_path}")
        else:
            logger.warning(f"Entity labels file not found: {entity_labels_path}")

        # Load relation labels: edges_labels.tsv (relation_code \t label)
        relation_labels_path = os.path.join(labels_dir, "edges_labels.tsv")
        if os.path.isfile(relation_labels_path):
            with open(relation_labels_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        self.relation_labels[row[0]] = row[1]
            logger.info(f"Loaded {len(self.relation_labels)} relation labels from {relation_labels_path}")
        else:
            logger.warning(f"Relation labels file not found: {relation_labels_path}")

    def get_entity_label(self, entity_id_str):
        """Map an entity vocab string (e.g. 'Compound::DB00808') to its human label (e.g. 'Loperamide')."""
        return self.entity_labels.get(entity_id_str, entity_id_str)

    def get_relation_label(self, relation_code):
        """Map a relation vocab string (e.g. 'CtD') to its human label (e.g. 'treats')."""
        return self.relation_labels.get(relation_code, relation_code)
    #END NEW CODE

    def create_graph(self):
        #open the file and read the triples
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                # map the entities and relations to their respective indices
                ent1, rel, ent2 = line[0], line[1], line[2]
                e1 = self.entity_vocab[ent1]
                r = self.relation_vocab[rel]
                e2 = self.entity_vocab[ent2]
                # store the triples in a dictionary
                #self.store[e1].append((r, e2))

                #ADD WEIGHTS TO THE EDGES FOR CLUSTERED IC BY EDGE TYPE
                if rel in self.edges_weight.keys():
                    w_e1 = self.edges_weight[rel].get(ent1, 0.5)
                    w_e2 = self.edges_weight[rel].get(ent2, 0.5)
                    # aggregate the IC of the source and target nodes
                    w = (w_e1 + w_e2) / 2
                else: 
                    w = 0.5 

                    
                # # ADD WEIGHTS TO THE EDGES FOR CLUSTERED IC AND NODE DEGREE 
                # w_e1 = self.edges_weight.get(ent1, 0.5)
                # w_e2 = self.edges_weight.get(ent2, 0.5)
                # w = (w_e1 + w_e2) / 2

                
                #store triples and weights 
                self.store[e1].append((r, e2, w))
        
        #build the array store
        for e1 in self.store:
            num_actions = 1
            ###   NO_OP RELATION
            ###self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP'] # COMMENT THIS LINE FOR REX
            self.array_store[e1, 0, 0] = e1
            for r, e2, w in self.store[e1]: #FOR THE WEIGHT
                if num_actions == self.array_store.shape[1]:
                    break
                self.array_store[e1,num_actions,0] = e2 #target ent
                self.array_store[e1,num_actions,1] = r # relation
                self.weights_store[e1, num_actions] = w # STORE THE WEIGHT

                num_actions += 1
        
        #clean up the temporary store
        del self.store
        self.store = None

    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts, visited_entities, prevent_cycles=True):
        ret = self.array_store[current_entities, :, :].copy()
        weights = self.weights_store[current_entities, :].copy()  # COPY WEIGHTS OF THE ACTIONS

        for i in range(current_entities.shape[0]):           
            # Get the candidate entities and relations for this step
            entities = ret[i, :, 0]
            relations = ret[i, :, 1]

             # AVOID SELF LOOPS 
            # Filter out visited entities when prevent_cycles is True and we have visited_entities
            if prevent_cycles and visited_entities is not None:
                # Create mask using NumPy's in1d function (tests whether each element of entities is in visited_entities[i])
                visited_mask = np.in1d(entities, visited_entities[i])
                
                # Apply mask in one operation
                entities[visited_mask] = self.ePAD
                relations[visited_mask] = self.rPAD
                weights[i, visited_mask] = 0.0
                ######## # AVOID SELF LOOPS

            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i] , entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD
                weights[i, mask] = 0.0  # MASCARA ALSO ADJUSTS WEIGHTS

            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[i//rollouts] and entities[j] != correct_e2:
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD
                        weights[i, j] = 0.0  # ADJUST WEIGHTS TO INVALID ENTITIES 


        # FINAL VALIDATION TO ENSURE WEIGHTS CONSISTENT WITH PADs
        weights[ret[:, :, 0] == self.ePAD] = 0.0
        weights[ret[:, :, 1] == self.rPAD] = 0.0


      
        return ret, weights
