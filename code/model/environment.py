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

import json
from dotenv import load_dotenv
load_dotenv()

# --llm_api 0 (default): loads Qwen 3.5 locally (needs GPU)
# --llm_api 1 --llm_model qwen: Qwen via HuggingFace API
# --llm_api 1 --llm_model gpt:  GPT via OpenAI API

def _extract_json_response(response):
    """Common helper to extract JSON array from LLM response text."""
    responseSplit = response
    if "<think>" in response and "</think>" in response:
        responseSplit = response.split("</think>", 1)[1].strip()
    json_start = responseSplit.find("[")
    json_end = responseSplit.rfind("]")
    if json_start != -1 and json_end > json_start:
        candidate = responseSplit[json_start:json_end+1]
        try:
            json.loads(candidate)  # validate
            return candidate, response
        except json.JSONDecodeError:
            pass
    return "", response

try:
    _llm_model = None
    _llm_tokenizer = None
    _llm_device = None
    _local_model_name = "Qwen/Qwen3.5-9B"  # default, overridden by --local_model

    def _init_llm():
        global _llm_model, _llm_tokenizer, _llm_device
        if _llm_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        MODEL_NAME = _local_model_name
        hf_token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        print(f"[LLM] Loading {MODEL_NAME}...")
        if torch.cuda.is_available():
            _llm_device = "cuda"
        elif torch.backends.mps.is_available():
            _llm_device = "mps"
        else:
            _llm_device = "cpu"
        _llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        if _llm_device == "cpu":
            model_dtype = torch.float32
        elif _llm_device == "mps":
            model_dtype = torch.float16   # bfloat16 causes inf/nan on MPS during sampling
        else:
            model_dtype = torch.bfloat16
        _llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=model_dtype,
            device_map="auto",
            token=hf_token,
        )
        print(f"[LLM] Ready on {_llm_device}")

    def _call_llm_local(messages, temperature=0):
        _init_llm()
        try:
            prompt = _llm_tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False
            )
            model_device = next(_llm_model.parameters()).device
            inputs = _llm_tokenizer(prompt, return_tensors="pt").to(model_device)
            # temperature=0 with do_sample=True causes inf/nan on MPS (division by ~0)
            if temperature <= 0:
                gen_kwargs = dict(do_sample=False)
            else:
                gen_kwargs = dict(do_sample=True, temperature=max(temperature, 0.01), top_p=0.9)
            with torch.no_grad():
                try:
                    outputs = _llm_model.generate(
                        **inputs, max_new_tokens=3072,
                        **gen_kwargs,
                        pad_token_id=_llm_tokenizer.pad_token_id,
                    )
                except RuntimeError as gen_err:
                    # Sampling can produce inf/nan logits on MPS with float16,
                    # crashing with "probability tensor contains either inf, nan or element < 0".
                    # When that happens, fall back to greedy decoding which skips the
                    # probability distribution entirely (always picks the top token).
                    # - max_new_tokens=512: capped (vs 3072) to limit generation time,
                    #   since we only need a short JSON response.
                    # - repetition_penalty=1.3: prevents greedy from entering repetitive
                    #   token loops (common failure mode that causes very slow generation).
                    # - warnings suppressed: transformers logs a spurious warning about
                    #   temperature/top_p/top_k not being valid for greedy; safe to ignore.
                    # inf/nan in sampling — just let it fail and skip this batch
                    raise

            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = _llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"[LLM] Call successful")
            return _extract_json_response(response)
        except Exception as e:
            print(f"[LLM LOCAL ERROR] {e}")
            return "", ""

    # Qwen via HuggingFace API (--llm_api 1 --llm_model qwen).
    # Uses httpx directly (openai SDK bug with extra_body + enable_thinking).
    _qwen_http_client = None
    try:
        import httpx as _httpx
    except ImportError as _e:
        _httpx = None
        logger.warning("Qwen API mode disabled: httpx not installed (%s).", _e)

    import time as _time

    def _call_llm_qwen_api(messages, temperature=0):
        global _qwen_http_client
        if _httpx is None:
            print("[LLM] Qwen API mode requested but httpx is unavailable.")
            return "", ""
        base_url = os.getenv("HF_API_BASE", "https://router.huggingface.co/together/v1")
        if _qwen_http_client is None:
            _qwen_http_client = _httpx.Client(timeout=120.0)
            print(f"[LLM] Qwen API mode via HuggingFace")
        try:
            model = os.getenv("HF_MODEL", "Qwen/Qwen3.5-9B")
            resp = _qwen_http_client.post(
                base_url + "/chat/completions",
                headers={
                    "Authorization": "Bearer " + os.getenv("HF_API_KEY", ""),
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 3072,
                    "temperature": temperature if temperature > 0 else 0.01,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            data = resp.json()
            # rate limit / error handling
            if "choices" not in data:
                err_obj = data.get("error", data) if isinstance(data, dict) else data
                err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                print(f"[LLM QWEN API ERROR] {err_msg}")
                # if rate limited, wait and retry once
                if resp.status_code == 429 or "rate" in err_msg.lower():
                    wait = 5
                    print(f"[LLM] Rate limited, waiting {wait}s...")
                    _time.sleep(wait)
                    resp = _qwen_http_client.post(
                        base_url + "/chat/completions",
                        headers={
                            "Authorization": "Bearer " + os.getenv("HF_API_KEY", ""),
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": messages,
                            "max_tokens": 3072,
                            "temperature": temperature if temperature > 0 else 0.01,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    data = resp.json()
                    if "choices" not in data:
                        return "", ""
                else:
                    return "", ""
            raw = data["choices"][0]["message"]["content"].strip()
            _time.sleep(0.5)  # small delay to avoid rate limits
            return _extract_json_response(raw)
        except Exception as e:
            print(f"[LLM QWEN API ERROR] {e}")
            return "", ""

    # GPT via OpenAI API (--llm_api 1 --llm_model gpt).
    _gpt_api_client = None

    def _call_llm_gpt_api(messages, temperature=0):
        global _gpt_api_client
        if _gpt_api_client is None:
            from openai import OpenAI
            _gpt_api_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                timeout=120.0,
            )
            print(f"[LLM] GPT API mode via OpenAI")
        try:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = _gpt_api_client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=3072,
                temperature=temperature if temperature > 0 else 0.01,
            )
            raw = resp.choices[0].message.content.strip()
            return _extract_json_response(raw)
        except Exception as e:
            print(f"[LLM GPT API ERROR] {e}")
            return "", ""

    # Default to local; Episode.__init__ overrides based on --llm_api / --llm_model
    _call_llm = _call_llm_local

except Exception as _e:
    _call_llm = None
    logger.warning("LLM not available: %s", _e)


# Dedicated env logger (separate file, no console)
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


# Completeness is best at 3; triangular mapping back to a 1–5 scale
COMPLETENESS_MAP = {1: 1.0, 2: 3.0, 3: 5.0, 4: 3.0, 5: 1.0}

def threshold_for_step(step: int) -> float:
    if step < 60:    return 0.50
    elif step < 80:  return 0.55
    elif step < 100: return 0.60
    else:            return 0.65


class Episode(object):
    _training_step = 0
    _test_threshold_override = None  # if set, replaces threshold_for_step at reward time

    @classmethod
    def set_training_step(cls, step):
        cls._training_step = step

    @classmethod
    def set_test_threshold_override(cls, value):
        cls._test_threshold_override = value

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher, IC_reward, adjust_factor, early_stopping, prevent_cycles, persona_path, agentic_ai_enabled, llm_api, llm_model, local_model  = params
        # Set LLM caller based on --llm_api / --llm_model
        global _call_llm, _local_model_name
        _local_model_name = local_model  # set before _init_llm() is called
        if llm_api:
            if llm_model == 'gpt':
                _call_llm = _call_llm_gpt_api
            else:
                _call_llm = _call_llm_qwen_api
        else:
            _call_llm = _call_llm_local
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
        self.IC_reward = IC_reward
        self.early_stopping = early_stopping
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

        # Load persona text (empty string if file missing)
        self.agentic_ai_enabled = bool(agentic_ai_enabled)
        self.persona_path = persona_path
        self.persona_text = ""
        if self.agentic_ai_enabled and self.persona_path and os.path.isfile(self.persona_path):
            with open(self.persona_path, "r", encoding="utf-8") as f:
                self.persona_text = f.read().strip()

        self.agentic_scores = None  # Final agentic scores for all rollouts
        self.llm_dimensions = {}    # Store v, c_conv, r for LLM-scored paths only

        self.done_mask = np.zeros(self.no_examples * self.num_rollouts, dtype=bool)

        # Each rollout tracks its own visited entities; initially just the start entity
        self.visited_entities = np.zeros((self.no_examples * self.num_rollouts, 1), dtype=np.int32)
        self.visited_entities[:, 0] = self.start_entities

        self.weight_history = []
        next_actions, next_weights = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.visited_entities, self.prevent_cycles)

        self.relation_history = []

        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        self.state['weights'] = next_weights
        self.state['visited_entities'] = self.visited_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation


    def _cache_ic_summaries(self):
        """
        Cache path-level IC mean per rollout.
        Sets:
        - self.ic_mean: np.ndarray shape [B]
        """
        self._recompute_from_path()
        self.ic_mean = self.recomputed_ic_mean

    def _recompute_from_path(self):
        """
        Recompute IC weights and relations directly from visited_entities
        using the graph structure (array_store / weights_store), bypassing
        potentially misaligned weight_history and relation_history from
        beam search reordering.

        Called once per episode; results are cached.

        Sets:
        - self.recomputed_ic_per_step: list of np.ndarray [B], sentinel=2.0 for padding
        - self.recomputed_rels_per_step: list of np.ndarray [B], sentinel=2.0 for padding
        - self.recomputed_ic_mean: np.ndarray [B]
        """
        if getattr(self, '_path_recomputed', False):
            return

        B = self.visited_entities.shape[0]
        T = self.visited_entities.shape[1] - 1  # number of edges

        if T == 0:
            self.recomputed_ic_per_step = []
            self.recomputed_rels_per_step = []
            self.recomputed_ic_mean = np.zeros(B, dtype=np.float32)
            self._path_recomputed = True
            return

        # Pre-allocate: sentinel 2.0 means padding (same convention as weight_history)
        all_weights = np.full((T, B), 2.0, dtype=np.float32)
        all_rels = np.full((T, B), 2.0, dtype=np.float32)

        for b in range(B):
            for t in range(T):
                e1 = int(self.visited_entities[b, t])
                e2 = int(self.visited_entities[b, t + 1])

                # Look up edge e1->e2 in the graph
                neighbors = self.grapher.array_store[e1]    # [max_actions, 2]
                edge_wts = self.grapher.weights_store[e1]    # [max_actions]

                match_idxs = np.where(neighbors[:, 0] == e2)[0]

                if len(match_idxs) > 0:
                    # Prefer non-self-loop (index > 0); disambiguate with
                    # relation_history hint when multiple edges exist
                    idx = match_idxs[0]
                    if len(match_idxs) > 1:
                        # Try relation_history as hint (best-effort)
                        if t < len(self.relation_history):
                            rel_hint = self.relation_history[t][b]
                            if rel_hint != 2.0:
                                for mi in match_idxs:
                                    if neighbors[mi, 1] == int(rel_hint):
                                        idx = mi
                                        break
                        # Otherwise prefer first non-self-loop edge
                        elif idx == 0 and len(match_idxs) > 1:
                            idx = match_idxs[1]

                    all_weights[t, b] = edge_wts[idx]
                    all_rels[t, b] = float(neighbors[idx, 1])
                else:
                    # Edge not in graph — fallback
                    all_weights[t, b] = 0.5
                    all_rels[t, b] = -1.0

                # If we reached the target entity, remaining steps are padding
                if self.early_stopping and e2 == self.end_entities[b]:
                    break

        self.recomputed_ic_per_step = [all_weights[t] for t in range(T)]
        self.recomputed_rels_per_step = [all_rels[t] for t in range(T)]

        # Compute mean IC per rollout (ignore sentinel 2.0)
        mask_2 = (all_weights == 2.0)
        w = all_weights.copy()
        w[mask_2] = np.nan
        ic_mean = np.nanmean(w, axis=0)
        self.recomputed_ic_mean = np.nan_to_num(ic_mean, nan=0.0)

        self._path_recomputed = True

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

        self._recompute_from_path()
        rels = (np.array(self.recomputed_rels_per_step, dtype=np.float32)
                if self.recomputed_rels_per_step
                else np.zeros((0, self.visited_entities.shape[0]), dtype=np.float32))

        # Map entity numeric ID -> human-readable label.
        # Flow: numeric ID -> rev_vocab -> vocab string (e.g. "Gene::134391") -> label (e.g. "SERPINC1")
        def name_e(eid):
            vocab_str = rev_e.get(int(eid), str(int(eid))) if rev_e else str(int(eid))
            return self.grapher.get_entity_label(vocab_str)

        # Map relation numeric ID -> human-readable label.
        # Flow: numeric ID -> rev_vocab -> vocab string (e.g. "CtD") -> label (e.g. "treats")
        def name_r(rid):
            vocab_str = rev_r.get(int(rid), str(int(rid))) if rev_r else str(int(rid))
            return self.grapher.get_relation_label(vocab_str)

        B = self.visited_entities.shape[0]

        # If keep_idxs is provided, we only process those rollouts; else all
        idxs = range(B) if keep_idxs is None else keep_idxs

        lines = []

        # The query relation name is the same for all rollouts in this batch
        if len(idxs) > 0:
            first_idx = idxs[0]
            query_rel_id = self.query_relation[first_idx]
            query_rel_name = name_r(query_rel_id)

            start_ent_name = name_e(self.start_entities[first_idx])
            end_ent_name = name_e(self.end_entities[first_idx])

            lines.append(f"Query: Explaining why {start_ent_name} --[{query_rel_name}]--> {end_ent_name}")
            lines.append(f"Finding paths that explain the '{query_rel_name}' relationship.\n")

        for out_i, b in enumerate(idxs, 1):
            path_segments = [name_e(self.visited_entities[b, 0])]

            # Steps = number of recorded relation hops (cannot exceed path_len-1)
            steps = min(rels.shape[0], self.visited_entities.shape[1] - 1)

            for t in range(steps):
                rid = rels[t, b] if rels.size else 2.0
                # Stop if this step is marked as padding / rollout finished
                if rid == 2.0:
                    break
                ent_id = self.visited_entities[b, t + 1]
                path_segments.append(f"--[{name_r(rid)}]--> {name_e(ent_id)}")

            lines.append(f"Path {out_i} [id={b}]: " + " ".join(path_segments))

        return "\n".join(lines)

    def get_scores_AgenticAI(self, keep_idxs):
        """
        Sync micro-batching: score ALL eligible paths by splitting into batches,
        sleeping briefly between requests, and retrying on failures.
        Returns a list of dicts (len == len(keep_idxs)) in the same order.
        """
        if not self.agentic_ai_enabled or _call_llm is None or not keep_idxs:
            print("[INFO] Agentic AI disabled or client/persona missing; skipping LLM scoring.")
            return None

        BATCH_SIZE = int(os.getenv("AGENTIC_BATCH_SIZE", "50"))
        SLEEP_BETWEEN = float(os.getenv("AGENTIC_SLEEP_BETWEEN", "0.4"))  # 0.3–0.5s
        MAX_RETRIES = int(os.getenv("AGENTIC_MAX_RETRIES", "4"))

        import time, random, json

        def _score_batch(batch_idxs, start_num):
            """One sync call with small retries + robust JSON extraction."""
            paths_text = self._build_paths_text(keep_idxs=batch_idxs)

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
                    messages = [{"role": "user", "content": prompt}]
                    resp, raw = _call_llm(messages, temperature=1.0)

                    if resp == "[]":
                        print(f"[WARN] LLM returned empty array. Raw response: {raw}")
                        raise ValueError("LLM returned empty array")

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
        
        # Group by example id so each prompt has a single (start, end) context
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
    
    def get_reward_agenticAI(self, wV=1/3, wC=1/3, wR=1/3, metric_only=False):
        # metric_only=True: skip the LLM call entirely. Used in test() so the
        # metric (rewards>0) never blocks on the LLM. The actual LLM scoring
        # of JSON-bound paths is then done by score_paths_for_json() in the
        # same batch (called from trainer.test()).
        training_step = getattr(Episode, '_training_step', 0)

        base = self.get_reward_ic_based()
        self._cache_ic_summaries()
        B = base.shape[0]

        if (not self.agentic_ai_enabled) or (_call_llm is None) or (not self.persona_text):
            # Keep logging arrays coherent so the rest of the pipeline/test logger works
            print("[INFO] Agentic AI disabled or client/persona missing; using IC-based rewards only.")
            self.agentic_scores = base.astype(np.float32)
            self.llm_dimensions = {}
            self.reward_kind = np.array(['ic_only'] * B, dtype=object)
            return base

        override = getattr(Episode, '_test_threshold_override', None)
        threshold = override if override is not None else threshold_for_step(training_step)

        # Three tiers of paths (boundaries match the paper: medium starts at 0.50)
        #   low:    0  < base <  0.5  → 0.10 fixed
        #   medium: 0.5 <= base <= threshold → 0.25 fixed
        #   high:   base > threshold → IC + LLM blend
        # low extends down to base > 0 so every successful rollout gets at
        # least 0.1 (matches the test metric which counts all successes).
        high_ic_idxs = np.where(base > threshold)[0].tolist()
        medium_ic_idxs = np.where((base >= 0.5) & (base <= threshold))[0].tolist()
        low_ic_idxs = np.where((base > 0) & (base < 0.5))[0].tolist()

        if metric_only:
            # Test-time path: no LLM here. Each correct path's agentic_score
            # defaults to its IC value; score_paths_for_json() then overwrites
            # the high-IC subset with the IC+LLM blend.
            out = base.astype(np.float32).copy()
            self.agentic_scores = out.copy()
            self.llm_dimensions = {}
            self.reward_kind = np.array(['none'] * B, dtype=object)
            self.reward_kind[(out > 0) & (out <= threshold)] = 'low_or_medium'
            self.reward_kind[out > threshold] = 'high_ic'
            return out

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

        self.agentic_scores = np.zeros((B,), dtype=np.float32)
        self.llm_dimensions = {}
        self.reward_kind = np.array(['none']*B, dtype=object)

        for idx in low_ic_idxs:
            out[idx] = 0.1
            self.agentic_scores[idx] = 0.10
            self.reward_kind[idx] = 'low_fixed'
        for idx in medium_ic_idxs:
            out[idx] = 0.25
            self.agentic_scores[idx] = 0.25
            self.reward_kind[idx] = 'medium_fixed'

        llm_rewards=[]
        if high_ic_idxs:
            scores_list = self.get_scores_AgenticAI(high_ic_idxs)
            if scores_list is None:
                # API failed: give high-IC paths a default
                for idx in high_ic_idxs:
                    out[idx] = 0.30
                    self.agentic_scores[idx] = 0.30
                    self.reward_kind[idx] = 'high_default'

            else:
                # Blend IC + LLM so neither dominates
                ALPHA = 0.5  # weight for IC; (1-ALPHA) for LLM
                for idx, scores in zip(high_ic_idxs, scores_list):
                    sv = float(scores["validity"])
                    comp_raw = float(scores["completeness"])
                    rel = float(scores["relevance"])

                    comp_conv = COMPLETENESS_MAP[int(comp_raw)]
                    raw = wV * sv + wC * comp_conv + wR * rel

                    llm_norm = (raw - 1.0) / 4.0
                    llm_norm = float(max(0.0, min(1.0, llm_norm)))

                    # Blend: IC ensures structural quality, LLM adds persona judgement
                    ic_val = float(self.ic_mean[idx])
                    final_blend = ALPHA * ic_val + (1 - ALPHA) * llm_norm

                    out[idx] = final_blend
                    self.agentic_scores[idx] = final_blend
                    self.reward_kind[idx] = 'llm+ic'

                    self.llm_dimensions[idx] = {
                        'validity': sv,
                        'completeness_conv': comp_conv,
                        'relevance': rel,
                        'ic_mean': ic_val,
                        'llm_score': llm_norm,
                    }

                    llm_rewards.append(final_blend)

                if llm_rewards:
                    s = (f"  LLM rewards: min={min(llm_rewards):.3f}, "
                        f"avg={np.mean(llm_rewards):.3f}, max={max(llm_rewards):.3f}")
                    print(s)
                    _env_logger.info(s)

        positive_rewards = out[out > 0]
        if len(positive_rewards) > 0:
            s = (f"[TOTAL] {len(positive_rewards)} paths rewarded "
            f"(avg={np.mean(positive_rewards):.3f})")
            print(s)
            _env_logger.info(s)

        # Bonus: machine-friendly JSON line for analysis (JSONL)
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


    def score_paths_for_json(self, indices, wV=1/3, wC=1/3, wR=1/3):
        """Call the persona LLM only on the given rollout indices and overwrite
        their entries in self.agentic_scores / self.llm_dimensions with the
        IC+LLM blend. Used in test() after we know which paths will actually be
        written to JSON, so we don't pay the LLM cost during the metric path.

        Safe to call when persona/agentic is disabled (no-op).
        Falls back silently to the existing IC values on LLM failure.
        """
        if not indices:
            return
        if (not self.agentic_ai_enabled) or (_call_llm is None) or (not self.persona_text):
            return

        self._cache_ic_summaries()
        if getattr(self, 'agentic_scores', None) is None:
            B = self.batch_size * self.num_rollouts
            self.agentic_scores = np.zeros((B,), dtype=np.float32)
        if not hasattr(self, 'llm_dimensions') or self.llm_dimensions is None:
            self.llm_dimensions = {}

        scores_list = self.get_scores_AgenticAI(list(indices))
        if scores_list is None:
            # Whole-batch failure: keep the IC values that get_reward_agenticAI
            # (metric_only) already wrote. Better than overwriting with a default.
            return

        ALPHA = 0.5
        for idx, scores in zip(indices, scores_list):
            try:
                sv = float(scores["validity"])
                comp_raw = float(scores["completeness"])
                rel = float(scores["relevance"])
            except Exception:
                continue
            comp_conv = COMPLETENESS_MAP[int(comp_raw)]
            raw = wV * sv + wC * comp_conv + wR * rel
            llm_norm = (raw - 1.0) / 4.0
            llm_norm = float(max(0.0, min(1.0, llm_norm)))
            ic_val = float(self.ic_mean[idx])
            final_blend = ALPHA * ic_val + (1 - ALPHA) * llm_norm
            self.agentic_scores[idx] = final_blend
            self.llm_dimensions[idx] = {
                'validity': sv,
                'completeness_conv': comp_conv,
                'relevance': rel,
                'ic_mean': ic_val,
                'llm_score': llm_norm,
            }



    def get_reward_ic_based(self):
        """
        Calculate reward based on the positive reward and the average IC weight.
        Uses 2.0 as a sentinel for padding (ignored in the mean).
        """
        self._recompute_from_path()
        weights_array = np.array(self.recomputed_ic_per_step)  # [T, B]

        mask_2 = (weights_array == 2.0)
        w = weights_array.copy()
        w[mask_2] = np.nan
        average_ic = np.nanmean(w, axis=0)  # [B]
        size = np.sum(~mask_2, axis=0)       # [B]

        success_mask = (self.current_entities == self.end_entities)

        if self.IC_reward:
            positive_part = self.positive_reward * average_ic
        else:
            positive_part = self.positive_reward
            print('[INFO] IC_reward disabled; using flat positive reward without IC scaling.')

        condlist   = [success_mask, ~success_mask]
        choicelist = [positive_part, self.negative_reward]
        final_reward = np.select(condlist, choicelist)
        return final_reward


    def __call__(self, action):
        if self.early_stopping:
            self.current_hop += 1
            bsz = self.no_examples * self.num_rollouts

            chosen_ents = self.state['next_entities'][np.arange(bsz), action]
            self.current_entities = chosen_ents

            chosen_rels = self.state['next_relations'][np.arange(bsz), action]
            chosen_weights = self.state['weights'][np.arange(bsz), action]

            # Any rollout that reached end_entity gets marked done
            newly_done = (chosen_ents == self.end_entities)
            prev_done  = self.done_mask.copy()

            # Pad only rollouts that were already done before this step
            chosen_weights[prev_done] = 2.0
            chosen_rels[prev_done]    = 2.0

            # Append real weight for the last hop of newly completed rollouts
            self.weight_history.append(chosen_weights)
            self.relation_history.append(chosen_rels)

            self.done_mask = np.logical_or(self.done_mask, newly_done)

            new_visited = np.zeros((bsz, self.visited_entities.shape[1] + 1), dtype=np.int32)
            for i in range(bsz):
                new_visited[i, :self.visited_entities.shape[1]] = self.visited_entities[i]
                new_visited[i, -1] = chosen_ents[i]
            self.visited_entities = new_visited

            # Still needed for rollouts not yet done
            next_actions, next_weights = self.grapher.return_next_actions(
                self.current_entities,
                self.start_entities,
                self.query_relation,
                self.end_entities,
                self.all_answers,
                (self.current_hop == self.path_len - 1),
                self.num_rollouts,
                self.visited_entities,
                self.prevent_cycles

            )

            self.state['next_relations']  = next_actions[:, :, 1]
            self.state['next_entities']   = next_actions[:, :, 0]
            self.state['current_entities'] = self.current_entities
            self.state['weights'] = next_weights

            return self.state


        else:
            self.current_hop += 1
            bsz = self.no_examples * self.num_rollouts

            self.current_entities = self.state['next_entities'][np.arange(bsz), action]

            chosen_rels = self.state['next_relations'][np.arange(bsz), action]
            self.relation_history.append(chosen_rels)

            self.weight_history.append(self.state['weights'][np.arange(bsz), action])

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
            self.state['weights'] = next_weights
            self.state['visited_entities'] = self.visited_entities

            return self.state


class env(object):
    def __init__(self, params, mode='train'):
        self.persona_path = params['persona_path']
        self.agentic_ai_enabled = params['agentic_ai_enabled']
        self.llm_api = params.get('llm_api', False)
        self.llm_model = params.get('llm_model', 'qwen')
        self.local_model = params.get('local_model', 'Qwen/Qwen3.5-9B')
        self.IC_reward = params['IC_reward']
        self.adjust_factor = params['IC_importance']
        self.early_stopping = params['early_stopping']
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.prevent_cycles = params['prevent_cycles']
        self.mode = mode
        # Optional separate env log file path, e.g. ".../env_metrics.log"
        self.env_log_file = params.get('env_log_file')
        configure_env_logger(self.env_log_file)
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
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'],
                                             edges_weight=params['edges_weight'],
                                             labels_dir=params['data_input_dir']
                                             )

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher, self.IC_reward, self.adjust_factor, self.early_stopping, self.prevent_cycles, self.persona_path, self.agentic_ai_enabled, self.llm_api, self.llm_model, self.local_model
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
