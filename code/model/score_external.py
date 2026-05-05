"""
Post-test LLM rerank for paths.json produced by trainer.test() under
--no_llm_rerank=1. Mirrors the in-loop scoring (Episode.score_paths_for_json /
get_scores_AgenticAI) but operates on the saved JSON instead of live Episode
state, so it can run after the test loop has finished and the GPU is free of
beam-search work.

Behavior:
  - For every path with ic_mean > threshold, send the path to the LLM with the
    persona prompt, parse {validity, completeness, relevance}, compute
    llm_norm = (raw - 1) / 4, and update the path entry:
        score.agentic_score = round(llm_norm, 4)
        score.final_score   = round(alpha*ic_mean + (1-alpha)*llm_norm, 4)
  - For every other path, leave score.final_score = ic_mean (already set by the
    no-LLM test loop) so that ALL paths have a comparable final_score during
    re-sort. This is what keeps a poorly-LLM-scored path from being unfairly
    pushed below an unscored neighbour, and vice-versa.
  - After scoring, sort each pair's `paths` list by final_score (desc).

Checkpointing:
  - Writes the JSON back to disk every CHECKPOINT_EVERY pairs scored (and at
    the end). Safe to interrupt and re-run; on resume any path that already has
    score.agentic_score set is skipped. Atomic via tmp + rename.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Callable, Iterable, List, Optional

# Match the in-loop completeness mapping (1..5 raw -> normalized).
COMPLETENESS_MAP = {1: 1.0, 2: 3.0, 3: 5.0, 4: 3.0, 5: 1.0}

CHECKPOINT_EVERY = int(os.getenv("RERANK_CHECKPOINT_EVERY", "25"))


def _atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _path_to_text(path_entry: dict) -> str:
    """Reconstruct a single readable path line from the JSON path entry.

    Output mirrors the format of Episode._build_paths_text:
        <node0> --[<rel0>]--> <node1> --[<rel1>]--> <node2> ...
    """
    nodes = path_entry.get("nodes", [])
    edges = path_entry.get("edges", [])
    if not nodes:
        return ""
    parts = [str(nodes[0].get("id", ""))]
    for i, e in enumerate(edges):
        rel = str(e.get("label", ""))
        nxt_idx = i + 1
        nxt = str(nodes[nxt_idx].get("id", "")) if nxt_idx < len(nodes) else str(e.get("target", ""))
        parts.append(f"--[{rel}]--> {nxt}")
    return " ".join(parts)


def _load_query_relation_map(test_data_path: Optional[str]) -> dict:
    """Build a (start_id, end_id) -> relation map from test.txt if available."""
    qr_map = {}
    if not test_data_path or not os.path.exists(test_data_path):
        return qr_map
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qr_map[(parts[0], parts[2])] = parts[1]
    return qr_map


def _build_prompt(persona_text: str,
                  paths_block: str,
                  id_list: list) -> str:
    """Cross-pair batch prompt. Each path line carries its own query context
    inline (start --[rel]--> end), so a single LLM call can score paths from
    many different (start, end) pairs at once. This dramatically improves
    amortization vs the per-pair prompt that sat idle on small pairs."""
    return f"""
You are evaluating drug-target/drug-disease explanation paths from the perspective of the following persona:

{persona_text}

You will see {len(id_list)} paths from possibly different drug-target queries. Each path begins with its own query context in parentheses (start --[query_relation]--> end), followed by the actual reasoning chain. Score each path independently against the persona, NOT relative to the others.

Score EACH path on three criteria:
1. Scientific Validity (V): 1-5. Scientific correctness, plausibility, and coherence based on biomedical knowledge.
2. Completeness (C): 1-5 where 3 is ideal. 1 = too simple, 5 = too complex. Reward paths that are sufficiently detailed without overload.
3. Relevance (R): 1-5. Usefulness for understanding why the prediction matters and how it connects to the task.

Paths to evaluate ({len(id_list)} total). Each line has an [id=...]:
{paths_block}

Return ONLY valid JSON with an array of exactly {len(id_list)} objects.
Each object MUST be: {{"id": <int from {id_list}>, "validity": <int>, "completeness": <int>, "relevance": <int>}}.
Use ONLY the ids from this set: {id_list}. Do not invent or omit ids. DOUBLE CHECK BEFORE RETURNING RESULTS.
Do NOT return any text outside the JSON array. Do not return thinking traces or internal monologue.
""".strip()


def _parse_response(resp: str, id_list: list) -> Optional[dict]:
    """Best-effort parse of the LLM response into {id -> {v,c,r}}."""
    if not resp:
        return None
    if "```" in resp:
        parts = resp.split("```")
        resp = "".join(p for p in parts if "[" in p and "]" in p)
    a, b = resp.find("["), resp.rfind("]")
    if a == -1 or b == -1 or b <= a:
        return None
    try:
        data = json.loads(resp[a:b + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "scores" in data:
        data = data["scores"]
    if not isinstance(data, list):
        return None

    def _clamp(x, lo=1, hi=5, default=3):
        try:
            xi = int(round(float(x)))
            return max(lo, min(hi, xi))
        except Exception:
            return default

    out = {}
    valid_ids = set(id_list)
    for obj in data:
        if not isinstance(obj, dict):
            continue
        try:
            oid = int(obj["id"])
        except Exception:
            continue
        if oid not in valid_ids:
            continue
        out[oid] = {
            "validity": _clamp(obj.get("validity", 3)),
            "completeness": _clamp(obj.get("completeness", 3)),
            "relevance": _clamp(obj.get("relevance", 3)),
        }
    return out


def score_paths_external(json_path: str,
                         persona_path: str,
                         threshold: float,
                         alpha: float,
                         call_llm: Callable[[list, float], tuple],
                         test_data_path: Optional[str] = None,
                         batch_size: int = 50,
                         sleep_between: float = 0.4,
                         max_retries: int = 4,
                         logger=None) -> None:
    """Rerank paths.json in place. `call_llm(messages, temperature)` must
    return `(parsed_response_str, raw_response_str)` like environment._call_llm."""

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)

    if not os.path.exists(json_path):
        _log(f"[RERANK] paths.json not found: {json_path}")
        return

    with open(persona_path, "r", encoding="utf-8") as f:
        persona_text = f.read().strip()

    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    pairs = doc.get("pairs", [])
    if not pairs:
        _log(f"[RERANK] No pairs in {json_path}")
        return

    qr_map = _load_query_relation_map(test_data_path)

    # Build a flat list of work units across ALL pairs. Each unit is everything
    # the prompt builder + applier needs: (pair_idx, path_idx, ic_mean,
    # start_label, end_label, query_rel). This lets us batch in groups of
    # `batch_size` paths at a time without being capped by per-pair path count.
    work = []
    skipped_resume = 0
    for pi, pair in enumerate(pairs):
        pair_paths = pair.get("paths", [])
        if not pair_paths:
            continue
        # Pair-level context (start/end labels and query relation).
        nodes_first = pair_paths[0].get("nodes", [])
        start_label = nodes_first[0].get("id", "?") if nodes_first else "?"
        end_label = nodes_first[-1].get("id", "?") if nodes_first else "?"
        pair_id = pair.get("id", "")
        query_rel = ""
        if " - " in pair_id and qr_map:
            s_id, e_id = pair_id.split(" - ", 1)
            query_rel = qr_map.get((s_id, e_id), "")
        for ji, p in enumerate(pair_paths):
            score = p.get("score", {})
            ic = float(score.get("ic_mean", 0.0))
            if ic <= threshold:
                continue
            if "agentic_score" in score:
                skipped_resume += 1
                continue
            work.append((pi, ji, ic, start_label, end_label, query_rel))

    total_eligible = sum(
        1 for pair in pairs for p in pair.get("paths", [])
        if float(p.get("score", {}).get("ic_mean", 0.0)) > threshold
    )
    _log(f"[RERANK] {len(pairs)} pairs, {total_eligible} paths > threshold {threshold:.2f}, "
         f"{skipped_resume} already scored, {len(work)} paths to score across pairs")

    if not work:
        _log("[RERANK] Nothing to do; ensuring paths are sorted then exiting.")
        _resort_all(pairs)
        _atomic_write_json(json_path, doc)
        return

    pairs_touched_since_ckpt = set()
    paths_done_total = 0
    batches_done = 0
    n_batches = (len(work) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_start in range(0, len(work), batch_size):
        batch = work[batch_start:batch_start + batch_size]
        id_list = list(range(len(batch)))
        lines = []
        for local_id, (pi, ji, _ic, sl, el, qr) in zip(id_list, batch):
            path_paths = pairs[pi].get("paths", [])
            if ji >= len(path_paths):
                continue
            txt = _path_to_text(path_paths[ji])
            rel_phrase = f"--[{qr}]-->" if qr else "->"
            # Each line: "Path N [id=N]: (start --[rel]--> end) <chain>"
            # The inline (start --[rel]--> end) gives the LLM per-path query context
            # so paths from different (start, end) pairs can mix in one batch.
            lines.append(f"Path {local_id} [id={local_id}]: ({sl} {rel_phrase} {el}) {txt}")
        paths_block = "\n".join(lines)

        prompt = _build_prompt(persona_text, paths_block, id_list)
        messages = [{"role": "user", "content": prompt}]

        parsed = None
        for attempt in range(1, max_retries + 1):
            try:
                resp, _raw = call_llm(messages, 1.0)
                parsed = _parse_response(resp, id_list)
                if parsed is not None and len(parsed) > 0:
                    break
            except Exception as e:
                _log(f"[RERANK] LLM call error (attempt {attempt}): {e}")
            time.sleep(sleep_between * attempt)

        if not parsed:
            # Whole-batch LLM failure: mark these paths as failed (no agentic
            # judgment available). final_score stays at ic_mean for now; the
            # post-rerank fallback step (apply_failure_fallback) computes a
            # data-driven replacement value from the actual LLM-score
            # distribution and re-ranks. We can't pick a fair value here
            # because the distribution is only known after the run completes.
            _log(f"[RERANK] Batch {batches_done+1}/{n_batches}: no usable LLM response, "
                 f"marking {len(batch)} paths as failed (fallback applied post-run)")
            for local_id, (pi, ji, _ic, *_rest) in zip(id_list, batch):
                pair_paths = pairs[pi].get("paths", [])
                if ji < len(pair_paths):
                    p_score = pair_paths[ji].setdefault("score", {})
                    p_score["agentic_score"] = None
                    p_score["llm_dimensions"] = {"failed": True}
                    pairs_touched_since_ckpt.add(pi)
            batches_done += 1
            paths_done_total += len(batch)
            continue

        # Apply scores back to original pair/path.
        for local_id, (pi, ji, ic, *_rest) in zip(id_list, batch):
            pair_paths = pairs[pi].get("paths", [])
            if ji >= len(pair_paths):
                continue
            scores = parsed.get(local_id)
            p_score = pair_paths[ji].setdefault("score", {})
            if scores is None:
                # Per-path miss within an otherwise-parsed batch: same as a
                # whole-batch failure. final_score stays at ic_mean and is
                # rewritten by apply_failure_fallback after the run completes.
                p_score["agentic_score"] = None
                p_score["llm_dimensions"] = {"failed": True}
            else:
                v = float(scores["validity"])
                c_raw = float(scores["completeness"])
                r = float(scores["relevance"])
                c_conv = COMPLETENESS_MAP.get(int(c_raw), 3.0)
                raw_blend = (v + c_conv + r) / 3.0
                llm_norm = max(0.0, min(1.0, (raw_blend - 1.0) / 4.0))
                final = alpha * ic + (1.0 - alpha) * llm_norm
                p_score["agentic_score"] = round(llm_norm, 4)
                p_score["final_score"] = round(final, 4)
                p_score["llm_dimensions"] = {
                    "validity": v,
                    "completeness_conv": c_conv,
                    "relevance": r,
                }
            pairs_touched_since_ckpt.add(pi)

        batches_done += 1
        paths_done_total += len(batch)

        # Checkpoint every CHECKPOINT_EVERY batches (atomic write of full doc).
        if batches_done % CHECKPOINT_EVERY == 0:
            _resort_pairs(pairs, pairs_touched_since_ckpt)
            _atomic_write_json(json_path, doc)
            elapsed = time.time() - t0
            paths_per_min = (paths_done_total / max(elapsed, 1e-9)) * 60.0
            remaining_paths = len(work) - paths_done_total
            eta_min = remaining_paths / max(paths_per_min, 1e-9)
            _log(f"[RERANK] checkpoint: batch {batches_done}/{n_batches} "
                 f"({paths_done_total}/{len(work)} paths, "
                 f"{paths_per_min:.1f} path/min, ETA {eta_min:.1f} min)")
            pairs_touched_since_ckpt.clear()

        time.sleep(sleep_between)

    # Final sort + write.
    _resort_all(pairs)
    _atomic_write_json(json_path, doc)
    elapsed = time.time() - t0
    _log(f"[RERANK] Done. {paths_done_total} paths scored in {batches_done} batches "
         f"in {elapsed/60:.1f} min. Updated {json_path}")


def apply_failure_fallback(json_path: str,
                           alpha: float = 0.5,
                           strategy: str = "mean",
                           logger=None) -> None:
    """Post-rerank pass that fixes paths the LLM failed on.

    Reads paths.json. Computes a fallback agentic_score from the actual
    distribution of successful LLM scores (paths whose llm_dimensions has
    'validity' rather than 'failed'). Applies that value to every failed
    path (agentic_score == None or llm_dimensions == {'failed': True}),
    rewrites final_score = alpha*ic + (1-alpha)*fallback, and re-sorts each
    touched pair's paths.

    Why this is a separate step: at LLM-failure time we don't know the score
    distribution, so a hardcoded constant (e.g. 0.5) can bias failed paths
    above or below their LLM-judged neighbours depending on the dataset.
    Computing the fallback after the run lets us pick a value that's
    statistically neutral relative to the actual scoring behaviour.

    Strategies:
      'mean'   - global mean of successful agentic_scores (default; least
                 biased estimate of "if we'd had data, what would we expect")
      'median' - global median of successful agentic_scores
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)

    if not os.path.exists(json_path):
        _log(f"[FALLBACK] paths.json not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    pairs = doc.get("pairs", [])
    if not pairs:
        return

    successful = []
    failed_units = []  # (pair_idx, path_idx, ic_mean)
    for pi, pair in enumerate(pairs):
        for ji, p in enumerate(pair.get("paths", [])):
            sc = p.get("score", {})
            ag = sc.get("agentic_score", "MISSING")
            dims = sc.get("llm_dimensions", {})
            is_failed = (ag is None) or (isinstance(dims, dict) and dims.get("failed"))
            if is_failed:
                failed_units.append((pi, ji, float(sc.get("ic_mean", 0.0))))
            elif isinstance(dims, dict) and "validity" in dims:
                successful.append(float(ag))

    if not successful:
        _log(f"[FALLBACK] No successful LLM scores found; cannot compute {strategy}. Skipping.")
        return
    if not failed_units:
        _log("[FALLBACK] No failed paths to fix. Skipping.")
        return

    if strategy == "median":
        s = sorted(successful)
        fallback_val = s[len(s) // 2]
    else:  # 'mean' (default)
        fallback_val = sum(successful) / len(successful)
    fallback_val = float(max(0.0, min(1.0, fallback_val)))

    _log(f"[FALLBACK] strategy={strategy}, value={fallback_val:.4f} "
         f"(from {len(successful)} successful scores). Applying to {len(failed_units)} failed paths.")

    pairs_touched = set()
    for pi, ji, ic in failed_units:
        pair_paths = pairs[pi].get("paths", [])
        if ji >= len(pair_paths):
            continue
        sc = pair_paths[ji].setdefault("score", {})
        sc["agentic_score"] = round(fallback_val, 4)
        sc["final_score"] = round(alpha * ic + (1.0 - alpha) * fallback_val, 4)
        existing_dims = sc.get("llm_dimensions", {})
        if isinstance(existing_dims, dict):
            existing_dims["fallback_strategy"] = strategy
            existing_dims["fallback_value"] = round(fallback_val, 4)
            sc["llm_dimensions"] = existing_dims
        pairs_touched.add(pi)

    for pi in pairs_touched:
        paths = pairs[pi].get("paths", [])
        paths.sort(key=lambda x: x.get("score", {}).get("final_score", 0.0), reverse=True)

    _atomic_write_json(json_path, doc)
    _log(f"[FALLBACK] Updated {len(failed_units)} paths across {len(pairs_touched)} pairs. "
         f"Wrote {json_path}")


def _resort_pairs(pairs: list, pair_idxs: Iterable[int]) -> None:
    for pi in pair_idxs:
        pair = pairs[pi]
        paths = pair.get("paths", [])
        if paths:
            paths.sort(key=lambda x: x.get("score", {}).get("final_score", 0.0), reverse=True)


def _resort_all(pairs: list) -> None:
    for pair in pairs:
        paths = pair.get("paths", [])
        if paths:
            paths.sort(key=lambda x: x.get("score", {}).get("final_score", 0.0), reverse=True)


# --- Ollama backend (HTTP, OpenAI-compatible endpoint) ---------------------
# Enabled by setting RERANK_BACKEND=ollama. Set RERANK_OLLAMA_MODEL to the
# tag (e.g. "qwen3:4b-q4_K_M" or "qwen3:1.7b"). Pulls must already be done
# (`ollama pull <tag>`).
def _make_ollama_caller(model: str, host: str = "http://localhost:11434"):
    import httpx
    client = httpx.Client(timeout=120.0)
    url = f"{host.rstrip('/')}/v1/chat/completions"

    def _call(messages, temperature):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False,
        }
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return content, content

    return _call


def get_call_llm():
    """Pick a backend based on RERANK_BACKEND env. Defaults to environment._call_llm."""
    backend = os.getenv("RERANK_BACKEND", "local").lower()
    if backend == "ollama":
        model = os.getenv("RERANK_OLLAMA_MODEL", "qwen3:4b-q4_K_M")
        host = os.getenv("RERANK_OLLAMA_HOST", "http://localhost:11434")
        return _make_ollama_caller(model, host)
    # Default: reuse the in-process Qwen from the same module trainer.py loaded.
    # Must import via the package path (code.model.environment) — `from environment
    # import ...` would load a second copy with default globals (Qwen3.5-9B), losing
    # the --local_model override that Episode.__init__ wrote into the original module.
    from code.model.environment import _call_llm  # type: ignore
    return _call_llm


# Standalone CLI usage: python score_external.py <json_path> <persona_path> [threshold] [alpha] [test_data]
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: score_external.py <paths.json> <persona.txt> [threshold=0.65] [alpha=0.5] [test_data.txt]")
        print("env: RERANK_BACKEND={local|ollama}, RERANK_OLLAMA_MODEL=<tag>, RERANK_CHECKPOINT_EVERY=<int>")
        sys.exit(1)
    json_p = sys.argv[1]
    persona_p = sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65
    alp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    test_d = sys.argv[5] if len(sys.argv) > 5 else None

    caller = get_call_llm()
    if caller is None:
        print("[RERANK] No LLM backend available. Aborting.")
        sys.exit(2)
    score_paths_external(json_p, persona_p, thr, alp, caller, test_data_path=test_d)
