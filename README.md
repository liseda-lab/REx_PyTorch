# REx_PyTorch

This repository includes:
-  The pytorch implementation for the REx [paper](https://www.ijcai.org/proceedings/2025/0515.pdf). The original implementation of REx in tensorflow can be found [here](https://github.com/liseda-lab/REx).
-   This implementation already includes a new approach called Adaptive REx with an arxiv version of the paper [here](https://arxiv.org/abs/2603.21846), but to use the original REX, just choose the `neutral_evaluator` files inside configs. 

## Guide to run the system

### Prerequisites
- UV installed on your machine. Any necessary dependencies will be automatically installed when you run the approach for the first time. For more information on UV, please check the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Running
Run the following command inside the main directory of the project:

```sh
uv run bash run.sh configs/{dataset}/{task}/{persona}
```

Where `{dataset}` is the name of the dataset you would like to run the approach on, and `{task}`is drug_repurposing or drug_target (interaction). `{persona}` is an extension of REx for adaptive explanations, for the original implementation of REx always choose `neutral_evaluator`.  Take this example: 

```sh
uv run bash run.sh configs/hetionet/drug_repurposing/neutral_evaluator.sh
```

### Datasets 
Datasets should have the following files:
```
dataset
    ├── graph.txt
    ├── dev.txt
    ├── test.txt
    ├── train.txt
    ├── graph_labels.tsv
    ├── edge_labels.tsv
    └── clustered_IC_classes_edgeType.json
    └── vocab
        └── entity_vocab.json
        └── relation_vocab.json
```

Where:
- `graph.txt` contains all triples of the KG except for `dev.txt`, `test.txt`.
- `dev.txt` contains all validation triples.
- `test.txt` contains all test triples.
- `train.txt` contains all train triples.
- `graph_labels.tsv` contains human-readable labels for nodes.
- `edge_labels.tsv` contains human-readable labels for edges.
- `clustered_IC_classes_edgeType.json` contains the IC scores for each edge types of the graph. It is a dictionary where the keys are the edge types and the values are dictionaries with the IC scores for each class.
- `vocab/entity_vocab.json` contains the vocabulary for the entities.
- `vocab/relation_vocab.json` contains the vocabulary for the relations.
- The vocab files are created by using the `create_vocab.py` file.


**Note1**: The existing datasets have the `graph.txt` file divided into smaller parts (`graph_part*.txt`) due to GitHub file size limits. The system **automatically assembles** `graph.txt` from these parts on the first run, meaning no manual steps are needed.

**Note2**: The adaptive version of REx uses a large language model for persona-shaped scoring. There are three modes controlled by `--llm_api` and `--llm_model`:

| Mode | Flag | Model | Requirement |
|------|------|-------|-------------|
| Local (default) | `--llm_api 0` | Set by `--local_model` | High performance GPU (~32GB of VRAM) necessary |
| Qwen API | `--llm_api 1 --llm_model qwen` | Qwen via HuggingFace | HF API key in `.env` |
| GPT API | `--llm_api 1 --llm_model gpt` | GPT via OpenAI | OpenAI key in `.env` |

For local mode, the `--local_model` parameter controls which model is loaded (default: `Qwen/Qwen3.5-9B`). The model weights are automatically downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`. It but can take several minutes depending on model size and connection speed:

| Model | Download Size | RAM/VRAM | Recommended for |
|-------|--------------|----------|-----------------|
| `Qwen/Qwen3-1.7B` | ~3.4 GB | ~5 GB | Quick smoke tests on CPU. Not recommended as a scoring judge — too small for reliable persona discrimination on a 1-5 scale and tends to drop items in batched JSON output |
| `Qwen/Qwen3-4B` | ~8 GB | ~10 GB | Testing on light GPU; **recommended judge for the external rerank** — validated end-to-end on oregano drug-target with mechanistic and insight personas |
| `Qwen/Qwen3.5-9B` | ~18 GB | ~20 GB | Training (in-loop reward signal); needs a powerful GPU with around 32 GB of VRAM. Used for our reported training runs |

When `viz_mode=1`, the local LLM is **forced to `Qwen/Qwen3-4B`** (overriding `--local_model`, even if a config sets a larger model), and only the final explanation JSON is saved — no metrics, no scores, no logs. This is useful for generating a handful of explanations at a time, or for running on a light GPU / CPU. Not advised for training.

No account or token is needed as the models are open source and download freely. Training was done using an RTX 5090 GPU. For API-based LLM calls, execution can be achieved on any system, even without a GPU.

### Test-time scoring modes

When `--agentic_ai_enabled=1`, two flags control how/when the persona LLM scores paths during/after the test phase. They are independent and the defaults reproduce the original in-loop behavior, so existing configs are unaffected.

| `--no_llm_rerank` | `--external_rerank` | What happens during test() | Use case |
|:---:|:---:|---|---|
| `0` (default) | `0` (default) | Each batch calls the LLM inline; paths.json is written once at the end with the IC+LLM blend already applied | Default behavior. Single-shot runs, backward-compatible |
| `1` | `0` | Test loop skips all LLM calls. paths.json has `final_score = ic_mean` only. No rerank, in-loop or external | Pure performance / metrics only — when persona scoring isn't needed |
| `1` | `1` | Test loop skips all LLM calls. After test() completes, a post-test batched rerank scores high-IC paths cross-pair, with a possibly lighter `--local_model`, then re-sorts each pair | Slow runs (long test sets, big personas like oregano DT). Lets you swap personas/models without re-running the test loop |

The external rerank exposes a few extra knobs:

| Flag / env var | Default | Effect |
|----------------|---------|--------|
| `--rerank_alpha` | `0.5` | Weight for IC in the blend: `final_score = alpha*ic_mean + (1-alpha)*llm_norm` |
| `RERANK_CHECKPOINT_EVERY` | `25` | After every N batches the scorer atomically rewrites paths.json. The pass is resumable: paths with `agentic_score` already set are skipped on resume |
| `RERANK_FALLBACK_STRATEGY` | `mean` | When the LLM returns unparseable JSON for a batch, those paths are filled in after the run with the `mean` (default) or `median` of the actual successful agentic_scores — so failed paths don't unfairly outrank LLM-judged-mediocre ones |
| `RERANK_BACKEND` | `local` | Set to `ollama` and provide `RERANK_OLLAMA_MODEL=<tag>` to use Ollama's OpenAI-compatible endpoint instead of the in-process `--local_model` |

**Models we validated this with:** training and in-loop scoring with `Qwen/Qwen3.5-9B`; external rerank with `Qwen/Qwen3-4B`. Both run locally (`--llm_api 0`). Smaller judges (e.g. `Qwen/Qwen3-1.7B`) are **not** recommended — they collapse calibration on a 1-5 scale and drop items in 50-object JSON batches, so the LLM contribution to `final_score` becomes near-constant and the rerank loses its signal.

**Test-time IC threshold behavior (important):** the threshold that decides which paths cross into "high_ic" (and get LLM-scored) depends on the mode:

* **In-loop scoring** (`--external_rerank=0`, the default): test() uses the model's actual training threshold at `best_step` — the same value the policy was rewarded against during training. No floor. Faithful to training, but if `best_step` is early in the curriculum (early stopping at threshold ~0.40), in-loop test will be **noticeably slower** than before because more paths qualify for LLM scoring.
* **External rerank** (`--external_rerank=1`): the threshold is floored at `0.65` (or the curriculum value at `best_step`, whichever is higher). The floor lives here because the rerank is the "make it efficient" mode and the cheap path is what matters for long test sets.

If you previously relied on the old hard-coded 0.65 floor for in-loop test speed, set `external_rerank=1` (and likely `no_llm_rerank=1`) on those configs to get the floor back.

**Recommended cost-sensitive workflow:** train with the larger model so the in-loop reward signal is high quality, then test+rerank with a smaller one for speed. Two separate `uv run` invocations sharing the same checkpoint:

```sh
# Phase 1: train (in-loop LLM scoring needed for the reward signal)
uv run bash run.sh configs/oregano/drug_target/train_mechanistic.sh
#   load_model=0   no_llm_rerank=0   external_rerank=0   local_model="Qwen/Qwen3.5-9B"

# Phase 2: test + external rerank, loading the trained checkpoint
uv run bash run.sh configs/oregano/drug_target/test_mechanistic.sh
#   load_model=1   no_llm_rerank=1   external_rerank=1   local_model="Qwen/Qwen3-4B"
#   model_load_dir=output/.../mechanistic/<run_timestamp>/model/best_ckpt.json
```

You can also run the rerank standalone against any existing paths.json (e.g. to swap personas without re-running the test loop):

```sh
uv run python code/model/score_external.py \
    output/.../<run>/test_beam/paths.json \
    personas/<persona>.txt \
    0.65 0.5 \
    datasets/.../test.txt
```


### Authors
- __Diogo Venes__
- __Susana Nunes__
- __Catia Pesquita__


For any comments or help needed, please send an email to: scnunes@ciencias.ulisboa.pt

