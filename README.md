# REx_PyTorch

This repository includes:
-  The pytorch implementation for the REx [paper](https://www.ijcai.org/proceedings/2025/0515.pdf). The original implementation of REx in tensorflow can be found [here](https://github.com/liseda-lab/REx).
-   This implementation already includes a new approach called Adaptive REx (more details soon), but to use the original REX, just choose the `neutral_evaluator` files inside configs. 

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
| `Qwen/Qwen3-1.7B` | ~3.4 GB | ~5 GB | Quick smoke tests. Too small to use as a scoring judge |
| `Qwen/Qwen3-4B` | ~8 GB | ~10 GB | External rerank judge (validated on oregano DT personas) |
| `Qwen/Qwen3.5-9B` | ~18 GB | ~20 GB | Training / in-loop reward signal (~32 GB VRAM) |

When `viz_mode=1`, the local LLM is forced to `Qwen/Qwen3-4B` regardless of `--local_model`, and only the final explanation JSON is saved (no metrics or logs). Useful for generating a handful of explanations on a light GPU / CPU.

No account or token is needed; models are open source. Training was done on an RTX 5090. API mode runs anywhere, even without a GPU.

### Test-time scoring modes (`--agentic_ai_enabled=1`)

Two independent flags. Defaults preserve the original in-loop behavior — existing configs are unaffected.

| `--no_llm_rerank` | `--external_rerank` | Behavior |
|:---:|:---:|---|
| 0 (default) | 0 (default) | In-loop LLM scoring during test (original behavior) |
| 1 | 0 | No LLM at all; paths.json with `final_score = ic_mean` |
| 1 | 1 | Fast test, then post-hoc batched rerank — recommended for very slow sets like oregano DT personas |

External-rerank knobs (only used when `--external_rerank=1`):
- `--rerank_alpha` (default `0.5`) — IC weight in `final_score = alpha*ic_mean + (1-alpha)*llm_norm`
- `RERANK_CHECKPOINT_EVERY` (default `25`) — atomic, resumable JSON checkpoints every N batches
- `RERANK_FALLBACK_STRATEGY` (default `mean`) — `mean` or `median` of successful scores fills in LLM-failed paths post-run

**Validated:** training with `Qwen3.5-9B`, external rerank with `Qwen3-4B`. Don't use `Qwen3-1.7B` as a judge — it loses calibration on the 1-5 scale and drops items in batched JSON.

**IC threshold:** the 0.65 floor only applies when `--external_rerank=1`. In-loop scoring uses the curriculum threshold at `best_step` (faithful to training; can be slow if best_step is early). To restore the previous hard-coded floor for an in-loop config, set both flags to 1.

The scorer can also run standalone against any saved paths.json — `uv run python code/model/score_external.py <paths.json> <persona.txt> [threshold] [alpha] [test_data.txt]`.


### Authors
- __Diogo Venes__
- __Susana Nunes__
- __Catia Pesquita__


For any comments or help needed, please send an email to: scnunes@ciencias.ulisboa.pt

