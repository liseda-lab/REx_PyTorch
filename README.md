# REx_PyTorch

This is the pytorch implementation for the REx [paper](https://www.ijcai.org/proceedings/2025/0515.pdf). The original implementation of REx in tensorflow can be found [here](https://github.com/liseda-lab/REx). This REx implementation already includes a new approach called Adaptive REx, but to use the original REX, just choose the `neutral_evaluator` files inside configs. More details will be here soon.

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


**Note1**: The existing datasets have the `graph.txt` file divided into smaller parts (`graph_part*.txt`) due to GitHub file size limits. The system **automatically assembles** `graph.txt` from these parts on the first run — no manual step needed.

**Note2**: The adaptive version of REx uses a large language model for persona-shaped scoring. There are three modes controlled by `--llm_api` and `--llm_model`:

| Mode | Flag | Model | Requirement |
|------|------|-------|-------------|
| Local (default) | `--llm_api 0` | Set by `--local_model` | High performance GPU (~RTX 5090) necessary |
| Qwen API | `--llm_api 1 --llm_model qwen` | Qwen via HuggingFace | HF API key in `.env` |
| GPT API | `--llm_api 1 --llm_model gpt` | GPT via OpenAI | OpenAI key in `.env` |

For local mode, the `--local_model` parameter controls which model is loaded (default: `Qwen/Qwen3.5-9B`). The model weights are automatically downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`. It but can take several minutes depending on model size and connection speed:

| Model | Download Size | RAM/VRAM | Recommended for |
|-------|--------------|----------|-----------------|
| `Qwen/Qwen3-1.7B` | ~3.4 GB | ~5 GB | Quick testing on CPU |
| `Qwen/Qwen3-4B` | ~8 GB | ~10 GB | Testing on light GPU |
| `Qwen/Qwen3.5-9B` | ~18 GB | ~20 GB | Training (needs powerful GPU) |

The parameter viz_mode changes the default mode to `Qwen/Qwen3-4B` and only saves a final json with the generated explanations without any scores, metrics or logs.This is useful for testing few explanations at a time and for generating explanations on CPU (not advised for training). 

No account or token is needed — the models are open source and download freely. Training was done using an RTX 5090 GPU. For API-based LLM calls, execution can be achieved on any system, even without a GPU.


### Authors
- __Diogo Venes__
- __Susana Nunes__
- __Catia Pesquita__


For any comments or help needed, please send an email to: scnunes@ciencias.ulisboa.pt

