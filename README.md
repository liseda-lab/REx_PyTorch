# REx_PyTorch
REx with port for PyTorch. The original implementation of REx can be found [here](https://github.com/liseda-lab/REx). This REx implementation already includes a new approach called Adaptive REx, but to use the original REX, just choose the non_evaluator folder. More details will be here soon.

## Guide to run the system

### Prerequisites
- UV installed on your machine

### Installing dependencies
To install the necessary dependencies, provided UV is installed, simply run the following command: 

```sh
uv sync
```


### Running
Once inside the folder, to run it, you will need to run the following command:

```sh
uv run bash run.sh configs/{dataset}/{task}/{persona}
```

Where `{dataset}` is the name of the dataset you would like to run the approach on, and `{task}`is drug_repurposing or drug_target (interaction). `{persona}` is an extension of REx for adaptive explanations, for the original implementation of REx choose non_evaluator. 

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


**Note1**: The existing dataset has the graph.txt file divided into one or more files. Just run the following command in the dataset directory:

```sh
cat graph_part*.txt > graph.txt
```

**Note2**: Current code downloads and uses a local version of the Qwen3.5 9B large language model via the HF Transformers library for the adaptive version of REx. Testing was done using an RTX 5090 GPU. GPUs with less VRAM will not be able to load both the LLM and the REx system onto GPU. If using an API for LLM calls, execution can be achieved on any system, even without a GPU.  

### Authors
- __Diogo Venes__
- __Susana Nunes__
- __Catia Pesquita__



For any comments or help needed, please send an email to: scnunes@ciencias.ulisboa.pt

## Acknowledgments
This work was supported by FCT through the fellowship 2023.00653.BD, and the LASIGE Research Unit, ref. UID/00408/2025. It was also partially supported by the KATY project (European Union Horizon 2020 grant No. 101017453), and by the CancerScan project which received funding from the European Union’s Horizon Europe Research and Innovation Action (EIC Pathfinder Open) under grant agreement No. 101186829. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Innovation Council and SMEs Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
