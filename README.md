# REx_PyTorch
Extension of REx with port for PyTorch. The original implementation of REx can be found [here] (https://github.com/liseda-lab/REx). 

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
uv run bash run.sh configs/{dataset}
```

Where `{dataset}` is the name of the dataset you would like to run the approach on. 

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

**Note2**: The CHI code can be used for normal REx too. In the config files there is a parameter called `agentic_ai_enabled`. This parameter is used to enable or disable the use of agentic AI for the CHI paper and can be turned off. 


### Authors
- __Diogo Venes__
- __Susana Nunes__
- __Catia Pesquita__



For any comments or help needed, please send an email to: to be added (Diogo Venes)

## Acknowledgments
To be added.
