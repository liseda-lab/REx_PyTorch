# REx_PyTorch
This implementation is a work in process for an extension of REx with pytorch. This README will be updated as the code is further developed. The title will also change. 

The original implementation of REx can be found here: https://github.com/liseda-lab/REx. CHI code is not yet public. 

The main differences from CHI with the original REx implementation are:
- early stopping with patience to avoid iterations without improvement
- reward is different to accept the personas and keep in memory the reward. 
- we use OPENAI API to generate the rewards but it can be easily changed to use other LLMs

Folder with CHI code and REx code will be deleted in the future when the new code is stable.


## guide to use both codes (to be deleted when the code is more stable):

### Prerequisites
- Docker installed on your machine

### Building the Docker Image
To build the Docker image, use the provided `Dockerfile`. Run the following command in the root directory of the project:

```sh
docker build -t REX-image .
```
Start a container from the built image:

```sh
docker run --gpus all -d --name rex_space -v $(pwd):/REx REX-image tail -f /dev/null

```

Create an interactive shell in the container to run commands:

```sh
docker exec -it rex_space bash
```

### Running
Once inside the container, to run it, you will need to run the following command:

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