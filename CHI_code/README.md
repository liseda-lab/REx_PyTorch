# Reflexive-XAI

**Towards Reflexive Explainability: Adapting AI Explanations to Human Expertise**  
- Resources for the anonymous submission for CHI 2026 (#1744) are available here.


---
## Credits
The reflexive approach in this repository builds on the **REx** framework, originally released in the repository [REx](https://github.com/liseda-lab/REx), which accompanied the paper: *Rewarding Explainability in Drug Repurposing with Knowledge Graphs* (Nunes et al., 2025).  

---

## Overview
This repository accompanies our submission, which introduces **reflexive explainability**: an approach to designing AI explanations that adapt to experts’ epistemic stances and decision contexts.  

Most explainable AI (XAI) systems assume a generic user model, overlooking that explanation needs vary by background, task, and interpretive strategy. Our work presents:  
1. **Reflexive explainability** – treating explanation as a bidirectional, context-sensitive process that adapts to users while encouraging reflection.  
2. **Agentic personas** – synthetic profiles derived from expert input and large language models, capturing diverse explanatory needs without requiring individual-level data.  
3. **A reinforcement learning framework** – using a persona-aligned reward function to generate explanations that better match expert reasoning.

We validate our approach through formative studies and a comparative user evaluation in the biomedical domain, showing that reflexive explanations are preferred and align more closely with expert assessments than uniform alternatives.

---

## Formative Studies
Before the comparative evaluation, we conducted two formative studies to understand expert needs and ground our personas:

1. **Study 1 – Survey on Explanation Preferences**  
   - **Tasks/Datasets**: *Drug Repurposing (DR)* and *Drug–Target Interaction (DTI)*, using examples sampled from Hetionet-derived biomedical knowledge graphs.  
   - **Focus**: Identifying heterogeneity in expert preferences for explanation depth, modality (graph vs text), and reliability indicators.  

2. **Study 2 – Persona Grounding**  
   - **Tasks/Datasets**: Same DR and DTI tasks, with domain experts providing qualitative feedback on explanations.  
   - **Focus**: Clustering interpretive strategies into *agentic personas* that capture distinct explanatory stances.  

Insights from these studies guided the synthesis of agentic personas and the design of reflexive explanations.

---

## Designing Agentic Personas and Reflexive Explanations

### Persona Synthesis
We synthesized coherent epistemic stances by combining clustered expert feedback with LLM-based narrative generation, producing personas that encode distinct explanatory preferences without relying on individual-level tracking.  

### Reflexive Explanation Generation
We operationalized reflexivity through a reinforcement learning framework: candidate explanations are filtered by relevance and scored by agentic personas, guiding the generation of explanations that adapt to diverse expert stances.

---

## Evaluation Criteria
We introduce **three evaluation criteria** grounded in philosophy of science:  
- **Relevance** – does the explanation provide meaningful, mechanistically informative content?  
- **Completeness** – does it offer sufficient causal depth without overwhelming complexity?  
- **Validity** – is the explanation biologically plausible and consistent with current knowledge?


## Repository Structure

```text
Reflexive-XAI/
│
├── evaluation/                 # Comparative evaluation and analyses
│   ├── credibility check        # Materials for the credibility check 
│   └── user study results       # Materials for the 22-participant comparative study
│
├── personas/                   # Agentic persona synthesis
│   ├── agentic_personas/       # Generate personas
│   ├── clustering/             # Clusters of interpretive strategies
│   ├── final_personas/         # Final personas (e.g., Elena, Leo)
│   └── verbalization/          # LLM-generated verbalizations 
│
├── reflexive_approach/         # Modified REx extension with reflexive explanainability
│   ├── configs/                # Training configs for RL with persona rewards
│   ├── code/                   # Core code for the reflexive approach
│   └── datasets/               # Dataset files for training and evaluation
│
└── README.md                   # ← You are here


```


---

## Prerequisites
- Docker installed on your machine

## Building the Docker Image
To build the Docker image, use the provided `Dockerfile`. Run the following command in the root directory of the project:

```sh
docker build -t reflex-image .
```
Start a container from the built image:

```sh
docker run --gpus all -d --name reflex_space -v $(pwd):/REx reflex-image tail -f /dev/null

```

Create an interactive shell in the container to run commands:

```sh
docker exec -it reflex_space bash
```

## Running Reflexive Approach
Once inside the container, to run it, you will need to run the following command:

```sh
uv run bash run.sh configs/{dataset}
```

Where `{dataset}` is the name of the dataset you would like to run the approach on. 

## Datasets 
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


**Note**: The existing dataset has the graph.txt file divided into one or more files. Just run the following command in the dataset directory:

```sh
cat graph_part*.txt > graph.txt

```


## Authors
- __Anon_submission__


For any comments or help needed, please send an email to: to be added.

## Acknowledgments
To be added.
