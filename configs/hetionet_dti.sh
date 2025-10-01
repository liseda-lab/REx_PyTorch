#!/usr/bin/env bash
data_input_dir="datasets/hetionet_dt/"
vocab_dir="datasets/hetionet_dt/vocab"
total_iterations=200
path_length=3
hidden_size=32
embedding_size=32
batch_size=128
learning_rate=0.0006
beta=0.05
num_rollouts=30
LSTM_layers=2
base_output_dir="output_hetionetDT/hetionet_leo_DT_bestThreshold/"
Lambda=0.02
eval_every=10
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
max_num_actions=400
size_flexibility=1
weighted_reward=1
load_model=0
model_load_dir=None
tensorboard_dir="tensorboard/hetionet_leo_DT/"

agentic_ai_enabled=1
persona_path="personas/leo_insight.txt"
