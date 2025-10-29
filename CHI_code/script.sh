#!/bin/bash
#SBATCH --job-name=rextorch
#SBATCH --output=rextorch.out
#SBATCH --partition=tier2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=liseda-01

uv run bash run.sh configs/hetionet_dr.sh