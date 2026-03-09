#!/bin/bash
#SBATCH --job-name=rextorch
#SBATCH --output=rextorchQwen359B.out
#SBATCH --partition=gpu_un
#SBATCH --nodelist=liseda-05
#SBATCH --gres=gpu:rtx5090:1

uv run bash run.sh configs/hetionet_dr.sh
