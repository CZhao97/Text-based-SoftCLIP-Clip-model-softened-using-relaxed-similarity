#!/bin/bash
#SBATCH --job-name=train_soft_model
#SBATCH --time=5-0:0:0  # 5 days

export TRANSFORMERS_CACHE=/w/284/markwang/CSC2516/huggingface/transformer

python3 train_and_test.py --gpu --save_model --save_dir models/bert_vitb_soft.pt

