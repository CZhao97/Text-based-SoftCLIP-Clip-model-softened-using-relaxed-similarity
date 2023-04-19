#!/bin/bash
#SBATCH --job-name=test_soft_model

export TRANSFORMERS_CACHE=/w/284/markwang/CSC2516/huggingface/transformer

CUDA_VISIBLE_DEVICES=0 python3 test.py --gpu --model_dir ../models/bert_vitb_hard.pt
