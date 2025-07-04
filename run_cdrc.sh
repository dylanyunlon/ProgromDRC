#!/bin/bash
# Activate environment
source cdrc_env/bin/activate

# Set environment variables
export HF_HOME="./cache/huggingface"
export TRANSFORMERS_CACHE="./cache/transformers"
export CUDA_VISIBLE_DEVICES=2  # Adjust as needed

# Run the experiment
python experiment_runner.py --config experiment_config.json --step all
