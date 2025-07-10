#!/bin/bash

# Example distributed analysis script
# This shows how to run attention analysis across multiple GPUs

export EXPERIMENT_PATH="results/5_by_5_mult/gpt2_20250710_095706"
export LAYERS="0,1,2,3,4,5,6,7,8,9,10,11"  # Analyze layers 0-5
export EPOCHS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" #,20,21,22,23,24,25,26,27,28,29"
# export EPOCHS="0,9,19,29,39,49,59,69,79,89,99,109,119,129,139"


echo "Running distributed attention analysis..."
echo "Experiment path: $EXPERIMENT_PATH"
echo "Layers to analyze: $LAYERS"
echo "Epochs to analyze: $EPOCHS"
echo "Configuration will be loaded from: $EXPERIMENT_PATH/args.json"


CUDA_VISIBLE_DEVICES=0,1 python3 src/analysis.py \
    --rpath $EXPERIMENT_PATH \
    --dataset train \
    --layers_to_collect $LAYERS \
    --epochs $EPOCHS \
    --batch_size 64

echo "Analysis complete! Results saved to $EXPERIMENT_PATH/analysis.json" 