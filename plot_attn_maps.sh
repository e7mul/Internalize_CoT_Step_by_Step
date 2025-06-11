#!/bin/bash

# Example distributed analysis script
# This shows how to run attention analysis across multiple GPUs

export EXPERIMENT_PATH="results/5_by_5_mult/gpt2_20250608_114605_remove_cot"
export LAYERS="0,1,2,3,4,5,6,7,8,9,10,11"  # Analyze layers 0-5
export EPOCHS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39"


echo "Running distributed attention analysis..."
echo "Experiment path: $EXPERIMENT_PATH"
echo "Layers to analyze: $LAYERS"
echo "Epochs to analyze: $EPOCHS"
echo "Configuration will be loaded from: $EXPERIMENT_PATH/args.json"


python3 src/attention_map_plots.py \
    --rpath $EXPERIMENT_PATH \
    --dataset train \
    --layers_to_collect $LAYERS \
    --epochs $EPOCHS \
    --batch_size 64 \
    --num_samples 5 \
    --max_size 1000 \

echo "Analysis complete! Results saved to $EXPERIMENT_PATH/attentions" 