export D=5
export FOLDER=data/${D}_by_${D}_mult
export MODEL=gpt2
export EPOCHS=40
export LR=5e-4
export BSZ=32
export ACCUMULATE=1
export SEED=13
export BASE_DATE=$(date +%Y%m%d_%H%M%S)

# Define arrays for the flag combinations
REMOVE_COT_OPTIONS=("" "--remove_cot")
REINIT_WEIGHTS_OPTIONS=("" "--reinitialize_weights")

# Loop through all combinations
for remove_cot in "${REMOVE_COT_OPTIONS[@]}"; do
    for reinit_weights in "${REINIT_WEIGHTS_OPTIONS[@]}"; do
        # Create experiment identifier
        EXP_ID=""
        if [[ -n "$remove_cot" ]]; then
            EXP_ID="${EXP_ID}_remove_cot"
        fi
        if [[ -n "$reinit_weights" ]]; then
            EXP_ID="${EXP_ID}_reinit_weights"
        fi
        if [[ -z "$EXP_ID" ]]; then
            EXP_ID="_baseline"
        fi
        
        # Set unique save directories for this experiment
        export SAVE=results/${D}_by_${D}_mult/gpt2_${BASE_DATE}${EXP_ID}
        export SAVE_CKPTS=results/${D}_by_${D}_mult/gpt2_${BASE_DATE}${EXP_ID}/checkpoints
        mkdir -p $SAVE
        
        echo "Starting experiment: ${EXP_ID}"
        echo "Save directory: ${SAVE}"
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        torchrun --nproc_per_node=8 --master_addr=localhost --master_port=29500 src/train.py \
            --model ${MODEL} \
            --distributed \
            --bf16 \
            --train_path ${FOLDER}/train.txt \
            --val_path ${FOLDER}/valid.txt \
            --test_path ${FOLDER}/test_bigbench.txt \
            --max_size -1 \
            ${remove_cot} \
            ${reinit_weights} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --batch_size ${BSZ} \
            --accumulate ${ACCUMULATE} \
            --seed ${SEED} \
            --reset_optimizer \
            --save_model ${SAVE_CKPTS} \
            --save_config ${SAVE} \
            > ${SAVE}/log.train 2>&1
        
        echo "Completed experiment: ${EXP_ID}"
        echo "Log saved to: ${SAVE}/log.train"
        echo ""
    done
done

echo "All experiments completed!" 