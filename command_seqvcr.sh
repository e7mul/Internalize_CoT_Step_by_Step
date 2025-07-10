#!/bin/bash
export D=5
export FOLDER=data/${D}_by_${D}_mult
export MODEL=gpt2
export EPOCHS=40
export LR=5e-4
export BSZ=32
export ACCUMULATE=1
export SEED=13
export DATE=$(date +%Y%m%d_%H%M%S)
export SAVE=results/${D}_by_${D}_mult/gpt2_seqvcr_pause_${DATE}
export SAVE_CKPTS=results/${D}_by_${D}_mult/gpt2_seqvcr_pause_${DATE}/checkpoints
mkdir -p $SAVE

# Seq-VCR and Pause Token Parameters
export LAMBDA_VAR=1.0
export LAMBDA_COV=0.004
export NUM_PAUSE_TOKENS=2
export PROJECTION_DIM=2048



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --nproc_per_node=8 --master_addr=localhost --master_port=29500 src/train.py \
python src/train.py \
    --model ${MODEL} \
    --distributed \
    --bf16 \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --test_path ${FOLDER}/test_bigbench.txt \
    --max_size 1000 \
    --remove_cot \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --accumulate ${ACCUMULATE} \
    --seed ${SEED} \
    --reset_optimizer \
    --save_model ${SAVE_CKPTS} \
    --save_config ${SAVE} \
    --enable_seq_vcr \
    --enable_pause_tokens \
    --lambda_var ${LAMBDA_VAR} \
    --lambda_cov ${LAMBDA_COV} \
    --num_pause_tokens ${NUM_PAUSE_TOKENS} \
    --projection_dim ${PROJECTION_DIM} \
    > ${SAVE}/log.train 2>&1 