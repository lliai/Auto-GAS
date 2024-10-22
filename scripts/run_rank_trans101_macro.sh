#!/bin/bash

SEED=${1}
echo $SEED
SS_NAME='macro'

LOG_PATH=./logs/rank/${SS_NAME}/${SEED}

mkdir -p $LOG_PATH

# SS_NAME='macro'
# ZC_TYPE='gaswot'

# CUDA_VISIBLE_DEVICES=0 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'l1'  > ${LOG_PATH}/log_trans101_l1_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'l2'  > ${LOG_PATH}/log_trans101_l2_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'ssim' > ${LOG_PATH}/log_trans101_ssim_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'ms-ssim' > ${LOG_PATH}/log_trans101_ms-ssim_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &

# 'nwot' 'fisher' 'zen' 'snip' 'flops' 'params' 'zico' 'grad_norm'
for ZC_TYPE in 'synflow_bn' 'params' 'nwot'
    do
        CURRENT_TIME=$(date +'%Y-%m-%d_%H:%M:%S')
        CUDA_VISIBLE_DEVICES=0 python exps/run_rank_trans101.py \
                                                    --sample_num 200 \
                                                    --ss_type ${SS_NAME} \
                                                    --seed ${SEED} \
                                                    --zc_type ${ZC_TYPE}  > ${LOG_PATH}/log_trans101_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1
    done
