#!/bin/bash

SEED=${1}
echo $SEED
SS_NAME='micro'
LOG_PATH=./logs/rank/${SS_NAME}/${SEED}

mkdir -p $LOG_PATH

# SS_NAME='macro'
# ZC_TYPE='gaswot'

# CUDA_VISIBLE_DEVICES=0 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'l1'  > ${LOG_PATH}/log_trans101_l1_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'l2'  > ${LOG_PATH}/log_trans101_l2_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'ssim' > ${LOG_PATH}/log_trans101_ssim_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python exps/run_rank_trans101.py --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --loss_type 'ms-ssim' > ${LOG_PATH}/log_trans101_ms-ssim_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1 &
#  'bn_score' 'mixup' 'size' 'grad_conflict' 'mgm' 'ntk' 'nst' 'sp1' 'sp2' 'at' 'pdist' 'cc' 'orm'

for ZC_TYPE in 'nwot' 'fisher' 'zen' 'snip' 'flops' 'params' 'zico' 'grad_norm'
    do
        CURRENT_TIME=$(date +'%Y-%m-%d_%H:%M:%S')
        CUDA_VISIBLE_DEVICES=1 python exps/run_rank_trans101.py --sample_num 50 --ss_type ${SS_NAME} --zc_type ${ZC_TYPE} --seed ${SEED} > ${LOG_PATH}/log_trans101_${SS_NAME}_${ZC_TYPE}_${CURRENT_TIME}.txt 2>&1
    done
