#!/bin/bash

SEED=${1}
echo $SEED
SS_NAME='macro'
TASK='naive_rank'
LOG_PATH=./logs/${TASK}/${SS_NAME}/${SEED}
SAMP=200

mkdir -p $LOG_PATH

# Define the function to rank the proxies
rank_proxies() {
    python exps/naive_rank_trans101.py --sample_num ${SAMP} \
                                       --ss_type ${SS_NAME} \
                                       --seed ${SEED} > ${LOG_PATH}/log_trans101_${SS_NAME}_${SEED}.txt 2>&1 &
}

# Call the function to rank the proxies
rank_proxies
