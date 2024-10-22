#!/bin/bash

SEED=${1}
echo $SEED
SS_NAME='macro'
TASK='naive_search'
LOG_PATH=./logs/${TASK}/${SS_NAME}/${SEED}

mkdir -p $LOG_PATH

TIME=$(date +'%Y-%m-%d_%H:%M:%S')
echo "Start time: ${TIME}"
# Run rnd search on trans101
ITER=20
SAMP=200
python exps/naive_search_trans101.py --iteration ${ITER} \
                                    --sample_num ${SAMP} \
                                    --ss_type ${SS_NAME} > ${LOG_PATH}/log_trans101_${SS_NAME}_${ITER}_${SAMP}_${SEED}.txt 2>&1 &

END_TIME=$(date +'%Y-%m-%d_%H:%M:%S')
echo "End time: ${END_TIME}"

# TODO List
# 1. search for 50 iterations under 50 samples to get the best auto-gas proxy. [running]
# 2. evaluate the best auto-gas proxy and gather data for plotting.
# 3. evaluate the existing proxies including param, synflow, nwot with 200 samples
# 4. evaluate the searched (version 1.0) proxy
