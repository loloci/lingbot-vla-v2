#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_DISABLE_TELEMETRY=1 
export DISABLE_TELEMETRY=1 

# Opt-in NCCL profile tuned for multi-node training on the measured 400 Gbps
# fabric. Keep the default disabled because the best protocol and channel count
# depend on the GPU and network topology. Individual values remain overridable.
if [ "${LINGBOT_NCCL_TUNED:-0}" = "1" ]; then
  export NCCL_PROTO="${NCCL_PROTO:-LL128}"
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
  export NCCL_NCHANNELS_PER_PEER="${NCCL_NCHANNELS_PER_PEER:-4}"
  export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
  export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-16777216}"
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}


torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
