#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_DISABLE_TELEMETRY=1 
export DISABLE_TELEMETRY=1 

# Perf knobs validated in report/05_muon-AG (§A.3) and
# report/06_fsdp2_forwardprefech_depth (§9). `${VAR-default}` omits the colon on
# purpose: the fp32 control cell is an *empty* LINGBOT_MUON_AG_DTYPE, which `:-`
# would swallow. Rollback / A-B cells, set them in the invoking environment:
#   LINGBOT_MUON_PIPELINE_DEPTH=1       legacy fully-serial Muon AG (bit-identical)
#   LINGBOT_MUON_CHUNK_ORDER=legacy     pre-260818 shape-key chunk order
#   LINGBOT_MUON_AG_BYTE_CAP=0          count-based chunking only
#   LINGBOT_MUON_AG_DTYPE=""            keep fp32 on the wire
#   LINGBOT_FSDP2_PREFETCH=0            no forward prefetch chain
export LINGBOT_MUON_PIPELINE_DEPTH=${LINGBOT_MUON_PIPELINE_DEPTH-2}
export LINGBOT_MUON_CHUNK_ORDER=${LINGBOT_MUON_CHUNK_ORDER-bytes_desc}
export LINGBOT_MUON_AG_BYTE_CAP=${LINGBOT_MUON_AG_BYTE_CAP-700000000}
export LINGBOT_MUON_AG_DTYPE=${LINGBOT_MUON_AG_DTYPE-bf16}
export LINGBOT_FSDP2_PREFETCH=${LINGBOT_FSDP2_PREFETCH-2}

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
