# NCCL tuning for multi-node training

`train.sh` provides an opt-in NCCL profile for the multi-node topology on
which LingBot-VLA was profiled. Enable it on every node before launching the
training command:

```bash
LINGBOT_NCCL_TUNED=1 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=<rank-0-host> \
bash train.sh tasks/vla/train_lingbotvla.py <config>
```

The profile supplies these defaults:

| Variable | Value |
| --- | ---: |
| `NCCL_PROTO` | `LL128` |
| `NCCL_MIN_NCHANNELS` | `8` |
| `NCCL_MAX_NCHANNELS` | `16` |
| `NCCL_NCHANNELS_PER_PEER` | `4` |
| `NCCL_IB_QPS_PER_CONNECTION` | `4` |
| `NCCL_BUFFSIZE` | `16777216` (16 MiB) |

Existing environment variables take precedence, so individual settings can
be adjusted without editing the launcher:

```bash
LINGBOT_NCCL_TUNED=1 NCCL_MAX_NCHANNELS=8 bash train.sh ...
```

## Measured results

The profile was measured on two nodes with eight Pro6000 GPUs per node and a
400 Gbps inter-node fabric. Relative to the default NCCL settings:

| Metric | Baseline | Tuned | Change |
| --- | ---: | ---: | ---: |
| AllReduce bandwidth, 4 MiB | 7.52 GB/s | 11.80 GB/s | +57% |
| AllReduce bandwidth, 4 GiB | 11.63 GB/s | 13.35 GB/s | +15% |
| Training step time, micro-batch 48 | 8.097 s | 7.804 s | -3.6% |
| Weak-scaling ratio | 0.9324 | 0.9675 | +3.50 pp |

Loss at the measured final step and peak GPU memory remained effectively
unchanged. These values are topology-specific; benchmark the profile before
using it as a production default on different GPUs, NICs, or switch fabrics.
