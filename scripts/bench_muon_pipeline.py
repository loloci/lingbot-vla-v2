"""Microbenchmark: DistributedMuon optimizer_step at pipeline depth 1/2/3.

Uses the REAL production shape groups (25 shapes / 44 chunks, extracted from
output/fa2full_solverD_1n_0731_1509/.../muon_chunk_stats_step43.json) so the
AG/NS skew that decides the payoff is reproduced exactly.

  torchrun --nproc_per_node=8 scripts/bench_muon_pipeline.py
"""

import os
import statistics
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (global_shape, n_params) — production Muon FSDP_GATHER_2D groups.
GROUPS = [
    ((32, 768), 36), ((55, 768), 1), ((128, 2560), 4), ((256, 2560), 8),
    ((704, 768), 72), ((768, 55), 2), ((768, 704), 36), ((768, 768), 145),
    ((768, 1536), 1), ((768, 4096), 36), ((1024, 768), 72), ((1024, 1024), 24),
    ((1024, 2560), 76), ((1024, 4096), 24), ((2560, 128), 4), ((2560, 2560), 16),
    ((2560, 4096), 40), ((2560, 5120), 2), ((2560, 9728), 36), ((3072, 1024), 24),
    ((4096, 768), 36), ((4096, 1024), 24), ((4096, 2560), 36), ((4096, 4096), 4),
    ((9728, 2560), 72),
]


def build(mesh):
    params = []
    for shape, count in GROUPS:
        for _ in range(count):
            full = torch.empty(*shape, dtype=torch.float32, device="cuda").normal_()
            p = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
            p.grad = distribute_tensor(
                torch.empty(*shape, dtype=torch.float32, device="cuda").normal_(),
                mesh, [Shard(0)])
            params.append(p)
    return params


def bench(mesh, order, depth, cap=0, ag_dtype="", warmup=3, iters=12):
    os.environ["LINGBOT_MUON_CHUNK_ORDER"] = order
    os.environ["LINGBOT_MUON_PIPELINE_DEPTH"] = str(depth)
    os.environ["LINGBOT_MUON_AG_BYTE_CAP"] = str(cap)
    os.environ["LINGBOT_MUON_AG_DTYPE"] = ag_dtype
    for mod in [m for m in sys.modules if m.endswith("optim.muon")]:
        del sys.modules[mod]
    from lingbotvla.optim.muon import DistributedMuon

    params = build(mesh)
    opt = DistributedMuon(params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                          nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")
    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()

    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del opt, params
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return statistics.median(ts), min(ts), peak


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))

    rows = []
    # cap>0 = byte-capped split; 7.0e8 B is the measured fp32 optimum (~54 ms).
    # ag_dtype bf16 halves every payload => the cap must halve with it to keep
    # the same millisecond budget per chunk.
    cases = [("legacy", 1, 0, ""), ("bytes_desc", 2, 0, ""),
             ("bytes_desc", 2, 700_000_000, ""), ("bytes_desc", 3, 700_000_000, ""),
             ("bytes_desc", 1, 0, "bf16"), ("bytes_desc", 2, 0, "bf16"),
             ("bytes_desc", 2, 700_000_000, "bf16"),
             ("bytes_desc", 2, 350_000_000, "bf16"),
             ("bytes_desc", 3, 350_000_000, "bf16"),
             ("bytes_desc", 2, 200_000_000, "bf16")]
    for order, depth, cap, agd in cases:
        med, best, peak = bench(mesh, order, depth, cap, agd)
        t = torch.tensor([med, best, peak], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        tag = f"{order}/cap{cap // 10**6}M" + (f"/{agd}" if agd else "/fp32")
        rows.append((tag, depth, t[0].item(), t[1].item(), t[2].item()))
        if rank == 0:
            print(f"  {tag:<28} d={depth}  median {t[0].item():8.1f} ms  "
                  f"min {t[1].item():8.1f}  peakVRAM {t[2].item():5.2f} GiB", flush=True)

    if rank == 0:
        base = rows[0][2]
        e6 = rows[2][2]  # bytes_desc/d2/cap700M/fp32 == the E6 production config
        print("\n  vs legacy/d1/fp32 (median, max over ranks):")
        for tag, depth, med, _, _ in rows[1:]:
            print(f"    {tag:<28} d={depth}  {med - base:+8.1f} ms "
                  f"({(med - base) / base * 100:+6.2f}%)  vs E6 {med - e6:+8.1f} ms "
                  f"({(med - e6) / e6 * 100:+6.2f}%)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
