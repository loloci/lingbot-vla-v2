"""A/B: Newton-Schulz sharded to chunk owners, on top of today's best Muon.

Single variable. Control = the production config (chunk order by descending AG
bytes, depth-2 AG/NS double buffer, on-wire byte cap 7e8, AG in bf16). Treatment
= the same plus LINGBOT_MUON_NS_SHARD=R, which routes each chunk's shards to one
owner, runs NS once there, and ships the output slices back over a staged xor
p2p exchange instead of all-gathering to every rank.

Real production shape groups (25 shapes / 48 chunks), so the AG/NS skew that
decides the payoff is reproduced. Two passes:
  verify  1/8-scale params, identical seed, bitwise compare vs the control
  time    full scale, median/min over 12 steps, max over ranks

  torchrun --nproc_per_node=8 scripts/bench_muon_nsshard.py [verify|time|both]
"""

import os
import statistics
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUPS = [
    ((32, 768), 36), ((55, 768), 1), ((128, 2560), 4), ((256, 2560), 8),
    ((704, 768), 72), ((768, 55), 2), ((768, 704), 36), ((768, 768), 145),
    ((768, 1536), 1), ((768, 4096), 36), ((1024, 768), 72), ((1024, 1024), 24),
    ((1024, 2560), 76), ((1024, 4096), 24), ((2560, 128), 4), ((2560, 2560), 16),
    ((2560, 4096), 40), ((2560, 5120), 2), ((2560, 9728), 36), ((3072, 1024), 24),
    ((4096, 768), 36), ((4096, 1024), 24), ((4096, 2560), 36), ((4096, 4096), 4),
    ((9728, 2560), 72),
]

# order, depth, byte cap, AG dtype — frozen at the production values so the only
# thing that moves between rows is the NS_SHARD round count.
PROD = ("bytes_desc", 2, 700_000_000, "bf16")


def build(mesh, scale=1, seed=None):
    """Params + grads. Same seed on every rank => identical global tensors."""
    if seed is not None:
        torch.manual_seed(seed)
    params = []
    for shape, count in GROUPS:
        for _ in range(max(1, count // scale)):
            full = torch.empty(*shape, dtype=torch.float32, device="cuda").normal_()
            p = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
            p.grad = distribute_tensor(
                torch.empty(*shape, dtype=torch.float32, device="cuda").normal_(),
                mesh, [Shard(0)])
            params.append(p)
    return params


def make_opt(params, ns):
    order, depth, cap, agd = PROD
    os.environ["LINGBOT_MUON_CHUNK_ORDER"] = order
    os.environ["LINGBOT_MUON_PIPELINE_DEPTH"] = str(depth)
    os.environ["LINGBOT_MUON_AG_BYTE_CAP"] = str(cap)
    os.environ["LINGBOT_MUON_AG_DTYPE"] = agd
    os.environ["LINGBOT_MUON_NS_SHARD"] = str(ns)
    for mod in [m for m in sys.modules if m.endswith("optim.muon")]:
        del sys.modules[mod]
    from lingbotvla.optim.muon import DistributedMuon
    return DistributedMuon(params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                           nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")


def run_steps(mesh, ns, scale, steps, seed):
    params = build(mesh, scale, seed)
    opt = make_opt(params, ns)
    for _ in range(steps):
        opt.step()
    out = [p.to_local().detach().clone() for p in params]
    del opt, params
    torch.cuda.empty_cache()
    return out


def verify(mesh, rounds, scale=8, steps=2):
    """Bitwise: same seed, same steps, control vs each R."""
    rank = dist.get_rank()
    ref = run_steps(mesh, 0, scale, steps, seed=1234)
    if rank == 0:
        print(f"\nbitwise vs ns_shard=0 ({len(ref)} params, 1/{scale} scale, "
              f"{steps} steps)", flush=True)
    for R in rounds:
        got = run_steps(mesh, R, scale, steps, seed=1234)
        bad = sum(0 if torch.equal(a, b) else 1 for a, b in zip(ref, got))
        mx = max((a - b).abs().max().item() for a, b in zip(ref, got))
        t = torch.tensor([float(bad), mx], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        del got
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"  R={R}   mismatching params {int(t[0].item()):4d}/{len(ref)}"
                  f"   max|diff| {t[1].item():.3e}", flush=True)


def time_one(mesh, ns, warmup=3, iters=12):
    params = build(mesh, 1)
    opt = make_opt(params, ns)
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
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del opt, params
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return statistics.median(ts), min(ts), peak


def timing(mesh, rounds):
    rank = dist.get_rank()
    if rank == 0:
        print(f"\noptimizer_step, {PROD[0]}/d{PROD[1]}/cap{PROD[2] // 10 ** 6}M/"
              f"{PROD[3]}, max over ranks", flush=True)
    rows = []
    for R in [0] + list(rounds):
        med, best, peak = time_one(mesh, R)
        t = torch.tensor([med, best, peak], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        rows.append((R, t[0].item(), t[1].item(), t[2].item()))
        if rank == 0:
            base = rows[0][1]
            print(f"  ns_shard={R}   median {t[0].item():8.1f} ms   min {t[1].item():8.1f}"
                  f"   peakVRAM {t[2].item():5.2f} GiB"
                  f"   {'' if R == 0 else f'{t[0].item() - base:+8.1f} ms ({(t[0].item() - base) / base * 100:+6.2f}%)'}",
                  flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    rounds = (1, 2, 3, 4, 6)
    if what in ("verify", "both"):
        verify(mesh, rounds)
    if what in ("time", "both"):
        timing(mesh, rounds)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
