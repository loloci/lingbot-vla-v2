"""Per-chunk AG accounting for the Muon pipeline: how much AG time is EXPOSED.

Runs the production shape groups at depth 1 vs 2 with LINGBOT_MUON_PROFILE=1 and
prints Sum(pack/ag/wait/ns/apply). ``wait_ms`` is the exposed part of each AG,
so Sum(wait) at d=2 vs Sum(ag) at d=1 is the hiding ratio.

  torchrun --nproc_per_node=8 scripts/prof_muon_pipeline.py
"""

import json
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.bench_muon_pipeline import GROUPS  # noqa: E402


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


def run(mesh, order, depth, rank, warmup=3):
    os.environ["LINGBOT_MUON_CHUNK_ORDER"] = order
    os.environ["LINGBOT_MUON_PIPELINE_DEPTH"] = str(depth)
    os.environ["LINGBOT_MUON_PROFILE"] = "1"
    for mod in [m for m in sys.modules if m.endswith("optim.muon")]:
        del sys.modules[mod]
    from lingbotvla.optim.muon import DistributedMuon

    params = build(mesh)
    opt = DistributedMuon(params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                          nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")
    opt.set_enable_nvtx(False)          # warmup without stats
    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()
    opt.set_enable_nvtx(True)
    opt.step()
    torch.cuda.synchronize()

    path = f"/tmp/muon_prof_{order}_d{depth}_r{rank}.json"
    opt.dump_stats(path, rank=rank)
    del opt, params
    torch.cuda.empty_cache()

    d = json.load(open(path))
    tot = {k: sum(c[k] or 0.0 for c in d["chunks"])
           for k in ("pack_ms", "ag_ms", "wait_ms", "ns_ms", "apply_ms")}
    tot["n"] = len(d["chunks"])
    tot["max_ag"] = max(c["ag_ms"] for c in d["chunks"])
    tot["max_wait"] = max(c["wait_ms"] or 0.0 for c in d["chunks"])
    return tot


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))

    for order, depth in [("legacy", 1), ("bytes_desc", 2), ("legacy", 2)]:
        t = run(mesh, order, depth, rank)
        dist.barrier()
        if rank == 0:
            print(f"{order}/d{depth}: chunks={t['n']}  pack {t['pack_ms']:7.1f}  "
                  f"ag {t['ag_ms']:8.1f}  wait(exposed) {t['wait_ms']:8.1f}  "
                  f"ns {t['ns_ms']:7.1f}  apply {t['apply_ms']:6.1f}  "
                  f"| max ag {t['max_ag']:6.1f} max wait {t['max_wait']:6.1f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
