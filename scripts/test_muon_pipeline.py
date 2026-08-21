"""Equivalence test for the Muon AG/NS pipeline (P1a-i + P1a-ii).

Runs DistributedMuon over FSDP2-style sharded DTensor params under several
(chunk order x pipeline depth) settings and asserts every setting produces
BITWISE-identical params. Only the schedule changes, so any diff is a bug.

  torchrun --nproc_per_node=8 scripts/test_muon_pipeline.py
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shapes mirror the production skew: a few giants + a long tail of small ones,
# and one group big enough to split across _MEGABATCH_MAX_GROUP_SIZE=32.
SHAPES = [
    ((2560, 9728), 3),
    ((9728, 2560), 2),
    ((2560, 2560), 40),   # 40 params => 2 chunks
    ((1536, 2560), 5),
    ((2560, 256), 7),
]


def build_params(mesh, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for shape, count in SHAPES:
        for _ in range(count):
            full = torch.randn(*shape, generator=g, dtype=torch.float32)
            p = torch.nn.Parameter(distribute_tensor(full.cuda(), mesh, [Shard(0)]))
            gr = torch.randn(*shape, generator=g, dtype=torch.float32)
            p.grad = distribute_tensor(gr.cuda(), mesh, [Shard(0)])
            out.append(p)
    return out


def run_case(mesh, order, depth, cap=0, ag_dtype="", steps=3):
    os.environ["LINGBOT_MUON_CHUNK_ORDER"] = order
    os.environ["LINGBOT_MUON_PIPELINE_DEPTH"] = str(depth)
    os.environ["LINGBOT_MUON_AG_BYTE_CAP"] = str(cap)
    os.environ["LINGBOT_MUON_AG_DTYPE"] = ag_dtype
    for mod in [m for m in sys.modules if m.endswith("optim.muon")]:
        del sys.modules[mod]
    from lingbotvla.optim.muon import DistributedMuon

    params = build_params(mesh, seed=1234)
    opt = DistributedMuon(params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                          nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")
    for s in range(steps):
        for i, p in enumerate(params):
            g = torch.Generator(device="cpu").manual_seed(9000 + 100 * s + i)
            gr = torch.randn(*p.shape, generator=g, dtype=torch.float32)
            p.grad = distribute_tensor(gr.cuda(), mesh, [Shard(0)])
        opt.step()
    torch.cuda.synchronize()
    flat = torch.cat([p.to_local().reshape(-1) for p in params])
    del opt, params
    torch.cuda.empty_cache()
    return flat


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))

    # (order, depth, ag_byte_cap, ag_dtype). cap>0 = byte-capped split; caps
    # bracket the production 7.0e8 and force a 1-param-per-chunk extreme.
    # ag_dtype="bf16" must ALSO be bitwise: NS casts to bf16 on entry anyway.
    cases = [("legacy", 1, 0, ""), ("bytes_desc", 1, 0, ""), ("bytes_desc", 2, 0, ""),
             ("bytes_desc", 3, 0, ""), ("bytes_desc", 2, 1_300_000_000, ""),
             ("bytes_desc", 2, 300_000_000, ""), ("bytes_desc", 1, 1, ""),
             ("legacy", 2, 1_300_000_000, ""),
             ("bytes_desc", 2, 0, "bf16"), ("bytes_desc", 1, 0, "bf16"),
             ("bytes_desc", 2, 700_000_000, "bf16"), ("legacy", 1, 0, "bf16")]
    ref = None
    ok = True
    for order, depth, cap, agd in cases:
        out = run_case(mesh, order, depth, cap, agd)
        tag = f"{order}/d{depth}/cap{cap}" + (f"/{agd}" if agd else "")
        if ref is None:
            ref = out
            if rank == 0:
                print(f"[ref ] {tag:<32} |p| sum={out.abs().sum().item():.6f}", flush=True)
            continue
        same = torch.equal(ref, out)
        maxdiff = (ref - out).abs().max().item()
        ok &= same
        if rank == 0:
            print(f"[test] {tag:<32} bitwise={'OK ' if same else 'FAIL'} "
                  f"maxdiff={maxdiff:.3e}", flush=True)

    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("RESULT:", "PASS" if flag.item() > 0 else "FAIL", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if flag.item() > 0 else 1)


if __name__ == "__main__":
    main()
