"""Gate B: measured all_to_all_single bandwidth on this box vs all_gather.

The NS-sharding plan (report/_docs/muon_ns_shard_a2a_proposal.md) replaces 48
all_gathers with 2R all_to_all_singles. Its 173 ms estimate uses 11.7 GB/s
reverse-engineered from a *ring* all_gather; a2a is a different pattern and
this node has zero NVLink (NCCL_P2P_LEVEL=SYS => every hop crosses the PCIe
root complex), so the number has to be measured.

Convention: GB/s = per-rank EGRESS bytes / wall, i.e. (ws-1)/ws of the send
buffer, matching how the 12.2 GB/s Muon-AG figure was computed.

  torchrun --nproc_per_node=8 scripts/probe_a2a_bw.py
"""

import os
import statistics

import torch
import torch.distributed as dist

# One production step moves B_full = 8.611 GiB of AG output. Under NS sharding
# each rank sends B_full/ws per direction; with R rounds that is per round:
#   fwd send = B_full / (ws * R)
GIB = 2 ** 30
B_FULL = 8.611 * GIB


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def bw(nbytes, ms, ws):
    """Per-rank egress GB/s."""
    return (nbytes * (ws - 1) / ws) / (ms * 1e-3) / 1e9


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    p0 = rank == 0
    if p0:
        print(f"world_size={ws}  P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL','')}  "
              f"ALGO={os.environ.get('NCCL_ALGO','')}\n", flush=True)

    # ---- 1. a2a, equal splits, sweep the per-rank send volume ----------------
    if p0:
        print("[1] all_to_all_single, EQUAL splits (best case for NCCL)")
        print("    per-rank send      ms      egress GB/s")
    for send_mib in (48, 96, 192, 384, 768, 1152):
        n = int(send_mib * 2 ** 20 // 2)  # bf16 elements
        n -= n % ws
        inp = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
        out = torch.empty_like(inp)
        ms = timeit(lambda: dist.all_to_all_single(out, inp))
        nb = inp.numel() * 2
        t = torch.tensor([ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ms = t.item()
        if p0:
            print(f"    {send_mib:5d} MiB     {ms:8.2f}   {bw(nb, ms, ws):8.2f}", flush=True)
        del inp, out
        torch.cuda.empty_cache()

    # ---- 2. a2a, UNEQUAL splits (the real shape of one NS-sharding round) ----
    # A round holds ~48/R chunks spread over ws owners; owners get different
    # chunk counts and chunks differ in size, so both split vectors are lopsided
    # and some entries are 0. NCCL can fall back to pairwise send/recv here.
    if p0:
        print("\n[2] all_to_all_single, UNEQUAL splits (one round, R rounds/step)")
        print("    R   per-rank send      ms      egress GB/s   (2R calls/step -> ms/step)")
    for R in (1, 2, 3, 6):
        per_rank_send = B_FULL / (ws * R)
        # Lopsided but sums to per_rank_send: weights 3,2,2,1,1,1,0,0 shuffled by
        # rank so no rank is a privileged destination.
        w = [3, 2, 2, 1, 1, 1, 0, 0][:ws] or [1]
        while len(w) < ws:
            w.append(1)
        tot = sum(w)
        in_splits = [int(per_rank_send * x / tot) // 2 for x in w]  # bf16 elems
        # Every rank uses the same input split vector; consistency then requires
        # output_split_sizes[i] == in_splits[rank] for all i.
        out_splits = [in_splits[rank]] * ws
        inp = torch.empty(sum(in_splits), dtype=torch.bfloat16, device="cuda").normal_()
        out = torch.empty(sum(out_splits), dtype=torch.bfloat16, device="cuda")
        ms = timeit(lambda: dist.all_to_all_single(out, inp, out_splits, in_splits))
        nb = inp.numel() * 2
        t = torch.tensor([ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ms = t.item()
        if p0:
            print(f"    {R}   {nb / 2**20:8.1f} MiB   {ms:8.2f}   {bw(nb, ms, ws):8.2f}"
                  f"        {2 * R * ms:8.1f}", flush=True)
        del inp, out
        torch.cuda.empty_cache()

    # ---- 3. all_gather anchor: reproduce the 12.2 GB/s production figure -----
    if p0:
        print("\n[3] all_gather anchor (today's path, per-chunk)")
        print("    out_bytes      ms      egress GB/s")
    for out_mib in (570, 285, 143):
        n = int(out_mib * 2 ** 20 // 2 // ws)
        inp = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
        gl = [torch.empty_like(inp) for _ in range(ws)]
        ms = timeit(lambda: dist.all_gather(gl, inp))
        nb = n * 2 * ws
        t = torch.tensor([ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ms = t.item()
        if p0:
            print(f"    {out_mib:5d} MiB   {ms:8.2f}   {bw(nb, ms, ws):8.2f}", flush=True)
        del inp, gl
        torch.cuda.empty_cache()

    # ---- 4. whole-step totals: today's 48 AGs vs 2R a2a ---------------------
    if p0:
        print("\n[4] per-step comm totals (bf16, B_full = 8.611 GiB)")
        print(f"    today  all_gather egress/rank = {B_FULL * (ws - 1) / ws / GIB:.2f} GiB")
        print(f"    a2a    fwd+rev    egress/rank = "
              f"{2 * B_FULL * (ws - 1) / ws / ws / GIB:.2f} GiB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
