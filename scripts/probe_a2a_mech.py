"""Gate B follow-up: why is all_to_all 10x slower than all_gather here?

Probe 1 measured a2a at 1.07-1.18 GB/s vs all_gather 12.0 GB/s on the same
buffers. Load-bearing question for the NS-sharding proposal: is the penalty
(a) the a2a *pattern* (point-to-point on a box with zero NVLink), or
(b) NCCL's a2a *implementation*, or
(c) the cross-PHB (SYS) hop specifically?

Same egress convention as probe 1: (ws-1)/ws * send bytes / wall.

  torchrun --nproc_per_node=8 scripts/probe_a2a_mech.py
"""

import os
import statistics

import torch
import torch.distributed as dist

MIB = 2 ** 20


def timeit(fn, warmup=4, iters=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
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


def report(tag, nbytes, ms, n_peers, p0):
    if p0:
        gb = (nbytes * n_peers / (n_peers + 1)) / (ms * 1e-3) / 1e9 if n_peers else 0
        print(f"    {tag:<34} {ms:8.2f} ms   {gb:7.2f} GB/s", flush=True)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    p0 = rank == 0
    SEND = 384 * MIB                      # per-rank send buffer for every case
    n = SEND // 2                         # bf16 elements
    n -= n % ws
    if p0:
        print(f"ws={ws}  P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL','<unset>')}  "
              f"P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE','<unset>')}  "
              f"send/rank={SEND/MIB:.0f} MiB\n", flush=True)

    inp = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
    out = torch.empty_like(inp)

    # ---- ring-shaped collectives (the 12 GB/s family) -----------------------
    if p0:
        print("[A] ring-shaped collectives over all 8 ranks")
    gl = [torch.empty(n // ws, dtype=torch.bfloat16, device="cuda") for _ in range(ws)]
    small = inp[: n // ws].contiguous()
    ms = timeit(lambda: dist.all_gather(gl, small))
    report("all_gather (out = 384 MiB)", SEND, ms, ws - 1, p0)
    ms = timeit(lambda: dist.reduce_scatter_tensor(small, inp))
    report("reduce_scatter (in = 384 MiB)", SEND, ms, ws - 1, p0)
    ms = timeit(lambda: dist.broadcast(inp, src=0))
    report("broadcast (384 MiB, root 0)", SEND, ms, ws - 1, p0)
    ms = timeit(lambda: dist.all_reduce(inp))
    report("all_reduce (384 MiB)", SEND, ms, ws - 1, p0)
    del gl

    # ---- point-to-point shaped, all 8 ranks --------------------------------
    if p0:
        print("\n[B] point-to-point shaped over all 8 ranks")
    ms = timeit(lambda: dist.all_to_all_single(out, inp))
    report("all_to_all_single", SEND, ms, ws - 1, p0)

    def manual_a2a():
        """Ring-ordered pairwise exchange: peer = rank ^ k, k = 1..ws-1."""
        chunk = n // ws
        ops = []
        for k in range(1, ws):
            peer = rank ^ k
            ops.append(dist.P2POp(dist.isend, inp[peer * chunk:(peer + 1) * chunk], peer))
            ops.append(dist.P2POp(dist.irecv, out[peer * chunk:(peer + 1) * chunk], peer))
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        out[rank * chunk:(rank + 1) * chunk].copy_(inp[rank * chunk:(rank + 1) * chunk])

    ms = timeit(manual_a2a)
    report("batch_isend_irecv (xor order)", SEND, ms, ws - 1, p0)

    def staged_a2a():
        """One pairwise exchange at a time, xor order — no concurrency."""
        chunk = n // ws
        for k in range(1, ws):
            peer = rank ^ k
            ops = [dist.P2POp(dist.isend, inp[peer * chunk:(peer + 1) * chunk], peer),
                   dist.P2POp(dist.irecv, out[peer * chunk:(peer + 1) * chunk], peer)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()

    ms = timeit(staged_a2a)
    report("pairwise, one peer at a time", SEND, ms, ws - 1, p0)

    # ---- intra-PHB subgroup (0-3 / 4-7): does p2p get fast inside a group? --
    if p0:
        print("\n[C] intra-PHB subgroup of 4 (0-3 | 4-7), same 384 MiB/rank")
    sub = dist.new_group(list(range(0, 4)))
    sub2 = dist.new_group(list(range(4, 8)))
    my = sub if rank < 4 else sub2
    n4 = n - n % 4
    i4, o4 = inp[:n4], out[:n4]
    ms = timeit(lambda: dist.all_to_all_single(o4, i4, group=my))
    report("all_to_all_single (ws=4, PHB)", SEND, ms, 3, p0)
    g4 = [torch.empty(n4 // 4, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    s4 = i4[: n4 // 4].contiguous()
    ms = timeit(lambda: dist.all_gather(g4, s4, group=my))
    report("all_gather (ws=4, PHB)", SEND, ms, 3, p0)

    # ---- single cross-PHB pair vs single intra-PHB pair ---------------------
    if p0:
        print("\n[D] one isolated pair, 384 MiB one-way")
    for tag, pairs in (("intra-PHB (0<->1, 4<->5)", {0: 1, 1: 0, 4: 5, 5: 4}),
                       ("cross-PHB (0<->4, 1<->5)", {0: 4, 4: 0, 1: 5, 5: 1})):
        peer = pairs.get(rank)
        if peer is not None:
            def one_pair():
                ops = [dist.P2POp(dist.isend, inp, peer), dist.P2POp(dist.irecv, out, peer)]
                for w in dist.batch_isend_irecv(ops):
                    w.wait()
            ms = timeit(one_pair, warmup=3, iters=8)
        else:
            ms = 0.0
        t = torch.tensor([ms], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if p0:
            print(f"    {tag:<34} {t.item():8.2f} ms   "
                  f"{SEND / (t.item() * 1e-3) / 1e9:7.2f} GB/s", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
