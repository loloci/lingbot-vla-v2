"""Gate B, part 5: reconcile 15.9 GB/s (part 3) vs 8.16 GB/s (part 4).

Same xor schedule, same ~140 MiB per exchange, two different runs, 2x apart.
Two candidate causes, and they need different responses:
  (a) code — part 3 shared one offset table for send and recv, part 4 gave recv
      its own layout (ws slots of S_rank).  If this is it, the layout matters.
  (b) run — warm-up order / environment drift.  If this is it, the number
      depends on how many exchanges preceded it, which changes the projection.
Both forms, same sizes, same process, twice each, ascending then descending.

  torchrun --nproc_per_node=8 scripts/probe_a2a_xor3.py
"""

import statistics

import torch
import torch.distributed as dist

MIB = 2 ** 20


def timeit(fn, warmup=3, iters=10):
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
    t = torch.tensor([statistics.median(ts)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def run_shared(send, recv, per, ws, rank):
    """part-3 form: recv slice at the SAME offset as the send slice."""
    for k in range(1, ws):
        peer = rank ^ k
        s0, s1 = peer * per, (peer + 1) * per
        ops = [dist.P2POp(dist.isend, send[s0:s1], peer),
               dist.P2POp(dist.irecv, recv[s0:s1], peer)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()


def run_split(send, recv, per, ws, rank):
    """part-4 form: recv laid out as ws slots, slot index = peer."""
    for k in range(1, ws):
        peer = rank ^ k
        ops = [dist.P2POp(dist.isend, send[peer * per:(peer + 1) * per], peer),
               dist.P2POp(dist.irecv, recv[peer * per:(peer + 1) * per], peer)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    p0 = rank == 0
    if p0:
        print(f"ws={ws}   per-exchange MiB -> egress GB/s, xor staged, max over ranks")
        print("    pass   per-exch     shared     split")

    sizes = [48, 96, 144, 144, 96, 48]        # ascending then descending
    for pas, mib in enumerate(sizes):
        per = int(mib * MIB // 2)
        send = torch.empty(per * ws, dtype=torch.bfloat16, device="cuda").normal_()
        recv = torch.empty_like(send)
        eg = per * 2 * (ws - 1)
        a = timeit(lambda: run_shared(send, recv, per, ws, rank))
        b = timeit(lambda: run_split(send, recv, per, ws, rank))
        if p0:
            print(f"    {pas}      {mib:5d} MiB   {eg / a * 1e-6:7.2f}   {eg / b * 1e-6:7.2f}"
                  f"     ({a:7.2f} / {b:7.2f} ms)", flush=True)
        del send, recv
        torch.cuda.empty_cache()

    # Same total volume, one big buffer vs many exchanges: does per-exchange size
    # alone explain it?  144 MiB x 7 == 1008 MiB egress either way.
    if p0:
        print("\n    one whole R=1 round (137.75 MiB/exchange) repeated 3x, split form")
    per = int(137.75 * MIB // 2)
    send = torch.empty(per * ws, dtype=torch.bfloat16, device="cuda").normal_()
    recv = torch.empty_like(send)
    eg = per * 2 * (ws - 1)
    for i in range(3):
        b = timeit(lambda: run_split(send, recv, per, ws, rank))
        if p0:
            print(f"    rep {i}   {b:7.2f} ms   {eg / b * 1e-6:7.2f} GB/s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
