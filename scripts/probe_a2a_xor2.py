"""Gate B, part 4: xor-staged exchange under the two conditions production adds.

Part 3 established the law on equal payloads (max over ranks):
    1 matching in flight   14.4-15.9 GB/s      2 matchings   1.79 GB/s
    all_to_all_single      1.07-1.20 GB/s      => 12-13x slower
Remaining unknowns before the NS-sharding patch is worth writing:
  [1] unequal per-peer payload — real ownership gives each owner a different
      byte count.  Send size to peer p = S_p (set by the OWNER), recv size from
      every peer = S_rank.  (Part 3 sized the recv slice wrong and deadlocked.)
  [2] does concurrent NS compute steal the 14.4 GB/s?  The whole design is
      "round r+1's exchange overlaps round r's NS".
  [3] the per-step projection for R = 1..6.

  torchrun --nproc_per_node=8 scripts/probe_a2a_xor2.py
"""

import statistics

import torch
import torch.distributed as dist

MIB = 2 ** 20
GIB = 2 ** 30
B_FULL = 8.611 * GIB          # AG output bytes per step, bf16, production


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


def xor_exchange(send, recv, send_off, recv_off, ws, rank):
    """One matching at a time: at step k, rank pairs with rank ^ k."""
    for k in range(1, ws):
        peer = rank ^ k
        a0, a1 = send_off[peer]
        b0, b1 = recv_off[peer]
        ops = [dist.P2POp(dist.isend, send[a0:a1], peer),
               dist.P2POp(dist.irecv, recv[b0:b1], peer)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()


def make_bufs(sizes, ws, rank):
    """send laid out by owner (S_p each); recv = ws slots of S_rank each."""
    send_off, acc = [], 0
    for s in sizes:
        send_off.append((acc, acc + s))
        acc += s
    mine = sizes[rank]
    recv_off = [(i * mine, (i + 1) * mine) for i in range(ws)]
    send = torch.empty(max(acc, 1), dtype=torch.bfloat16, device="cuda").normal_()
    recv = torch.empty(max(mine * ws, 1), dtype=torch.bfloat16, device="cuda")
    egress = (acc - mine) * 2
    return send, recv, send_off, recv_off, egress


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    p0 = rank == 0
    assert ws & (ws - 1) == 0

    # ---- 1. unequal per-peer payload --------------------------------------
    if p0:
        print(f"ws={ws}   [1] unequal per-peer payload, xor staged (1102 MiB/rank"
              " = one whole step, R=1)")
        print("    distribution        ms    max-egress GB/s")
    TOT = int(B_FULL / ws) // 2                      # bf16 elems, one rank, R=1
    for tag, w in (("equal 1:1", [1] * 8),
                   ("LPT-real 1.06:1", [17, 17, 17, 16, 16, 16, 16, 16]),
                   ("lopsided 3:1", [3, 3, 2, 2, 1, 1, 1, 1]),
                   ("2 owners idle", [3, 2, 2, 2, 2, 2, 0, 0])):
        w = w[:ws]
        tot = sum(w)
        sizes = [TOT * x // tot for x in w]
        send, recv, so, ro, eg = make_bufs(sizes, ws, rank)
        ms = timeit(lambda: xor_exchange(send, recv, so, ro, ws, rank))
        e = torch.tensor([float(eg)], device="cuda")
        dist.all_reduce(e, op=dist.ReduceOp.MAX)
        if p0:
            print(f"    {tag:<18} {ms:7.2f}   {e.item() / ms * 1e-6:7.2f}", flush=True)
        del send, recv
        torch.cuda.empty_cache()

    # ---- 2. does concurrent NS compute steal the bandwidth? ----------------
    # NS is batched bf16 matmul; use a production-ish shape (72 x 2560 x 2560).
    if p0:
        print("\n[2] exchange on a side stream vs the same exchange under NS-like"
              " compute\n    (551 MiB/rank = R=2 round)")
    n = int(B_FULL / ws / 2) // 2 // ws * ws
    chunk = n // ws
    sizes = [chunk] * ws
    send, recv, so, ro, eg = make_bufs(sizes, ws, rank)
    e = torch.tensor([float(eg)], device="cuda")
    dist.all_reduce(e, op=dist.ReduceOp.MAX)
    eg = e.item()
    comm = torch.cuda.Stream()
    A = torch.randn(72, 2560, 2560, dtype=torch.bfloat16, device="cuda")
    Bm = torch.randn(72, 2560, 2560, dtype=torch.bfloat16, device="cuda")

    def solo():
        with torch.cuda.stream(comm):
            xor_exchange(send, recv, so, ro, ws, rank)
        torch.cuda.current_stream().wait_stream(comm)

    ms_solo = timeit(solo, warmup=2, iters=6)

    def loaded():
        for _ in range(6):
            torch.bmm(A, Bm)                      # default (compute) stream
        with torch.cuda.stream(comm):
            xor_exchange(send, recv, so, ro, ws, rank)
        torch.cuda.current_stream().wait_stream(comm)

    ms_load = timeit(loaded, warmup=2, iters=6)

    def comp_only():
        for _ in range(6):
            torch.bmm(A, Bm)

    ms_comp = timeit(comp_only, warmup=2, iters=6)
    if p0:
        print(f"    exchange alone            {ms_solo:7.2f} ms  {eg / ms_solo * 1e-6:6.2f} GB/s")
        print(f"    compute alone             {ms_comp:7.2f} ms")
        print(f"    both (wall)               {ms_load:7.2f} ms   "
              f"serial would be {ms_solo + ms_comp:7.2f}   "
              f"overlap {100 * (1 - (ms_load - max(ms_solo, ms_comp)) / min(ms_solo, ms_comp)):5.1f}%",
              flush=True)
    del A, Bm, send, recv
    torch.cuda.empty_cache()

    # ---- 3. per-step projection --------------------------------------------
    if p0:
        print("\n[3] projected per-step comm, xor staged (fwd + rev, LPT-real split)")
        print("    R    per round/rank     ms/round    ms/step")
    for R in (1, 2, 3, 4, 6):
        per = int(B_FULL / ws / R) // 2
        w = [17, 17, 17, 16, 16, 16, 16, 16][:ws]
        tot = sum(w)
        sizes = [per * x // tot for x in w]
        send, recv, so, ro, eg = make_bufs(sizes, ws, rank)
        ms = timeit(lambda: xor_exchange(send, recv, so, ro, ws, rank), warmup=2, iters=8)
        if p0:
            print(f"    {R}    {per * 2 / MIB:8.1f} MiB     {ms:8.2f}   {2 * R * ms:8.1f}",
                  flush=True)
        del send, recv
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
