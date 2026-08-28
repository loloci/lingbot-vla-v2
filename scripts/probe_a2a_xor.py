"""Gate B, part 3: is the xor-scheduled pairwise exchange a usable a2a?

Probe 2 found the a2a penalty is CONCURRENCY, not topology:
  all_to_all_single / batch_isend_irecv (all 7 peers in flight)  1.2 GB/s
  7 xor-matched pairwise exchanges, one matching at a time      14.3 GB/s
  one isolated pair                                            25-30 GB/s
At step k every rank pairs with ``rank ^ k``, which is a perfect matching for
k = 1..ws-1 when ws is a power of two => 4 disjoint pairs per step, no link
shared. This probe checks the number is real (max over ranks), holds across
sizes, and survives the unequal per-peer payloads that NS-ownership implies.

  torchrun --nproc_per_node=8 scripts/probe_a2a_xor.py
"""

import statistics

import torch
import torch.distributed as dist

MIB = 2 ** 20
GIB = 2 ** 30
B_FULL = 8.611 * GIB          # AG output bytes per step, bf16, production


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
    ms = statistics.median(ts)
    t = torch.tensor([ms], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)   # slowest rank decides the round
    return t.item()


def xor_exchange(send, recv, offs, ws, rank, concurrent=1):
    """``concurrent`` matchings in flight at once (1 = fully staged)."""
    ks = list(range(1, ws))
    for i in range(0, len(ks), concurrent):
        ops = []
        for k in ks[i:i + concurrent]:
            peer = rank ^ k
            s0, s1 = offs[peer]
            ops.append(dist.P2POp(dist.isend, send[s0:s1], peer))
            ops.append(dist.P2POp(dist.irecv, recv[s0:s1], peer))
        for w in dist.batch_isend_irecv(ops):
            w.wait()


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    p0 = rank == 0
    assert ws & (ws - 1) == 0, "xor matching needs power-of-two world size"

    # ---- 1. size sweep, equal per-peer payload -----------------------------
    if p0:
        print(f"ws={ws}   [1] equal per-peer payload, xor staged vs all_to_all_single")
        print("    send/rank    xor ms   xor GB/s      a2a ms   a2a GB/s   speedup")
    for send_mib in (48, 96, 192, 384, 768, 1152):
        n = (send_mib * MIB // 2) // ws * ws
        chunk = n // ws
        send = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
        recv = torch.empty_like(send)
        offs = [(i * chunk, (i + 1) * chunk) for i in range(ws)]
        nb = n * 2 * (ws - 1) / ws                       # egress bytes
        mx = timeit(lambda: xor_exchange(send, recv, offs, ws, rank))
        ma = timeit(lambda: dist.all_to_all_single(recv, send))
        if p0:
            print(f"    {send_mib:5d} MiB  {mx:8.2f}   {nb / mx * 1e-6:7.2f}   "
                  f"{ma:9.2f}   {nb / ma * 1e-6:7.2f}   {ma / mx:6.2f}x", flush=True)
        del send, recv
        torch.cuda.empty_cache()

    # ---- 2. concurrency knob: how many matchings may overlap? --------------
    if p0:
        print("\n[2] matchings in flight (384 MiB/rank) — where the cliff is")
    n = (384 * MIB // 2) // ws * ws
    chunk = n // ws
    send = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
    recv = torch.empty_like(send)
    offs = [(i * chunk, (i + 1) * chunk) for i in range(ws)]
    nb = n * 2 * (ws - 1) / ws
    for c in (1, 2, 3, 7):
        ms = timeit(lambda: xor_exchange(send, recv, offs, ws, rank, concurrent=c))
        if p0:
            print(f"    concurrent={c}   {ms:8.2f} ms   {nb / ms * 1e-6:7.2f} GB/s", flush=True)
    del send, recv
    torch.cuda.empty_cache()

    # ---- 3. unequal per-peer payload (what NS ownership actually produces) --
    # LPT balance is 1.06 on real chunk costs, but BYTES per owner are lumpier
    # than cost; test both a near-balanced and a 3:1 lopsided distribution.
    if p0:
        print("\n[3] unequal per-peer payload, xor staged (384 MiB/rank total)")
    for tag, w in (("balanced  1.06:1", [17, 17, 17, 16, 16, 16, 16, 16]),
                   ("lopsided  3:1", [3, 3, 2, 2, 1, 1, 1, 1]),
                   ("2 owners idle", [3, 2, 2, 2, 2, 2, 0, 0])):
        tot = sum(w[:ws])
        sizes = [(384 * MIB // 2) * x // tot for x in w[:ws]]
        offs, acc = [], 0
        for s in sizes:
            offs.append((acc, acc + s))
            acc += s
        send = torch.empty(acc, dtype=torch.bfloat16, device="cuda").normal_()
        recv = torch.empty_like(send)
        nb = (acc - sizes[rank]) * 2
        ms = timeit(lambda: xor_exchange(send, recv, offs, ws, rank))
        nbt = torch.tensor([float(nb)], device="cuda")
        dist.all_reduce(nbt, op=dist.ReduceOp.MAX)
        if p0:
            print(f"    {tag:<18} {ms:8.2f} ms   {nbt.item() / ms * 1e-6:7.2f} GB/s", flush=True)
        del send, recv
        torch.cuda.empty_cache()

    # ---- 4. the number that decides the proposal ----------------------------
    # NS sharding sends B_full/ws per rank per direction, in R rounds.
    if p0:
        print("\n[4] projected per-step comm for NS sharding (fwd + rev, xor staged)")
        print("    R    per round/rank      ms/round     ms/step")
    for R in (1, 2, 3, 4, 6):
        per_dir = B_FULL / ws / R                        # bytes this rank sends
        n = int(per_dir // 2) // ws * ws
        chunk = n // ws
        send = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
        recv = torch.empty_like(send)
        offs = [(i * chunk, (i + 1) * chunk) for i in range(ws)]
        ms = timeit(lambda: xor_exchange(send, recv, offs, ws, rank))
        if p0:
            print(f"    {R}    {n * 2 / MIB:8.1f} MiB      {ms:8.2f}     {2 * R * ms:8.1f}",
                  flush=True)
        del send, recv
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
