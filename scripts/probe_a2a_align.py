"""Gate B, part 6: the 2x gap is payload ALIGNMENT, not schedule.

Part 5 (aligned, 144 MiB = 551*256KiB per exchange)   15.9 GB/s   9.09 ms/exch
Part 4 (ragged, 144,438,788 B per exchange, 4B-aligned) 8.16 GB/s  17.71 ms/exch
Same xor schedule, same volume, deterministic across reruns => it is the
element count / slice offset, and NS-ownership payloads (sums of N_c*L_c*T_c)
are ragged by nature.  If padding each per-owner payload to a boundary buys 2x,
that is a one-line design rule for the patch.

Sweeps the alignment of the per-exchange element count, holding the payload
within 0.2% of 144 MiB, and separates size-alignment from offset-alignment.

  torchrun --nproc_per_node=8 scripts/probe_a2a_align.py
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


def xor_exchange(send, recv, per, ws, rank, stride=None):
    """stride = slot pitch; per = bytes actually sent (stride >= per)."""
    st = per if stride is None else stride
    for k in range(1, ws):
        peer = rank ^ k
        ops = [dist.P2POp(dist.isend, send[peer * st:peer * st + per], peer),
               dist.P2POp(dist.irecv, recv[peer * st:peer * st + per], peer)]
        for w in dist.batch_isend_irecv(ops):
            w.wait()


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    p0 = rank == 0

    BASE = 72_219_394        # part-4's ragged count, 4B-aligned in bytes
    if p0:
        print(f"ws={ws}  xor staged, ~144 MiB per exchange, max over ranks\n")
        print("[1] size AND offset aligned to A bytes (count rounded up)")
        print("      A        elems      MiB/exch     ms   GB/s")
    for A in (4, 16, 64, 512, 4096, 65536, 262144, 2097152):
        ae = max(A // 2, 1)                          # alignment in bf16 elems
        per = (BASE + ae - 1) // ae * ae
        send = torch.empty(per * ws, dtype=torch.bfloat16, device="cuda").normal_()
        recv = torch.empty_like(send)
        eg = per * 2 * (ws - 1)
        ms = timeit(lambda: xor_exchange(send, recv, per, ws, rank))
        if p0:
            print(f"    {A:8d}  {per:11d}  {per * 2 / MIB:8.2f}  {ms:7.2f}  "
                  f"{eg / ms * 1e-6:6.2f}", flush=True)
        del send, recv
        torch.cuda.empty_cache()

    # [2] Which one matters: the transfer SIZE or the buffer OFFSET?
    # stride = 2 MiB-aligned slot (=> every slice starts aligned) while the
    # transferred count stays ragged.
    if p0:
        print("\n[2] offset aligned to 2 MiB, size left ragged (BASE elems)")
    ae = 2097152 // 2
    stride = (BASE + ae - 1) // ae * ae
    send = torch.empty(stride * ws, dtype=torch.bfloat16, device="cuda").normal_()
    recv = torch.empty_like(send)
    eg = BASE * 2 * (ws - 1)
    ms = timeit(lambda: xor_exchange(send, recv, BASE, ws, rank, stride=stride))
    if p0:
        print(f"    ragged size, aligned offset   {ms:7.2f} ms  {eg / ms * 1e-6:6.2f} GB/s",
              flush=True)
    ms = timeit(lambda: xor_exchange(send, recv, stride, ws, rank, stride=stride))
    if p0:
        print(f"    aligned size, aligned offset  {ms:7.2f} ms  "
              f"{stride * 2 * (ws - 1) / ms * 1e-6:6.2f} GB/s", flush=True)
    del send, recv
    torch.cuda.empty_cache()

    # [3] does the same rule apply to all_gather?  (today's production path)
    if p0:
        print("\n[3] all_gather anchor, ragged vs aligned input (same ~144 MiB in)")
    for tag, cnt in (("ragged", BASE), ("2 MiB-aligned", stride)):
        inp = torch.empty(cnt, dtype=torch.bfloat16, device="cuda").normal_()
        gl = [torch.empty_like(inp) for _ in range(ws)]
        ms = timeit(lambda: dist.all_gather(gl, inp))
        if p0:
            print(f"    {tag:<16} {ms:7.2f} ms  "
                  f"{cnt * 2 * ws * (ws - 1) / ws / ms * 1e-6:6.2f} GB/s", flush=True)
        del inp, gl
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
