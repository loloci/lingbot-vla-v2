"""Why cap so small it yields 1 param/chunk is not bitwise.

Hypothesis: cuBLAS dispatches ``baddbmm`` with batch==1 to a different kernel
than batch>1, so a matrix's NS result depends on how many siblings share its
batched call. Nothing to do with the plan construction. Run on 1 GPU.
"""

import torch

torch.manual_seed(0)
M = K = 2560
NS = 5
a, b, c = 3.4445, -4.7750, 2.0315


def ns(x):  # same math as batched_newton_schulz, bf16, batched path
    o = x.to(torch.bfloat16)
    o = o / o.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)
    for _ in range(NS):
        A = o @ o.mT
        g = torch.baddbmm(A, A, A, beta=b, alpha=c)
        o = torch.baddbmm(o, g, o, beta=a)
    return o


full = torch.randn(32, M, K, device="cuda", dtype=torch.float32)
for n in (1, 2, 3, 8, 13, 32):
    got = ns(full[:n].clone())[0]
    ref = ns(full[:32].clone())[0]
    d = (got.float() - ref.float()).abs().max().item()
    print(f"square {M}x{K} batch={n:<3} vs 32 maxdiff={d:.3e} {'OK' if d == 0 else 'DIFF'}")

# The test's real shapes: shard(0) over 8 ranks then cat back to global.
for shape, nmax in (((2560, 9728), 3), ((9728, 2560), 2), ((1536, 2560), 5), ((2560, 256), 7)):
    full = torch.randn(nmax, *shape, device="cuda", dtype=torch.float32)
    ref = ns(full.clone())[0]
    for n in range(1, nmax + 1):
        d = (ns(full[:n].clone())[0].float() - ref.float()).abs().max().item()
        print(f"{shape} batch={n} vs {nmax} maxdiff={d:.3e} {'OK' if d == 0 else 'DIFF'}")
