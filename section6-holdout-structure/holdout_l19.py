#!/usr/bin/env python3
"""H-013 extension: exact holdout sets at l=19, budgets 20,21,22 (j*(19)=23).
Also: x2/x4 inclusion checks, doubling-chain maxrun (= dlog run length), mod-3^6 classes,
and the bootstrap-tightness prediction maxrun(H(19,j)) == 23 - j for j=21,22."""
import sys
import numpy as np

def coverage(l, j):
    q = 3 ** l
    dp = [np.zeros(q, dtype=bool) for _ in range(j + 1)]
    dp[0][0] = True
    for a in range(2 * j - 1, -1, -1):
        c_hi = min(j - 1, 2 * j - 1 - a)
        c_lo = max(0, j - 1 - a)
        for c in range(c_hi, c_lo - 1, -1):
            t = (pow(2, a, q) * pow(3, c, q)) % q
            src, dst = dp[c], dp[c + 1]
            if t == 0:
                dst |= src
            else:
                dst[t:] |= src[:q - t]
                dst[:t] |= src[q - t:]
    out = dp[j].copy()
    del dp
    return out

def maxrun_chains(hold, q):
    s = set(int(x) for x in hold)
    best, best_start = 0, None
    for h in s:
        inv2h = (h * pow(2, -1, q)) % q
        if inv2h not in s:                      # chain start
            r, x = 1, h
            while (x * 2) % q in s:
                x = (x * 2) % q
                r += 1
            if r > best:
                best, best_start = r, h
    return best, best_start

l = 19
q = 3 ** l
JSTAR_L = 23
prev_hold = None
for j in (20, 21, 22):
    cov = coverage(l, j)
    um = np.ones(q, dtype=bool); um[::3] = False
    hold = np.nonzero(um & ~cov)[0]
    del cov, um
    mr, start = maxrun_chains(hold, q)
    cls = sorted(set(int(x) % 729 for x in hold))
    print(f"l=19 j={j}: |H|={len(hold)}, maxrun={mr} (chain start {start}), "
          f"bootstrap j*<={j+mr} (true {JSTAR_L}) "
          f"{'TIGHT' if j+mr == JSTAR_L else ('OK' if j+mr > JSTAR_L else 'FALSIFIED')}",
          flush=True)
    print(f"  mod-3^6 classes ({len(cls)}): {cls[:40]}{'...' if len(cls) > 40 else ''}", flush=True)
    if len(hold) <= 10:
        print(f"  residues: {[int(x) for x in hold]}", flush=True)
    if prev_hold is not None:
        s_prev = set(int(x) for x in prev_hold)
        inv2, inv4 = pow(2, -1, q), pow(4, -1, q)
        bad2 = [int(x) for x in hold if (int(x) * inv2) % q not in s_prev]
        bad4 = [int(x) for x in hold if (int(x) * inv4) % q not in s_prev]
        print(f"  inclusion H({j}) in 2H({j-1}): {'OK' if not bad2 else f'VIOLATED {bad2[:3]}'} ; "
              f"in 4H({j-1}): {'OK' if not bad4 else f'VIOLATED {bad4[:3]}'}", flush=True)
    prev_hold = hold
np.save('holdouts_l19_j22.npy', prev_hold)
print("done")
