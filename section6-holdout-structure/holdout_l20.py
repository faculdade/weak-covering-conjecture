#!/usr/bin/env python3
"""H-013 extension: exact holdout sets at l=20, budgets 22,23 (j*(20)=24), bigint-bitset DP
(3^20 bools would need 84GB as bytes; as bits it is 436MB/layer)."""
import sys

def coverage_bigint(l, j):
    q = 3 ** l
    mask = (1 << q) - 1
    dp = [0] * (j + 1)
    dp[0] = 1
    for a in range(2 * j - 1, -1, -1):
        c_hi = min(j - 1, 2 * j - 1 - a)
        c_lo = max(0, j - 1 - a)
        for c in range(c_hi, c_lo - 1, -1):
            t = (pow(2, a, q) * pow(3, c, q)) % q
            src = dp[c]
            if src:
                rolled = ((src << t) | (src >> (q - t))) & mask if t else src
                dp[c + 1] |= rolled
        print(f"  a={a} done", file=sys.stderr, flush=True)
    return dp[j]

def bits_to_list(x, q):
    """Set-bit positions of a q-bit int, via numpy (the per-bit shift loop is quadratic)."""
    import numpy as np
    nbytes = (q + 7) // 8
    arr = np.frombuffer(x.to_bytes(nbytes, 'little'), dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='little')[:q]
    return [int(i) for i in np.nonzero(bits)[0]]

def maxrun_chains(s, q):
    best, best_start = 0, None
    inv2 = pow(2, -1, q)
    for h in s:
        if (h * inv2) % q not in s:
            r, x = 1, h
            while (x * 2) % q in s:
                x = (x * 2) % q
                r += 1
            if r > best:
                best, best_start = r, h
    return best, best_start

l = 20
q = 3 ** l
JSTAR_L = 24
prev = None
for j in (21, 22, 23):
    cov = coverage_bigint(l, j)
    # holdouts: units (not div by 3) not covered
    units_pat = int(''.join('011' * 1), 2)  # per 3 bits: positions 0 mod 3 excluded
    # build units mask as bigint: bits set at all x with x%3!=0
    # pattern LSB-first per 3 bits: x=0 -> bit0 (excluded), x=1,2 included -> bits 1,2 set: 0b110
    chunk = 0b110
    units = 0
    step = 3
    # build via repeated doubling of pattern
    pat, bits = chunk, 3
    while bits < q:
        pat = pat | (pat << bits)
        bits *= 2
    units = pat & ((1 << q) - 1)
    holdbits = units & ~cov
    hold = bits_to_list(holdbits, q)
    s = set(hold)
    mr, start = maxrun_chains(s, q)
    cls = sorted(set(x % 729 for x in hold))
    print(f"l=20 j={j}: |H|={len(hold)}, maxrun={mr} (chain start {start}), "
          f"bootstrap j*<={j+mr} (true {JSTAR_L}) "
          f"{'TIGHT' if j+mr == JSTAR_L else ('OK' if j+mr > JSTAR_L else 'FALSIFIED')}", flush=True)
    print(f"  mod-3^6 classes ({len(cls)}): {cls[:40]}{'...' if len(cls) > 40 else ''}", flush=True)
    if len(hold) <= 10:
        print(f"  residues: {hold}", flush=True)
    if prev is not None:
        inv2, inv4 = pow(2, -1, q), pow(4, -1, q)
        bad2 = [x for x in hold if (x * inv2) % q not in prev]
        bad4 = [x for x in hold if (x * inv4) % q not in prev]
        print(f"  inclusion H({j}) in 2H({j-1}): {'OK' if not bad2 else f'VIOLATED {bad2[:3]}'} ; "
              f"in 4H({j-1}): {'OK' if not bad4 else f'VIOLATED {bad4[:3]}'}", flush=True)
    prev = s
print("done")
