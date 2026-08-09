#!/usr/bin/env python3
"""Same witness-level check for (7,11) -> (8,12): does worst repair cost grow with l?"""
import math
from itertools import combinations
import numpy as np

def enumerate_witnesses(j, mod):
    n = math.comb(2 * j, j)
    res = np.empty(n, dtype=np.int64)
    dep = np.empty(n, dtype=np.int8)
    msk = np.empty(n, dtype=np.int64)
    pow3 = [3 ** i for i in range(j)]
    k = 0
    for comb in combinations(range(2 * j), j):
        F = 0
        for i in range(j):
            F += (1 << comb[j - 1 - i]) * pow3[i]
        res[k] = F % mod
        dep[k] = sum(1 for b in comb if b >= j)
        m = 0
        for b in comb:
            m |= 1 << b
        msk[k] = m
        k += 1
    return res, dep, msk

def popcount64(arr):
    x = arr.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

l = 7
jp, jc = 11, 12
q_par, q_child = 3 ** l, 3 ** (l + 1)

pres7, pdep, pmask = enumerate_witnesses(jp, q_child)
cres7, cdep, cmask = enumerate_witnesses(jc, q_child)
print(f"enumerated {len(pres7)} (j={jp}) and {len(cres7)} (j={jc}) tuples")
pres6 = pres7 % q_par
units6 = np.array([x for x in range(q_par) if x % 3 != 0])

dmin6 = np.full(q_par, 127, dtype=np.int8); np.minimum.at(dmin6, pres6, pdep)
dmin7 = np.full(q_child, 127, dtype=np.int8); np.minimum.at(dmin7, cres7, cdep)

deltas, viol = {}, 0
for p in units6:
    for eps in range(3):
        dd = int(dmin7[p + eps * q_par]) - int(dmin6[p])
        deltas[dd] = deltas.get(dd, 0) + 1
        viol += dd > 1
print(f"B. Delta d distribution: {dict(sorted(deltas.items()))}; violations of Dd<=1: {viol}")
units7 = np.array([x for x in range(q_child) if x % 3 != 0])
print(f"B'. D_7={int(dmin6[units6].max())}  D_8={int(dmin7[units7].max())}")

hit7 = np.zeros(q_child, dtype=bool); hit7[pres7] = True
missing = [int(r) for p in units6 for r in (p, p+q_par, p+2*q_par) if not hit7[r]]
full = sum(1 for p in units6 if all(hit7[p+e*q_par] for e in range(3)))
print(f"C. full lift-mask parents: {full}/{len(units6)} ({100*full/len(units6):.2f}%); "
      f"missing children: {len(missing)}")

order_p = np.argsort(pres6, kind='stable'); order_c = np.argsort(cres7, kind='stable')
pstart = np.searchsorted(pres6[order_p], np.arange(q_par + 1))
cstart = np.searchsorted(cres7[order_c], np.arange(q_child + 1))

hist, worst, argmax, missing_detail = {}, 0, None, []
for p in units6:
    Pm = pmask[order_p[pstart[p]:pstart[p+1]]]
    for eps in range(3):
        r = p + eps * q_par
        Cm = cmask[order_c[cstart[r]:cstart[r+1]]]
        x = np.bitwise_xor.outer(Pm, Cm)
        d = int(popcount64(x).min())
        hist[d] = hist.get(d, 0) + 1
        if d > worst:
            worst, argmax = d, (int(p), int(r))
        if r in missing:
            missing_detail.append((int(p), int(r), d))
print(f"D. min-edit distribution: {dict(sorted(hist.items()))}")
print(f"   worst child: {argmax} at distance {worst}")
print(f"   missing children (parent, child, min-edit): {missing_detail}")
