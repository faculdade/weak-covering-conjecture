#!/usr/bin/env python3
"""Verifies Proposition 22 (corner-redundancy implies tightness).

Checks, directly against the exact H(l,j) sets, that whenever corner-redundancy
holds at every relevant width (verified separately, l=3..13, by
round5_width_reformulation.py), maxrun(H(l,j)) decreases by exactly 1 at each
budget step from j to j*(l), which is exactly the claim
j*(l) = j + maxrun(H(l,j)) proven in the paper.

Also independently verifies the underlying set identity
D(l,W+1) = D(l,W) ∩ 2*D(l,W) (equivalent to corner-redundancy at width W) by
direct enumeration, matching the paper's Proposition 22 proof.
"""
from itertools import combinations


def R(j):
    vals = set()
    for tup in combinations(range(2 * j), j):
        tup = sorted(tup, reverse=True)
        v = sum(2 ** b * 3 ** i for i, b in enumerate(tup))
        vals.add(v)
    return vals


def H(l, j):
    q = 3 ** l
    units = set(x for x in range(q) if x % 3 != 0)
    covered = set(v % q for v in R(j))
    return units - covered


def U(l, W):
    q = 3 ** l
    s = set()
    for tup in combinations(range(W + 1), l):
        tup = sorted(tup, reverse=True)
        s.add(sum((2 ** b) * (3 ** i) for i, b in enumerate(tup)) % q)
    return s


def D(l, W):
    q = 3 ** l
    units = set(x for x in range(q) if x % 3 != 0)
    return units - U(l, W)


def maxrun(Hset, q):
    if not Hset:
        return 0
    best = 0
    Hset_ = set(Hset)
    for x in Hset:
        length = 1
        cur = (x * 2) % q
        while cur in Hset_:
            length += 1
            cur = (cur * 2) % q
        best = max(best, length)
    return best


JSTAR = {3: 6, 4: 7, 5: 9, 6: 10, 7: 11, 8: 12}

print("Set-level check: D(l,W+1) = D(l,W) cap 2*D(l,W) at every corner-redundant width")
for l in range(3, 8):
    for W in range(2 * l + 1, 2 * l + 6):
        q = 3 ** l
        DW, DW1 = D(l, W), D(l, W + 1)
        rhs = DW & set((2 * x) % q for x in DW)
        print(f"  l={l} W={W}: D(l,W+1)==D(l,W) cap 2D(l,W): {DW1 == rhs}")

print("\nmaxrun-level check: exact -1 decrement from j to j*(l), l+2<=j<=j*(l)")
for l in range(3, 9):
    q = 3 ** l
    for j in range(l + 2, JSTAR[l] + 1):
        seq = [maxrun(H(l, jp), q) for jp in range(j, JSTAR[l] + 1)]
        ok = all(seq[i + 1] == seq[i] - 1 for i in range(len(seq) - 1))
        tight = (j + seq[0] == JSTAR[l])
        print(f"  l={l} j={j}: maxrun sequence {seq}, exact -1 each step: {ok}, "
              f"j+maxrun(H(l,j))=j*(l): {tight}")
