#!/usr/bin/env python3
"""Verifies Proposition 17 (Near-extinction: forward containment).

H(l,j*(l)-1) subset 2*{x in H(l,j*(l)-2) : x = 2 (mod 3)}, proven directly
from Theorem 9 (holdout doubling inclusion) and Theorem 16 (last-holdout
parity); no new machinery. This checks the containment (and, for context,
the full empirical equality of Empirical Result 18) against the exact
holdout sets.
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


JSTAR = {2: 4, 3: 6, 4: 7, 5: 9, 6: 10, 7: 11}

for l in range(2, 8):
    q = 3 ** l
    J = JSTAR[l]
    HJ1 = H(l, J - 1)
    HJ2 = H(l, J - 2)
    target = set(x for x in HJ2 if x % 3 == 2)
    rhs = set((2 * x) % q for x in target)
    subset_ok = HJ1 <= rhs
    equal_ok = HJ1 == rhs
    print(f"l={l} J={J}: Proposition 17 (subset): {subset_ok}; "
          f"Empirical Result 18 (equality): {equal_ok}")
