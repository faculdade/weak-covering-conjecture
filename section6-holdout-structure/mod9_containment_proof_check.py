#!/usr/bin/env python3
"""Verifies Proposition 18 (mod-9 two-class containment) two ways.

1. The algebraic identity {4^(J-1)+3, 4^(J-1)+6} = {4^J, 4^(J+1)} (mod 9) for
   every J, which is the last step of the proposition's proof.
2. Direct enumeration: H(l, j*(l)-1) mod 9 is contained in {4^J, 4^(J+1)} at
   every l checked (matches Empirical Result 19's range, l=3..16, run via
   ../section6-holdout-structure/mod9_class_law.py for l=5..16; l=3,4 checked
   here directly since mod9_class_law.py starts at l=5).
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


print("Algebraic identity check: {4^(J-1)+3, 4^(J-1)+6} == {4^J, 4^(J+1)} (mod 9)")
for J in range(1, 15):
    term0 = pow(2, 2 * J - 2, 9)
    candidates = {(term0 + 3) % 9, (term0 + 6) % 9}
    claimed = {pow(4, J, 9), pow(4, J + 1, 9)}
    print(f"  J={J}: candidates={candidates}, claimed={claimed}, match={candidates == claimed}")

print("\nDirect check at l=3,4 (mod9_class_law.py covers l=5..16)")
JSTAR = {3: 6, 4: 7}
for l in (3, 4):
    J = JSTAR[l]
    Hj = H(l, J - 1)
    m9 = sorted(set(x % 9 for x in Hj))
    fourJ, fourJ1 = pow(4, J, 9), pow(4, J + 1, 9)
    contained = set(m9) <= {fourJ, fourJ1}
    excludes_fourJ = fourJ not in m9
    print(f"  l={l} J={J}: H mod 9 = {m9}, contained in {{{fourJ},{fourJ1}}}: {contained}, "
          f"excludes 4^J: {excludes_fourJ}")
