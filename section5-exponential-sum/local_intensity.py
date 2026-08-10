#!/usr/bin/env python3
"""Depth-c local intensity of the last holdouts, Section 5.3.

For a residue z modulo 3^l, its depth-c cell is its class modulo 3^c
(c < l). The depth-c local intensity lambda_c(z) is the total exact hit
count of R_{m-1,m} within that class, divided by 3^(l-c) (the class
size): the rate a uniform spread of hits over the class would give. A
non-homogeneous Poisson model fit to that intensity assigns z hole
probability exp(-lambda_c(z)).

This reconstructs, with a precise definition, the "cell / local
intensity / conductor-depth" language used informally in earlier
project notes, and reports it at depths c=8,9,10 for the exact
last-holdout set H(l,j*(l)-1), l=10,...,15, the numbers cited in the
paper.

l=15 (m=19, T=C(38,19)=35,345,263,800) takes roughly 10-15 minutes on
this project's machine; l=10..14 finish in well under a minute total.
"""
from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_inverse import make_histogram

JSTAR = {10: 15, 11: 16, 12: 17, 13: 18, 14: 19, 15: 20}
DEPTHS = (8, 9, 10)


def depth_c_intensity(hist, l: int, c: int, z: int) -> float:
    r = l - c
    coarse_mod = 3 ** c
    y = z % coarse_mod
    total = sum(int(hist[d * coarse_mod + y]) for d in range(3 ** r))
    return total / (3 ** r)


def main() -> None:
    levels = range(10, 16) if "--with15" in sys.argv else range(10, 15)
    for l in levels:
        j = JSTAR[l]
        m = j - 1
        q = 3 ** l
        hist = make_histogram(l, m)
        units = range(0, q, 1)
        holdouts = [z for z in units if z % 3 != 0 and hist[z] == 0]
        print(f"l={l} m={m} q={q} #holdouts={len(holdouts)} holdouts={holdouts}")
        for z in holdouts:
            for c in DEPTHS:
                if c >= l:
                    print(f"    c={c}: out of range (l={l})")
                    continue
                lam = depth_c_intensity(hist, l, c, z)
                print(f"    z={z} c={c}: lambda={lam:.4f} hole_probability=e^-{lam:.4f}")


if __name__ == "__main__":
    main()
