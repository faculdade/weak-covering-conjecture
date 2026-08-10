#!/usr/bin/env python3
"""Finest-depth (c=l-1) local intensity check, leave-one-out, Section 5.3.

local_intensity.py checks depths c=8,9,10 only. At those depths, with l up
to 15, the averaging class is large (3^(l-c) points), which dilutes local
information and lets lambda_c(z) run into the thousands even for a holdout.
This script checks the finest depth the paper's own definition allows short
of the degenerate case c=l (a class of only 3 siblings, c=l-1), computing
each residue's own local intensity with its own hit count excluded from the
average (leave-one-out), to avoid the self-referential bias a class that
small would otherwise carry for every residue, not just holdouts.

Reports, per level: the expected total hole count sum_z exp(-lambda_c(z))
under a non-homogeneous Poisson model against the actual observed count,
and, for l=12,13,14, every unit's rank by this intensity, to locate where
the actual holdouts fall in the low-intensity tail.
"""
from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_inverse import make_histogram
from local_intensity import JSTAR


def depth_c_intensity_loo(hist, l: int, c: int, z: int) -> float:
    r = l - c
    coarse_mod = 3 ** c
    y = z % coarse_mod
    n = 3 ** r
    total = sum(int(hist[d * coarse_mod + y]) for d in range(n)) - int(hist[z])
    return total / (n - 1)


def main() -> None:
    do_rank = "--rank" in sys.argv
    levels = range(10, 15) if not do_rank else (12, 13, 14)
    for l in levels:
        j = JSTAR[l]
        m = j - 1
        q = 3 ** l
        hist = make_histogram(l, m)
        units = [z for z in range(q) if z % 3 != 0]
        holdouts = [z for z in units if hist[z] == 0]
        c = l - 1
        if do_rank:
            intens = sorted((depth_c_intensity_loo(hist, l, c, z), z) for z in units)
            rank_of = {z: i + 1 for i, (_, z) in enumerate(intens)}
            hold_ranks = sorted(rank_of[z] for z in holdouts)
            lowest20_covered = all(hist[z] > 0 for _, z in intens[:20])
            print(f"l={l} |units|={len(units)} |holdouts|={len(holdouts)} "
                  f"holdout_ranks={hold_ranks} lowest20_all_covered={lowest20_covered}")
        else:
            expected = sum(
                math.exp(-depth_c_intensity_loo(hist, l, c, z)) for z in units
            )
            print(f"l={l} j={j} m={m} |holdouts|={len(holdouts)} c={c} "
                  f"expected_holes_LOO={expected:.4f}")


if __name__ == "__main__":
    main()
