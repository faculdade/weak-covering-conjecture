#!/usr/bin/env python3
"""Local-intensity-only null for the l=14 phase-scramble diagnostic, Section 5.3.

The constrained phase null (constrained_phase_null.py) fixes every |S(t)|,
i.e. every piece of phase information, and asks whether the actual phases
still look exchangeable once the zero-off-units constraint is respected.
That null carries no information about the level-(l-1) local intensity
documented earlier in Section 5.3 (lambda_c ranging from 2.0 to 108.4
against a global mean in the thousands).

This script builds the opposite null: keep the level-(l-1) parent count
N_{l-1}(r) exactly as observed, for every parent r, and split it
multinomially (1/3,1/3,1/3) among its three lifts, discarding all
information about the primitive-frequency phases. If this null alone
reproduces the actual array's max/RMS statistic, the observed extremity
does not need phase structure to explain it: pure local-intensity skew
already accounts for it.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from analyze_inverse import make_histogram


def multinomial_null(l: int, m: int, trials: int, seed: int) -> None:
    q = 3**l
    q_parent = q // 3
    hist = make_histogram(l, m).astype(np.int64)
    parents = hist[:q_parent] + hist[q_parent:2*q_parent] + hist[2*q_parent:]

    lift_imbalance = 3 * hist.astype(np.float64) - np.tile(
        parents.astype(np.float64), 3
    )
    actual_max = float(np.max(np.abs(lift_imbalance)))
    actual_rms = float(np.sqrt(np.mean(lift_imbalance**2)))
    actual_ratio = actual_max / actual_rms
    print(f"actual: max={actual_max:.6g} rms={actual_rms:.6g} "
          f"ratio={actual_ratio:.6g}")
    print(f"parents: mean={parents.mean():.4f} max={parents.max()} "
          f"n={parents.size}")

    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(trials):
        n1 = rng.binomial(parents, 1.0 / 3.0)
        rem = parents - n1
        n2 = rng.binomial(rem, 0.5)
        n3 = rem - n2
        sim_hist = np.concatenate([n1, n2, n3]).astype(np.float64)
        sim_imbalance = 3 * sim_hist - np.tile(parents.astype(np.float64), 3)
        max_sim = float(np.max(np.abs(sim_imbalance)))
        rms_sim = float(np.sqrt(np.mean(sim_imbalance**2)))
        ratios.append(max_sim / rms_sim)

    ratios = np.asarray(ratios)
    print(
        f"multinomial (local-intensity-only) null ({trials} trials, "
        f"seed={seed}): mean={float(np.mean(ratios)):.4f} "
        f"range=[{float(np.min(ratios)):.4f},{float(np.max(ratios)):.4f}] "
        f"stderr={float(np.std(ratios) / math.sqrt(trials)):.4f}"
    )
    exceed = float(np.mean(ratios > actual_ratio))
    print(f"fraction of null trials exceeding the actual ratio: {exceed:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=14)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    multinomial_null(args.l, args.m, args.trials, args.seed)
