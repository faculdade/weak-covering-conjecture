#!/usr/bin/env python3
"""Constrained phase-scramble null for the l=14, m=16 diagnostic.

The unconstrained scramble in analyze_inverse.py randomizes every primitive
phase independently, subject only to conjugate symmetry.  That destroys the
identically-zero-off-units property F(z)=0 for z divisible by 3.

This script builds a null that keeps that property.  F(z)=0 for all z=3s is
equivalent (by a DFT-duality argument on the subgroup 3(Z/qZ), verified by
hand before writing this script) to: for every "primitive parent" residue r
modulo 3^(l-1) with 3 not dividing r, the three lifted frequencies
t = r, r+3^(l-1), r+2*3^(l-1) satisfy S(t1)+S(t2)+S(t3) = 0.

With |S(t1)|,|S(t2)|,|S(t3)| fixed at their actual values (which do sum to
zero for some phase choice, since the real data achieves it), the solution
space for a non-degenerate triangle is exactly two families: a common
rotation of the actual phases, or a common rotation of their negation
(mirror reflection). Both preserve the zero sum, verified below.

F is real, so S(3^l-t) = conj(S(t)) must hold throughout (used already in
the paper, Section 5.1). The conjugate partner of the triple at parent r is
the triple at parent 3^(l-1)-r: only ONE triple per conjugate pair is given
an independent rotation and reflection; the partner's three phases are set
by conjugation, not drawn independently. Getting this wrong (randomizing
every triple independently, ignoring the conjugate pairing) produces a
complex-valued field and a materially different, wrong max/RMS statistic;
both the real-valuedness and the exact conjugate-symmetry relation are
checked directly below, not just assumed from the construction.
"""

from __future__ import annotations

import math

import numpy as np

from analyze_inverse import make_histogram


def constrained_scramble(l: int, m: int, trials: int, seed: int) -> None:
    q = 3**l
    q_parent = q // 3
    histogram = make_histogram(l, m).astype(np.float64)
    transform = np.fft.fft(histogram)

    # Actual primitive-frequency array, for reference.
    parents = (
        histogram[:q_parent] + histogram[q_parent : 2 * q_parent]
        + histogram[2 * q_parent :]
    )
    lift_imbalance = 3 * histogram - np.tile(parents, 3)
    actual_max = float(np.max(np.abs(lift_imbalance))) * q_parent
    actual_rms_all = float(np.sqrt(np.mean(lift_imbalance**2))) * q_parent
    actual_ratio = actual_max / actual_rms_all
    print(f"actual: max={actual_max:.6g} rms_all={actual_rms_all:.6g} "
          f"ratio={actual_ratio:.6g}")

    # Independent primitive parents: 0 < r < q_parent/2, 3 not dividing r.
    r_values = np.arange(1, q_parent // 2 + 1)
    r_values = r_values[r_values % 3 != 0]

    # Sanity check: verify the zero-sum identity on the ACTUAL data first.
    max_triple_sum_error = 0.0
    for r in r_values[:2000]:
        t1, t2, t3 = int(r), int(r + q_parent), int(r + 2 * q_parent)
        s = transform[t1] + transform[t2] + transform[t3]
        max_triple_sum_error = max(max_triple_sum_error, abs(s))
    print(f"triple-sum identity check (first 2000 r): "
          f"max|S(t1)+S(t2)+S(t3)|={max_triple_sum_error:.3e} "
          f"(scale: |S(t1)|~{abs(transform[int(r_values[0])]):.3e})")

    rng = np.random.default_rng(seed)
    ratios = []
    for trial in range(trials):
        randomized = np.zeros_like(transform)
        n = r_values.size
        reflections = rng.choice([1.0, -1.0], size=n)
        rotations = rng.uniform(0, 2 * math.pi, size=n)
        for idx, r in enumerate(r_values):
            t1, t2, t3 = int(r), int(r + q_parent), int(r + 2 * q_parent)
            s_refl = reflections[idx]
            phi = rotations[idx]
            for t in (t1, t2, t3):
                mag = abs(transform[t])
                beta = math.atan2(transform[t].imag, transform[t].real)
                new_phase = s_refl * beta + phi
                randomized[t] = mag * complex(math.cos(new_phase), math.sin(new_phase))
                randomized[q - t] = np.conjugate(randomized[t])

        # Verify construction: zero-sum per triple, exactly (float tolerance).
        if trial == 0:
            worst = 0.0
            for r in r_values:
                t1, t2, t3 = int(r), int(r + q_parent), int(r + 2 * q_parent)
                worst = max(worst, abs(randomized[t1] + randomized[t2] + randomized[t3]))
            print(f"constrained-null triple-sum check (trial 0): max|sum|={worst:.3e}")

            # Conjugate symmetry S(q-t)=conj(S(t)) is what keeps the inverse
            # transform real; check it directly over every entry actually
            # set, not just assumed from how randomized[q-t] was assigned.
            nonzero = np.flatnonzero(randomized)
            conj_err = max(
                abs(randomized[int(t)] - np.conjugate(randomized[(q - int(t)) % q]))
                for t in nonzero
            )
            print(f"constrained-null conjugate-symmetry check (trial 0): "
                  f"max|S(t)-conj(S(q-t))|={conj_err:.3e} over {nonzero.size} nonzero entries")

        raw_delta = np.fft.ifft(randomized)
        if trial == 0:
            max_imag = float(np.max(np.abs(raw_delta.imag)))
            max_real = float(np.max(np.abs(raw_delta.real)))
            print(f"constrained-null real-valuedness check (trial 0): "
                  f"max|Im|={max_imag:.3e} against max|Re|~{max_real:.3e}")
        random_delta = raw_delta.real
        # Verify off-units vanishing directly.
        if trial == 0:
            off_units = random_delta[0::3]
            print(f"constrained-null off-units max|F|={np.max(np.abs(off_units)):.3e} "
                  f"(scale: on-units max~{np.max(np.abs(random_delta)):.3e})")

        rms_all = float(np.sqrt(np.mean(random_delta**2)))
        max_all = float(np.max(np.abs(random_delta)))
        ratios.append(max_all / rms_all)

    ratios = np.asarray(ratios)
    mean_ratio = float(np.mean(ratios))
    print(
        f"constrained null ({trials} trials, seed={seed}): "
        f"max/RMS_all mean={mean_ratio:.6g} "
        f"range=[{float(np.min(ratios)):.6g},{float(np.max(ratios)):.6g}] "
        f"stderr={float(np.std(ratios)/math.sqrt(trials)):.4g}"
    )
    residual = actual_ratio / mean_ratio
    print(f"residual = actual_ratio / constrained_mean = {actual_ratio:.6g} / "
          f"{mean_ratio:.6g} = {residual:.6g}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=14)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    constrained_scramble(args.l, args.m, args.trials, args.seed)
