#!/usr/bin/env python3
"""Inverse-Fourier diagnostics for the R_{m-1,m} hit counts.

The C++ helper constructs the exact hit-count histogram N_l(z).  This script
also Fourier transforms and inverts that histogram, both as a check and to
compare the triangle-inequality envelope with the actual signed primitive-
frequency sum.

For q = 3^l, the nonprimitive frequencies t = 3s recover one third of the
level-(l-1) parent count.  Consequently the primitive-frequency sum is exactly

    sum_{3 does not divide t} S_l(t) e(-tz/q)
      = (q/3) (3 N_l(z) - N_{l-1}(z mod q/3)).

This identity is the relevant one for testing cancellation between primitive
frequencies: it measures imbalance among the three lifts of each parent, not
deviation from a global mean.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import tempfile
import time

import numpy as np

from count_spectrum import BINARY, choose_m, compile_helper


def make_histogram(l: int, m: int) -> np.ndarray:
    q = 3**l
    total = math.comb(2 * m, m)
    with tempfile.TemporaryDirectory(prefix="inverse-frequency-") as tmp:
        path = pathlib.Path(tmp) / "histogram.u32"
        subprocess.run([str(BINARY), str(l), str(m), str(path)], check=True)
        histogram = np.fromfile(path, dtype=np.uint32)
    if histogram.size != q:
        raise RuntimeError(f"expected {q} bins, got {histogram.size}")
    if int(histogram.sum(dtype=np.uint64)) != total:
        raise RuntimeError("histogram mass mismatch")
    return histogram


def analyze(l: int, m: int, do_fft: bool, scramble_trials: int, seed: int) -> None:
    started = time.monotonic()
    q = 3**l
    q_parent = q // 3
    total = math.comb(2 * m, m)
    histogram = make_histogram(l, m)
    print(f"l={l} m={m} q={q} T={total} T/q={total/q:.12g}")

    # Every represented value is a unit.  Use the unit-space mean when
    # comparing occupancy with the uniform random-balls benchmark.
    units = np.concatenate((histogram[1::3], histogram[2::3]))
    unit_count = 2 * q_parent
    unit_mean = total / unit_count
    unit_zeros = int(np.count_nonzero(units == 0))
    poisson_zero_expectation = unit_count * math.exp(-unit_mean)
    exact_random_zero_expectation = unit_count * math.exp(
        total * math.log1p(-1 / unit_count)
    )
    print(
        "units: "
        f"mean={unit_mean:.12g} min={int(units.min())} "
        f"max={int(units.max())} zeros={unit_zeros} "
        f"zero_fraction={unit_zeros/unit_count:.12g} "
        f"std={float(np.std(units)):.12g} "
        f"Fano={float(np.var(units))/unit_mean:.12g}"
    )
    print(
        "iid-uniform occupancy: "
        f"expected_zeros_exact={exact_random_zero_expectation:.12g} "
        f"poisson_approx={poisson_zero_expectation:.12g} "
        f"actual/expected={unit_zeros/exact_random_zero_expectation:.12g}"
    )

    # Aggregate the three lifts to obtain N_{l-1} for the same tuple family.
    parents = (
        histogram[:q_parent].astype(np.uint64)
        + histogram[q_parent : 2 * q_parent]
        + histogram[2 * q_parent :]
    )
    lift_imbalance = 3 * histogram.astype(np.int64) - np.tile(parents, 3)
    primitive_signed_scale = q_parent
    primitive_max_signed = (
        int(np.max(np.abs(lift_imbalance))) * primitive_signed_scale
    )
    parent_positive = parents > 0
    complete_lifts = (
        (histogram[:q_parent] > 0)
        & (histogram[q_parent : 2 * q_parent] > 0)
        & (histogram[2 * q_parent :] > 0)
    )
    print(
        "lifting: "
        f"positive_parents={int(np.count_nonzero(parent_positive))} "
        f"complete_lifts={int(np.count_nonzero(complete_lifts))} "
        f"defective_positive_parents="
        f"{int(np.count_nonzero(parent_positive & ~complete_lifts))} "
        f"max_abs(3N-parent)={int(np.max(np.abs(lift_imbalance)))}"
    )
    sample_indices = [1, 2]
    for residue_class in (1, 2):
        class_counts = histogram[residue_class::3]
        zero_offsets = np.flatnonzero(class_counts == 0)
        if zero_offsets.size:
            sample_indices.append(residue_class + 3 * int(zero_offsets[0]))
        sample_indices.append(
            residue_class + 3 * int(np.argmax(class_counts))
        )
        class_imbalance = np.abs(lift_imbalance[residue_class::3])
        sample_indices.append(
            residue_class + 3 * int(np.argmax(class_imbalance))
        )
    print("sample counts (z:N,parent,primitive_delta,unit_mean_delta):")
    for z in dict.fromkeys(sample_indices):
        parent = int(parents[z % q_parent])
        count = int(histogram[z])
        print(
            f"  {z}:{count},{parent},{count-parent/3:.12g},"
            f"{count-unit_mean:.12g}"
        )

    if not do_fft:
        print(f"seconds={time.monotonic() - started:.3f}", flush=True)
        return

    histogram_float = histogram.astype(np.float64)
    transform = np.fft.fft(histogram_float)
    parseval_lhs = float(np.vdot(transform, transform).real)
    parseval_rhs = float(q * np.dot(histogram_float, histogram_float))
    if not math.isclose(parseval_lhs, parseval_rhs, rel_tol=3e-12):
        raise RuntimeError("Parseval check failed")

    primitive_1 = transform[1::3]
    primitive_2 = transform[2::3]
    primitive_l1 = float(np.abs(primitive_1).sum() + np.abs(primitive_2).sum())
    primitive_l2_sq = float(
        np.vdot(primitive_1, primitive_1).real
        + np.vdot(primitive_2, primitive_2).real
    )
    primitive_rms_signed = math.sqrt(primitive_l2_sq)

    # FFT sign conventions only permute/conjugate the frequencies.  Inverting
    # the transform must recover the exact direct histogram.
    recovered = np.fft.ifft(transform)
    inverse_max_imag = float(np.max(np.abs(recovered.imag)))
    inverse_max_error = float(
        np.max(np.abs(recovered.real - histogram_float))
    )
    recovered_rounded = np.rint(recovered.real).astype(np.int64)
    inverse_mismatches = int(
        np.count_nonzero(recovered_rounded != histogram.astype(np.int64))
    )

    # The RMS is not a model prediction: by Parseval it is identically the
    # RMS over z of the actual signed primitive sum.  The Gaussian benchmark
    # concerns only the extreme/RMS ratio.
    actual_primitive_rms = primitive_signed_scale * math.sqrt(
        float(np.mean(lift_imbalance.astype(np.float64) ** 2))
    )
    gaussian_max_multiplier = math.sqrt(2 * math.log(2 * q))
    print(
        "primitive Fourier: "
        f"L1={primitive_l1:.12g} L2={primitive_rms_signed:.12g} "
        f"actual_max_signed={primitive_max_signed} "
        f"max/L1={primitive_max_signed/primitive_l1:.12g} "
        f"max/RMS={primitive_max_signed/primitive_rms_signed:.12g} "
        f"iid_Gaussian_max/RMS~{gaussian_max_multiplier:.12g}"
    )
    print(
        "Parseval signed-RMS identity: "
        f"spectrum_RMS={primitive_rms_signed:.12g} "
        f"residue_RMS={actual_primitive_rms:.12g} "
        f"ratio={actual_primitive_rms/primitive_rms_signed:.12g}"
    )
    print(
        "inverse FFT check: "
        f"max_real_error={inverse_max_error:.12g} "
        f"max_imag={inverse_max_imag:.12g} "
        f"rounded_mismatches={inverse_mismatches}"
    )

    if scramble_trials:
        # Null model: keep every |S(t)| fixed but randomize the primitive
        # phases independently, subject only to conjugate symmetry.  Add the
        # resulting primitive component to the exact lower-level baseline.
        # The output is continuous rather than an integer count; <= 0 is the
        # natural analogue of a failure of positivity.
        del recovered
        rng = np.random.default_rng(seed)
        half_t = np.arange(1, (q + 1) // 2, dtype=np.int64)
        half_t = half_t[half_t % 3 != 0]
        half_magnitudes = np.abs(transform[half_t])
        baseline = np.tile(parents.astype(np.float64) / 3, 3)
        maxima = []
        rms_values = []
        nonpositive_units = []
        for _ in range(scramble_trials):
            phases = rng.uniform(0, 2 * math.pi, half_t.size)
            randomized = np.zeros_like(transform)
            randomized[half_t] = half_magnitudes * np.exp(1j * phases)
            randomized[q - half_t] = np.conjugate(randomized[half_t])
            random_delta = np.fft.ifft(randomized).real
            maxima.append(float(np.max(np.abs(random_delta))))
            rms_values.append(float(np.sqrt(np.mean(random_delta**2))))
            pseudo_counts = baseline + random_delta
            nonpositive_units.append(
                int(np.count_nonzero(pseudo_counts[1::3] <= 0))
                + int(np.count_nonzero(pseudo_counts[2::3] <= 0))
            )
        ratios = np.asarray(maxima) / np.asarray(rms_values)
        print(
            f"phase-scramble null ({scramble_trials} trials, seed={seed}): "
            f"max/RMS mean={float(np.mean(ratios)):.12g} "
            f"range=[{float(np.min(ratios)):.12g},{float(np.max(ratios)):.12g}] "
            f"nonpositive_units mean={float(np.mean(nonpositive_units)):.12g} "
            f"range=[{min(nonpositive_units)},{max(nonpositive_units)}]"
        )
    print(f"seconds={time.monotonic() - started:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int)
    parser.add_argument(
        "--scale", choices=("bare", "counting", "plus2"), default="counting"
    )
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument("--scramble-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    compile_helper()
    analyze(
        args.l,
        args.m if args.m is not None else choose_m(args.l, args.scale),
        not args.no_fft,
        args.scramble_trials,
        args.seed,
    )


if __name__ == "__main__":
    main()
