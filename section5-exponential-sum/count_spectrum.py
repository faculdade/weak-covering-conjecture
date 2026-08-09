#!/usr/bin/env python3
"""Exact primitive-frequency tails for R_{m-1,m} modulo 3^l.

The C++ helper enumerates every decreasing m-tuple once and constructs the
exact residue histogram.  One FFT then evaluates the complete spectrum.  This
is O(binomial(2m,m) + 3^l log(3^l)), rather than doing a separate tuple sum at
each of the 3^l frequencies.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import tempfile
import time

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "histogram.cpp"
BINARY = HERE / "histogram"
ALPHA = math.log(3, 4)
DEFAULT_THRESHOLDS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                      0.0005, 0.0002, 0.0001)
THRESHOLD_GUARD = 1e-10


def compile_helper() -> None:
    if BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return
    subprocess.run(
        ["g++", "-O3", "-std=c++17", "-o", str(BINARY), str(SOURCE)],
        check=True,
    )


def choose_m(l: int, scale: str) -> int:
    if scale == "bare":
        return math.ceil(ALPHA * l)
    if scale == "counting":
        return math.ceil(ALPHA * l + 0.5 * math.log(l, 4))
    if scale == "plus2":
        return math.ceil(ALPHA * l) + 2
    raise ValueError(scale)


def spectrum(l: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    q = 3**l
    total = math.comb(2 * m, m)
    with tempfile.TemporaryDirectory(prefix="exceptional-frequency-") as tmp:
        histogram_path = pathlib.Path(tmp) / "histogram.u32"
        subprocess.run(
            [str(BINARY), str(l), str(m), str(histogram_path)], check=True
        )
        histogram_u32 = np.fromfile(histogram_path, dtype=np.uint32)
    if histogram_u32.size != q:
        raise RuntimeError(f"expected {q} bins, got {histogram_u32.size}")
    if int(histogram_u32.sum(dtype=np.uint64)) != total:
        raise RuntimeError("histogram mass does not equal binomial(2m,m)")

    histogram = histogram_u32.astype(np.float64)
    transform = np.fft.fft(histogram)
    # numpy uses the opposite sign from S(t), which does not change magnitudes.
    magnitudes = np.abs(transform) / total
    primitive_t = np.flatnonzero(np.arange(q) % 3 != 0)

    # Parseval is an independent numerical check on the transform.
    lhs = float(np.vdot(transform, transform).real)
    rhs = float(q * np.dot(histogram, histogram))
    if not math.isclose(lhs, rhs, rel_tol=3e-12):
        raise RuntimeError(f"Parseval check failed: {lhs} versus {rhs}")
    return primitive_t, magnitudes[primitive_t]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l-min", type=int, default=5)
    parser.add_argument("--l-max", type=int, default=14)
    parser.add_argument(
        "--scale", choices=("bare", "counting", "plus2"), default="counting"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--rms-multipliers",
        type=float,
        nargs="*",
        default=(),
        help="also count above c/sqrt(binomial(2m,m)) for each supplied c",
    )
    args = parser.parse_args()
    compile_helper()

    labels = " ".join(f"N>{eta:g}" for eta in args.thresholds)
    rms_labels = " ".join(f"N>{c:g}/sqrtT" for c in args.rms_multipliers)
    print(f"scale={args.scale}")
    print(
        f"l m e T/q max primitive_L1 primitive_RMS*sqrtT "
        f"{labels} {rms_labels} seconds",
        flush=True,
    )
    for l in range(args.l_min, args.l_max + 1):
        started = time.monotonic()
        m = choose_m(l, args.scale)
        q = 3**l
        total = math.comb(2 * m, m)
        _, values = spectrum(l, m)
        separations = [float(np.min(np.abs(values - eta)))
                       for eta in args.thresholds]
        if min(separations) <= THRESHOLD_GUARD:
            raise RuntimeError(
                "a coefficient is too close to a requested threshold for a "
                "reliable double-precision count"
            )
        counts = [int(np.count_nonzero(values > eta)) for eta in args.thresholds]
        rms_counts = [
            int(np.count_nonzero(values > c / math.sqrt(total)))
            for c in args.rms_multipliers
        ]
        fields = [
            l,
            m,
            f"{m - ALPHA * l:.6f}",
            f"{total / q:.9g}",
            f"{values.max():.9g}",
            f"{values.sum():.9g}",
            f"{math.sqrt(float(np.mean(values**2)) * total):.9g}",
            *counts,
            *rms_counts,
            f"{time.monotonic() - started:.3f}",
        ]
        print(" ".join(map(str, fields)), flush=True)


if __name__ == "__main__":
    main()
