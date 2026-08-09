#!/usr/bin/env python3
"""Split the primitive-frequency L1 mass of S_l(t) at a fixed threshold.

Section 5 of the paper quotes, at l=18, the number of primitive frequencies with
|S(t)|/T above 0.001 and the share of the total L1 norm they carry. This script
prints both, reusing count_spectrum.spectrum for the histogram and the FFT.

    python3 l1_tail.py --l 18 --threshold 0.001
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from count_spectrum import choose_m, compile_helper, spectrum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--scale", choices=("bare", "counting", "plus2"),
                        default="counting")
    parser.add_argument("--threshold", type=float, default=0.001)
    args = parser.parse_args()

    compile_helper()
    m = args.m if args.m is not None else choose_m(args.l, args.scale)
    _, values = spectrum(args.l, m)

    total_l1 = float(values.sum())
    above = values[values > args.threshold]
    tail_l1 = float(above.sum())

    print(f"l={args.l} m={m} T=C(2m,m)={math.comb(2 * m, m)}")
    print(f"primitive frequencies            {values.size}")
    print(f"max |S(t)|/T                     {values.max():.9g}")
    print(f"primitive L1 of |S(t)|/T         {total_l1:.9g}")
    print(f"count with |S(t)|/T > {args.threshold:g}       {above.size}")
    print(f"their L1 contribution            {tail_l1:.9g}")
    print(f"share of the L1 norm             {tail_l1 / total_l1:.6%}")


if __name__ == "__main__":
    main()
