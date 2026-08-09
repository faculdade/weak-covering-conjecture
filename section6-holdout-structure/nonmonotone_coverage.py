#!/usr/bin/env python3
"""Pointwise coverage is NOT monotone in the budget j: explicit counterexample.

Residue r = 128368 mod 3^13 is covered by R_{j-1,j} for j=13,14, NOT covered for j=15,16,
and covered again for j=17. Verified two ways: the coverage DP, and an explicit budget-14
witness (printed below, checkable by hand: 14 strictly decreasing exponents <= 27,
F = sum 2^(alpha_i) 3^i == 128368 mod 3^13).

Context: r is the 'fresh defect' whose x4 image is the last holdout 513472 of level 13
(see README). Its budget-14 witnesses all contain exponent 0 ('0-locked'), which is exactly
the case where no residue-preserving extension to budget 15 exists (prepending a smaller
exponent is impossible, and every other insertion changes the residue).

Consequence worth recording: the folklore 'coverage is monotone in j' used to justify
bisection in jstar-fast is FALSE pointwise; what is true (and is proved by this round's
Theorem 1) is set-level monotonicity of full coverage: H(l,j) empty => H(l,j+1) subset
2H(l,j) empty. The bisection is therefore sound, but for a nontrivial reason.
"""
import numpy as np
from verify_witness_maps_and_inclusions import coverage

l, q, r = 13, 3 ** 13, 128368
for j in range(13, 18):
    print(f"budget j={j}: covered({r}) = {bool(coverage(l, j)[r])}")

alphas = [26, 23, 22, 16, 15, 14, 13, 9, 6, 5, 4, 3, 1, 0]
assert all(alphas[i] > alphas[i + 1] for i in range(len(alphas) - 1))
assert len(alphas) == 14 and alphas[0] <= 2 * 14 - 1
F = sum((1 << a) * 3 ** i for i, a in enumerate(alphas))
print(f"explicit budget-14 witness {alphas}: F mod 3^13 = {F % q} (target {r}, "
      f"match={F % q == r})")
