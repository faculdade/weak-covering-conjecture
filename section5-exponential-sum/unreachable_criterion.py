"""Verifies the paper's Proposition "The full-spectrum criterion is unreachable" (Section 5.1):
sum_{t!=0} S(t) = -T exactly, and S(3^(l-1)) + S(2*3^(l-1)) = -T exactly, for R_{j,k}, at every
(l,j,k) checked. Direct enumeration, cmath, no FFT: this is a check of an algebraic identity, not
a large-scale spectrum computation (see l1_tail.py / count_spectrum.py for those).
"""
import cmath
from itertools import combinations

CASES = [(1, 1, 1), (2, 1, 1), (3, 2, 1), (4, 3, 2), (2, 2, 2), (5, 4, 3)]

def S(l, j, k, t):
    mod = 3 ** l
    total = 0j
    for combo in combinations(range(j + k + 1), j + 1):
        a = sorted(combo, reverse=True)
        val = sum((2 ** a[i]) * (3 ** i) for i in range(j + 1))
        total += cmath.exp(2j * cmath.pi * t * val / mod)
    return total

def main():
    all_ok = True
    for (l, j, k) in CASES:
        mod = 3 ** l
        T = sum(1 for _ in combinations(range(j + k + 1), j + 1))
        s_full = sum(S(l, j, k, t) for t in range(mod))
        s_half = S(l, j, k, mod // 3)
        s_2half = S(l, j, k, 2 * mod // 3)
        ok_full = abs(s_full) < 1e-6
        ok_pair = abs((s_half + s_2half) + T) < 1e-6
        ok_re = abs(s_half.real + T / 2) < 1e-6
        status = "OK" if (ok_full and ok_pair and ok_re) else "VIOLATED"
        if status == "VIOLATED":
            all_ok = False
        print(f"l={l} j={j} k={k}: T={T}  sum_t S(t)={s_full:.6f}  "
              f"S(3^(l-1))+S(2*3^(l-1))={s_half + s_2half:.6f} (want {-T})  "
              f"Re S(3^(l-1))={s_half.real:.4f} (want {-T/2})  {status}")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

if __name__ == "__main__":
    main()
