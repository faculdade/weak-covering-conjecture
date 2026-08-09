#!/usr/bin/env python3
"""Re-check a stored window-k certificate without re-running the solver.

Theorem 3 of the paper needs, for a given window k, one policy sigma, one value
rho_k and one potential h such that at every unit state s modulo 3^k the move
sigma(s) is legal and safe, and

    d_sigma(s) + h(s'(e)) <= rho_k + h(s)    for every adversary digit e.

The additive constant is C_k = max h - min h. This script reads
certificate_k<K>.json(.gz), rebuilds the state set and the transitions from k
alone, and checks all of that. It never calls the solver, so it is an
independent check on the solver's output, not a repetition of it.

    python3 verify_certificate.py 3 4 5 6 7 8 9 10 11 12 13 14
    python3 verify_certificate.py            # every certificate in this folder

Exit status is nonzero if any check fails.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
import sys
from fractions import Fraction
from math import gcd

HERE = pathlib.Path(__file__).resolve().parent

# The values quoted in Table 3 of the paper.
TABLE3 = {
    3: Fraction(2), 4: Fraction(5, 3), 5: Fraction(3, 2), 6: Fraction(3, 2),
    7: Fraction(7, 5), 8: Fraction(25, 19), 9: Fraction(5, 4),
    10: Fraction(11, 9), 11: Fraction(6, 5), 12: Fraction(7, 6),
    13: Fraction(119, 104), 14: Fraction(9, 8),
}


def load(k: int) -> dict:
    plain = HERE / f"certificate_k{k}.json"
    packed = HERE / f"certificate_k{k}.json.gz"
    if packed.exists():
        with gzip.open(packed, "rt") as f:
            return json.load(f)
    with open(plain) as f:
        return json.load(f)


def check(k: int) -> bool:
    cert = load(k)
    if cert["k"] != k:
        print(f"k={k}: certificate is for k={cert['k']}")
        return False

    mod = 3 ** k
    mod3 = 3 * mod
    rho = Fraction(*cert["rho"])
    Ck = Fraction(*cert["Ck"])
    sigma = {int(s): d for s, d in cert["sigma"].items()}
    h = {int(s): Fraction(*v) for s, v in cert["h"].items()}

    states = [z for z in range(mod) if z % 3]
    if sorted(sigma) != states or sorted(h) != states:
        print(f"k={k}: policy or potential does not cover every unit state")
        return False

    # Clear denominators once: every comparison below is between integers.
    den = 1
    for v in h.values():
        den = den * v.denominator // gcd(den, v.denominator)
    hn = {s: int(v * den) for s, v in h.items()}
    rp, rq = rho.numerator, rho.denominator

    pow2 = {}
    illegal = unsafe = violated = 0
    for s in states:
        d = sigma[s]
        if (pow(2, d, 3) * s) % 3 != 1:
            illegal += 1
            continue
        if d not in pow2:
            pow2[d] = pow(2, d + 1, mod3)
        pd = pow2[d]
        for e in (0, 1, 2):
            num = (pd * (s + e * mod) - 2) % mod3
            if num % 3:
                unsafe += 1
                continue
            nxt = (num // 3) % mod
            if nxt % 3 == 0:
                unsafe += 1
                continue
            # d + h[nxt] <= rho + h[s], cleared of denominators
            if (d * den + hn[nxt]) * rq > rp * den + hn[s] * rq:
                violated += 1

    spread = max(h.values()) - min(h.values())
    ok = (illegal == 0 and unsafe == 0 and violated == 0 and spread == Ck)
    if k in TABLE3 and rho != TABLE3[k]:
        print(f"k={k}: rho={rho} disagrees with Table 3's {TABLE3[k]}")
        ok = False

    print(f"k={k:2d}  n={len(states):8d}  rho_k={str(rho):>7}={float(rho):.6f}  "
          f"C_k={str(Ck):>8}={float(Ck):7.4f}  "
          f"illegal={illegal} unsafe={unsafe} violations={violated}  "
          f"{'OK' if ok else 'FAILED'}")
    return ok


def main() -> None:
    if len(sys.argv) > 1:
        ks = [int(x) for x in sys.argv[1:]]
    else:
        ks = sorted(
            int(re.search(r"certificate_k(\d+)\.json", p.name).group(1))
            for p in HERE.glob("certificate_k*.json*")
        )
    print("checking legality, safety, the telescoping inequality at every "
          "state and digit, and C_k = max h - min h")
    if not all([check(k) for k in ks]):
        sys.exit(1)
    print("all certificates verified")


if __name__ == "__main__":
    main()
