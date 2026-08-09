#!/usr/bin/env python3
"""Independent small-level check of the Rust H-014 computation.

This deliberately uses the original R_{j-1,j} definition and Python sets,
not the reduced packed-bitset implementation in src/main.rs.
"""

from itertools import combinations


RUST_EXPECTED = {
    1: (0, 0),
    2: (1, 1),
    3: (3, 2),
    4: (10, 2),
    5: (28, 3),
    6: (77, 3),
    7: (208, 3),
    8: (552, 3),
    9: (1430, 3),
    10: (3628, 4),
}


def direct_holdouts(ell):
    j = ell + 1
    q = 3**ell
    image = set()
    # combinations are ascending; reverse them to obtain alpha_0>...>alpha_{j-1}.
    for chosen in combinations(range(2 * j), j):
        value = sum(pow(2, a, q) * pow(3, i, q)
                    for i, a in enumerate(reversed(chosen))) % q
        image.add(value)
    return {x for x in range(q) if x % 3 and x not in image}


def maxrun(hold, q):
    if not hold:
        return 0
    inv2 = pow(2, -1, q)
    best = 0
    for x in hold:
        if x * inv2 % q not in hold:
            run = 1
            y = x
            while 2 * y % q in hold:
                y = 2 * y % q
                run += 1
            best = max(best, run)
    return best


for ell, expected in RUST_EXPECTED.items():
    hold = direct_holdouts(ell)
    got = (len(hold), maxrun(hold, 3**ell))
    assert got == expected, (ell, got, expected)
    print(f"ell={ell}: direct_original |H|={got[0]} maxrun={got[1]} PASS")

print("all direct-original checks match the Rust reduced-DP results")
