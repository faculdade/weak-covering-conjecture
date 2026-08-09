"""Verifies the paper's Lemma "No smaller budget covers": for l = 2..L_MAX, no j < l
has R_{j-1,j} covering (Z/3^l Z)^*. Direct enumeration, independent of the Rust DP.
"""
import sys
from itertools import combinations

L_MAX = 13

def image_mod(l, j):
    mod = 3 ** l
    hit = set()
    if j == 0:
        return hit
    top = 2 * j - 1
    for combo in combinations(range(top + 1), j):
        a = sorted(combo, reverse=True)
        val = sum((2 ** a[i]) * (3 ** i) for i in range(j))
        hit.add(val % mod)
    return hit

def units_mod(l):
    mod = 3 ** l
    return {x for x in range(mod) if x % 3 != 0}

def main():
    all_ok = True
    for l in range(2, L_MAX + 1):
        u = units_mod(l)
        target = len(u)
        for j in range(1, l):
            img = image_mod(l, j) & u
            covers = (img == u)
            if covers:
                all_ok = False
                print(f"VIOLATED: l={l}, j={j} < l covers (Z/3^{l}Z)^* "
                      f"({len(img)}/{target} units hit)")
        print(f"l={l}: OK, no j<{l} covers ({l - 1} budgets checked)")
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
