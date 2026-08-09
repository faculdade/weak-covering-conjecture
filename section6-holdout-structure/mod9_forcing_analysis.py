#!/usr/bin/env python3
"""Round-5 H-013: per-budget mod-9 class content of H(l,j) and the forcing analysis.

For each level l and budget j (l <= j < j*), write the mod-9 classes of H(l,j) as
discrete logs base 2 (2 is a primitive root mod 9: classes 1,2,4,8,7,5 = 2^0..2^5),
a subset E(j) of Z/6.  Theorem 1 (H(l,j+1) subset 2H ^ 4H) projects to

    E(j+1) subset (E(j)+1) ^ (E(j)+2)      [shifts in Z/6]

and iterates to E(j+t) subset intersection_{s=t..2t} (E(j)+s).  This script checks,
per level:
  (a) the mod-3 dichotomy: every pre-final H(l,j) contains both classes mod 3
      (single class mod 3 forces extinction next step, so only the final set may
      collapse);
  (b) the budget at which E(j) first becomes a proper subset of Z/6;
  (c) whether iterating the shadow map from each budget FORCES the observed
      single class at j*-1 (deterministic forcing = the iterated image is a
      1-element set);
  (d) the mod-9 run bound j* <= j + maxcyclicrun(E(j)) (proper subsets only);
  (e) ends of maximal doubling runs in full dlog space, reduced mod 6: do all
      maximal-run ends agree mod 6 at the last budgets?

Levels 5..16 are computed here with the exact numpy DP (same as E-009 round 4).
Levels 17..21 use histograms/dumps from the Rust extractor (see
run_rust_sweep.sh output pasted into the round-5 note).

Also dumps H(l,j) for j in {j*-2, j*-1} as raw u64 files compatible with the
Rust extractor's --dump, for the fresh-vs-inherited analysis.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_witness_maps_and_inclusions import coverage, units_mask, JSTAR

DLOG9 = {1: 0, 2: 1, 4: 2, 8: 3, 7: 4, 5: 5}
POW9 = {v: k for k, v in DLOG9.items()}
DUMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")
os.makedirs(DUMPDIR, exist_ok=True)


def shadow_step(E):
    """One budget step of the mod-9 shadow map on a subset of Z/6."""
    return {(e + 1) % 6 for e in E} & {(e + 2) % 6 for e in E}


def cyclic_maxrun(E, n):
    if len(E) == n:
        return n
    best = 0
    for e in E:
        if (e - 1) % n not in E:
            r = 1
            while (e + r) % n in E:
                r += 1
            best = max(best, r)
    return best


def run_ends_mod6(hold, l):
    """Maximal doubling runs of H via membership walks; return (maxlen, ends-of-maximal-runs mod 6 info).

    A run start is x in H with x/2 not in H; the run is x, 2x, 4x, ...
    The 'end' is the last element; its dlog mod 6 equals DLOG9[end mod 9].
    """
    q = 3 ** l
    inv2 = pow(2, -1, q)
    s = set(int(x) for x in hold)
    runs = []
    for x in s:
        if (x * inv2) % q not in s:
            r = 1
            y = x
            while (y * 2) % q in s:
                y = (y * 2) % q
                r += 1
            runs.append((r, y))
    if not runs:
        return 0, {}
    maxlen = max(r for r, _ in runs)
    ends = {}
    for r, y in runs:
        if r == maxlen:
            c = y % 9
            ends.setdefault(c, 0)
            ends[c] += 1
    return maxlen, ends


def main():
    for l in range(5, 17):
        q = 3 ** l
        um = units_mask(l)
        jm = JSTAR[l]
        H = {}
        for j in range(l, jm):
            cov = coverage(l, j)
            H[j] = np.nonzero(um & ~cov)[0]
        print(f"\n=== l={l}  j*={jm} ===")
        E = {}
        for j in range(l, jm):
            hold = H[j]
            cls9 = {}
            for c in (1, 2, 4, 5, 7, 8):
                n = int(np.sum(hold % 9 == c))
                if n:
                    cls9[c] = n
            E[j] = {DLOG9[c] for c in cls9}
            m3 = sorted(set(int(x) % 3 for x in hold))
            mixed = "mixed" if m3 == [1, 2] else f"SINGLE mod3={m3}"
            proper = "" if len(E[j]) == 6 else f"  PROPER (missing dlogs {sorted(set(range(6)) - E[j])})"
            runb = ""
            if len(E[j]) < 6:
                mr = cyclic_maxrun(E[j], 6)
                runb = f"  mod9-run-bound: j*<= {j + mr} (true {jm})"
            mrun, ends = run_ends_mod6(hold, l)
            print(f"  j={j}: |H|={len(hold)}  mod9={cls9}  E={sorted(E[j])}"
                  f"  {mixed}{proper}{runb}  maxrun={mrun} maxrun-ends(mod9 cls:count)={ends}")
        # dichotomy check on final set
        final = H[jm - 1]
        m3f = sorted(set(int(x) % 3 for x in final))
        print(f"  final-set mod3 classes: {m3f} (law: single class 1)"
              f"  {'OK' if m3f == [1] else 'UNEXPECTED'}")
        # forcing: iterate shadow map from each budget to jm-1
        obs_final = {DLOG9[int(final[0]) % 9]} if len(set(int(x) % 9 for x in final)) == 1 else \
            {DLOG9[c] for c in set(int(x) % 9 for x in final)}
        for j in range(l, jm - 1):
            X = set(E[j])
            for _ in range(jm - 1 - j):
                X = shadow_step(X)
            verdict = ("FORCED-SINGLE" if len(X) == 1 else f"not forced (|image|={len(X)})")
            ok = "consistent" if obs_final <= X else "INCONSISTENT"
            print(f"  forcing from j={j}: iterated shadow image {sorted(X)} -> {verdict}; "
                  f"observed {sorted(obs_final)} {ok}")
        # dumps for cross-level work
        for j in (jm - 2, jm - 1):
            if j in H:
                arr = np.asarray(H[j], dtype=np.uint64)
                path = os.path.join(DUMPDIR, f"l{l}_j{j}.u64")
                arr.tofile(path)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
