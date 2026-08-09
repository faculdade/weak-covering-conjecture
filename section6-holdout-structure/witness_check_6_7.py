#!/usr/bin/env python3
"""H-012 round 4: full witness-level child-fiber check (l,j)=(6,10) -> (7,11) under Dd<=1.

Definitions (project conventions, E-007/round6_durfee.py and round3_fable_dp.py):
  R_{j-1,j} = { F(alpha) = sum_{i=0}^{j-1} 2^{alpha_i} 3^i :
                2j-1 >= alpha_0 > alpha_1 > ... > alpha_{j-1} >= 0 }
  j*(l) = least j such that R_{j-1,j} covers all units mod 3^l.
  Durfee depth of a witness: d = #{ i : alpha_i >= j }   (verified below against
  the round-6 durfee_size() implementation on the partition lambda_r = alpha_{r-1}-(j-r)).

Checks performed, all exhaustive (full enumeration of both witness universes):
  A. coverage sanity at (6,10) mod 3^6 and (7,11) mod 3^7.
  B. per-child fiber check: d'_min(child) <= d_min(parent) + 1  (the literal Dd<=1 test).
  C. lift-mask: which of the 3 children of each parent are hit *directly* by parent
     witnesses (F mod 3^7 of a (6,10)-witness); missing children identified.
  D. repair cost: for every child, min symmetric-difference |S ^ S'| between any
     (6,10)-witness S of the parent and any (7,11)-witness S' of the child
     (masks over exponents; |S|=10, |S'|=11, so the distance is odd, >=1).
     Micro-move theory predicts <=3 (one slot moved + one exponent prepended).
"""
import math, sys
from itertools import combinations
import numpy as np

def durfee_size_reference(alphas, j):
    lam = [alphas[r - 1] - (j - r) for r in range(1, j + 1)]
    d = 0
    for q in range(1, j + 1):
        if lam[q - 1] >= q:
            d = q
        else:
            break
    return d

def enumerate_witnesses(j, mod):
    """Return arrays (res mod `mod`, durfee depth, mask) over all j-subsets of [0,2j-1],
    descending assignment: largest exponent gets 3^0."""
    n = math.comb(2 * j, j)
    res = np.empty(n, dtype=np.int64)
    dep = np.empty(n, dtype=np.int8)
    msk = np.empty(n, dtype=np.int64)
    pow3 = [3 ** i for i in range(j)]
    k = 0
    for comb in combinations(range(2 * j), j):  # ascending b_0<...<b_{j-1}
        # alpha_i (descending) = comb[j-1-i]; F = sum 2^alpha_i 3^i
        F = 0
        for i in range(j):
            F += (1 << comb[j - 1 - i]) * pow3[i]
        res[k] = F % mod
        d = 0
        for b in comb:
            if b >= j:
                d += 1
        dep[k] = d
        m = 0
        for b in comb:
            m |= 1 << b
        msk[k] = m
        k += 1
    return res, dep, msk

def main():
    # --- sanity: durfee via count-of-high-exponents == round-6 partition definition
    import random
    random.seed(4)
    for _ in range(2000):
        j = random.randint(2, 12)
        alphas = sorted(random.sample(range(0, 2 * j), j), reverse=True)
        d1 = durfee_size_reference(alphas, j)
        d2 = sum(1 for a in alphas if a >= j)
        assert d1 == d2, (alphas, j, d1, d2)
    print("sanity: durfee depth == #{alpha_i >= j}, 2000 random trials, exact")

    l = 6
    jp, jc = 10, 11             # parent and child budgets (j*(6), j*(7))
    q_par, q_child = 3 ** l, 3 ** (l + 1)

    pres7, pdep, pmask = enumerate_witnesses(jp, q_child)  # parent residues taken mod 3^7!
    cres7, cdep, cmask = enumerate_witnesses(jc, q_child)
    print(f"enumerated: {len(pres7)} parent tuples (j={jp}), {len(cres7)} child tuples (j={jc})")

    pres6 = pres7 % q_par

    units6 = np.array([x for x in range(q_par) if x % 3 != 0])
    units7 = np.array([x for x in range(q_child) if x % 3 != 0])

    # A. coverage
    cov6 = np.zeros(q_par, dtype=bool); cov6[pres6] = True
    cov7 = np.zeros(q_child, dtype=bool); cov7[cres7] = True
    print(f"A. coverage (6,{jp}): {cov6[units6].sum()}/{len(units6)} units;"
          f"  (7,{jc}): {cov7[units7].sum()}/{len(units7)} units")

    # d_min per residue
    dmin6 = np.full(q_par, 127, dtype=np.int8)
    np.minimum.at(dmin6, pres6, pdep)
    dmin7 = np.full(q_child, 127, dtype=np.int8)
    np.minimum.at(dmin7, cres7, cdep)

    # B. per-child Dd<=1
    viol = []
    deltas = {}
    for p in units6:
        for eps in range(3):
            r = p + eps * q_par
            dd = int(dmin7[r]) - int(dmin6[p])
            deltas[dd] = deltas.get(dd, 0) + 1
            if dd > 1:
                viol.append((int(p), int(r), int(dmin6[p]), int(dmin7[r])))
    print(f"B. Delta d = d'_min(child) - d_min(parent) distribution over {len(units6)*3} children: "
          f"{dict(sorted(deltas.items()))}")
    print(f"B. children violating Dd<=1: {len(viol)}"
          + (f"  e.g. {viol[:10]}" if viol else "   -->  Dd<=1 HOLDS for every child"))

    # B'. stratum-ceiling version (what an inductive strata argument actually uses):
    D6 = int(dmin6[units6].max())
    D7 = int(dmin7[units7].max())
    print(f"B'. stratum ceiling D_l = max over units of d_min:  D_6={D6}  D_7={D7}  "
          f"(Dd<=1 at ceiling level: {'HOLDS' if D7 <= D6 + 1 else 'FAILS'})")

    # C. lift-mask: children directly hit by parent witnesses
    hit7 = np.zeros(q_child, dtype=bool); hit7[pres7] = True
    missing = [int(r) for p in units6 for r in (p, p + q_par, p + 2 * q_par) if not hit7[r]]
    full_mask_parents = sum(1 for p in units6
                            if all(hit7[p + e * q_par] for e in range(3)))
    print(f"C. parents with full lift-mask (all 3 children hit directly): "
          f"{full_mask_parents}/{len(units6)} ({100*full_mask_parents/len(units6):.2f}%)")
    print(f"C. missing children (not directly hit): {len(missing)}: {sorted(missing)}")

    # group witnesses by residue for D
    order_p = np.argsort(pres6, kind='stable')
    order_c = np.argsort(cres7, kind='stable')
    pstart = np.searchsorted(pres6[order_p], np.arange(q_par + 1))
    cstart = np.searchsorted(cres7[order_c], np.arange(q_child + 1))

    def popcount64(arr):
        x = arr.astype(np.uint64)
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

    # D. min edit distance for every child (and specially for missing ones)
    print("D. repair-cost (min |S xor S'|) analysis:")
    editdist_hist = {}
    missing_detail = []
    maxdist_seen = 0
    argmax = None
    for p in units6:
        Pm = pmask[order_p[pstart[p]:pstart[p + 1]]]
        for eps in range(3):
            r = p + eps * q_par
            Cm = cmask[order_c[cstart[r]:cstart[r + 1]]]
            if len(Cm) == 0:
                print(f"   child {r}: NO WITNESS AT (7,{jc}) -- coverage failure!")
                continue
            x = np.bitwise_xor.outer(Pm, Cm)
            dmin_edit = int(popcount64(x).min())
            editdist_hist[dmin_edit] = editdist_hist.get(dmin_edit, 0) + 1
            if dmin_edit > maxdist_seen:
                maxdist_seen = dmin_edit
                argmax = (int(p), int(r))
            if r in missing:
                missing_detail.append((int(p), int(r), dmin_edit))
    print(f"   min-edit distribution over all children: {dict(sorted(editdist_hist.items()))}")
    print(f"   worst child: {argmax} at distance {maxdist_seen}")
    print(f"   missing children detail (parent, child, min-edit): {missing_detail}")

if __name__ == "__main__":
    main()
