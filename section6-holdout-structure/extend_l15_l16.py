#!/usr/bin/env python3
"""Extension: l=15,16 (incl. the j* plateau), the maxrun-at-(l+1) table, and the exact
x4 genealogy of the singleton holdout chain l=10..14."""
import numpy as np
from verify_witness_maps_and_inclusions import coverage, units_mask, dlog_table, maxrun_circular, JSTAR

def holdouts(l, j):
    return np.nonzero(units_mask(l) & ~coverage(l, j))[0]

# ---- exact x4 genealogy of the holdout chain across levels ----------------------
print("=== x4 genealogy: is h_(l+1) * 4^-1 mod 3^(l+1) a lift of (an element of) H(l, j*(l)-1)? ===")
chain = {10: [37912, 47389], 11: [33550], 12: [212932, 483148, 488494],
         13: [513472], 14: [2532112, 2952745, 3648211]}
for l in range(10, 14):
    q_next = 3 ** (l + 1)
    inv4 = pow(4, -1, q_next)
    prev = chain[l]
    for h in chain[l + 1]:
        h4 = (h * inv4) % q_next
        par = h4 % (3 ** l)
        eps = (h4 - par) // (3 ** l)
        match = par in prev
        print(f"  l={l+1}: h={h}: h/4 mod 3^{l+1} = {h4} = parent {par} + {eps}*3^{l}  "
              f"parent in H({l},{JSTAR[l+1]-2})={match}")

# ---- l=15, 16 ------------------------------------------------------------------
for l in (15, 16):
    jm = JSTAR[l]
    dtab, ordr = dlog_table(l)
    q = 3 ** l
    inv2, inv4 = pow(2, -1, q), pow(4, -1, q)
    H = {}
    for j in range(l, jm + 1):
        H[j] = holdouts(l, j)
    print(f"\nl={l} (j*={jm}), |H|: {[(j, len(H[j])) for j in sorted(H)]}")
    for j in sorted(H)[:-1]:
        h_cur = set(int(x) for x in H[j])
        bad2 = [int(x) for x in H[j+1] if (int(x)*inv2) % q not in h_cur]
        bad4 = [int(x) for x in H[j+1] if (int(x)*inv4) % q not in h_cur]
        mr = maxrun_circular(dtab[H[j]], ordr)
        status = "OK(tight)" if jm == j + mr else ("OK" if jm <= j + mr else "FALSIFIED")
        print(f"  j={j}: |H|={len(H[j])}, maxrun={mr}, bootstrap j*<={j+mr} (true {jm}) {status}; "
          f"2H incl: {'OK' if not bad2 else bad2[:3]}; 4H incl: {'OK' if not bad4 else bad4[:3]}")
    hold = H[jm - 1]
    cls = sorted(set(int(x) % 729 for x in hold))
    print(f"  H-013: |H(l,j*-1)|={len(hold)}, mod-3^6 classes: "
          f"{cls if len(cls) <= 15 else str(len(cls)) + ' classes'}")
    if len(hold) <= 6:
        print(f"        residues: {sorted(int(x) for x in hold)}")
