#!/usr/bin/env python3
"""Mod-3 / mod-9 class structure of the last holdout sets H(l, j*(l)-1), l=5..16.

Checks (see README, 'mod-9 single-class law'):
  - every element == 1 (mod 3);
  - single class mod 9 per level;
  - class(l+1) == 4^(Delta j*) * class(l) (mod 9) at every transition;
  - #(class-2 mod 3 elements of H(l, j*-2)) == |H(l, j*-1)|.
"""
import numpy as np
from verify_witness_maps_and_inclusions import coverage, units_mask, JSTAR

prev_cls = None
prev_l = None
for l in range(5, 17):
    jm = JSTAR[l]
    hold = np.nonzero(units_mask(l) & ~coverage(l, jm - 1))[0]
    m3 = sorted(set(int(x) % 3 for x in hold))
    m9 = sorted(set(int(x) % 9 for x in hold))
    hold2 = np.nonzero(units_mask(l) & ~coverage(l, jm - 2))[0]
    c2 = int(np.sum(hold2 % 3 == 2))
    ok3 = "OK" if m3 == [1] else "VIOLATED"
    ok9 = "OK(single)" if len(m9) == 1 else f"NOT SINGLE ({m9})"
    okbij = "OK" if c2 == len(hold) else f"MISMATCH ({c2} vs {len(hold)})"
    line = (f"l={l}: |H(j*-1)|={len(hold)}, mod3={m3} {ok3}, mod9={m9} {ok9}, "
            f"class2-of-H(j*-2)={c2} {okbij}")
    if prev_cls is not None and len(m9) == 1:
        dj = jm - JSTAR[prev_l]
        pred = (pow(4, dj, 9) * prev_cls) % 9
        line += f"; x4^Dj law: pred {pred}, got {m9[0]} {'OK' if pred == m9[0] else 'VIOLATED'}"
    prev_cls = m9[0] if len(m9) == 1 else None
    prev_l = l
    print(line, flush=True)
