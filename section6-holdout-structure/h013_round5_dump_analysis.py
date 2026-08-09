#!/usr/bin/env python3
"""Round-5 H-013 dump analysis: ancestry set-equalities, run ends, fresh-vs-inherited.

Inputs: raw u64 holdout dumps written by
  dumps-h013/ (the h013_sweep binary in ../section2-jstar-computation), l=17..21, and
  dumps/ (written by mod9_forcing_analysis.py in this folder), l=5..16.

Checks, per level l (b = j*(l)-1 the last budget):
  A. Theorem-1 inclusions on the dumped budgets (y/2, y/4 stay holdouts one budget down).
  B. Ancestry set-equality: for s=1..(#dumped budgets-1),
        2^{-s} H(l,b)  ==  { x in H(l,b-s) : dlog9(x) == e_final - s (mod 6) }
     (the "tail class" of the earlier budget is EXACTLY the ancestor set, not a superset).
  C. Maximal doubling runs at each dumped budget: maxrun, and the mod-9 classes of the
     ends of the maximal runs (single class?).
  D. Bijection fact: H(l,b) == 2 * { x in H(l,b-1) : x == 2 mod 3 }.
  E. Cross-level fresh/inherited 2x2 for transitions l -> l+1 (Delta j* = 1):
     for y in H(l+1, b+1): via-4 ancestor y/4 in H(l+1,b) is a lift of H(l,b) or fresh;
     same for the via-2 ancestor y/2; cross-tab, and mod-729 family split
     (A = 4^(b+1-13), B = A - 243 mod 729, other).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUST_DUMPS = os.path.join(HERE, "dumps-h013")
PY_DUMPS = os.path.join(HERE, "dumps")

JSTAR = {5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 15, 11: 16, 12: 17, 13: 18, 14: 19,
         15: 20, 16: 20, 17: 21, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26}
DLOG9 = {1: 0, 2: 1, 4: 2, 8: 3, 7: 4, 5: 5}
POW9 = {v: k for k, v in DLOG9.items()}


def load(l, j):
    for d in (RUST_DUMPS, PY_DUMPS):
        p = os.path.join(d, f"l{l}_j{j}.u64")
        if os.path.exists(p):
            return np.fromfile(p, dtype=np.uint64)
    return None


def runs_info(hold_set, q):
    inv2 = pow(2, -1, q)
    best = 0
    ends = {}
    allruns = []
    for x in hold_set:
        if (x * inv2) % q not in hold_set:
            r = 1
            y = x
            while (y * 2) % q in hold_set:
                y = (y * 2) % q
                r += 1
            allruns.append((r, y))
            if r > best:
                best = r
    for r, y in allruns:
        if r == best:
            ends.setdefault(y % 9, 0)
            ends[y % 9] += 1
    return best, ends


def main():
    sets = {}   # (l, j) -> python set
    for l in range(16, 23):
        jm = JSTAR.get(l)
        if jm is None:
            continue
        for j in range(l, jm):
            arr = load(l, j)
            if arr is not None:
                sets[(l, j)] = set(int(v) for v in arr)

    for l in range(16, 23):
        jm = JSTAR[l]
        b = jm - 1
        if (l, b) not in sets:
            print(f"l={l}: final dump missing, skipping level checks")
            continue
        q = 3 ** l
        inv2 = pow(2, -1, q)
        inv4 = pow(4, -1, q)
        final = sets[(l, b)]
        e_final = {DLOG9[x % 9] for x in final}
        print(f"\n=== l={l} j*={jm}: |H(b)|={len(final)}, final mod9 dlogs={sorted(e_final)} ===")
        ef = next(iter(e_final)) if len(e_final) == 1 else None

        avail = sorted(j for (ll, j) in sets if ll == l)
        # A + C: inclusions and run structure
        for j in avail:
            hs = sets[(l, j)]
            mr, ends = runs_info(hs, q)
            print(f"  j={j}: |H|={len(hs)} maxrun={mr} maxrun-end classes mod9={ends}")
            if (l, j + 1) in sets:
                nxt = sets[(l, j + 1)]
                bad2 = sum(1 for y in nxt if (y * inv2) % q not in hs)
                bad4 = sum(1 for y in nxt if (y * inv4) % q not in hs)
                print(f"    Theorem1 into j={j}: bad2={bad2} bad4={bad4} "
                      f"{'OK' if bad2 == 0 and bad4 == 0 else 'VIOLATED'}")
        # B: ancestry set equality
        if ef is not None:
            for s in range(1, 4):
                if (l, b - s) not in sets:
                    break
                anc = {(y * pow(inv2, s, q)) % q for y in final}
                tail_cls = POW9[(ef - s) % 6]
                tail = {x for x in sets[(l, b - s)] if x % 9 == tail_cls}
                verdict = "EQUAL" if anc == tail else \
                    f"anc<=tail: {anc <= tail}, |anc|={len(anc)}, |tail|={len(tail)}"
                print(f"  ancestry s={s}: 2^-{s}H(b) vs class-{tail_cls} part of H({b - s}): {verdict}")
        # D: bijection fact
        if (l, b - 1) in sets:
            m2 = {x for x in sets[(l, b - 1)] if x % 3 == 2}
            img = {(2 * x) % q for x in m2}
            print(f"  bijection: 2*(mod3=2 part of H(b-1)) == H(b): {img == final} "
                  f"(|mod3=2 part|={len(m2)})")

    # E: cross-level transitions
    print("\n=== cross-level fresh/inherited (Delta j*=1 transitions) ===")
    for l in range(16, 22):
        jm, jm1 = JSTAR[l], JSTAR.get(l + 1)
        if jm1 != jm + 1:
            print(f"l={l}->{l+1}: Delta j* != 1, skipping")
            continue
        b_lo, b_hi = jm - 1, jm1 - 1        # b_hi = b_lo + 1
        need = [(l, b_lo), (l + 1, b_lo), (l + 1, b_hi)]
        if any(k not in sets for k in need):
            print(f"l={l}->{l+1}: missing dumps {[k for k in need if k not in sets]}")
            continue
        qh = 3 ** (l + 1)
        ql = 3 ** l
        inv2 = pow(2, -1, qh)
        inv4 = pow(4, -1, qh)
        Hl = sets[(l, b_lo)]
        Hmid = sets[(l + 1, b_lo)]
        Hn = sets[(l + 1, b_hi)]
        A = pow(4, b_hi - 13, 729)
        B = (A - 243) % 729
        table = {}
        fam_table = {}
        for y in Hn:
            y2 = (y * inv2) % qh
            y4 = (y * inv4) % qh
            assert y2 in Hmid and y4 in Hmid, "Theorem 1 violated at transition!"
            lift2 = (y2 % ql) in Hl
            lift4 = (y4 % ql) in Hl
            key = (lift2, lift4)
            table[key] = table.get(key, 0) + 1
            fam = "A" if y % 729 == A else ("B" if y % 729 == B else "other")
            fam_table[(fam, key)] = fam_table.get((fam, key), 0) + 1
        print(f"l={l}->{l+1} (budgets {b_lo}->{b_hi}): |H_next|={len(Hn)}")
        for key in sorted(table, reverse=True):
            print(f"   (via2-lift={key[0]}, via4-lift={key[1]}): {table[key]}")
        for fk in sorted(fam_table):
            print(f"   family {fk[0]} {fk[1]}: {fam_table[fk]}")


if __name__ == "__main__":
    main()
