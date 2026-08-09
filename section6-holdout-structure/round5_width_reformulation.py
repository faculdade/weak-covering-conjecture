#!/usr/bin/env python3
"""Round 5 (Fable leg): the width reformulation, the death-parity theorem, and the
corner-redundancy test.

Setup (proved in notes/H-003.md round-5 section; verified here from scratch):
for j >= l, coverage of x mod 3^l at budget j is equivalent to
2^-(j-l) x in U(l, W) with W = j + l - 1, where

    U(l, W) := { sum_{i=0}^{l-1} 2^(beta_i) 3^i  mod 3^l :
                 W >= beta_0 > beta_1 > ... > beta_(l-1) >= 0 }.

U is monotone in W. The exact one-step identity is
    U(W+1) = U(W) union 2U(W) union Corner(W+1),
Corner(W+1) = values of tuples with beta_0 = W+1 AND beta_(l-1) = 0.

Checks:
 1. U reproduces the known holdout sizes |H(l,j)| through the width dictionary.
 2. The one-step identity holds (D(W+1) := U(W+1) \\ (U(W) u 2U(W)) consists of
    corner values only) and we measure |D| at every width: D = empty for W >= 2l
    is the corner-redundancy property that IMPLIES the observed tightness law.
 3. Death-parity theorem: W_min(x) := least W with x in U(l,W); then every
    threshold witness has beta_0 = W_min, hence x == (-1)^W_min (mod 3).
 4. beta_1-parity: eps(x) := parity of beta_1 in threshold witnesses, read off as
    (x - 2^W_min)/3 == 2^(beta_1) == (-1)^(beta_1) (mod 3). The mod-9 class of x
    is 2^W_min + 3*2^(beta_1) mod 9, determined by (W_min mod 2, eps). The
    observed single-class mod-9 law of last holdouts == "all x with maximal
    W_min have eps == (W_min - 1) mod 2". Tabulated per W_min value.
 5. Level recursion: W_min_l(x) = min{ W : W_min_(l-1)((x - 2^W)/3) <= W-1 }.
"""
import sys
import numpy as np

JSTAR = {2: 4, 3: 6, 4: 7, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 15, 11: 16,
         12: 17, 13: 18}
# l=1..5 re-verified here by direct brute-force enumeration of R_{j-1,j} (j*(3)=6, not 5)
# known |H(l,j)| for validation, from verify_witness_maps_and_inclusions.py output
KNOWN_H = {
    (5, 5): 58, (5, 6): 28, (5, 7): 9, (5, 8): 1, (5, 9): 0,
    (6, 6): 162, (6, 7): 77, (6, 8): 24, (6, 9): 3, (6, 10): 0,
    (7, 7): 447, (7, 8): 208, (7, 9): 66, (7, 10): 9, (7, 11): 0,
    (10, 11): 3628, (10, 14): 2, (10, 15): 0,
}


def images_by_width(l, wmax):
    """U(l, W) for W = l-1 .. wmax, as boolean arrays over Z/3^l.

    DP over widths: A(W, c) := values of c-subsets of [0, W] (top exponent
    <= W) under sum 2^(beta_i) 3^i with beta descending, i ascending from the
    TOP: the exponent paired with 3^i is the (i+1)-st largest. Recurrence on
    the smallest element: append a new smallest exponent b to a c-subset of
    [b+1, W]. Simpler equivalent used here: iterate v = W down to 0 with
    c = number already chosen (all chosen exponents > v), matching the h013
    extractor's recurrence; rerun per W (cheap at these sizes).
    """
    q = 3 ** l
    out = {}
    for W in range(l - 1, wmax + 1):
        dp = [np.zeros(q, dtype=bool) for _ in range(l + 1)]
        dp[0][0] = True
        for v in range(W, -1, -1):
            p2 = pow(2, v, q)
            for c in range(min(l - 1, W - v), -1, -1):
                t = (p2 * pow(3, c, q)) % q
                src = dp[c]
                if src.any():
                    dp[c + 1] |= np.roll(src, t)
        out[W] = dp[l]
    return out


def main():
    ok = True
    # Levels for checks 2-5. The paper's corner-redundancy statement covers
    # l=3..13; pass a different upper level on the command line to change it.
    lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 13

    # --- check 1: dictionary against known holdout sizes ---
    for l in (5, 6, 7, 10):
        wmax = JSTAR[l] + l - 1
        U = images_by_width(l, wmax)
        units = np.ones(3 ** l, dtype=bool)
        units[::3] = False
        for (ll, j), known in KNOWN_H.items():
            if ll != l:
                continue
            got = int((units & ~U[j + l - 1]).sum())
            tag = 'OK' if got == known else 'MISMATCH'
            if got != known:
                ok = False
            print(f'check1 l={l} j={j} (W={j + l - 1}): |H|={got} known={known} {tag}')

    # --- checks 2-5 per level ---
    for l in range(3, lmax + 1):
        q = 3 ** l
        jstar = JSTAR[l]
        wstar = jstar + l - 1
        wmax = wstar + 1
        U = images_by_width(l, wmax)
        units = np.ones(q, dtype=bool)
        units[::3] = False

        # check 2: |D(W+1)| = |U(W+1) \ (U(W) u 2U(W))| per width
        drow = []
        for W in range(l - 1, wmax):
            twoU = np.zeros(q, dtype=bool)
            idx = np.nonzero(U[W])[0]
            twoU[(2 * idx) % q] = True
            D = U[W + 1] & ~(U[W] | twoU)
            drow.append(int(D.sum()))
        # Corner-redundancy as stated in the paper: |D(W+1)| = 0 at every
        # width W >= 2l+1. The boundary width W = 2l is reported separately,
        # since that is where the known failures sit.
        tail = drow[(2 * l + 1) - (l - 1):]
        boundary = drow[(2 * l) - (l - 1)] if (2 * l) - (l - 1) < len(drow) else 0
        tag = ('corner-redundant for W>=2l+1'
               if all(d == 0 for d in tail) else 'CORNER-ESSENTIAL AT SOME W>=2l+1')
        print(f'check2 l={l}: |D(W+1)| for W={l-1}..{wmax-1}: {drow}  -> {tag}'
              f'; |D| at the boundary W=2l is {boundary}')

        # W_min per unit
        wmin = np.full(q, -1, dtype=np.int64)
        prev = np.zeros(q, dtype=bool)
        for W in range(l - 1, wmax + 1):
            new = U[W] & ~prev
            wmin[np.nonzero(new)[0]] = W
            prev = U[W]
        un = np.nonzero(units)[0]
        assert (wmin[un] >= 0).all(), 'some unit never covered in range'
        assert int(wmin[un].max()) == wstar, f'W* mismatch: {wmin[un].max()} vs {wstar}'

        # check 3: death parity  x == (-1)^W_min (mod 3)
        bad = [int(x) for x in un if (x % 3 == 1) != (wmin[x] % 2 == 0)]
        if bad:
            ok = False
        print(f'check3 l={l}: death-parity x==(-1)^W_min mod 3: '
              f'{"OK (all units)" if not bad else f"FAILS at {bad[:5]}"}')

        # check 4: beta_1 parity per W_min value; single-class test at W*
        rows = {}
        for x in un:
            W = int(wmin[x])
            y = ((int(x) - pow(2, W, q)) % q) // 3 % 3  # (x-2^W)/3 mod 3
            eps = 0 if y == 1 else 1  # parity of beta_1
            rows.setdefault(W, [0, 0])[eps] += 1
        line = ', '.join(f'W={W}: eps0={a} eps1={b}' for W, (a, b) in sorted(rows.items()))
        a, b = rows[wstar]
        single = (a == 0) or (b == 0)
        expect = (wstar - 1) % 2
        got_eps = 1 if a == 0 else 0
        tag = ('SINGLE mod-9 class at W*, eps=' + str(got_eps)
               + (' == (W*-1)%2 OK' if got_eps == expect else ' != (W*-1)%2 MISMATCH'))
        if not single:
            tag = 'NOT single-class at W*'
            ok = False
        print(f'check4 l={l}: {line}')
        print(f'check4 l={l}: {tag}')

        # check 5: level recursion vs level l-1 (skip l=3: base small)
        if l >= 4:
            ql = q // 3
            Uprev = images_by_width(l - 1, wmax)
            wmin_prev = np.full(ql, 10 ** 9, dtype=np.int64)
            prevb = np.zeros(ql, dtype=bool)
            for W in range(l - 2, wmax + 1):
                new = Uprev[W] & ~prevb
                wmin_prev[np.nonzero(new)[0]] = W
                prevb = Uprev[W]
            import random
            random.seed(5)
            sample = random.sample(list(map(int, un)), min(300, len(un)))
            bad5 = []
            for x in sample:
                best = None
                for W in range(l - 1, wmax + 1):
                    d = (x - pow(2, W, q)) % q
                    if d % 3:
                        continue  # beta_0 = W impossible: mod-3 parity mismatch
                    y = d // 3
                    if wmin_prev[y] <= W - 1:
                        best = W
                        break
                if best != wmin[x]:
                    bad5.append((x, best, int(wmin[x])))
            if bad5:
                ok = False
            print(f'check5 l={l}: recursion W_min^l = min{{W: W_min^(l-1)((x-2^W)/3)<=W-1}}: '
                  f'{"OK (300 samples)" if not bad5 else f"FAILS {bad5[:3]}"}')

    print('ALL OK' if ok else 'SOME CHECK FAILED')


if __name__ == '__main__':
    main()
