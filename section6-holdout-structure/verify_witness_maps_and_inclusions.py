#!/usr/bin/env python3
"""H-012/H-013 round 4: test the budget-increment inclusions and the run-length bootstrap.

Claimed lemma (hand-derived this round, to be verified here on every computable case):
  Let H(l,j) = units mod 3^l NOT covered by R_{j-1,j} (holdout set at level l, budget j).
  For j >= l:
    (x2 map)  alpha -> alpha+1 (all exponents), prepend 0:      F' = 2F + 3^j  == 2F  (mod 3^l)
    (x4 map)  alpha -> alpha+2 (all exponents), prepend b<=1:   F' = 4F + 2^b 3^j == 4F (mod 3^l)
  Both produce valid budget-(j+1) witnesses, so
    covered(r, j) => covered(2r, j+1) and covered(4r, j+1)
    i.e.  H(l,j+1) subseteq 2*H(l,j)  and  H(l,j+1) subseteq 4*H(l,j).
  Corollary (run-length bootstrap): writing K = dlog_2(H(l,j)) (2 is a primitive root mod 3^l),
  H(l,j+t) subseteq intersection_{s=t..2t} 2^s H(l,j); nonempty requires K to contain t+1
  consecutive values.  So maxrun(K) <= t  =>  j*(l) <= j+t;  in particular j*(l) <= j + |H(l,j)|.

This script computes H(l,j) exactly by DP for l=5..14, j=l..j*(l), then:
  1. verifies both inclusions exactly, every (l,j);
  2. computes dlog structure and maxrun(K), verifies the bootstrap prediction against true j*;
  3. verifies the witness-map lemma directly on random witnesses;
  4. H-013: holdout sets at j*(l)-1, their mod-3^6 classes, the x4 cross-level law,
     and the decomposition inherited-vs-new (children of parent holdouts vs fresh defects).
"""
import math
import numpy as np

JSTAR = {1:1,2:4,3:6,4:7,5:9,6:10,7:11,8:12,9:13,10:15,11:16,12:17,13:18,14:19,
         15:20,16:20,17:21,18:22,19:23,20:24,21:25,22:26,23:27}

def coverage(l, j):
    """Bool array over Z/3^l: covered by R_{j-1,j}. Vectorized DP (round3_fable_dp style)."""
    q = 3 ** l
    dp = [np.zeros(q, dtype=bool) for _ in range(j + 1)]
    dp[0][0] = True
    for a in range(2 * j - 1, -1, -1):
        c_hi = min(j - 1, 2 * j - 1 - a)
        c_lo = max(0, j - 1 - a)
        for c in range(c_hi, c_lo - 1, -1):
            t = (pow(2, a, q) * pow(3, c, q)) % q
            src, dst = dp[c], dp[c + 1]
            if t == 0:
                dst |= src
            else:
                dst[t:] |= src[:q - t]
                dst[:t] |= src[q - t:]
    return dp[j]

def units_mask(l):
    q = 3 ** l
    m = np.ones(q, dtype=bool)
    m[::3] = False
    return m

def dlog_table(l):
    """dlog[u] = k with 2^k == u (mod 3^l), for units u; -1 elsewhere."""
    q = 3 ** l
    ordr = 2 * 3 ** (l - 1)
    tab = np.full(q, -1, dtype=np.int64)
    x = 1
    for k in range(ordr):
        tab[x] = k
        x = (x * 2) % q
    return tab, ordr

def maxrun_circular(ks, ordr):
    """longest run of consecutive integers (mod ordr) in the set ks."""
    if len(ks) == 0:
        return 0
    s = set(int(k) for k in ks)
    if len(s) == ordr:
        return ordr
    best = 0
    for k in s:
        if (k - 1) % ordr not in s:      # run start
            r = 1
            while (k + r) % ordr in s:
                r += 1
            best = max(best, r)
    return best

def main():
    # ---- 3. witness-map lemma, direct random verification -------------------
    import random
    random.seed(12)
    for _ in range(500):
        j = random.randint(4, 12)
        l = random.randint(2, j)
        q = 3 ** l
        S = sorted(random.sample(range(0, 2 * j), j))          # ascending
        F = sum((1 << S[j - 1 - i]) * 3 ** i for i in range(j))
        # x2 map
        S2 = [0] + [b + 1 for b in S]
        F2 = sum((1 << S2[j - i]) * 3 ** i for i in range(j + 1))
        assert len(S2) == j + 1 and S2[-1] <= 2 * (j + 1) - 1 and len(set(S2)) == j + 1
        assert F2 % q == (2 * F) % q, (S, l, j)
        # x4 map, b=0 and b=1
        for b in (0, 1):
            S4 = [b] + [x + 2 for x in S]
            F4 = sum((1 << S4[j - i]) * 3 ** i for i in range(j + 1))
            assert S4[-1] <= 2 * (j + 1) - 1 and len(set(S4)) == j + 1
            assert F4 % q == (4 * F) % q, (S, l, j, b)
    print("3. witness maps verified: 500 random (l,j,S), x2 and x4 (b=0,1), exact")

    # ---- 1/2/4. holdout sets, inclusions, bootstrap, H-013 ------------------
    prev_holdout = {}    # l -> H(l, j*(l)-1) as sorted array (for cross-level law)
    for l in range(5, 15):
        q = 3 ** l
        um = units_mask(l)
        n_units = int(um.sum())
        dtab, ordr = dlog_table(l)
        jm = JSTAR[l]
        H = {}
        for j in range(max(l, 5), jm + 1):
            cov = coverage(l, j)
            hold = np.nonzero(um & ~cov)[0]
            H[j] = hold
        js = sorted(H.keys())
        print(f"\nl={l} (j*={jm}), |H(l,j)| for j={js[0]}..{jm}: "
              f"{[len(H[j]) for j in js]}")
        # sanity: coverage at j* complete, at j*-1 not
        assert len(H[jm]) == 0, f"j*({l}) table mismatch!"
        if jm - 1 in H:
            assert len(H[jm - 1]) > 0, f"j*({l}) table mismatch (covered too early)!"

        inv2 = pow((q + 1) // 2, 1, q)   # placeholder; compute proper inverse below
        inv2 = pow(2, -1, q)
        inv4 = pow(4, -1, q)
        for j in js[:-1]:
            h_next = H[j + 1]
            h_cur = set(int(x) for x in H[j])
            bad2 = [int(x) for x in h_next if (int(x) * inv2) % q not in h_cur]
            bad4 = [int(x) for x in h_next if (int(x) * inv4) % q not in h_cur]
            ok2 = "OK" if not bad2 else f"VIOLATED {bad2[:5]}"
            ok4 = "OK" if not bad4 else f"VIOLATED {bad4[:5]}"
            # bootstrap check at this j
            ks = dtab[H[j]]
            mr = maxrun_circular(ks, ordr)
            pred = j + mr          # theorem: j*(l) <= j + maxrun
            status = "OK" if jm <= pred else "THEOREM FALSIFIED"
            print(f"  j={j}: |H|={len(H[j])}, maxrun(dlog)={mr}, "
                  f"bootstrap j*<= {pred} (true {jm}) {status}; "
                  f"H(j+1) in 2H: {ok2}; in 4H: {ok4}")

        # ---- H-013 block: the j*-1 holdout set ------------------------------
        hold = H[jm - 1] if jm - 1 in H else None
        if hold is not None and len(hold) > 0:
            cls6 = sorted(set(int(x) % 729 for x in hold))
            print(f"  H-013: H(l,j*-1) size {len(hold)}"
                  + (f", residues {sorted(int(x) for x in hold)}" if len(hold) <= 6 else "")
                  + f", mod 3^6 classes {cls6 if len(cls6) <= 12 else str(len(cls6)) + ' classes'}")
            # x4 law across levels: classes vs 4 * previous level's classes (mod 729)
            if (l - 1) in prev_holdout and len(prev_holdout[l - 1]) > 0:
                prev_cls = set(int(x) % 729 for x in prev_holdout[l - 1])
                pred_cls = set((4 * c) % 729 for c in prev_cls)
                inter = pred_cls & set(cls6)
                print(f"  H-013: 4*prev-level classes {sorted(pred_cls)} ; "
                      f"overlap with current: {sorted(inter)}")
            # inherited vs new, same budget: children of H(l-1, jm-1) vs H(l, jm-1)
            if l - 1 >= 5 and jm - 1 >= l - 1:
                cov_prev = coverage(l - 1, jm - 1)
                um_prev = units_mask(l - 1)
                hold_prev_sb = np.nonzero(um_prev & ~cov_prev)[0]  # H(l-1, j*(l)-1)
                children = set()
                for p in hold_prev_sb:
                    for e in range(3):
                        children.add(int(p) + e * 3 ** (l - 1))
                inherited = [int(x) for x in hold if int(x) in children]
                fresh = [int(x) for x in hold if int(x) not in children]
                print(f"  H-013: same-budget decomposition: |H(l-1,{jm-1})|={len(hold_prev_sb)}, "
                      f"inherited children in H(l,{jm-1}): {len(inherited)}, fresh defects: {len(fresh)}"
                      + (f" {fresh[:6]}" if len(fresh) <= 6 else ""))
        prev_holdout[l] = hold if hold is not None else np.array([], dtype=np.int64)

if __name__ == "__main__":
    main()
