# Sections 6 and 7: the holdout sets

`H(l,j)` is the set of units modulo `3^l` that `R_{j-1,j}` fails to cover. Everything here is
exact: full dynamic programs or full enumeration, never sampling. Scripts print `OK` / `PASS`
lines, and any `VIOLATED`, `FALSIFIED` or `MISMATCH` in the output means a claim failed to
reproduce.

Run each with `python3 <script>` from this folder. `numpy` is required.

## Theorem 8, Corollary 10, Empirical Result 11

```
python3 verify_witness_maps_and_inclusions.py      # 4 seconds, l=5..14
python3 extend_l15_l16.py                          # 33 seconds, l=15,16
```

Checks the two witness shift maps, the inclusion `H(l,j+1) subset 2H(l,j) cap 4H(l,j)` at every
budget, and the bootstrap `j*(l) <= j + maxrun(H(l,j))`. At every budget from `l+1` to `j*(l)` the
bootstrap is not just true but tight, which is Empirical Result 11. The `l=16` case is the sharp
one: `H(16,19)` has 1182 elements and `maxrun = 1`, so Corollary 10 alone forces coverage at 20.

Larger levels, two independent ways:

```
python3 h013_round5_dump_analysis.py               # 23 seconds, l=16..21, from stored dumps
python3 holdout_l19.py                             # 15 minutes, 27 GiB, recomputes l=19
python3 holdout_l20.py                             # about 1 hour, 11 GiB, recomputes l=20
```

`h013_round5_dump_analysis.py` reads the raw `u64` holdout dumps in `dumps-h013/` (produced by
`h013_sweep` in `../section2-jstar-computation/`) and `dumps/` (produced by
`mod9_forcing_analysis.py` here). It confirms the Theorem 8 inclusions and the bootstrap
tightness up to `l=21`, including `|H(19,20)| = 3,195,464`, the case above three million that
Empirical Result 11 mentions. The last line, `l=21->22: missing dumps`, is expected: `l=22` was
never dumped. `holdout_l19.py` recomputes the `l=19` sets from scratch with a separate numpy
implementation and gets the same holdout counts and the same maxruns.

## Empirical Result 12, the failure of the cost-1 repair rule

```
python3 witness_check_6_7.py                       # 9 seconds
python3 witness_check_7_8.py                       # 45 seconds
```

Exhaustive over both witness universes (184,756 and 705,432 tuples at the first transition,
705,432 and 2,704,156 at the second). Expected: at `(6,10) -> (7,11)`, 100 of 1458 children
exceed their parent's minimum Durfee depth by more than one, with the excess distribution
`{-2: 6, -1: 214, 0: 661, 1: 477, 2: 94, 3: 6}`; at `(7,11) -> (8,12)`, 332 of 4374, distribution
`{-3: 2, -2: 20, -1: 666, 0: 2072, 1: 1282, 2: 304, 3: 28}`. The minimum-edit repair distances are
`{1: 1392, 3: 9, 5: 52, 7: 5}` and `{1: 4205, 3: 7, 5: 135, 7: 27}`, so the worst case is 7 at
both transitions and 1 for about 95% of children.

## Theorem 14, Empirical Results 15 and 16, and the 1547 counterexample

```
python3 mod9_class_law.py                          # 9 seconds, l=5..16
python3 round5_secondary_closure.py                # 4 seconds, l=2..14
python3 mod9_forcing_analysis.py                   # 21 seconds, rewrites dumps/
```

`mod9_class_law.py` checks that every element of `H(l, j*(l)-1)` is `1 mod 3` (Theorem 14), that
the set occupies a single class mod 9 (Empirical Result 16), and that the class follows
`class(l+1) = 4^(delta j*) class(l)` across every transition, plateau and double step included.

`round5_secondary_closure.py` adds the near-extinction equality
`H(l,J-1) = 2 { x in H(l,J-2) : x = 2 mod 3 }` (Empirical Result 15), which passes at every
`l=2..14`, and the counterexample to an unrestricted converse of the run-length bootstrap:
`1547` lies in `2H(7,8) cap 4H(7,8)` yet is covered at budget 9, and both of its budget-9
witnesses span the full exponent range.

## The width reformulation and corner-redundancy (Section 7)

```
python3 round5_width_reformulation.py              # 15 seconds, l=3..13
```

Verifies Lemma 13's width dictionary against known holdout sizes, the death-parity theorem, and
the one-step identity `U(l,W+1) = U(l,W) u 2U(l,W) u Corner(l,W+1)`. `|D(W+1)|` counts the values
at width `W+1` that neither `U(l,W)` nor `2U(l,W)` produces. It is zero at every `W >= 2l+1` for
every `l=3..13`, which is the corner-redundancy property. The boundary width `W = 2l` is the only
place it fails, and the script reports that entry separately: 0 for `l<=6`, then 1, 1, 2, 4, 4, 11,
9 for `l=7..13`.

Pass a different upper level as an argument, for example `python3 round5_width_reformulation.py 11`.

## Also here

```
python3 nonmonotone_coverage.py                    # under a second
```

Pointwise coverage is not monotone in the budget. Residue `128368` modulo `3^13` is covered at
budgets 13 and 14, not at 15 and 16, and again at 17. The script prints an explicit budget-14
witness that can be checked by hand. Full coverage is still monotone, but only by Theorem 8.

`maxrun_from_dump.py <dumpfile>` reports the maxrun and the mod-9 class content of any stored
dump.
