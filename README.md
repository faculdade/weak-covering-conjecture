# Reproducibility repository: Wirsching's Weak Covering Conjecture

Companion code and data for the paper *"Wirsching's Weak Covering Conjecture: an extended
computation, a conditional bound, and the structure of what resists it"* (Renato Augusto Tavares,
Universidade Federal de Goiás).

Every table, theorem certificate and numerical claim in the paper is produced by a script here.
The folders follow the paper's section numbers.

| paper claim | folder | command |
|---|---|---|
| Table 1, `j*(l)` for `l=1..23` | `section2-jstar-computation/` | `jstar-fast run 1 20` |
| Table 2, AIC/BIC/LOOCV and the plateau test | `section3-growth-models/` | `python3 experiment.py` |
| Lemma 2 (Grounding), Empirical Result 3 (the game's worst case) | `section4-mean-payoff-game/` | `python3 step_a_grounding.py` |
| Table 3, `rho_k`/`C_k`; Theorem 4 and Corollary 5 | `section4-mean-payoff-game/` | `python3 verify_certificate.py` |
| Section 5, the `l=18` exponential sum, the phase scramble, and the local-intensity computation | `section5-exponential-sum/` | `python3 l1_tail.py --l 18`, `python3 local_intensity.py` |
| Theorem 9, Corollary 10, Corollary 11, Empirical Result 12, Empirical Result 13 | `section6-holdout-structure/` | `python3 verify_witness_maps_and_inclusions.py` |
| Theorem 16, Proposition 18, Empirical Results 17, 19 | `section6-holdout-structure/` | `python3 mod9_class_law.py`, `python3 mod9_containment_proof_check.py` |
| Lemma 14, Proposition 15, Lemma 21, and the corner-redundancy test of Section 7 | `section6-holdout-structure/` | `python3 round5_width_reformulation.py` |
| Proposition 22 (corner-redundancy implies tightness) | `section6-holdout-structure/` | `python3 corner_redundancy_tightness.py` |
| Proposition 20, `maxrun(H(l,l+1))` through `l=22` | `section7-maxrun/` | `h014-maxrun 22 8` |

Each folder has its own README giving what it checks, how to run it, the output to expect, and
what the run costs.

## Requirements

Python 3.10 or later with `numpy` and `scipy`, a C++17 compiler (`g++`, used by helpers in
`section5-exponential-sum/`), and a Rust toolchain (`cargo`) for the compiled experiments.

```
pip install -r requirements.txt
```

## Scale

Most checks run in seconds, and every certificate in Section 4 re-verifies in under a minute.
Table 1 through `l=20` takes about 14 minutes. The expensive items, each with its cost in the
folder README, are `j*(21)`, `j*(22)` and `j*(23)` (263 GiB of state and about 104 hours at
`l=23`), the `l=18` spectrum (95 seconds, 22 GiB), the `local_intensity.py --with15` run (about
10-15 minutes), `maxrun` at `l=22` (2 minutes, 54 GiB), and the `l=20` holdout recomputation
(about an hour).

Stored holdout sets (`section6-holdout-structure/dumps*`) run through `l=21`; the covering search
in `section2-jstar-computation/` reproduces Table 1 through `l=23` by rerunning the DP, not from a
stored dump, since the `l=21..23` state is too large to keep as a checked-in file.

## Reference machine

AMD Ryzen 7 5700U, 8 cores and 16 threads, 62 GiB RAM, no GPU. Every timing quoted in this
repository was measured there.
