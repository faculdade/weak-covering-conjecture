# Reproducibility repository: Wirsching's Weak Covering Conjecture

Companion code and data for the paper *"Wirsching's Weak Covering Conjecture: an extended
computation, a proven bound, and the structure of what resists it"* (Renato Augusto Tavares,
Universidade Federal de Goiás).

Every table, theorem certificate and numerical claim in the paper is produced by a script here.
The folders follow the paper's section numbers.

| paper claim | folder | command |
|---|---|---|
| Table 1, `j*(l)` for `l=1..23` | `section2-jstar-computation/` | `jstar-fast run 1 20` |
| Table 2, AIC/BIC/LOOCV and the plateau test | `section3-growth-models/` | `python3 experiment.py` |
| Table 3, `rho_k`; Theorem 3 and Corollary 4 | `section4-mean-payoff-game/` | `python3 verify_certificate.py` |
| Empirical Result 2 (soundness of the game) | `section4-mean-payoff-game/` | `python3 step_a_grounding.py` |
| Section 5, the `l=18` exponential sum and the phase scramble | `section5-exponential-sum/` | `python3 l1_tail.py --l 18` |
| Theorem 8, Corollary 10, Empirical Results 11, 12, 15, 16 | `section6-holdout-structure/` | `python3 verify_witness_maps_and_inclusions.py` |
| Theorem 14 and the corner-redundancy test of Section 7 | `section6-holdout-structure/` | `python3 round5_width_reformulation.py` |
| Proposition 17, `maxrun(H(l,l+1))` through `l=22` | `section7-maxrun/` | `h014-maxrun 22 8` |

Each folder has its own README giving what it checks, how to run it, the output to expect, and
what the run costs.

## Requirements

Python 3.10 or later with `numpy` and `scipy`, a C++17 compiler (`g++`, used by one helper in
`section5-exponential-sum/`), and a Rust toolchain (`cargo`) for the two compiled experiments.

```
pip install -r requirements.txt
```

## Scale

Most checks run in seconds, and every certificate in Section 4 re-verifies in under a minute.
Table 1 through `l=20` takes about 14 minutes. The expensive items, each with its cost in the
folder README, are `j*(21)`, `j*(22)` and `j*(23)` (263 GiB of state and about 104 hours at
`l=23`), the `l=18` spectrum (95 seconds, 22 GiB), `maxrun` at `l=22` (2 minutes, 54 GiB), and
the `l=20` holdout recomputation (about an hour).

## Reference machine

AMD Ryzen 7 5700U, 8 cores and 16 threads, 62 GiB RAM, no GPU. Every timing quoted in this
repository was measured there.
