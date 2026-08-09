# Section 4: the mean-payoff-game bound (Table 3, Theorem 4, Corollary 5)

The window-`k` relaxation of the covering budget is a mean-payoff game on the `2*3^(k-1)` unit
states modulo `3^k`. Its value `rho_k` gives `j*(l) <= rho_k * l + C_k`. At `k=14`,
`rho_14 = 9/8`, which is Corollary 5.

Theorem 4 does not need the game's theory. It needs, for each `k`, one policy `sigma`, one value
`rho_k`, and one potential `h` satisfying

```
d_sigma(s) + h(s'(e)) <= rho_k + h(s)     at every unit state s and every adversary digit e,
```

with `C_k = max h - min h`. Those twelve objects are stored here as `certificate_k<k>.json`, and
`verify_certificate.py` re-checks all of them without calling the solver.

## Check the certificates

```
python3 verify_certificate.py
```

It rebuilds the state set and the transitions from `k` alone, then checks, at every state: that
the stored move is legal (`2^d z == 1 mod 3`), that it is safe (all three lifts stay units), that
the telescoping inequality holds for all three digits, that `C_k` equals the spread of `h`, and
that `rho_k` agrees with Table 3. Expected output, one line per `k`, all `OK`:

| k | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|----|----|----|----|----|
| `rho_k` | 2 | 5/3 | 3/2 | 3/2 | 7/5 | 25/19 | 5/4 | 11/9 | 6/5 | 7/6 | 119/104 | 9/8 |
| `C_k` | 5 | 19/3 | 15/2 | 9 | 10 | 207/19 | 47/4 | 115/9 | 69/5 | 44/3 | 1621/104 | 33/2 |

`k=3..11` take a few seconds together, `k=12` and `k=13` about 13 seconds between them,
`k=14` about 28 seconds. The three
largest certificates are stored gzipped (`certificate_k14.json` is 166 MB uncompressed, past
GitHub's per-file limit); the verifier reads either form.

## Recompute a certificate

```
python3 mpg4.py 3 4 5 6 7 8 9        # nested Howard strategy improvement, exact rationals
```

`mpg4.py` writes `certificate_k<k>.json` on every run and prints `rho_k`, the adversary lower
bound, and `C_k`. Upper and lower bound coincide at every `k`, which is what makes `rho_k` exact
rather than merely an upper bound. Cost grows with `n = 2*3^(k-1)`: `k<=9` is minutes, `k=12`
about 76 minutes, `k=13` about 5.5 hours. `k=14` was run once, overnight.

`mpg.py` is a second, Karp-based value-iteration solver, `O(n^2)` in memory, usable up to `k=7`:

```
python3 mpg.py 3 4 5
```

## Empirical Result 3 and the policy cross-check

```
python3 step_a_grounding.py     # about 30 seconds
```

reproduces `j*(l)` from the full-precision game against direct brute-force enumeration of
`R_{j-1,j}` for `l=1..6`, and against the known table for `l=1..12`. That is the evidence behind
Empirical Result 3, on which Theorem 4 is conditional.

```
python3 verify_on_real.py       # about 1 minute
```

runs the extracted window-`k` policies for `k=4` and `k=7` against the actual covering dynamic
program, over every unit residue modulo `3^l` for `l` up to 14. Expected: `fails = 0` in every
row, no illegal and no unsafe move anywhere.

`mpg3.py` holds the shared action construction and the max-mean-cycle routine. `certificate.py`
is the original in-solver extractor, kept because it is what produced the stored files; use
`verify_certificate.py` to check them.
