# Section 3: growth models for `e(l)` (Table 2)

Fits four functional forms to `e(l) = j*(l) - l log_4 3` on the tail `l=10..23` by ordinary least
squares, and runs the plateau-frequency test. The `j*(l)` table is hardcoded at the top of
`experiment.py`; it comes from `../section2-jstar-computation/`.

## Run

```
python3 experiment.py
```

Needs `numpy` and `scipy`. Runs in under a second.

## Expected output

Table 2 of the paper:

| model | k | dAIC | dBIC | LOOCV RMSE |
|---|---|---|---|---|
| constant | 1 | 16.091 | 15.452 | 0.5200 |
| logarithmic | 2 | 1.512 | 1.512 | 0.2991 |
| square-root | 2 | 0.722 | 0.722 | 0.2909 |
| slow-linear | 2 | 0.000 | 0.000 | 0.2837 |

The script also prints the regressor correlations (all at least 0.993), the slope confidence
intervals, and the plateau test: 1 plateau in the 22 increments `l=1..23` against
`Bin(22, 0.2053)` gives `p=0.0426`, and the same test on the 13 increments of the tail `l=10..23`
gives `p=0.2199`. Both figures appear in Section 3.

`e(l)` is a deterministic sequence, so these statistics compare fit and extrapolation quality.
They do not carry the sampling interpretation of an inference over noisy data, and the script
says so in its own docstring.
