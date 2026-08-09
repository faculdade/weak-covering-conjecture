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
intervals, and binomial plateau-test diagnostics against both the full range and the tail, as
exploratory output; the paper itself no longer reports either binomial p-value (Round 10: the tail
increment total, `j*(23)-j*(10)=27-15=12`, falls outside the `{10,11}` any constant-rounding model
can produce over 13 steps, a deterministic refutation stated directly in Section 3 instead of a
probability against a null already impossible on its own terms).

`e(l)` is a deterministic sequence, so these statistics compare fit and extrapolation quality.
They do not carry the sampling interpretation of an inference over noisy data, and the script
says so in its own docstring.
