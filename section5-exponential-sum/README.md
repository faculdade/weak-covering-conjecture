# Section 5: the exponential sum

For the family `R_{m-1,m}`, with `T = C(2m,m)`,

```
S_l(t) = sum over 2m-1 >= a_0 > ... > a_{m-1} >= 0 of e( t * sum_i 2^{a_i} 3^i / 3^l ).
```

`histogram.cpp` enumerates every decreasing `m`-tuple once and builds the exact residue histogram;
one FFT then gives the whole spectrum. Both scripts here check the histogram mass against
`C(2m,m)` and check Parseval before reporting anything. The main scale used is
`m = ceil(log_4(3) l + 0.5 log_4 l)`, which keeps `T/3^l` of constant order.

`g++` is required. The helper is compiled on first use.

## The full-spectrum criterion is unreachable

```
python3 unreachable_criterion.py     # under a second
```

Every element of `R_{j,k}` is a unit modulo 3 (no term below the top carries a positive power of
3), so `N(0)=0` and `sum_{t!=0} S(t) = -T` exactly; the triangle inequality then forces
`sum_{t!=0} |S(t)| >= T` always, so the naive positivity criterion `sum_{t!=0}|S(t)| < T` can never
hold. The script checks the exact identity directly (no FFT) at six small `(l,j,k)`, plus the
sharper, localized form `S(3^(l-1)) + S(2*3^(l-1)) = -T` that the paper's Proposition (Section 5.1)
uses.

## The `l=18` mass split

```
python3 l1_tail.py --l 18 --threshold 0.001
```

About 95 seconds and 22 GiB of memory. Expected:

```
l=18 m=16 T=C(2m,m)=601080390
primitive frequencies            258280326
max |S(t)|/T                     0.0164133735
primitive L1 of |S(t)|/T         5226.01462
count with |S(t)|/T > 0.001      8014
their L1 contribution            12.2219984
share of the L1 norm             0.233868%
```

Those 8,014 coefficients are the visible spikes. Their share of the `L1` norm is under a quarter
of one percent, which is the point Section 5 makes: excising a sparse exceptional set leaves
essentially all of the mass behind, in the exponentially numerous population near the average
square-root scale.

## The threshold tables

```
python3 count_spectrum.py --l-min 5 --l-max 14 --scale counting \
    --thresholds .02 .01 .005 .002 .001
```

runs in under two seconds and gives, per level, `max |S|/T`, the primitive `L1`, and the counts
above each fixed threshold. `l=5..14` matches the table in Section 5's discussion; `l=15..18`
extends it at the cost above. Adding `--rms-multipliers .5 1 2 4 8` counts above `c/sqrt(T)`
instead, the threshold that actually controls the `L1` norm, where the counts grow by roughly a
factor of three per level.

## Inverse transform and the phase scramble

```
python3 analyze_inverse.py --l 14 --m 16 --scramble-trials 8    # 12 seconds
python3 analyze_inverse.py --l 18                               # 2 minutes, over 20 GiB
```

The first is the phase-randomization experiment. Holding every `|S(t)|` fixed and randomizing the
phases of conjugate-symmetric pairs, at `(l,m) = (14,16)` with seed 20260808, gives eight trials
with `max/RMS` in `[5.1076, 5.2355]`, against `15.7867` for the actual phases. The seed is fixed,
so the interval reproduces exactly.

The second checks that an FFT followed by an inverse FFT recovers all 387,420,489 integer hit
counts (maximum real error `2.8e-14`, zero rounding mismatches) and reports `max/RMS = 19.9746`
against a Gaussian-phase heuristic of `6.3982`.

## The constrained phase-scramble null

```
python3 constrained_phase_null.py --l 14 --m 16 --trials 30 --seed 777    # under a minute
```

The unconstrained scramble above does not preserve `F(z)=0` off units, so it is not the right null
for that constraint on its own. This script builds one that does: `F` vanishing on the subgroup
`3(Z/3^l Z)` is equivalent to `S(t1)+S(t2)+S(t3)=0` for every primitive-frequency triple
`t = r, r+3^(l-1), r+2*3^(l-1)` (parent residue `r` modulo `3^(l-1)`, `3` not dividing `r`), three
fixed magnitudes forming a closed triangle. For a generic (scalene) triple the phases are pinned
down, up to a common rotation and an independent reflection (negating all three phases before
rotating also sums to zero). The script draws both independently per triple, keeps every `|S(t)|`
fixed, and vanishes off units exactly by construction (checked directly, not just consistent with
the identity). Expected at `(l,m)=(14,16)`, seed `777`, 30 trials: `max/RMS_all` averaging `6.35`,
range `[5.93,7.20]`, against the real array's `15.79`, a residual factor of about `2.49`. Both the
zero-sum identity on the actual data and its exact preservation in the constructed null are checked
and printed (`max|sum|` at the `1e-9`--`1e-11` scale against magnitudes near `10^4`).

## Local intensity of the last holdouts

```
python3 local_intensity.py            # l=10..14, under a minute
python3 local_intensity.py --with15   # adds l=15, roughly 10-15 minutes (T=35,345,263,800)
```

For each residue z, the depth-c cell is its class modulo `3^c`, and the depth-c local intensity
`lambda_c(z)` is the total exact hit count within that class divided by `3^(l-c)` (the class
size). A non-homogeneous Poisson model fit to that intensity gives z hole probability
`exp(-lambda_c(z))`. Run against the exact last-holdout set `H(l,j*(l)-1)`, l=10..15, at depths
`c=8,9,10`, every one of the eleven holdouts has `lambda_c` from `2.0` to `108.4` (hole probability
`e^-2` to `e^-108`), against a global occupancy mean in the thousands over the same levels: every
holdout sits in a systematically low-intensity cell, at every depth checked.
