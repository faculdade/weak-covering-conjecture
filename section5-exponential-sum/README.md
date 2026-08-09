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
