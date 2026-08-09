# Section 7: `maxrun(H(l,l+1))` through `l=22` (Proposition 20)

Proposition 20 says that if `maxrun(H(l,l+1))` is bounded by `C` for all `l`, then
`j*(l) <= l + C + 1`. This is the exact computation of that quantity, level by level.

`maxrun(S)` is the length of the longest doubling chain `x, 2x, ..., 2^(t-1)x` inside `S`, all
residues modulo `3^l`. Since 2 is a primitive root modulo `3^l`, it equals the longest run of
consecutive values in the base-2 discrete-log picture of `S`.

## Build and check

```
cargo build --release
python3 verify_small.py                     # 5 seconds
```

`verify_small.py` enumerates the original `R_{j-1,j}` tuples directly with Python `combinations`
for `l=1..10` and compares against the Rust reduced DP. All ten agree.

## Run

```
./target/release/h014-maxrun <ell> <sparse_cutoff> [--progress]
```

`ell=1..18` with cutoff 7 takes about two seconds in total. The larger levels, measured on the
reference machine:

| ell | cutoff | wall | peak RSS |
|---|---|---|---|
| 19 | 7 | 3.7 s | 1.7 GiB |
| 20 | 7 | 12.3 s | 5.5 GiB |
| 21 | 7 | 38.4 s | 17.9 GiB |
| 22 | 8 | 120.0 s | 53.7 GiB |

`ell=23` would need well over physical memory on a 62 GiB machine and was not attempted.

## Expected output

`|H(l,l+1)|` and the maxrun, with one explicit maximal chain per level as a certificate
(consecutive entries differ by a factor of 2 modulo `3^l`, and every entry is a holdout):

| ell | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|----|----|
| `\|H\|` | 0 | 1 | 3 | 10 | 28 | 77 | 208 | 552 | 1,430 | 3,628 | 8,957 |
| maxrun | 0 | 1 | 2 | 2 | 3 | 3 | 3 | 3 | 3 | 4 | 4 |

| ell | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|----|----|----|----|----|----|----|----|----|----|----|
| `\|H\|` | 21,423 | 49,807 | 112,071 | 243,529 | 505,049 | 997,201 | 1,851,056 | 3,195,464 | 5,044,358 | 7,156,001 | 8,951,079 |
| maxrun | 4 | 4 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

The sequence for `ell=1..22` is `0,1,2,2,3,3,3,3,3,4,4,4,4,4,4,3,3,3,3,3,3,3`. It rises once and
falls once, and never exceeds 4. That is finite evidence, not a proof of boundedness.

The `|H(19,20)| = 3,195,464` and `|H(20,21)| = 5,044,358` entries also appear, computed by two
other implementations, in `../section6-holdout-structure/`.

## Two state representations

The DP uses a sparse list for low layers and packed bitsets above `sparse_cutoff`. Setting
`sparse_cutoff=0` forces packed-only, a second representation of the same computation:

```
./target/release/h014-maxrun 19 0
./target/release/h014-maxrun 20 0
```

Both reproduce the hybrid results exactly. Each run also checks
`image_size + holdout_count = 2 * 3^(ell-1)` before reporting.
