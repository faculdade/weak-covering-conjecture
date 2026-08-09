# Section 2: the exact computation of `j*(l)` (Table 1)

`j*(l)` is the least `j` such that

```
R_{j-1,j} = { sum_{i=0}^{j-1} 2^{a_i} 3^i : 2j-1 >= a_0 > ... > a_{j-1} >= 0 }
```

covers every unit residue modulo `3^l`. This is the Rust implementation that produced Table 1.
The algorithm is a descending-exponent 0/1 dynamic program over residues modulo `3^l`, held as
`Vec<u64>` bitsets and advanced by cyclic rotation, with rayon splitting each shift and OR across
cores.

## Build

```
cargo build --release
```

## Check it first

```
./target/release/jstar-fast validate     # reference values l=1..7, plus one pointwise check
./target/release/bruteforce 4            # direct enumeration of R_{j-1,j}, l<=4, no DP
```

`bruteforce` shares no code path with the DP. It is the only fully independent check available,
since direct enumeration stops being tractable past `l=4`.

Expected output of `validate`: `j*(1..7) = 1, 4, 6, 7, 9, 10, 11`, then the pointwise check at
`l=2, j=2` (image `{1,2,5,7}` modulo 9, missing `{4,8}`), then `ALL VALIDATIONS PASSED`.

## Reproduce Table 1

```
./target/release/jstar-fast run 1 20
```

prints one line per level. The values, with `e(l) = j*(l) - l log_4 3`:

| l | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|----|----|----|
| `j*(l)` | 1 | 4 | 6 | 7 | 9 | 10 | 11 | 12 | 13 | 15 | 16 | 17 |

| l | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|----|----|----|----|----|----|----|----|----|----|----|
| `j*(l)` | 18 | 19 | 20 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 |

The plateau at `l=15, 16` is real, and it is the single zero increment in the whole table.

## Cost

`l=1..16` takes about 6 seconds, `l=17..19` about 3.5 minutes, `l=20` about 10 minutes, all on the
reference machine. Memory is exactly `(l+1) * 3^l / 8` bytes, which is `9 GiB` at `l=20`.

The last three levels are expensive and are not meant to be re-run casually:

| l | state | wall time | notes |
|---|---|---|---|
| 21 | 27 GiB | 748 s | fits in physical RAM |
| 22 | 84 GiB | 10,749 s | needed 500 GiB of swap |
| 23 | 263 GiB | 375,615 s | about 104 hours |

Three attempts at `l=24` failed before completion (two memory-policy kills, one apparent
out-of-memory), which is why the table stops at 23.

`run` appends one line per settled budget to `checkpoints/progress_l<l>.txt` so a restart does
not redo a budget already ruled out. Delete that folder for a run from scratch. There is no
full-state resume: killing the process loses the current attempt.

## Other binaries

- `jstar-fast size <l> <j>` computes a single image size, useful for testing one budget alone.
- `h013_sweep <l> <j> [--summary] [--dump <path>]` and `h013_holdouts <l> <j>` compute the
  holdout set `H(l,j)`, the units modulo `3^l` that `R_{j-1,j}` misses. The raw `u64` dumps in
  `../section6-holdout-structure/dumps-h013/` were produced by `h013_sweep --dump`.
