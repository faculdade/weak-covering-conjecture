#!/usr/bin/env python3
"""Round 5: maxrun + class statistics from an h013_holdouts dump.

Usage: python3 maxrun_from_dump.py <dumpfile> [<dumpfile> ...]

Reads the `ell=... j=...` header and the `holdouts=[...]` line of each dump
produced by experiments/E-001-jstar-fast/src/bin/h013_holdouts, then reports:
  |H(l,j)|, maxrun (longest doubling chain x,2x,...,2^(t-1)x inside H), the
  number of maximal chains and up to three example chain heads, mod-9 class
  histogram, and the iid null-model calibration ln(N)/ln(N/|H|) with
  N = phi(3^l) = 2*3^(l-1) (expected longest success run of an iid set of the
  same density on the dlog circle).

maxrun is invariant under the extractor's global 2^(j-l) rescaling, since the
doubling map commutes with it.
"""
import math
import sys


def process(path):
    ell = j = None
    hold = None
    with open(path) as f:
        for line in f:
            if line.startswith('ell='):
                parts = dict(p.split('=', 1) for p in line.split() if '=' in p)
                ell, j = int(parts['ell']), int(parts['j'])
            elif line.startswith('holdouts=['):
                hold = eval(line[len('holdouts='):])  # list literal from our own tool
                break
    assert ell is not None and hold is not None, f'{path}: bad dump'
    q = 3 ** ell
    n_units = 2 * 3 ** (ell - 1)
    s = set(hold)
    inv2 = pow(2, -1, q)
    best, heads, nmax = 0, [], 0
    for h in s:
        if (h * inv2) % q not in s:  # chain head
            r, x = 1, h
            while (x * 2) % q in s:
                x = (x * 2) % q
                r += 1
            if r > best:
                best, heads, nmax = r, [h], 1
            elif r == best:
                nmax += 1
                if len(heads) < 3:
                    heads.append(h)
    mod9 = {}
    for x in hold:
        mod9[x % 9] = mod9.get(x % 9, 0) + 1
    p = len(hold) / n_units
    null = math.log(n_units) / math.log(1 / p) if 0 < p < 1 else float('nan')
    print(f'l={ell} j={j}: |H|={len(hold)} density={p:.3e} '
          f'maxrun={best} (maximal chains: {nmax}, heads e.g. {heads}) '
          f'null-model ln(N)/ln(1/p)={null:.2f}')
    print(f'  mod-9 histogram: {dict(sorted(mod9.items()))}')
    return ell, j, len(hold), best


if __name__ == '__main__':
    for path in sys.argv[1:]:
        process(path)
