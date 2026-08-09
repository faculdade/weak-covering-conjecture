"""Independent, from-scratch Python re-implementation of the j*(l) covering
search, using native Python bignums as bitsets (not numpy, not Rust): a
genuinely different code path and memory representation from
experiments/E-001-jstar-fast/src/main.rs, following main.rs's own documented
recurrence (read from source, not copied) as an independent spec:

  modu = 3^ell, max_exp = ell + j - 1
  state[c] = bitset of residues mod modu reachable as a sum of c terms,
  each term 2^v * 3^c for a chosen exponent v, with each v used in at most
  one c-slot (0/1 knapsack): iterate v from max_exp down to 0, and for each
  v, iterate c from ell-1 down to 0, updating state[c+1] |= rotate(state[c],
  2^v * 3^c mod modu) using state[c]'s value from BEFORE this v-iteration.

  j*(l) is the least j with image_size(l, j) == 2*3^(l-1) (full coverage of
  the unit group).
"""
import sys
import time

def image_size(ell: int, j: int) -> int:
    assert j >= ell
    modu = 3 ** ell
    max_exp = ell + j - 1
    mask = (1 << modu) - 1
    state = [0] * (ell + 1)
    state[0] = 1  # bit 0 set: the empty sum
    for v in range(max_exp, -1, -1):
        p2 = pow(2, v, modu)
        for c in range(ell - 1, -1, -1):
            s = state[c]
            if s == 0:
                continue
            p3 = pow(3, c, modu)
            offset = (p2 * p3) % modu
            if offset == 0:
                rotated = s
            else:
                rotated = ((s << offset) | (s >> (modu - offset))) & mask
            state[c + 1] |= rotated
    return bin(state[ell]).count("1")

def find_j_star(ell: int, j_start: int = None, j_max: int = 200):
    target = 2 * 3 ** (ell - 1)
    j = max(ell, 1) if j_start is None else max(j_start, ell, 1)
    while j <= j_max:
        t0 = time.time()
        sz = image_size(ell, j)
        dt = time.time() - t0
        print(f"  l={ell} j={j}: image_size={sz} (target {target}) [{dt:.2f}s]", flush=True)
        if sz == target:
            return j, sz
        j += 1
    return None

if __name__ == "__main__":
    ell = int(sys.argv[1])
    j_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    j_max = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    t0 = time.time()
    result = find_j_star(ell, j_start, j_max)
    print(f"j*({ell}) = {result[0]} (image size {result[1]}), total time {time.time()-t0:.2f}s")
