"""
Step A: Independently ground the "controlled 3-adic dynamics" Bellman game against
(1) a direct brute-force computation of j*(l) from the R_{j-1,j} covering definition, and
(2) the known exact table.

This confirms *why* the game equals j*(l), from first principles, not by trusting round 19.

Known table j*(l), l=1..23 (from experiments/E-002 DATA, itself from jstar-fast):
 1,4,6,7,9,10,11,12,13,15,16,17,18,19,20,20,21,22,23,24,25,26,27
"""
from functools import lru_cache
from itertools import combinations

KNOWN = {1:1,2:4,3:6,4:7,5:9,6:10,7:11,8:12,9:13,10:15,11:16,12:17,13:18,
         14:19,15:20,16:20,17:21,18:22,19:23,20:24,21:25,22:26,23:27}

# ---------------------------------------------------------------------------
# (1) DIRECT brute-force j*(l) from the covering definition.
#
# R_{J-1,J} = { sum_{i=0}^{J-1} 2^{a_i} 3^i : 2J-1 >= a_0 > a_1 > ... > a_{J-1} >= 0 }.
# j*(l) = smallest J such that { v mod 3^l : v in R_{J-1,J} } contains every unit mod 3^l.
#
# Only exponent positions i=0..l-1 matter mod 3^l (3^i == 0 for i>=l). So for a given J,
# enumerate all strictly-decreasing exponent tuples a_0>...>a_{J-1}>=0 with a_0<=2J-1,
# reduce value mod 3^l, and test coverage of the units. (Feasible for small l.)
# ---------------------------------------------------------------------------
def covers_mod(J, l):
    mod = 3**l
    units = set(x for x in range(mod) if x % 3 != 0)
    hit = set()
    top = 2*J - 1
    # choose J distinct exponents from 0..top, strictly decreasing (i.e. any J-subset)
    # value mod 3^l: position i (the i-th largest exponent) carries weight 3^i, i<l only.
    for combo in combinations(range(top+1), J):
        exps = sorted(combo, reverse=True)  # a_0 > a_1 > ...
        v = 0
        for i in range(min(J, l)):
            v = (v + pow(2, exps[i], mod) * pow(3, i, mod)) % mod
        hit.add(v % mod)
        if units <= hit:
            return True
    return units <= hit

def jstar_bruteforce(l, Jmax=40):
    for J in range(1, Jmax+1):
        if covers_mod(J, l):
            return J
    return None

# ---------------------------------------------------------------------------
# (2) The controlled-3-adic-dynamics Bellman game (round 19's reformulation),
# reimplemented independently from the stated definitions.
#   state z (3-adic unit); legal d>=0 iff 2^d z == 1 (mod 3); T_d(z)=(2^{d+1} z - 2)/3.
#   j*(l) = max over unit z0 mod 3^l of min over l-step policies of sum(d_i),
#   modulus shrinking one power of 3 per step.
# ---------------------------------------------------------------------------
def T(d, z, mod):
    num = (pow(2, d+1, 3*mod) * z - 2) % (3*mod)
    if num % 3 != 0:
        return None
    return (num // 3) % mod

def legal_ds(z, upto):
    zm3 = z % 3
    return [d for d in range(upto) if (pow(2, d, 3) * zm3) % 3 == 1]

def units_mod(m):
    return [x for x in range(m) if x % 3 != 0] if m > 1 else [1]

def jstar_game(l, max_d_try=80):
    @lru_cache(maxsize=None)
    def cost(step, z):
        remaining = l - step
        if remaining == 0:
            return 0
        mod_next = 3**(remaining-1)
        best = None
        for d in legal_ds(z, max_d_try):
            if mod_next > 1:
                znext = T(d, z, mod_next)
                if znext is None or znext % 3 == 0:
                    continue
            else:
                znext = 0
            sub = cost(step+1, znext)
            if sub is None:
                continue
            c = d + sub
            if best is None or c < best:
                best = c
        return best
    worst = 0
    for z0 in units_mod(3**l):
        c = cost(0, z0)
        if c is None:
            return None
        worst = max(worst, c)
    return worst

if __name__ == "__main__":
    print("l | known | game DP | brute-force covering")
    print("-"*48)
    for l in range(1, 9):
        kg = KNOWN[l]
        g = jstar_game(l)
        # brute force gets expensive; cap l where it's still fast
        bf = jstar_bruteforce(l) if l <= 6 else None
        bf_s = str(bf) if bf is not None else "(skip)"
        flag = "OK" if (g == kg and (bf is None or bf == kg)) else "MISMATCH"
        print(f"{l:2d} | {kg:5d} | {g:7d} | {bf_s:>8}   {flag}")
    print()
    # extend game DP past round 19's l=10 to l=12 as a further independent check
    for l in range(9, 13):
        g = jstar_game(l)
        kg = KNOWN[l]
        print(f"l={l}: game DP = {g}, known = {kg}, match = {g==kg}")
