"""
Independent, round-19-style check: take the OPTIMAL window-k policy sigma_d found
by the mean-payoff solver and RUN it on the actual finite j*(l) covering DP
(modulus shrinking one power of 3 per step, over EVERY unit residue mod 3^l).
Confirm the worst-case total cost tracks rho_k * l + O(1), i.e. the abstract game
value is a real, achievable upper bound on j*(l).
"""
from fractions import Fraction
from mpg4 import solve

def T(d, z, mod):
    num = (pow(2, d+1, 3*mod) * z - 2) % (3*mod)
    if num % 3 != 0:
        return None
    return (num // 3) % mod

def units_mod(m):
    return [x for x in range(m) if x % 3 != 0] if m > 1 else [1]

def run_policy_real(sigma_d, k, l):
    """Apply window-k policy to the real l-step problem over all unit targets."""
    modk = 3**k
    worst = -1; worst_z = None; fails = 0
    for z0 in units_mod(3**l):
        z = z0; total = 0; ok = True
        for step in range(l):
            remaining = l - step
            key = z % modk
            d = sigma_d.get(key)
            if d is None:
                ok = False; break
            mod_next = 3**(remaining-1)
            if mod_next > 1:
                zn = T(d, z, mod_next)
                if zn is None or zn % 3 == 0:
                    ok = False; break
            else:
                zn = 0
            total += d; z = zn
        if not ok:
            fails += 1; continue
        if total > worst:
            worst = total; worst_z = z0
    return worst, worst_z, fails

if __name__ == "__main__":
    KNOWN = {1:1,2:4,3:6,4:7,5:9,6:10,7:11,8:12,9:13,10:15,11:16,12:17,13:18,14:19}
    for k in (4, 7):
        r = solve(k)
        rho = r['rho']
        sigma_d = {s: r['sigma'][s][0] for s in r['states']}
        print(f"\n=== window k={k}, rho_k={rho}={float(rho):.4f} ===")
        print(f"{'l':>3} {'worst cost':>10} {'rho*l':>8} {'cost/l':>7} {'known j*':>8} {'fails':>6}")
        for l in range(k, 15):
            worst, wz, fails = run_policy_real(sigma_d, k, l)
            print(f"{l:3d} {worst:10d} {float(rho*l):8.2f} {worst/l:7.3f} "
                  f"{str(KNOWN.get(l,'?')):>8} {fails:6d}")
