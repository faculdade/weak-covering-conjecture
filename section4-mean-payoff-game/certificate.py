"""
Extract, for each window k, the explicit certificate for j*(l) <= rho_k * l + C_k:
a potential h on states with  d(s) + h[s'(e)] <= rho_k + h[s]  for ALL e in {0,1,2}
(the optimal window-k policy sigma). Then C_k = max_s h - min_s h is the additive
constant. Verify the inequality holds at every state and every hidden digit e.
"""
from fractions import Fraction
from mpg4 import solve
from mpg3 import howard_max_mean, induced_out

def certify(k):
    r = solve(k)
    rho = r['rho']; states = r['states']; sigma = r['sigma']
    out = induced_out(states, sigma)
    rho2, lam, h, succ = howard_max_mean(states, out)
    assert rho2 == rho, (rho, rho2)
    # verify telescoping inequality for ALL e
    viol = 0; worst_slack = None
    for s in states:
        d, tr = sigma[s]
        for e in (0,1,2):
            lhs = d + h[tr[e]]
            rhs = rho + h[s]
            if lhs > rhs:
                viol += 1
            slack = rhs - lhs
            if worst_slack is None or slack < worst_slack:
                worst_slack = slack
    C = max(h.values()) - min(h.values())
    return rho, C, viol, len(states)

if __name__ == "__main__":
    import math
    lg = math.log(3)/math.log(4)
    print(f"log_4(3) = {lg:.6f}")
    print(f"{'k':>3} {'rho_k':>8} {'=dec':>8} {'C_k(bias spread)':>18} {'cert-violations':>16}")
    for k in range(3, 11):
        rho, C, viol, n = certify(k)
        print(f"{k:3d} {str(rho):>8} {float(rho):8.5f} {str(C):>18} {viol:16d}   (n={n})")
