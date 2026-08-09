"""
Nested Howard strategy improvement for the min-max mean-payoff cylinder game.
Exact, O(n) memory, converges in few iterations independent of n.

  Outer: MIN improves its positional policy sigma.
  Inner: given sigma, MAX (adversary) best-responds = max-mean-cycle (Howard),
         yielding gain lam[s] and bias h[s].
  MIN improvement: at each state pick the action minimising, lexicographically,
     ( gain  = max_e lam[succ_e] ,  bias = d + max_{e in argmax-gain} h[succ_e] ).
  rho_k = max_s lam[s] at convergence.

Validated to reproduce the Karp-proven rho_3=2, rho_4=5/3, rho_5=3/2.
Cross-checked at convergence by an adversary-potential LOWER bound.
"""
from fractions import Fraction
from mpg3 import build_actions, howard_max_mean, _evaluate, value_iteration, min_policy, induced_out

def gain_bias_of_action(d, tr, lam, h):
    g = max(lam[tr[0]], lam[tr[1]], lam[tr[2]])
    b = d + max(h[tr[e]] for e in (0,1,2) if lam[tr[e]] == g)
    return g, b

def nested_howard(states, actions, sigma0, verbose=False):
    sigma = dict(sigma0)
    history = []
    for it in range(100000):
        out = induced_out(states, sigma)
        rho, lam, h, _ = howard_max_mean(states, out)
        history.append(rho)
        improved = False
        newsig = {}
        for s in states:
            cd, ctr = sigma[s]
            cur = gain_bias_of_action(cd, ctr, lam, h)
            best = cur; ba = sigma[s]
            for (d, tr) in actions[s]:
                gb = gain_bias_of_action(d, tr, lam, h)
                if gb < best:
                    best = gb; ba = (d, tr)
            newsig[s] = ba
            if ba != sigma[s] and best < cur:
                improved = True
        sigma = newsig
        if verbose:
            print(f"    outer {it}: rho={rho}={float(rho):.6f} improved={improved}", flush=True)
        if not improved:
            break
    return rho, sigma, lam, h

def adversary_lower_bound(states, actions, lam, h):
    """Fix adversary positional strategy tau(s,action)=argmax over (gain,bias) of e,
    using the evaluated (lam,h). Then MIN best-responds over all actions -> min mean
    cycle = a lower bound on rho_k. Uses Howard for MIN mean-cycle via negation."""
    out = {}
    for s in states:
        edges = []
        for (d, tr) in actions[s]:
            # adversary picks e maximizing (lam, h)
            be = max((0,1,2), key=lambda e: (lam[tr[e]], h[tr[e]]))
            edges.append((d, tr[be]))
        out[s] = edges
    # min mean cycle = -(max mean cycle of negated weights)
    negout = {s: [(-w, u) for (w, u) in out[s]] for s in states}
    negrho, _, _, _ = howard_max_mean(states, negout)
    return -negrho

def solve(k, dcap=40, verbose=False):
    states, actions = build_actions(k, dcap)
    # seed with a short VI to get a decent starting policy
    V = value_iteration(states, actions, min(300, 50 + 20*k))
    sigma0 = min_policy(states, actions, V)
    rho, sigma, lam, h = nested_howard(states, actions, sigma0, verbose=verbose)
    lb = adversary_lower_bound(states, actions, lam, h)
    maxd = max(sigma[s][0] for s in states)
    Ck = max(h.values()) - min(h.values())
    return dict(k=k, n=len(states), rho=rho, lb=lb, proven=(rho==lb), maxd=maxd, dcap=dcap,
                Ck=Ck, sigma=sigma, h=h, lam=lam, states=states, actions=actions)

if __name__ == "__main__":
    import sys, json
    ks = [int(x) for x in sys.argv[1:]] or [3,4,5,6]
    known = {3:Fraction(2),4:Fraction(5,3),5:Fraction(3,2)}
    for k in ks:
        r = solve(k)
        chk = ""
        if k in known:
            chk = f"  [proven-Karp={known[k]}, match={r['rho']==known[k]}]"
        print(f"k={k}: rho_k={r['rho']}={float(r['rho']):.6f}  lower={r['lb']}={float(r['lb']):.6f}  "
              f"tight={r['proven']}  n={r['n']}  maxd={r['maxd']}  C_k={r['Ck']}={float(r['Ck']):.6f}{chk}", flush=True)
        # save the certificate (policy + potentials) so C_k / A3-style diamond checks never need a re-solve
        cert = {
            "k": k, "n": r["n"], "rho": [r["rho"].numerator, r["rho"].denominator],
            "Ck": [r["Ck"].numerator, r["Ck"].denominator],
            "sigma": {str(s): r["sigma"][s][0] for s in r["states"]},
            "h": {str(s): [r["h"][s].numerator, r["h"][s].denominator] for s in r["states"]},
            "lam": {str(s): [r["lam"][s].numerator, r["lam"][s].denominator] for s in r["states"]},
        }
        with open(f"certificate_k{k}.json", "w") as f:
            json.dump(cert, f)
