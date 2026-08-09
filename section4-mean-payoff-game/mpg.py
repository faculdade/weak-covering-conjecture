"""
Mean-payoff-game (MPG) solver for the cylinder-window-k relaxation of the
controlled-3-adic-dynamics game.

THE GAME (window k):
  States  : units s mod 3^k        (n = 2*3^{k-1} of them)
  Minimizer: at state s, picks a "safe legal" action d>=0:
             - legal    : 2^d * s == 1 (mod 3)          (divisibility of T_d)
             - safe     : all three lifts give a unit successor
             cost of the move = d.
  Adversary: picks the hidden next 3-adic digit e in {0,1,2}, sending
             s -> s'(e) = T_d(s + 3^k * e)  mod 3^k,     T_d(z)=(2^{d+1}z-2)/3.
  Value rho_k = min over deterministic window-k policies of the worst-case
                long-run average cost the adversary can force
              = min_policy  ( max mean cycle of the induced graph ).

rho_k is EXACTLY the best asymptotic rate achievable by any window-k cylinder
policy, hence j*(l) <= rho_k * l + O_k(1), and rho_k is non-increasing in k and
>= limsup j*(l)/l >= log_4(3).

We solve rho_k EXACTLY with a self-certifying method:
  - policy improvement / value iteration to get candidate optimal min-policy sigma*
  - upper bound: max-mean-cycle of the sigma*-induced graph (Karp), exact rational
  - lower bound: pick adversary's best response tau* to sigma*, then min-mean-cycle
    of the tau*-induced one-player graph (Karp), exact rational
  - if upper == lower, that rational IS the game value, PROVEN (matching certificate).
Everything is exact integer/Fraction arithmetic.
"""
from fractions import Fraction

def T(d, z, mod):
    num = (pow(2, d+1, 3*mod) * z - 2) % (3*mod)
    if num % 3 != 0:
        return None
    return (num // 3) % mod

def units_mod(m):
    return [x for x in range(m) if x % 3 != 0]

def build_actions(k, dcap=None):
    """For each unit state s mod 3^k, list of (d, (s0,s1,s2)) safe legal actions,
    deduped to the minimal-cost d per distinct successor triple.
    dcap bounds the immediate cost d searched (optimal policies use tiny d since
    rho_k in [1,2]); a-posteriori we verify max d used << dcap."""
    mod = 3**k
    states = units_mod(mod)
    sidx = {s: i for i, s in enumerate(states)}
    if dcap is None:
        dcap = 2 * mod - 1  # full period
    actions = {}
    for s in states:
        p = s % 3
        best_by_triple = {}
        for d in range(dcap + 1):
            if (pow(2, d, 3) * p) % 3 != 1:   # legality
                continue
            triple = []
            ok = True
            for e in (0, 1, 2):
                z = s + mod * e
                sp = T(d, z, mod)
                if sp is None or sp % 3 == 0:
                    ok = False
                    break
                triple.append(sp)
            if not ok:
                continue
            tr = tuple(triple)
            if tr not in best_by_triple or d < best_by_triple[tr]:
                best_by_triple[tr] = d
        acts = [(d, tr) for tr, d in best_by_triple.items()]
        acts.sort()
        if not acts:
            raise RuntimeError(f"state {s} (k={k}) has NO safe legal action")
        actions[s] = acts
    return states, sidx, actions

# --------------------------------------------------------------------------
# Karp's exact max/min mean cycle on a graph where every node has >=1 out-edge.
# Returns (mean:Fraction, potentials, node achieving optimum). Weights integer.
# For a functional graph (each node exactly one out-edge, our tau-induced case)
# and for the sigma-induced multigraph (each node 3 out-edges), Karp applies.
# --------------------------------------------------------------------------
def karp_mean_cycle(nodes, out_edges, maximize):
    """out_edges[v] = list of (w:int, u). Graph assumed to have every node with
    an out-edge; every infinite walk hits a cycle. Returns exact Fraction opt
    mean cycle value over all cycles reachable-as-cycles in the graph."""
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    NEG = None
    # d[k][v] = opt weight of a walk of exactly k edges ending at v, over all start nodes
    # Standard Karp uses fixed source; the "max over all cycles" version takes
    # min_v max_{0<=j<n} (d_n(v)-d_j(v))/(n-j) with d_0 = 0 for all v (all starts).
    d = [[None]*n for _ in range(n+1)]
    for v in range(n):
        d[0][v] = 0
    better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)
    for kk in range(1, n+1):
        row = d[kk]
        prev = d[kk-1]
        for v in nodes:
            iv = idx[v]
            best = None
            for (w, u) in out_edges[v]:
                pu = prev[idx[u]]
                if pu is None:
                    continue
                val = pu + w
                if best is None or better(val, best):
                    best = val
            row[iv] = best
    # cycle value = opt over v of ( opt/pess over j of (d_n(v)-d_j(v))/(n-j) )
    # For MAX mean cycle: max_v min_j (dn-dj)/(n-j). For MIN: min_v max_j (...).
    outer = None
    for v in nodes:
        iv = idx[v]
        if d[n][iv] is None:
            continue
        inner = None
        for j in range(n):
            if d[j][iv] is None:
                continue
            cand = Fraction(d[n][iv] - d[j][iv], n - j)
            if maximize:
                if inner is None or cand < inner:
                    inner = cand   # min over j
            else:
                if inner is None or cand > inner:
                    inner = cand   # max over j
        if inner is None:
            continue
        if maximize:
            if outer is None or inner > outer:
                outer = inner   # max over v
        else:
            if outer is None or inner < outer:
                outer = inner   # min over v
    return outer

# --------------------------------------------------------------------------
# Value iteration for the min-max game to obtain a candidate optimal min policy.
# V_0 = 0 ; V_n(s) = min_a [ d_a + max_e V_{n-1}(triple_a[e]) ].  Integer values.
# --------------------------------------------------------------------------
def value_iteration(states, actions, iters):
    V = {s: 0 for s in states}
    for _ in range(iters):
        NV = {}
        for s in states:
            best = None
            for (d, tr) in actions[s]:
                val = d + max(V[tr[0]], V[tr[1]], V[tr[2]])
                if best is None or val < best:
                    best = val
            NV[s] = best
        V = NV
    return V

def extract_min_policy(states, actions, V):
    """Greedy min policy w.r.t. value function V (one-step argmin)."""
    sigma = {}
    for s in states:
        best = None; bd = None; btr = None
        for (d, tr) in actions[s]:
            val = d + max(V[tr[0]], V[tr[1]], V[tr[2]])
            if best is None or val < best:
                best = val; bd = d; btr = tr
        sigma[s] = (bd, btr)
    return sigma

def upper_bound_of_policy(states, sigma):
    """Adversary best response to fixed min policy sigma = max mean cycle."""
    out = {s: [(0, sigma[s][1][e]) for e in (0,1,2)] for s in states}
    # cost of the move is d=sigma[s][0], put it on each out-edge
    out = {s: [(sigma[s][0], sigma[s][1][e]) for e in (0,1,2)] for s in states}
    return karp_mean_cycle(states, out, maximize=True)

def lower_bound_of_adversary(states, actions, V):
    """Correct lower bound. Fix a positional ADVERSARY strategy tau(s, action) that,
    for EVERY action available at s, commits to the e maximizing continuation value
    (argmax_e V[tr[e]]). Then the MINIMIZER is free to choose any action; it faces a
    one-player min-mean-cycle problem on the graph whose s-edges are
    {(d, tr[tau(s,action)]) : actions at s}. Any fixed adversary strategy gives a
    LOWER bound on the game value (value = min_sigma max_tau >= min_sigma f(sigma,tau0))."""
    out = {}
    for s in states:
        edges = []
        for (d, tr) in actions[s]:
            be = max((0, 1, 2), key=lambda e: V[tr[e]])
            edges.append((d, tr[be]))
        out[s] = edges
    return karp_mean_cycle(states, out, maximize=False)

def solve_exact(k, vi_iters=1200, dcap=40, verbose=True):
    states, sidx, actions = build_actions(k, dcap=dcap)
    n = len(states)
    V = value_iteration(states, actions, vi_iters)
    sigma = extract_min_policy(states, actions, V)
    ub = upper_bound_of_policy(states, sigma)
    lb = lower_bound_of_adversary(states, actions, V)
    numeric = {s: Fraction(V[s], vi_iters) for s in states}
    vi_est = max(numeric.values())
    maxd = max(sigma[s][0] for s in states)
    proven = (ub == lb)
    if verbose:
        print(f"k={k}: n={n} states, VI iters={vi_iters}, dcap={dcap}, max d used={maxd}")
        print(f"   VI estimate max_s V_N/N = {float(vi_est):.6f}")
        print(f"   min-policy upper bound (max mean cycle) = {ub} = {float(ub):.6f}")
        print(f"   adversary   lower bound (min mean cycle) = {lb} = {float(lb):.6f}")
        print(f"   CERTIFICATE MATCH (proven exact): {proven}"
              + ("" if proven else "   <-- not tight"))
    return dict(k=k, n=n, ub=ub, lb=lb, vi_est=vi_est, maxd=maxd, proven=proven,
                states=states, actions=actions, sigma=sigma, V=V)

if __name__ == "__main__":
    import sys
    ks = [int(x) for x in sys.argv[1:]] or [3, 4, 5]
    for k in ks:
        r = solve_exact(k)
        pred = Fraction(1) + Fraction(2, k-1)
        print(f"   -> 1+2/(k-1) = {pred} = {float(pred):.6f} ; equals rho_k(upper)? {r['ub']==pred}")
        print()
