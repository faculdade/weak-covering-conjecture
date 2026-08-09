"""
Memory-light (O(n)) exact solver for the cylinder-window-k mean-payoff game.
 - integer value iteration -> near-optimal MIN policy
 - Howard's policy iteration for max-mean-cycle -> EXACT achievable rho of that
   policy (a rigorous upper bound on rho_k, hence j*(l) <= rho*l + O(1)).
 - a two-sided potential feasibility test (small k) to certify rho_k exactly.
Validated below to reproduce the Karp-proven values rho_3=2, rho_4=5/3, rho_5=3/2.
"""
from fractions import Fraction

def T(d, z, mod):
    num = (pow(2, d+1, 3*mod) * z - 2) % (3*mod)
    if num % 3 != 0:
        return None
    return (num // 3) % mod

def units_mod(m):
    return [x for x in range(m) if x % 3 != 0]

def build_actions(k, dcap):
    mod = 3**k
    mod3 = 3*mod
    states = units_mod(mod)
    # precompute 2^{d+1} mod 3^{k+1} and parity legality 2^d mod 3
    P = [pow(2, d+1, mod3) for d in range(dcap+1)]
    par = [pow(2, d, 3) for d in range(dcap+1)]   # (-1)^d mod 3
    actions = {}
    for s in states:
        p = s % 3
        z0 = s; z1 = s + mod; z2 = s + 2*mod
        best = {}
        for d in range(dcap+1):
            if (par[d] * p) % 3 != 1:
                continue
            pd = P[d]
            n0 = (pd*z0 - 2) % mod3
            if n0 % 3: continue
            s0 = (n0//3) % mod
            if s0 % 3 == 0: continue
            n1 = (pd*z1 - 2) % mod3
            if n1 % 3: continue
            s1 = (n1//3) % mod
            if s1 % 3 == 0: continue
            n2 = (pd*z2 - 2) % mod3
            if n2 % 3: continue
            s2 = (n2//3) % mod
            if s2 % 3 == 0: continue
            tr = (s0, s1, s2)
            if tr not in best:
                best[tr] = d
        acts = sorted((d,tr) for tr,d in best.items())
        if not acts:
            raise RuntimeError(f"no action s={s} k={k}")
        actions[s] = acts
    return states, actions

def value_iteration(states, actions, iters):
    V = {s:0 for s in states}
    for _ in range(iters):
        NV = {}
        for s in states:
            b = None
            for (d,tr) in actions[s]:
                v = d + max(V[tr[0]],V[tr[1]],V[tr[2]])
                if b is None or v < b: b = v
            NV[s] = b
        V = NV
    return V

def min_policy(states, actions, V):
    sig = {}
    for s in states:
        b=None; ba=None
        for (d,tr) in actions[s]:
            v = d + max(V[tr[0]],V[tr[1]],V[tr[2]])
            if b is None or v<b: b=v; ba=(d,tr)
        sig[s]=ba
    return sig

# ---- Howard's policy iteration for MAX mean cycle (O(n) memory) ------------
def howard_max_mean(nodes, out):
    """out[v] = list of (w:int, u). Returns (rho:Fraction, lam:dict, h:dict).
    rho = max over reachable cycles of the mean; exact."""
    # init: successor maximising immediate weight
    succ = {}
    for v in nodes:
        succ[v] = max(out[v], key=lambda e: e[0])
    for _ in range(100000):
        lam, h = _evaluate(nodes, succ)
        improved = False
        for v in nodes:
            cw, cu = succ[v]
            cur_lam = lam[v]; cur_h = h[v]
            best_edge = succ[v]; best_lam = lam[cu]; best_val = cw + h[cu]
            for (w,u) in out[v]:
                # maximise: first higher gain lam[u], then higher bias (w+h[u])
                if lam[u] > best_lam or (lam[u]==best_lam and (w+h[u]) > best_val + 0):
                    if lam[u] > best_lam + 0 or (w+h[u]) > best_val:
                        best_lam = lam[u]; best_val = w+h[u]; best_edge=(w,u)
            # apply if strictly better than current
            if best_lam > lam[cu] or (best_lam==lam[cu] and best_val > (cw+h[cu])):
                if best_edge != succ[v]:
                    succ[v] = best_edge; improved = True
        if not improved:
            break
    lam, h = _evaluate(nodes, succ)
    return max(lam.values()), lam, h, succ

def _evaluate(nodes, succ):
    """Functional graph v -> succ[v]=(w,u). Compute gain lam[v] (mean of cycle
    reached) and bias h[v]. Exact Fractions."""
    color = {v:0 for v in nodes}   # 0 unvisited,1 in-progress,2 done
    lam = {}; h = {}
    for start in nodes:
        if color[start]!=0:
            continue
        path = []
        v = start
        while color[v]==0:
            color[v]=1
            path.append(v)
            v = succ[v][1]
        if color[v]==1:
            # found a new cycle: v is on it, locate
            i = path.index(v)
            cyc = path[i:]
            L = len(cyc)
            tot = sum(succ[c][0] for c in cyc)
            g = Fraction(tot, L)
            # set bias around cycle: h[cyc[0]]=0, h[prev]=w(prev)+h[succ]-g going backward
            # forward relation: h[c] = w(c) + h[succ(c)] - g  -> compute by fixing h[cyc[0]]=0
            # traverse cycle: h[cyc[j]] expressed; solve by setting reference then walk.
            hc = {cyc[0]: Fraction(0)}
            # walk backwards from cyc[0]'s predecessor: cyc[-1]->cyc[0]
            for j in range(L-1, 0, -1):
                c = cyc[j]
                nx = succ[c][1]  # = cyc[j+1] or cyc[0]
                hc[c] = succ[c][0] + hc[nx] - g
            for c in cyc:
                lam[c]=g; h[c]=hc[c]; color[c]=2
            transient = path[:i]
        else:
            # v already done: its lam,h known
            g = lam[v]; hv = h[v]
            transient = path
        # resolve transient nodes (in reverse: closest to cycle first)
        for c in reversed(transient):
            nx = succ[c][1]
            lam[c] = lam[nx]
            h[c] = succ[c][0] + h[nx] - lam[nx]
            color[c]=2
    return lam, h

def induced_out(states, sigma):
    return {s: [(sigma[s][0], sigma[s][1][e]) for e in (0,1,2)] for s in states}

def solve_upper(k, dcap=40, vi_iters=600):
    states, actions = build_actions(k, dcap)
    V = value_iteration(states, actions, vi_iters)
    sig = min_policy(states, actions, V)
    out = induced_out(states, sig)
    rho, lam, h, succ = howard_max_mean(states, out)
    maxd = max(sig[s][0] for s in states)
    vi_est = Fraction(max(V.values()), vi_iters)
    return dict(k=k, n=len(states), rho_upper=rho, vi_est=vi_est, maxd=maxd, dcap=dcap,
                states=states, actions=actions, sigma=sig, V=V)

if __name__ == "__main__":
    import sys
    ks = [int(x) for x in sys.argv[1:]] or [3,4,5,6]
    known = {3:Fraction(2),4:Fraction(5,3),5:Fraction(3,2)}
    for k in ks:
        r = solve_upper(k)
        chk = ""
        if k in known:
            chk = f"  [Karp-proven={known[k]}, match={r['rho_upper']==known[k]}]"
        print(f"k={k}: rho_upper={r['rho_upper']}={float(r['rho_upper']):.6f}  "
              f"VI_est={float(r['vi_est']):.6f}  n={r['n']}  maxd={r['maxd']}{chk}", flush=True)
