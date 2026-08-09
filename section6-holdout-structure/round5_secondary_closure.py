#!/usr/bin/env python3
"""Round 5 secondary closure checks for the last-holdout laws.

This is deliberately separate from the round-4 checker.  It verifies:

1. the exact mod-3 theorem for H(l,j*(l)-1);
2. the exact near-extinction equality
       H(l,J-1) = 2 {x in H(l,J-2): x == 2 (mod 3)};
3. the unconditional two-class mod-9 bound and the observed choice of its
   second class;
4. an explicit counterexample to an unrestricted converse to the run-length
   bootstrap: H(6,6) has a 5-chain although H(6,10) is empty;
5. an explicit counterexample to reversing the one-step set inclusion, at
   j=l+1, together with all of its full-width witnesses;
6. the sharper candidate needed to finish the mod-9 proof: in normalized
   coordinates, after slack at least 3, newly covered holes use only the
   opposite-parity/full-width mod-9 class.  This is checked, not proved.

The support DP is imported from the already independently checked E-009
implementation; every assertion below is an exact finite-set assertion.
"""

import sys
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_witness_maps_and_inclusions import (  # noqa: E402
    JSTAR,
    coverage,
    dlog_table,
    maxrun_circular,
    units_mask,
)


def holdout(l, j):
    return set(int(x) for x in np.nonzero(units_mask(l) & ~coverage(l, j))[0])


def normalized_holdout(l, j, hold=None):
    """Complement of R_{l-1,j}; H(l,j)=2^(j-l) normalized_H(l,j)."""
    q = 3**l
    if hold is None:
        hold = holdout(l, j)
    inv = pow(pow(2, j - l, q), -1, q)
    return {(inv * x) % q for x in hold}


def check_last_holdouts(max_l=14):
    for l in range(2, max_l + 1):
        q = 3**l
        J = JSTAR[l]
        last = holdout(l, J - 1)
        previous = holdout(l, J - 2)

        # The two statements proved in the accompanying note.
        assert last and {x % 3 for x in last} == {1}
        rhs = {(2 * x) % q for x in previous if x % 3 == 2}
        assert last == rhs

        # The support recurrence proves the subset; choosing only the second
        # class is the still-empirical mod-9 single-class assertion.
        possible = {pow(4, J, 9), pow(4, J + 1, 9)}
        actual = {x % 9 for x in last}
        assert actual <= possible
        assert actual == {pow(4, J + 1, 9)}
        print(
            f"l={l:2d} J={J:2d}: |last|={len(last):5d}, "
            f"mod9={sorted(actual)}, near-extinction equality PASS"
        )


def check_candidate_repair(max_l=14):
    """Check the precise unproved lemma that would imply the mod-9 law."""
    for l in range(2, max_l + 1):
        J = JSTAR[l]
        normalized = {
            j: normalized_holdout(l, j) for j in range(l, J + 1)
        }
        for j in range(l, J):
            removed = normalized[j] - normalized[j + 1]
            N = l + j  # new top exponent in R_{l-1,j+1}
            target_mod3 = pow(2, N, 3)
            assert all(x % 3 == target_mod3 for x in removed)

            alternative = pow(2, N + 2, 9)
            observed = pow(2, N + 4, 9)
            classes = {x % 9 for x in removed}
            assert classes <= {alternative, observed}
            if j - l >= 3:
                # Stronger than the transition statement: the alternative
                # class is absent from the whole old holdout set.
                assert all(x % 9 != alternative for x in normalized[j])
                assert alternative not in classes

    print(
        f"candidate cylinder-covering lemma PASS exactly for l=2..{max_l}, "
        "all computed budgets/transitions with j-l>=3 (finite check only)"
    )


def check_converse_counterexample():
    l, j, q = 6, 6, 3**6
    chain = [187, 374, 19, 38, 76]
    h = holdout(l, j)
    assert all(x in h for x in chain)
    assert all(chain[i + 1] == (2 * chain[i]) % q for i in range(4))
    dlog, order = dlog_table(l)
    assert maxrun_circular(dlog[np.array(sorted(h))], order) == 5
    assert not holdout(l, 10)
    print(
        "unrestricted converse FALSE: H(6,6) contains the exact 5-chain "
        f"{chain}, but H(6,10) is empty"
    )


def check_reverse_inclusion_counterexample():
    """Falsify 2H(j) intersection 4H(j) subset H(j+1), even at j=l+1."""
    l, j, q, y = 7, 8, 3**7, 1547
    inv2, inv4 = pow(2, -1, q), pow(4, -1, q)
    p2, p4 = (y * inv2) % q, (y * inv4) % q
    old = holdout(l, j)
    new = holdout(l, j + 1)
    assert (p4, p2) == (2027, 1867)
    assert p4 in old and p2 in old
    assert (2 * p4) % q == p2 and (2 * p2) % q == y
    assert y not in new

    witnesses = []
    for selected in itertools.combinations(range(2 * (j + 1)), j + 1):
        alpha = tuple(reversed(selected))
        value = sum((1 << a) * 3**i for i, a in enumerate(alpha)) % q
        if value == y:
            witnesses.append(alpha)
    expected = [
        (17, 12, 11, 9, 5, 3, 2, 1, 0),
        (17, 14, 10, 6, 5, 4, 2, 1, 0),
    ]
    assert witnesses == expected
    assert all(w[0] == 17 and w[-1] == 0 for w in witnesses)
    print(
        "reverse one-step inclusion FALSE at j=l+1: y=1547 is in "
        "2H(7,8) intersection 4H(7,8), but is covered at j=9; its exactly "
        "two j=9 witnesses are both doubly saturated/full-width"
    )


if __name__ == "__main__":
    check_last_holdouts()
    check_candidate_repair()
    check_converse_counterexample()
    check_reverse_inclusion_counterexample()
