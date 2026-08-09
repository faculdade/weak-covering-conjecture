// Independent cross-check for H-001, computed by a genuinely different
// method than main.rs: direct enumeration of R_{j-1,j} itself (no
// R_{ell-1,j} reduction, no bitset/rotation DP), used only for small l
// where C(2j,j) is tractable (l<=4). Confirms both the reduction identity
// and the bitset DP logic in main.rs independently, per the project's
// honesty rule: never trust the fast/clever method without a naive check.
//
// R_{j-1,j} := { sum_{i=0}^{j-1} 2^{a_i} 3^i : 2j-1 >= a_0 > a_1 > ... > a_{j-1} >= 0 }

use std::collections::HashSet;
use std::env;

fn pow_u64(base: u64, exp: u32) -> u64 {
    base.checked_pow(exp).expect("overflow")
}

fn pow_mod128(base: u64, mut exp: u64, modu: u64) -> u64 {
    let mut result: u128 = 1;
    let mut b: u128 = (base % modu) as u128;
    let m = modu as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        b = (b * b) % m;
        exp >>= 1;
    }
    result as u64
}

/// Recursively choose `remaining` strictly decreasing exponents, all <=
/// `upper_bound`, all >= 0. `remaining` is the count still needed
/// (including the one chosen at this call), so at every level the loop
/// variable must be >= remaining-1 to leave room for the rest.
fn recurse(
    remaining: u32,
    upper_bound: i64, // i64, not u32: can legitimately go to -1 at the deepest call, where it's
    // unused (remaining==0 hits the base case before ever reading it); u32 would underflow that
    // subtraction and panic under debug-mode overflow checks even though the value is dead.
    chosen: &mut Vec<u32>,
    modu: u64,
    image: &mut HashSet<u64>,
) {
    if remaining == 0 {
        let mut sum: u128 = 0;
        for (i, &a) in chosen.iter().enumerate() {
            let term = (pow_mod128(2, a as u64, modu) as u128
                * pow_mod128(3, i as u64, modu) as u128)
                % modu as u128;
            sum = (sum + term) % modu as u128;
        }
        image.insert(sum as u64);
        return;
    }
    let lo = remaining as i64 - 1; // must leave room for (remaining-1) more values below, down to 0
    let mut a = upper_bound;
    while a >= lo {
        chosen.push(a as u32);
        recurse(remaining - 1, a - 1, chosen, modu, image);
        chosen.pop();
        a -= 1;
    }
}

/// Enumerate all strictly decreasing sequences of `j` exponents chosen from
/// {0, ..., 2j-1}, compute the image of R_{j-1,j} mod 3^ell directly.
fn image_size_direct(ell: u32, j: u32) -> u64 {
    let modu = pow_u64(3, ell);
    let max_exp = 2 * j - 1;
    let mut chosen: Vec<u32> = Vec::with_capacity(j as usize);
    let mut image: HashSet<u64> = HashSet::new();
    recurse(j, max_exp as i64, &mut chosen, modu, &mut image);
    image.len() as u64
}

fn find_j_star_direct(ell: u32, j_max: u32) -> Option<(u32, u64)> {
    let target = 2u64 * pow_u64(3, ell - 1);
    for j in 1..=j_max {
        let sz = image_size_direct(ell, j);
        if sz == target {
            return Some((j, sz));
        }
    }
    None
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let ell_max: u32 = if args.len() > 1 {
        args[1].parse().unwrap()
    } else {
        4
    };
    let reference: [(u32, u32); 4] = [(1, 1), (2, 4), (3, 6), (4, 7)];
    println!("=== Independent brute-force cross-check (direct enumeration, no reduction/DP) ===");
    let mut all_ok = true;
    for ell in 1..=ell_max {
        let t0 = std::time::Instant::now();
        let result = find_j_star_direct(ell, 20);
        let expected = reference.iter().find(|(e, _)| *e == ell).map(|(_, j)| *j);
        let ok = result.map(|(j, _)| j) == expected;
        all_ok &= ok;
        println!(
            "  l={}: j*={:?} (expected {:?}) {}  time={:.3}s",
            ell,
            result.map(|(j, _)| j),
            expected,
            if ok { "OK" } else { "MISMATCH!" },
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "\n{}",
        if all_ok {
            "BRUTE-FORCE CROSS-CHECK PASSED"
        } else {
            "BRUTE-FORCE CROSS-CHECK FAILED - investigate before trusting the fast implementation"
        }
    );
}
