// Fast reimplementation of the j*(l) covering computation for Wirsching's
// Weak Covering Conjecture (H-001). Reimplements experiment_wcc.py's bitset
// + cyclic rotation DP using a native Vec<u64> bitset instead of Python
// arbitrary-precision integers.
//
// R_{j,k} := { sum_{i=0}^{j} 2^{a_i} 3^i : j+k >= a_0 > a_1 > ... > a_j >= 0 }
// For j >= l: R_{j-1,j} mod 3^l = 2^{j-l} * R_{l-1,j} mod 3^l (a unit multiple,
// doesn't change coverage), reducing enumeration from C(2j,j) to C(j+l,l).
//
// DP: state[c] = bitset of residues mod 3^l reachable as a sum of c terms
// (each term is 2^v * 3^c for some chosen exponent v), processing candidate
// exponents v from high to low, c from high to low within each v (0/1
// knapsack pattern: prevents using the same v twice per step).

use rayon::prelude::*;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

// Below this many words, run the per-word loop serially: rayon's dispatch
// overhead isn't worth it for small bitsets (early/small l), but for large l
// (where memory and time actually matter) the loop is embarrassingly
// parallel: each output word depends only on at most two fixed input words,
// with no dependency between output words, so splitting ranges across
// threads needs no synchronization.
const PAR_THRESHOLD: usize = 50_000;

fn pow_mod(base: u64, mut exp: u64, modu: u64) -> u64 {
    if modu == 1 {
        return 0;
    }
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

#[derive(Clone)]
struct BigBitset {
    words: Vec<u64>,
    nbits: u64,
}

impl BigBitset {
    fn nwords(nbits: u64) -> usize {
        ((nbits + 63) / 64) as usize
    }

    fn top_mask(nbits: u64) -> u64 {
        let used = nbits - (Self::nwords(nbits) as u64 - 1) * 64;
        if used >= 64 {
            u64::MAX
        } else {
            (1u64 << used) - 1
        }
    }

    fn zero(nbits: u64) -> Self {
        BigBitset {
            words: vec![0u64; Self::nwords(nbits)],
            nbits,
        }
    }

    fn with_bit0(nbits: u64) -> Self {
        let mut b = Self::zero(nbits);
        b.words[0] = 1;
        b
    }

    fn mask_top(&mut self) {
        let n = self.words.len();
        self.words[n - 1] &= Self::top_mask(self.nbits);
    }

    fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    fn shl(&self, k: u64) -> Self {
        let nwords = self.words.len();
        if k >= self.nbits {
            return Self::zero(self.nbits);
        }
        let word_shift = (k / 64) as usize;
        let bit_shift = (k % 64) as u32;
        let mut out = vec![0u64; nwords];
        let src = &self.words;
        let compute = move |i: usize| -> u64 {
            if bit_shift == 0 {
                src[i - word_shift]
            } else {
                let hi = src[i - word_shift] << bit_shift;
                let lo = if i > word_shift {
                    src[i - word_shift - 1] >> (64 - bit_shift)
                } else {
                    0
                };
                hi | lo
            }
        };
        let span = &mut out[word_shift..nwords];
        if span.len() >= PAR_THRESHOLD {
            span.par_iter_mut()
                .enumerate()
                .for_each(|(off, o)| *o = compute(word_shift + off));
        } else {
            for (off, o) in span.iter_mut().enumerate() {
                *o = compute(word_shift + off);
            }
        }
        let mut b = BigBitset {
            words: out,
            nbits: self.nbits,
        };
        b.mask_top();
        b
    }

    fn shr(&self, k: u64) -> Self {
        let nwords = self.words.len();
        if k >= self.nbits {
            return Self::zero(self.nbits);
        }
        let word_shift = (k / 64) as usize;
        let bit_shift = (k % 64) as u32;
        let mut out = vec![0u64; nwords];
        let src = &self.words;
        let compute = move |i: usize| -> u64 {
            let lo = src[i + word_shift] >> bit_shift;
            let hi = if bit_shift > 0 && i + word_shift + 1 < nwords {
                src[i + word_shift + 1] << (64 - bit_shift)
            } else {
                0
            };
            lo | hi
        };
        let span = &mut out[0..(nwords - word_shift)];
        if span.len() >= PAR_THRESHOLD {
            span.par_iter_mut().enumerate().for_each(|(i, o)| *o = compute(i));
        } else {
            for (i, o) in span.iter_mut().enumerate() {
                *o = compute(i);
            }
        }
        BigBitset {
            words: out,
            nbits: self.nbits,
        }
    }

    /// Rotates left and ORs the result into `target`, one full-size
    /// temporary at a time (shl's result is OR'd in and dropped before shr's
    /// result is even computed), not two held simultaneously: this is what
    /// keeps peak memory at state-size-plus-one, not state-size-plus-two.
    fn rotate_left_and_or_into(&self, offset: u64, target: &mut Self) {
        let shifted = self.shl(offset);
        target.or_assign(&shifted);
        drop(shifted);
        if offset != 0 {
            let wrapped = self.shr(self.nbits - offset);
            target.or_assign(&wrapped);
        }
    }

    fn or_assign(&mut self, other: &Self) {
        let dst = &mut self.words;
        let src = &other.words;
        if dst.len() >= PAR_THRESHOLD {
            dst.par_iter_mut().enumerate().for_each(|(i, d)| *d |= src[i]);
        } else {
            for i in 0..dst.len() {
                dst[i] |= src[i];
            }
        }
    }

    fn popcount(&self) -> u64 {
        self.words.iter().map(|w| w.count_ones() as u64).sum()
    }
}

// --- Progress log (state checkpointing removed 2026-07-27; see
// notes/H-001.md "Plan for l=24 onward" for why: it dominated l=23's wall
// time once the state reached hundreds of GiB, and the systemd-oomd kill
// that originally motivated it is now fixed at the root cause, oomd
// disabled). What remains is the cheap part: one line per j already
// confirmed NOT covering (with its image size), appended as each attempt
// finishes, so a restart after several completed-but-failed j's doesn't
// redo them. This is a tiny text file, not a multi-hundred-GiB state dump,
// so it stays.
fn checkpoint_dir() -> &'static str {
    "checkpoints"
}

fn progress_path(ell: u32) -> String {
    format!("{}/progress_l{}.txt", checkpoint_dir(), ell)
}

/// The highest j for this ell already confirmed (in a possibly earlier,
/// interrupted process run) NOT to cover - so a restart's j-search doesn't
/// redo work whose answer is already known and logged.
fn confirmed_max_failed_j(ell: u32) -> Option<u32> {
    let content = std::fs::read_to_string(progress_path(ell)).ok()?;
    content
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter_map(|tok| tok.parse::<u32>().ok())
        .max()
}

fn append_progress(ell: u32, j: u32, size: u64) {
    if std::fs::create_dir_all(checkpoint_dir()).is_err() {
        return;
    }
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(progress_path(ell))
    {
        let _ = writeln!(f, "{} {}", j, size);
    }
}

/// Image size of R_{j-1,j} mod 3^ell, via the R_{ell-1,j} reduction (a unit
/// multiple, same population count).
fn image_size(ell: u32, j: u32) -> u64 {
    // The R_{j-1,j} = 2^{j-ell} * R_{ell-1,j} reduction requires j >= ell;
    // silently wrong (not just slow) if violated, since e(l) is o(l) means
    // j*(l) < l is possible for large enough l (unreachable at this
    // implementation's actual memory ceiling of l~21, but cheap to guard).
    debug_assert!(j >= ell, "reduction R_{{j-1,j}}=2^(j-ell)*R_{{ell-1,j}} requires j >= ell");
    let modu = 3u64.checked_pow(ell).expect("3^ell overflows u64");
    let nbits = modu;
    let max_exp = ell + j - 1; // exponents range 0..=max_exp, choose ell of them

    let mut state: Vec<BigBitset> = (0..=ell).map(|_| BigBitset::zero(nbits)).collect();
    state[0] = BigBitset::with_bit0(nbits);

    let mut v = max_exp;
    loop {
        let p2 = pow_mod(2, v as u64, modu);
        // Sequential, decreasing c: matches the original single-threaded
        // 0/1-knapsack structure exactly (reads state[c] before it could
        // ever be written this v-iteration, since only targets > c have
        // been touched so far). Parallelism comes from inside each
        // rotate/OR call (word-level, no cross-c dependency), which keeps
        // at most one extra full-size bitset alive at a time instead of up
        // to `ell` of them - this is what brings peak memory down from
        // ~2*(ell+1)*3^ell/8 bytes to ~(ell+2)*3^ell/8.
        for c in (0..ell as usize).rev() {
            if state[c].is_zero() {
                continue;
            }
            let p3 = pow_mod(3, c as u64, modu);
            let offset = ((p2 as u128 * p3 as u128) % modu as u128) as u64;
            let (left, right) = state.split_at_mut(c + 1);
            left[c].rotate_left_and_or_into(offset, &mut right[0]);
        }
        if v == 0 {
            break;
        }
        v -= 1;
    }
    state[ell as usize].popcount()
}

fn find_j_star(ell: u32, j_start: u32, j_max: u32) -> Option<(u32, u64)> {
    let target = 2u64 * 3u64.checked_pow(ell - 1).expect("2*3^(ell-1) overflows u64");
    // image_size's reduction is only valid for j >= ell (see its own debug_assert); never
    // evaluate it outside that domain. Note this clamp assumes j*(l) >= l, true throughout the
    // range this project actually reaches (l<=21, memory-limited); since e(l) is o(l) and the
    // entropy threshold log_4(3) < 1, j*(l) < l becomes possible only around l~38+, unreached
    // here but worth revisiting if the memory ceiling is ever overcome.
    let mut j = j_start.max(ell).max(1);
    if let Some(confirmed) = confirmed_max_failed_j(ell) {
        if confirmed + 1 > j {
            eprintln!(
                "[checkpoint] l={}: progress log shows j<={} already confirmed not covering, starting search at j={}",
                ell, confirmed, confirmed + 1
            );
            j = confirmed + 1;
        }
    }
    while j <= j_max {
        let sz = image_size(ell, j);
        if sz == target {
            return Some((j, sz));
        }
        append_progress(ell, j, sz);
        j += 1;
    }
    None
}

/// Validation reference table j*(l) for l=1..7, from experiment_wcc.py
/// (E-098, previous project), itself verified against the Wirsching 1998
/// primary source (literature/notes/L-001.md).
fn validate_reference_table() -> bool {
    let reference: [(u32, u32); 7] = [(1, 1), (2, 4), (3, 6), (4, 7), (5, 9), (6, 10), (7, 11)];
    println!("=== Validation against reference table (l=1..7) ===");
    let mut all_ok = true;
    for (ell, expected) in reference {
        let t0 = Instant::now();
        let result = find_j_star(ell, 1, 200);
        let ok = result.map(|(j, _)| j) == Some(expected);
        all_ok &= ok;
        println!(
            "  l={}: j*={:?} (expected {}) {}  time={:.3}s",
            ell,
            result.map(|(j, _)| j),
            expected,
            if ok { "OK" } else { "MISMATCH!" },
            t0.elapsed().as_secs_f64()
        );
    }
    // pointwise check: l=2, j=2 covers {1,2,5,7} mod 9, missing {4,8}
    let sz22 = image_size(2, 2);
    let point_ok = sz22 == 4;
    println!(
        "\n  pointwise check l=2,j=2: image size={} (expected 4, missing {{4,8}} of 6 total) {}",
        sz22,
        if point_ok { "OK" } else { "MISMATCH!" }
    );
    all_ok && point_ok
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "validate" {
        let ok = validate_reference_table();
        if !ok {
            eprintln!("\nVALIDATION FAILED - do not proceed");
            std::process::exit(1);
        }
        println!("\nALL VALIDATIONS PASSED");
        return;
    }

    if args.len() > 1 && args[1] == "run" {
        // args: run <ell_start> <ell_end>
        let ell_start: u32 = args[2].parse().unwrap();
        let ell_end: u32 = args[3].parse().unwrap();
        let log4_3 = 3f64.ln() / 4f64.ln();
        let mut prev_j: u32 = 1;
        let t_total = Instant::now();
        for ell in ell_start..=ell_end {
            let t0 = Instant::now();
            let j_start = if prev_j > 2 { prev_j - 2 } else { 1 };
            let result = find_j_star(ell, j_start, 200);
            let dt = t0.elapsed().as_secs_f64();
            match result {
                Some((j_star, _sz)) => {
                    prev_j = j_star;
                    let excess = j_star as f64 - ell as f64 * log4_3;
                    println!(
                        "l={:2}  j*={:3}  j*/l={:.4}  excess={:7.3}  time={:10.3}s  total={:10.1}s",
                        ell,
                        j_star,
                        j_star as f64 / ell as f64,
                        excess,
                        dt,
                        t_total.elapsed().as_secs_f64()
                    );
                }
                None => {
                    println!("l={}: j* not found within limit - stopping", ell);
                    break;
                }
            }
        }
        return;
    }

    if args.len() > 1 && args[1] == "size" {
        // args: size <ell> <j> - single image_size call, for testing
        // checkpoint save/resume in isolation from find_j_star's j-search.
        let ell: u32 = args[2].parse().unwrap();
        let j: u32 = args[3].parse().unwrap();
        let t0 = Instant::now();
        let sz = image_size(ell, j);
        println!("l={} j={}: image_size={} time={:.3}s", ell, j, sz, t0.elapsed().as_secs_f64());
        return;
    }

    eprintln!("usage: E-001-jstar-fast validate | run <ell_start> <ell_end> | size <ell> <j>");
}
