//! Exact maxrun(H(ell,ell+1)) computation for H-014.
//!
//! We use
//!   R_{j-1,j} = 2^(j-ell) R_{ell-1,j} (mod 3^ell), j >= ell.
//! Multiplication by the common unit 2^(j-ell) preserves doubling-chain
//! lengths, so maxrun can be measured directly in the reduced complement.
//!
//! The descending-exponent DP is identical to E-001's independently checked
//! extractor.  The only representation optimization is exact: layers up to a
//! configurable cutoff retain one residue per exponent subset (duplicates are
//! harmless and preserve the 0/1-knapsack recurrence); higher layers are
//! packed bitsets.  This saves several full 3^ell-bit layers at large ell.

use rayon::prelude::*;
use std::env;
use std::time::Instant;

const PAR_THRESHOLD: usize = 50_000;

fn pow_mod(base: u64, mut exp: u64, modu: u64) -> u64 {
    let mut result: u128 = 1;
    let mut b = (base % modu) as u128;
    let m = modu as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % m;
        }
        b = b * b % m;
        exp >>= 1;
    }
    result as u64
}

struct Bitset {
    words: Vec<u64>,
    nbits: u64,
}

impl Bitset {
    fn nwords(nbits: u64) -> usize {
        nbits.div_ceil(64) as usize
    }

    fn top_mask(nbits: u64) -> u64 {
        let used = nbits - (Self::nwords(nbits) as u64 - 1) * 64;
        if used == 64 {
            u64::MAX
        } else {
            (1u64 << used) - 1
        }
    }

    fn zero(nbits: u64) -> Self {
        Self {
            words: vec![0; Self::nwords(nbits)],
            nbits,
        }
    }

    fn read_linear(&self, start: u64, len: u32) -> u64 {
        debug_assert!(len <= 64 && start + len as u64 <= self.nbits);
        if len == 0 {
            return 0;
        }
        let word = (start / 64) as usize;
        let bit = (start % 64) as u32;
        let mut value = self.words[word] >> bit;
        if bit != 0 && bit + len > 64 {
            value |= self.words[word + 1] << (64 - bit);
        }
        if len == 64 {
            value
        } else {
            value & ((1u64 << len) - 1)
        }
    }

    fn read_circular64(&self, start: u64) -> u64 {
        let first = (self.nbits - start).min(64) as u32;
        let low = self.read_linear(start, first);
        if first == 64 {
            low
        } else {
            low | self.read_linear(0, 64 - first) << first
        }
    }

    fn rotate_left_or_into(&self, offset: u64, target: &mut Self) {
        debug_assert_eq!(self.nbits, target.nbits);
        let nbits = self.nbits;
        let top_mask = Self::top_mask(nbits);
        let apply = |(word_index, dst): (usize, &mut u64)| {
            let dest_start = word_index as u64 * 64;
            let source_start = (dest_start + nbits - offset) % nbits;
            let mut shifted = self.read_circular64(source_start);
            if word_index + 1 == self.words.len() {
                shifted &= top_mask;
            }
            *dst |= shifted;
        };
        if target.words.len() >= PAR_THRESHOLD {
            target.words.par_iter_mut().enumerate().for_each(apply);
        } else {
            target.words.iter_mut().enumerate().for_each(apply);
        }
    }

    fn insert(&mut self, bit: u64) {
        self.words[(bit / 64) as usize] |= 1u64 << (bit % 64);
    }

    fn contains(&self, bit: u64) -> bool {
        self.words[(bit / 64) as usize] >> (bit % 64) & 1 != 0
    }

    fn popcount(&self) -> u64 {
        self.words
            .par_iter()
            .map(|word| word.count_ones() as u64)
            .sum()
    }
}

enum Layer {
    /// One entry per exponent subset. Repeated residues are intentionally
    /// retained; this is cheaper than hashing while subset counts are small.
    Sparse(Vec<u64>),
    Dense(Bitset),
}

impl Layer {
    fn empty_sparse() -> Self {
        Self::Sparse(Vec::new())
    }

    fn add_shifted_from(&mut self, source: &Layer, offset: u64, modu: u64) {
        match (self, source) {
            (Layer::Sparse(dst), Layer::Sparse(src)) => {
                dst.reserve(src.len());
                dst.extend(src.iter().map(|&x| {
                    let y = x + offset;
                    if y >= modu {
                        y - modu
                    } else {
                        y
                    }
                }));
            }
            (Layer::Dense(dst), Layer::Sparse(src)) => {
                for &x in src {
                    let y = x + offset;
                    dst.insert(if y >= modu { y - modu } else { y });
                }
            }
            (Layer::Dense(dst), Layer::Dense(src)) => src.rotate_left_or_into(offset, dst),
            (Layer::Sparse(_), Layer::Dense(_)) => {
                unreachable!("a dense source cannot feed a sparse target")
            }
        }
    }
}

fn reduced_image(ell: u32, sparse_cutoff: u32, progress: bool) -> (Bitset, f64) {
    let j = ell + 1;
    let modu = 3u64.checked_pow(ell).expect("3^ell overflows u64");
    let max_exp = ell + j - 1; // = 2 ell
    let cutoff = sparse_cutoff.min(ell.saturating_sub(1));
    let mut state: Vec<Layer> = (0..=ell)
        .map(|c| {
            if c <= cutoff {
                Layer::empty_sparse()
            } else {
                Layer::Dense(Bitset::zero(modu))
            }
        })
        .collect();
    match &mut state[0] {
        Layer::Sparse(v) => v.push(0),
        Layer::Dense(_) => unreachable!(),
    }

    let started = Instant::now();
    for v in (0..=max_exp).rev() {
        let processed_before = (max_exp - v) as usize;
        let c_hi = (ell as usize - 1).min(processed_before);
        // A c below this cannot reach ell even if every remaining exponent,
        // including v, is selected.
        let c_lo = (ell as i64 - (v as i64 + 1)).max(0) as usize;
        if c_lo <= c_hi {
            let p2 = pow_mod(2, v as u64, modu);
            for c in (c_lo..=c_hi).rev() {
                let p3 = pow_mod(3, c as u64, modu);
                let offset = (p2 as u128 * p3 as u128 % modu as u128) as u64;
                let (left, right) = state.split_at_mut(c + 1);
                right[0].add_shifted_from(&left[c], offset, modu);
            }
        }
        if progress {
            eprintln!(
                "ell={} exponent={} active_c={}..{} elapsed_seconds={:.3}",
                ell,
                v,
                c_lo,
                if c_lo <= c_hi { c_hi } else { c_lo },
                started.elapsed().as_secs_f64()
            );
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    match state.pop().unwrap() {
        Layer::Dense(image) => (image, seconds),
        Layer::Sparse(_) => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug)]
struct RunResult {
    holdout_count: u64,
    maxrun: u64,
    start: u64,
}

fn better_run(a: RunResult, b: RunResult) -> RunResult {
    let (maxrun, start) = if a.maxrun > b.maxrun || (a.maxrun == b.maxrun && a.start < b.start) {
        (a.maxrun, a.start)
    } else {
        (b.maxrun, b.start)
    };
    RunResult {
        holdout_count: a.holdout_count + b.holdout_count,
        maxrun,
        start,
    }
}

fn unit_masks() -> [u64; 3] {
    let mut masks = [0u64; 3];
    for (start_mod3, mask) in masks.iter_mut().enumerate() {
        for bit in 0..64 {
            if (start_mod3 + bit) % 3 != 0 {
                *mask |= 1u64 << bit;
            }
        }
    }
    masks
}

fn holdout_runs(image: &Bitset) -> RunResult {
    let modu = image.nbits;
    let inv2 = modu.div_ceil(2);
    assert_eq!(2 * inv2 % modu, 1);
    let masks = unit_masks();
    let last = image.words.len() - 1;
    image
        .words
        .par_iter()
        .enumerate()
        .map(|(word_index, &word)| {
            let base = word_index as u64 * 64;
            let mut missing = !word & masks[(base % 3) as usize];
            if word_index == last {
                missing &= Bitset::top_mask(modu);
            }
            let mut out = RunResult {
                holdout_count: missing.count_ones() as u64,
                maxrun: 0,
                start: u64::MAX,
            };
            while missing != 0 {
                let bit = missing.trailing_zeros() as u64;
                let x = base + bit;
                let prev = (x as u128 * inv2 as u128 % modu as u128) as u64;
                if image.contains(prev) {
                    let mut run = 1u64;
                    let mut y = x;
                    loop {
                        y = (2 * y) % modu;
                        if image.contains(y) {
                            break;
                        }
                        run += 1;
                    }
                    if run > out.maxrun || (run == out.maxrun && x < out.start) {
                        out.maxrun = run;
                        out.start = x;
                    }
                }
                missing &= missing - 1;
            }
            out
        })
        .reduce(
            || RunResult {
                holdout_count: 0,
                maxrun: 0,
                start: u64::MAX,
            },
            better_run,
        )
}

fn chain(start: u64, len: u64, modu: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(len as usize);
    let mut x = start;
    for _ in 0..len {
        out.push(x);
        x = 2 * x % modu;
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        eprintln!("usage: h014-maxrun <ell> [sparse_cutoff=7] [--progress]");
        std::process::exit(2);
    }
    let ell: u32 = args[1].parse().unwrap();
    let sparse_cutoff: u32 = args
        .get(2)
        .filter(|s| s.as_str() != "--progress")
        .map(|s| s.parse().unwrap())
        .unwrap_or(7);
    let progress = args.iter().any(|s| s == "--progress");
    assert!(ell >= 1);
    let modu = 3u64.checked_pow(ell).expect("3^ell overflows u64");
    let (image, dp_seconds) = reduced_image(ell, sparse_cutoff, progress);
    let scan_started = Instant::now();
    let result = holdout_runs(&image);
    let scan_seconds = scan_started.elapsed().as_secs_f64();
    let image_size = image.popcount();
    let unit_target = 2 * 3u64.pow(ell - 1);
    assert_eq!(image_size + result.holdout_count, unit_target);
    assert!(image_size > 0);
    assert_eq!(result.holdout_count == 0, result.maxrun == 0);
    let reduced_values = chain(result.start, result.maxrun, modu);
    assert!(reduced_values.iter().all(|&x| !image.contains(x)));
    let actual_start = if result.maxrun == 0 {
        u64::MAX
    } else {
        2 * result.start % modu
    };
    let actual_values = chain(actual_start, result.maxrun, modu);
    assert_eq!(
        actual_values,
        reduced_values
            .iter()
            .map(|&x| 2 * x % modu)
            .collect::<Vec<_>>()
    );
    if result.maxrun > 0 {
        let predecessor = (result.start as u128 * modu.div_ceil(2) as u128 % modu as u128) as u64;
        assert!(image.contains(predecessor));
        assert!(image.contains(2 * reduced_values[reduced_values.len() - 1] % modu));
    }
    println!(
        "ell={} j={} modulus={} sparse_cutoff={} image_size={} unit_target={} holdout_count={} maxrun={} reduced_chain_start={} reduced_chain={:?} actual_chain_start={} actual_chain={:?} dp_seconds={:.6} scan_seconds={:.6} total_seconds={:.6}",
        ell, ell + 1, modu, sparse_cutoff, image_size, unit_target,
        result.holdout_count, result.maxrun, result.start, reduced_values,
        actual_start, actual_values,
        dp_seconds, scan_seconds, dp_seconds + scan_seconds
    );
}
