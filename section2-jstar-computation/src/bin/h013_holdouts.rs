// Exact H-013 holdout extractor.
//
// At ell <= j, the project uses
//   R_{j-1,j} mod 3^ell = 2^(j-ell) R_{ell-1,j} mod 3^ell.
// This program computes the reduced image R_{ell-1,j} with the same
// descending-exponent 0/1 DP as src/main.rs, extracts its missing units,
// and multiplies them by 2^(j-ell) to report holdouts in the original
// R_{j-1,j} convention used in notes/H-003.md.
//
// Unlike src/main.rs, the circular shift is ORed into its target in one
// word pass. This avoids full-size shift temporaries and makes ell=20 fit
// comfortably in physical memory.

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
            result = (result * b) % m;
        }
        b = (b * b) % m;
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
        ((nbits + 63) / 64) as usize
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

    fn with_bit0(nbits: u64) -> Self {
        let mut out = Self::zero(nbits);
        out.words[0] = 1;
        out
    }

    fn is_zero(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    // Read len <= 64 consecutive bits without crossing nbits. The result's
    // low bit is the bit at start.
    fn read_linear(&self, start: u64, len: u32) -> u64 {
        debug_assert!(len <= 64);
        debug_assert!(start + len as u64 <= self.nbits);
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

    // Read 64 consecutive bits on the cyclic nbits-bit circle.
    fn read_circular64(&self, start: u64) -> u64 {
        debug_assert!(start < self.nbits);
        let first = (self.nbits - start).min(64) as u32;
        let low = self.read_linear(start, first);
        if first == 64 {
            low
        } else {
            low | (self.read_linear(0, 64 - first) << first)
        }
    }

    // OR rotate_left(self, offset) into target. For destination bit d, the
    // source is (d-offset) mod nbits.
    fn rotate_left_or_into(&self, offset: u64, target: &mut Self) {
        debug_assert_eq!(self.nbits, target.nbits);
        let nbits = self.nbits;
        let top_mask = Self::top_mask(nbits);
        let src = self;
        let apply = |(word_index, dst): (usize, &mut u64)| {
            let dest_start = word_index as u64 * 64;
            let source_start = (dest_start + nbits - offset) % nbits;
            let mut shifted = src.read_circular64(source_start);
            if word_index + 1 == src.words.len() {
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

    fn contains(&self, bit: u64) -> bool {
        (self.words[(bit / 64) as usize] >> (bit % 64)) & 1 != 0
    }

    fn popcount(&self) -> u64 {
        self.words.iter().map(|word| word.count_ones() as u64).sum()
    }
}

fn reduced_image(ell: u32, j: u32) -> Bitset {
    assert!(j >= ell, "the reduction requires j >= ell");
    let modu = 3u64.checked_pow(ell).expect("3^ell overflows u64");
    let max_exp = ell + j - 1;
    let mut state: Vec<Bitset> = (0..=ell).map(|_| Bitset::zero(modu)).collect();
    state[0] = Bitset::with_bit0(modu);

    for v in (0..=max_exp).rev() {
        let p2 = pow_mod(2, v as u64, modu);
        for c in (0..ell as usize).rev() {
            if state[c].is_zero() {
                continue;
            }
            let p3 = pow_mod(3, c as u64, modu);
            let offset = ((p2 as u128 * p3 as u128) % modu as u128) as u64;
            let (left, right) = state.split_at_mut(c + 1);
            left[c].rotate_left_or_into(offset, &mut right[0]);
        }
    }
    state.pop().unwrap()
}

fn unit_masks() -> [u64; 3] {
    let mut masks = [0u64; 3];
    for start_mod3 in 0..3 {
        for bit in 0..64 {
            if (start_mod3 + bit) % 3 != 0 {
                masks[start_mod3 as usize] |= 1u64 << bit;
            }
        }
    }
    masks
}

fn reduced_holdouts(image: &Bitset) -> Vec<u64> {
    let masks = unit_masks();
    let last = image.words.len() - 1;
    let mut holes = Vec::new();
    for (word_index, &word) in image.words.iter().enumerate() {
        let start = word_index as u64 * 64;
        let mut missing = !word & masks[(start % 3) as usize];
        if word_index == last {
            missing &= Bitset::top_mask(image.nbits);
        }
        while missing != 0 {
            let bit = missing.trailing_zeros() as u64;
            holes.push(start + bit);
            missing &= missing - 1;
        }
    }
    holes
}

fn to_base3(mut x: u64, width: u32) -> String {
    let mut digits = vec![b'0'; width as usize];
    for digit in digits.iter_mut().rev() {
        *digit += (x % 3) as u8;
        x /= 3;
    }
    assert_eq!(x, 0);
    String::from_utf8(digits).unwrap()
}

fn class_histogram(values: &[u64], modu: u64) -> Vec<(u64, u64)> {
    let mut counts = vec![0u64; modu as usize];
    for &value in values {
        counts[(value % modu) as usize] += 1;
    }
    let mut histogram: Vec<(u64, u64)> = counts
        .into_iter()
        .enumerate()
        .filter(|&(_, count)| count != 0)
        .map(|(class, count)| (class as u64, count))
        .collect();
    histogram.sort_unstable_by_key(|&(class, count)| (std::cmp::Reverse(count), class));
    histogram
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: h013_holdouts <ell> <j>");
        std::process::exit(2);
    }
    let ell: u32 = args[1].parse().unwrap();
    let j: u32 = args[2].parse().unwrap();
    let modu = 3u64.checked_pow(ell).expect("3^ell overflows u64");
    let target = 2 * 3u64.pow(ell - 1);
    let scale = pow_mod(2, (j - ell) as u64, modu);

    let started = Instant::now();
    let image = reduced_image(ell, j);
    let seconds = started.elapsed().as_secs_f64();
    let reduced_holes = reduced_holdouts(&image);
    let mut holdouts: Vec<u64> = reduced_holes
        .iter()
        .map(|&value| ((value as u128 * scale as u128) % modu as u128) as u64)
        .collect();
    holdouts.sort_unstable();

    assert!(holdouts.iter().all(|&value| value % 3 != 0));
    assert_eq!(image.popcount() + holdouts.len() as u64, target);
    for (&reduced, actual) in reduced_holes.iter().zip(
        reduced_holes
            .iter()
            .map(|&value| ((value as u128 * scale as u128) % modu as u128) as u64),
    ) {
        assert!(!image.contains(reduced));
        assert_eq!(actual % 3, reduced * scale % 3);
    }

    println!("ell={} j={} modulus={} scale={} image_size={} unit_target={} holdout_count={} time_seconds={:.6}",
             ell, j, modu, scale, image.popcount(), target, holdouts.len(), seconds);
    println!("holdouts={:?}", holdouts);
    println!(
        "class_histogram_mod_3^5={:?}",
        class_histogram(&holdouts, 243)
    );
    println!(
        "class_histogram_mod_3^6={:?}",
        class_histogram(&holdouts, 729)
    );
    println!("holdouts_base3:");
    for &value in &holdouts {
        println!("{} {}", value, to_base3(value, ell));
    }
}
