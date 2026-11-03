//! A fast non-cryptographic hasher for the interpreter's hot maps (scope variables, property
//! tables, pointer-keyed registries). Same multiply-rotate scheme rustc uses internally (FxHash):
//! JS workloads hash short ASCII keys millions of times per second, where SipHash's
//! DoS-resistance costs far more than it buys.

use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            self.add(u64::from_le_bytes(bytes[..8].try_into().unwrap()));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            self.add(u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64);
            bytes = &bytes[4..];
        }
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    /// The multiply leaves its entropy in the high bits, and a key's known-zero low bits (a cell
    /// address's alignment: 7 bits for 128-byte cells) stay zero in the product's low bits, which
    /// hashbrown takes for the bucket. Rotating the high bits down spreads such keys over the
    /// table, as rustc-hash 2's `finish` does; SpiderMonkey's tables index by the product's high
    /// bits instead (mfbt/HashTable.h, `hash >> hashShift`), and JSC and V8 hash pointers through
    /// an avalanche mix (`intHash`, `ComputeLongHash`).
    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}

pub type FastMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FastSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

#[cfg(test)]
mod tests {
    use super::FxHasher;
    use std::hash::Hasher;

    fn hash_usize(key: usize) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_usize(key);
        hasher.finish()
    }

    /// Cell addresses share their alignment's zero low bits (7 for 128-byte cells), which a
    /// multiplicative product keeps; the low bits a hashbrown table buckets by must still
    /// spread them. Without the rotation 1024 cells 128 bytes apart reach 8 of 1024 buckets (16
    /// at 192 bytes); with it, 862 (a random hash reaches about 647).
    #[test]
    fn aligned_addresses_spread_over_the_low_bits() {
        for align in [8_usize, 64, 128, 192, 4096] {
            let buckets: std::collections::HashSet<u64> = (0..1024)
                .map(|i| hash_usize(0x7f12_3450_0000 + i * align) & 1023)
                .collect();
            assert!(
                buckets.len() > 400,
                "{align}-byte strides reach only {} of 1024 buckets",
                buckets.len()
            );
        }
    }
}
