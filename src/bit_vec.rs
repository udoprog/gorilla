//! An efficient BitVec implementation for loading segments of data.

type BitWord = u64;
const BITS_SIZE: usize = 8 * 8;

pub struct Iter<'a> {
    bit_vec: &'a BitVec,
    offset: usize,
}

#[inline]
fn index_offset(offset: usize) -> (usize, usize) {
    (offset / BITS_SIZE, offset % BITS_SIZE)
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.offset >= self.bit_vec.size {
            return None;
        }

        let (index, offset) = index_offset(self.offset);

        self.offset += 1;

        Some((self.bit_vec.bits[index] >> offset) & 1 == 1)
    }
}

impl<'a> Iter<'a> {
    pub fn step(&mut self, size: usize) {
        self.offset += size;
    }

    /// Load the given size from the current offset.
    pub fn load(&mut self, size: usize) -> Option<u64> {
        if size == 0 || size > BITS_SIZE {
            return Some(0);
        }

        let end = self.offset + size;

        if end > self.bit_vec.size {
            return None;
        }

        let start = index_offset(self.offset);
        let end = index_offset(self.offset + size);

        self.offset += size;

        let mut total: u64 = 0;

        let bits = BITS_SIZE;
        let range = end.0 - start.0;

        // first segment
        {
            let diff = if range == 0 {
                end.1 - start.1
            } else {
                bits - start.1
            };

            let segment = self.bit_vec.bits[start.0];

            if diff == bits {
                total = segment;
            } else {
                let mask = (1u64 << diff) - 1;
                let shift = start.1;
                total += ((segment >> shift) as u64 & mask) as u64;
            }
        };

        // last segment
        if range > 0 && end.1 > 0 {
            let diff = end.1;
            let shift = (range * bits) - start.1;
            let mask: u64 = (1u64 << diff) - 1;
            let segment: u64 = self.bit_vec.bits[end.0 - 1];

            total += (((segment & mask) as u64) << shift) as u64;
        };

        Some(total)
    }
}

pub struct BitVecBuilder {
    bits: Vec<BitWord>,
    size: usize,
    word: BitWord,
}

impl BitVecBuilder {
    pub fn new() -> BitVecBuilder {
        BitVecBuilder {
            bits: Vec::new(),
            size: 0usize,
            word: 0,
        }
    }

    pub fn push(&mut self, value: bool) {
        let offset = self.size % BITS_SIZE;

        if self.size > 0 && offset == 0 {
            self.bits.push(self.word);
            self.word = 0;
        }

        self.size += 1;

        self.word += if value { 1 } else { 0 } << offset;
    }

    pub fn build(mut self) -> BitVec {
        if self.size > 0 {
            self.bits.push(self.word);
        }

        BitVec {
            bits: self.bits,
            size: self.size,
        }
    }
}

pub struct BitVec {
    bits: Vec<BitWord>,
    size: usize,
}

impl BitVec {
    pub fn empty() -> BitVec {
        BitVec {
            bits: Vec::new(),
            size: 0usize,
        }
    }

    pub fn builder() -> BitVecBuilder {
        BitVecBuilder::new()
    }

    pub fn iter(&self) -> Iter {
        Iter {
            bit_vec: self,
            offset: 0usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_vec() {
        let mut builder = BitVec::builder();

        for i in 0..256 {
            builder.push(if i % 2 == 0 { true } else { false });
        }

        let vec = builder.build();

        {
            let mut it = vec.iter();

            for i in 0..256 {
                let expected = if i % 2 == 0 { true } else { false };
                assert_eq!(Some(expected), it.next());
            }
        }

        for i in 0..128 {
            let mut it = vec.iter();
            it.step(i);
            let expected: u64 = if i % 2 == 0 {
                0x5555555555555555
            } else {
                0xaaaaaaaaaaaaaaaa
            };
            let loaded = it.load(64);
            assert_eq!(Some(expected), loaded);
        }
    }
}
