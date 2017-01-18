extern crate bit_vec;
extern crate byteorder;

use std::mem;
use bit_vec::BitVec;

use byteorder::{ByteOrder, NativeEndian};

pub struct Block {
    bits: BitVec,
}

pub struct BlockBuilder {
    /// storage
    bits: BitVec,
    /// n - 1 timestamp/value encoded
    p0: Option<(u64, f64)>,
    /// n - 2 timestamp/value encoded
    p1: Option<(u64, f64)>,
    /// last observed leading/trailing lengths
    ///
    /// first entry encodes the number of leading zeros in the encoded value.
    /// second entry encodes the length of the meaningful bits.
    leading_trailing: Option<(u64, u64)>,
}

pub struct BlockIterator<'a> {
    /// iterator over storage
    bits: bit_vec::Iter<'a>,
    /// n - 1 timestamp/value decoded
    p0: Option<(u64, f64)>,
    /// n - 2 timestamp/value decoded
    p1: Option<(u64, f64)>,
    /// last observed leading/trailing lengths
    ///
    /// first entry encodes the number of leading zeros in the encoded value.
    /// second entry encodes the length of the meaningful bits.
    leading_trailing: Option<(u64, u64)>,
}

/// Macro to read and supply the given Option expression
macro_rules! read_or_set {
    ($x:expr, $reader:expr) => {
        match $x {
            Some(p) => p,
            None => {
                let p = tryopt!($reader());
                $x = Some(p);
                return Some(p);
            },
        }
    }
}

macro_rules! write_or_set {
    ($x:expr, $writer:expr, $t:expr, $v:expr) => {
        match $x {
            Some(p) => p,
            None => {
                $x = Some(($t, $v));
                return $writer($t, $v);
            },
        }
    }
}

macro_rules! tryopt {
    ($s:expr) => {
        match $s { Some(s) => s, None => return None, }
    }
}

const W1: u64 = 7;
const W1_S: i64 = 1 - (1 << (W1 - 1));
const W1_E: i64 = 1 << (W1 - 1);

const W2: u64 = 9;
const W2_S: i64 = 1 - (1 << (W2 - 1));
const W2_E: i64 = 1 << (W2 - 1);

const W3: u64 = 12;
const W3_S: i64 = 1 - (1 << (W3 - 1));
const W3_E: i64 = 1 << (W3 - 1);

const W4: u64 = 32;
const W4_S: i64 = 1 - (1 << (W4 - 1));
const W4_E: i64 = 1 << (W4 - 1);

// extend to support larger deltas...
const W5: u64 = 48;
const W5_S: i64 = 1 - (1 << (W5 - 1));
const W5_E: i64 = 1 << (W5 - 1);

impl<'a> Iterator for BlockIterator<'a> {
    type Item = (u64, f64);

    fn next(&mut self) -> Option<(u64, f64)> {
        // TODO: partial compression for the second (p1) point?

        let (t0, v0) = read_or_set!(self.p0, || self.read_full());
        let (t1, v1) = read_or_set!(self.p1, || self.read_partial(v0));

        let time = tryopt!(self.read_time(t0, t1));
        let v = tryopt!(self.read_value(v1));

        // shift the two last seen values.
        self.p0 = self.p1;
        self.p1 = Some((time, v));

        return Some((time, v));
    }
}

impl<'a> BlockIterator<'a> {
    #[inline]
    fn read_time(&mut self, t0: u64, t1: u64) -> Option<u64> {
        let time_bits = match tryopt!(self.read_leading(5, true)) {
            0 => 0,
            1 => W1,
            2 => W2,
            3 => W3,
            4 => W4,
            5 => W5,
            _ => return None,
        };

        let d = tryopt!(self.read_i64(time_bits));
        let r = d + ((t1 + t1) as i64 - t0 as i64);

        Some(r as u64)
    }

    #[inline]
    fn read_value(&mut self, v1: f64) -> Option<f64> {
        // value bit is zero, current value is same as last.
        if !tryopt!(self.bits.next()) {
            return Some(v1);
        }

        let (width, trailing) = tryopt!(self.read_previous_bits());

        let l_xor = tryopt!(self.read_u64(width));
        let v_xor = if trailing == 64 {
            0x0
        } else {
            l_xor << trailing
        };

        let uv1: u64 = unsafe { mem::transmute(v1) };
        let value = uv1 ^ v_xor;

        Some(unsafe { mem::transmute(value) })
    }

    #[inline]
    fn read_previous_bits(&mut self) -> Option<(u64, u64)> {
        // control bit is false, inherit previous value
        if !tryopt!(self.bits.next()) {
            let (leading, trailing) = tryopt!(self.leading_trailing);
            let width = (64 - (leading + trailing)) as u64;
            return Some((width, trailing));
        }

        // control bit is true, read new bits.
        let leading = tryopt!(self.read_u64(5));
        let width = match tryopt!(self.read_u64(6)) {
            0 => 64,
            v => v,
        };

        let shift = leading + width;
        let trailing = (64 - shift) as u64;

        self.leading_trailing = Some((leading, trailing));
        Some((width, trailing))
    }

    /// Read the number of leading bits that matches the given value.
    #[inline]
    fn read_leading(&mut self, max_bits: u16, m: bool) -> Option<u16> {
        let mut dod_bits = 0;

        while dod_bits < max_bits && tryopt!(self.bits.next()) == m {
            dod_bits += 1;
        }

        Some(dod_bits)
    }

    #[inline]
    fn read_i64(&mut self, bits: u64) -> Option<i64> {
        let value = tryopt!(self.read_u64(bits));

        let value = if bits > 0 && value >> bits - 1 == 1 {
            // value as i64 - (1 << bits)
            let x = std::u64::MAX >> (64 - bits);
            !(((value - 1) ^ x) as i64) + 1
        } else {
            value as i64
        };

        Some(value)
    }

    #[inline]
    fn read_u64(&mut self, bits: u64) -> Option<u64> {
        if bits <= 0 {
            return Some(0u64);
        }

        let mut value: u64 = 0u64;

        for o in (0..bits).rev() {
            if tryopt!(self.bits.next()) {
                value += 1 << o;
            }
        }

        Some(value)
    }

    #[inline]
    fn read_full(&mut self) -> Option<(u64, f64)> {
        let mut buffer = [0u8; 16];

        tryopt!(self.read_buffer(&mut buffer[..]));

        let t = NativeEndian::read_u64(&buffer[0..8]);
        let v = NativeEndian::read_f64(&buffer[8..16]);

        Some((t, v))
    }

    #[inline]
    fn read_partial(&mut self, v0: f64) -> Option<(u64, f64)> {
        let mut buffer = [0u8; 8];

        tryopt!(self.read_buffer(&mut buffer[..]));

        let t = NativeEndian::read_u64(&buffer[0..8]);
        let v = tryopt!(self.read_value(v0));

        Some((t, v))
    }

    #[inline]
    fn read_buffer(&mut self, buffer: &mut [u8]) -> Option<()> {
        for a in buffer {
            for i in (0u8..8u8).rev() {
                if tryopt!(self.bits.next()) {
                    *a += 1 << i;
                }
            }
        }

        Some(())
    }
}

impl Block {
    pub fn iter<'a>(&'a self) -> BlockIterator<'a> {
        BlockIterator {
            bits: self.bits.iter(),
            p0: None,
            p1: None,
            leading_trailing: None,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.bits.to_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Block {
        Block { bits: BitVec::from_bytes(bytes) }
    }
}

impl BlockBuilder {
    pub fn new() -> BlockBuilder {
        BlockBuilder {
            bits: BitVec::new(),
            p0: None,
            p1: None,
            leading_trailing: None,
        }
    }

    pub fn finalize(self) -> Block {
        Block { bits: self.bits }
    }

    pub fn iter(&self) -> BlockIterator {
        BlockIterator {
            bits: self.bits.iter(),
            p0: None,
            p1: None,
            leading_trailing: None,
        }
    }

    pub fn push(&mut self, t: u64, v: f64) {
        let (t0, v0) = write_or_set!(self.p0, |t, v| self.write_full(t, v), t, v);
        let (t1, v1) = write_or_set!(self.p1, |t, v| self.write_partial(v0, t, v), t, v);

        self.push_time(t0, t1, t);
        self.push_value(v1, v);

        self.p0 = Some((t1, v1));
        self.p1 = Some((t, v));
    }

    #[inline]
    fn push_time(&mut self, t0: u64, t1: u64, t: u64) {
        let dod = (t as i64 - t1 as i64) - (t1 as i64 - t0 as i64);

        match dod {
            0 => {
                self.write_bits(&[false]);
            }
            W1_S...W1_E => {
                self.write_bits(&[true, false]);
                self.write_signed(dod, W1);
            }
            W2_S...W2_E => {
                self.write_bits(&[true, true, false]);
                self.write_signed(dod, W2);
            }
            W3_S...W3_E => {
                self.write_bits(&[true, true, true, false]);
                self.write_signed(dod, W3);
            }
            W4_S...W4_E => {
                self.write_bits(&[true, true, true, true, false]);
                self.write_signed(dod, W4);
            }
            W5_S...W5_E => {
                self.write_bits(&[true, true, true, true, true]);
                self.write_signed(dod, W5);
            }
            _ => {
                panic!("Unsupported Delta: {}", dod);
            }
        };
    }

    #[inline]
    fn push_value(&mut self, v1: f64, v: f64) {
        let v_xor = {
            let uv1: u64 = unsafe { mem::transmute(v1) };
            let uv: u64 = unsafe { mem::transmute(v) };
            uv1 ^ uv
        };

        if v_xor == 0 {
            self.bits.push(false);
            return;
        }

        self.bits.push(true);

        let leading = v_xor.leading_zeros() as u64;
        let trailing = v_xor.trailing_zeros() as u64;

        if let Some((last_leading, last_trailing)) = self.leading_trailing {
            // if the number of trailing/leading zeros are compatible with what came before.
            if leading >= last_leading && trailing >= last_trailing {
                let width = (64 - (last_leading + last_trailing)) as u64;

                self.bits.push(false);
                self.write_unsigned(v_xor >> last_trailing, width);
                return;
            }
        }

        let width = (64 - (leading + trailing)) as u64;

        self.bits.push(true);
        self.write_unsigned(leading, 5);
        self.write_unsigned(width, 6);
        self.write_unsigned(v_xor >> trailing, width);

        self.leading_trailing = Some((leading, trailing));
    }

    #[inline]
    fn write_bits(&mut self, bits: &[bool]) {
        for b in bits {
            self.bits.push(*b);
        }
    }

    #[inline]
    fn write_unsigned(&mut self, value: u64, bits: u64) {
        for o in (0..bits).rev() {
            self.bits.push(((value >> o) & 0x1) == 0x1);
        }
    }

    #[inline]
    fn write_signed(&mut self, value: i64, bits: u64) {
        for o in (0..bits).rev() {
            self.bits.push(((value >> o) & 0x1) == 0x1);
        }
    }

    /// Write a full sample to the bitvector.
    #[inline]
    fn write_full(&mut self, t: u64, v: f64) {
        let mut buffer = [0u8; 16];

        NativeEndian::write_u64(&mut buffer[0..8], t);
        NativeEndian::write_f64(&mut buffer[8..16], v);

        self.write_buffer(&buffer[..]);
    }

    #[inline]
    fn write_partial(&mut self, v0: f64, t: u64, v: f64) {
        let mut buffer = [0u8; 8];
        NativeEndian::write_u64(&mut buffer[0..8], t);
        self.write_buffer(&buffer[..]);
        self.push_value(v0, v);
    }

    #[inline]
    fn write_buffer(&mut self, buffer: &[u8]) {
        for a in buffer {
            for i in (0u8..8u8).rev() {
                self.bits.push(((a >> i) & 1) == 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    fn build_block(values: &[(u64, f64)]) -> Block {
        let mut b = BlockBuilder::new();

        for &(t, v) in values {
            b.push(t, v);
        }

        b.finalize()
    }

    fn test_values(values: &[(u64, f64)]) {
        let r = build_block(values);

        let mut c = 0;

        for (i, (t, v)) in r.iter().enumerate() {
            let (rt, rv) = values[i];

            assert_eq!(rt, t);

            if rv.is_finite() && v.is_finite() {
                assert_eq!(rv, v);
            } else {
                assert_eq!(float_to_bits(rv), float_to_bits(v));
            }

            c += 1;
        }

        assert_eq!(values.len(), c);
    }

    fn float_to_bits(f: f64) -> u64 {
        unsafe { mem::transmute(f) }
    }

    fn f2b(bits: u64) -> f64 {
        unsafe { mem::transmute(bits) }
    }

    #[test]
    fn test_1() {
        test_values(&[(0, 3.1), (60, 3.2), (120, 3.3), (180, 3.4), (240, 3.5), (300, 3.6)]);
    }

    /// This should cause the encoding to attempt to encode the maximum possible distance in
    /// value.
    #[test]
    fn test_value_limits() {
        test_values(&[(0, f2b(0xffffffffffffffff)),
                      (1, f2b(0x0)),
                      (2, f2b(0xffffffffffffffff)),
                      (3, f2b(0x0)),
                      (4, f2b(0xffffffffffffffff)),
                      (4, f2b(0x0))]);
    }

    /// This should cause the encoding to attempt to encode the maximum possible distance in time.
    #[test]
    fn test_time_limits() {
        test_values(&[(0, 0f64),
                      (0x3fffffffffff, 0f64),
                      (0, 0f64),
                      (0x3fffffffffff, 0f64),
                      (0, 0f64),
                      (0x3fffffffffff, 0f64)]);
    }

    #[test]
    fn test_repeated_values() {
        let mut vec: Vec<(u64, f64)> = Vec::new();
        vec.push((0, 10.0));
        vec.push((10, 20.0));

        for i in 0..4096 {
            vec.push((10 * (i + 2), 33.0 + i as f64));
        }

        let b = build_block(&vec);
        let bytes = b.to_bytes();

        assert_eq!(bytes.len(), 6292);
    }
}
