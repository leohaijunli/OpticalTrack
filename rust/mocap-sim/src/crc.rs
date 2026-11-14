//! MAVLink X.25 CRC (CRC-16/MCRF4XX), used for frame checksums and CRC_EXTRA.

#[derive(Clone, Copy)]
pub struct X25 {
    pub crc: u16,
}

impl Default for X25 {
    fn default() -> Self {
        Self::new()
    }
}

impl X25 {
    pub fn new() -> Self {
        Self { crc: 0xFFFF }
    }

    pub fn accumulate(&mut self, byte: u8) {
        let mut tmp = byte ^ (self.crc & 0xFF) as u8;
        tmp = (tmp ^ (tmp << 4)) & 0xFF;
        self.crc = ((self.crc >> 8)
            ^ ((tmp as u16) << 8)
            ^ ((tmp as u16) << 3)
            ^ ((tmp as u16) >> 4))
            & 0xFFFF;
    }

    pub fn accumulate_buf(&mut self, data: &[u8]) {
        for &b in data {
            self.accumulate(b);
        }
    }

    pub fn accumulate_str(&mut self, s: &str) {
        self.accumulate_buf(s.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::X25;

    #[test]
    fn x25_known_vector() {
        // CRC of the ASCII string "123456789" with CRC-16/MCRF4XX is 0x6F91.
        let mut c = X25::new();
        c.accumulate_str("123456789");
        assert_eq!(c.crc, 0x6F91);
    }
}
