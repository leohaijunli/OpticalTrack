//! MAVLink 1 / MAVLink 2 frame reader and writer.
//!
//! Accepts raw `.mavlink` streams as well as QGroundControl `.tlog` files
//! (8-byte big-endian microsecond timestamp prefix per frame).  Unknown or
//! corrupted frames are resynchronised byte-by-byte instead of aborting.

use crate::crc::X25;
use crate::dialect::{Dialect, MessageDef, Value};

pub const MAGIC_V1: u8 = 0xFE;
pub const MAGIC_V2: u8 = 0xFD;

#[derive(Debug, Clone)]
pub struct Record {
    /// Timestamp from a .tlog prefix, in seconds (None for raw streams).
    pub t_log: Option<f64>,
    pub sysid: u8,
    pub compid: u8,
    pub msgid: u32,
    pub payload: Vec<u8>,
    pub version: u8,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub frames: usize,
    pub bad_crc: usize,
    pub unknown_msg: usize,
    pub resync_bytes: usize,
    pub per_msg: Vec<(String, usize)>,
}

impl Stats {
    pub fn bump(&mut self, name: &str) {
        if let Some(e) = self.per_msg.iter_mut().find(|(n, _)| n == name) {
            e.1 += 1;
        } else {
            self.per_msg.push((name.to_string(), 1));
        }
    }
}

pub fn read_log(bytes: &[u8], d: &Dialect, stats: &mut Stats) -> Vec<Record> {
    // Heuristic: a tlog starts with a 64-bit timestamp, so byte 0 is almost
    // never a magic. If a plausible frame shows up at offset 8, use tlog mode.
    let tlog_mode = if bytes.is_empty() {
        false
    } else {
        !is_plausible_start(bytes, 0, d) && is_plausible_start(bytes, 8, d)
    };

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let t_stamp = if tlog_mode && i + 8 <= bytes.len() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i..i + 8]);
            Some(u64::from_be_bytes(b) as f64 * 1e-6)
        } else {
            None
        };
        let hdr = if tlog_mode { 8 } else { 0 };
        let p = i + hdr;

        if p >= bytes.len() {
            break;
        }
        let magic = bytes[p];
        if magic != MAGIC_V1 && magic != MAGIC_V2 {
            stats.resync_bytes += 1;
            i += 1;
            continue;
        }

        let header_len = if magic == MAGIC_V2 { 10 } else { 6 };
        if p + header_len > bytes.len() {
            break;
        }
        let len = bytes[p + 1] as usize;
        let (incompat, msgid, payload_start) = if magic == MAGIC_V2 {
            (
                bytes[p + 2],
                (bytes[p + 7] as u32) | ((bytes[p + 8] as u32) << 8) | ((bytes[p + 9] as u32) << 16),
                p + 10,
            )
        } else {
            (0u8, bytes[p + 5] as u32, p + 6)
        };
        let signed = incompat & 0x01 != 0;
        let total = header_len + len + 2 + if signed { 13 } else { 0 };
        if p + total > bytes.len() {
            break;
        }

        let payload = &bytes[payload_start..payload_start + len];
        let crc_pos = payload_start + len;

        let def = d.by_id(msgid);
        let ok = match def {
            Some(m) => validate_crc(&bytes[p..], header_len, len, m.crc_extra),
            None => {
                stats.unknown_msg += 1;
                true // cannot validate; keep the frame so `probe` can report it
            }
        };
        if !ok {
            // move to the next candidate magic (or past the timestamp prefix)
            stats.bad_crc += 1;
            i += 1;
            continue;
        }

        stats.frames += 1;
        stats.bump(def.map(|m| m.name.as_str()).unwrap_or("UNKNOWN"));
        out.push(Record {
            t_log: t_stamp,
            sysid: bytes[p + if magic == MAGIC_V2 { 5 } else { 3 }],
            compid: bytes[p + if magic == MAGIC_V2 { 6 } else { 4 }],
            msgid,
            payload: payload.to_vec(),
            version: if magic == MAGIC_V2 { 2 } else { 1 },
        });
        i += hdr + total;
    }
    out
}

fn is_plausible_start(bytes: &[u8], at: usize, _d: &Dialect) -> bool {
    at < bytes.len() && (bytes[at] == MAGIC_V1 || bytes[at] == MAGIC_V2)
}

fn validate_crc(frame: &[u8], header_len: usize, payload_len: usize, crc_extra: u8) -> bool {
    let mut c = X25::new();
    // Everything after the magic byte: header + transmitted payload. MAVLink 2
    // checksums are computed before trailing-zero truncation, so the bytes on
    // the wire are exactly what both ends hash.
    let n = header_len - 1 + payload_len;
    c.accumulate_buf(&frame[1..1 + n]);
    c.accumulate(crc_extra);
    let rx = u16::from_le_bytes([frame[n + 1], frame[n + 2]]);
    c.crc == rx
}

/// Serialise a MAVLink 2 frame, truncating trailing payload zeros as the
/// specification requires. Note that the checksum is computed over the *full*
/// payload (zero padded); truncation happens afterwards.
pub fn encode_v2(m: &MessageDef, values: &[(&str, Value)], seq: u8, sysid: u8, compid: u8) -> Vec<u8> {
    let full = m.encode(values);
    let mut payload = full.clone();
    while payload.len() > 1 && *payload.last().unwrap() == 0 {
        payload.pop();
    }
    let mut frame = Vec::with_capacity(12 + payload.len());
    frame.push(MAGIC_V2);
    frame.push(payload.len() as u8);
    frame.push(0); // incompat flags
    frame.push(0); // compat flags
    frame.push(seq);
    frame.push(sysid);
    frame.push(compid);
    frame.push((m.id & 0xFF) as u8);
    frame.push(((m.id >> 8) & 0xFF) as u8);
    frame.push(((m.id >> 16) & 0xFF) as u8);
    frame.extend_from_slice(&payload);

    let mut c = X25::new();
    c.accumulate_buf(&frame[1..]);
    c.accumulate(m.crc_extra);
    frame.extend_from_slice(&c.crc.to_le_bytes());
    frame
}

/// Look up a decoded field by name.
pub fn field<'a>(dec: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    dec.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d() -> Dialect {
        let base = env!("CARGO_MANIFEST_DIR");
        Dialect::load(&[
            &format!("{base}/data/minimal.xml"),
            &format!("{base}/data/standard.xml"),
            &format!("{base}/data/common.xml"),
        ])
        .unwrap()
    }

    #[test]
    fn roundtrip_attitude() {
        let dl = d();
        let m = dl.by_name("ATTITUDE").unwrap();
        let vals = [
            ("time_boot_ms", Value::Num(vec![1234.0])),
            ("roll", Value::Num(vec![0.1])),
            ("pitch", Value::Num(vec![-0.2])),
            ("yaw", Value::Num(vec![0.3])),
            ("rollspeed", Value::Num(vec![0.01])),
            ("pitchspeed", Value::Num(vec![0.02])),
            ("yawspeed", Value::Num(vec![0.03])),
        ];
        let bytes = encode_v2(m, &vals, 7, 1, 1);
        let mut st = Stats::default();
        let recs = read_log(&bytes, &dl, &mut st);
        assert_eq!(recs.len(), 1);
        assert_eq!(st.bad_crc, 0);
        let dec = m.decode(&recs[0].payload);
        assert!((field(&dec, "yaw").unwrap().f() - 0.3).abs() < 1e-6);
        assert_eq!(field(&dec, "time_boot_ms").unwrap().f(), 1234.0);
    }
}
