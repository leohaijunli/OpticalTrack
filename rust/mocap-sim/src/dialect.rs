//! MAVLink dialect model built directly from the official XML message definitions.
//!
//! Nothing about field order, sizes or checksum salts is hardcoded: the wire
//! layout (fields sorted by descending element size, extensions appended last)
//! and the `CRC_EXTRA` byte are both derived from the XML, then checked against
//! canonical values in the tests below.

use crate::crc::X25;
use crate::xml;

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    /// Element type without array brackets, e.g. `float`, `uint8_t`.
    pub elem: String,
    pub count: usize,
    pub ext: bool,
}

impl Field {
    pub fn elem_size(&self) -> usize {
        match self.elem.as_str() {
            "uint8_t" | "int8_t" | "char" => 1,
            "uint16_t" | "int16_t" => 2,
            "uint32_t" | "int32_t" | "float" => 4,
            "uint64_t" | "int64_t" | "double" => 8,
            _ => 1,
        }
    }
    pub fn byte_len(&self) -> usize {
        self.elem_size() * self.count
    }
    pub fn is_signed(&self) -> bool {
        self.elem.starts_with("int")
    }
    pub fn is_float(&self) -> bool {
        matches!(self.elem.as_str(), "float" | "double")
    }
}

#[derive(Clone, Debug)]
pub struct MessageDef {
    pub id: u32,
    pub name: String,
    pub fields: Vec<Field>,
    /// Indices into `fields`, in wire order.
    pub wire: Vec<usize>,
    pub crc_extra: u8,
    pub payload_len: usize,
}

#[derive(Clone, Debug)]
pub enum Value {
    /// Numeric scalar, or numeric array flattened to `count` elements.

    Num(Vec<f64>),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn f(&self) -> f64 {
        match self {
            Value::Num(v) => v.first().copied().unwrap_or(0.0),
            Value::Bytes(_) => 0.0,
        }
    }
    pub fn arr(&self) -> Vec<f64> {
        match self {
            Value::Num(v) => v.clone(),
            Value::Bytes(b) => b.iter().map(|&x| x as f64).collect(),
        }
    }
}

#[derive(Default)]
pub struct Dialect {
    pub messages: Vec<MessageDef>,
}

impl Dialect {
    pub fn load(paths: &[&str]) -> std::io::Result<Self> {
        let mut d = Dialect::default();
        for p in paths {
            let src = std::fs::read_to_string(p)?;
            d.add_xml(&src);
        }
        d.messages.sort_by_key(|m| m.id);
        Ok(d)
    }

    pub fn add_xml(&mut self, src: &str) {
        let tags = xml::tags(src);
        let mut i = 0;
        while i < tags.len() {
            let t = &tags[i];
            if t.name == "message" && !t.closing {
                let id: u32 = match t.attr("id").and_then(|s| s.parse().ok()) {
                    Some(v) => v,
                    None => {
                        i += 1;
                        continue;
                    }
                };
                let name = t.attr("name").unwrap_or("UNKNOWN").to_string();
                let mut fields: Vec<Field> = Vec::new();
                let mut in_ext = false;
                let mut j = i + 1;
                while j < tags.len() {
                    let u = &tags[j];
                    if u.name == "message" && u.closing {
                        break;
                    }
                    if u.name == "extensions" && !u.closing {
                        in_ext = true;
                    }
                    if u.name == "field" && !u.closing {
                        if let (Some(ft), Some(fname)) = (u.attr("type"), u.attr("name")) {
                            let (elem, count) = parse_type(ft);
                            fields.push(Field {
                                name: fname.to_string(),
                                elem,
                                count,
                                ext: in_ext,
                            });
                        }
                    }
                    j += 1;
                }
                if self.messages.iter().all(|m| m.id != id) {
                    self.messages.push(build(id, name, fields));
                }
                i = j;
            }
            i += 1;
        }
    }

    pub fn by_id(&self, id: u32) -> Option<&MessageDef> {
        self.messages.iter().find(|m| m.id == id)
    }

    pub fn by_name(&self, name: &str) -> Option<&MessageDef> {
        self.messages.iter().find(|m| m.name == name)
    }
}

fn parse_type(t: &str) -> (String, usize) {
    if let Some(open) = t.find('[') {
        let elem = &t[..open];
        let count = t[open + 1..]
            .trim_end_matches(']')
            .parse::<usize>()
            .unwrap_or(1);
        (crc_type(elem).to_string(), count)
    } else {
        (crc_type(t).to_string(), 1)
    }
}

fn crc_type(elem: &str) -> &str {
    match elem {
        "uint8_t_mavlink_version" => "uint8_t",
        other => other,
    }
}

fn build(id: u32, name: String, fields: Vec<Field>) -> MessageDef {
    // Wire order: non-extension fields sorted by descending element size,
    // stable so equal-size fields keep declaration order; extensions appended
    // in declaration order (MAVLink 2 semantics).
    let mut base: Vec<usize> = (0..fields.len()).filter(|&i| !fields[i].ext).collect();
    base.sort_by_key(|&i| std::cmp::Reverse(fields[i].elem_size()));

    let mut wire = base;
    wire.extend((0..fields.len()).filter(|&i| fields[i].ext));

    let payload_len = wire.iter().map(|&i| fields[i].byte_len()).sum();

    // CRC_EXTRA: message name, then "type name" for every non-extension field
    // in wire order, all space separated; array fields append the element count
    // as a raw byte.
    let mut c = X25::new();
    c.accumulate_str(&name);
    c.accumulate(b' ');
    for &i in wire.iter().filter(|&&i| !fields[i].ext) {
        c.accumulate_str(&fields[i].elem);
        c.accumulate(b' ');
        c.accumulate_str(&fields[i].name);
        c.accumulate(b' ');
        if fields[i].count > 1 {
            c.accumulate(fields[i].count as u8);
        }
    }
    let crc_extra = ((c.crc & 0xFF) ^ (c.crc >> 8)) as u8;

    MessageDef {
        id,
        name,
        fields,
        wire,
        crc_extra,
        payload_len,
    }
}

impl MessageDef {
    /// Decode a (possibly zero-truncated MAVLink 2) payload into named values.
    pub fn decode(&self, payload: &[u8]) -> Vec<(String, Value)> {
        let mut out = Vec::with_capacity(self.fields.len());
        let mut off = 0usize;
        for &fi in &self.wire {
            let f = &self.fields[fi];
            let n = f.byte_len();
            let buf = if off + n <= payload.len() {
                &payload[off..off + n]
            } else {
                // v2 payload truncation: missing trailing bytes are zero
                &[][..]
            };
            let val = if buf.is_empty() {
                if f.count == 1 {
                    Value::Num(vec![0.0])
                } else {
                    Value::Num(vec![0.0; f.count])
                }
            } else if f.elem == "char" {
                Value::Bytes(buf.to_vec())
            } else {
                let mut nums = Vec::with_capacity(f.count);
                let es = f.elem_size();
                for k in 0..f.count {
                    let b = &buf[k * es..(k + 1) * es];
                    nums.push(read_num(b, &f.elem));
                }
                Value::Num(nums)
            };
            out.push((f.name.clone(), val));
            off += n;
        }
        out
    }

    pub fn encode(&self, values: &[(&str, Value)]) -> Vec<u8> {
        let mut payload = vec![0u8; self.payload_len];
        let mut off = 0usize;
        for &fi in &self.wire {
            let f = &self.fields[fi];
            if let Some((_, v)) = values.iter().find(|(k, _)| *k == f.name) {
                let es = f.elem_size();
                match v {
                    Value::Bytes(b) => {
                        for (k, &byte) in b.iter().take(f.count).enumerate() {
                            payload[off + k] = byte;
                        }
                    }
                    Value::Num(nums) => {
                        for k in 0..f.count {
                            let x = nums.get(k).copied().unwrap_or(0.0);
                            if k * es + es <= f.byte_len() {
                                write_num(&mut payload[off + k * es..off + (k + 1) * es], &f.elem, x);
                            }
                        }
                    }
                }
            }
            off += f.byte_len();
        }
        payload
    }
}

fn read_num(b: &[u8], elem: &str) -> f64 {
    match elem {
        "uint8_t" | "char" => b[0] as f64,
        "int8_t" => b[0] as i8 as f64,
        "uint16_t" => u16::from_le_bytes([b[0], b[1]]) as f64,
        "int16_t" => i16::from_le_bytes([b[0], b[1]]) as f64,
        "uint32_t" => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
        "int32_t" => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
        "float" => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
        "uint64_t" => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64,
        "int64_t" => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64,
        "double" => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        _ => 0.0,
    }
}

fn write_num(b: &mut [u8], elem: &str, x: f64) {
    match elem {
        "uint8_t" | "char" => b[0] = x as u8,
        "int8_t" => b[0] = (x as i8) as u8,
        "uint16_t" => b[..2].copy_from_slice(&(x as u16).to_le_bytes()),
        "int16_t" => b[..2].copy_from_slice(&(x as i16).to_le_bytes()),
        "uint32_t" => b[..4].copy_from_slice(&(x as u32).to_le_bytes()),
        "int32_t" => b[..4].copy_from_slice(&(x as i32).to_le_bytes()),
        "float" => b[..4].copy_from_slice(&(x as f32).to_le_bytes()),
        "uint64_t" => b[..8].copy_from_slice(&(x as u64).to_le_bytes()),
        "int64_t" => b[..8].copy_from_slice(&(x as i64).to_le_bytes()),
        "double" => b[..8].copy_from_slice(&x.to_le_bytes()),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialect() -> Dialect {
        let base = env!("CARGO_MANIFEST_DIR");
        Dialect::load(&[
            &format!("{base}/data/minimal.xml"),
            &format!("{base}/data/standard.xml"),
            &format!("{base}/data/common.xml"),
        ])
        .expect("dialect xml")
    }

    #[test]
    fn crc_extra_matches_canonical_values() {
        let d = dialect();
        // Values published by every MAVLink generator.
        let expect = [
            ("HEARTBEAT", 50u8),
            ("SYS_STATUS", 124),
            ("SYSTEM_TIME", 137),
            ("PING", 237),
            ("ATTITUDE", 39),
            ("LOCAL_POSITION_NED", 185),
            ("GLOBAL_POSITION_INT", 104),
            ("RC_CHANNELS", 118),
            ("SET_POSITION_TARGET_LOCAL_NED", 143),
            ("HIGHRES_IMU", 93),
            ("ATT_POS_MOCAP", 109),
            ("ODOMETRY", 91),
            ("TIMESYNC", 34),
            ("VFR_HUD", 20),
            ("STATUSTEXT", 83),
            ("COMMAND_LONG", 152),
            ("GPS_RAW_INT", 24),
            ("SERVO_OUTPUT_RAW", 222),
            ("PARAM_VALUE", 220),
            ("EXTENDED_SYS_STATE", 130),
            ("DISTANCE_SENSOR", 85),
            ("BATTERY_STATUS", 154),
            ("OPTICAL_FLOW", 175),
            ("SCALED_IMU", 170),
            ("RAW_IMU", 144),
        ];
        for (name, want) in expect {
            let m = d.by_name(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(m.crc_extra, want, "CRC_EXTRA mismatch for {name}");
        }
    }

    #[test]
    fn wire_order_sorts_by_size_stable() {
        let d = dialect();
        let m = d.by_name("ATTITUDE").unwrap();
        let names: Vec<&str> = m.wire.iter().map(|&i| m.fields[i].name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "time_boot_ms",
                "roll",
                "pitch",
                "yaw",
                "rollspeed",
                "pitchspeed",
                "yawspeed"
            ]
        );
        assert_eq!(m.payload_len, 28);
    }
}
