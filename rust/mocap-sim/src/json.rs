//! Minimal JSON emission helpers (no external crates).

pub struct J {
    buf: String,
}

impl J {
    pub fn new() -> Self {
        J { buf: String::new() }
    }
    pub fn raw(&mut self, s: &str) {
        self.buf.push_str(s);
    }
    pub fn str(&mut self, s: &str) {
        self.buf.push('"');
        for c in s.chars() {
            match c {
                '"' => self.buf.push_str("\\\""),
                '\\' => self.buf.push_str("\\\\"),
                '\n' => self.buf.push_str("\\n"),
                '\r' => self.buf.push_str("\\r"),
                '\t' => self.buf.push_str("\\t"),
                c if (c as u32) < 0x20 => self.buf.push_str(&format!("\\u{:04x}", c as u32)),
                c => self.buf.push(c),
            }
        }
        self.buf.push('"');
    }
    pub fn num(&mut self, v: f64) {
        if v.is_finite() {
            if v == v.round() && v.abs() < 1e15 {
                self.buf.push_str(&format!("{:.0}", v));
            } else {
                self.buf.push_str(&format!("{:.7}", v));
            }
        } else {
            self.buf.push_str("null");
        }
    }
    pub fn key(&mut self, k: &str) {
        self.str(k);
        self.buf.push(':');
    }
    pub fn kv_num(&mut self, k: &str, v: f64) {
        self.key(k);
        self.num(v);
    }
    pub fn kv_str(&mut self, k: &str, v: &str) {
        self.key(k);
        self.str(v);
    }
    pub fn kv_arr3(&mut self, k: &str, v: [f64; 3]) {
        self.key(k);
        self.arr3(v);
    }
    pub fn arr3(&mut self, v: [f64; 3]) {
        self.buf.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                self.buf.push(',');
            }
            self.num(*x);
        }
        self.buf.push(']');
    }
    pub fn arr4(&mut self, v: [f64; 4]) {
        self.buf.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                self.buf.push(',');
            }
            self.num(*x);
        }
        self.buf.push(']');
    }
    pub fn finish(self) -> String {
        self.buf
    }
}
