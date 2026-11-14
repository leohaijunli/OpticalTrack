//! A deliberately tiny XML tag scanner.  It only needs to survive the MAVLink
//! message-definition dialect files, so it extracts tags + attributes and skips
//! comments, processing instructions, doctypes and text content.

#[derive(Debug, Clone, Default)]
pub struct Tag {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub closing: bool,
}

impl Tag {
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

pub fn tags(src: &str) -> Vec<Tag> {
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        let rest = &src[i..];

        if rest.starts_with("<!--") {
            match rest.find("-->") {
                Some(p) => {
                    i += p + 3;
                    continue;
                }
                None => break,
            }
        }
        // <!DOCTYPE ...> and <?xml ...?>
        if rest.starts_with("<?") || rest.starts_with("<!") {
            match rest.find('>') {
                Some(p) => {
                    i += p + 1;
                    continue;
                }
                None => break,
            }
        }
        if rest.starts_with("<![CDATA[") {
            match rest.find("]]>") {
                Some(p) => {
                    i += p + 3;
                    continue;
                }
                None => break,
            }
        }

        let mut j = i + 1;
        let closing = j < b.len() && b[j] == b'/';
        if closing {
            j += 1;
        }
        let name_start = j;
        while j < b.len() && !b[j].is_ascii_whitespace() && b[j] != b'>' && b[j] != b'/' {
            j += 1;
        }
        let name = src[name_start..j].to_string();

        let mut attrs: Vec<(String, String)> = Vec::new();
        let mut self_closing = false;
        loop {
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= b.len() {
                break;
            }
            if b[j] == b'>' {
                j += 1;
                break;
            }
            if b[j] == b'/' {
                self_closing = true;
                j += 1;
                continue;
            }
            let key_start = j;
            while j < b.len()
                && b[j] != b'='
                && !b[j].is_ascii_whitespace()
                && b[j] != b'>'
                && b[j] != b'/'
            {
                j += 1;
            }
            let key = src[key_start..j].to_string();
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'=' {
                j += 1;
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                let quote = if j < b.len() && (b[j] == b'"' || b[j] == b'\'') {
                    let q = b[j];
                    j += 1;
                    q
                } else {
                    0
                };
                let val_start = j;
                if quote != 0 {
                    while j < b.len() && b[j] != quote {
                        j += 1;
                    }
                } else {
                    while j < b.len() && !b[j].is_ascii_whitespace() && b[j] != b'>' {
                        j += 1;
                    }
                }
                let val = src[val_start..j].to_string();
                if j < b.len() && quote != 0 {
                    j += 1;
                }
                attrs.push((key, unescape(&val)));
            }
        }
        i = j;
        out.push(Tag {
            name,
            attrs,
            closing,
        });
        let _ = self_closing;
    }
    out
}

fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}
