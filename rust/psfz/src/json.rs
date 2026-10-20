//! A minimal JSON reader and writer (objects, arrays, numbers, strings, true, false, null): enough for device files
//! and reports, with no outside crate.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(BTreeMap<String, J>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(m) => m.get(k),
            _ => None,
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            J::Num(x) => Some(*x),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn arr(&self) -> Option<&Vec<J>> {
        match self {
            J::Arr(v) => Some(v),
            _ => None,
        }
    }
}

pub fn parse(s: &str) -> Result<J, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    ws(b, &mut i);
    if i != b.len() {
        return Err(format!("json: trailing text at {}", i));
    }
    Ok(v)
}

fn ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && (b[*i] as char).is_ascii_whitespace() {
        *i += 1;
    }
}

fn value(b: &[u8], i: &mut usize) -> Result<J, String> {
    ws(b, i);
    if *i >= b.len() {
        return Err("json: unexpected end".into());
    }
    match b[*i] {
        b'{' => {
            *i += 1;
            let mut m = BTreeMap::new();
            ws(b, i);
            if *i < b.len() && b[*i] == b'}' {
                *i += 1;
                return Ok(J::Obj(m));
            }
            loop {
                ws(b, i);
                let k = match value(b, i)? {
                    J::Str(s) => s,
                    _ => return Err("json: key is not a string".into()),
                };
                ws(b, i);
                if *i >= b.len() || b[*i] != b':' {
                    return Err("json: ':' expected".into());
                }
                *i += 1;
                let v = value(b, i)?;
                m.insert(k, v);
                ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == b'}' {
                    *i += 1;
                    return Ok(J::Obj(m));
                }
                return Err("json: ',' or '}' expected".into());
            }
        }
        b'[' => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if *i < b.len() && b[*i] == b']' {
                *i += 1;
                return Ok(J::Arr(v));
            }
            loop {
                v.push(value(b, i)?);
                ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == b']' {
                    *i += 1;
                    return Ok(J::Arr(v));
                }
                return Err("json: ',' or ']' expected".into());
            }
        }
        b'"' => {
            *i += 1;
            let mut s = String::new();
            while *i < b.len() && b[*i] != b'"' {
                if b[*i] == b'\\' && *i + 1 < b.len() {
                    *i += 1;
                    s.push(match b[*i] {
                        b'n' => '\n',
                        b't' => '\t',
                        c => c as char,
                    });
                } else {
                    s.push(b[*i] as char);
                }
                *i += 1;
            }
            *i += 1;
            Ok(J::Str(s))
        }
        b't' if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(J::Bool(true))
        }
        b'f' if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(J::Bool(false))
        }
        b'n' if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(J::Null)
        }
        _ => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *i += 1;
            }
            let t = std::str::from_utf8(&b[st..*i]).unwrap();
            t.parse::<f64>().map(J::Num).map_err(|_| format!("json: bad number '{}'", t))
        }
    }
}

/// Strict JSON (RFC 8259), for records written by a model (gate 1): number grammar as the RFC (no "+1", "01", "1.",
/// ".5", NaN or Infinity); strings as UTF-8 with the RFC's escapes only and no raw control characters; a duplicate
/// key is an error. A number literal that overflows f64 (1e999) parses to +-inf, which gate 1's finiteness check
/// reports. `parse` above stays as it is for device files and reports.
pub fn parse_strict(s: &str) -> Result<J, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = sval(b, &mut i, 0)?;
    ws(b, &mut i);
    if i != b.len() {
        return Err(format!("json: trailing text at byte {}", i));
    }
    Ok(v)
}

fn sval(b: &[u8], i: &mut usize, depth: usize) -> Result<J, String> {
    if depth > 64 {
        return Err("json: nested deeper than 64".into());
    }
    ws(b, i);
    if *i >= b.len() {
        return Err("json: unexpected end".into());
    }
    match b[*i] {
        b'{' => {
            *i += 1;
            let mut m = BTreeMap::new();
            ws(b, i);
            if *i < b.len() && b[*i] == b'}' {
                *i += 1;
                return Ok(J::Obj(m));
            }
            loop {
                ws(b, i);
                if *i >= b.len() || b[*i] != b'"' {
                    return Err(format!("json: key expected at byte {}", i));
                }
                let k = sstr(b, i)?;
                ws(b, i);
                if *i >= b.len() || b[*i] != b':' {
                    return Err(format!("json: ':' expected at byte {}", i));
                }
                *i += 1;
                let v = sval(b, i, depth + 1)?;
                if m.insert(k.clone(), v).is_some() {
                    return Err(format!("json: duplicate key '{}'", k));
                }
                ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == b'}' {
                    *i += 1;
                    return Ok(J::Obj(m));
                }
                return Err(format!("json: ',' or '}}' expected at byte {}", i));
            }
        }
        b'[' => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if *i < b.len() && b[*i] == b']' {
                *i += 1;
                return Ok(J::Arr(v));
            }
            loop {
                v.push(sval(b, i, depth + 1)?);
                ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == b']' {
                    *i += 1;
                    return Ok(J::Arr(v));
                }
                return Err(format!("json: ',' or ']' expected at byte {}", i));
            }
        }
        b'"' => Ok(J::Str(sstr(b, i)?)),
        b't' if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(J::Bool(true))
        }
        b'f' if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(J::Bool(false))
        }
        b'n' if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(J::Null)
        }
        b'-' | b'0'..=b'9' => snum(b, i),
        c => Err(format!("json: unexpected '{}' at byte {}", c as char, i)),
    }
}

fn snum(b: &[u8], i: &mut usize) -> Result<J, String> {
    let st = *i;
    let digits = |i: &mut usize| {
        let s = *i;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
        *i - s
    };
    if b[*i] == b'-' {
        *i += 1;
    }
    if *i < b.len() && b[*i] == b'0' {
        *i += 1;
    } else if digits(i) == 0 {
        return Err(format!("json: bad number at byte {}", st));
    }
    if *i < b.len() && b[*i] == b'.' {
        *i += 1;
        if digits(i) == 0 {
            return Err(format!("json: bad number at byte {}", st));
        }
    }
    if *i < b.len() && (b[*i] == b'e' || b[*i] == b'E') {
        *i += 1;
        if *i < b.len() && (b[*i] == b'+' || b[*i] == b'-') {
            *i += 1;
        }
        if digits(i) == 0 {
            return Err(format!("json: bad number at byte {}", st));
        }
    }
    if *i < b.len() && b[*i].is_ascii_digit() {
        return Err(format!("json: bad number at byte {}", st)); // a leading zero followed by digits
    }
    let t = std::str::from_utf8(&b[st..*i]).unwrap();
    t.parse::<f64>().map(J::Num).map_err(|_| format!("json: bad number '{}'", t))
}

fn sstr(b: &[u8], i: &mut usize) -> Result<String, String> {
    *i += 1; // the opening quote
    let mut out: Vec<u8> = Vec::new();
    loop {
        if *i >= b.len() {
            return Err("json: unterminated string".into());
        }
        let c = b[*i];
        *i += 1;
        match c {
            b'"' => break,
            0x00..=0x1f => return Err("json: control character in a string".into()),
            b'\\' => {
                if *i >= b.len() {
                    return Err("json: unterminated escape".into());
                }
                let e = b[*i];
                *i += 1;
                let ch = match e {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let hex4 = |i: &mut usize| -> Result<u32, String> {
                            if *i + 4 > b.len() {
                                return Err("json: short \\u escape".into());
                            }
                            let h = std::str::from_utf8(&b[*i..*i + 4]).map_err(|_| "json: bad \\u escape")?;
                            *i += 4;
                            u32::from_str_radix(h, 16).map_err(|_| "json: bad \\u escape".to_string())
                        };
                        let mut u = hex4(i)?;
                        if (0xd800..0xdc00).contains(&u) {
                            if !(b[*i..].starts_with(b"\\u")) {
                                return Err("json: lone surrogate".into());
                            }
                            *i += 2;
                            let lo = hex4(i)?;
                            if !(0xdc00..0xe000).contains(&lo) {
                                return Err("json: bad surrogate pair".into());
                            }
                            u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
                        }
                        char::from_u32(u).ok_or("json: bad \\u escape")?
                    }
                    _ => return Err("json: bad escape".into()),
                };
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
            _ => out.push(c),
        }
    }
    String::from_utf8(out).map_err(|_| "json: a string is not UTF-8".into())
}

pub fn write(v: &J) -> String {
    match v {
        J::Null => "null".into(),
        J::Bool(b) => b.to_string(),
        J::Num(x) => {
            if x.fract() == 0.0 && x.abs() < 1e15 {
                format!("{}", *x as i64)
            } else {
                format!("{:e}", x)
            }
        }
        J::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")),
        J::Arr(a) => format!("[{}]", a.iter().map(write).collect::<Vec<_>>().join(",")),
        J::Obj(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, v)| format!("\"{}\":{}", k, write(v))).collect::<Vec<_>>().join(",")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let s = r#"{"a": [1, 2.5, -3e-2, "x"], "b": {"c": null, "d": true}}"#;
        let v = parse(s).unwrap();
        assert_eq!(parse(&write(&v)).unwrap(), v);
        assert_eq!(v.get("a").unwrap().arr().unwrap()[2].num().unwrap(), -0.03);
    }

    #[test]
    fn strict() {
        assert!(parse_strict(r#"{"a": [1, 2.5, -3e-2, "xé\n"], "b": {"c": null, "d": true}}"#).is_ok());
        assert_eq!(parse_strict(r#""日本""#).unwrap(), J::Str("日本".into()));
        for bad in ["+1", "01", "1.", ".5", "NaN", "Infinity", "-", "1e", r#"{"a":1,"a":2}"#, "[1,]", r#""\x""#,
                    "\"a\u{1}\"", r#"{"a" 1}"#, "tru", "[1] 2"] {
            assert!(parse_strict(bad).is_err(), "{}", bad);
        }
        assert_eq!(parse_strict("1e999").unwrap(), J::Num(f64::INFINITY));
        assert_eq!(parse_strict("-0").unwrap(), J::Num(-0.0));
    }
}
