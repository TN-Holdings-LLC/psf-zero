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
}
