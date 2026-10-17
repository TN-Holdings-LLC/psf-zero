//! Reading OpenQASM 3 (the subset Qiskit's qasm3.dumps writes for gate-level circuits) and writing it.
//!
//! Read: `qubit[n] name;` and `bit[n] name;` registers (several are concatenated in declaration order), gate
//! applications `name(params) a, b;`, `c[i] = measure q[j];`, `measure q[j] -> c[i];`, `barrier ...;` (ignored).
//! `gate name(...) ... { ... }` definitions are skipped: a gate is known by its name only, and only names in
//! `compile::KNOWN` are accepted (Qiskit writes rzz, rxx, ryy as definitions with these names). Parameters are
//! numbers with + - * / ( ) and pi.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Op {
    Gate { name: String, qubits: Vec<usize>, params: Vec<f64> },
    Measure { qubit: usize, clbit: usize },
}

#[derive(Clone, Debug)]
pub struct Circuit {
    pub nq: usize,
    pub nc: usize,
    pub ops: Vec<Op>,
}

fn strip_comments(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '/' && it.peek() == Some(&'/') {
            for d in it.by_ref() {
                if d == '\n' {
                    out.push('\n');
                    break;
                }
            }
        } else if c == '/' && it.peek() == Some(&'*') {
            it.next();
            let mut prev = ' ';
            for d in it.by_ref() {
                if prev == '*' && d == '/' {
                    break;
                }
                prev = d;
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Evaluates a parameter expression.
pub fn eval(e: &str) -> Result<f64, String> {
    let toks: Vec<char> = e.chars().filter(|c| !c.is_whitespace()).collect();
    let mut i = 0;
    let v = expr(&toks, &mut i)?;
    if i != toks.len() {
        return Err(format!("qasm: cannot read the parameter '{}'", e));
    }
    Ok(v)
}

fn expr(t: &[char], i: &mut usize) -> Result<f64, String> {
    let mut v = term(t, i)?;
    while *i < t.len() && (t[*i] == '+' || t[*i] == '-') {
        let op = t[*i];
        *i += 1;
        let w = term(t, i)?;
        v = if op == '+' { v + w } else { v - w };
    }
    Ok(v)
}

fn term(t: &[char], i: &mut usize) -> Result<f64, String> {
    let mut v = unary(t, i)?;
    while *i < t.len() && (t[*i] == '*' || t[*i] == '/') {
        let op = t[*i];
        *i += 1;
        let w = unary(t, i)?;
        v = if op == '*' { v * w } else { v / w };
    }
    Ok(v)
}

fn unary(t: &[char], i: &mut usize) -> Result<f64, String> {
    if *i < t.len() && t[*i] == '-' {
        *i += 1;
        return Ok(-unary(t, i)?);
    }
    if *i < t.len() && t[*i] == '+' {
        *i += 1;
        return unary(t, i);
    }
    atom(t, i)
}

fn atom(t: &[char], i: &mut usize) -> Result<f64, String> {
    if *i >= t.len() {
        return Err("qasm: parameter ends early".into());
    }
    if t[*i] == '(' {
        *i += 1;
        let v = expr(t, i)?;
        if *i >= t.len() || t[*i] != ')' {
            return Err("qasm: ')' expected".into());
        }
        *i += 1;
        return Ok(v);
    }
    let rest: String = t[*i..].iter().collect();
    if rest.starts_with("pi") {
        *i += 2;
        return Ok(std::f64::consts::PI);
    }
    if rest.starts_with('π') {
        *i += 1;
        return Ok(std::f64::consts::PI);
    }
    let st = *i;
    while *i < t.len() && (t[*i].is_ascii_digit() || t[*i] == '.' || t[*i] == 'e' || t[*i] == 'E'
        || ((t[*i] == '-' || t[*i] == '+') && *i > st && (t[*i - 1] == 'e' || t[*i - 1] == 'E')))
    {
        *i += 1;
    }
    let s: String = t[st..*i].iter().collect();
    s.parse::<f64>().map_err(|_| format!("qasm: cannot read the number '{}'", s))
}

fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c)
            }
            ')' => {
                depth -= 1;
                cur.push(c)
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear()
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub fn read(text: &str) -> Result<Circuit, String> {
    let text = strip_comments(text);
    // statements: split at ';' outside braces; gate definitions (with braces) are dropped whole
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    for c in text.chars() {
        match c {
            '{' => {
                depth += 1;
                cur.push(c)
            }
            '}' => {
                depth -= 1;
                cur.push(c);
                if depth == 0 {
                    stmts.push(cur.trim().to_string());
                    cur.clear();
                }
            }
            ';' if depth == 0 => {
                stmts.push(cur.trim().to_string());
                cur.clear()
            }
            _ => cur.push(c),
        }
    }
    let mut qregs: HashMap<String, (usize, usize)> = HashMap::new();
    let mut cregs: HashMap<String, (usize, usize)> = HashMap::new();
    let (mut nq, mut nc) = (0usize, 0usize);
    let mut ops = Vec::new();

    fn reg_decl(s: &str, kw: &str) -> Option<(String, usize)> {
        let r = s.strip_prefix(kw)?.trim();
        if let Some(r) = r.strip_prefix('[') {
            let close = r.find(']')?;
            let size = r[..close].trim().parse().ok()?;
            Some((r[close + 1..].trim().to_string(), size))
        } else {
            Some((r.trim().to_string(), 1))
        }
    }
    fn index(s: &str, regs: &HashMap<String, (usize, usize)>) -> Result<Vec<usize>, String> {
        let s = s.trim();
        if let Some(p) = s.find('[') {
            let name = s[..p].trim();
            let k: usize = s[p + 1..s.len() - 1].trim().parse().map_err(|_| format!("qasm: bad index '{}'", s))?;
            let (off, size) = regs.get(name).ok_or(format!("qasm: unknown register '{}'", name))?;
            if k >= *size {
                return Err(format!("qasm: index out of range '{}'", s));
            }
            Ok(vec![off + k])
        } else {
            let (off, size) = regs.get(s).ok_or(format!("qasm: unknown register '{}'", s))?;
            Ok((*off..off + size).collect())
        }
    }

    for st in stmts {
        if st.is_empty() || st.starts_with("OPENQASM") || st.starts_with("include") || st.starts_with("gate ")
            || st.starts_with("barrier")
        {
            continue;
        }
        if st.starts_with("qubit") {
            let (name, size) = reg_decl(&st, "qubit").ok_or(format!("qasm: cannot read '{}'", st))?;
            qregs.insert(name, (nq, size));
            nq += size;
            continue;
        }
        if st.starts_with("bit") {
            let (name, size) = reg_decl(&st, "bit").ok_or(format!("qasm: cannot read '{}'", st))?;
            cregs.insert(name, (nc, size));
            nc += size;
            continue;
        }
        if st.contains("measure") {
            let (q, c) = if let Some(eq) = st.find('=') {
                (st[eq + 1..].trim().trim_start_matches("measure").trim().to_string(), st[..eq].trim().to_string())
            } else if let Some(ar) = st.find("->") {
                (st[..ar].trim().trim_start_matches("measure").trim().to_string(), st[ar + 2..].trim().to_string())
            } else {
                return Err(format!("qasm: cannot read '{}'", st));
            };
            let qs = index(&q, &qregs)?;
            let cs = index(&c, &cregs)?;
            if qs.len() != cs.len() {
                return Err(format!("qasm: '{}' pairs registers of different sizes", st));
            }
            for (a, b) in qs.into_iter().zip(cs) {
                ops.push(Op::Measure { qubit: a, clbit: b });
            }
            continue;
        }
        // a gate application
        let (head, args) = match st.find('(') {
            Some(p) if st[..p].trim().chars().all(|c| c.is_alphanumeric() || c == '_') => {
                let close = st.rfind(')').ok_or(format!("qasm: cannot read '{}'", st))?;
                (st[..close + 1].to_string(), st[close + 1..].to_string())
            }
            _ => {
                let sp = st.find(char::is_whitespace).ok_or(format!("qasm: cannot read '{}'", st))?;
                (st[..sp].to_string(), st[sp..].to_string())
            }
        };
        let (name, params) = match head.find('(') {
            Some(p) => {
                let ps = split_args(&head[p + 1..head.len() - 1]);
                (head[..p].trim().to_string(), ps.iter().map(|e| eval(e)).collect::<Result<Vec<_>, _>>()?)
            }
            None => (head.trim().to_string(), vec![]),
        };
        let mut qubits = Vec::new();
        for a in split_args(&args) {
            let v = index(&a, &qregs)?;
            if v.len() != 1 {
                return Err(format!("qasm: '{}' applies a gate to a whole register (not read)", st));
            }
            qubits.push(v[0]);
        }
        ops.push(Op::Gate { name, qubits, params });
    }
    Ok(Circuit { nq, nc, ops })
}

/// OpenQASM 3 of a compiled circuit on physical qubits ($k), with its classical bits.
pub fn write(nc: usize, ops: &[(String, Vec<usize>, Vec<f64>)], meas: &[(usize, usize)], note: &str) -> String {
    let mut s = String::from("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
    s += &format!("// {}\n", note);
    if ops.iter().any(|o| o.0 == "ecr") {
        s += "// ecr is the device's native two-qubit gate (IBM Eagle)\n";
        // ECR = RZX(pi/4) (X on _a) RZX(-pi/4), RZX(t) = (h on _b) exp(-i t/2 ZZ) (h on _b): checked against Qiskit's
        // ECRGate by the harness
        s += "gate ecr _a, _b { h _b; cx _a, _b; rz(pi/4) _b; cx _a, _b; h _b; x _a; h _b; cx _a, _b; rz(-pi/4) _b; cx _a, _b; h _b; }\n";
    }
    if nc > 0 {
        s += &format!("bit[{}] c;\n", nc);
    }
    for (name, qs, ps) in ops {
        let q = qs.iter().map(|k| format!("${}", k)).collect::<Vec<_>>().join(", ");
        if ps.is_empty() {
            s += &format!("{} {};\n", name, q);
        } else {
            let p = ps.iter().map(|v| format!("{:.17}", v)).collect::<Vec<_>>().join(", ");
            s += &format!("{}({}) {};\n", name, p, q);
        }
    }
    for (q, c) in meas {
        s += &format!("c[{}] = measure ${};\n", c, q);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_qiskit_style() {
        let t = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ngate rzz(_gate_p_0) _gate_q_0, _gate_q_1 {\n  cx _gate_q_0, _gate_q_1;\n  rz(_gate_p_0) _gate_q_1;\n}\nbit[2] meas;\nqubit[2] q;\nry(-pi/4) q[0];\nrzz(0.5*pi) q[0], q[1];\nbarrier q[0], q[1];\nmeas[0] = measure q[0];\nmeas[1] = measure q[1];\n";
        let c = read(t).unwrap();
        assert_eq!((c.nq, c.nc, c.ops.len()), (2, 2, 4));
        match &c.ops[1] {
            Op::Gate { name, qubits, params } => {
                assert_eq!(name, "rzz");
                assert_eq!(qubits, &vec![0, 1]);
                assert!((params[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
            }
            _ => panic!(),
        }
        assert!((eval("-(1.5e-1+pi)/2").unwrap() + (0.15 + std::f64::consts::PI) / 2.0).abs() < 1e-15);
    }
}
