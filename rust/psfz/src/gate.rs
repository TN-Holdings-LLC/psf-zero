//! Gate 1 (P0, 2026-10-20): checks an IR v0 record (`psf-intent/0`, IR r2: psf-zero
//! `docs/design/IR_v0_spec_r2_2026-10-20.md` and its schema `psf-intent-0.r2.schema.json`) in a fixed order and
//! either refuses it with one code, declines it (the decline record), or passes it with the built output's counts.
//!
//! Every check is a formula on the record and the device file; placement is psfz's own (`place::chain`,
//! `place::ring`), so gate 1 and the build agree by construction. Nothing is searched, scored or retried, and the
//! model's text never reaches anything but the strict JSON parser and the schema.
//!
//! Conventions this module fixes where the IR leaves them open (delegated to home by the owner, 2026-10-20; listed in
//! the gate's README note): TFIM's rzz angle is 2 J dt and its rx angle 2 h dt per first-order step (Qiskit's
//! rzz(t) = exp(-i t ZZ / 2), rx(t) = exp(-i t X / 2)); second order is Strang's splitting rx(h dt) rzz rx(h dt), so
//! k = 1 rzz layer per step at both orders (all ZZ terms commute); the TFIM chain starts in |0...0> and measures every
//! qubit; the ring classifier measures its readout qubit only, into clbit 0.

use crate::compile;
use crate::device::{Device, Lattice};
use crate::json::{self, J};
use crate::place;
use crate::qasm;
use crate::sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

pub const VERSION: &str = "psfz gate 1 (IR r2), 2026-10-20";
const IR: &str = "psf-intent/0";

/// Limits and where device files live. `skip` is for fault injection in tests only; the command line never sets it.
#[derive(Clone, Debug)]
pub struct Config {
    pub device_dir: PathBuf,
    pub cz_max: f64,
    pub sx_max: f64,
    pub chi_max: f64,
    pub skip: Vec<String>,
}

impl Config {
    pub fn from_json(j: &J, base: &Path) -> Result<Config, String> {
        let num = |k: &str| j.get(k).and_then(J::num).ok_or(format!("config: '{}' missing", k));
        let dir = j.get("device_dir").and_then(J::str).ok_or("config: 'device_dir' missing")?;
        Ok(Config { device_dir: base.join(dir), cz_max: num("CZ_max")?, sx_max: num("SX_max")?,
                    chi_max: num("chi_max")?, skip: Vec::new() })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Passed(BTreeMap<String, J>),
    Declined(String),
    Refused { code: String, check: String, field: String, formula: String, reason: String },
}

fn refuse(code: &str, check: &str, field: &str, formula: &str, reason: String) -> Outcome {
    Outcome::Refused { code: code.into(), check: check.into(), field: field.into(), formula: formula.into(), reason }
}

impl Outcome {
    pub fn code_check(&self) -> (String, String) {
        match self {
            Outcome::Passed(_) => ("pass".into(), "-".into()),
            Outcome::Declined(_) => ("declined".into(), "D".into()),
            Outcome::Refused { code, check, .. } => (code.clone(), check.clone()),
        }
    }

    pub fn to_json(&self, record_sha: &str) -> J {
        let mut m = BTreeMap::new();
        m.insert("gate".to_string(), J::Str(VERSION.into()));
        m.insert("record_sha256".to_string(), J::Str(record_sha.into()));
        match self {
            Outcome::Passed(p) => {
                m.insert("passed".into(), J::Bool(true));
                for (k, v) in p {
                    m.insert(k.clone(), v.clone());
                }
            }
            Outcome::Declined(r) => {
                m.insert("declined".into(), J::Bool(true));
                m.insert("reason".into(), J::Str(r.clone()));
            }
            Outcome::Refused { code, check, field, formula, reason } => {
                m.insert("refused".into(), J::Bool(true));
                m.insert("code".into(), J::Str(code.clone()));
                m.insert("check".into(), J::Str(check.clone()));
                m.insert("field".into(), J::Str(field.clone()));
                m.insert("formula".into(), J::Str(formula.clone()));
                m.insert("reason".into(), J::Str(reason.clone()));
            }
        }
        J::Obj(m)
    }
}

// ------------------------------------------------------------------------------------------------ check 3: schema

type Sch = Result<(), (String, String)>; // (field path, reason)

fn bad(path: &str, why: &str) -> Sch {
    Err((path.to_string(), why.to_string()))
}

fn obj<'a>(v: &'a J, path: &str) -> Result<&'a BTreeMap<String, J>, (String, String)> {
    match v {
        J::Obj(m) => Ok(m),
        _ => Err((path.into(), "not an object".into())),
    }
}

fn keys(m: &BTreeMap<String, J>, path: &str, allowed: &[&str], required: &[&str]) -> Sch {
    for k in m.keys() {
        if !allowed.contains(&k.as_str()) {
            return bad(&format!("{}{}", path, k), "unknown field (additionalProperties: false)");
        }
    }
    for k in required {
        if !m.contains_key(*k) {
            return bad(&format!("{}{}", path, k), "missing (required)");
        }
    }
    Ok(())
}

fn number(m: &BTreeMap<String, J>, path: &str, k: &str, lo: Option<(f64, bool)>, hi: f64) -> Sch {
    // lo: (bound, exclusive)
    let p = format!("{}{}", path, k);
    match m.get(k) {
        None => Ok(()),
        Some(J::Num(x)) => {
            if let Some((b, ex)) = lo {
                if (ex && *x <= b) || (!ex && *x < b) {
                    return bad(&p, &format!("{} is below the minimum {}{}", x, b, if ex { " (exclusive)" } else { "" }));
                }
            }
            if *x > hi {
                return bad(&p, &format!("{} is above the maximum {}", x, hi));
            }
            Ok(())
        }
        Some(_) => bad(&p, "not a number"),
    }
}

fn integer(m: &BTreeMap<String, J>, path: &str, k: &str, lo: f64, hi: f64) -> Sch {
    let p = format!("{}{}", path, k);
    match m.get(k) {
        None => Ok(()),
        Some(J::Num(x)) if x.fract() == 0.0 => {
            if *x < lo || *x > hi {
                bad(&p, &format!("{} is outside {} to {}", x, lo, hi))
            } else {
                Ok(())
            }
        }
        Some(_) => bad(&p, "not an integer"),
    }
}

fn enum_str(m: &BTreeMap<String, J>, path: &str, k: &str, allowed: &[&str]) -> Sch {
    match m.get(k) {
        None => Ok(()),
        Some(J::Str(s)) if allowed.contains(&s.as_str()) => Ok(()),
        Some(_) => bad(&format!("{}{}", path, k), &format!("not one of {:?}", allowed)),
    }
}

fn device_path_ok(s: &str) -> bool {
    let r = match s.strip_prefix("file:") {
        Some(r) => r,
        None => return false,
    };
    let ok_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/');
    !r.is_empty() && r.len() <= 255 && r.chars().all(ok_char) && !r.starts_with('/') && !s.contains("..")
}

/// The r2 schema, written out as code; `tests` and the N2 harness compare it with the JSON Schema file.
pub fn schema(v: &J) -> Sch {
    let m = obj(v, "")?;
    if m.contains_key("decline") {
        keys(m, "", &["ir", "decline"], &["ir", "decline"])?;
        return enum_str(m, "", "decline", &["out_of_family"]);
    }
    keys(m, "", &["ir", "family", "params", "output", "device", "device_sha256", "twin"],
         &["ir", "family", "params", "output", "device"])?;
    if m.get("ir").and_then(J::str) != Some(IR) {
        return bad("ir", "not psf-intent/0");
    }
    enum_str(m, "", "family", &["tfim_chain", "ring_classifier", "rzz_layer"])?;
    enum_str(m, "", "output", &["template", "circuit"])?;
    match m.get("device") {
        Some(J::Str(s)) if device_path_ok(s) => {}
        _ => return bad("device", "not 'file:' and a relative path of [A-Za-z0-9._/-], without '..'"),
    }
    if let Some(h) = m.get("device_sha256") {
        match h {
            J::Str(s) if s.len() == 64 && s.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)) => {}
            _ => return bad("device_sha256", "not 64 lowercase hex digits"),
        }
    }
    if let Some(t) = m.get("twin") {
        let tm = obj(t, "twin")?;
        keys(tm, "twin.", &["requested"], &["requested"])?;
        if !matches!(tm.get("requested"), Some(J::Bool(_))) {
            return bad("twin.requested", "not a boolean");
        }
    }
    let p = obj(m.get("params").unwrap(), "params")?;
    match m.get("family").and_then(J::str).unwrap() {
        "ring_classifier" => {
            keys(p, "params.", &["n", "layers", "encoding", "encoding_scale", "readout"],
                 &["n", "layers", "encoding", "encoding_scale", "readout"])?;
            integer(p, "params.", "n", 2.0, 64.0)?;
            integer(p, "params.", "layers", 1.0, 64.0)?;
            enum_str(p, "params.", "encoding", &["ry"])?;
            number(p, "params.", "encoding_scale", Some((0.0, true)), PI)?;
            match p.get("readout") {
                Some(J::Str(s)) if s.len() >= 2 && s.len() <= 3 && s.starts_with('z')
                    && s[1..].chars().all(|c| c.is_ascii_digit()) => Ok(()),
                _ => bad("params.readout", "not z followed by one or two digits"),
            }
        }
        "tfim_chain" => {
            keys(p, "params.", &["n", "J", "h", "dt", "steps", "order", "slots"], &["n", "J", "h", "dt", "steps", "order"])?;
            integer(p, "params.", "n", 2.0, 100.0)?;
            number(p, "params.", "J", Some((-10.0, false)), 10.0)?;
            number(p, "params.", "h", Some((-10.0, false)), 10.0)?;
            number(p, "params.", "dt", Some((0.0, true)), 1.0)?;
            integer(p, "params.", "steps", 1.0, 64.0)?;
            match p.get("order") {
                Some(J::Num(x)) if *x == 1.0 || *x == 2.0 => {}
                _ => return bad("params.order", "not 1 or 2"),
            }
            enum_str(p, "params.", "slots", &["field", "none"])
        }
        _ => {
            keys(p, "params.", &["graph", "n", "angles"], &["graph", "n", "angles"])?;
            enum_str(p, "params.", "graph", &["chain", "ring"])?;
            integer(p, "params.", "n", 2.0, 100.0)?;
            match p.get("angles") {
                Some(J::Str(s)) if s == "slots" => Ok(()),
                Some(J::Arr(a)) => {
                    if a.is_empty() || a.len() > 100 {
                        return bad("params.angles", "not 1 to 100 angles");
                    }
                    for (k, x) in a.iter().enumerate() {
                        match x {
                            J::Num(v) if v.abs() <= 2.0 * PI => {}
                            _ => return bad(&format!("params.angles[{}]", k), "not a number in [-2 pi, 2 pi]"),
                        }
                    }
                    Ok(())
                }
                _ => bad("params.angles", "not \"slots\" or an array of angles"),
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ the checks

fn all_finite(v: &J, path: &str) -> Option<String> {
    match v {
        J::Num(x) if !x.is_finite() => Some(path.to_string()),
        J::Arr(a) => a.iter().enumerate().find_map(|(k, x)| all_finite(x, &format!("{}[{}]", path, k))),
        J::Obj(m) => m.iter().find_map(|(k, x)| all_finite(x, &if path.is_empty() { k.clone() } else { format!("{}.{}", path, k) })),
        _ => None,
    }
}

/// Rzz(theta) on the device: CZ count of psfz step 5 (0 for theta = 0 or pi, 1 for +-pi/2, else 2; TOL as psfz).
pub fn cz_per_rzz(theta: f64) -> usize {
    let r = crate::linalg::wrap(theta).abs();
    if r < compile::TOL || (r - PI).abs() < compile::TOL {
        0
    } else if (r - PI / 2.0).abs() < compile::TOL {
        1
    } else {
        2
    }
}

/// Fold an rzz angle into [0, pi/2] by the IR draft's identities (period, sign, shift by pi).
pub fn fold_rzz(theta: f64) -> f64 {
    let t = theta.rem_euclid(PI);
    t.min(PI - t)
}

fn n_of(p: &J, k: &str) -> f64 {
    p.get(k).and_then(J::num).unwrap()
}

fn fmt(x: f64) -> String {
    // Rust's Display never uses an exponent, so the QASM reader can take it
    format!("{}", x)
}

pub fn tfim_qasm(n: usize, j: f64, h: f64, dt: f64, steps: usize, order: usize) -> String {
    let mut s = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{}] q;\nbit[{}] c;\n", n, n);
    let zz = 2.0 * j * dt;
    for _ in 0..steps {
        if order == 2 {
            for q in 0..n {
                s += &format!("rx({}) q[{}];\n", fmt(h * dt), q);
            }
        }
        for start in [0usize, 1] {
            let mut q = start;
            while q + 1 < n {
                s += &format!("rzz({}) q[{}], q[{}];\n", fmt(zz), q, q + 1);
                q += 2;
            }
        }
        for q in 0..n {
            s += &format!("rx({}) q[{}];\n", fmt(if order == 2 { h * dt } else { 2.0 * h * dt }), q);
        }
    }
    for q in 0..n {
        s += &format!("c[{}] = measure q[{}];\n", q, q);
    }
    s
}

/// The ring classifier's structure (as the template check of 2026-10-19 and TOQB's data re-uploading ring): per
/// layer, ry(encoding) on every qubit, ry and rz on every qubit, CZ around the ring; a final ry on every qubit; the
/// readout qubit measured into clbit 0. The angles are placeholders: in a template every one is a slot.
pub fn ring_qasm(n: usize, layers: usize, readout: usize) -> String {
    let mut s = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{}] q;\nbit[1] c;\n", n);
    for _ in 0..layers {
        for q in 0..n {
            s += &format!("ry(0.1) q[{}];\n", q);
        }
        for q in 0..n {
            s += &format!("ry(0.2) q[{}];\nrz(0.3) q[{}];\n", q, q);
        }
        for q in 0..n {
            s += &format!("cz q[{}], q[{}];\n", q, (q + 1) % n);
        }
    }
    for q in 0..n {
        s += &format!("ry(0.4) q[{}];\n", q);
    }
    s += &format!("c[0] = measure q[{}];\n", readout);
    s
}

fn counts(out: &compile::Output) -> BTreeMap<String, J> {
    let mut m = BTreeMap::new();
    for g in ["cz", "ecr", "sx", "x", "rz"] {
        m.insert(g.to_string(), J::Num(out.ops.iter().filter(|o| o.0 == g).count() as f64));
    }
    m.insert("slots".into(), J::Num(out.slots.len() as f64));
    m.insert("kind".into(), J::Str(out.kind.clone()));
    m.insert("layout".into(), J::Arr(out.layout.iter().map(|&q| J::Num(q as f64)).collect()));
    m
}

/// Gate 1 on the record's text. Never panics on any text: every failure is an outcome.
pub fn check(text: &str, cfg: &Config) -> Outcome {
    let skip = |c: &str| cfg.skip.iter().any(|s| s == c);
    // 0. parse
    let rec = match json::parse_strict(text) {
        Ok(v) => v,
        Err(e) => return refuse("SCHEMA_INVALID", "0", "", "the text is strict JSON (RFC 8259)", e),
    };
    // 1. version
    match rec.get("ir") {
        Some(J::Str(s)) if s == IR => {}
        Some(J::Str(s)) => return refuse("IR_VERSION", "1", "ir", "ir = \"psf-intent/0\"", format!("unknown IR version '{}'", s)),
        _ => return refuse("SCHEMA_INVALID", "1", "ir", "an object whose 'ir' is a string", "no IR version".into()),
    }
    // 2. finite numbers
    if let Some(p) = all_finite(&rec, "") {
        return refuse("NUMBER_NOT_FINITE", "2", &p, "every number is finite", format!("{} is not finite", p));
    }
    // 3. schema
    if let Err((field, why)) = schema(&rec) {
        return refuse("SCHEMA_INVALID", "3", &field, "validates against psf-intent-0.r2.schema.json", why);
    }
    // D. decline
    if let Some(r) = rec.get("decline").and_then(J::str) {
        return Outcome::Declined(r.to_string());
    }
    let family = rec.get("family").and_then(J::str).unwrap().to_string();
    let output = rec.get("output").and_then(J::str).unwrap().to_string();
    let p = rec.get("params").unwrap().clone();
    // 4. device file
    let rel = rec.get("device").and_then(J::str).unwrap()["file:".len()..].to_string();
    let path = cfg.device_dir.join(&rel);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => return refuse("DEVICE_UNREADABLE", "4", "device", "the file is readable",
                                format!("{}: {}", rel, e.kind())),
    };
    let dsha = sha256::hex(&bytes);
    if let Some(want) = rec.get("device_sha256").and_then(J::str) {
        if want != dsha {
            return refuse("DEVICE_HASH_MISMATCH", "4", "device_sha256", "SHA-256(file) = device_sha256",
                          format!("the file's SHA-256 is {}", dsha));
        }
    }
    let dev_json = match std::str::from_utf8(&bytes).map_err(|e| e.to_string()).and_then(|s| json::parse(s)) {
        Ok(j) => j,
        Err(e) => return refuse("DEVICE_UNREADABLE", "4", "device", "the file parses as a psfz device", e),
    };
    let d = match Device::from_json(&dev_json) {
        Ok(d) => d,
        Err(e) => return refuse("DEVICE_UNREADABLE", "4", "device", "the file parses as a psfz device", e),
    };
    let l = match Lattice::read(&d) {
        Ok(l) => l,
        Err(e) => return refuse("DEVICE_UNREADABLE", "4", "device", "psfz reads the device as a heavy-hex lattice", e),
    };
    // 5. device gate set
    if output == "template" && d.gate2 == "ecr" {
        return refuse("TEMPLATE_ECR_UNSUPPORTED", "5", "output", "two-qubit gate = cz when output = template",
                      format!("{} offers ecr; r5 templates need cz", d.name));
    }
    let mut derived = BTreeMap::new();
    let (circ, template) = match family.as_str() {
        "ring_classifier" => {
            let n = n_of(&p, "n") as usize;
            let layers = n_of(&p, "layers") as usize;
            let k: usize = p.get("readout").and_then(J::str).unwrap()[1..].parse().unwrap();
            if !skip("R1") && k >= n {
                return refuse("READOUT_RANGE", "R1", "params.readout", "readout = z<k> with 0 <= k < n",
                              format!("z{} on a {}-qubit ring", k, n));
            }
            if !skip("R2") {
                if n % 2 == 1 {
                    return refuse("RING_ODD", "R2", "params.n", "n even", format!("n = {} (every heavy-hex cycle is even)", n));
                }
                if n < 12 {
                    return refuse("RING_TOO_SHORT", "R2", "params.n", "n >= 12", format!("n = {} (heavy-hex has no cycle shorter than 12)", n));
                }
            }
            if !skip("R3") && place::ring(&d, &l, n).is_none() {
                return refuse("NO_WORKING_PLACEMENT", "R3", "params.n", &format!("ring(D, {}) != none", n),
                              format!("no ring of {} working qubits and couplers in psfz's fixed candidate order", n));
            }
            let (cz, sx) = ((n * layers) as f64, (n * (2 * layers + 2)) as f64);
            if !skip("R4") {
                if cz > cfg.cz_max {
                    return refuse("RESOURCE_LIMIT", "R4", "params.layers", &format!("CZ = n L = {} <= CZ_max = {}", cz, cfg.cz_max),
                                  format!("CZ {} > {}", cz, cfg.cz_max));
                }
                if sx > cfg.sx_max {
                    return refuse("RESOURCE_LIMIT", "R4", "params.layers", &format!("sx = n (2 L + 2) = {} <= SX_max = {}", sx, cfg.sx_max),
                                  format!("sx {} > {}", sx, cfg.sx_max));
                }
            }
            if !skip("R5") && output != "template" {
                return refuse("OUTPUT_KIND", "R5", "output", "output = template",
                              "a ring classifier's trainable angles come at bind time, not from the record".into());
            }
            derived.insert("cz_formula".to_string(), J::Num(cz));
            derived.insert("sx_formula".to_string(), J::Num(sx));
            (ring_qasm(n, layers, k), true)
        }
        "tfim_chain" => {
            let n = n_of(&p, "n") as usize;
            let (jj, hh, dt) = (n_of(&p, "J"), n_of(&p, "h"), n_of(&p, "dt"));
            let steps = n_of(&p, "steps") as usize;
            let order = n_of(&p, "order") as usize;
            if !skip("T1") && place::chain(&d, &l, n).is_none() {
                return refuse("CHAIN_TOO_LONG", "T1", "params.n", &format!("chain(D, {}) != none (n <= lambda_chain(D))", n),
                              format!("no working row-and-bridge path of {} qubits on {}", n, d.name));
            }
            let c = cz_per_rzz(2.0 * jj * dt);
            let cz = ((n - 1) * steps * c) as f64;
            if !skip("T2") && cz > cfg.cz_max {
                return refuse("RESOURCE_LIMIT", "T2", "params.steps",
                              &format!("CZ = (n - 1) k steps c(2 J dt) = {} <= CZ_max = {}", cz, cfg.cz_max),
                              format!("CZ {} > {}", cz, cfg.cz_max));
            }
            let twin = matches!(rec.get("twin").and_then(|t| t.get("requested")), Some(J::Bool(true)));
            let per_edge = if c > 0 { steps } else { 0 };
            let bound = (0..n - 1).map(|i| (i + 1).min(n - i - 1).min(per_edge)).max().unwrap_or(0);
            if twin && !skip("T3") && 2f64.powi(bound as i32) > cfg.chi_max {
                return refuse("TWIN_BOND_LIMIT", "T3", "twin.requested",
                              &format!("max_i min(2^(i+1), 2^(n-i-1), 2^(k steps)) = 2^{} <= chi_max = {}", bound, cfg.chi_max),
                              format!("bond bound 2^{} = {} > {}", bound, 2f64.powi(bound as i32), cfg.chi_max));
            }
            derived.insert("cz_formula".to_string(), J::Num(cz));
            derived.insert("twin_bound_log2".to_string(), J::Num(bound as f64));
            let field_slots = p.get("slots").and_then(J::str).unwrap_or("field") == "field";
            (tfim_qasm(n, jj, hh, dt, steps, order), output == "template" && field_slots)
        }
        _ => {
            let graph = p.get("graph").and_then(J::str).unwrap().to_string();
            let n = n_of(&p, "n") as usize;
            if !skip("Z1") && graph == "ring" {
                if n % 2 == 1 {
                    return refuse("RING_ODD", "Z1", "params.n", "n even", format!("n = {}", n));
                }
                if n < 12 {
                    return refuse("RING_TOO_SHORT", "Z1", "params.n", "n >= 12", format!("n = {}", n));
                }
            }
            let angles = p.get("angles").and_then(J::arr).cloned();
            let need = if graph == "ring" { n } else { n - 1 };
            if let Some(a) = &angles {
                if !skip("Z2") && a.len() != need {
                    return refuse("PARAM_COUNT", "Z2", "params.angles", &format!("len(angles) = {}", need),
                                  format!("{} angles for a {} of {} qubits", a.len(), graph, n));
                }
            }
            let placed = if graph == "ring" { place::ring(&d, &l, n) } else { place::chain(&d, &l, n) };
            let placed = match placed {
                Some(v) => v,
                None if skip("Z3") => Vec::new(),
                None => {
                    return if graph == "ring" {
                        refuse("NO_WORKING_PLACEMENT", "Z3", "params.n", &format!("ring(D, {}) != none", n), "no working ring".into())
                    } else {
                        refuse("CHAIN_TOO_LONG", "Z3", "params.n", &format!("chain(D, {}) != none", n), "no working chain".into())
                    };
                }
            };
            let native = dev_json.get("rzz_native");
            if !skip("Z4") {
                let listed: BTreeSet<(usize, usize)> = native.and_then(|r| r.get("couplers")).and_then(J::arr)
                    .map(|a| a.iter().filter_map(|c| {
                        let c = c.arr()?;
                        let (x, y) = (c.first()?.num()? as usize, c.get(1)?.num()? as usize);
                        Some((x.min(y), x.max(y)))
                    }).collect()).unwrap_or_default();
                let edges: Vec<(usize, usize)> = (0..placed.len().saturating_sub(1)).map(|k| (placed[k], placed[k + 1]))
                    .chain(if graph == "ring" && !placed.is_empty() { Some((placed[placed.len() - 1], placed[0])) } else { None })
                    .map(|(x, y)| (x.min(y), x.max(y))).collect();
                if native.is_none() || !edges.iter().all(|e| listed.contains(e)) {
                    return refuse("DEVICE_NO_RZZ", "Z4", "device", "the device file lists an rzz native on every placed coupler",
                                  format!("{} lists no rzz native for this placement", d.name));
                }
            }
            if let Some(a) = &angles {
                if !skip("Z5") {
                    let lo = native.and_then(|r| r.get("angle_min_exclusive")).and_then(J::num).unwrap_or(0.0);
                    let hi = native.and_then(|r| r.get("angle_max")).and_then(J::num).unwrap_or(-1.0);
                    for (k, x) in a.iter().enumerate() {
                        let f = fold_rzz(x.num().unwrap());
                        if !(f > lo && f <= hi) {
                            return refuse("RZZ_ANGLE_RANGE", "Z5", &format!("params.angles[{}]", k),
                                          &format!("{} < fold(theta) <= {}", lo, hi),
                                          format!("{} folds to {}, outside ({}, {}]", x.num().unwrap(), f, lo, hi));
                        }
                    }
                }
            } else if !skip("Z6") {
                return refuse("TEMPLATE_2Q_SLOT_UNSUPPORTED", "Z6", "params.angles", "angles is an array",
                              "two-qubit angles cannot be template slots in r5".into());
            }
            return refuse("BUILD_REFUSED", "9", "family", "psfz builds the record's circuit",
                          "psfz has no fractional-gate natives yet: an rzz layer cannot be built (IR draft §3 note)".into());
        }
    };
    // 9. psfz build
    let c = match qasm::read(&circ) {
        Ok(c) => c,
        Err(e) => return refuse("BUILD_REFUSED", "9", "family", "psfz builds the record's circuit", format!("internal circuit text: {}", e)),
    };
    match compile::compile(&c, &d, &l, template) {
        Err(why) => refuse("BUILD_REFUSED", "9", "family", "psfz builds the record's circuit", why),
        Ok(out) => {
            let mut res = BTreeMap::new();
            res.insert("family".to_string(), J::Str(family));
            res.insert("output".to_string(), J::Str(output));
            let mut dv = BTreeMap::new();
            dv.insert("file".to_string(), J::Str(rel));
            dv.insert("sha256".to_string(), J::Str(dsha));
            dv.insert("name".to_string(), J::Str(d.name.clone()));
            dv.insert("two_qubit_gate".to_string(), J::Str(d.gate2.clone()));
            res.insert("device".to_string(), J::Obj(dv));
            let built = counts(&out);
            let cz_built = built.get("cz").and_then(J::num).unwrap() + built.get("ecr").and_then(J::num).unwrap();
            let matches = derived.get("cz_formula").and_then(J::num).map(|f| f == cz_built).unwrap_or(true)
                && derived.get("sx_formula").and_then(J::num).map_or(true, |f| Some(f) == built.get("sx").and_then(J::num));
            derived.insert("formula_matches_build".to_string(), J::Bool(matches));
            res.insert("derived".to_string(), J::Obj(derived));
            res.insert("build".to_string(), J::Obj(built));
            Outcome::Passed(res)
        }
    }
}

/// `psfz gate RECORD.json CONFIG.json OUT.json`: exit 0 passed or declined, 2 refused, 1 input error. OUT must not
/// exist (a result is never overwritten). CONFIG: {"device_dir", "CZ_max", "SX_max", "chi_max"}; device_dir is
/// relative to the config file.
pub fn run(args: &[String]) -> Result<i32, String> {
    let out = Path::new(&args[3]);
    if out.exists() {
        return Err(format!("{} exists: a result is never overwritten", args[3]));
    }
    let text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {}", args[1], e))?;
    let cfg_path = Path::new(&args[2]);
    let cfg_j = json::parse(&std::fs::read_to_string(cfg_path).map_err(|e| format!("{}: {}", args[2], e))?)?;
    let cfg = Config::from_json(&cfg_j, cfg_path.parent().unwrap_or(Path::new(".")))?;
    let res = check(&text, &cfg);
    std::fs::write(out, json::write(&res.to_json(&sha256::hex(text.as_bytes()))))
        .map_err(|e| format!("{}: {}", args[3], e))?;
    let (code, chk) = res.code_check();
    eprintln!("psfz gate: {} ({})", code, chk);
    Ok(if matches!(res, Outcome::Refused { .. }) { 2 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n2_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/2026-10-20/n2")
    }

    fn cfg(n2: &J) -> Config {
        let c = n2.get("config_test_only").unwrap();
        Config { device_dir: n2_dir().join("fixtures"), cz_max: c.get("CZ_max").unwrap().num().unwrap(),
                 sx_max: c.get("SX_max").unwrap().num().unwrap(), chi_max: c.get("chi_max").unwrap().num().unwrap(),
                 skip: Vec::new() }
    }

    fn load_n2() -> J {
        let path = n2_dir().join("n2.json");
        let text = std::fs::read_to_string(&path).unwrap();
        // N2 is fixed: its SHA-256 was recorded on 2026-10-20 at 06:35 CEST
        assert_eq!(sha256::hex(text.as_bytes()), "458e53df0d3df9f869a1beb4958994ecf802ee30b7719104322c30f991d781ef");
        json::parse(&text).unwrap()
    }

    #[test]
    fn n2_milestone_m0() {
        let n2 = load_n2();
        let c = cfg(&n2);
        let mut lines = Vec::new();
        let mut fail = 0;
        for r in n2.get("refusals").unwrap().arr().unwrap() {
            let id = r.get("id").unwrap().str().unwrap();
            let got = check(r.get("text").unwrap().str().unwrap(), &c).code_check();
            let e = r.get("expected").unwrap();
            let want = (e.get("code").unwrap().str().unwrap().to_string(), e.get("check").unwrap().str().unwrap().to_string());
            if got != want {
                fail += 1;
            }
            lines.push(format!("{} {:?} want {:?}", id, got, want));
        }
        // R19: fault injection, R10's record with check R3 disabled
        let r10 = n2.get("refusals").unwrap().arr().unwrap().iter().find(|r| r.get("id").unwrap().str() == Some("R10")).unwrap();
        let mut c19 = c.clone();
        c19.skip = vec!["R3".into()];
        let got = check(r10.get("text").unwrap().str().unwrap(), &c19).code_check();
        if got != ("BUILD_REFUSED".to_string(), "9".to_string()) {
            fail += 1;
        }
        lines.push(format!("R19 {:?}", got));
        for k in n2.get("controls").unwrap().arr().unwrap() {
            let id = k.get("id").unwrap().str().unwrap();
            let out = check(k.get("text").unwrap().str().unwrap(), &c);
            let want = k.get("expected").unwrap().str().unwrap();
            let ok = match (&out, want) {
                (Outcome::Passed(m), "pass") => {
                    m.get("derived").and_then(|d| d.get("formula_matches_build")) == Some(&J::Bool(true))
                }
                (Outcome::Declined(_), "declined") => true,
                _ => false,
            };
            if !ok {
                fail += 1;
            }
            let b = match &out {
                Outcome::Passed(m) => json::write(m.get("build").unwrap()),
                _ => String::new(),
            };
            lines.push(format!("{} {:?} want {} {}", id, out.code_check(), want, b));
        }
        println!("{}", lines.join("\n"));
        assert_eq!(fail, 0, "\n{}", lines.join("\n"));
    }

    #[test]
    fn controls_give_the_template_check_counts() {
        let n2 = load_n2();
        let c = cfg(&n2);
        let ctl = |id: &str| n2.get("controls").unwrap().arr().unwrap().iter()
            .find(|k| k.get("id").unwrap().str() == Some(id)).unwrap().get("text").unwrap().str().unwrap().to_string();
        for (id, cz, sx) in [("K1", 96.0, 216.0), ("K2", 192.0, 408.0)] {
            match check(&ctl(id), &c) {
                Outcome::Passed(m) => {
                    let b = m.get("build").unwrap();
                    assert_eq!(b.get("cz").unwrap().num(), Some(cz), "{}", id);
                    assert_eq!(b.get("sx").unwrap().num(), Some(sx), "{}", id);
                }
                o => panic!("{} {:?}", id, o),
            }
        }
    }

    #[test]
    fn folds_and_counts() {
        assert!((fold_rzz(1.0) - 1.0).abs() < 1e-15);
        assert!((fold_rzz(-0.3) - 0.3).abs() < 1e-15);
        assert!((fold_rzz(PI - 0.2) - 0.2).abs() < 1e-12);
        assert!((fold_rzz(2.0 * PI + 0.1) - 0.1).abs() < 1e-12);
        assert_eq!(cz_per_rzz(0.0), 0);
        assert_eq!(cz_per_rzz(PI), 0);
        assert_eq!(cz_per_rzz(-PI / 2.0), 1);
        assert_eq!(cz_per_rzz(0.2), 2);
    }

    #[test]
    fn never_panics_on_odd_text() {
        let n2 = load_n2();
        let c = cfg(&n2);
        for t in ["", "[]", "null", "{}", r#"{"ir": 1}"#, r#"{"ir": "psf-intent/0"}"#, "\u{feff}{}", "{\"ir\":\"psf-intent/0\",",
                  r#"{"ir":"psf-intent/0","family":"tfim_chain","params":[],"output":"template","device":"file:F1.json"}"#] {
            let _ = check(t, &c);
        }
    }
}
