//! psfz -- PSF-Zero in Rust, prototype r4 (2026-10-18): circuits whose two-qubit interaction graph is a chain or a
//! ring, compiled for an IBM heavy-hex device (CZ: Heron; ECR: Eagle) by formula only. OpenQASM 3 in and out; the
//! device from a JSON file. No search, no estimate of noise, no simulation, no check at run time: placement by
//! coordinates, order by commutation levels, every rewrite a closed form tested once in the unit tests.
//!
//!     psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]
//!     psfz template DEVICE.json IN.qasm OUT.json          (r5 prototype: every single-qubit gate with parameters is a
//!                                                          slot; the gate list never depends on the values)
//!     psfz bind TEMPLATE.json VALUES.json OUT.qasm        (VALUES: one list of parameters per slot, in input order)
//!     psfz gate RECORD.json CONFIG.json OUT.json          (P0, 2026-10-20: gate 1 on an IR v0 record, IR r2; exit 0
//!                                                          passed or declined, 2 refused with one code)
//!     psfz twin DEVICE.json IN.qasm OUT.json [CHI_MAX]    (prototype: compile, then simulate the chain output as an
//!                                                          MPS, noiseless; <Z> per classical bit; refused before it
//!                                                          starts if the formula bound exceeds CHI_MAX, default 4096)
//!
//! Exit 0: compiled. Exit 2: refused (the reason on stderr and in the report). Exit 1: an input error. `compile` is
//! r4's, unchanged (its report says r4: release 2026-10-18.1 uses psfz's output only for that version).

mod compile;
mod device;
mod gate;
mod json;
mod kak;
mod linalg;
mod mps;
mod place;
mod qasm;
mod sha256;
mod template;

use json::J;
use std::collections::BTreeMap;
use std::time::Instant;

const USAGE: &str = "usage: psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json] \
| template DEVICE.json IN.qasm OUT.json \
| bind TEMPLATE.json VALUES.json OUT.qasm \
| gate RECORD.json CONFIG.json OUT.json \
| twin DEVICE.json IN.qasm OUT.json [CHI_MAX]";

#[inline]
fn read_file(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))
}

#[inline]
fn write_file(path: &str, content: impl AsRef<[u8]>) -> Result<(), String> {
    std::fs::write(path, content).map_err(|e| format!("{}: {}", path, e))
}

fn refuse_with_report(out_path: Option<&str>, mut rep: BTreeMap<String, J>, why: String) -> Result<i32, String> {
    eprintln!("psfz: refused: {}", why);
    rep.insert("ok".into(), J::Bool(false));
    rep.insert("reason".into(), J::Str(why));
    if let Some(p) = out_path {
        write_file(p, json::write(&J::Obj(rep)))?;
    }
    Ok(2)
}

fn run_template(args: &[String]) -> Result<i32, String> {
    let dev_text = read_file(&args[1])?;
    let src = read_file(&args[2])?;
    let d = device::Device::from_json(&json::parse(&dev_text)?)?;
    let q = qasm::read(&src)?;
    let t0 = Instant::now();
    let res = match device::Lattice::read(&d) {
        Ok(l) => compile::compile(&q, &d, &l, true),
        Err(e) => Err(format!("the device is not read as a heavy-hex lattice: {}", e)),
    };
    match res {
        Ok(out) => {
            let mut j = template::to_json(&out, &d.name);
            if let J::Obj(m) = &mut j {
                m.insert("build_s".into(), J::Num(t0.elapsed().as_secs_f64()));
            }
            write_file(&args[3], json::write(&j))?;
            Ok(0)
        }
        Err(why) => {
            eprintln!("psfz: refused: {}", why);
            Ok(2)
        }
    }
}

fn run_bind(args: &[String]) -> Result<i32, String> {
    let t = json::parse(&read_file(&args[1])?)?;
    let v = json::parse(&read_file(&args[2])?)?;
    let raw_slots = v.arr().ok_or_else(|| "values: not an array".to_string())?;
    let mut values: Vec<Vec<f64>> = Vec::with_capacity(raw_slots.len());
    for (i, slot) in raw_slots.iter().enumerate() {
        let arr = slot.arr().ok_or_else(|| format!("values[{}]: not an array", i))?;
        let mut row = Vec::with_capacity(arr.len());
        for (j, x) in arr.iter().enumerate() {
            let num = x.num().ok_or_else(|| format!("values[{}][{}]: not a number", i, j))?;
            if !num.is_finite() {
                return Err(format!("values[{}][{}]: non-finite number", i, j));
            }
            row.push(num);
        }
        values.push(row);
    }

    let t0 = Instant::now();
    let ops = template::bind(&t, &values)?;
    let dt = t0.elapsed().as_secs_f64();

    let mut meas: Vec<(usize, usize)> = Vec::new();
    if let Some(m_arr) = t.get("measure").and_then(|m| m.arr()) {
        meas.reserve(m_arr.len());
        for (i, pair) in m_arr.iter().enumerate() {
            let p = pair.arr().ok_or_else(|| format!("template.measure[{}]: not an array", i))?;
            if p.len() < 2 {
                return Err(format!("template.measure[{}]: expected [qubit, clbit]", i));
            }
            let q = p[0].num().ok_or_else(|| format!("template.measure[{}][0]: not a number", i))?;
            let c = p[1].num().ok_or_else(|| format!("template.measure[{}][1]: not a number", i))?;
            meas.push((q as usize, c as usize));
        }
    }

    let nc = t.get("num_clbits").and_then(|x| x.num()).unwrap_or(0.0) as usize;
    write_file(&args[3], qasm::write(nc, &ops, &meas, &format!("{}: bound", template::VERSION)))?;
    eprintln!("bind_s {:.6}", dt);
    Ok(0)
}

fn run_twin(args: &[String]) -> Result<i32, String> {
    let dev_text = read_file(&args[1])?;
    let src = read_file(&args[2])?;
    let chi_max: usize = match args.get(4) {
        Some(s) => s.parse().map_err(|_| format!("invalid CHI_MAX: {}", s))?,
        None => 4096,
    };
    let d = device::Device::from_json(&json::parse(&dev_text)?)?;
    let q = qasm::read(&src)?;
    let t0 = Instant::now();
    let l = match device::Lattice::read(&d) {
        Ok(lat) => lat,
        Err(e) => {
            return refuse_with_report(
                Some(&args[3]),
                BTreeMap::new(),
                format!("the device is not read as a heavy-hex lattice: {}", e),
            );
        }
    };
    let out = match compile::compile(&q, &d, &l, false) {
        Ok(o) => o,
        Err(why) => return refuse_with_report(Some(&args[3]), BTreeMap::new(), why),
    };
    let t1 = Instant::now();

    let walk = match mps::chain_order(&out.ops) {
        Ok(w) => w,
        Err(why) => return refuse_with_report(Some(&args[3]), BTreeMap::new(), why),
    };
    let site: BTreeMap<usize, usize> = walk.iter().enumerate().map(|(i, &qb)| (qb, i)).collect();
    let n = walk.len();
    let plan = match mps::plan_ops(&out.ops, &site) {
        Ok(p) => p,
        Err(why) => return refuse_with_report(Some(&args[3]), BTreeMap::new(), why),
    };
    let bound_cz = mps::bound_log2(&out.ops, &site, n);
    let bound = mps::bound_log2_blocks(&plan, n);
    let bmax = bound.iter().copied().max().unwrap_or(0);

    let mut rep = BTreeMap::new();
    rep.insert(
        "version".into(),
        J::Str("psfz twin prototype (2026-11-13), MPS, noiseless, one split per CZ-rotation-CZ block".into()),
    );
    rep.insert("svd".into(), J::Str(mps::svd_backend().into()));
    rep.insert("sites".into(), J::Num(n as f64));
    rep.insert("bound_log2".into(), J::Arr(bound.iter().map(|&b| J::Num(b as f64)).collect()));
    rep.insert("bound_log2_cz_count".into(), J::Arr(bound_cz.iter().map(|&b| J::Num(b as f64)).collect()));
    rep.insert(
        "two_site_gates".into(),
        J::Num(plan.iter().filter(|o| matches!(o, mps::TwinOp::Two(..))).count() as f64),
    );
    rep.insert("compile_s".into(), J::Num((t1 - t0).as_secs_f64()));

    let exceeds_chi = u32::try_from(bmax)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
        .map_or(true, |req| req > chi_max);
    if bmax >= 63 || exceeds_chi {
        let why = format!("the formula bound 2^{} exceeds CHI_MAX {}", bmax, chi_max);
        return refuse_with_report(Some(&args[3]), rep, why);
    }

    let mut st = mps::Mps::zero(n);
    for op in &plan {
        match op {
            mps::TwinOp::One(s_, m) => st.apply1(*s_, m),
            mps::TwinOp::Two(s_, w, _) => st.apply2(*s_, w),
        }
    }
    let z = st.z_all();
    let t2 = Instant::now();
    let mut zc = vec![f64::NAN; out.nc];
    for &(qb, c) in &out.meas {
        if let (Some(&s_), Some(slot)) = (site.get(&qb), zc.get_mut(c)) {
            *slot = z[s_];
        }
    }
    rep.insert("ok".into(), J::Bool(true));
    rep.insert("simulate_s".into(), J::Num((t2 - t1).as_secs_f64()));
    rep.insert("max_chi".into(), J::Num(st.max_chi as f64));
    rep.insert("chi".into(), J::Arr(st.chi.iter().map(|&c| J::Num(c as f64)).collect()));
    rep.insert("dropped_weight".into(), J::Num(st.dropped));
    rep.insert("z_clbits".into(), J::Arr(zc.iter().map(|&v| J::Num(v)).collect()));
    write_file(&args[3], json::write(&J::Obj(rep)))?;
    Ok(0)
}

fn run_compile(args: &[String]) -> Result<i32, String> {
    let dev_text = read_file(&args[1])?;
    let src = read_file(&args[2])?;
    let t0 = Instant::now();
    let d = device::Device::from_json(&json::parse(&dev_text)?)?;
    let lat = device::Lattice::read(&d);
    let t1 = Instant::now();
    let c = qasm::read(&src)?;
    let res = match &lat {
        Ok(l) => compile::compile(&c, &d, l, false),
        Err(e) => Err(format!("the device is not read as a heavy-hex lattice: {}", e)),
    };
    let t2 = Instant::now();

    let mut rep = BTreeMap::new();
    rep.insert("version".into(), J::Str("psfz r4 (2026-10-18)".into()));
    rep.insert("device".into(), J::Str(d.name.clone()));
    rep.insert("device_s".into(), J::Num((t1 - t0).as_secs_f64()));
    rep.insert("compile_s".into(), J::Num((t2 - t1).as_secs_f64()));

    let code = match res {
        Ok(out) => {
            let text = qasm::write(
                out.nc,
                &out.ops,
                &out.meas,
                &format!("psfz r4: {} on {}, layout {:?}", out.kind, d.name, out.layout),
            );
            write_file(&args[3], text)?;
            rep.insert("ok".into(), J::Bool(true));
            rep.insert("kind".into(), J::Str(out.kind.clone()));
            rep.insert("layout".into(), J::Arr(out.layout.iter().map(|&q| J::Num(q as f64)).collect()));
            rep.insert("num_clbits".into(), J::Num(out.nc as f64));
            rep.insert(
                "ops".into(),
                J::Arr(
                    out.ops
                        .iter()
                        .map(|(n, q, p)| {
                            J::Arr(vec![
                                J::Str(n.clone()),
                                J::Arr(q.iter().map(|&x| J::Num(x as f64)).collect()),
                                J::Arr(p.iter().map(|&x| J::Num(x)).collect()),
                            ])
                        })
                        .collect(),
                ),
            );
            rep.insert(
                "measure".into(),
                J::Arr(
                    out.meas
                        .iter()
                        .map(|&(q, c)| J::Arr(vec![J::Num(q as f64), J::Num(c as f64)]))
                        .collect(),
                ),
            );
            0
        }
        Err(why) => {
            eprintln!("psfz: refused: {}", why);
            rep.insert("ok".into(), J::Bool(false));
            rep.insert("reason".into(), J::Str(why));
            2
        }
    };
    if let Some(p) = args.get(4) {
        write_file(p, json::write(&J::Obj(rep)))?;
    }
    Ok(code)
}

fn run(args: &[String]) -> Result<i32, String> {
    match (args.first().map(String::as_str), args.len()) {
        (Some("gate"), 4) => gate::run(args),
        (Some("twin"), 4 | 5) => run_twin(args),
        (Some("template"), 4) => run_template(args),
        (Some("bind"), 4) => run_bind(args),
        (Some("compile"), 4 | 5) => run_compile(args),
        _ => Err(USAGE.into()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("psfz: {}", e);
            std::process::exit(1);
        }
    }
}

