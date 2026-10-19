//! psfz -- PSF-Zero in Rust, prototype r4 (2026-10-18): circuits whose two-qubit interaction graph is a chain or a
//! ring, compiled for an IBM heavy-hex device (CZ: Heron; ECR: Eagle) by formula only. OpenQASM 3 in and out; the
//! device from a JSON file. No search, no estimate of noise, no simulation, no check at run time: placement by
//! coordinates, order by commutation levels, every rewrite a closed form tested once in the unit tests.
//!
//!     psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]
//!     psfz template DEVICE.json IN.qasm OUT.json          (r5 prototype: every single-qubit gate with parameters is a
//!                                                          slot; the gate list never depends on the values)
//!     psfz bind TEMPLATE.json VALUES.json OUT.qasm        (VALUES: one list of parameters per slot, in input order)
//!     psfz twin DEVICE.json IN.qasm OUT.json [CHI_MAX]     (prototype: compile, then simulate the chain output as an
//!                                                          MPS, noiseless; <Z> per classical bit; refused before it
//!                                                          starts if the formula bound exceeds CHI_MAX, default 4096)
//!
//! Exit 0: compiled. Exit 2: refused (the reason on stderr and in the report). Exit 1: an input error. `compile` is
//! r4's, unchanged (its report says r4: release 2026-10-18.1 uses psfz's output only for that version).

mod compile;
mod device;
mod json;
mod kak;
mod linalg;
mod mps;
mod place;
mod qasm;
mod template;

use json::J;
use std::collections::BTreeMap;
use std::time::Instant;

fn run_template(args: &[String]) -> Result<i32, String> {
    let dev_text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {}", args[1], e))?;
    let src = std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {}", args[2], e))?;
    let d = device::Device::from_json(&json::parse(&dev_text)?)?;
    let t0 = Instant::now();
    let res = match device::Lattice::read(&d) {
        Ok(l) => compile::compile(&qasm::read(&src)?, &d, &l, true),
        Err(e) => Err(format!("the device is not read as a heavy-hex lattice: {}", e)),
    };
    match res {
        Ok(out) => {
            let mut j = template::to_json(&out, &d.name);
            if let J::Obj(m) = &mut j {
                m.insert("build_s".into(), J::Num(t0.elapsed().as_secs_f64()));
            }
            std::fs::write(&args[3], json::write(&j)).map_err(|e| format!("{}: {}", args[3], e))?;
            Ok(0)
        }
        Err(why) => {
            eprintln!("psfz: refused: {}", why);
            Ok(2)
        }
    }
}

fn run_bind(args: &[String]) -> Result<i32, String> {
    let t = json::parse(&std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {}", args[1], e))?)?;
    let v = json::parse(&std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {}", args[2], e))?)?;
    let values: Vec<Vec<f64>> = v.arr().ok_or("values: not an array")?.iter()
        .map(|s| s.arr().map(|a| a.iter().map(|x| x.num().unwrap_or(f64::NAN)).collect()).unwrap_or_default())
        .collect();
    let t0 = Instant::now();
    let ops = template::bind(&t, &values)?;
    let dt = t0.elapsed().as_secs_f64();
    let meas: Vec<(usize, usize)> = t.get("measure").and_then(|m| m.arr()).map(|a| a.iter().map(|p| {
        let p = p.arr().unwrap();
        (p[0].num().unwrap() as usize, p[1].num().unwrap() as usize)
    }).collect()).unwrap_or_default();
    let nc = t.get("num_clbits").and_then(|x| x.num()).unwrap_or(0.0) as usize;
    std::fs::write(&args[3], qasm::write(nc, &ops, &meas, &format!("{}: bound", template::VERSION)))
        .map_err(|e| format!("{}: {}", args[3], e))?;
    eprintln!("bind_s {:.6}", dt);
    Ok(0)
}

fn run_twin(args: &[String]) -> Result<i32, String> {
    use std::collections::BTreeMap as Map;
    let dev_text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {}", args[1], e))?;
    let src = std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {}", args[2], e))?;
    let chi_max: usize = args.get(4).map(|x| x.parse().unwrap_or(4096)).unwrap_or(4096);
    let d = device::Device::from_json(&json::parse(&dev_text)?)?;
    let l = device::Lattice::read(&d).map_err(|e| format!("the device is not read as a heavy-hex lattice: {}", e))?;
    let t0 = Instant::now();
    let out = match compile::compile(&qasm::read(&src)?, &d, &l, false) {
        Ok(o) => o,
        Err(why) => {
            eprintln!("psfz: refused: {}", why);
            let mut rep = Map::new();
            rep.insert("ok".into(), J::Bool(false));
            rep.insert("reason".into(), J::Str(why));
            std::fs::write(&args[3], json::write(&J::Obj(rep))).map_err(|e| format!("{}: {}", args[3], e))?;
            return Ok(2);
        }
    };
    let t1 = Instant::now();
    let walk = mps::chain_order(&out.ops)?;
    let site: Map<usize, usize> = walk.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let n = walk.len();
    let bound = mps::bound_log2(&out.ops, &site, n);
    let bmax = bound.iter().copied().max().unwrap_or(0);
    let mut rep = Map::new();
    rep.insert("version".into(), J::Str("psfz twin prototype (2026-10-19), MPS, noiseless".into()));
    rep.insert("sites".into(), J::Num(n as f64));
    rep.insert("bound_log2".into(), J::Arr(bound.iter().map(|&b| J::Num(b as f64)).collect()));
    rep.insert("compile_s".into(), J::Num((t1 - t0).as_secs_f64()));
    if bmax >= 63 || (1usize << bmax) > chi_max {
        rep.insert("ok".into(), J::Bool(false));
        rep.insert("reason".into(), J::Str(format!("the formula bound 2^{} exceeds CHI_MAX {}", bmax, chi_max)));
        std::fs::write(&args[3], json::write(&J::Obj(rep))).map_err(|e| format!("{}: {}", args[3], e))?;
        eprintln!("psfz: refused: bond bound 2^{} > {}", bmax, chi_max);
        return Ok(2);
    }
    let mut st = mps::Mps::zero(n);
    for (g, q, p) in &out.ops {
        if g == "cz" {
            let (a, b) = (site[&q[0]], site[&q[1]]);
            if a.max(b) - a.min(b) != 1 {
                return Err("twin: a CZ between sites that are not neighbours".into());
            }
            st.cz(a.min(b));
        } else {
            let m = mps::gate1(g, p).ok_or(format!("twin: gate {} is not read", g))?;
            st.apply1(site[&q[0]], &m);
        }
    }
    let z = st.z_all();
    let t2 = Instant::now();
    let mut zc = vec![f64::NAN; out.nc];
    for &(q, c) in &out.meas {
        if let Some(&s_) = site.get(&q) {
            zc[c] = z[s_];
        }
    }
    rep.insert("ok".into(), J::Bool(true));
    rep.insert("simulate_s".into(), J::Num((t2 - t1).as_secs_f64()));
    rep.insert("max_chi".into(), J::Num(st.max_chi as f64));
    rep.insert("chi".into(), J::Arr(st.chi.iter().map(|&c| J::Num(c as f64)).collect()));
    rep.insert("dropped_weight".into(), J::Num(st.dropped));
    rep.insert("z_clbits".into(), J::Arr(zc.iter().map(|&v| J::Num(v)).collect()));
    std::fs::write(&args[3], json::write(&J::Obj(rep))).map_err(|e| format!("{}: {}", args[3], e))?;
    Ok(0)
}

fn run(args: &[String]) -> Result<i32, String> {
    if (args.len() == 4 || args.len() == 5) && args[0] == "twin" {
        return run_twin(args);
    }
    if args.len() == 4 && args[0] == "template" {
        return run_template(args);
    }
    if args.len() == 4 && args[0] == "bind" {
        return run_bind(args);
    }
    if args.len() < 4 || args[0] != "compile" {
        return Err("usage: psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json] | template DEVICE.json IN.qasm \
                    OUT.json | bind TEMPLATE.json VALUES.json OUT.qasm".into());
    }
    let dev_text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {}", args[1], e))?;
    let src = std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {}", args[2], e))?;
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
            let text = qasm::write(out.nc, &out.ops, &out.meas,
                                   &format!("psfz r4: {} on {}, layout {:?}", out.kind, d.name, out.layout));
            std::fs::write(&args[3], text).map_err(|e| format!("{}: {}", args[3], e))?;
            rep.insert("ok".into(), J::Bool(true));
            rep.insert("kind".into(), J::Str(out.kind.clone()));
            rep.insert("layout".into(), J::Arr(out.layout.iter().map(|&q| J::Num(q as f64)).collect()));
            rep.insert("num_clbits".into(), J::Num(out.nc as f64));
            rep.insert(
                "ops".into(),
                J::Arr(out.ops.iter().map(|(n, q, p)| J::Arr(vec![
                    J::Str(n.clone()),
                    J::Arr(q.iter().map(|&x| J::Num(x as f64)).collect()),
                    J::Arr(p.iter().map(|&x| J::Num(x)).collect()),
                ])).collect()),
            );
            rep.insert("measure".into(), J::Arr(out.meas.iter().map(|&(q, c)| J::Arr(vec![J::Num(q as f64), J::Num(c as f64)])).collect()));
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
        std::fs::write(p, json::write(&J::Obj(rep))).map_err(|e| format!("{}: {}", p, e))?;
    }
    Ok(code)
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
