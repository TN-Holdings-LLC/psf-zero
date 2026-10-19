//! psfz -- PSF-Zero in Rust, prototype r4 (2026-10-18): circuits whose two-qubit interaction graph is a chain or a
//! ring, compiled for an IBM heavy-hex device (CZ: Heron; ECR: Eagle) by formula only. OpenQASM 3 in and out; the
//! device from a JSON file. No search, no estimate of noise, no simulation, no check at run time: placement by
//! coordinates, order by commutation levels, every rewrite a closed form tested once in the unit tests.
//!
//!     psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]
//!     psfz template DEVICE.json IN.qasm OUT.json          (r5 prototype: every single-qubit gate with parameters is a
//!                                                          slot; the gate list never depends on the values)
//!     psfz bind TEMPLATE.json VALUES.json OUT.qasm        (VALUES: one list of parameters per slot, in input order)
//!
//! Exit 0: compiled. Exit 2: refused (the reason on stderr and in the report). Exit 1: an input error. `compile` is
//! r4's, unchanged (its report says r4: release 2026-10-18.1 uses psfz's output only for that version).

mod compile;
mod device;
mod json;
mod kak;
mod linalg;
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

fn run(args: &[String]) -> Result<i32, String> {
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
