//! psfz -- PSF-Zero in Rust, prototype r3 (2026-10-17): circuits whose two-qubit interaction graph is a chain or a
//! ring, compiled for an IBM heavy-hex device (CZ: Heron; ECR: Eagle) by formula only. OpenQASM 3 in and out; the
//! device from a JSON file. No search, no estimate of noise, no simulation, no check at run time: placement by
//! coordinates, order by commutation levels, every rewrite a closed form tested once in the unit tests.
//!
//!     psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]
//!
//! Exit 0: compiled. Exit 2: refused (the reason on stderr and in the report). Exit 1: an input error.

mod compile;
mod device;
mod json;
mod kak;
mod linalg;
mod place;
mod qasm;

use json::J;
use std::collections::BTreeMap;
use std::time::Instant;

fn run(args: &[String]) -> Result<i32, String> {
    if args.len() < 4 || args[0] != "compile" {
        return Err("usage: psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]".into());
    }
    let dev_text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {}", args[1], e))?;
    let src = std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {}", args[2], e))?;
    let t0 = Instant::now();
    let d = device::Device::from_json(&json::parse(&dev_text)?)?;
    let lat = device::Lattice::read(&d);
    let t1 = Instant::now();
    let c = qasm::read(&src)?;
    let res = match &lat {
        Ok(l) => compile::compile(&c, &d, l),
        Err(e) => Err(format!("the device is not read as a heavy-hex lattice: {}", e)),
    };
    let t2 = Instant::now();
    let mut rep = BTreeMap::new();
    rep.insert("version".into(), J::Str("psfz r3 (2026-10-17)".into()));
    rep.insert("device".into(), J::Str(d.name.clone()));
    rep.insert("device_s".into(), J::Num((t1 - t0).as_secs_f64()));
    rep.insert("compile_s".into(), J::Num((t2 - t1).as_secs_f64()));
    let code = match res {
        Ok(out) => {
            let text = qasm::write(out.nc, &out.ops, &out.meas,
                                   &format!("psfz r3: {} on {}, layout {:?}", out.kind, d.name, out.layout));
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
