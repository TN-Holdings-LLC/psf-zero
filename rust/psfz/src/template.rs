//! Templates (r5 prototype, 2026-10-19): the output of `compile(.., true)` as JSON, and binding new slot values in
//! closed form. A template fixes the gate list; `bind` recomputes only the angles of the runs that carry slots: for
//! each run, the product of its factors (fixed matrices and the slots' gates at the new values) and that product's
//! ZYZ angles written in the general form. No compilation, no decision, no search.

use crate::compile::{one, zsx_general, Fac, Output};
use crate::json::J;
use crate::linalg::*;
use std::collections::BTreeMap;

pub const VERSION: &str = "psfz r5 template prototype (2026-10-19)";

fn m2_json(m: &M2) -> J {
    J::Arr(m.iter().flat_map(|c| [J::Num(c.re), J::Num(c.im)]).collect())
}

fn m2_read(j: &J) -> Result<M2, String> {
    let v = j.arr().ok_or("template: a fixed factor is not an array")?;
    if v.len() != 8 {
        return Err("template: a fixed factor does not have 8 numbers".into());
    }
    let n = |k: usize| v[k].num().ok_or("template: a factor entry is not a number");
    let mut m = [ZERO; 4];
    for k in 0..4 {
        m[k] = C::new(n(2 * k)?, n(2 * k + 1)?);
    }
    Ok(m)
}

pub fn to_json(out: &Output, device: &str) -> J {
    let mut t = BTreeMap::new();
    t.insert("version".into(), J::Str(VERSION.into()));
    t.insert("device".into(), J::Str(device.into()));
    t.insert("kind".into(), J::Str(out.kind.clone()));
    t.insert("layout".into(), J::Arr(out.layout.iter().map(|&q| J::Num(q as f64)).collect()));
    t.insert("num_clbits".into(), J::Num(out.nc as f64));
    t.insert(
        "ops".into(),
        J::Arr(out.ops.iter().map(|(n, q, p)| J::Arr(vec![
            J::Str(n.clone()),
            J::Arr(q.iter().map(|&x| J::Num(x as f64)).collect()),
            J::Arr(p.iter().map(|&x| J::Num(x)).collect()),
        ])).collect()),
    );
    t.insert("measure".into(), J::Arr(out.meas.iter().map(|&(q, c)| J::Arr(vec![J::Num(q as f64), J::Num(c as f64)])).collect()));
    t.insert(
        "slots".into(),
        J::Arr(out.slots.iter().map(|s| {
            let mut o = BTreeMap::new();
            o.insert("gate".into(), J::Str(s.gate.clone()));
            o.insert("qubit".into(), J::Num(s.qubit as f64));
            o.insert("nparams".into(), J::Num(s.nparams as f64));
            J::Obj(o)
        }).collect()),
    );
    // a factor is a slot index (number) or a fixed matrix (8 numbers: re, im of m00, m01, m10, m11)
    t.insert(
        "runs".into(),
        J::Arr(out.runs.iter().map(|r| {
            let mut o = BTreeMap::new();
            o.insert("qubit".into(), J::Num(r.qubit as f64));
            o.insert("rz".into(), J::Arr(r.rz.iter().map(|&k| J::Num(k as f64)).collect()));
            o.insert("factors".into(), J::Arr(r.facs.iter().map(|f| match f {
                Fac::S(k, _) => J::Num(*k as f64),
                Fac::M(m) => m2_json(m),
            }).collect()));
            J::Obj(o)
        }).collect()),
    );
    J::Obj(t)
}

/// The template's operations with the runs' angles for slot values `values` (one list of parameters per slot).
pub fn bind(t: &J, values: &[Vec<f64>]) -> Result<Vec<(String, Vec<usize>, Vec<f64>)>, String> {
    let ops_j = t.get("ops").and_then(|v| v.arr()).ok_or("template: no ops")?;
    let mut ops = Vec::with_capacity(ops_j.len());
    for o in ops_j {
        let a = o.arr().ok_or("template: an op is not an array")?;
        let name = a[0].str().ok_or("template: an op without a name")?.to_string();
        let qs = a[1].arr().ok_or("template: op qubits")?.iter().map(|x| x.num().unwrap_or(0.0) as usize).collect();
        let ps = a[2].arr().ok_or("template: op params")?.iter().map(|x| x.num().unwrap_or(0.0)).collect();
        ops.push((name, qs, ps));
    }
    let slots = t.get("slots").and_then(|v| v.arr()).ok_or("template: no slots")?;
    if slots.len() != values.len() {
        return Err(format!("template: {} slots, {} values given", slots.len(), values.len()));
    }
    let mut mats = Vec::with_capacity(slots.len());
    for (k, s) in slots.iter().enumerate() {
        let g = s.get("gate").and_then(|v| v.str()).ok_or("template: a slot without a gate")?;
        let np = s.get("nparams").and_then(|v| v.num()).unwrap_or(0.0) as usize;
        if values[k].len() != np {
            return Err(format!("template: slot {} takes {} parameters, {} given", k, np, values[k].len()));
        }
        mats.push(one(g, &values[k]).ok_or("template: a slot's gate is not known")?);
    }
    let runs = t.get("runs").and_then(|v| v.arr()).ok_or("template: no runs")?;
    for r in runs {
        let rz: Vec<usize> = r.get("rz").and_then(|v| v.arr()).ok_or("template: run rz")?
            .iter().map(|x| x.num().unwrap_or(0.0) as usize).collect();
        let mut m = eye();
        for f in r.get("factors").and_then(|v| v.arr()).ok_or("template: run factors")? {
            let g = match f {
                J::Num(k) => mats[*k as usize],
                other => m2_read(other)?,
            };
            m = mul2(&g, &m);
        }
        let angles: Vec<f64> = zsx_general(&m, rz.len() == 2).into_iter().filter(|(n, _)| *n == "rz").map(|(_, a)| a[0]).collect();
        for (i, k) in rz.iter().enumerate() {
            ops[*k].2 = vec![angles[i]];
        }
    }
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::zsx_general;
    use std::f64::consts::PI;

    #[test]
    fn general_form_holds_for_every_angle() {
        let mats = [eye(), x(), y(), z(), h(), sx(), rz(0.3), ry(PI), rx(PI), ry(PI / 2.0), ry(0.0), ry(-PI),
                    u3(0.4, 1.1, -2.3), u3(2.9, -0.2, 0.5), mul2(&h(), &rz(0.7)), u3(1e-12, 0.3, 0.2), u3(PI - 1e-12, 0.1, 0.0)];
        for m in mats.iter() {
            let mut back = eye();
            for (n, a) in zsx_general(m, false) {
                let g = match n {
                    "rz" => rz(a[0]),
                    "sx" => sx(),
                    _ => unreachable!(),
                };
                back = mul2(&g, &back);
            }
            assert!(eq_phase(m, &back, 1e-9), "{:?}", m);
            assert_eq!(zsx_general(m, false).len(), 5);
            assert_eq!(zsx_general(m, true).len(), 4);
        }
    }
}
