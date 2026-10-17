//! The compiler: a gate-level circuit whose two-qubit interaction graph is a chain or a ring, built for the device
//! by formula. Steps, each a closed form:
//!
//! 1. Every gate becomes single-qubit matrices and two-qubit Pauli rotations PP(theta) = exp(-i theta/2 P P),
//!    P in {X, Y, Z}: a diagonal two-qubit gate is ZZ(-gamma/2) and single-qubit phases (gamma: its phases' second
//!    difference); cx is ZZ between fixed basis changes; rxx, ryy, rzz are themselves.
//! 2. Every diagonal operation after which only diagonal operations and final measurements follow on its qubits is
//!    dropped (it does not change the outcome distribution).
//! 3. Order: every operation gets a level, one more than the highest level of an earlier operation on a shared qubit
//!    that does not commute with it (diagonal operations commute; Pauli rotations commute when their Paulis do);
//!    emitted by level, two-qubit rotations first, even edges of the chain or ring before odd ones. Operations that
//!    do not commute keep their order: exact. Rotations on the same pair that end up next to each other are merged
//!    (they commute): exp(-i/2 (x XX + y YY + z ZZ)).
//! 4. Placement by coordinates (`place`).
//! 5. ZZ(theta) on the device: none for theta = 0, Z Z for theta = pi, one CZ and phases for theta = +-pi/2, else
//!    H CZ RX(theta) CZ H; XX or YY alone: ZZ between basis changes; two or three kinds: the 3-CZ canonical form
//!    (Vatan and Williams' circuit, its signs fixed for these conventions and checked in the tests). On an ECR device
//!    each CZ is the coupler's ECR with fixed single-qubit gates (item 104). Where a form puts its basis changes on
//!    one qubit of the pair: in the general ZZ form, the input's higher-numbered qubit (as items 87/91 order a term's
//!    support), on an ECR device the coupler's ECR control; in the canonical form, the qubit whose pending single-qubit run already needs pulses
//!    (so they merge for free), else the one free earlier on the schedule so far (ties: the lower index). Rules on
//!    the circuit built so far, not comparisons of alternatives.
//! 6. Single-qubit runs merged into one matrix and written as rz, sx, x in closed form (ZYZ); the last rz before a
//!    measurement is dropped.

use crate::device::{Device, Lattice};
use crate::linalg::*;
use crate::place;
use crate::qasm::{Circuit, Op};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

pub const TOL: f64 = 1e-10;
pub const KNOWN: &[&str] = &[
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u1", "u2", "u3", "u", "U",
    "cx", "CX", "cz", "cp", "cphase", "crz", "rzz", "rxx", "ryy",
];

/// A single-qubit matrix, or a two-qubit Pauli rotation exp(-i theta/2 P P) with P = X, Y, Z (0, 1, 2).
#[derive(Clone, Debug)]
pub enum Ir {
    U(usize, M2),
    Pr(usize, usize, u8, f64),
}

#[derive(Clone, Debug)]
pub struct Output {
    pub kind: String,
    pub layout: Vec<usize>,
    pub ops: Vec<(String, Vec<usize>, Vec<f64>)>,
    pub meas: Vec<(usize, usize)>,
    pub nc: usize,
}

fn one(name: &str, p: &[f64]) -> Option<M2> {
    let s = |k: usize| p.get(k).copied().unwrap_or(0.0);
    Some(match name {
        "id" => eye(),
        "x" => x(),
        "y" => y(),
        "z" => z(),
        "h" => h(),
        "s" => phase(PI / 2.0),
        "sdg" => phase(-PI / 2.0),
        "t" => phase(PI / 4.0),
        "tdg" => phase(-PI / 4.0),
        "sx" => sx(),
        "sxdg" => dag2(&sx()),
        "rx" => rx(s(0)),
        "ry" => ry(s(0)),
        "rz" => rz(s(0)),
        "p" | "u1" => phase(s(0)),
        "u2" => u3(PI / 2.0, s(0), s(1)),
        "u3" | "u" | "U" => u3(s(0), s(1), s(2)),
        _ => return None,
    })
}

/// A diagonal two-qubit gate with phases phi[x][y] (x: first qubit) as ZZ(theta) and phases on each qubit.
fn diag2(a: usize, b: usize, phi: [[f64; 2]; 2], out: &mut Vec<Ir>) {
    let gamma = phi[1][1] - phi[1][0] - phi[0][1] + phi[0][0];
    let theta = -gamma / 2.0;
    // ZZ(theta) gives phase -theta/2 * (-1)^(x+y); the rest is linear
    let zz = |x: usize, y: usize| -theta / 2.0 * if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
    let r = |x: usize, y: usize| phi[x][y] - zz(x, y);
    out.push(Ir::Pr(a, b, 2, theta));
    out.push(Ir::U(a, phase(r(1, 0) - r(0, 0))));
    out.push(Ir::U(b, phase(r(0, 1) - r(0, 0))));
}

/// Step 1.
pub fn lower_input(c: &Circuit) -> Result<(Vec<Ir>, Vec<(usize, usize)>), String> {
    let mut ir = Vec::new();
    let mut meas = Vec::new();
    let mut measured = vec![false; c.nq];
    let mut cl_used = BTreeSet::new();
    for op in &c.ops {
        match op {
            Op::Measure { qubit, clbit } => {
                if measured[*qubit] || !cl_used.insert(*clbit) {
                    return Err("a qubit or a classical bit is measured twice (not read)".into());
                }
                measured[*qubit] = true;
                meas.push((*qubit, *clbit));
            }
            Op::Gate { name, qubits, params } => {
                if !KNOWN.contains(&name.as_str()) {
                    return Err(format!("the gate '{}' is not read", name));
                }
                if qubits.iter().any(|q| measured[*q]) {
                    return Err("a gate after a measurement (only final measurements are read)".into());
                }
                let p = |k: usize| params.get(k).copied().unwrap_or(0.0);
                if qubits.len() == 1 {
                    ir.push(Ir::U(qubits[0], one(name, params).unwrap()));
                    continue;
                }
                if qubits.len() != 2 {
                    return Err(format!("'{}' acts on {} qubits", name, qubits.len()));
                }
                let (a, b) = (qubits[0], qubits[1]);
                match name.as_str() {
                    "cz" => diag2(a, b, [[0.0, 0.0], [0.0, PI]], &mut ir),
                    "cp" | "cphase" => diag2(a, b, [[0.0, 0.0], [0.0, p(0)]], &mut ir),
                    "crz" => diag2(a, b, [[0.0, 0.0], [-p(0) / 2.0, p(0) / 2.0]], &mut ir),
                    "rzz" => ir.push(Ir::Pr(a, b, 2, p(0))),
                    "cx" | "CX" => {
                        ir.push(Ir::U(b, h()));
                        diag2(a, b, [[0.0, 0.0], [0.0, PI]], &mut ir);
                        ir.push(Ir::U(b, h()));
                    }
                    "rxx" => ir.push(Ir::Pr(a, b, 0, p(0))),
                    "ryy" => ir.push(Ir::Pr(a, b, 1, p(0))),
                    _ => return Err(format!("the gate '{}' is not read", name)),
                }
            }
        }
    }
    Ok((ir, meas))
}

fn is_diagonal(op: &Ir) -> bool {
    match op {
        Ir::U(_, m) => is_diag(m, TOL),
        Ir::Pr(_, _, k, _) => *k == 2,
    }
}

fn qubits(op: &Ir) -> Vec<usize> {
    match op {
        Ir::U(q, _) => vec![*q],
        Ir::Pr(a, b, _, _) => vec![*a, *b],
    }
}

/// Whether two operations commute: both diagonal, or two Pauli rotations whose Paulis commute (they differ on an even
/// number of shared qubits), or no shared qubit.
fn commute(p: &Ir, q: &Ir) -> bool {
    let (qp, qq) = (qubits(p), qubits(q));
    if !qp.iter().any(|x| qq.contains(x)) {
        return true;
    }
    if is_diagonal(p) && is_diagonal(q) {
        return true;
    }
    if let (Ir::Pr(a, b, k, _), Ir::Pr(c, d, l, _)) = (p, q) {
        let shared = [*a, *b].iter().filter(|x| **x == *c || **x == *d).count();
        return k == l || shared % 2 == 0;
    }
    false
}

/// Step 2.
pub fn drop_final_diagonal(ir: Vec<Ir>, nq: usize) -> Vec<Ir> {
    let mut clean = vec![true; nq]; // no non-diagonal operation later on this qubit
    let mut keep = vec![true; ir.len()];
    for (k, op) in ir.iter().enumerate().rev() {
        let qs = qubits(op);
        if is_diagonal(op) && qs.iter().all(|&q| clean[q]) {
            keep[k] = false;
        } else if !is_diagonal(op) {
            for q in qs {
                clean[q] = false;
            }
        }
    }
    ir.into_iter().zip(keep).filter(|(_, k)| *k).map(|(o, _)| o).collect()
}

/// The interaction graph as a chain or a ring: (kind, logical qubits in walk order).
pub fn read_graph(ir: &[Ir], nq: usize) -> Result<(String, Vec<usize>), String> {
    let mut adj = vec![BTreeSet::new(); nq];
    for op in ir {
        if let Ir::Pr(a, b, _, _) = op {
            if a == b {
                return Err("a two-qubit gate on one qubit".into());
            }
            adj[*a].insert(*b);
            adj[*b].insert(*a);
        }
    }
    if adj.iter().any(|s| s.len() > 2) {
        return Err("a qubit interacts with more than two others: not a chain or a ring".into());
    }
    if adj.iter().any(|s| s.is_empty()) {
        return Err("a qubit without two-qubit gates (not read by this prototype)".into());
    }
    let edges: usize = adj.iter().map(|s| s.len()).sum::<usize>() / 2;
    let ring = edges == nq;
    if !ring && edges != nq - 1 {
        return Err("the interaction graph is not one chain or one ring".into());
    }
    let start = if ring { 0 } else { (0..nq).find(|&q| adj[q].len() == 1).unwrap() };
    let mut walk = vec![start];
    let mut prev = usize::MAX;
    let mut cur = start;
    while walk.len() < nq {
        let next = *adj[cur].iter().filter(|&&v| v != prev).min().ok_or("the interaction graph is not connected")?;
        if walk.contains(&next) {
            return Err("the interaction graph is not one chain or one ring".into());
        }
        walk.push(next);
        prev = cur;
        cur = next;
    }
    if ring && !adj[cur].contains(&start) {
        return Err("the interaction graph is not one ring".into());
    }
    if ring && nq % 2 == 1 {
        return Err("an odd ring does not embed in a heavy-hex device".into());
    }
    Ok((if ring { "ring" } else { "chain" }.to_string(), walk))
}

/// Step 3.
pub fn order(ir: Vec<Ir>, nq: usize, walk: &[usize]) -> Vec<Ir> {
    let mut rank = vec![0usize; nq];
    for (i, &q) in walk.iter().enumerate() {
        rank[q] = i;
    }
    let n = nq;
    let mut level = vec![0usize; ir.len()];
    let mut last: Vec<Vec<usize>> = vec![Vec::new(); nq]; // earlier operations per qubit
    for k in 0..ir.len() {
        let mut lv = 0;
        for q in qubits(&ir[k]) {
            for &j in &last[q] {
                if !commute(&ir[k], &ir[j]) {
                    lv = lv.max(level[j] + 1);
                }
            }
        }
        level[k] = lv;
        for q in qubits(&ir[k]) {
            last[q].push(k);
        }
    }
    let key = |k: usize| -> (usize, usize, usize, usize, usize) {
        match &ir[k] {
            Ir::Pr(a, b, _, _) => {
                let (ra, rb) = (rank[*a], rank[*b]);
                // the edge's position along the walk: i for the edge (i, i + 1), n - 1 for the ring's closing edge
                let pos = if (ra + 1) % n == rb { ra } else { rb };
                (level[k], 0, pos % 2, pos, k)
            }
            Ir::U(q, _) => (level[k], 1, 0, rank[*q], k),
        }
    };
    let mut idx: Vec<usize> = (0..ir.len()).collect();
    idx.sort_by_key(|&k| key(k));
    idx.into_iter().map(|k| ir[k].clone()).collect()
}

/// Rotations on one pair, merged: (a, b, [x, y, z]) for exp(-i/2 (x XX + y YY + z ZZ)).
#[derive(Clone, Debug)]
pub enum Blk {
    U(usize, M2),
    Can(usize, usize, [f64; 3]),
}

/// Step 3's merge: Pauli rotations on the same pair next to each other in the order become one block.
pub fn merge(ir: Vec<Ir>) -> Vec<Blk> {
    let mut out: Vec<Blk> = Vec::new();
    for op in ir {
        match op {
            Ir::U(q, m) => out.push(Blk::U(q, m)),
            Ir::Pr(a, b, k, t) => {
                if let Some(Blk::Can(c, d, ang)) = out.last_mut() {
                    if (*c, *d) == (a, b) || (*c, *d) == (b, a) {
                        ang[k as usize] += t;
                        continue;
                    }
                }
                let mut ang = [0.0; 3];
                ang[k as usize] = t;
                out.push(Blk::Can(a, b, ang));
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
enum Nat {
    U(usize, M2),
    Two(&'static str, usize, usize),
}

/// Step 5 for one ZZ(theta) on physical qubits a, b.
fn lower_zz(a: usize, b: usize, theta: f64, out: &mut Vec<Nat>) {
    let r = wrap(theta);
    if r.abs() < TOL {
        return;
    }
    if (r.abs() - PI).abs() < TOL {
        out.push(Nat::U(a, z()));
        out.push(Nat::U(b, z()));
        return;
    }
    if (r.abs() - PI / 2.0).abs() < TOL {
        // ZZ(theta) phases minus CZ's are linear: one CZ and a phase on each qubit
        let zz = |x: usize, y: usize| -r / 2.0 * if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
        let res = |x: usize, y: usize| zz(x, y) - if x == 1 && y == 1 { PI } else { 0.0 };
        out.push(Nat::Two("cz", a, b));
        out.push(Nat::U(a, phase(res(1, 0) - res(0, 0))));
        out.push(Nat::U(b, phase(res(0, 1) - res(0, 0))));
        return;
    }
    out.push(Nat::U(b, h()));
    out.push(Nat::Two("cz", a, b));
    out.push(Nat::U(b, rx(r)));
    out.push(Nat::Two("cz", a, b));
    out.push(Nat::U(b, h()));
}

/// Step 5 for a merged block exp(-i/2 (x XX + y YY + z ZZ)) on physical qubits a, b.
fn lower_can(a: usize, b: usize, ang: [f64; 3], out: &mut Vec<Nat>) {
    let nz: Vec<usize> = (0..3).filter(|&k| wrap(ang[k]).abs() > TOL).collect();
    if nz.is_empty() {
        return;
    }
    if nz.len() == 1 {
        let k = nz[0];
        let (pre, post) = match k {
            0 => (h(), h()),
            1 => {
                let w = mul2(&phase(PI / 2.0), &h()); // W Z W^dag = Y
                (dag2(&w), w)
            }
            _ => (eye(), eye()),
        };
        if k != 2 {
            out.push(Nat::U(a, pre));
            out.push(Nat::U(b, pre));
        }
        lower_zz(a, b, ang[k], out);
        if k != 2 {
            out.push(Nat::U(a, post));
            out.push(Nat::U(b, post));
        }
        return;
    }
    // Vatan and Williams' 3-CNOT circuit for exp(i(al XX + be YY + ga ZZ)), al = -x/2, be = -y/2, ga = -z/2, its
    // signs fixed for these conventions (tests: canonical_form), each CNOT(c -> t) = H_t CZ H_t
    let (x_, y_, z_) = (ang[0], ang[1], ang[2]);
    out.push(Nat::U(b, rz(-PI / 2.0)));
    out.push(Nat::U(a, h()));
    out.push(Nat::Two("cz", a, b));
    out.push(Nat::U(a, h()));
    out.push(Nat::U(a, rz(z_ - PI / 2.0)));
    out.push(Nat::U(b, ry(PI / 2.0 - x_)));
    out.push(Nat::U(b, h()));
    out.push(Nat::Two("cz", a, b));
    out.push(Nat::U(b, h()));
    out.push(Nat::U(b, ry(y_ - PI / 2.0)));
    out.push(Nat::U(a, h()));
    out.push(Nat::Two("cz", a, b));
    out.push(Nat::U(a, h()));
    out.push(Nat::U(a, rz(PI / 2.0)));
}

/// The schedule of the native operations emitted so far (as the final merge will run them): per qubit the end of
/// its last two-qubit gate and its pending single-qubit run.
struct Track<'a> {
    d: &'a Device,
    clock: BTreeMap<usize, f64>,
    pend: BTreeMap<usize, M2>,
}

fn pulses(m: &M2) -> usize {
    zsx(m).iter().filter(|(n, _)| *n != "rz").count()
}

impl<'a> Track<'a> {
    fn apply(&mut self, op: &Nat) {
        match op {
            Nat::U(q, m) => {
                let cur = self.pend.get(q).copied().unwrap_or_else(eye);
                self.pend.insert(*q, mul2(m, &cur));
            }
            Nat::Two(_, a, b) => {
                let s = self.ready(*a).max(self.ready(*b));
                let dur = self.d.dur_ps(*a, *b).unwrap_or(0) as f64 * 1e-12;
                self.clock.insert(*a, s + dur);
                self.clock.insert(*b, s + dur);
                self.pend.remove(a);
                self.pend.remove(b);
            }
        }
    }
    fn busy(&self, q: usize) -> bool {
        self.pend.get(&q).map_or(false, |m| pulses(m) > 0)
    }
    fn ready(&self, q: usize) -> f64 {
        self.clock.get(&q).copied().unwrap_or(0.0)
            + self.pend.get(&q).map_or(0, pulses) as f64 * self.d.sx_dur.get(q).copied().unwrap_or(0.0)
    }
    /// (the qubit that takes the form's basis changes, the other)
    fn pick(&self, p: usize, q: usize) -> (usize, usize) {
        let (bp, bq) = (self.busy(p), self.busy(q));
        if bp != bq {
            return if bp { (p, q) } else { (q, p) };
        }
        let (rp, rq) = (self.ready(p), self.ready(q));
        if rp < rq || (rp == rq && p < q) {
            (p, q)
        } else {
            (q, p)
        }
    }
}

/// Item 104's form: CZ(x, y) = [rz(3pi/4), sx, rz(pi/2) on y] ECR(x, y) [sx, sx, rz(-pi/2) on x; rz(pi/2), sx,
/// rz(-pi/4) on y], up to a global phase.
fn cz_as_ecr(x_: usize, y_: usize, out: &mut Vec<Nat>) {
    out.push(Nat::U(y_, rz(3.0 * PI / 4.0)));
    out.push(Nat::U(y_, sx()));
    out.push(Nat::U(y_, rz(PI / 2.0)));
    out.push(Nat::Two("ecr", x_, y_));
    out.push(Nat::U(x_, sx()));
    out.push(Nat::U(x_, sx()));
    out.push(Nat::U(x_, rz(-PI / 2.0)));
    out.push(Nat::U(y_, rz(PI / 2.0)));
    out.push(Nat::U(y_, sx()));
    out.push(Nat::U(y_, rz(-PI / 4.0)));
}

/// A single-qubit matrix as rz, sx, x in time order (closed form; tested against the matrix).
pub fn zsx(m: &M2) -> Vec<(&'static str, Vec<f64>)> {
    let (p, t, l) = zyz(m);
    let mut v: Vec<(&'static str, Vec<f64>)> = Vec::new();
    if t.abs() < TOL {
        v.push(("rz", vec![p + l]));
    } else if (t - PI / 2.0).abs() < TOL {
        v.push(("rz", vec![l - PI / 2.0]));
        v.push(("sx", vec![]));
        v.push(("rz", vec![p + PI / 2.0]));
    } else if (t - PI).abs() < TOL {
        v.push(("rz", vec![l + PI]));
        v.push(("x", vec![]));
        v.push(("rz", vec![p]));
    } else {
        v.push(("rz", vec![l]));
        v.push(("sx", vec![]));
        v.push(("rz", vec![t + PI]));
        v.push(("sx", vec![]));
        v.push(("rz", vec![p + PI]));
    }
    v.into_iter()
        .filter(|(n, a)| !(*n == "rz" && wrap(a[0]).abs() < TOL))
        .map(|(n, a)| (n, a.into_iter().map(wrap).collect()))
        .collect()
}

pub fn compile(c: &Circuit, d: &Device, l: &Lattice) -> Result<Output, String> {
    let (ir, meas) = lower_input(c)?;
    let (kind, walk) = read_graph(&ir, c.nq)?;
    let ir = drop_final_diagonal(ir, c.nq);
    let blocks = merge(order(ir, c.nq, &walk));
    let phys_walk = match kind.as_str() {
        "chain" => place::chain(d, l, c.nq).ok_or("no working path of this length on the snake")?,
        _ => place::ring(d, l, c.nq).ok_or("no working ring of this length among the shapes")?,
    };
    let mut layout = vec![0usize; c.nq];
    for (k, &q) in walk.iter().enumerate() {
        layout[q] = phys_walk[k];
    }
    // step 5
    let mut nat = Vec::new();
    let mut track = Track { d, clock: BTreeMap::new(), pend: BTreeMap::new() };
    for op in &blocks {
        let start = nat.len();
        match op {
            Blk::U(q, m) => nat.push(Nat::U(layout[*q], *m)),
            Blk::Can(a, b, ang) => {
                let (pa, pb) = (layout[*a], layout[*b]);
                let several = ang.iter().filter(|t| wrap(**t).abs() > TOL).count() > 1;
                if several {
                    let (role, other) = track.pick(pa, pb);
                    lower_can(role, other, *ang, &mut nat); // the form's basis changes are on its first qubit
                } else {
                    // ZZ's general form changes basis on its second qubit
                    let later_b = if d.gate2 == "ecr" {
                        // on ECR the form's basis changes go to the coupler's ECR control (a rule taken from the
                        // exploratory comparison: there it saved one pulse per edge on FakeBrisbane's Ising chains)
                        !d.couplers.contains_key(&(pa, pb))
                    } else {
                        *b > *a // the input's higher-numbered qubit, as items 87/91 order a term's support
                    };
                    if later_b {
                        lower_can(pa, pb, *ang, &mut nat);
                    } else {
                        lower_can(pb, pa, *ang, &mut nat);
                    }
                }
            }
        }
        for op in &nat[start..] {
            track.apply(op);
        }
    }
    if d.gate2 == "ecr" {
        let mut e = Vec::new();
        for op in nat {
            match op {
                Nat::Two(_, a, b) => {
                    if d.couplers.contains_key(&(a, b)) {
                        cz_as_ecr(a, b, &mut e)
                    } else {
                        cz_as_ecr(b, a, &mut e)
                    }
                }
                u => e.push(u),
            }
        }
        nat = e;
    } else {
        nat = nat
            .into_iter()
            .map(|op| match op {
                Nat::Two(g, a, b) if !d.couplers.contains_key(&(a, b)) => Nat::Two(g, b, a),
                o => o,
            })
            .collect();
    }
    // step 6
    let mut pend: BTreeMap<usize, M2> = BTreeMap::new();
    let mut ops: Vec<(String, Vec<usize>, Vec<f64>)> = Vec::new();
    let flush = |q: usize, pend: &mut BTreeMap<usize, M2>, ops: &mut Vec<(String, Vec<usize>, Vec<f64>)>| {
        if let Some(m) = pend.remove(&q) {
            for (n, a) in zsx(&m) {
                ops.push((n.to_string(), vec![q], a));
            }
        }
    };
    for op in nat {
        match op {
            Nat::U(q, m) => {
                let cur = pend.get(&q).copied().unwrap_or_else(eye);
                pend.insert(q, mul2(&m, &cur));
            }
            Nat::Two(g, a, b) => {
                if !d.working(a, b) {
                    return Err(format!("internal: {} on ({}, {}) is not a working coupler", g, a, b));
                }
                flush(a, &mut pend, &mut ops);
                flush(b, &mut pend, &mut ops);
                ops.push((g.to_string(), vec![a, b], vec![]));
            }
        }
    }
    // the last run before a measurement: its final rz is dropped (diagonal before a Z measurement)
    let rest: Vec<usize> = pend.keys().copied().collect();
    for q in rest {
        let m = pend.remove(&q).unwrap();
        let mut seq = zsx(&m);
        if seq.last().map_or(false, |(n, _)| *n == "rz") {
            seq.pop();
        }
        for (n, a) in seq {
            ops.push((n.to_string(), vec![q], a));
        }
    }
    let meas = meas.into_iter().map(|(q, cb)| (layout[q], cb)).collect();
    Ok(Output { kind, layout, ops, meas, nc: c.nc })
}

#[cfg(test)]
mod tests {
    use super::*;

    type M4 = [[C; 4]; 4];
    fn id4() -> M4 {
        let mut m = [[ZERO; 4]; 4];
        for i in 0..4 {
            m[i][i] = ONE;
        }
        m
    }
    fn mul4(a: &M4, b: &M4) -> M4 {
        let mut m = [[ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    m[i][j] = m[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        m
    }
    /// u on qubit q (0: first, the high bit of the index; 1: second)
    fn lift(u: &M2, q: usize) -> M4 {
        let mut m = [[ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let (xi, yi, xj, yj) = (i >> 1, i & 1, j >> 1, j & 1);
                m[i][j] = if q == 0 {
                    if yi == yj { u[2 * xi + xj] } else { ZERO }
                } else if xi == xj {
                    u[2 * yi + yj]
                } else {
                    ZERO
                };
            }
        }
        m
    }
    fn zz4(t: f64) -> M4 {
        let mut m = [[ZERO; 4]; 4];
        for i in 0..4 {
            let s = if (i >> 1) ^ (i & 1) == 0 { 1.0 } else { -1.0 };
            m[i][i] = C::expi(-t / 2.0 * s);
        }
        m
    }
    fn cz4() -> M4 {
        let mut m = id4();
        m[3][3] = -ONE;
        m
    }
    /// Qiskit's ECRGate on (q0, q1), in this module's order (q0 the high bit)
    fn ecr4() -> M4 {
        let r = 1.0 / 2f64.sqrt();
        let q = [
            [ZERO, C::new(r, 0.0), ZERO, C::new(0.0, r)],
            [C::new(r, 0.0), ZERO, C::new(0.0, -r), ZERO],
            [ZERO, C::new(0.0, r), ZERO, C::new(r, 0.0)],
            [C::new(0.0, -r), ZERO, C::new(r, 0.0), ZERO],
        ];
        let p = [0usize, 2, 1, 3];
        let mut m = [[ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[p[i]][p[j]] = q[i][j];
            }
        }
        m
    }
    fn eq4(a: &M4, b: &M4) -> bool {
        let (mut bi, mut bj) = (0, 0);
        for i in 0..4 {
            for j in 0..4 {
                if a[i][j].abs() > a[bi][bj].abs() {
                    bi = i;
                    bj = j;
                }
            }
        }
        let f = a[bi][bj].div(b[bi][bj]);
        (0..4).all(|i| (0..4).all(|j| (a[i][j] - b[i][j] * f).abs() < 1e-9))
    }
    /// The 4 x 4 matrix of a list of native operations on qubits 0 and 1 (time order).
    fn mat(ops: &[Nat]) -> M4 {
        let mut m = id4();
        for op in ops {
            let g = match op {
                Nat::U(q, u) => lift(u, *q),
                Nat::Two("cz", _, _) => cz4(),
                Nat::Two("ecr", a, _) => {
                    assert_eq!(*a, 0);
                    ecr4()
                }
                _ => unreachable!(),
            };
            m = mul4(&g, &m);
        }
        m
    }

    #[test]
    fn zz_forms() {
        for t in [0.0, PI, -PI, PI / 2.0, -PI / 2.0, 0.37, -2.1, 3.0, 1.5707963, 5.0 * PI / 2.0] {
            let mut v = Vec::new();
            lower_zz(0, 1, t, &mut v);
            assert!(eq4(&mat(&v), &zz4(t)), "theta {}", t);
        }
    }

    #[test]
    fn diag2_forms() {
        for phi in [[[0.0, 0.0], [0.0, PI]], [[0.0, 0.0], [0.0, 0.7]], [[0.0, 0.0], [-0.4, 0.4]], [[0.1, -0.3], [0.8, 2.0]]] {
            let mut ir = Vec::new();
            diag2(0, 1, phi, &mut ir);
            let mut m = id4();
            for op in &ir {
                let g = match op {
                    Ir::U(q, u) => lift(u, *q),
                    Ir::Pr(_, _, _, t) => zz4(*t),
                };
                m = mul4(&g, &m);
            }
            let mut want = [[ZERO; 4]; 4];
            for i in 0..4 {
                want[i][i] = C::expi(phi[i >> 1][i & 1]);
            }
            assert!(eq4(&m, &want));
        }
    }

    fn pauli4(k: usize, t: f64) -> M4 {
        // exp(-i t/2 P P) = (V V) ZZ(t) (V^dag V^dag) with V Z V^dag = P
        let v = match k {
            0 => h(),
            1 => mul2(&phase(PI / 2.0), &h()),
            _ => eye(),
        };
        let vd = dag2(&v);
        let m = mul4(&mul4(&lift(&vd, 0), &lift(&vd, 1)), &id4());
        let m = mul4(&zz4(t), &m);
        mul4(&mul4(&lift(&v, 0), &lift(&v, 1)), &m)
    }

    #[test]
    fn canonical_form() {
        for ang in [[0.3, -0.7, 1.1], [0.0, 0.4, 0.0], [1.2, 0.0, 0.0], [0.5, 0.5, 0.5], [-2.0, 1.0, 0.0],
                    [PI / 2.0, PI / 2.0, 0.3], [0.0, 0.0, PI / 2.0]] {
            let mut v = Vec::new();
            lower_can(0, 1, ang, &mut v);
            let want = mul4(&pauli4(0, ang[0]), &mul4(&pauli4(1, ang[1]), &pauli4(2, ang[2])));
            assert!(eq4(&mat(&v), &want), "{:?}", ang);
            assert!(v.iter().filter(|o| matches!(o, Nat::Two(..))).count() <= 3);
        }
    }

    #[test]
    fn cz_from_ecr() {
        let mut v = Vec::new();
        cz_as_ecr(0, 1, &mut v);
        assert!(eq4(&mat(&v), &cz4()));
    }

    #[test]
    fn zsx_forms() {
        let mats = [eye(), x(), y(), z(), h(), sx(), rz(0.3), ry(PI), rx(PI), ry(PI / 2.0), rx(-PI / 2.0),
                    u3(0.4, 1.1, -2.3), u3(2.9, -0.2, 0.5), mul2(&h(), &rz(0.7))];
        for m in mats.iter() {
            let mut back = eye();
            for (n, a) in zsx(m) {
                let g = match n {
                    "rz" => rz(a[0]),
                    "sx" => sx(),
                    "x" => x(),
                    _ => unreachable!(),
                };
                back = mul2(&g, &back);
            }
            assert!(eq_phase(m, &back, 1e-9), "{:?}", zsx(m));
        }
    }

    #[test]
    fn ryy_and_rxx_conjugations() {
        // W Z W^dag = Y with W = S H
        let w = mul2(&phase(PI / 2.0), &h());
        assert!(eq_phase(&mul2(&mul2(&w, &z()), &dag2(&w)), &y(), 1e-12));
        assert!(eq_phase(&mul2(&mul2(&h(), &z()), &h()), &x(), 1e-12));
    }
}
