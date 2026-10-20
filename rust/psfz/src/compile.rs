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
//!
//! Templates (r5 prototype, 2026-10-19; `compile(.., true)`, `psfz template`): every single-qubit gate of the input
//! with parameters is a slot. A slot is carried through steps 1-6 as a factor (never multiplied into a fixed matrix
//! early), every decision above that reads a value is taken from the structure instead (a slot is diagonal when its
//! gate is diagonal for every value: rz, p, u1; a block with a slot is lowered part by part, not by KAK; a pending run
//! with a slot counts as two pulses), and a run with a slot is always written in the general form rz sx rz sx rz
//! (the last rz dropped before a measurement). So the output's gate list depends on the input's structure and the
//! device only, and each run's three angles are a closed-form function (ZYZ of the product of its factors) of the
//! slots in it: `template::bind` evaluates only that. Two-qubit gates with parameters are not slots (their values
//! are fixed in the template); ECR devices are refused. Without templates every path is r4's, unchanged.

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

/// A single-qubit matrix, or a two-qubit Pauli rotation exp(-i theta/2 P P) with P = X, Y, Z (0, 1, 2); in a
/// template, `S` is a single-qubit operation that depends on slots: its matrix at the given values and its factors.
#[derive(Clone, Debug)]
pub enum Ir {
    U(usize, M2),
    Pr(usize, usize, u8, f64),
    S(usize, M2, Vec<Fac>),
}

/// A factor of a template run (time order within the run): a fixed matrix, or input slot k with its gate's rotation
/// axis (`AX_X`, `AX_Y`, `AX_Z`: rx, ry, rz / p / u1; `AX_NONE`: u, u2, u3, which have no single axis).
#[derive(Clone, Debug)]
pub enum Fac {
    M(M2),
    S(usize, u8),
}

pub const AX_X: u8 = 0;
pub const AX_Y: u8 = 1;
pub const AX_Z: u8 = 2;
pub const AX_NONE: u8 = 3;

fn slot_axis(name: &str) -> u8 {
    match name {
        "rx" => AX_X,
        "ry" => AX_Y,
        "rz" | "p" | "u1" => AX_Z,
        _ => AX_NONE,
    }
}

/// Class Z (r5, workplace proposal of 2026-10-20 §1): a run F_0 S_1 F_1 ... S_m F_m (time order) equals
/// G · T_m ··· T_1 with G = F_m ··· F_0 and T_k = A_k^† S_k A_k, A_k = F_(k-1) ··· F_0. When every slot's axis, turned by
/// A_k, is ±z (to TOL), the T_k are rz(±θ_k) and the run is G · rz(φ), φ = Σ s_k θ_k, for every value. Reads only the
/// slots' axes and the fixed factors. Returns (G, [(slot, sign)]) or None. `class_z_left` is the mirror image.
pub fn class_z(facs: &[Fac]) -> Option<(M2, Vec<(usize, f64)>)> {
    let mut a = eye();
    let mut terms = Vec::new();
    for f in facs {
        match f {
            Fac::M(m) => a = mul2(m, &a),
            Fac::S(k, ax) => {
                let sig = match *ax {
                    AX_X => x(),
                    AX_Y => y(),
                    AX_Z => z(),
                    _ => return None,
                };
                // A^† σ A = a_x X + a_y Y + a_z Z; a_z = Re(entry 00)
                let b = mul2(&mul2(&dag2(&a), &sig), &a);
                let az = b[0].re;
                if az.abs() < 1.0 - TOL {
                    return None;
                }
                terms.push((*k, az.signum()));
            }
        }
    }
    Some((a, terms))
}

/// Class Z on the left (home, 2026-10-20): with B_k = F_m ··· F_k the fixed product after slot k, the run equals
/// U_m ··· U_1 · G with U_k = B_k S_k B_k^†; when every slot's axis turned by B_k is ±z, the run is rz(φ) · G for every
/// value. Tried when `class_z` does not hold (a fixed order, not a comparison).
pub fn class_z_left(facs: &[Fac]) -> Option<(M2, Vec<(usize, f64)>)> {
    let mut b = eye();
    let mut terms = Vec::new();
    for f in facs.iter().rev() {
        match f {
            Fac::M(m) => b = mul2(&b, m),
            Fac::S(k, ax) => {
                let sig = match *ax {
                    AX_X => x(),
                    AX_Y => y(),
                    AX_Z => z(),
                    _ => return None,
                };
                let c = mul2(&mul2(&b, &sig), &dag2(&b));
                let az = c[0].re;
                if az.abs() < 1.0 - TOL {
                    return None;
                }
                terms.push((*k, az.signum()));
            }
        }
    }
    terms.reverse();
    Some((b, terms))
}

/// A class-Z run: (G, terms, true when φ stands on the left of G).
pub fn class_z_any(facs: &[Fac]) -> Option<(M2, Vec<(usize, f64)>, bool)> {
    if let Some((g, t)) = class_z(facs) {
        return Some((g, t, false));
    }
    class_z_left(facs).map(|(g, t)| (g, t, true))
}

/// The fixed ZYZ angle t of G decides the form of a class-Z run: rz only (t = 0), rz x rz (t = π), rz sx rz (t = π/2),
/// else the general form. Returns the number of pulses.
fn class_z_pulses(g: &M2) -> usize {
    let (_, t, _) = zyz(g);
    if t.abs() < TOL {
        0
    } else if (t - PI).abs() < TOL || (t - PI / 2.0).abs() < TOL {
        1
    } else {
        2
    }
}

/// An input gate with parameters, made a slot in a template.
#[derive(Clone, Debug)]
pub struct Slot {
    pub gate: String,
    pub qubit: usize,
    pub nparams: usize,
}

/// A run of a template with slots: its qubit, its factors and, either in the general form the indices of its rz
/// operations in `Output::ops` (3, or 2 before a measurement), or for a class-Z run (`zform`) the one rz whose angle is
/// base + Σ sign · θ_slot (None when the run vanishes before a measurement).
#[derive(Clone, Debug)]
pub struct RunMap {
    pub qubit: usize,
    pub facs: Vec<Fac>,
    pub rz: Vec<usize>,
    pub zform: Option<(Option<usize>, f64, Vec<(usize, f64)>)>,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub kind: String,
    pub layout: Vec<usize>,
    pub ops: Vec<(String, Vec<usize>, Vec<f64>)>,
    pub meas: Vec<(usize, usize)>,
    pub nc: usize,
    pub slots: Vec<Slot>,
    pub runs: Vec<RunMap>,
}

pub fn one(name: &str, p: &[f64]) -> Option<M2> {
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

/// Step 1 (with `template`, every single-qubit gate with parameters becomes a slot).
pub fn lower_input(c: &Circuit, template: bool) -> Result<(Vec<Ir>, Vec<(usize, usize)>, Vec<Slot>), String> {
    let mut slots = Vec::new();
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
                    let m = one(name, params).unwrap();
                    if template && !params.is_empty() {
                        let k = slots.len();
                        slots.push(Slot { gate: name.clone(), qubit: qubits[0], nparams: params.len() });
                        ir.push(Ir::S(qubits[0], m, vec![Fac::S(k, slot_axis(name))]));
                    } else {
                        ir.push(Ir::U(qubits[0], m));
                    }
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
    Ok((ir, meas, slots))
}

fn is_diagonal(op: &Ir) -> bool {
    match op {
        Ir::U(_, m) => is_diag(m, TOL),
        Ir::Pr(_, _, k, _) => *k == 2,
        Ir::S(_, _, f) => f.iter().all(|x| match x {
            Fac::M(m) => is_diag(m, TOL),
            Fac::S(_, ax) => *ax == AX_Z,
        }),
    }
}

fn qubits(op: &Ir) -> Vec<usize> {
    match op {
        Ir::U(q, _) | Ir::S(q, _, _) => vec![*q],
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
            Ir::U(q, _) | Ir::S(q, _, _) => (level[k], 1, 0, rank[*q], k),
        }
    };
    let mut idx: Vec<usize> = (0..ir.len()).collect();
    idx.sort_by_key(|&k| key(k));
    idx.into_iter().map(|k| ir[k].clone()).collect()
}

/// A merged block on one pair: one rotation kind set (`Can`: exp(-i/2 (x XX + y YY + z ZZ))), or several two-qubit
/// rotations with the single-qubit operations between them (`Gen`, kept as its parts).
#[derive(Clone, Debug)]
pub enum Blk {
    U(usize, M2),
    S(usize, M2, Vec<Fac>),
    Can(usize, usize, [f64; 3]),
    Gen(usize, usize, Vec<Ir>),
}

/// Step 3's merge (as items 99-101): consecutive two-qubit rotations on one pair, in the order of step 3, become one
/// block; single-qubit operations on the pair's qubits between two of its rotations are taken into the block, those
/// after its last rotation stay outside. A block of commuting rotations alone is `Can`; anything else is `Gen`.
pub fn merge(ir: Vec<Ir>) -> Vec<Blk> {
    struct Open {
        pair: (usize, usize),
        parts: Vec<Ir>,
        tail: Vec<Ir>,
        slot: usize,
    }
    let mut out: Vec<Option<Blk>> = Vec::new();
    let mut opens: Vec<Option<Open>> = Vec::new();
    let mut open_of: BTreeMap<usize, usize> = BTreeMap::new(); // qubit -> index into opens
    fn close(k: usize, opens: &mut Vec<Option<Open>>, open_of: &mut BTreeMap<usize, usize>, out: &mut Vec<Option<Blk>>) {
        if let Some(o) = opens[k].take() {
            for q in [o.pair.0, o.pair.1] {
                if open_of.get(&q) == Some(&k) {
                    open_of.remove(&q);
                }
            }
            let terms: Vec<&Ir> = o.parts.iter().filter(|p| matches!(p, Ir::Pr(..))).collect();
            let commuting = o.parts.iter().all(|p| matches!(p, Ir::Pr(..)));
            out[o.slot] = Some(if terms.len() == 1 || commuting {
                let mut ang = [0.0; 3];
                for p in &o.parts {
                    if let Ir::Pr(_, _, k, t) = p {
                        ang[*k as usize] += t;
                    }
                }
                Blk::Can(o.pair.0, o.pair.1, ang)
            } else {
                Blk::Gen(o.pair.0, o.pair.1, o.parts)
            });
            for t in o.tail {
                match t {
                    Ir::U(q, m) => out.push(Some(Blk::U(q, m))),
                    Ir::S(q, m, f) => out.push(Some(Blk::S(q, m, f))),
                    _ => {}
                }
            }
        }
    }
    for op in ir {
        match op {
            Ir::Pr(a, b, _, _) => {
                let ka = open_of.get(&a).copied();
                if ka.is_some() && ka == open_of.get(&b).copied() {
                    let o = opens[ka.unwrap()].as_mut().unwrap();
                    let tail = std::mem::take(&mut o.tail);
                    o.parts.extend(tail);
                    o.parts.push(op);
                    continue;
                }
                for q in [a, b] {
                    if let Some(k) = open_of.get(&q).copied() {
                        close(k, &mut opens, &mut open_of, &mut out);
                    }
                }
                out.push(None);
                opens.push(Some(Open { pair: (a, b), parts: vec![op], tail: vec![], slot: out.len() - 1 }));
                let k = opens.len() - 1;
                open_of.insert(a, k);
                open_of.insert(b, k);
            }
            Ir::U(q, m) => {
                if let Some(k) = open_of.get(&q).copied() {
                    opens[k].as_mut().unwrap().tail.push(Ir::U(q, m));
                } else {
                    out.push(Some(Blk::U(q, m)));
                }
            }
            Ir::S(q, m, f) => {
                if let Some(k) = open_of.get(&q).copied() {
                    opens[k].as_mut().unwrap().tail.push(Ir::S(q, m, f));
                } else {
                    out.push(Some(Blk::S(q, m, f)));
                }
            }
        }
    }
    let rest: Vec<usize> = (0..opens.len()).filter(|&k| opens[k].is_some()).collect();
    for k in rest {
        close(k, &mut opens, &mut open_of, &mut out);
    }
    out.into_iter().flatten().collect()
}

/// The 4 x 4 matrix of a block's parts on (a, b), a first.
fn block_matrix(a: usize, parts: &[Ir]) -> crate::kak::M4 {
    use crate::kak::{lift, mul4, pp, id4};
    let mut m = id4();
    for p in parts {
        let g = match p {
            Ir::U(q, u) | Ir::S(q, u, _) => lift(u, if *q == a { 0 } else { 1 }),
            Ir::Pr(_, _, k, t) => pp(*k as usize, *t),
        };
        m = mul4(&g, &m);
    }
    m
}

#[derive(Clone, Debug)]
enum Nat {
    U(usize, M2),
    Two(&'static str, usize, usize),
    S(usize, M2, Vec<Fac>),
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
    if nz.len() == 2 {
        // two kinds: CX (X_a I) CX = XX, CX (I Z_b) CX = ZZ, so Can(x, 0, z) = CX (RX_a(x) RZ_b(z)) CX, 2 CZ; the
        // other pairs of kinds by a local Clifford on both qubits (S: X -> Y; RX(-pi/2): Z -> Y), checked in the tests
        let (l, lx, lz): (M2, f64, f64) = match (nz[0], nz[1]) {
            (0, 2) => (eye(), ang[0], ang[2]),
            (1, 2) => (phase(PI / 2.0), ang[1], ang[2]),
            _ => (rx(-PI / 2.0), ang[0], ang[1]),
        };
        let ld = dag2(&l);
        out.push(Nat::U(a, ld));
        out.push(Nat::U(b, ld));
        out.push(Nat::U(b, h()));
        out.push(Nat::Two("cz", a, b));
        out.push(Nat::U(b, h()));
        out.push(Nat::U(a, rx(lx)));
        out.push(Nat::U(b, rz(lz)));
        out.push(Nat::U(b, h()));
        out.push(Nat::Two("cz", a, b));
        out.push(Nat::U(b, h()));
        out.push(Nat::U(a, l));
        out.push(Nat::U(b, l));
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
    slot: BTreeSet<usize>, // qubits whose pending run has a slot (template): pulses by class, not by values
    facs: BTreeMap<usize, Vec<Fac>>, // the pending run's factors (template)
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
                if self.slot.contains(q) {
                    self.facs.entry(*q).or_default().push(Fac::M(*m));
                } else {
                    self.facs.insert(*q, vec![Fac::M(mul2(m, &cur))]);
                }
            }
            Nat::S(q, m, f) => {
                let cur = self.pend.get(q).copied().unwrap_or_else(eye);
                self.pend.insert(*q, mul2(m, &cur));
                if !self.slot.contains(q) {
                    self.facs.insert(*q, vec![Fac::M(cur)]);
                }
                self.facs.entry(*q).or_default().extend(f.iter().cloned());
                self.slot.insert(*q);
            }
            Nat::Two(_, a, b) => {
                let s = self.ready(*a).max(self.ready(*b));
                let dur = self.d.dur_ps(*a, *b).unwrap_or(0) as f64 * 1e-12;
                self.clock.insert(*a, s + dur);
                self.clock.insert(*b, s + dur);
                self.pend.remove(a);
                self.pend.remove(b);
                self.slot.remove(a);
                self.slot.remove(b);
                self.facs.remove(a);
                self.facs.remove(b);
            }
        }
    }
    fn npulses(&self, q: usize) -> usize {
        if self.slot.contains(&q) {
            match self.facs.get(&q).and_then(|f| class_z_any(f)) {
                Some((g, _, _)) => class_z_pulses(&g),
                None => 2,
            }
        } else {
            self.pend.get(&q).map_or(0, pulses)
        }
    }
    fn busy(&self, q: usize) -> bool {
        self.npulses(q) > 0
    }
    fn ready(&self, q: usize) -> f64 {
        self.clock.get(&q).copied().unwrap_or(0.0)
            + self.npulses(q) as f64 * self.d.sx_dur.get(q).copied().unwrap_or(0.0)
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

/// r4: the forms CZ(x, y) = (A on x, B on y) ECR(x, y) (C on x, D on y), up to a global phase, with C and D
/// single-qubit Cliffords: 64 of them (A and B follow from C and D). Built in a fixed order: the 24 Cliffords by
/// closure under H and S, then every (C, D) whose CZ (ECR (C x D))^dag is a product A x B. Item 104's form is one
/// way of writing one of them; tested in `ecr_forms_are_cz`.
fn ecr_forms() -> Vec<(M2, M2, M2, M2)> {
    use crate::kak::{dag4, eq4, id4, kron, mul4, M4};
    let mut cl: Vec<M2> = vec![eye()];
    let mut i = 0;
    while i < cl.len() {
        for g in [h(), phase(PI / 2.0)] {
            let n = mul2(&g, &cl[i]);
            if !cl.iter().any(|m| eq_phase(m, &n, 1e-9)) {
                cl.push(n);
            }
        }
        i += 1;
    }
    // Qiskit's ECRGate on (x, y), x the high bit of the index
    let r = 1.0 / 2f64.sqrt();
    let q = [
        [ZERO, C::new(r, 0.0), ZERO, C::new(0.0, r)],
        [C::new(r, 0.0), ZERO, C::new(0.0, -r), ZERO],
        [ZERO, C::new(0.0, r), ZERO, C::new(r, 0.0)],
        [C::new(0.0, -r), ZERO, C::new(r, 0.0), ZERO],
    ];
    let p = [0usize, 2, 1, 3];
    let mut ecr: M4 = [[ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            ecr[p[i]][p[j]] = q[i][j];
        }
    }
    let mut cz: M4 = id4();
    cz[3][3] = C::new(-1.0, 0.0);
    let mut out = Vec::new();
    for c_ in &cl {
        for d_ in &cl {
            let m = mul4(&cz, &dag4(&mul4(&ecr, &kron(c_, d_))));
            // m = A (x) B read from the rows and columns through its largest entry
            let (mut bi, mut bj) = (0, 0);
            for i in 0..4 {
                for j in 0..4 {
                    if m[i][j].abs() > m[bi][bj].abs() {
                        bi = i;
                        bj = j;
                    }
                }
            }
            let (yi, yj, xi, xj) = (bi & 1, bj & 1, bi >> 1, bj >> 1);
            let am: M2 = [m[yi][yj], m[yi][2 + yj], m[2 + yi][yj], m[2 + yi][2 + yj]];
            let bm: M2 = [m[2 * xi][2 * xj], m[2 * xi][2 * xj + 1], m[2 * xi + 1][2 * xj], m[2 * xi + 1][2 * xj + 1]];
            if eq4(&kron(&am, &bm), &m, 1e-9) {
                let unit = |u: &M2| u.map(|z| z.scale(1.0 / (u[0] * u[3] - u[1] * u[2]).abs().sqrt()));
                out.push((unit(&am), unit(&bm), *c_, *d_));
            }
        }
    }
    out
}

/// r4: each CZ of the CZ-level list becomes an ECR in the form (of `ecr_forms`) that lets the ECR start earliest,
/// given the single-qubit runs pending on its two qubits as the final merge will run them; ties: fewer pulses after
/// it, then the earlier form. One gate at a time, in program order, nothing revisited. (Release 2026-10-17.1 and
/// psfz r1-r3 use item 104's form for every CZ; in the exploratory comparison of 2026-10-18 the extra pulse it leaves
/// on one qubit was what made psfz longer than the release on some Eagle inputs.)
fn ecr_adaptive(d: &Device, nat: Vec<Nat>) -> Vec<Nat> {
    let forms = ecr_forms();
    let mut tr = Track { d, clock: BTreeMap::new(), pend: BTreeMap::new(), slot: BTreeSet::new(), facs: BTreeMap::new() };
    let mut e = Vec::new();
    let after = |tr: &Track, q: usize, u: &M2| -> f64 {
        let cur = tr.pend.get(&q).copied().unwrap_or_else(eye);
        tr.clock.get(&q).copied().unwrap_or(0.0) + pulses(&mul2(u, &cur)) as f64 * d.sx_dur.get(q).copied().unwrap_or(0.0)
    };
    for op in nat {
        match op {
            Nat::Two(_, a, b) => {
                let (x_, y_) = if d.couplers.contains_key(&(a, b)) { (a, b) } else { (b, a) };
                let mut best: Option<(f64, usize, usize)> = None;
                for (i, (am, bm, cm, dm)) in forms.iter().enumerate() {
                    let s = after(&tr, x_, cm).max(after(&tr, y_, dm));
                    let post = pulses(am) + pulses(bm);
                    let better = match best {
                        None => true,
                        Some((bs, bp, _)) => s < bs - 1e-15 || ((s - bs).abs() <= 1e-15 && post < bp),
                    };
                    if better {
                        best = Some((s, post, i));
                    }
                }
                let (am, bm, cm, dm) = forms[best.unwrap().2];
                for o in [Nat::U(x_, cm), Nat::U(y_, dm), Nat::Two("ecr", x_, y_), Nat::U(x_, am), Nat::U(y_, bm)] {
                    tr.apply(&o);
                    e.push(o);
                }
            }
            u => {
                tr.apply(&u);
                e.push(u);
            }
        }
    }
    e
}

/// Item 104's form: CZ(x, y) = [rz(3pi/4), sx, rz(pi/2) on y] ECR(x, y) [sx, sx, rz(-pi/2) on x; rz(pi/2), sx,
/// rz(-pi/4) on y], up to a global phase. (r4: kept for its test; the compiler now chooses among `ecr_forms`.)
#[cfg(test)]
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

/// A run with a slot in the general form rz(l) sx rz(t + pi) sx rz(p + pi) (ZYZ angles p, t, l of its matrix); before
/// a measurement (`last`) without the final rz. No angle is dropped, whatever its value.
pub fn zsx_general(m: &M2, last: bool) -> Vec<(&'static str, Vec<f64>)> {
    let (p, t, l) = zyz(m);
    let mut v = vec![("rz", vec![wrap(l)]), ("sx", vec![]), ("rz", vec![wrap(t + PI)]), ("sx", vec![])];
    if !last {
        v.push(("rz", vec![wrap(p + PI)]));
    }
    v
}

pub fn compile(c: &Circuit, d: &Device, l: &Lattice, template: bool) -> Result<Output, String> {
    if template && d.gate2 == "ecr" {
        return Err("templates are built for CZ devices only (r5 prototype)".into());
    }
    let (ir, meas, slots) = lower_input(c, template)?;
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
    let mut track = Track { d, clock: BTreeMap::new(), pend: BTreeMap::new(), slot: BTreeSet::new(), facs: BTreeMap::new() };
    for op in &blocks {
        let start = nat.len();
        match op {
            Blk::U(q, m) => nat.push(Nat::U(layout[*q], *m)),
            Blk::S(q, m, f) => nat.push(Nat::S(layout[*q], *m, f.clone())),
            Blk::Gen(a, b, parts) => {
                let (pa, pb) = (layout[*a], layout[*b]);
                // a block with a slot is lowered part by part (its KAK form would depend on the values)
                let kak = if parts.iter().any(|p| matches!(p, Ir::S(..))) {
                    None
                } else {
                    crate::kak::kak(&block_matrix(*a, parts))
                };
                match kak {
                    Some(k) => {
                        nat.push(Nat::U(pa, k.b1));
                        nat.push(Nat::U(pb, k.b2));
                        let (role, other) = track.pick(pa, pb);
                        // Can(x, y, z) is the same for either order of its qubits
                        if (role, other) == (pa, pb) {
                            lower_can(pa, pb, k.ang, &mut nat);
                        } else {
                            lower_can(pb, pa, k.ang, &mut nat);
                        }
                        nat.push(Nat::U(pa, k.a1));
                        nat.push(Nat::U(pb, k.a2));
                    }
                    None => {
                        // the decomposition did not hold: the parts one by one (exact, more CZ)
                        for p in parts {
                            match p {
                                Ir::U(q, m) => nat.push(Nat::U(layout[*q], *m)),
                                Ir::S(q, m, f) => nat.push(Nat::S(layout[*q], *m, f.clone())),
                                Ir::Pr(x_, y_, k, t) => {
                                    let mut ang = [0.0; 3];
                                    ang[*k as usize] = *t;
                                    lower_can(layout[*x_], layout[*y_], ang, &mut nat);
                                }
                            }
                        }
                    }
                }
            }
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
        nat = ecr_adaptive(d, nat);
    } else {
        nat = nat
            .into_iter()
            .map(|op| match op {
                Nat::Two(g, a, b) if !d.couplers.contains_key(&(a, b)) => Nat::Two(g, b, a),
                o => o,
            })
            .collect();
    }
    // step 6 (a pending run: its matrix, its factors, whether it has a slot)
    type Pend = (M2, Vec<Fac>, bool);
    let mut pend: BTreeMap<usize, Pend> = BTreeMap::new();
    let mut ops: Vec<(String, Vec<usize>, Vec<f64>)> = Vec::new();
    let mut runs: Vec<RunMap> = Vec::new();
    let emit = |q: usize, run: Pend, last: bool, ops: &mut Vec<(String, Vec<usize>, Vec<f64>)>, runs: &mut Vec<RunMap>| -> Result<(), String> {
        let (m, facs, has_slot) = run;
        if has_slot {
            if let Some((g, terms, left)) = class_z_any(&facs) {
                // the run is G · rz(φ) (or rz(φ) · G) for every value; φ at these values from G^† M (or M G^†)
                let d = if left { mul2(&m, &dag2(&g)) } else { mul2(&dag2(&g), &m) };
                if d[1].abs() > 1e-9 || d[2].abs() > 1e-9 {
                    return Err("internal: a class-Z run is not G · rz(φ)".into());
                }
                let phi = (d[3].div(d[0])).arg();
                let (p, t, l) = zyz(&g);
                // (name, angle, depends on φ): the forms of zsx with l -> l + φ (right) or p -> p + φ (left)
                let (r, lf) = (!left, left);
                let mut seq: Vec<(&str, f64, bool)> = if t.abs() < TOL {
                    vec![("rz", p + l, true)]
                } else if (t - PI / 2.0).abs() < TOL {
                    vec![("rz", l - PI / 2.0, r), ("sx", 0.0, false), ("rz", p + PI / 2.0, lf)]
                } else if (t - PI).abs() < TOL {
                    vec![("rz", l + PI, r), ("x", 0.0, false), ("rz", p, lf)]
                } else {
                    vec![("rz", l, r), ("sx", 0.0, false), ("rz", t + PI, false), ("sx", 0.0, false), ("rz", p + PI, lf)]
                };
                if last && seq.last().map_or(false, |o| o.0 == "rz") {
                    seq.pop();
                }
                let mut zidx = None;
                let mut base = 0.0;
                for (n, a, dep) in seq {
                    if n != "rz" {
                        ops.push((n.to_string(), vec![q], vec![]));
                    } else if dep {
                        zidx = Some(ops.len());
                        base = a;
                        ops.push((n.to_string(), vec![q], vec![wrap(a + phi)]));
                    } else if wrap(a).abs() >= TOL {
                        // a fixed angle: dropped when zero (structural: it depends on G only)
                        ops.push((n.to_string(), vec![q], vec![wrap(a)]));
                    }
                }
                runs.push(RunMap { qubit: q, facs, rz: Vec::new(), zform: Some((zidx, base, terms)) });
            } else {
                let mut rz = Vec::new();
                for (n, a) in zsx_general(&m, last) {
                    if n == "rz" {
                        rz.push(ops.len());
                    }
                    ops.push((n.to_string(), vec![q], a));
                }
                runs.push(RunMap { qubit: q, facs, rz, zform: None });
            }
        } else {
            let mut seq = zsx(&m);
            if last && seq.last().map_or(false, |(n, _)| *n == "rz") {
                seq.pop();
            }
            for (n, a) in seq {
                ops.push((n.to_string(), vec![q], a));
            }
        }
        Ok(())
    };
    for op in nat {
        match op {
            Nat::U(q, m) => {
                let cur = pend.remove(&q).unwrap_or_else(|| (eye(), Vec::new(), false));
                let mut f = cur.1;
                if template {
                    f.push(Fac::M(m));
                }
                pend.insert(q, (mul2(&m, &cur.0), f, cur.2));
            }
            Nat::S(q, m, sf) => {
                let cur = pend.remove(&q).unwrap_or_else(|| (eye(), Vec::new(), false));
                let mut f = cur.1;
                f.extend(sf);
                pend.insert(q, (mul2(&m, &cur.0), f, true));
            }
            Nat::Two(g, a, b) => {
                if !d.working(a, b) {
                    return Err(format!("internal: {} on ({}, {}) is not a working coupler", g, a, b));
                }
                for q in [a, b] {
                    if let Some(run) = pend.remove(&q) {
                        emit(q, run, false, &mut ops, &mut runs)?;
                    }
                }
                ops.push((g.to_string(), vec![a, b], vec![]));
            }
        }
    }
    // the last run before a measurement: its final rz is dropped (diagonal before a Z measurement)
    let rest: Vec<usize> = pend.keys().copied().collect();
    for q in rest {
        let run = pend.remove(&q).unwrap();
        emit(q, run, true, &mut ops, &mut runs)?;
    }
    let meas = meas.into_iter().map(|(q, cb)| (layout[q], cb)).collect();
    Ok(Output { kind, layout, ops, meas, nc: c.nc, slots, runs })
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
                    Ir::U(q, u) | Ir::S(q, u, _) => lift(u, *q),
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
                    [PI / 2.0, PI / 2.0, 0.3], [0.0, 0.0, PI / 2.0], [0.7, 0.0, -0.4], [0.0, 1.3, 0.6], [0.9, -0.2, 0.0]] {
            let mut v = Vec::new();
            lower_can(0, 1, ang, &mut v);
            let want = mul4(&pauli4(0, ang[0]), &mul4(&pauli4(1, ang[1]), &pauli4(2, ang[2])));
            assert!(eq4(&mat(&v), &want), "{:?}", ang);
            let kinds = ang.iter().filter(|t| wrap(**t).abs() > TOL).count();
            assert!(v.iter().filter(|o| matches!(o, Nat::Two(..))).count() <= kinds.max(1).min(3).max(if kinds == 1 { 2 } else { kinds }));
        }
    }

    #[test]
    fn general_block() {
        // ZZ, a field, XX, YY on one pair: one block, decomposed, lowered: equal to the parts' product
        let parts = vec![Ir::Pr(0, 1, 2, 0.4), Ir::U(0, rz(0.3)), Ir::Pr(0, 1, 0, -0.9), Ir::U(1, rx(1.1)),
                         Ir::Pr(1, 0, 1, 0.25)];
        let m = block_matrix(0, &parts);
        let k = crate::kak::kak(&m).expect("kak");
        let mut v = vec![Nat::U(0, k.b1), Nat::U(1, k.b2)];
        lower_can(0, 1, k.ang, &mut v);
        v.push(Nat::U(0, k.a1));
        v.push(Nat::U(1, k.a2));
        let mut want = [[ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                want[i][j] = m[i][j];
            }
        }
        assert!(eq4(&mat(&v), &want));
        assert!(v.iter().filter(|o| matches!(o, Nat::Two(..))).count() <= 3);
        let b = merge(parts.clone());
        assert_eq!(b.len(), 1);
        assert!(matches!(b[0], Blk::Gen(..)));
    }

    #[test]
    fn cz_from_ecr() {
        let mut v = Vec::new();
        cz_as_ecr(0, 1, &mut v);
        assert!(eq4(&mat(&v), &cz4()));
    }

    #[test]
    fn ecr_forms_are_cz() {
        let f = ecr_forms();
        assert_eq!(f.len(), 64);
        for (am, bm, cm, dm) in f {
            let v = vec![Nat::U(0, cm), Nat::U(1, dm), Nat::Two("ecr", 0, 1), Nat::U(0, am), Nat::U(1, bm)];
            assert!(eq4(&mat(&v), &cz4()));
        }
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
    fn class_z_rules() {
        let s_ = phase(PI / 2.0);
        // H rx H = rz: class Z, sign +1
        let f = vec![Fac::M(h()), Fac::S(0, AX_X), Fac::M(h())];
        let (g, t) = class_z(&f).expect("H rx H is class Z");
        assert!(eq_phase(&g, &eye(), 1e-12) && t == vec![(0, 1.0)]);
        // S rx S^dag turns x into y: not class Z
        assert!(class_z(&[Fac::M(dag2(&s_)), Fac::S(0, AX_X), Fac::M(s_)]).is_none());
        // rx rz rx: not class Z; a chain of rz is; u3 never is
        assert!(class_z(&[Fac::S(0, AX_X), Fac::S(1, AX_Z), Fac::S(2, AX_X)]).is_none());
        assert!(class_z(&[Fac::S(0, AX_Z), Fac::M(z()), Fac::S(1, AX_Z)]).is_some());
        assert!(class_z(&[Fac::S(0, AX_NONE)]).is_none());
        // M · rx with M x M^dag = z: class Z on the left only
        // (rz(0.4) H) · rx: right axis H x H = z turned by nothing is x (not right); left: rz(.4) H x H rz(-.4) = z
        let fm = mul2(&rz(0.4), &h());
        assert!(class_z(&[Fac::S(0, AX_X), Fac::M(fm)]).is_none());
        let (g, t, left) = class_z_any(&[Fac::S(0, AX_X), Fac::M(fm)]).unwrap();
        assert!(left);
        let m = mul2(&fm, &rx(1.1));
        assert!(eq_phase(&m, &mul2(&rz(t[0].1 * 1.1), &g), 1e-12));
        // X rz X = rz(-θ): sign -1
        assert_eq!(class_z(&[Fac::M(x()), Fac::S(0, AX_Z), Fac::M(x())]).unwrap().1, vec![(0, -1.0)]);
        // the identity G · rz(φ) at random values
        for th in [0.3, -1.7, 2.9] {
            let m = mul2(&h(), &mul2(&rx(th), &h()));
            let (g, _) = class_z(&[Fac::M(h()), Fac::S(0, AX_X), Fac::M(h())]).unwrap();
            assert!(eq_phase(&m, &mul2(&g, &rz(th)), 1e-12));
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
