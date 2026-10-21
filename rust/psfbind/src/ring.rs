//! The ring template of `psf_ai_compile` a22 in Rust (2026-10-21): item 43's choice (`model_template`: the ring
//! template for a ring that does not embed in the device), item 35's builder (`_ring_template_one`) with item 28's
//! reading, logical circuit and N2 schedule, item 31's schedule, item 87's placement (`_chain_lattice`,
//! `_chain_snake`, `_chain_place`) and item 84's choice between the two schedules (`ring_template` by `_better`: an
//! operation the device does not offer or reports at error >= 1 loses, then the as-soon-as-possible duration, then
//! the two-qubit count, a full tie keeping N2). Ported line by line from psf_ai_compile a22 and psf_compile c61 so
//! that the template is the Python's (TOQB's tests compare the gate list, the fixed angles, the runs, the layout and
//! the values' order).
//!
//! Input: one JSON object (`build`): the device (every entry of its Target: name, qubits or null for a global entry,
//! error, duration), the input circuit (name, qubits, clbits, the angle of rz / ry / rx, the 2 x 2 matrix of any other
//! single-qubit operation), the fixed two-qubit blocks (Qiskit's `TwoQubitBasisDecomposer(CZ)` of each sequence of
//! cz / swap, computed and checked in Python: constants given to the builder, not chosen by it), max_error, and
//! optionally the one schedule to build ("n2" or "q4").
//!
//! Output: {"status": "built" | "none" | "refused" | "out_of_scope", ...}. "none" is the Python's None (no line, no
//! schedule). "refused" names what is not ported, and the caller then builds with the Python: a circuit not read as a
//! CZ ring classifier (item 41's other diagonal ring gates; the compile template of item 100), a ring that embeds in
//! the working couplers (item 42; decided here on the written CZ edges by an exact cycle search, where the Python uses
//! item 91's reader and VF2 -- they agree on the tested inputs), a device without cz, a block missing from the table,
//! an ambiguous bridge in the snake. "out_of_scope" (owner, 2026-10-21: formula only): no snake piece holds the ring,
//! so the Python would place it by VF2's search; the template is not built, and the caller does not fall back.

use crate::json::{self, J};
use crate::{mul, zyz, C, M};
use std::collections::{BTreeMap, BTreeSet};

const PI: f64 = std::f64::consts::PI;

/// Why a build stops: Python's None, something not ported, or a case outside the declared scope.
#[derive(Debug)]
pub enum Stop {
    None(String),
    Refused(String),
    /// The owner's rule of 2026-10-21 (formula only): a ring the snake piece cannot hold would need VF2's search;
    /// it is outside the scope, never placed by search.
    OutOfScope(String),
}

fn refused<T>(s: &str) -> Result<T, Stop> {
    Err(Stop::Refused(s.to_string()))
}

// ------------------------------------------------------------------------------------------------ device

#[derive(Clone, Copy, Debug, Default)]
pub struct Props {
    pub err: Option<f64>,
    pub dur: Option<f64>,
}

#[derive(Default, Debug)]
pub struct Entries {
    pub by: BTreeMap<Vec<usize>, Props>,
    pub global: Option<Props>,
}

#[derive(Debug)]
pub struct Device {
    pub n: usize,
    pub ops: BTreeMap<String, Entries>,
}

impl Device {
    fn has(&self, name: &str) -> bool {
        self.ops.contains_key(name)
    }
}

fn usz(j: &J) -> Result<usize, String> {
    match j.num() {
        Some(v) if v >= 0.0 && v.fract() == 0.0 => Ok(v as usize),
        _ => Err("expected a non-negative integer".to_string()),
    }
}

pub fn device_from(j: &J) -> Result<Device, String> {
    let n = usz(j.get("num_qubits").ok_or("device: num_qubits")?)?;
    let mut ops: BTreeMap<String, Entries> = BTreeMap::new();
    for name in j.get("names").and_then(|v| v.arr()).ok_or("device: names")? {
        ops.entry(name.str().ok_or("device: a name")?.to_string()).or_default();
    }
    for e in j.get("entries").and_then(|v| v.arr()).ok_or("device: entries")? {
        let e = e.arr().ok_or("device: an entry")?;
        if e.len() != 4 {
            return Err("device: an entry has 4 fields".into());
        }
        let name = e[0].str().ok_or("device: an entry's name")?.to_string();
        let p = Props { err: e[2].num(), dur: e[3].num() };
        let ent = ops.get_mut(&name).ok_or("device: an entry of an operation not in names")?;
        match e[1].arr() {
            Some(qs) => {
                let qs: Result<Vec<usize>, String> = qs.iter().map(usz).collect();
                ent.by.insert(qs?, p);
            }
            None => ent.global = Some(p),
        }
    }
    Ok(Device { n, ops })
}

// ------------------------------------------------------------------------------------------------ input and blocks

#[derive(Clone, Debug)]
pub struct Op {
    pub name: String,
    pub qs: Vec<usize>,
    pub cs: Vec<usize>,
    pub angle: Option<f64>,
    pub mat: Option<M>,
}

fn mat_from(j: &J) -> Option<M> {
    let v = j.arr()?;
    let f: Vec<f64> = v.iter().filter_map(|x| x.num()).collect();
    if v.len() != 8 || f.len() != 8 {
        return None;
    }
    Some([C::new(f[0], f[1]), C::new(f[2], f[3]), C::new(f[4], f[5]), C::new(f[6], f[7])])
}

fn mat_to(m: &M) -> J {
    J::Arr(m.iter().flat_map(|z| [J::Num(z.re), J::Num(z.im)]).collect())
}

pub fn input_from(j: &J) -> Result<(usize, Vec<Op>), String> {
    let n = usz(j.get("num_qubits").ok_or("input: num_qubits")?)?;
    let mut out = Vec::new();
    for o in j.get("ops").and_then(|v| v.arr()).ok_or("input: ops")? {
        let o = o.arr().ok_or("input: an op")?;
        if o.len() != 5 {
            return Err("input: an op has 5 fields".into());
        }
        let name = o[0].str().ok_or("input: a name")?.to_string();
        let qs: Result<Vec<usize>, String> = o[1].arr().ok_or("input: qubits")?.iter().map(usz).collect();
        let cs: Result<Vec<usize>, String> = o[2].arr().ok_or("input: clbits")?.iter().map(usz).collect();
        out.push(Op { name, qs: qs?, cs: cs?, angle: o[3].num(), mat: mat_from(&o[4]) });
    }
    Ok((n, out))
}

#[derive(Clone, Debug)]
pub enum BlockOp {
    One(usize, M),
    Cz,
}

pub type Blocks = BTreeMap<Vec<String>, Vec<BlockOp>>;

pub fn blocks_from(j: &J) -> Result<Blocks, String> {
    let mut out = BTreeMap::new();
    for b in j.arr().ok_or("blocks")? {
        let b = b.arr().ok_or("blocks: an entry")?;
        let kinds: Vec<String> =
            b[0].arr().ok_or("blocks: kinds")?.iter().filter_map(|v| v.str().map(|s| s.to_string())).collect();
        let mut seq = Vec::new();
        for o in b[1].arr().ok_or("blocks: ops")? {
            let o = o.arr().ok_or("blocks: an op")?;
            match o[0].str() {
                Some("cz") => seq.push(BlockOp::Cz),
                Some("1q") => seq.push(BlockOp::One(usz(&o[1])?, mat_from(&o[2]).ok_or("blocks: a matrix")?)),
                _ => return Err("blocks: an op".into()),
            }
        }
        out.insert(kinds, seq);
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------ item 28: reading

#[derive(Debug)]
pub struct Read {
    pub cycle: Vec<usize>,
    pub segments: Vec<Vec<usize>>, // per layer: the input op indices of its single-qubit operations
    pub tail: Vec<usize>,
    pub meas: Vec<(usize, usize)>,
}

/// `_ring_read`: the CZ ring classifier's layers, or None.
pub fn ring_read(n: usize, ops: &[Op]) -> Option<Read> {
    if n < 4 {
        return None;
    }
    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut block: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut blocks: Vec<BTreeSet<(usize, usize)>> = Vec::new();
    let mut meas = Vec::new();
    let mut touched: BTreeSet<usize> = BTreeSet::new();
    let mut measured: BTreeSet<usize> = BTreeSet::new();
    for (k, op) in ops.iter().enumerate() {
        let nm = op.name.as_str();
        if nm == "barrier" {
            continue;
        }
        if op.cs.len() > (if nm == "measure" { 1 } else { 0 }) {
            return None;
        }
        if nm == "measure" {
            if op.qs.len() != 1 || op.cs.len() != 1 || measured.contains(&op.qs[0]) {
                return None;
            }
            measured.insert(op.qs[0]);
            meas.push((op.qs[0], op.cs[0]));
            continue;
        }
        if op.qs.iter().any(|q| measured.contains(q)) {
            return None;
        }
        if op.qs.len() == 1 {
            if touched.contains(&op.qs[0]) {
                if block.len() != n {
                    return None;
                }
                blocks.push(std::mem::take(&mut block));
                segments.push(std::mem::take(&mut cur));
                touched.clear();
            }
            cur.push(k);
            continue;
        }
        if op.qs.len() != 2 || nm != "cz" {
            return None;
        }
        let e = (op.qs[0].min(op.qs[1]), op.qs[0].max(op.qs[1]));
        if block.contains(&e) {
            return None;
        }
        block.insert(e);
        touched.insert(op.qs[0]);
        touched.insert(op.qs[1]);
    }
    if !block.is_empty() {
        if block.len() != n {
            return None;
        }
        blocks.push(std::mem::take(&mut block));
        segments.push(std::mem::take(&mut cur));
    }
    if blocks.is_empty() || blocks.iter().any(|b| *b != blocks[0]) {
        return None;
    }
    if !measured.is_empty() && measured.len() != n {
        return None;
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in &blocks[0] {
        if b >= n {
            return None;
        }
        adj[a].push(b);
        adj[b].push(a);
    }
    if adj.iter().any(|v| v.len() != 2) {
        return None;
    }
    let mut cycle = vec![0usize];
    let mut nxt = *adj[0].iter().min().unwrap();
    while nxt != 0 {
        cycle.push(nxt);
        let prev = cycle[cycle.len() - 2];
        nxt = *adj[nxt].iter().find(|&&v| v != prev).unwrap();
        if cycle.len() > n {
            return None;
        }
    }
    if cycle.len() != n {
        return None;
    }
    Some(Read { cycle, segments, tail: cur, meas })
}

// ------------------------------------------------------------------------------------------------ N2 and item 31

pub type Rounds = Vec<(Vec<(usize, usize)>, Vec<(usize, usize)>)>;

fn folded(n: usize) -> Vec<usize> {
    let (mut order, mut lo, mut hi) = (Vec::new(), 0i64, n as i64 - 1);
    while lo <= hi {
        order.push(lo as usize);
        if hi != lo {
            order.push(hi as usize);
        }
        lo += 1;
        hi -= 1;
    }
    order
}

fn ring_edges(n: usize) -> BTreeSet<(usize, usize)> {
    (0..n).map(|i| (i.min((i + 1) % n), i.max((i + 1) % n))).collect()
}

fn adjacent(occ: &[usize], edges: &mut BTreeSet<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut czs = Vec::new();
    for p in 0..occ.len() - 1 {
        let e = (occ[p].min(occ[p + 1]), occ[p].max(occ[p + 1]));
        if edges.remove(&e) {
            czs.push((occ[p], occ[p + 1]));
        }
    }
    czs
}

/// `ring_rounds` (N2).
pub fn ring_rounds(n: usize) -> Result<(Vec<usize>, Rounds), String> {
    let order = folded(n);
    let mut edges = ring_edges(n);
    let (mut occ, mut rounds, mut parity) = (order.clone(), Vec::new(), 1u8);
    loop {
        let czs = adjacent(&occ, &mut edges);
        let mut swaps = Vec::new();
        if !edges.is_empty() {
            let mut p = if parity == 1 { 1 } else { 2 };
            while p + 2 < n {
                swaps.push((p, p + 1));
                occ.swap(p, p + 1);
                p += 2;
            }
            parity ^= 1;
        }
        rounds.push((czs, swaps));
        if edges.is_empty() {
            return Ok((order, rounds));
        }
        if rounds.len() > 4 * n {
            return Err("N2: no schedule".into());
        }
    }
}

/// `ring_rounds_q4` (item 31).
pub fn ring_rounds_q4(n: usize) -> Result<(Vec<usize>, Rounds), String> {
    let order = folded(n);
    let mut edges = ring_edges(n);
    let mut occ = order.clone();
    let mut rounds: Rounds = Vec::new();
    let sa: Vec<(usize, usize)> = (1..n - 1).step_by(4).map(|p| (p, p + 1)).collect();
    let sb: Vec<(usize, usize)> = (3..n - 1).step_by(4).map(|p| (p, p + 1)).collect();
    let both: Vec<(usize, usize)> = sa.iter().chain(sb.iter()).cloned().collect();
    let mut czs = adjacent(&occ, &mut edges);
    for sw in [sa, both] {
        if edges.is_empty() {
            break;
        }
        rounds.push((czs, sw.clone()));
        for (p, p1) in sw {
            occ.swap(p, p1);
        }
        czs = adjacent(&occ, &mut edges);
    }
    rounds.push((czs, Vec::new()));
    if !edges.is_empty() {
        return Err("item 31: no schedule".into());
    }
    Ok((order, rounds))
}

// ------------------------------------------------------------------------------------------------ item 28: logical

#[derive(Clone, Debug, PartialEq)]
pub enum L {
    One(usize, usize),               // (input op index, wire)
    Two(&'static str, usize, usize), // cz / swap on two wires
    Measure(usize, usize),           // (wire, clbit)
}

impl L {
    fn wires(&self) -> Vec<usize> {
        match self {
            L::One(_, w) | L::Measure(w, _) => vec![*w],
            L::Two(_, a, b) => vec![*a, *b],
        }
    }
}

struct Lines {
    wire_at: Vec<usize>,
    ridx: Vec<usize>,
    pos: Vec<usize>,
    occ: Vec<usize>,
    out: Vec<L>,
}

impl Lines {
    fn here(&self, q: usize) -> usize {
        self.wire_at[self.pos[self.ridx[q]]]
    }
    fn swaps(&mut self, sw: &[(usize, usize)]) {
        for &(p, p1) in sw {
            self.out.push(L::Two("swap", self.wire_at[p], self.wire_at[p1]));
            let (a, b) = (self.occ[p], self.occ[p1]);
            self.occ[p] = b;
            self.occ[p1] = a;
            self.pos[a] = p1;
            self.pos[b] = p;
        }
    }
    fn czs(&mut self, cz: &[(usize, usize)]) {
        for &(a, b) in cz {
            self.out.push(L::Two("cz", self.wire_at[self.pos[a]], self.wire_at[self.pos[b]]));
        }
    }
}

/// `_ring_logical`: (the logical circuit on the input's wires, p_of_wire), Ok(None) for an unmeasured ring with an
/// odd number of layers; Err when the schedule fails (the Python's RuntimeError).
pub fn ring_logical(n: usize, ops: &[Op], read: &Read, q4: bool) -> Result<Option<(Vec<L>, Vec<usize>)>, String> {
    let (order, rounds) = if q4 { ring_rounds_q4(n)? } else { ring_rounds(n)? };
    if read.meas.is_empty() && read.segments.len() % 2 == 1 {
        return Ok(None);
    }
    let mut ridx = vec![0usize; n];
    for (i, &q) in read.cycle.iter().enumerate() {
        ridx[q] = i;
    }
    let wire_at: Vec<usize> = (0..n).map(|p| read.cycle[order[p]]).collect();
    let mut p_of_wire = vec![0usize; n];
    for (p, &w) in wire_at.iter().enumerate() {
        p_of_wire[w] = p;
    }
    let mut pos = vec![0usize; n];
    for p in 0..n {
        pos[order[p]] = p;
    }
    let mut s = Lines { wire_at, ridx, pos, occ: order.clone(), out: Vec::new() };
    for (layer, seg) in read.segments.iter().enumerate() {
        for &k in seg {
            let w = s.here(ops[k].qs[0]);
            s.out.push(L::One(k, w));
        }
        if layer % 2 == 0 {
            for (cz, sw) in &rounds {
                s.czs(cz);
                s.swaps(sw);
            }
        } else {
            for (cz, sw) in rounds.iter().rev() {
                let rev: Vec<(usize, usize)> = sw.iter().rev().cloned().collect();
                s.swaps(&rev);
                s.czs(cz);
            }
        }
    }
    for &k in &read.tail {
        let w = s.here(ops[k].qs[0]);
        s.out.push(L::One(k, w));
    }
    for &(q, cb) in &read.meas {
        let w = s.here(q);
        s.out.push(L::Measure(w, cb));
    }
    Ok(Some((s.out, p_of_wire)))
}

// ------------------------------------------------------------------------------------------------ item 87: placement

fn round_ps(d: f64) -> i64 {
    (d * 1e12).round_ties_even() as i64 // Python's round(): halves to even
}

pub struct Lattice {
    pub k: BTreeSet<usize>,
    pub kadj: BTreeMap<usize, BTreeSet<usize>>,
    pub fadj: BTreeMap<usize, BTreeSet<usize>>,
    pub snake: Vec<usize>,
}

/// `_chain_lattice`.
pub fn chain_lattice(dev: &Device, max_error: f64) -> Result<Option<Lattice>, Stop> {
    let nq = dev.n;
    let cz = match dev.ops.get("cz") {
        Some(e) => e,
        None => return refused("the device has no cz (item 87 reads target['cz'])"),
    };
    let mut err: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut dur: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for (qa, pr) in &cz.by {
        if qa.len() != 2 || qa[0] >= nq || qa[1] >= nq {
            return refused("a cz entry not on two device qubits");
        }
        let k = (qa[0].min(qa[1]), qa[0].max(qa[1]));
        let e = pr.err.unwrap_or(0.0);
        let d = pr.dur.unwrap_or(0.0);
        let ce = err.entry(k).or_insert(0.0);
        *ce = ce.max(e);
        let cd = dur.entry(k).or_insert(0.0);
        *cd = cd.max(d);
    }
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nq];
    for &(a, b) in err.keys() {
        adj[a].insert(b);
        adj[b].insert(a);
    }
    let mut bad: BTreeSet<usize> = BTreeSet::new();
    for name in ["sx", "measure"] {
        if let Some(ent) = dev.ops.get(name) {
            for (qa, pr) in &ent.by {
                if let Some(e) = pr.err {
                    if e >= max_error {
                        bad.insert(qa[0]);
                    }
                }
            }
        }
    }
    let mut gadj: BTreeMap<usize, BTreeSet<usize>> =
        (0..nq).filter(|q| !bad.contains(q)).map(|q| (q, BTreeSet::new())).collect();
    for (&(a, b), &e) in &err {
        if e < max_error && !bad.contains(&a) && !bad.contains(&b) {
            gadj.get_mut(&a).unwrap().insert(b);
            gadj.get_mut(&b).unwrap().insert(a);
        }
    }
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut comps: Vec<BTreeSet<usize>> = Vec::new();
    for &q in gadj.keys() {
        if seen.contains(&q) {
            continue;
        }
        let mut comp: BTreeSet<usize> = [q].into_iter().collect();
        let mut fr = vec![q];
        while let Some(u) = fr.pop() {
            for &v in &gadj[&u] {
                if comp.insert(v) {
                    fr.push(v);
                }
            }
        }
        seen.extend(comp.iter().cloned());
        comps.push(comp);
    }
    if comps.is_empty() {
        return Ok(None);
    }
    // max by (len, -min): the first maximal
    let mut kbest = 0;
    for i in 1..comps.len() {
        let (a, b) = (&comps[i], &comps[kbest]);
        let ka = (a.len(), -(*a.iter().next().unwrap() as i64));
        let kb = (b.len(), -(*b.iter().next().unwrap() as i64));
        if ka > kb {
            kbest = i;
        }
    }
    let k = comps.swap_remove(kbest);
    let kadj: BTreeMap<usize, BTreeSet<usize>> =
        k.iter().map(|&q| (q, gadj[&q].iter().filter(|v| k.contains(v)).cloned().collect())).collect();
    let mut lens: BTreeMap<i64, usize> = BTreeMap::new();
    for &a in &k {
        for &b in &kadj[&a] {
            if a < b {
                *lens.entry(round_ps(dur[&(a, b)])).or_insert(0) += 1;
            }
        }
    }
    // max by (count, -length): the largest count, ties the shortest (ascending iteration, strict >)
    let mut modal: Option<(i64, usize)> = None;
    for (&x, &c) in &lens {
        if modal.map_or(true, |(_, mc)| c > mc) {
            modal = Some((x, c));
        }
    }
    let fadj: BTreeMap<usize, BTreeSet<usize>> = k
        .iter()
        .map(|&q| {
            let s = kadj[&q]
                .iter()
                .filter(|&&v| modal.map_or(false, |(m, _)| round_ps(dur[&(q.min(v), q.max(v))]) == m))
                .cloned()
                .collect();
            (q, s)
        })
        .collect();
    let snake = chain_snake(nq, &adj)?;
    Ok(Some(Lattice { k, kadj, fadj, snake }))
}

/// `_chain_snake`.
pub fn chain_snake(nq: usize, adj: &[BTreeSet<usize>]) -> Result<Vec<usize>, Stop> {
    if nq == 0 {
        return Ok(Vec::new());
    }
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut cur = vec![0usize];
    for q in 1..nq {
        if adj[q - 1].contains(&q) {
            cur.push(q);
        } else {
            rows.push(std::mem::replace(&mut cur, vec![q]));
        }
    }
    rows.push(cur);
    let rows: Vec<Vec<usize>> = rows.into_iter().filter(|r| r.len() > 2).collect();
    let mut inrow: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        for (j, &q) in r.iter().enumerate() {
            inrow.insert(q, (i, j));
        }
    }
    let mut path: Vec<usize> = Vec::new();
    let mut col: usize = 0;
    for (i, r) in rows.iter().enumerate() {
        let leftward = i % 2 == 1;
        let (exit_col, bq): (usize, Option<usize>);
        if i + 1 < rows.len() {
            let mut br: Vec<(usize, usize)> = Vec::new();
            for b in 0..nq {
                if inrow.contains_key(&b) {
                    continue;
                }
                let to_next = adj[b].iter().any(|c| inrow.get(c).map_or(false, |&(ri, _)| ri == i + 1));
                for a in &adj[b] {
                    if let Some(&(ri, j)) = inrow.get(a) {
                        if ri == i && to_next {
                            br.push((j, b));
                        }
                    }
                }
            }
            if br.is_empty() {
                path.extend(r.iter().cloned());
                return Ok(path);
            }
            let pick = if leftward { *br.iter().min().unwrap() } else { *br.iter().max().unwrap() };
            exit_col = pick.0;
            bq = Some(pick.1);
        } else {
            exit_col = if leftward { 0 } else { r.len() - 1 };
            bq = None;
        }
        let start = if i > 0 { col } else { 0 };
        if leftward {
            let mut j = start as i64;
            while j >= exit_col as i64 {
                if j as usize >= r.len() {
                    return refused("the snake indexes past a row");
                }
                path.push(r[j as usize]);
                j -= 1;
            }
        } else {
            let mut j = start;
            while j <= exit_col {
                if j >= r.len() {
                    return refused("the snake indexes past a row");
                }
                path.push(r[j]);
                j += 1;
            }
        }
        if let Some(bq) = bq {
            path.push(bq);
            let below: Vec<usize> =
                adj[bq].iter().filter(|c| inrow.get(c).map_or(false, |&(ri, _)| ri == i + 1)).cloned().collect();
            if below.len() != 1 {
                return refused("a bridge with more than one qubit below (the Python's set order decides)");
            }
            col = inrow[&below[0]].1;
        }
    }
    Ok(path)
}

/// `_chain_place`: the longest surviving snake piece, centred. Where the Python would fall back to VF2's search, this
/// stops with OutOfScope (the owner's rule of 2026-10-21: formula only, no search).
pub fn chain_place(n: usize, k: &BTreeSet<usize>, kadj: &BTreeMap<usize, BTreeSet<usize>>, snake: &[usize])
    -> Result<Vec<usize>, Stop> {
    let mut pieces: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for &q in snake {
        if k.contains(&q) && (cur.is_empty() || kadj[cur.last().unwrap()].contains(&q)) {
            cur.push(q);
        } else {
            if !cur.is_empty() {
                pieces.push(std::mem::take(&mut cur));
            }
            cur = if k.contains(&q) { vec![q] } else { Vec::new() };
        }
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    if !pieces.is_empty() {
        let index = |q: usize| snake.iter().position(|&x| x == q).unwrap() as i64;
        let mut bi = 0;
        for i in 1..pieces.len() {
            let ka = (pieces[i].len(), -index(pieces[i][0]));
            let kb = (pieces[bi].len(), -index(pieces[bi][0]));
            if ka > kb {
                bi = i;
            }
        }
        let best = &pieces[bi];
        if best.len() >= n {
            let s = (best.len() - n) / 2;
            return Ok(best[s..s + n].to_vec());
        }
    }
    Err(Stop::OutOfScope("no snake piece long enough: the placement would need VF2's search (outside the scope)".into()))
}

// ------------------------------------------------------------------------------------------------ item 42: embedding

/// True when the n-cycle of the ring embeds in the device's working couplers (item 96's graph: couplers of every
/// two-qubit operation, without failed qubits (sx or readout error >= max_error) or couplers failed in either
/// direction (the first of cz, ecr, cx)): an exact search for a simple cycle of length n.
pub fn ring_embeds(dev: &Device, n: usize, max_error: f64) -> Result<bool, Stop> {
    let mut cm: BTreeSet<(usize, usize)> = BTreeSet::new();
    for e in dev.ops.values() {
        if e.global.is_some() && e.by.keys().any(|q| q.len() == 2) {
            return refused("a global two-qubit operation (the coupling map is not built from entries)");
        }
        for q in e.by.keys() {
            if q.len() == 2 {
                cm.insert((q[0], q[1]));
            }
        }
    }
    let mut fq: BTreeSet<usize> = BTreeSet::new();
    for name in ["sx", "measure"] {
        if let Some(ent) = dev.ops.get(name) {
            for (qa, pr) in &ent.by {
                if pr.err.map_or(false, |e| e >= max_error) {
                    fq.insert(qa[0]);
                }
            }
        }
    }
    let mut fedges: BTreeSet<(usize, usize)> = BTreeSet::new();
    if let Some(g) = ["cz", "ecr", "cx"].iter().find(|g| dev.has(g)) {
        for (qa, pr) in &dev.ops[*g].by {
            if qa.len() == 2 && pr.err.map_or(false, |e| e >= max_error) {
                fedges.insert((qa[0], qa[1]));
            }
        }
    }
    let mut und: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(a, b) in &cm {
        if fq.contains(&a) || fq.contains(&b) || fedges.contains(&(a, b)) || fedges.contains(&(b, a)) {
            continue;
        }
        und.insert((a.min(b), a.max(b)));
    }
    let working = dev.n as i64 - fq.len() as i64;
    if n as i64 > working || und.is_empty() {
        return Ok(false);
    }
    let mut g: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(a, b) in &und {
        g.entry(a).or_default().push(b);
        g.entry(b).or_default().push(a);
    }
    let mut budget: u64 = 200_000_000;
    fn dfs(g: &BTreeMap<usize, Vec<usize>>, s: usize, u: usize, depth: usize, n: usize, on: &mut BTreeSet<usize>,
           budget: &mut u64) -> Option<bool> {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        if depth == n {
            return Some(g[&u].contains(&s));
        }
        for &v in &g[&u] {
            if v > s && !on.contains(&v) {
                on.insert(v);
                let r = dfs(g, s, v, depth + 1, n, on, budget);
                on.remove(&v);
                match r {
                    Some(true) => return Some(true),
                    None => return None,
                    _ => {}
                }
            }
        }
        Some(false)
    }
    for &s in g.keys() {
        let mut on: BTreeSet<usize> = [s].into_iter().collect();
        match dfs(&g, s, s, 1, n, &mut on, &mut budget) {
            Some(true) => return Ok(true),
            None => return refused("the cycle search ran out of its budget"),
            _ => {}
        }
    }
    Ok(false)
}

// ------------------------------------------------------------------------------------------------ item 35: the template

#[derive(Clone, Debug)]
pub enum Factor {
    Fixed(M),
    Var(usize, String),
}

#[derive(Clone, Debug)]
pub enum Param {
    None,
    Num(f64),
    Var(usize),
}

#[derive(Clone, Debug)]
pub struct OutOp {
    pub name: &'static str,
    pub qs: Vec<usize>,
    pub cs: Vec<usize>,
    pub param: Param,
}

#[derive(Clone, Debug)]
pub struct Template {
    pub schedule: &'static str,
    pub num_qubits: usize,
    pub ops: Vec<OutOp>,
    pub var_runs: Vec<Vec<Factor>>,
    pub var_kinds: Vec<(usize, String)>,
    pub layout: Vec<usize>,
    pub use_rx: bool,
    pub nvar: usize,
}

enum Item {
    Run(usize, Vec<Factor>),
    Cz(usize, usize),
    Measure(usize, usize),
}

fn flush(p: usize, pending: &mut BTreeMap<usize, Vec<Factor>>, items: &mut Vec<Item>) {
    let v = pending.get_mut(&p).unwrap();
    if !v.is_empty() {
        items.push(Item::Run(p, std::mem::take(v)));
    }
}

fn op(name: &'static str, qs: Vec<usize>, param: Param) -> OutOp {
    OutOp { name, qs, cs: Vec::new(), param }
}

/// `_ring_template_one` for a circuit already read as a ring (`read`), on the device's line `line`.
pub fn template_one(dev: &Device, n: usize, ops: &[Op], read: &Read, line: &[usize], blocks: &Blocks, q4: bool)
    -> Result<Template, Stop> {
    let built = ring_logical(n, ops, read, q4).map_err(Stop::None)?;
    let (logical, p_of_wire) = match built {
        Some(b) => b,
        None => return Err(Stop::None("an unmeasured ring with an odd number of layers".into())),
    };
    let phys: Vec<usize> = (0..n).map(|w| line[p_of_wire[w]]).collect();
    let src: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, o)| !matches!(o.name.as_str(), "barrier" | "measure" | "cz"))
        .map(|(k, _)| k)
        .collect();
    // pending in phys's order (the Python's dict order: the final flush goes wire 0, 1, ...)
    let mut pending: BTreeMap<usize, Vec<Factor>> = phys.iter().map(|&p| (p, Vec::new())).collect();
    let mut items: Vec<Item> = Vec::new();
    let mut used = vec![false; logical.len()];
    let mut si = 0usize;
    let mut var_kinds: Vec<(usize, String)> = Vec::new();
    for i in 0..logical.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        match &logical[i] {
            L::One(_, w) => {
                if si >= src.len() {
                    return Err(Stop::None("more single-qubit operations than the input's".into()));
                }
                let o = &ops[src[si]];
                let p = phys[*w];
                if matches!(o.name.as_str(), "rz" | "ry" | "rx") {
                    if o.angle.is_none() {
                        return refused("a parameterised rotation in the input");
                    }
                    pending.get_mut(&p).unwrap().push(Factor::Var(var_kinds.len(), o.name.clone()));
                    var_kinds.push((src[si], o.name.clone()));
                } else {
                    match o.mat {
                        Some(m) => pending.get_mut(&p).unwrap().push(Factor::Fixed(m)),
                        None => return refused("a fixed single-qubit operation without its matrix"),
                    }
                }
                si += 1;
            }
            L::Measure(w, cb) => {
                let p = phys[*w];
                flush(p, &mut pending, &mut items);
                items.push(Item::Measure(p, *cb));
            }
            L::Two(nm, w1, w2) => {
                let (a, b) = ((*w1).min(*w2), (*w1).max(*w2));
                let mut kinds: Vec<String> = vec![nm.to_string()];
                for j in i + 1..logical.len() {
                    if used[j] {
                        continue;
                    }
                    let wj = logical[j].wires();
                    if !wj.contains(&a) && !wj.contains(&b) {
                        continue;
                    }
                    if let L::Two(nj, x, y) = &logical[j] {
                        if (*x).min(*y) == a && (*x).max(*y) == b {
                            kinds.push(nj.to_string());
                            used[j] = true;
                            continue;
                        }
                    }
                    break;
                }
                let (pa, pb) = (phys[a], phys[b]);
                let seq = match blocks.get(&kinds) {
                    Some(s) => s,
                    None => return refused(&format!("the block {:?} is not in the table", kinds)),
                };
                for bo in seq {
                    match bo {
                        BlockOp::Cz => {
                            flush(pa, &mut pending, &mut items);
                            flush(pb, &mut pending, &mut items);
                            items.push(Item::Cz(pa, pb));
                        }
                        BlockOp::One(wq, m) => {
                            pending.get_mut(&(if *wq == 0 { pa } else { pb })).unwrap().push(Factor::Fixed(*m));
                        }
                    }
                }
            }
        }
    }
    for w in 0..n {
        flush(phys[w], &mut pending, &mut items);
    }
    if si != src.len() {
        return Err(Stop::None("fewer single-qubit operations than the input's".into()));
    }
    let use_rx = dev.has("rx");
    let half = PI / 2.0;
    let isvar = |f: &Factor| matches!(f, Factor::Var(..));
    let nvar = items.iter().filter(|it| matches!(it, Item::Run(_, fs) if fs.iter().any(isvar))).count();
    let mut out: Vec<OutOp> = Vec::new();
    let mut var_runs: Vec<Vec<Factor>> = Vec::new();
    let mut k = 0usize;
    for it in items {
        match it {
            Item::Cz(a, b) => out.push(op("cz", vec![a, b], Param::None)),
            Item::Measure(p, c) => out.push(OutOp { name: "measure", qs: vec![p], cs: vec![c], param: Param::None }),
            Item::Run(p, factors) => {
                let mut merged: Vec<Factor> = Vec::new();
                let mut acc: Option<M> = None;
                for f in factors {
                    match f {
                        Factor::Var(..) => {
                            if let Some(a) = acc.take() {
                                merged.push(Factor::Fixed(a));
                            }
                            merged.push(f);
                        }
                        Factor::Fixed(m) => {
                            acc = Some(match acc {
                                None => m,
                                Some(a) => mul(&m, &a),
                            });
                        }
                    }
                }
                if let Some(a) = acc {
                    merged.push(Factor::Fixed(a));
                }
                if merged.iter().any(isvar) {
                    var_runs.push(merged);
                    if use_rx {
                        out.push(op("rz", vec![p], Param::Var(k)));
                        out.push(op("rx", vec![p], Param::Var(k + 1)));
                        out.push(op("rz", vec![p], Param::Var(k + 2)));
                    } else {
                        out.push(op("rz", vec![p], Param::Var(k)));
                        out.push(op("sx", vec![p], Param::None));
                        out.push(op("rz", vec![p], Param::Var(k + 1)));
                        out.push(op("sx", vec![p], Param::None));
                        out.push(op("rz", vec![p], Param::Var(k + 2)));
                    }
                    k += 3;
                } else {
                    let m = match &merged[0] {
                        Factor::Fixed(m) => *m,
                        _ => unreachable!(),
                    };
                    let (theta, phi, lam) = zyz(&m);
                    if theta.abs() < 1e-12 {
                        if ((phi + lam + PI).rem_euclid(2.0 * PI) - PI).abs() > 1e-12 {
                            out.push(op("rz", vec![p], Param::Num(phi + lam)));
                        }
                    } else if use_rx {
                        out.push(op("rz", vec![p], Param::Num(lam - half)));
                        out.push(op("rx", vec![p], Param::Num(theta)));
                        out.push(op("rz", vec![p], Param::Num(phi + half)));
                    } else if (theta - half).abs() < 1e-12 {
                        out.push(op("rz", vec![p], Param::Num(lam - half)));
                        out.push(op("sx", vec![p], Param::None));
                        out.push(op("rz", vec![p], Param::Num(phi + half)));
                    } else {
                        out.push(op("rz", vec![p], Param::Num(lam)));
                        out.push(op("sx", vec![p], Param::None));
                        out.push(op("rz", vec![p], Param::Num(theta + PI)));
                        out.push(op("sx", vec![p], Param::None));
                        out.push(op("rz", vec![p], Param::Num(phi + PI)));
                    }
                }
            }
        }
    }
    Ok(Template {
        schedule: if q4 { "q4" } else { "n2" },
        num_qubits: dev.n,
        ops: out,
        var_runs,
        var_kinds,
        layout: phys,
        use_rx,
        nvar,
    })
}

// ------------------------------------------------------------------------------------------------ item 84: the choice

/// `_st_cost` is infinite: an operation not offered, not on these qubits (and no global entry), or at error >= 1.
fn cost_inf(dev: &Device, ops: &[OutOp]) -> bool {
    for o in ops {
        let Some(ent) = dev.ops.get(o.name) else {
            return true;
        };
        match ent.by.get(&o.qs) {
            Some(p) => {
                if p.err.unwrap_or(0.0) >= 1.0 {
                    return true;
                }
            }
            None => {
                if ent.global.is_none() {
                    return true;
                }
            }
        }
    }
    false
}

/// `_duration_s`: the as-soon-as-possible makespan on the reported durations, None when one is missing.
pub fn duration_s(dev: &Device, ops: &[OutOp]) -> Option<f64> {
    let mut clock: BTreeMap<usize, f64> = BTreeMap::new();
    for o in ops {
        let ent = dev.ops.get(o.name)?;
        let props = match ent.by.get(&o.qs) {
            Some(p) => Some(*p),
            None => ent.global,
        };
        let d = match props.and_then(|p| p.dur) {
            Some(d) => d,
            None if o.name == "rz" => 0.0,
            None => return None,
        };
        let s = o.qs.iter().map(|x| *clock.get(x).unwrap_or(&0.0)).fold(f64::NEG_INFINITY, f64::max);
        let s = if o.qs.is_empty() { 0.0 } else { s };
        for &x in &o.qs {
            clock.insert(x, s + d);
        }
    }
    Some(clock.values().cloned().fold(0.0f64, |a, b| if b > a { b } else { a }))
}

/// `_duration_cx`: the makespan in CX units (cx 1, swap 3, rz 0, other one-qubit 0.3, other two-qubit 1).
pub fn duration_cx(ops: &[OutOp]) -> f64 {
    let mut clock: BTreeMap<usize, f64> = BTreeMap::new();
    for o in ops {
        let d = match o.name {
            "cx" => 1.0,
            "swap" => 3.0,
            "rz" => 0.0,
            _ => {
                if o.qs.len() >= 2 {
                    1.0
                } else {
                    0.3
                }
            }
        };
        let s = o.qs.iter().map(|x| *clock.get(x).unwrap_or(&0.0)).fold(f64::NEG_INFINITY, f64::max);
        let s = if o.qs.is_empty() { 0.0 } else { s };
        for &x in &o.qs {
            clock.insert(x, s + d);
        }
    }
    clock.values().cloned().fold(0.0f64, |a, b| if b > a { b } else { a })
}

fn count_2q(ops: &[OutOp]) -> usize {
    ops.iter().filter(|o| o.qs.len() == 2).count()
}

/// `_better(a, b)`: True when `a` is to replace `b`.
pub fn better(dev: &Device, a: &[OutOp], b: &[OutOp]) -> bool {
    let (fa, fb) = (cost_inf(dev, a), cost_inf(dev, b));
    if fa != fb {
        return fb;
    }
    let (da, db, tol) = match (duration_s(dev, a), duration_s(dev, b)) {
        (Some(x), Some(y)) => (x, y, 1e-12),
        _ => (duration_cx(a), duration_cx(b), 1e-9),
    };
    if (da - db).abs() > tol {
        return da < db;
    }
    let (na, nb) = (count_2q(a), count_2q(b));
    if na != nb {
        return na < nb;
    }
    false
}

// ------------------------------------------------------------------------------------------------ item 43: the entry

pub struct Candidate {
    pub schedule: &'static str,
    pub outcome: String,
    pub duration_s: Option<f64>,
    pub cz: usize,
    pub cost_inf: bool,
}

/// `model_template` for a ring that does not embed (else refused), `ring_template`'s choice between N2 and item 31.
pub fn model_template(dev: &Device, n: usize, ops: &[Op], blocks: &Blocks, max_error: f64, only: Option<&str>)
    -> Result<(Option<Template>, Vec<Candidate>), Stop> {
    let read = match ring_read(n, ops) {
        Some(r) => r,
        None => return refused("not read as a CZ ring classifier (the compile template's case: not ported)"),
    };
    if ring_embeds(dev, n, max_error)? {
        return refused("the ring embeds in the working couplers (item 42: the compile template, not ported)");
    }
    let lat = match chain_lattice(dev, max_error)? {
        Some(l) => l,
        None => return Ok((None, Vec::new())),
    };
    let line = chain_place(n, &lat.k, &lat.fadj, &lat.snake)?;
    let mut best: Option<Template> = None;
    let mut cands = Vec::new();
    for q4 in [false, true] {
        let name = if q4 { "q4" } else { "n2" };
        if only.map_or(false, |o| o != name) {
            continue;
        }
        match template_one(dev, n, ops, &read, &line, blocks, q4) {
            Ok(t) => {
                cands.push(Candidate {
                    schedule: name,
                    outcome: "built".into(),
                    duration_s: duration_s(dev, &t.ops),
                    cz: count_2q(&t.ops),
                    cost_inf: cost_inf(dev, &t.ops),
                });
                let take = match &best {
                    None => true,
                    Some(b) => better(dev, &t.ops, &b.ops),
                };
                if take {
                    best = Some(t);
                }
            }
            Err(Stop::None(why)) => {
                cands.push(Candidate { schedule: name, outcome: why, duration_s: None, cz: 0, cost_inf: false })
            }
            Err(e) => return Err(e),
        }
    }
    Ok((best, cands))
}

fn jnum(x: usize) -> J {
    J::Num(x as f64)
}

fn obj(kv: Vec<(&str, J)>) -> J {
    J::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub fn template_json(t: &Template) -> J {
    let ops = t
        .ops
        .iter()
        .map(|o| {
            let p = match o.param {
                Param::None => J::Null,
                Param::Num(x) => J::Num(x),
                Param::Var(k) => obj(vec![("p", jnum(k))]),
            };
            J::Arr(vec![
                J::Str(o.name.to_string()),
                J::Arr(o.qs.iter().map(|&q| jnum(q)).collect()),
                J::Arr(o.cs.iter().map(|&c| jnum(c)).collect()),
                p,
            ])
        })
        .collect();
    let runs = t
        .var_runs
        .iter()
        .map(|fs| {
            J::Arr(
                fs.iter()
                    .map(|f| match f {
                        Factor::Fixed(m) => mat_to(m),
                        Factor::Var(i, kind) => J::Arr(vec![jnum(*i), J::Str(kind.clone())]),
                    })
                    .collect(),
            )
        })
        .collect();
    obj(vec![
        ("schedule", J::Str(t.schedule.to_string())),
        ("num_qubits", jnum(t.num_qubits)),
        ("ops", J::Arr(ops)),
        ("var_runs", J::Arr(runs)),
        ("var_kinds", J::Arr(t.var_kinds.iter().map(|(k, nm)| J::Arr(vec![jnum(*k), J::Str(nm.clone())])).collect())),
        ("layout", J::Arr(t.layout.iter().map(|&q| jnum(q)).collect())),
        ("use_rx", J::Bool(t.use_rx)),
        ("nvar", jnum(t.nvar)),
    ])
}

/// The whole call: JSON in, JSON out (never panics on bad input: an error is a "refused" with its reason).
pub fn build(input: &str) -> String {
    let res = (|| -> Result<J, Stop> {
        let j = json::parse(input).map_err(Stop::Refused)?;
        let dev = device_from(j.get("device").ok_or(Stop::Refused("no device".into()))?).map_err(Stop::Refused)?;
        let (n, ops) = input_from(j.get("input").ok_or(Stop::Refused("no input".into()))?).map_err(Stop::Refused)?;
        let blocks = blocks_from(j.get("blocks").ok_or(Stop::Refused("no blocks".into()))?).map_err(Stop::Refused)?;
        let max_error = j.get("max_error").and_then(|v| v.num()).unwrap_or(0.5);
        let only = j.get("schedule").and_then(|v| v.str());
        for o in &ops {
            if o.qs.iter().any(|&q| q >= n) {
                return Err(Stop::Refused("an input qubit out of range".into()));
            }
        }
        let (best, cands) = model_template(&dev, n, &ops, &blocks, max_error, only)?;
        let cj = J::Arr(
            cands
                .iter()
                .map(|c| {
                    obj(vec![
                        ("schedule", J::Str(c.schedule.into())),
                        ("outcome", J::Str(c.outcome.clone())),
                        ("duration_s", c.duration_s.map_or(J::Null, J::Num)),
                        ("cz", jnum(c.cz)),
                        ("cost_inf", J::Bool(c.cost_inf)),
                    ])
                })
                .collect(),
        );
        Ok(match best {
            Some(t) => obj(vec![("status", J::Str("built".into())), ("template", template_json(&t)), ("candidates", cj)]),
            None => obj(vec![("status", J::Str("none".into())), ("candidates", cj)]),
        })
    })();
    json::write(&match res {
        Ok(j) => j,
        Err(Stop::Refused(r)) => obj(vec![("status", J::Str("refused".into())), ("reason", J::Str(r))]),
        Err(Stop::None(r)) => obj(vec![("status", J::Str("none".into())), ("reason", J::Str(r))]),
        Err(Stop::OutOfScope(r)) => obj(vec![("status", J::Str("out_of_scope".into())), ("reason", J::Str(r))]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_cover_the_ring() {
        for n in 4..20 {
            let (_, r) = ring_rounds(n).unwrap();
            let total: usize = r.iter().map(|(c, _)| c.len()).sum();
            assert_eq!(total, n);
        }
        let (_, r) = ring_rounds_q4(10).unwrap();
        assert_eq!(r.iter().map(|(c, _)| c.len()).sum::<usize>(), 10);
    }

    #[test]
    fn folded_order() {
        assert_eq!(folded(5), vec![0, 4, 1, 3, 2]);
        assert_eq!(folded(4), vec![0, 3, 1, 2]);
    }

    #[test]
    fn python_round_is_half_even() {
        assert_eq!((2.5f64).round_ties_even(), 2.0);
        assert_eq!((3.5f64).round_ties_even(), 4.0);
    }
}
