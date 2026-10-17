//! The device: qubits, couplers and the coordinates of IBM's heavy-hex lattice, all read from the device file.
//!
//! Device file (JSON): {"name", "num_qubits", "two_qubit_gate": "cz" | "ecr",
//!   "qubits": [{"sx_error", "sx_duration", "meas_error"}, ...] (null allowed; other keys are ignored),
//!   "couplers": [[a, b, error, duration], ...] (as the device offers them; ECR is directed)}.
//! A qubit or coupler whose reported error is 0.5 or more is failed (PSF-Zero's item 31); a coupler is working when
//! it and both its qubits are.

use crate::json::J;
use std::collections::{BTreeMap, BTreeSet};

pub const FAILED: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub n: usize,
    pub gate2: String,
    pub qubit_ok: Vec<bool>,
    /// directed couplers the device offers, with (error, duration)
    pub couplers: BTreeMap<(usize, usize), (Option<f64>, Option<f64>)>,
    /// undirected adjacency over all couplers
    pub adj: Vec<BTreeSet<usize>>,
    /// single-qubit pulse length (sx), seconds; 0 when not reported
    pub sx_dur: Vec<f64>,
}

impl Device {
    pub fn from_json(j: &J) -> Result<Device, String> {
        let n = j.get("num_qubits").and_then(J::num).ok_or("device: num_qubits missing")? as usize;
        let gate2 = j.get("two_qubit_gate").and_then(J::str).ok_or("device: two_qubit_gate missing")?.to_string();
        if gate2 != "cz" && gate2 != "ecr" {
            return Err(format!("device: two-qubit gate '{}' is not cz or ecr", gate2));
        }
        let name = j.get("name").and_then(J::str).unwrap_or("device").to_string();
        let qs = j.get("qubits").and_then(J::arr).ok_or("device: qubits missing")?;
        if qs.len() != n {
            return Err("device: qubits list length differs from num_qubits".into());
        }
        let f = |q: &J, k: &str| q.get(k).and_then(J::num);
        let mut qubit_ok = Vec::with_capacity(n);
        let mut sx_dur = Vec::with_capacity(n);
        for q in qs {
            let bad = f(q, "sx_error").map_or(false, |e| e >= FAILED) || f(q, "meas_error").map_or(false, |e| e >= FAILED);
            qubit_ok.push(!bad);
            sx_dur.push(f(q, "sx_duration").unwrap_or(0.0));
        }
        let mut couplers = BTreeMap::new();
        let mut adj = vec![BTreeSet::new(); n];
        for c in j.get("couplers").and_then(J::arr).ok_or("device: couplers missing")? {
            let v = c.arr().ok_or("device: a coupler is not a list")?;
            let a = v[0].num().ok_or("device: coupler qubit")? as usize;
            let b = v[1].num().ok_or("device: coupler qubit")? as usize;
            couplers.insert((a, b), (v.get(2).and_then(J::num), v.get(3).and_then(J::num)));
            adj[a].insert(b);
            adj[b].insert(a);
        }
        Ok(Device { name, n, gate2, qubit_ok, couplers, adj, sx_dur })
    }

    /// The coupler's duration in picoseconds (the longer direction), if reported.
    pub fn dur_ps(&self, a: usize, b: usize) -> Option<i64> {
        let d = |k: &(usize, usize)| self.couplers.get(k).and_then(|(_, d)| *d);
        match (d(&(a, b)), d(&(b, a))) {
            (Some(x), Some(y)) => Some((x.max(y) * 1e12).round() as i64),
            (Some(x), None) | (None, Some(x)) => Some((x * 1e12).round() as i64),
            _ => None,
        }
    }

    /// The most common duration among working couplers (ties: the shorter), as PSF-Zero's item 87 prefers.
    pub fn modal_ps(&self) -> Option<i64> {
        let mut count: BTreeMap<i64, usize> = BTreeMap::new();
        for &(a, b) in self.couplers.keys() {
            if a < b || !self.couplers.contains_key(&(b, a)) {
                if self.working(a, b) {
                    if let Some(d) = self.dur_ps(a, b) {
                        *count.entry(d).or_default() += 1;
                    }
                }
            }
        }
        count.iter().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0))).map(|(d, _)| *d)
    }

    /// Working, and of the most common duration (`modal`), when that is asked for.
    pub fn usable(&self, a: usize, b: usize, modal: Option<i64>) -> bool {
        self.working(a, b) && modal.map_or(true, |m| self.dur_ps(a, b) == Some(m))
    }

    /// The coupler between a and b in either direction is offered and working.
    pub fn working(&self, a: usize, b: usize) -> bool {
        if !self.qubit_ok[a] || !self.qubit_ok[b] {
            return false;
        }
        let ok = |k: &(usize, usize)| self.couplers.get(k).map_or(false, |(e, _)| e.map_or(true, |e| e < FAILED));
        ok(&(a, b)) || ok(&(b, a))
    }
}

/// Heavy-hex coordinates: rows of consecutive qubits joined by couplers (more than two qubits), each row's column
/// offset (so that a bridge joins equal columns), and the bridges of each band (between row i and row i + 1).
#[derive(Clone, Debug)]
pub struct Lattice {
    pub rows: Vec<Vec<usize>>,
    pub offset: Vec<i64>,
    /// band i: (column, bridge qubit), sorted by column
    pub bridges: Vec<Vec<(i64, usize)>>,
}

impl Lattice {
    pub fn read(d: &Device) -> Result<Lattice, String> {
        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut cur = vec![0usize];
        for q in 1..d.n {
            if d.adj[q].contains(&(q - 1)) {
                cur.push(q);
            } else {
                rows.push(cur);
                cur = vec![q];
            }
        }
        rows.push(cur);
        let rows: Vec<Vec<usize>> = rows.into_iter().filter(|r| r.len() > 2).collect();
        let mut at = vec![None; d.n]; // qubit -> (row, index)
        for (i, r) in rows.iter().enumerate() {
            for (j, &q) in r.iter().enumerate() {
                at[q] = Some((i, j as i64));
            }
        }
        let mut offset: Vec<Option<i64>> = vec![None; rows.len()];
        if !rows.is_empty() {
            offset[0] = Some(0);
        }
        let mut links = Vec::new(); // (band, upper index, lower index, bridge)
        for b in 0..d.n {
            if at[b].is_some() {
                continue;
            }
            let ends: Vec<(usize, i64)> = d.adj[b].iter().filter_map(|&a| at[a]).collect();
            if ends.len() < 2 {
                continue; // a qubit hanging off one row (FakeTorino's 129-132): not used by chains or rings
            }
            if ends.len() != 2 || ends[0].0.abs_diff(ends[1].0) != 1 {
                return Err(format!("lattice: qubit {} is not a bridge between two rows", b));
            }
            let (u, l) = if ends[0].0 < ends[1].0 { (ends[0], ends[1]) } else { (ends[1], ends[0]) };
            links.push((u.0, u.1, l.1, b));
        }
        for _ in 0..rows.len() {
            for &(band, ju, jl, _) in &links {
                if let (Some(o), None) = (offset[band], offset[band + 1]) {
                    offset[band + 1] = Some(o + ju - jl);
                }
            }
        }
        let offset: Vec<i64> = offset.into_iter().map(|o| o.ok_or("lattice: a row is not joined to row 0")).collect::<Result<_, _>>()?;
        let mut bridges = vec![Vec::new(); rows.len().saturating_sub(1)];
        for &(band, ju, jl, b) in &links {
            if offset[band] + ju != offset[band + 1] + jl {
                return Err(format!("lattice: bridge {} joins different columns", b));
            }
            bridges[band].push((offset[band] + ju, b));
        }
        for v in bridges.iter_mut() {
            v.sort();
        }
        Ok(Lattice { rows, offset, bridges })
    }

    /// The qubit of row `row` at absolute column `c`.
    pub fn at(&self, row: usize, c: i64) -> Option<usize> {
        let j = c - self.offset[row];
        if j < 0 || j as usize >= self.rows[row].len() {
            None
        } else {
            Some(self.rows[row][j as usize])
        }
    }

    /// Row qubits from column a to column b inclusive, in that direction.
    pub fn segment(&self, row: usize, a: i64, b: i64) -> Option<Vec<usize>> {
        let step = if b >= a { 1 } else { -1 };
        let mut v = Vec::new();
        let mut c = a;
        loop {
            v.push(self.at(row, c)?);
            if c == b {
                break;
            }
            c += step;
        }
        Some(v)
    }
}
