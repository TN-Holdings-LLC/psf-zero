//! Placement by coordinates, no search: a chain goes on the heavy-hex snake, a ring on the boundary of a band of
//! hexagons. Candidate positions are produced by formula in a fixed order; the first one whose qubits and couplers all
//! work is used. Nothing is scored, nothing is matched against a graph, nothing backtracks.

use crate::device::{Device, Lattice};

/// Heron's snake (PSF-Zero's item 87, W5): row 0 left to right, down the bridge at its far end, the next row back,
/// and so on.
pub fn snake(l: &Lattice) -> Vec<usize> {
    let mut path = Vec::new();
    let nrows = l.rows.len();
    let mut col = l.offset[0];
    for i in 0..nrows {
        let leftward = i % 2 == 1;
        let row_first = l.offset[i];
        let row_last = l.offset[i] + l.rows[i].len() as i64 - 1;
        let start = if i == 0 { row_first } else { col };
        if i + 1 < nrows {
            let br = &l.bridges[i];
            if br.is_empty() {
                path.extend(l.segment(i, start, if leftward { row_first } else { row_last }).unwrap());
                return path;
            }
            let (exit, bq) = if leftward { br[0] } else { *br.last().unwrap() };
            match l.segment(i, start, exit) {
                Some(seg) => path.extend(seg),
                None => return path,
            }
            path.push(bq);
            col = exit;
        } else {
            path.extend(l.segment(i, start, if leftward { row_first } else { row_last }).unwrap());
        }
    }
    path
}

/// n physical qubits forming a path: the longest working piece of the snake (ties: the first), centred; first on
/// couplers of the device's most common duration, then on every working coupler (as item 87).
pub fn chain(d: &Device, l: &Lattice, n: usize) -> Option<Vec<usize>> {
    chain_on(d, l, n, d.modal_ps()).or_else(|| chain_on(d, l, n, None))
}

fn chain_on(d: &Device, l: &Lattice, n: usize, modal: Option<i64>) -> Option<Vec<usize>> {
    let s = snake(l);
    let mut pieces: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for &q in &s {
        let ok = d.qubit_ok[q] && cur.last().map_or(true, |&p| d.usable(p, q, modal));
        if ok {
            cur.push(q);
        } else {
            if !cur.is_empty() {
                pieces.push(std::mem::take(&mut cur));
            }
            if d.qubit_ok[q] {
                cur.push(q);
            }
        }
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    let mut best: Option<&Vec<usize>> = None;
    for p in &pieces {
        if best.map_or(true, |b| p.len() > b.len()) {
            best = Some(p);
        }
    }
    let b = best?;
    if b.len() < n {
        return None;
    }
    let st = (b.len() - n) / 2;
    Some(b[st..st + n].to_vec())
}

/// The bridge of band `j` whose column is nearest to `c` (ties: the smaller column if `left`, else the larger).
fn nearest(l: &Lattice, j: usize, c: i64, left: bool) -> Option<(i64, usize)> {
    let mut best: Option<(i64, usize)> = None;
    for &(bc, q) in &l.bridges[j] {
        let better = match best {
            None => true,
            Some((b, _)) => {
                let (d1, d0) = ((bc - c).abs(), (b - c).abs());
                d1 < d0 || (d1 == d0 && if left { bc < b } else { bc > b })
            }
        };
        if better {
            best = Some((bc, q));
        }
    }
    best
}

/// How a ring's side steps from band to band: always outward (away from the other side), always inward,
/// alternating (outward first, or inward first: a straight side), or once one way and then the other (a bulge or a
/// waist).
#[derive(Clone, Copy, Debug)]
pub enum Step {
    Out,
    In,
    OutIn,
    InOut,
    OnceOut,
    OnceIn,
}

impl Step {
    /// Whether the k-th step (k = 1 for the step into the second band) goes outward.
    fn outward(self, k: usize) -> bool {
        match self {
            Step::Out => true,
            Step::In => false,
            Step::OutIn => k % 2 == 1,
            Step::InOut => k % 2 == 0,
            Step::OnceOut => k == 1,
            Step::OnceIn => k != 1,
        }
    }
}

/// The ring of shape (top band i0, height h, left bridge li and right bridge ri of band i0, and how each side steps
/// between bands), as a cycle of qubits; None when the shape does not close on this lattice.
pub fn ring_shape(l: &Lattice, i0: usize, h: usize, li: usize, ri: usize, ls: Step, rs: Step) -> Option<Vec<usize>> {
    let bands = l.bridges.len();
    if i0 + h > bands || li >= ri || ri >= l.bridges[i0].len() {
        return None;
    }
    let mut left = vec![l.bridges[i0][li]];
    let mut right = vec![l.bridges[i0][ri]];
    for j in i0 + 1..i0 + h {
        let k = j - i0;
        left.push(nearest(l, j, left.last().unwrap().0, ls.outward(k))?);
        right.push(nearest(l, j, right.last().unwrap().0, !rs.outward(k))?);
    }
    // the sides stay apart on every intermediate row
    for k in 1..h {
        let lmax = left[k - 1].0.max(left[k].0);
        let rmin = right[k - 1].0.min(right[k].0);
        if lmax >= rmin {
            return None;
        }
    }
    if left[h - 1].0 >= right[h - 1].0 || left[0].0 >= right[0].0 {
        return None;
    }
    let mut cyc = Vec::new();
    cyc.extend(l.segment(i0, left[0].0, right[0].0)?); // top row, left to right
    for k in 0..h {
        cyc.push(right[k].1); // down the right bridge of band i0 + k
        let row = i0 + k + 1;
        if k + 1 < h {
            let seg = l.segment(row, right[k].0, right[k + 1].0)?;
            cyc.extend(seg);
        } else {
            cyc.extend(l.segment(row, right[k].0, left[k].0)?); // bottom row, right to left
        }
    }
    for k in (0..h).rev() {
        cyc.push(left[k].1); // up the left bridge of band i0 + k
        if k > 0 {
            let row = i0 + k;
            cyc.extend(l.segment(row, left[k].0, left[k - 1].0)?);
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    if !cyc.iter().all(|q| seen.insert(*q)) {
        return None;
    }
    Some(cyc)
}

/// n physical qubits forming a cycle: shapes in the order (height, the sides' steps, top band, left bridge, right
/// bridge); the first of length n whose qubits and couplers all work, first among couplers of the most common
/// duration. Every cycle in a heavy-hex lattice has even length.
pub fn ring(d: &Device, l: &Lattice, n: usize) -> Option<Vec<usize>> {
    ring_on(d, l, n, d.modal_ps()).or_else(|| ring_on(d, l, n, None))
}

fn ring_on(d: &Device, l: &Lattice, n: usize, modal: Option<i64>) -> Option<Vec<usize>> {
    let bands = l.bridges.len();
    for h in 1..=bands {
        let all = [Step::Out, Step::In, Step::OutIn, Step::InOut, Step::OnceOut, Step::OnceIn];
        let mut steps = Vec::new();
        for &a in &all {
            for &b in &all {
                steps.push((a, b));
            }
        }
        if h == 1 {
            steps.truncate(1); // one band: no step
        }
        for &(ls, rs) in &steps {
            for i0 in 0..bands {
                if i0 + h > bands {
                    continue;
                }
                let nb = l.bridges[i0].len();
                for li in 0..nb {
                    for ri in li + 1..nb {
                        if let Some(c) = ring_shape(l, i0, h, li, ri, ls, rs) {
                            if c.len() == n && (0..n).all(|k| d.usable(c[k], c[(k + 1) % n], modal)) {
                                return Some(c);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}
