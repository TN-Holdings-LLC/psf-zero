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
///
/// r4: when the snake has no such piece on couplers of the most common duration, the longest row-and-bridge path
/// on them (`row_bridge_path`) is tried before the snake on every working coupler, and the longest row-and-bridge path
/// on every working coupler last.
pub fn chain(d: &Device, l: &Lattice, n: usize) -> Option<Vec<usize>> {
    let m = d.modal_ps();
    chain_on(d, l, n, m)
        .or_else(|| centred(row_bridge_path(d, l, m), n))
        .or_else(|| chain_on(d, l, n, None))
        .or_else(|| centred(row_bridge_path(d, l, None), n))
}

fn centred(p: Vec<usize>, n: usize) -> Option<Vec<usize>> {
    if p.len() < n {
        return None;
    }
    let st = (p.len() - n) / 2;
    Some(p[st..st + n].to_vec())
}

/// The longest path of the row-and-bridge family (r4): it runs along a stretch of one row, crosses a bridge to the
/// next row, runs along a stretch of that row (in either direction), crosses a bridge, and so on downward. The snake
/// is one member (every turn at the row's far end); the other members turn at any bridge. Every member is a simple
/// path by construction (each row is entered once). The longest is computed exactly, row by row from the bottom
/// (g(r, c): the most qubits of a member entering row r at column c), in O(rows x columns x bridges); nothing is
/// matched against a graph and nothing backtracks. Ties: ending in the row before a bridge, a bridge on the left
/// before one on the right, the upper start row, the left start.
pub fn row_bridge_path(d: &Device, l: &Lattice, modal: Option<i64>) -> Vec<usize> {
    let nr = l.rows.len();
    let qok = |r: usize, c: i64| l.at(r, c).map_or(false, |q| d.qubit_ok[q]);
    // the working run of row r around column c: (lo, hi)
    let run = |r: usize, c: i64| -> (i64, i64) {
        let (mut lo, mut hi) = (c, c);
        while qok(r, lo - 1) && d.usable(l.at(r, lo - 1).unwrap(), l.at(r, lo).unwrap(), modal) {
            lo -= 1;
        }
        while qok(r, hi + 1) && d.usable(l.at(r, hi).unwrap(), l.at(r, hi + 1).unwrap(), modal) {
            hi += 1;
        }
        (lo, hi)
    };
    // the usable bridges of band r: (column, bridge qubit)
    let usable_bridges = |r: usize| -> Vec<(i64, usize)> {
        l.bridges[r]
            .iter()
            .copied()
            .filter(|&(b, q)| {
                qok(r, b)
                    && qok(r + 1, b)
                    && d.qubit_ok[q]
                    && d.usable(l.at(r, b).unwrap(), q, modal)
                    && d.usable(q, l.at(r + 1, b).unwrap(), modal)
            })
            .collect()
    };
    // g[r][c - offset]: (qubits, choice) with choice None = end in the row, Some(i) = the i-th usable bridge
    let mut g: Vec<Vec<(usize, Option<usize>)>> = vec![Vec::new(); nr];
    let mut ub: Vec<Vec<(i64, usize)>> = vec![Vec::new(); nr];
    for r in (0..nr).rev() {
        if r + 1 < nr {
            ub[r] = usable_bridges(r);
        }
        let mut row = vec![(0usize, None); l.rows[r].len()];
        for (j, slot) in row.iter_mut().enumerate() {
            let c = l.offset[r] + j as i64;
            if !qok(r, c) {
                continue;
            }
            let (lo, hi) = run(r, c);
            let mut best = (((hi - c).max(c - lo) + 1) as usize, None);
            for (i, &(b, _)) in ub[r].iter().enumerate() {
                if b < lo || b > hi {
                    continue;
                }
                let below = g[r + 1][(b - l.offset[r + 1]) as usize].0;
                let v = (b - c).unsigned_abs() as usize + 2 + below;
                if v > best.0 {
                    best = (v, Some(i));
                }
            }
            *slot = best;
        }
        g[r] = row;
    }
    // the start: (qubits, row, start column, first bridge or None for a stretch of one row)
    let mut start: Option<(usize, usize, i64, Option<usize>)> = None;
    for r in 0..nr {
        let len = l.rows[r].len() as i64;
        let mut c = l.offset[r];
        while c < l.offset[r] + len {
            if qok(r, c) {
                let (lo, hi) = run(r, c);
                let v = (hi - lo + 1) as usize;
                if start.map_or(true, |s| v > s.0) {
                    start = Some((v, r, lo, None));
                }
                c = hi + 1;
            } else {
                c += 1;
            }
        }
        if r + 1 < nr {
            for (i, &(b, _)) in ub[r].iter().enumerate() {
                let (lo, hi) = run(r, b);
                let s = if b - lo >= hi - b { lo } else { hi };
                let v = (b - s).unsigned_abs() as usize + 2 + g[r + 1][(b - l.offset[r + 1]) as usize].0;
                if start.map_or(true, |x| v > x.0) {
                    start = Some((v, r, s, Some(i)));
                }
            }
        }
    }
    let Some((_, r0, s0, first)) = start else { return Vec::new() };
    let mut path = Vec::new();
    let (mut r, mut c, mut choice) = (r0, s0, first);
    if first.is_none() {
        let (_, hi) = run(r, c);
        return l.segment(r, c, hi).unwrap();
    }
    loop {
        match choice {
            Some(i) => {
                let (b, q) = ub[r][i];
                path.extend(l.segment(r, c, b).unwrap());
                path.push(q);
                r += 1;
                c = b;
                choice = g[r][(c - l.offset[r]) as usize].1;
            }
            None => {
                let (lo, hi) = run(r, c);
                let e = if hi - c >= c - lo { hi } else { lo };
                path.extend(l.segment(r, c, e).unwrap());
                break;
            }
        }
    }
    path
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
