//! A matrix-product-state twin for psfz's chain outputs (prototype, 2026-10-19; noiseless so far).
//!
//! The compiled circuit is simulated on its chain of physical qubits as an MPS, sites in chain order. No bond
//! dimension is chosen by trial: before simulating, each cut's bond dimension is bounded by a formula,
//! chi_i <= min(2^(i+1), 2^(n-i-1), 2^(c_i)), with c_i the number of CZ crossing cut i (a CZ has operator Schmidt rank
//! 2, so it at most doubles the rank across its own cut and changes no other cut); if the bound exceeds `chi_max` the
//! run is refused before it starts. While simulating, each two-site update is split by an exact SVD (one-sided Jacobi,
//! no outside crate) that drops only numerically zero singular values (below 1e-12 of the largest); the dropped
//! weight is summed and reported, so the result is exact up to that weight. Single-qubit gates (and, later, the
//! single-qubit noise of trajectories) never change a bond dimension, so the bound holds for noisy trajectories too.
//!
//! The SVD is pure Rust by default (one-sided Jacobi). With the `lapack` feature (Cargo.toml) it is LAPACK's zgesdd
//! from a system library: the same factorisation, much faster at large bond dimensions.

use crate::linalg::*;

/// The SVD used by the MPS: LAPACK's zgesdd with the `lapack` feature, else the pure-Rust Jacobi method. Same
/// contract as `svd_jacobi`.
pub fn svd(a: Vec<Vec<C>>) -> (Vec<Vec<C>>, Vec<f64>, Vec<Vec<C>>) {
    #[cfg(feature = "lapack")]
    {
        svd_lapack(&a)
    }
    #[cfg(not(feature = "lapack"))]
    {
        svd_jacobi(a)
    }
}

pub fn svd_backend() -> &'static str {
    if cfg!(feature = "lapack") {
        "lapack zgesdd"
    } else {
        "pure Rust one-sided Jacobi"
    }
}

#[cfg(feature = "lapack")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn zgesdd_(jobz: *const u8, m: *const i32, n: *const i32, a: *mut f64, lda: *const i32, s: *mut f64,
               u: *mut f64, ldu: *const i32, vt: *mut f64, ldvt: *const i32, work: *mut f64, lwork: *const i32,
               rwork: *mut f64, iwork: *mut i32, info: *mut i32, jobz_len: usize);
}

/// LAPACK's zgesdd (jobz 'S') on the m x k matrix with columns `a[j]`; same contract as `svd_jacobi` (min(m, k)
/// singular values).
#[cfg(feature = "lapack")]
pub fn svd_lapack(a: &[Vec<C>]) -> (Vec<Vec<C>>, Vec<f64>, Vec<Vec<C>>) {
    let k = a.len();
    let m = if k > 0 { a[0].len() } else { 0 };
    let mn = m.min(k);
    // column-major, complex as interleaved (re, im)
    let mut buf = Vec::with_capacity(2 * m * k);
    for col in a {
        for z in col {
            buf.push(z.re);
            buf.push(z.im);
        }
    }
    let (mi, ki, mni) = (m as i32, k as i32, mn as i32);
    let mut s = vec![0.0f64; mn];
    let mut u = vec![0.0f64; 2 * m * mn];
    let mut vt = vec![0.0f64; 2 * mn * k];
    let mx = m.max(k);
    let lrwork = (mn * (5 * mn + 7)).max(mn * (2 * mx + 2 * mn + 1)).max(1);
    let mut rwork = vec![0.0f64; lrwork];
    let mut iwork = vec![0i32; 8 * mn.max(1)];
    let mut info = 0i32;
    let mut wq = [0.0f64; 2];
    let query = -1i32;
    unsafe {
        zgesdd_(b"S".as_ptr(), &mi, &ki, buf.as_mut_ptr(), &mi, s.as_mut_ptr(), u.as_mut_ptr(), &mi,
                vt.as_mut_ptr(), &mni, wq.as_mut_ptr(), &query, rwork.as_mut_ptr(), iwork.as_mut_ptr(), &mut info, 1);
    }
    let lwork = (wq[0] as i32).max(1);
    let mut work = vec![0.0f64; 2 * lwork as usize];
    unsafe {
        zgesdd_(b"S".as_ptr(), &mi, &ki, buf.as_mut_ptr(), &mi, s.as_mut_ptr(), u.as_mut_ptr(), &mi,
                vt.as_mut_ptr(), &mni, work.as_mut_ptr(), &lwork, rwork.as_mut_ptr(), iwork.as_mut_ptr(), &mut info, 1);
    }
    assert!(info == 0, "zgesdd failed: info {}", info);
    let uc: Vec<Vec<C>> = (0..mn).map(|r| (0..m).map(|i| C::new(u[2 * (r * m + i)], u[2 * (r * m + i) + 1])).collect()).collect();
    // V[i][r] = conj(VT[r][i]); VT is mn x k, column-major
    let vc: Vec<Vec<C>> = (0..mn).map(|r| (0..k).map(|i| C::new(vt[2 * (i * mn + r)], -vt[2 * (i * mn + r) + 1])).collect()).collect();
    (uc, s, vc)
}

/// One-sided Jacobi SVD of an m x k complex matrix (column-major columns `a[j]`): returns (U columns, sigma, V
/// columns) with A = U diag(sigma) V^H, sigma in decreasing order.
#[cfg_attr(feature = "lapack", allow(dead_code))]
pub fn svd_jacobi(mut a: Vec<Vec<C>>) -> (Vec<Vec<C>>, Vec<f64>, Vec<Vec<C>>) {
    let k = a.len();
    let m = if k > 0 { a[0].len() } else { 0 };
    let mut v: Vec<Vec<C>> = (0..k).map(|j| (0..k).map(|i| if i == j { ONE } else { ZERO }).collect()).collect();
    let dot = |x: &[C], y: &[C]| -> C { x.iter().zip(y).fold(ZERO, |s, (p, q)| s + p.conj() * *q) };
    let norm2 = |x: &[C]| -> f64 { x.iter().map(|z| z.re * z.re + z.im * z.im).sum() };
    for _sweep in 0..60 {
        let mut off = 0.0f64;
        for p in 0..k {
            for q in (p + 1)..k {
                let al = norm2(&a[p]);
                let be = norm2(&a[q]);
                let g = dot(&a[p], &a[q]);
                let ga = g.abs();
                if ga <= 1e-15 * (al * be).sqrt() || ga == 0.0 {
                    continue;
                }
                off = off.max(ga / (al * be).sqrt());
                // make the inner product real: column q times e^{-i phi}
                let ph = C::new(g.re / ga, -g.im / ga);
                for i in 0..m {
                    a[q][i] = a[q][i] * ph;
                }
                for i in 0..k {
                    v[q][i] = v[q][i] * ph;
                }
                let zeta = (be - al) / (2.0 * ga);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (a[p][i], a[q][i]);
                    a[p][i] = x.scale(c) - y.scale(s);
                    a[q][i] = x.scale(s) + y.scale(c);
                }
                for i in 0..k {
                    let (x, y) = (v[p][i], v[q][i]);
                    v[p][i] = x.scale(c) - y.scale(s);
                    v[q][i] = x.scale(s) + y.scale(c);
                }
            }
        }
        if off < 1e-14 {
            break;
        }
    }
    let mut sig: Vec<f64> = a.iter().map(|c| norm2(c).sqrt()).collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| sig[y].partial_cmp(&sig[x]).unwrap());
    let u: Vec<Vec<C>> = order.iter().map(|&j| {
        let s = sig[j];
        a[j].iter().map(|z| if s > 0.0 { z.scale(1.0 / s) } else { ZERO }).collect()
    }).collect();
    let vv: Vec<Vec<C>> = order.iter().map(|&j| v[j].clone()).collect();
    sig = order.iter().map(|&j| sig[j]).collect();
    (u, sig, vv)
}

/// MPS: site s holds a tensor of shape (l[s], 2, l[s+1]), stored as t[s][(a * 2 + p) * r + b].
pub struct Mps {
    pub n: usize,
    pub chi: Vec<usize>, // chi[s] = left bond of site s; chi[0] = chi[n] = 1
    pub t: Vec<Vec<C>>,
    pub dropped: f64,
    pub max_chi: usize,
}

impl Mps {
    pub fn zero(n: usize) -> Mps {
        Mps { n, chi: vec![1; n + 1], t: (0..n).map(|_| vec![ONE, ZERO]).collect(), dropped: 0.0, max_chi: 1 }
    }

    pub fn apply1(&mut self, s: usize, u: &M2) {
        let (l, r) = (self.chi[s], self.chi[s + 1]);
        let t = &mut self.t[s];
        for a in 0..l {
            for b in 0..r {
                let x0 = t[(a * 2) * r + b];
                let x1 = t[(a * 2 + 1) * r + b];
                t[(a * 2) * r + b] = u[0] * x0 + u[1] * x1;
                t[(a * 2 + 1) * r + b] = u[2] * x0 + u[3] * x1;
            }
        }
    }

    /// CZ on sites s, s + 1 (diagonal: phase -1 on |11>), then the exact split.
    #[allow(dead_code)]
    pub fn cz(&mut self, s: usize) {
        let mut w = [ZERO; 16];
        for i in 0..4 {
            w[i * 4 + i] = if i == 3 { -ONE } else { ONE };
        }
        self.apply2(s, &w);
    }

    /// A two-site gate W on sites s, s + 1 (row-major 4 x 4, index 2 p + q with p the left site's bit), then one exact
    /// split.
    pub fn apply2(&mut self, s: usize, w: &[C; 16]) {
        let (l, m, r) = (self.chi[s], self.chi[s + 1], self.chi[s + 2]);
        // theta[(a, p), (q, b)]
        let mut th = vec![ZERO; l * 2 * 2 * r];
        for a in 0..l {
            for p in 0..2 {
                for c in 0..m {
                    let x = self.t[s][(a * 2 + p) * m + c];
                    if x.abs() == 0.0 {
                        continue;
                    }
                    for q in 0..2 {
                        for b in 0..r {
                            let y = self.t[s + 1][(c * 2 + q) * r + b];
                            th[((a * 2 + p) * 2 + q) * r + b] = th[((a * 2 + p) * 2 + q) * r + b] + x * y;
                        }
                    }
                }
            }
        }
        let mut th2 = vec![ZERO; l * 4 * r];
        for a in 0..l {
            for b in 0..r {
                for i in 0..4 {
                    let mut acc = ZERO;
                    for j in 0..4 {
                        let wij = w[i * 4 + j];
                        if wij.abs() != 0.0 {
                            acc = acc + wij * th[(a * 4 + j) * r + b];
                        }
                    }
                    th2[(a * 4 + i) * r + b] = acc;
                }
            }
        }
        let th = th2;
        // columns of the (l*2) x (2*r) matrix
        let rows = l * 2;
        let cols = 2 * r;
        let colv: Vec<Vec<C>> = (0..cols).map(|j| (0..rows).map(|i| th[i * cols + j]).collect()).collect();
        let (u, sig, v) = svd(colv);
        let smax = sig.first().copied().unwrap_or(0.0);
        let keep = sig.iter().take_while(|&&x| x > 1e-12 * smax).count().max(1);
        self.dropped += sig[keep..].iter().map(|x| x * x).sum::<f64>();
        // left tensor: U[:, :keep] (rows (a, p)); right: diag(sig) V^H (rows k, cols (q, b))
        let mut tl = vec![ZERO; l * 2 * keep];
        for a in 0..l {
            for p in 0..2 {
                for k in 0..keep {
                    tl[(a * 2 + p) * keep + k] = u[k][a * 2 + p];
                }
            }
        }
        let mut tr = vec![ZERO; keep * 2 * r];
        for k in 0..keep {
            for q in 0..2 {
                for b in 0..r {
                    tr[(k * 2 + q) * r + b] = v[k][q * r + b].conj().scale(sig[k]);
                }
            }
        }
        self.t[s] = tl;
        self.t[s + 1] = tr;
        self.chi[s + 1] = keep;
        self.max_chi = self.max_chi.max(keep);
    }

    /// <Z> of every site (environments from both ends; the state's norm divided out). Each environment step is two
    /// contractions of order chi^3 (2026-10-20: was one of order chi^4).
    pub fn z_all(&self) -> Vec<f64> {
        let n = self.n;
        // left[s][a * l + a2]: environment of sites < s (bra index a, ket index a2)
        let mut left: Vec<Vec<C>> = vec![vec![ONE]];
        for s in 0..n {
            let (l, r) = (self.chi[s], self.chi[s + 1]);
            let e = &left[s];
            let t = &self.t[s];
            // tmp[(a, p), b2] = sum_a2 e[a, a2] t[a2, p, b2]
            let mut tmp = vec![ZERO; l * 2 * r];
            for a in 0..l {
                for a2 in 0..l {
                    let w = e[a * l + a2];
                    if w.abs() == 0.0 {
                        continue;
                    }
                    for p in 0..2 {
                        for b2 in 0..r {
                            tmp[(a * 2 + p) * r + b2] = tmp[(a * 2 + p) * r + b2] + w * t[(a2 * 2 + p) * r + b2];
                        }
                    }
                }
            }
            // ne[b, b2] = sum_(a, p) conj(t[a, p, b]) tmp[(a, p), b2]
            let mut ne = vec![ZERO; r * r];
            for ap in 0..l * 2 {
                for b in 0..r {
                    let x = t[ap * r + b].conj();
                    if x.abs() == 0.0 {
                        continue;
                    }
                    for b2 in 0..r {
                        ne[b * r + b2] = ne[b * r + b2] + x * tmp[ap * r + b2];
                    }
                }
            }
            left.push(ne);
        }
        // right[s][b * r + b2]: environment of sites >= s
        let mut right: Vec<Vec<C>> = vec![vec![ZERO]; n + 1];
        right[n] = vec![ONE];
        for s in (0..n).rev() {
            let (l, r) = (self.chi[s], self.chi[s + 1]);
            let e = &right[s + 1];
            let t = &self.t[s];
            // tmp[(a2, p), b] = sum_b2 t[a2, p, b2] e[b, b2]
            let mut tmp = vec![ZERO; l * 2 * r];
            for a2p in 0..l * 2 {
                for b in 0..r {
                    let mut acc = ZERO;
                    for b2 in 0..r {
                        acc = acc + t[a2p * r + b2] * e[b * r + b2];
                    }
                    tmp[a2p * r + b] = acc;
                }
            }
            // ne[a, a2] = sum_(p, b) conj(t[a, p, b]) tmp[(a2, p), b]
            let mut ne = vec![ZERO; l * l];
            for a in 0..l {
                for p in 0..2 {
                    for b in 0..r {
                        let x = t[(a * 2 + p) * r + b].conj();
                        if x.abs() == 0.0 {
                            continue;
                        }
                        for a2 in 0..l {
                            ne[a * l + a2] = ne[a * l + a2] + x * tmp[(a2 * 2 + p) * r + b];
                        }
                    }
                }
            }
            right[s] = ne;
        }
        let norm = left[n][0].re;
        (0..n).map(|s| {
            let (l, r) = (self.chi[s], self.chi[s + 1]);
            let t = &self.t[s];
            let e = &left[s];
            // tmp[(a, p), b2] = sum_a2 e[a, a2] t[a2, p, b2]; then sum conj(t[a, p, b]) sign(p) tmp right[b, b2]
            let mut tmp = vec![ZERO; l * 2 * r];
            for a in 0..l {
                for a2 in 0..l {
                    let w = e[a * l + a2];
                    if w.abs() == 0.0 {
                        continue;
                    }
                    for p in 0..2 {
                        for b2 in 0..r {
                            tmp[(a * 2 + p) * r + b2] = tmp[(a * 2 + p) * r + b2] + w * t[(a2 * 2 + p) * r + b2];
                        }
                    }
                }
            }
            let rt = &right[s + 1];
            let mut acc = ZERO;
            for a in 0..l {
                for p in 0..2 {
                    let sgn = if p == 0 { 1.0 } else { -1.0 };
                    for b in 0..r {
                        let x = t[(a * 2 + p) * r + b].conj();
                        if x.abs() == 0.0 {
                            continue;
                        }
                        let mut inner = ZERO;
                        for b2 in 0..r {
                            inner = inner + tmp[(a * 2 + p) * r + b2] * rt[b * r + b2];
                        }
                        acc = acc + (x * inner).scale(sgn);
                    }
                }
            }
            acc.re / norm
        }).collect()
    }
}

/// The chain order of a compiled chain circuit's physical qubits (from its CZ couplers), or an error.
pub fn chain_order(ops: &[(String, Vec<usize>, Vec<f64>)]) -> Result<Vec<usize>, String> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut adj: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (g, q, _) in ops {
        if g == "cz" {
            adj.entry(q[0]).or_default().insert(q[1]);
            adj.entry(q[1]).or_default().insert(q[0]);
        } else if q.len() == 2 {
            return Err(format!("mps: two-qubit gate {} is not read", g));
        }
    }
    for (_, q, _) in ops {
        adj.entry(q[0]).or_default();
    }
    if adj.values().any(|s| s.len() > 2) {
        return Err("mps: not a chain (a qubit has more than two neighbours)".into());
    }
    let ends: Vec<usize> = adj.iter().filter(|(_, s)| s.len() <= 1).map(|(q, _)| *q).collect();
    if adj.len() > 1 && ends.len() != 2 {
        return Err("mps: the couplers do not form one chain (rings are not simulated yet)".into());
    }
    let mut walk = vec![ends[0]];
    let mut prev = usize::MAX;
    while walk.len() < adj.len() {
        let cur = *walk.last().unwrap();
        let next = *adj[&cur].iter().find(|&&x| x != prev).ok_or("mps: the chain is not connected")?;
        prev = cur;
        walk.push(next);
    }
    Ok(walk)
}

/// The formula bound of every cut's bond dimension (log2): min(i + 1, n - i - 1, CZ crossing cut i).
pub fn bound_log2(ops: &[(String, Vec<usize>, Vec<f64>)], site: &std::collections::BTreeMap<usize, usize>, n: usize) -> Vec<usize> {
    let mut c = vec![0usize; n.saturating_sub(1)];
    for (g, q, _) in ops {
        if g == "cz" {
            let (a, b) = (site[&q[0]], site[&q[1]]);
            c[a.min(b)] += 1;
        }
    }
    (0..n.saturating_sub(1)).map(|i| c[i].min(i + 1).min(n - i - 1)).collect()
}

/// One operation of the twin: a single-qubit matrix on a site, or a two-site gate on (s, s + 1) with its operator
/// Schmidt rank.
pub enum TwinOp {
    One(usize, M2),
    Two(usize, [C; 16], usize),
}

/// The operator Schmidt rank of a 4 x 4 two-qubit gate (singular values of its realignment above 1e-12 of the largest).
pub fn schmidt_rank(w: &[C; 16]) -> usize {
    // R[(p, p'), (q, q')] = W[(p, q), (p', q')]
    let cols: Vec<Vec<C>> = (0..4).map(|qq| (0..4).map(|pp| {
        let (p, p2, q, q2) = (pp >> 1, pp & 1, qq >> 1, qq & 1);
        w[(2 * p + q) * 4 + (2 * p2 + q2)]
    }).collect()).collect();
    let (_, sig, _) = svd_jacobi(cols);
    let smax = sig.iter().cloned().fold(0.0, f64::max);
    sig.iter().filter(|&&x| x > 1e-12 * smax).count()
}

/// The twin's operations from a compiled chain circuit (workplace proposal of 2026-10-20 §2): two CZ on the same
/// edge with, between them, only single-qubit operations on one end and only rz on the other become one two-site gate
/// W = CZ (D ⊗ U) CZ, applied with one split at its final bond dimension. Read from the gate list (structural).
pub fn plan_ops(ops: &[(String, Vec<usize>, Vec<f64>)], site: &std::collections::BTreeMap<usize, usize>) -> Result<Vec<TwinOp>, String> {
    let mut used = vec![false; ops.len()];
    let mut out = Vec::new();
    let czm = {
        let mut w = [ZERO; 16];
        for i in 0..4 {
            w[i * 4 + i] = if i == 3 { -ONE } else { ONE };
        }
        w
    };
    let m1 = |k: usize| -> Result<M2, String> {
        gate1(&ops[k].0, &ops[k].2).ok_or(format!("twin: gate {} is not read", ops[k].0))
    };
    for i in 0..ops.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let (g, q, _) = &ops[i];
        if g != "cz" {
            out.push(TwinOp::One(site[&q[0]], m1(i)?));
            continue;
        }
        let (a, b) = (q[0], q[1]);
        // look for CZ (ops on one end) CZ on the same edge; try both ends as the one with general operations
        let mut found: Option<(usize, Vec<usize>, Vec<usize>, usize)> = None; // (j, on x, on y, x)
        for x_ in [b, a] {
            let (mut on_x, mut on_y) = (Vec::new(), Vec::new());
            let mut ok = false;
            let mut jj = 0;
            for k in (i + 1)..ops.len() {
                if used[k] {
                    continue;
                }
                let (gk, qk, _) = &ops[k];
                let touches = qk.contains(&a) || qk.contains(&b);
                if !touches {
                    continue;
                }
                if gk == "cz" {
                    if (qk[0] == a && qk[1] == b) || (qk[0] == b && qk[1] == a) {
                        ok = true;
                        jj = k;
                    }
                    break;
                }
                if qk.len() != 1 {
                    break;
                }
                if qk[0] == x_ {
                    on_x.push(k);
                } else if gk == "rz" {
                    on_y.push(k);
                } else {
                    break;
                }
            }
            if ok {
                found = Some((jj, on_x, on_y, x_));
                break;
            }
        }
        let (sa, sb) = (site[&a], site[&b]);
        if sa.max(sb) - sa.min(sb) != 1 {
            return Err("twin: a CZ between sites that are not neighbours".into());
        }
        let left_is_a = sa < sb;
        let w = match found {
            None => czm,
            Some((j, on_x, on_y, x_)) => {
                let mut ux = eye();
                for k in &on_x {
                    ux = mul2(&m1(*k)?, &ux);
                    used[*k] = true;
                }
                let mut dy = eye();
                for k in &on_y {
                    dy = mul2(&m1(*k)?, &dy);
                    used[*k] = true;
                }
                used[j] = true;
                // W on (a, b): CZ · (op on a ⊗ op on b) · CZ
                let (ua, ub) = if x_ == a { (ux, dy) } else { (dy, ux) };
                let mut mid = [ZERO; 16];
                for pa in 0..2 {
                    for pb in 0..2 {
                        for qa in 0..2 {
                            for qb in 0..2 {
                                mid[(2 * pa + pb) * 4 + (2 * qa + qb)] = ua[2 * pa + qa] * ub[2 * pb + qb];
                            }
                        }
                    }
                }
                let sgn = |i: usize| if i == 3 { -ONE } else { ONE };
                let mut w = [ZERO; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        w[r * 4 + c] = sgn(r) * mid[r * 4 + c] * sgn(c);
                    }
                }
                w
            }
        };
        // to (left site, right site) order
        let w = if left_is_a {
            w
        } else {
            let mut v = [ZERO; 16];
            for p in 0..2 {
                for q in 0..2 {
                    for p2 in 0..2 {
                        for q2 in 0..2 {
                            v[(2 * q + p) * 4 + (2 * q2 + p2)] = w[(2 * p + q) * 4 + (2 * p2 + q2)];
                        }
                    }
                }
            }
            v
        };
        let r = schmidt_rank(&w);
        out.push(TwinOp::Two(sa.min(sb), w, r));
    }
    Ok(out)
}

/// The formula bound of every cut's bond dimension (log2) from the twin's own operations: min(i + 1, n - i - 1,
/// Σ over two-site gates on cut i of ceil(log2 rank)). Gate 1 (IR T3) must use the same count.
pub fn bound_log2_blocks(plan: &[TwinOp], n: usize) -> Vec<usize> {
    let mut c = vec![0usize; n.saturating_sub(1)];
    for op in plan {
        if let TwinOp::Two(s, _, r) = op {
            c[*s] += (*r as f64).log2().ceil() as usize;
        }
    }
    (0..n.saturating_sub(1)).map(|i| c[i].min(i + 1).min(n - i - 1)).collect()
}

pub fn gate1(name: &str, p: &[f64]) -> Option<M2> {
    Some(match name {
        "rz" => rz(p[0]),
        "sx" => sx(),
        "x" => x(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for (m, k) in [(4, 4), (6, 3), (3, 7), (8, 8), (40, 24)] {
            let a: Vec<Vec<C>> = (0..k).map(|_| (0..m).map(|_| C::new(rnd(), rnd())).collect()).collect();
            let (u, s, v) = svd(a.clone());
            for j in 0..k {
                for i in 0..m {
                    let mut x = ZERO;
                    for r in 0..s.len() {
                        x = x + (u[r][i] * v[r][j].conj()).scale(s[r]);
                    }
                    assert!((x - a[j][i]).abs() < 1e-10, "({}, {})", m, k);
                }
            }
        }
    }

    #[cfg(feature = "lapack")]
    #[test]
    fn lapack_matches_jacobi() {
        let mut seed = 777u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a: Vec<Vec<C>> = (0..16).map(|_| (0..32).map(|_| C::new(rnd(), rnd())).collect()).collect();
        let (_, s1, _) = svd_lapack(&a);
        let (_, s2, _) = svd_jacobi(a);
        for (x, y) in s1.iter().zip(&s2) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn bell_pair_and_ghz() {
        let mut s = Mps::zero(3);
        s.apply1(0, &h());
        s.apply1(1, &h());
        s.cz(0);
        s.apply1(1, &h());
        s.apply1(2, &h());
        s.cz(1);
        s.apply1(2, &h());
        let z = s.z_all();
        for v in z {
            assert!(v.abs() < 1e-12);
        }
        assert_eq!(s.max_chi, 2);
    }
}
