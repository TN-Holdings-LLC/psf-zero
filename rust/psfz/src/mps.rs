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

use crate::linalg::*;

/// One-sided Jacobi SVD of an m x k complex matrix (column-major columns `a[j]`): returns (U columns, sigma, V
/// columns) with A = U diag(sigma) V^H, sigma in decreasing order.
pub fn svd(mut a: Vec<Vec<C>>) -> (Vec<Vec<C>>, Vec<f64>, Vec<Vec<C>>) {
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
    pub fn cz(&mut self, s: usize) {
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
        for a in 0..l {
            for b in 0..r {
                let k = ((a * 2 + 1) * 2 + 1) * r + b;
                th[k] = -th[k];
            }
        }
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

    /// <Z> of every site (environments from both ends; the state's norm divided out).
    pub fn z_all(&self) -> Vec<f64> {
        let n = self.n;
        // left[s]: (chi[s] x chi[s]) environment of sites < s
        let mut left: Vec<Vec<C>> = vec![vec![ONE]];
        for s in 0..n {
            let (l, r) = (self.chi[s], self.chi[s + 1]);
            let e = &left[s];
            let mut ne = vec![ZERO; r * r];
            let t = &self.t[s];
            for a in 0..l {
                for a2 in 0..l {
                    let w = e[a * l + a2];
                    if w.abs() == 0.0 {
                        continue;
                    }
                    for p in 0..2 {
                        for b in 0..r {
                            let x = t[(a * 2 + p) * r + b].conj() * w;
                            for b2 in 0..r {
                                ne[b * r + b2] = ne[b * r + b2] + x * t[(a2 * 2 + p) * r + b2];
                            }
                        }
                    }
                }
            }
            left.push(ne);
        }
        let mut right: Vec<Vec<C>> = vec![vec![ZERO]; n + 1];
        right[n] = vec![ONE];
        for s in (0..n).rev() {
            let (l, r) = (self.chi[s], self.chi[s + 1]);
            let e = right[s + 1].clone();
            let mut ne = vec![ZERO; l * l];
            let t = &self.t[s];
            for b in 0..r {
                for b2 in 0..r {
                    let w = e[b * r + b2];
                    if w.abs() == 0.0 {
                        continue;
                    }
                    for p in 0..2 {
                        for a in 0..l {
                            let x = t[(a * 2 + p) * r + b].conj() * w;
                            for a2 in 0..l {
                                ne[a * l + a2] = ne[a * l + a2] + x * t[(a2 * 2 + p) * r + b2];
                            }
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
            let mut acc = ZERO;
            for a in 0..l {
                for a2 in 0..l {
                    let w = left[s][a * l + a2];
                    if w.abs() == 0.0 {
                        continue;
                    }
                    for p in 0..2 {
                        let sgn = if p == 0 { 1.0 } else { -1.0 };
                        for b in 0..r {
                            let x = t[(a * 2 + p) * r + b].conj() * w;
                            for b2 in 0..r {
                                acc = acc + (x * t[(a2 * 2 + p) * r + b2] * right[s + 1][b * r + b2]).scale(sgn);
                            }
                        }
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
        for (m, k) in [(4, 4), (6, 3), (3, 7), (8, 8)] {
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
