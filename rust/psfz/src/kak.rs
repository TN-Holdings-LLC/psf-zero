//! The two-qubit Weyl (KAK) decomposition in closed form, with no outside crate: U = (A1 x A2) Can(x, y, z) (B1 x B2)
//! up to a global phase, Can(x, y, z) = exp(-i/2 (x XX + y YY + z ZZ)). The steps are the textbook ones: U in the
//! magic basis, M = U'^T U' (symmetric, unitary), M diagonalised by a real rotation (Jacobi on Re M, then on Im M inside
//! each group of equal eigenvalues: no random mixing, no retry), the local parts read back out of the magic basis.
//! A 4 x 4 matrix's index is 2 x + y (x: the first qubit). The decomposition is checked against U in the tests; the
//! caller also checks it (`check`) and lowers the block term by term when it does not hold.

use crate::linalg::*;

pub type M4 = [[C; 4]; 4];

pub fn id4() -> M4 {
    let mut m = [[ZERO; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = ONE;
    }
    m
}

pub fn mul4(a: &M4, b: &M4) -> M4 {
    let mut m = [[ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = ZERO;
            for k in 0..4 {
                s = s + a[i][k] * b[k][j];
            }
            m[i][j] = s;
        }
    }
    m
}

pub fn dag4(a: &M4) -> M4 {
    let mut m = [[ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[j][i].conj();
        }
    }
    m
}

/// u on qubit q (0: the first, the high bit of the index; 1: the second).
pub fn lift(u: &M2, q: usize) -> M4 {
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

pub fn kron(a: &M2, b: &M2) -> M4 {
    mul4(&lift(a, 0), &lift(b, 1))
}

/// exp(-i t/2 P P) for P = X, Y, Z (k = 0, 1, 2).
pub fn pp(k: usize, t: f64) -> M4 {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    let p: M2 = match k {
        0 => x(),
        1 => y(),
        _ => z(),
    };
    let pp = kron(&p, &p);
    let mut m = [[ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let d = if i == j { C::new(c, 0.0) } else { ZERO };
            m[i][j] = d + pp[i][j] * C::new(0.0, -s);
        }
    }
    m
}

pub fn can(ang: [f64; 3]) -> M4 {
    mul4(&pp(0, ang[0]), &mul4(&pp(1, ang[1]), &pp(2, ang[2])))
}

fn magic() -> M4 {
    let r = 1.0 / 2f64.sqrt();
    let (o, i) = (C::new(r, 0.0), C::new(0.0, r));
    // columns: (|00>+|11>), i(|00>-|11>), i(|01>+|10>), (|01>-|10>), each / sqrt 2
    [[o, i, ZERO, ZERO], [ZERO, ZERO, i, o], [ZERO, ZERO, i, -o], [o, -i, ZERO, ZERO]]
}

/// Cyclic Jacobi on a real symmetric n x n matrix: (eigenvalues, eigenvectors as columns).
fn jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// A real orthogonal P (det +1) with P^T M P diagonal, for M symmetric and unitary.
fn diagonalise(m: &M4) -> Option<[[f64; 4]; 4]> {
    let re: Vec<Vec<f64>> = (0..4).map(|i| (0..4).map(|j| m[i][j].re).collect()).collect();
    let im: Vec<Vec<f64>> = (0..4).map(|i| (0..4).map(|j| m[i][j].im).collect()).collect();
    let (vals, v) = jacobi(re);
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|&i, &j| vals[i].partial_cmp(&vals[j]).unwrap());
    let mut p = vec![vec![0.0; 4]; 4]; // columns in order
    for (c, &k) in order.iter().enumerate() {
        for r in 0..4 {
            p[r][c] = v[r][k];
        }
    }
    // groups of equal eigenvalues of Re M: diagonalise Im M inside each
    let sv: Vec<f64> = order.iter().map(|&k| vals[k]).collect();
    let mut start = 0;
    while start < 4 {
        let mut end = start + 1;
        while end < 4 && (sv[end] - sv[start]).abs() < 1e-9 {
            end += 1;
        }
        let g = end - start;
        if g > 1 {
            let mut sub = vec![vec![0.0; g]; g];
            for a in 0..g {
                for b in 0..g {
                    let mut s = 0.0;
                    for r in 0..4 {
                        for t in 0..4 {
                            s += p[r][start + a] * im[r][t] * p[t][start + b];
                        }
                    }
                    sub[a][b] = s;
                }
            }
            let (_, w) = jacobi(sub);
            let old: Vec<Vec<f64>> = (0..4).map(|r| (start..end).map(|c| p[r][c]).collect()).collect();
            for r in 0..4 {
                for b in 0..g {
                    p[r][start + b] = (0..g).map(|a| old[r][a] * w[a][b]).sum();
                }
            }
        }
        start = end;
    }
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = p[r][c];
        }
    }
    if det4r(&out) < 0.0 {
        for row in out.iter_mut() {
            row[0] = -row[0];
        }
    }
    // check: P^T M P diagonal
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                let mut s = ZERO;
                for r in 0..4 {
                    for t in 0..4 {
                        s = s + m[r][t].scale(out[r][i] * out[t][j]);
                    }
                }
                if s.abs() > 1e-8 {
                    return None;
                }
            }
        }
    }
    Some(out)
}

fn det4r(m: &[[f64; 4]; 4]) -> f64 {
    let mut a = *m;
    let mut det = 1.0;
    for c in 0..4 {
        let p = (c..4).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap()).unwrap();
        if a[p][c].abs() < 1e-300 {
            return 0.0;
        }
        if p != c {
            a.swap(p, c);
            det = -det;
        }
        det *= a[c][c];
        for r in c + 1..4 {
            let f = a[r][c] / a[c][c];
            for k in c..4 {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    det
}

/// K = A x B (up to a phase) as (A, B), each with determinant of modulus 1.
fn factor(k: &M4) -> (M2, M2) {
    let (mut bi, mut bj) = (0, 0);
    for i in 0..4 {
        for j in 0..4 {
            if k[i][j].abs() > k[bi][bj].abs() {
                bi = i;
                bj = j;
            }
        }
    }
    let (i0, k0, j0, l0) = (bi >> 1, bi & 1, bj >> 1, bj & 1);
    let mut b: M2 = [k[2 * i0][2 * j0], k[2 * i0][2 * j0 + 1], k[2 * i0 + 1][2 * j0], k[2 * i0 + 1][2 * j0 + 1]];
    let piv = b[2 * k0 + l0];
    let mut a: M2 = [ZERO; 4];
    for i in 0..2 {
        for j in 0..2 {
            a[2 * i + j] = k[2 * i + k0][2 * j + l0].div(piv);
        }
    }
    let norm = |m: &mut M2| {
        let d = (m[0] * m[3] - m[1] * m[2]).sqrt();
        for e in m.iter_mut() {
            *e = e.div(d);
        }
    };
    norm(&mut a);
    norm(&mut b);
    (a, b)
}

pub struct Kak {
    pub a1: M2,
    pub a2: M2,
    pub ang: [f64; 3],
    pub b1: M2,
    pub b2: M2,
}

/// U = (a1 x a2) Can(ang) (b1 x b2), up to a global phase; None when the diagonalisation does not hold.
pub fn kak(u: &M4) -> Option<Kak> {
    let bm = magic();
    let bd = dag4(&bm);
    let up = mul4(&bd, &mul4(u, &bm));
    // M = U'^T U'
    let mut ut = [[ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            ut[i][j] = up[j][i];
        }
    }
    let m = mul4(&ut, &up);
    let p = diagonalise(&m)?;
    // D = P^T M P, theta_k = arg(D_kk) / 2
    let mut th = [0.0; 4];
    for k in 0..4 {
        let mut s = ZERO;
        for r in 0..4 {
            for t in 0..4 {
                s = s + m[r][t].scale(p[r][k] * p[t][k]);
            }
        }
        th[k] = s.arg() / 2.0;
    }
    // O1 = U' P diag(e^{-i theta}): real orthogonal; fix its determinant by shifting theta_0 by pi
    let o1 = |th: &[f64; 4]| -> [[f64; 4]; 4] {
        let mut o = [[0.0; 4]; 4];
        for i in 0..4 {
            for k in 0..4 {
                let mut s = ZERO;
                for r in 0..4 {
                    s = s + up[i][r].scale(p[r][k]);
                }
                o[i][k] = (s * C::expi(-th[k])).re;
            }
        }
        o
    };
    let mut o = o1(&th);
    if det4r(&o) < 0.0 {
        th[0] += std::f64::consts::PI;
        o = o1(&th);
    }
    // the canonical angles: in the magic basis XX, YY, ZZ are diagonal with signs s_P; theta = phi + sum c_P s_P
    let sgn = |k: usize| -> [f64; 4] {
        let p2: M2 = match k {
            0 => x(),
            1 => y(),
            _ => z(),
        };
        let d = mul4(&bd, &mul4(&kron(&p2, &p2), &bm));
        [d[0][0].re, d[1][1].re, d[2][2].re, d[3][3].re]
    };
    let (sx_, sy_, sz_) = (sgn(0), sgn(1), sgn(2));
    // the four sign vectors with (1,1,1,1) are orthogonal (each of length 2): project
    let proj = |s: &[f64; 4]| -> f64 { (0..4).map(|k| s[k] * th[k]).sum::<f64>() / 4.0 };
    let (c1, c2, c3) = (proj(&sx_), proj(&sy_), proj(&sz_));
    let ang = [-2.0 * c1, -2.0 * c2, -2.0 * c3];
    // K1 = B O1 B^dag, K2 = B P^T B^dag
    let to_c = |r: &[[f64; 4]; 4]| -> M4 {
        let mut m = [[ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = C::new(r[i][j], 0.0);
            }
        }
        m
    };
    let mut pt = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            pt[i][j] = p[j][i];
        }
    }
    let k1 = mul4(&bm, &mul4(&to_c(&o), &bd));
    let k2 = mul4(&bm, &mul4(&to_c(&pt), &bd));
    let (a1, a2) = factor(&k1);
    let (b1, b2) = factor(&k2);
    let r = Kak { a1, a2, ang, b1, b2 };
    if check(u, &r) {
        Some(r)
    } else {
        None
    }
}

/// Whether the decomposition multiplies back to U, up to a global phase.
pub fn check(u: &M4, r: &Kak) -> bool {
    let back = mul4(&kron(&r.a1, &r.a2), &mul4(&can(r.ang), &kron(&r.b1, &r.b2)));
    eq4(u, &back, 1e-9)
}

pub fn eq4(a: &M4, b: &M4, tol: f64) -> bool {
    let (mut bi, mut bj) = (0, 0);
    for i in 0..4 {
        for j in 0..4 {
            if a[i][j].abs() > a[bi][bj].abs() {
                bi = i;
                bj = j;
            }
        }
    }
    if b[bi][bj].abs() < 1e-14 {
        return false;
    }
    let f = a[bi][bj].div(b[bi][bj]);
    (0..4).all(|i| (0..4).all(|j| (a[i][j] - b[i][j] * f).abs() < tol))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(k: u64) -> f64 {
        let mut v = k.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xA0761D6478BD642F;
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        (v % 1_000_000) as f64 / 1_000_000.0 * 6.283185307179586 - 3.141592653589793
    }

    fn rand_u(k: u64) -> M4 {
        let mut m = id4();
        for j in 0..6 {
            let s = 40 * k + 7 * j;
            let l = kron(&u3(rnd(s), rnd(s + 1), rnd(s + 2)), &u3(rnd(s + 3), rnd(s + 4), rnd(s + 5)));
            m = mul4(&l, &m);
            m = mul4(&pp((j % 3) as usize, rnd(s + 6)), &m);
        }
        m
    }

    #[test]
    fn decomposes_random_and_special() {
        for k in 0..500 {
            let u = rand_u(k);
            assert!(kak(&u).is_some(), "random {}", k);
        }
        // degenerate and special cases
        let special = [id4(), can([0.3, 0.3, 0.3]), can([PI2, 0.0, 0.0]), can([PI2, PI2, 0.0]), can([PI2, PI2, PI2]),
                       can([0.4, 0.0, 0.0]), can([0.4, 0.4, 0.0]), kron(&h(), &sx()), can([1.0, -1.0, 0.5])];
        for (i, u) in special.iter().enumerate() {
            assert!(kak(u).is_some(), "special {}", i);
        }
    }
    const PI2: f64 = std::f64::consts::FRAC_PI_2;

    #[test]
    fn magic_basis_makes_locals_real() {
        let b = magic();
        let l = kron(&u3(0.3, 1.2, -0.7), &u3(2.1, -0.4, 0.9));
        let d = l[0][0] * l[1][1] - l[0][1] * l[1][0]; // any entry: normalise the global phase via a 4th root of det
        let _ = d;
        let m = mul4(&dag4(&b), &mul4(&l, &b));
        // up to a global phase the matrix is real: m / m[k] is real for the largest entry k
        let (mut bi, mut bj) = (0, 0);
        for i in 0..4 {
            for j in 0..4 {
                if m[i][j].abs() > m[bi][bj].abs() {
                    bi = i;
                    bj = j;
                }
            }
        }
        let f = m[bi][bj];
        for i in 0..4 {
            for j in 0..4 {
                assert!(m[i][j].div(f).im.abs() < 1e-12);
            }
        }
    }
}
