//! psfbind (2026-10-21): the template binding of `psf_ai_compile.RingTemplate.angles` in Rust, for many value vectors
//! at once, through a C ABI (no dependencies; Python calls it with ctypes).
//!
//! A template is a list of runs; each run is a list of factors, applied in order (U = f_k ... f_2 f_1): a factor is a
//! fixed 2 x 2 complex matrix or a rotation rz / ry / rx of one of the input's values. For each run the product U is
//! written as ZYZ angles (theta, phi, lam), U = e^{i a} Rz(phi) Ry(theta) Rz(lam), exactly as `_t_zyz` (the same
//! special cases at 1e-15), and the run's three outputs are (lam - pi/2, theta, phi + pi/2) when `use_rx`, else
//! (lam, theta + pi, phi + pi), as `RingTemplate.angles`. Nothing is chosen: the arithmetic follows the Python line by
//! line (the same order of products), so the results agree to rounding (checked in TOQB's tests).

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub const fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }
    /// Principal square root, as Python's cmath.sqrt.
    fn sqrt(self) -> C {
        if self.re == 0.0 && self.im == 0.0 {
            return C::new(0.0, self.im);
        }
        let r = self.abs();
        let a = ((r + self.re.abs()) / 2.0).sqrt();
        if self.re >= 0.0 {
            C::new(a, self.im / (2.0 * a))
        } else {
            let b = self.im.abs() / (2.0 * a);
            C::new(b, if self.im >= 0.0 { a } else { -a })
        }
    }
    fn expi(t: f64) -> C {
        C::new(t.cos(), t.sin())
    }
}

pub type M = [C; 4];

/// A @ B for tuples (a, b, c, d) = [[a, b], [c, d]], as `_mul`.
fn mul(a: &M, b: &M) -> M {
    [
        a[0].mul(b[0]).add(a[1].mul(b[2])),
        a[0].mul(b[1]).add(a[1].mul(b[3])),
        a[2].mul(b[0]).add(a[3].mul(b[2])),
        a[2].mul(b[1]).add(a[3].mul(b[3])),
    ]
}

/// rz / ry / rx(t) as `_rot` (kind 1, 2, 3).
pub fn rot(kind: u8, t: f64) -> M {
    let z = C::new(0.0, 0.0);
    match kind {
        1 => [C::expi(-0.5 * t), z, z, C::expi(0.5 * t)],
        2 => {
            let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
            [C::new(c, 0.0), C::new(-s, 0.0), C::new(s, 0.0), C::new(c, 0.0)]
        }
        _ => {
            let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
            [C::new(c, 0.0), C::new(0.0, -s), C::new(0.0, -s), C::new(c, 0.0)]
        }
    }
}

/// (theta, phi, lam) as `_t_zyz`.
pub fn zyz(u: &M) -> (f64, f64, f64) {
    let (a, b, c, d) = (u[0], u[1], u[2], u[3]);
    let sq = a.mul(d).sub(b.mul(c)).sqrt();
    let (a, c, d) = (a.div(sq), c.div(sq), d.div(sq));
    let theta = 2.0 * c.abs().atan2(a.abs());
    let p11 = d.phase();
    let p10 = if c.abs() > 1e-15 { c.phase() } else { 0.0 };
    if c.abs() <= 1e-15 {
        return (0.0, 2.0 * p11, 0.0);
    }
    if a.abs() <= 1e-15 {
        return (PI, 2.0 * p10, 0.0);
    }
    (theta, p11 + p10, p11 - p10)
}

/// One run's three outputs (RingTemplate.angles).
pub fn run_angles(factors: &[(u8, usize, M)], values: &[f64], use_rx: bool) -> [f64; 3] {
    let one = C::new(1.0, 0.0);
    let zero = C::new(0.0, 0.0);
    let mut u: M = [one, zero, zero, one];
    for (kind, idx, m) in factors {
        let f = if *kind == 0 { *m } else { rot(*kind, values[*idx]) };
        u = mul(&f, &u);
    }
    let (theta, phi, lam) = zyz(&u);
    let half = PI / 2.0;
    if use_rx {
        [lam - half, theta, phi + half]
    } else {
        [lam, theta + PI, phi + PI]
    }
}

/// C ABI. Runs: `n_runs`, `run_start[n_runs + 1]` (factor offsets); factors: `kind[nf]` (0 fixed, 1 rz, 2 ry, 3 rx),
/// `index[nf]` (the value index for a rotation), `fixed[8 nf]` (re, im of m00, m01, m10, m11 for a fixed factor).
/// Values: `batch` rows of `n_values`; output: `batch` rows of `3 n_runs`. Returns 0, or -1 on a bad index.
#[no_mangle]
pub unsafe extern "C" fn psfbind_angles(
    n_runs: usize,
    run_start: *const usize,
    kind: *const u8,
    index: *const usize,
    fixed: *const f64,
    use_rx: i32,
    batch: usize,
    n_values: usize,
    values: *const f64,
    out: *mut f64,
) -> i32 {
    let rs = std::slice::from_raw_parts(run_start, n_runs + 1);
    let nf = rs[n_runs];
    let kind = std::slice::from_raw_parts(kind, nf);
    let index = std::slice::from_raw_parts(index, nf);
    let fixed = std::slice::from_raw_parts(fixed, 8 * nf);
    let mut facs: Vec<(u8, usize, M)> = Vec::with_capacity(nf);
    for k in 0..nf {
        if kind[k] != 0 && index[k] >= n_values {
            return -1;
        }
        let f = &fixed[8 * k..8 * k + 8];
        facs.push((kind[k], index[k], [C::new(f[0], f[1]), C::new(f[2], f[3]), C::new(f[4], f[5]), C::new(f[6], f[7])]));
    }
    let vals = std::slice::from_raw_parts(values, batch * n_values);
    let outs = std::slice::from_raw_parts_mut(out, batch * 3 * n_runs);
    for b in 0..batch {
        let v = &vals[b * n_values..(b + 1) * n_values];
        for r in 0..n_runs {
            let a = run_angles(&facs[rs[r]..rs[r + 1]], v, use_rx != 0);
            outs[b * 3 * n_runs + 3 * r..b * 3 * n_runs + 3 * r + 3].copy_from_slice(&a);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn zyz_reconstructs() {
        let u = mul(&rot(1, 0.3), &mul(&rot(2, 1.1), &rot(1, -0.7)));
        let (t, p, l) = zyz(&u);
        let v = mul(&rot(1, p), &mul(&rot(2, t), &rot(1, l)));
        // equal up to a global phase
        let ph = u[0].div(v[0]);
        for k in 0..4 {
            let w = v[k].mul(ph);
            assert!(close(w.re, u[k].re) && close(w.im, u[k].im));
        }
    }

    #[test]
    fn sqrt_matches_principal_branch() {
        for (re, im) in [(1.0, 0.0), (-1.0, 0.0), (-1.0, -0.0), (0.3, -2.0), (-4.0, 1e-20)] {
            let s = C::new(re, im).sqrt();
            let back = s.mul(s);
            assert!(close(back.re, re) && close(back.im, im));
            assert!(s.re >= 0.0);
        }
    }

    #[test]
    fn diagonal_and_antidiagonal_cases() {
        assert_eq!(zyz(&rot(1, 0.8)).0, 0.0);
        let x = [C::new(0.0, 0.0), C::new(1.0, 0.0), C::new(1.0, 0.0), C::new(0.0, 0.0)];
        assert!(close(zyz(&x).0, PI));
    }
}
