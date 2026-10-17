//! Complex numbers and 2 x 2 matrices: all the arithmetic the compiler needs.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub const fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    pub fn expi(a: f64) -> C {
        C::new(a.cos(), a.sin())
    }
    pub fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    pub fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    pub fn sqrt(self) -> C {
        let r = self.abs().sqrt();
        let a = self.arg() / 2.0;
        C::new(r * a.cos(), r * a.sin())
    }
    pub fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
}

impl Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Neg for C {
    type Output = C;
    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }
}

pub const ZERO: C = C::new(0.0, 0.0);
pub const ONE: C = C::new(1.0, 0.0);
pub const I: C = C::new(0.0, 1.0);

/// A 2 x 2 matrix, row major: [[m[0], m[1]], [m[2], m[3]]].
pub type M2 = [C; 4];

pub fn mul2(a: &M2, b: &M2) -> M2 {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

pub fn dag2(a: &M2) -> M2 {
    [a[0].conj(), a[2].conj(), a[1].conj(), a[3].conj()]
}

pub fn eye() -> M2 {
    [ONE, ZERO, ZERO, ONE]
}

pub fn rx(t: f64) -> M2 {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    [C::new(c, 0.0), C::new(0.0, -s), C::new(0.0, -s), C::new(c, 0.0)]
}
pub fn ry(t: f64) -> M2 {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    [C::new(c, 0.0), C::new(-s, 0.0), C::new(s, 0.0), C::new(c, 0.0)]
}
pub fn rz(t: f64) -> M2 {
    [C::expi(-t / 2.0), ZERO, ZERO, C::expi(t / 2.0)]
}
pub fn phase(l: f64) -> M2 {
    [ONE, ZERO, ZERO, C::expi(l)]
}
pub fn u3(t: f64, p: f64, l: f64) -> M2 {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    [C::new(c, 0.0), -C::expi(l).scale(s), C::expi(p).scale(s), C::expi(p + l).scale(c)]
}
pub fn h() -> M2 {
    let r = 1.0 / 2f64.sqrt();
    [C::new(r, 0.0), C::new(r, 0.0), C::new(r, 0.0), C::new(-r, 0.0)]
}
pub fn x() -> M2 {
    [ZERO, ONE, ONE, ZERO]
}
pub fn y() -> M2 {
    [ZERO, -I, I, ZERO]
}
pub fn z() -> M2 {
    [ONE, ZERO, ZERO, -ONE]
}
pub fn sx() -> M2 {
    [C::new(0.5, 0.5), C::new(0.5, -0.5), C::new(0.5, -0.5), C::new(0.5, 0.5)]
}

/// True when a and b are equal up to a global phase, within tol (used by the tests).
#[allow(dead_code)]
pub fn eq_phase(a: &M2, b: &M2, tol: f64) -> bool {
    let k = (0..4).max_by(|&i, &j| a[i].abs().partial_cmp(&a[j].abs()).unwrap()).unwrap();
    if b[k].abs() < 1e-14 {
        return false;
    }
    let f = a[k].div(b[k]);
    (0..4).all(|i| (a[i] - b[i] * f).abs() < tol)
}

pub fn is_diag(a: &M2, tol: f64) -> bool {
    a[1].abs() < tol && a[2].abs() < tol
}

/// Angle reduced to (-pi, pi].
pub fn wrap(a: f64) -> f64 {
    let mut r = a % (2.0 * PI);
    if r <= -PI {
        r += 2.0 * PI;
    }
    if r > PI {
        r -= 2.0 * PI;
    }
    r
}

/// ZYZ angles (phi, theta, lambda) with U = e^{ia} Rz(phi) Ry(theta) Rz(lambda), theta in [0, pi].
pub fn zyz(u: &M2) -> (f64, f64, f64) {
    let det = u[0] * u[3] - u[1] * u[2];
    let s = det.sqrt();
    let v = [u[0].div(s), u[1].div(s), u[2].div(s), u[3].div(s)]; // in SU(2)
    let theta = 2.0 * v[2].abs().atan2(v[0].abs());
    let sum = 2.0 * v[3].arg(); // phi + lambda
    let diff = 2.0 * v[2].arg(); // phi - lambda
    if v[2].abs() < 1e-12 {
        return (sum, 0.0, 0.0);
    }
    if v[0].abs() < 1e-12 {
        return (diff, theta, 0.0);
    }
    ((sum + diff) / 2.0, theta, (sum - diff) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(k: u64) -> f64 {
        // a fixed pseudo-random sequence for the tests (xorshift)
        let mut x = k.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % 1_000_000) as f64 / 1_000_000.0 * 2.0 * PI - PI
    }

    #[test]
    fn zyz_reconstructs() {
        for k in 0..2000 {
            let u = mul2(&mul2(&rz(rnd(3 * k)), &ry(rnd(3 * k + 1).abs())), &rz(rnd(3 * k + 2)));
            let u = mul2(&u, &phase(rnd(k + 7)));
            let (p, t, l) = zyz(&u);
            let back = mul2(&mul2(&rz(p), &ry(t)), &rz(l));
            assert!(eq_phase(&u, &back, 1e-9), "k {}", k);
        }
        for m in [eye(), x(), y(), z(), h(), sx(), rz(0.3), ry(PI), rx(PI)] {
            let (p, t, l) = zyz(&m);
            assert!(eq_phase(&m, &mul2(&mul2(&rz(p), &ry(t)), &rz(l)), 1e-9));
        }
    }
}
