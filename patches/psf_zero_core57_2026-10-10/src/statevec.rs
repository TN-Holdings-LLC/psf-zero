//! Item 57b (candidate 2026-10-10.c30): the per-gate loops of `excitation_cost`, `hybrid_cost` and item 39's
//! `_apply_ops`, moved out of Python. Standard library only, so that it can be built and tested anywhere
//! (`rustc --edition 2021 --test statevec.rs`); the Python binding (lib.rs, pyo3) is a thin wrapper that passes one
//! byte buffer in and a float (or a state) out.
//!
//! Conventions, as in psf_compile.py:
//! - the state is a C-order tensor with axis j for the j-th touched qubit, so axis j is bit (k - 1 - j) of the flat
//!   index;
//! - a gate matrix on qubits q[0..m] is Qiskit's: bit j of its row and column index belongs to q[j] (little-endian);
//! - a single-qubit gate is not applied to the state at once: it waits in `pend` until a wider gate on its qubit,
//!   and each qubit's 2x2 reduced state `rho` follows it meanwhile (item 49).
//!
//! The arithmetic is the Python code's, step for step, with sums formed in a fixed sequential order. NumPy may sum
//! in another order (pairwise or by BLAS), so values agree to rounding, not bit for bit.

use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    #[inline]
    pub fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }
    #[inline]
    pub fn conj(self) -> C64 {
        C64 { re: self.re, im: -self.im }
    }
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl std::ops::Add for C64 {
    type Output = C64;
    #[inline]
    fn add(self, o: C64) -> C64 {
        C64 { re: self.re + o.re, im: self.im + o.im }
    }
}

impl std::ops::AddAssign for C64 {
    #[inline]
    fn add_assign(&mut self, o: C64) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl std::ops::Mul for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, o: C64) -> C64 {
        C64 { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

/// A square matrix of side 2^m, row-major.
#[derive(Clone, Debug)]
pub struct Mat {
    pub dim: usize,
    pub a: Vec<C64>,
}

impl Mat {
    pub fn eye(dim: usize) -> Mat {
        let mut a = vec![C64::ZERO; dim * dim];
        for i in 0..dim {
            a[i * dim + i] = C64::ONE;
        }
        Mat { dim, a }
    }
    #[inline]
    pub fn at(&self, r: usize, c: usize) -> C64 {
        self.a[r * self.dim + c]
    }
    /// self @ other
    pub fn matmul(&self, o: &Mat) -> Mat {
        let d = self.dim;
        let mut out = vec![C64::ZERO; d * d];
        for r in 0..d {
            for c in 0..d {
                let mut s = C64::ZERO;
                for j in 0..d {
                    s += self.at(r, j) * o.at(j, c);
                }
                out[r * d + c] = s;
            }
        }
        Mat { dim: d, a: out }
    }
    /// The conjugate transpose.
    pub fn dagger(&self) -> Mat {
        let d = self.dim;
        let mut out = vec![C64::ZERO; d * d];
        for r in 0..d {
            for c in 0..d {
                out[c * d + r] = self.at(r, c).conj();
            }
        }
        Mat { dim: d, a: out }
    }
    /// np.kron(self, o): rows of `self` are the high part of the index.
    pub fn kron(&self, o: &Mat) -> Mat {
        let (da, db) = (self.dim, o.dim);
        let d = da * db;
        let mut out = vec![C64::ZERO; d * d];
        for ra in 0..da {
            for ca in 0..da {
                let x = self.at(ra, ca);
                for rb in 0..db {
                    for cb in 0..db {
                        out[(ra * db + rb) * d + ca * db + cb] = x * o.at(rb, cb);
                    }
                }
            }
        }
        Mat { dim: d, a: out }
    }
}

/// |0><0|, the reduced state of a qubit no gate has touched.
fn rho0() -> Mat {
    let mut m = Mat::eye(2);
    m.a[3] = C64::ZERO;
    m
}

/// One instruction: its touched-qubit positions (axes), matrix and the Target's properties for it.
#[derive(Clone, Debug)]
pub struct Op {
    pub pos: Vec<usize>,
    pub mat: Mat,
    /// The instruction has an entry in the Target (`props is not None`).
    pub has_props: bool,
    /// `props.error`, None if not reported.
    pub error: Option<f64>,
    /// `props.duration`, None if not reported.
    pub duration: Option<f64>,
}

/// The input of an estimate: k touched qubits (axes), each with its T1 and T2 (None if not reported), and the ops.
#[derive(Clone, Debug)]
pub struct Problem {
    pub k: usize,
    pub t1: Vec<Option<f64>>,
    pub t2: Vec<Option<f64>>,
    pub ops: Vec<Op>,
}

#[derive(Debug, PartialEq)]
pub enum Error {
    Truncated(&'static str),
    BadMagic,
    BadVersion(u32),
    BadKind(u32),
    TooManyQubits(usize),
    BadOp(usize, &'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub const MAX_QUBITS: usize = 16; // RESYNTH_MAX_QUBITS
pub const MAGIC: u32 = 0x3735_4650; // "PF57" little-endian
pub const VERSION: u32 = 1;

/// The state |0...0> on k qubits.
pub fn zero_state(k: usize) -> Vec<C64> {
    let mut s = vec![C64::ZERO; 1usize << k];
    s[0] = C64::ONE;
    s
}

#[inline]
fn bit_of_axis(k: usize, axis: usize) -> usize {
    k - 1 - axis
}

/// Applies `mat` (side 2^m, Qiskit's little-endian on `pos`) to the flat C-order state on k qubits, in place.
pub fn apply(state: &mut [C64], k: usize, pos: &[usize], mat: &Mat) {
    let m = pos.len();
    let d = 1usize << m;
    debug_assert_eq!(mat.dim, d);
    let bits: Vec<usize> = pos.iter().map(|&p| bit_of_axis(k, p)).collect();
    let mask: usize = bits.iter().map(|&b| 1usize << b).sum();
    // offset of each sub-index r (bit j of r -> bit bits[j] of the flat index)
    let off: Vec<usize> = (0..d)
        .map(|r| (0..m).filter(|&j| r >> j & 1 == 1).map(|j| 1usize << bits[j]).sum())
        .collect();
    let mut v = vec![C64::ZERO; d];
    let n = state.len();
    let mut base = 0usize;
    while base < n {
        if base & mask == 0 {
            for c in 0..d {
                v[c] = state[base + off[c]];
            }
            for r in 0..d {
                let row = &mat.a[r * d..(r + 1) * d];
                let mut s = C64::ZERO;
                for c in 0..d {
                    s += row[c] * v[c];
                }
                state[base + off[r]] = s;
            }
        }
        base += 1;
    }
}

/// A two-qubit gate (side 4, Qiskit's little-endian on `pos`), applied in place in one pass over the state, which
/// also forms the 2x2 reduced states of both qubits from the new amplitudes (item 57a's single pass, fused with the
/// gate). Returns (rho of pos[0], rho of pos[1]).
pub fn apply2_rho(state: &mut [C64], k: usize, pos: &[usize], mat: &Mat) -> (Mat, Mat) {
    debug_assert_eq!(mat.dim, 4);
    let (b0, b1) = (bit_of_axis(k, pos[0]), bit_of_axis(k, pos[1]));
    let (s0, s1) = (1usize << b0, 1usize << b1);
    let (lo, hi) = if b0 < b1 { (b0, b1) } else { (b1, b0) };
    let mut m = [C64::ZERO; 16];
    m.copy_from_slice(&mat.a);
    let (mut a00, mut a11, mut a01) = (0.0f64, 0.0f64, C64::ZERO); // pos[0]
    let (mut c00, mut c11, mut c01) = (0.0f64, 0.0f64, C64::ZERO); // pos[1]
    let (slo, shi) = (1usize << lo, 1usize << hi);
    let n = state.len();
    // bases with both bits zero, in contiguous runs of length slo: outer steps over the high bit, mid over the low
    let mut outer = 0usize;
    while outer < n {
        let mut mid = outer;
        while mid < outer + shi {
            for base in mid..mid + slo {
                let (i0, i1, i2, i3) = (base, base | s0, base | s1, base | s0 | s1); // r: bit 0 = pos[0], bit 1 = pos[1]
                let v = [state[i0], state[i1], state[i2], state[i3]];
                let w0 = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3];
                let w1 = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3];
                let w2 = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3];
                let w3 = m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3];
                state[i0] = w0;
                state[i1] = w1;
                state[i2] = w2;
                state[i3] = w3;
                let (n0, n1, n2, n3) = (w0.norm_sqr(), w1.norm_sqr(), w2.norm_sqr(), w3.norm_sqr());
                a00 += n0 + n2;
                a11 += n1 + n3;
                a01 += w0 * w1.conj() + w2 * w3.conj();
                c00 += n0 + n1;
                c11 += n2 + n3;
                c01 += w0 * w2.conj() + w1 * w3.conj();
            }
            mid += 2 * slo;
        }
        outer += 2 * shi;
    }
    let mk = |r00: f64, r11: f64, r01: C64| Mat { dim: 2, a: vec![C64::new(r00, 0.0), r01, r01.conj(), C64::new(r11, 0.0)] };
    (mk(a00, a11, a01), mk(c00, c11, c01))
}

/// `_rho1`: the 2x2 reduced density matrix of the qubit on `axis`.
pub fn rho1(state: &[C64], k: usize, axis: usize) -> Mat {
    let b = 1usize << bit_of_axis(k, axis);
    let (mut r00, mut r11, mut r01) = (0.0f64, 0.0f64, C64::ZERO);
    for i in 0..state.len() {
        if i & b != 0 {
            continue;
        }
        let a0 = state[i];
        let a1 = state[i | b];
        r00 += a0.norm_sqr();
        r11 += a1.norm_sqr();
        r01 += a0 * a1.conj(); // np.vdot(b, a) = sum conj(b) a
    }
    Mat { dim: 2, a: vec![C64::new(r00, 0.0), r01, r01.conj(), C64::new(r11, 0.0)] }
}

/// `_embed_1q`: on `qubits` (little-endian), mats[q] on each qubit q that has one, the identity elsewhere.
fn embed_1q(pre: &[(usize, Mat)], qubits: &[usize]) -> Mat {
    let mut out = Mat::eye(1);
    for &q in qubits.iter().rev() {
        let m = pre.iter().find(|(p, _)| *p == q).map(|(_, m)| m.clone()).unwrap_or_else(|| Mat::eye(2));
        out = out.kron(&m);
    }
    out
}

/// Item 49's bookkeeping shared by both estimates: the state, each qubit's reduced state and pending 1q products.
struct Walk {
    k: usize,
    state: Vec<C64>,
    rho: Vec<Option<Mat>>,
    pend: Vec<Option<Mat>>,
}

impl Walk {
    fn new(k: usize) -> Walk {
        Walk { k, state: zero_state(k), rho: vec![None; k], pend: vec![None; k] }
    }
    /// `_p1`
    fn p1(&self, i: usize) -> f64 {
        self.rho[i].as_ref().map(|r| r.at(1, 1).re).unwrap_or(0.0)
    }
    /// The gate step of both estimates, after the "before" costs.
    fn step(&mut self, op: &Op) {
        if op.pos.len() == 1 {
            let i = op.pos[0];
            let p = self.pend[i].take();
            self.pend[i] = Some(match p {
                None => op.mat.clone(),
                Some(p) => op.mat.matmul(&p),
            });
            let r = self.rho[i].clone().unwrap_or_else(rho0);
            self.rho[i] = Some(op.mat.matmul(&r).matmul(&op.mat.dagger()));
            return;
        }
        let mut pre: Vec<(usize, Mat)> = Vec::new();
        for &x in &op.pos {
            if let Some(m) = self.pend[x].take() {
                pre.push((x, m));
            }
        }
        let mat = if pre.is_empty() { op.mat.clone() } else { op.mat.matmul(&embed_1q(&pre, &op.pos)) };
        if op.pos.len() == 2 {
            let (r0, r1) = apply2_rho(&mut self.state, self.k, &op.pos, &mat);
            self.rho[op.pos[0]] = Some(r0);
            self.rho[op.pos[1]] = Some(r1);
            return;
        }
        apply(&mut self.state, self.k, &op.pos, &mat);
        for &i in &op.pos {
            self.rho[i] = Some(rho1(&self.state, self.k, i));
        }
    }
}

/// `excitation_cost`, from the ops (positions in place of physical qubits; T1 by position).
pub fn excitation_cost(p: &Problem) -> f64 {
    let mut w = Walk::new(p.k.max(1));
    let mut cost = 0.0f64;
    for op in &p.ops {
        if op.has_props {
            if let Some(e) = op.error {
                cost += -(1.0 - e).max(1e-300).ln();
            }
        }
        let dur = if op.has_props { op.duration.unwrap_or(0.0) } else { 0.0 };
        if dur != 0.0 {
            for &i in &op.pos {
                if let Some(t1) = p.t1[i] {
                    if t1 != 0.0 {
                        cost += dur / t1 * w.p1(i);
                    }
                }
            }
        }
        w.step(op);
    }
    cost
}

/// `hybrid_cost` without `readout_cost` (which the caller adds, as it reads measurements, not gates).
pub fn hybrid_cost_gates(p: &Problem) -> f64 {
    let thermal = |i: usize, t: f64| -> Option<(f64, f64)> {
        let t1 = p.t1[i]?;
        if t == 0.0 || t1 == 0.0 {
            return None;
        }
        let t2 = match p.t2[i] {
            Some(t2) if t2 != 0.0 => t2.min(2.0 * t1),
            _ => 2.0 * t1,
        };
        Some((t1, t2))
    };
    let mut w = Walk::new(p.k.max(1));
    let mut cost = 0.0f64;
    for op in &p.ops {
        let m = op.pos.len();
        let t = if op.has_props { op.duration.unwrap_or(0.0) } else { 0.0 };
        if op.has_props {
            for &i in &op.pos {
                if let Some(th) = thermal(i, t) {
                    cost += t / th.0 * w.p1(i);
                }
            }
        }
        w.step(op);
        if !op.has_props {
            continue;
        }
        let mut thermal_f = 1.0f64;
        for &i in &op.pos {
            let (t1, t2) = match thermal(i, t) {
                Some(th) => th,
                None => continue,
            };
            let rate = (1.0 / t2 - 1.0 / (2.0 * t1)).max(0.0);
            let ez = match &w.rho[i] {
                Some(r) => r.at(0, 0).re - r.at(1, 1).re,
                None => 1.0,
            };
            cost += (1.0 - (-t * rate).exp()) / 2.0 * (1.0 - ez * ez);
            thermal_f *= (1.0 + 2.0 * (-t / t2).exp() + (-t / t1).exp()) / 4.0;
        }
        let d = (1usize << m) as f64;
        let err = op.error.unwrap_or(0.0);
        cost += (err - (1.0 - (d * thermal_f + 1.0) / (d + 1.0))).max(0.0) * (d + 1.0) / d;
    }
    cost
}

/// `_apply_ops`: the ops applied to `state` in order, without item 49's waiting (as in the checks).
pub fn apply_ops(state: &mut [C64], k: usize, ops: &[Op]) {
    for op in ops {
        if op.pos.len() == 2 {
            apply2_rho(state, k, &op.pos, &op.mat);
        } else {
            apply(state, k, &op.pos, &op.mat);
        }
    }
}

// ---- the byte buffer passed from Python -------------------------------------------------------------------

/// What the buffer asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Excitation,
    HybridGates,
    ApplyOps,
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], Error> {
        if self.i + n > self.b.len() {
            return Err(Error::Truncated(what));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self, what: &'static str) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn f64s(&mut self, n: usize, what: &'static str) -> Result<Vec<f64>, Error> {
        let s = self.take(8 * n, what)?;
        Ok(s.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn u8s(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], Error> {
        self.take(n, what)
    }
}

fn opt(x: f64) -> Option<f64> {
    if x.is_nan() {
        None
    } else {
        Some(x)
    }
}

/// Parses the buffer:
///   u32 magic, u32 version, u32 kind (0 excitation, 1 hybrid gates, 2 apply ops), u32 k, u32 n_ops, u32 n_pos,
///   u32 n_mat (number of f64 in the matrices);
///   f64 t1[k], f64 t2[k] (NaN: not reported);
///   u8 m[n_ops], u8 pos[n_pos], u8 has_props[n_ops];
///   f64 error[n_ops], f64 duration[n_ops] (NaN: not reported);
///   f64 mat[n_mat] (each op's (2^m)^2 entries, row-major, real and imaginary parts interleaved);
///   for kind 2 only: f64 state[2 * 2^k] (interleaved).
/// Returns the kind, the problem, and the state for kind 2.
pub fn parse(buf: &[u8]) -> Result<(Kind, Problem, Option<Vec<C64>>), Error> {
    let mut r = Reader { b: buf, i: 0 };
    if r.u32("magic")? != MAGIC {
        return Err(Error::BadMagic);
    }
    let v = r.u32("version")?;
    if v != VERSION {
        return Err(Error::BadVersion(v));
    }
    let kind = match r.u32("kind")? {
        0 => Kind::Excitation,
        1 => Kind::HybridGates,
        2 => Kind::ApplyOps,
        x => return Err(Error::BadKind(x)),
    };
    let k = r.u32("k")? as usize;
    if k > MAX_QUBITS {
        return Err(Error::TooManyQubits(k));
    }
    let n_ops = r.u32("n_ops")? as usize;
    let n_pos = r.u32("n_pos")? as usize;
    let n_mat = r.u32("n_mat")? as usize;
    let t1: Vec<Option<f64>> = r.f64s(k, "t1")?.into_iter().map(opt).collect();
    let t2: Vec<Option<f64>> = r.f64s(k, "t2")?.into_iter().map(opt).collect();
    let ms = r.u8s(n_ops, "m")?;
    let pos = r.u8s(n_pos, "pos")?;
    let hp = r.u8s(n_ops, "has_props")?;
    let err = r.f64s(n_ops, "error")?;
    let dur = r.f64s(n_ops, "duration")?;
    let mat = r.f64s(n_mat, "mat")?;
    let (mut ip, mut im) = (0usize, 0usize);
    let mut ops = Vec::with_capacity(n_ops);
    for j in 0..n_ops {
        let m = ms[j] as usize;
        if m == 0 || m > 6 {
            return Err(Error::BadOp(j, "width"));
        }
        if ip + m > n_pos {
            return Err(Error::BadOp(j, "positions"));
        }
        let p: Vec<usize> = pos[ip..ip + m].iter().map(|&x| x as usize).collect();
        ip += m;
        if p.iter().any(|&x| x >= k) {
            return Err(Error::BadOp(j, "position out of range"));
        }
        for a in 0..m {
            if p[a + 1..].contains(&p[a]) {
                return Err(Error::BadOp(j, "repeated position"));
            }
        }
        let d = 1usize << m;
        if im + 2 * d * d > n_mat {
            return Err(Error::BadOp(j, "matrix"));
        }
        let a: Vec<C64> = (0..d * d).map(|e| C64::new(mat[im + 2 * e], mat[im + 2 * e + 1])).collect();
        im += 2 * d * d;
        ops.push(Op { pos: p, mat: Mat { dim: d, a }, has_props: hp[j] != 0, error: opt(err[j]), duration: opt(dur[j]) });
    }
    if ip != n_pos || im != n_mat {
        return Err(Error::Truncated("counts do not match"));
    }
    let state = if kind == Kind::ApplyOps {
        let s = r.f64s(2 << k, "state")?;
        Some(s.chunks_exact(2).map(|c| C64::new(c[0], c[1])).collect())
    } else {
        None
    };
    if r.i != buf.len() {
        return Err(Error::Truncated("trailing bytes"));
    }
    Ok((kind, Problem { k, t1, t2, ops }, state))
}

/// The state as interleaved little-endian f64 bytes (what the binding returns for kind 2).
pub fn state_bytes(s: &[C64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 * s.len());
    for c in s {
        out.extend_from_slice(&c.re.to_le_bytes());
        out.extend_from_slice(&c.im.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h() -> Mat {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Mat { dim: 2, a: vec![C64::new(s, 0.0), C64::new(s, 0.0), C64::new(s, 0.0), C64::new(-s, 0.0)] }
    }

    /// Qiskit's CX with control q[0], target q[1]: |q1 q0>: 01 -> 11, 11 -> 01.
    fn cx() -> Mat {
        let mut m = Mat { dim: 4, a: vec![C64::ZERO; 16] };
        for (r, c) in [(0, 0), (3, 1), (2, 2), (1, 3)] {
            m.a[r * 4 + c] = C64::ONE;
        }
        m
    }

    fn op(pos: Vec<usize>, mat: Mat) -> Op {
        Op { pos, mat, has_props: false, error: None, duration: None }
    }

    #[test]
    fn bell_state_with_qiskit_conventions() {
        // two qubits; axis 0 = touched qubit 0 = bit 1 of the flat index
        let mut s = zero_state(2);
        apply(&mut s, 2, &[0], &h());
        apply(&mut s, 2, &[0, 1], &cx());
        let a = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s[0].re - a).abs() < 1e-15 && (s[3].re - a).abs() < 1e-15);
        assert!(s[1].norm_sqr() < 1e-30 && s[2].norm_sqr() < 1e-30);
        let r = rho1(&s, 2, 0);
        assert!((r.at(0, 0).re - 0.5).abs() < 1e-15 && (r.at(1, 1).re - 0.5).abs() < 1e-15);
        assert!(r.at(0, 1).norm_sqr() < 1e-30);
    }

    #[test]
    fn control_is_the_first_qarg() {
        // X on axis 1 only, then CX(q[0]=axis 1 control, q[1]=axis 0 target): both end in |1>
        let x = Mat { dim: 2, a: vec![C64::ZERO, C64::ONE, C64::ONE, C64::ZERO] };
        let mut s = zero_state(2);
        apply(&mut s, 2, &[1], &x);
        apply(&mut s, 2, &[1, 0], &cx());
        assert!((s[3].re - 1.0).abs() < 1e-15);
    }

    #[test]
    fn waiting_single_qubit_gates_are_applied_with_the_next_wide_gate() {
        let p = Problem {
            k: 2,
            t1: vec![None, None],
            t2: vec![None, None],
            ops: vec![op(vec![0], h()), op(vec![0, 1], cx())],
        };
        let mut w = Walk::new(2);
        for o in &p.ops {
            w.step(o);
        }
        let a = std::f64::consts::FRAC_1_SQRT_2;
        assert!((w.state[0].re - a).abs() < 1e-15 && (w.state[3].re - a).abs() < 1e-15);
    }

    #[test]
    fn fused_two_qubit_pass_equals_apply_then_rho1() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for k in 2..7 {
            for a in 0..k {
                for b in 0..k {
                    if a == b {
                        continue;
                    }
                    let st: Vec<C64> = (0..1usize << k).map(|_| C64::new(rnd(), rnd())).collect();
                    let mat = Mat { dim: 4, a: (0..16).map(|_| C64::new(rnd(), rnd())).collect() };
                    let (mut x, mut y) = (st.clone(), st.clone());
                    apply(&mut x, k, &[a, b], &mat);
                    let (r0, r1) = apply2_rho(&mut y, k, &[a, b], &mat);
                    for i in 0..x.len() {
                        assert!((x[i].re - y[i].re).abs() < 1e-14 && (x[i].im - y[i].im).abs() < 1e-14);
                    }
                    for (r, ax) in [(r0, a), (r1, b)] {
                        let e = rho1(&x, k, ax);
                        for j in 0..4 {
                            assert!((r.a[j].re - e.a[j].re).abs() < 1e-12 && (r.a[j].im - e.a[j].im).abs() < 1e-12);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn parse_rejects_bad_buffers() {
        assert_eq!(parse(&[0u8; 3]).unwrap_err(), Error::Truncated("magic"));
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(parse(&b).unwrap_err(), Error::BadMagic);
    }
}
