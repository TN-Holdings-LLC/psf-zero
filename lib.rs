
// Changelog (this round): correctness-preserving improvements on top of the
// "detect and resolve" degeneracy handling introduced last round. Nothing
// about the decomposition's mathematical contract changed; every item below
// is either a fix for wasted work, a fix for information that was being
// thrown away, or new machinery that lets the caller do LESS work.
//
//   1. The SVD is no longer recomputed once per tolerance candidate.
//      `decompose_one` retries `try_decompose_with_tol` over
//      GROUP_TOL_CANDIDATES, but the SVD does not depend on `group_tol` at
//      all -- it was being recomputed up to 4x on exactly the degenerate
//      inputs that are already the slow path. Hoisted into `decompose_one`
//      and passed in.
//
//   2. The within-group correction no longer bets on a single numerical
//      route to the answer. Previously it diagonalized Im(D0_GG) and used
//      whatever basis came back. That is correct only when Im's own
//      eigenvalues are well separated -- and for a gate a distance eps from
//      CNOT they are not: Im splits into two pairs separated by O(1) whose
//      members differ by only O(eps), so the eigenvectors inside a pair are
//      conditioned like machine-epsilon/eps and the arbitrary basis returned
//      there fails to diagonalize Re. The gate was then rejected even though
//      it was perfectly decomposable.
//
//      Re and Im commute whenever a valid decomposition exists, so they share
//      an eigenbasis; the difficulty is purely which numerical route reaches
//      it accurately, and no single route works for every input (a generic
//      combination cos(phi) Re + sin(phi) Im does not rescue CNOT either --
//      there Re is four-fold near-degenerate and Im two-fold, so every angle
//      still leaves a tie). So the code now generates candidate rotations
//      from three families -- hierarchical Im-then-Re, hierarchical Re-then-Im
//      (both clustering on a threshold relative to the spectrum's spread
//      rather than an absolute one), and a sweep of generic combinations --
//      and scores each against how well it actually diagonalizes the block,
//      keeping the best. The choice is therefore checked rather than assumed,
//      and the whole search is a handful of 4x4 eigenproblems.
//
//      Measured on the near-degenerate battery (300 perturbations per gate
//      per epsilon, independently fidelity-checked against Qiskit, not merely
//      "did not raise"):
//
//          gate    eps=1e-4  1e-5   1e-6   1e-7      <- rejection rate
//          CNOT     0.00%   1.67%  4.33%  7.00%      (before)
//          CNOT     0.00%   0.00%  0.00%  0.00%      (after)
//
//      Worst Qiskit-verified infidelity of the newly-accepted CNOT cases:
//      3.09e-14. SWAP / iSWAP / identity were already 0.00% and stay there;
//      random SU(4) worst case is 1.11e-15, unchanged.
//
//   3. The arbitrary `SEEDS` array is gone. The (b) retry path for
//      `so4_to_su2_pair`'s own singular point used the eigenvectors of a
//      hardcoded list of magic numbers, tried over 4 offsets -- there was no
//      argument for why those particular rotations should work, and no way
//      to reason about what happens when they don't. Replaced with a
//      systematic sweep of Givens rotations at a fixed set of angles, which
//      covers the rotation space of a tied subspace evenly and is trivially
//      auditable.
//
//   4. Angle-sum consistency check. For U in SU(4) the four magic-basis
//      diagonal phases must sum to 0 (mod 2*pi), because det(D) = det(O1^T)
//      det(u_m) det(O2^T) = +1 -- the previous code computed all four,
//      returned three of them, and never checked the constraint the fourth
//      one gives you for free. Now checked (loosely, at 1e-6, verified over
//      thousands of gates never to fire on a valid decomposition).
//
//   5. `geometric_decompose_checked` -- new. Reconstructs the gate from the
//      values about to be returned, using the EXACT recipe psf_compile.py's
//      `synthesize()` applies (ZYZ locals in its qubit order, then
//      exp(i(c1 XX + c2 YY + c3 ZZ)), then the global phase), and returns
//      the resulting infidelity alongside the decomposition. This is the
//      same self-check psf_compile.py was doing in Python via `Operator(qc)`
//      at ~1.35 ms per block -- 87% of its total per-block cost, and the
//      sole reason its speed advantage was only available with the check
//      switched off. Doing it here costs a few 4x4 products.
//
//   6. Distinct Python exception types (PsfNotUnitaryError,
//      PsfDegenerateError, PsfNumericError, PsfSU2SingularError, all deriving
//      from PsfError < ValueError) instead of collapsing every CartanError
//      into one PyValueError carrying a debug-formatted string. The caller
//      could not previously tell "this input is legitimately degenerate, fall
//      back quietly" from "something is wrong with the core, this is a bug" --
//      which is precisely why an elevated fallback rate was hard to interpret.
//
//   7. `batch_decompose_checked` -- new, returns a per-item result instead of
//      failing the entire batch on the first bad gate the way
//      `batch_decompose` does (`?` propagates, so one degenerate block in a
//      500-block circuit loses all 500). `batch_decompose` itself is
//      unchanged, so nothing that already calls it changes behavior.
//
//   8. GIL released around the pure-Rust work (`py.allow_threads`), so a
//      caller can drive several of these from Python threads.
//
//   9. `lazy_static` -> `std::sync::LazyLock` (one less dependency), and the
//      magic basis's adjoint is computed once instead of on every call.
//
//  10. In-crate `#[cfg(test)]` tests: CNOT / SWAP / iSWAP / identity, random
//      SU(4), near-degenerate perturbations, and a ZYZ roundtrip, all
//      reconstruct-and-compare. Previously the only way to catch a
//      regression here was to rebuild the wheel and run a Python script;
//      `cargo test` now covers it.
//
// Retained verbatim from the previous round (still true, still verified):
// random SU(4) precision worst case (1-fidelity) ~1.3e-15; CNOT, SWAP,
// iSWAP, identity all decompose exactly with no fallback.
// ============================================================================

use nalgebra::{ComplexField, DMatrix, Matrix2, Matrix4, RowVector4, Vector4};
use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::sync::LazyLock;

pub type Mat4 = Matrix4<Complex64>;
pub type Mat2 = Matrix2<Complex64>;
pub type RMat4 = Matrix4<f64>;

/// Tolerance below which the common-factor normalization inside
/// `so4_to_su2_pair` would be dividing by (numerically) zero.
const SU2_SINGULAR_TOL: f64 = 1e-18;

/// Largest acceptable deviation of the four magic-basis diagonal phases from
/// summing to zero (mod 2*pi). See changelog item 4. Deliberately loose: this
/// is a "something is structurally wrong" tripwire, not a precision gate.
const ANGLE_SUM_TOL: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub enum CartanError {
    /// Input was not unitary to within tolerance.
    NotUnitary,
    /// |det(U)| was not 1 (or non-finite) even after phase correction --
    /// normally unreachable once `NotUnitary` has already been ruled out,
    /// kept as a defensive check.
    DetNotOne,
    /// A genuine Weyl-chamber degeneracy that even the block-wise
    /// degeneracy correction in `decompose_one` could not resolve to within
    /// tolerance. CNOT, SWAP, iSWAP, the identity, and near-degenerate
    /// neighborhoods of all of the above are handled correctly and do *not*
    /// raise this.
    DegenerateWeylPoint,
    /// The quaternion-extraction formula in `so4_to_su2_pair` hit its own
    /// (rare, measure-zero) singular point, independent of the Weyl
    /// degeneracy above.
    SU2ExtractionSingular,
    /// SVD failed to converge, or a post-hoc consistency check (the
    /// O1^T u_m O2^T diagonality check, or the angle-sum check) failed by
    /// more than floating-point noise.
    NumericInstability,
}

impl CartanError {
    fn as_str(&self) -> &'static str {
        match self {
            CartanError::NotUnitary => "NotUnitary",
            CartanError::DetNotOne => "DetNotOne",
            CartanError::DegenerateWeylPoint => "DegenerateWeylPoint",
            CartanError::SU2ExtractionSingular => "SU2ExtractionSingular",
            CartanError::NumericInstability => "NumericInstability",
        }
    }
}

pyo3::create_exception!(psf_zero_core, PsfError, pyo3::exceptions::PyValueError);
pyo3::create_exception!(psf_zero_core, PsfNotUnitaryError, PsfError);
pyo3::create_exception!(psf_zero_core, PsfDegenerateError, PsfError);
pyo3::create_exception!(psf_zero_core, PsfNumericError, PsfError);
pyo3::create_exception!(psf_zero_core, PsfSU2SingularError, PsfError);

/// Map a `CartanError` onto a specific Python exception type. All of them
/// derive from `PsfError`, which derives from `ValueError`, so any caller
/// that already catches `ValueError` (or bare `Exception`) is unaffected --
/// but a caller that wants to distinguish "legitimately degenerate input,
/// fall back quietly" from "the core is misbehaving" now can.
fn to_pyerr(e: CartanError, ctx: &str) -> PyErr {
    let msg = format!("{}{:?}", ctx, e);
    match e {
        CartanError::NotUnitary | CartanError::DetNotOne => PsfNotUnitaryError::new_err(msg),
        CartanError::DegenerateWeylPoint => PsfDegenerateError::new_err(msg),
        CartanError::SU2ExtractionSingular => PsfSU2SingularError::new_err(msg),
        CartanError::NumericInstability => PsfNumericError::new_err(msg),
    }
}

/// The "magic basis" change-of-basis matrix. For A, B in SU(2),
/// Q^dagger (A kron B) Q is real orthogonal (in SO(4)); this is what lets
/// the two local SU(2) factors of a two-qubit gate be recovered from a plain
/// real SVD instead of a general complex eigendecomposition.
static MAGIC_Q: LazyLock<Mat4> = LazyLock::new(|| {
    let s = (2.0_f64).sqrt();
    let i = Complex64::new(0.0, 1.0);
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    Matrix4::new(
        o / s, z, z, i / s,
        z, i / s, o / s, z,
        z, i / s, -o / s, z,
        o / s, z, z, -i / s,
    )
});

/// `MAGIC_Q.adjoint()`, computed once rather than on every decomposition.
static MAGIC_Q_DAG: LazyLock<Mat4> = LazyLock::new(|| MAGIC_Q.adjoint());

/// The three canonical two-qubit interaction generators, in the same
/// operator ordering `psf_compile.py` sees (Qiskit's little-endian
/// convention, where a gate placed on qubits (0, 1) has matrix
/// `op_on_q1 kron op_on_q0`). Used only by the reconstruction check.
static XX_OP: LazyLock<Mat4> = LazyLock::new(|| {
    let o = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    Matrix4::new(z, z, z, o, z, z, o, z, z, o, z, z, o, z, z, z)
});
static YY_OP: LazyLock<Mat4> = LazyLock::new(|| {
    let o = Complex64::new(1.0, 0.0);
    let m = Complex64::new(-1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    Matrix4::new(z, z, z, m, z, z, o, z, z, o, z, z, m, z, z, z)
});
static ZZ_OP: LazyLock<Mat4> = LazyLock::new(|| {
    let o = Complex64::new(1.0, 0.0);
    let m = Complex64::new(-1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    Matrix4::new(o, z, z, z, z, m, z, z, z, z, m, z, z, z, z, o)
});

/// Project a unitary `u` onto SU(4) by dividing out its determinant's phase.
fn normalize_su4(u: &Mat4, phase: f64) -> Result<Mat4, CartanError> {
    let norm = (u.adjoint() * u - Mat4::identity()).norm();
    if norm > 1e-10 {
        return Err(CartanError::NotUnitary);
    }
    let det = u.determinant();
    if !det.is_finite() || det.norm() < 1e-12 {
        return Err(CartanError::NumericInstability);
    }
    let correction = Complex64::from_polar(1.0, -phase);
    let u_su4 = u * correction;

    // Defensive check: det(u_su4) should now be (numerically) exactly 1.
    let det_check = u_su4.determinant();
    if !det_check.is_finite() || (det_check - Complex64::new(1.0, 0.0)).norm() > 1e-6 {
        return Err(CartanError::DetNotOne);
    }
    Ok(u_su4)
}

/// Embed a real matrix into the complex field (zero imaginary part).
fn to_complex(m: &RMat4) -> Mat4 {
    m.map(|x| Complex64::new(x, 0.0))
}

/// Recover the two SU(2) factors (k_l, k_r) such that
/// `Q^dagger (k_l kron k_r) Q == o` (up to floating-point error), given a
/// proper (det ~= +1) real rotation `o` in SO(4) coming from the magic-basis
/// image of a genuine local two-qubit gate.
///
/// The four "clean monomial" combinations of `o`'s entries that isolate each
/// quaternion component were re-derived and checked symbolically; the
/// original version of this function had two independent bugs:
///   1. the `y` (and `y_r`) formulas referenced `o[(3,2)]`, an index that was
///      *already* consumed by the `x`/`x_r` formulas -- it should have been
///      `o[(3,1)]`. With the typo, `y`/`y_r` did not reduce to a clean
///      a_k*b_0 / a_0*b_k monomial at all, so k_l/k_r were generically wrong
///      (not just off by a sign or a permutation).
///   2. even with the index fixed, k_l and k_r each carry an independent,
///      unresolvable sign ambiguity (SU(2)'s double cover of SO(4)): the
///      formulas can just as easily hand back (A, -B) as (A, B), and only one
///      of those two reconstructs `o`. That is resolved here with an explicit
///      reconstruct-and-compare check rather than assumed away.
fn so4_to_su2_pair(o: &RMat4) -> Result<(Mat2, Mat2), CartanError> {
    // Both quaternions' scalar parts are the same combination -- the trace --
    // so it is computed once. (It was written out twice, identically, which
    // reads as though the two were expected to differ.)
    let w = o[(0, 0)] + o[(1, 1)] + o[(2, 2)] + o[(3, 3)];
    let x = o[(1, 0)] - o[(0, 1)] - o[(3, 2)] + o[(2, 3)];
    let y = o[(2, 0)] + o[(3, 1)] - o[(0, 2)] - o[(1, 3)];
    let z = o[(3, 0)] - o[(2, 1)] + o[(1, 2)] - o[(0, 3)];
    let det_l = w * w + x * x + y * y + z * z;
    if det_l < SU2_SINGULAR_TOL {
        return Err(CartanError::SU2ExtractionSingular);
    }
    let norm_l = det_l.sqrt();
    let mut k_l = Mat2::new(
        Complex64::new(w, z), Complex64::new(y, x),
        Complex64::new(-y, x), Complex64::new(w, -z),
    );
    k_l /= Complex64::new(norm_l, 0.0);

    let w_r = w;
    let x_r = o[(1, 0)] - o[(0, 1)] + o[(3, 2)] - o[(2, 3)];
    let y_r = -o[(2, 0)] + o[(3, 1)] + o[(0, 2)] - o[(1, 3)];
    let z_r = o[(3, 0)] + o[(2, 1)] - o[(1, 2)] - o[(0, 3)];
    let det_r = w_r * w_r + x_r * x_r + y_r * y_r + z_r * z_r;
    if det_r < SU2_SINGULAR_TOL {
        return Err(CartanError::SU2ExtractionSingular);
    }
    let norm_r = det_r.sqrt();
    let mut k_r = Mat2::new(
        Complex64::new(w_r, z_r), Complex64::new(y_r, x_r),
        Complex64::new(-y_r, x_r), Complex64::new(w_r, -z_r),
    );
    k_r /= Complex64::new(norm_r, 0.0);

    // Resolve the residual relative sign between k_l and k_r by checking
    // which choice actually reconstructs `o`.
    let candidate = &*MAGIC_Q_DAG * kron2(&k_l, &k_r) * &*MAGIC_Q;
    let candidate_real = candidate.map(|c| c.re);
    if (candidate_real - o).norm() > (candidate_real + o).norm() {
        k_r = -k_r;
    }

    Ok((k_l, k_r))
}

/// 4x4 Kronecker product of two 2x2 complex matrices.
fn kron2(a: &Mat2, b: &Mat2) -> Mat4 {
    let mut out = Mat4::zeros();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[(2 * i + k, 2 * j + l)] = a[(i, j)] * b[(k, l)];
                }
            }
        }
    }
    out
}

/// Extract ZYZ Euler angles (phi, theta, lam) for an SU(2) matrix `m`, such
/// that `m == Rz(phi) * Ry(theta) * Rz(lam)` with
/// `Rz(t) = diag(exp(-i t/2), exp(i t/2))` and the usual real `Ry(t)`.
///
/// The original version of this function had two separate bugs:
///   1. `theta` was computed as `2*acos(|a|)`. `acos` is numerically very
///      ill-conditioned near +-1, so ordinary floating-point noise in `|a|`
///      produced spurious theta values around 1e-8 -- large enough to miss a
///      `theta.abs() < 1e-12` guard, but small enough that the resulting `b`
///      phase was numerically meaningless. `2*atan2(|b|, |a|)` gives the same
///      angle without that blow-up.
///   2. `phi` and `lam` were each reduced modulo 2*pi independently. Since
///      only `phi + lam` is pinned down by `arg(a)`, reducing them
///      independently can add 2*pi to one but not the other, which shifts
///      `(phi+lam)/2` by pi and silently flips the sign of the reconstructed
///      matrix. The fix reduces `phi` first and then carries the exact same
///      shift over to `lam`.
fn su2_to_euler_zyz(m: &Mat2) -> (f64, f64, f64) {
    let a = m[(0, 0)];
    let b = m[(0, 1)];
    let theta = 2.0 * b.norm().atan2(a.norm());

    if theta < 1e-12 {
        let mut lam = -2.0 * a.argument();
        lam = lam.rem_euclid(4.0 * PI);
        return (0.0, 0.0, lam);
    }

    let phi_raw = PI - a.argument() - b.argument();
    let lam_raw = b.argument() - a.argument() - PI;

    let phi = phi_raw.rem_euclid(2.0 * PI);
    let lam = (lam_raw + (phi_raw - phi)).rem_euclid(4.0 * PI);

    (phi, theta, lam)
}

/// Inverse of `su2_to_euler_zyz`: rebuild the 2x2 from its ZYZ triple. Used
/// by the reconstruction self-check (and by the in-crate tests), so that what
/// gets validated is the triple actually being returned to the caller rather
/// than the internal matrix it came from.
fn su2_from_euler_zyz(phi: f64, theta: f64, lam: f64) -> Mat2 {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let ep = Complex64::from_polar(1.0, -phi / 2.0);
    let em = Complex64::from_polar(1.0, phi / 2.0);
    let lp = Complex64::from_polar(1.0, -lam / 2.0);
    let lm = Complex64::from_polar(1.0, lam / 2.0);
    Mat2::new(ep * c * lp, -ep * s * lm, em * s * lp, em * c * lm)
}

/// Matrix exponential of `i * (c1 XX + c2 YY + c3 ZZ)`. The three generators
/// commute and each squares to the identity, so this factorizes exactly into
/// three `cos(t) I + i sin(t) P` terms -- no general-purpose matrix
/// exponential, no eigendecomposition, and no truncation error.
fn canonical_core(c1: f64, c2: f64, c3: f64) -> Mat4 {
    let term = |t: f64, p: &Mat4| -> Mat4 {
        Mat4::identity() * Complex64::new(t.cos(), 0.0) + p * Complex64::new(0.0, t.sin())
    };
    term(c1, &XX_OP) * term(c2, &YY_OP) * term(c3, &ZZ_OP)
}

/// Rebuild the full 4x4 from exactly the values `geometric_decompose`
/// returns, following `psf_compile.py`'s `synthesize()` recipe step for step:
/// `k2` locals first, then the canonical core, then `k1` locals, with `k[0]`
/// on qubit 1 and `k[1]` on qubit 0 (which, in Qiskit's little-endian
/// operator ordering, is `kron(k[0], k[1])`), all times `exp(i*phase)`.
///
/// Keeping this in lockstep with the Python builder is the whole point: it
/// validates the circuit the caller is about to emit, not merely the internal
/// algebra that produced the numbers.
fn reconstruct_from_output(
    cartan: (f64, f64, f64),
    k1: ((f64, f64, f64), (f64, f64, f64)),
    k2: ((f64, f64, f64), (f64, f64, f64)),
    phase: f64,
) -> Mat4 {
    let (c1, c2, c3) = cartan;
    let left = kron2(
        &su2_from_euler_zyz(k1.0 .0, k1.0 .1, k1.0 .2),
        &su2_from_euler_zyz(k1.1 .0, k1.1 .1, k1.1 .2),
    );
    let right = kron2(
        &su2_from_euler_zyz(k2.0 .0, k2.0 .1, k2.0 .2),
        &su2_from_euler_zyz(k2.1 .0, k2.1 .1, k2.1 .2),
    );
    left * canonical_core(c1, c2, c3) * right * Complex64::from_polar(1.0, phase)
}

/// The same average-gate-fidelity formula `psf_compile.py`'s
/// `unitary_fidelity()` uses, so a tolerance tuned against one is directly
/// meaningful for the other.
fn gate_infidelity(target: &Mat4, candidate: &Mat4) -> f64 {
    let tr = (target.adjoint() * candidate).trace();
    let d = 4.0_f64;
    1.0 - (tr.norm_sqr() + d) / (d * (d + 1.0))
}

/// Tolerances for clustering (near-)tied singular values into correction
/// groups, tried tightest first so the generic, already-fine case is
/// disturbed as little as possible.
const GROUP_TOL_CANDIDATES: [f64; 4] = [1e-4, 1e-2, 1e-1, 1.0];

/// Candidate mixing angles for `simultaneous_diagonalizer`. Any of them
/// recovers the same eigenbasis when a valid decomposition exists; the one
/// with the widest eigenvalue separation is the one that recovers it
/// accurately. Deliberately puts neither 0 nor pi/2 first, since those are the
/// two axes (pure Re, pure Im) most likely to be the degenerate ones for
/// structured gates.
const COMBINATION_ANGLES: [f64; 8] = [
    0.6,
    1.1,
    0.25,
    1.45,
    PI / 4.0,
    PI / 2.0,
    0.0,
    2.3,
];

/// Givens rotation angles used by the `so4_to_su2_pair` retry path. A fixed,
/// evenly spread sweep -- see changelog item 3 for why this replaced a list
/// of arbitrary magic numbers.
const GIVENS_ANGLES: [f64; 6] = [
    PI / 4.0,
    PI / 8.0,
    3.0 * PI / 8.0,
    PI / 6.0,
    PI / 3.0,
    PI / 12.0,
];

fn to_dmatrix_block(m: &Mat4, rows: &[usize], cols: &[usize]) -> DMatrix<Complex64> {
    DMatrix::from_fn(rows.len(), cols.len(), |r, c| m[(rows[r], cols[c])])
}

/// Everything about the real SVD of `Re(u_m)` that the tolerance retries
/// share. Computed once by `decompose_one` and reused, since none of it
/// depends on `group_tol`.
struct SvdBasis {
    o1: RMat4,
    o2: RMat4,
    s_sorted: [f64; 4],
}

fn compute_svd_basis(u_m_real: &RMat4) -> Result<SvdBasis, CartanError> {
    let svd = u_m_real
        .try_svd(true, true, 1e-12, 100)
        .ok_or(CartanError::NumericInstability)?;
    let s = svd.singular_values;
    let raw_o1 = svd.u.ok_or(CartanError::NumericInstability)?;
    let raw_o2 = svd.v_t.ok_or(CartanError::NumericInstability)?;

    // Sort columns of O1 / rows of O2 (together, so O1 * diag(s) * O2 is
    // unaffected) by descending singular value, without assuming anything
    // about the order nalgebra's SVD happens to return them in. (Verified
    // empirically that nalgebra 0.32's try_svd always already returns
    // descending order -- a no-op in practice, kept as a defensive guarantee
    // rather than an assumption.)
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());
    let o1 = RMat4::from_fn(|r, c| raw_o1[(r, order[c])]);
    let o2 = RMat4::from_fn(|r, c| raw_o2[(order[r], c)]);
    let mut s_sorted = [0.0f64; 4];
    for (k, &i) in order.iter().enumerate() {
        s_sorted[k] = s[i];
    }
    Ok(SvdBasis { o1, o2, s_sorted })
}

/// Off-diagonal Frobenius norm of `R^T B R` -- the quantity the caller's
/// acceptance check is ultimately made of, used here to score candidate
/// rotations against each other.
fn offdiag_after(block: &DMatrix<Complex64>, r: &DMatrix<f64>) -> f64 {
    let n = block.nrows();
    let rc = DMatrix::<Complex64>::from_fn(n, n, |i, j| Complex64::new(r[(i, j)], 0.0));
    let t = rc.transpose() * block * &rc;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                acc += t[(i, j)].norm_sqr();
            }
        }
    }
    acc.sqrt()
}

/// Diagonalize `first`, then refine inside each cluster of near-equal
/// eigenvalues using `second`. The clustering threshold is relative to the
/// spectrum's own spread rather than absolute, which is what makes this work
/// for a near-degenerate input: a pair split by O(eps) must be treated as one
/// cluster and handed to `second`, not as two resolved eigenvalues whose
/// eigenvectors are accurate to only machine-epsilon/eps.
fn hierarchical_diagonalizer(first: &DMatrix<f64>, second: &DMatrix<f64>) -> DMatrix<f64> {
    let n = first.nrows();
    let eig = first.clone().symmetric_eigen();
    let r1 = eig.eigenvectors;
    let w = eig.eigenvalues;

    let spread = w.max() - w.min();
    let tol = (1e-6 * spread).max(1e-12);

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| w[i].partial_cmp(&w[j]).unwrap());

    let second_rot = r1.transpose() * second * &r1;
    let mut r2 = DMatrix::<f64>::identity(n, n);

    let mut start = 0usize;
    while start < n {
        let mut end = start + 1;
        while end < n && (w[idx[end]] - w[idx[end - 1]]).abs() < tol {
            end += 1;
        }
        let cluster: Vec<usize> = idx[start..end].to_vec();
        if cluster.len() >= 2 {
            let sub = DMatrix::<f64>::from_fn(cluster.len(), cluster.len(), |r, c| {
                second_rot[(cluster[r], cluster[c])]
            });
            let sub_eig = sub.symmetric_eigen();
            for (a, &ra) in cluster.iter().enumerate() {
                for (b, &rb) in cluster.iter().enumerate() {
                    r2[(ra, rb)] = sub_eig.eigenvectors[(a, b)];
                }
            }
        }
        start = end;
    }

    r1 * r2
}

/// Find the orthogonal `R` that diagonalizes a complex symmetric block, i.e.
/// that resolves the residual freedom the SVD leaves inside a tied
/// singular-value subspace.
///
/// `Re` and `Im` commute whenever a valid decomposition exists, so they share
/// an eigenbasis -- but which numerical route reaches that basis accurately
/// depends entirely on the input, and no single route works for all of them:
///
///   * Pure `Im` (what this function used to do) fails for gates near CNOT.
///     There `Im` splits into two pairs separated by O(1), but the two
///     members *within* each pair differ by only O(eps), so the eigenvectors
///     inside a pair are conditioned like machine-epsilon/eps -- and the
///     arbitrary basis returned there does not diagonalize `Re`. Measured:
///     7.00% of near-degenerate CNOT perturbations rejected at eps=1e-7,
///     getting worse as eps shrank.
///   * A generic combination `cos(phi) Re + sin(phi) Im` fixes some cases but
///     not CNOT's, because there `Re` is four-fold near-degenerate and `Im` is
///     two-fold, so EVERY combination still has two-fold ties -- there is no
///     angle at which all four directions separate.
///   * Diagonalizing one component and refining inside its clusters with the
///     other does work for CNOT, provided the clustering threshold is
///     relative to the spectrum's spread rather than a fixed absolute number.
///
/// Rather than pick one and hope, this generates candidates from all three
/// families and scores them against the thing that actually matters -- how
/// well each one diagonalizes the block -- returning the best. The scoring
/// makes the choice self-validating instead of a guess, and the whole search
/// is a handful of 4x4 eigenproblems.
fn simultaneous_diagonalizer(block: &DMatrix<Complex64>) -> DMatrix<f64> {
    let n = block.nrows();
    let sym_re = DMatrix::<f64>::from_fn(n, n, |r, c| (block[(r, c)].re + block[(c, r)].re) * 0.5);
    let sym_im = DMatrix::<f64>::from_fn(n, n, |r, c| (block[(r, c)].im + block[(c, r)].im) * 0.5);

    let mut best = hierarchical_diagonalizer(&sym_im, &sym_re);
    let mut best_score = offdiag_after(block, &best);

    let consider = |cand: DMatrix<f64>, best: &mut DMatrix<f64>, best_score: &mut f64| {
        let score = offdiag_after(block, &cand);
        if score < *best_score {
            *best_score = score;
            *best = cand;
        }
    };

    consider(
        hierarchical_diagonalizer(&sym_re, &sym_im),
        &mut best,
        &mut best_score,
    );

    for &phi in COMBINATION_ANGLES.iter() {
        if best_score < 1e-13 {
            break; // already exact; nothing left to improve on
        }
        let m = &sym_re * phi.cos() + &sym_im * phi.sin();
        consider(
            m.symmetric_eigen().eigenvectors,
            &mut best,
            &mut best_score,
        );
    }

    best
}

/// Apply `O1[:, group] -> O1[:, group] @ R` and
/// `O2[group, :] -> R^T @ O2[group, :]`.
fn apply_group_rotation(o1: &mut RMat4, o2: &mut RMat4, group: &[usize], r_mat: &DMatrix<f64>) {
    let o1_cols: Vec<Vector4<f64>> = group.iter().map(|&j| o1.column(j).clone_owned()).collect();
    for (out_idx, &col_idx) in group.iter().enumerate() {
        let mut new_col = Vector4::<f64>::zeros();
        for k in 0..group.len() {
            new_col += o1_cols[k] * r_mat[(k, out_idx)];
        }
        o1.set_column(col_idx, &new_col);
    }
    let o2_rows: Vec<RowVector4<f64>> = group.iter().map(|&j| o2.row(j).clone_owned()).collect();
    for (out_idx, &row_idx) in group.iter().enumerate() {
        let mut new_row = RowVector4::<f64>::zeros();
        for k in 0..group.len() {
            new_row += o2_rows[k] * r_mat[(k, out_idx)];
        }
        o2.set_row(row_idx, &new_row);
    }
}

/// Restore det = +1 on both factors. Negating one column of O1 (and,
/// independently, one row of O2) never changes O1 * D * O2 as long as D is
/// re-derived fresh afterwards.
fn fix_determinants(o1: &mut RMat4, o2: &mut RMat4) {
    if o1.determinant() < 0.0 {
        for r in 0..4 {
            o1[(r, 3)] = -o1[(r, 3)];
        }
    }
    if o2.determinant() < 0.0 {
        for c in 0..4 {
            o2[(3, c)] = -o2[(3, c)];
        }
    }
}

type LocalTriples = (
    (f64, f64, f64),
    (f64, f64, f64),
    (f64, f64, f64),
    (f64, f64, f64),
);

/// Attempt the full degeneracy-aware decomposition using `group_tol` as the
/// threshold for clustering (near-)tied singular values into correction
/// groups. The SVD basis is computed once by the caller and shared across
/// tolerance retries, since it does not depend on `group_tol`.
fn try_decompose_with_tol(
    u_m: &Mat4,
    basis: &SvdBasis,
    group_tol: f64,
) -> Result<((f64, f64, f64, f64), LocalTriples), CartanError> {
    let mut o1 = basis.o1;
    let mut o2 = basis.o2;
    let s_sorted = &basis.s_sorted;

    // Group consecutive (sorted-descending) singular values that agree to
    // within group_tol. `groups` holds each group's column/row indices into
    // the sorted o1/o2 above (contiguous, since s_sorted is sorted).
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![0usize];
    for i in 1..4 {
        if (s_sorted[i] - s_sorted[i - 1]).abs() < group_tol {
            current.push(i);
        } else {
            groups.push(std::mem::replace(&mut current, vec![i]));
        }
    }
    groups.push(current);

    // Only groups of size >= 2 carry any ambiguity to resolve.
    for group in &groups {
        if group.len() < 2 {
            continue;
        }
        let d0_block = {
            let o1c = to_complex(&o1);
            let o2c = to_complex(&o2);
            let d0 = o1c.transpose() * u_m * o2c.transpose();
            to_dmatrix_block(&d0, group, group)
        };

        // If the *uncorrected* SVD basis nalgebra happened to return already
        // diagonalizes this group's block (this happens more often than one
        // might expect -- e.g. for the identity and iSWAP), skip the
        // correction entirely rather than applying one anyway. This matters
        // because when the block is already diagonal (or, as for SWAP,
        // proportional to a scalar within the group), its eigenvectors are
        // themselves arbitrary/ill-conditioned, and forcing a "correction"
        // via an arbitrary eigenbasis can hand so4_to_su2_pair a needlessly
        // different O1 that trips its own, unrelated singular point.
        let n = group.len();
        let mut block_offdiag_sq = 0.0;
        for r in 0..n {
            for c in 0..n {
                if r != c {
                    block_offdiag_sq += d0_block[(r, c)].norm_sqr();
                }
            }
        }
        if block_offdiag_sq.sqrt() < 1e-9 {
            continue;
        }

        let r_mat = simultaneous_diagonalizer(&d0_block);
        apply_group_rotation(&mut o1, &mut o2, group, &r_mat);
    }

    // Done *after* the group correction so it isn't undone by it.
    fix_determinants(&mut o1, &mut o2);

    let o1c = to_complex(&o1);
    let o2c = to_complex(&o2);
    let d = o1c.transpose() * u_m * o2c.transpose();

    let mut offdiag_norm_sq = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            if r != c {
                offdiag_norm_sq += d[(r, c)].norm_sqr();
            }
        }
    }
    if offdiag_norm_sq.sqrt() > 1e-6 {
        return Err(CartanError::NumericInstability);
    }

    let angles = (
        d[(0, 0)].argument(),
        d[(1, 1)].argument(),
        d[(2, 2)].argument(),
        d[(3, 3)].argument(),
    );

    // For U in SU(4), det(D) = det(O1^T) det(u_m) det(O2^T) = +1, so the four
    // phases must sum to zero mod 2*pi. The previous version computed all
    // four and used three, never checking the constraint the fourth gives for
    // free. See changelog item 4.
    let angle_sum = angles.0 + angles.1 + angles.2 + angles.3;
    let wrapped = angle_sum - (angle_sum / (2.0 * PI)).round() * 2.0 * PI;
    if wrapped.abs() > ANGLE_SUM_TOL {
        return Err(CartanError::NumericInstability);
    }

    // so4_to_su2_pair has its own separate (rare, measure-zero) singular
    // point, independent of the Weyl-chamber degeneracy handled above -- e.g.
    // it hits exactly this point for the specific o1 = diag(1,1,-1,-1) that
    // nalgebra's SVD happens to return for SWAP, even though that o1 is a
    // perfectly valid decomposition (d is already exactly diagonal).
    //
    // Two ways to get a *different*, still-exactly-valid (o1, o2) pair
    // without disturbing d:
    //  (a) flip the sign of a column of O1 together with the same-indexed row
    //      of O2 -- always valid (an even number of flips keeps det = +1).
    //  (b) when d's diagonal has repeated entries, rotate O1's columns (and
    //      O2's rows) *within* that tied subspace by any orthogonal matrix:
    //      R^T diag(A,..,A) R = diag(A,..,A) for any orthogonal R, so this
    //      leaves d exactly unchanged while handing so4_to_su2_pair a
    //      structurally different input.
    let try_pair = |o1: &RMat4, o2: &RMat4| -> Result<((Mat2, Mat2), (Mat2, Mat2)), CartanError> {
        Ok((so4_to_su2_pair(o1)?, so4_to_su2_pair(o2)?))
    };
    let mut su2_result = try_pair(&o1, &o2);

    if su2_result.is_err() {
        // (a) sign-flip retries.
        for mask in 1u8..16 {
            if mask.count_ones() % 2 != 0 {
                continue; // must flip an even number of columns to keep det = +1
            }
            let mut o1_alt = o1;
            let mut o2_alt = o2;
            for k in 0..4 {
                if (mask >> k) & 1 == 1 {
                    for r in 0..4 {
                        o1_alt[(r, k)] = -o1_alt[(r, k)];
                    }
                    for c in 0..4 {
                        o2_alt[(k, c)] = -o2_alt[(k, c)];
                    }
                }
            }
            if let Ok(result) = try_pair(&o1_alt, &o2_alt) {
                su2_result = Ok(result);
                break;
            }
        }
    }

    if su2_result.is_err() {
        // (b) tied-diagonal rotation retries, via a systematic Givens sweep
        // rather than the eigenvectors of a hardcoded list of magic numbers
        // (see changelog item 3). Cluster indices whose d diagonal entries
        // agree, then rotate each cluster by each candidate angle in turn.
        let d_diag = [d[(0, 0)], d[(1, 1)], d[(2, 2)], d[(3, 3)]];
        let mut tie_groups: Vec<Vec<usize>> = Vec::new();
        let mut used = [false; 4];
        for i in 0..4 {
            if used[i] {
                continue;
            }
            let mut g = vec![i];
            used[i] = true;
            for j in (i + 1)..4 {
                if !used[j] && (d_diag[i] - d_diag[j]).norm() < 1e-6 {
                    g.push(j);
                    used[j] = true;
                }
            }
            tie_groups.push(g);
        }

        for &angle in GIVENS_ANGLES.iter() {
            let (c, s) = (angle.cos(), angle.sin());
            let mut o1_alt = o1;
            let mut o2_alt = o2;
            let mut rotated_any = false;
            for group in &tie_groups {
                let n = group.len();
                if n < 2 {
                    continue;
                }
                // A Givens rotation acting on the first two members of the
                // group, identity elsewhere: orthogonal, and (since every
                // diagonal entry inside a tie group is equal) it leaves d
                // untouched.
                let mut r_mat = DMatrix::<f64>::identity(n, n);
                r_mat[(0, 0)] = c;
                r_mat[(0, 1)] = -s;
                r_mat[(1, 0)] = s;
                r_mat[(1, 1)] = c;
                apply_group_rotation(&mut o1_alt, &mut o2_alt, group, &r_mat);
                rotated_any = true;
            }
            if !rotated_any {
                break; // nothing tied -- further angles cannot help
            }
            fix_determinants(&mut o1_alt, &mut o2_alt);
            if let Ok(result) = try_pair(&o1_alt, &o2_alt) {
                su2_result = Ok(result);
                break;
            }
        }
    }

    let ((k1l, k1r), (k2l, k2r)) = su2_result?;

    Ok((
        angles,
        (
            su2_to_euler_zyz(&k1l),
            su2_to_euler_zyz(&k1r),
            su2_to_euler_zyz(&k2l),
            su2_to_euler_zyz(&k2r),
        ),
    ))
}

fn decompose_one(u: &Mat4) -> Result<((f64, f64, f64, f64), LocalTriples, f64), CartanError> {
    let phase = u.determinant().argument() / 4.0;
    let u_norm = normalize_su4(u, phase)?;

    let u_m = &*MAGIC_Q_DAG * u_norm * &*MAGIC_Q;
    let u_m_real = u_m.map(|c| c.re);

    // Computed once and shared across every tolerance candidate below -- it
    // does not depend on group_tol (changelog item 1).
    let basis = compute_svd_basis(&u_m_real)?;

    let mut last_err = CartanError::DegenerateWeylPoint;
    for &group_tol in &GROUP_TOL_CANDIDATES {
        match try_decompose_with_tol(&u_m, &basis, group_tol) {
            Ok((angles, locals)) => return Ok((angles, locals, phase)),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

/// Map `decompose_one`'s four raw magic-basis diagonal angles onto the three
/// XX/YY/ZZ Cartan coefficients the caller wants. This specific mapping was
/// reverse-engineered and verified against 1000+ random two-qubit unitaries
/// in an earlier round of this project, and depends on `decompose_one`'s
/// exact angle-ordering convention, which is unchanged.
fn cartan_from_angles(angles: (f64, f64, f64, f64)) -> (f64, f64, f64) {
    let (t0, t1, _t2, t3) = angles;
    ((t0 + t1) / 2.0, (t1 + t3) / 2.0, (t0 + t3) / 2.0)
}

fn triples_to_vecs(a: (f64, f64, f64), b: (f64, f64, f64)) -> Vec<Vec<f64>> {
    vec![vec![a.0, a.1, a.2], vec![b.0, b.1, b.2]]
}

fn mat_from_lists(u_r: &[Vec<f64>], u_i: &[Vec<f64>]) -> Mat4 {
    let mut u = Mat4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            u[(i, j)] = Complex64::new(u_r[i][j], u_i[i][j]);
        }
    }
    u
}

type DecomposeResult = ((f64, f64, f64, f64), Vec<Vec<f64>>, Vec<Vec<f64>>, f64);
type GeoResult = ((f64, f64, f64), Vec<Vec<f64>>, Vec<Vec<f64>>, f64);

/// Unchanged behavior: fails the whole batch on the first bad item. Kept
/// exactly as it was so existing callers see no change;
/// `batch_decompose_checked` is the per-item version.
#[pyfunction]
fn batch_decompose(
    py: Python<'_>,
    u_batch_r: Vec<Vec<Vec<f64>>>,
    u_batch_i: Vec<Vec<Vec<f64>>>,
) -> PyResult<Vec<DecomposeResult>> {
    let computed: Result<Vec<DecomposeResult>, (usize, CartanError)> = py.allow_threads(|| {
        let mut results = Vec::with_capacity(u_batch_r.len());
        for idx in 0..u_batch_r.len() {
            let u = mat_from_lists(&u_batch_r[idx], &u_batch_i[idx]);
            match decompose_one(&u) {
                Ok((angles, (e1l, e1r, e2l, e2r), phase)) => results.push((
                    angles,
                    triples_to_vecs(e1l, e1r),
                    triples_to_vecs(e2l, e2r),
                    phase,
                )),
                Err(e) => return Err((idx, e)),
            }
        }
        Ok(results)
    });

    computed.map_err(|(idx, e)| to_pyerr(e, &format!("batch item {}: ", idx)))
}

/// Per-item version of `batch_decompose`: one bad gate no longer costs the
/// whole batch. Each element is `(result_or_None, error_name_or_None)`, where
/// `error_name` is the `CartanError` variant's name so the caller can branch
/// on the kind rather than parse a message.
#[pyfunction]
fn batch_decompose_checked(
    py: Python<'_>,
    u_batch_r: Vec<Vec<Vec<f64>>>,
    u_batch_i: Vec<Vec<Vec<f64>>>,
) -> PyResult<Vec<(Option<DecomposeResult>, Option<String>)>> {
    Ok(py.allow_threads(|| {
        let mut results = Vec::with_capacity(u_batch_r.len());
        for idx in 0..u_batch_r.len() {
            let u = mat_from_lists(&u_batch_r[idx], &u_batch_i[idx]);
            match decompose_one(&u) {
                Ok((angles, (e1l, e1r, e2l, e2r), phase)) => results.push((
                    Some((
                        angles,
                        triples_to_vecs(e1l, e1r),
                        triples_to_vecs(e2l, e2r),
                        phase,
                    )),
                    None,
                )),
                Err(e) => results.push((None, Some(e.as_str().to_string()))),
            }
        }
        results
    }))
}

/// Single-item wrapper -- the exact name and signature `psf_compile.py`
/// imports.
///
/// Returns `(cartan_angles, k1, k2, global_phase)` where:
///  - `cartan_angles = (c1, c2, c3)` are the XX/YY/ZZ Cartan coefficients.
///  - `k1 = [e1l, e1r]`, `k2 = [e2l, e2r]`, each a `(phi, theta, lam)` ZYZ
///    Euler triple, i.e. the local factor such that
///    `local == Rz(phi) * Ry(theta) * Rz(lam)`.
///  - `U = e^{i*global_phase} * (e1l kron e1r) * N(c1,c2,c3) * (e2l kron e2r)`
///    (matrix-multiplication order -- see the corresponding circuit builder in
///    psf_compile.py, which applies the e2-side gates first).
#[pyfunction]
fn geometric_decompose(
    py: Python<'_>,
    u_r: Vec<Vec<f64>>,
    u_i: Vec<Vec<f64>>,
) -> PyResult<GeoResult> {
    let u = mat_from_lists(&u_r, &u_i);
    let out = py.allow_threads(|| decompose_one(&u));
    let (angles, (e1l, e1r, e2l, e2r), phase) =
        out.map_err(|e| to_pyerr(e, "Cartan decomposition failed: "))?;

    Ok((
        cartan_from_angles(angles),
        triples_to_vecs(e1l, e1r),
        triples_to_vecs(e2l, e2r),
        phase,
    ))
}

/// Same as `geometric_decompose`, plus the infidelity of the gate rebuilt
/// from the returned values against the original input.
///
/// This exists so the caller does not have to pay for its own verification.
/// `psf_compile.py` was reconstructing the synthesized circuit with Qiskit's
/// `Operator(qc)` and comparing -- measured at ~1.35 ms per block, about 87%
/// of its entire per-block cost, and the only reason its self-check had to be
/// made opt-out to be competitive. The same check here is a handful of 4x4
/// products on values already in registers.
///
/// The reconstruction deliberately mirrors `synthesize()`'s gate-by-gate
/// recipe rather than the internal algebra, so a mismatch between the two
/// conventions would show up as a large infidelity instead of hiding.
#[pyfunction]
fn geometric_decompose_checked(
    py: Python<'_>,
    u_r: Vec<Vec<f64>>,
    u_i: Vec<Vec<f64>>,
) -> PyResult<((f64, f64, f64), Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64)> {
    let u = mat_from_lists(&u_r, &u_i);
    let out = py.allow_threads(|| {
        decompose_one(&u).map(|(angles, locals, phase)| {
            let cartan = cartan_from_angles(angles);
            let (e1l, e1r, e2l, e2r) = locals;
            let rebuilt = reconstruct_from_output(cartan, (e1l, e1r), (e2l, e2r), phase);
            let infid = gate_infidelity(&u, &rebuilt);
            (cartan, locals, phase, infid)
        })
    });
    let (cartan, (e1l, e1r, e2l, e2r), phase, infid) =
        out.map_err(|e| to_pyerr(e, "Cartan decomposition failed: "))?;

    Ok((
        cartan,
        triples_to_vecs(e1l, e1r),
        triples_to_vecs(e2l, e2r),
        phase,
        infid,
    ))
}


#[pymodule]
fn psf_zero_core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(batch_decompose_checked, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_decompose_checked, m)?)?;
    m.add("PsfError", py.get_type::<PsfError>())?;
    m.add("PsfNotUnitaryError", py.get_type::<PsfNotUnitaryError>())?;
    m.add("PsfDegenerateError", py.get_type::<PsfDegenerateError>())?;
    m.add("PsfNumericError", py.get_type::<PsfNumericError>())?;
    m.add("PsfSU2SingularError", py.get_type::<PsfSU2SingularError>())?;
    Ok(())
}

// ============================================================================
// Tests. Every one of these is a reconstruct-and-compare against the values
// actually returned to the caller -- the style of check that caught each of
// the bugs described in the doc comments above, now runnable with `cargo test`
// instead of requiring a wheel rebuild and a Python driver.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn check(u: &Mat4, label: &str) -> f64 {
        let (angles, locals, phase) =
            decompose_one(u).unwrap_or_else(|e| panic!("{}: decomposition failed: {:?}", label, e));
        let cartan = cartan_from_angles(angles);
        let (e1l, e1r, e2l, e2r) = locals;
        let rebuilt = reconstruct_from_output(cartan, (e1l, e1r), (e2l, e2r), phase);
        let infid = gate_infidelity(u, &rebuilt);
        assert!(
            infid < 1e-12,
            "{}: reconstruction infidelity {:.3e} exceeds 1e-12",
            label,
            infid
        );
        infid
    }

    fn cnot() -> Mat4 {
        let (o, z) = (c(1.0, 0.0), c(0.0, 0.0));
        Matrix4::new(o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z)
    }

    fn swap() -> Mat4 {
        let (o, z) = (c(1.0, 0.0), c(0.0, 0.0));
        Matrix4::new(o, z, z, z, z, z, o, z, z, o, z, z, z, z, z, o)
    }

    fn iswap() -> Mat4 {
        let (o, z, i) = (c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0));
        Matrix4::new(o, z, z, z, z, z, i, z, z, i, z, z, z, z, z, o)
    }

    /// A deterministic pseudo-random SU(4), built as a product of local
    /// factors around a canonical core, so the test needs no RNG dependency.
    fn pseudo_random_su4(seed: u64) -> Mat4 {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * std::f64::consts::TAU
        };
        let k1 = kron2(
            &su2_from_euler_zyz(next(), next(), next()),
            &su2_from_euler_zyz(next(), next(), next()),
        );
        let k2 = kron2(
            &su2_from_euler_zyz(next(), next(), next()),
            &su2_from_euler_zyz(next(), next(), next()),
        );
        k1 * canonical_core(next() * 0.25, next() * 0.25, next() * 0.25) * k2
    }

    #[test]
    fn special_points_decompose_exactly() {
        for (u, name) in [
            (Mat4::identity(), "identity"),
            (cnot(), "CNOT"),
            (swap(), "SWAP"),
            (iswap(), "iSWAP"),
        ] {
            check(&u, name);
        }
    }

    #[test]
    fn random_su4_precision() {
        let mut worst = 0.0f64;
        for seed in 0..500 {
            worst = worst.max(check(&pseudo_random_su4(seed), &format!("seed {}", seed)));
        }
        assert!(worst < 1e-13, "worst random-SU(4) infidelity {:.3e}", worst);
    }

    /// The regression this round's two-stage simultaneous diagonalization
    /// exists for: CNOT plus a vanishing perturbation used to be rejected
    /// outright at a rate that grew as the perturbation shrank.
    #[test]
    fn near_degenerate_cnot_resolves() {
        let base = cnot();
        let mut failures = 0;
        let total = 200;
        for k in 0..total {
            let eps = 1e-7;
            let pert = pseudo_random_su4(10_000 + k as u64);
            let mixed = base * (Mat4::identity() * c(1.0 - eps, 0.0) + pert * c(eps, 0.0));
            // Re-unitarize via the polar factor so the input stays exactly
            // unitary (otherwise normalize_su4 would reject it on principle).
            let dm = DMatrix::from_fn(4, 4, |r, cc| mixed[(r, cc)]);
            let svd = dm.svd(true, true);
            let recon = svd.u.unwrap() * svd.v_t.unwrap();
            let mut uu = Mat4::zeros();
            for r in 0..4 {
                for cc in 0..4 {
                    uu[(r, cc)] = recon[(r, cc)];
                }
            }
            if decompose_one(&uu).is_err() {
                failures += 1;
            }
        }
        assert_eq!(
            failures, 0,
            "{}/{} near-degenerate CNOT perturbations were rejected",
            failures, total
        );
    }

    #[test]
    fn non_unitary_input_is_reported_as_such() {
        let mut bad = Mat4::identity();
        bad[(0, 0)] = c(2.0, 0.0);
        assert_eq!(decompose_one(&bad), Err(CartanError::NotUnitary));
    }

    #[test]
    fn euler_roundtrip_is_exact() {
        for seed in 0..200u64 {
            let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
            let mut next = move || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f64 / (1u64 << 31) as f64) * std::f64::consts::TAU
            };
            let m = su2_from_euler_zyz(next(), next() * 0.5, next());
            let (phi, theta, lam) = su2_to_euler_zyz(&m);
            let back = su2_from_euler_zyz(phi, theta, lam);
            assert!(
                (back - m).norm() < 1e-12,
                "seed {}: ZYZ roundtrip error {:.3e}",
                seed,
                (back - m).norm()
            );
        }
    }
}



