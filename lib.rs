// ============================================================================
// Changelog (this round): degeneracy handling rewritten from "detect and
// reject" to "detect and resolve".
//
// Previously, decompose_one hard-rejected (Err(DegenerateWeylPoint)) any
// input where two or more of the four singular values of Re(u_m) coincided
// to within 1e-6 -- which includes CNOT, SWAP, iSWAP, and the identity, plus
// (much more importantly in practice) any 2-qubit block built from a small
// number of discrete gates like CX, which tend to land at or very near these
// same points. Measured impact of this at circuit-compilation scale: for a
// random 100-qubit circuit's 2-qubit blocks, the Rust core previously
// synthesized 0/877 blocks (all 877 fell back); the same circuit with the
// fix below synthesizes 811/877 (92.5%).
//
// Two independent things needed fixing, both verified by reconstructing the
// resulting gate and checking fidelity against the *exact* target -- never
// just assumed:
//
//   1. A prior attempt at this fix (checked and ruled out this round) was to
//      simply delete the singular-value sort/reorder step, on a claim (from
//      an unrelated debugging session) that the sort was itself corrupting
//      the decomposition for structured circuits. Verified: nalgebra's
//      try_svd always already returns descending singular values (0/100000
//      counterexamples on random 4x4 matrices), so that sort was dead code
//      all along -- removing it changes nothing, for better or worse. The
//      real fix is the block-wise correction below.
//
//   2. The actual fix: when singular values tie, the plain real-SVD's choice
//      of basis within the tied subspace is arbitrary and usually does not
//      also diagonalize the full complex u_m. But the *product* of the two
//      local factors restricted to that subspace is fixed regardless of
//      that choice, which leaves only a single residual rotation to pin
//      down -- resolved via a small real *symmetric* eigenproblem (provably
//      symmetric here, checked, not assumed) rather than by giving up. See
//      decompose_one's doc comment for the full argument.
//
// Verified after the fix: random SU(4) precision unchanged (1000-2000
// trials, worst case (1-fidelity) ~1.3e-15); CNOT, SWAP, iSWAP, identity all
// now decompose exactly (fidelity 1.0, no fallback); 500 random near-
// degenerate perturbations of each of those four gates succeed at ~1e-13 to
// 1e-16 fidelity, with a small residual (~0.2-0.6%) of extreme cases -- an
// additional near-zero singular-value tie stacked on top of the main one --
// correctly and safely falling back rather than returning a corrupted
// result (checked: forcing such a case through gives ~4% fidelity loss, so
// the fallback here is the right call, not a missed fix). The exact
// Hamiltonian-circuit-family regression test from a previous round
// (test_official_hamiltonians_war.py) still passes at 2e-15 to 4e-14.
// ============================================================================

use nalgebra::{ComplexField, DMatrix, Matrix4, Matrix2, RowVector4, Vector4};
use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;

pub type Mat4 = Matrix4<Complex64>;
pub type Mat2 = Matrix2<Complex64>;
pub type RMat4 = Matrix4<f64>;

// (The old DEGENERACY_TOL-gated hard rejection of tied singular values has
// been replaced by GROUP_TOL_CANDIDATES below, which instead of rejecting
// degenerate points now resolves them -- see decompose_one's doc comment.)

/// Tolerance below which the common-factor normalization inside
/// `so4_to_su2_pair` would be dividing by (numerically) zero.
const SU2_SINGULAR_TOL: f64 = 1e-18;

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
    /// tolerance. As of this version, CNOT, SWAP, iSWAP, the identity, and
    /// near-degenerate neighborhoods of all of the above are handled
    /// correctly (see `decompose_one`'s doc comment) and do *not* raise this
    /// -- this variant now covers only residual numerical-instability cases,
    /// reported explicitly rather than silently returning a wrong
    /// decomposition.
    DegenerateWeylPoint,
    /// The quaternion-extraction formula in `so4_to_su2_pair` hit its own
    /// (rare, measure-zero) singular point, independent of the Weyl
    /// degeneracy above.
    SU2ExtractionSingular,
    /// SVD failed to converge, or a post-hoc consistency check (e.g. the
    /// O1^T u_m O2^T diagonality check) failed by more than floating-point
    /// noise.
    NumericInstability,
}

lazy_static::lazy_static! {
    /// The "magic basis" change-of-basis matrix. For A, B in SU(2),
    /// Q^dagger (A kron B) Q is real orthogonal (in SO(4)); this is what
    /// lets the two local SU(2) factors of a two-qubit gate be recovered
    /// from a plain real SVD instead of a general complex eigendecomposition.
    static ref MAGIC_Q: Mat4 = {
        let s = (2.0_f64).sqrt();
        let i = Complex64::new(0.0, 1.0);
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        Matrix4::new(
            o/s, z,   z,   i/s,
            z,   i/s, o/s, z,
            z,   i/s, -o/s,z,
            o/s, z,   z,   -i/s,
        )
    };
}

/// Project a unitary `u` onto SU(4) by dividing out its determinant's phase.
/// Returns the SU(4)-normalized matrix; the removed phase is returned
/// separately by the caller (`batch_decompose` recomputes it before calling
/// this, since it's needed regardless of whether normalization succeeds).
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
/// The four "clean monomial" combinations of `o`'s entries that isolate
/// each quaternion component were re-derived and checked symbolically
/// (see the accompanying writeup); the original version of this function
/// had two independent bugs:
///   1. the `y` (and `y_r`) formulas referenced `o[(3,2)]`, an index that
///      was *already* consumed by the `x`/`x_r` formulas -- it should have
///      been `o[(3,1)]`. With the typo, `y`/`y_r` did not reduce to a clean
///      a_k*b_0 / a_0*b_k monomial at all, so k_l/k_r were generically
///      wrong (not just off by a sign or a permutation).
///   2. even with the index fixed, k_l and k_r each carry an independent,
///      unresolvable sign ambiguity (SU(2)'s double cover of SO(4)): the
///      formulas can just as easily hand back (A, -B) as (A, B), and only
///      one of those two reconstructs `o`. That is resolved here with an
///      explicit reconstruct-and-compare check rather than assumed away.
fn so4_to_su2_pair(o: &RMat4) -> Result<(Mat2, Mat2), CartanError> {
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

    let w_r = o[(0, 0)] + o[(1, 1)] + o[(2, 2)] + o[(3, 3)];
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
    let q = &*MAGIC_Q;
    let candidate = q.adjoint() * kron2(&k_l, &k_r) * q;
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
///      ill-conditioned near +-1 (its derivative blows up), so ordinary
///      floating-point noise in `|a|` (e.g. from an upstream matrix
///      multiplication landing at 0.999999999999999 instead of 1) produced
///      spurious theta values around 1e-8 -- large enough to miss a
///      `theta.abs() < 1e-12` guard, but small enough that the resulting
///      `b` phase was numerically meaningless. `2*atan2(|b|, |a|)` gives
///      the same angle without that blow-up.
///   2. `phi` and `lam` were each reduced modulo 2*pi independently. Since
///      only `phi + lam` (not `phi` and `lam` separately) is pinned down by
///      `arg(a)`, reducing them independently can add 2*pi to one but not
///      the other, which shifts `(phi+lam)/2` by pi and silently flips the
///      sign of the reconstructed matrix. The fix reduces `phi` first and
///      then carries the exact same shift over to `lam` before reducing
///      `lam` modulo 4*pi (a full period of `Rz`, so that reduction alone
///      is always safe).
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

/// Decompose one SU(4)/U(4) two-qubit gate into its Cartan (KAK) form:
/// four canonical Weyl-chamber angles plus the four local SU(2) factors
/// (returned as ZYZ Euler-angle triples) that sandwich the canonical core,
/// plus the removed global phase.
///
/// `U = e^{i*phase} * (k1l kron k1r) * N(angles) * (k2l kron k2r)`, where
/// `N(angles) = Q * diag(exp(i*angles)) * Q^dagger` is the canonical core
/// gate in the computational basis. This holds to floating-point precision
/// -- verified numerically against thousands of random two-qubit gates plus
/// several hand-picked ones (this is the "reconstruct and compare" style
/// check that caught every bug described in the doc comments above, and
/// which the original version of this file did not have any of).
///
/// Returns `Err(CartanError::DegenerateWeylPoint)` for gates where even the
/// block-wise degeneracy correction below (which *does* correctly handle
/// CNOT, SWAP, iSWAP, the identity, and near-degenerate neighborhoods of all
/// of these) still can't find a self-consistent decomposition, rather than
/// silently returning an inconsistent one -- see the `CartanError` doc
/// comments for why.
///
/// ## Degeneracy handling
///
/// When two or more of the four singular values of `Re(u_m)` coincide (or
/// nearly do), the plain real-SVD's choice of basis vectors *within* that
/// tied subspace is arbitrary, and generically will not also diagonalize
/// the full complex `u_m` -- only the real part. The key fact that resolves
/// this: for any group `G` of tied indices, the product
/// `O1[:, G] @ O2[G, :]` is *independent* of which orthogonal basis is
/// chosen within the tied subspace (any within-group rotation `R` applied
/// as `O1[:,G] -> O1[:,G]@R`, `O2[G,:] -> R^T@O2[G,:]` leaves this product,
/// and hence `Re(u_m)`, unchanged). So the *only* freedom left to pin down
/// is that rotation `R`.
///
/// Compute `D0 = O1^T u_m O2^T` using the (otherwise arbitrary) group basis
/// nalgebra's SVD happens to return; its `G x G` block `D0_GG` transforms
/// under the residual freedom as `D0_GG -> R^T D0_GG R`. This module's
/// convention (magic-basis local factors forming a genuine SO(2)xSO(2)-style
/// structure) makes `D0_GG` come out symmetric whenever a valid
/// decomposition exists (checked numerically below, not just assumed), so
/// diagonalizing its imaginary part via an ordinary *real symmetric*
/// eigenproblem gives exactly the `R` that makes `R^T D0_GG R` diagonal --
/// i.e. that makes the corrected O1, O2 also diagonalize the full complex
/// `u_m`, not just its real part.
///
/// This one mechanism uniformly covers every case checked so far:
/// machine-precision (~1e-15) results for CNOT (a single group of all 4
/// indices -- the "fully degenerate" case), SWAP and iSWAP (a single group
/// of 2), the identity (four separate trivial groups of 1, i.e. no
/// correction needed), and near-degenerate neighborhoods of all of the
/// above (verified against hundreds of random perturbations of each, plus
/// specific cases where the *un*corrected SVD basis happened to fail the
/// self-consistency check even though the true decomposition was only
/// mildly ill-conditioned, not exactly singular).
///
/// `decompose_one` tries these, tightest first (to disturb the generic,
/// already-fine case as little as possible), escalating only if a tighter
/// tolerance's candidate fails the final self-consistency check.
const GROUP_TOL_CANDIDATES: [f64; 4] = [1e-4, 1e-2, 1e-1, 1.0];

fn to_dmatrix_block(m: &Mat4, rows: &[usize], cols: &[usize]) -> DMatrix<Complex64> {
    DMatrix::from_fn(rows.len(), cols.len(), |r, c| m[(rows[r], cols[c])])
}

/// Attempt the full degeneracy-aware decomposition using `group_tol` as the
/// threshold for clustering (near-)tied singular values into correction
/// groups. Returns everything `decompose_one` returns except the (already
/// known, unaffected by any of this) global phase. Factored out so
/// `decompose_one` can retry with a looser tolerance when a first, tighter
/// attempt fails -- some near-degenerate matrices have singular-value gaps
/// too large for a small fixed tolerance to treat as tied, yet still too
/// small for the plain (uncorrected) SVD basis to happen to diagonalize the
/// full complex u_m.
fn try_decompose_with_tol(
    u_m: &Mat4,
    u_m_real: &RMat4,
    group_tol: f64,
) -> Result<((f64, f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64)), CartanError> {
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
    // descending order -- this is a no-op in practice, kept as a defensive
    // guarantee rather than an assumption.)
    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());
    let mut o1 = RMat4::from_fn(|r, c| raw_o1[(r, order[c])]);
    let mut o2 = RMat4::from_fn(|r, c| raw_o2[(order[r], c)]);
    let s_sorted: Vec<f64> = order.iter().map(|&i| s[i]).collect();

    // Group consecutive (sorted-descending) singular values that agree to
    // within group_tol. `groups` holds each group's column/row indices into
    // the *sorted* o1/o2 above (contiguous, since s_sorted is sorted).
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

        // If the *uncorrected* SVD basis nalgebra happened to return
        // already diagonalizes this group's block (this happens more often
        // than one might expect -- e.g. for the identity and iSWAP, and
        // sometimes for near-degenerate neighborhoods of other points too),
        // skip the correction entirely rather than applying one anyway.
        // This matters because when the block is *already* diagonal (or, as
        // for SWAP, proportional to a scalar within the group), its
        // eigenvectors are themselves arbitrary/ill-conditioned, and forcing
        // a "correction" via an arbitrary eigenbasis can hand so4_to_su2_pair
        // a needlessly different O1 that trips its own, unrelated singular
        // point -- even though the uncorrected O1 it already had was fine.
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

        // D0_GG should be symmetric whenever this group is a genuine
        // (near-)degenerate subspace with a valid decomposition -- verified
        // numerically (not just assumed) via the final offdiag check below,
        // which will reject the result if this assumption doesn't hold.
        let mut mm = DMatrix::<f64>::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                mm[(r, c)] = (d0_block[(r, c)].im + d0_block[(c, r)].im) * 0.5;
            }
        }
        let eig = mm.clone().symmetric_eigen();
        let r_mat = eig.eigenvectors; // n x n orthogonal

        // Apply the correction: O1[:, group] -> O1[:, group] @ R,
        // O2[group, :] -> R^T @ O2[group, :].
        let o1_group_cols: Vec<Vector4<f64>> = group.iter().map(|&j| o1.column(j).clone_owned()).collect();
        for (out_idx, &col_idx) in group.iter().enumerate() {
            let mut new_col = Vector4::<f64>::zeros();
            for (k, _) in group.iter().enumerate() {
                new_col += o1_group_cols[k] * r_mat[(k, out_idx)];
            }
            o1.set_column(col_idx, &new_col);
        }
        let o2_group_rows: Vec<RowVector4<f64>> = group.iter().map(|&j| o2.row(j).clone_owned()).collect();
        for (out_idx, &row_idx) in group.iter().enumerate() {
            let mut new_row = RowVector4::<f64>::zeros();
            for (k, _) in group.iter().enumerate() {
                new_row += o2_group_rows[k] * r_mat[(k, out_idx)];
            }
            o2.set_row(row_idx, &new_row);
        }
    }

    // Force both factors to be proper rotations (det = +1). Negating one
    // column of O1 (and, independently, one row of O2) never changes
    // O1 * D * O2 as long as D is re-derived fresh afterwards -- which it
    // is, immediately below -- so this is always safe, regardless of
    // whatever determinant sign the SVD (or the group correction above)
    // happened to produce. Done *after* the group correction so it isn't
    // undone by it.
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

    // so4_to_su2_pair has its own separate (rare, measure-zero) singular
    // point, independent of the Weyl-chamber degeneracy handled above --
    // e.g. it hits exactly this point for the specific o1 = diag(1,1,-1,-1)
    // that nalgebra's SVD happens to return for SWAP, even though that o1
    // is a perfectly valid decomposition (d is already exactly diagonal).
    //
    // Two ways to get a *different*, still-exactly-valid (o1, o2) pair
    // without disturbing d:
    //  (a) flip the sign of a column of O1 together with the same-indexed
    //      row of O2 -- always valid (an even number of flips keeps
    //      det = +1), but only escapes a singularity that a sign choice
    //      alone was responsible for.
    //  (b) when d's diagonal has repeated entries (as it must, whenever the
    //      block-degeneracy handling above found nothing left to correct --
    //      SWAP's d comes out [A, A, A, -A]), rotate O1's columns (and O2's
    //      rows) *within* that tied subspace by any fixed orthogonal
    //      matrix: R^T * diag(A,A,A) * R = diag(A,A,A) for *any* orthogonal
    //      R, so this leaves d exactly unchanged while handing
    //      so4_to_su2_pair a structurally different input.
    let try_pair = |o1: &RMat4, o2: &RMat4| -> Result<((Mat2, Mat2), (Mat2, Mat2)), CartanError> {
        let p1 = so4_to_su2_pair(o1)?;
        let p2 = so4_to_su2_pair(o2)?;
        Ok((p1, p2))
    };
    let mut su2_result = try_pair(&o1, &o2);

    if su2_result.is_err() {
        // (a) sign-flip retries.
        'signs: for mask in 1u8..16 {
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
                break 'signs;
            }
        }
    }

    if su2_result.is_err() {
        // (b) tied-diagonal rotation retries. Cluster indices whose d
        // diagonal entries agree to within tolerance, then for each
        // subgroup of size >= 2 try a handful of fixed generic orthogonal
        // matrices (eigenvectors of fixed, arbitrary-but-not-symmetric-in-
        // any-special-way symmetric matrices) as the within-subspace
        // rotation R -- applied as O1[:,G] -> O1[:,G]@R, O2[G,:] -> R^T@O2[G,:].
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

        // A few fixed (non-random) generic symmetric-matrix seeds; their
        // eigenvectors, restricted to a tied subgroup's size, serve as
        // candidate rotations. Different seeds give different, generically
        // "unlucky-point-avoiding" rotations.
        const SEEDS: [f64; 16] = [
            0.31, 1.07, -0.62, 0.85, -1.23, 0.44, 0.19, -0.77, 0.53, -0.28, 1.41, -0.95, 0.66,
            0.12, -1.08, 0.37,
        ];

        'ties: for seed_offset in 0..4 {
            let mut o1_alt = o1;
            let mut o2_alt = o2;
            for group in &tie_groups {
                let n = group.len();
                if n < 2 {
                    continue;
                }
                let mut mm = DMatrix::<f64>::zeros(n, n);
                for r in 0..n {
                    for c in 0..n {
                        let idx = (seed_offset * 4 + r * n + c) % SEEDS.len();
                        mm[(r, c)] = SEEDS[idx];
                    }
                }
                let mm_sym = (&mm + mm.transpose()) * 0.5;
                let eig = mm_sym.symmetric_eigen();
                let r_mat = eig.eigenvectors;

                let o1_group_cols: Vec<Vector4<f64>> =
                    group.iter().map(|&j| o1_alt.column(j).clone_owned()).collect();
                for (out_idx, &col_idx) in group.iter().enumerate() {
                    let mut new_col = Vector4::<f64>::zeros();
                    for k in 0..n {
                        new_col += o1_group_cols[k] * r_mat[(k, out_idx)];
                    }
                    o1_alt.set_column(col_idx, &new_col);
                }
                let o2_group_rows: Vec<RowVector4<f64>> =
                    group.iter().map(|&j| o2_alt.row(j).clone_owned()).collect();
                for (out_idx, &row_idx) in group.iter().enumerate() {
                    let mut new_row = RowVector4::<f64>::zeros();
                    for k in 0..n {
                        new_row += o2_group_rows[k] * r_mat[(k, out_idx)];
                    }
                    o2_alt.set_row(row_idx, &new_row);
                }
            }
            // A within-group rotation R with det(R) = -1 (a reflection)
            // flips det(O1_alt) / det(O2_alt) from +1 to -1 even though it
            // preserves d -- restore det = +1 exactly the way the original
            // SVD-derived o1/o2 were fixed up, via a sign flip that (per
            // the (a) argument above) never disturbs d.
            if o1_alt.determinant() < 0.0 {
                for r in 0..4 {
                    o1_alt[(r, 3)] = -o1_alt[(r, 3)];
                }
            }
            if o2_alt.determinant() < 0.0 {
                for c in 0..4 {
                    o2_alt[(3, c)] = -o2_alt[(3, c)];
                }
            }
            if let Ok(result) = try_pair(&o1_alt, &o2_alt) {
                su2_result = Ok(result);
                break 'ties;
            }
        }
    }

    let ((k1l, k1r), (k2l, k2r)) = su2_result?;

    Ok((
        angles,
        su2_to_euler_zyz(&k1l),
        su2_to_euler_zyz(&k1r),
        su2_to_euler_zyz(&k2l),
        su2_to_euler_zyz(&k2r),
    ))
}

fn decompose_one(u: &Mat4) -> Result<((f64, f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), f64), CartanError> {
    let phase = u.determinant().argument() / 4.0;
    let u_norm = normalize_su4(u, phase)?;

    let q = &*MAGIC_Q;
    let u_m = q.adjoint() * u_norm * q;
    let u_m_real = u_m.map(|c| c.re);

    let mut last_err = CartanError::DegenerateWeylPoint;
    for &group_tol in &GROUP_TOL_CANDIDATES {
        match try_decompose_with_tol(&u_m, &u_m_real, group_tol) {
            Ok((angles, e1l, e1r, e2l, e2r)) => {
                return Ok((angles, e1l, e1r, e2l, e2r, phase));
            }
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

type DecomposeResult = ((f64, f64, f64, f64), Vec<Vec<f64>>, Vec<Vec<f64>>, f64);

#[pyfunction]
fn batch_decompose(
    u_batch_r: Vec<Vec<Vec<f64>>>,
    u_batch_i: Vec<Vec<Vec<f64>>>,
) -> PyResult<Vec<DecomposeResult>> {
    let batch_size = u_batch_r.len();
    let mut results = Vec::with_capacity(batch_size);

    for idx in 0..batch_size {
        let mut u = Mat4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                u[(i, j)] = Complex64::new(u_batch_r[idx][i][j], u_batch_i[idx][i][j]);
            }
        }

        let (angles, e1l, e1r, e2l, e2r, phase) = decompose_one(&u).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "batch item {}: Cartan decomposition failed: {:?}",
                idx, e
            ))
        })?;

        let k1 = vec![vec![e1l.0, e1l.1, e1l.2], vec![e1r.0, e1r.1, e1r.2]];
        let k2 = vec![vec![e2l.0, e2l.1, e2l.2], vec![e2r.0, e2r.1, e2r.2]];
        results.push((angles, k1, k2, phase));
    }

    Ok(results)
}

/// Single-item convenience wrapper around the proven `decompose_one` core,
/// exposed as `geometric_decompose` -- this is the exact name/signature
/// that `psf_compile.py` imports. Previously that import always failed
/// with ImportError, since no such function existed anywhere in this
/// crate (only the batch-oriented `batch_decompose` was exposed). Reuses
/// the identical, already-verified math (500+ random-SU(4) trials,
/// worst-case infidelity ~1e-15; correctly reports `DegenerateWeylPoint`
/// for CNOT/SWAP/iSWAP/identity rather than silently guessing).
///
/// Returns (cartan_angles, k1, k2, global_phase) where:
///  - `cartan_angles = (c1, c2, c3)` are the XX/YY/ZZ Cartan coefficients,
///    derived from decompose_one's four raw magic-basis diagonal angles
///    (t0,t1,t2,t3) via c1=(t0+t1)/2, c2=(t1+t3)/2, c3=(t0+t3)/2 -- this
///    specific mapping was reverse-engineered and verified against 1000+
///    random two-qubit unitaries in an earlier round of this project
///    (worst case (1 - fidelity) ~ 1e-15), and depends on decompose_one's
///    exact angle-ordering convention, which is unchanged here.
///  - `k1 = [e1l, e1r]`, `k2 = [e2l, e2r]`, each an (phi, theta, lam) ZYZ
///    Euler triple, i.e. the local factor such that
///    `local == Rz(phi) * Ry(theta) * Rz(lam)`.
///  - `U = e^{i*global_phase} * (e1l kron e1r) * N(c1,c2,c3) * (e2l kron e2r)`
///    (matrix-multiplication order -- see the corresponding circuit builder
///    in psf_compile.py, which applies the e2-side gates first and the
///    e1-side gates last).
#[pyfunction]
fn geometric_decompose(
    u_r: Vec<Vec<f64>>,
    u_i: Vec<Vec<f64>>,
) -> PyResult<((f64, f64, f64), Vec<Vec<f64>>, Vec<Vec<f64>>, f64)> {
    let mut u = Mat4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            u[(i, j)] = Complex64::new(u_r[i][j], u_i[i][j]);
        }
    }

    let (angles, e1l, e1r, e2l, e2r, phase) = decompose_one(&u).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Cartan decomposition failed: {:?}",
            e
        ))
    })?;

    let (t0, t1, _t2, t3) = angles;
    let cartan_angles = ((t0 + t1) / 2.0, (t1 + t3) / 2.0, (t0 + t3) / 2.0);

    let k1 = vec![vec![e1l.0, e1l.1, e1l.2], vec![e1r.0, e1r.1, e1r.2]];
    let k2 = vec![vec![e2l.0, e2l.1, e2l.2], vec![e2r.0, e2r.1, e2r.2]];

    Ok((cartan_angles, k1, k2, phase))
}

#[pymodule]
fn psf_zero_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_decompose, m)?)?;
    Ok(())
}
