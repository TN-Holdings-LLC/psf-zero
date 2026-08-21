use nalgebra::{ComplexField, Matrix2, Matrix4};
use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;

pub type Mat4 = Matrix4<Complex64>;
pub type Mat2 = Matrix2<Complex64>;
pub type RMat4 = Matrix4<f64>;

const DEGENERACY_TOL: f64 = 1e-6;
const SU2_SINGULAR_TOL: f64 = 1e-18;

#[derive(Debug, Clone, PartialEq)]
pub enum CartanError {
    NotUnitary,
    DetNotOne,
    DegenerateWeylPoint,
    SU2ExtractionSingular,
    NumericInstability,
}

lazy_static::lazy_static! {
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

    let det_check = u_su4.determinant();
    if !det_check.is_finite() || (det_check - Complex64::new(1.0, 0.0)).norm() > 1e-6 {
        return Err(CartanError::DetNotOne);
    }
    Ok(u_su4)
}

fn to_complex(m: &RMat4) -> Mat4 {
    m.map(|x| Complex64::new(x, 0.0))
}

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

    let q = &*MAGIC_Q;
    let candidate = q.adjoint() * kron2(&k_l, &k_r) * q;
    let candidate_real = candidate.map(|c| c.re);
    if (candidate_real - o).norm() > (candidate_real + o).norm() {
        k_r = -k_r;
    }

    Ok((k_l, k_r))
}

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

fn decompose_one(u: &Mat4) -> Result<((f64, f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), f64), CartanError> {
    let phase = u.determinant().argument() / 4.0;
    let u_norm = normalize_su4(u, phase)?;

    let q = &*MAGIC_Q;
    let u_m = q.adjoint() * u_norm * q;
    let u_m_real = u_m.map(|c| c.re);

    let svd = u_m_real
        .try_svd(true, true, 1e-12, 100)
        .ok_or(CartanError::NumericInstability)?;
    let s = svd.singular_values;
    let mut o1 = svd.u.ok_or(CartanError::NumericInstability)?;
    let mut o2 = svd.v_t.ok_or(CartanError::NumericInstability)?;

    let mut sorted_s = [s[0], s[1], s[2], s[3]];
    sorted_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in sorted_s.windows(2) {
        if (w[1] - w[0]).abs() < DEGENERACY_TOL {
            return Err(CartanError::DegenerateWeylPoint);
        }
    }

    let mut order = [0usize, 1, 2, 3];
    order.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());
    let o1_sorted = RMat4::from_fn(|r, c| o1[(r, order[c])]);
    let o2_sorted = RMat4::from_fn(|r, c| o2[(order[r], c)]);
    o1 = o1_sorted;
    o2 = o2_sorted;

    if o1.determinant() < 0.0 {
        for r in 0..4 { o1[(r, 3)] = -o1[(r, 3)]; }
    }
    if o2.determinant() < 0.0 {
        for c in 0..4 { o2[(3, c)] = -o2[(3, c)]; }
    }

    let o1c = to_complex(&o1);
    let o2c = to_complex(&o2);
    let d = o1c.transpose() * u_m * o2c.transpose();

    let mut offdiag_norm_sq = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            if r != c { offdiag_norm_sq += d[(r, c)].norm_sqr(); }
        }
    }
    if offdiag_norm_sq.sqrt() > 1e-6 {
        return Err(CartanError::NumericInstability);
    }

    let mut a0 = d[(0, 0)].argument() / 2.0;
    let mut a1 = d[(1, 1)].argument() / 2.0;
    let mut a2 = d[(2, 2)].argument() / 2.0;
    let mut a3 = d[(3, 3)].argument() / 2.0;

    // =========================================================
    // WEYL CHAMBER FOLD (Restoring strict positive alcove bounds)
    // =========================================================
    let mut angles = vec![a0, a1, a2, a3];
    angles.sort_by(|x, y| y.partial_cmp(x).unwrap()); // Descending

    if angles[0] + angles[1] > PI / 2.0 {
        let c1 = PI / 2.0 - angles[1];
        let c2 = PI / 2.0 - angles[0];
        let c3 = angles[2];
        let c4 = -angles[3]; // Keep consistency with sign invariants
        angles = vec![c1, c2, c3, c4];
        angles.sort_by(|x, y| y.partial_cmp(x).unwrap());
    }

    let (k1l, k1r) = so4_to_su2_pair(&o1)?;
    let (k2l, k2r) = so4_to_su2_pair(&o2)?;

    Ok((
        (angles[0], angles[1], angles[2], angles[3]),
        su2_to_euler_zyz(&k1l),
        su2_to_euler_zyz(&k1r),
        su2_to_euler_zyz(&k2l),
        su2_to_euler_zyz(&k2r),
        phase,
    ))
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
        let mut has_nan = false;
        for i in 0..4 {
            for j in 0..4 {
                let re = u_batch_r[idx][i][j];
                let im = u_batch_i[idx][i][j];
                if !re.is_finite() || !im.is_finite() { has_nan = true; }
                u[(i, j)] = Complex64::new(re, im);
            }
        }
        if has_nan {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "batch item {}: contains NaN or infinite values.", idx
            )));
        }

        let (angles, e1l, e1r, e2l, e2r, phase) = decompose_one(&u).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "batch item {}: Cartan decomposition failed: {:?}", idx, e
            ))
        })?;

        let k1 = vec![vec![e1l.0, e1l.1, e1l.2], vec![e1r.0, e1r.1, e1r.2]];
        let k2 = vec![vec![e2l.0, e2l.1, e2l.2], vec![e2r.0, e2r.1, e2r.2]];
        results.push((angles, k1, k2, phase));
    }

    Ok(results)
}

#[pymodule]
fn psf_zero_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_decompose, m)?)?;
    Ok(())
}
