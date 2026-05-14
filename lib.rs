use nalgebra::{Matrix4, Matrix2, ComplexField};
use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;

pub type Mat4 = Matrix4<Complex64>;
pub type Mat2 = Matrix2<Complex64>;

#[derive(Debug, Clone, PartialEq)]
pub enum CartanError {
    NotUnitary,
    DetNotOne,
    EigenDecompositionFailed,
    NumericInstability,
}

// =========================================================
// 1. Magic Basis (Absolute Cosmic Coordinates Template)
// =========================================================
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

// =========================================================
// 2. SU(4) Normalization (Noise Removal and Normalization)
// =========================================================
fn normalize_su4(u: &Mat4) -> Result<Mat4, CartanError> {
    let norm = (u.adjoint() * u - Mat4::identity()).norm();
    if norm > 1e-10 {
        return Err(CartanError::NotUnitary);
    }

    let det = u.determinant();
    if !det.is_finite() || det.norm() < 1e-12 {
        return Err(CartanError::NumericInstability);
    }

    let phase = det.argument() / 4.0;
    let correction = Complex64::from_polar(1.0, -phase);
    let u_su4 = u * correction;

    if (u_su4.determinant() - Complex64::new(1.0, 0.0)).norm() > 1e-10 {
        return Err(CartanError::DetNotOne);
    }
    Ok(u_su4)
}

// =========================================================
// 3. SU(2) → ZYZ Euler Angles (With Strict Fail-Safes)
// =========================================================
fn su2_to_euler_zyz(m: &Mat2) -> (f64, f64, f64) {
    let a = m[(0, 0)];
    let b = m[(0, 1)];

    // Prevent NaN caused by floating-point inaccuracies
    let theta = 2.0 * (a.norm().min(1.0)).acos();

    if theta.abs() < 1e-12 {
        let mut lam = 2.0 * a.argument();
        if lam < 0.0 { lam += 2.0 * PI; }
        return (0.0, 0.0, lam % (2.0 * PI));
    }

    let mut phi = b.argument() + a.argument();
    let mut lam = b.argument() - a.argument();
    
    // Correct negative angles into the positive range
    if phi < 0.0 { phi += 2.0 * PI; }
    if lam < 0.0 { lam += 2.0 * PI; }

    (phi % (2.0 * PI), theta, lam % (2.0 * PI))
}

// =========================================================
// 4. SO(4) → SU(2)×SU(2) Pair 
// 🚨 FIX: Strict Typing - Now accepts only Real Matrices (Matrix4<f64>)
// =========================================================
fn so4_to_su2_pair(o: &nalgebra::Matrix4<f64>) -> (Mat2, Mat2) {
    let sign = if o.determinant() < 0.0 { -1.0 } else { 1.0 };

    // Extract Left (K_L)
    let w = o[(0,0)] + o[(1,1)] + sign*o[(2,2)] + sign*o[(3,3)];
    let x = o[(1,0)] - o[(0,1)] - sign*o[(3,2)] + sign*o[(2,3)];
    let y = o[(2,0)] + o[(3,2)] - sign*o[(0,2)] - sign*o[(1,3)];
    let z = o[(3,0)] - o[(2,1)] + sign*o[(1,2)] - sign*o[(0,3)];

    let mut k_l = Mat2::new(
        Complex64::new(w, z), Complex64::new(y, x),
        Complex64::new(-y, x), Complex64::new(w, -z),
    );
    k_l /= k_l.determinant().norm().sqrt(); // Strict normalization to SU(2)

    // Extract Right (K_R)
    let w_r = o[(0,0)] + o[(1,1)] - sign*o[(2,2)] - sign*o[(3,3)];
    let x_r = o[(1,0)] + o[(0,1)] + sign*o[(3,2)] + sign*o[(2,3)];
    let y_r = o[(2,0)] - o[(3,2)] + sign*o[(0,2)] - sign*o[(1,3)];
    let z_r = o[(3,0)] + o[(2,1)] + sign*o[(1,2)] + sign*o[(0,3)];

    let mut k_r = Mat2::new(
        Complex64::new(w_r, z_r), Complex64::new(y_r, x_r),
        Complex64::new(-y_r, x_r), Complex64::new(w_r, -z_r),
    );
    k_r /= k_r.determinant().norm().sqrt(); // Strict normalization to SU(2)

    (k_l, k_r)
}

// =========================================================
// 5. Main Geometric Decomposition (The Core Algorithm)
// =========================================================
#[pyfunction]
fn geometric_decompose(
    u_r: Vec<Vec<f64>>,
    u_i: Vec<Vec<f64>>,
) -> PyResult<((f64, f64, f64), Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    
    // Reconstruct the matrix from Python data
    let mut u = Mat4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            u[(i, j)] = Complex64::new(u_r[i][j], u_i[i][j]);
        }
    }

    let u = normalize_su4(&u).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Normalization failed: input is not a valid unitary matrix.")
    })?;

    let q = &*MAGIC_Q;
    let u_m = q.adjoint() * u * q;
    let m = u_m.transpose() * u_m;

    // 🚨 FIX: Extract Cartan coordinates directly from the Argument (Phase) of complex eigenvalues
    let eigen = m.complex_eigenvalues();
    if eigen.len() != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Eigen decomposition failed"));
    }

    let mut angles: Vec<f64> = eigen.iter()
        .map(|v| v.argument().abs() / 2.0)
        .collect();
    angles.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Weyl Chamber Reflection (Folding coordinates into the primary chamber)
    if angles[0] + angles[1] > PI / 2.0 {
        let c1 = PI / 2.0 - angles[1];
        let c2 = PI / 2.0 - angles[0];
        angles[0] = c1;
        angles[1] = c2;
        angles.sort_by(|a, b| b.partial_cmp(a).unwrap());
    }

    // 🚨 FIX: Safely separate local operations (K1, K2) by extracting the "Real Part" of U_m
    // and performing a Real SVD to ensure no complex data entanglement is destroyed.
    let mut u_m_real = nalgebra::Matrix4::<f64>::zeros();
    for i in 0..4 {
        for j in 0..4 {
            u_m_real[(i, j)] = u_m[(i, j)].re;
        }
    }

    let svd = u_m_real.svd(true, true);
    let o1 = svd.u.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("SVD U extraction failed"))?;
    let o2_t = svd.v_t.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("SVD V_T extraction failed"))?;
    let o2 = o2_t.transpose();

    // Pass ONLY the Real Orthogonal Matrices (SO(4)) to the parser function
    let (k1l, k1r) = so4_to_su2_pair(&o1);
    let (k2l, k2r) = so4_to_su2_pair(&o2);

    let a1 = su2_to_euler_zyz(&k1r);
    let a2 = su2_to_euler_zyz(&k1l);
    let a3 = su2_to_euler_zyz(&k2r);
    let a4 = su2_to_euler_zyz(&k2l);

    let k1 = vec![vec![a1.0, a1.1, a1.2], vec![a2.0, a2.1, a2.2]];
    let k2 = vec![vec![a3.0, a3.1, a3.2], vec![a4.0, a4.1, a4.2]];

    Ok(((angles[0], angles[1], angles[2]), k1, k2))
}

// =========================================================
// 6. Python Module Exposure
// =========================================================
#[pymodule]
fn psf_zero_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(geometric_decompose, m)?)?;
    Ok(())
}
