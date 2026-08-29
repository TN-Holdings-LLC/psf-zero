use nalgebra::{ComplexField, Matrix2, Matrix4};
use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;

pub type Mat4 = Matrix4<Complex64>;
pub type Mat2 = Matrix2<Complex64>;
pub type RMat4 = Matrix4<f64>;

const SU2_SINGULAR_TOL: f64 = 1e-18;

#[derive(Debug, Clone, PartialEq)]
pub enum CartanError {
    NotUnitary,
    DetNotOne,
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
    let det = u.determinant();
    if !det.is_finite() || det.norm() < 1e-12 {
        return Err(CartanError::NumericInstability);
    }
    let correction = Complex64::from_polar(1.0, -phase);
    Ok(u * correction)
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

    let phase = u.determinant().argument() / 4.0;
    let u_norm = normalize_su4(&u, phase).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Normalization failed")
    })?;

    let q = &*MAGIC_Q;
    let u_m = q.adjoint() * u_norm * q;
    let m = u_m.transpose() * u_m;

    let m_real = m.map(|x| x.re);
    let sym_eigen = nalgebra::linalg::SymmetricEigen::new(m_real);
    let o2_real = sym_eigen.eigenvectors.transpose();
    let o2_complex = o2_real.map(|x| Complex64::new(x, 0.0));
    
    let diag_m = o2_complex * m * o2_complex.transpose();

    let mut angles: Vec<f64> = (0..4)
        .map(|i| diag_m[(i, i)].argument().abs() / 2.0)
        .map(|a| if a.is_finite() { a } else { 0.0 })
        .collect();
    
    while angles.len() < 4 { angles.push(0.0); }
    angles.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    
    if angles[0] + angles[1] > PI / 2.0 {
        let c1 = PI / 2.0 - angles[1];
        let c2 = PI / 2.0 - angles[0];
        angles[0] = c1;
        angles[1] = c2;
        angles.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut u_m_real = nalgebra::Matrix4::<f64>::zeros();
    for i in 0..4 {
        for j in 0..4 {
            u_m_real[(i, j)] = if u_m[(i, j)].re.is_finite() { u_m[(i, j)].re } else { 0.0 };
        }
    }

    let svd = u_m_real.svd(true, true);
    
    let mut o1 = svd.u.unwrap_or_else(|| nalgebra::Matrix4::identity());
    let mut o2 = svd.v_t.unwrap_or_else(|| nalgebra::Matrix4::identity());
    
    
    if o1.determinant() < 0.0 {
        for r in 0..4 { o1[(r, 3)] = -o1[(r, 3)]; }
    }
    if o2.determinant() < 0.0 {
        for c in 0..4 { o2[(3, c)] = -o2[(3, c)]; } // v_t なので行(3)を反転
    }
    let o2_t = o2.transpose();

    
    let (k1l, k1r) = so4_to_su2_pair(&o1).unwrap_or((Mat2::identity(), Mat2::identity()));
    let (k2l, k2r) = so4_to_su2_pair(&o2_t).unwrap_or((Mat2::identity(), Mat2::identity()));

    let a1 = su2_to_euler_zyz(&k1r);
    let a2 = su2_to_euler_zyz(&k1l);
    let a3 = su2_to_euler_zyz(&k2r);
    let a4 = su2_to_euler_zyz(&k2l);

    let k1 = vec![vec![a1.0, a1.1, a1.2], vec![a2.0, a2.1, a2.2]];
    let k2 = vec![vec![a3.0, a3.1, a3.2], vec![a4.0, a4.1, a4.2]];

    Ok(((angles[0], angles[1], angles[2]), k1, k2, phase))
}

#[pymodule]
fn psf_zero_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(geometric_decompose, m)?)?;
    Ok(())
}
