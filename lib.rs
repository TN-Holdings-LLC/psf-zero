use nalgebra::{Matrix4, ComplexField};
use num_complex::Complex64;
use std::f64::consts::PI;

pub type Mat4 = Matrix4<Complex64>;

#[derive(Debug, Clone, PartialEq)]
pub enum CartanError {
    NotUnitary,
    DetNotOne,
    EigenDecompositionFailed,
    NumericInstability,
}

// =========================================================
// 1. Magic Basis (Immutable Template)
// =========================================================
lazy_static::lazy_static! {
    static ref MAGIC_Q: Mat4 = {
        let s = (2.0_f64).sqrt();
        let i = Complex64::new(0.0, 1.0);
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);

        Matrix4::new(
            o/s,  z,   z,   i/s,
            z,    i/s, o/s, z,
            z,    i/s, -o/s,z,
            o/s,  z,   z,   -i/s,
        )
    };
}

// =========================================================
// 2. SU(4) Normalization (Global Phase Stripping)
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
// 3. Cartan Coordinate Extraction (Absolute, O(1))
// =========================================================
pub fn cartan_coordinates(u: &Mat4) -> Result<(f64, f64, f64), CartanError> {
    let u = normalize_su4(u)?;

    let q = &*MAGIC_Q;
    let u_m = q.adjoint() * u * q;
    let m = u_m.transpose() * u_m;   // M = U_M^T * U_M

    let eigen = m.complex_eigenvalues();
    if eigen.len() != 4 {
        return Err(CartanError::EigenDecompositionFailed);
    }

    // Magic Basis固有値の引数からCartan座標を直接抽出
    let c1 = (eigen[0] * eigen[1]).argument().abs() / 2.0;
    let c2 = (eigen[0] * eigen[2]).argument().abs() / 2.0;
    let c3 = (eigen[0] * eigen[3]).argument().abs() / 2.0;

    let mut v = [c1, c2, c3];
    v.sort_by(|a, b| b.partial_cmp(a).unwrap()); // c1 ≥ c2 ≥ c3

    // Weyl Chamber Reflection
    if v[0] + v[1] > PI / 2.0 {
        let new_c1 = PI / 2.0 - v[1];
        let new_c2 = PI / 2.0 - v[0];
        v[0] = new_c1;
        v[1] = new_c2;
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
    }

    Ok((v[0], v[1], v[2]))
}

// =========================================================
// Unit Tests
// =========================================================
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_returns_zero() {
        let u = Mat4::identity();
        let (c1, c2, c3) = cartan_coordinates(&u).unwrap();
        assert_relative_eq!(c1, 0.0, epsilon = 1e-12);
        assert_relative_eq!(c2, 0.0, epsilon = 1e-12);
        assert_relative_eq!(c3, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn pure_rzz() {
        let theta = 0.7;
        let phase = Complex64::new(0.0, -theta / 2.0).exp();
        let u = Mat4::from_diagonal(&[phase, phase.conj(), phase.conj(), phase]);

        let (c1, c2, c3) = cartan_coordinates(&u).unwrap();
        assert_relative_eq!(c1, theta.abs() / 2.0, epsilon = 1e-10);
        assert_relative_eq!(c2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(c3, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn non_unitary_rejected() {
        let mut u = Mat4::identity();
        u[(0, 0)] += Complex64::new(0.01, 0.0);
        assert!(matches!(cartan_coordinates(&u), Err(CartanError::NotUnitary)));
    }
}
