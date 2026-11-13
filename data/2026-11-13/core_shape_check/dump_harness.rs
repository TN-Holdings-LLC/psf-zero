
// ---- home comparison harness (2026-11-13), identical in the old and the proposed core ----
#[cfg(test)]
mod dumpcmp {
    use super::*;
    use std::io::Write;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64 / (1u64 << 53) as f64) * std::f64::consts::TAU
        }
    }
    fn rand_su4(seed: u64) -> Mat4 {
        let mut n = lcg(seed);
        let k1 = kron2(&su2_from_euler_zyz(n(), n(), n()), &su2_from_euler_zyz(n(), n(), n()));
        let k2 = kron2(&su2_from_euler_zyz(n(), n(), n()), &su2_from_euler_zyz(n(), n(), n()));
        k1 * canonical_core(n() * 0.25, n() * 0.25, n() * 0.25) * k2
    }
    fn locals_at(theta: [f64; 4], seed: u64) -> Mat4 { tests::from_magic(theta, seed) }

    #[test]
    fn dump_battery() {
        let out = std::env::var("DUMP_OUT").unwrap_or_else(|_| "dump.txt".into());
        let mut f = std::fs::File::create(out).unwrap();
        let o = Complex64::new(1.0, 0.0);
        let z = Complex64::new(0.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let mut cases: Vec<(String, Mat4)> = vec![
            ("identity".into(), Mat4::identity()),
            ("cnot".into(), Matrix4::new(o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z)),
            ("swap".into(), Matrix4::new(o, z, z, z, z, z, o, z, z, o, z, z, z, z, z, o)),
            ("iswap".into(), Matrix4::new(o, z, z, z, z, z, i, z, z, i, z, z, z, z, z, o)),
        ];
        for s in 0..2000u64 { cases.push((format!("rand{}", s), rand_su4(s))); }
        let text = include_str!("test_data/eigen15.txt");
        for (k, line) in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).enumerate() {
            let v: Vec<f64> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
            cases.push((format!("eigen15_{}", k), Mat4::from_fn(|a, b| Complex64::new(v[2 * (4 * a + b)], v[2 * (4 * a + b) + 1]))));
        }
        for &a in &[0.3, 0.9, 1.4] { for &dl in &[0.0, 1e-9, 1e-7, 1e-5] { for s in 0..5u64 {
            let (t0, t1, t2) = (a, std::f64::consts::PI - a + dl, 0.4);
            cases.push((format!("costie_{}_{:e}_{}", a, dl, s), locals_at([t0, t1, t2, -(t0 + t1 + t2)], s)));
        }}}
        for &a in &[0.35, 1.1] { for &dl in &[0.0, 1e-8, 1e-6] { for s in 0..10u64 {
            let (t0, t1, t2) = (a, std::f64::consts::PI - a + dl, -0.5 * std::f64::consts::PI + 0.5 * dl);
            cases.push((format!("compound_{}_{:e}_{}", a, dl, s), locals_at([t0, t1, t2, -(t0 + t1 + t2)], s)));
        }}}
        // exactly repeated phases and Clifford-like points (where +0.0 / -0.0 eigenvalues can appear)
        let h = std::f64::consts::FRAC_PI_2;
        for (k, th) in [[0.0, 0.0, 0.0, 0.0], [h, h, -h, -h], [h, -h, h, -h], [0.0, 0.0, h, -h], [std::f64::consts::PI, 0.0, 0.0, -std::f64::consts::PI]].iter().enumerate() {
            for s in 0..20u64 { cases.push((format!("exact{}_{}", k, s), locals_at(*th, s))); }
        }
        for (name, u) in cases {
            match decompose_one(&u) {
                Ok((ang, (a, b, c, d), ph)) => {
                    let vals = [ang.0, ang.1, ang.2, ang.3, a.0, a.1, a.2, b.0, b.1, b.2, c.0, c.1, c.2, d.0, d.1, d.2, ph];
                    let bits: Vec<String> = vals.iter().map(|x| format!("{:016x}", x.to_bits())).collect();
                    let cartan = cartan_from_angles(ang);
                    let inf = gate_infidelity(&u, &reconstruct_from_output(cartan, (a, b), (c, d), ph));
                    writeln!(f, "{} OK {:.3e} {}", name, inf, bits.join(" ")).unwrap();
                }
                Err(e) => writeln!(f, "{} ERR {:?}", name, e).unwrap(),
            }
        }
    }
}
