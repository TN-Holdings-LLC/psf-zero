# psfz twin: time against the bond dimension (exploratory, 2026-10-20, about 03:35 CEST)

Noiseless TFIM chains (the generator of `rust/psfz/harness/twin_check.py`, seed 20261020) on FakeFez, simulated by
`psfz twin` built with the `lapack` feature, one core, in the cloud workspace. Run from the repository root:
`python data/2026-10-20/psfz_twin_chi_scaling/chi_scale.py rust/psfz/target-lapack/release/psfz`.

| n, steps | bound = bond reached | two-site gates | simulate |
|---|---|---|---|
| 16, 6 | 64 | 90 | 0.05 s |
| 18, 7 | 128 | 119 | 0.40 s |
| 18, 8 | 256 | 136 | 2.04 s |
| 20, 9 | 512 | 171 | 11.08 s |

About 5 x per doubling of the bond dimension. Used in `docs/design/review_notes_2026-10-20b.md` (W2) to set the noisy
twin's χ_max by time. The figure for χ = 1,024 there is an extrapolation, not a measurement.
