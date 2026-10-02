# b17 score (SMOKE -- not a result)

versions {"psf": "2026-10-01.1", "layout": "2026-10-01.1", "core": "2026-09-29.1", "qiskit": "2.5.2", "python": "3.12.13", "git_head": "4839966"}
P0: PASS -- files 8 of 8 (missing []), short [], compile errors 0, control failures 0

## Failures (1 - F_avg > 1e-6) / circuits, worst 1 - F_avg

| workload | QK1 | QK2 | QK3 | QK3CZ | QK3U | PSF | PSFNG |
|---|---|---|---|---|---|---|---|
| W1 | 0/42 (7.4e-15) | 0/42 (9.3e-08) | 0/42 (9.3e-08) | 0/42 (9.3e-08) | 0/42 (9.3e-08) | 0/42 (7.7e-15) | 5/42 (9.8e-03) |
| W2 | 4/6 (5.0e-01) | 4/6 (5.3e-01) | 4/6 (5.3e-01) | 0/6 (8.5e-11) | 0/6 (4.6e-15) | 0/6 (4.0e-15) | 4/6 (3.1e-02) |
| W3 | 0/6 (8.9e-16) | 0/6 (2.3e-15) | 0/6 (2.3e-15) | 0/6 (2.1e-15) | 0/6 (1.8e-15) | 0/6 (2.1e-15) | 0/6 (2.1e-15) |
| W4 | 0/12 (8.9e-16) | 0/12 (6.7e-13) | 0/12 (6.7e-13) | 0/12 (7.5e-13) | 0/12 (6.7e-13) | 0/12 (8.9e-16) | 0/12 (8.9e-16) |

## W1 by cell: QK3 failures / circuits (rows dt, columns r = Jz/Jx)

| dt | 0 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|
| 0.001 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 0.01 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 0.1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |

PSF guard rejections of the ZSX decomposer (all workloads): 25

## Predictions

- H1 (QK2 and QK3 fail on >= 10% of W2): **CONFIRMED**
- H2 (QK1-3 never fail on Haar blocks, W3): **CONFIRMED**
- H3 (the PSF release never fails, any workload): **CONFIRMED**
- H4 (PSF with the guard off fails on >= 1% of W2): **CONFIRMED**
- H5 (a realistic physics workload hits the bug: QK3 fails on >= 1 W1 circuit): **REFUTED**

Reported: median compile s QK1 0.005, QK2 0.008, QK3 0.008, QK3CZ 0.008, QK3U 0.007, PSF 0.012, PSFNG 0.011
Reported: mean two-qubit count QK1 62.9, QK2 30.7, QK3 30.7, QK3CZ 30.8, QK3U 30.8, PSF 33.8, PSFNG 33.3
