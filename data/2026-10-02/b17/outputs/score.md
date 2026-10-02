# b17 score

versions {"psf": "2026-10-01.1", "layout": "2026-10-01.1", "core": "2026-09-29.1", "qiskit": "2.5.2", "python": "3.12.13", "git_head": "f0095e6"}
P0: PASS -- files 8 of 8 (missing []), short [], compile errors 0, control failures 0

## Failures (1 - F_avg > 1e-6) / circuits, worst 1 - F_avg

| workload | QK1 | QK2 | QK3 | QK3CZ | QK3U | PSF | PSFNG |
|---|---|---|---|---|---|---|---|
| W1 | 0/2100 (1.4e-14) | 0/2100 (8.9e-08) | 0/2100 (8.9e-08) | 0/2100 (8.9e-08) | 0/2100 (8.9e-08) | 0/2100 (1.6e-14) | 248/2100 (1.6e-02) |
| W2 | 707/1000 (6.8e-01) | 707/1000 (6.8e-01) | 707/1000 (6.8e-01) | 0/1000 (3.5e-09) | 0/1000 (6.6e-15) | 0/1000 (5.9e-15) | 599/1000 (6.6e-01) |
| W3 | 0/1000 (5.2e-15) | 0/1000 (4.8e-15) | 0/1000 (4.8e-15) | 0/1000 (3.5e-09) | 0/1000 (5.2e-15) | 0/1000 (4.8e-15) | 0/1000 (4.8e-15) |
| W4 | 0/900 (3.1e-15) | 0/900 (3.8e-09) | 0/900 (3.8e-09) | 0/900 (3.8e-09) | 0/900 (2.9e-09) | 0/900 (3.1e-15) | 0/900 (3.1e-15) |

## W1 by cell: QK3 failures / circuits (rows dt, columns r = Jz/Jx)

| dt | 0 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|
| 0.001 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| 0.01 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| 0.1 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |

PSF guard rejections of the ZSX decomposer (all workloads): 1841

## Predictions

- H1 (QK2 and QK3 fail on >= 10% of W2): **CONFIRMED**
- H2 (QK1-3 never fail on Haar blocks, W3): **CONFIRMED**
- H3 (the PSF release never fails, any workload): **CONFIRMED**
- H4 (PSF with the guard off fails on >= 1% of W2): **CONFIRMED**
- H5 (a realistic physics workload hits the bug: QK3 fails on >= 1 W1 circuit): **REFUTED**

Reported: median compile s QK1 0.005, QK2 0.007, QK3 0.008, QK3CZ 0.008, QK3U 0.007, PSF 0.010, PSFNG 0.009
Reported: mean two-qubit count QK1 45.7, QK2 24.1, QK3 24.0, QK3CZ 24.3, QK3U 24.3, PSF 26.1, PSFNG 25.7
