# c49 (T2P) evaluation, scored 2026-10-13T07:48:33+00:00 (2026-10-13 09:48:33 CEST)

devices: FakeFez; circuits: 27

**H0 (harness check):** C49_esp equals C48 on 26 of 26 compiles

| device | V1 share cost T2P <= C48 | V2 gmean exposure T2P/C48 (BA) | V3 gmean ESP T2P/C48 | V3 min ESP ratio | V4 low-T2 qubits T2P / C48 | V5 median 2q ratio | V5 median duration ratio | V6 median time ratio |
|---|---|---|---|---|---|---|---|---|
| FakeFez | 1.000 (27) | 1.000 (20) | 1.0000 | 1.0000 | 19 / 19 | 1.000 | 1.000 | 1.06 |

**V7 (reported).**

| device | T2P_half/T2P exposure (gmean) | T2P_half/T2P ESP (gmean) | T2P/QK3 ESP | T2P/QK3 exposure | C48/QK3 ESP | C48/QK3 exposure | cost T2P - QK3 (mean) | cost C48 - QK3 (mean) |
|---|---|---|---|---|---|---|---|---|
| FakeFez | 1.000 | 1.0000 | 1.3108 | 0.630 | 1.3108 | 0.630 | -0.5388 | -0.5388 |

| ID | verdict |
|---|---|
| V0 | CONFIRMED (53 outputs, exactness checked on 53; failures: []) |
| V1 | CONFIRMED |
| V2 | REFUTED |
| V3 | CONFIRMED |
| V4 | REFUTED |
| V5 | CONFIRMED (pooled medians: 2q 1.000, duration 1.000) |
| V6 | CONFIRMED (pooled median 1.06) |
