# C43-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-11 13:49 CEST, before the full run. Seen before: a smoke run of 4 jobs (gen_random_n8 on four devices;
C43 = C41 there), the two C42-VAL cases used to develop item 75 (qasmbench bv_n30 on FakeFez, bv_n140 on FakeMarrakesh:
c43 gave one output over three calls), the exactness sweep and the readability survey below. No test of C43-VAL was
used in C41-VAL or C42-VAL.

- Prototype: psf_compile 2026-10-11.c43 (sha256 8fdeb421a2e757780d2e2ada3ece93d45eb0d4f3f289c268fcd6ade1dfb9b1f0), built by make_c43.py
  (sha256 76e29c4598323e017239087082497077ecb6c10ddb490d0198c3a08e9517ce22) from c42 (sha256 9277594a...518c). Release 2026-10-11.3 is c41 with its version
  lines changed; c43 = c42 + the following:
  - item 74, revised: only the near-failed elements (error in [0.1, 0.5)) that the first output used are avoided in
    the second compile (a copy of the Target with those marked failed); the better on the real Target is kept;
  - item 75: a deterministic exact re-placement (rustworkx VF2 with id_order, every placement scored by the summed
    -ln(1 - error) on the device's Target) in place of Qiskit's VF2PostLayout in the preset pass managers, item 33's
    re-placement and item 68's relabel; without a Target, no re-placement;
  - item 76: item 73 also reads circuits whose final Clifford is a qubit permutation (QFT with its swaps).
- Root cause of the run-to-run variation (found before this file): Qiskit 2.5.2's VF2PostLayout returns a different
  `post_layout` from call to call with a fixed seed (traced pass by pass on a 6-qubit random circuit).
- test_c43.py (sha256 cdad961450cac923a6ab8f9ca1c837cdbd5aa56d1ce0a26f1eba75b21b8e4b9c): 13 tests passed (c41's item-73 tests on c43, QFT with swaps and
  a swapped cx-rz-cx circuit exact, the same output over three calls with and without the device, item 74's marked
  elements, never worse than item 74 off on a device with a near-failed coupler).
- Exactness sweep (benchmarks/c41_exact_sweep.py on c43): 43 of 43 Benchpress files of <= 7 qubits read by items 73/76
  give the input's measured distribution (c41_exact_sweep.txt). Readability: 312 of 519 Benchpress QASM files (c41:
  213; the 99 new ones are the QFT files).
- Script: c43_val.py (sha256 1e469216d99fc5943da65abc5bc25668b1888a626b1660b6052de12b95691210). 35 tests, fixed by rules in the script: Benchpress QFT
  N = 5, 6, 7, 9, 10, 12, 14, 20, 28, 40 (10); QAOA N = 6, 9, 13, 17, 22, 27, 45, 70 (8); quantum volume N = 6, 9, 14,
  20, 28 (5); generated with seeds "C43-VAL|...": random circuits n = 8-48 (6), Bernstein-Vazirani n = 20, 40, 80 (3),
  Qiskit basis-gate exports of Trotter evolutions n = 12 (3).
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
- Arms: QK3 (level 3, seed 0), C41 (release 2026-10-11.3 but its version lines), C42, C43; then C41 and C43 again in the
  same process (repeat check).
- Runnable: best ESP of QK3, C41, C42, C43 >= 0.01; ESP of an output on a failed element = 0. Ratios: geometric means.
  "Unread": items 72-73/76 do not read the input; "tried": item 74 compiled its alternative.

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| V1 | C43 outputs on a failed element, per device | 0 | any |
| V2 | C43's second call gives its first call's output by value, per device | all | any differs |
| V3 | unread tests where item 74 did not try, runnable, pooled: ESP C43 / C41 (item 75's effect) | >= 1.00 | < 0.98 |
| V4 | runnable tests with ESP C43 below 0.95 x C41's, pooled | <= 10% | > 20% |
| V5 | tests where item 74 tried, runnable, pooled: ESP C43 / C41 (with the share of "no alternative", reported) | >= 1.05 | < 1.01 |
| V6 | QFT tests (item 76), runnable, pooled: ESP C43 / C41 | >= 1.02 | < 1.00 |
| V7 | QAOA QASM tests, runnable, pooled: ESP C43 / C41 (reported) | -- | -- |
| V8 | median compile time C43 / C41 (reported) | <= 2.0 | -- |

Also reported: how often C41 repeats its own output.

Between the confirmed and refuted bounds: ambiguous.

Proposed rule for taking items 74-76 into the next bundled release (the owner decides): V1 and V2 confirmed on every
device, V3 not refuted, V4 confirmed, and V5 or V6 confirmed.
