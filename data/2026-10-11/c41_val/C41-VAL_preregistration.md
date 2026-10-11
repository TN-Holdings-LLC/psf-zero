# C41-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-11 10:51 CEST, before the full run. Seen before: a smoke run of 4 jobs (control_random_n6 on four
devices) and one job (qaoa_N5 on FakeTorino); the readability survey of Benchpress's QASM files (which files item 73
reads, term counts; no compilation); the exactness sweep below.

- Prototype: psf_compile 2026-10-11.c41 = c40 + item 73 (sha256 e41abde63c54094e2ad16e98054104ca2e4c2b4fee9f98edbd002aab39a9d0dc),
  built by make_c41.py (sha256 8a79f0dccd1b30a610da23aabd50aaf8700ce37d42bf11058bdfb13008b636ea) from c40 (sha256 9ab1478d...1411).
- test_c41.py (sha256 1e4575c98d743791c8706c281745f1044f2f049aedc5f6765510b1c6e18bf0ed): 9 tests passed (exact construction on 14 kinds of input
  incl. cx-rz-cx QAOA, Heisenberg with rx/ry basis changes, Qiskit basis-gate exports, cp/crz/t gadgets, Toffolis, QFT
  without swaps, mixed gates with a u gate and an X-basis end; unreadable inputs as the release by value).
- Exactness sweep (sweep2.py with meascheck.py): all 37 Benchpress QASM circuits of <= 7 qubits that item 73 reads,
  compiled by STEINER on a 7-qubit test device: the measured distribution equals the input's for 6 random product
  input states each (tolerance 1e-6): 37 / 37. (Qiskit level 3 removes diagonal gates before measurements, so the
  unitary itself is not the test with measurements.) Item 73's outputs equal the inputs up to a global phase.
- Script: c41_val.py (sha256 c9a1d4e17d191ba6fab28dcfe028cc9ee09ac0f403694067ef9584ac0f488bea). 89 tests, fixed by rules in the script:
  - qaoa_qasm: Benchpress qaoa_barabasi_albert_N{5,8,10,12,15,20,25,30,40,60,100}_3reps (11);
  - heisenberg_qasm: square_heisenberg_N{4,9,16,25,36,49,64,81,100} (9);
  - qasmbench: every QASMBench file (not *_transpiled) that item 73 reads, <= 156 qubits (39);
  - feynman: every Feynman file that item 73 reads (14);
  - gen_export: generated evolutions (tfim LieTrotter 2 steps n = 10, 16; Heisenberg n = 10, 16; random Paulis n = 10)
    compiled by Qiskit to cx, rz, sx, x, h at level 1 (5); gen_qaoa_gates: cx-rz-cx QAOA p = 2 on 3-regular graphs,
    n = 12, 20, 28 (3); seeds "C41-VAL|...";
  - control (not read by item 73): qft_N010, qft_N020, qv_N010, clifford_20, feynman adder, gf2^4_mult, random n = 6, 8 (8).
  A circuit without measurements gets measure_all().
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston (a test runs where it fits).
- Arms: QK3 (level 3, seed 0), R1 (release 2026-10-11.1), C40, C41; psf arms with the plain call with backend=.
- Runnable: best ESP of the four >= 0.01; ESP of an output on a failed element = 0. Ratios: geometric means.

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| Z1 | C41 outputs on a failed element, per device | 0 | any |
| Z2 | every test: ESP C41 >= ESP R1, per device (by design of item 69) | all | any lower |
| Z3 | every test: ESP C41 >= ESP C40, per device | all | any lower |
| Z4 | controls: C41's output = C40's = R1's by value, per device | all | any differs |
| Z5 | qaoa_qasm, runnable, pooled: ESP C41 / R1 | >= 1.10 | < 1.02 |
| Z6 | heisenberg_qasm, runnable, pooled: ESP C41 / R1 | >= 1.05 | < 1.01 |
| Z7 | qasmbench and feynman, runnable, pooled: ESP C41 / R1, and on how many tests STEINER is chosen (reported) | -- | -- |
| Z8 | gen_export + gen_qaoa_gates, runnable, pooled: ESP C41 / R1 | >= 1.05 | < 1.01 |
| Z9 | median compile time of C41 on tests it reads (reported with R1's and the maximum) | <= 5 s | -- |

Between the confirmed and refuted bounds: ambiguous.

Proposed rule for taking item 73 into the next bundled release (the owner decides): Z1-Z4 confirmed on every device,
and at least one of Z5, Z6, Z8 confirmed pooled.
