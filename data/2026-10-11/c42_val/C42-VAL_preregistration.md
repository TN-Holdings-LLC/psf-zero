# C42-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-11 11:26 CEST, before the full run; revised 11:35 CEST for run 2 (below). Seen before: a smoke run of 4 jobs (gen_random_n10 on four devices,
item 74 not triggered), and one development case from C41-VAL (square_heisenberg_N81 on FakeBoston: the usual path's
ESP 2.0e-6 became 0.047 with item 74), which is why C42-VAL uses none of C41-VAL's circuits.

- Prototype: psf_compile 2026-10-11.c42 = c41 + item 74 (sha256 9277594afd30d477077fa0d9808072b23fe6736a26d81dcb097aafe48f2e518c),
  built by make_c42.py (sha256 9c880a68ace894dd6f8e9c70e2d11542e8edf41caaea5b29a9e3b319458f3919) from c41 (sha256 e41abde6...d0dc).
- Item 74: with the device's Target, when the output uses an element whose error is in [0.1, 0.5) -- a two-qubit gate
  on such a coupler, any gate on a qubit with such an sx error, a measurement of a qubit with such a readout error --
  the input is compiled again with every element of error >= 0.1 avoided, and the output with the lower summed
  -ln(1 - error) is returned (on a tie, or if the second compile cannot avoid them, the first). The threshold 0.1 was
  fixed before any measurement of it.
- test_c42.py (sha256 986413dd1b9cc0a611093715eec80368594b05b42aaf42033369446bce087b6a): 6 tests passed; c41's 9 tests also pass on c42.
- **Run 1 was stopped** at 11:33 CEST after 23 of 783 jobs: its test list included QASM files of 170,000-660,000 gates
  (factor247_n15, vqe_uccsd_n28, bwt_n37, square_root_n60, hwb12, ...) whose jobs would have used most of the time
  budget. None of its results were looked at; its partial data is kept as c42_val_1_stopped_partial. Run 2 differs
  only by one rule: QASM inputs with more than 50,000 instructions are left out (MAX_OPS), which removes 9 tests.
- Script: c42_val.py (sha256 a7acdee9a3518f6c45037114d0ba836d7a450672b64a43417b92bccd24cf2b75; run 1's was 4b266d67...953a). 122 tests, fixed by rules in the
  script, none used in C41-VAL: Benchpress QAOA N = 7, 11, 18, 35, 50, 80, 120 (7); square-lattice Heisenberg N = 121, 144 (2); QFT N = 8,
  16, 24, 32, 48, 64, 96 (7); quantum volume N = 8, 12, 16, 24, 32 (5); dtc cx, clifford_100 (2); every Feynman (32)
  and QASMBench (64) file with <= 156 qubits and <= 50,000 instructions outside C41-VAL's tests; random circuits
  n = 10, 20, 40 with seeds "C42-VAL|..." (3). A circuit without measurements gets measure_all().
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston (a test runs where it fits). Their
  near-failed elements (error in [0.1, 0.5)): couplers 0 / 1 / 2 / 0 / 2 / 4, qubits (sx or readout) 19 / 6 / 11 / 1 /
  0 / 1.
- Arms: QK3 (level 3, seed 0), R1 (release 2026-10-11.1), C41, C42; psf arms with the plain call with backend=.
- Runnable: best ESP of the four >= 0.01; ESP of an output on a failed element = 0. Ratios: geometric means.
  "Tried": item 74 compiled the alternative (NEAR_STATS["tried"] > 0 in the C42 call).

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| W1 | C42 outputs on a failed element, per device | 0 | any |
| W2 | every test: ESP C42 >= ESP C41, per device (by design) | all | any lower |
| W3 | tests where item 74 did not try: C42's output = C41's by value, per device | all | any differs |
| W4 | tests where item 74 tried, runnable, pooled over all devices: ESP C42 / C41 | >= 1.10 | < 1.02 |
| W5 | share of tests where item 74 tries, per device (reported) | -- | -- |
| W6 | tests where item 74 tried: median compile time C42 / C41 | <= 2.5 | -- |
| W7 | all runnable tests: ESP C42 / R1 and C42 / QK3, per device (reported) | -- | -- |

Between the confirmed and refuted bounds: ambiguous.

Proposed rule for taking item 74 into the next bundled release (the owner decides): W1-W3 confirmed on every device
and W4 confirmed.
