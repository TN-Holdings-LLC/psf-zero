# C45-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-12 05:45 CEST, before the full run. Seen before: a smoke run of 4 jobs (gen_random_n14 on four devices);
development on C44-VAL's tests (already used, `dev45_development.jsonl`, below) and on gen_random_n48 and gen_bv_n100
(C43-VAL's and C44-VAL's). No test of C45-VAL was used in C41-VAL to C44-VAL.

## What was found before this file

- **Diagnosis (gen_random_n48, FakeBoston).** Qiskit level 3's ESP is 0.029 with its own VF2PostLayout and 5.5e-8
  without it; c44 gave 2.8e-5. Qiskit's gain on this circuit is entirely its re-placement right after routing. c44's
  replacement of that pass (item 75) gave up at the routing stage whenever the circuit had a gate the device does not
  run natively, i.e. on almost every circuit.
- **Tried and not kept.** The same routing-stage re-placement inside PSF-Zero's own pipeline: ESP 0.996 times c44's on
  118 runnable development tests, and 0.547 on one (gen_bv_n100, FakeTorino). PSF-Zero's routed circuits rarely have an
  isomorphic copy on better qubits; the proxy's choice before translation narrowed item 33's later search.
- **Tried and not kept.** Error-aware regions (compile again inside a connected region of n + slack qubits grown by
  error): gen_random_n48 ESP 0.012 on FakeBoston (QK3 0.029), 0.0054 on FakeAachen (QK3 0.0035), 0.0004 on FakeKingston
  (QK3 0.0033), at 5-90 s. Not in c45.
- **A bug.** `compile_for_hardware(qc, target=t)` without `backend=` left the output untranslated (basis gates were
  taken from the Target only with `backend=`). Fixed in c45 (item 79). Releases 2026-10-11.3 and .4 have it.
- **Exactness sweep of c45's full call** (`c45_exact_sweep.txt`): 61 Benchpress files of <= 7 qubits on the 7-qubit
  test device; 60 give the input's distribution to 1e-6; hhl_n7 differs by 1.3e-6, from Qiskit level 3's output
  (level 3 removes rotations of about 1e-6 rad even with approximation_degree=1.0; PSF-Zero's own output differs by
  6e-8). 48 of the 61 outputs are level 3's. 9 files with classical control were skipped.

## The prototype

- psf_compile 2026-10-12.c45 (sha256 dc51b0492f01b4c431056d6c3a6dda740f07b0b56f9ea8530bef049076ed8859), built by
  make_c45.py (sha256 00a5d86da6c0c7b9f875ca29c73d877d9253cbc10ea61960d3f9b445886d6d88) from c44:
  - item 78: item 75's deterministic re-placement scores the routing stage's gates by their translated cost (swap 3
    native two-qubit gates, cx-like 1, others 2; single-qubit gates 0-2 sx; no placement on failed elements; at most
    25,000 trials and 3,000,000 VF2 calls per search order) -- **only inside item 80's level-3 compile**;
    PSF-Zero's own pipeline is c44's;
  - item 79: basis gates from a Target given alone;
  - item 80: with the device and no caller's map or layout (`level3_candidate="auto"`), Qiskit's level-3 preset on the
    device's Target, under item 75/78's deterministic re-placement, is a candidate; it is returned when its summed
    -ln(1 - error) is lower than PSF-Zero's (a tie keeps PSF-Zero's), refused if it uses a failed element, and checked
    by item 39's `_implements` only when the input has no measurement and at most 16 qubits ("unverified" otherwise).
- test_c45.py (sha256 34d795a036b65d656ae6f013529dfae3d33079e0a038e6c871145bcf01ace345): 23 tests passed (c44's 15 on
  c45; items 78-80; `level3_candidate=False` gives c44's output by value).
- Development (C44-VAL's tests, already seen; 162 complete, 137 runnable): ESP C45 / QK3 1.182, C45 / C44 1.085,
  C45n / C44 1.000 (C45n = c45 with `level3_candidate=False`), C45n / QK3 1.090; level 3's output kept in 81 of 174;
  median time C45 / C44 1.44, largest 14 s; GHZ below QK3 (0.977).

## The run

- Script: c45_val.py (sha256 14e993ea513cd51318de266c0c61b86f02607f934b47eef31b23aa38da0a9079), run in a clone of the
  public release 2026-10-11.4 with c44 and c45 under `patches/`. 24 tests: Benchpress QFT N = 17, 21, 25, 33, 44 (5);
  QAOA N = 16, 23, 29, 42, 65 (5); quantum volume N = 13, 15, 19, 26 (4); clifford_20 (1); generated with seeds
  "C45-VAL|...": random circuits n = 14, 26, 36, 44, 52 (5), Bernstein-Vazirani n = 50, 90 (2), GHZ n = 40, 100 (2).
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
- Arms: QK3 (Qiskit level 3, seed 0), R4 (release 2026-10-11.4), C44, C45, C45n (`level3_candidate=False`); then C45
  again in the same process (C45r).
- Runnable: best ESP of the five >= 0.01; an output on a failed element has ESP 0. Ratios: geometric means.

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| X1 | C45 outputs on a failed element, per device | 0 | any |
| X2 | C45's second call gives the same output, per device | all | any differs |
| X3 | runnable, pooled: ESP C45 / QK3; and runnable tests below 0.98 x QK3's | >= 1.10; <= 10% | < 1.05; > 20% |
| X4 | ESP C45 / C44, runnable, pooled; and complete tests where C45's -log10 ESP is above C44's by > 1e-9 | >= 1.03; 0 | < 1.00; any |
| X5 | C45n's output equals C44's by value, every job where both compile | all | any differs |
| X6 | median compile time C45 / C44 | <= 2.0 | > 2.5 |
| X7 | level 3's output kept / tried, and how many of those unverified (reported) | -- | -- |

The two numbers to report: **display** ESP C45 / QK3 (with Qiskit's output as a candidate); **essential** ESP C45n / QK3
(PSF-Zero's own pipeline against Qiskit) and the share of tests where Qiskit's output is returned.

Proposed rule for taking items 78-80 into the next release (the owner decides): X1, X2, X5 confirmed on every device,
X4's count 0, X3 and X6 not refuted.
