# C44-VAL -- fixed before the run (local; to be recorded in an Addendum with the results)

Written 2026-10-12 03:39 CEST, before the full run. Seen before: a smoke run of 4 jobs (gen_random_n10 on four devices),
and the development cases of item 75's revision: gen_random_n48 (C43-VAL's) on FakeAachen, FakeBoston, FakeKingston,
gen_random_n32, qaoa_N45 and qv_N28 (C43-VAL's) on FakeTorino. No test of C44-VAL was used in C41-VAL to C43-VAL.

Also done before this file (recorded with the results):
- RECHECK-17: the 17 test-device pairs that C42-VAL's psf arms refused, compiled again. Release 2026-10-11.4 compiles
  15 of them on no failed element; release 2026-10-11.3 compiles the same 15 when only Qiskit's VF2PostLayout is
  removed; the other 2 (cat_n130 on FakeTorino, cc_n151 on FakeKingston) cannot avoid failed elements (Qiskit level 3's
  outputs use them) and all releases refuse them.
- TIMING-PROBE: of 154 compiles with release 2026-10-11.4 (C42-VAL's tests on FakeTorino, C43-VAL's on FakeFez), one
  (dtc_100) reached the layout search's 2 s time budget.

- Prototype: psf_compile 2026-10-11.c44 (sha256 34875ef3b7ddd7ec6bee08ee1e5535767e20a865d07315b5792fb033b04fd732), built by make_c44.py
  (sha256 e54702288a756d754d150bc8e76ceffc8276ae902793bbfa617b0a1067f97b91) from c43 (= release 2026-10-11.4 but its version lines):
  - item 75 revised: the deterministic re-placement searches in three orders (c43's index order; device qubits by their
    own error and circuit qubits by two-qubit gate count; rustworkx's own VF2 order), each up to the trial limit, and
    keeps the lowest score;
  - item 77: `layout_search_time_budget_s` defaults to None (no time budget; call limits only).
- test_c44.py (sha256 867d79d83bf1625930e163828d3787d67f9b5fdcd8944e6057955a931c72e1df): 15 tests passed (c43's 13, c44's re-placement never scores above
  c43's on the same routed circuit, the same output under a clock that jumps 10 s per reading). Exactness sweep on c44:
  43 of 43 (c44_exact_sweep.txt).
- Script: c44_val.py (sha256 f0f46aa22b0ed2f0f96c90f6969d18b098d19ea498d2378438266b451c9afee5), run in a clone of the public release 2026-10-11.4 (psf_compile.py
  normalized sha256 f08f333d...8f2e). 29 tests: Benchpress QFT N = 11, 13, 15, 18, 22, 26, 30, 36 (8); QAOA N = 14, 19,
  24, 32, 38, 55 (6); quantum volume N = 7, 11, 18, 22 (4); generated with seeds "C44-VAL|...": random circuits n = 10,
  20, 30, 40, 48, 56 (6), Bernstein-Vazirani n = 30, 60, 100 (3), GHZ n = 30, 80 (2).
- Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
- Arms: QK3 (level 3, seed 0), R3 (release 2026-10-11.3), R4 (release 2026-10-11.4), C44, C44off (near_failed_error=None);
  then C44 again in the same process.
- Runnable: best ESP of the five >= 0.01; an output on a failed element has ESP 0. Ratios: geometric means.

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| U1 | C44 outputs on a failed element, per device | 0 | any |
| U2 | C44's second call gives the same output, per device | all | any differs |
| U3 | runnable, pooled: ESP C44 / R4 | >= 1.00 | < 0.99 |
| U4 | runnable tests with ESP C44 below 0.95 x R4's, pooled | <= 5% | > 10% |
| U5 | runnable, pooled: ESP C44 / R3, and tests below 0.95 x R3's (reported) | -- | -- |
| U6 | median compile time C44 / R4 | <= 1.5 | > 2.0 |
| U7 | pairs R3 refused and C44 compiled (reported); pairs C44 refused while QK3's output avoids failed elements | -- ; 0 | -- ; any |

Item 74's decision, fixed now: on the tests where item 74 tried in C44 (runnable), if the pooled ESP C44 / C44off is
>= 1.03 and the median time C44 / C44off on them is <= 1.6, item 74 stays on by default; otherwise the next release
sets `near_failed_error=None` by default.

Proposed rule for taking items 75 (revised) and 77 into the next release (the owner decides): U1, U2 confirmed on
every device, U3 and U4 not refuted, U6 not refuted, and U7's second part 0.
