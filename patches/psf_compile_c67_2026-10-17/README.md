# Prototype c67 (home, 2026-10-17): item 104, devices whose two-qubit gate is ECR (IBM Eagle)

`make_c67.py` = release 2026-10-16.2 (`psf_compile.py`) + `item104_ecr_code.py` -> `psf_compile.py`, with option A's
`_T100Formula._frame_items` and `_T63A_I` removed (unreachable since 2026-10-16.2). Not validated; not a release.

**Why.** Every formula path builds from CZ, so 2026-10-16.2 refuses every ECR device (`OutOfScopeError`: "the
device's two-qubit gate is not cz").

**Item 104.** CZ and ECR are the same two-qubit class: CZ(a, b) = fixed single-qubit gates around ECR(a, b) (global
phase pi/4; multiplied out and checked at import). On an ECR device (no CZ) the call runs the CZ path on a *CZ view* of
the device: CZ on every coupler with ECR in either direction, with that ECR's error and duration; every other
instruction as the device gives it. Each CZ of the output is then written as the coupler's ECR in its direction with
its fixed gates, and the single-qubit runs are merged into rz, sx and x by Qiskit's one-qubit synthesis (a closed form
per run). Placement, scope and refusals are the CZ path's, on the view. Not covered: `compile_template` on ECR devices
(still refused).

**Checks.**

- `check_c67_cz.py`: on CZ devices, c67 against 2026-10-16.2, bit for bit, on C63-VAL's 49,590 inputs (templates)
  and 2,090 default calls (`check_c67_cz_result.txt`).
- Exploratory, not pre-registered (`data/2026-10-17/ecr_explore/`): C63-VAL's 19 tests on FakeSherbrooke,
  FakeBrisbane, FakeKyiv and FakeOsaka, against Qiskit level 3 -- see Addendum 515.
