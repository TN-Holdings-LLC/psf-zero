# spare-qubit-cliff: Combined Addenda, Part 9 of 9 (Addendum 248 onward)

**Continued from [Part 8](spare-qubit-cliff-combined-135.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 7](spare-qubit-cliff-combined-108.md)).** Same conventions as every prior part.

**Note on this part specifically** (2026-09-29): Part 8 had grown past 700 KB, so a new
Part starts here, at the workplace records of 2026-09-29 (Addenda 248-259, first merged
into Part 8 in commit 7151e79 and moved here unchanged). Addendum 212, reserved for the
Stage-2 results of Addendum 211 (Part 8), will be recorded in this Part when it is
available, out of numerical order.

---

<!-- ===== Addendum 248 (source: spare-qubit-cliff-addendum-248-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Workplace pre-registration (sandbox): does PSF-Zero meet the CommutativeCancellation angle cutoff of Addendum 247 B at routing level 1 (default) and at level 3?

## Addendum 248 -- Pre-registration: does PSF-Zero meet Qiskit's CommutativeCancellation angle cutoff (Addendum 247 B) at routing level 1 and at level 3? (2026-09-29)

**Status: pre-registration, locked at the Project save time of this
document**, before the scored run. A small dry run (section 6) was made
before locking; its outcome is disclosed. **The run is made in the
workplace sandbox**; times are sandbox times.

## 1. Why this experiment exists

Addendum 247 B (home) traced Qiskit L3's two inexact real-target outputs to
`CommutativeCancellation` dropping a merged Z rotation below a fixed
cutoff. The home handover (item 2) asks two things of the workplace:

- confirm the rule in Qiskit's Rust source;
- check whether PSF-Zero's `compile_for_hardware` path meets this pass.

### 1.1 Source reading (done before this pre-registration; descriptive)

Qiskit 2.5.2 (tag, commit `c1c01ad`),
`crates/transpiler/src/passes/commutation_cancellation.rs`:

- `const _CUTOFF_PRECISION: f64 = 1e-5`.
- `is_multiple_of_pi(angle, factor)` computes
  `(angle / (factor * PI)).rem_euclid(1.0)` and returns true when the result
  is within `_CUTOFF_PRECISION` of 0 or of 1.
- A merged rotation is dropped when `is_multiple_of_pi(total_angle, 4.)`,
  that is, when its angle is within 4 pi x 1e-5 = 1.2566e-4 of a multiple of
  4 pi. Within 2 pi x 1e-5 of an odd multiple of 2 pi it is also dropped, and
  a phase of pi is added.
- Merged X rotations are replaced by `x` or `sx` gates when they are within
  pi x 1e-5 or (pi/2) x 1e-5 of a multiple.

This matches the boundaries measured in Addendum 247 B to all printed
digits. It confirms the interpretation left open there: the constant is
applied relative to the period being tested.

On Qiskit main (commit `1298f96`, 2026-09-28), the Python class
`CommutativeCancellation` now accepts `approximation_degree` (default 1.0).
The value is passed only to the commutation analysis
(`analyze_commutations`). The angle cutoff still uses the fixed
`_CUTOFF_PRECISION = 1e-5`, so it is not adjustable on main either.

In Qiskit 2.5.2's preset pass managers
(`preset_passmanagers/builtin_plugins.py`), `CommutativeCancellation` runs
at optimization levels 2 and 3 (init and optimization stages), **not at
level 1**. `compile_for_hardware` routes with `transpile(...,
optimization_level=routing_optimization_level)`, default 1.

## 2. Fixed design ([`cutoff_psf_levels.py`](../../benchmarks/cutoff_psf_levels.py))

Arms:

- **Q3:** `transpile` at `optimization_level=3`.
- **P1:** `compile_for_hardware`, routing level 1 (the default).
- **P3:** `compile_for_hardware`, `routing_optimization_level=3`.

PSF-Zero is the release pair: `psf_compile.py` 2026-09-28.1 with
`CORE_VERSION` 2026-09-28.1 and `REFINE_THRESHOLD` 1e-14.

**Part M (minimal circuits):**

- Circuit: `rz(-pi/2) q0 . cz(0, 1) . rz(delta + pi/2) q0` (merged angle
  delta), 2 qubits, line coupling map, basis cz/rz/sx/x.
- 12 offsets: delta = ±5e-5, ±1.0e-4, ±1.2e-4 (inside the cutoff) and
  ±1.3e-4, ±2e-4, ±1e-3 (outside).
- Error: phase-aligned Frobenius distance of the 4x4 operators, with the
  final layout applied.

**Part R (realistic circuits):**

- FakeNighthawk (`loop_endurance.nighthawk()`: 120 qubits, native
  cz/id/x/sx/rz).
- 50 circuits of 60 disjoint `pair24` blocks with random angles
  (seed = 5000 + i), so 3,000 pairs per arm.
- Q3 compiles with the backend's `target`. P1 and P3 compile with its
  coupling map and native basis, `entangling_basis="cx"`,
  `layout_search=True`, `seed_transpiler=0`.
- Error: per-pair phase-aligned distance (the construction of
  `real_target_cliff.pair_check`).

## 3. Pre-registered predictions

Harness gate **C0**: versions as above; 36 M cells and 150 R cells; per-pair
check applicable (no two-qubit gate joining pairs) in at least 90% of each
arm's circuits (only applicable circuits are scored).

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| K1 | Q3 drops the merged rotation inside the cutoff (minimal) | error = abs(delta) within 1% for the 6 inside offsets, and <= 1e-12 for the 6 outside | any inside offset exact, or any outside offset > 1e-10 |
| K2 | P1 never meets the cutoff (minimal) | every offset <= 1e-12 | any > 1e-10 |
| K3 | P3 behaves like Q3 (minimal) | as K1 | as K1 |
| K4 | P1 exact on realistic circuits | 0 of the pairs > 1e-12, worst <= 1e-12 | any pair > 1e-10 |
| K5 | Q3's inexact pairs are cutoff-sized | worst pair error <= 5e-4 | any > 1e-3 (another mechanism) |
| K6 | P3 meets the cutoff on realistic circuits | at least one pair > 1e-12 | (none; 0 pairs = ambiguous) |

Between the bounds: ambiguous. K1-K3 are controls with known outcome (see
section 6); the new predictions are K4-K6. **Reported without prediction:**
the number and sizes of inexact pairs per arm, two-qubit counts, compile
times (sandbox).

## 4. What this can and cannot establish

It tests, in this sandbox and on FakeNighthawk (not ibm_kingston, whose
Target is only at home), whether PSF-Zero's default path avoids the cutoff
and whether level 3 brings it back. A 0 in K5's count would not mean Qiskit
L3 is exact: the rate seen at home was 2 in 729 pairs. The mechanism
attribution for any inexact pair is by size only; no pass trace is part of
this run.

## 5. Environment, files, run commands

- Workplace sandbox: Python 3.11.15, Qiskit 2.5.2, numpy 2.4.4,
  qiskit-ibm-runtime 0.50.0 (fake provider only; no account).
- A clean clone of the repository at `f4b4a6c` (`psf_compile.py`
  2026-09-28.1), with the script placed in `benchmarks/`.
- Fixed core: a sandbox build of the release source (`src/lib.rs`
  `bf3bf537...`), `CORE_VERSION` 2026-09-28.1.

[`cutoff_psf_levels.py`](../../benchmarks/cutoff_psf_levels.py) (Project: `psf-zero/benchmarks/`), 10,805 bytes,
normalized SHA-256
`064558605d749a15ae3ca5b8eead366739d273ab8fd419b3928326aed0ca830a`.

```
python -u benchmarks/cutoff_psf_levels.py run   2>&1 | tee cutoff_run.txt
python -u benchmarks/cutoff_psf_levels.py score 2>&1 | tee cutoff_score.txt
```

Expected run time about 16 minutes: Q3 takes about 19 s per 120-qubit
circuit here, the Nighthawk layout cliff.

## 6. Dry run before locking (outcome disclosed)

A copy with N_R = 2 (seeds 5000 and 5001, which are also the first two
seeds of the scored run):

- **Part M (complete, 36 cells):** Q3 and P3 dropped the rotation at all 6
  inside offsets (error = abs(delta) to 4 digits) and were exact outside
  (<= 3.1e-16). P1 was exact everywhere (0.0; it keeps `rz, cz, rz`).
  **So K1-K3 are known before the lock**; Part M is deterministic and is
  re-run only as a control.
- **Part R (2 circuits x 3 arms):** no pair above 1e-12 in any arm (worst Q3
  9.8e-14, P1 1.0e-14, P3 1.5e-13); two-qubit count 180 for every arm; Q3
  about 18.5 s per compile, P1 and P3 about 0.04 s.

One change after the dry run: C0 originally required the per-pair check to
be applicable in every circuit. It was relaxed to 90% per arm, with only
applicable circuits scored, so that one routed Q3 output cannot void the
whole run. The thresholds of K1-K6 were written before the dry run and were
not changed.

---

<!-- ===== Addendum 249 (source: spare-qubit-cliff-addendum-249-2026-09-29.md) ===== -->

> **Note added when merging:** K1-K5 confirmed, K6 ambiguous: the default path (routing level 1) never runs the pass and was exact everywhere; at level 3 PSF-Zero drops the same rotations as Qiskit on minimal circuits; on 3,000 FakeNighthawk pairs no arm, not even Qiskit L3, produced an inexact pair.

## Addendum 249 -- PSF-Zero's default path never meets Qiskit's CommutativeCancellation cutoff; at routing level 3 it does (minimal circuits); on 3,000 FakeNighthawk pairs no arm, not even Qiskit L3, produced an inexact pair (K1-K5 confirmed, K6 ambiguous) (2026-09-29)

**Scored against:** `cutoff-psf-levels-preregistration-2026-09-29.md`,
locked in the Project before the scored run (script [`cutoff_psf_levels.py`](../../benchmarks/cutoff_psf_levels.py)
saved at the same time). Thresholds applied exactly as written.

**Run:** workplace sandbox (not the pod, not home): Intel(R) Xeon(R)
Processor @ 2.80GHz, 2 CPUs, `Linux-6.18.44-fc-v37-x86_64-with-glibc2.39`,
Python 3.11.15, Qiskit 2.5.2, numpy 2.4.4, qiskit-ibm-runtime 0.50.0 (fake
provider only). Clean clone at `f4b4a6c`; `psf_compile.py` 2026-09-28.1
(`3616efc8...` on the LOADED line), `CORE_VERSION` 2026-09-28.1 (sandbox
build of `bf3bf537...`), `REFINE_THRESHOLD` 1e-14. Script normalized
SHA-256 `064558605d749a15ae3ca5b8eead366739d273ab8fd419b3928326aed0ca830a`,
checked before the run and printed by it. Run 00:29:39-00:45:48 UTC.
**Times are workplace-sandbox times.** **Amendments after the lock:** none.

## 1. Scoring

| ID | Prediction | Verdict | Numbers |
|---|---|---|---|
| C0 | versions; 36 M and 150 R cells; per-pair check applicable in >= 90% per arm | passed | applicable 50/50 in every arm |
| K1 | Q3 drops the merged rotation inside the cutoff (minimal) | CONFIRMED (control) | error = abs(delta) at all 6 inside offsets; <= 3.1e-16 at the 6 outside |
| K2 | P1 never meets the cutoff (minimal) | **CONFIRMED** | 0.0 at every offset (output keeps `rz, cz, rz`) |
| K3 | P3 behaves like Q3 (minimal) | CONFIRMED (control) | identical to Q3 |
| K4 | P1 exact on realistic circuits | **CONFIRMED** | 0 of 3,000 pairs > 1e-12; worst 1.12e-14 |
| K5 | Q3's inexact pairs are cutoff-sized | **CONFIRMED** | worst 3.13e-13; **0 of 3,000 pairs inexact** |
| K6 | P3 meets the cutoff on realistic circuits | **AMBIGUOUS** | 0 of 3,000 pairs > 1e-12; worst 2.63e-13 |

K1-K3 were known from the dry run (Part M is deterministic) and act as
controls here.

## 2. Numbers (Part R, 50 circuits x 60 pairs, sandbox)

| arm | median compile | two-qubit count | pairs > 1e-12 | worst pair |
|---|---|---|---|---|
| Q3 (transpile L3, target) | 18.54 s | 180 in all 50 | 0 | 3.13e-13 |
| P1 (PSF-Zero, routing level 1) | 0.038 s | 180 in all 50 | 0 | 1.12e-14 |
| P3 (PSF-Zero, routing level 3) | 0.044 s | 180 in all 50 | 0 | 2.63e-13 |

## 3. Reading

1. **The rule is confirmed in Qiskit's source and in behaviour.** Qiskit
   2.5.2's `is_multiple_of_pi` drops a merged Z rotation within
   4 pi x 1e-5 of a multiple of 4 pi. On Qiskit main (2026-09-28) the new
   `approximation_degree` argument of `CommutativeCancellation` reaches only
   the commutation analysis; the angle cutoff is still fixed (see the
   pre-registration, section 1.1).
2. **PSF-Zero's default path is immune by construction.** The pass does not
   run at optimization level 1, and `compile_for_hardware` routes at level 1
   by default. On the minimal circuit it keeps the two rotations and is
   exact at every offset (K2). On 3,000 realistic pairs no output was
   inexact (K4). The 12 exact PSF-Zero outputs of Addendum 246 are
   therefore expected on this path, not luck.
3. **At routing level 3 PSF-Zero is exposed like Qiskit.** On the minimal
   circuit P3 drops exactly the same rotations as Q3 (K3). Anyone who passes
   `routing_optimization_level=3` (for example to close the routing CX gap, as
   in the routing-arms experiment of 2026-09-28) accepts this cutoff too. On the
   realistic circuits here it did not occur (K6 ambiguous).
4. **The rate depends on the device or the circuits.** Qiskit L3 had 2
   inexact pairs in 729 on ibm_kingston's Target at home (Addendum 246). It
   had 0 in 3,000 here on FakeNighthawk. At the home rate about 8 would be
   expected here, and seeing 0 would then have a probability of about
   3e-4. So the difference is unlikely to be chance. Possible reasons, not
   tested:
   - Nighthawk's square lattice against kingston's heavy-hex;
   - FakeNighthawk's error data steering the layout and decomposition
     differently;
   - home's pair-block widths (spares 0-16) against the full 120 qubits
     here.

## 4. What this does not establish

- Nothing on ibm_kingston's Target (not available at work).
- The rate of the cutoff on other devices or circuit families.
- No pass trace was part of this run; since no inexact pair occurred,
  none was needed.

## 5. Files

| File | What it is |
|---|---|
| [`cutoff_psf_levels_2026-09-29.json`](../../data/2026-09-29/cutoff_psf_levels/cutoff_psf_levels_2026-09-29.json) | all 36 M and 150 R cells, environment |
| [`cutoff_run.txt`](../../data/2026-09-29/cutoff_psf_levels/cutoff_run.txt), [`cutoff_score.txt`](../../data/2026-09-29/cutoff_psf_levels/cutoff_score.txt) | logs |
| [`sandbox_env_cutoff.txt`](../../data/2026-09-29/cutoff_psf_levels/sandbox_env_cutoff.txt) | start time, script hash, commit, CPU |
| [`cutoff_psf_levels.py`](../../benchmarks/cutoff_psf_levels.py), `cutoff-psf-levels-preregistration-2026-09-29.md` | the locked script and pre-registration |

---

<!-- ===== Addendum 250 (source: spare-qubit-cliff-addendum-250-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Workplace pre-registration (sandbox) of an eigen-route fallback in the Rust core, candidate CORE_VERSION 2026-09-29.1; its section 1 holds the exploratory diagnosis of the 15 fallbacks of 247 A.

## Addendum 250 -- Pre-registration: an eigen-route fallback in the Rust core for the 15 remaining fallbacks (Addendum 247 A) -- candidate CORE_VERSION 2026-09-29.1 (2026-09-29)

**Status: pre-registration, locked at the Project save time of this
document**, before the scored runs. Exploratory probes (section 1) and
small dry runs (section 6) were made before locking; their outcomes are
disclosed. **All runs are in the workplace sandbox.**

## 1. What the exploratory probes found (not pre-registered)

Scripts are in the hand-off zip.

1. **Reproduced.** [`capture15.py`](../../data/2026-09-29/core_eigen_route/exploratory/capture15.py) captured the 15 blocks that the release
   core (2026-09-28.1) rejects in part E of the v4 run (all
   `PsfNumericError`, as in Addendum 247 A).
2. **Which check fails.** A debug build of the release source (identical
   except for `eprintln!` lines gated by an environment variable;
   [`diag15.py`](../../data/2026-09-29/core_eigen_route/exploratory/diag15.py)) shows that every one of the 15 fails the off-diagonal check
   of `try_decompose_with_tol`. The off-diagonal norm of D is 1.3e-6 to
   1.5e-4 against the limit 1e-6, for every `group_tol` candidate. None
   reaches the angle-sum check or the SU(2) extraction.
3. **Why.** In every case two singular values of Re(u_m) nearly tie (gaps
   2.3e-5 to 1.4e-4), or one is near 0 (lap 11,864: 5.1e-5). The
   magic-basis phases themselves stay well apart; for example lap 4,003 has
   phases -0.0212 and +0.0184, whose cosines differ by only 5.5e-5.
   Singular values are |cos(theta)|, which is flat near 0 and pi, so the
   SVD route is ill-conditioned there although the decomposition is not.
   This explains why Addendum 247 A found the blocks near several different
   loci, including a = b + |c|: the relevant ties are in |cos(theta)|, not
   in the Weyl coordinates.
4. **nalgebra's SVD specifically.**
   - numpy's (LAPACK) SVD of the same 15 matrices gives off-diagonal norms
     of 4e-16 to 4e-13.
   - nalgebra's `try_svd(..., 1e-12, 100)` returns left and right vectors
     that are not rotated consistently inside the near-tied pair (the
     error scales roughly as 1e-9 / gap). The group correction assumes one
     common rotation, so it cannot remove the mismatch.
   - A numpy prototype of an eigen route gives 2e-16 to 6e-16 on all 15
     ([`proto_eig.py`](../../data/2026-09-29/core_eigen_route/exploratory/proto_eig.py)). It takes O2 from an eigenbasis of the real symmetric
     pencil of M = u_m^T u_m, whose eigenvalues exp(2 i theta) stay
     separated, and O1 from u_m O2^T.
5. **Two candidate fixes, built and compared** ([`battery.py`](../../data/2026-09-29/core_eigen_route/exploratory/battery.py); 15 v4 blocks,
   the 33,000 part-E blocks captured on 2026-09-28, 20,000 random U(4) and
   4,000 near-degenerate blocks):
   - **F1**, tightening the SVD tolerance to 5 eps: accepted 0 of the 15,
     newly rejected 13 part-E and 2 near-degenerate blocks, and changed the
     output bits of about a third of the accepted blocks. **Rejected.**
   - **F2**, the eigen route tried only after every SVD candidate has
     failed: accepted 15 of 15 (raw residual median 1.7e-15, max 1.5e-10),
     plus the one part-E block the release core rejected in the 33,000.
     Bit-identical on all 57,000 blocks the release core accepts; no block
     newly rejected. **Chosen.**

## 2. The candidate (fixed design)

`src/lib.rs` of the release (normalized SHA-256 `bf3bf537...d234`) with
patch `lib_rs_eigen_route_2026-09-29.patch` (122 lines, SHA-256
`b9d4fd18...177d`). Result: normalized SHA-256
`5364630ec4e3648fe944b4103d76e8caa440e722c27ee6621b0ac51fdb26ad8f`. The
patch changes three things:

- `decompose_one`: after the SVD candidates fail (or the SVD itself fails),
  it tries the bases from `eigen_route_bases`. These are eight mixing angles
  (the existing `COMBINATION_ANGLES`), ordered by off-diagonal norm, each
  passed through the unchanged `try_decompose_with_tol` with `group_tol` 0.
- It adds the core changelog item 12.
- `CORE_VERSION` becomes "2026-09-29.1".

`psf_compile.py` stays 2026-09-28.1. The sandbox build uses rustc 1.95.0
and maturin, cp311.

## 3. Pre-registered predictions

**T0 (before the runs):** `cargo test --release --lib` 9 passed (done at
build time: 9 passed). pytest on the six release test files with the
candidate core: 83 passed. If T0 fails, nothing is run.

**Part 1: [`core_eigen_route_check.py`](../../benchmarks/core_eigen_route_check.py), release against candidate on fresh
sets.**

The sets:

- RAND: 50,000 Haar U(4), new seeds.
- NEAR: 5,000 near-degenerate perturbations of CNOT, SWAP, iSWAP and the
  identity, eps 1e-4 to 1e-8.
- TIE: 10,000 blocks with two near-tied |cos| built from Haar SO(4)
  factors.
- EFRESH: the blocks of 30,000 fresh part-E training circuits, about
  330,000, seed 303.
- V4: the 15 blocks.

The harness gate C0 requires both core versions and the set sizes.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| E1 | nothing changes where the release core succeeds | returned values bit-identical on every block the release core accepts (RAND, NEAR, TIE, EFRESH) | any differs |
| E2 | no new rejections | 0 blocks newly rejected (all sets) | any |
| E3 | the candidate accepts the fresh training blocks | every EFRESH block accepted, newly accepted ones with raw residual <= 1e-8 | any rejected, or a raw residual > 1e-6 |
| E4 | the release core still fails on fresh blocks at about the v4 rate | 1 to 100 EFRESH blocks rejected (v4: 15 in about 330,000) | 0, or more than 1,000 |
| E5 | (control, known) the candidate accepts the 15 V4 blocks | 15 of 15 | fewer |

**Part 2: [`long_loop_100k_v5.py`](../../benchmarks/long_loop_100k_v5.py), the release gate run with the
candidate.**

This is the home v4 run (laps, seeds, floor, checks and flags unchanged),
compared with the v4 CSVs in `data/`.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| Q1 | the training loop's core fallbacks disappear | E fallbacks = 0 (v4: 15) | >= 3 |
| Q2 | no repair path is needed | `exact_rebuilt` + `psf_rerouted` + `best_effort` = 0 (v4: 4) | >= 3 |
| Q3 | the cliff parts are unchanged | F + C fallbacks 0, and C drift at laps 1,000 / 10,000 / 20,000 within 1% of v4 | any fallback, or any > 10% off |
| Q4 | the training outputs are unchanged elsewhere | every E loss check <= 1e-13, and at every checked lap that was not a v4 fallback lap the value is within 1e-15 of v4's (rounding across machines) | any check > 1e-13, or any such difference > 1e-13 |
| Q5 | nothing else breaks | L1, L2, L5 clear | any trips |

Between the bounds: ambiguous. Timing (L3, L6, medians) is printed, not
scored.

## 4. What this can and cannot establish

It tests whether the eigen route removes the core's remaining fallbacks at
full scale without changing any output the release core already produces.
It does not test Python 3.12 or the pod build (home would rebuild from the
patch). It does not explain why nalgebra's SVD is less accurate than
LAPACK's for near-tied singular values, and it does not test other
circuit families at 100,000-compile scale. Adoption is home's decision.

## 5. Files, integrity, run commands

- [`core_eigen_route_check.py`](../../benchmarks/core_eigen_route_check.py): 9,686 bytes, normalized SHA-256
  `113df1c6e1855c279ad7f00ac6eef570413ad2991c7320ad2e592859190191f8`.
- [`long_loop_100k_v5.py`](../../benchmarks/long_loop_100k_v5.py): 16,439 bytes, normalized SHA-256
  `12b4c1c122fcaa758ec3cc0b2c6e468a2e06655cbfd4fb61cad0e04d46a0df56`.
- The scripts run from a clean clone at `f4b4a6c`, in `benchmarks/`. The
  cores are unpacked sandbox wheels, put first on the path:
  `<release_core_dir>` (2026-09-28.1) and `<candidate_core_dir>`
  (2026-09-29.1).

```
PYTHONPATH=<candidate_core_dir> python -m pytest -q <the six release test files>
python -u benchmarks/core_eigen_route_check.py collect <release_core_dir>   release   <repo>
python -u benchmarks/core_eigen_route_check.py collect <candidate_core_dir> candidate <repo>
python -u benchmarks/core_eigen_route_check.py score
PYTHONPATH=<candidate_core_dir> python -u benchmarks/long_loop_100k_v5.py > long_loop_100k_v5.txt 2>&1
```

Part 1 takes a few minutes. Part 2 takes about an hour; it runs in
parallel with Part 1, since timing is not scored.

## 6. Dry runs before locking (outcome disclosed)

- **Part 1 at reduced sizes (RAND 300, NEAR 100, TIE 300, EFRESH 2,200
  blocks from 200 circuits):**
  - identical outputs on every release-accepted block; no new rejections;
    candidate 15 of 15 on V4;
  - the release core rejected 0 of the 2,200 EFRESH blocks, as expected at
    that scale;
  - two design changes followed. The TIE set did not stress the release
    core at either tie range tried (0.005-0.05, then 0.0005-0.005: 0
    rejections of 300), so E4 was moved from TIE to the new EFRESH set.
    TIE was kept with the original range as an identity set.
- **Part 2 smoke run (1/100 scale):** V0 passed; Q1 and Q2 0; Q5 clear.
  In the smoke run, 3 of 30 E checks differed from v4's values by one unit
  in the last place (1.11e-16 against 0.000e+00): rounding across machines.
  So Q4 was changed from "identical as printed" to "within 1e-15".

---

<!-- ===== Addendum 251 (source: spare-qubit-cliff-addendum-251-2026-09-29.md) ===== -->

> **Note added when merging:** All predictions confirmed: bit-identical to the release core on 394,988 blocks where that core succeeds, no new rejections, 0 fallbacks and 0 repairs over 100,000 compiles (v5, sandbox). This record does not adopt the candidate; the release stays CORE_VERSION 2026-09-28.1.

## Addendum 251 -- The eigen-route fallback (candidate CORE_VERSION 2026-09-29.1) removes every remaining core fallback without changing any output the release core produces: bit-identical on 394,988 blocks, the 27 newly accepted blocks within raw residual 1.5e-10 (12 of them within 2.3e-15), 100,000-compile run with 0 fallbacks and 0 repairs (E1-E5, Q1-Q5 confirmed) (2026-09-29)

**Scored against:** `core-eigen-route-preregistration-2026-09-29.md`,
locked in the Project before the scored runs (scripts
[`core_eigen_route_check.py`](../../benchmarks/core_eigen_route_check.py) and [`long_loop_100k_v5.py`](../../benchmarks/long_loop_100k_v5.py), and the patch,
saved at the same time). Thresholds applied exactly as written.

**Run:** workplace sandbox: Intel(R) Xeon(R) Processor @ 2.80GHz, 2 CPUs,
`Linux-6.18.44-fc-v37-x86_64-with-glibc2.39`, Python 3.11.15, Qiskit 2.5.2,
numpy 2.4.4.

- **Code:** clean clone at `f4b4a6c` (`psf_compile.py` 2026-09-28.1,
  `3616efc8...`, unchanged). Release core 2026-09-28.1 (sandbox build of
  `bf3bf537...`). Candidate core 2026-09-29.1 (sandbox build, rustc
  1.95.0, of `5364630e...` = release + `lib_rs_eigen_route_2026-09-29.patch`).
- **Script hashes** were checked before the runs ([`sandbox_env_eigen.txt`](../../data/2026-09-29/core_eigen_route/sandbox_env_eigen.txt)):
  `113df1c6...` and `12b4c1c1...`.
- **Timeline (UTC):** start 01:02:21. Part 1 ran 01:02-01:05 and Part 2
  01:02-02:07, in parallel. **Times are sandbox times** and are not scored.
- **Amendments after the lock:** none.

## 1. Scoring

**T0:** `cargo test --release --lib` 9 passed (candidate, at build time);
pytest on the six release test files with the candidate core: 83 passed.

| ID | Prediction | Verdict | Numbers |
|---|---|---|---|
| C0 | versions and set sizes | passed | 2026-09-28.1 / 2026-09-29.1; RAND 50,000, NEAR 5,000, TIE 10,000, EFRESH 330,000, V4 15 |
| E1 | bit-identical where the release core succeeds | **CONFIRMED** | 50,000 + 4,999 + 10,000 + 329,989 = 394,988 of 394,988 |
| E2 | no new rejections | **CONFIRMED** | 0 |
| E3 | candidate accepts all fresh training blocks | **CONFIRMED** | 330,000 of 330,000; the 11 newly accepted have raw residual <= 2.3e-15 |
| E4 | release still fails at about the v4 rate | **CONFIRMED** | 11 of 330,000 (v4: 15 in about 330,000) |
| E5 | (control) the 15 V4 blocks | CONFIRMED | 15 of 15, worst raw residual 1.5e-10 |
| Q1 | no E fallbacks | **CONFIRMED** | 0 (v4: 15) |
| Q2 | no repair path | **CONFIRMED** | exact_rebuilt 0 + psf_rerouted 0 + best_effort 0 (v4: 4 + 0 + 0) |
| Q3 | cliff parts unchanged | **CONFIRMED** | F + C fallbacks 0; C drift 6.0565e-12 / 6.0810e-11 / 1.2191e-10 against v4's 6.056e-12 / 6.081e-11 / 1.219e-10 (<= 0.01%) |
| Q4 | training outputs unchanged elsewhere | **CONFIRMED** | worst loss check 3.3e-16; 2,999 checked laps compared, 2,905 identical as printed, largest difference 2.2e-16 |
| Q5 | nothing else breaks | **CONFIRMED** | L1, L2, L5 clear |

Also: in NEAR, the release core rejected 1 of 5,000 blocks; the candidate
accepted it with a raw residual of 2.1e-15. Not predicted, and consistent
with E2 and E3.

## 2. Other numbers (reported without prediction)

- **Whole-run `GUARD_STATS` (v5):** checked 29, `zsx_rejected` 7, inexact
  11, all repair counters 0. v4's were checked 44 and inexact 22, with
  `exact_rebuilt` 4. The difference, 15 checks and 11 inexact, is exactly
  the 15 fallback blocks that no longer reach the Python checks.
- **Timing flags** L3 and L6 were clear in all three parts. Medians: F
  36.6 ms, C 37.3 ms, E 10.2 ms (sandbox, with Part 1 running alongside for
  the first three minutes).
- **Raw residuals of the release core on fresh sets** (maximum over
  accepted blocks) are unchanged by the candidate, which is bit-identical
  there:
  - RAND 5.9e-9, NEAR 1.8e-7, TIE 8.9e-7, EFRESH 7.9e-7;
  - the Python polish brings these to <= 1e-14 (unchanged behaviour).

## 3. Reading

1. **Cause (exploratory, section 1 of the pre-registration):**
   - The SVD route works on Re(u_m), whose singular values are
     |cos(theta)|. They nearly tie when two well-separated phases have
     nearly equal cosines.
   - nalgebra's SVD then returns left and right vectors that are not
     rotated consistently, and the group correction cannot repair that.
   - All 15 v4 fallbacks, and the 11 fresh ones here, are of this kind.
   - It also explains the loci listed in Addendum 247 A, including
     a = b + |c|. The tie is in |cos(theta)|, not in the Weyl coordinates.
2. **Fix:** O2 is taken from the real symmetric pencil of M = u_m^T u_m,
   whose eigenvalues exp(2 i theta) stay apart there. This route is tried
   only when every SVD candidate has failed.
3. **Effect at full scale:**
   - The last core fallbacks and Python repairs disappear: 0 in 100,000
     compiles.
   - The C drift is the v4 value to four digits, and the training outputs
     agree to within one unit in the last place.
   - Across 394,988 fresh blocks, the candidate returns exactly the release
     core's values wherever the release core succeeds. Adopting it cannot
     change any output that does not already involve a fallback.
4. **For home's decision:** the patch is small (one route added to
   `decompose_one` plus a helper, changelog item 12, `CORE_VERSION`
   2026-09-29.1).
   - The hand-off contains the full new `lib.rs`, the patch, and a
     hash-checked apply script.
   - Home would rebuild (`maturin develop --release`) and could repeat v5
     at home, as v4 was, before release.

## 4. What this does not establish

- Python 3.12, the pod or home builds (home would rebuild from the patch).
- Why nalgebra's SVD is less accurate than LAPACK's for near-tied singular
  values. The SVD route itself was not changed. F1 (a tighter tolerance)
  made acceptance worse.
- Other circuit families at 100,000-compile scale.

## 5. Files

| File | What it is |
|---|---|
| [`sandbox_outputs/eigen_collect_release.txt`](../../data/2026-09-29/core_eigen_route/eigen_collect_release.txt), [`eigen_collect_candidate.txt`](../../data/2026-09-29/core_eigen_route/eigen_collect_candidate.txt), [`eigen_score.txt`](../../data/2026-09-29/core_eigen_route/eigen_score.txt) | Part 1 logs (the per-block npz files, 54 MB each, and the 84 MB EFRESH cache are regenerable from the fixed seeds and not included) |
| [`sandbox_outputs/long_loop_100k_v5.txt`](../../data/2026-09-29/core_eigen_route/long_loop_100k_v5.txt), `long_loop_{F,C,E}_2026-09-29_v5.csv` | Part 2 log and per-lap data |
| [`sandbox_outputs/t0_pytest.txt`](../../data/2026-09-29/core_eigen_route/t0_pytest.txt), [`sandbox_env_eigen.txt`](../../data/2026-09-29/core_eigen_route/sandbox_env_eigen.txt) | T0 and environment |
| `exploratory/` | [`capture15.py`](../../data/2026-09-29/core_eigen_route/exploratory/capture15.py), `blocks15.npz`, [`diag15.py`](../../data/2026-09-29/core_eigen_route/exploratory/diag15.py) + [`diag15.txt`](../../data/2026-09-29/core_eigen_route/exploratory/diag15.txt) (with `lib_rs_debug_eprintln.patch`), [`proto_eig.py`](../../data/2026-09-29/core_eigen_route/exploratory/proto_eig.py) + [`proto_eig.txt`](../../data/2026-09-29/core_eigen_route/exploratory/proto_eig.txt), [`battery.py`](../../data/2026-09-29/core_eigen_route/exploratory/battery.py) + [`battery_summary.txt`](../../data/2026-09-29/core_eigen_route/exploratory/battery_summary.txt) |
| `candidate_core/` | `lib.rs` (2026-09-29.1), `lib_rs_eigen_route_2026-09-29.patch`, [`apply_update_2026-09-29.py`](../../patches/core_eigen_route_2026-09-29.1/apply_update_2026-09-29.py) |
| [`core_eigen_route_check.py`](../../benchmarks/core_eigen_route_check.py), [`long_loop_100k_v5.py`](../../benchmarks/long_loop_100k_v5.py), `core-eigen-route-preregistration-2026-09-29.md` | the locked scripts and pre-registration |

---

<!-- ===== Addendum 252 (source: spare-qubit-cliff-addendum-252-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Workplace pre-registration (sandbox, FakeKingston 156 qubits): the cliff on a fully occupied heavy-hex device (pairs and 3-qubit paths), with the candidate layout 2026-09-29.c1 as an extra arm.

## Addendum 252 -- Pre-registration: the layout cliff on a fully occupied heavy-hex device (FakeKingston, 156 qubits), and a candidate fix for PSF-Zero's layout search on 3-qubit paths (psf_smart_layout 2026-09-29.c1) (2026-09-29)

**Status: pre-registration, locked at the Project save time of this
document**, before the scored run. Exploratory probes and two harness dry
runs on a 27-qubit device were made before locking; their outcomes are
disclosed in sections 1 and 6. **All runs are in the workplace sandbox. No
IBM account, no network access to IBM and no QPU are used** (the device is a
fake-provider snapshot).

## 1. Why, and what the probes found (not pre-registered)

**The open question (Addendum 246, home).**

- On ibm_kingston's live Target, disjoint pair blocks fill only 2M = 128 of
  the 156 qubits, where M = 64 is the maximum matching.
- So 28 physical qubits were always free, the layout search was never under
  pressure, and R2 and R3 came out ambiguous.
- Whether a *saturated* heavy-hex device shows the cliff is open. Testing it
  needs a circuit family that can occupy all 156 qubits.

**Stage 0 facts of FakeKingston** ([`probe_stage0.py`](../../data/2026-09-29/full_heavyhex_cliff/exploratory/probe_stage0.py)):

- 156 qubits, 176 edges, degree min/median/max 1/2/3, bipartite, connected.
- Maximum matching 64 pairs, leaving 28 qubits unmatched. This agrees with
  home's Stage 0 of the live Target (156 qubits, degree 1/2/3, matching 64).
  Whether the two coupling maps are identical edge by edge is not checked
  here.
- **A cover of all 156 qubits by 2- and 3-qubit paths exists:** 36 pairs
  plus 28 paths of 3 qubits (a {P2,P3}-factor). The construction attaches
  each unmatched qubit to a distinct matched edge next to it, via a bipartite
  matching.

**Harness dry run 1** (27-qubit FakeAuckland, heavy-hex; section 6). It
found two things on that small device.

1. **Qiskit L3 hits a cliff on the pairs-plus-triples family (T) at full
   occupation.**
   - Timing: about 8.5 s at spare 0, against 0.10-0.12 s at spare 2 and 4.
   - It also did not find a swap-free layout: 54 two-qubit gates against 51
     without SWAPs.
   - The family S (pairs plus qubits with single-qubit gates only) showed no
     cliff.
2. **A bug in PSF-Zero's layout search.**
   - For T, `smart_vf2_layout` returned "infeasible" at once (0 orderings
     tried, 0.001 s). `compile_for_hardware` then fell back to Qiskit's
     level-1 layout, which inserted SWAPs: 63 two-qubit gates against 51
     without SWAPs, and against Qiskit L3's 54.
   - Cause: the feasibility check `_has_feasible_matching(coupling_map,
     len(interaction_pairs))` requires a physical matching as large as the
     *number of interaction edges*.
   - That is right only for a matching-shaped interaction graph, and that
     case is already handled earlier by the matching shortcut (Addendum 192).
   - A 3-qubit path has 2 edges but needs only 1 disjoint physical edge. So
     any circuit with 3-qubit paths and more edges than the device's
     matching is wrongly declared impossible.

**The candidate `psf_smart_layout.py` 2026-09-29.c1** (built after dry run
1; [`probe_pf_auckland.py`](../../data/2026-09-29/full_heavyhex_cliff/exploratory/probe_pf_auckland.py)).

- **(a) Corrected feasibility check.** It requires a physical matching as
  large as the maximum matching of the interaction graph, which is a correct
  necessary condition.
  - With (a) alone, on FakeAuckland, family T at spare 0: the VF2 stages
    tried 9 orderings in 0.23 s and found nothing.
  - At spare 2 they found a layout in phase 1.
- **(b) Stage 0b, a short-path shortcut.** When every component of the
  interaction graph is a path of 2 or 3 qubits (at least one of 3), and no
  edge weights are given, the layout is built as in Stage 0 above: a maximum
  matching, plus one unmatched neighbour per 3-qubit path.
  - It is a sufficient construction, not a complete search. If it fails, the
    ordinary VF2 stages run, now with the corrected check.
  - With (b), on FakeAuckland, family T is placed at once (`path_direct`) at
    spare 0 and 2.
- **Unchanged:**
  - `psf_compile.py` (2026-09-28.1) and the Rust core;
  - the matching shortcut;
  - VF2 stages 1 and 2;
  - the weighted path (`edge_weights`), where the shortcut is not used.
- **Tests:** the six release test files (83) and a new
  [`test_short_path_layout.py`](../../patches/psf_smart_layout_c1_2026-09-29/test_short_path_layout.py) (10) pass with the candidate: 93 passed.

## 2. Design ([`full_heavyhex_cliff.py`](../../benchmarks/full_heavyhex_cliff.py))

**Device.** FakeKingston (qiskit-ibm-runtime 0.50.0 fake provider), native
gates cz, rz, sx, x, id. Stage 0 prints the qubit and edge counts, the
degree range, the maximum matching M, and the {P2,P3}-factor.

**Circuit families.** `pair24` is `loop_endurance.add_pair24`. Random angles
are drawn with numpy seed = family offset (T 0, S 100,000, B 200,000)
+ 1000 x spare + input. There are 3 inputs per cell.

- **T (pairs + triples):**
  - k3 = N - 2M = 28 triples and k2 = M - k3 = 36 pairs.
  - A triple (a, b, c) is pair24 on (a, b) followed by pair24 on (b, c), so
    its interaction graph is a 3-qubit path.
  - At spare 0 this uses all 156 qubits, and the only swap-free layouts are
    {P2,P3}-factors of the device.
- **S (pairs + single-qubit-only qubits):**
  - M = 64 pairs, plus 28 qubits that carry two rz-ry-rz layers each.
  - All 156 logical qubits exist, but the interaction graph is still a
    matching.
- **B (bridge):** M = 64 pairs and nothing else (n = 128). This is the
  Addendum 245 family at its spare 0, on this device. Spare 0 only.
- **Spare s (T and S):** removes s/2 pair blocks, so n = 156 - s, with s in
  {0, 2, 8, 16}.

**Arms.** Each compile runs in its own spawned child process. The timer
runs inside the child, around the compile call only. A child is killed at
180 s, which is recorded as DNF.

- **Q3:** `transpile(qc, target=target, optimization_level=3,
  seed_transpiler=0)`.
- **P:** `compile_for_hardware(qc, coupling_map=target's,
  basis_gates=native, entangling_basis="cx", layout_search=True,
  on_unsupported="raise", seed_transpiler=0)`, with the release
  [`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) (`LAYOUT_VERSION` 2026-09-26.m1).
- **PX:** the same call, with the candidate `psf_smart_layout.py`
  (2026-09-29.c1) first on `sys.path`.

**Recorded per compile:**

- time, and the routed two-qubit count;
- the swap-free count: 3 per pair and 6 per triple;
- an exact per-block check, when no two-qubit gate joins different blocks.
  Blocks have 1, 2 or 3 qubits, and the check is the phase-aligned Frobenius
  distance;
- a digest of the output (gate names, qubit indices, parameters);
- for P and PX, the layout module's version and hash. Also, outside the
  timed call, what that module's `smart_vf2_layout` finds for the same
  interaction graph with `compile_for_hardware`'s limits (50,000 per
  attempt, 2 s, 2,000,000 fallback).

## 3. Pre-registered predictions

**G0 (gate, before any compile):** Stage 0 finds a {P2,P3}-factor with 28
triples and 36 pairs. If not, nothing is run.

**C0 (harness):** no compile ends in an error, every P and PX compile
finishes, and the P rows load layout 2026-09-26.m1 while the PX rows load
2026-09-29.c1. If C0 fails, the predictions are read with that caveat.

"Median" is over the 3 inputs, with DNF counted as 180 s. "Within 1 s" is
the compile call's time.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| H1 | Qiskit L3 has a cliff on T at full occupation | Q3 median spare 0 / Q3 median spare 16 >= 10 | < 3 |
| H2 | the release PSF-Zero inserts SWAPs on T at spare 0 (the bug) | P two-qubit count above the swap-free count in 3 of 3 | 0 of 3 |
| H3 | the candidate places T at spare 0 swap-free and fast | PX swap-free in 3 of 3 **and** within 1 s in 3 of 3 | either in 1 or fewer |
| H4 | the cliff gap on T | Q3 median / PX median at spare 0 >= 10 | < 3 |
| H5 | no cliff on S (occupying all qubits with a matching-shaped interaction graph is not enough) | Q3 median spare 0 / spare 16 < 3 | >= 10 |
| H6 | PSF-Zero meets 1 s on S at spare 0 | P and PX both within 1 s in 3 of 3 | either in 1 or fewer |
| H7 | the candidate changes nothing on matching-shaped circuits | PX output digest equal to P's in all 15 B and S compiles | any differs |
| H8 | the candidate never pays in two-qubit gates | PX never above Q3 in any compile where both finish | above in any |
| H9 | PSF-Zero's outputs are exact | P and PX per-block distance <= 1e-12 wherever the check applies | > 1e-12 anywhere |

Between the bounds: ambiguous.

**Expectations stated before running, and how the dry run informed them.**

- **H1:** the 27-qubit dry run had a ratio of about 77. Whether the 156-qubit
  device behaves the same is the point of the run. Qiskit's search limits do
  not scale with the device, so the cliff could be longer, shorter, or
  absent.
- **H2:** follows from the code as read. The question is only whether
  Qiskit's level-1 fallback happens to find a swap-free layout on this
  device; on FakeAuckland it did not.
- **H3 and H4:** expected from the construction and from the dry run.
- **H5:** expected, because Qiskit's layout step places qubits without
  two-qubit gates separately from the search.
- **H7:** expected, because Stage 0b needs a 3-qubit path and the
  feasibility check is reached only after the matching shortcut.

**Reported without prediction:**

- every compile time and the per-cell medians, and the B ratio Q3/P;
- P's two-qubit counts against Q3's;
- Qiskit L3's per-block distances;
- the layout diagnostics.

## 4. What this can and cannot establish

It tests whether a saturated heavy-hex device shows the cliff for one
circuit family that can occupy it, and whether the candidate layout search
removes PSF-Zero's SWAPs there at no cost elsewhere.

It does not establish:

- the live ibm_kingston Target (home can run the same script with
  `--target-pkl`);
- other circuit families that fill a heavy-hex device, such as longer paths
  or trees, which the shortcut does not cover;
- execution on hardware;
- the weighted layout path.

Times are sandbox times and are not compared with home or the pod.
Adopting the candidate is home's decision.

## 5. Files, integrity, run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| [`full_heavyhex_cliff.py`](../../benchmarks/full_heavyhex_cliff.py) (in `benchmarks/`) | 21,637 | `1264fadacab44403d0dbc40a5cab8c0c60281b013f6f2c33e6ba91181eb502e4` |
| candidate `psf_smart_layout.py` (2026-09-29.c1) | 27,664 | `e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa` |
| candidate [`test_short_path_layout.py`](../../patches/psf_smart_layout_c1_2026-09-29/test_short_path_layout.py) | 3,975 | `3ce3bdf1dca71e1e6af7265c76b0ed2f11c40aec9ceccf33b56c6fd73abf84a1` |
| release [`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) (base) | 22,450 | `a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875` |
| `psf_smart_layout_c1_2026-09-29.patch` (git diff against `f4b4a6c`, both files) | 11,028 | raw SHA-256 `e42beb1aa9f4aa1c40a2705baca1bd780426c95b892ea47d16302b620041733f` |

Run from a clean clone at `f4b4a6c`, with the script in `benchmarks/`, the
release core 2026-09-28.1 (sandbox build of `bf3bf537...`) first on the
path, and the candidate module in `<candidate_dir>`:

```
PYTHONPATH=<release_core_dir> python -u benchmarks/full_heavyhex_cliff.py run --px-dir <candidate_dir> > full_heavyhex_cliff.txt 2>&1
PYTHONPATH=<release_core_dir> python -u benchmarks/full_heavyhex_cliff.py score > full_heavyhex_score.txt 2>&1
```

## 6. Dry runs before locking (outcome disclosed)

Both dry runs used FakeAuckland (27 qubits, maximum matching 10, 7
unmatched, so T = 7 triples + 3 pairs), spares 0, 2, 4, and a 60 s cap.

**Dry run 1** used the harness with arms Q3 and P only. Its findings are in
section 1:

- Qiskit L3 on T at spare 0: 8.5 s median, 54 two-qubit gates.
- P on T at spare 0: 0.13 s, 63 two-qubit gates, layout search "infeasible".
- Everything else was swap-free and exact.

**Dry run 2** used the final harness with Q3, P and PX. With the 27-qubit
device and spare 4 in place of 16, every prediction H1-H9 scored CONFIRMED:

- H1: 76.9.
- PX on T at spare 0: 51 two-qubit gates (swap-free) in 0.106 s.
- H7: 12 of 12 digests equal.
- H9: worst 1.5e-14.

Changes to the design after dry run 1:

- the PX arm and the candidate module were added;
- the predictions were rewritten around them;
- the layout diagnostic was changed to use `compile_for_hardware`'s limits.

No FakeKingston compile was run before locking.

---

<!-- ===== Addendum 253 (source: spare-qubit-cliff-addendum-253-2026-09-29.md) ===== -->

> **Note added when merging:** H1-H9 confirmed: a truly full heavy-hex device shows the cliff (Qiskit L3 24-29 s); the release layout search wrongly rejected 3-qubit paths and fell back, adding 45-51 two-qubit gates; the candidate places them swap-free in 0.21 s and leaves pair-only outputs unchanged. Not adopted here.

## Addendum 253 -- A fully occupied heavy-hex device shows the cliff: on FakeKingston (156 qubits), with pairs and 3-qubit paths filling every qubit, Qiskit L3 takes 24-29 s and adds 18-39 two-qubit gates up to spare 8; the release PSF-Zero falls back and adds 45-51 (a layout-search bug); the candidate psf_smart_layout 2026-09-29.c1 places them swap-free in 0.21 s (H1-H9 confirmed) (2026-09-29)

**Scored against:** `full-heavyhex-cliff-preregistration-2026-09-29.md`,
locked in the Project before the scored run. The script
[`full_heavyhex_cliff.py`](../../benchmarks/full_heavyhex_cliff.py) and the patch `psf_smart_layout_c1_2026-09-29.patch`
were saved at the same time. Thresholds were applied exactly as written.

**Run:** workplace sandbox.

- **Machine:** Intel(R) Xeon(R) Processor @ 2.80GHz, 2 CPUs,
  `Linux-6.18.44-fc-v37-x86_64-with-glibc2.39`. Python 3.11.15, Qiskit
  2.5.2, qiskit-ibm-runtime 0.50.0 (fake provider only; no account, no
  network access to IBM).
- **Code:** clean clone at `f4b4a6c`, with `psf_compile.py` 2026-09-28.1
  (`3616efc8...`) and `CORE_VERSION` 2026-09-28.1 (sandbox build of
  `bf3bf537...`).
- **Layout modules:** release 2026-09-26.m1 (`a639efde...`) and candidate
  2026-09-29.c1 (`e25952a3...`), both printed per compile.
- **Hashes:** the script's hash `1264fada...` was checked before the run
  ([`sandbox_env_hh.txt`](../../data/2026-09-29/full_heavyhex_cliff/sandbox_env_hh.txt)) and printed by it.
- **Timeline:** 02:28:01-02:34 UTC. **Times are sandbox times.**
- **Amendments after the lock:** none.

## 1. Scoring

**G0:** passed. The {P2,P3}-factor was found, with 28 triples and 36 pairs.
FakeKingston has 156 qubits, 176 edges, degree 1/2/3, and a maximum
matching of 64.

**C0:** passed. All 81 compiles finished with no errors, and the P and PX
rows loaded the expected layout modules.

| ID | Prediction | Verdict | Numbers |
|---|---|---|---|
| H1 | Qiskit L3 has a cliff on T | **CONFIRMED** | median spare 0 / spare 16 = 24.943 / 0.281 s = 88.7 |
| H2 | release PSF-Zero inserts SWAPs on T at spare 0 | **CONFIRMED** | 327 two-qubit gates in 3 of 3 (swap-free: 276) |
| H3 | candidate swap-free and within 1 s on T at spare 0 | **CONFIRMED** | 276 in 3 of 3; 0.214-0.219 s |
| H4 | cliff gap Q3 / PX on T at spare 0 | **CONFIRMED** | 24.943 / 0.214 s = 116.4 |
| H5 | no cliff on S | **CONFIRMED** | 0.090 / 0.100 s = 0.9 |
| H6 | P and PX within 1 s on S at spare 0 | **CONFIRMED** | 3 of 3 each (0.052-0.060 s) |
| H7 | candidate changes nothing on matching-shaped circuits | **CONFIRMED** | output digest equal in 15 of 15 (B and S) |
| H8 | PX never above Qiskit L3 in two-qubit gates | **CONFIRMED** | 0 of 27 |
| H9 | PSF-Zero exact per block | **CONFIRMED** | 45 of 54 P and PX compiles applicable, worst 1.7e-14 |

## 2. Numbers (medians of 3 inputs; sandbox; compile call only)

| family | spare | n | Q3 s | Q3 2q | P s | P 2q | PX s | PX 2q | swap-free 2q |
|---|---|---|---|---|---|---|---|---|---|
| B | 0 | 128 | 0.346 | 192 | 0.052 | 192 | 0.057 | 192 | 192 |
| T | 0 | 156 | **24.943** | 315 | 0.276 | 327 | 0.214 | **276** | 276 |
| T | 2 | 154 | **24.187** | 309 | 0.275 | 324 | 0.215 | **273** | 273 |
| T | 8 | 148 | **28.372** | 282 | 0.295 | 309 | 0.228 | **264** | 264 |
| T | 16 | 140 | 0.281 | 252 | 0.210 | 252 | 0.209 | 252 | 252 |
| S | 0 | 156 | 0.090 | 192 | 0.054 | 192 | 0.056 | 192 | 192 |
| S | 2 | 154 | 0.092 | 189 | 0.060 | 189 | 0.054 | 189 | 189 |
| S | 8 | 148 | 0.098 | 180 | 0.055 | 180 | 0.054 | 180 | 180 |
| S | 16 | 140 | 0.100 | 168 | 0.052 | 168 | 0.052 | 168 | 168 |

- **Spread:** within each cell, the three inputs gave the same two-qubit
  count, and times within 5% (Q3 on T: 24.2-29.2 s over the three cliff
  cells).
- **Qiskit L3 on T, spare 0 / 2 / 8:** 39 / 36 / 18 two-qubit gates above
  the swap-free count, which equals 13 / 12 / 6 SWAPs at 3 each.
- **Release PSF-Zero on T, spare 0 / 2 / 8:** 51 / 51 / 45 above. That is
  also above Qiskit L3 in all 9 of those compiles (reported without
  prediction).
- **Release PSF-Zero on T, spare 16:** swap-free. Its own search still
  declared the layout infeasible there, but Qiskit's level-1 fallback found
  a swap-free layout.
- **Layout diagnostics:**
  - The release search returned "infeasible" with 0 orderings tried at every
    T spare.
  - The candidate used `path_direct` at every T spare, and
    `matching_direct` for B and S. This is the same as the release for B
    and S.
- **Qiskit L3 per-block distance** (where no SWAP joins blocks, 18 of 27):
  worst 3.4e-13, and none above 1e-12.
- **B (the Addendum 245 family on this snapshot):** Q3 / P = 6.6. At home
  on the live ibm_kingston Target it was 5.1, but the timing machines differ
  and are not compared.

The medians, two-qubit counts and worst distances above were recomputed
from the CSV and match the log and the score file.

**Note on the pre-registration's section 1 (exploratory, not scored).** The
"0.23 s, 9 orderings, nothing found" figure for the corrected check alone
used the module's default limits (fallback 300,000). The probe was rerun
after the scored run with `compile_for_hardware`'s limits (fallback
2,000,000; [`probe_pf_auckland_r2.txt`](../../data/2026-09-29/full_heavyhex_cliff/exploratory/probe_pf_auckland_r2.txt)). It still tried 9 orderings and found
nothing at spare 0, now in 1.17 s. At spare 2 it found a layout in phase 1,
after 0.004 s. The conclusion is unchanged: on the 27-qubit T layout, the
corrected check alone is not enough, and the short-path shortcut is what
places it.

## 3. Reading

1. **The cliff exists on heavy-hex once the device is actually saturated.**
   - Addendum 246 could not reach saturation with pairs alone. Filling all
     156 qubits with pairs and 3-qubit paths makes Qiskit L3 take about
     25 s, against 0.28 s with room to spare.
   - The cliff reaches further than on FakeNighthawk: it is still there at
     spare 8 (28 s) and gone at spare 16. On FakeNighthawk it was gone by
     spare 4.
   - Qiskit L3 is not only slow there. It also misses the swap-free layout
     that exists, and adds up to 39 two-qubit gates.
2. **The earlier statement that the cliff does not arise on heavy-hex
   (Addenda 39-40, 135-136, and section 1 of 178) needs narrowing.** It holds for circuits made of disjoint pairs, whose layouts
   heavy-hex can never saturate. It does not hold for circuits whose
   interaction graph can cover the device.
   - Family S shows that simply having 156 logical qubits is not enough:
     with single-qubit-only qubits in place of the triples there is no cliff
     (0.09 s).
3. **The release PSF-Zero is not immune on this family.**
   - The feasibility check in `psf_smart_layout.smart_vf2_layout` counts
     interaction edges instead of the disjoint edges a layout needs. So it
     rejects every circuit with 3-qubit paths whose edge count exceeds the
     device's matching. That was every T circuit here, even at spare 16.
   - `compile_for_hardware` then falls back to Qiskit's level-1 layout. It
     stays fast (0.28 s) but adds 45-51 two-qubit gates, more than Qiskit L3
     does.
   - The PSF-Zero claim "same circuits as Qiskit, much faster, at the cliff"
     therefore did not hold on this family with the release.
4. **The candidate removes the problem without touching anything else.**
   - It places every T circuit swap-free in about 0.21 s. That is 116 times
     faster than Qiskit L3 at spare 0, with 39 fewer two-qubit gates. The
     outputs are exact per block.
   - On every matching-shaped circuit (B and S), its output is identical to
     the release's.
   - The change is limited to `psf_smart_layout.py`: the corrected check and
     a new Stage 0b. `psf_compile.py` and the Rust core are unchanged.

## 4. What this does not establish

- **The live ibm_kingston Target.** FakeKingston agrees with home's Stage 0
  (156 qubits, degree 1/2/3, matching 64), but edge-by-edge identity and
  Qiskit's use of the live error data were not checked. Home can run
  `full_heavyhex_cliff.py run --px-dir <candidate_dir> --target-pkl <file>
  --tag home`.
- **Other families that fill heavy-hex:** longer paths, trees, or 3-qubit
  paths whose middle is not a matched edge's end. The candidate's shortcut
  covers only components of 2 or 3 qubits. Anything else goes through the
  VF2 stages, which, with only the corrected check, did not find the 27-qubit
  T layout in the dry run.
- **Exactness of the outputs with SWAPs** (Q3 and P on T up to spare 8).
  The per-block check does not apply there, and a whole-circuit check at
  148-156 qubits is not computable.
- **The weighted layout path** (`layout_edge_errors`), where the shortcut is
  not used.
- **Hardware execution.**

## 5. Files

| File | What it is |
|---|---|
| [`sandbox_outputs/full_heavyhex_cliff.txt`](../../data/2026-09-29/full_heavyhex_cliff/full_heavyhex_cliff.txt), [`full_heavyhex_score.txt`](../../data/2026-09-29/full_heavyhex_cliff/full_heavyhex_score.txt) | run log and scoring |
| [`sandbox_outputs/full_heavyhex_cliff_2026-09-29.csv`](../../data/2026-09-29/full_heavyhex_cliff/full_heavyhex_cliff_2026-09-29.csv), `.json` | all 81 compiles, with Stage 0 and the factor |
| [`sandbox_outputs/sandbox_env_hh.txt`](../../data/2026-09-29/full_heavyhex_cliff/sandbox_env_hh.txt) | start time, hashes, commit, CPU |
| `candidate_layout/` | `psf_smart_layout.py` (2026-09-29.c1), [`test_short_path_layout.py`](../../patches/psf_smart_layout_c1_2026-09-29/test_short_path_layout.py), `psf_smart_layout_c1_2026-09-29.patch`, and a hash-checked apply script |
| `exploratory/` | [`probe_stage0.py`](../../data/2026-09-29/full_heavyhex_cliff/exploratory/probe_stage0.py), [`probe_pf_auckland.py`](../../data/2026-09-29/full_heavyhex_cliff/exploratory/probe_pf_auckland.py) and output |
| `dry_runs/` | the two FakeAuckland dry runs (logs and scores) |
| [`full_heavyhex_cliff.py`](../../benchmarks/full_heavyhex_cliff.py), `full-heavyhex-cliff-preregistration-2026-09-29.md` | the locked script and pre-registration |

---

<!-- ===== Addendum 254 (source: spare-qubit-cliff-addendum-254-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Workplace pre-registration (sandbox CPU): the PennyLane compounding loop of Addenda 183-184 on the fully occupied FakeKingston, release stack and candidate stack against Qiskit L3, 20 laps.

## Addendum 254 -- Pre-registration: the timed PennyLane compounding loop (Addenda 183-184) on a fully occupied heavy-hex device (FakeKingston, 156 qubits), with the candidate stack of 2026-09-29 (core 2026-09-29.1 + psf_smart_layout 2026-09-29.c1), 20 laps (2026-09-29)

**Status: pre-registration, locked at the Project save time of this
document**, before the scored runs. Two harness dry runs on the 27-qubit
FakeAuckland were made before locking; they are disclosed in section 5.
**All runs are in the workplace sandbox, on CPU. No IBM account, no network
access to IBM and no QPU are used.**

## 1. Why

Two results earlier today set this up.

- **Pairs and 3-qubit paths filling FakeKingston make a cliff.** Qiskit L3
  took about 25 s per compile. The release PSF-Zero added SWAPs because of
  a layout-search bug. The candidate layout placed the circuits swap-free in
  0.21 s. That was one compile per circuit.
- **The candidate core** 2026-09-29.1 removes the remaining core fallbacks,
  and its outputs are bit-identical elsewhere.

Addendum 183-184's loop is what a user of PennyLane actually does: convert
a tape, compile, bring the result back as a tape, repeat. It showed on
FakeNighthawk that the cliff recurs on every lap. It also caught a per-lap
drift in PSF-Zero, which was fixed later.

This experiment asks, for the heavy-hex case and the combined candidate
stack:

- whether each lap meets a 1 s deadline;
- whether the lap stays swap-free and can be mapped back to the tape;
- whether the circuit's meaning, as PennyLane computes it, survives 20
  compounded laps.

## 2. Design ([`pl_heavyhex_chain.py`](../../benchmarks/pl_heavyhex_chain.py))

**Device and circuits.**

- Device: FakeKingston, with the Stage 0 gate of [`full_heavyhex_cliff.py`](../../benchmarks/full_heavyhex_cliff.py)
  (the {P2,P3}-factor must exist).
- Circuits: family T.
  - At spare 0: 28 triples and 36 pairs, 156 wires.
  - At spare 16: 28 triples and 28 pairs, 140 wires, the no-cliff control.
  - A pair block is 20 Haar-random `qml.QubitUnitary` on (a, b), as in
    Addendum 183.
  - A triple block is 10 on (a, b) followed by 10 on (b, c).
  - Seed: numpy 1000 x spare.

**The lap.**

1. `tape_to_qiskit`, then compile. The timer runs around the compile call
   only.
2. Map the result back to logical qubits (initial layout = final layout
   required).
3. `qiskit_to_tape` (two-qubit gates wrapped as unitaries, as in 183).
4. The new tape is the next lap's input.

**Meaning check.** Each block's `qml.matrix` of the current tape is compared
with the same block's matrix of the lap-0 tape, by phase-aligned Frobenius
distance. So the check is compounded over laps.

**Laps with SWAPs.** If the output joins different blocks or permutes
qubits, it cannot be mapped back block by block. That lap is recorded as
"swap", with no meaning check, and the next lap reuses the same input tape.

**Arms.** One process per arm, run one after another. Each arm runs 20 laps
at spare 0, then 20 at spare 16.

- **Q3:** `transpile(qc, target=target, optimization_level=3,
  seed_transpiler=0)`.
- **P (release):** `compile_for_hardware(..., entangling_basis="cx",
  layout_search=True, on_unsupported="raise", seed_transpiler=0)`, with
  core 2026-09-28.1 and `psf_smart_layout` 2026-09-26.m1.
- **PN (candidate stack):** the same call, with core 2026-09-29.1 (sandbox
  build of `5364630e...`) and `psf_smart_layout` 2026-09-29.c1
  (`e25952a3...`).
- `psf_compile.py` is 2026-09-28.1 in both P and PN.

## 3. Pre-registered predictions

**G0:** Stage 0 finds the {P2,P3}-factor.

**C0:** every arm completes 20 laps at both spares. The P rows report core
and layout (2026-09-28.1, 2026-09-26.m1), and the PN rows report
(2026-09-29.1, 2026-09-29.c1).

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| D1 | PN meets the deadline at full occupation | spare 0: PN within 1 s in 20 of 20 laps | 10 or fewer |
| D2 | the cliff recurs on every lap for Qiskit L3 | spare 0: Q3 within 1 s in 0 of 20 | 10 or more |
| D3 | PN stays swap-free and returns to PennyLane every lap | spare 0: PN two-qubit count = swap-free count (276) and mapped back in 20 of 20 | 10 or fewer |
| D4 | PN keeps the meaning over 20 compounded laps | worst block distance to lap 0, both spares, every checked lap, <= 1e-12 | any > 1e-10 |
| D5 | (control) the release P has SWAPs at spare 0 | P output not mappable in 20 of 20 laps | 0 |
| D6 | (control) no cliff with room to spare | spare 16: Q3 within 1 s in 18 or more of 20 | 10 or fewer |
| D7 | PN with room to spare | spare 16: PN within 1 s, swap-free and mapped back in 20 of 20 | 10 or fewer |

Between the bounds: ambiguous.

**Expectations stated before running.**

- D1, D3, D5 and D6 follow from today's single-compile results. Here they
  are tested with the PennyLane round trip in the loop, over 20 laps, and
  with the candidate core added.
- D2: Qiskit L3 took 24-29 s per compile on this family, so the cliff
  should recur on every lap.
- D4 is the genuinely new part. Since the drift fix, the release stack's
  repeated-compile loop C reached 6.1e-12 at lap 1,000 (Addendum 244, v4).
  20 laps should stay far below 1e-12, but PennyLane's own matrices and the
  3-qubit blocks were not part of that measurement.

**Reported without prediction:**

- per-lap compile and lap times (lap time includes the PennyLane
  conversions);
- two-qubit counts;
- layout changes between laps;
- the block distances of Q3 and P wherever they could be checked (at spare
  16).

## 4. What this can and cannot establish

It can show whether the candidate stack meets a 1 s deadline on a saturated
heavy-hex device inside a PennyLane loop, and keeps the circuit's meaning
over 20 compounded laps.

It cannot establish:

- the live ibm_kingston Target;
- GPU simulation (this runs on CPU; the pod is not used);
- training with gradients, since the circuits have fixed matrices;
- more than 20 laps;
- other circuit families.

Times are sandbox times. Adoption of either candidate is home's decision.

## 5. Files, integrity, run commands; dry runs

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| [`pl_heavyhex_chain.py`](../../benchmarks/pl_heavyhex_chain.py) (in `benchmarks/`) | 14,966 | `84aafa1f56680e4e77be2e6261b2d45e669c5fa48a3db25f88e8180ccf8e005f` |

It imports [`full_heavyhex_cliff.py`](../../benchmarks/full_heavyhex_cliff.py) (locked earlier today, `1264fada...`),
[`loop_endurance.py`](../../benchmarks/loop_endurance.py) and [`psf_pennylane_gpu_prototype.py`](../../benchmarks/psf_pennylane_gpu_prototype.py) from the clean
clone at `f4b4a6c`. The versions are Qiskit 2.5.2, PennyLane 0.45.1 and
Python 3.11.15.

```
PYTHONPATH=<release_core_dir>   python -u benchmarks/pl_heavyhex_chain.py run --arm Q3 > pl_chain_Q3.txt 2>&1
PYTHONPATH=<release_core_dir>   python -u benchmarks/pl_heavyhex_chain.py run --arm P  > pl_chain_P.txt 2>&1
PYTHONPATH=<candidate_core_dir> python -u benchmarks/pl_heavyhex_chain.py run --arm PN --layout-dir <candidate_layout_dir> > pl_chain_PN.txt 2>&1
python -u benchmarks/pl_heavyhex_chain.py score > pl_chain_score.txt 2>&1
```

**Dry runs** (FakeAuckland, 3 laps, spares 0 and 2):

- **Dry run 1** used a first version of the script (`c05a5fa4...`). C0
  failed: the PN rows reported layout 2026-09-26.m1, because
  `loop_endurance` imports `psf_smart_layout` at module level, before the
  candidate folder was put on the path. The fix moves that path insertion to
  the start of `run`. In that run PN therefore behaved like P (SWAPs at
  spare 0).
- **Dry run 2** used the locked script:
  - C0 passed, and D1-D7 all scored CONFIRMED at that scale;
  - PN at spare 0: 51 two-qubit gates, swap-free, 0.015 s median, block
    distance 8e-15 to 1.6e-14 over 3 laps;
  - Q3 at spare 0: 8.8 s per lap, with SWAPs.

No FakeKingston lap was run before locking.

---

<!-- ===== Addendum 255 (source: spare-qubit-cliff-addendum-255-2026-09-29.md) ===== -->

> **Note added when merging:** D1-D7 confirmed: the candidate stack met 1 s on every lap (0.071 s), swap-free, within 1.4e-13 of PennyLane's block matrices after 20 laps; Qiskit L3 took 26.6 s per lap; the release stack could not map its SWAP outputs back to blocks.

## Addendum 255 -- In a PennyLane loop on a fully occupied heavy-hex device (FakeKingston, 156 wires), the candidate stack (core 2026-09-29.1 + psf_smart_layout 2026-09-29.c1) meets 1 s on 20 of 20 laps, swap-free, with the meaning kept to 1.4e-13 over 20 compounded laps; Qiskit L3 misses on every lap (26.6 s) (D1-D7 confirmed) (2026-09-29)

**Scored against:** `pl-heavyhex-chain-preregistration-2026-09-29.md`,
locked in the Project before the scored runs. The script
[`pl_heavyhex_chain.py`](../../benchmarks/pl_heavyhex_chain.py) was saved at the same time. Thresholds were applied
exactly as written.

**Run:** workplace sandbox.

- **Machine:** Intel(R) Xeon(R) Processor @ **2.10GHz**, 2 CPUs. The
  sandbox was restarted between this morning's runs and this one: the
  earlier runs today had 2.80GHz, so times are not compared across them.
- **Software:** Python 3.11.15, Qiskit 2.5.2, PennyLane 0.45.1,
  qiskit-ibm-runtime 0.50.0 (fake provider only).
- **Code:** clean clone at `f4b4a6c`, with `psf_compile.py` 2026-09-28.1
  (`3616efc8...`) in both PSF-Zero arms.
  - P: core 2026-09-28.1 and layout 2026-09-26.m1 (`a639efde...`).
  - PN: core 2026-09-29.1 and layout 2026-09-29.c1 (`e25952a3...`).
  - The versions were printed by each arm.
- **Hashes:** script `84aafa1f...`, checked before the run
  ([`sandbox_env_pl.txt`](../../data/2026-09-29/pl_heavyhex_chain/sandbox_env_pl.txt)) and printed by it.
- **Timeline:** 04:01:39-04:12 UTC, with the arms run one after another.
  **Times are sandbox times.**
- **Amendments after the lock:** none.

## 1. Scoring

**G0:** passed. **C0:** passed. Every arm completed 20 laps at spares 0 and
16, with the expected versions.

| ID | Prediction | Verdict | Numbers |
|---|---|---|---|
| D1 | PN meets 1 s at full occupation | **CONFIRMED** | 20 of 20 laps; median 0.071 s, max 0.217 s |
| D2 | the cliff recurs on every lap for Qiskit L3 | **CONFIRMED** | 0 of 20; median 26.6 s, max 28.2 s |
| D3 | PN swap-free and back in PennyLane every lap | **CONFIRMED** | 20 of 20, 276 two-qubit gates every lap |
| D4 | PN keeps the meaning over 20 compounded laps | **CONFIRMED** | worst block distance to lap 0: 1.45e-13 |
| D5 | (control) release P has SWAPs at spare 0 | CONFIRMED | 20 of 20 (327 two-qubit gates) |
| D6 | (control) no cliff at spare 16 | CONFIRMED | Q3 within 1 s in 20 of 20; median 0.306 s |
| D7 | PN at spare 16 | **CONFIRMED** | 20 of 20 within 1 s, swap-free and mapped back |

## 2. Numbers (20 laps each; sandbox)

| spare | arm | compile median | lap median (with PennyLane conversion) | total compile | two-qubit | laps mapped back | block distance lap 1 / lap 20 |
|---|---|---|---|---|---|---|---|
| 0 | Q3 | 26.597 s | 26.73 s | 533.8 s | 321 (swap-free 276) | 0 (SWAPs) | - |
| 0 | P | 0.159 s | 0.27 s | 3.3 s | 327 | 0 (SWAPs) | - |
| 0 | PN | **0.071 s** | 0.79 s | 1.6 s | **276** | **20** | 1.2e-14 / 1.4e-13 |
| 16 | Q3 | 0.306 s | 0.97 s | 6.3 s | 252 | 20 | 1.8e-13 / 4.6e-13 |
| 16 | P | 0.064 s | 0.74 s | 1.4 s | 252 | 20 | 1.8e-14 / 1.2e-13 |
| 16 | PN | 0.056 s | 0.63 s | 1.2 s | 252 | 20 | 1.9e-14 / 1.2e-13 |

**Also reported without prediction:**

- **Drift per lap.** The block distance grows linearly with the lap, by
  least squares over the checked laps:
  - PN: 7.0e-15 per lap at spare 0 (r² 0.986) and 5.7e-15 at spare 16
    (r² 0.995);
  - P at spare 16: 5.7e-15 (r² 0.995);
  - Qiskit L3 at spare 16: 2.3e-14 (r² 0.79), about four times PSF-Zero's.
- **Layouts.** No PSF-Zero arm changed its layout between laps. Qiskit L3
  changed once, at spare 16.
- **P against PN at spare 16.** Their distances are close but not
  identical. The layouts differ: P falls back to Qiskit's level-1 layout,
  and PN uses the short-path shortcut.
- **Lap time.** For PN the lap is dominated by the PennyLane conversions
  (about 0.6-0.8 s), not by the compile (0.06-0.07 s). At spare 0 the P
  laps are shorter (0.27 s) only because a SWAP lap skips the conversion
  back.

The medians, totals and slopes were recomputed from the three CSVs.

## 3. Reading

1. **In a PennyLane loop on a saturated heavy-hex device, the candidate
   stack is what makes the deadline and the circuit both hold.**
   - Qiskit L3 pays the cliff on every lap: 534 s of compiling for 20 laps,
     with 45 extra two-qubit gates each time.
   - The release PSF-Zero is fast, but its output never comes back to
     PennyLane block by block, because of the layout bug found earlier
     today.
   - The candidate stack is fast (1.6 s for 20 laps), swap-free, and every
     lap returns a tape whose blocks match PennyLane's lap-0 matrices to
     1.4e-13.
2. **The drift is small and linear.**
   - About 6e-15 to 7e-15 per lap for PSF-Zero, against Qiskit L3's 2.3e-14
     here.
   - At that rate, 1e-12 would be reached after roughly 150 laps. This
     agrees with the release gate's C loop (6.1e-12 at lap 1,000,
     Addendum 244).
   - It is the compounding of floating-point rounding over repeated round
     trips, not an error of any single compile (every lap-1 distance is
     about 1e-14).
3. **The candidate core did not change anything visible here.** No core
   fallback case arose in these circuits. The core's own effect was shown in
   this morning's experiment. This run shows that the two candidates work
   together in the PennyLane loop.

## 4. What this does not establish

- The live ibm_kingston Target.
- GPU simulation: this ran on CPU, and the pod was not used.
- Training with gradients, since the blocks here have fixed matrices.
- More than 20 laps (the linear drift is extrapolated, not measured).
- Other circuit families.
- Whether a lap with SWAPs keeps the meaning: those laps could not be
  checked block by block.

## 5. Files

| File | What it is |
|---|---|
| [`sandbox_outputs/pl_chain_Q3.txt`](../../data/2026-09-29/pl_heavyhex_chain/pl_chain_Q3.txt), [`pl_chain_P.txt`](../../data/2026-09-29/pl_heavyhex_chain/pl_chain_P.txt), [`pl_chain_PN.txt`](../../data/2026-09-29/pl_heavyhex_chain/pl_chain_PN.txt), [`pl_chain_score.txt`](../../data/2026-09-29/pl_heavyhex_chain/pl_chain_score.txt) | logs and scoring |
| `sandbox_outputs/pl_heavyhex_chain_{Q3,P,PN}_2026-09-29.csv` | every lap (120 rows) |
| [`sandbox_outputs/sandbox_env_pl.txt`](../../data/2026-09-29/pl_heavyhex_chain/sandbox_env_pl.txt), `run_all.sh` | start time, hashes, CPU, the exact commands |
| `dry_runs/` | the two FakeAuckland dry runs (logs and scores) |
| [`pl_heavyhex_chain.py`](../../benchmarks/pl_heavyhex_chain.py), `pl-heavyhex-chain-preregistration-2026-09-29.md` | the locked script and pre-registration |

---

<!-- ===== Addendum 256 (source: spare-qubit-cliff-addendum-256-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Workplace pre-registration, run on a RunPod RTX 4090: the same loop on FakeAuckland (27 qubits) with a whole-circuit check of every lap's output on lightning.gpu, 30 laps.

## Addendum 256 -- Pre-registration: the PennyLane compounding loop on a fully occupied heavy-hex device with a whole-circuit check on lightning.gpu (FakeAuckland, 27 qubits; RunPod RTX 4090), candidate stack against release and Qiskit L3, 30 laps (2026-09-29)

**Status: pre-registration, locked at the Project save time of this
document**, before the pod run. Designed and dry-run in the workplace
sandbox; **the scored run is on a RunPod pod (RTX 4090)**. No IBM account, no
network access to IBM and no QPU are used (fake-provider snapshot).

## 1. Why

The CPU loop earlier today (`pl-heavyhex-chain`, FakeKingston, 156 wires)
could check the circuit only block by block. It also could not check laps
whose output contained SWAPs, because a 156-qubit statevector cannot be
simulated.

On a 27-qubit heavy-hex device the whole circuit fits on the GPU (2 GiB in
complex128). So every lap's *compiled output* can be simulated as one
circuit and compared with the lap-0 tape. This includes the SWAP outputs of
Qiskit L3 and the release PSF-Zero. The workplace dry run of the heavy-hex
cliff showed that FakeAuckland has the cliff for the pairs-plus-triples
family: Qiskit L3 took 8.5 s at spare 0 in the sandbox, with SWAPs, against
about 0.1 s at spare 2 and 4.

## 2. Design ([`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py), self-contained)

**Device and circuits.**

- Device: FakeAuckland (27 qubits, maximum matching 10). Stage 0 gate: a
  {P2,P3}-factor exists (7 triples + 3 pairs).
- Family T:
  - At spare 0: 7 triples and 3 pairs, 27 wires.
  - At spare 4: 7 triples and 1 pair, 23 wires, the control.
  - Pair block: 20 Haar-random `qml.QubitUnitary`. Triple block: 10 on
    (a, b), then 10 on (b, c).
  - Seed: 1000 x spare.

**The lap.** The loop is as in `pl_heavyhex_chain` (tape -> Qiskit ->
compile, timed -> back to a tape), 30 laps. There are two checks per lap.

- **Whole-circuit check on `lightning.gpu`** (new):
  - The routed physical circuit is simulated from |0...0> on 27 wires.
  - The observables are <Z> and <X> of every logical qubit, and <ZZ> of
    every block edge. They are read at each logical qubit's *final* physical
    position, so SWAPs and permutations are allowed.
  - The values are compared with the lap-0 logical tape's values on the same
    device. The score is the maximum absolute difference.
  - This check applies to every lap, SWAP laps included.
- **Block check** (`qml.matrix` against lap 0), when the output is
  swap-free. In that case the output also becomes the next lap's tape.
  Otherwise the next lap reuses the same input.

**Harness checks** (per arm and spare):

- **C0 (GPU against CPU):** the first three blocks (9 wires) are simulated
  on `lightning.gpu` and on `lightning.qubit`.
- **C1 (sensitivity control):** an extra RX(1e-6), inserted after the first
  gate of block 0, must change the observables by at least 1e-9.

**Arms.** One process each, run one after another.

- **Q3:** `transpile(..., optimization_level=3, target, seed_transpiler=0)`.
- **P (release):** `compile_for_hardware(..., layout_search=True,
  entangling_basis="cx", on_unsupported="raise", seed_transpiler=0)` with
  core 2026-09-28.1 and layout 2026-09-26.m1.
- **PN (candidate stack):** the same call with core 2026-09-29.1 and layout
  2026-09-29.c1.

**Pod setup** (`setup_gpu_2026-09-29.sh`, idempotent, deletes nothing):

- the venv with Qiskit 2.5.2, PennyLane 0.45.1, lightning and lightning-gpu
  0.45.0, qiskit-ibm-runtime 0.50.0 (fake provider only), and maturin;
- rustup if missing;
- the repository at `f4b4a6c`;
- both cores built the same way (maturin wheel from `src/lib.rs`: release
  `bf3bf537...`, candidate `5364630e...` via
  `lib_rs_eigen_route_2026-09-29.patch`), installed into separate folders;
- the candidate layout (`e25952a3...`) extracted from
  `psf_smart_layout_c1_2026-09-29.patch`.

The run script (`run_gpu_2026-09-29.sh`) refuses to run unless the script's
normalized SHA-256 equals the locked value.

## 3. Pre-registered predictions

**G0:** the {P2,P3}-factor is found.

**C0:**

- all arms complete 30 laps at both spares;
- versions P (2026-09-28.1, 2026-09-26.m1) and PN (2026-09-29.1,
  2026-09-29.c1);
- device `lightning.gpu`;
- GPU against CPU on the 9-wire sub-circuit <= 1e-12.

**C1:** the RX(1e-6) control changes the values by >= 1e-9.

If C0 or C1 fails, the E predictions are read with that caveat.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| E1 | PN meets 1 s at full occupation | spare 0: PN within 1 s in 30 of 30 | 15 or fewer |
| E2 | the cliff recurs for Qiskit L3 | spare 0: Q3 within 1 s in 0 of 30 | 15 or more |
| E3 | PN swap-free and back in PennyLane every lap | spare 0: 30 of 30 | 15 or fewer |
| E4 | PN correct as a whole circuit, every lap, compounded | max whole-circuit difference, both spares, <= 1e-12 | > 1e-10 |
| E5 | the SWAP outputs of Q3 and P are also correct | spare 0: max whole-circuit difference over their 60 laps <= 1e-10 | > 1e-6 in any lap |
| E6 | (control) no cliff with room to spare | spare 4: Q3 within 1 s in 28 or more of 30 | 15 or fewer |
| E7 | PN with room to spare | spare 4: PN within 1 s, swap-free and mapped back in 30 of 30 | 15 or fewer |

Between the bounds: ambiguous.

**Expectations stated before running.**

- E1, E3, E6 and E7 follow from the sandbox dry runs.
- E2 expects the pod's CPU to leave Qiskit L3 well above 1 s (8.5 s in the
  sandbox).
- E5 is the genuinely new part: whether routing with SWAPs preserves the
  circuit exactly. The refutation bound of 1e-6 would catch an error on the
  scale of Qiskit's `CommutativeCancellation` cutoff (about 1e-4; Addenda
  247 B and the cutoff study earlier today). Whether this family triggers
  that cutoff here is not known.

**Reported without prediction:**

- compile times and GPU-check times;
- the per-lap drift of the whole-circuit difference and of the block
  distance;
- two-qubit counts;
- layout changes.

## 4. What this can and cannot establish

It can show, on a GPU:

- whether every compiled output in the loop is correct as a whole circuit,
  including the SWAP outputs;
- whether the candidate stack keeps meeting 1 s while staying swap-free on a
  saturated heavy-hex device.

It cannot establish:

- anything about the 156-qubit device, which is too large to simulate (the
  CPU loop covered it block by block);
- the live ibm_kingston Target;
- training with gradients;
- more than 30 laps.

The observables are first-order sensitive to errors (C1 checks this), but a
finite set of expectation values is not a full operator check. Times are
RunPod-pod times and are not compared with home or the sandbox.

## 5. Files, integrity, run commands; dry runs

| File | Bytes | Hash |
|---|---|---|
| [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py) | 19,867 | normalized SHA-256 `a0081c2917591de57a2e3a9d4914441c47f030282c5128b3f18642b7dd8402ea` |
| `lib_rs_eigen_route_2026-09-29.patch` | 5,650 | raw SHA-256 `b9d4fd18...177d` (as locked this morning) |
| `psf_smart_layout_c1_2026-09-29.patch` | 11,028 | raw SHA-256 `e42beb1a...733f` (as locked earlier today) |
| `setup_gpu_2026-09-29.sh`, `run_gpu_2026-09-29.sh` | -- | saved with this document. The setup checks the patched `lib.rs` (`5364630e...`) and layout (`e25952a3...`) hashes, and the run checks the script's hash. |

On the pod, from the bundle folder:

```
bash setup_gpu_2026-09-29.sh 2>&1 | tee setup_log.txt
bash run_gpu_2026-09-29.sh   2>&1 | tee run_log.txt
```

**Dry runs** (workplace sandbox, on CPU with `lightning.qubit`, on a
19-qubit `GenericBackendV2` heavy-hex (d = 3), 3 laps, spares 0 and 2):

- The harness worked for all arms: C1 detected the control (4.2e-7), and
  every whole-circuit difference was <= 1.2e-13, SWAP laps of P included.
- C0 failed only on the device name (`lightning.qubit`), as expected
  without a GPU.
- On that generic device Qiskit L3 found a swap-free layout (0.4 s). So E2
  scored "refuted" there. The generic device's error data differ from
  FakeAuckland's, and FakeAuckland showed the cliff in the earlier dry run.

The setup script was also tested in the sandbox, with a separate home
folder. Steps 1-5 (environment, repository, both cores, candidate layout)
succeeded and gave `CORE_VERSION` 2026-09-28.1 and 2026-09-29.1. It stopped
at `nvidia-smi`, as expected without a GPU.

No FakeAuckland lap with this script was run before locking.

---

<!-- ===== Addendum 257 (source: spare-qubit-cliff-addendum-257-2026-09-29.md) ===== -->

> **Note added when merging:** All predictions confirmed. The SWAP-containing outputs of Qiskit L3 and the release stack were also correct as whole circuits (<= 4.9e-14): at full occupancy they lose gates and time, not correctness.

## Addendum 257 -- On the GPU, every compiled output of the PennyLane loop on a fully occupied heavy-hex device (FakeAuckland, 27 qubits) is correct as a whole circuit, including the SWAP outputs of Qiskit L3 and the release PSF-Zero (<= 4.9e-14); the candidate stack meets 1 s on 30 of 30 laps, swap-free; Qiskit L3 misses on every lap (9.6 s) (C0, C1, E1-E7 confirmed) (2026-09-29)

**Scored against:** `pl-heavyhex-gpu-preregistration-2026-09-29.md`,
locked in the Project before the pod run. The script [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py)
and the setup and run scripts were saved at the same time. Thresholds were
applied exactly as written.

**Run:** RunPod pod, not the workplace sandbox and not home.

- **Hardware:** NVIDIA GeForce RTX 4090 (24,564 MiB, driver 570.169), AMD
  EPYC 7R32 (96 threads). The pod was freshly initialized and set up by
  `setup_gpu_2026-09-29.sh`.
- **Software:** Python 3.12.3, Qiskit 2.5.2, PennyLane 0.45.1, numpy
  2.5.3, repository `f4b4a6c`.
- **Cores:** both built on the pod the same way, from `src/lib.rs`: release
  `bf3bf537...` and candidate `5364630e...`. The candidate layout is
  `e25952a3...`. The hashes were printed in [`pod_env_gpu.txt`](../../data/2026-09-29/pl_heavyhex_gpu/pod_env_gpu.txt).
- **Script hash:** `a0081c29...`, checked by the run script before the run.
- **Timeline:** 04:37:19-04:49:37 UTC. **Times are RunPod-pod times.**
- **Amendments after the lock:** none.

**Re-scored from files.** The outputs zip was downloaded to the workplace
PC and read here. Running `score` on the three CSVs gave output identical to
the pod's [`pl_gpu_score.txt`](../../data/2026-09-29/pl_heavyhex_gpu/pl_gpu_score.txt).

## 1. Scoring

**G0:** passed (7 triples + 3 pairs).

**C0:** passed.

- All arms completed 30 laps at spares 0 and 4.
- Versions: P (2026-09-28.1, 2026-09-26.m1) and PN (2026-09-29.1,
  2026-09-29.c1).
- Device `lightning.gpu`.
- GPU against CPU on the 9-wire sub-circuit: max 1.3e-15.

**C1:** passed. The RX(1e-6) control changed the values by >= 2.9e-7.

| ID | Prediction | Verdict | Numbers |
|---|---|---|---|
| E1 | PN meets 1 s at full occupation | **CONFIRMED** | 30 of 30; median 0.015 s, max 0.037 s |
| E2 | the cliff recurs for Qiskit L3 | **CONFIRMED** | 0 of 30; median 9.56 s, max 9.97 s |
| E3 | PN swap-free and back in PennyLane every lap | **CONFIRMED** | 30 of 30 (51 two-qubit gates) |
| E4 | PN correct as a whole circuit, every lap | **CONFIRMED** | max 2.9e-14 (both spares) |
| E5 | the SWAP outputs of Q3 and P are correct | **CONFIRMED** | max 4.9e-14 over 60 laps |
| E6 | no cliff at spare 4 | CONFIRMED | Q3 within 1 s in 30 of 30 (median 0.113 s) |
| E7 | PN at spare 4 | CONFIRMED | 30 of 30 within 1 s, swap-free and mapped back |

## 2. Numbers (30 laps each; pod)

| spare | arm | compile median | total compile | two-qubit (swap-free) | mapped | whole-circuit diff (range) | block distance slope |
|---|---|---|---|---|---|---|---|
| 0 | Q3 | 9.558 s | 287.3 s | 54 (51) | 0 | 4.88e-14 every lap | - |
| 0 | P | 0.035 s | 1.1 s | 63 (51) | 0 | 1.10e-14 every lap | - |
| 0 | PN | 0.015 s | 0.5 s | 51 (51) | 30 | 8.8e-15 - 2.9e-14 | 3.1e-15 / lap |
| 4 | Q3 | 0.113 s | 3.4 s | 45 (45) | 30 | 6.6e-14 - 1.3e-13 | 1.6e-14 / lap |
| 4 | P | 0.015 s | 0.4 s | 45 (45) | 30 | 9.6e-15 - 2.7e-14 | 2.9e-15 / lap |
| 4 | PN | 0.014 s | 0.4 s | 45 (45) | 30 | 9.3e-15 - 2.6e-14 | 2.9e-15 / lap |

**Reported without prediction:**

- **SWAP laps.** Q3 and P at spare 0 are constant across laps, because
  their input is reused and their compile is deterministic.
- **GPU-check time.** The whole-circuit check took about 2.0-2.4 s per lap,
  longer than any PSF-Zero compile. For PN a lap took about 2.2-2.5 s in
  all, most of it the check.
- **Drift.** The drift of the whole-circuit difference is about 6e-16 to
  7e-16 per lap for PSF-Zero, and 2.0e-15 for Qiskit L3 at spare 4.

## 3. Reading

1. **Routing with SWAPs did not change the meaning of any output here.**
   - In the CPU loop, the SWAP laps of Qiskit L3 and the release PSF-Zero
     could not be checked. On the GPU, the whole routed circuit was
     simulated, and it agreed with the logical tape to <= 4.9e-14.
   - What those arms lose at full occupation is cost: 3 to 12 extra
     two-qubit gates here, and for Qiskit L3 about 9.6 s per compile. It is
     not correctness.
   - Qiskit's `CommutativeCancellation` cutoff did not show up on this
     family.
2. **The candidate stack is fast, swap-free and correct on every lap.** It
   compiled in 15 ms, with the minimum 51 two-qubit gates. Its output was
   correct as a whole circuit to 3e-14, and it drifted slowly and linearly
   over 30 compounded laps.
3. **With room to spare the stacks agree.** At spare 4, P and PN are
   swap-free, fast, and drift at the same rate.

## 4. What this does not establish

- The 156-qubit device: too large to simulate. The CPU loop covered it
  block by block.
- The live ibm_kingston Target.
- Training with gradients.
- More than 30 laps: the 100,000-lap run is pre-registered separately.
- A full operator check: the whole-circuit check compares 71 expectation
  values (or 61 at spare 4). These are first-order sensitive (C1), but they
  are not the full operator.

## 5. Files

| File | What it is |
|---|---|
| `pod_outputs/pl_gpu_{Q3,P,PN}.txt`, [`pl_gpu_score.txt`](../../data/2026-09-29/pl_heavyhex_gpu/pl_gpu_score.txt), `pl_heavyhex_gpu_{Q3,P,PN}_2026-09-29.csv`, [`pod_env_gpu.txt`](../../data/2026-09-29/pl_heavyhex_gpu/pod_env_gpu.txt) | the pod outputs zip, unchanged |
| [`rescore.txt`](../../data/2026-09-29/pl_heavyhex_gpu/rescore.txt) | the re-scoring from the CSVs (identical to the pod's) |
| `pod_scripts/` | `setup_gpu_2026-09-29.sh`, `run_gpu_2026-09-29.sh` and the two patches as used on the pod |
| `dry_runs/` | the sandbox dry runs (CPU, 19-qubit generic heavy-hex) |
| [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py), `pl-heavyhex-gpu-preregistration-2026-09-29.md` | the locked script and pre-registration |

---

<!-- ===== Addendum 258 (source: spare-qubit-cliff-addendum-258-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Workplace pre-registration of 100,000 laps on the pod and a 500-lap home short version (Part II). Amendment 1 (appended) stopped Part I at 30,000 laps for time and scaled the thresholds; it was written while the run was in progress, with results seen up to lap 1,000 only.

## Addendum 258 -- Pre-registration: 100,000 laps of the PennyLane compounding loop on a fully occupied heavy-hex device with the candidate stack (RunPod RTX 4090), and a 500-lap short version for home (tens of seconds); no IBM (2026-09-29)

**Status: pre-registration, locked at the Project save time of this
document**, before either run. Designed and dry-run in the workplace
sandbox.

- **Part I** (100,000 laps) runs on the RunPod pod set up today
  (`setup_gpu_2026-09-29.sh`).
- **Part II** (500 laps) is for home tonight, next to the Stage 2 work.

**Neither part uses an IBM account, network access to IBM, or a QPU.** The
device is the FakeAuckland snapshot.

## 1. Why

This afternoon's GPU loop (`pl-heavyhex-gpu`, 30 laps) confirmed E1-E7 on
FakeAuckland:

- the candidate stack met 1 s on every lap;
- it stayed swap-free;
- its compiled output was correct as a whole circuit to 3e-14.

The release gate (Addenda 243-244, v4; workplace v5) ran 100,000 compiles,
but not with PennyLane in the loop, not on a saturated heavy-hex device, and
not with the candidate layout. Part I asks whether the loop holds for
100,000 laps:

- the deadline;
- swap-free every lap;
- no core fallbacks;
- drift that stays linear, not accelerating;
- memory that stays flat.

Part II is a short version home can run in tens of seconds. It confirms the
same stack on home's machine and compares lap 500 with the pod.

## 2. Design ([`pl_heavyhex_100k.py`](../../benchmarks/pl_heavyhex_100k.py); reuses the locked [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py))

**Device and circuits.** FakeAuckland, family T as in [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py)
(same seeds). At spare 0 that is 7 triples and 3 pairs on 27 wires. At
spare 4 it is 7 triples and 1 pair on 23 wires.

**Parts of the 100,000-lap run.** Each part runs in its own process, and the
three run in parallel on the pod's 96 cores.

| part | stack | spare |
|---|---|---|
| A | candidate (core 2026-09-29.1 + layout 2026-09-29.c1) | 0 |
| B | candidate | 4 |
| C | release (core 2026-09-28.1 + layout 2026-09-26.m1) | 4 |

**Each lap:**

1. tape -> Qiskit;
2. `compile_for_hardware(layout_search=True, entangling_basis="cx",
   on_unsupported="raise", seed_transpiler=0)`, timed, with the synthesis
   cache cleared first as in v4/v5;
3. back to logical qubits -> tape, which is the next lap's input.

**Recorded every lap:**

- compile time and two-qubit count;
- mapped back or not;
- core fallbacks, from psf_compile's warning;
- block distance to lap 0 (`qml.matrix`).

**At checkpoints** (laps 1, 10, 100, 1,000, 5,000 and every 5,000 up to
100,000; 24 in all):

- the whole-circuit check of that lap's compiled output on `lightning.gpu`
  (<Z>, <X> of every qubit and <ZZ> of every block edge, at the final
  positions) against lap 0;
- RSS.

RSS is also recorded every 1,000 laps. The C1 control (an extra RX(1e-6))
runs once per part.

**Part II (short):** part A only, 500 laps, checkpoints at 1, 10, 100 and
500, run by `short_500_2026-09-29.sh`.

- On its first run the script builds the candidate core into
  `~/core_cand_0929` and extracts the candidate layout into
  `~/layout_cand_0929`, outside home's repository. Both are hash-checked
  (`5364630e...`, `e25952a3...`).
- It changes nothing in the repository and deletes nothing.

## 3. Pre-registered predictions

### Part I (pod, 100,000 laps)

**C0:**

- all three parts complete 100,000 laps;
- versions A and B (2026-09-29.1, 2026-09-29.c1) and C (2026-09-28.1,
  2026-09-26.m1);
- device `lightning.gpu`;
- C1 >= 1e-9 in every part.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| H1 | A meets 1 s | within 1 s in >= 99,990 of 100,000 laps | < 99,000 |
| H2 | A stays swap-free and returns to PennyLane | 100,000 of 100,000 | < 99,900 |
| H3 | no core fallbacks with the candidate core | A + B: 0 | >= 3 |
| H4 | A's drift does not accelerate | block distance at lap 100,000 <= 2 x the straight line fitted to laps 1-1,000 | > 10 x |
| H5 | the compiled output stays correct as a whole circuit | A and B, every checkpoint: max difference <= 1e-8 | > 1e-6 |
| H6 | memory stays flat | RSS growth from lap 1,000 to the end <= 100 MB in every part | > 500 MB in any |
| H7 | candidate and release drift alike where both are swap-free | B / C block distance at lap 100,000 between 0.5 and 2 | < 0.2 or > 5 |
| H8 | A's timing is stable | median of the last 10% <= 1.5 x the first 10% | > 3 x |

**Reported without prediction:**

- the release core's fallbacks in C (the release rejects about 1 in 30,000
  training blocks; Addendum 247 A);
- compile-time distributions;
- GPU-check times;
- the drift slopes;
- wall time.

### Part II (home, 500 laps)

**S0:** 500 laps, candidate versions, device `lightning.gpu`, C1 >= 1e-9.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| S1 | within 1 s | 500 of 500 | 450 or fewer |
| S2 | swap-free and mapped back | 500 of 500 | 450 or fewer |
| S3 | no core fallbacks | 0 | >= 3 |
| S4 | meaning kept | lap-500 block distance <= 1e-11 **and** whole-circuit max <= 1e-10 | block > 1e-9 or whole > 1e-8 |
| S5 | tens of seconds | the loop (500 laps with 4 GPU checks) <= 120 s | > 600 s |
| S6 | same drift as on the pod (only if the pod's part A CSV is given) | home / pod lap-500 block distance between 0.5 and 2 | < 0.1 or > 10 |

Between the bounds: ambiguous.

**Expectations stated before running.**

- The drift should be linear, at about 2.6e-15 to 7e-15 per lap. That is
  about 3e-12 at lap 1,000 in the dry runs, and about 1e-10 to 7e-10 by lap
  100,000. That is far above today's 1e-12-scale checks, but well inside
  H5's 1e-8.
- H4 allows a random-walk (slower-than-linear) growth, and refutes only a
  clear acceleration.
- H7 expects the two stacks to drift alike at spare 4 (in the 30-lap GPU run
  their distances agreed to three digits).
- H1 allows ten late laps, for pod noise.

**Time.** The sandbox took about 0.12 s per lap on 2 CPUs. On the pod,
Part I is expected to take about 3 hours; the pod must stay running. Part
II is expected to take 40-90 s at home, most of it the 500 laps.

## 4. What this can and cannot establish

It can show whether the candidate stack's PennyLane loop on a saturated
heavy-hex device holds over 100,000 laps, and how its rounding drift grows.

It cannot establish:

- anything about the 156-qubit device, or the live ibm_kingston Target;
- training with gradients;
- the Qiskit L3 arm, which is not run: at about 9.6 s per lap it would take
  11 days.

Part II is a stack check at home. It is not an IBM experiment and says
nothing about Stage 2. Times are pod or home times and are not compared
with each other or with the sandbox.

## 5. Files, integrity, run commands; dry runs

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| [`pl_heavyhex_100k.py`](../../benchmarks/pl_heavyhex_100k.py) | 16,396 | `dad56b1b2f2351b413e6f2b096ee4f9b2e3ae6d8db405760a5fcb0ae523288cb` |
| [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py) (reused, locked earlier today) | 19,867 | `a0081c2917591de57a2e3a9d4914441c47f030282c5128b3f18642b7dd8402ea` |
| `run_100k_2026-09-29.sh` (pod), `short_500_2026-09-29.sh` (home) | -- | saved with this document; both refuse to run unless the two hashes above match |

Pod, from the bundle folder:

```
nohup bash run_100k_2026-09-29.sh > run_100k_log.txt 2>&1 &
```

Home, with the project's Python environment active:

```
REPO=<psf-zero clone> bash short_500_2026-09-29.sh [<pod part A CSV>]
```

**Dry runs** (workplace sandbox, CPU, `lightning.qubit`):

- **Per-lap time.** FakeAuckland, part A, 200 laps, no checkpoints: 0.124 s
  per lap (compile 17 ms). The block distance at lap 200 was 6.0e-13, and
  every lap was mapped.
- **Harness.** A 19-qubit generic heavy-hex (d = 3), parts A, B and C in
  parallel, 1,200 laps, checkpoints 1, 10, 100, 1,000, 1,200.
  - Every H prediction that can be scored at that length came out CONFIRMED:
    fallbacks 0, drift ratio 1.00, whole-circuit max 1.8e-12, RSS growth
    <= 4 MB, B / C = 1.00.
  - H1 and H2 were "refuted" only because the dry run had 1,200 laps, not
    100,000.
  - Two scorer crashes that happened only in runs shorter than 1,000 laps
    were fixed before locking.
- **Short script.** Tested in the sandbox with a separate home folder: it
  built the candidate core, extracted the layout, passed the hash checks and
  started the loop. The loop was stopped at the first whole-circuit check,
  because a 27-qubit statevector on 2 CPUs is too slow without a GPU.

No FakeAuckland lap of [`pl_heavyhex_100k.py`](../../benchmarks/pl_heavyhex_100k.py) was run on a GPU before
locking.

---

### Amendment 1 to the pl-heavyhex-100k pre-registration (2026-09-29): Part I is stopped at 30,000 laps, for time; thresholds scaled

**Status: amendment, written and saved in the Project while the pod run was
in progress, before any result beyond lap 1,000 had been seen.** It amends
`pl-heavyhex-100k-preregistration-2026-09-29.md`, Part I only. Part II (the
500-lap short version for home) is unchanged.

## 1. Why, and what had been seen

- **Timing.** The pod run started at about 05:31 UTC (14:31 JST) with parts
  A, B and C in parallel. At about 0.105 s per lap it would take about 2.9
  hours. The owner asked for about one hour, for time reasons.
- **What had been seen before this amendment.** One progress check, quoted
  in the chat:
  - A: laps 10 and 100;
  - B: laps 100 and 1,000;
  - C: laps 10 and 100.
  - All were swap-free, with 0 fallbacks, and block distances of 3.6e-14 to
    2.9e-12.
  - Nothing beyond lap 1,000 had been seen.
- **The per-lap time cannot be reduced without changing the locked
  script.** The lap is dominated by the PennyLane conversions and the block
  matrices, and the loop is sequential by design (each lap's output is the
  next lap's input). So the run is shortened, not sped up.

## 2. The change

1. **The run is stopped once all three parts have printed lap 30,000.**
   - A watcher on the pod waits for the line `lap  30000` in all three logs,
     then stops the three `pl_heavyhex_100k.py run` processes.
   - `run_100k_2026-09-29.sh` then continues as written: its own scoring
     fails C0, as expected, since 100,000 laps were not reached, and it
     writes the outputs zip.
   - At lap 30,000 each process has flushed its CSV (30,000 is a
     checkpoint). Rows after lap 30,000, including a possibly half-written
     last row, are ignored.
   - Expected duration: about 55 minutes from the start.
2. **Scoring is on laps 1-30,000**, with [`score_30k_amendment.py`](../../benchmarks/score_30k_amendment.py) (6,561
   bytes, normalized SHA-256
   `9fb59f1918d13ca88c6cf91c149989fbcbeefa6acefa9b19e61b1dd9d07205f6`,
   saved with this amendment).

| ID | Original (100,000 laps) | Amended (30,000 laps) |
|---|---|---|
| C0 | all parts 100,000 laps | all parts have laps 1-30,000; other conditions unchanged |
| H1 | >= 99,990 within 1 s; refuted < 99,000 | >= 29,997; refuted < 29,700 (same rates, 0.01% and 1%) |
| H2 | 100,000 swap-free and mapped; refuted < 99,900 | 30,000; refuted < 29,970 (same rate, 0.1%) |
| H3 | A + B fallbacks 0; refuted >= 3 | unchanged (counted over laps 1-30,000) |
| H4 | lap 100,000 <= 2 x the laps 1-1,000 line; refuted > 10 x | same bounds, at lap 30,000 |
| H5 | every checkpoint <= 1e-8; refuted > 1e-6 | same, checkpoints 1, 10, 100, 1,000, 5,000, 10,000, 15,000, 20,000, 25,000, 30,000 |
| H6 | RSS growth lap 1,000 -> end <= 100 MB; refuted > 500 MB | same bounds, lap 1,000 -> 30,000 |
| H7 | B / C at lap 100,000 in 0.5-2; refuted < 0.2 or > 5 | same bounds, at lap 30,000 |
| H8 | last / first 10% median <= 1.5; refuted > 3 | same bounds, over laps 1-30,000 |

**Expectation, unchanged in kind.** A linear drift of about 3e-15 per lap
gives about 1e-10 at lap 30,000.

## 3. What this changes in what can be concluded

The run establishes 30,000 compounded laps, not 100,000. The release gate's
100,000 compiles (v4, v5) remain the only 100,000-scale evidence, and they
did not include PennyLane in the loop. Any statement from this run is
limited to 30,000 laps.

---

<!-- ===== Addendum 259 (source: spare-qubit-cliff-addendum-259-2026-09-29.md) ===== -->

> **Note added when merging:** H1-H8 confirmed under Amendment 1: 30,000 laps within 1 s, swap-free, no core fallbacks, flat memory; drift linear in laps (about 3e-15 per lap, 8.8e-11 at lap 30,000). A 100,000-lap PennyLane loop is not established. The home short version (Part II) had not been run when this was merged.

## Addendum 259 -- 30,000 compounded PennyLane laps on a fully occupied heavy-hex device (FakeAuckland) with the candidate stack: 1 s met on every lap, swap-free every lap, 0 core fallbacks, drift strictly linear (2.9e-15 per lap, 8.8e-11 at lap 30,000), memory flat; candidate and release drift alike (within 0.14%) where both are swap-free (H1-H8 confirmed under Amendment 1) (2026-09-29)

**Scored against:** `pl-heavyhex-100k-preregistration-2026-09-29.md`, as
changed by **Amendment 1**
(`pl-heavyhex-100k-amendment1-2026-09-29.md`).

- **The amendment.** It was saved in the Project while the run was in
  progress, when nothing beyond lap 1,000 had been seen. It stops Part I at
  30,000 laps for time, and scales the thresholds.
- **The scorer.** Scoring used [`score_30k_amendment.py`](../../benchmarks/score_30k_amendment.py) (`9fb59f19...`),
  saved with the amendment.
- **The pod's own scoring.** `run_100k_2026-09-29.sh` ran the original
  100,000-lap scorer automatically after the stop ([`pl_100k_score.txt`](../../data/2026-09-29/pl_heavyhex_30k/pl_100k_score.txt), kept
  unchanged). It reports C0 FAILED and H1/H2 "REFUTED", only because
  30,000 laps are fewer than 100,000. It does not apply under the
  amendment.

Part II (the 500-lap short version for home) is not part of this document.

**Run:** RunPod pod.

- **Hardware and software:** RTX 4090 (driver 570.169), AMD EPYC 7R32 (96
  threads), Python 3.12.3, Qiskit 2.5.2, PennyLane 0.45.1, repository
  `f4b4a6c`. The cores and layout were as set up for the GPU loop
  (`bf3bf537...`, `5364630e...`, `e25952a3...`, printed in
  [`pod_env_100k.txt`](../../data/2026-09-29/pl_heavyhex_30k/pod_env_100k.txt)).
- **Script hashes:** `dad56b1b...` and `a0081c29...`, checked by the run
  script before the run.
- **Timeline:** start 05:30:55 UTC. Parts A, B and C ran in parallel. The
  watcher stopped all three at 06:28:33 UTC, after all had passed lap
  30,000.
- **Rows ignored:** A had exactly 30,000 rows. B and C, which were faster,
  had reached laps 34,002 and 33,243. Their rows after lap 30,000 are
  ignored.
- **Times are RunPod-pod times.**

## 1. Scoring (Amendment 1, laps 1-30,000)

**C0:** passed.

- All three parts have laps 1-30,000.
- Versions: A and B (2026-09-29.1, 2026-09-29.c1); C (2026-09-28.1,
  2026-09-26.m1).
- Device `lightning.gpu`; C1 >= 2.9e-7.

| ID | Prediction (amended) | Verdict | Numbers |
|---|---|---|---|
| H1 | A within 1 s in >= 29,997 of 30,000 | **CONFIRMED** | 30,000; median 15.1 ms, p99 119 ms, max 197 ms |
| H2 | A swap-free and mapped back in 30,000 | **CONFIRMED** | 30,000 (51 two-qubit gates every lap) |
| H3 | candidate core fallbacks in A + B = 0 | **CONFIRMED** | 0 |
| H4 | A drift does not accelerate (lap 30,000 <= 2 x the laps 1-1,000 line) | **CONFIRMED** | predicted 8.746e-11, measured 8.794e-11, ratio 1.01 |
| H5 | whole-circuit check <= 1e-8 at every checkpoint | **CONFIRMED** | max 2.45e-11 (A, lap 30,000) |
| H6 | RSS growth lap 1,000 -> 30,000 <= 100 MB | **CONFIRMED** | -14, -11, -18 MB (it fell) |
| H7 | B / C block distance at lap 30,000 in 0.5-2 | **CONFIRMED** | 9.1164e-11 / 9.1164e-11 = 1.00 |
| H8 | A last / first 10% median <= 1.5 | **CONFIRMED** | 15.07 / 15.23 ms = 0.99 |

## 2. Numbers (reported without prediction)

**Drift of the block distance** (least squares over all 30,000 laps):

| part | slope per lap | distance at lap 10,000 / 20,000 / 30,000 | residual, straight line / square-root fit |
|---|---|---|---|
| A | 2.93e-15 | 2.92e-11 / 5.86e-11 / 8.79e-11 | 1.9e-13 / 5.0e-12 |
| B | 3.06e-15 | 2.99e-11 / 6.07e-11 / 9.12e-11 | 1.8e-13 / 5.4e-12 |
| C | 3.06e-15 | 2.99e-11 / 6.07e-11 / 9.12e-11 | 1.8e-13 / 5.4e-12 |

- The growth is linear, not a random walk: the straight line fits about 25
  times better than a square-root curve.
- The whole-circuit difference also grows linearly, by about 8e-16 per lap:
  8.2e-13 at lap 1,000, 8.4e-12 at 10,000, 2.5e-11 at 30,000.

**Other numbers:**

- **Candidate against release at spare 4.** B and C agree closely on every
  lap: the largest relative difference from lap 100 on is 0.14%, and 498
  of the 30,000 laps are identical. At lap 30,000 they differ by 9e-17
  (relative 1e-6).
- **Release core fallbacks.** C had 0 in its 30,000 laps, and 0 up to lap
  33,243. On these random-unitary blocks the release core met none of its
  near-tie cases.
- **Lap time.** The median lap took 0.108 s (A) and 0.096-0.098 s (B, C),
  with compile 14-15 ms. The rest is the PennyLane conversions and the block
  matrices. A checkpoint lap adds about 2.4 s of GPU check.
- **For the short version at home (S6).** Part A's block distance at lap
  500 is 1.511411e-12 (lap 1,000: 2.948950e-12).

The numbers above were recomputed here from the CSVs.

## 3. Reading

1. **The candidate stack holds for 30,000 compounded laps inside a
   PennyLane loop on a saturated heavy-hex device.** Every lap met 1 s
   (worst 0.2 s). Every lap was swap-free and came back to PennyLane. There
   were no core fallbacks, and memory and timing stayed flat.
2. **The only thing that grows is the rounding drift, and it grows
   linearly.**
   - The rate is about 3e-15 per lap in the block distance, and about 8e-16
     per lap in the whole-circuit observables.
   - At that rate the drift would reach about 3e-10 at 100,000 laps and
     1e-8 at about 3 million. This is an extrapolation, not a measurement.
   - It is the same kind of drift the release gate's C loop showed (1.2e-10
     at 20,000 laps, Addendum 244). It comes from repeating the conversion
     round trip, not from any single compile: every lap-1 value is about
     1e-14.
3. **The candidate changes nothing that matters where the release already
   works.** At spare 4 the two stacks produced the same drift over 30,000
   laps, within 0.14% at every lap from lap 100 on. The candidate's gain is at full occupation (part A), where
   the release cannot compile without SWAPs.

## 4. What this does not establish

- **100,000 laps.** The run was stopped at 30,000 by Amendment 1. The only
  100,000-scale evidence remains the release gate v4/v5, which had no
  PennyLane in the loop.
- **The 156-qubit device and the live ibm_kingston Target.**
- **Training with gradients.**
- **The Qiskit L3 arm**, which was not run.

## 5. Files

| File | What it is |
|---|---|
| `pod_outputs/` | the pod outputs zip, unchanged: three logs, three CSVs (with rows beyond 30,000 for B and C), [`pl_100k_score.txt`](../../data/2026-09-29/pl_heavyhex_30k/pl_100k_score.txt) (the original scorer, not applicable under the amendment), [`pod_env_100k.txt`](../../data/2026-09-29/pl_heavyhex_30k/pod_env_100k.txt) |
| [`score_30k_amendment.txt`](../../data/2026-09-29/pl_heavyhex_30k/score_30k_amendment.txt) | the amended scoring, run here on the CSVs |
| [`score_30k_amendment.py`](../../benchmarks/score_30k_amendment.py), `pl-heavyhex-100k-amendment1-2026-09-29.md` | the amendment and its scorer |
| [`pl_heavyhex_100k.py`](../../benchmarks/pl_heavyhex_100k.py), [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py), `pl-heavyhex-100k-preregistration-2026-09-29.md` | the locked scripts and pre-registration |
| `pod_scripts/run_100k_2026-09-29.sh` | the run script as used on the pod |
| `dry_runs/` | the sandbox dry runs |


---

<!-- ===== Addendum 260 (source: spare-qubit-cliff-addendum-260-2026-09-29.md) ===== -->

> **Note added when merging:** Home run of Part II of Addendum 258 (500 laps, candidate stack, RTX 4070): S0-S6 confirmed; the block distance equals the pod's at all 500 laps to the printed digits.

## Addendum 260 -- The 500-lap home short version (Part II of Addendum 258): S0-S6 all confirmed; the candidate stack's block distance at every one of the 500 laps equals the pod's run to all printed digits, on a different GPU, CPU and build (2026-09-29)

**Pre-registered in**: Addendum 258, Part II (`pl-heavyhex-100k-preregistration-2026-09-29.md`),
locked at the workplace before either run. Run at home with the kit handed over as
`work_2026-09-29_home_short_kit` (checked with `check_intake.py`: all listed files
verified), using `short_500_2026-09-29.sh` (now `benchmarks/pod/`).

**Run (home, 2026-09-29 12:10:57 UTC):** WSL2, AMD Ryzen 5 5500, NVIDIA GeForce RTX 4070
(12,282 MiB; Windows driver 616.92), Python 3.12.13, Qiskit 2.5.2, PennyLane 0.45.1,
`lightning.gpu`. Repository at `b8f3ec3`, not changed by the run. The script checked the
locked files before running ([`pl_heavyhex_100k.py`](../../benchmarks/pl_heavyhex_100k.py) `dad56b1b...`, [`pl_heavyhex_gpu.py`](../../benchmarks/pl_heavyhex_gpu.py)
`a0081c29...`) and the release files it builds from (`src/lib.rs` `bf3bf537...`,
[`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) `a639efde...`). It built the candidate core outside the
repository (release `lib.rs` + `lib_rs_eigen_route_2026-09-29.patch`, hash-checked; cargo
release build, maturin wheel for CPython 3.12) and extracted the candidate layout. The
run log prints `CORE_VERSION 2026-09-29.1 | LAYOUT 2026-09-29.c1 e25952a33bac |
psf_compile 2026-09-28.1 3616efc8b8a7`. Part A of the pre-registration: FakeAuckland
(27 qubits, fully occupied with pairs and 3-qubit paths), 500 compounded PennyLane laps,
whole-circuit GPU checks at laps 1, 10, 100 and 500. **Times are home times** (not
compared with the pod or the workplace).

## 1. Scoring (thresholds as pre-registered)

| ID | Prediction | Result |
|---|---|---|
| S0 | 500 laps, candidate versions, `lightning.gpu`, C1 control detected (>= 1e-9) | **passed** (C1 5.06e-7) |
| S1 | within 1 s in 500/500 (refuted <= 450) | **confirmed**: 500/500 |
| S2 | swap-free and mapped back in 500/500 (refuted <= 450) | **confirmed**: 500/500, 51 two-qubit gates every lap |
| S3 | core fallbacks 0 (refuted >= 3) | **confirmed**: 0 |
| S4 | lap-500 block distance <= 1e-11 and whole-circuit max <= 1e-10 | **confirmed**: 1.51e-12 and 4.22e-13 |
| S5 | whole run <= 120 s (refuted > 600) | **confirmed**: 64 s |
| S6 | lap-500 block distance / pod's (1.511411e-12) in 0.5-2 | **confirmed**: 1.511411e-12 / 1.511411e-12 = 1.00 |

Reported without prediction: compile median 13.2 ms (min 11.3, max 108 ms); the four GPU
checks took 4.8-5.1 s each; total wall including imports 72 s.

## 2. Comparison with the pod (done here, not pre-registered)

The home CSV was compared lap by lap with the pod's Part A CSV (Addendum 259,
[`data/2026-09-29/pl_heavyhex_30k/pl_100k_A_2026-09-29.csv`](../../data/2026-09-29/pl_heavyhex_30k/pl_100k_A_2026-09-29.csv), laps 1-500):

- **block distance to lap 0: identical as printed (7 significant digits) at all 500
  laps**; two-qubit count, mapped flag and fallbacks identical at every lap;
- whole-circuit GPU check at the shared checkpoints: home 1.121e-14, 1.477e-14,
  8.448e-14 against the pod's 1.099e-14, 1.488e-14, 8.449e-14 at laps 1, 10, 100 (the
  pod had no checkpoint at lap 500). These differ in the last digits, as expected for
  statevector simulations on different GPUs; they are all at the 1e-14-1e-13 level.

So the candidate stack's compiled circuits, as seen through PennyLane's block matrices,
evolve identically over 500 compounded laps on the pod (RTX 4090, AMD EPYC, Linux, core
built there) and at home (RTX 4070, Ryzen, WSL2, core built here).

## 3. What this does not establish

The candidates are still not the release (this is a replication of Part II, not the
adoption check). Qiskit L3 and the release stack were not run here. 500 laps, one
device model, one circuit family.

## 4. Files (`data/2026-09-29/pl_heavyhex_short_home/`)

| File | What it is |
|---|---|
| [`env_short.txt`](../../data/2026-09-29/pl_heavyhex_short_home/env_short.txt) | start time, GPU, CPU, Python, repository commit, total wall |
| [`pl_100k_A_short.txt`](../../data/2026-09-29/pl_heavyhex_short_home/pl_100k_A_short.txt) | run log (versions, script hashes, checkpoints) |
| [`pl_100k_A_short.csv`](../../data/2026-09-29/pl_heavyhex_short_home/pl_100k_A_short.csv) | every lap (500 rows) |
| [`score_short.txt`](../../data/2026-09-29/pl_heavyhex_short_home/score_short.txt) | scoring |
| [`short_500_home_console.txt`](../../data/2026-09-29/pl_heavyhex_short_home/short_500_home_console.txt) | console output (end of the core build and the scoring) |


---

<!-- ===== Addendum 261 (source: spare-qubit-cliff-addendum-261-preregistration-2026-09-29.md) ===== -->

> **Note added when merging:** Home pre-registration, pushed before running: Part I of Addendum 258 at the original 100,000 laps overnight, H1-H8 unchanged, plus lap-by-lap replication against the pod's first 30,000 laps (R1-R3).

## Addendum 261 -- Pre-registration: Part I of Addendum 258 at the originally registered 100,000 laps, at home (RTX 4070), overnight; plus a lap-by-lap replication check against the pod's first 30,000 laps (2026-09-29)

**Written and pushed before running.** Results will be recorded as Addendum 262.

## 1. Why

Part I of Addendum 258 (the PennyLane compounding loop on a fully occupied heavy-hex
device, 100,000 laps) was stopped at 30,000 laps on the pod for time (Amendment 1;
Addendum 259). Its 100,000-lap claim is therefore not established. The home short
version (Addendum 260) showed that the candidate stack's block distance on the home
machine equals the pod's at all 500 laps to the printed digits. This run completes
Part I as originally registered, on the home machine, overnight.

## 2. Design

Exactly Part I of Addendum 258: the locked `pl_heavyhex_100k.py` (`dad56b1b...`) with the
locked `pl_heavyhex_gpu.py` (`a0081c29...`); FakeAuckland (27 qubits), family T;
parts A (candidate stack, spare 0), B (candidate stack, spare 4), C (release stack,
spare 4) in parallel; 100,000 laps each; the default checkpoints (1, 10, 100, 1,000,
5,000, then every 5,000); `lightning.gpu`. Tag `home`.

- Stacks: candidate core 2026-09-29.1 (`~/core_cand_0929`, built from the release
  `lib.rs` + the patch, `5364630e...`) with candidate layout 2026-09-29.c1
  (`e25952a3...`); release core 2026-09-28.1 (installed in the environment; repository
  `lib.rs` `bf3bf537...`) with the repository's layout 2026-09-26.m1 (`a639efde...`).
- Run script `benchmarks/pod/run_100k_home_2026-09-29.sh` (normalized SHA-256
  `02177d34e7d2eb0ab889c6c58db4a8ae44d7b0c55678ec00f3e382cfae8e2945`). It checks every hash and both core versions,
  refuses to start if more than 4,000 MiB of GPU memory is already in use, records the
  environment, runs the three parts, then scores with the script's own `score` (the
  original 100,000-lap thresholds).
- Machine: home, WSL2, AMD Ryzen 5 5500, RTX 4070 12 GB, Python 3.12.13. Times are
  home times, not compared with the pod.

## 3. Predictions

**H1-H8 of Addendum 258, Part I, unchanged** (100,000 laps; the thresholds as written
there, applied by the script's `score`).

**Added here** (replication; possible because the pod's CSVs are in the repository,
`data/2026-09-29/pl_heavyhex_30k/`):

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| R1 | A's block distance equals the pod's at laps 1-30,000 | identical as printed (7 significant digits) at all 30,000 laps | any lap differs by more than 1% |
| R2 | the same for B and C | identical as printed at all 30,000 laps of each | any lap differs by more than 1% |
| R3 | two-qubit count, mapped flag and fallbacks equal the pod's at laps 1-30,000 | equal at every lap in A, B and C | any difference |

Between the bounds: ambiguous. R1-R3 are scored at home the next day from the CSVs
(the comparison script will be published with the results). The whole-circuit GPU
values are not compared (they differ in the last digits between GPUs; Addendum 260).

## 4. What this cannot establish

One machine, one device model, one circuit family; the candidate stack is still not
the release. If the PC sleeps or the run stops, the run is reported as far as it got
and no threshold is scaled after the fact.


---

<!-- ===== Addendum 262 (source: spare-qubit-cliff-addendum-262-2026-09-29.md) ===== -->

> **Note added when merging:** H1-H8 of Addendum 258 confirmed at 100,000 laps at home; R1-R3 confirmed: laps 1-30,000 of all three parts equal the pod's run to the printed digits.

## Addendum 262 -- The PennyLane loop on a fully occupied heavy-hex device over 100,000 laps at home: H1-H8 of Addendum 258 confirmed at the original thresholds, and the first 30,000 laps of all three parts equal the pod's run lap by lap to the printed digits (R1-R3 confirmed) (2026-09-29/30)

**Pre-registered in**: Addendum 261 (pushed as `47809d7` before the run started).
Scored with the locked script's own `score` (H1-H8, the original 100,000-lap thresholds of
Addendum 258, Part I) and with [`benchmarks/compare_100k_home_pod.py`](../../benchmarks/compare_100k_home_pod.py) (R1-R3; written after
the run, applying the thresholds of Addendum 261 exactly).

**Run (home):** 2026-09-29 13:58:27 to 16:45:09 UTC (2 h 47 min for the three parts in
parallel). WSL2, AMD Ryzen 5 5500 (12 threads), NVIDIA GeForce RTX 4070 12 GB (Windows
driver 616.92), Python 3.12.13, Qiskit 2.5.2, PennyLane 0.45.1, `lightning.gpu`. Repository at
`47809d7`. The environment record shows the locked scripts (`dad56b1b...`, `a0081c29...`),
the release core 2026-09-28.1 (repository `lib.rs` `bf3bf537...`), the candidate core
2026-09-29.1 (`5364630e...`), the release layout (`a639efde...`) and the candidate layout
(`e25952a3...`); each part's log prints its versions. **Times are home times.**

**One recording gap:** the environment record says `run script ?` instead of the run
script's hash. The script changed into the output folder before hashing itself by its
relative path. The script that ran is `benchmarks/pod/run_100k_home_2026-09-29.sh` as
committed in `47809d7` with Addendum 261 (normalized SHA-256 `02177d34...`, stated there);
the run was started from that repository checkout.

## 1. Scoring

**C0:** all parts 100,000 laps; versions as expected; `lightning.gpu`; the RX(1e-6) control
detected (smallest 2.87e-7) -> passed.

| ID | Prediction (Addendum 258, Part I) | Result |
|---|---|---|
| H1 | A within 1 s in >= 99,990 of 100,000 | **confirmed**: 100,000 (median 13.1 ms, max 131 ms) |
| H2 | A swap-free and mapped back in 100,000 | **confirmed**: 100,000, 51 two-qubit gates every lap |
| H3 | candidate core fallbacks in A + B: 0 | **confirmed**: 0 |
| H4 | A's drift at lap 100,000 <= 2 x the line fitted to laps 1-1,000 | **confirmed**: 2.951e-10 against 2.915e-10 extrapolated (ratio 1.01) |
| H5 | whole-circuit GPU check at every checkpoint of A and B <= 1e-8 | **confirmed**: max 8.04e-11 |
| H6 | RSS growth from lap 1,000 to the end <= 100 MB in every part | **confirmed**: 0, 0, 0 MB |
| H7 | B / C block distance at lap 100,000 in 0.5-2 | **confirmed**: 3.018e-10 / 3.018e-10 = 1.00 |
| H8 | A's last-10% / first-10% median compile time <= 1.5 | **confirmed**: 1.00 |

| ID | Prediction (Addendum 261) | Result |
|---|---|---|
| R1 | A's block distance equals the pod's at laps 1-30,000 | **confirmed**: identical as printed at 30,000 of 30,000 laps |
| R2 | the same for B and C | **confirmed**: 30,000 of 30,000 in each |
| R3 | two-qubit count, mapped flag and fallbacks equal the pod's at laps 1-30,000 | **confirmed**: equal at every lap of A, B and C |

Reported without prediction: release core fallbacks in C: 0; C mapped back on all 100,000
laps. Compile medians A 13.1, B 12.5, C 12.9 ms; p99 100.5, 21.8, 105.2 ms.

## 2. Reading

1. **The 100,000-lap claim of Addendum 258 is now established at the original
   thresholds**, on a second machine. Amendment 1's 30,000-lap stop (Addendum 259) is no
   longer the limit of the record.
2. **Drift stays linear to 100,000 laps:** about 2.95e-15 per lap on A, 2.95e-10 at the
   end, within 1% of the straight line from the first 1,000 laps. The whole-circuit check
   stays at 8e-11. No acceleration, no memory growth, no slow-down.
3. **Machine independence:** the home run reproduces the pod's run lap by lap (block
   distance to 7 significant digits, gate counts, flags) for all 30,000 overlapping laps of
   all three parts, although GPU, CPU, OS and both core builds differ. With Addendum 260
   (500 laps) this is a strong reproducibility result for the candidate stack.
4. **Candidate and release at spare 4** end at the same distance (3.018e-10) but are not
   identical lap by lap (13,059 of 100,000 laps equal as printed; they differ from lap 1,
   6.906e-15 against 7.326e-15). The two layout modules place the circuit differently at
   spare 4 (the candidate takes the new short-path shortcut), so the compiled circuits
   presumably differ in their placement and single-qubit parts (not examined); the drift
   rate is the same.

## 3. What this does not establish

The candidates are still not the release (this is not the adoption check). One device
model (FakeAuckland), one circuit family; no Qiskit L3 arm in this run.

## 4. Files

| File | What it is |
|---|---|
| `benchmarks/pod/run_100k_home_2026-09-29.sh` | the run script (Addendum 261) |
| [`benchmarks/compare_100k_home_pod.py`](../../benchmarks/compare_100k_home_pod.py) | R1-R3 scoring against the pod's CSVs |
| `data/2026-09-29/pl_heavyhex_100k_home/pl_100k_{A,B,C}_home.csv` | every lap (100,000 rows each) |
| `data/2026-09-29/pl_heavyhex_100k_home/pl_100k_{A,B,C}_home.txt` | run logs |
| [`data/2026-09-29/pl_heavyhex_100k_home/pl_100k_score_home.txt`](../../data/2026-09-29/pl_heavyhex_100k_home/pl_100k_score_home.txt), [`compare_100k_home_pod.txt`](../../data/2026-09-29/pl_heavyhex_100k_home/compare_100k_home_pod.txt) | H1-H8 and R1-R3 scoring |
| [`data/2026-09-29/pl_heavyhex_100k_home/home_env_100k.txt`](../../data/2026-09-29/pl_heavyhex_100k_home/home_env_100k.txt), [`run_100k_home_log.txt`](../../data/2026-09-29/pl_heavyhex_100k_home/run_100k_home_log.txt) | environment and console log |


---

<!-- ===== Addendum 263 (source: spare-qubit-cliff-addendum-263-2026-09-30.md) ===== -->

> **Note added when merging:** Workplace pre-registration (RunPod B200): a go/no-go test of vLLM x PSF-Zero with explicit stop-loss criteria (G1-G3). From 2026-09-30 the vLLM line is on the record, by the owner's decision. Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run. Document names used in Addenda 263-270: `vllm-invest-preregistration` = 263, `vllm-invest-results` = 264, `vllm-w3-pilot-exploratory` = 265, `vllm-v8-pilot2-exploratory` = 266, `vllm-v9-pilot3-exploratory` = 267, `vllm-invest2-preregistration` = 268, `vllm-invest2-results` = 269, `vllm-v10-eval-preregistration` (with `-rev1` and `-amendment1`) = 270.

## Addendum 263 -- Pre-registration: is vLLM x PSF-Zero worth further investment? A go/no-go test with explicit stop-loss criteria on one RunPod B200 (2026-09-30)

**Status: pre-registration, locked at the Project save time of this
document**, before any scored run. Designed and dry-run in the workplace
sandbox. The scored run is on a RunPod B200 pod. No IBM account, no network
access to IBM, no QPU: the device is the FakeAuckland snapshot.

## 1. Why, and what changes today

Until yesterday the vLLM line was exploratory and unrecorded (hand-off 4-8,
`work_2026-09-29_e2e_vllm_exploratory.zip`). From today it is **on the
record**, by the owner's decision, with a **stop-loss rule**:

- the test gives the idea the strongest conditions we can buy today (the
  largest GPU on offer, the strongest open models that fit it, unquantized
  where possible), so that a negative result cannot be blamed on weak
  conditions;
- failures are recorded like successes;
- if the criteria below say it is not worth it, the line is **cut
  immediately**. Reopening it needs a new pre-registration.

Cost is not a constraint (owner, 2026-09-30). The only limits are the wall
time rules in section 5.

What the exploratory runs of 2026-09-29 showed (7B, pod RTX 4090, v1-v3;
home RTX 4070, v5), for context only: GHZ5 and Bell3 solved; W3 and QFT3
never solved by the 7B model; PSF-Zero compiled in 15-27 ms and the model
took more than 90% of the time; on the fully filled 27-qubit task, free-text
7B circuits compiled by PSF-Zero in about 12 ms with 17 two-qubit gates
against Qiskit L3's 6.4 s and 20 (home), but circuits built from triangles
compiled to 48 against L3's 39 (home, `--structured`), the known limit of
the c1 short-path shortcut.

## 2. Design ([`e2e_vllm_psf_v6.py`](../../benchmarks/e2e_vllm_psf_v6.py), `run_invest_2026-09-30.sh`, [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py))

**Hardware.** One RunPod B200 (183 GB, driver 595.91.07), 192 CPUs, 2 TB RAM,
300 GB disk. All times in this test are **B200-pod times** and are not
compared with any other machine.

**Models** (served one at a time by vLLM, `--gpu-memory-utilization 0.90
--max-model-len 32768`, `VLLM_USE_FLASHINFER_SAMPLER=0`):

| tag | model | reply budget |
|---|---|---|
| qwen7b | Qwen/Qwen2.5-7B-Instruct (bf16) | 4,000 tokens; the baseline that failed yesterday |
| qwen72b | Qwen/Qwen2.5-72B-Instruct (bf16, not quantized) | 4,000 tokens |
| gptoss120b | openai/gpt-oss-120b (as published) | 16,000 tokens, `reasoning_effort=high` |

**Tasks** (the target state is computed by the script and never shown to
the model):

| task | qubits | what |
|---|---|---|
| ghz5 | 5 | (&#124;00000> + &#124;11111>)/sqrt(2) |
| w3 | 3 | W state |
| bell3 | 6 | three Bell pairs |
| qft3 | 3 | QFT of &#124;101> |
| fill27 | 27 | the whole FakeAuckland device filled: seven GHZ-3 and three Bell pairs |

**Per task and run, up to 6 rounds:**

1. The model reasons and gives a JSON circuit (free text; the last JSON
   block counts). A few gate aliases are accepted (cnot, sdag, ...);
   parameters on parameter-free gates are ignored and reported back.
2. Logical check on the CPU (lightning.qubit), component by component (the
   target is a product over groups).
3. PSF-Zero `compile_for_hardware` (candidate core 2026-09-29.1 + candidate
   layout 2026-09-29.c1, `layout_search=True`, `entangling_basis="cx"`,
   synthesis cache cleared before each compile) and Qiskit
   `transpile(optimization_level=3)` on the same input.
4. Compiled check of the PSF output, read at the final layout.
5. Feedback: fidelities, the wrong groups with their amplitudes next to the
   target's, and for tasks of at most 4 qubits the state after every gate.
   The temperature starts at 0.2 and rises by 0.3 (up to 1.0) when the same
   wrong answer or the same error repeats.

- **Solved** = compiled fidelity >= 0.9999 in any round.
- **Early stop:** after a solve, stop when two further rounds do not improve
  the device two-qubit count.
- **Request seed** = 1000 x run + round. Each task-run is its own process;
  the 15 task-runs of a model run in parallel against its server.
- History keeps the system prompt, the task and the last two exchanges.
- **HTTP 400** (context): retry once with only the task and the last
  feedback, at half the reply budget.
- Other HTTP errors cost the round and are recorded.
- An unreachable server ends the task-run.

**Baseline for G2.** Qiskit `StatePreparation` of each group's target (qubits
passed in reverse, because the script's qubit 0 is the most significant
bit), decomposed to cx + u at level 0, compiled by PSF-Zero and by L3 the
same way, with its compiled fidelity checked.

**G3 re-timing pass.** After all model runs, with nothing else running,
every distinct valid fill27 circuit from all models is compiled by PSF-Zero
and by L3, **5 times each, sequentially**, and the median is used. The
per-round times recorded during the parallel runs are reported, but they are
not used for G3.

**Scale.** 3 models x 5 tasks x 3 runs = 45 task-runs, at most 270 model
calls.

## 3. Criteria (scored only by [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py), from files)

**G1: can the best model do the job?**

- Best model = most solved task-runs out of 15. Ties go to more w3 + qft3
  solves.
- **go:** >= 12/15 solved, and w3 >= 2/3, and qft3 >= 2/3.
- **stop:** <= 7/15.
- Otherwise ambiguous.

**G2: does the model add anything over a generic method?** On the best
model's solved task-runs, compare the device two-qubit count (PSF-Zero) of
its best circuit with the StatePreparation baseline compiled by PSF-Zero.

- **go:** the model is <= baseline on >= 80% of them.
- **stop:** the model is worse on > 50% of them, or nothing is solved.
- Otherwise ambiguous.

**G3: does PSF-Zero matter inside the loop, at full-device scale?** Over the
distinct fill27 circuits from all models that are correct after compiling
(re-timing pass):

- **go:** median PSF-Zero time <= 1 s, and median per-circuit time ratio
  L3/PSF-Zero >= 10, and PSF-Zero two-qubit count <= L3 on >= 80% of the
  circuits.
- **stop:** median ratio < 3, or PSF-Zero > L3 on > 50% of the circuits, or
  median PSF-Zero time > 1 s, or no correct fill27 circuit at all.
- Otherwise ambiguous.

**Decision.**

- **INVEST** only if G1 = go, G2 != stop and G3 = go.
- **Anything else is CUT**, including any ambiguous G1 or G3.
- A CUT is final for this line. It is recorded with the failures, and
  reopening needs a new pre-registration with a different premise (not just
  a bigger model).

## 4. Expectations (written before the run; they do not change the decision)

- **P1.** qwen7b repeats yesterday: ghz5 and bell3 solved in every run; w3
  and qft3 solved in at most 1 of 3 runs each.
- **P2.** At least one of qwen72b and gptoss120b solves w3 in >= 2 of 3 runs.
- **P3.** On correct fill27 circuits, PSF-Zero is >= 10x faster than L3
  (median). Whether its two-qubit count is <= L3 depends on the circuit
  shape (paths versus triangles); no prediction is made.
- **Overall:** no prediction. G1 is the open question.

## 5. Wall time rules and failures

- A model whose server is not ready within 40 minutes is recorded as a
  server failure ([`SERVER_FAILED.txt`](../../data/2026-09-30/vllm_invest/smoke_run/qwen7b/SERVER_FAILED.txt) with the log tail). Its 15 task-runs
  count as not solved.
- Each task-run process is killed after 90 minutes. A missing result counts
  as not solved.
- **Nothing is re-run to improve a score.** A re-run is allowed only for an
  infrastructure fault outside the model (for example, the pod restarting).
  It must be recorded as an amendment before the re-run, and the first
  attempt's files are kept.

## 6. Dry runs (disclosed)

- **Sandbox, mock model** (canned replies):
  - all five tasks run end to end;
  - the checker catches a wrong W3 parameter (fidelity 0.889) and an
    incomplete fill27 circuit (0.0078 = 1/2 x (1/4)^3);
  - the correct mock circuits reach fidelity 1 - 1e-14.
- **Sandbox, fake HTTP server:**
  - an HTTP 400 is followed by a retry at half the budget, with seeds
    2001, 2001, 2002, ...;
  - an HTTP 500 costs one round;
  - an empty reply is recorded as an error;
  - a triangle fill27 circuit is checked correctly (PSF 38 = L3 38).
- **Sandbox, full run script with a fake server** (2 models x 2 tasks x
  1 run): server start and stop, the re-timing pass, the scorer and the zip
  all worked.
- **Sandbox timing note** (not a result): fill27 L3 took 4-8 s against
  PSF-Zero's 13-100 ms, on 2 sandbox CPUs.
- **Pod smoke run** (`SMOKE=1`: qwen7b, ghz5 and fill27, run 9, 2 rounds,
  into `~/invest_smoke_0930`), allowed after this lock. It checks the real
  server path on the B200 and **is not scored**. Its output is kept and
  handed off.

## 7. What this will not establish

- Anything about real hardware fidelity (no QPU).
- Models other than the three listed, or prompting other than section 2.
  The structured-output mode of home's v5 is not part of this test.
- Whether the idea could work with fine-tuning or tools. A CUT here means
  "not worth investing on this premise".

## 8. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/e2e_vllm_psf_v6.py`](../../benchmarks/e2e_vllm_psf_v6.py) | `4b24734fd315f188fb14d3e67253c55819e4dc771bb321f1a3330fb30552fd10` |
| [`benchmarks/score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py) | `1e154fb1caddad67bda4f399f2eaaf6d8665e64c78a1ab7313c37f8a17c986ee` |
| `benchmarks/pod/run_invest_2026-09-30.sh` | `64446a98cf0073959871ac9e2f53496f6afc108a7672f31cccd6c115cf221168` |

The stack on the pod is the one `setup_gpu_2026-09-29.sh` builds:

- `psf_compile` 2026-09-28.1;
- candidate core 2026-09-29.1;
- candidate layout 2026-09-29.c1.

Setup on this pod printed `SETUP DONE` with both cores and
`pl_heavyhex_gpu.py a0081c29...`.

---

<!-- ===== Addendum 264 (source: spare-qubit-cliff-addendum-264-2026-09-30.md) ===== -->

> **Note added when merging:** Decision CUT: G1 ambiguous (gpt-oss-120b 12/15, but W3 0/3: all 18 of its W3 rounds used the 16,000-token reply budget without giving a circuit), G2 go, G3 go (on correct 27-qubit circuits PSF-Zero 10-21 ms against Qiskit L3's 5.9-11.1 s, 17 against 20 two-qubit gates). The CUT stands as the result of these conditions; the line was reopened only under a new pre-registration (Addendum 268). Data: `data/2026-09-30/vllm_invest/` (pod outputs unpacked).

## Addendum 264 -- Results: vLLM x PSF-Zero go/no-go on one RunPod B200. Decision: CUT (2026-09-30)

**Pre-registration:** `vllm-invest-preregistration-2026-09-30.md`, locked at
its Project save time (2026-09-30 about 00:11 UTC), before any run on the
pod. The scored run started at 00:33 UTC and finished at 01:13 UTC.

**Nothing was re-run to improve a score.** The locked scripts ran unchanged:
`env.txt` on the pod records the same raw SHA-256 values that were checked
before the run (`05e2ee77...`, `0b1136e8...`, `0434e66d...`).

## 0. In one line

Under the pre-registered conditions the line is **not worth further
investment**, and it is **cut**.

- **G1 = ambiguous.** The best model, gpt-oss-120b, solved 12/15 task-runs
  but W3 0/3. No model solved W3 in any run.
- **G2 = go.**
- **G3 = go.**

The rule is that anything other than G1 = go and G3 = go is CUT.

Two results stand independently of the cut:

- On the fully filled 27-qubit device, every correct circuit compiled by
  PSF-Zero in 10-21 ms against Qiskit L3's 5.9-11.1 s (500-660x). PSF-Zero
  used 17 two-qubit gates against L3's 20 on all 7 distinct correct
  circuits.
- The deciding failure was a budget failure, not a wrong answer: in all 18
  W3 rounds gpt-oss-120b used its whole 16,000-token reply budget on
  reasoning and never produced a circuit. Section 3 says what that does and
  does not mean.

## 1. Environment (B200 pod; times are B200-pod times, not compared with any other machine)

- **Hardware:** NVIDIA B200 (183,359 MiB, driver 595.91.07); 192 CPUs (Intel
  Xeon Platinum 8568Y+); 2 TB RAM.
- **Software:** vLLM 0.30.0, torch 2.13.0+cu130, Qiskit 2.5.2, PennyLane
  0.45.1.
- **PSF-Zero stack:** psf_compile 2026-09-28.1, candidate core 2026-09-29.1,
  candidate layout 2026-09-29.c1, built by `setup_gpu_2026-09-29.sh`.
- **GPU memory in use** with each server up: 163,796 MiB (7B), 163,802 MiB
  (72B) and 165,102 MiB (gpt-oss-120b).
- **Server start:** gpt-oss-120b took 932 s to be ready, inside the 40-minute
  limit.
- **Environment additions made before the scored run** (disclosed; none of
  them changes a locked file):
  1. The first smoke attempt failed at server start: FlashInfer's JIT build
     found no `ninja`.
  2. `ninja` was installed into the vLLM venv and linked into
     `/usr/local/bin`.
  3. `/usr/local/cuda/bin` (nvcc 12.8.93) was put on `PATH`, with
     `CUDA_HOME=/usr/local/cuda`, in the terminal that then ran both the
     second smoke attempt and the scored run.
- **Smoke run** (disclosed, not scored; qwen7b, ghz5 and fill27, run 9, 2
  rounds):
  - fill27 was solved;
  - ghz5 was not. The model used H + CZ chains and CRY chains, and the
    fidelities 0.25 and 0.2608 were checked by hand.
  - [`SERVER_FAILED.txt`](../../data/2026-09-30/vllm_invest/smoke_run/qwen7b/SERVER_FAILED.txt) in the smoke zip is from the first (ninja) attempt.
    Its `vllm_server.log` was overwritten by the second attempt.

## 2. Results (scored by [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py), from files)

| model | solved /15 | ghz5 | w3 | bell3 | qft3 | fill27 | reply errors | HTTP 400 | missing |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | **12** | 3/3 | **0/3** | 3/3 | 3/3 | 3/3 | 24 | 0 | 0 |
| Qwen2.5-72B | 9 | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 | 4 | 0 | 0 |
| Qwen2.5-7B | 7 | 3/3 | 0/3 | 2/3 | 0/3 | 2/3 | 4 | 0 | 0 |

**Model time**, summed over the task-runs of each model (the 15 ran in
parallel):

- 7B: 377 s (median 2.8-5.9 s per call by task);
- 72B: 1,731 s (11-35 s per call);
- gpt-oss-120b: 2,731 s (8-84 s per call).

**G1 = ambiguous.** gpt-oss-120b solved 12/15 and QFT3 3/3, but W3 was 0/3.
The go rule needs W3 >= 2/3.

**G2 = go.** gpt-oss-120b's device two-qubit count was at most the
StatePreparation baseline on 11 of its 12 solved task-runs:

| task | model (PSF) | baseline (PSF) | baseline (L3) |
|---|---|---|---|
| bell3 | 3 | 3 | 3 |
| ghz5 | 4 | 47 | 34 |
| fill27 | 17 | 52 | 46 |
| qft3 run 1 | 0 | 7 | 3 |
| qft3 run 2 | 3 | 7 | 3 |
| qft3 run 3 | 12 | 7 | 3 |

- QFT of a basis state is a product state. gpt-oss-120b found the
  zero-two-qubit circuit in run 1.
- Run 3 used the textbook QFT, which is the one case worse than the
  baseline.

**G3 = go.** The re-timing pass ran sequentially with nothing else running,
5 repetitions each, and the median is used. It found 13 distinct valid
fill27 circuits, 7 of them correct.

- **Correct circuits:**
  - PSF-Zero median 16.9 ms (range 9.7-21.3 ms);
  - L3 5.87-11.10 s;
  - per-circuit ratio 501-659x (median 612x);
  - two-qubit count PSF-Zero 17 against L3 20 on all 7;
  - no swaps in any of them.
- **Wrong circuits (not scored):**
  - 15 = 15 on three circuits;
  - 30 = 30 on one;
  - PSF-Zero lower on two (34 against 37).

**Decision: CUT.**

**Expectations** (from section 4 of the pre-registration):

| | expectation | outcome |
|---|---|---|
| P1 | qwen7b: ghz5 and bell3 solved every run; w3, qft3 at most 1/3 | **not confirmed**: bell3 2/3 (run 3 ended at fidelity 0.0156). The rest held (w3 0/3, qft3 0/3). |
| P2 | qwen72b or gpt-oss-120b: w3 >= 2/3 | **not confirmed**: both 0/3 |
| P3 | PSF-Zero >= 10x faster than L3 on correct fill27 | **confirmed**: minimum 501x |

## 3. The failures, as they are

**gpt-oss-120b, W3 (the deciding cell).**

- In all 18 rounds (3 runs x 6) the reply ended with `finish_reason=length`
  after 41,920-52,708 characters of reasoning, and no JSON was produced.
- The same happened in 6 of its 16 QFT3 rounds, but QFT3 was still solved
  3/3.
- The 16,000-token budget was pre-registered and was the same in every
  round.

What this means:

- By the rules, running out of budget is model behaviour, not an
  infrastructure fault, so it counts as not solved and is not re-run.
- **The budget was decisive.** Had gpt-oss-120b solved W3 in 2 of 3 runs, G1
  would have been go, and with G2 go and G3 go the decision would have been
  INVEST.
- All 18 rounds ran out in the same way, so the model did not look close to
  an answer. Even so, whether a larger budget would have produced a correct
  W3 was not tested and is not known.
- Raising the budget and running again would chase a pass by changing the
  conditions. Under section 3 of the pre-registration that needs a new
  pre-registration with a different premise.

**Qwen2.5-72B and 7B, W3.** Both answered every round with wrong circuits:

- best compiled fidelity 0.037-0.094 (72B) and 0.333 (7B);
- 72B also used `ccx` once.

**QFT3.**

- 72B always used the textbook circuit on the wrong input (best 0.568).
- 7B never exceeded 0.071.

**Reply errors that were not the model's reasoning.** Qwen2.5-72B wrote
parameters twice as `2 * Math.acos(...)` (capital M), which the parser does
not accept.

- Evaluated by hand, both circuits give W3 fidelity 0.037, so neither
  acceptance would have changed a count.
- The other errors are the model's: an unknown gate (`crx`, `ccx`), qubit
  index 3 on a 3-qubit task, and `h` on more than one qubit.

## 4. Checks of the harness (workplace sandbox, from the zip)

- The zip (`invest_outputs_0930.zip`, SHA-256 `c7a976d6...`, 318 files)
  matches its MANIFEST: 0 mismatches.
- Re-scoring the files with the locked scorer gives the same verdicts and
  numbers. The only differences are float summation order in four sums, at
  the 1e-13 level.
- An independent numpy simulator (no PennyLane, no Qiskit;
  [`indep_check_2026-09-30.py`](../../benchmarks/indep_check_2026-09-30.py)) recomputed the logical fidelity of every
  parsed circuit:
  - 141 circuits on the small tasks: largest difference 4.4e-16, and 0
    disagreements on solved or not solved;
  - 32 fill27 circuits, checked group by group: difference 0.

## 5. What this does and does not establish

**It establishes** that under these conditions a vLLM-served open model
coupled to PSF-Zero does not reliably solve the pre-registered five tasks:

- the best open models that fit one B200, unquantized, with free-text
  reasoning, 6 rounds of state feedback, and 4,000 tokens (Qwen) or 16,000
  tokens with high reasoning effort (gpt-oss) per reply;
- in particular, none of them produced a 3-qubit W state in any of 9
  task-runs.

**It does not establish:**

- that the idea fails with a larger reply budget, tools (for example a
  simulator the model can call), fine-tuning, or other models;
- anything about real hardware.

**It separately establishes, for PSF-Zero itself** (B200-pod CPU,
sequential): on correct fill27 circuits written by the models, PSF-Zero
compiles in about 17 ms against L3's 6-11 s, with fewer two-qubit gates (17
against 20).

## 6. Files

- **Pod outputs:** `invest_outputs_0930.zip`, per model and run:
  - `rounds.jsonl` (every reply, feedback and number);
  - `result.json`;
  - best circuits;
  - server logs;
  - `retime_fill27.csv`;
  - `score.md` and `score.json`;
  - `env.txt`.
- **Smoke run:** `invest_smoke_0930.zip` (SHA-256 `2f9036bf...`).
- **Locked scripts:**
  - [`benchmarks/e2e_vllm_psf_v6.py`](../../benchmarks/e2e_vllm_psf_v6.py);
  - [`benchmarks/score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py);
  - `benchmarks/pod/run_invest_2026-09-30.sh`.
- **Independent check:** [`benchmarks/indep_check_2026-09-30.py`](../../benchmarks/indep_check_2026-09-30.py).

---

<!-- ===== Addendum 265 (source: spare-qubit-cliff-addendum-265-2026-09-30.md) ===== -->

> **Note added when merging:** Exploratory, after the CUT of Addendum 264, not scored: v7 (32,000-token replies, the cp gate, a short-reasoning prompt) on W3 only, RunPod H200: high effort 3/3, medium 1/3. Data: `data/2026-09-30/vllm_w3_pilot/`.

## Addendum 265 -- Exploratory (not pre-registered, not scored): W3 pilot of v7 with gpt-oss-120b on a RunPod H200 (2026-09-30)

**Status: exploratory.** This pilot came after the CUT of
`vllm-invest-results-2026-09-30.md`, at the owner's request ("improve only,
then look at the weak point with a short test first"). It does not change
that decision. Its only use is to decide whether to pre-register one new
run.

## Why

The deciding failure of the scored run was W3 with gpt-oss-120b. In all 18
rounds it used the whole 16,000-token reply budget on reasoning and gave no
circuit. v7 changes only the three points named after the CUT:

1. reply budget 32,000 tokens;
2. the controlled-phase gate `cp` is allowed;
3. the system prompt asks for short reasoning and always a final JSON.

Everything else is v6.

## Setup

- **Hardware:** RunPod **H200** (143,771 MiB, driver 595.91.07), 96 CPUs,
  2 TB RAM. B200 was out of capacity. GPU times are H200-pod times.
- **Software:** vLLM 0.30.0 (pinned, as on the B200), torch 2.13.0+cu130,
  the same PSF-Zero stack (setup printed `SETUP DONE`, both cores).
- **Server:** `--max-model-len 40960`; ready after 171 s.
- **Scripts:**
  - [`e2e_vllm_psf_v7.py`](../../benchmarks/e2e_vllm_psf_v7.py) (raw SHA-256 `8dd7a285...`);
  - `pilot_w3_2026-09-30.sh` (`c2c16432...`).
- **Design:** gpt-oss-120b, task w3 only, `reasoning_effort` high and medium,
  3 runs each, up to 6 rounds, 6 task-runs in parallel.
- **Outputs:** `pilot_outputs_0930.zip` (SHA-256 `2624fd49...`).

## What happened

| arm | run | solved | first exact round | rounds that hit the 32,000-token limit (no answer) |
|---|---|---|---|---|
| high | 1 | yes | 1 | 2 of 6 |
| high | 2 | yes | 2 | 4 of 6 |
| high | 3 | yes | 5 | 4 of 6 |
| medium | 1 | yes | 2 | 0 of 4 |
| medium | 2 | no (best 0.742) | - | 0 of 6 |
| medium | 3 | no (best 0.445) | - | 0 of 6 |

**Arm totals:** high **3/3** solved, medium **1/3**.

**Compared with the scored run** (v6, 16,000 tokens, B200): 0/3 solved, and
18/18 rounds hit the limit.

**The correct circuits** are the standard construction:

- RY(1.231) = RY(2 acos sqrt(2/3)) on qubit 0;
- a controlled RY(pi/2) with X conjugation;
- then CNOTs to distribute the excitation.

Device two-qubit counts after PSF-Zero were 4-7, against the StatePreparation
baseline's 7.

**Independent check.** A numpy simulator without PennyLane or Qiskit
([`indep_check_2026-09-30.py`](../../benchmarks/indep_check_2026-09-30.py) plus `cp`) recomputed all 24 parsed circuits:
the largest difference was 2.8e-17, and it found the same 9 correct circuits.

## What it suggests, and what it does not

**What it suggests:**

- With the three changes, the high arm now reaches a correct W3 in every
  run, within the 6-round cap.
- Medium answers every round but is less accurate.

**What it does not show:**

- **The high arm is still fragile:** 10 of its 18 rounds still hit the
  32,000-token limit. Run 3 needed 5 rounds.
- **The sample is small:** three runs are not enough to be confident of the
  >= 2/3 pre-registered bar.
- **Nothing about the other four tasks or the full criteria.** Only W3 was
  run.
- **A different GPU** (H200, not B200). The model and vLLM version are the
  same.

**Next, if the owner agrees:**

- one new pre-registered run with v7 unchanged (high, 32,000 tokens) on the
  H200;
- gpt-oss-120b plus Qwen2.5-7B as the control;
- all five tasks x 3 runs, with the same G1-G3 criteria and decision rule.

No further tuning on W3 before that run.

---

<!-- ===== Addendum 266 (source: spare-qubit-cliff-addendum-266-2026-09-30.md) ===== -->

> **Note added when merging:** Exploratory, not scored: v8 (a salvage request after a cut reply, best-circuit memory): W3 high 3/3, W3 medium 1/3, QFT3 high 3/3, all three with zero two-qubit gates. Data: `data/2026-09-30/vllm_v8_pilot2/`.

## Addendum 266 -- Exploratory (not pre-registered, not scored): pilot 2 of v8 with gpt-oss-120b on a RunPod H200 (2026-09-30)

**Status: exploratory**, like pilot 1 (`vllm-w3-pilot-exploratory-2026-09-30.md`).
It does not change the CUT of `vllm-invest-results-2026-09-30.md`. It is the
last tuning step on W3. The next step, if any, is one pre-registered run on
all five tasks with fresh seeds.

## What changed from v7 (two hints from pilot 1)

4. **Salvage.** When a reply ends at the token limit with no JSON, one short
   follow-up request is sent in the same round (reasoning effort low, 4,000
   tokens): "give your best circuit now".
   - Pilot 1 lost 10 of 18 high rounds this way.
5. **Memory.** The feedback carries the best circuit so far and its fidelity.
   - The history keeps only the last two exchanges, and pilot 1 showed answers
     oscillating and regressing.

A sandbox test of v8 caught one bug before the pod run: the memory lines
had broken an `if/else`, so correct rounds also received the "NOT correct"
text. It was fixed before any pod use.

## Setup

- **Hardware and software:** the same H200 pod as pilot 1 (vLLM 0.30.0,
  gpt-oss-120b, `--max-model-len 40960`).
- **Seeds:** fresh (runs 4-6).
- **Arms:**
  - w3 at high effort;
  - w3 at medium effort;
  - qft3 at high effort, the other task that had hit the token limit.
- **Scale:** 3 runs each, up to 6 rounds, 32,000-token replies, 9 task-runs
  in parallel.
- **Scripts:**
  - [`e2e_vllm_psf_v8.py`](../../benchmarks/e2e_vllm_psf_v8.py) (raw SHA-256 `2d8a8c49...`);
  - `pilot2_v8_2026-09-30.sh` (`1ac42ad9...`).
- **Outputs:** `pilot2_outputs_0930.zip` (SHA-256 `c2c5f0ec...`).

## What happened

| arm | solved | first exact round per run | rounds cut at the limit | of those, salvaged to a correct circuit |
|---|---|---|---|---|
| w3 high | **3/3** | 1, 2, 1 | 7 of 12 | 2 |
| w3 medium | 1/3 | -, -, 3 | 0 of 17 | - |
| qft3 high | **3/3** | 2, 1, 1 | 0 of 14 | - |

**W3, high effort.**

- 3/3 again, now with **6/6 across the two pilots**. The first exact round
  came earlier than in pilot 1 (1, 2, 1 against 1, 2, 5).
- Every round now yields a circuit. There were no "no JSON" errors, against
  10 in pilot 1's high arm.
- Salvage requests took 1.6-4.8 s each, and 2 of the 7 gave a correct
  circuit.
- The best device two-qubit count reached 4, against the StatePreparation
  baseline's 7.

**QFT3, high effort.**

- 3/3, with no round cut at the limit. v6 on the B200 had 6 of 16 rounds
  cut.
- **All three runs found a zero-two-qubit circuit**, because the QFT of a
  basis state is a product state: H on each qubit plus single-qubit phases.
  On the B200, v6 found such a circuit in 1 of 3 runs.

**W3, medium effort.**

- Still 1/3. The memory did not stop the oscillation: run 4 went 0.55,
  0.74, 0.28, 0.06, 0.87, 0.44.
- Medium answers quickly but does not converge.

**Regressions after a solve.** When asked for fewer two-qubit gates, a
solved run often proposed a wrong circuit (fidelity 0). The best exact
circuit is kept, so the solved count is not affected, but those rounds are
spent.

**Independent check.** A numpy simulator without PennyLane or Qiskit
([`indep_check_2026-09-30.py`](../../benchmarks/indep_check_2026-09-30.py) plus `cp`) recomputed every parsed circuit:

- 29 W3 circuits and 14 QFT3 circuits;
- the recorded fidelities were reproduced exactly;
- the same 10 + 10 correct circuits were found.

## What it suggests, and what it does not

**What it suggests:** with v8 at high effort, gpt-oss-120b now solves the two
tasks it struggled with (W3 6/6 over two pilots, QFT3 3/3 with optimal
circuits) within the 6-round cap.

**What it does not show:**

- **Small samples:** 3 and 6 runs.
- **Only two of the five tasks were run.**
- **W3 still hits the limit in about half of its high rounds.** Salvage
  turns those into answers, but mostly wrong ones.
- **W3 has now been the tuning target twice**, so its numbers are optimistic
  by construction. The scored test must use fresh seeds and all five tasks.
- **A different GPU from the scored run** (H200, not B200).

**Next, if the owner agrees:**

- one pre-registered run with v8 unchanged, at high effort, 32,000 tokens,
  with salvage and memory on;
- on the H200, with gpt-oss-120b plus Qwen2.5-7B as the control;
- all five tasks x 3 runs on fresh seeds;
- the same G1-G3 criteria and decision rule as the 2026-09-30 test.

A CUT there ends the line.

---

<!-- ===== Addendum 267 (source: spare-qubit-cliff-addendum-267-2026-09-30.md) ===== -->

> **Note added when merging:** Exploratory, not scored: v9 (stronger salvage, 48,000-token replies, an optional simulator tool): W3 without the tool 3/3; the tool slowed convergence (up to 168,000 tokens and 25 minutes in one round) and was dropped. W3 was the tuning target of all three pilots (Addenda 265-267), so their W3 numbers are optimistic. Data: `data/2026-09-30/vllm_v9_pilot3/`.

## Addendum 267 -- Exploratory (not pre-registered, not scored): pilot 3 of v9 with gpt-oss-120b on a RunPod H200 (2026-09-30)

**Status: exploratory**, like pilots 1 and 2 (`vllm-w3-pilot-exploratory-2026-09-30.md`,
`vllm-v8-pilot2-exploratory-2026-09-30.md`). The CUT of
`vllm-invest-results-2026-09-30.md` stands. This was the third tuning step,
at the owner's request ("if there is still room, improve again and
retest").

## What changed from v8 (hints from pilot 2)

6. **Salvage** at medium effort and 8,000 tokens. At low effort it had given
   2 correct circuits of 7.
7. **Reply budget of 48,000 tokens.** W3 had still hit 32,000 in 7 of 12
   high rounds.
8. **Optional simulate() tool** (`--tool-sim`).
   - The model may simulate its own candidate circuit during a round (up to 8
     calls per round).
   - The tool returns only the amplitudes of that circuit: never the target,
     never a fidelity.
   - The tool's numpy simulator matched PennyLane on 300 random circuits to
     1.6e-16. Here, 88 tool answers were re-checked independently with no
     missing basis state.

## Setup

- **Hardware:** the same H200 pod.
- **Server:** vLLM 0.30.0 with `--tool-call-parser openai
  --enable-auto-tool-choice` ([`server_mode.txt`](../../data/2026-09-30/vllm_v9_pilot3/pod_outputs/server_mode.txt): tools on), and
  `--max-model-len 65536`.
- **Model and seeds:** gpt-oss-120b, reasoning effort high, fresh seeds
  (runs 7-9).
- **Scale:** 10 task-runs in parallel.
- **Scripts:**
  - [`e2e_vllm_psf_v9.py`](../../benchmarks/e2e_vllm_psf_v9.py) (raw SHA-256 `11f747fa...`);
  - `pilot3_v9_2026-09-30.sh` (`cae07919...`).
- **Outputs:** `pilot3_outputs_0930.zip` (SHA-256 `911929ef...`).

## What happened

| arm | solved | first exact round per run | rounds with no usable circuit | typical tokens per round |
|---|---|---|---|---|
| w3, no tool | **3/3** | 2, 3, 1 | 0 of 12 (1 hit the limit; salvage gave a wrong circuit) | 8k-46k |
| w3, tool | 3/3 | 2, **5, 5** | 5 of 17 (ended at the 8-call cap without JSON) | 14k-**168k** |
| guard: ghz5, bell3, fill27 (tool offered) | 3/3 | 1, 1, 1 | 0 | 0.5k-3k (no tool calls) |
| guard: qft3 (tool offered) | 1/1 | 1 | 1 of 4 | 1.4k-53k |

**W3 without the tool.**

- 3/3 solved.
- Across the three pilots, the high arm without the tool has now solved W3
  in **9 of 9 runs**: v7 at rounds 1, 2, 5; v8 at 1, 2, 1; v9 at 2, 3, 1.
- With 48,000 tokens, only 1 of 12 rounds hit the limit.

**W3 with the tool.**

- Also 3/3, but late: runs 8 and 9 first solved in round 5. Round 5 of run 9
  made no tool call.
- **It was much more expensive:** up to 168,000 tokens and 25 minutes in one
  round.
- 5 of 17 rounds used all 8 calls and then gave no circuit.
- In this form the tool made the model explore, not converge.

**Guard tasks** (not used for tuning so far):

- ghz5, bell3 and fill27 were solved in round 1 with small replies. The
  model did not call the tool on them.
- qft3 was solved in round 1. In later rounds, while trying to remove
  two-qubit gates, it called the tool 7-8 times and lost those rounds.
- No regression was seen on the untuned tasks.

**Independent check.** A numpy simulator without PennyLane or Qiskit
recomputed every parsed circuit (w3 24, qft3 3, ghz5 4, bell3 3):

- largest difference 4.4e-16;
- same correct circuits (14, 2, 4, 3).

## What it suggests, and what it does not

**What it suggests:**

- The useful v9 change is the larger budget, together with the stronger
  salvage.
- The simulator tool, as offered here, does not help. It slows convergence
  and multiplies token use.
- If a scored run follows, it should use v9 **without** `--tool-sim`.

**What it does not show:**

- **Small samples** (3 runs per arm).
- **W3 has now been the tuning target three times**, so its numbers are
  optimistic by construction.
- **Only one guard run per untuned task.**
- **Other tool designs** (fewer calls, or a tool that answers yes/no to
  "is this the target") were not tried. A yes/no tool would leak the target
  and is not fair.

**Proposal:**

- stop tuning;
- one pre-registered run with v9 (no tool, high effort, 48,000 tokens) on
  the H200;
- gpt-oss-120b plus Qwen2.5-7B as the control;
- all five tasks x 3 runs on fresh seeds;
- the same G1-G3 and decision rule.

The owner's idea of feedback in words (a verbal description of how the
state differs, generated mechanically, not how to fix it) could be a second,
pre-registered arm in that run, instead of another W3 pilot.

---

<!-- ===== Addendum 268 (source: spare-qubit-cliff-addendum-268-2026-09-30.md) ===== -->

> **Note added when merging:** Workplace pre-registration (RunPod H200) of a second go/no-go under a new premise: the v9 harness without the tool, gpt-oss-120b and Qwen2.5-7B, five tasks x 3 runs on fresh seeds, criteria unchanged from Addendum 263; a CUT here would end the line. Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 268 -- Pre-registration: second vLLM x PSF-Zero go/no-go, v9 without the tool, on a RunPod H200 (2026-09-30)

**Status: pre-registration, locked at the Project save time of this
document**, before any scored run of this test. No IBM account, no network
access to IBM, no QPU: the device is the FakeAuckland snapshot.

## 1. Why a second test, and what is different

The first test of today (`vllm-invest-preregistration-2026-09-30.md`,
results in `vllm-invest-results-2026-09-30.md`) ended in **CUT**:

- G1 was ambiguous: gpt-oss-120b solved 12/15 but W3 0/3, because all 18 of
  its W3 rounds used the 16,000-token budget on reasoning;
- G2 and G3 were go.

That CUT stands as the result of that test. Its rule said that reopening
needs a new pre-registration with a different premise. This is that
pre-registration.

**The new premise:** the harness changes found in three exploratory pilots
(all on the H200, gpt-oss-120b, recorded in
`vllm-w3-pilot-exploratory-2026-09-30.md`,
`vllm-v8-pilot2-exploratory-2026-09-30.md` and
`vllm-v9-pilot3-exploratory-2026-09-30.md`):

1. short-reasoning prompt with a mandatory final JSON;
2. the controlled-phase gate `cp`;
3. a larger reply budget (48,000 tokens for gpt-oss);
4. a salvage request when a reply is cut at the limit;
5. the best circuit so far carried in the feedback.

The simulate() tool tried in pilot 3 is **not** used: it slowed convergence
and multiplied token use. A restricted-tool variant was never tried, so it
is not introduced here either.

**Disclosed tuning history** (why these numbers are optimistic for W3):

- W3 was the tuning target in all three pilots, where the high-effort arm
  without the tool solved it 9 of 9 times;
- QFT3 was piloted once (3/3);
- ghz5, bell3 and fill27 were each piloted once with the tool offered (not
  used), all solved in round 1.

This test uses **fresh seeds** and **all five tasks**, and the criteria are
unchanged from the first test.

## 2. Design ([`e2e_vllm_psf_v9.py`](../../benchmarks/e2e_vllm_psf_v9.py) without `--tool-sim`, `run_invest2_2026-09-30.sh`, [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py))

- **Hardware:** RunPod **H200** (143,771 MiB, driver 595.91.07), 96 CPUs,
  2 TB RAM. The first test ran on a B200; times are H200-pod times and are
  not compared across machines.
- **Software:**
  - vLLM 0.30.0 (as in the first test), torch 2.13.0+cu130;
  - PSF-Zero stack from `setup_gpu_2026-09-29.sh` (psf_compile 2026-09-28.1,
    candidate core 2026-09-29.1, candidate layout 2026-09-29.c1);
  - `ninja`, and `PATH` and `CUDA_HOME` set for `/usr/local/cuda` inside the
    run script.

**Models** (served one at a time, `--gpu-memory-utilization 0.90`):

| tag | model | reply budget | reasoning effort | `--max-model-len` |
|---|---|---|---|---|
| qwen7b | Qwen/Qwen2.5-7B-Instruct | 8,000 | not sent | 32,768 |
| gptoss120b | openai/gpt-oss-120b | 48,000 | high (salvage at medium, 8,000 tokens) | 65,536 |

Qwen2.5-72B is dropped. It was not the best model in the first test, and
the question is whether the deciding model now passes.

- **Tasks:** ghz5, w3, bell3, qft3 and fill27, identical to the first test.
- **Runs:** 3 per task and model. The request seeds use run numbers 11, 12
  and 13 (seed = 1000 x run + round), and results are written to
  `run1`-`run3`.
- **Rounds:** up to 6, with the same solve rule (compiled fidelity >= 0.9999)
  and early stop as in the first test.
- **Parallelism:** the 15 task-runs of a model run in parallel against its
  server.
- **Scoring:** after the model runs, the G3 re-timing pass (5 repetitions,
  median) and scoring by the **unchanged** [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py), from
  files.

## 3. Criteria (unchanged from the first test)

- **G1:**
  - go if the best model solves >= 12/15, with w3 >= 2/3 and qft3 >= 2/3;
  - stop if it solves <= 7/15.
- **G2:** device two-qubit count (PSF) against the StatePreparation baseline
  on the best model's solved task-runs.
  - go if it is <= baseline on >= 80% of them;
  - stop if it is worse on > 50% of them, or nothing is solved.
- **G3** (re-timing pass, correct fill27 circuits):
  - go if the median PSF time is <= 1 s, the median ratio L3/PSF is >= 10,
    and PSF two-qubit count <= L3 on >= 80% of the circuits;
  - stop if the median ratio is < 3, or PSF > L3 on > 50% of the circuits,
    or the median PSF time is > 1 s, or there is no correct circuit.
- **Decision:**
  - **INVEST** only if G1 = go, G2 != stop and G3 = go;
  - anything else is **CUT**, and a CUT here **ends the vLLM line**. No
    further pilots or reopening on this premise.

## 4. Expectations (written before the run; they do not change the decision)

- **P1.** gpt-oss-120b solves ghz5, bell3 and fill27 in every run (as in the
  first test).
- **P2.** gpt-oss-120b solves w3 in >= 2 of 3 runs (pilots: 9/9 on other
  seeds).
- **P3.** gpt-oss-120b solves qft3 in >= 2 of 3 runs.
- **P4.** qwen7b does not solve w3 or qft3 in more than 1 of 3 runs each.
- **Overall:** INVEST is expected but not certain. W3 still hit the limit in
  1 of 12 pilot rounds, and three runs per task leave little margin.

## 5. Wall-time rules and failures

- **Server start:** a server not ready within 40 minutes is recorded as a
  server failure, and its task-runs count as not solved.
- **Task-run limit:** each task-run process is killed after 120 minutes, and
  a missing result counts as not solved.
- **No re-runs to improve a score.** A re-run is allowed only for an
  infrastructure fault outside the model, recorded as an amendment first,
  with the first attempt's files kept.

## 6. Dry runs (disclosed)

- **Sandbox, full run script with a fake vLLM server** (2 models x 5 tasks x
  3 runs, then re-timing, scoring and zip): completed.
  - The decision it printed comes from canned replies and means nothing.
  - The Qwen arm was checked to send no reasoning_effort (salvage effort
    empty).
- **Sandbox, v9 components:**
  - the tool path and its fallback are tested but not used here;
  - salvage and memory were tested in pilot 2 and 3.
- **Pod smoke run** (`SMOKE=1`: qwen7b, ghz5 and w3, seed run 99, 2 rounds,
  into `~/invest2_smoke_0930`), allowed after this lock. It checks the Qwen
  download and server path on the H200 and **is not scored**.

## 7. Locked files (normalized SHA-256; raw SHA-256 in brackets)

| file | normalized | raw |
|---|---|---|
| [`benchmarks/e2e_vllm_psf_v9.py`](../../benchmarks/e2e_vllm_psf_v9.py) (same file as pilot 3) | `5f9f769c3b5d69f338b4296b3cbbf898a97aedfa2817f7323622d371df23eb0b` | `11f747fa...` |
| [`benchmarks/score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py) (unchanged since the morning lock) | `1e154fb1caddad67bda4f399f2eaaf6d8665e64c78a1ab7313c37f8a17c986ee` | `0b1136e8...` |
| `benchmarks/pod/run_invest2_2026-09-30.sh` | `d999fbc4b33acdf48af645b7de0c4437348aa4b9e5d06ccd486f9b69e57a60b9` | `291e263e...` |

---

<!-- ===== Addendum 269 (source: spare-qubit-cliff-addendum-269-2026-09-30.md) ===== -->

> **Note added when merging:** Decision INVEST, at the smallest possible margin: G1 go (14/15; W3 2/3 against a bar of 2/3, on the task used for tuning), G2 go (14/14), G3 go (PSF-Zero 10-11 ms against L3's 7.4 s, 17 against 20 two-qubit gates), the PSF-Zero advantage reproduced on a second machine. The CUT of Addendum 264 remains the result of the v6 conditions. After this result the owner made the vLLM line a continuing project. Data: `data/2026-09-30/vllm_invest2/`.

## Addendum 269 -- Results: second vLLM x PSF-Zero go/no-go (v9 without the tool, RunPod H200). Decision: INVEST (2026-09-30)

**Pre-registration:** `vllm-invest2-preregistration-2026-09-30.md`, locked at
its Project save time (about 05:26 UTC), before any scored run of this test.
The scored run started at 05:34 UTC and finished at 06:06 UTC.

**Nothing was re-run.** `env.txt` records the raw SHA-256 values of the
three locked files as checked before the run:

- [`e2e_vllm_psf_v9.py`](../../benchmarks/e2e_vllm_psf_v9.py): `11f747fa...`;
- [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py): `0b1136e8...`;
- `run_invest2_2026-09-30.sh`: `291e263e...`.

## 0. In one line

Under the pre-registered conditions the second test passes:

- **G1 = go:** gpt-oss-120b solved 14/15, with w3 2/3 and qft3 3/3;
- **G2 = go:** 14/14 solved task-runs at or below the StatePreparation
  baseline;
- **G3 = go:** PSF-Zero 10-11 ms against L3's 7.4 s on the correct fill27
  circuits, with 17 two-qubit gates against 20.

The decision is **INVEST**. The margin on W3 is the smallest possible (2 of 3
against a bar of 2 of 3), and W3 was the tuning target of three pilots; see
section 3.

This result is the second test. It does not erase the first test's CUT
(`vllm-invest-results-2026-09-30.md`), which remains the result of the v6
conditions.

## 1. Environment (H200 pod; times are H200-pod times, not compared with the B200 run)

- **Hardware:** NVIDIA H200 (143,771 MiB, driver 595.91.07); 96 CPUs (Intel
  Xeon Platinum 8568Y+); 2 TB RAM.
- **Software:** vLLM 0.30.0, torch 2.13.0+cu130, Qiskit 2.5.2, PennyLane
  0.45.1.
- **PSF-Zero stack:** psf_compile 2026-09-28.1, candidate core 2026-09-29.1,
  layout 2026-09-29.c1.
- **GPU memory in use** with each server up: 128,705 MiB (7B) and 129,467 MiB
  (gpt-oss-120b).
- **Smoke run** (disclosed, not scored; qwen7b, ghz5 and w3, seed run 99, 2
  rounds): the Qwen download and server path worked on the H200 (ready in
  120 s). Both tasks went unsolved in 2 rounds.

## 2. Results (scored by the unchanged [`score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py), from files)

| model | solved /15 | ghz5 | w3 | bell3 | qft3 | fill27 | reply errors | HTTP 400 | missing | model s (sum) |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | **14** | 3/3 | **2/3** | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 5,937 |
| Qwen2.5-7B | 7 | 3/3 | 0/3 | 3/3 | 0/3 | 1/3 | 0 | 0 | 0 | 278 |

**G1 = go.** gpt-oss-120b solved 14/15; w3 2/3 and qft3 3/3 meet the bar.

**G2 = go.** The best circuit was at most the StatePreparation baseline
(device two-qubit count after PSF-Zero) on **14 of 14** solved task-runs:

| task | model | baseline (PSF) |
|---|---|---|
| bell3 | 3 | 3 |
| ghz5 | 4 | 47 |
| fill27 | 17 | 52 |
| w3 | 6, 6 | 7 |
| qft3 | 0, 0, 3 | 7 |

In two of three QFT3 runs the model found the zero-two-qubit circuit.

**G3 = go.** The re-timing pass (sequential, 5 repetitions, median) found 12
distinct valid fill27 circuits, 2 of them correct.

- **Correct circuits:** PSF-Zero 10.1 and 11.1 ms against L3 7.37 and 7.40 s,
  a ratio of 665-731x. Two-qubit counts were 17 against 20.
- **Wrong circuits (not scored):** mostly equal counts. On two circuits L3
  was lower (22 against 13, and 33 against 24). Both were wrong circuits.

**Decision: INVEST.**

**Expectations** (from section 4 of the pre-registration):

| | expectation | outcome |
|---|---|---|
| P1 | gpt-oss: ghz5, bell3, fill27 every run | confirmed (3/3 each) |
| P2 | gpt-oss: w3 >= 2/3 | confirmed, at the edge (2/3) |
| P3 | gpt-oss: qft3 >= 2/3 | confirmed (3/3) |
| P4 | qwen7b: w3, qft3 at most 1/3 each | confirmed (0/3, 0/3) |

## 3. How close it was, as it is

**W3 with gpt-oss-120b**, compiled fidelity per round (L = cut at 48,000
tokens, S = salvage):

- run 1: L+S 0.056, L+S 0.648, **1.000**, L+S 1.000, 1.000, 0.324;
- run 2: 0.016, 0.870, 0.049, 0.206, 0.648, 0.111 (**not solved**);
- run 3: 0.049, 0.278, 0.870, 0.241, **1.000** (round 5), 0.056.

What this shows:

- Two of three runs solved W3, one at round 3 and one at round 5 of 6. One
  more miss would have made G1 ambiguous and the decision CUT.
- 3 of the 18 W3 rounds still hit the 48,000-token limit. Salvage turned one
  of them into a correct circuit.
- **Disclosed tuning:** W3 was the tuning target in three exploratory
  pilots, where the same configuration solved it 9/9 on other seeds. The
  scored 2/3 is lower than the pilots suggested. That is consistent with
  the pilots being optimistic, as the pre-registration warned.

**QFT3 run 3** solved in round 2. Later attempts to remove two-qubit gates
failed (fidelity 0), which does not affect the count.

**Qwen2.5-7B** is unchanged in kind from the first test: W3 and QFT3 are
never solved, and fill27 fell to 1/3 (2/3 in the first test). The v9
changes did not help the small model.

## 4. Checks of the harness (workplace sandbox, from the zip)

- The zip (`invest2_outputs_0930.zip`, SHA-256 `01865ca5...`, 220 files)
  matches its MANIFEST: 0 mismatches.
- Re-scoring the files with the locked scorer gives the same G1-G3, best
  model and decision.
- An independent numpy simulator without PennyLane or Qiskit
  ([`indep_check_2026-09-30.py`](../../benchmarks/indep_check_2026-09-30.py), plus `cp`) recomputed:
  - 111 parsed circuits on the small tasks: largest difference 4.4e-16, and
    0 disagreements on solved or not;
  - 25 fill27 circuits, group by group: difference 0.

## 5. What this does and does not establish

**It establishes** that under the pre-registered v9 conditions, vLLM-served
gpt-oss-120b coupled to PSF-Zero clears the go/no-go bar set in the morning,
with PSF-Zero's advantage at full-device scale reproduced on a second
machine and a second model setup. The conditions were: short-reasoning
prompt, cp gate, 48,000-token replies at high effort, salvage, best-circuit
memory, no tools, one H200.

**It does not establish:**

- **Reliability.** Three runs per task, and W3 passed at the minimum. A
  repeat could fail G1.
- **Anything beyond these five tasks**, or anything on real hardware.
- **That the pilots' tuning generalises.** The untuned tasks were solved,
  but they were already solved under v6.
- **Cost-effectiveness.** gpt-oss-120b spent 5,937 model-seconds on 15
  task-runs; PSF-Zero compile time was a fraction of a second in total.

**What INVEST means here**, by the rule set in the morning: the line is worth
further investment. The next pre-registered steps should target the weak
point directly (for example more runs on W3-like states, or new state
families), not repeat this test.

## 6. Files

- **Pod outputs:** `invest2_outputs_0930.zip`, per model and run:
  - `rounds.jsonl`;
  - `result.json`;
  - best circuits;
  - server logs;
  - `retime_fill27.csv`;
  - `score.md` and `score.json`;
  - `env.txt`.
- **Locked scripts:**
  - [`benchmarks/e2e_vllm_psf_v9.py`](../../benchmarks/e2e_vllm_psf_v9.py);
  - [`benchmarks/score_vllm_invest.py`](../../benchmarks/score_vllm_invest.py);
  - `benchmarks/pod/run_invest2_2026-09-30.sh`.
- **Independent check:** [`benchmarks/indep_check_2026-09-30.py`](../../benchmarks/indep_check_2026-09-30.py).

---

<!-- ===== Addendum 270 (source: spare-qubit-cliff-addendum-270-2026-09-30.md) ===== -->

> **Note added when merging:** Workplace pre-registration of the v10 evaluation (three candidates per round and verbal feedback) against v9 on five held-out tasks never shown to any model. Revision 1 (before any run: held-out tasks only) and Amendment 1 (the pod went down for a billing reason before any task-run finished; re-run with the same files, seeds and criteria) are appended. **No result yet.** Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run. The original `score_v10_eval.py` and `run_v10_eval_2026-09-30.sh`, superseded by Revision 1 before any run, are not in this repository; the scripts that govern are [`benchmarks/score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py) and `benchmarks/pod/run_v10_eval_r1_2026-09-30.sh`.

## Addendum 270 -- Pre-registration: v10 evaluation (several candidates per round + verbal feedback) against v9, on tuned and held-out tasks, gpt-oss-120b on a RunPod H200 (2026-09-30)

**Status: pre-registration, locked at the Project save time of this
document**, before any model sees the held-out tasks. No IBM account, no
network access to IBM, no QPU: the device is the FakeAuckland snapshot.

## 1. Why

The second go/no-go test (`vllm-invest2-results-2026-09-30.md`) passed with
the smallest possible margin: W3 2/3 against a bar of 2/3. It passed on
tasks that had been used for tuning. The owner made the line a continuing
project and chose two changes for v10:

- **(9) Several candidates per round** (`--n-candidates 3`, vLLM `n`). In the
  scored W3 failure, the answers oscillated (0.016, 0.870, 0.049, 0.206,
  0.648, 0.111), so more draws per round should raise the chance of an exact
  one.
  - The harness parses every candidate, checks it against the target, and
    keeps the best for the round.
  - Candidates are sampled at temperature >= 0.8.
- **(10) Verbal feedback** (`--verbal`), the owner's idea. For a wrong group
  of at most 6 qubits, the feedback also states in words how the state
  differs from the target:
  - basis states missing or extra;
  - magnitudes too large or too small;
  - relative sign or phase wrong;
  - whether the target is an equal-magnitude superposition.

  It is computed mechanically from the two state vectors and **never says
  how to fix the circuit**.

The owner also chose to judge the change on **held-out tasks**, to guard
against overfitting to the tasks used in the pilots.

## 2. Tasks

**Tuned tasks** (used in pilots or earlier tests): ghz5, w3, bell3, qft3,
fill27, identical to before.

**Held-out tasks** were defined for this test, and no model has seen them
before this lock. Only the sandbox mock replies written by hand were run on
them.

| task | qubits | target |
|---|---|---|
| w4 | 4 | W state (&#124;0001> + &#124;0010> + &#124;0100> + &#124;1000>)/2 |
| dicke42 | 4 | Dicke state D(4,2): the equal superposition of the six basis states with two 1s |
| ghz3i | 3 | (&#124;000> + i&#124;111>)/sqrt(2) |
| singlet3 | 6 | three singlets (&#124;01> - &#124;10>)/sqrt(2) on (0,1), (2,3), (4,5) |
| fill27g9 | 27 | the whole FakeAuckland device filled with nine GHZ-3 states on (0,1,2) ... (24,25,26) |

## 3. Design ([`e2e_vllm_psf_v10.py`](../../benchmarks/e2e_vllm_psf_v10.py), `run_v10_eval_2026-09-30.sh`, `score_v10_eval.py`)

**Hardware and software** are as in the second go/no-go test:

- RunPod H200, vLLM 0.30.0, gpt-oss-120b (`--max-model-len 65536`);
- PSF-Zero stack from `setup_gpu_2026-09-29.sh`.

Times are H200-pod times.

**Two arms** run at the same time against one server, both with high
reasoning effort, 48,000-token replies, salvage at medium effort,
best-circuit memory and no tool:

- **v9 arm:** 1 candidate per round, numeric feedback only. This is the v9
  harness of the INVEST test, run through the v10 script.
- **v10 arm:** `--n-candidates 3 --verbal`.

**Runs and timing:**

- 10 tasks x 3 runs per arm, 60 task-runs in parallel;
- request seeds use runs 21-23, written to `run1`-`run3`;
- up to 6 rounds, with the same solve rule (compiled fidelity >= 0.9999)
  and early stop as before;
- each task-run is killed after 180 minutes, and a missing result counts as
  not solved.

**Afterwards:**

- re-timing of the correct fill27 and fill27g9 circuits (sequential, 5
  repetitions, median);
- `score_v10_eval.py`, from files.

## 4. Criteria

- **H1, held-out generalisation** (v10 arm, 15 held-out task-runs):
  - go if >= 12/15 are solved;
  - stop if <= 7/15;
  - otherwise ambiguous.
- **H2, v10 against v9** (30 task-runs each):
  - better if v10 solves >= v9 + 3;
  - worse if v10 solves <= v9 - 2;
  - otherwise no clear difference.
- **H3, G2 on the held-out tasks** (v10 arm): compare the device two-qubit
  count (PSF-Zero) of the best circuit with the StatePreparation baseline on
  its solved held-out task-runs.
  - go if it is <= baseline on >= 80% of them;
  - stop if it is worse on > 50% of them, or nothing is solved.
- **H4, G3 on fill27g9** (correct circuits of both arms), with the thresholds
  of the go/no-go tests:
  - go if the median PSF time is <= 1 s, the median ratio L3/PSF is >= 10,
    and PSF 2q <= L3 on >= 80% of the circuits;
  - stop if the median ratio is < 3, or PSF > L3 on > 50% of the circuits,
    or the median PSF time is > 1 s, or there is no correct circuit.
- **G3 on fill27** is reported with the same thresholds, for comparison.
- **Decisions:**
  - **Default harness:** ADOPT v10 if H2 = better and H1 != stop; otherwise
    KEEP v9.
  - **Line:** CONTINUE unless both arms solve <= 7/15 held-out task-runs,
    in which case CUT: the approach does not generalise beyond the tasks it
    was tuned on.
  - H3 and H4 are reported and do not change these two decisions. A stop in
    H4 is a finding about PSF-Zero on a new tiling, not about the model.

## 5. Expectations (written before the run; they do not change any decision)

- **P1.** Both arms solve ghz3i and singlet3 in every run. They are close
  relatives of GHZ and Bell.
- **P2.** w4 and dicke42 are the hard held-out tasks. The v10 arm solves more
  of them than the v9 arm.
- **P3.** H2: better, but a small effect is likely with 3 runs per task. "No
  clear difference" would not surprise.
- **P4.** H4 is not go on time. In the sandbox dry run, PSF-Zero took 1.24-1.30
  s on a correct fill27g9 circuit (2-CPU sandbox; median of 3 and 5 repetitions)
  against L3's 8.7-8.8 s, a ratio of about 7. The nine-GHZ-3 tiling does not
  take the fast short-path route of layout c1. The pod CPU is faster, but the
  ratio is expected to stay below 10. The sandbox check is disclosed here; it
  was not tuned on.

## 6. Dry runs (disclosed)

- **Sandbox, fake vLLM server with `n` choices:**
  - one wrong, one cut and one correct W3 candidate;
  - the harness kept the correct one, recorded all three, and sampled at
    temperature 0.8.
- **Sandbox, verbal feedback:** checked on hand-made W3 and QFT3 states.
  Missing and extra basis states, magnitudes and sign or phase are stated
  correctly.
- **Sandbox, held-out targets:** printed and checked (normalised; the
  StatePreparation baseline reaches fidelity 1 after PSF-Zero). Hand-written
  mock circuits ran through the pipeline:
  - ghz3i, singlet3 and fill27g9 correct;
  - w4 and dicke42 deliberately wrong.
- **Sandbox, full run script with a fake server** (1 run per arm): both
  re-timing passes, the scorer and the zip worked. The fake server does not
  know the held-out tasks, so their verdicts there mean nothing.
- **Pod smoke run** (`SMOKE=1`: v10 arm, ghz5 and bell3 only, which are tuned
  tasks, seed run 97, 2 rounds), allowed after this lock. It is **not
  scored**, and it does not touch any held-out task.

## 7. Locked files (normalized SHA-256; raw in brackets)

| file | normalized | raw |
|---|---|---|
| [`benchmarks/e2e_vllm_psf_v10.py`](../../benchmarks/e2e_vllm_psf_v10.py) | `7203a243722dbdd3ab1718b09d2a12cb96e7c33c16732e015abf1f7cc7052c43` | `e00487a0...` |
| `benchmarks/score_v10_eval.py` | `69d757376b4a3f49ccfe071f7037398b505ea708ed97330f552313d32789fe9a` | `dc52d625...` |
| `benchmarks/pod/run_v10_eval_2026-09-30.sh` | `8b4f510f8eefbc2d33c857d9e62e9daf1012c2c7153eb26098780f4d4c4f344b` | `25ff91ee...` |

### Revision 1 (before any run) of the v10 evaluation pre-registration -- held-out tasks only, to fit in about one hour (2026-09-30)

**Status: revision of `vllm-v10-eval-preregistration-2026-09-30.md`, locked at
the Project save time of this document.** No run of the original design, and
no smoke run, happened before this revision. The original document is kept
unchanged. Where they differ, this revision governs.

## Why

The owner asked for the evaluation to take about one hour. The original
design (10 tasks x 3 runs x 2 arms, 60 task-runs, with the v10 arm drawing 3
candidates per round) was estimated at 1.5-3 hours.

The five tuned tasks were already scored this afternoon with the v9
harness, which solved 14/15 in the INVEST test. They carry no information
about generalisation, and W3 was the tuning target. They are therefore
dropped. The held-out tasks, which answer both questions (does v10 help,
and does the approach generalise), are kept in full.

## What changes

- **Tasks:** only the five held-out tasks (w4, dicke42, ghz3i, singlet3,
  fill27g9), 3 runs per arm, 30 task-runs in all. Arms, seeds (runs 21-23),
  budgets, rounds and the solve rule are unchanged.
- **H1** is unchanged: v10 arm, held-out, go if >= 12/15 solved, stop if
  <= 7/15.
- **H2** is rescaled to 15 task-runs per arm:
  - better if v10 solves >= v9 + 2;
  - worse if v10 solves <= v9 - 2;
  - otherwise no clear difference.
- **H3** and **H4** are unchanged.
- **G3 on fill27** is not run. It was measured in the INVEST test.
- **Decisions** are unchanged:
  - ADOPT v10 if H2 = better and H1 != stop, otherwise KEEP v9;
  - line CUT only if both arms solve <= 7/15 held-out task-runs.
- **Expectations:**
  - P1, P2 and P4 are unchanged.
  - P3 now reads: "H2 better, but with 15 task-runs per arm, no clear
    difference would not surprise".
- **Time estimate:** about 45-75 minutes on the H200, including the server
  start, the re-timing and the scoring.

## Dry run (disclosed)

- **Sandbox, the revised run script with a fake server** that answers the
  held-out tasks with hand-written mock circuits:
  - server start, the 30 task-runs, the fill27g9 re-timing, the revised
    scorer and the zip all worked;
  - two fill27g9 processes were killed by the sandbox's 7 GB memory limit
    with 30 processes at once. The same task-run alone completed. The pod
    has 2 TB.
- **Sandbox timing** of fill27g9 on the mock circuit (2-CPU sandbox, as
  disclosed in P4): PSF-Zero 1.27-1.31 s, L3 8.4-8.8 s.
- **Pod smoke run** (`SMOKE=1`: v10 arm, ghz5 and bell3, tuned tasks only, seed
  run 97, 2 rounds), allowed after this lock. It is not scored.

## Locked files (normalized SHA-256; raw in brackets)

| file | normalized | raw |
|---|---|---|
| [`benchmarks/e2e_vllm_psf_v10.py`](../../benchmarks/e2e_vllm_psf_v10.py) (unchanged) | `7203a243722dbdd3ab1718b09d2a12cb96e7c33c16732e015abf1f7cc7052c43` | `e00487a0...` |
| [`benchmarks/score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py) | `c1f599e6055b1a9895c273dbe524f1060e61753f9495466f50aa4efaf2634234` | `cdb53a78...` |
| `benchmarks/pod/run_v10_eval_r1_2026-09-30.sh` | `43c94396a1e1a85cf8b445dbdeb77ad8e0460ecb7db8a6e049427353d0768c32` | `8f595bcc...` |

The original `score_v10_eval.py` and `run_v10_eval_2026-09-30.sh` are kept
in the Project but are not used.

### Amendment 1 to the v10 evaluation (revision 1): infrastructure interruption and re-run (2026-09-30)

**Status: amendment, written and saved before the re-run.** It follows
section 5 of the go/no-go pre-registrations (carried into the v10
evaluation), which allows a re-run only for an infrastructure fault outside
the model, recorded as an amendment first.

## What happened

- The scored run of `run_v10_eval_r1_2026-09-30.sh` started on the H200 pod,
  with the locked hashes printed in its log. The server was ready after 81 s
  and 30 task-runs started.
- Shortly after, the pod went down because of a **billing-card issue** on the
  hosting account. The owner reported this; it was not caused by the
  harness or the model.
- At the last check before the interruption, **0 of 15 task-runs had
  finished in each arm**. No result, fidelity or score of the held-out tasks
  had been seen by anyone.

## What is kept

The first attempt's files are kept if the pod still has them. Stopping a pod
resets its container disk, so they may be lost; that is recorded as such.
Nothing from them can enter the score, because no task-run finished.

## Re-run

- **Unchanged:**
  - the locked files ([`e2e_vllm_psf_v10.py`](../../benchmarks/e2e_vllm_psf_v10.py) `e00487a0...`,
    [`score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py) `cdb53a78...`, `run_v10_eval_r1_2026-09-30.sh`
    `8f595bcc...`);
  - the seeds (runs 21-23);
  - the criteria and decisions of revision 1.
- **Same seeds:** they are kept because nothing from the first attempt was
  observed.
- **Environment:** if the pod has to be rebuilt, the setup is repeated as
  before:
  - `setup_gpu_2026-09-29.sh`;
  - vLLM 0.30.0 plus ninja in `~/vllm_env`;
  - gpt-oss-120b in `/workspace/hf_cache`.

  Any difference in the rebuilt environment (GPU type, driver, vLLM version)
  is recorded in the results. A different GPU type is allowed only if the
  H200 is unavailable, and must be stated.
- **Smoke run:** the disclosed smoke run (tuned tasks only, not scored) may be
  repeated after a rebuild.


---

<!-- ===== Addendum 271 (source: spare-qubit-cliff-addendum-271-2026-09-30.md) ===== -->

> **Note added when merging:** Results of the pre-registered v10 evaluation (Addendum 270, Revision 1, Amendment 1), re-run at home on a RunPod H200 with the same files and seeds: on five held-out tasks v9 solved 15/15 and v10 14/15; H1 go, H2 no clear difference, H3 go, H4 stop (PSF-Zero 1.2 s against L3's 8.0 s on the nine-GHZ-3 tiling). Decisions: KEEP v9, CONTINUE. Data: `data/2026-09-30/vllm_v10_eval/`.

## Addendum 271 -- Results: v10 evaluation on five held-out tasks (re-run under Amendment 1, RunPod H200). v9 solved 15/15 and v10 14/15; H1 go, H2 no clear difference, H3 go, H4 stop. Decisions: KEEP v9, CONTINUE (2026-09-30)

**Pre-registration:** Addendum 270, as governed by its Revision 1 (held-out tasks only) and Amendment 1 (re-run after
the first attempt's pod went down with no task-run finished). All three were locked at the workplace Project's save
time before any model saw a held-out task. This re-run used the same locked files, the same seeds (runs 21-23) and the
same criteria. Nothing was re-run to improve a score.

**Run:** at home, through VS Code on a RunPod H200 pod. `env.txt` records the start at 2026-09-30 13:52:06 UTC. The
server was ready at 13:53:27, and its last request was at 14:49:36. The fill27g9 re-timing and the scoring followed.
The locked files ran unchanged; `env.txt` records the same raw SHA-256 values that the kit and Revision 1 state:

- [`e2e_vllm_psf_v10.py`](../../benchmarks/e2e_vllm_psf_v10.py): `e00487a0...`;
- [`score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py): `cdb53a78...`;
- `run_v10_eval_r1_2026-09-30.sh`: `8f595bcc...`.

The outputs zip as received is `v10eval_outputs_0930.zip` (SHA-256 `7d9b21a3...`, 218 files plus its MANIFEST).

## 0. In one line

On five tasks that no model had seen, the v9 harness (the one that passed Addendum 269) solved all 15 task-runs. The
new v10 harness (three candidates per round and verbal feedback) solved 14. It did not beat v9 and used three times
the tokens, so **v9 stays the default harness and the line continues.** H4 is a stop, as predicted: on the new
nine-GHZ-3 tiling PSF-Zero took 1.2 s against Qiskit L3's 8.0 s. That is a finding about PSF-Zero's candidate layout,
not about the model.

## 1. Environment (H200 pod; times are H200-pod times, not compared with any other machine)

- **Hardware:** NVIDIA H200 (143,771 MiB, driver 595.91.07); 96 CPUs (Intel Xeon Platinum 8568Y+).
- **Software:** vLLM 0.30.0, torch 2.13.0+cu130, Qiskit 2.5.2, PennyLane 0.45.1.
- **PSF-Zero stack:** psf_compile 2026-09-28.1, candidate core 2026-09-29.1, candidate layout 2026-09-29.c1. The setup
  script `setup_gpu_2026-09-29.sh` printed `SETUP DONE` with both cores and `pl_heavyhex_gpu.py a0081c29...`.
- **GPU memory in use** with the server up: 129,467 MiB.
- **Same as the day's workplace H200 pods:** GPU, driver and CPU count are identical to those of Addenda 265-269, so
  Amendment 1's "record any difference" has nothing to record.

**Two pods were rejected before anything ran on them.** Both reported driver 570.124.06 (CUDA 12.8), which cannot run
the pinned torch 2.13.0+cu130 (CUDA 13.0 needs driver 580 or later). They were terminated after the first check,
before any install, model download or smoke run. The third pod was chosen with RunPod's CUDA-version filter set to 13.0
or later.

**Smoke run** (disclosed, not scored; v10 arm, ghz5 and bell3, tuned tasks only, seed run 97, 2 rounds): the server was
ready after 171 s, and both tasks were solved without errors. Its output stayed on the pod and was not downloaded; the
console lines are the record.

## 2. Results (scored by the locked [`score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py), from files)

| arm | held-out /15 | w4 | dicke42 | ghz3i | singlet3 | fill27g9 | missing | reply errors | HTTP 400 |
|---|---|---|---|---|---|---|---|---|---|
| v9 (1 candidate, numeric feedback) | **15** | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 |
| v10 (3 candidates, verbal feedback) | **14** | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 |

- **H1 = go.** The v10 arm solved 14/15 held-out task-runs (go needs >= 12).
- **H2 = no clear difference.** v10 minus v9 is -1 (better needs >= +2, worse needs <= -2).
- **H3 = go.** On all 14 of the v10 arm's solved task-runs, the device two-qubit count after PSF-Zero was at most the
  StatePreparation baseline's.
- **H4 = stop.** On the three distinct correct fill27g9 circuits:
  - median PSF-Zero time 1.21 s, above the 1 s limit;
  - median ratio L3/PSF 6.6 (range 6.4-6.8), below 10;
  - PSF-Zero two-qubit count <= L3 on 2 of the 3 circuits.
- **Decisions** (Revision 1):
  - **Default harness: KEEP v9**, because H2 is not "better";
  - **Line: CONTINUE**, because neither arm is at or below 7/15.

### 2.1 Expectations (written before the run)

| | expectation | outcome |
|---|---|---|
| P1 | both arms solve ghz3i and singlet3 in every run | **confirmed** (3/3 each, all in round 1) |
| P2 | w4 and dicke42 are the hard tasks, and v10 solves more of them than v9 | **half**: they were the hard tasks (980-3,355 s of model time per task-run), but v9 solved 6/6 and v10 5/6 |
| P3 | H2 better, but "no clear difference" would not surprise | **no clear difference** |
| P4 | H4 not go on time | **confirmed**: 1.21 s, ratio 6.6 |

### 2.2 Cost of the two arms

Summed over the 15 task-runs of each arm, which ran in parallel against one server:

| arm | completion tokens | model time (s) |
|---|---|---|
| v9 | 823,352 | 13,714 |
| v10 | 2,500,505 | 16,397 |

v10 used about 3.0 times the tokens and 1.2 times the model time, and solved one task-run fewer.

- Most of the cost sits in w4 and dicke42 (980-3,355 s per task-run in both arms). ghz3i, singlet3 and fill27g9 were
  solved in round 1 of every run, at 100-170 s per task-run.
- 17 of v10's 150 candidates ended at the 48,000-token limit without a circuit.
- Six rounds (2 in v10, 4 in v9) were cut at the limit and went to the salvage request.

### 2.3 The one failure

v10, dicke42, run 1:

- 6 rounds, best compiled fidelity 0.742;
- 663,326 completion tokens and 3,355 s of model time, the most of any task-run.

The two other v10 dicke42 runs solved it in round 1, and v9 solved all three (in rounds 1, 4 and 1).

## 3. Two-qubit counts (device count after compiling; best circuit of each solved task-run)

| task | v9: model (PSF) | v10: model (PSF) | baseline (PSF / L3) | model circuit through L3 |
|---|---|---|---|---|
| w4 | 9, 9, 9 | 10, 9, 9 | 17 / 13 | 6-10 |
| dicke42 | 12, 16, 15 | 13, 11 | 17 / 13 | 9-16 |
| ghz3i | 2, 2, 2 | 2, 2, 2 | 7 / 7 | 2 |
| singlet3 | 3, 3, 3 | 3, 3, 3 | 3 / 3 | 3 |
| fill27g9 | 18, 18, 18 | 18, 18, 18 | 63 / 53 | 18 |

**Reported without prediction:** on W4 and Dicke(4,2), Qiskit L3 compiling the same model circuit often reached fewer
two-qubit gates than PSF-Zero:

- W4: 6 against 9 in five of the six solved task-runs;
- Dicke(4,2): 10 against 12 and 9 against 11.

On these small, dense 4-qubit circuits L3's resynthesis finds shorter circuits than PSF-Zero's block-by-block
synthesis. It is the opposite of the fill27 and fill27g9 results, where PSF-Zero is equal or better. The model's
circuits were at or below the StatePreparation baseline either way (H3).

## 4. PSF-Zero on the nine-GHZ-3 tiling (H4)

Re-timing: sequential, 5 repetitions, median. There were 3 distinct correct circuits, one of which was written 15 times.

| circuit | PSF-Zero | L3 | two-qubit PSF / L3 |
|---|---|---|---|
| `34540ad3...` (15 occurrences) | 1.199 s | 8.141 s | 18 / 18 |
| `2b48f48b...` (3) | 1.210 s | 8.041 s | 18 / 18 |
| `9af9e699...` (1) | 1.266 s | 8.139 s | 24 / 18 |

- All three are correct after compiling (fidelity within 1e-14 of 1), and there are no swaps in any of them.
- On the tilings of Addenda 264 and 269 (seven GHZ-3 states and three Bell pairs), PSF-Zero took 10-21 ms. Here it took
  1.2 s: the nine-GHZ-3 tiling does not take the candidate layout's short-path route.
- One circuit (a round-1 answer, not the best of its run) got 24 two-qubit gates against L3's 18.
- This confirms, on the pod's CPU, what Addendum 270 disclosed from the sandbox (1.24-1.31 s against 8.4-8.8 s). It
  was not tuned on. The improvement of layout c1 for this tiling can now start, with its own pre-registration.

## 5. Checks (home)

- **Manifest:** all 218 files of the outputs zip match its MANIFEST (normalized SHA-256 for text, raw for binary).
- **Re-scoring:** the locked [`score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py) run again on the unpacked files gives an identical [`score_v10.md`](../../data/2026-09-30/vllm_v10_eval/pod_outputs/score_v10.md)
  and an identical [`score_v10.json`](../../data/2026-09-30/vllm_v10_eval/pod_outputs/score_v10.json).
- **Independent numpy check** ([`benchmarks/indep_check_v10_heldout.py`](../../benchmarks/indep_check_v10_heldout.py); no PennyLane, no Qiskit):
  - The script builds the five held-out targets from their definitions in Addendum 270, not from the harness. All five
    are symmetric under relabelling within each group (the singlets up to a global sign), so the check does not depend
    on the harness's bit order.
  - It recomputed all 103 recorded round circuits (fill27g9 group by group; no gate crossed groups): largest difference
    4.4e-16, and 0 disagreements on solved or not solved.
  - It gives v9 15/15 and v10 14/15, with the same per-task counts as the scorer.
  - The outputs keep, for the v10 arm's non-chosen candidates, only their finish reason, reasoning length and fidelity,
    not the circuits. Those candidates could not be re-simulated.

## 6. What this does and does not establish

**It establishes:**

- With the v9 harness, gpt-oss-120b coupled to PSF-Zero solved 15 of 15 runs of five tasks that no model had seen and
  that were not used for tuning, within 6 rounds.
- Together with Addendum 269, the harness that passed at the margin on its tuning tasks also held on new ones.
- v10's two changes, several candidates per round and verbal feedback, did not help on these tasks and tripled the
  token cost.

**It does not establish:**

- **Reliability at scale.** There were three runs per task.
- **Generality.** Three of the five held-out tasks (ghz3i, singlet3, fill27g9) are close relatives of the tuned GHZ and
  Bell tasks and were solved in round 1. The hard two, W4 and Dicke(4,2), are relatives of W3. States of a different
  kind remain untested.
- **Cost-effectiveness.** The w4 and dicke42 runs took 16-56 minutes of model time each.
- **Anything about real hardware.**

## 7. Files

| file | what it is |
|---|---|
| `data/2026-09-30/vllm_v10_eval/pod_outputs/` | the unpacked `v10eval_outputs_0930.zip`: per arm and run, `rounds.jsonl`, `result.json`, best circuits; logs, server log, [`retime_fill27g9.csv`](../../data/2026-09-30/vllm_v10_eval/pod_outputs/retime_fill27g9.csv), [`score_v10.md`](../../data/2026-09-30/vllm_v10_eval/pod_outputs/score_v10.md) and `.json`, `env.txt`, `MANIFEST.tsv` |
| [`data/2026-09-30/vllm_v10_eval/indep_check_output.txt`](../../data/2026-09-30/vllm_v10_eval/indep_check_output.txt) | output of the independent check |
| [`benchmarks/indep_check_v10_heldout.py`](../../benchmarks/indep_check_v10_heldout.py) | the independent check (targets rebuilt from Addendum 270) |
| [`benchmarks/e2e_vllm_psf_v10.py`](../../benchmarks/e2e_vllm_psf_v10.py), [`benchmarks/score_v10_eval_r1.py`](../../benchmarks/score_v10_eval_r1.py), `benchmarks/pod/run_v10_eval_r1_2026-09-30.sh` | the locked files (already in the repository with Addendum 270) |


---

<!-- ===== Addendum 272 (source: spare-qubit-cliff-addendum-272-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration of the PSF-Zero core fixes c2 (psf_compile and psf_smart_layout), made after the weaknesses found in Addendum 271. Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run. Document names used in Addenda 272-289: core-fix-c2 = 272-274, ai-compile-a0 = 275-277, ai-compile-a1 = 278-279, noisy-fidelity = 280-281, ai-compile-a2 = 282-283, vllm-a2-replay = 284-285, ai-compile-a4 = 286-287, long-loop-a5 = 288-289. Note on section 1: on the H200 pod of Addendum 271, layout c1 placed the nine-GHZ-3 circuits swap-free after about 1.2 s (18 or 24 two-qubit gates); the sandbox re-check here found SWAPs (30). A time-budgeted search that finishes on the faster pod CPU and not on 2 sandbox CPUs would explain both; not tested.

## Addendum 272 -- Pre-registration: PSF-Zero core fixes of 2026-10-01 -- psf_compile 2026-10-01.c2 (cheaper short blocks, SWAP elision, routing-SWAP absorption) and psf_smart_layout 2026-10-01.c2 (exact packing of 2- and 3-qubit paths), on held-out inputs (2026-10-01)

**Status: pre-registration. It is locked at the Project save time of this
document**, before the scored run. Development used other inputs; they and
one harness dry run are disclosed in sections 1 and 7. **All runs are in the
workplace sandbox (Linux, 2 CPUs, Python 3.11.15, Qiskit 2.5.2). No IBM
account, no network access to IBM and no QPU are used.** Devices are
fake-provider snapshots.

## 1. Why, and what development found (not pre-registered)

**The two weaknesses (Addendum 271, v10 held-out evaluation, sandbox
re-checks).** Both were in PSF-Zero itself, not in the language model.

- **Slow layout on the nine-GHZ-3 tiling (fill27g9).** Nine disjoint 3-qubit
  paths fill FakeAuckland's 27 qubits exactly.
  - Layout c1's short-path shortcut (a maximum matching plus one neighbour
    per path) does not find this tiling.
  - The VF2 stages then search for 1.1-1.2 s (sandbox) and still route with
    SWAPs (30 two-qubit gates against 18 swap-free).
- **More two-qubit gates than Qiskit L3 on small dense circuits.**
  - Examples: model-written W4 and Dicke(4,2) circuits.
  - Short same-pair runs such as (cry, cx) are left alone by
    `block_gate_floor = 8`. They cost 3 CX where 2 suffice.

**Development data.** All of these are now excluded from the scored inputs:

- the 149 model circuits of 2026-09-30 (`data/2026-09-30/**/best_circuit.json`);
- random dense circuits with Python `random` seeds 7, 8 and 9 (60 circuits
  each, FakeAuckland and FakeKingston);
- `random_circuit` seeds 1-6;
- `dense_pair_blocks` and the families with seed 0;
- synthetic tilings: Auckland k3 = 7, 8 and 9; Kingston k3 = 39, 50 and 52.

**How the candidates changed during development**, in order:

1. **Layout c2.** After the c1 shortcut fails, an exact depth-first packing
   search runs. It places the 3-qubit paths and pairs on vertex-disjoint
   physical paths and edges.
   - Budget: min(1 s, half the layout budget). It runs only if a matching of
     the needed size exists.
   - Results: fill27g9 is placed in about 0.11 s (11-13 ms median inside
   `compile_for_hardware`, against 1,209 ms for c1).
   - Auckland k3 = 8 and 9 and Kingston k3 = 39, 50 and 52 are placed.
   - The failures checked were proved infeasible.
   - Kingston (39, 17) needed 0.79 s, so the budget was raised from 0.5 to
     1.0 s.
2. **Compile c1: cost-aware consolidation.**
   - Rule: a block at or below the floor is consolidated if it has two or
     more 2-qubit gates and its Weyl-optimal CX count is below its CX cost as
     written.
   - Model circuits: 0 fidelity changes, and no circuit got worse. W4 went
     from 55 to 40 two-qubit gates in total, W3 from 209 to 196, Dicke from
     89 to 85.
   - An apparent regression on a 40-qubit random circuit (346 against 342)
     was a counting artifact: gates were counted as written. Counted in CX
     after translation, it was 641 → 632.
   - Under `entangling_basis="canonical"` the rule made things worse
     (641 → 648). **It is therefore restricted to `"cx"`.**
   - The `Operator`-based block matrix was replaced by a numpy builder
     (600 of 600 blocks equal to `Operator`). Blocks costing more than 3 CX
     skip the matrix.
3. **Remaining gap to L3.** It came from SWAPs.
   - Model-written QFT circuits end in a SWAP, and Qiskit 2.5 levels 2 and 3
     remove such SWAPs by relabelling: ElidePermutations, and
     Split2QUnitaries(split_swap=True) for a SWAP written as a `unitary`, as
     PennyLane tapes arrive.
   - Routing SWAPs next to a gate on the same pair can also be merged
     (SWAP + CRY: 5 CX as written, 3 needed).
   - Compile c2 adds both: `elide_permutations="auto"` and
     `post_routing_resynthesis="auto"`.
   - **One bug was found and fixed during development.** The first version
     read the elided permutation from `PassManager.property_set` after
     `run()`. That returns nothing in Qiskit 2.5, so elision silently never
     applied (outputs were correct, just not improved). It is now read via
     the run callback.
   - The first post-routing version re-synthesised every block after
     translation. That cost +60 % time on `dense_pair_blocks` 156q for no
     gain. It now runs in the `post_routing` stage, on blocks that contain a
     SWAP only.
   - Cost of that change on the development set: one model W4 circuit went
     back from 33 to 34 two-qubit gates (L3: 33).
4. **Development results of the final candidates** (sandbox, development
   inputs; all outputs equivalent):
   - **Model circuits (149):** 1,146 → 1,030 two-qubit gates. The release
     stack here is compile 2026-09-28.1 with layout 2026-09-29.c1, as in the
     9/30 runs; both stacks use core 2026-09-29.1. 0 circuits worse, 0
     fidelity changes.
     - Circuits above L3, counted over unique circuits with the release
       compile and the candidate compile (both with layout c2):
       QFT3 9/19 → 0/19, W3 13/30 → 1/30, W4 5/5 → 1/5 (by 1 gate),
       Dicke 3/6 → 2/6.
     - Sums over the unique circuits: W4 46 → 34 (L3 33), Dicke 89 → 83
       (L3 79), QFT3 115 → 86 (L3 86).
   - **Random dense, 3-5 qubits** (seeds 7, 8 on Auckland; 9 on Kingston):
     sums 1,291 / 1,350 / 1,377 (release) against 824 / 827 / 836
     (candidate) and 807 / 793 / 791 (L3).
     - Candidate above release: 0, 0 and 1 circuits of 60. The one case
       comes from a different routing; before routing, the candidate had
       fewer CX.
   - **Larger circuits on Kingston:** never more two-qubit gates.
     - Example: ghz_star 60q, 348 → 183.
     - `dense_pair_blocks` and `k_chains` are unchanged in gates, with
       0-10 % more time.
     - `random_circuit` circuits are 5-11 % fewer gates, with 15-50 % more
       time.

## 2. The candidates

**Rust core.** Neither candidate touches it. All arms run on the sandbox
build of core **2026-09-29.1** (the candidate core of 9/29, the same as the
v10 runs).

**`psf_compile.py` 2026-10-01.c2.** The base is release 2026-09-28.1
(changelog items 28-30 in the file).

- **(28)** Cost-aware consolidation of short blocks, only for
  `entangling_basis="cx"`.
- **(29)** `compile_for_hardware(elide_permutations="auto")`:
  - ElidePermutations + Split2QUnitaries(split_swap=True) run before
    compiling;
  - the permutation is carried into the preset pipeline's final layout;
  - the output qubits must be read through `out.layout.final_index_layout()`,
    as the e2e checker and the IBM pipeline already do.
- **(30)** `compile_for_hardware(post_routing_resynthesis="auto")`: blocks
  with a routing SWAP are re-synthesised by the PSF core in the
  `post_routing` stage.
- **"auto" = on for `"cx"` only.** The canonical path is meant to be
  bit-identical to the release.
- **Unchanged:**
  - the core interface;
  - `block_gate_floor`;
  - `routing_optimization_level = 1`;
  - the layout search call;
  - `verify` and `tol`.

**`psf_smart_layout.py` 2026-10-01.c2.** The base is candidate 2026-09-29.c1:
the corrected feasibility check and the short-path shortcut.

- After the shortcut fails, `packing_layout()` runs: an exact packing search
  with budget min(1.0 s, 0.5 × layout budget).
- Everything else is as in c1.

## 3. Design ([`benchmarks/core_fix_c2_eval.py`](../../benchmarks/core_fix_c2_eval.py))

All compiles use `compile_for_hardware(coupling_map, basis_gates=native,
entangling_basis="cx", layout_search=True, seed_transpiler=0)`, unless noted.
The layout module is switched between arms by replacing
`sys.modules["psf_smart_layout"]`.

**Part L (layout).** Tilings of k3 disjoint 3-qubit paths (h, cx, ry, cx, rz)
and k2 pairs (h, cx, ry). Logical labels are shuffled (seed 5000 + 100 ×
device index + combination index). One circuit per combination.

| Device | Qubits / max matching | (k3, k2) |
|---|---|---|
| FakeAuckland | 27 / 10 | (6,4), (5,6), (4,7), (6,3) |
| FakeTorino | 133 / 56 | (21,35), (25,29), (30,21), (37,11), (44,0), (20,36) |
| FakeKingston | 156 / 64 | (28,36), (32,30), (40,18), (46,9), (30,32) |
| FakeNighthawk | 120 / 60 | (10,45), (20,30), (40,0), (26,21) |

- **Ground truth:** `packing_layout` with a 30 s budget, outside any timed
  call. "Feasible" = a layout is found. "Infeasible" = the search ends with
  none in under 28.5 s. "Unknown" = it times out.
- **Arms:** layout m1 (release 2026-09-26.m1), c1 and c2, all with compile
  c2.
- **Recorded:** the two-qubit count against the swap-free count
  (2 × k3 + k2); median time of 3; an output digest; and a component-wise
  equivalence check.
- **The equivalence check:** statevector with 3 random inputs per logical
  component, read at `final_index_layout`, with ancillas in |0>. It is done
  only where a component's physical support is 12 qubits or fewer.

**Part C (small dense, both FakeAuckland and FakeKingston).**

- **Random:** 90 circuits per device, Python `random` seeds 1001-1090.
  - 3-5 qubits and 6-20 gates. 30 % single-qubit gates: h, x, s, t, ry.
  - 2-qubit gates: cx, cry, crz, cp, swap, cz. Half of the SWAPs are written
    as `unitary`.
- **Textbook:** 13 circuits, each in gate form and in PennyLane-style form
  (every multi-qubit gate as `unitary`).
  - W3-W5: a cry cascade with cx.
  - Dicke(4,2): cx-cry-cx steps.
  - QFT3-QFT5 with final swaps.
  - GHZ star 4 and 5.
  - A 4-qubit swap network with phases.
  - A 5-qubit reversal by SWAPs.
- **Arms:**
  - REL = compile 2026-09-28.1;
  - CAND = compile 2026-10-01.c2;
  - both with layout c2;
  - Qiskit `transpile(optimization_level=3, seed_transpiler=0)` as the
    reference (L3).
- **Also recorded:**
  - **(K)** For the 90 Auckland random circuits, the REL and CAND digests
    with `entangling_basis="canonical"`.
  - **(M)** For Auckland seeds 1001-1030, `measure_all()` is added, the
    circuit is compiled by CAND and run on AerSimulator (20,000 shots), and
    the TVD to the exact distribution is recorded.

**Part R (larger, FakeKingston).** REL against CAND (layout c2 for both),
median time of 3.

- **"other":** `random_circuit` (16q d20 s2001, 32q d20 s2002, 48q d15 s2003,
  80q d10 s2004, 120q d8 s2005), plus `random_regular` and `ghz_star` (80q,
  seed 1).
- **"unchanged":** `dense_pair_blocks` (60, 120 and 156q, seed 1), plus
  `k_chains` and `linear_chain` (80q, seed 1). These are families with no
  short cheaper blocks and no SWAPs before routing.

## 4. Pre-registered predictions

**C0 (harness):** no compile raises, and the loaded versions are
2026-09-28.1, 2026-10-01.c2 and layouts 2026-09-26.m1, 2026-09-29.c1 and
2026-10-01.c2. "Fidelity OK" means > 1 - 1e-9.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| L1 | c2 places every feasible tiling swap-free, fast | all ground-truth-feasible combinations: c2 two-qubit count = swap-free count **and** median time <= 1.0 s | 2 or more feasible combinations miss either |
| L2 | c2 never costs gates and changes nothing c1 already solves | c2 two-qubit <= min(m1, c1) everywhere **and** c2 digest = c1 digest wherever c1 was swap-free | c2 above min(m1, c1) anywhere |
| L3 | the packing budget bounds the extra time | on infeasible/unknown combinations, c2 time <= c1 time + 1.2 s | exceeded on 2 or more |
| L4 | layout outputs are exact | no checked output below 1 - 1e-9 (all arms) | any checked output below |
| C1 | CAND outputs are exact | no checked C/T output below 1 - 1e-9 **and** at most 5 % not checkable | any checked output below |
| C2 | CAND uses fewer two-qubit gates than REL | per device: sum CAND < sum REL **and** CAND > REL in <= 5 % of circuits | sum not lower, or > 10 % worse |
| C3 | CAND is close to Qiskit L3 | per device: sum CAND <= 1.08 × sum L3 **and** CAND > L3 in <= 25 % | > 1.15 × or > 40 % |
| C4 | textbook circuits never get worse | CAND <= REL on all 52 textbook compiles | above on any |
| C5 | the canonical path is unchanged | REL and CAND canonical digests equal on all 90 | any differs |
| C6 | measured circuits come out right (final-layout bookkeeping) | TVD <= 0.03 on all 30 | above on any |
| R1 | no gate regression on larger circuits | CAND two-qubit <= REL on all 12 | above on any |
| R2 | small time cost where nothing changes | "unchanged" cases: CAND <= 1.25 × REL + 10 ms | any > 1.5 × REL + 20 ms |
| R3 | bounded time cost elsewhere | "other" cases: CAND <= 2 × REL + 20 ms | any > 3 × REL + 50 ms |
| R4 | larger outputs are exact | no checked output below 1 - 1e-9 | any checked output below |

Between the bounds: ambiguous.

**Decision rule.** If C0 holds and all 14 predictions are CONFIRMED, the
recommendation to home is to adopt both candidates. Adopting them is home's
decision. Otherwise the failing items are reported, and nothing is re-run to
improve a score. A re-run is allowed only for an infrastructure fault, and
only after an amendment.

**Expectations stated before running:**

- **L1:** about 0.7 s was needed for Kingston (39, 17) in development, so a
  Torino or Kingston combination could exceed 1 s. That would be a real
  failure of the budget, and it is not excluded.
- **C3:** development ratios were 1.02-1.06. Qiskit's L3 also uses
  commutation-based cancellation, which the candidate does not.
- **C2:** the 5 % allowance is for routing differences, which development
  showed (1 of 60 on Kingston).

**Reported without prediction:**

- all per-circuit counts and times;
- REL's own equivalence;
- the ground-truth search times;
- a supplementary repeat of Parts C and R with the release core 2026-09-28.1
  (if run).

## 5. What this can and cannot establish

**It tests** whether the two weaknesses found in Addendum 271 are fixed on
inputs not used in development, and whether the fixes cost gates,
correctness or much time elsewhere.

**It does not establish:**

- hardware fidelity: compile time and gate counts are not fidelity;
- the live IBM Targets;
- the weighted layout path (`layout_edge_errors`), where the packing search
  is not used;
- `entangling_basis="canonical"` improvements (deliberately off);
- circuits with mid-circuit measurement beyond one smoke test;
- the GPU tests of the repository, which need lightning.gpu and were not run
  here.

Times are sandbox times. They are not compared with home or the pod.

## 6. Files, integrity, run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| [`benchmarks/core_fix_c2_eval.py`](../../benchmarks/core_fix_c2_eval.py) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |
| candidate `psf_compile.py` (2026-10-01.c2) | 95,818 | `e22dc6ddf4bf717d722a3f7ce01e23ecbe096d41730f7fe54c0d29eeaa4a3bd1` |
| candidate `psf_smart_layout.py` (2026-10-01.c2) | 32,708 | `f0d38519adad04864de42c456564a8321584ae3feadd744101a4cb719070fc14` |
| candidate [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) | 6,857 | `d2a188d694585588f6950176daf6d9ff256956f44c4a24e0f51d795a0d06e8fb` |
| release `psf_compile.py` (2026-09-28.1, base) | 84,542 | `3616efc8b8a7bea184d6170fb7379d509d0bfec9828f4eb2f703dd4d26d8c60b` |
| release [`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) (2026-09-26.m1) | 22,450 | `a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875` |
| candidate layout 2026-09-29.c1 ([`patches/psf_smart_layout_c1_2026-09-29/`](../../patches/psf_smart_layout_c1_2026-09-29/)) | 27,664 | `e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa` |
| `core_fix_c2_2026-10-01.patch` (git diff against `9131cee`, 4 files) | 62,762 | raw SHA-256 `46223e8c3fb1f2d70edbfbd37a7e868568c407668dada4d135ce876c819b5dd3` |

The patch reproduces the four candidate files byte for byte on a clean
`9131cee`. [`apply_core_fix_c2_2026-10-01.py`](../../patches/core_fix_c2_2026-10-01/apply_core_fix_c2_2026-10-01.py) installs them only over the
listed bases and keeps backups.

Run from a clean clone at `9131cee` with the candidate files in
`<cand>/compile` and `<cand>/layout`, and core 2026-09-29.1 in `<core>`:

```
PYTHONPATH=<core> python -u benchmarks/core_fix_c2_eval.py run \
  --rel-compile psf_compile.py --cand-compile <cand>/compile/psf_compile.py \
  --layout-m1 benchmarks/psf_smart_layout.py \
  --layout-c1 patches/psf_smart_layout_c1_2026-09-29/psf_smart_layout.py \
  --layout-c2 <cand>/layout/psf_smart_layout.py --families benchmarks \
  --out core_fix_c2_raw.json > core_fix_c2_run.txt 2>&1
python benchmarks/core_fix_c2_eval.py score --out core_fix_c2_raw.json > core_fix_c2_score.txt
```

## 7. Before locking (disclosed)

**Harness dry run** (`--dry`): development cases and seeds offset by 900,000.

- Inputs: layout Auckland (7,3) and (9,0) and Kingston (39,17); 6 random
  circuits per device; 4 textbook compiles per device; 3 larger circuits.
- Every item scored CONFIRMED there.
- c2 placed Kingston (39,17) swap-free in 0.71 s, against 2.08 s and 260
  two-qubit gates for c1.
- After the dry run, C1's scoring was changed: an output too large to check
  now counts as "not checkable" rather than as a failure, with a 5 % cap.

**Repository tests** (pytest on `benchmarks/test_*.py`, candidate tree
against release tree, same core):

- The same 13 tests fail in both. They need lightning.gpu or an IBM-style
  GPU device.
- 7 files cannot be collected in either (pytket is missing).
- The only new failures are the two version-string assertions:
  - `test_release_2026_09_28.py::test_version`, which expects 2026-09-28.1;
  - `test_short_path_layout.py::test_version_is_candidate`, which expects
    2026-09-29.c1.
- The new `test_core_fix_c2.py` passes 10 of 10.
- Totals: 124 passed with the candidates, against 106 with the release. The
  candidate tree has the two extra test files.

**No scored input was compiled before locking.**

---

<!-- ===== Addendum 273 (source: spare-qubit-cliff-addendum-273-2026-10-01.md) ===== -->

> **Note added when merging:** 13 of 14 CONFIRMED, L2 AMBIGUOUS, none refuted; the pre-registered automatic adoption rule did not fire. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/core_fix_c2/as_run/`](../../data/2026-10-01/core_fix_c2/as_run/); the outputs are in [`data/2026-10-01/core_fix_c2/`](../../data/2026-10-01/core_fix_c2/).

## Addendum 273 -- Results: PSF-Zero core fixes of 2026-10-01 (psf_compile 2026-10-01.c2, psf_smart_layout 2026-10-01.c2) on held-out inputs -- 13 of 14 CONFIRMED, L2 AMBIGUOUS (2026-10-01)

**Pre-registration:** `docs/findings/core-fix-c2-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 272 of this Part).
It was locked at its Project save time before this run. This run compiled
the scored inputs for the first time. **Environment:** workplace sandbox,
Linux, 2 CPUs, Python 3.11.15, Qiskit 2.5.2. Core 2026-09-29.1 (sandbox
build) in all arms. Fake-provider devices; no IBM access. Times are sandbox
times, median of 3, and are compared only within this run.

## 1. Verdict

**C0 (harness):** OK. Versions loaded: 2026-09-28.1 and 2026-10-01.c2;
layouts 2026-09-26.m1, 2026-09-29.c1 and 2026-10-01.c2. No compile raised.

| ID | Result | Numbers |
|---|---|---|
| L1 | CONFIRMED | 10 of 10 ground-truth-feasible tilings placed swap-free by c2 within 1 s (slowest: Kingston (40,18), 0.955 s) |
| L2 | **AMBIGUOUS** | c2 never above min(m1, c1) (0 of 19); but 1 of 6 tilings that c1 already placed swap-free got a different output from c2 (section 2) |
| L3 | CONFIRMED | 0 of 9 infeasible/unknown tilings where c2 took more than c1 + 1.2 s |
| L4 | CONFIRMED | 30 of 57 layout outputs checkable, 0 below 1 - 1e-9 |
| C1 | CONFIRMED | 224 C/T outputs, 0 not equivalent, 0 not checkable (REL also 0) |
| C2 | CONFIRMED | Auckland: REL 1,905 → CAND 1,147; Kingston: 2,013 → 1,168. CAND above REL in 0 % |
| C3 | CONFIRMED | CAND / L3 = 1.027 (Auckland), 1.041 (Kingston); CAND above L3 in 13.3 % / 15.6 % |
| C4 | CONFIRMED | 0 of 44 textbook compiles above REL (see the count correction in section 4) |
| C5 | CONFIRMED | 90 of 90 canonical-basis outputs identical, REL vs CAND |
| C6 | CONFIRMED | 30 of 30 measured circuits within TVD 0.03 (max 0.0089, 20,000 shots) |
| R1 | CONFIRMED | 0 of 12 larger circuits above REL |
| R2 | CONFIRMED | "unchanged" families: +2.5 to +11.5 ms (largest ratio 1.29 at 120q, 33.5 → 43.2 ms, within 1.25 × + 10 ms) |
| R3 | CONFIRMED | "other": at most 1.51 × (32q random, 84 → 127 ms) |
| R4 | CONFIRMED | 6 of 24 checkable, 0 below 1 - 1e-9 |

**Decision under the pre-registered rule:** not every item is CONFIRMED, so
the run **does not** produce the automatic "adopt" recommendation. The single
non-confirmed item is explained in section 2. Whether to adopt the
candidates anyway is home's decision. Nothing was re-run.

## 2. L2 AMBIGUOUS: what happened

**The case is FakeNighthawk with (k3, k2) = (20, 30).**

| Layout | Two-qubit gates | Time | How the layout was found |
|---|---|---|---|
| c1 | 70 (swap-free) | 0.039 s | `smart_vf2_layout`: VF2 phase 1, ordering `bfs_from_min_degree`, 0.144 s standalone |
| c2 | 70 (swap-free) | 0.019 s | `path_packing` (phase 0), 0.006 s standalone |

**Why the prediction assumed otherwise.**

- The second clause of L2 assumed that a tiling c1 places swap-free is
  always placed by c1's short-path shortcut. Since c2 runs only after that
  shortcut fails, c2 would then leave the output unchanged.
- Here c1's shortcut failed and its ordinary VF2 stage succeeded. c2's
  packing search runs before VF2 and found a different, equally swap-free
  layout first.

**Consequence.** The outputs differ, with the same gate count. Both are
exact (component check passed) and c2 is faster. Under the pre-registered
bounds this is AMBIGUOUS: not confirmed, and not refuted, since c2 is never
above min(m1, c1). The assumption behind the clause was wrong. The
candidate's behaviour matches its design.

## 3. Layout details (Part L)

| Device | (k3,k2) | Ground truth (30 s) | m1 2q | c1 2q / s | c2 2q / s | Swap-free |
|---|---|---|---|---|---|---|
| Auckland | (6,4) | feasible | 19 | 16 / 0.007 | 16 / 0.007 | 16 |
| Auckland | (5,6) | infeasible | 37 | 37 / 0.020 | 37 / 0.021 | 16 |
| Auckland | (4,7) | infeasible | 18 | 18 / 0.023 | 18 / 0.024 | 15 |
| Auckland | (6,3) | feasible | 15 | 15 / 0.007 | 15 / 0.006 | 15 |
| Torino | (21,35) | unknown | 147 | 77 / 0.022 | 77 / 0.021 | 77 |
| Torino | (25,29) | unknown | 178 | 178 / 2.07 | 178 / 2.06 | 79 |
| Torino | (30,21) | unknown | 192 | 192 / 2.04 | 192 / 2.06 | 81 |
| Torino | (37,11) | feasible | 195 | 195 / 2.05 | **85 / 0.068** | 85 |
| Torino | (44,0) | feasible | 244 | 244 / 2.08 | **88 / 0.029** | 88 |
| Torino | (20,36) | unknown | 160 | 76 / 0.023 | 76 / 0.020 | 76 |
| Kingston | (28,36) | unknown | 202 | 92 / 0.029 | 92 / 0.026 | 92 |
| Kingston | (32,30) | unknown | 214 | 214 / 2.15 | 214 / 2.24 | 94 |
| Kingston | (40,18) | feasible | 219 | 219 / 2.12 | **98 / 0.955** | 98 |
| Kingston | (46,9) | feasible | 263 | 263 / 2.09 | **101 / 0.034** | 101 |
| Kingston | (30,32) | unknown | 227 | 227 / 2.08 | 227 / 2.18 | 92 |
| Nighthawk | (10,45) | feasible | 92 | 92 / 1.90 | **65 / 0.016** | 65 |
| Nighthawk | (20,30) | feasible | 113 | 70 / 0.039 | 70 / 0.019 | 70 |
| Nighthawk | (40,0) | feasible | 152 | 152 / 2.09 | **80 / 0.020** | 80 |
| Nighthawk | (26,21) | feasible | 124 | 124 / 2.01 | **73 / 0.019** | 73 |

**What this shows beyond the scored items (not predicted).**

- **c2 fixes most c1 misses.** In 7 combinations c1 spent about 2 s and
  still routed with SWAPs. c2 placed them swap-free in 0.016-0.955 s, with
  1.4-2.8 × fewer two-qubit gates.
- **The packing search is not efficient enough on the large heavy-hex
  devices when many pairs are mixed with the 3-qubit paths.**
  - 7 of 19 ground-truth runs ended "unknown" at 30 s.
  - Three of them (Torino (21,35) and (20,36), Kingston (28,36)) are in fact
    feasible: c1's shortcut found a swap-free layout at once. So the
    exhaustive search is far from deciding these instances.
  - In 4 "unknown" combinations (Torino (25,29) and (30,21), Kingston (32,30)
    and (30,32)) neither c1 nor c2 found a swap-free layout. They may or may
    not be feasible.
  - This is the remaining layout weakness.
  - **Next step:** a better search order (for example, choose the most
    constrained uncovered qubit, or seed from c1's matching construction)
    before the full search.
- **Kingston (40,18) needed 0.955 s**, close to the 1 s bound.

## 4. Compile details (Parts C and R)

**Random dense circuits (90 per device).**

| Device | CAND vs REL: better / same / worse | CAND vs L3: below / equal / above |
|---|---|---|
| FakeAuckland | 82 / 8 / 0 | 3 / 75 / 12 |
| FakeKingston | 84 / 6 / 0 | 4 / 72 / 14 |

**Textbook circuits (44 compiles).**

- **Equal to L3:** CAND equals L3 everywhere except three cases:
  - QFT3 in gate form on Auckland: 9 against 7;
  - QFT5 in gate form on Auckland: 31 against 29;
  - GHZ star 5 on Kingston: 10 against 7, unchanged from REL. It is a
    degree-4 star on a degree-3 device, and routing differs.
- **Largest gains:**
  - Reverse5: 19-22 → 4 (the SWAPs are elided);
  - SwapNet4: 30 → 18 on Auckland, 15 on Kingston;
  - Dicke(4,2): 30 → 18;
  - W3-W5: 6/9/12 → 4/6/8.

**Correction to the pre-registration text (append-only note).**

- Section 3 of the pre-registration says "13 circuits" and C4 says "all 52
  textbook compiles". The locked script `core_fix_c2_eval.py` (hash in the
  pre-registration) defines 11 circuits:
  - W3, W4, W5;
  - Dicke(4,2);
  - QFT3, QFT4, QFT5;
  - GHZ star 4 and 5;
  - SwapNet4;
  - Reverse5.
- Each comes in two forms on two devices, so there are 44 compiles. The
  text count was a counting error in writing. The scored set is the
  script's. C4 is scored on 44, and the result is 0 of 44.

**Larger circuits (FakeKingston).**

| Circuit | 2q REL → CAND | Time REL → CAND |
|---|---|---|
| random_circuit 16q d20 | 970 → 843 | 54 → 66 ms |
| random_circuit 32q d20 | 2,411 → 2,342 | 84 → 127 ms |
| random_circuit 48q d15 | 4,116 → 3,756 | 140 → 153 ms |
| random_circuit 80q d10 | 1,446 → 1,204 | 109 → 111 ms |
| random_circuit 120q d8 | 1,354 → 1,253 | 109 → 136 ms |
| dense_pair_blocks 60q | 90 → 90 | 26 → 29 ms |
| dense_pair_blocks 120q | 180 → 180 | 34 → 43 ms |
| dense_pair_blocks 156q | 282 → 282 | 109 → 120 ms |
| k_chains 80q | 237 → 237 | 51 → 53 ms |
| linear_chain 80q | 237 → 237 | 50 → 54 ms |
| random_regular 80q | 1,356 → 1,296 | 138 → 171 ms |
| ghz_star 80q | 468 → 243 | 102 → 132 ms |

**Not run:** the supplementary repeat with the release core 2026-09-28.1
(optional in the pre-registration).

## 5. What this establishes, and what not

**Established on held-out inputs:**

- The two weaknesses of Addendum 271 are fixed.
  - Disjoint 3-qubit-path tilings that c1 missed are placed swap-free in
    well under 1 s, wherever the exhaustive search could decide feasibility.
  - Small dense circuits now come within 3-4 % of Qiskit L3's two-qubit
    count, against 70-80 % above before (REL / L3 = 1.71 and 1.79).
- The fixes cost:
  - no gates anywhere tested;
  - no correctness anywhere tested: 0 non-equivalent outputs, final-layout
    bookkeeping right for measured circuits, canonical path bit-identical;
  - up to about 1.5 × time on random circuits;
  - at most 1.29 × on the families PSF-Zero is built for.

**Not established:**

- hardware fidelity (fewer two-qubit gates is not measured fidelity);
- the live IBM Targets;
- the weighted layout path;
- the canonical basis;
- the 4 large heavy-hex tilings that neither layout could place, and whose
  feasibility is unknown.

## 6. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`core_fix_c2_raw.json`](../../data/2026-10-01/core_fix_c2/outputs/scored/core_fix_c2_raw.json) | 68,556 | `ebc54f1f235c13473cf9541915ee98b3c9338a1a4c3e9f8a5ae5f506f92bb440` |
| [`core_fix_c2_run.txt`](../../data/2026-10-01/core_fix_c2/outputs/scored/core_fix_c2_run.txt) | 45,859 | `f17de2f6ecd6aadc373aac955e22ebe7a03a3382ecde88723c80aff9f7307dc7` |
| [`core_fix_c2_score.txt`](../../data/2026-10-01/core_fix_c2/outputs/scored/core_fix_c2_score.txt) | 5,856 | `95b517c204e4ed8d5a46a7a6288078ca6aaaa4fd45513ada6b7bc5a8491fc9b9` |

The candidate files, the patch and the apply script are those listed in
the pre-registration, section 6. Their hashes are unchanged, and the run's
META line records them.

---

<!-- ===== Addendum 274 (source: spare-qubit-cliff-addendum-274-2026-10-01.md) ===== -->

> **Note added when merging:** The owner adopted c2 on the evidence (2026-10-01). At home the same evening the owner also adopted the Rust core 2026-09-29.1, on which every evaluation since 2026-09-29 ran, so that the tested combination is the release: commit `d358e87` makes psf_compile and psf_smart_layout 2026-10-01.1 and the core 2026-09-29.1 ([`patches/core_fix_c2_2026-10-01/release_2026-10-01.py`](../../patches/core_fix_c2_2026-10-01/release_2026-10-01.py)). Checks there: cargo test 9 of 9; 103 of 103 in the core, synthesis, layout, release and c2 test files run together. Run as one session, the whole [`benchmarks/`](../../benchmarks/) suite has collection errors that predate this release (one test module leaves a stand-in `psf_zero_core` behind; 9 such errors before, the same plus the new c2 test file after).

## Addendum 274 -- Adoption decision: psf_compile 2026-10-01.c2 and psf_smart_layout 2026-10-01.c2 (2026-10-01, workplace)

**Decision.** The project owner decided on 2026-10-01 to adopt both candidates evaluated in
`core-fix-c2-results-2026-10-01.md`.

**What the evaluation showed.**

- 13 of 14 predictions were CONFIRMED and none was refuted.
- L2 was AMBIGUOUS. In one tiling (FakeNighthawk, 20 and 30), c2 produced a different swap-free layout
  from c1, with the same 70 two-qubit gates. It was faster (0.039 s → 0.019 s), and both outputs were
  exact.
- The pre-registered automatic rule did not fire, because it required all 14 to be CONFIRMED. The
  owner adopted on the evidence, not on the rule. This is recorded so the distinction is not lost.

**Scope of what is adopted.**

| File | Version | Base |
|---|---|---|
| `psf_compile.py` | 2026-10-01.c2 | release 2026-09-28.1 |
| [`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) | 2026-10-01.c2 | release 2026-09-26.m1 |

- The layout file includes the 2026-09-29.c1 changes: the corrected feasibility check and the
  short-path shortcut. Adopting c2 therefore also adopts c1.
- [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) and [`benchmarks/core_fix_c2_eval.py`](../../benchmarks/core_fix_c2_eval.py) are added.
- The Rust core is not part of this decision.
  - The evaluation ran on core 2026-09-29.1, the 9/29 candidate, which is not yet adopted.
  - The Python changes do not depend on the core version.
  - Running the c2 files on the release core 2026-09-28.1 was not measured.

**Still to do (at home, where the repository is pushed).**

1. Apply the change from `core_fix_c2_2026-10-01.patch` (git diff against `9131cee`), or with
   [`apply_core_fix_c2_2026-10-01.py`](../../patches/core_fix_c2_2026-10-01/apply_core_fix_c2_2026-10-01.py).
2. Decide the release version strings. The files currently say `2026-10-01.c2` (candidate).
3. Update the two version-string tests, which expect 2026-09-28.1 and 2026-09-29.c1.
4. Run the test suite.
5. Give the pre-registration, results and this note Addendum numbers in Part 9. The next number is
   planned to be 272.
6. Add a README update following the psf-zero-repo-publish rules. Report numbers with their sandbox
   conditions, and never compare timings across machines.
7. Decide separately whether to adopt core 2026-09-29.1.

**Known limitation carried into the release.** Four large heavy-hex tilings remain unplaced by both
c1 and c2, with feasibility unknown:

- FakeTorino (25, 29) and (30, 21);
- FakeKingston (32, 30) and (30, 32).

---

<!-- ===== Addendum 275 (source: spare-qubit-cliff-addendum-275-2026-10-01.md) ===== -->

> **Note added when merging:** Exploratory, not pre-registered: the first AI front end a0 ([`benchmarks/psf_ai_compile_a0.py`](../../benchmarks/psf_ai_compile_a0.py)), developed before Addendum 276. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/ai_compile_a0/as_run/`](../../data/2026-10-01/ai_compile_a0/as_run/); the outputs are in [`data/2026-10-01/ai_compile_a0/`](../../data/2026-10-01/ai_compile_a0/).

## Addendum 275 -- Exploratory (not pre-registered): psf_ai_compile 2026-10-01.a0, a PSF-Zero front end for model-written circuits (2026-10-01, workplace)

**Status.** This is an exploratory prototype, measured on *development* inputs only. No claim here is
a confirmed result. A pre-registered test on new inputs would be the next step.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Python 3.11.15, Qiskit 2.5.2.
- **Compiler and core:** the adopted psf_compile 2026-10-01.c2 and psf_smart_layout 2026-10-01.c2, on
  core 2026-09-29.1. Neither file is changed.
- **Devices:** fake-provider only; no IBM access.

## 1. Idea (the project owner's, 2026-09-30/10-01)

- `lib.rs` and `psf_compile.py` stay as they are.
- What sits around them is specialised for the vLLM use: small circuits, many SWAPs and redundant
  gates, PennyLane-style `unitary` gates, and many compiles per task.
- The separation chosen is a separate module on top: `psf_ai_compile.py`. It calls
  `compile_for_hardware()` and nothing below it, so the compiler's release and its speed claims are
  untouched.

## 2. What the prototype does (`compile_for_model_circuit`)

For circuits of at most 8 qubits it keeps the best result (fewest 2-qubit gates, then 2-qubit depth)
over the following combinations:

- **Starting points:**
  - the input as written;
  - the input after Qiskit's `CommutativeCancellation`.
- **Routing seeds:** 0, 1, 2 and 3. It stops early when two seeds in a row give the same layout and
  count, which happens when the layout search has fixed the layout.
- **Polish of each routed result:** `CommutativeCancellation` on the physical circuit, then
  re-synthesis by the PSF-Zero core of every 2-qubit block whose optimal CX count is below what it holds,
  then translation. This repeats while the count drops, up to 3 rounds. The routed layout (initial and
  final) is kept.

Larger circuits go straight to `compile_for_hardware()`.

## 3. Development results

These are the same inputs used to develop c2. They are not held out.

**Random dense circuits, 3-5 qubits, 60 per row.** All 360 outputs (c2 and a0) are equivalent to their
input; statevector, read at the final layout.

| Device, seed | c2 | a0 | Qiskit L3 | a0 above c2 | a0 above L3 | Median time c2 / a0 |
|---|---|---|---|---|---|---|
| FakeAuckland, 7 | 824 | 788 | 807 | 0 | 3 | 11 / 49 ms |
| FakeAuckland, 8 | 829 | 787 | 793 | 0 | 1 | 10 / 48 ms |
| FakeKingston, 9 | 840 | 794 | 791 | 0 | 4 | 28 / 128 ms |

**Ablation** (sum of 2-qubit gates, median time):

| Configuration | Auckland seed 7 | Kingston seed 9 |
|---|---|---|
| c2 only (1 seed) | 824, 11 ms | 840, 29 ms |
| + 4 routing seeds | 801, 24 ms | 809, 59 ms |
| + commuted start | 819, 18 ms | 824, 53 ms |
| + polish | 818, 12 ms | 831, 31 ms |
| all three (a0) | 788, 53 ms | 794, 130 ms |

Routing seeds give the largest single gain, and the three combine.

**The 149 model circuits of 2026-09-30** (FakeAuckland, unique circuits). There were 0 fidelity changes
against c2 (component-wise check of the e2e harness).

| Task | Unique | c2 | a0 | L3 | a0 above L3 |
|---|---|---|---|---|---|
| dicke42 | 6 | 83 | 75 | 79 | 0 |
| w3 | 30 | 144 | 136 | 142 | 1 |
| w4 | 5 | 34 | 33 | 33 | 0 |
| qft3 | 19 | 86 | 86 | 86 | 0 |
| fill27 (27 qubits, fast path) | 9 | 138 | 138 | 174 | 0 |
| bell3, fill27g9, ghz3i, ghz5, singlet3 | 18 | 76 | 76 | 76 | 0 |

**Time.** The median a0 compile for these circuits is 10-54 ms. That is small next to the model's
generation time in the vLLM loop, which takes seconds per round.

## 4. Reading

- On development inputs the prototype reaches or beats Qiskit L3's 2-qubit count on small circuits.
  - The sums are below L3 on both Auckland rows, about equal on Kingston, and below L3 on the model's
    Dicke and W3 circuits.
  - It is never worse than c2.
  - It costs 4-5 × c2's time, still tens of milliseconds.
- **What this does not change:**
  - The vLLM pass rate. The model's failures in the go/no-go tests were wrong circuits (mostly phases),
    not compiler output.
  - What improves is the gate-count side: comparisons with L3 and with the StatePreparation baseline.

**Not established:**

- new inputs (a pre-registration would use fresh seeds and new textbook or model circuits);
- hardware fidelity;
- error-aware layout choice, where a small circuit could afford to score every embedding by the device's
  error rates. This was not tried here, and the e2e harness passes no error rates.

## 5. Next steps (proposed)

1. Pre-register a held-out test of a0 against c2 and L3: new random seeds, new textbook circuits, and
   both devices.
2. If confirmed, wire it into the e2e harness (v11) as the compile step, behind a flag.
3. Optionally add error-aware layout scoring for small circuits, when a Target with error rates is
   available.

**Files:** `psf_ai_compile.py`, 5,926 bytes, normalized SHA-256
`7105fb3591df4648f64f4c260d1e3ad427e5323ee70d36aa0f44483f12c69949`.

---

<!-- ===== Addendum 276 (source: spare-qubit-cliff-addendum-276-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration of a0 on held-out inputs. Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run. The a0 module was then named `psf_ai_compile.py`; in this repository it is [`benchmarks/psf_ai_compile_a0.py`](../../benchmarks/psf_ai_compile_a0.py) and [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) is the latest version (a5).

## Addendum 276 -- Pre-registration: psf_ai_compile 2026-10-01.a0 (PSF-Zero front end for model-written circuits) on held-out inputs (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.
The prototype and its development results are in `ai-compile-a0-exploratory-2026-10-01.md`, which lists
the development inputs. One harness dry run is disclosed in section 6.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Python 3.11.15, Qiskit 2.5.2.
- **Compiler:** the adopted psf_compile 2026-10-01.c2 and psf_smart_layout 2026-10-01.c2.
- **Core:** 2026-09-29.1, sandbox build.
- **Devices:** fake-provider only; no IBM access.

## 1. What is tested

`psf_ai_compile.compile_for_model_circuit()` works on circuits of at most 8 qubits:

- it starts from two points, the input and the input after CommutativeCancellation;
- it tries routing seeds 0-3 for each;
- it polishes each routed result (commutative cancellation plus PSF-core re-synthesis of blocks that save
  CX);
- it keeps the result with the fewest 2-qubit gates.

Larger circuits go straight to `compile_for_hardware()`. `psf_compile.py` and the core are not changed.

## 2. Design (`ai_a0_eval.py`; helpers from the locked `core_fix_c2_eval.py`)

**Arms:**

- **C2:** `compile_for_hardware(entangling_basis="cx", layout_search=True, seed_transpiler=0)`.
- **A0:** `compile_for_model_circuit()` with default settings.
- **L3:** Qiskit `transpile(optimization_level=3, seed_transpiler=0)`, as the reference.

**Devices:** FakeAuckland (cx), FakeKingston (cz), FakeTorino (cz). FakeTorino was not used in a0's
development.

**Inputs.** None of these were used in development.

- **R:** random dense circuits from Python `random` seeds 3001-3060, on each device.
  - 3-5 qubits and 6-20 gates.
  - Single-qubit gates: h, x, s, t, ry.
  - 2-qubit gates: cx, cry, crz, cp, swap, cz. Half of the SWAPs are written as `unitary`.
- **T:** 12 new textbook circuits, each in gate form and in PennyLane-style `unitary` form, on each
  device (72 compiles):
  - inverse QFT4 with leading swaps;
  - W6 (cry cascade);
  - a linear GHZ6;
  - Bernstein-Vazirani with 5 qubits;
  - a QAOA ring of 4 and of 5 qubits (p = 1);
  - a 4-qubit hardware-efficient ansatz (2 layers, cz ring);
  - a 4-qubit Heisenberg chain (2 Trotter steps, rxx/ryy/rzz);
  - Grover on 3 qubits (one iteration, ccz);
  - QPE with 3 counting qubits;
  - a 5-qubit cyclic shift by SWAPs;
  - two Bell pairs exchanged by a SWAP.
- **M:** measured copies of the first 30 Auckland R circuits, compiled by A0 and run on AerSimulator
  (20,000 shots). The TVD to the exact distribution is recorded.
- **F:** 8 `random_circuit` inputs of 10-20 qubits on FakeKingston (seeds 4001-4008), above the 8-qubit
  limit.

**Recorded:** 2-qubit counts; A0 wall time; and component-wise equivalence (statevector, read at the
final layout, ancillas in |0>, where the support is 12 qubits or fewer).

## 3. Predictions

**C0:** the loaded versions are 2026-10-01.c2 (compile and layout) and 2026-10-01.a0.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| A1 | A0 outputs are exact | no checked R/T output below 1 - 1e-9, and at most 5 % not checkable | any checked output below |
| A2 | A0 never uses more 2-qubit gates than C2 (holds by construction; this checks the implementation) | 0 R/T circuits with A0 > C2 | any |
| A3 | A0 reaches Qiskit L3 | every device: sum A0 <= 1.02 × sum L3, **and** A0 > L3 in <= 10 % of R circuits | any device > 1.06 × or > 20 % |
| A4 | A0 improves on C2 | every device: sum A0 <= 0.97 × sum C2 | any device > 0.99 × |
| A5 | textbook circuits at L3 level | A0 <= L3 in >= 90 % of the 72 T compiles | < 75 % |
| A6 | A0 is fast enough for the vLLM loop | per-device median A0 time (R and T) <= 250 ms, **and** max <= 2 s | any median > 500 ms |
| A7 | measured circuits come out right | TVD <= 0.03 on all 30 | above on any |
| A8 | larger circuits fall through unchanged | A0 output digest = C2 digest on all 8 F inputs | any differs |

Between the bounds: ambiguous. No re-runs to improve a score.

**Expectations from development (Auckland and Kingston, 60 circuits each).**

| Prediction | Development values |
|---|---|
| A3: A0 / L3 | 0.976, 0.992, 1.004 |
| A3: share above L3 | 1.7-6.7 % |
| A4: A0 / C2 | 0.945-0.956 |
| A6: median time | 48-128 ms |

FakeTorino is new to a0, so its numbers could fall outside these ranges.

## 4. What this can and cannot establish

**It can establish** whether the front end keeps its development advantage on new circuits and on a new
device, at a time cost acceptable inside the vLLM loop.

**It cannot establish:**

- hardware fidelity;
- the vLLM pass rate, which depends on the model's circuits rather than on the compiler;
- error-aware layout choice, which a0 does not have;
- live IBM Targets.

Times are sandbox times, not compared with home or the pod.

## 5. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `ai_a0_eval.py` | 11,849 | `0e50983993a52accefe557388518182baede91cb6e3a2034077b4dd6493c215d` |
| `psf_ai_compile.py` (2026-10-01.a0) | 5,926 | `7105fb3591df4648f64f4c260d1e3ad427e5323ee70d36aa0f44483f12c69949` |
| `core_fix_c2_eval.py` (helpers, locked earlier today) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |
| `psf_compile.py` 2026-10-01.c2 | 95,818 | `e22dc6ddf4bf717d722a3f7ce01e23ecbe096d41730f7fe54c0d29eeaa4a3bd1` |
| `psf_smart_layout.py` 2026-10-01.c2 | 32,708 | `f0d38519adad04864de42c456564a8321584ae3feadd744101a4cb719070fc14` |

```
PYTHONPATH=<core 2026-09-29.1> python -u ai_a0_eval.py run --compile <psf_compile.py> \
  --layout <psf_smart_layout.py> --ai psf_ai_compile.py --out ai_a0_raw.json > ai_a0_run.txt 2>&1
python ai_a0_eval.py score --out ai_a0_raw.json > ai_a0_score.txt
```

The two eval scripts must be in the same folder.

## 6. Dry run before locking (disclosed)

**Inputs:** `--dry` uses seeds offset by 700,000. That gives 4 R circuits per device, the first 2
textbook circuits (IQFT4 and W6, 12 compiles), 4 M circuits and 2 F inputs.

**Results:**

- A1, A2, A3, A6, A7 and A8 scored CONFIRMED.
- A4 and A5 scored AMBIGUOUS. With 4 circuits per device the A0/C2 ratios were 0.91-0.97, and 10 of the
  12 textbook compiles had A0 <= L3: IQFT4 on FakeTorino came out 17 against L3's 16.

**Changes after the dry run:** none to the predictions or thresholds. One cosmetic line in the scorer
was cleaned.

**No scored input was compiled before locking.**

---

<!-- ===== Addendum 277 (source: spare-qubit-cliff-addendum-277-2026-10-01.md) ===== -->

> **Note added when merging:** 7 of 8 CONFIRMED, A5 AMBIGUOUS (3-qubit gates and ring-shaped circuits). Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/ai_compile_a0/as_run/`](../../data/2026-10-01/ai_compile_a0/as_run/); the outputs are in [`data/2026-10-01/ai_compile_a0/`](../../data/2026-10-01/ai_compile_a0/).

## Addendum 277 -- Results: psf_ai_compile 2026-10-01.a0 on held-out inputs -- 7 of 8 CONFIRMED, A5 (textbook circuits) AMBIGUOUS (2026-10-01)

**Pre-registration:** `docs/findings/ai-compile-a0-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 276 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Qiskit 2.5.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Devices:** fake-provider only.
- **Hashes:** the run's META line records the same hashes as the pre-registration.

## 1. Verdict

**C0:** OK.

| ID | Result | Numbers |
|---|---|---|
| A1 | CONFIRMED | 252 R/T outputs, 0 not equivalent, 0 not checkable |
| A2 | CONFIRMED | 0 of 252 with A0 above C2 |
| A3 | CONFIRMED | A0/L3 = 0.975 (Auckland), 0.989 (Kingston), 0.977 (Torino); A0 above L3 in 3.3 %, 8.3 %, 5.0 % |
| A4 | CONFIRMED | A0/C2 = 0.949, 0.957, 0.941 |
| A5 | **AMBIGUOUS** | A0 <= L3 in 77.8 % of the 72 textbook compiles (confirm >= 90 %, refute < 75 %) |
| A6 | CONFIRMED | median 47 / 133 / 114 ms, max 0.31 s |
| A7 | CONFIRMED | 30 of 30 measured circuits within TVD 0.03 (max 0.0112) |
| A8 | CONFIRMED | 8 of 8 inputs above 8 qubits: output identical to C2 |

Not every item is CONFIRMED, so the pre-registered decision line reads "NOT ALL CONFIRMED". Nothing was
re-run.

## 2. Random circuits (R, 60 per device)

| Device | Sum of 2-qubit gates: C2 / A0 / L3 | A0 below / equal / above L3 |
|---|---|---|
| FakeAuckland | 850 / 807 / 828 | 12 / 46 / 2 |
| FakeKingston | 879 / 841 / 850 | 13 / 42 / 5 |
| FakeTorino (new to a0) | 884 / 832 / 852 | 17 / 40 / 3 |

The development advantage held on new circuits and on the new device: below Qiskit L3 in sum everywhere,
and 5-6 % below C2.

## 3. Textbook circuits (T): where A5 fell short

**Sums over the 72 compiles:** C2 849, A0 833, L3 822.

**A0 above L3 (16 compiles).** Each circuit counts twice, once in gate form and once in unitary form.

- **Grover3 on all three devices** (6 compiles): 18-19 against 17. The circuit contains 3-qubit `ccz`
  gates. A0's improvements all act on 2-qubit blocks, so the 3-qubit gates are decomposed by Qiskit's
  default path, and L3 does better on them.
- **Ring-shaped interaction graphs on the heavy-hex Kingston and Torino** (10 compiles):
  - HEA4 (cz ring): 18-20 against 17;
  - QAOAring5 on Kingston: 18 against 16;
  - QAOAring4 on Torino: 12 against 11;
  - IQFT4 on Torino: 17 against 16.

  A ring cannot be embedded in heavy-hex, so routing is needed. L3's layout and routing search (more
  Sabre trials plus VF2PostLayout) finds a cheaper routing than four seeds of level-1 routing.
- HEA4 in `unitary` form stayed at 20, where the gate form reached 18. The commuted starting point does
  not see through `unitary` gates.

**A0 below L3 (8 compiles):**

- QAOAring5 and QPE3 on Auckland (14 against 16, 13 against 15);
- IQFT4 on Kingston (17 against 18);
- QPE3 on Torino (13 against 15).

## 4. Reading

- On random small dense circuits, which are the kind a model writes, the front end is at or below
  Qiskit L3 on all three devices. It costs about 50-130 ms per compile, without changing
  `psf_compile.py` or the core.
- **Two weaknesses remain,** both visible in the textbook set:
  1. 3-qubit gates, which are handled by Qiskit's default decomposition rather than optimised;
  2. ring-shaped interaction graphs that need routing on heavy-hex.
- **Candidate remedies for a next version (a1):**
  - more routing seeds, or a level-3-style routing call for small circuits, keeping the best;
  - decomposing 3-qubit gates before the front end works on 2-qubit blocks;
  - converting `unitary` gates to standard gates when they match one, before the commuted starting
    point.

  Each would need a new pre-registration on new inputs.
- The vLLM pass rate is not affected. This front end changes gate counts, not which circuits are
  correct.

## 5. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`ai_a0_raw.json`](../../data/2026-10-01/ai_compile_a0/outputs/scored/ai_a0_raw.json) | 50,391 | `f738fe6182019611284ef04f9eb748c164d783159b2967b8370c9aa610fc7c60` |
| [`ai_a0_run.txt`](../../data/2026-10-01/ai_compile_a0/outputs/scored/ai_a0_run.txt) | 40,766 | `39652cf826315e2278e11902acd5ab1d9d433d627222afa95a81512bac898880` |
| [`ai_a0_score.txt`](../../data/2026-10-01/ai_compile_a0/outputs/scored/ai_a0_score.txt) | 6,323 | `07cbfc4e35d205eebbcc80f835e74c0508ef496d5029c6a0af980250ce265514` |

---

<!-- ===== Addendum 278 (source: spare-qubit-cliff-addendum-278-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration of a1 (3-qubit decomposition, L3 placement candidates). Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 278 -- Pre-registration: psf_ai_compile 2026-10-01.a1 on held-out inputs (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Qiskit 2.5.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Devices:** fake-provider only; no IBM access.

## 1. Why a1, and what changed from a0 (development, not pre-registered)

The held-out test of a0 (`ai-compile-a0-results-2026-10-01.md`) scored 7 of 8 CONFIRMED. A5 was
AMBIGUOUS: only 77.8 % of the textbook compiles were at or below L3. The misses came from two sources:

- 3-qubit gates (Grover3);
- ring-shaped interaction graphs that need routing on heavy-hex (HEA4, QAOA rings, IQFT4).

a1 keeps a0 unchanged as the first part of its search, so its candidate set contains a0's. It adds:

1. **Split starting points.** When the input has gates on 3 or more qubits, two more starting points
   use those gates decomposed: as written, and after commutative cancellation.
2. **Expanded starting point.** When the input has 2-qubit `unitary` gates, one more starting point
   re-expresses every 2-qubit gate in CX form (`psf_compile.compile`) and then commutatively cancels.
3. **Level-3 layout candidate.** One more candidate per starting point routes from the initial layout
   that Qiskit's level-3 search picks on the (split) input. Only the layout is borrowed; synthesis,
   routing (level 1) and the polish stay PSF-Zero's.
4. **Seeds.** The extra starting points use the first routing seed only; a0's two starting points keep
   all four seeds.

**Development data:** the a0 held-out set, now seen. That is R seeds 3001-3060 on Auckland, Kingston
and Torino, and the a0 textbook set. All outputs were equivalent.

| Device | R sums: a0 / a1 / L3 | a1 above L3 (R) | T sums: a0 / a1 / L3 | a1 above L3 (T) | a1 median time (R) |
|---|---|---|---|---|---|
| FakeAuckland | 807 / 796 / 828 | 1 of 60 | 273 / 270 / 278 | 0 of 24 | 101 ms |
| FakeKingston | 841 / 816 / 850 | 0 of 60 | 283 / 272 / 274 | 0 of 24 | 224 ms |
| FakeTorino | 832 / 810 / 852 | 0 of 60 | 277 / 266 / 270 | 0 of 24 | 206 ms |

a1 was never above a0 on these inputs.

**Ablation on Kingston (R + T), before item 4 was added:**

- full a1: 1,084 two-qubit gates;
- without the level-3 layout candidate: 1,117;
- without the expanded start: 1,089;
- without the split: 1,084 (the a0 textbook set has one 3-qubit-gate circuit).

## 2. Design (`ai_a1_eval.py`; helpers from the locked `core_fix_c2_eval.py`)

**Arms:**

- **C2:** `compile_for_hardware(cx, layout_search=True, seed 0)`.
- **A0:** psf_ai_compile 2026-10-01.a0.
- **A1:** psf_ai_compile 2026-10-01.a1.
- **L3:** Qiskit level 3, seed 0, as the reference.

**Devices:** FakeAuckland, FakeKingston, FakeTorino and **FakeFez**. FakeFez has not been used before in
these tests.

**Inputs.** None of these were used in developing a0 or a1.

- **R:** random dense circuits from Python `random` seeds 5001-5060 on each device. The generator is the
  same as before: 3-5 qubits and 6-20 gates, with half of the SWAPs written as `unitary`.
- **T:** a third textbook set of 12 circuits, each in gate form and in `unitary` form, on each device
  (96 compiles). The `unitary` matrices are taken via `Operator`, so multi-controlled gates work too.
  - CCZchain4: two overlapping ccz gates;
  - CuccaroMajUma3: the MAJ and UMA blocks of the ripple-carry adder;
  - Grover4: mcx;
  - QAOAring6 with rzz;
  - HEA5ring with a cx ring;
  - IsingRing5: 2 Trotter steps;
  - DJ4;
  - DraperAdd2;
  - GHZstar6;
  - W4tree, which contains a ccx;
  - TeleportUnitary;
  - Clifford4: `random_clifford(4, seed=5101)`.
- **M:** measured copies of the first 30 Auckland R circuits, compiled by A1 and run on AerSimulator
  (20,000 shots).
- **F:** 8 `random_circuit` inputs of 10-20 qubits on FakeKingston (seeds 6001-6008).

## 3. Predictions

**C0:** the versions loaded are c2, c2, a0 and a1.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| B1 | A1 outputs are exact | no checked R/T output below 1 - 1e-9, and at most 5 % not checkable | any checked output below |
| B2 | A1 improves on A0 | every device: R+T sum A1 < A0, **and** A1 > A0 in <= 2 % of R+T circuits | any device: sum not lower, or > 5 % worse |
| B3 | A1 is at or below L3 on random circuits | every device: R sum A1 <= 1.00 × L3, **and** A1 > L3 in <= 5 % | any device > 1.04 × or > 15 % |
| B4 | textbook circuits at L3 level (a0 missed this) | A1 <= L3 in >= 90 % of the 96 T compiles | < 75 % |
| B5 | fast enough for the vLLM loop | per-device median A1 time (R+T) <= 400 ms, **and** max <= 2 s | any median > 800 ms |
| B6 | measured circuits come out right | TVD <= 0.03 on all 30 | above on any |
| B7 | larger circuits fall through unchanged | A1 digest = C2 digest on all 8 F | any differs |

Between the bounds: ambiguous. No re-runs to improve a score.

**Expectations stated before running:**

- **B2:** A1 > A0 should not happen, since the candidate set contains a0's. A nonzero count would point
  to an implementation fault.
- **B3:** development R ratios were 0.95-0.96.
- **B4:** the third set has more 3-qubit gates and rings than the a0 set. A6-style misses are the main
  risk.
- **B5:** development medians were 100-224 ms on R; FakeFez is the size of Kingston.

## 4. What this cannot establish

- hardware fidelity;
- the vLLM pass rate;
- error-aware layout;
- live Targets.

Times are sandbox times.

## 5. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `ai_a1_eval.py` | 13,441 | `b30a46508fa858363d115958b7d246f75a2ae1de6852c1ddb602963c0c639f5a` |
| `psf_ai_compile.py` 2026-10-01.a1 | 9,336 | `625e569dee1e69fae0890b5f89ea8acc8b659869bcf6cbbae5594df9ecd6b281` |
| `psf_ai_compile.py` 2026-10-01.a0 | 5,926 | `7105fb3591df4648f64f4c260d1e3ad427e5323ee70d36aa0f44483f12c69949` |
| `core_fix_c2_eval.py` (helpers) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |
| `psf_compile.py` / `psf_smart_layout.py` 2026-10-01.c2 | | as in the c2 pre-registration |

```
PYTHONPATH=<core 2026-09-29.1> python -u ai_a1_eval.py run --compile <psf_compile.py> --layout <psf_smart_layout.py> \
  --a0 psf_ai_compile_a0.py --a1 psf_ai_compile_a1.py --out ai_a1_raw.json > ai_a1_run.txt 2>&1
python ai_a1_eval.py score --out ai_a1_raw.json > ai_a1_score.txt
```

## 6. Dry runs before locking (disclosed)

**Dry-run inputs:** `--dry` uses R seeds offset by 800,000 (3 per device), 2 F inputs, and its own two
textbook circuits.

**First dry run.**

- It used the first two circuits of the then textbook set, ToffoliAdder and SwapTest5.
- It showed a1 above a0 on SwapTest5 in `unitary` form on Kingston and Fez (27 against 26). The early a1
  replaced a0's starting points with the split ones.
- **Changes made:**
  - a1 was changed to keep a0's starting points first (item 1 above);
  - ToffoliAdder and SwapTest5 were moved out of the scored set into the dry-run set, and replaced by
    CCZchain4 and CuccaroMajUma3, which no run has compiled.
  - The `unitary` form was switched to `Operator` matrices, because MCXGate has no `to_matrix`.

**Second dry run.** All CONFIRMED except B5 (Kingston 398 ms, Fez 433 ms medians, on 7 items per
device of which 4 have 3-qubit gates). Item 4 (one seed on the extra starting points) was then added.

**Third dry run, with the final files.** All 7 CONFIRMED. Medians were 144-318 ms.

**No scored input was compiled before locking.**

---

<!-- ===== Addendum 279 (source: spare-qubit-cliff-addendum-279-2026-10-01.md) ===== -->

> **Note added when merging:** 7 of 7 CONFIRMED. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/ai_compile_a1/as_run/`](../../data/2026-10-01/ai_compile_a1/as_run/); the outputs are in [`data/2026-10-01/ai_compile_a1/`](../../data/2026-10-01/ai_compile_a1/).

## Addendum 279 -- Results: psf_ai_compile 2026-10-01.a1 on held-out inputs -- 7 of 7 CONFIRMED (2026-10-01)

**Pre-registration:** `docs/findings/ai-compile-a1-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 278 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Qiskit 2.5.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Devices:** fake-provider only.
- **Hashes:** the run's META line records the same hashes as the pre-registration. Nothing was re-run.

## 1. Verdict

**C0:** OK.

| ID | Result | Numbers |
|---|---|---|
| B1 | CONFIRMED | 336 R/T outputs, 0 not equivalent, 0 not checkable |
| B2 | CONFIRMED | A1 below A0 in sum on every device; A1 above A0 in 0 of 336 |
| B3 | CONFIRMED | A1/L3 on R = 0.975 (Auckland), 0.973 (Kingston), 0.968 (Torino), 0.973 (Fez); A1 above L3 in 0 of 240 |
| B4 | CONFIRMED | A1 <= L3 in 99.0 % of the 96 textbook compiles (95 of 96) |
| B5 | CONFIRMED | median 94 / 208 / 204 / 206 ms, max 0.77 s |
| B6 | CONFIRMED | 30 of 30 measured circuits within TVD 0.03 (max 0.0148) |
| B7 | CONFIRMED | 8 of 8 inputs above 8 qubits: output identical to C2 |

**Decision line:** ALL CONFIRMED.

## 2. Random circuits (R, 60 per device)

| Device | Sums: C2 / A0 / A1 / L3 | A1 below / equal / above L3 |
|---|---|---|
| FakeAuckland | 799 / 778 / 770 / 790 | 11 / 49 / 0 |
| FakeKingston | 829 / 804 / 794 / 816 | 12 / 48 / 0 |
| FakeTorino | 821 / 796 / 788 / 814 | 13 / 47 / 0 |
| FakeFez | 829 / 804 / 794 / 816 | 12 / 48 / 0 |

FakeKingston and FakeFez gave identical sums in all four arms. In this fake-provider version the two
seem to share the same coupling map; this was not checked edge by edge. Fez therefore does not add an
independent topology here.

## 3. Textbook circuits (T, third set, 96 compiles)

**Totals:** C2 2,381, A0 2,307, **A1 2,145**, L3 2,208. A1 is below L3 in 31 compiles, equal in 64 and
above in 1.

| Circuit (both forms, 4 devices) | C2 | A0 | A1 | L3 |
|---|---|---|---|---|
| CCZchain4 | 136 | 128 | 116 | 116 |
| CuccaroMajUma3 | 204 | 192 | 192 | 192 |
| Grover4 | 832 | 822 | 730 | 736 |
| QAOAring6 | 180 | 174 | 166 | 174 |
| HEA5ring | 212 | 212 | 204 | 204 |
| IsingRing5 | 292 | 290 | 282 | 284 |
| DJ4 | 24 | 24 | 24 | 24 |
| DraperAdd2 | 108 | 88 | 88 | 88 |
| GHZstar6 | 74 | 70 | 70 | 88 |
| W4tree | 139 | 131 | 122 | 126 |
| TeleportUnitary | 56 | 56 | 40 | 56 |
| Clifford4 | 124 | 120 | 111 | 120 |

**The one compile above L3:** W4tree in gate form on FakeAuckland, 16 against 14. That circuit contains
a ccx.

**For comparison:** a0 would have met B4's bound in only 68.8 % of these compiles. The two a0
weaknesses (3-qubit gates, ring-shaped graphs on heavy-hex) are what a1 fixed.

## 4. Reading

- On held-out inputs the AI front end is now at or below Qiskit L3 everywhere tested:
  - random small dense circuits on four devices: 0 of 240 above L3, sums 2.5-3.2 % below;
  - textbook circuits with 3-qubit gates and rings: 95 of 96 at or below L3, sum 2.9 % below.
- It never loses to a0, keeps every output exact (final-layout bookkeeping included), and takes about
  0.1-0.2 s per compile.
- `psf_compile.py` and the Rust core were not changed for this. The gains come from the front end:
  - several starting points;
  - several routing seeds;
  - the level-3 initial layout as one extra candidate;
  - a commutation-plus-PSF-re-synthesis polish.
- **Borrowed from Qiskit:** CommutativeCancellation, decomposition of 3-qubit gates, and the level-3
  *layout* choice.
- **Not borrowed:** level-3 synthesis and routing are not used. Every 2-qubit block that is
  re-synthesised goes through the PSF-Zero core.

**Not established:**

- hardware fidelity;
- the vLLM pass rate (the front end changes gate counts, not which circuits are correct);
- error-aware layout;
- live Targets;
- circuits above 8 qubits (they fall through to c2 unchanged, by design).

## 5. Next steps (proposed)

1. Use a1 as the compile step of the e2e harness (v11), behind a flag, and record the gate counts the
   model's circuits now get.
2. Optionally add error-aware layout scoring when a Target with error rates is available.

## 6. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`ai_a1_raw.json`](../../data/2026-10-01/ai_compile_a1/outputs/scored/ai_a1_raw.json) | 65,910 | `ebaeb84525d124497650296231a0014990c5897f82f57144ff57e64c932059fb` |
| [`ai_a1_run.txt`](../../data/2026-10-01/ai_compile_a1/outputs/scored/ai_a1_run.txt) | 52,638 | `18c86f79f415a98e780c0f98f8e530e1d8d718cf217638ec846195fe34c4d547` |
| [`ai_a1_score.txt`](../../data/2026-10-01/ai_compile_a1/outputs/scored/ai_a1_score.txt) | 9,058 | `9b95d3fba5d06ca1b03afd32d916d9aa950df6a6006a129731c9aa3107cac8e5` |

---

<!-- ===== Addendum 280 (source: spare-qubit-cliff-addendum-280-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration: do fewer two-qubit gates give higher fidelity under the fake devices' noise (Qiskit Aer noise models from published calibration; not real hardware)? Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 280 -- Pre-registration: do fewer 2-qubit gates give higher fidelity under the fake devices' noise? Released stack vs c2 vs psf_ai_compile a1 vs Qiskit L3 (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Core:** 2026-09-29.1, sandbox build.
- **Hardware:** no IBM access and no hardware. This is a noise *simulation*.

## 1. Why

Every PSF-Zero comparison so far counts 2-qubit gates. Whether fewer gates actually give better results
has not been measured. This test is the inexpensive step before any hardware run (which happens only at
home, on the owner's signal).

## 2. Design (`nf_eval.py`; helpers from the locked `core_fix_c2_eval.py`)

**Arms.** Every arm uses `compile_for_hardware(entangling_basis="cx", layout_search=True,
seed_transpiler=0)` unless noted.

| Arm | What it is |
|---|---|
| REL | the released stack: psf_compile 2026-09-28.1 + psf_smart_layout 2026-09-26.m1 |
| C2 | the adopted stack: psf_compile and psf_smart_layout 2026-10-01.c2 |
| A1 | psf_ai_compile 2026-10-01.a1 on top of C2 |
| L3 | Qiskit `transpile(optimization_level=3, seed_transpiler=0)` |

**Noise.** `qiskit_aer.noise.NoiseModel.from_backend(<fake device>)`:

- per-gate depolarizing error plus thermal relaxation, from the snapshot's gate errors, T1/T2 and gate
  lengths;
- no idle (delay) noise, no crosstalk;
- readout is irrelevant, because nothing is measured.

**Fidelity.**

- F = <psi|rho|psi>.
  - psi is the ideal output of the logical circuit from |0...0>.
  - rho is the density matrix of the compiled circuit, taken on its final-layout qubits (AerSimulator,
    density_matrix method; unused qubits truncated).
- Infidelity = 1 - F.
- The same simulation without noise checks exactness (N0).

**Devices:** FakeAuckland (27 qubits, cx) and FakeTorino (133 qubits, cz).

**Inputs:** new random dense circuits from Python `random` seeds 7001-7040 on each device (40 per device).
The generator is the same as before: 3-5 qubits, 6-20 gates, half of the SWAPs written as `unitary`.

**Secondary set, reported without prediction:** the unique model-written circuits of 2026-09-30 with
at most 8 qubits, on FakeAuckland. They were used in development.

## 3. Predictions

**C0:** the versions loaded are 2026-09-28.1, 2026-09-26.m1, and 2026-10-01.c2 / c2 / a1.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| N0 | every compiled output is exact without noise | noiseless F >= 1 - 1e-9 for all 4 arms, all circuits | any below |
| N1 | A1 beats the released stack under noise | every device: mean infidelity A1 < REL, **and** A1 lower than REL in >= 70 % of circuits where their 2-qubit counts differ | any device: mean A1 >= REL |
| N2 | A1 is at Qiskit L3's level under noise | every device: mean infidelity A1 <= 1.05 × L3 | any device > 1.15 × L3 |
| N3 | the 2-qubit count is the right proxy | within a circuit, between two arms with different 2-qubit counts, the one with fewer 2-qubit gates has the lower infidelity in >= 70 % of such pairs, per device | < 55 % on any device |
| N4 | A1 is not worse than C2 under noise | every device: mean infidelity A1 <= C2 | any device > 1.02 × C2 |

Between the bounds: ambiguous. No re-runs to improve a score.

**Expectations stated before running.**

- **N1:** REL has about 40 % more 2-qubit gates than A1 on such circuits, so a clear gap is expected.
- **N2:** A1 has about 3 % fewer 2-qubit gates than L3, so the infidelities should be close. Layout
  matters too, because on Torino the error rates differ from qubit to qubit and the arms may pick
  different physical qubits.
- **N3:** on Torino, the physical qubits chosen can outweigh a one-gate difference. That is why the bound
  is 70 % and not higher.

## 4. What this cannot establish

- **Hardware.** The noise model is a simplified snapshot: no idle noise, no crosstalk, no drift.
- Circuits with measurement.
- Larger circuits.
- The vLLM pass rate.

**A positive result here is a reason to run the hardware test, not a substitute for it.**

## 5. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `nf_eval.py` | 11,813 | `9a2fe1a3beb18da63dfd58440f74ba4c330266753040917a5b8ad0756e6bbf71` |
| `psf_ai_compile.py` 2026-10-01.a1 | 9,336 | `625e569dee1e69fae0890b5f89ea8acc8b659869bcf6cbbae5594df9ecd6b281` |
| `core_fix_c2_eval.py` (helpers) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |
| release `psf_compile.py` 2026-09-28.1 | 84,542 | `3616efc8b8a7bea184d6170fb7379d509d0bfec9828f4eb2f703dd4d26d8c60b` |
| release [`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) 2026-09-26.m1 | 22,450 | `a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875` |
| c2 `psf_compile.py` / `psf_smart_layout.py` | 95,818 / 32,708 | `e22dc6dd...a3bd1` / `f0d38519...70fc14` |

```
PYTHONPATH=<core>:<repo>/benchmarks python -u nf_eval.py run --rel-compile <repo>/psf_compile.py \
  --rel-layout <repo>/benchmarks/psf_smart_layout.py --compile <c2>/psf_compile.py --layout <c2>/psf_smart_layout.py \
  --a1 psf_ai_compile_a1.py --model-dir <repo>/data/2026-09-30 --v10-dir <dir of e2e_vllm_psf_v10.py> \
  --out nf_raw.json > nf_run.txt 2>&1
python nf_eval.py score --out nf_raw.json > nf_score.txt
```

## 6. Dry run before locking (disclosed)

**Inputs:** `--dry` uses seeds offset by 900,000, 3 circuits per device, and no model circuits.

**First scoring.** N0, N1, N2 and N4 were CONFIRMED. N3 was then defined as a Spearman correlation
across all compiled outputs. It scored 0.87 on Auckland and -0.09 on Torino, and so came out REFUTED.

- **Why that was the wrong measure:** it pools different circuits, and their sizes and the physical
  qubits they land on dominate the infidelity.
- **Change:** N3 was redefined as the within-circuit pairwise comparison above. On the same dry-run data
  it gives 85.7 % (Auckland) and 81.8 % (Torino).
- This change was made after seeing dry-run data, on dev seeds only, and it is disclosed here.

The model-circuit loading path was smoke-tested on one file (loading only, no compile).

**No scored input was compiled before locking.**

---

<!-- ===== Addendum 281 (source: spare-qubit-cliff-addendum-281-2026-10-01.md) ===== -->

> **Note added when merging:** N0, N1, N3 CONFIRMED; N2 and N4 AMBIGUOUS. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/noisy_fidelity/as_run/`](../../data/2026-10-01/noisy_fidelity/as_run/); the outputs are in [`data/2026-10-01/noisy_fidelity/`](../../data/2026-10-01/noisy_fidelity/).

## Addendum 281 -- Results: noisy-simulation fidelity, released stack vs c2 vs a1 vs Qiskit L3 -- N0, N1, N3 CONFIRMED; N2, N4 AMBIGUOUS (2026-10-01)

**Pre-registration:** `docs/findings/noisy-fidelity-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 280 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2, core 2026-09-29.1.
- **Noise:** `NoiseModel.from_backend` of FakeAuckland and FakeTorino (gate noise only).
- **Hashes:** the run's META line records the pre-registered hashes. Nothing was re-run.

## 1. Verdict

**C0:** OK.

| ID | Result | Numbers |
|---|---|---|
| N0 | CONFIRMED | all 4 arms exact without noise on all 80 random and 77 model circuits |
| N1 | CONFIRMED | mean infidelity REL → A1: Auckland 0.136 → 0.085, Torino 0.110 → 0.059; A1 lower than REL in 97 % / 92 % of circuits with different 2-qubit counts |
| N2 | **AMBIGUOUS** | A1 / L3 mean infidelity: Auckland 1.058 (bound 1.05), Torino 0.874 |
| N3 | CONFIRMED | within a circuit, the arm with fewer 2-qubit gates had the lower infidelity in 91.6 % (Auckland, 155 pairs) and 92.4 % (Torino, 144 pairs) |
| N4 | **AMBIGUOUS** | A1 / C2 mean infidelity: Auckland 0.966, Torino 1.017 (bound 1.00, refute above 1.02) |

## 2. Numbers (40 random circuits per device)

| Device | Mean infidelity REL / C2 / A1 / L3 | 2-qubit sums | 1-qubit sums |
|---|---|---|---|
| FakeAuckland | 0.1360 / 0.0880 / 0.0850 / 0.0803 | 932 / 530 / 486 / 505 | 1,704 / 1,522 / 1,620 / 1,430 |
| FakeTorino | 0.1097 / 0.0581 / 0.0591 / 0.0676 | 980 / 524 / 501 / 511 | 3,804 / 2,539 / 2,519 / 2,162 |

**Model-written circuits of 9/30** (77 unique, at most 8 qubits, FakeAuckland; reported without a
prediction): mean infidelity REL 0.0580, C2 0.0491, A1 0.0454, L3 0.0436.

## 3. Reading

- **The 2-qubit count is the right first target.**
  - Within a circuit, fewer 2-qubit gates meant lower infidelity in about 92 % of comparisons on both
    devices (N3).
  - The improvement from the released stack to c2 and a1 shows up as a large fidelity gain: infidelity
    falls by 38 % on Auckland and 46 % on Torino (N1).
- **It is not the only factor once the 2-qubit counts are close.**
  - **A1 against L3:** in the 26 Auckland circuits where A1 and L3 have the same 2-qubit count, A1 was
    worse in 16. Its mean infidelity there was 0.0715 against L3's 0.0675, and one case was 0.094 against
    0.051 at 9 two-qubit gates each.
  - **A1 against C2 on Torino:** A1 has fewer 2-qubit gates (501 against 524), but its mean infidelity
    is 1.7 % higher.
  - **Why, as far as the data shows:**
    - **Which physical qubits and couplers a circuit lands on.** The snapshots' error rates vary from
      qubit to qubit, and no arm was given error rates, so this is effectively luck.
    - **Single-qubit gate count.** A1 has more single-qubit gates than L3: 1,620 against 1,430 on
      Auckland, 2,519 against 2,162 on Torino.
  - The prediction-free causal split between these two factors was not measured.
- **Next lever (proposed a2).** a1 already produces several candidate circuits. It currently picks the
  one with the fewest 2-qubit gates. It could instead pick by *estimated fidelity* from the device's own
  error rates: the product of (1 - error) over the gates used, on the qubits used. It could also pass
  edge errors to the layout search, which PSF-Zero already supports (`layout_edge_errors`).
  - This needs a Target with error rates, which the fake devices have and the live devices provide.
  - It also needs a new pre-registration.

**Not established:** hardware results. The noise model has no idle noise, crosstalk or drift, and
nothing here is measured on a QPU.

## 4. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`nf_raw.json`](../../data/2026-10-01/noisy_fidelity/outputs/scored/nf_raw.json) | 81,799 | `83e475986c5205db694248e4c62ed1eb2cba340d2f2636426e4d0e70230dc8bf` |
| [`nf_run.txt`](../../data/2026-10-01/noisy_fidelity/outputs/scored/nf_run.txt) | 64,774 | `86acf2f3f3a55b134cf384c93811c4685beb0cbfe56ed63721e9be53de5765cf` |
| [`nf_score.txt`](../../data/2026-10-01/noisy_fidelity/outputs/scored/nf_score.txt) | 1,223 | `46e17691035e2c5d6c01839aee1ba061d786fadd2ac94b2409f483096fb5fe9c` |

---

<!-- ===== Addendum 282 (source: spare-qubit-cliff-addendum-282-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration of a2 (error-aware placement and selection). Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 282 -- Pre-registration: psf_ai_compile 2026-10-01.a2 (error-aware placement and selection) under noisy simulation (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Devices and noise:** fake-provider devices and their noise models. No IBM access, no hardware.

## 1. Why a2, and what development showed (not pre-registered)

**The starting point.** The noisy-fidelity test (`noisy-fidelity-results-2026-10-01.md`) showed two
things:

- fewer 2-qubit gates meant lower infidelity in about 92 % of within-circuit comparisons;
- once the 2-qubit counts were close, the physical qubits chosen decided the result. a1 was not given
  any error rates, and lost to L3 on FakeAuckland (1.06 ×).

**a2 = a1 plus the following, used only when a `target` with error rates is given:**

1. **Re-placement.** Each of the best candidates (at most 6, within 2 two-qubit gates of the fewest) is
   re-placed on the device.
   - Every coupling-preserving placement of its used qubits and couplers is scored by the target's own
     gate errors (subgraph isomorphism, at most 5,000 placements).
   - This is the idea of Qiskit's VF2PostLayout, applied to PSF-Zero's routed circuit.
2. **Selection by estimated fidelity.** The final choice minimises the sum of -log(1 - error) over the
   gates, rather than the 2-qubit count.
3. **Error-aware layout candidate.** One more candidate per starting point routes from the initial
   layout that Qiskit level 3 picks *with* the target. Only the layout is borrowed.

Without a target, a2 is identical to a1.

**Development data:** the seeds of the noisy-fidelity test (7001-7040), FakeAuckland and FakeTorino.
They are now excluded. The table gives mean infidelity.

| Device | A1 | A2 | L3 | L3T (Qiskit L3 with target) | A2 lower than L3T | A2 median time |
|---|---|---|---|---|---|---|
| FakeAuckland | 0.0850 | 0.0654 | 0.0803 | 0.0617 | 13 of 40 | 161 ms |
| FakeTorino | 0.0591 | 0.0356 | 0.0676 | 0.0364 | 24 of 40 | 383 ms |

- **2-qubit sums:** A2 487 / 501, A1 486 / 501.
- **Before item 3 was added,** A2 was 0.0654 / 0.0357; item 3 changed almost nothing.
- **A diagnostic on Auckland:** the estimated cost and the simulated infidelity correlate at 0.75 across
  compiled outputs. Some outputs with equal estimates differ by up to 2 × in simulated infidelity. The
  estimate ignores where in the circuit an error falls and which qubits are traced out.
- **Not done, on purpose:** selection by simulating the noise model itself. That would select by the
  evaluation metric, and it would not carry over to real hardware.

## 2. Design (`a2_eval.py`)

**Arms.**

| Arm | What it is |
|---|---|
| A1 | psf_ai_compile a1, no error information |
| A2 | psf_ai_compile a2 with `target=<device Target>` |
| L3 | Qiskit level 3, no error information |
| L3T | Qiskit level 3 with `target=<device Target>`, error-aware |

**Noise.** `NoiseModel.from_backend` of each fake device: per-gate depolarizing plus thermal relaxation;
no idle noise, crosstalk or readout.

**Fidelity.** <psi|rho|psi> on the final-layout qubits (density matrix).

**Devices:** FakeAuckland, FakeTorino, **FakeKingston** (Kingston's noise has not been used before).

**Inputs.** None of these were used in development.

- **Random dense circuits** from Python `random` seeds 8001-8040, 3-5 qubits, the same generator as
  before, on each device.
- **The A2-without-target check:** A2 called without a target, on the first 20 Auckland circuits,
  compared with A1.

## 3. Predictions

**C0:** the versions loaded are c2, c2, a1 and a2.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| D1 | all outputs are exact without noise | noiseless F >= 1 - 1e-9 for every arm and circuit | any below |
| D2 | error-aware a2 beats a1 under noise | every device: mean infidelity A2 <= 0.90 × A1 | any device: A2 >= A1 |
| D3 | a2 is at error-aware Qiskit L3's level | every device: A2 <= 1.10 × L3T | any device > 1.25 × |
| D4 | a2 beats Qiskit L3 without error information | every device: A2 <= 0.95 × L3 | any device: A2 > L3 |
| D5 | a2 does not pay in 2-qubit gates | every device: 2-qubit sum A2 <= 1.03 × A1 | any device > 1.10 × |
| D6 | fast enough for the vLLM loop | per-device median A2 time <= 600 ms **and** max <= 3 s | any median > 1.2 s |
| D7 | without a target, a2 is a1 | identical output digests on all 20 | any differs |

Between the bounds: ambiguous. No re-runs to improve a score.

**Expectations stated before running.**

- **D3:** development ratios were 1.06 (Auckland) and 0.98 (Torino). Auckland is the risk.
- **D2:** development ratios were 0.77 and 0.60. On a device where A1 already lands on good qubits,
  the gain could be smaller.
- **D6:** development medians were 161-383 ms. Kingston has 156 qubits, so it is expected to be
  slowest.

## 4. What this cannot establish

- **Hardware results.** The simplified noise model has no idle noise, crosstalk or drift. The error
  rates are a snapshot.
- The vLLM pass rate.
- Larger circuits.

A positive result is a reason to try a2 on hardware with the live Target's error rates. That happens at
home.

## 5. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `a2_eval.py` | 8,406 | `88c098ea963df2c704219e235bf68c983e975485f3c642f1446433eafa70722c` |
| `psf_ai_compile.py` 2026-10-01.a2 | 15,874 | `dda42b284f2f0aed430d72d6ae701af996892623f3fca53753bde20774ce19e6` |
| `psf_ai_compile.py` 2026-10-01.a1 | 9,336 | `625e569dee1e69fae0890b5f89ea8acc8b659869bcf6cbbae5594df9ecd6b281` |
| `core_fix_c2_eval.py` (helpers) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |

```
PYTHONPATH=<core> python -u a2_eval.py run --compile <c2>/psf_compile.py --layout <c2>/psf_smart_layout.py \
  --a1 psf_ai_compile_a1.py --a2 psf_ai_compile_a2.py --out a2_raw.json > a2_run.txt 2>&1
python a2_eval.py score --out a2_raw.json > a2_score.txt
```

## 6. Dry run before locking (disclosed)

**Inputs:** `--dry` uses seeds offset by 900,000, 3 per device.

**Results:**

- D1 and D4-D7 were CONFIRMED.
- D2 and D3 were AMBIGUOUS:
  - A2/A1 was 0.67, 0.83 and 0.93;
  - A2/L3T was 1.16 on Auckland and 0.97 on Torino and Kingston.

**Changes after the dry run:** none.

**No scored input was compiled before locking.**

---

<!-- ===== Addendum 283 (source: spare-qubit-cliff-addendum-283-2026-10-01.md) ===== -->

> **Note added when merging:** 7 of 7 CONFIRMED. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/ai_compile_a2/as_run/`](../../data/2026-10-01/ai_compile_a2/as_run/); the outputs are in [`data/2026-10-01/ai_compile_a2/`](../../data/2026-10-01/ai_compile_a2/).

## Addendum 283 -- Results: psf_ai_compile 2026-10-01.a2 (error-aware) under noisy simulation -- 7 of 7 CONFIRMED (2026-10-01)

**Pre-registration:** `docs/findings/ai-compile-a2-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 282 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Noise:** each fake device's own noise model.
- **Hashes:** the run's META line records the pre-registered hashes. Nothing was re-run.

## 1. Verdict

**C0:** OK. **Decision line:** ALL CONFIRMED.

| ID | Result | Numbers |
|---|---|---|
| D1 | CONFIRMED | all 480 outputs (4 arms × 120 circuits) exact without noise |
| D2 | CONFIRMED | A2/A1 mean infidelity: 0.772 (Auckland), 0.559 (Torino), 0.842 (Kingston) |
| D3 | CONFIRMED | A2/L3T: 1.000, 0.972, 0.977 |
| D4 | CONFIRMED | A2/L3: 0.769, 0.498, 0.812 |
| D5 | CONFIRMED | 2-qubit sums A2/A1: 554/554, 552/551, 551/551 |
| D6 | CONFIRMED | median 167 / 359 / 417 ms, max 1.13 s |
| D7 | CONFIRMED | without a target, A2's output equals A1's in 20 of 20 |

## 2. Numbers (40 new random circuits per device)

| Device | Mean infidelity A1 / A2 / L3 / L3T | A2 lower than L3T | 2-qubit sums A1 / A2 / L3 / L3T |
|---|---|---|---|
| FakeAuckland | 0.0870 / 0.0672 / 0.0874 / 0.0672 | 20 of 40 | 554 / 554 / 564 / 564 |
| FakeTorino | 0.0714 / 0.0399 / 0.0801 / 0.0410 | 29 of 40 | 551 / 552 / 564 / 564 |
| FakeKingston | 0.0218 / 0.0183 / 0.0226 / 0.0188 | 28 of 40 | 551 / 551 / 558 / 558 |

## 3. Reading

- **Giving the AI front end the device's error rates (a2) cuts the simulated error a lot:**
  - 23 % on Auckland, 44 % on Torino and 16 % on Kingston, compared with a1;
  - it costs essentially no 2-qubit gates (one extra gate on Torino, in total).
- **Against error-aware Qiskit level 3 (L3T), a2 is level or slightly better:**
  - equal on Auckland, 2-3 % lower on Torino and Kingston;
  - it has the lower error in 77 of 120 circuits.
- **Against Qiskit level 3 without error rates,** a2 has 19-50 % lower error.
- **Without a target,** a2 is exactly a1, so it is safe to use where no error rates are available.
- **Nothing in PSF-Zero's compiler or core was changed.**
  - From the error rates, a2 picks the qubits (re-placement) and the candidate (estimated fidelity).
  - The level-3 layout it borrows is only one more starting layout; synthesis and routing remain
    PSF-Zero's.

**Not established:**

- **Hardware results.** The noise model has no idle noise, crosstalk or drift, and the error rates are a
  fixed snapshot.
- The vLLM pass rate.
- Circuits above 8 qubits, which fall through to c2.

**Next step (at home, on the owner's signal):** run a2 with the live device's Target against L3T on a
small set of circuits on hardware.

## 4. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`a2_raw.json`](../../data/2026-10-01/ai_compile_a2/outputs/scored/a2_raw.json) | 63,742 | `9c20727e3034484bccadae25ea6e91d786a73de6456684785186c605de29aaf7` |
| [`a2_run.txt`](../../data/2026-10-01/ai_compile_a2/outputs/scored/a2_run.txt) | 48,627 | `8db0ad5b014ea9039b8f43fbee3cfe7b99864842e334cbb36d776b58dbed1420` |
| [`a2_score.txt`](../../data/2026-10-01/ai_compile_a2/outputs/scored/a2_score.txt) | 1,396 | `19e41944c54eeec00528b4530f9179f83c00d9cc387681c2fe6c6e5bc0cb894a` |

---

<!-- ===== Addendum 284 (source: spare-qubit-cliff-addendum-284-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration: a2 on 153 model-written circuits of 2026-09-30 that no earlier test used, and the harness v11 ([`benchmarks/e2e_vllm_psf_v11.py`](../../benchmarks/e2e_vllm_psf_v11.py)). Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 284 -- Pre-registration: the AI front end a2 in the vLLM setting -- replay of 153 unused model-written circuits under noisy simulation, and the e2e harness v11 (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **No GPU, no pod, no IBM access.** The language model is not run again. Its circuits from 9/30 are
  replayed.

## 1. What is tested, and why this way

**The question.** What does the AI front end a2 change for the circuits a language model actually
writes in the vLLM loop?

**Why not on the pod.** Running the model again would mostly re-measure the model; the pass rate does
not depend on the compiler. This test replays the circuits the models wrote on 9/30 instead.

**Inputs.**

- Every circuit in the pod runs' `rounds.jsonl` (field `spec`) that is **not** one of the
  `best_circuit.json` files, unique per task, with at most 8 qubits. The best circuits were used in
  developing c2 and a0; these were never compiled in any test.
- **153 circuits:** w3 83, qft3 28, dicke42 16, w4 12, ghz5 7, ghz3i 4, singlet3 2, bell3 1.
- They come from gpt-oss-120b, Qwen2.5-7B and Qwen2.5-72B, in the go/no-go, pilot and v10 runs.
- They include wrong answers. That does not matter here: the comparison is between compilers of the
  same logical circuit.
- They are converted exactly as the harness does (`to_tape` → `tape_to_qiskit`), so 2-qubit gates
  arrive as `unitary`.

**Arms.**

| Arm | What it is |
|---|---|
| C2 | `compile_for_hardware(cx, layout_search=True, seed 0)`, the compile step of v10, adopted version |
| A2 | psf_ai_compile 2026-10-01.a2 with the device Target, the compile step of v11 `--compiler ai` |
| L3 | Qiskit level 3 without error rates |
| L3T | Qiskit level 3 with the Target, the harness's own "Q3" comparison |

**Devices:** FakeAuckland (the e2e device) and FakeTorino.

**Noise and fidelity:** `NoiseModel.from_backend`; fidelity on the final-layout qubits (density matrix).

## 2. Predictions

**C0:** the versions loaded are c2, c2 and a2. Circuits the harness itself would reject are skipped and
counted.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| E1 | all outputs are exact without noise | every arm, every circuit: F >= 1 - 1e-9 | any below |
| E2 | a2 beats the compile step v10 used | every device: mean infidelity A2 <= 0.90 × C2 | any device: A2 >= C2 |
| E3 | a2 is at the harness's Qiskit comparison | every device: A2 <= 1.05 × L3T | any device > 1.20 × |
| E4 | a2 does not use more 2-qubit gates than L3T | every device: A2 sum <= L3T sum | any device > 1.05 × |
| E5 | fast enough per round | per-device median A2 time <= 0.6 s, **and** max <= 3 s | any median > 1.2 s |

Between the bounds: ambiguous. No re-runs to improve a score.

**Reported without prediction:** per-task means, and better/worse counts against L3T.

**Expectations.** These are similar to the a2 test on random circuits. Model-written circuits are smaller
and simpler, so the differences between arms may be smaller. On FakeAuckland (27 qubits) every arm has
few qubits to choose from.

## 3. Harness v11 (exploratory, done before locking, disclosed)

**What v11 is:** `e2e_vllm_psf_v11.py` = v10 plus `--compiler {psf,ai}` and `--ai-module`. With `ai` the
compile step calls `compile_for_model_circuit(qc, coupling_map, basis_gates, target=target)`. Everything
else is unchanged: the fidelity checks, the Qiskit comparison, the feedback text.

**Mock-LLM run** (`--mock-llm`, 3 rounds, all 10 tasks, FakeAuckland, against a repository tree with the
c2 patch applied):

- Both settings completed without error.
- With `ai`, the compiled fidelity equalled the logical fidelity in every round, so the final-layout
  bookkeeping works through the harness.
- qft3's correct mock circuit compiled to 7 two-qubit gates, against 9 with `psf` and 9 for the
  harness's L3.
- The StatePreparation baselines dropped:
  - ghz5 41 → 34;
  - qft3 7 → 3.
- **Compile time per round with `ai`:** 40-200 ms for 3-5 qubits, about 0.5 s for the 6-qubit tasks.
  The 27-qubit tasks fall through to c2 and stay at about 12 ms.

These runs are not scored.

## 4. What this cannot establish

- The model's behaviour with a2 in the loop. The feedback contains gate counts, so a model *could* react
  to lower counts. That needs a pod run.
- Hardware results.
- The 27-qubit tasks, which fall through to c2.

## 5. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `rp_eval.py` | 9,832 | `c5ded2a900078b909491d2883750c67e718f2edbb61b39779520b59e6f2e595b` |
| `e2e_vllm_psf_v11.py` | 58,424 | `96474c6f8722c08fb7d069124a8ad6bd444b007a39e92e5e56509582984940c5` |
| `psf_ai_compile.py` 2026-10-01.a2 | 15,874 | `dda42b284f2f0aed430d72d6ae701af996892623f3fca53753bde20774ce19e6` |
| `core_fix_c2_eval.py` (helpers) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |

```
PYTHONPATH=<core>:<repo>/benchmarks python -u rp_eval.py run --compile <c2>/psf_compile.py \
  --layout <c2>/psf_smart_layout.py --a2 psf_ai_compile.py --data <repo>/data/2026-09-30 \
  --v10-dir <dir of e2e_vllm_psf_v10.py> --out rp_raw.json > rp_run.txt 2>&1
python rp_eval.py score --out rp_raw.json > rp_score.txt
```

## 6. Dry run before locking (disclosed)

**Inputs:** `--dry` uses circuits from the `sandbox_dryrun` folders. These are mock and fake-model
circuits, never the scored pod circuits; 3 were usable.

**Results:** all 5 items CONFIRMED.

**Changes after the dry run:** none to the predictions. Before the dry run, its input was changed from a
slice of the scored set to the `sandbox_dryrun` folders, so that no scored circuit is compiled before
locking.

**No scored circuit was compiled before locking.**

---

<!-- ===== Addendum 285 (source: spare-qubit-cliff-addendum-285-2026-10-01.md) ===== -->

> **Note added when merging:** E1, E3, E4, E5 CONFIRMED; E2 (FakeAuckland) AMBIGUOUS. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/vllm_a2_replay/as_run/`](../../data/2026-10-01/vllm_a2_replay/as_run/); the outputs are in [`data/2026-10-01/vllm_a2_replay/`](../../data/2026-10-01/vllm_a2_replay/).

## Addendum 285 -- Results: a2 on 153 unused model-written circuits (noisy simulation) -- E1, E3, E4, E5 CONFIRMED; E2 AMBIGUOUS (FakeAuckland) (2026-10-01)

**Pre-registration:** `docs/findings/vllm-a2-replay-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 284 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Run:** nothing was re-run. 153 circuits, 0 skipped.

## 1. Verdict

**C0:** OK.

| ID | Result | Numbers |
|---|---|---|
| E1 | CONFIRMED | 1,224 outputs (4 arms × 153 × 2 devices), all exact without noise |
| E2 | **AMBIGUOUS** | A2/C2 mean infidelity: 0.978 (Auckland; bound 0.90), 0.511 (Torino) |
| E3 | CONFIRMED | A2/L3T: 0.955 (Auckland), 0.946 (Torino) |
| E4 | CONFIRMED | 2-qubit sums A2/L3T: 885/963 (Auckland), 894/961 (Torino) |
| E5 | CONFIRMED | median 119 / 243 ms, max 0.96 s |

## 2. Numbers

| Device | Mean infidelity C2 / A2 / L3 / L3T | 2-qubit sums C2 / A2 / L3 / L3T | A2 better / worse than L3T |
|---|---|---|---|
| FakeAuckland | 0.0554 / 0.0542 / 0.0519 / 0.0568 | 963 / 885 / 965 / 963 | 85 / 67 |
| FakeTorino | 0.0386 / 0.0197 / 0.0366 / 0.0208 | 965 / 894 / 959 / 961 | 79 / 63 |

**Per task (mean infidelity C2 → A2).**

| Task | FakeAuckland | FakeTorino |
|---|---|---|
| dicke42 | 0.139 → 0.112 | 0.126 → 0.045 |
| qft3 | 0.035 → 0.037 | 0.024 → 0.013 |
| w3 | 0.045 → 0.049 | 0.029 → 0.017 |

**2-qubit counts:** A2 has 8 % fewer 2-qubit gates than C2 on Auckland (885 against 963) and 7 % fewer
on Torino (894 against 965). For qft3 the figure is 113 against 137, mostly SWAP elision.

## 3. Reading

**On FakeTorino** a2 does what the earlier tests predicted for the model's own circuits:

- it halves the simulated error of the compile step v10 used;
- it beats Qiskit level 3 with the target, with lower mean infidelity and fewer 2-qubit gates.

**On FakeAuckland** the picture is different, and it was not anticipated.

- **Both error-aware compilers are behind the error-blind Qiskit L3.** A2 is at 0.0542 and L3T at
  0.0568, while L3 reaches 0.0519.
- **A2 against C2:** A2 is only 2 % better than C2 overall, and slightly worse on w3 and qft3, although
  it uses fewer 2-qubit gates.
- **A2 against L3T:** A2 still beats Qiskit's own error-aware level 3 (E3).
- **One hypothesis, not tested here:** on this device, choosing qubits by the reported gate error rates
  does not track the simulated noise well. The noise model adds thermal relaxation from each qubit's
  T1/T2 and the gate durations, and the reported error numbers may not reflect that.
  - This contrasts with the a2 test on random circuits, where on Auckland A2 had 23 % lower infidelity
    than L3 and equalled L3T. The model's circuits are smaller and simpler, and here the choice of
    qubits seems to matter differently.
  - A cost that also uses T1/T2 and gate durations would be the next thing to try. It would need a new
    pre-registration.
- **Not established:**
  - hardware results;
  - the model's behaviour with a2 in the loop (a pod run);
  - the 27-qubit tasks, which fall through to c2.

## 4. Harness v11

`e2e_vllm_psf_v11.py` (`--compiler ai --ai-module psf_ai_compile.py`) passed its mock-LLM check before
locking. Section 3 of the pre-registration has the details. It is ready for a pod run if the owner wants
one.

## 5. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`rp_raw.json`](../../data/2026-10-01/vllm_a2_replay/outputs/scored/rp_raw.json) | 183,771 | `f8a590b9f2ccb8c99746b10a43de039784f96fb9e6d516cf1e22968eddaedc77` |
| [`rp_run.txt`](../../data/2026-10-01/vllm_a2_replay/outputs/scored/rp_run.txt) | 148,828 | `3a15538393cb49c16fa1c3496c9cad5fe93a00d61ee7e3eb3c2640badc457b5a` |
| [`rp_score.txt`](../../data/2026-10-01/vllm_a2_replay/outputs/scored/rp_score.txt) | 2,760 | `e7bce3de31d97d31c15eccd9834808fc6513473e9a9edf86ede9f4b1d85c6cb7` |

---

<!-- ===== Addendum 286 (source: spare-qubit-cliff-addendum-286-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration: the FakeAuckland anomaly and a4 (state-aware error estimate). Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 286 -- Pre-registration: why error-aware placement failed on FakeAuckland, and psf_ai_compile 2026-10-01.a4 (state-aware error estimate) (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Devices and noise:** fake-provider devices and their noise models. No IBM access, no hardware.

## 1. The FakeAuckland puzzle, and what development found (not pre-registered)

**The puzzle.** In the replay of model-written circuits (`vllm-a2-replay-results-2026-10-01.md`), both
error-aware compilers lost to error-blind Qiskit level 3 on FakeAuckland:

| Compiler | Mean infidelity |
|---|---|
| A2 | 0.0542 |
| Qiskit L3 with target (L3T) | 0.0568 |
| Qiskit L3 without target (L3) | 0.0519 |

**Finding 1: reported errors understate decoherence on some qubits.**

- On FakeAuckland, the reported gate error is below what the same snapshot's T1/T2 and gate duration
  allow on 16 of 56 cx gates and on some single-qubit gates.
  - Qubit 24 has T2 = 26 µs.
  - cx(24,25) is reported at 0.0055, but its decoherence limit is 0.0089.
- Aer's device noise model applies the decoherence limit. The channels it builds have average
  infidelity up to 1.6× (cx) and 2.1× (sx) the reported value.
- So placing by reported error lures circuits onto qubits 22-26.
- On FakeTorino no cz gate is affected.

**Candidate a3:** use max(reported, decoherence limit) per gate. On the replay set (FakeAuckland) it gave
0.0432 against A2's 0.0542. On the a2 random seeds 8001-8040 it gave 0.0701 against 0.0672, worse.

**Finding 2: the score itself was the problem.**

- On 20 random circuits × 4 compilers (FakeAuckland), the sum of average gate infidelities ranked the
  compiled versions of the same circuit correctly in only 63 of 117 pairs, about chance.
- **The cause:** relaxation and dephasing barely affect a qubit sitting in a basis state, while
  depolarizing noise affects every state. In one example a3 moved a circuit from the short-T2 qubits
  22-26 to long-T2 qubits 3-11. Its average-infidelity score fell from 0.149 to 0.132, but the simulated
  infidelity rose from 0.057 to 0.091.

**Candidate a4: a state-aware first-order estimate.**

- Every gate's error is split into Pauli components:
  - Pauli-twirled thermal relaxation on each qubit, from T1, T2 and the gate duration;
  - plus a depolarizing remainder, up to the reported error.
- Each component P with probability p costs p × (1 - <P>²), with <P> taken on the ideal state of the
  routed circuit right after the gate (statevector of the used qubits).
- The expectation values do not depend on where the circuit is placed, so every placement is scored from
  precomputed sums.
- On the same 117 pairs it ranked 109 correctly. Its correlation with the simulated infidelity was 0.96,
  against 0.85 for the average-infidelity score.

**Caution.** The estimate makes the same physical assumptions as Aer's device noise model:
depolarizing plus thermal relaxation per gate. A noisy-simulation test is therefore favourable to it by
construction. Its value on hardware is untested.

**Development results of a4.** These are on development inputs: the a2 seeds 8001-8040 and the replay
set. The table gives mean infidelity.

| Set | A2 | A4 | L3 | L3T | A4 median time |
|---|---|---|---|---|---|
| Auckland random (8001-8040) | 0.0672 | 0.0540 | 0.0874 | 0.0672 | 278 ms |
| Auckland replay (153 model circuits) | 0.0542 | 0.0422 | 0.0519 | 0.0568 | 149 ms |
| Torino random | 0.0399 | 0.0396 | 0.0801 | 0.0410 | 373 ms |
| Kingston random | 0.0183 | 0.0180 | 0.0226 | 0.0188 | 368 ms |

- **2-qubit sums:** essentially unchanged (Auckland random: A4 562 against A2 554).
- **Against L3:** A4 was lower in 40 of 40 random circuits on each device, and in 133 of 153 replay
  circuits.

## 2. Design (`a4_eval.py`)

**Arms.**

| Arm | What it is |
|---|---|
| A2 | psf_ai_compile a2 with the target |
| A4 | psf_ai_compile a4 with the target |
| L3 | Qiskit level 3, no target |
| L3T | Qiskit level 3 with the target |

A1 is used only for the no-target identity check.

**Noise and fidelity:** `NoiseModel.from_backend`; fidelity on the final-layout qubits (density matrix).

**Devices:** FakeAuckland, FakeTorino, FakeKingston.

**Inputs (new):** random dense circuits from Python `random` seeds 9001-9040, 3-5 qubits, the same
generator as before. Also, A4 without a target against A1 on the first 20 Auckland circuits.

## 3. Predictions

**C0:** the versions loaded are c2, c2, a1, a2 and a4. F2 and F3 are split by device, as development
suggests.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| F1 | all outputs are exact without noise | all arms, all circuits | any below 1 - 1e-9 |
| F2a | a4 fixes the Auckland problem | FakeAuckland: mean infidelity A4 <= 0.90 × A2 | > 1.00 × |
| F2b | a4 is not worse elsewhere | FakeTorino and FakeKingston: A4 <= 1.00 × A2 | either > 1.05 × |
| F3a | a4 beats error-aware Qiskit on Auckland | FakeAuckland: A4 <= 0.90 × L3T | > 1.00 × |
| F3b | a4 is at least level with error-aware Qiskit elsewhere | FakeTorino and FakeKingston: A4 <= 1.00 × L3T | either > 1.05 × |
| F4 | a4 beats Qiskit L3 without error information | every device: A4 <= 0.90 × L3 | any > 1.00 × |
| F5 | a4 keeps the 2-qubit count | every device: sum A4 <= 1.03 × A2 | any > 1.10 × |
| F6 | fast enough for the vLLM loop | every median <= 0.8 s, max <= 4 s | any median > 1.6 s |
| F7 | without a target, a4 is a1 | 20 of 20 identical | any differs |

Between the bounds: ambiguous. No re-runs to improve a score.

## 4. What this cannot establish

- Hardware. See the caution in section 1: the estimate and the simulator share their physical model.
- The vLLM pass rate.
- Larger circuits.

## 5. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `a4_eval.py` | 9,810 | `7658769db090682820e2b4954a5af526444935af6cad5436cc094c870e02ec61` |
| `psf_ai_compile.py` 2026-10-01.a4 | 25,676 | `06ac6750705ce5159753d283d043ac4fff936f4c3023bf7c0199143b38fa1a13` |
| `psf_ai_compile.py` 2026-10-01.a2 / a1 | 15,874 / 9,336 | `dda42b28...ce19e6` / `625e569d...b281` |
| `core_fix_c2_eval.py` (helpers) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |

```
PYTHONPATH=<core> python -u a4_eval.py run --compile <c2>/psf_compile.py --layout <c2>/psf_smart_layout.py \
  --a1 psf_ai_compile_a1.py --a2 psf_ai_compile_a2.py --a4 psf_ai_compile_a4.py --out a4_raw.json > a4_run.txt 2>&1
python a4_eval.py score --out a4_raw.json > a4_score.txt
```

## 6. Dry run before locking (disclosed)

**Inputs:** `--dry` uses seeds offset by 900,000, 3 per device.

**Results:** all 9 items CONFIRMED. A4/A2 was 0.88, 1.00 and 0.98; A4/L3T was 0.80, 0.91 and 0.94.

**Changes after the dry run:** none.

**No scored input was compiled before locking.**

---

<!-- ===== Addendum 287 (source: spare-qubit-cliff-addendum-287-2026-10-01.md) ===== -->

> **Note added when merging:** 9 of 9 CONFIRMED. The estimate shares its physics with the simulator that scores it, so these results favour it by construction; a real-device check is needed. Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/ai_compile_a4/as_run/`](../../data/2026-10-01/ai_compile_a4/as_run/); the outputs are in [`data/2026-10-01/ai_compile_a4/`](../../data/2026-10-01/ai_compile_a4/).

## Addendum 287 -- Results: psf_ai_compile 2026-10-01.a4 (state-aware error estimate) under noisy simulation -- 9 of 9 CONFIRMED (2026-10-01)

**Pre-registration:** `docs/findings/ai-compile-a4-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 286 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, Qiskit 2.5.2, qiskit-aer 0.17.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Run:** new seeds 9001-9040. Nothing was re-run.

## 1. Verdict

**C0:** OK. **Decision line:** ALL CONFIRMED.

| ID | Result | Numbers |
|---|---|---|
| F1 | CONFIRMED | all 480 outputs exact without noise |
| F2a | CONFIRMED | Auckland A4/A2 = 0.867 |
| F2b | CONFIRMED | Torino 0.994, Kingston 0.983 |
| F3a | CONFIRMED | Auckland A4/L3T = 0.851 |
| F3b | CONFIRMED | Torino 0.937, Kingston 0.936 |
| F4 | CONFIRMED | A4/L3 = 0.615, 0.353, 0.801 |
| F5 | CONFIRMED | 2-qubit sums A4/A2 = 582/574, 584/583, 583/583 |
| F6 | CONFIRMED | median 221 / 407 / 426 ms, max 0.66 s |
| F7 | CONFIRMED | without a target, A4's output equals A1's in 20 of 20 |

## 2. Numbers (40 new random circuits per device)

| Device | Mean infidelity A2 / A4 / L3 / L3T | A4 lower than L3T |
|---|---|---|
| FakeAuckland | 0.0694 / 0.0601 / 0.0978 / 0.0707 | 29 of 40 |
| FakeTorino | 0.0425 / 0.0422 / 0.1196 / 0.0451 | 33 of 40 |
| FakeKingston | 0.0195 / 0.0192 / 0.0239 / 0.0205 | 37 of 40 |

## 3. Reading

- **The FakeAuckland puzzle is explained, and the fix holds on new circuits.** It had two layers:
  - **Reported errors understate decoherence.** On FakeAuckland the reported gate errors are below the
    decoherence limit implied by the same snapshot's T1/T2 on several qubits, so error-aware placement
    was lured onto them.
  - **The score could not rank candidates.** The sum of average gate infidelities ranked a circuit's
    compiled versions about as well as chance, because relaxation and dephasing barely affect qubits in
    basis states.
- **What a4 does about it.** a4 scores with a state-aware first-order estimate, built from T1, T2, gate
  durations and reported errors and evaluated on the circuit's own ideal state. On FakeAuckland it lowers
  the simulated error by 13 % against a2 and by 15 % against Qiskit level 3 with the target.
- **Elsewhere,** a4 is level with a2 (within 2 %) and 6 % better than Qiskit L3T. It is lower than L3T in
  99 of 120 circuits.
- **Unchanged:** the 2-qubit counts, exactness, and the no-target behaviour (identical to a1).
- **Caution, repeated from the pre-registration.** The estimate makes the same physical assumptions as
  Aer's device noise model, so these simulations favour it by construction. The decoherence finding, that
  reported error is below the T1/T2 limit, is a property of the snapshot data. On a live device it can
  be checked directly from the Target.
- **Not established:** hardware results; the vLLM pass rate; larger circuits.

## 4. How to use

- **In the e2e harness v11:** `--compiler ai --ai-module psf_ai_compile_a4.py`. The harness passes the
  device Target.
- **Directly:** `compile_for_model_circuit(qc, coupling_map, basis_gates, target=target)`.

## 5. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`a4_raw.json`](../../data/2026-10-01/ai_compile_a4/outputs/scored/a4_raw.json) | 64,369 | `4f20cd6f93169cc44f10cf454e64f134ae26389680e31174928319c94c3469bb` |
| [`a4_run.txt`](../../data/2026-10-01/ai_compile_a4/outputs/scored/a4_run.txt) | 49,249 | `dfb74ac4abe5f9e7052fa56511de4f8ced70e11d0c0e5654cea797c3b27fe9bb` |
| [`a4_score.txt`](../../data/2026-10-01/ai_compile_a4/outputs/scored/a4_score.txt) | 1,559 | `da00e00bb60bf2ee75270e014190b3d40a24344eaf386aa0bfd8ea675cf6eda1` |

---

<!-- ===== Addendum 288 (source: spare-qubit-cliff-addendum-288-2026-10-01.md) ===== -->

> **Note added when merging:** Workplace pre-registration of a5 (cache keyed by value) and a 30,000-lap test. Locked at the workplace Project's save time, not by a git commit; it entered this repository after the run.

## Addendum 288 -- Pre-registration: 30,000-lap test of psf_ai_compile 2026-10-01.a5 (cache keyed by value, not identity) (2026-10-01)

**Status: pre-registration, locked at the Project save time of this document**, before the scored run.

- **Environment:** workplace sandbox, Linux, 2 CPUs, Qiskit 2.5.2.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Device:** FakeAuckland only; no IBM access.
- **Timings:** sandbox only, compared within this run.

## 1. Why

**The owner's question:** does repeating the AI front end tens of thousands of times make a drift appear?

**What can and cannot drift.**

- The arithmetic cannot accumulate: every compile starts from scratch.
- State kept between calls can go stale.

**The defect found before this test.** a4 (and a3) cache per-gate error parameters under the key
`id(target)`. Python reuses the id of a freed object, so after a Target was replaced, for example by a
calibration update, a4 kept using the old numbers silently. In a check that replaced the Target every lap
with alternating calibrations, 199 of 200 laps were served stale values.

**The fix, a5.** Every cache entry is keyed by the values it was computed from: the gate error, the
duration, and T1/T2 of the gate's qubits. Object identity is never used, and the caches' size is bounded.

**Checks after the fix:** 0 stale laps of 200; identical outputs to a4 for a fixed Target on 10 circuits
(seeds 10001-10010); median time 0.18 s against a4's 0.17 s.

## 2. Design (`lt_eval.py`)

**Pool:** 50 new random dense circuits, Python `random` seeds 10101-10150, 3-5 qubits, the same
generator as before. Lap *i* compiles circuit *i* mod 50.

**Calibrations.** The Target object is replaced every 1,000 laps, after the old object is freed. The two
calibrations alternate:

- **A:** FakeAuckland as shipped.
- **B:** cx errors × 1.6 on every cx touching qubits 0-13, and T2 × 0.5 on qubits 14-26.

**Recorded per lap:**

- the output digest;
- the wall time;
- the cost a5 reports for its choice;
- an independent reference cost of the returned circuit, from `stateaware.py`: the same estimate written
  separately, without any cache. It matched a4's estimator to 6 decimals in development.

**Also recorded:** RSS every 250 laps, and exactness (noiseless, component-wise) every 500 laps.

**Runs.**

- **a5:** two worker processes in parallel, laps 0-14,999 and 15,000-29,999. The calibration follows the
  global lap index.
- **Control:** a4, laps 0-2,999, run after the workers.

## 3. Predictions

**G0:** the versions loaded are a5 in the workers and a4 in the control.

| ID | Prediction | Confirmed if | Refuted if |
|---|---|---|---|
| G1 | no drift in results | 0 digest mismatches for the same (calibration, circuit), within each worker and between the two workers | any |
| G2 | calibration changes are followed immediately | the reported cost equals the independent reference (within 2e-6) on all 30,000 laps | any lap off |
| G3 | the test can see the defect | the a4 control has >= 1 lap off | 0 laps off |
| G4 | no slowdown | per worker: median of the last 1,000 laps / median of laps 101-1,100 <= 1.20 | > 1.50 |
| G5 | no memory growth | per worker: RSS at the end - RSS at lap 2,000 <= 50 MB | > 200 MB |
| G6 | outputs stay exact | every sampled output has F >= 1 - 1e-9 | any below |
| G7 | the fix changes nothing else | for calibration A, a5's digests equal a4's on all 50 circuits (first time each is seen) | any differs |

Between the bounds: ambiguous. No re-runs to improve a score.

## 4. Files and run commands

| File | Bytes | Normalized SHA-256 |
|---|---|---|
| `lt_eval.py` | 9,111 | `222fed923c5998eec3f38794c3c7eba461a4d58b8eedc8f04a5732d42a76ba44` |
| `psf_ai_compile.py` 2026-10-01.a5 | 27,043 | `91fb1ea49bb99135a8bd6f37cff091e5abd31fd097aabf4744ac5d3e58e8c172` |
| `stateaware.py` (reference) | 4,227 | `47d852b8f0bd45dbda360c0996fad907d2cbf6ea401b31aa7a0be1cc94d2caf7` |
| `psf_ai_compile.py` 2026-10-01.a4 (control) | 25,676 | `06ac6750705ce5159753d283d043ac4fff936f4c3023bf7c0199143b38fa1a13` |
| `core_fix_c2_eval.py` (helpers) | 27,244 | `9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac` |

```
PYTHONPATH=<core> python -u lt_eval.py worker --wid 0 --start 0     --laps 15000 --ai psf_ai_compile.py --compile <c2> --layout <c2> --out w0.json &
PYTHONPATH=<core> python -u lt_eval.py worker --wid 1 --start 15000 --laps 15000 --ai psf_ai_compile.py --compile <c2> --layout <c2> --out w1.json &
# after both finish:
PYTHONPATH=<core> python -u lt_eval.py worker --wid 9 --start 0 --laps 3000 --ai psf_ai_compile_a4.py --compile <c2> --layout <c2> --out control.json
python lt_eval.py score --files w0.json w1.json --control control.json
```

## 5. Dry run before locking (disclosed)

**Inputs:** a separate pool (seeds offset by 900,000) and 1 epoch per 20 laps. There were 60 laps each
for two a5 workers and the a4 control.

**Results:**

| Run | Reference mismatches | Digest mismatches | Median time |
|---|---|---|---|
| a5 workers | 0 | 0 | 175-190 ms |
| a4 control | 12 of 60 laps | 0 | 180 ms |

The a4 control showed the defect, so the harness can see it.

The scorer needs at least 2,000 laps per worker, so these short runs were inspected directly rather than
scored. Nothing was changed after the dry run.

**No scored lap was run before locking.**

---

<!-- ===== Addendum 289 (source: spare-qubit-cliff-addendum-289-2026-10-01.md) ===== -->

> **Note added when merging:** 7 of 7 CONFIRMED; a5 is the latest AI front end ([`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py)). Workplace sandbox (2 CPUs), fake-provider devices, no IBM access; times are sandbox times. The files exactly as run are in [`data/2026-10-01/long_loop_a5/as_run/`](../../data/2026-10-01/long_loop_a5/as_run/); the outputs are in [`data/2026-10-01/long_loop_a5/`](../../data/2026-10-01/long_loop_a5/).

## Addendum 289 -- Results: 30,000-lap test of psf_ai_compile 2026-10-01.a5 -- 7 of 7 CONFIRMED; the a4 control shows the defect; the scored a4 results are unaffected (2026-10-01)

**Pre-registration:** `docs/findings/long-loop-a5-preregistration-2026-10-01.md` (a workplace document, not in the repository; its text is Addendum 288 of this Part). It was locked at its
Project save time before this run.

- **Environment:** workplace sandbox, 2 CPUs, two worker processes in parallel, then the control.
- **Compiler and core:** psf_compile and psf_smart_layout 2026-10-01.c2; core 2026-09-29.1.
- **Device:** FakeAuckland.
- **Run:** nothing was re-run.

## 1. Verdict

**G0:** OK. **Decision line:** ALL CONFIRMED.

| ID | Result | Numbers |
|---|---|---|
| G1 | CONFIRMED | 0 digest mismatches within the workers, and 0 of 100 (calibration, circuit) keys differing between the two workers |
| G2 | CONFIRMED | reported cost = independent reference on 30,000 of 30,000 laps |
| G3 | CONFIRMED | the a4 control was off on 420 of 3,000 laps |
| G4 | CONFIRMED | median lap time 201.7 → 208.2 ms (ratio 1.032) and 201.4 → 208.4 ms (1.035); max lap 0.58 s / 0.99 s |
| G5 | CONFIRMED | RSS growth after lap 2,000: +1 MB and +0 MB (about 204-205 MB throughout) |
| G6 | CONFIRMED | 60 of 60 sampled outputs exact |
| G7 | CONFIRMED | a5 = a4 on calibration A for 50 of 50 circuits |

## 2. The control, in detail

- **When it went wrong.** In the a4 control, all 420 bad laps fall in laps 1,000-1,999. That is the first
  epoch after the Target was replaced: calibration B, built after the calibration-A object had been freed.
- **How badly.** The reported cost differed from the independent reference by up to 48 %. Not every lap
  was affected: gates first seen after the switch were computed fresh.
- **Why it stayed hidden.** The control's digests were self-consistent (0 mismatches). A determinism
  check alone would not have caught it; the independent reference did.

## 3. Were earlier results affected? (check added after the run, not pre-registered)

- **The concern.** The scored a4 run (`ai-compile-a4-results-2026-10-01.md`) built the FakeAuckland,
  FakeTorino and FakeKingston Targets one after another in one process, so an id could have been reused
  across devices.
- **The check.** The A4 arm was recomputed with a5 in a fresh process per device, and the noisy infidelity
  compared circuit by circuit with the recorded A4 values.
- **Result:** 40 of 40 identical on each device (120 of 120, |ΔF| = 0, the same 2-qubit counts). The
  scored a4 results stand.
- **Other runs:** a2 and earlier versions had no such cache. a3 was never scored.

## 4. Reading

- **On the owner's question: no drift appeared in 30,000 laps.** Results were identical every time for
  the same input and calibration, in both processes. There was no slowdown (+3 %, within noise) and no
  memory growth.
- **Calibration changes are now followed on the next call.**
- **The one drift mechanism that existed was found and fixed before this run:** a cache keyed by object
  identity. This run confirms both the fix (a5) and that the test can see the defect (a4 control).
- **Use a5 rather than a4 wherever Targets are replaced during a run.** Typical cases are long harness
  runs and live devices with daily calibrations. For one fixed Target, a5's output is identical to a4's.

## 5. Files

| File | Bytes | SHA-256 (raw) |
|---|---|---|
| [`w0.json`](../../data/2026-10-01/long_loop_a5/outputs/scored/w0.json) | 315,340 | `115e464a5ec7292e0d138caa01a92e9909b73d822cd5e96afa592ed9d22ca705` |
| [`w1.json`](../../data/2026-10-01/long_loop_a5/outputs/scored/w1.json) | 315,419 | `45f2bc420b4feda128c2067df6a18706070f39ca48387e18690313d3034234b6` |
| [`control.json`](../../data/2026-10-01/long_loop_a5/outputs/scored/control.json) | 105,108 | `9df010249c753af5f300e2badf31ddcef056784c19036cd84ed3bf3eb94640f3` |
| [`score.txt`](../../data/2026-10-01/long_loop_a5/outputs/scored/score.txt) | 1,087 | `d1afd69aae1c628265afe72e0d9627a4b2ad16daec7032f5feb2f04c0e8f3e77` |
| `a4_validity_check.py` | (section 3) | |


---

<!-- ===== Addendum 290 (source: spare-qubit-cliff-addendum-290-2026-10-01.md) ===== -->

> **Note added when merging:** Home pre-registration, independent of the workplace QML test of the same day. Locked by the git commit that adds this Addendum and its two scripts, pushed before the scored run.

## Addendum 290 -- Pre-registration: does a small quantum classifier keep its accuracy, and can it learn, when every circuit goes through a given compiler onto a noisy fake device? Previous release, release 2026-10-01.1, AI front end a5 and error-aware Qiskit L3 (2026-10-01)

**Status: pre-registration, written at home before any scored run.** It is locked by a git commit that
contains this document and the two scripts, pushed before the scored run starts. The workplace ran a quantum
machine-learning test on the same day. Its design and results were not seen here; this is an independent test
of the same question, not a reproduction.

No IBM account, no network access to IBM, no QPU. The devices are fake-provider snapshots and the noise is
Qiskit Aer's noise model built from them.

## 1. Question

The owner asked whether "an AI gets smarter using these circuits". Here that means a variational quantum
classifier, and two questions:

- **Q1, keeping:** a classifier trained without noise is run on a noisy device, with every circuit compiled by a
  given compiler. How much of its accuracy and margin does it keep?
- **Q2, learning:** the classifier is trained from scratch with the compiler and the noisy device inside the
  loss. Does it learn as well as the same training would without noise?

The compiler changes only the physical implementation of each circuit (placement, routing, gate synthesis). The
model and its parameters are the same in every arm, so every difference between arms comes from the compiled
circuits under noise.

## 2. Design (`benchmarks/qml_home_eval.py`, `benchmarks/run_qml_home_2026-10-01.sh`)

**Model.**

- 4 qubits, 2 layers, data re-uploading, 20 parameters.
- Layer l: RY(pi x_q) on every qubit; then RY(a_lq) RZ(b_lq) on every qubit; then CZ on the ring (0,1), (1,2),
  (2,3), (3,0).
- After the layers: RY(c_q) on every qubit.
- Output z = <Z> of logical qubit 0; prediction sign(z); loss mean (y - z)^2.
- Heavy-hex devices have no 4-cycle, so the ring always needs routing.

**Data (teacher-student).**

- A teacher is the same model with parameters drawn uniformly in [-pi, pi] from a seed.
- Labels are sign(z_teacher). Points with |z_teacher| < 0.25 are skipped, and the classes are balanced.
- Train: 8 + 8 points (seed 31). Test: 16 + 16 points (seed 32). x is uniform in [-1, 1]^4.
- **Teacher rule:** the first teacher seed from 21 upwards for which noiseless SPSA (seed 40, 80 steps) reaches
  test accuracy >= 0.9. It uses numpy only, with no compiler and no noise. Checked before this lock, numpy only:
  the rule selects **teacher 23** (theta* test accuracy 0.969).

**Training.**

- SPSA with a = 0.6, c = 0.2, A = 5, alpha = 0.602, gamma = 0.101.
- The initial point is normal(0, 0.3) from the seed.
- The perturbations come from the same seeded stream, so for a given seed every arm sees the same initial point
  and the same perturbations (common random numbers).

**Arms** (all with the installed core 2026-09-29.1):

| tag | compiler |
|---|---|
| REL | psf_compile 2026-09-28.1 + psf_smart_layout 2026-09-26.m1 (taken from git `9131cee`, hash-checked), `compile_for_hardware(entangling_basis="cx", layout_search=True, seed_transpiler=0)` |
| C2 | release psf_compile 2026-10-01.1 + psf_smart_layout 2026-10-01.1, same call |
| A5 | [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) 2026-10-01.a5, `compile_for_model_circuit(qc, coupling_map, basis_gates, target=<device Target>)` |
| L3T | Qiskit `transpile(target=<device Target>, optimization_level=3, seed_transpiler=0)` |

The REL arm runs on core 2026-09-29.1, not on its original core 2026-09-28.1. The two give identical output
wherever the older core succeeds (Addendum 251).

**Noise and readout.**

- `qiskit_aer.noise.NoiseModel.from_backend(<fake device>)` with `AerSimulator(method="density_matrix")`.
- z is read exactly from the density matrix of the final-layout qubits: no shot noise, no readout error, no idle
  noise, no crosstalk.

**Q1.**

- theta* = noiseless SPSA (seed 40, 80 steps).
- The 32 test circuits go through every arm on FakeAuckland, FakeTorino and FakeKingston: 384 compiles.
- Recorded per circuit: noisy z, noiseless z of the compiled circuit, two-qubit count and compile time.

**Q2.**

- SPSA from scratch for 40 steps, seeds 41 and 42, on FakeAuckland and FakeTorino, for all four arms (16 runs).
- Each loss evaluation compiles and simulates the 16 training circuits: 1,280 compiles per run, plus 48 at the
  end.
- **IDEAL:** the same SPSA (same seeds and steps) on the noiseless numpy model. Checked before this lock, numpy
  only: test accuracy 1.000 (seed 41) and 0.625 (seed 42), mean 0.8125.

**Runner.** Q1 and the 16 Q2 runs, 6 processes in parallel, each single-threaded, then `score`. It is run from
the repository checkout at the lock commit. Times are home
times (WSL2, Ryzen 5 5500) and are reported only.

## 3. Predictions (scored only by `qml_home_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- the numpy model and Qiskit's Statevector agree on z for all 48 points (<= 1e-9);
- every compiled Q1 circuit, simulated without noise, gives the logical z (<= 1e-6);
- theta* reaches test accuracy >= 0.9;
- all 16 Q2 runs finish.

The margin is y·z.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | Q1: release C2 keeps at least the previous release's margin | mean margin C2 >= REL - 0.005 on all 3 devices | C2 < REL - 0.02 on any device |
| H2 | Q1: A5 keeps at least error-aware L3's margin | A5 >= L3T - 0.005 on >= 2 of 3 devices | A5 < L3T - 0.02 on >= 2 devices |
| H3 | Q1: on the two lower-noise devices the classifier stays smart | noisy test accuracy >= theta* accuracy - 2/32 for every arm on FakeTorino and FakeKingston | any arm below theta* accuracy - 6/32 there |
| H4 | Q2: learning through the compiler and the noise works | every arm's mean (over seeds) noisy test accuracy >= IDEAL mean - 0.10, on both devices | any arm below IDEAL mean - 0.25, or a run missing |
| H5 | Q2: the better compilers train to a lower loss | mean final noisy training loss C2 <= REL + 0.01 and A5 <= L3T + 0.01, on both devices | C2 > REL + 0.05 on both devices, or A5 > L3T + 0.05 on both devices |

**Reported without prediction:**

- FakeAuckland accuracy in Q1;
- two-qubit counts;
- compile time per compile and per run (home);
- training curves;
- the noiseless accuracy of the parameters each Q2 run ends with.

**Expectations, stated now:**

- C2 should route the ring with fewer two-qubit gates than REL. This is what H1 and the C2 half of H5 test.
  This line was written before the smoke run. On its two FakeAuckland points, REL and C2 gave the same
  two-qubit count and the same margin (section 5). The predictions were not changed after it.
- A5's estimate shares its physics with the simulator that scores it (Addenda 286-289), so H2 and the A5 half of
  H5 favour A5 by construction. A pass there is weaker evidence than a pass in H1.
- FakeAuckland has the qubits whose published errors are below their T1/T2 bound (Addendum 287). L3T may be
  misled there.

## 4. What this will not establish

- Anything about real hardware.
- Any model other than this one, or any device other than the three fake devices named here.
- Reliability: there are 2 training seeds.
- Q2 depends on SPSA as configured. The IDEAL arm shows how much of any shortfall is the optimizer's.

## 5. Development and dry runs (disclosed)

- **Design checks in numpy only (no compiler, no noise):**
  - A first dataset, sign(x0 x1 + x2 x3), with one encoding layer, did not learn (test accuracy 0.28). It was
    replaced by the teacher-student design.
  - Teacher 7 with data seeds 11 and 12 and SPSA seeds 100-102 learned noiselessly (test accuracy 0.94-1.0).
    These seeds are not used in the scored run.
  - Scanning teachers 21-30 led to the teacher rule above, which selects 23.
- **Smoke run at home** (`SMOKE=1`; data seeds 931 and 932, SPSA seed 941, 2 Q2 steps, FakeAuckland only, 2
  Q1 test points):
  - it checks the plumbing and the timing;
  - it is not scored;
  - its output is kept with the results.
  - **Result, run 2026-10-01 at home:** it ran end to end.
    - P0: numpy vs Statevector 8.9e-16; compiled noiseless vs logical 4.3e-15; theta* test accuracy 0.938.
    - Q1, 2 points on FakeAuckland (noisy accuracy / mean margin / median two-qubit count): REL 1.000 / 0.4894 / 17,
      C2 1.000 / 0.4894 / 17, A5 1.000 / 0.4930 / 17, L3T 1.000 / 0.4822 / 17.
    - Q2, 2 steps on FakeAuckland: noisy test accuracy 0.781 in every arm (IDEAL 0.781); compile time per compile
      about 0.03 s (REL, C2, L3T) and 0.17 s (A5), home.
    - Its scoring counted the FakeTorino Q2 runs as missing, because the smoke run uses FakeAuckland only.
    - **After the smoke run, `score` was changed to use the run's own device list** (`q2_devices` in the
      configuration). Nothing in the run path changed. The scored run uses the files locked below.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/qml_home_eval.py`](../../benchmarks/qml_home_eval.py) | `0a30fb22b709420418ed130a18531810616f405b42768a73157587e545b3ca32` |
| [`benchmarks/run_qml_home_2026-10-01.sh`](../../benchmarks/run_qml_home_2026-10-01.sh) | `3c734482bdaa14032d0d9e55f0111f730993b11d41c08d9a271027ff5df6ce72` |

The runner expects `qml_home_eval.py` in its own folder, so both are in [`benchmarks/`](../../benchmarks/).


---

<!-- ===== Addendum 291 (source: spare-qubit-cliff-addendum-291-2026-10-01.md) ===== -->

> **Note added when merging:** Results of the home pre-registration in Addendum 290 (lock commit 680cf08). Scored by the locked script and re-scored by an independent script written after the lock.

## Addendum 291 -- Results: the home QML test (Addendum 290). All five predictions confirmed; the classifier keeps its accuracy and learns through every compiler, and the test turned out to discriminate the compilers only through the margin (2026-10-01)

**Status: results of the pre-registered test in Addendum 290.** Scored by the locked `qml_home_eval.py score`, and
re-scored by an independent script written after the lock (section 7). No IBM account, no QPU: fake-provider
devices with Qiskit Aer noise models, at home (WSL2, Ryzen 5 5500).

## 1. Provenance

- Lock commit `680cf08` (Addendum 290 with [`benchmarks/qml_home_eval.py`](../../benchmarks/qml_home_eval.py) and [`benchmarks/run_qml_home_2026-10-01.sh`](../../benchmarks/run_qml_home_2026-10-01.sh))
  was pushed before the scored run; `origin/main` was at `680cf08` while the run was going.
- All 17 result files (q1.json and 16 q2 files) carry `git_head 680cf08` and the locked normalized hash of the
  script, `0a30fb22…`; one set of file hashes across all of them.
- The raw SHA-256 of the two scripts that ran (`62b01461…`, `77c97a2b…`, in `env.txt`) is the raw hash of the files
  that were locked.
- Versions: REL psf_compile 2026-09-28.1 + layout 2026-09-26.m1 (from `9131cee`, hash-checked); C2 release
  2026-10-01.1 + layout 2026-10-01.1; A5 2026-10-01.a5; core 2026-09-29.1; Qiskit 2.5.2, Aer 0.17.2, Python 3.12.13.
- Run started 2026-10-01 10:53 UTC; Q1 and all 16 Q2 runs finished after 4,208 s (6 processes in parallel).

## 2. Verdicts

| ID | Prediction | Verdict | Deciding numbers |
|---|---|---|---|
| P0 | harness | **PASS** | numpy vs Statevector 6.7e-16; compiled noiseless vs logical 3.3e-15; theta* 0.969; 16 of 16 Q2 runs |
| H1 | C2 margin >= REL - 0.005 on all 3 devices | **CONFIRMED** | C2 - REL: Auckland +0.0000, Torino +0.0045, Kingston +0.0018 |
| H2 | A5 margin >= L3T - 0.005 on >= 2 of 3 | **CONFIRMED** (3 of 3) | A5 - L3T: Auckland +0.0124, Torino +0.0004, Kingston -0.0003 |
| H3 | noisy accuracy >= theta* - 2/32 on Torino and Kingston, every arm | **CONFIRMED** | every arm 0.969 = theta* |
| H4 | every arm's Q2 noisy test accuracy >= IDEAL mean - 0.10 | **CONFIRMED** | arms 0.812-0.828; IDEAL 0.8125 |
| H5 | final train loss C2 <= REL + 0.01 and A5 <= L3T + 0.01, both devices | **CONFIRMED** | C2 - REL: +0.0000, -0.0029; A5 - L3T: -0.0174, -0.0012 |

## 3. Q1: theta* through each compiler

theta*: noiseless test accuracy 0.969 (31 of 32), noiseless mean margin 0.5449.

| device | arm | noisy accuracy | noisy mean margin | margin kept | margin lost to noise | two-qubit gates | compile s (median, home) |
|---|---|---|---|---|---|---|---|
| FakeAuckland | REL | 0.969 | 0.5094 | 0.935 | 0.0355 | 17 | 0.008 |
| FakeAuckland | C2 | 0.969 | 0.5094 | 0.935 | 0.0355 | 17 | 0.009 |
| FakeAuckland | A5 | 0.969 | 0.5184 | 0.951 | 0.0265 | 17 | 0.186 |
| FakeAuckland | L3T | 0.969 | 0.5061 | 0.929 | 0.0389 | 17 | 0.010 |
| FakeTorino | REL | 0.969 | 0.5075 | 0.931 | 0.0374 | 26 | 0.026 |
| FakeTorino | C2 | 0.969 | 0.5120 | 0.940 | 0.0329 | 20 | 0.031 |
| FakeTorino | A5 | 0.969 | 0.5304 | 0.973 | 0.0145 | 17 | 0.498 |
| FakeTorino | L3T | 0.969 | 0.5299 | 0.973 | 0.0150 | 17 | 0.017 |
| FakeKingston | REL | 0.969 | 0.5306 | 0.974 | 0.0143 | 26 | 0.029 |
| FakeKingston | C2 | 0.969 | 0.5323 | 0.977 | 0.0126 | 20 | 0.035 |
| FakeKingston | A5 | 0.969 | 0.5384 | 0.988 | 0.0065 | 17 | 0.666 |
| FakeKingston | L3T | 0.969 | 0.5387 | 0.989 | 0.0062 | 17 | 0.018 |

- The two-qubit count was the same for all 32 test circuits within each device and arm.
- **Accuracy did not discriminate anything.** The one misclassified test point (index 27) is misclassified by
  theta* without noise as well, and it is the only error in every one of the 12 device/arm cells. Noise did not flip
  a single prediction. The noise in these models mostly shrinks z towards 0 rather than changing its sign. The
  correct point with the smallest noiseless margin (0.019) kept at least 0.012 in every cell; for it, noise removed
  up to 39%.
- **The margin did discriminate.** Noise removed 1-7% of the mean margin, and how much depended on the compiler:
  - On FakeAuckland, REL and C2 produced the same 17-gate circuits and the same numbers; A5 lost the least (0.0265),
    L3T the most (0.0389).
  - On FakeTorino and FakeKingston, C2 cut the routed ring from 26 to 20 two-qubit gates, and its margin loss fell
    by 12% on each (Torino 0.0374 -> 0.0329; Kingston 0.0143 -> 0.0126).
  - A5 and L3T reached 17 gates there and lost about half as much margin as C2 (Torino 0.015 vs 0.033).

## 4. Q2: learning with the compiler and the noise in the loop

IDEAL (noiseless numpy, same SPSA seeds and steps): seed 41 accuracy 1.000, loss 0.212; seed 42 accuracy 0.625,
loss 0.706; mean accuracy 0.8125.

| device | arm | seed 41: noisy test acc / train loss | seed 42: noisy test acc / train loss | mean acc | mean loss | compile s per run | wall s per run |
|---|---|---|---|---|---|---|---|
| FakeAuckland | REL | 1.000 / 0.2399 | 0.656 / 0.7418 | 0.828 | 0.4908 | 15 | 265 |
| FakeAuckland | C2 | 1.000 / 0.2399 | 0.656 / 0.7418 | 0.828 | 0.4908 | 16 | 266 |
| FakeAuckland | A5 | 1.000 / 0.2276 | 0.625 / 0.7086 | 0.812 | 0.4681 | 268 | 524 |
| FakeAuckland | L3T | 1.000 / 0.2401 | 0.656 / 0.7308 | 0.828 | 0.4854 | 17 | 268 |
| FakeTorino | REL | 1.000 / 0.2364 | 0.625 / 0.7226 | 0.812 | 0.4795 | 38 | 2,089 |
| FakeTorino | C2 | 1.000 / 0.2331 | 0.625 / 0.7202 | 0.812 | 0.4766 | 61 | 2,103 |
| FakeTorino | A5 | 1.000 / 0.2218 | 0.656 / 0.7194 | 0.828 | 0.4706 | 631 | 2,669 |
| FakeTorino | L3T | 1.000 / 0.2226 | 0.656 / 0.7211 | 0.828 | 0.4718 | 22 | 1,841 |

- Every run made 1,328 compiles (40 SPSA steps x 2 evaluations x 16 training circuits, plus 48 at the end).
- **Learning through the compiler and the noise worked as well as learning without noise.** Every arm's mean test
  accuracy is within one test point (1/32) of IDEAL. The training curves track each other closely: mean of the two
  evaluated losses, seed 41, step 1 / 10 / 20 / 40 = about 1.9 / 0.47-0.50 / 0.35-0.36 / 0.25-0.26 in every arm.
- **Seed 42 is poor in every arm, and in IDEAL too** (accuracy 0.625-0.656 against IDEAL 0.625). That shortfall is
  the optimizer's (40 SPSA steps from that start), not the compiler's or the noise's.
- Final training loss ordered the arms the same way as the Q1 margin: A5 and L3T lowest on Torino, A5 lowest on
  Auckland; C2 equal to REL on Auckland (identical circuits) and 0.003 lower on Torino.
- Wall time on FakeTorino is dominated by simulation, not compilation: about 1,840-2,110 s per run for REL, C2
  and L3T, of which 22-61 s is compiling. A5's compile time (268 s per run on Auckland, 631 s on Torino) is 10-40 times
  the others and adds directly to the run time.

## 5. What the results mean

- **For the owner's question** ("does an AI get smarter through these circuits?"): for this small classifier on
  these fake devices, yes in the sense tested. It keeps its accuracy exactly (31 of 32), and it learns as well with
  the compiler and the device noise inside training as without. On these noise levels and a 17-26 two-qubit-gate
  circuit, the compiler choice does not change a single prediction.
- **What the compiler changes is the margin, by a few percent.** Fewer two-qubit gates and better placement keep
  more of it. That matters when there are more qubits, deeper circuits or noisier hardware; here it is too small
  to move the accuracy.
- **H1 is "not worse", not "better".** The release C2 improves on REL where REL routes the ring poorly (26 -> 20
  gates on FakeTorino and FakeKingston, margin +0.0045 and +0.0018) and is identical where REL already reaches 17
  (FakeAuckland). The expectation in Addendum 290 that C2 needs fewer two-qubit gates than REL held on 2 of 3
  devices.
- **Reported without prediction: the release still leaves room against error-aware Qiskit L3 on the larger
  devices.** For this circuit on FakeTorino and FakeKingston, C2 uses 20 two-qubit gates where L3T and A5 use 17.
  C2 loses about twice as much margin to noise as L3T (Torino 0.033 vs 0.015; Kingston 0.013 vs 0.006). This is
  one 4-qubit ring circuit, not a general comparison, but it is a concrete routing target: a 4-cycle on heavy-hex.
- **H2 and the A5 half of H5 favour A5 by construction** (Addendum 290, section 3). A5's error estimate shares its
  physics with the simulator that scores it. A5's clear lead is on FakeAuckland (+0.0124 margin over L3T, -0.017
  training loss), the device where some published two-qubit errors are below their T1/T2 bound (Addendum 287). On
  FakeTorino and FakeKingston, A5 and L3T are tied (within 0.0004 margin). A5 costs 10-40 times more compile time
  than L3T.
- **H3 was an easy test.** In this simulation, noise shrinks margins and did not flip any prediction. Separating
  the arms by accuracy would need a model with many small-margin points, much deeper circuits, or readout and shot
  noise.

## 6. What this does not establish

The limits in Addendum 290, section 4, all apply:

- nothing about real hardware;
- nothing about other models or other devices;
- two training seeds;
- z is read exactly from the density matrix, with no shots, no readout error, no idle noise and no crosstalk.

On real hardware, shot noise alone (about 1/sqrt(shots) on z) is of the same size as the margin differences
between compilers here.

## 7. Deviations and checks (disclosed)

- **Smoke scoring.** The smoke run used the pre-fix `score`, which counted the FakeTorino Q2 runs as missing (H4
  REFUTED). This was disclosed in Addendum 290, section 5. Re-scored with the locked script, the smoke output gives
  H4 and H5 CONFIRMED. It is still not a result (`dev/score_rescored_with_locked_script.md`).
- **Independent re-scoring, written after the lock** ([`benchmarks/qml_home_rescore.py`](../../benchmarks/qml_home_rescore.py)):
  - It does not import the scored script. It re-implements the model with dense 16x16 matrices (agreement with the
    scored model 5.6e-16 on random inputs).
  - It rebuilds teacher 23, the data, theta* (difference 8.6e-15) and IDEAL from the pre-registered seeds.
  - It checks provenance (commit, hashes, versions) in all 17 files, and the number of compiles and steps in every
    Q2 run. It also checks that the noiseless accuracy of every Q2 result reproduces.
  - It re-derives P0 and H1-H5 from the raw rows with the thresholds copied from Addendum 290.
  - Result: P0 PASS, H1-H5 CONFIRMED, no flags; identical to the locked `score` (`outputs/rescore.txt`). It was run
    once at home and once again on the copied data, with the same output.
- **Redaction.** Local paths in `env.txt` and `run.log` were replaced with `<repo>/` and `<home folder>/` before
  publishing. The result JSON files are unchanged, and the hashes in section 8 are of the published files.
- The run log was renamed from `qml_home_1001_run.log` to `run.log`. The release-file copies that `q2` extracts
  from `9131cee` (`rel_9131cee/`) are not included; they are in git.

## 8. Data (`data/2026-10-01/qml_home/`)

- `outputs/`: the scored run.
  - `q1.json`: the 384 Q1 rows, theta*, IDEAL.
  - `q2_<arm>_<device>_<seed>.json`: 16 files, each with the training log, the final parameters and the
    statistics.
  - `log_*.txt`, `env.txt`, `run.log`, `score.md` (locked score), `score_log.txt`, `rescore.txt`.
- `dev/`: the smoke run, not a result. It contains its `score.md` from the pre-fix scorer, plus
  `score_rescored_with_locked_script.md`.

SHA-256 (raw) of the main result files:

| file | SHA-256 |
|---|---|
| `outputs/q1.json` | `366ae3065fed4d8102abc791701d7580a17dfab3d3f74da2ea3d57e8a65fe98d` |
| `outputs/score.md` | `65a1cc665e23c3a4d8ee7db72604b34db7eb295dc7489296d6b93bb4a52a3332` |


---

<!-- ===== Addendum 292 (source: spare-qubit-cliff-addendum-292-2026-10-01.md) ===== -->

> **Note added when merging:** Home pre-registration of Phase A0 (snapshot data only, no IBM access). Locked by the git commit that adds this Addendum and benchmarks/a0_target_check.py, pushed before the scored run.

## Addendum 292 -- Pre-registration: Phase A0 Target check. How often is a reported gate error below the decoherence floor implied by the same snapshot's T1, T2 and gate duration, across all fake-provider devices? (2026-10-01)

**Status: pre-registration, written at home before any scored run.** It is locked by the git commit that adds this
document and [`benchmarks/a0_target_check.py`](../../benchmarks/a0_target_check.py), pushed before the scored run. No IBM account, no network access to
IBM, no QPU: the data are the device snapshots shipped with `qiskit-ibm-runtime`.

## 1. Why

- **Addendum 286 found the effect on one device.** On FakeAuckland, 16 of 56 cx gates are reported with an error
  below the decoherence limit of their own T1, T2 and duration. Single-qubit gates are affected too. On FakeTorino,
  no cz gate is affected.
- **Aer already applies that limit.** qiskit-aer's `NoiseModel.from_backend` applies max(reported error, floor) per
  gate: when the reported error is below the relaxation infidelity, it adds no depolarizing part.
- **So a noisy-simulation test favours any compiler that knows the floor** (Addenda 286-291), and the a5 lead over
  error-aware Qiskit L3 is real only where the floor binds.
- **Before building a "Target checker" (Phase A), A0 asks where the floor binds:**
  - on all device snapshots, not one;
  - by device generation;
  - and whether the current generation (Heron, cz) is affected at all.
- **The real-device question (A1) is separate.** Reading a live Target needs IBM access and the owner's go-ahead.
  A0 predicts what A1 is likely to find.

## 2. Definitions (`benchmarks/a0_target_check.py`)

- **Population:**
  - Every class named `Fake*` in `qiskit_ibm_runtime.fake_provider` that is a `BackendV2`.
  - Excluded: the fractional-gate test backend, generic backends and the base classes.
  - Duplicates by backend name are dropped.
  - Every exclusion or construction failure is listed in the output.
- **Gates:** the native two-qubit gate (cx, ecr or cz) and the single-qubit gates sx and x, per qubit tuple as in
  the Target.
- **Floor:** 1 - average gate fidelity of zero-temperature thermal relaxation on each qubit of the gate for the
  gate's duration, with T2 truncated to 2·T1.
  - In closed form, per qubit the process fidelity is (1 + 2e^(-t/T2) + e^(-t/T1))/4. These multiply over the
    qubits, and F_avg = (d·F_pro + 1)/(d + 1) with d = 2^n.
  - This is the relaxation part of qiskit-aer 0.17.2's `NoiseModel.from_backend`, at its default temperature 0.
- **Usable gate:** 0 < reported error < 0.5, a duration, and T1 and T2 known for its qubits. Errors >= 0.5 mark
  disabled gates.
- **Below floor:** a usable gate with reported error < floor.
- **Device class:** the native two-qubit gate in its Target.
  - cx: Falcon, Hummingbird and older.
  - ecr: Eagle.
  - cz: Heron.
  - A device with more than one is "mixed": reported, but not in the class predictions.
- **Device fraction:** below-floor two-qubit gates / usable two-qubit gates on that device.
- **Pooled fraction:** the same, over all devices of a class.

## 3. Predictions (scored only by `a0_target_check.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- the floor recomputed from the raw data by the scorer equals the stored one (<= 1e-12);
- the closed form equals qiskit's `average_gate_fidelity` of the `thermal_relaxation_error` channels built as Aer
  builds them (<= 1e-9, every gate);
- the infidelity of the error that `NoiseModel.from_backend` actually puts on each usable gate equals
  max(reported, floor) (<= 1e-9, every usable gate, none missing);
- at least 30 devices, including FakeAuckland, FakeTorino and FakeKingston.

**Replication** of Addendum 286 (same snapshots, independent code):

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| R1 | FakeAuckland as in Addendum 286 | 16 cx below floor, max floor/reported 1.5-1.7 for cx and 2.0-2.2 for sx | cx count outside 14-18 |
| R2 | FakeTorino has no cz below floor | 0 | > 0 |

**Predictions** (the author has seen no below-floor statistic for any device other than the two in Addendum 286):

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the current generation is rarely affected | cz class: median device fraction <= 0.02 | > 0.10 |
| H2 | the cx generation is often affected | cx class: median device fraction >= 0.05 | < 0.01 |
| H3 | the problem is a cx-era problem more than a cz-era one | cx > cz in both median device fraction and pooled fraction | cx <= cz in both |
| H4 | short T2 is the mechanism | >= 80% of below-floor two-qubit gates have a qubit whose T2 is below its device's median T2 | < 50% |
| H5 | FakeKingston, used in Addenda 287-291, is not affected | device fraction <= 0.02 | > 0.10 |

**Reported without prediction:**

- the ecr class;
- single-qubit (sx) fractions by class;
- every device's row (qubits, snapshot date, counts, largest floor/reported);
- per device, how much max(reported, floor) raises the summed two-qubit error;
- the rank correlation of snapshot date with the device fraction;
- excluded rows;
- T2 truncations.

**Reasoning behind the predictions:**

- The floor grows with gate duration over T2.
- cx gates on Falcon-era devices take about 300-500 ns, with T2 of tens of µs on weak qubits.
- Heron cz gates take under 100 ns, with T2 typically above 100 µs.
- ecr gates (Eagle) take about 500-700 ns with longer T2, so no direction is predicted for them.

## 4. What it means, stated now

- **If H1 and H5 hold:**
  - A1 (reading the live Targets of current Heron devices) is expected to find few or no violations.
  - Phase A on current hardware reduces to flagging individual gates.
  - The ties between a5 and L3T on FakeTorino and FakeKingston (Addendum 291) are what that predicts.
- **If H1 fails:** the problem is not historical, and a Target checker is relevant to current devices.
- **Either way, A0 shows only that snapshots contradict themselves.** Which side is wrong on hardware (the reported
  error, or the T1/T2 used to compute the floor) cannot be told from snapshots. That needs A1, and then measurement.

## 5. What this will not establish

- **Anything about live devices** (A1).
- **Which number in a snapshot is wrong.** T1, T2 and gate errors are measured at different times. T2 may be a
  Ramsey or an echo value. The reported duration may include padding. Any of these makes a gate look below the
  floor without the device being so.
- **That max(reported, floor) is the right correction on hardware.**

## 6. Development (disclosed)

- **The closed form was checked in numpy, without qiskit,** against an explicit Kraus-operator computation of the
  same channel: 2,000 random one- and two-qubit cases, maximum difference 4.4e-16.
- **The script ran end to end on stub qiskit modules** (synthetic devices), to check the plumbing and the scorer.
  No real snapshot was read.
- **The smoke run at home** (`--smoke`: FakeAuckland only; it prints only the harness checks and row counts, no
  below-floor statistic):
  - Result, run 2026-10-01 at home (Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.49.0, Python 3.12.13):
    - 110 rows (27 sx, 27 x, 56 cx), all usable.
    - Closed form vs Aer-built channels: 2.0e-15. Applied noise-model infidelity vs max(reported, floor):
      1.8e-15, 110 of 110 checked, none missing.
  - No change was made after the smoke run.
- **The scored run** reads every device once. Nothing is re-run to improve a score.

## 7. Locked file (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/a0_target_check.py`](../../benchmarks/a0_target_check.py) | `f9747e4b46368d56338fdffb3a13e13e4fa5ebf6355951ff5765ae77c8debf6e` |

**Run:**

```
python benchmarks/a0_target_check.py run --out <dir> > <dir>/run.txt 2>&1
python benchmarks/a0_target_check.py score --out <dir>
```


---

<!-- ===== Addendum 293 (source: spare-qubit-cliff-addendum-293-2026-10-01.md) ===== -->

> **Note added when merging:** Results of the home pre-registration in Addendum 292 (lock commit 7c1993b). Scored by the locked script and re-checked by an independent script written after the run.

## Addendum 293 -- Results: Phase A0 Target check (Addendum 292). Reported gate errors below the T1/T2 floor are common on cx and ecr devices and rare on cz (Heron) devices; Kingston is affected on a few short-T2 qubits. P0 pass; R1, R2 and H1-H4 confirmed; H5 ambiguous (2026-10-01)

**Status: results of the pre-registered check in Addendum 292.**

- Lock commit `7c1993b` was pushed before the scored run.
- Scored by the locked `a0_target_check.py score`, and re-checked by an independent script written after the run
  (section 6).
- Snapshot data only: no IBM account, no QPU.

## 1. Provenance

- **Software:** Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.49.0, Python 3.12.13; at home (WSL2).
- **Run:** started 2026-10-01 12:33 UTC and took 330 s.
- **Population:** 67 fake-provider devices, none skipped, 13,982 gate rows. 437 rows were excluded from the
  fractions: no error, error >= 0.5, no duration, or missing T1/T2. FakeKyoto has no usable ecr row.
- **FakeNighthawk** prints a warning that its properties "are not intended to represent typical nighthawk error
  values". It is kept, as pre-registered; removing it does not change any verdict (section 6).

## 2. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | closed form vs Aer-built channels 2.0e-15 on all 13,982 rows; the noise model's applied infidelity vs max(reported, floor) 3.4e-15 on all 13,545 usable rows, none missing; 67 devices |
| R1 | **CONFIRMED** | FakeAuckland: 16 of 56 cx below floor; largest floor/reported 1.613 (cx), 2.096 (sx) -- Addendum 286 reproduced by independent code |
| R2 | **CONFIRMED** | FakeTorino: 0 of 278 cz |
| H1 | **CONFIRMED** | cz class: median device fraction 0.000 (11 devices) |
| H2 | **CONFIRMED** | cx class: median device fraction 0.125 (43 devices) |
| H3 | **CONFIRMED** | cx vs cz: median 0.125 vs 0.000; pooled 0.181 vs 0.010 |
| H4 | **CONFIRMED** | 92.8% of the 636 below-floor two-qubit gates touch a qubit whose T2 is below its device median |
| H5 | **AMBIGUOUS** | FakeKingston: 18 of 338 cz below floor (0.053), between the bounds (CONFIRMED <= 0.02, REFUTED > 0.10) |

## 3. By device class

| class | devices | devices with any 2q gate below floor | median device fraction | pooled fraction | pooled sx fraction | rise in summed 2q error under max(): median / max |
|---|---|---|---|---|---|---|
| cx (Falcon, Hummingbird and older) | 43 | 30 | 0.125 | 0.181 (313 / 1,731) | 0.230 | 2.1% / 284% |
| ecr (Eagle) | 10 | 9 | 0.178 | 0.224 (281 / 1,255) | 0.261 | 8.4% / 36% |
| cz (Heron and later) | 11 | 5 | 0.000 | 0.010 (38 / 3,670) | 0.069 | 0.0% / 0.3% |
| mixed (FakeCairo, cx and ecr) | 1 | 1 | 0.160 | 0.160 | 0.185 | 8.0% |

- Rank correlation of snapshot date with device fraction: 0.069 over 65 dated devices. **Snapshot age does not
  explain the effect.**
- Qubits whose T2 > 2·T1 (truncated, as Aer does): 58.

## 4. Reading

- **The effect is not specific to FakeAuckland, and not a matter of old snapshots.**
  - 30 of 43 cx devices and 9 of 10 ecr devices have at least one two-qubit gate reported below its own
    decoherence floor.
  - The ecr (Eagle) class, which was not predicted, is the most affected. This includes snapshots dated April 2026:
    FakeBrussels 51 of 138, FakeStrasbourg 50 of 142, FakeOsaka 52 of 137.
  - The longer gate is the likely reason. Median gate durations in these snapshots are ecr 594 ns, cx 434 ns and
    cz 68 ns.
- **The current cz generation is mostly clean, but not entirely.**
  - The median cz device has no violation, and the pooled fraction is 1%.
  - 5 of 11 cz devices have some violations, and they sit on a few qubits.
  - On FakeKingston, all 18 below-floor cz gates touch one of six qubits with T2 = 15-26 µs: qubits 10, 37, 88,
    123, 137 and 139 (device median 144 µs).
  - This is why H5 missed its CONFIRMED bound: the prediction that Kingston is unaffected was too strong.
- **The mechanism is short T2 (H4).** Some violations are extreme, for example:
  - FakeKawasaki ecr(112,126) is reported at 0.0085, but its floor is 0.234.
  - FakeToronto cx(21,23) is reported at 0.0123, with floor 0.169.
  - These are qubits whose T1/T2 say they are close to unusable, while their gate errors say they are fine. A
    compiler that trusts the reported error will place work on them.
- **For the a5 / L3T comparisons (Addenda 287-291).**
  - FakeTorino has no below-floor cz gate; FakeKingston has 5% of its cz gates below floor, on short-T2 qubits.
  - A5 and L3T tied on both in Addendum 291. A5's clear lead was on FakeAuckland (cx).
  - This is consistent with A5's advantage coming from the floor. It is not a test of that.
- **What it means for Phase A.**
  - A Target checker is relevant to current hardware: to the Eagle (ecr) devices broadly, and to individual
    short-T2 qubits on Heron.
  - The earlier suggestion that Auckland might be an old-snapshot artefact is not supported.
  - A1 (reading live Targets, with the owner's go-ahead) can now be pre-registered with concrete expectations:
    - Eagle devices: many violations;
    - Heron devices: few, on short-T2 qubits.

## 5. What this does not establish

As stated in Addendum 292:

- **Snapshots contradict themselves; they do not say which number is wrong.** T1, T2 and gate errors are measured
  at different times. T2 may be a Ramsey or an echo value. The reported duration may include padding.
- **Nothing about live devices.**
- **Not that max(reported, floor) is the right correction on hardware.** Short T2 from a single measurement can be
  transient, for example two-level-system defects that move.

## 6. Independent check (written after the run)

[`benchmarks/a0_verify.py`](../../benchmarks/a0_verify.py) does not import the scored script. It recomputes every floor from explicit Kraus
operators instead of the closed form (largest difference from the stored floors 6.7e-16). It then re-derives the
below-floor sets, the class statistics and the deciding numbers. All of them match the locked score: the class
medians and pooled fractions in section 3, H4 = 0.928 of 636, and Auckland, Torino and Kingston at 16/56, 0/278
and 18/338.

Without FakeNighthawk, the cz-class median is still 0.000 (10 devices).

## 7. Data (`data/2026-10-01/a0/`)

- `outputs/a0_raw.json.gz`: every gate row and device record.
- `outputs/a0_score.md`: the locked score, including the per-device table.
- `outputs/run.txt`: the run log. One local path in a library warning was replaced with `<venv>/`.
- `outputs/verify.txt`: the output of the independent check.
- `dev/a0_raw_smoke.json.gz`: the smoke run, FakeAuckland only.

| file | SHA-256 (raw) |
|---|---|
| `outputs/a0_raw.json.gz` | `9aadd7aa8f804f76c59ad951209e3e199154a89794942966e53556a3bccf314d` |
| `outputs/a0_score.md` | `47ec9ec85ad4a338aaa096e0cbc6d7d1eb094f4cef4ff1b4404b37c48a5f0134` |


---

<!-- ===== Addendum 294 (source: spare-qubit-cliff-addendum-294-2026-10-01.md) ===== -->

> **Note added when merging:** Not pre-registered: diagnosis of an upstream bug report (Qiskit issue #17057), run at home. The owner posted the findings to the issue in their own words.

## Addendum 294 -- Qiskit issue #17057: the wrong ZSX/CX synthesis near the two-CX boundary predates the Rust port; cause located and a one-line fix candidate tested (2026-10-01)

**Status: diagnosis and follow-up of an upstream bug report.** This was not pre-registered. It is an
investigation of a bug, not a test of a prediction.

- **Runs:** at home (WSL2), Qiskit 2.5.2 plus released Qiskit versions in throw-away venvs.
- **Issue:** https://github.com/Qiskit/qiskit/issues/17057. It was filed by the owner and was labelled `bug`.
- **Maintainer response (2026-09-30):** a maintainer replied that it is a real, serious compilation error and
  suggested a typo made during the Python-to-Rust translation. A synthesis maintainer was asked to look at it after
  a holiday.
- **This Addendum:** the owner's follow-up comment (2026-10-01) reports the results below.

## 1. The report (summary)

- `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` returns a circuit with 1 - F_avg = 6.987e-2.
- The input is exp(i(0.6 XX + 0.3 YY + c ZZ)) with c of about 3e-8 to 3e-7. The output uses 3 cx.
- The infidelity is the same at every c in the band.
- The default Euler basis and the cz basis are not affected.

## 2. Where the error is

- **The affected path.** With euler_basis ZSX (or ZSXX), a CX gate and three CX uses, the decomposer takes the
  "pulse-optimal" path `_get_sx_vz_3cx_efficient_euler` (Python up to 1.0) / `get_sx_vz_3cx_efficient_euler`
  (Rust from 1.1, `crates/synthesis/src/two_qubit_decompose/basis_decomposer.rs`). The two versions were compared
  line by line: Euler-angle index mapping, matrix order, branch conditions, `atol=1e-10`. The Rust code is a
  faithful port.
- **The angle that picks the branch.** The path splits the KAK pieces on qubit 0 into ZXZ angles, and on qubit 1
  into XZX angles. Their sum x12 = euler_q0[1][2] + euler_q0[2][0] decides the branch. Normally x12 = 3π, and the
  "π-multiple" branch emits rz(±θ1), with θ1 = euler_q0[1][1].
- **What goes wrong near the boundary.** Near c = 0, the middle unitary on qubit 0 (`u1ra·rz(-2c)·u1rb`) has ZXZ
  angle θ2 = 2c ≈ 0. Its split into λ and φ is then ill-conditioned:
  - x12 = 3π + δ, with sin(x12) ≈ 3.9e-17 / c, as measured;
  - for c below about 4e-7, δ exceeds `atol = 1e-10`, and x12 fails the π-multiple test;
  - it then falls into the branch commented "non-optimal but doesn't seem to occur currently", which emits only
    Rx(x12) and never emits rz(θ1).
- **The numbers agree.** In the reported example θ1 = 0.6. A missing Rz(0.6) on one qubit of two gives
  1 - F_avg = 1 - (16cos²(0.3)/4 + 1)/5 = 0.069866, i.e. exactly the reported 6.987e-2. This is also why the error
  does not depend on c.

## 3. When the error was introduced (released versions, reproducer at four values of c)

| Qiskit | c = 1e-8 | c = 1e-7 | c = 2e-7 | c = 1e-5 | implementation |
|---|---|---|---|---|---|
| 0.45.3 | 2 cx, ok | 3 cx, **6.987e-2** | 3 cx, ok | ok | Python |
| 1.0.2 | 2 cx, ok | 3 cx, **6.987e-2** | 3 cx, ok | ok | Python |
| 1.1.2, 1.2.4, 1.4.6, 2.0.3, 2.2.3, 2.4.2, 2.5.2 | 2 cx, ok | 3 cx, **6.987e-2** | 3 cx, **6.987e-2** | ok | Rust |

**The error predates the Rust port.** The port only widened the affected band slightly, through the numerical
details of the Euler-angle extraction (c = 2e-7 is correct in 1.0.2 and wrong from 1.1.2 on).

## 4. Fix candidate and its test

**Candidate.** In that branch (x12 neither 0 nor a multiple of π), emit rz(θ1) on qubit 0 before the x12
rotation:

```rust
if x12_is_non_zero && !x12_is_pi_mult {
    gates.push((StandardGate::RZ.into(), smallvec![euler_q0[1][1]], smallvec![0]));
}
// then the existing `if x12_is_half_pi { ... } else if x12_is_non_zero && !x12_is_pi_mult { ... }`
```

**Test setup.** A Python port of the 0.45 function, fed with the KAK pieces from 2.5.2's
`decomp3_supercontrolled`. It was run with and without the fix (`diag_17057_v2.py`):

| set | shipped 2.5.2 | 0.45 port | 0.45 port with fix |
|---|---|---|---|
| reported family, c = 1e-9 … 0.1 | wrong for c in 2e-8 … 3e-7 | wrong for c <= 3e-7 | all <= 3.3e-16 |
| 300 Haar-random targets | 0 wrong | 0 wrong | 0 wrong (worst 1.1e-15) |
| 300 random near-boundary targets (random a, b; c log-uniform in 1e-9 … 1e-5; random local unitaries) | **122 wrong, worst 0.36** | 191 wrong | **0 wrong** (worst 8.6e-10) |

- "Wrong" means 1 - F_avg > 1e-9.
- The 0.45 port is forced to the 3-cx path for every target, which is why it counts more failures than the shipped
  decomposer: the shipped one takes 2 cx for some of them.
- The fix's worst value, 8.6e-10, is within the decomposer's default tolerance of 1 - 1e-9.
- **The error is not always 0.07.** It is the infidelity of the missing rz(θ1), so it depends on θ1. The worst
  case seen was 0.36.

**Limits:**

- the fix was tested on the Python port, not on a rebuilt Rust crate;
- the x12 = π/2 branch was never reached in these tests;
- the failure rate of 122 out of 300 depends on how the near-boundary set was drawn.

The root cause is that the branch test uses an absolute tolerance on an angle that is ill-conditioned when θ2 ≈ 0.
How to fix it is the maintainers' decision.

**Workaround:** `pulse_optimize=False` gives the exact result at every c tested.

## 5. What was posted

The owner posted the findings of sections 2-4 to the issue in their own words on 2026-10-01. The post included the
candidate fix, the limits above and the AI-assistance disclosure. The issue remains open.

## 6. Files (`data/2026-10-01/qiskit_17057/`)

| file | what | SHA-256 (raw) |
|---|---|---|
| `diag_17057.py` | v1: shipped vs the 0.45 port, branch variables | `1571af0e296b76bce4d0e3193ae5c6282f4033688cb9969eab4f6539b51f83cd` |
| `diag_17057_v2.py` | v2: adds the fix candidate and the random checks | `baa37864ad937ce36182e540233dd9a55b4dada3001edb343184dde81d858ee1` |
| `bisect_17057.sh` | released versions in throw-away venvs | `e084d9d702d261ccdc9bd0af496a970ed1935675c21b9f8edc84542edc6f381a` |
| `diag_17057_v1_output.txt`, `bisect_output.txt` | outputs, as printed | |
| `diag_17057_v2_output_tail.txt` | the last four lines of the v2 output (it was run with `tail -4`) | |


---

<!-- ===== Addendum 295 (source: spare-qubit-cliff-addendum-295-2026-10-01.md) ===== -->

> **Note added when merging:** A decision record on the A0 finding of Addendum 293; no test.

## Addendum 295 -- Decision: the A0 finding (reported gate errors below the T1/T2 floor) is not filed upstream for now; the route to A1 and to publication (2026-10-01)

**Status: a decision record, not a test.** It concerns the result of Addendum 293.

## 1. The question

Addendum 293 found reported gate errors below the decoherence floor implied by the same snapshot's T1, T2 and gate
duration:

- on 18% of cx gates and 22% of ecr gates;
- on 1% of cz gates, concentrated on a few short-T2 qubits.

The owner asked whether this should be reported upstream like issue #17057 (Addendum 294).

## 2. Decision: not now

| possible place | why not, now |
|---|---|
| qiskit-aer | It is not a bug. `NoiseModel.from_backend` applies max(reported, floor) by design. Addenda 292-293 used this, and P0 verified it on all 13,545 usable gates. |
| qiskit-ibm-runtime fake backends | It is not a code bug. The snapshots copy IBM's published calibration data, and the inconsistency is in that data. |
| Qiskit error-aware layout (VF2PostLayout, Target-based scoring) | This could become a feature request: weigh T1/T2 and duration, not only the reported error. But snapshot data alone invites the obvious question "does this happen on live devices?", and it cannot be answered yet. |

- **#17057 was a code defect.** It could be reproduced and fixed independently of any device.
- **A0 is an observation about calibration data.** Its significance depends on live devices, and on which number
  is wrong.

## 3. The route

1. **A1, pre-registered.** Read the live Targets of current devices, with the owner's go-ahead (no jobs are
   needed). Count the same quantity with `a0_target_check.py`'s definitions.
   - **Expectations from A0:** Eagle (ecr) devices, many violations; Heron (cz) devices, few, on short-T2 qubits.
   - **Record:** the calibration timestamps, so that the gap in time between the T1/T2 and the gate-error
     measurements can be examined.
2. **If A1 confirms the pattern on live devices:**
   - write it up as a short research note (data, method, the Aer max() behaviour, the effect on error-aware
     placement);
   - then, separately, propose to Qiskit an optional T1/T2-aware error score for layout, citing the note.
3. **If A1 finds no violations on live devices:** record that. The A0 pattern is then a property of the snapshots,
   and no upstream action follows.

**Rules that carry over:**

- the owner posts upstream personally, in their own words;
- account names, instance names and CRN lines are redacted before anything is published.

## 4. Not changed by this decision

- The results of Addendum 293.
- The caution that snapshots do not say whether the reported error or the T1/T2 is wrong.
- A5's status: its advantage over error-aware Qiskit L3 in noisy simulation is favoured by construction, and is
  untested on hardware.


---

<!-- ===== Addendum 296 (source: spare-qubit-cliff-addendum-296-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration (part 2 of the QML test of Addenda 290-291). Locked by the git commit that adds this Addendum and its two scripts, pushed before the scored run.

## Addendum 296 -- Pre-registration: home QML test, part 2. Learning with 8 seeds on three devices, and keeping accuracy with a deeper classifier, a fragile test set, readout error and shots (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, [`benchmarks/qml_home2_eval.py`](../../benchmarks/qml_home2_eval.py) and
  [`benchmarks/run_qml_home2_2026-10-02.sh`](../../benchmarks/run_qml_home2_2026-10-02.sh), pushed before the scored run.
- **No hardware:** no IBM account, no QPU. Fake-provider devices and Qiskit Aer noise models only.

## 1. Why: where Addendum 291 was weak

Addendum 291 (the results of Addendum 290) confirmed all five predictions, but it had three weaknesses:

1. **Learning (Q2) used only two training seeds, on two devices.** FakeKingston was not in Q2.
2. **Accuracy could not tell the compilers apart.** Noise flipped no prediction, so the compilers differed only in
   margin, by a few per cent.
3. **There was an open item.** On the larger devices, the release (C2) used 20 two-qubit gates where error-aware
   Qiskit L3 (L3T) and A5 used 17. C2 lost about twice as much margin to noise.

Part 2 addresses each of them:

- **Q2W** widens the learning test.
- **Q1D** is built so that accuracy can register noise:
  - a deeper model, so there are more two-qubit gates;
  - a "fragile" test set of small-margin points;
  - the measured qubit's readout error and finite shots, applied in the score.
- **H3** tests the open item at greater depth.

## 2. Design (`benchmarks/qml_home2_eval.py`, `benchmarks/run_qml_home2_2026-10-02.sh`)

**Arms, devices and noise:** exactly as in Addendum 290.

- Arms: REL from git 9131cee, hash-checked; C2 is release 2026-10-01.1; A5 is psf_ai_compile a5 with the Target;
  L3T is Qiskit level 3 with the Target. All run on core 2026-09-29.1.
- Devices: FakeAuckland, FakeTorino, FakeKingston.
- Noise: `NoiseModel.from_backend` with Aer `density_matrix`. z is read exactly from the density matrix of the
  final-layout qubits.

One change: each loss evaluation's circuits are simulated in one Aer job instead of one job per circuit. This
changes only the run time.

**Q2W (learning, wide).**

- Model and data: the Addendum 290 model (4 qubits, 2 layers), the same teacher rule (it selects teacher 23) and
  the same data seeds (train 31, test 32).
- Training: SPSA from scratch with the same constants, 40 steps.
- Scope: seeds 41-48, all four arms, all three devices, 96 runs.
- Recorded at the end of each run: per test point, the noisy z and the readout error of the measured physical
  qubit.
- IDEAL: the same SPSA without noise. Checked in numpy before this lock, test accuracy by seed:

  | seed | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
  |---|---|---|---|---|---|---|---|---|
  | accuracy | 1.000 | 0.625 | 1.000 | 1.000 | 0.781 | 0.906 | 0.688 | 1.000 |

  Mean 0.875. Seeds 41 and 42 reproduce Addendum 291.

**Q1D (keeping, deep).**

- **Model:** the same family with 4 layers, 36 parameters.
- **Teacher rule:** for teacher seeds 81, 82, …:
  - training set: 32 + 32 points (seed 91), with |z_teacher| >= 0.25;
  - θ* is trained without noise by Adam on central finite-difference gradients (seed 93, 400 iterations,
    lr 0.05);
  - the first teacher whose θ* reaches test accuracy >= 0.85 on test set A is used.
- **Test set A:** 16 + 16 points (seed 92), with |z_teacher| >= 0.25, as before.
- **Checked in numpy before this lock:** teachers 81-83 reach 0.812, 0.750 and 0.531. The rule selects teacher 84:
  θ* test accuracy 0.875 (28 of 32), noiseless mean margin on A 0.447.
- **Test set B ("fragile"):** 32 + 32 points (seed 94) with 0.01 <= |z_θ*| <= 0.10, labelled by the sign of θ*. A
  "flip" is a noisy sign that differs from θ*'s noiseless sign.
- All 96 circuits go through every arm on all three devices: 1,152 compiles.

**Readout and shots (in the score).**

- **Readout:** the measured qubit's readout error e (the Target's measure error of the physical qubit that carries
  logical qubit 0) is applied as a symmetric flip: p0 → p0(1 - e) + (1 - p0)e.
- **Shots:** z is estimated from 4,000 shots, repeated 20 times.
- **Same random numbers in every arm:** the uniforms are seeded by device, set and point only. A difference between
  arms therefore comes from the compiled circuit, not from sampling luck.

## 3. Predictions (scored only by `qml_home2_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- numpy agrees with Qiskit's Statevector for both models (<= 1e-9);
- every compiled Q1D circuit, simulated without noise, gives the logical z (<= 1e-6);
- deep θ* >= 0.85 and Q2W θ* >= 0.9;
- all 96 Q2W runs finish.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | Q1D: C2 keeps at least REL's accuracy and margin | on every device: B shot-flip rate C2 <= REL + 0.01 and A mean margin C2 >= REL - 0.005 | on any device: C2 > REL + 0.03 in B shot flips, or C2 < REL - 0.02 in A margin |
| H2 | Q1D: A5 keeps at least L3T's | the same two conditions, A5 vs L3T, on >= 2 of 3 devices | the REFUTED condition, A5 vs L3T, on >= 2 devices |
| H3 | Q1D: the open item persists at depth: error-aware L3 keeps more margin than the release on the larger devices | A margin L3T >= C2 + 0.005 on FakeTorino and FakeKingston | L3T < C2 on either |
| H4 | Q1D: the fragile set registers noise | exact B flip rate >= 0.05 for every arm on FakeAuckland | < 0.01 for every arm there |
| H5 | Q2W: learning through the compiler and the noise works as well as without noise | every arm's mean noisy test accuracy over 8 seeds >= IDEAL mean - 0.05, on every device | any arm < IDEAL mean - 0.15, or a run missing |
| H6 | Q2W: the better compilers train to a lower loss | mean final train loss C2 <= REL + 0.005 and A5 <= L3T + 0.005, on every device | C2 > REL + 0.02 on >= 2 devices, or A5 > L3T + 0.02 on >= 2 devices |

**Reported without prediction:**

- exact and shot-based accuracy on A;
- the exact B flip rates on FakeTorino and FakeKingston;
- Q2W shot-based test accuracy;
- noiseless accuracy of each Q2W result;
- two-qubit counts;
- compile and wall times (home).

**Expectations, stated now:**

- **Where noise acts.** In Addendum 291's data, noise shrank z by 1-7% and pushed it towards +1 by about +0.015 on
  FakeAuckland and about +0.003 on the larger devices, for the 2-layer circuit. Twice the depth should roughly
  double both. B points with -0.03 < z < 0 on FakeAuckland should then flip without shots (H4).
- **Why H1 and H3 can both hold.** H1 only asks that C2 is not worse than REL. H3 says that C2 still trails L3T
  on Heron, which was the case in Addendum 291 (20 against 17 two-qubit gates for 2 layers).
- **H2 and the A5 half of H6 favour A5 by construction.** A5's error estimate shares its physics with the
  simulator, as noted in Addenda 286-291. Addendum 293 adds a detail: the floor binds on FakeAuckland, not at all on
  FakeTorino, and on six short-T2 qubits of FakeKingston.

## 4. What this will not establish

- Anything about real hardware.
- Other models or devices.
- Readout is modelled as a symmetric flip of the one measured qubit, with no crosstalk, drift or idle noise.
- Q2W depends on SPSA as configured. IDEAL shows how much of any shortfall is the optimizer's.

## 5. Development (disclosed)

- **Q1D configuration.** Chosen with numpy-only checks:
  - 4-layer teachers trained by SPSA (120 steps, 16 training points) did not learn (test accuracy 0.38-0.81 over
    teachers 51-70).
  - 3-layer, and 2-layer-teacher / deeper-student, variants were also tried. Both learned poorly.
  - The final choice is 4 layers with 64 training points and Adam on exact (finite-difference) gradients.
- **Analysis of Addendum 291's published data, to set the expectations above:** a linear fit of noisy z on
  noiseless z per device and arm.
- **The shot model, the fragile band and the thresholds** were fixed before any Q1D or Q2W circuit was compiled.
- **The scorer** was run on synthetic files to check the plumbing.
- **Smoke run at home** (`SMOKE=1`: other data seeds, 1 + 1 deep points per set, 10 Adam iterations, Q2W seed 941
  with 2 steps on all three devices). It checks the plumbing and the timing, and is not scored. Result:
  - **Run 2026-10-02 at home.** It ran end to end: Q1D, 12 Q2W runs (2 steps each), then the score.
    All 12 runs had finished after 101 s.
  - **Wall time per smoke run (home):**
    - REL, C2 and L3T: 5 s (FakeAuckland), 18-25 s (FakeTorino, FakeKingston);
    - A5: 27 s, 62 s and 77 s.
    Batching the simulation works: in Addendum 291 one FakeTorino loss evaluation took about 25 s.
  - **Two-qubit counts of the 4-layer circuit (1 + 1 smoke points, reported only):**
    - FakeAuckland: 37 in every arm;
    - FakeTorino and FakeKingston: REL 58, C2 44, A5 37, L3T 37.
  - **Its verdict lines are not results.** They come from 1 + 1 Q1D points, a 10-iteration θ* and 2 SPSA steps.
  - **No change was made after the smoke run.** The design, the thresholds and both scripts are as they were
    before it.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/qml_home2_eval.py`](../../benchmarks/qml_home2_eval.py) | `9e12f5bf34c29542ebe608a0c672519a52a0039d796dff5fc19ea600a2a6b95f` |
| [`benchmarks/run_qml_home2_2026-10-02.sh`](../../benchmarks/run_qml_home2_2026-10-02.sh) | `47cc2275ca837fc970753074b6eb9a27f5b83e7fc9438de820de963b26bce106` |

**Run** (from the repository checkout at the lock commit):

```
setsid nohup bash benchmarks/run_qml_home2_2026-10-02.sh <repo> <out> > <log> 2>&1 < /dev/null &
```


---

<!-- ===== Addendum 297 (source: spare-qubit-cliff-addendum-297-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the home pre-registration in Addendum 296 (lock commit 4839966). Scored by the locked script and re-checked by an independent script written after the run.

## Addendum 297 -- Results: home QML test, part 2 (Addendum 296). All six predictions confirmed. With a fragile test set, noise now flips predictions, and how many depends on the compiler: on FakeTorino the release halves its gap to error-aware Qiskit L3 but still flips twice as many points; learning over 8 seeds × 3 devices matches noiseless training (2026-10-02)

**Status: results of the pre-registered test in Addendum 296.**

- **Lock:** commit `4839966`, pushed before the scored run.
- **Scoring:** by the locked `qml_home2_eval.py score`, and re-checked by an independent script written after the
  run (section 6).
- **Setting:** fake-provider devices with Aer noise models, at home (WSL2, Ryzen 5 5500, 6 processes in
  parallel). The run started 2026-10-02 00:30 UTC; all 96 Q2W runs and Q1D finished after about 73 minutes.

## 1. Verdicts

| ID | Verdict | Deciding numbers |
|---|---|---|
| P0 | **PASS** | numpy vs Statevector 8.0e-16 / 6.7e-16; compiled noiseless vs logical 1.4e-13; deep θ* 0.875 (teacher 84), Q2W θ* 0.969 (teacher 23); 96 of 96 Q2W runs |
| H1 | **CONFIRMED** | C2 vs REL, B shot flips: Auckland 0.205 vs 0.205, Torino 0.094 vs 0.123, Kingston 0.045 vs 0.052; A margin C2 >= REL on every device |
| H2 | **CONFIRMED** (3 of 3) | A5 vs L3T, B shot flips: 0.145 vs 0.155, 0.062 vs 0.059, 0.040 vs 0.041; A margins within 0.006 |
| H3 | **CONFIRMED** | A margin L3T - C2: Torino +0.043, Kingston +0.010 |
| H4 | **CONFIRMED** | exact B flip rate on FakeAuckland: 0.203 (REL, C2), 0.172 (A5, L3T) |
| H5 | **CONFIRMED** | every arm's mean noisy test accuracy 0.867-0.879, against IDEAL 0.875 |
| H6 | **CONFIRMED** | final train loss C2 - REL: 0.000, -0.006, -0.002; A5 - L3T: -0.014, -0.001, -0.001 |

## 2. Q1D: the deep classifier through each compiler

- θ* (4 layers): noiseless test accuracy 0.875 on A, mean margin 0.447.
- B: 64 points with 0.01 <= |z_θ*| <= 0.10.
- "Shots" means 4,000 shots × 20 repetitions, with the measured qubit's readout error, using the same random
  numbers in every arm.

| device | arm | 2q gates | A acc (exact) | A margin | B flips (exact) | B flips (shots) | readout error of the measured qubit (mean) |
|---|---|---|---|---|---|---|---|
| FakeAuckland | REL | 37 | 0.875 | 0.351 | 0.203 | 0.205 | 0.0075 |
| FakeAuckland | C2 | 37 | 0.875 | 0.351 | 0.203 | 0.205 | 0.0075 |
| FakeAuckland | A5 | 37 | 0.875 | 0.377 | 0.172 | 0.145 | 0.0067 |
| FakeAuckland | L3T | 37 | 0.875 | 0.371 | 0.172 | 0.155 | 0.0064 |
| FakeTorino | REL | 58 | 0.875 | 0.355 | 0.109 | 0.123 | 0.0481 |
| FakeTorino | C2 | 44 | 0.875 | 0.371 | 0.078 | 0.094 | 0.0481 |
| FakeTorino | A5 | 37 | 0.875 | 0.415 | 0.047 | 0.062 | 0.0111 |
| FakeTorino | L3T | 37 | 0.875 | 0.414 | 0.047 | 0.059 | 0.0118 |
| FakeKingston | REL | 58 | 0.875 | 0.417 | 0.000 | 0.052 | 0.0062 |
| FakeKingston | C2 | 44 | 0.875 | 0.421 | 0.000 | 0.045 | 0.0062 |
| FakeKingston | A5 | 37 | 0.875 | 0.431 | 0.000 | 0.040 | 0.0114 |
| FakeKingston | L3T | 37 | 0.875 | 0.432 | 0.000 | 0.041 | 0.0081 |

**Reading.**

- **The fragile set does what Addendum 291 lacked: noise now changes predictions, and how often depends on the
  compiler.**
  - On FakeAuckland, gate noise alone flips 17-20% of the small-margin points.
  - On FakeTorino it flips 5-11%.
  - On FakeKingston gate noise flips none; shot noise flips 4-5%.
  - The well-separated set A is still not affected (0.875 everywhere): accuracy on clear-cut inputs survives this
    noise.
- **On FakeTorino, the release C2 sits between the previous release and error-aware L3.**
  - It cuts the routed 4-layer ring from 58 to 44 two-qubit gates, and the exact flip rate from 0.109 to 0.078.
  - L3T and A5 route it with 37 gates and flip 0.047.
  - The open item of Addendum 291 therefore persists at depth, and the margin gap grows: L3T - C2 was +0.018 for
    2 layers and is +0.043 for 4.
- **Part of the FakeTorino shot gap is readout placement, not gates.**
  - REL and C2 put logical qubit 0 on a qubit with readout error 0.048; L3T and A5 chose ones near 0.011.
  - The exact flips (no readout, no shots) still differ (0.078 against 0.047), so the gates account for the rest.
  - The PSF-Zero layout does not consider readout error. This is a concrete improvement target alongside the
    routing.
- **A5 and L3T are tied on the Heron devices; A5 is ahead on FakeAuckland** (margin +0.006, shot flips 0.145 against
  0.155). As pre-registered, H2 favours A5 by construction.

## 3. Q2W: learning with the compiler in the loop, 8 seeds × 3 devices

IDEAL (noiseless SPSA, same seeds): 1.000, 0.625, 1.000, 1.000, 0.781, 0.906, 0.688, 1.000; mean 0.875.

| device | arm | noisy test acc | shot test acc | noisy train loss | noiseless acc of result | wall s per run |
|---|---|---|---|---|---|---|
| FakeAuckland | REL | 0.871 | 0.868 | 0.410 | 0.875 | 36 |
| FakeAuckland | C2 | 0.871 | 0.868 | 0.410 | 0.875 | 36 |
| FakeAuckland | A5 | 0.871 | 0.873 | 0.386 | 0.875 | 288 |
| FakeAuckland | L3T | 0.875 | 0.873 | 0.399 | 0.875 | 37 |
| FakeTorino | REL | 0.867 | 0.868 | 0.398 | 0.875 | 170 |
| FakeTorino | C2 | 0.871 | 0.869 | 0.393 | 0.875 | 176 |
| FakeTorino | A5 | 0.875 | 0.874 | 0.377 | 0.875 | 736 |
| FakeTorino | L3T | 0.875 | 0.874 | 0.378 | 0.875 | 156 |
| FakeKingston | REL | 0.875 | 0.875 | 0.374 | 0.879 | 226 |
| FakeKingston | C2 | 0.875 | 0.875 | 0.372 | 0.879 | 237 |
| FakeKingston | A5 | 0.875 | 0.876 | 0.368 | 0.879 | 993 |
| FakeKingston | L3T | 0.879 | 0.876 | 0.368 | 0.879 | 207 |

**Reading.**

- **Training through the compiler and the device noise reached the same accuracy as noiseless training, on every
  device and in every arm.** The largest shortfall is 0.008, a quarter of one test point. Addendum 291's two-seed
  result holds over eight seeds and on FakeKingston.
- **The final training loss orders the arms as the Q1D margins do,** but the differences (<= 0.025) do not reach
  test accuracy.
- **Batching the simulation changed only the time.** Seeds 41 and 42 reproduce Addendum 291 exactly on FakeAuckland
  and FakeTorino, in every arm: test accuracy and final training loss to four decimals. The FakeTorino runs were
  about 12 times faster for REL, C2 and L3T (156-176 s against about 1,840-2,100 s).

## 4. What this means

- **For the owner's question** ("does an AI get smarter through these circuits?"):
  - With these fake devices, a small classifier learns as well through any of the four compilers as without
    noise, and it keeps its accuracy on clear-cut inputs.
  - On borderline inputs, noise does flip predictions. Fewer two-qubit gates and better placement flip fewer.
- **For PSF-Zero, two concrete targets, both measured:**
  1. **Routing of a 4-cycle on heavy-hex:** 44 two-qubit gates against Qiskit L3's 37 for 4 layers. This is about
     half of the remaining flip gap on FakeTorino.
  2. **Readout-aware placement of the measured qubit:** 0.048 against 0.011 readout error on FakeTorino.

## 5. What this does not establish

As in Addendum 296, section 4:

- nothing about real hardware;
- one model family;
- readout is modelled as a symmetric flip of one qubit;
- A5's advantages are favoured by construction.

## 6. Independent check (written after the run)

[`benchmarks/qml2_verify.py`](../../benchmarks/qml2_verify.py) does not import the scored script. It re-implements the model with dense 16×16
matrices and checks the following:

| check | result |
|---|---|
| provenance of all 97 files (commit `4839966`, script hash, versions) | no flags |
| Q2W data and IDEAL, rebuilt | 1.000, 0.625, 1.000, 1.000, 0.781, 0.906, 0.688, 1.000 |
| deep test sets A and B, rebuilt from teacher 84 and the stored θ* | logical z within 7.5e-16, labels identical, θ* accuracy 0.875 |
| every Q2W run | 1,328 compiles and 40 steps; noiseless accuracy of its result reproduced |
| P0 and H1-H6, from the raw rows | same verdicts as the locked score; output in `outputs/verify.txt` |

## 7. Data (`data/2026-10-02/qml_home2/`)

- **`outputs/`:**
  - `q1.json`: 1,152 Q1D rows, IDEAL, θ*;
  - `q2_<arm>_<device>_<seed>.json`: 96 files;
  - logs, `env.txt`, `run.log`, `score.md`, `score_log.txt`, `verify.txt`.
- **`dev/`:** the smoke run (not a result).
- **Redaction:** local paths in `env.txt` and the run log were replaced with `<repo>/` and `<home folder>/`. The
  release-file copies extracted from `9131cee` are not included; they are in git.


---

<!-- ===== Addendum 298 (source: spare-qubit-cliff-addendum-298-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of B17 (Qiskit #17057 in practice). Locked by the git commit that adds this Addendum and its two scripts, pushed before the scored run.

## Addendum 298 -- Pre-registration: does Qiskit issue #17057 bite in practice through transpile(), and does the PSF-Zero release stay exact on the same workloads? (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, [`benchmarks/b17_practice_eval.py`](../../benchmarks/b17_practice_eval.py) and
  [`benchmarks/run_b17_2026-10-02.sh`](../../benchmarks/run_b17_2026-10-02.sh), pushed before the scored run.
- **No hardware:** no IBM account, no QPU. Pure compilation and exact operator comparison.

## 1. Why

- **Addendum 294** located #17057 in the pulse-optimal 3-CX path that Qiskit's `TwoQubitBasisDecomposer` takes
  with CX and the ZSX Euler basis. It showed the bug on isolated two-qubit unitaries. Near the two-CX boundary it
  failed 122 of 300 times, with a worst 1 - F_avg of 0.36.
- **A maintainer judged such inputs probably rare.** The open question is how often it happens through ordinary
  `transpile()` to basis [cx, rz, sx, x], on workloads people actually compile.
- **The PSF-Zero release guards its own use of that decomposer** (psf_compile changelog item 17, Addendum 195).
  It has not been tested on such workloads end to end.

## 2. Design (`benchmarks/b17_practice_eval.py`)

**Workloads,** each at n = 4 and n = 6 qubits; every circuit depends only on its own seed:

| ID | workload | circuits per n |
|---|---|---|
| W1 | XYZ-Heisenberg Trotter chain, 4 steps (see below) | 7 × 3 × 50 = 1,050 |
| W2 | four near-boundary two-qubit unitaries on random qubit pairs, between layers of random single-qubit unitaries (see below) | 500 |
| W3 | the same with Haar-random two-qubit unitaries | 500 |
| W4 | hardware-efficient ansatz: 3 layers of ry(t) rz(t) and a cx ladder, t ~ N(0, s), s ∈ {1e-4, 1e-3, 1e-2} | 3 × 150 = 450 |

- **W1, one Trotter step:**
  - per bond, rxx(2 Jx dt) ryy(2 Jy dt) rzz(2 Jz dt), applied to even bonds, then odd bonds;
  - then rz(2 h_i dt) on each qubit.
- **W1 parameters:**
  - Jx, Jy ~ U[0.5, 1.5]; h_i ~ U[-1, 1];
  - Jz = r·Jx, with r ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1};
  - dt ∈ {1e-3, 1e-2, 0.1}.

  Small dt or a small anisotropy r gives blocks with a small third Weyl coordinate.
- **W2 blocks:** each is a canonical core with a, b ~ U[0, π/4] and c log-uniform in [1e-10, 1e-4], between random
  local unitaries. This is Addendum 294's near-boundary set, now inside circuits.

**Compilers.** The coupling map is a line of n qubits.

| ID | compiler | role |
|---|---|---|
| QK1, QK2, QK3 | `transpile(basis_gates=[cx, rz, sx, x], optimization_level=1/2/3, seed_transpiler=0)` | under test |
| QK3CZ | the same at level 3, basis [cz, rz, sx, x] | control |
| QK3U | the same at level 3, basis [cx, u] | control |
| PSF | release `compile_for_hardware(entangling_basis="cx", basis [cx, rz, sx, x], layout_search=True, seed_transpiler=0)` | under test |
| PSFNG | the same module loaded separately, with `USE_CX_GUARD = False` | positive control |

**Metric.**

- 1 - F_avg between `Operator(original)` and `Operator.from_circuit(compiled)`, which undoes the layout.
- A **failure** is 1 - F_avg > 1e-6, or a compile error.
- Also recorded:
  - the two-qubit count;
  - the compile time;
  - for PSF, how many times its guard rejected the ZSX decomposer.

## 3. Predictions (scored only by `b17_practice_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- all 8 chunks are present, with the full number of circuits;
- no compile errors;
- the two controls (QK3CZ, QK3U) have 0 failures on every workload.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | ordinary transpile reproduces the bug on near-boundary blocks | QK2 and QK3 fail on >= 10% of W2 | QK2 and QK3 have 0 W2 failures |
| H2 | generic blocks are safe | QK1-QK3 have 0 failures on W3 | any failure on W3 |
| H3 | the PSF-Zero release is exact on every workload | PSF: 0 failures in all four workloads | any failure |
| H4 | the guard is what protects it | PSFNG fails on >= 1% of W2 | PSFNG has 0 W2 failures |
| H5 | a realistic physics workload hits the bug | QK3 fails on >= 1 W1 circuit | 0 W1 failures |

**Reported without prediction:**

- QK1 on every workload; all compilers on W4;
- the W1 failure map by (dt, r);
- n = 4 against n = 6;
- the number of guard rejections;
- two-qubit counts and compile times.

**Expectations, stated now:**

- **H1.** Each W2 circuit has four near-boundary blocks. Addendum 294 found 122 of 300 isolated blocks failing,
  so most W2 circuits should fail. Whether `transpile`'s consolidation and resynthesis takes the failing path for
  every block is what H1 measures.
- **H5 is a genuine open question.** The failing band is narrow: c of about 2e-8 to 3e-7 at the reported a, b. A
  Trotter step has c = Jz·dt. With dt = 1e-3 and r = 1e-4, c is about 1e-7. Whether consolidation of several
  steps keeps c in the band is not known in advance.
- **QK1.** At level 1, `transpile` may not consolidate rxx/ryy/rzz into unitary blocks, but it must synthesize
  W2/W3's explicit unitaries. No prediction is made.

## 4. What this will not establish

- How common these workloads are in practice.
- Other bases. ecr and cz do not use the affected path; the cz control checks the latter.
- Wider circuits.
- That the guard catches every case. It catches every case in these workloads, if H3 holds.

## 5. Development (disclosed)

- **The scorer** was run on synthetic files to check the plumbing.
- **The smoke run at home** (`SMOKE=1`: 1 seed per W1 cell, 3 / 3 / 2 circuits for W2-W4), 2026-10-02. It ran
  end to end with no compile errors, and the controls (QK3CZ, QK3U) had 0 failures.
  - **Failures / circuits:**

    | workload | QK1 | QK2 | QK3 | PSF | PSFNG |
    |---|---|---|---|---|---|
    | W1 | 0/42 | 0/42 | 0/42 | 0/42 | **5/42** |
    | W2 | 4/6 | 4/6 | 4/6 | 0/6 | 4/6 |
    | W3, W4 | 0 | 0 | 0 | 0 | 0 |

  - **Notes:**
    - The largest exact-arm W1 deviation was 9.3e-8 (QK2, QK3, QK3CZ, QK3U alike), below the 1e-6 failure line.
    - The PSF guard rejected the ZSX decomposer 25 times.
    - PSF with the guard off failing on the physics workload while Qiskit does not was not anticipated. It is
      reported here and not turned into a prediction.
  - **Status:** its verdict lines (H5 REFUTED among them) come from 2 circuits per W1 cell and are not results.
  - **Nothing was changed after the smoke run:** design, sizes, threshold and both scripts are as before it.
- **No scored circuit was compiled before the lock.** Every scored circuit is drawn from seeds disjoint from the
  smoke run's.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/b17_practice_eval.py`](../../benchmarks/b17_practice_eval.py) | `04c0a80121170041768569737c7331cb7acbb8672a04729a977333f02e3a736a` |
| [`benchmarks/run_b17_2026-10-02.sh`](../../benchmarks/run_b17_2026-10-02.sh) | `2d6b383321c1559cc8dc84d97c7cecda8403509cfce95558059c01c8b588492f` |


---

<!-- ===== Addendum 299 (source: spare-qubit-cliff-addendum-299-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the home pre-registration in Addendum 298 (lock commit f0095e6). Scored by the locked script and re-counted by an independent script written after the run.

## Addendum 299 -- Results: B17 (Addendum 298). Qiskit #17057 hits explicit near-boundary unitaries at every optimization level (707 of 1,000 circuits), but no physics Trotter circuit through plain transpile (0 of 2,100). The PSF-Zero release is exact everywhere, and its guard is what makes it so: without it, 248 of the Trotter circuits fail, exactly in the c ~ 1e-8 to 1e-7 band (2026-10-02)

**Status: results of the pre-registered test in Addendum 298.**

- **Lock:** commit `f0095e6`, pushed before the scored run.
- **Scoring:** by the locked `b17_practice_eval.py score`, and re-counted by an independent script written after the
  run (section 5).
- **Setting:** home (WSL2), 6 processes; 5,000 circuits × 7 compilers, no compile errors.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 8 of 8 chunks complete; 0 compile errors; controls (QK3CZ, QK3U) 0 failures in 10,000 compiles |
| H1 | **CONFIRMED** | W2 (near-boundary unitaries): QK2 and QK3 fail on 707 of 1,000; worst 1 - F_avg 0.68 |
| H2 | **CONFIRMED** | W3 (Haar unitaries): QK1-QK3 0 of 1,000 |
| H3 | **CONFIRMED** | PSF release: 0 failures on all 5,000 circuits |
| H4 | **CONFIRMED** | PSF with the guard off: 599 of 1,000 W2 circuits fail |
| H5 | **REFUTED** | W1 (Heisenberg Trotter): QK3 0 of 2,100 |

## 2. Failures (1 - F_avg > 1e-6) / circuits

| workload | QK1 | QK2 | QK3 | QK3CZ | QK3U | PSF | PSFNG |
|---|---|---|---|---|---|---|---|
| W1 Trotter | 0/2100 | 0/2100 | 0/2100 | 0/2100 | 0/2100 | 0/2100 | **248/2100** |
| W2 near-boundary | **707/1000** | **707/1000** | **707/1000** | 0/1000 | 0/1000 | 0/1000 | **599/1000** |
| W3 Haar | 0/1000 | 0/1000 | 0/1000 | 0/1000 | 0/1000 | 0/1000 | 0/1000 |
| W4 small-angle ansatz | 0/900 | 0/900 | 0/900 | 0/900 | 0/900 | 0/900 | 0/900 |

**Notes.**

- QK1 fails exactly as often as QK2 and QK3 on W2: an explicit unitary is synthesised at every optimization level.
- On W1 every exact arm deviates by up to 8.9e-8, the cz and u controls included. That is the synthesis's own
  approximation tolerance, below the 1e-6 failure line.

## 3. Reading

- **Through plain `transpile`, the bug needs an explicit unitary near the boundary.**
  - With such unitaries (W2), 70.7% of circuits came out wrong at every optimization level, with errors up to
    0.68.
  - With a physics workload whose blocks come from rxx/ryy/rzz gates (W1), none of 2,100 circuits failed. This
    includes the cells whose per-step third coordinate lies in the failing band (below).
  - The prediction that a realistic workload would hit it is refuted. This supports the maintainer's view that
    such inputs are rare in typical use, with the caveat that explicit two-qubit unitaries near the boundary (from
    numerical optimization, or from block consolidation in another tool) fail most of the time.
- **PSF-Zero's own pipeline does reach the band on the physics workload, and the guard catches every case.**
  - Without the guard (PSFNG), 248 W1 circuits fail. They sit exactly in the cells where one Trotter step's
    third coordinate c = r·Jx·dt is about 1e-8 to 1e-7.

    | dt | r = 1e-5 | r = 1e-4 | every other r |
    |---|---|---|---|
    | 1e-3 | 48 of 100 | 100 of 100 | 0 |
    | 1e-2 | 100 of 100 | 0 | 0 |
    | 0.1 | 0 | 0 | 0 |

  - The guarded release rejected the ZSX decomposer 1,841 times: 979 in W1 and 862 in W2.
  - In all 847 circuits where PSFNG failed, the guarded run had at least one rejection. With the guard, the output
    was exact in every case.
  - PSF-Zero emits each block's canonical core on its own, so it meets the narrow band whenever one step's
    coordinate lies in it. Qiskit's consolidation evidently does not produce such blocks from these gates.
- **For the Qiskit issue.** These are facts the owner may choose to add:
  - explicit near-boundary unitaries fail at every optimization level, at 70.7% in this sample;
  - a Heisenberg Trotter workload through `transpile` did not fail in 2,100 circuits;
  - a pipeline that synthesizes per-step blocks does reach the band.

  As with every upstream post, the owner decides whether and how.

## 4. What this does not establish

As in Addendum 298:

- **How common such workloads are.**
- **The guard's completeness beyond these workloads.**
- **Why Qiskit's consolidation avoids the band on W1.** That was not investigated; the observation is only that it
  did.

## 5. Independent check

[`benchmarks/b17_verify.py`](../../benchmarks/b17_verify.py) (written after the run) reads the raw jsonl files only. It checks the following, and all
of it matches the locked score:

- the commit `f0095e6`, the script hash, the release version and the circuit counts in all 8 files;
- every count in section 2;
- P0 and H1-H5;
- the maps in section 3.

Output in `outputs/verify.txt`.

## 6. Data (`data/2026-10-02/b17/`)

- `outputs/`: the 8 jsonl files (one row per circuit, all 7 compilers), the logs, `env.txt`, `run.log`,
  `score.md`, `score_log.txt` and `verify.txt`.
- `dev/`: the smoke run.
- Local paths in `env.txt` and the run log were replaced with `<repo>/` and `<home folder>/`.


---

<!-- ===== Addendum 300 (source: spare-qubit-cliff-addendum-300-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of GAP. Locked by the git commit that adds this Addendum and its two scripts, pushed before the scored run. H6 was added after the smoke run and is marked as such.

## Addendum 300 -- Pre-registration: the gap map. Where the PSF-Zero release trails, ties or beats error-aware Qiskit L3 and A5, by circuit family, on three noisy fake devices (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, [`benchmarks/gap_eval.py`](../../benchmarks/gap_eval.py) and [`benchmarks/run_gap_2026-10-02.sh`](../../benchmarks/run_gap_2026-10-02.sh),
  pushed before the scored run.
- **No hardware:** no IBM account, no QPU. Fake-provider devices and Aer noise models only.

## 1. Why

Addendum 297 found two concrete targets for the release (C2) on one circuit family, the QML ring ansatz:

1. **Routing of a 4-cycle on heavy-hex.** C2 used 44 two-qubit gates against error-aware Qiskit L3's (L3T) 37 for
   4 layers, and lost more margin to noise.
2. **Readout-blind placement of the measured qubit.**

Before working on them, this test measures how general the gap is. Does it appear on every circuit that needs a
cycle? Does it disappear on chains? And how does C2 fare on dense random circuits, where its synthesis rather than
its routing decides?

## 2. Design (`benchmarks/gap_eval.py`)

**Families.** Inputs are |0…0⟩, and every circuit depends only on its own seed.

| ID | family | instances |
|---|---|---|
| F1 | ring ansatz (the QML family, CZ ring) | n = 4 and 6; L = 2, 4, 6; 36 per (n, L): 216 |
| F2 | QAOA MaxCut on random 3-regular graphs, random angles | n = 6; p = 1, 2; 60 per p: 120 |
| F3o | XYZ-Heisenberg Trotter chain, open boundary, dt = 0.1, 4 steps | n = 6: 75 |
| F3p | the same with a periodic boundary (a 6-cycle) | n = 6: 75 |
| F4 | quantum-volume-style layers of Haar SU(4) on random pairs, depth n | n = 4, 5, 6; 45 per n: 135 |
| F5 | GHZ chain plus a random single-qubit layer | n = 4, 6, 8; 24 per n: 72 |

**Arms, devices and noise.**

- Arms: C2, A5 and L3T, as in Addenda 290-297. REL is dropped: Addendum 297 placed it below C2 throughout.
- Devices: FakeAuckland, FakeTorino, FakeKingston.
- Noise: `NoiseModel.from_backend` with Aer `density_matrix`. One Aer job per (device, arm, family).

**Metric.**

- Infidelity 1 - ⟨ψ|ρ|ψ⟩ of the final-layout qubits' noisy state against the ideal output ψ.
- Also recorded:
  - the two-qubit count and depth;
  - the number of physical qubits touched;
  - the mean readout error of the final-layout qubits;
  - the compile time.
- A compiled circuit that touches more than 11 physical qubits is not simulated (the density matrix would be too
  large) and is counted as "too wide".

**Comparisons are paired:** the same circuits in every arm.

## 3. Predictions (scored only by `gap_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- all 45 job files are present;
- every simulated circuit's noiseless infidelity is <= 1e-9;
- at most 5% of the circuits are too wide.

Two subsets are used: **cycles** = F1 + F3p, and **chains** = F3o + F5.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | cycles cost C2 extra two-qubit gates on Heron | C2 uses more 2q gates than L3T in >= 80% of cycle circuits, on FakeTorino and FakeKingston | <= 50% on either |
| H2 | chains do not | C2's 2q count <= L3T's in >= 90% of chain circuits, on every device | < 70% on any device |
| H3 | the extra gates cost fidelity | cycles on Heron: mean infidelity C2 >= 1.10 × L3T, on both devices | C2 <= L3T on either |
| H4 | on dense random circuits C2 is level with L3T | F4: mean infidelity C2 <= 1.05 × L3T, on every device | > 1.20 × on any device |
| H5 | A5 at least matches L3T overall | mean infidelity over all families A5 <= L3T on >= 2 of 3 devices | A5 > 1.05 × L3T on >= 2 |
| H6 | **added after the smoke run (section 5):** C2 trails L3T on chains too, where the 2q counts are equal, so placement rather than routing | chains: mean infidelity C2 >= 1.10 × L3T, on every device | C2 <= L3T on any device |

**Reported without prediction:**

- F2 (QAOA);
- FakeAuckland for H1 and H3;
- depths;
- readout error of the final-layout qubits by arm;
- compile times;
- qubits touched.

**Expectations, stated now:**

- H1 and H3 generalise Addendum 297's single family.
- H2 is the control: without cycles there should be no routing gap.
- H4 is genuinely uncertain. C2's synthesis was competitive in earlier tests, but F4 also needs routing.
- H5 favours A5 by construction, as in Addenda 286-297.

## 4. What this will not establish

- Real hardware.
- Circuits wider than 8 logical qubits.
- That the gaps found are the only ones.
- Readout error does not enter the metric, because the state is read from the density matrix. It is reported
  separately.

## 5. Development (disclosed)

- **The scorer** was run on synthetic files to check the plumbing.
- **The smoke run at home** (`SMOKE=1`: 1 circuit per sub-family; 144 circuits per arm and device in all), 2026-10-02.
  - It ran end to end. P0 PASS: 45 of 45 files, noiseless infidelity at most 4.8e-15, none too wide.
  - Summed job time 122 s.
- **Two changes after the smoke run, before any scored circuit was compiled:**
  1. **The sizes were tripled** (from 12, 20, 25, 15, 8 to 36, 60, 75, 45, 24 per sub-family), because the run
     proved light. This changes precision, not which circuits are drawn first. The smoke seeds stay disjoint from the
     scored ones.
  2. **H6 was added, informed by the smoke run.**
     - In the smoke run, C2's infidelity was 1.2-1.5 times L3T's on the chain families (F3o, F5) at equal
       two-qubit counts.
     - That points at placement: C2's layout search, called as here without `layout_edge_errors`, ignores gate and
       readout errors, while L3T's does not.
     - The smoke run is 1 circuit per sub-family, so this is a hypothesis formed on 3-6 circuits per device. H6 is
       marked as post-smoke and carries less weight than H1-H5.
- **The smoke run's verdict lines are not results.** They read H1 REFUTED, H2 CONFIRMED, H3 CONFIRMED, H4 REFUTED
  and H5 CONFIRMED. H1-H5 were not changed after seeing them.
- **No scored circuit was compiled before the lock.** Scored and smoke circuits use disjoint seeds.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/gap_eval.py`](../../benchmarks/gap_eval.py) | `6659bd0bd2054d4484374da393a1f84fa6138607babc019b8eb91effa2a2ff2a` |
| [`benchmarks/run_gap_2026-10-02.sh`](../../benchmarks/run_gap_2026-10-02.sh) | `4185e0c983a3cfaf99e24a66d32bbcebe403106db790fa62d53a145506fe5b67` |


---

<!-- ===== Addendum 301 (source: spare-qubit-cliff-addendum-301-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the home pre-registration in Addendum 300 (lock commit 4acca8c). P0 failed on a too-strict noiseless threshold (9 of 6,237 rows, all Qiskit L3T approximations below 1.9e-8), so nothing is scored; the results are reported descriptively and re-checked by an independent script.

## Addendum 301 -- Results: GAP (Addendum 300). P0 FAILED on a too-strict harness threshold, so nothing is scored. Descriptively, the release trails error-aware Qiskit L3 almost everywhere: it has lower infidelity in only 0.3-12% of 2,079 circuits per device. On chains with identical two-qubit counts it is 1.25-1.47 times worse, which points at error-blind placement (2026-10-02)

**Status: results of the pre-registered test in Addendum 300.**

- **Lock:** commit `4acca8c`, pushed before the scored run.
- **Setting:** home (WSL2), 6 processes; 45 jobs, 6,237 circuit compilations in 377 s; none too wide.

## 1. P0 failed: no prediction is scored

- **What failed.** P0 required every simulated circuit's noiseless infidelity to be <= 1e-9.
  - 9 of 6,237 rows exceeded it, with values from 1.4e-9 to 1.9e-8.
  - All 9 are L3T, on the same three open-chain Heisenberg circuits (F3o seeds 13, 27, 45), on all three devices.
- **The cause is the threshold, not the harness.** Qiskit's level-3 transpile with a Target approximates two-qubit
  synthesis when the Target's error rates make an approximation worthwhile. The 1e-9 threshold did not allow for
  that, whereas B17 used 1e-6. This is a design error in Addendum 300.
- **What follows from the pre-registration.** As pre-registered, nothing below P0 is scored. The verdict lines that
  the locked score printed after its P0 FAIL line are not results:

  | H1 | H2 | H3 | H4 | H5 | H6 |
  |---|---|---|---|---|---|
  | REFUTED | CONFIRMED | CONFIRMED | REFUTED | CONFIRMED | CONFIRMED |

- **The quantities are reported descriptively (section 2).** These 9 rows change L3T's mean infidelity by less
  than 1e-8 against values of 0.01-0.5, so the description is not affected. It is still not a scored test.
- **Lesson for future harnesses.** A noiseless-equivalence check must use a tolerance that admits each arm's
  intended approximation (1e-6, as in B17), or must pass `approximation_degree=1.0` to the arms that approximate.

## 2. Descriptive results

Mean noisy infidelity / mean two-qubit count, and C2/L3T paired infidelity ratio:

| family | device | C2 | A5 | L3T | C2/L3T | share of circuits where C2 uses more 2q gates than L3T |
|---|---|---|---|---|---|---|
| F1 ring ansatz | Auckland | 0.392 / 53.5 | 0.335 / 53.5 | 0.339 / 53.8 | 1.16 | 0.17 |
| | Torino | 0.581 / 59.2 | 0.177 / 53.7 | 0.179 / 53.7 | **3.25** | 1.00 |
| | Kingston | 0.097 / 57.3 | 0.086 / 53.5 | 0.087 / 53.8 | 1.12 | 0.50 |
| F2 QAOA | Auckland | 0.366 / 47.2 | 0.304 / 44.5 | 0.319 / 45.4 | 1.15 | 0.57 |
| | Torino | 0.353 / 48.4 | 0.168 / 45.5 | 0.172 / 46.4 | **2.06** | 0.40 |
| | Kingston | 0.088 / 48.1 | 0.074 / 45.1 | 0.076 / 45.8 | 1.15 | 0.55 |
| F3o open chain | Auckland | 0.465 / 60.0 | 0.375 / 60.0 | 0.337 / 60.0 | 1.38 | 0.00 |
| | Torino | 0.241 / 60.0 | 0.193 / 60.0 | 0.194 / 60.0 | 1.24 | 0.00 |
| | Kingston | 0.142 / 60.0 | 0.096 / 60.0 | 0.098 / 60.0 | 1.46 | 0.00 |
| F3p 6-cycle | Auckland | 0.583 / 114.0 | 0.571 / 114.0 | 0.465 / 114.0 | 1.25 | 0.00 |
| | Torino | 0.337 / 114.0 | 0.323 / 114.0 | 0.321 / 114.0 | 1.05 | 0.00 |
| | Kingston | 0.206 / 114.0 | 0.167 / 114.0 | 0.167 / 114.0 | 1.23 | 0.00 |
| F4 random SU(4) | Auckland | 0.314 / 42.2 | 0.285 / 41.0 | 0.305 / 41.7 | 1.03 | 0.21 |
| | Torino | 0.302 / 42.2 | 0.153 / 41.1 | 0.161 / 42.0 | **1.87** | 0.15 |
| | Kingston | 0.095 / 42.5 | 0.076 / 41.1 | 0.079 / 41.7 | 1.20 | 0.21 |
| F5 GHZ chain | Auckland | 0.063 / 5.0 | 0.047 / 5.0 | 0.051 / 5.0 | 1.24 | 0.00 |
| | Torino | 0.030 / 5.0 | 0.023 / 5.0 | 0.023 / 5.0 | 1.30 | 0.00 |
| | Kingston | 0.018 / 5.0 | 0.012 / 5.0 | 0.012 / 5.0 | 1.54 | 0.00 |

**The quantities named in H1-H6, descriptive only:**

| quantity | FakeAuckland | FakeTorino | FakeKingston |
|---|---|---|---|
| cycles: share of circuits where C2 uses more 2q gates than L3T (H1) | – | 0.74 | 0.37 |
| chains: share where C2's 2q count <= L3T's (H2) | 1.00 | 1.00 | 1.00 |
| cycles: C2/L3T infidelity (H3) | – | 2.41 | 1.16 |
| F4: C2/L3T infidelity (H4) | 1.03 | 1.87 | 1.20 |
| all families: A5/L3T infidelity (H5) | 1.02 | 0.98 | 0.98 |
| chains: C2/L3T infidelity (H6) | 1.36 | 1.25 | 1.47 |

**C2 has lower infidelity than L3T** in 7.8% (FakeAuckland), 0.3% (FakeTorino) and 11.7% (FakeKingston) of
circuits.

**Mean readout error of the final-layout qubits** (not part of the metric):

| arm | FakeAuckland | FakeTorino | FakeKingston |
|---|---|---|---|
| C2 | 0.008 | 0.055 | 0.029 |
| A5 | 0.008 | 0.038 | 0.039 |
| L3T | 0.008 | 0.038 | 0.039 |

**Median compile time:** C2 0.025 s, A5 0.66 s, L3T 0.015 s.

## 3. Reading (exploratory)

- **The release trails error-aware Qiskit L3 broadly, not just on cycles.**
  - On every family and device, C2's mean infidelity is 1.03-3.25 times L3T's.
  - It is lower than L3T's in under 12% of circuits.
  - This is a sharper statement than Addendum 297, which saw the gap on one family.
- **Placement, more than routing, is the main cause.**
  - On the chain families, C2 and L3T use the same number of two-qubit gates (60 and 5 on every device), yet C2's
    infidelity is 1.24-1.54 times L3T's.
  - On cycles, C2 used more two-qubit gates than L3T in only 37% of circuits on FakeKingston and 74% on
    FakeTorino. Its infidelity is still 1.16 and 2.41 times L3T's.
  - What C2 and L3T differ in, beyond gate counts, is where they put the circuit. C2's layout search, called as here
    without `layout_edge_errors`, uses no error information. L3T scores layouts by the Target's errors.
  - Readout plays no part in this metric. So the gap comes from gate errors and decoherence on the qubits chosen,
    not from readout.
- **FakeTorino is the outlier.** C2 is 3.25 times worse on F1, 2.06 on F2 and 1.87 on F4. This suggests that
  C2's error-blind layout lands on one or more very poor qubits or couplers of that snapshot. It has not been
  examined which ones; it is the first thing to check (section 5).
- **A5 is level with L3T** (0.98-1.02). A5 is the AI front end that chooses among candidate compilations by an
  error estimate built from the Target. That is consistent with the reading that the release's deficit lies in using
  no error information. A5's estimate shares its
  physics with the simulator, so this support is limited, as noted in Addenda 286-297.
- **Routing still matters on cycles.** Addendum 297's 44-against-37 count on the 4-layer ring reappears: on F1,
  FakeTorino, C2 used more two-qubit gates in every circuit. But it is the smaller of the two effects.

## 4. What this does not establish

- Real hardware.
- The cause of the FakeTorino outlier.
- That error-aware placement would close the gap. That is a prediction for a next test.

## 5. Next steps (proposed, not run)

1. **Diagnose FakeTorino.** List the physical qubits and couplers C2 uses for F1 on FakeTorino, with their Target
   errors, T1 and T2.
2. **Error-aware layout for the release.** PSF-Zero already has an opt-in `layout_edge_errors` (changelog item 16,
   matching layouts only). Extend error weighting to the general layout search, then re-run this map with the P0
   threshold fixed at 1e-6, pre-registered.

## 6. Independent check

[`benchmarks/gap_verify.py`](../../benchmarks/gap_verify.py) (written after the run) reads the raw json only. It confirms:

- the provenance and circuit counts of all 45 files (commit `4acca8c`, script hash `6659bd0b…`);
- the 9 P0 rows above, all L3T F3o seeds 13, 27 and 45, maximum 1.85e-8, all below 1e-6;
- every quantity in section 2.

Output in `outputs/verify.txt`.

## 7. Data (`data/2026-10-02/gap/`)

- `outputs/`: 45 job files (one row per circuit), logs, `env.txt`, `run.log`, `score.md` (locked score, including
  its P0 FAIL), `score_log.txt` and `verify.txt`.
- `dev/`: the smoke run.
- Local paths in `env.txt` and the run log were replaced.


---

<!-- ===== Addendum 302 (source: spare-qubit-cliff-addendum-302-2026-10-02.md) ===== -->

> **Note added when merging:** Exploratory diagnosis of the FakeTorino outlier of Addendum 301; not a test.

## Addendum 302 -- Diagnosis: on FakeTorino the release places a 6-qubit ring across a failed coupler (reported error 1.0), 7-14 times per circuit (2026-10-02)

**Status: exploratory diagnosis, not a test.** It explains the FakeTorino outlier of Addendum 301. The script
([`data/2026-10-02/c3/diag/torino_diag.py`](../../data/2026-10-02/c3/diag/torino_diag.py)) and its output, run at home at commit `bc2c2f9`, are in the data folder.

## 1. What was looked at

For the first circuit of every (n, L) cell of GAP family F1 (same seeds as the scored run), the script lists the
following for the release (C2) and for error-aware Qiskit L3 (L3T):

- the physical qubits used;
- the couplers used, with their reported CZ error;
- the worst sx error, and the shortest T1 and T2;
- the readout error of the final-layout qubits.

## 2. Findings

**FakeTorino's coupling map (from `Target.build_coupling_map()`) still contains failed couplers.**

- 300 CZ entries; median error 0.0042; 26 with error >= 0.05.
- 22 directed entries, on 11 couplers, report an error of exactly 1.0. These are the couplers around qubits 19, 58,
  86 and 97, plus (21, 34).
- From the A0 data (Addendum 293):
  - FakeKingston has 7 such couplers, and 5 qubits whose sx error is 1.0;
  - FakeAuckland has none.

**For the 6-qubit ring, C2 uses one of them.** It chooses qubits 0, 1, 2, 3, 15, 19, and the ring crosses (15, 19),
whose error is 1.0:

| L | uses of (15, 19) | 2q gates (C2 / L3T) |
|---|---|---|
| 2 | 7 | 36 / 33 |
| 4 | 12 | 74 / 70 |
| 6 | 14 | 113 / 108 |

- Each use applies a fully depolarizing two-qubit channel in the noise model.
- The same region also has the readout errors 0.167 and 0.257, and a qubit with T2 = 36 µs.
- L3T chooses regions whose worst coupler is 0.0037-0.0042 and whose shortest T2 is >= 126 µs.

**For the 4-qubit ring, C2 does not touch a failed coupler.**

- Its couplers (0.0037-0.0049) are a little worse than L3T's (0.0034-0.0035).
- It routes the ring with more gates: 20 / 44 / 68 against 17 / 37 / 57. That is the routing gap of Addendum 297.

**Why.**

- `compile_for_hardware` sees only `coupling_map`. Its layout search (`smart_vf2_layout`) and the routing treat
  every listed edge as usable.
- The layout search returns an embedding without regard to errors. For the 6-qubit ring that is qubits 0-3, 15
  and 19; for the 4-qubit ring it is 114-116 and 129, or 0-3.
- `layout_edge_errors` (changelog item 16) would weight edges, but it is opt-in and applies only to matching
  layouts. A ring is not a matching.

## 3. Consequence

- Through `compile_for_hardware`, the release can place gates on a coupler the device reports as failed. This is
  not limited to fake snapshots: a live Target can list a failed coupler with error 1.0, and the coupling map built
  from it includes that edge.
- It explains why FakeTorino, with 11 failed couplers, was the outlier in Addendum 301 (C2/L3T 3.25 on F1).
- FakeKingston (7 failed couplers, 5 failed qubits) was affected less in Addendum 301. FakeAuckland (none) was
  not affected.

## 4. Next

**Candidate 2026-10-02.c3 (Addendum 303):**

- **What it adds:** an opt-in `target=` argument to `compile_for_hardware`. It removes from the coupling map every
  edge whose reported 2-qubit error is >= 0.5, and every edge touching a qubit whose sx error is >= 0.5, before the
  layout search and the routing.
- **What it does not do:** weight the remaining errors. That is the larger, separate step, and Addendum 301's chain
  results say it is needed too.
- **How it is tested:** with a pre-registration, on the GAP circuits.


---

<!-- ===== Addendum 303 (source: spare-qubit-cliff-addendum-303-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of candidate psf_compile 2026-10-02.c3. Locked by the git commit that adds this Addendum, the candidate with its tests, and the evaluation scripts, pushed before the scored run. The first design was changed after smoke run 1, before the lock; both are disclosed in section 5.

## Addendum 303 -- Pre-registration: candidate psf_compile 2026-10-02.c3 (avoidance of failed couplers and qubits). Does it remove the FakeTorino outlier without changing anything else? (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c3_2026-10-02/psf_compile.py`](../../patches/psf_compile_c3_2026-10-02/psf_compile.py), with its tests) and [`benchmarks/c3_eval.py`](../../benchmarks/c3_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake-provider devices and Aer noise only.
- **Not a release:** the candidate becomes one only by a separate adoption decision.

## 1. The candidate (changelog item 31)

- **The new argument:** `compile_for_hardware(..., target=None, prune_max_error=0.5)`.
- **What it does with `target` given:**
  1. The circuit is compiled exactly as without it.
  2. If the result uses no failed element, it is returned unchanged. A failed element is a 2-qubit gate on a
     directed edge whose native 2-qubit gate error is >= 0.5, or any gate on a qubit whose sx error is >= 0.5.
  3. Otherwise it is compiled again on `prune_coupling_map(...)`: the coupling map without the failed edges and
     without every edge that touches a failed qubit. Qubit indices and `size()` are unchanged.
- **Without `target`, nothing changes.** A test checks this gate for gate against the release.
- **Motivation:** Addendum 302. On FakeTorino the release crosses a coupler with error 1.0 up to 14 times per
  circuit.
- **What it is not:** error weighting.

## 2. Design (`benchmarks/c3_eval.py`)

**Circuits:** the five GAP families (Addendum 300), generated by the locked [`benchmarks/gap_eval.py`](../../benchmarks/gap_eval.py) with the same
seeds and sizes: 2,079 circuits per device and arm.

**Arms:**

| arm | what it is |
|---|---|
| C2 | the release, called exactly as in GAP |
| C3 | the candidate, the same call plus `target=<device Target>` |
| L3T | Qiskit level 3 with the Target, now with `approximation_degree=1.0` so its noiseless output is exact |

**Devices and metric:**

- FakeAuckland, FakeTorino, FakeKingston.
- Metric as in GAP: infidelity of the final-layout qubits' state under `NoiseModel.from_backend`.
- Also recorded per circuit: 2-qubit gates on failed edges, gates on failed qubits, and whether C3 recompiled.

**Known before the lock (published A0 data, Addendum 293):**

| device | failed couplers (error 1.0) | failed qubits (sx error 1.0) |
|---|---|---|
| FakeAuckland | 0 | 0 |
| FakeTorino | 11 | 0 |
| FakeKingston | 7 | 5 |

## 3. Predictions (scored only by `c3_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- all 45 job files are present;
- every simulated circuit's noiseless infidelity is <= 1e-6;
- at most 5% of the circuits are too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | C3 never uses a failed element | 0 failed-edge or failed-qubit uses by C3, over all circuits and devices | any use |
| H2 | the FakeTorino outlier is gone | FakeTorino C3/L3T mean infidelity <= 1.30 on each of F1, F2, F4 (C2 had 3.25, 2.06, 1.87 in GAP) | any of them >= 1.80 |
| H3 | nothing else changes | on every circuit where C2 used no failed element, C3 has the same 2q count and the same infidelity (to 1e-12) | any difference |
| H4 | recompiling helps where it happens | on the circuits where C2 used a failed element, C3 has lower infidelity in >= 90% | < 70% |
| H5 | the placement gap remains; this is not error weighting | chains (F3o + F5): C3/L3T >= 1.10 on every device | <= 1.0 on any device |

**Reported without prediction:**

- the family × device table;
- failed-element uses of C2 and L3T;
- the number of recompiles;
- compile times.

**Expectations, stated now (after the smoke runs below):**

- **H1 and H3 test the plumbing.**
- **H2 and H4 are at risk.**
  - A recompile is an error-blind re-draw of the layout. It avoids the failed elements but can land on a region that
    is merely poor.
  - In smoke run 2 the FakeTorino F4 circuit, after recompiling, was still 2.61 times L3T.
  - H2 was not changed after seeing that.

## 4. What this will not establish

- Real hardware.
- That 0.5 is the best threshold.
- Anything about error weighting.

## 5. Development (disclosed)

### 5.1 First design and smoke run 1: rejected

The first design pruned the coupling map before every compile. In smoke run 1 (1 circuit per sub-family) it removed
every failed-edge use, but it also changed circuits that had never touched a failed element:

| family | device | C3/C2 infidelity (failed-edge uses by C2) |
|---|---|---|
| F1 | FakeKingston | 5.66 (0) |
| F2 | FakeKingston | 3.84 (0) |
| F3p | FakeKingston | 3.99 (0) |
| F3p | FakeTorino | 2.56 (0) |
| F4 | FakeKingston | 2.96 (0) |
| F4 | FakeTorino | 1.16 (15) |
| F1 | FakeTorino | 0.76 (33) |

- **Cause:** removing edges changes the order in which the error-blind layout search meets candidates, so unaffected
  circuits were placed elsewhere, sometimes on poor regions.
- **Changes made before any scored circuit was compiled:**
  - the design was changed to recompile only when the first result uses a failed element (section 1);
  - H3 and H4 replaced the first draft's "C3/C2 <= 1.02" and "FakeAuckland identical" predictions;
  - H1 now also counts failed qubits.

### 5.2 Tests

`test_c3_prune.py`, 5 tests, all pass at home:

- the version string;
- exact pruning on all three devices;
- FakeTorino rings avoid failed edges and stay exact (checked on the touched qubits only);
- unaffected outputs are identical with `target`;
- without `target`, the output is identical to the release.

The first version of the test built a full 133-qubit operator and failed for that reason. It was fixed before smoke
run 2.

### 5.3 Smoke run 2 (the locked design; not a result)

- P0 passed.
- Its verdict lines:

  | H1 | H2 | H3 | H4 | H5 |
  |---|---|---|---|---|
  | CONFIRMED | REFUTED | CONFIRMED (44 of 44) | CONFIRMED (4 of 4) | CONFIRMED |

  H2 read F1 1.51, F2 1.19, F4 2.61.
- These come from 1 circuit per sub-family and are not results.

### 5.4 Other development

- The scorer was run on synthetic files.
- Smoke and scored circuits use disjoint seeds.
- No scored circuit was compiled before the lock.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c3_2026-10-02/psf_compile.py`](../../patches/psf_compile_c3_2026-10-02/psf_compile.py) | `8fd5e50087c1aa6abd382c8a6852d3a015d9e07fdb66ef22a7b5bd91360fd36d` |
| [`patches/psf_compile_c3_2026-10-02/test_c3_prune.py`](../../patches/psf_compile_c3_2026-10-02/test_c3_prune.py) | `55276d3e3f60137801e6f9c898224c35a23d5b5bd4613cbdda9ffb5e6670a450` |
| [`benchmarks/c3_eval.py`](../../benchmarks/c3_eval.py) | `a56aa67676142fe8d254c67d91f120e854ed0c64b673b5fe3e750b22941196f4` |
| [`benchmarks/run_c3_2026-10-02.sh`](../../benchmarks/run_c3_2026-10-02.sh) | `7c82f6126e66ad15c7e6c3171a7de7fb5c8b503240dfe01044367f8ed6ba990d` |


---

<!-- ===== Addendum 304 (source: spare-qubit-cliff-addendum-304-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the home pre-registration in Addendum 303 (lock commit 158967c). Scored by the locked script and re-checked by an independent script written after the run. Section 4 corrects the explanation given in Addendum 301 for its P0 failure.

## Addendum 304 -- Results: candidate psf_compile 2026-10-02.c3 (Addendum 303). Four of five confirmed, H2 ambiguous: c3 never touches a failed element, leaves 1,926 of 1,926 unaffected circuits bit-for-bit unchanged, and improves all 153 affected ones (mean infidelity 0.956 → 0.375); the FakeTorino gap to Qiskit L3 halves but does not close. Correction to Addendum 301 (2026-10-02)

**Status: results of the pre-registered test in Addendum 303.**

- **Lock:** commit `158967c`, pushed before the scored run.
- **Scoring:** by the locked `c3_eval.py score`, and re-checked by an independent script written after the run
  (section 5).
- **Setting:** home (WSL2), 6 processes; 45 jobs, 6,237 circuit compilations, 117 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 45 of 45 files; noiseless infidelity max 1.9e-8 (<= 1e-6); none too wide |
| H1 | **CONFIRMED** | C3: 0 failed-edge and 0 failed-qubit uses (C2: 1,632 failed-edge uses on FakeTorino) |
| H2 | **AMBIGUOUS** | FakeTorino C3/L3T: F1 1.511, F2 1.189, F4 1.696 (CONFIRMED needed <= 1.30 on all three; REFUTED needed any >= 1.80) |
| H3 | **CONFIRMED** | 1,926 of 1,926 circuits where C2 used no failed element: C3 identical (2q count and infidelity to 1e-12) |
| H4 | **CONFIRMED** | 153 of 153 circuits where C2 used a failed element: C3 lower infidelity |
| H5 | **CONFIRMED** | chains C3/L3T: Auckland 1.360, Torino 1.246, Kingston 1.467 |

## 2. Numbers

**FakeTorino** is the only device where C2 used failed elements, and where C3 recompiled:

| family | C2/L3T | C3/L3T | C3/C2 | C2 failed-edge uses |
|---|---|---|---|---|
| F1 ring ansatz | 3.252 | 1.511 | 0.465 | 1,188 |
| F2 QAOA | 2.058 | 1.189 | 0.578 | 183 |
| F4 random SU(4) | 1.872 | 1.696 | 0.906 | 261 |
| F3o, F3p, F5 | unchanged | unchanged | 1.000 | 0 |

**The 153 recompiled circuits** (all on FakeTorino): mean infidelity C2 0.956, C3 0.375, L3T 0.238.

**FakeAuckland and FakeKingston:**

- C3/C2 = 1.000 in every family.
- C2 used none of FakeKingston's 7 failed couplers or 5 failed qubits on these circuits, so nothing was
  recompiled there.

## 3. Reading

- **c3 does exactly what it was built for, and nothing else.**
  - Failed couplers are never used.
  - Every circuit that did not touch one is unchanged, bit for bit (1,926 of 1,926).
  - Every circuit that did touch one improves (153 of 153). Their mean infidelity falls from 0.956, which is
    essentially a fully depolarized output, to 0.375.
- **It does not close the FakeTorino gap.**
  - C3/L3T on FakeTorino is 1.19-1.70 against 1.86-3.25 before. That is about halved, not brought to the
    1.30 that H2 required.
  - On the recompiled circuits C3 is still 1.6 times L3T (0.375 against 0.238): the recompiled layout avoids the
    failed coupler but is otherwise chosen without error information.
  - This is the risk Addendum 303 stated before the run.
- **The rest of the gap is the error-blind layout (H5).** On chains, where both use the same two-qubit gates,
  C3 is 1.25-1.47 times L3T on every device, unchanged from Addendum 301.
- **Adoption.** c3 is safe to adopt as an opt-in. It changes nothing without `target`, and nothing with `target`
  unless a failed element would otherwise be used. Adoption is the owner's decision; this test supports it. It
  is not the answer to the release's deficit. That needs error-weighted layout.

## 4. Correction to Addendum 301

Addendum 301 explained its P0 failure (9 L3T rows with noiseless infidelity up to 1.9e-8) as Qiskit approximating
two-qubit synthesis from the Target's error rates. This run disproves that: L3T was called here with
`approximation_degree=1.0`, which disables that approximation, and the same 9 rows (F3o seeds 13, 27, 45 on all
three devices) still show the same values, up to 1.85e-8.

The deviation therefore comes from somewhere inside Qiskit's exact path. The likely source is the default
fidelity tolerance (1 - 1e-9 per block) with which two-qubit Weyl decompositions are specialized, accumulated over
the 60 two-qubit gates of these circuits. This was not investigated further. The rest of Addendum 301 stands: the
threshold of 1e-9 was too strict, and 1e-6 (used here) is adequate.

## 5. Independent check

[`benchmarks/c3_verify.py`](../../benchmarks/c3_verify.py) (written after the run) reads the raw json only. It checks the following, and all of it
matches the locked score:

- the commit `158967c`, the script and candidate hashes, the candidate version and the counts in all 45 files;
- P0 and the 9 rows behind its maximum;
- H1-H5;
- the recompile counts per device (Auckland 0, Torino 153, Kingston 0);
- the recompiled circuits' means.

Output in `outputs/verify.txt`.

## 6. Data (`data/2026-10-02/c3/`)

- `outputs/`: 45 job files, logs, `env.txt`, `run.log`, `score.md`, `score_log.txt`, `verify.txt`.
- `dev/smoke1/`: the first design's smoke run (Addendum 303, section 5.1).
- `dev/smoke2/`: the locked design's smoke run.
- `diag/`: from Addendum 302.
- Local paths were replaced.


---

<!-- ===== Addendum 305 (source: spare-qubit-cliff-addendum-305-2026-10-02.md) ===== -->

> **Note added when merging:** Adoption record: candidate 2026-10-02.c3 becomes release psf_compile 2026-10-02.1 (owner's decision, 2026-10-02).

## Addendum 305 -- Adoption: candidate 2026-10-02.c3 becomes release psf_compile 2026-10-02.1 (2026-10-02)

**Status: an adoption record.**

- **Decision:** on 2026-10-02 the owner adopted candidate psf_compile 2026-10-02.c3 as the release, after its
  pre-registered evaluation (Addenda 303-304).
- **Verdicts at adoption:** four of five predictions confirmed; H2 ambiguous.

## 1. What changes

- **`psf_compile.py` becomes 2026-10-02.1.** It is the candidate with only the version strings changed:
  - the `VERSION:` line;
  - the `VERSION` constant;
  - the changelog heading of item 31.
- **The one functional change is opt-in** (changelog item 31): `compile_for_hardware(..., target=...,
  prune_max_error=0.5)` recompiles on a pruned coupling map only when the first result uses a failed coupler or
  qubit. Without `target`, output is identical to 2026-10-01.1, as checked by the candidate's test against that
  release.
- **Unchanged:** `psf_smart_layout` 2026-10-01.1 and the Rust core 2026-09-29.1.
- **Tests:**
  - [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) and [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py) now expect the new version
    string;
  - [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py) adds the candidate's tests, adapted to the release file.
- **README:**
  - a new "Current version (2026-10-02)" block, which states the known gap;
  - the 2026-10-01 block is kept as "Previous release".

## 2. Basis

From Addendum 304, on the five GAP families × 3 fake devices:

- failed elements are never used;
- 1,926 of 1,926 unaffected circuits are unchanged bit for bit;
- all 153 affected circuits improve (mean infidelity 0.956 → 0.375);
- the FakeTorino gap to Qiskit L3 with the Target falls from 1.87-3.25 to 1.19-1.70.

## 3. What adoption does not claim

- **That the release matches Qiskit L3 with the Target.** It does not: its infidelity is 1.03-1.70 times L3T's on
  every family and device tested, because its layout search ignores gate errors (Addenda 301, 304).
- **Any real-hardware result.**
- **That 0.5 is the right threshold for every device.**
- **`target` is not used by default.** Callers must pass it.

## 4. Next

Error-weighted layout for the general layout search, pre-registered against the same GAP circuits. The P0
noiseless threshold stays at 1e-6.


---

<!-- ===== Addendum 306 (source: spare-qubit-cliff-addendum-306-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of candidate psf_compile 2026-10-02.c4. Locked by the git commit that adds this Addendum, the candidate with its tests, and the evaluation scripts, pushed before the scored run. The smoke run, unfavourable to the candidate, is disclosed in section 5; nothing was changed after it.

## Addendum 306 -- Pre-registration: candidate psf_compile 2026-10-02.c4 (error-aware layout through the device Target). Does handing placement to Qiskit's error-aware layout stage close the placement gap? (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c4_2026-10-02/psf_compile.py`](../../patches/psf_compile_c4_2026-10-02/psf_compile.py), with its tests) and [`benchmarks/c4_eval.py`](../../benchmarks/c4_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **Smoke run:** it was run before the lock and did **not** look good for this candidate (section 5). The test is
  run anyway, unchanged, because 1 circuit per cell is weak evidence and a pre-registered negative result is still
  a result.

## 1. The candidate (changelog item 32)

`compile_for_hardware(..., target=..., error_aware_layout=True)`:

- PSF-Zero compresses the circuit as before.
- The routing call is given the device `target` instead of `coupling_map`/`basis_gates`, and `layout_search` is
  skipped. Qiskit's layout stage at `routing_optimization_level` (default 1) then places the circuit, using the
  Target's errors: VF2Layout, SabreLayout and VF2PostLayout.
- Permutation elision and post-routing re-synthesis are kept.
- Item 31 remains as a backstop.
- Default False: identical to release 2026-10-02.1, as checked by test.

## 2. Design (`benchmarks/c4_eval.py`)

- **Circuits:** the same five GAP families (Addendum 300), via the locked `gap_eval.family()`: 2,079 per device and
  arm.
- **Arms:**

  | arm | what it is |
  |---|---|
  | C3 | release 2026-10-02.1 with `target` (Addendum 304's C3) |
  | C4 | the candidate with `target` and `error_aware_layout=True` |
  | L3T | Qiskit level 3 with the Target and `approximation_degree=1.0` |

- **Devices, noise and metric:** as in Addendum 303.
- **Each family × device "cell" compares paired mean infidelities:** 6 families (F3 split into open and periodic) × 3
  devices = 18 cells.

## 3. Predictions (scored only by `c4_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- all 45 job files are present;
- every simulated circuit's noiseless infidelity is <= 1e-6;
- at most 5% of the circuits are too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the placement gap on chains closes | chains (F3o + F5): C4/L3T <= 1.05 on every device | >= 1.20 on any device |
| H2 | C4 improves on the release | C4/C3 <= 1.00 in >= 16 of 18 cells and no cell > 1.10 | fewer than 12 cells <= 1.00, or any cell > 1.20 |
| H3 | C4 is close to L3T everywhere | C4/L3T <= 1.10 in every cell | any cell > 1.30 |
| H4 | C4 never uses a failed element | 0 failed-edge or failed-qubit uses | any |
| H5 | C4 is not slow | median compile time C4 <= 2 × C3 | > 3 × C3 |

**Reported without prediction:**

- the cell table (C3/L3T, C4/L3T, C4/C3, mean two-qubit counts);
- backstop recompiles;
- compile times.

**Expectations, stated after the smoke run:**

- H1-H3 are likely to be refuted (section 5).
- The run's value is to measure, on 2,079 circuits per device, whether Qiskit's level-1 error-aware layout stage
  helps PSF-Zero's compressed circuits or not.

## 4. What this will not establish

- Real hardware.
- Other routing levels.
- Why the layout stage chooses as it does.

## 5. Development (disclosed)

- **Tests** (`test_c4_layout.py`), 4 tests, all pass at home:
  - the version strings;
  - the default (`error_aware_layout=False`) identical to release 2026-10-02.1, with and without `target`;
  - error-aware outputs on FakeTorino and FakeKingston are exact, use only Target instructions and avoid failed
    elements;
  - without `target` the flag has no effect.
- **Smoke run at home (1 circuit per sub-family, 54 s):**
  - P0 passed.
  - C4/C3 was <= 1.00 in only 7 of 18 cells, ranging 0.54 (FakeTorino F4) to 1.53 (FakeTorino F3p).
  - C4/L3T ranged 0.98-1.66; on chains it was 1.25 / 1.34 / 1.36.
  - The mean two-qubit counts of C4 equalled C3's in every cell: the layout stage re-placed the same routed
    circuit rather than routing differently.
  - Its verdict lines:

    | H1 | H2 | H3 | H4 | H5 |
    |---|---|---|---|---|
    | REFUTED | REFUTED | REFUTED | CONFIRMED | CONFIRMED |

    C4's median compile time was 0.019 s against C3's 0.035 s.
  - These are not results.
- **No threshold or design element was changed after the smoke run.**
- The scorer was run on synthetic files.
- Scored and smoke circuits use disjoint seeds.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c4_2026-10-02/psf_compile.py`](../../patches/psf_compile_c4_2026-10-02/psf_compile.py) | `0a1502f9a6fddcea8cb01105e7053c3c5b5e61359e356330c66d2ca51bab445d` |
| [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py) | `2c820390ab4167635e4612a2b054df2937f96f6441dfbab1580887044224d139` |
| [`benchmarks/c4_eval.py`](../../benchmarks/c4_eval.py) | `449789117b2fff27c48da21d4ff89f27f3e593239d210b110864ae4a6645058c` |
| [`benchmarks/run_c4_2026-10-02.sh`](../../benchmarks/run_c4_2026-10-02.sh) | `eb3f1f40d6646614be487c5e449ccc4495e665a4313b5dcf8f982ed6e0e9f4ab` |


---

<!-- ===== Addendum 307 (source: spare-qubit-cliff-addendum-307-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 306 (lock commit 29dd762), scored by the locked script and re-checked by benchmarks/c4_verify.py, written after the lock and before the results were seen. Candidate c4 is not adopted.

## Addendum 307 -- Results: candidate psf_compile 2026-10-02.c4 (Addendum 306). H1-H3 refuted, H4-H5 confirmed: handing placement to Qiskit's level-1 error-aware layout stage does not close the gap to Qiskit L3. It helps on FakeAuckland, is neutral on FakeTorino and hurts on FakeKingston, and its wins and losses are systematic by device and family. Not adopted (2026-10-02)

**Status: results of the pre-registered test in Addendum 306.**

- **Lock:** commit `29dd762`, pushed before the scored run.
- **Scoring:** by the locked `c4_eval.py score`, and re-checked by an independent script written after the lock and
  before the results were seen (section 5).
- **Setting:** home (WSL2), 6 processes; 45 jobs, 6,237 circuit compilations, 117 s.
- **As expected after the smoke run** (Addendum 306, section 5): the verdicts are the same as the smoke run's.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 45 of 45 files; noiseless infidelity max 1.85e-8 (<= 1e-6); 0 of 6,237 too wide |
| H1 | **REFUTED** | chains (F3o + F5) C4/L3T: FakeAuckland 1.258, FakeTorino 1.320, FakeKingston 1.301 (REFUTED at >= 1.20) |
| H2 | **REFUTED** | C4/C3 <= 1.00 in 7 of 18 cells (range 0.830-1.527) |
| H3 | **REFUTED** | C4/L3T 1.039-1.634; 10 of 18 cells > 1.30 |
| H4 | **CONFIRMED** | 0 failed-edge and 0 failed-qubit uses by C4; 0 backstop recompiles |
| H5 | **CONFIRMED** | median compile time C3 0.028 s, C4 0.014 s (L3T 0.015 s) |

## 2. Numbers

**Cells** (paired mean infidelity ratios):

| family | device | C3/L3T | C4/L3T | C4/C3 |
|---|---|---|---|---|
| F1 ring ansatz | FakeAuckland | 1.156 | 1.106 | 0.957 |
| | FakeTorino | 1.511 | 1.291 | 0.855 |
| | FakeKingston | 1.115 | 1.528 | 1.371 |
| F2 QAOA | FakeAuckland | 1.147 | 1.045 | 0.911 |
| | FakeTorino | 1.189 | 1.214 | 1.021 |
| | FakeKingston | 1.148 | 1.488 | 1.296 |
| F3o open chain | FakeAuckland | 1.377 | 1.288 | 0.936 |
| | FakeTorino | 1.240 | 1.313 | 1.059 |
| | FakeKingston | 1.459 | 1.269 | 0.870 |
| F3p periodic chain | FakeAuckland | 1.252 | 1.381 | 1.103 |
| | FakeTorino | 1.051 | 1.606 | 1.527 |
| | FakeKingston | 1.233 | 1.634 | 1.326 |
| F4 random SU(4) | FakeAuckland | 1.031 | 1.039 | 1.007 |
| | FakeTorino | 1.696 | 1.407 | 0.830 |
| | FakeKingston | 1.195 | 1.608 | 1.346 |
| F5 GHZ chain | FakeAuckland | 1.244 | 1.048 | 0.842 |
| | FakeTorino | 1.299 | 1.375 | 1.059 |
| | FakeKingston | 1.538 | 1.578 | 1.026 |

**Per circuit, C4 against C3** (descriptive, from the independent script):

| device | pooled C4/C3 | C4 lower | same 2q count |
|---|---|---|---|
| FakeAuckland | 0.978 | 509 of 693 | 693 |
| FakeTorino | 1.002 | 308 of 693 | 674 |
| FakeKingston | 1.265 | 126 of 693 | 693 |
| all | geometric mean 1.053 | 943 of 2,079 (C3 lower in 1,134; equal in 2) | 2,060 |

**C4 lower than C3, by family:**

| device | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| FakeAuckland | 199 / 216 | 106 / 120 | 75 / 150 | 57 / 135 | 72 / 72 |
| FakeTorino | 180 / 216 | 43 / 120 | 0 / 150 | 85 / 135 | 0 / 72 |
| FakeKingston | 0 / 216 | 3 / 120 | 75 / 150 | 0 / 135 | 48 / 72 |

## 3. Reading

- **C4 changes where the circuit is placed, not how it is routed.**
  - The two-qubit counts equal C3's in 2,060 of 2,079 circuits (all 19 differences on FakeTorino, F2 and F4).
  - Yet the infidelity differs in 2,077. The level-1 layout stage re-places the same routed circuit on other physical
    qubits.
- **The placement it chooses is not better under the noise model, on balance.**
  - FakeAuckland: better (C4 lower in 509 of 693; every F5 chain).
  - FakeTorino: neutral (pooled 1.002), but every chain (F3 and F5, 222 circuits) is worse.
  - FakeKingston: worse (pooled 1.265; C4 lower in none of 216 F1 rings and none of 135 F4 circuits).
- **The wins and losses are systematic, not noise.**
  - Whole families go one way on a device: 0 of 216, 0 of 150, 72 of 72.
  - This is what one expects if the layout stage ranks regions by a score that disagrees with the noise model in a
    consistent way, so that it repeatedly picks the same regions.
- **Why the score might disagree (not tested here).**
  - Qiskit's layout stage ranks by the Target's reported errors.
  - `NoiseModel.from_backend` applies, per gate, the larger of the reported error and the T1/T2 relaxation floor
    (Addendum 293).
  - Where a region's reported errors lie below its floor, it looks better to the layout stage than it is under the
    noise model.
  - This is the hypothesis that motivates the next candidate. It is stated here, not established.
- **What did work.**
  - C4 never touched a failed element (H4).
  - It compiles in half the time of C3, because it skips PSF-Zero's layout search (H5).
  - In the 7 cells where it won (four on FakeAuckland, two on FakeTorino, FakeKingston F3o) the gains were real:
    C4/C3 0.83-0.96.
- **Decision.** c4 is not adopted. `error_aware_layout` stays in the candidate patch only; release 2026-10-02.1 is
  unchanged.

## 4. Reproducibility

The C3 and L3T arms of this run were compared, circuit by circuit, with the same arms of the c3 run (Addendum 304):

- C3 (release 2026-10-02.1 here, candidate 2026-10-02.c3 there): 2,079 of 2,079 identical (two-qubit count and
  infidelity to 1e-12).
- L3T: 2,079 of 2,079 identical.

This confirms both that the run is deterministic and that the release equals the adopted candidate.

## 5. Independent check

[`benchmarks/c4_verify.py`](../../benchmarks/c4_verify.py) (written after the lock, before the results were seen) reads the raw json only. It
checks the following, and all of it matches the locked score:

- the commit `29dd762`, the script and candidate hashes, the versions and the counts in all 45 files;
- P0;
- H1-H5;
- the cell table;
- the per-circuit comparison in section 2.

Output in `outputs/verify.txt`.

## 6. Next

**Candidate c5 (design to follow):** keep PSF-Zero's routing, and choose the placement itself.

- Enumerate VF2 embeddings of the routed circuit's interaction graph.
- Score each by the errors of the edges and qubits it actually uses, weighted by use count.
- Compare two scores: the reported errors, and the floor-aware errors (the larger of the reported error and the
  T1/T2 floor).
- If the floor-aware score fixes the FakeKingston and FakeTorino-chain losses, the hypothesis in section 3 is
  supported.

A short exploratory diagnosis (which regions C3, C4 and L3T choose, with reported and floor-aware scores) is to come
first, in the style of Addendum 302.

## 7. Data (`data/2026-10-02/c4/`)

- `outputs/`: 45 job files, their logs, `env.txt`, `run.log`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 308 (source: spare-qubit-cliff-addendum-308-2026-10-02.md) ===== -->

> **Note added when merging:** Exploratory diagnosis at home (commit aea5871), not a test. It reuses the measured infidelities of the c4 run (Addendum 307) and recompiles only; diag_extra.py was written after seeing the summary.

## Addendum 308 -- Diagnosis: c4 lost on the Heron devices because Qiskit's level-1 layout stage ranks placements by an averaged per-qubit error that mixes in readout, not because of the T1/T2 floor. Level 3 adds a final re-placement on exact per-gate errors, which is why L3T wins (2026-10-02)

**Status: exploratory diagnosis, not a test.**

- **Question:** it follows up Addendum 307, where c4's wins and losses against C3 were systematic by device and
  family.
- **Scripts and outputs:** in [`data/2026-10-02/c5/diag/`](../../data/2026-10-02/c5/diag/).
  - `region_diag.py` and `run_region_diag.sh`.
  - `diag_extra.py`, written after seeing the summary.
- **Where and when:** run at home at commit `aea5871`; 3 devices in parallel, 67 s.

## 1. What was done

**Recompiling.** Every circuit of the c4 run (2,079 per device, same generator and seeds) was compiled again with
C3, C4 and L3T. There was no simulation:

- the measured infidelities were taken from the c4 run;
- a row was used only if the recompiled two-qubit count matched the recorded one;
- this held for all 6,237 arm-rows.

**Scoring.** The placement each arm chose was scored three ways, summing -log(1 - e) over the gates actually
placed:

| score | e per gate |
|---|---|
| S_rep | the Target's reported error of that gate on those qubits |
| S_eff | 1 - average gate fidelity of the QuantumError that `NoiseModel.from_backend` attaches to that gate, i.e. what the simulation applies, floor-aware |
| S_avg | a replica of Qiskit's average error map, which takes per qargs the mean error over every instruction on them; for a single qubit that includes `measure`, i.e. readout |

**Two hypotheses were in view:**

- the T1/T2 floor (Addendum 307, section 3): S_eff should then order the arms as the measured infidelity does,
  and S_rep should not;
- readout mixed into the averaged map: S_avg should then follow C4's choices and not the measured infidelity.

## 2. Findings

**Which score orders the arms as the measured infidelity does** (sign agreement of the score difference with the
infidelity difference, per circuit):

| device | score | C4 vs C3 | C4 vs L3T | C3 vs L3T |
|---|---|---|---|---|
| FakeAuckland | S_rep | 0.935 | 0.874 | 0.929 |
| | S_eff | 0.933 | 0.965 | 0.983 |
| | S_avg | 0.899 | 0.833 | 0.908 |
| FakeTorino | S_rep | 0.994 | 0.999 | 0.996 |
| | S_eff | 0.997 | 1.000 | 0.996 |
| | S_avg | 0.478 | 0.160 | 0.752 |
| FakeKingston | S_rep | 1.000 | 1.000 | 0.968 |
| | S_eff | 1.000 | 1.000 | 0.968 |
| | S_avg | 0.040 | 0.085 | 0.372 |

**Which arm has the lowest score** (circuits out of 693):

| device | by S_avg (C3 / C4 / L3T) | by S_rep | by measured infidelity |
|---|---|---|---|
| FakeAuckland | 33 / 136 / 524 | 6 / 25 / 662 | 37 / 72 / 584 |
| FakeTorino | 0 / 584 / 109 | 9 / 3 / 681 | 6 / 2 / 685 |
| FakeKingston | 147 / 545 / 1 | 91 / 0 / 602 | 81 / 0 / 612 |

**Mean S_avg / S_rep**, which shows how far the averaged map departs from the per-gate errors:

| device | C3 | C4 | L3T |
|---|---|---|---|
| FakeAuckland | 1.3 | 1.3 | 1.4 |
| FakeTorino | 6.9 | 3.6 | 7.6 |
| FakeKingston | 12.5 | 6.8 | 18.8 |

## 3. Reading

- **The T1/T2 floor is not the cause.**
  - On FakeTorino and FakeKingston, S_eff and S_rep agree with each other and with the measured infidelity almost
    perfectly (0.97-1.00).
  - The floor matters a little only on FakeAuckland. That is the cx device with longer gates, where S_eff orders
    C4 against L3T better than S_rep does (0.965 against 0.874).
  - The hypothesis of Addendum 307, section 3, is not supported as the main cause.
- **The averaged map is the cause.**
  - On FakeKingston, C4 has the lowest S_avg in 545 of 693 circuits, but the lowest S_rep and the lowest measured
    infidelity in none.
  - On FakeTorino, C4 has the lowest S_avg in 584 circuits and the lowest infidelity in 2.
  - S_avg's agreement with the measured order falls to 0.04-0.48 for C4.
  - C4 optimizes the averaged map well, and the averaged map is the wrong target for this metric.
- **Why the averaged map misleads on Heron** (from the Qiskit 2.5.2 source):
  - `build_average_error_map` (`crates/transpiler/src/passes/vf2/vf2_layout.rs`) averages, per qubit, the errors of
    every instruction on it, `measure` included.
  - Readout errors are typically one to two orders of magnitude larger than sx errors, so the qubit term is mostly
    readout. Readout values were not extracted in this run; the ratio S_avg / S_rep is the evidence. On the Heron
    devices it is 3.6-19. On FakeAuckland it is 1.3, which is where C4 helped.
  - The metric here (the state before measurement) contains no readout at all.
  - The replica differs from Qiskit in one detail: Qiskit also counts instructions with no recorded error, as 0, in
    the denominator. Where every qubit has the same instruction set, this rescales the map rather than reordering
    it. This was not checked against Qiskit's own map.
- **Why L3T is not misled.**
  - The level-1 routing stage that c4 uses runs `VF2PostLayout(strict_direction=False)`, which uses the averaged map
    (`qiskit/transpiler/preset_passmanagers/common.py`).
  - Level 3 additionally ends its optimization stage with `VF2PostLayout(strict_direction=True)`
    (`builtin_plugins.py`, level 3), which scores the exact per-instruction errors of the gates in the circuit.
  - L3T has the lowest S_rep in 602-681 of 693 circuits on every device.
- **Caveat.**
  - On hardware, readout does matter for circuits that are measured. "Readout mixed into the qubit term" is wrong
    for this metric, but not wrong in every use.
  - The right treatment is to charge gate errors per gate and readout once per measured qubit, not to average them
    together.

## 4. Consequence for c5

The placement step should score the routed circuit by the exact per-gate errors, the way level 3's final pass does,
not by the averaged map. Two designs are possible:

1. After PSF-Zero's own routing, run `VF2PostLayout(target, strict_direction=True)` and `ApplyLayout`, and keep the
   result only if it is valid and scores lower.
2. Enumerate the VF2 embeddings with our own scorer: S_rep, optionally floor-aware (S_eff), plus readout charged
   once per measured qubit when the circuit measures.

Design 1 reuses Qiskit's tested pass; design 2 controls the score. Either needs a pre-registration. The c4 and
this diagnosis suggest that whichever is chosen should be judged on chains first. There the two-qubit counts are
equal, so only placement differs.

## 5. Data (`data/2026-10-02/c5/diag/`)

- Scripts: `region_diag.py`, `run_region_diag.sh`, `diag_extra.py`.
- `outputs/`:
  - `diag_<device>.json` (per circuit and arm: the three scores, the qubits and edges used, the measured
    infidelity);
  - `summary.md`, `extra.txt`;
  - logs and `env.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 309 (source: spare-qubit-cliff-addendum-309-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of candidate psf_compile 2026-10-02.c5. Locked by the git commit that adds this Addendum, the candidate with its tests, and the evaluation scripts, pushed before the scored run. The predictions were written before the smoke run; the smoke run and one test fix are disclosed in section 5.

## Addendum 309 -- Pre-registration: candidate psf_compile 2026-10-02.c5 (exact error-weighted re-placement after routing). Does Qiskit level 3's final re-placement, added to PSF-Zero's own compile, close the placement gap? (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c5_2026-10-02/psf_compile.py`](../../patches/psf_compile_c5_2026-10-02/psf_compile.py), with its tests) and [`benchmarks/c5_eval.py`](../../benchmarks/c5_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before the smoke run** and were not changed after it. The smoke run
  and one test fix are disclosed in section 5.

## 1. The candidate (changelog item 33)

`compile_for_hardware(..., target=..., placement_refine=True)`:

- **Same as the release up to routing:** PSF-Zero compresses, lays out and routes exactly as release 2026-10-02.1.
- **Then one re-placement step:** the routing pass manager ends, after its optimization stage, with the step that
  Qiskit level 3 runs at the end of its own optimization stage:
  - `VF2PostLayout(target, strict_direction=True, seed=-1)`, with level 3's limits for it (call limit 300,000,
    max trials 2,500);
  - `ApplyLayout` when it finds a placement that scores better.
- **What "scores better" means:** the score is the sum of -log(1 - error) of the gates as placed, using each
  instruction's own reported error. Readout enters only through `measure` instructions, and the circuits here
  have none.
- **It relabels physical qubits only.** Gates, their number and the routing are unchanged. The routing permutation
  and final layout are carried over by `ApplyLayout`'s own update.
- **Safety:** item 31 stays as the backstop. A failed coupler or qubit (error 1.0) costs `f64::MAX` in the score,
  so the re-placement never moves onto one.
- **Defaults:** requires `target`. The default False is identical to the release, as checked by test.
- **Motivation:** Addendum 308.
  - Qiskit's level-1 layout stage (c4) ranks by an averaged per-qubit error that mixes in readout.
  - Level 3's final exact re-placement is why L3T wins.
  - The exact per-gate score (S_rep) ordered C3, C4 and L3T as the measured infidelity did in 93-100% of
    circuits.

## 2. Design (`benchmarks/c5_eval.py`)

- **Circuits:** the five GAP families (Addendum 300), via the locked `gap_eval.family()`: 2,079 per device and
  arm.
- **Arms:**

  | arm | what it is |
  |---|---|
  | C3 | release 2026-10-02.1 with `target` (as in Addenda 304 and 307) |
  | C5 | the candidate with `target` and `placement_refine=True` |
  | L3T | Qiskit level 3 with the Target and `approximation_degree=1.0` |

- **Devices, noise and metric:** as in Addenda 303 and 306.
- **Cells:** 6 families (F3 split into open and periodic) × 3 devices = 18 cells of paired mean infidelity
  ratios.
- **Also recorded per circuit:** whether the re-placement was applied, and whether the item-31 backstop
  recompiled.

## 3. Predictions (scored only by `c5_eval.py score`; written before the smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- all 45 job files are present;
- every simulated circuit's noiseless infidelity is <= 1e-6;
- at most 5% of the circuits are too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the placement gap on chains closes | chains (F3o + F5): C5/L3T <= 1.05 on every device | >= 1.15 on any device |
| H2 | C5 improves on the release almost everywhere | C5/C3 <= 1.00 in >= 17 of 18 cells, and no cell > 1.02 | fewer than 14 cells <= 1.00, or any cell > 1.10 |
| H3 | it rarely makes a circuit worse | per circuit, C5 infidelity <= C3's in >= 90% of circuits on every device | < 80% on any device |
| H4 | it only relabels | on every circuit where neither arm used the item-31 backstop, C5 has the same two-qubit count and depth as C3 | any difference |
| H5 | it never uses a failed element | 0 failed-edge or failed-qubit uses by C5 | any |
| H6 | it is not slow | median compile time C5 <= 3 × C3 | > 10 × C3 |

**Reported without prediction:**

- the cell table (C3/L3T, C5/L3T, C5/C3, mean two-qubit counts, re-placements applied);
- backstop recompiles of both arms;
- compile times;
- the C5/L3T range.

**Expectations, stated after the smoke run (section 5); the predictions were not changed:**

- **FakeAuckland is the risk for H1-H3.** Addendum 308 found that only there does the T1/T2 floor make the
  simulated error differ from the reported one: S_rep's sign agreement was 0.93 there against 0.99-1.00 on the
  Heron devices. C5 scores by the reported error.
- **Off the chains, a gap to L3T should remain.** F1, F2 and F4 still differ from L3T in their two-qubit counts,
  because of routing.

## 4. What this will not establish

- Real hardware, or measured circuits (readout is not in the metric).
- Whether a floor-aware score would do better on FakeAuckland.
- Routing.

## 5. Development (disclosed)

### 5.1 Tests

`test_c5_placement.py`, 7 tests:

- the version string;
- the default identical to release 2026-10-02.1, with and without `target`;
- `placement_refine` without `target` raises;
- on each of FakeAuckland, FakeTorino and FakeKingston, refined outputs are:
  - exact (checked on the touched qubits);
  - free of failed elements;
  - a relabelling of the unrefined compile: same gates, and a reported-error score no higher.

**One test fix.** In the first version, the reference for "same gates" was the release with `target`. On
FakeTorino the unrefined 6-qubit ring uses a failed coupler (Addendum 302), so item 31 recompiled it on the
pruned map, with a different routing. The refined call had moved off the failed coupler and needed no
recompile. The test therefore failed (1 failed, 5 passed), which was a fault of the test and not of the
candidate.

The reference is now the compile without `target`, which is the refined call's own first pass. Calls in which the
backstop still recompiles are checked only for exactness and avoidance. The candidate and the evaluation scripts
were not changed. The fixed test was run at home before the lock (section 5.4).

### 5.2 Smoke run (1 circuit per sub-family, 53 s; not a result)

- P0 passed.
- Chains C5/L3T: FakeAuckland 1.102, FakeTorino 1.009, FakeKingston 1.005.
- C5/C3 <= 1.00 in 17 of 18 cells; the maximum was 1.022 (FakeAuckland F4).
- Per circuit, C5 <= C3: FakeAuckland 0.875, FakeTorino 1.000, FakeKingston 1.000.
- The re-placement was applied in 16, 16 and 13 C5 circuits (FakeAuckland, FakeTorino, FakeKingston).
- Backstop recompiles: C3 4, C5 0. The re-placement moved those circuits off the failed couplers before item 31
  was needed.
- Median compile time: C3 0.032 s, C5 0.037 s, L3T 0.018 s.
- Its verdict lines:

  | H1 | H2 | H3 | H4 | H5 | H6 |
  |---|---|---|---|---|---|
  | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED |

### 5.3 Other development

- The scorer was run on the c4 run's files relabelled as C5. It reproduced Addendum 307's H1-H2 numbers.
- Smoke and scored circuits use disjoint seeds.
- No scored circuit was compiled before the lock.

### 5.4 Test run at home before the lock

The fixed test was run at home on the installed files and passed (7 passed) before this document was
committed; the commit command was chained on that run.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c5_2026-10-02/psf_compile.py`](../../patches/psf_compile_c5_2026-10-02/psf_compile.py) | `daa3a44dd96ffed925bcc5b2edf8d627d480d54976f4affd7579263edcac8240` |
| [`patches/psf_compile_c5_2026-10-02/test_c5_placement.py`](../../patches/psf_compile_c5_2026-10-02/test_c5_placement.py) | `d46be65edf63778f4448b3d9d4500996b217e9f980aff68ad62df4791a89afdd` |
| [`benchmarks/c5_eval.py`](../../benchmarks/c5_eval.py) | `44fe82dea70717478aad64be3fd0854cb95c92ef04f8f8ba7daac4a3155d52ce` |
| [`benchmarks/run_c5_2026-10-02.sh`](../../benchmarks/run_c5_2026-10-02.sh) | `ffdb04a5e9ec10fd031985bc25b79c9b983a6f06d71bbbf0cf322c78267414a0` |


---

<!-- ===== Addendum 310 (source: spare-qubit-cliff-addendum-310-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 309 (lock commit ce574b0), scored by the locked script and re-checked by benchmarks/c5_verify.py, written after the locked score was seen and before the raw files were read.

## Addendum 310 -- Results: candidate psf_compile 2026-10-02.c5 (Addendum 309). Five of six confirmed, H1 ambiguous: the exact re-placement improves on the release in all 18 cells, matches Qiskit L3 on chains on both Heron devices (0.998 and 1.000), never moves a gate count, and makes the failed-coupler backstop unnecessary. FakeAuckland, where the T1/T2 floor matters, keeps a gap (2026-10-02)

**Status: results of the pre-registered test in Addendum 309.**

- **Lock:** commit `ce574b0`, pushed before the scored run.
- **Scoring:** by the locked `c5_eval.py score`, and re-checked by an independent script (section 5). That
  script was written after the locked score was seen and before the raw files were read.
- **Setting:** home (WSL2), 6 processes; 45 jobs, 6,237 circuit compilations, 116 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 45 of 45 files; noiseless infidelity max 1.85e-8 (<= 1e-6); 0 of 6,237 too wide |
| H1 | **AMBIGUOUS** | chains (F3o + F5) C5/L3T: FakeAuckland 1.096, FakeTorino 0.998, FakeKingston 1.000 (CONFIRMED needed <= 1.05 on all three; REFUTED needed any >= 1.15) |
| H2 | **CONFIRMED** | C5/C3 <= 1.00 in 18 of 18 cells (0.574-0.981) |
| H3 | **CONFIRMED** | per circuit C5 <= C3: FakeAuckland 0.932, FakeTorino 0.994, FakeKingston 1.000 |
| H4 | **CONFIRMED** | 1,926 of 1,926 circuits without a backstop recompile: same two-qubit count and depth as C3 |
| H5 | **CONFIRMED** | 0 failed-edge and 0 failed-qubit uses by C5 |
| H6 | **CONFIRMED** | median compile time C3 0.027 s, C5 0.028 s (L3T 0.015 s) |

## 2. Numbers

**Cells** (paired mean infidelity ratios):

| family | device | C3/L3T | C5/L3T | C5/C3 |
|---|---|---|---|---|
| F1 ring ansatz | FakeAuckland | 1.156 | 1.005 | 0.870 |
| | FakeTorino | 1.511 | 1.072 | 0.709 |
| | FakeKingston | 1.115 | 1.060 | 0.951 |
| F2 QAOA | FakeAuckland | 1.147 | 1.040 | 0.907 |
| | FakeTorino | 1.189 | 1.015 | 0.853 |
| | FakeKingston | 1.148 | 1.036 | 0.902 |
| F3o open chain | FakeAuckland | 1.377 | 1.110 | 0.806 |
| | FakeTorino | 1.240 | 0.998 | 0.805 |
| | FakeKingston | 1.459 | 1.000 | 0.685 |
| F3p periodic chain | FakeAuckland | 1.252 | 1.229 | 0.981 |
| | FakeTorino | 1.051 | 1.011 | 0.961 |
| | FakeKingston | 1.233 | 1.086 | 0.881 |
| F4 random SU(4) | FakeAuckland | 1.031 | 1.000 | 0.970 |
| | FakeTorino | 1.696 | 0.973 | 0.574 |
| | FakeKingston | 1.195 | 0.985 | 0.824 |
| F5 GHZ chain | FakeAuckland | 1.244 | 1.000 | 0.804 |
| | FakeTorino | 1.299 | 1.000 | 0.770 |
| | FakeKingston | 1.538 | 1.000 | 0.650 |

**Per device** (independent script):

| device | pooled C5/C3 | pooled C5/L3T | C5 <= L3T | re-placed | C5 worse than C3 |
|---|---|---|---|---|---|
| FakeAuckland | 0.902 | 1.059 | 280 of 693 | 693 | 47 (at most 1.168×) |
| FakeTorino | 0.750 | 1.022 | 353 of 693 | 693 | 4 (at most 1.019×) |
| FakeKingston | 0.861 | 1.040 | 299 of 693 | 619 | 0 |

**The 153 circuits item 31 recompiled for C3** (all on FakeTorino):

- C5 needed no recompile: the re-placement had already moved them off the failed couplers.
- Mean infidelity: C3 0.375, C5 0.232, L3T 0.238.
- C5 kept C3's first-pass two-qubit count in 134 of them.

**Backstop recompiles:** C3 153, C5 0.

## 3. Reading

- **The placement gap is closed on the Heron devices.**
  - On chains, where the two-qubit gates are the same, C5/L3T is 0.998 (FakeTorino) and 1.000 (FakeKingston),
    against 1.24-1.54 for the release.
  - Over all cells there, C5/L3T is 0.97-1.09. What remains is in families whose two-qubit counts still differ
    from L3T's, i.e. routing.
- **It is a strict improvement on the release.**
  - All 18 cells are better (0.574-0.981).
  - Per circuit it is never worse on FakeKingston, and worse in 4 of 693 on FakeTorino (by at most 1.9%).
  - It only relabels qubits (H4): gate counts and depth are unchanged.
  - It costs about 1 ms per compile.
- **It also supersedes the backstop in practice.**
  - The circuits that crossed a failed coupler are moved off it by the re-placement itself.
  - They end slightly better than L3T (0.232 against 0.238).
  - Item 31 stays as the safety net.
- **FakeAuckland keeps a gap, as stated in advance.**
  - Chains 1.096, F3p 1.229; C5 is worse than C3 on 47 of 693 circuits there.
  - This matches Addendum 308: only on FakeAuckland does the simulated error differ from the reported one (the
    T1/T2 floor), and the re-placement scores by the reported error.
  - A floor-aware score is the obvious next refinement. It is not established here.
- **Adoption** is the owner's decision. This test supports adopting c5 as an opt-in (`placement_refine=True` with
  `target`).

## 4. Reproducibility

The C3 and L3T arms of this run were compared, circuit by circuit, with the same arms of the c4 run
(Addendum 307):

- C3: 2,079 of 2,079 identical (two-qubit count and infidelity to 1e-12);
- L3T: 2,079 of 2,079 identical.

## 5. Independent check

[`benchmarks/c5_verify.py`](../../benchmarks/c5_verify.py) reads the raw json only. It checks the following, and all of it matches the locked
score:

- the commit `ce574b0`, the script and candidate hashes, the versions and the counts in all 45 files;
- P0;
- H1-H6;
- the cell table;
- the per-device figures and the backstop subset in section 2.

Output in `outputs/verify.txt`.

## 6. Data (`data/2026-10-02/c5/`)

- `outputs/`: 45 job files, their logs, `env.txt`, `run.log`, `score.md`, `score_log.txt`, `verify.txt`.
- `diag/`: from Addendum 308.
- Local paths were replaced.


---

<!-- ===== Addendum 311 (source: spare-qubit-cliff-addendum-311-2026-10-02.md) ===== -->

> **Note added when merging:** Adoption record: candidate 2026-10-02.c5 becomes release psf_compile 2026-10-02.2 (owner's decision, 2026-10-02).

## Addendum 311 -- Release psf_compile 2026-10-02.2: candidate 2026-10-02.c5 adopted by the owner (opt-in exact error-weighted re-placement, `placement_refine=True`) (2026-10-02)

**Status: adoption record.**

- **Decision:** the owner adopted candidate 2026-10-02.c5 on 2026-10-02, after seeing the scored results of
  Addendum 310 (five of six predictions confirmed, H1 ambiguous because of FakeAuckland).
- **Basis:** this Addendum records what changed; Addendum 310 is the basis.

## 1. What changed

- **`psf_compile.py`** becomes release 2026-10-02.2.
  - The code is the candidate's ([`patches/psf_compile_c5_2026-10-02/psf_compile.py`](../../patches/psf_compile_c5_2026-10-02/psf_compile.py), Addendum 309) with three
    lines changed: the `VERSION:` header, the changelog heading of item 33, and the `VERSION` constant.
  - Changelog item 33 is now part of the release.
- **`placement_refine` stays opt-in (default False)**, as tested. Without it the release is identical to
  2026-10-02.1, gate for gate (checked by test). Making it the default for calls with `target` would be a separate
  decision.
- **Tests pinned to the release version now expect "2026-10-02.2":**
  - [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py);
  - [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py);
  - [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py);
  - [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py).
- **About the last of these:** it belongs to the files locked by Addendum 306. Its `test_version` pinned the
  release at the time, so it would otherwise fail from now on. Only that line was changed, with a comment.
  - Normalized SHA-256 before: `2c820390ab4167635e4612a2b054df2937f96f6441dfbab1580887044224d139` (as locked).
  - After: `a42b5ea9b6abdaab58ca578902cf9cd46f6d89aab356ebc1ceb721f752009cbe`.
  - The c4 evaluation is complete (Addendum 307), and its job files record the hashes it actually ran with.
- **New [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py)**, adapted from the candidate's tests.
  - The previous release is represented by [`patches/psf_compile_c3_2026-10-02/psf_compile.py`](../../patches/psf_compile_c3_2026-10-02/psf_compile.py), which differs from
    release 2026-10-02.1 only in its version lines.
  - The tests: version; default identical to the previous release with and without `target`; `placement_refine`
    without `target` raises; on FakeAuckland, FakeTorino and FakeKingston, refined outputs are exact, avoid failed
    elements, and relabel the unrefined compile with a reported-error score no higher.
- **`README.md`:**
  - a new "Current version (2026-10-02, second release)" block;
  - the 2026-10-02.1 block becomes "Previous release (2026-10-02.1)", and its known gap is marked as addressed.
- **Data:** [`data/2026-10-02/c5/outputs/`](../../data/2026-10-02/c5/outputs/) (Addendum 310) and [`benchmarks/c5_verify.py`](../../benchmarks/c5_verify.py).

## 2. Checks before the commit

These tests were run at home on the applied files, and the commit command was chained on them passing:

- [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py);
- [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py);
- [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py);
- [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py);
- [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py);
- [`patches/psf_compile_c5_2026-10-02/test_c5_placement.py`](../../patches/psf_compile_c5_2026-10-02/test_c5_placement.py).

## 3. Release file (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| `psf_compile.py` (2026-10-02.2) | `2603bc2decea47b2047fe8ddf17896d79f39e74eda887f0b596f6a28f4daa08b` |
| [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py) | `cab184e63b308d7f5de4dbe31f6309b1f83d801c978d4dd56943e9e556c2312a` |

## 4. What remains

- **FakeAuckland:** a floor-aware score is untested.
- **Routing:** where PSF-Zero still uses more two-qubit gates than level 3.
- **Measured circuits:** readout charged once per measured qubit.
- **Real hardware.**


---

<!-- ===== Addendum 312 (source: spare-qubit-cliff-addendum-312-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of psf_ai_compile 2026-10-02.a6 (the AI front end with release 2026-10-02.2 inside). Locked by the git commit that adds this Addendum, the candidate with its tests, and the evaluation scripts, pushed before the scored run. The predictions were written before the smoke run, which is disclosed in section 5.

## Addendum 312 -- Pre-registration: the AI front end integrated with release 2026-10-02.2. Does psf_ai_compile a6 (a5 with the release's exact re-placement inside every compile) keep or improve a5, does the AI front end still add anything over the release alone, and is a fast mode enough? (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_ai_compile_a6_2026-10-02/psf_ai_compile.py`](../../patches/psf_ai_compile_a6_2026-10-02/psf_ai_compile.py), with its tests) and [`benchmarks/ai6_eval.py`](../../benchmarks/ai6_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before the smoke run.**

## 1. The candidate (psf_ai_compile 2026-10-02.a6)

a5 is the AI front end for model-written circuits (Addenda 275-289). It does four things:

- it generates several PSF-Zero compiles per circuit (starting points, seeds, Qiskit level-3 layouts);
- it polishes them;
- it re-places the best few by a state-aware error estimate that includes the T1/T2 floor, enumerating up to 5,000
  embeddings and keeping the current placement unless one scores better;
- it returns the best by that estimate.

In GAP it was level with Qiskit L3T (0.98-1.02), at a median 0.66 s per circuit.

**a6 changes two things, and only when a `target` is given:**

1. **The release inside.** Every internal `compile_for_hardware()` call gets `target=` and `placement_refine=True`
   (release 2026-10-02.2, items 31 and 33).
   - Each candidate starts from the release's exact error-weighted placement and avoids failed couplers. a5's
     internal compiles saw no target.
   - The state-aware re-placement still runs. Where its 5,000-embedding search is cut off, the better starting
     placement can survive.
2. **A fast mode.** `state_aware_placement=False` skips the state-aware re-placement and returns the candidate with
   the fewest two-qubit gates (then two-qubit depth), as placed by the release.

Without a target, a6 is a5, as checked by test.

## 2. Design (`benchmarks/ai6_eval.py`)

**Circuit sets:**

| set | what it is |
|---|---|
| GAP | the five GAP families (Addendum 300) via the locked `gap_eval.family()`, same seeds: 2,079 per device, all with at most 8 qubits |
| MODEL | the 153-circuit replay set of Addendum 285: every circuit the models wrote in the 2026-09-30 pod runs that is not a best_circuit.json, unique per task, with at most 8 qubits, converted as the e2e harness does |

Neither set is new. GAP was used in Addenda 301-310, and MODEL in Addenda 285 and, through a3/a4's development, in
286-287. This test compares compilers on them; it does not tune anything on them.

**Arms** (release 2026-10-02.2):

| arm | what it is |
|---|---|
| C5 | the release alone: `compile_for_hardware(..., target, placement_refine=True)` |
| A5 | psf_ai_compile a5 ([`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py)) with the target |
| A6 | the candidate with the target |
| A6F | the candidate with the target and `state_aware_placement=False` |
| L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

**Devices, noise and metric:**

- FakeAuckland, FakeTorino, FakeKingston, with the metric of Addenda 303-310.
- All ratios are pooled mean infidelities per device and set. GAP pools all five families.
- Also recorded: instructions not in the Target ("off-target", reported only).

## 3. Predictions (scored only by `ai6_eval.py score`; written before the smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- all 90 job files are present;
- every noiseless infidelity is <= 1e-6;
- at most 5% of the circuits are too wide;
- at least 150 model circuits were converted.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | integration does no harm | A6/A5 <= 1.00 on every device, GAP and MODEL | any > 1.02 |
| H2 | integration helps where a5's search is cut off | GAP: A6/A5 <= 0.98 on both Heron devices | >= 1.00 on both |
| H3 | the AI front end still adds something over the release on model-written circuits | MODEL: A6/C5 <= 0.95 on every device | any >= 1.00 |
| H4 | ... and does not lose to it on GAP | GAP: A6/C5 <= 1.00 on every device | any > 1.05 |
| H5 | the fast mode is nearly as good on the Heron devices, at half the time or less | A6F/A6 <= 1.05 on both Heron devices, GAP and MODEL, and median compile time A6F <= 0.5 × A6 | any of those ratios > 1.15, or median A6F > A6 |
| H6 | A6 is level with or better than L3T | A6/L3T <= 1.00 on every device, GAP and MODEL | any > 1.05 |
| H7 | the release's placement never uses a failed element | 0 failed-edge or failed-qubit uses by C5, A6 and A6F | any |
| H8 | integration costs little time | median compile time A6 <= 1.3 × A5 | > 2 × A5 |

**Reported without prediction:**

- the per-set and per-family tables;
- mean two-qubit counts;
- compile times;
- off-target instructions by arm.

## 4. What this will not establish

- Hardware.
- The vLLM loop itself (pass rates, or the model's behaviour with a6).
- Circuits above 8 qubits: none are in either set, so a6's fast path for them is not tested.

## 5. Development (disclosed)

### 5.1 Tests

`test_ai6.py`, 6 tests, all pass at home:

- the version strings;
- without a target, a6 equals a5 gate for gate;
- on FakeAuckland, FakeTorino and FakeKingston, a6 outputs (both modes) are exact on the touched qubits and avoid
  failed elements;
- a6 refuses a psf_compile without `placement_refine`.

### 5.2 Smoke run (not a result)

The smoke run used 1 circuit per GAP sub-family and the first 6 sandbox dry-run (mock) circuits for MODEL, and took
122 s. The lock command was chained on its P0 line reading PASS.

- **A6/A5 was 1.000 in every family-device cell and in MODEL.** On these circuits a6 returned a5's result: a5's
  state-aware re-placement reached the same placement whatever the starting placement.
- **GAP pooled:**
  - A6/C5: FakeAuckland 0.963, FakeTorino 0.955, FakeKingston 0.940;
  - A6/L3T: 0.996, 0.989, 0.980.
- **MODEL (mock circuits, mean 1.7 two-qubit gates):**
  - A6/C5: 0.822, 0.990, 0.988;
  - A6/L3T: 0.864, 0.930, 0.903.
- **Fast mode:** A6F/A6 on the Heron devices was 1.002-1.012.
- **Median compile time:** C5 0.035 s, A5 0.707 s, A6 0.669 s, A6F 0.312 s, L3T 0.017 s.
- **Failed elements and off-target instructions:** none, in any arm.
- **Its verdict lines:**

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 |
  |---|---|---|---|---|---|---|---|
  | CONFIRMED | REFUTED | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

**Expectations, stated after the smoke run; the predictions were not changed:**

- **H2 is likely to be refuted.** The integration does not change a5's choices when the state-aware search finds
  the same optimum from any start.
- **H3 depends on the real model circuits**, which are larger than the mock ones (Addendum 285: 2-qubit sums of
  about 6 per circuit).

### 5.3 Other development

- The scorer was run on synthetic files built from the c5 run.
- The smoke run's MODEL circuits are the sandbox mock circuits, never the scored set.
- No scored circuit was compiled before the lock.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_ai_compile_a6_2026-10-02/psf_ai_compile.py`](../../patches/psf_ai_compile_a6_2026-10-02/psf_ai_compile.py) | `2b8d11fac91138af4352bd1d35c8383197dca9d43763811c4715319849f3e046` |
| [`patches/psf_ai_compile_a6_2026-10-02/test_ai6.py`](../../patches/psf_ai_compile_a6_2026-10-02/test_ai6.py) | `1b4920cbc1a31b858409a0c555cda7b2d5306054c00f390997b9c5f396384b41` |
| [`benchmarks/ai6_eval.py`](../../benchmarks/ai6_eval.py) | `adef4385dbee523f38f8a9ff04f8cc9e90b9dc4bbfa084ae3bfdfa850adcd7f0` |
| [`benchmarks/run_ai6_2026-10-02.sh`](../../benchmarks/run_ai6_2026-10-02.sh) | `7d3cb7e8b1b8e861d150321686a8b83a346d0aef3ac68a30c2ac3af5f37dd87e` |


---

<!-- ===== Addendum 313 (source: spare-qubit-cliff-addendum-313-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 312 (lock commit c8dc903), scored by the locked script and re-checked by benchmarks/ai6_verify.py, written after the locked score was seen and before the raw files were read. Section 4 corrects a circuit count stated in earlier Addenda; section 5 describes the link update made to this Part at the same time.

## Addendum 313 -- Results: the AI front end with release 2026-10-02.2 inside (Addendum 312). Four of eight confirmed, four ambiguous, none refuted. a6 returns a5's result (identical on 2,070 of 2,079 GAP and 449 of 459 model-written circuits): a5's state-aware re-placement already finds what the release's placement offers. The AI front end clearly adds to the release on model-written circuits (A6/C5 0.73-0.92), and its fast mode keeps most of that at half the time on the Heron devices. Also: corrections (circuit counts) and links in Part 9 (2026-10-02)

**Status: results of the pre-registered test in Addendum 312.**

- **Lock:** commit `c8dc903`, pushed before the scored run.
- **Scoring:** by the locked `ai6_eval.py score`, and re-checked by [`benchmarks/ai6_verify.py`](../../benchmarks/ai6_verify.py). That script was
  written after the locked score was seen and before the raw files were read.
- **Setting:** home (WSL2), 6 processes, PennyLane 0.45.1; 90 jobs, 12,690 circuit compilations.
- **Model circuits:** 153 collected, 153 converted, 0 skipped.

## 1. Verdicts

| ID | Verdict | Numbers (unrounded where it matters) |
|---|---|---|
| P0 | **PASS** | 90 of 90 files; noiseless infidelity max 1.85e-8; 0 too wide; 153 model circuits |
| H1 | **AMBIGUOUS** | A6/A5: GAP 1.00004 / 0.99999 / 1.00002, MODEL 0.99988 / 0.99941 / 0.99998 (Auckland / Torino / Kingston). Two exceed 1.00 by less than 1e-4; none exceeds 1.02 |
| H2 | **AMBIGUOUS** | GAP A6/A5: FakeTorino 0.99999, FakeKingston 1.00002 (CONFIRMED needed <= 0.98 on both; REFUTED needed >= 1.00 on both) |
| H3 | **CONFIRMED** | MODEL A6/C5: 0.731 / 0.922 / 0.914 |
| H4 | **CONFIRMED** | GAP A6/C5: 0.967 / 0.963 / 0.945 |
| H5 | **CONFIRMED** | A6F/A6 on the Heron devices: GAP 1.002 / 1.006, MODEL 1.020 / 1.021; median compile time A6F 0.297 s, A6 0.612 s |
| H6 | **AMBIGUOUS** | A6/L3T: GAP 1.024 / 0.984 / 0.983, MODEL 0.740 / 0.932 / 0.934 (FakeAuckland GAP above 1.00, below 1.05) |
| H7 | **CONFIRMED** | 0 failed-element uses by C5, A6 and A6F (also 0 by A5 and L3T) |
| H8 | **CONFIRMED** | median compile time A5 0.613 s, A6 0.612 s |

## 2. Numbers

**Pooled infidelity ratios:**

| set | device | A6/A5 | A6/C5 | A6F/A6 | A6F/C5 | A6/L3T | A6F/L3T | C5/L3T |
|---|---|---|---|---|---|---|---|---|
| GAP | FakeAuckland | 1.000 | 0.967 | 1.018 | 0.985 | 1.024 | 1.042 | 1.059 |
| GAP | FakeTorino | 1.000 | 0.963 | 1.002 | 0.965 | 0.984 | 0.986 | 1.022 |
| GAP | FakeKingston | 1.000 | 0.945 | 1.006 | 0.950 | 0.983 | 0.988 | 1.040 |
| MODEL | FakeAuckland | 1.000 | 0.731 | 1.284 | 0.939 | 0.740 | 0.951 | 1.013 |
| MODEL | FakeTorino | 0.999 | 0.921 | 1.020 | 0.940 | 0.932 | 0.951 | 1.012 |
| MODEL | FakeKingston | 1.000 | 0.914 | 1.021 | 0.933 | 0.934 | 0.954 | 1.022 |

**Where A6 differs from A5 at all:**

- GAP F4 only: 2, 3 and 4 circuits of 135 (Auckland, Torino, Kingston).
- MODEL: 3, 3 and 4 circuits of 153.
- Every other circuit is identical.

**MODEL per task** (FakeAuckland; mean infidelity C5 → A6, mean two-qubit gates C5 → A6):

| task | n | infidelity C5 → A6 | 2q C5 → A6 |
|---|---|---|---|
| w3 | 83 | 0.0511 → 0.0360 | 5.4 → 5.1 |
| qft3 | 28 | 0.0409 → 0.0269 | 4.9 → 4.1 |
| dicke42 | 16 | 0.1244 → 0.0954 | 13.5 → 11.9 |
| w4 | 12 | 0.0713 → 0.0581 | 7.7 → 7.7 |
| ghz5 | 7 | 0.0519 → 0.0443 | 6.1 → 5.9 |

On the Heron devices the same tasks gain less (A6/C5 0.91-0.92 pooled). There the release's placement is already
good, so the remaining gain is mostly the fewer two-qubit gates (qft3 4.9 → 4.2, dicke42 13.3-14.5 → 12.4-12.8).

## 3. Reading

- **Integrating the release into a5 changes nothing that matters.**
  - a5 re-places its best candidates by a state-aware estimate, keeping the starting placement only if nothing
    better is found. In practice it finds the same placement from the error-blind start as from the release's
    exact one.
  - The pre-registered reason for H2 (the 5,000-embedding cap) does not bite on these circuits.
  - H1 and H2 are ambiguous only because the ratios sit at 1.0000 ± 0.0006.
- **The AI front end adds clearly to the release, most on model-written circuits.**
  - A6/C5 is 0.73-0.92 on MODEL and 0.95-0.97 on GAP.
  - Two sources:
    - fewer two-qubit gates on model-written circuits (commutation clean-up, several starting points, polish);
    - on FakeAuckland, a placement score that includes the T1/T2 floor and the circuit's state, which the release's
      reported-error score lacks (Addendum 308).
- **The fast mode.**
  - **On the Heron devices** it keeps almost all of a6's quality (within 2.1%) at 0.49 times the time.
  - **On FakeAuckland** it loses most of the MODEL gain (A6F/A6 1.284), because there the state-aware placement is
    what matters.
  - **Against the release alone**, A6F is better everywhere (A6F/C5 0.93-0.99).
- **Against Qiskit L3T:**
  - A6 is better on MODEL on every device (0.74-0.93) and on GAP on the Heron devices (0.98).
  - On FakeAuckland GAP it trails by 2.4%, as A5 did in Addendum 301.
- **Reproducibility.** A5's outputs equal the GAP run's A5 arm (Addendum 301), circuit by circuit: 2,079 of 2,079
  (two-qubit count and infidelity to 1e-12). That run used release 2026-10-01.1 underneath, and release 2026-10-02.2
  without its opt-in arguments is identical to it.
- **Decision (owner):** whether a6 replaces a5 as [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py). What a6 adds over a5 is the fast
  mode; its default mode equals a5 in practice. For the vLLM loop:
  - A6F suits Heron targets;
  - A6 / A5 suits devices with below-floor reported errors (cx/ecr devices, Addendum 293).

## 4. Corrections to earlier Addenda (circuit counts)

The GAP generator yields **693 circuits per device**: F1 216, F2 120, F3 150, F4 135, F5 72. That is **2,079 over
the three devices**. Several earlier texts say "2,079 per device":

- Addendum 301 (title);
- Addendum 303 (section 2);
- Addendum 306 (section 2 and its expectations);
- Addendum 308 (section 1);
- Addendum 309 (section 2);
- Addendum 312 (section 2);
- the README block for release 2026-10-02.2.

The correct reading is "693 per device, 2,079 in all". Every prediction and verdict in those tests is a per-device
or pooled ratio computed from the actual files, so none changes.

- **The locked pre-registrations** (303, 306, 309, 312) are left as they are; this Addendum is the correction.
- **The README** is corrected in this update.
- **The 2,079 counts in Addenda 307 and 310** (for example "C3: 2,079 of 2,079 identical") are totals over the three
  devices and are correct.

## 5. Links in Part 9 (presentation only)

**Code-formatted paths became links.**

- From Addendum 272 on, every code-formatted path to a file or folder that exists in the repository is now a
  relative link: `` [`path`](../../path) ``, as Addendum 270 already did.
- No other character of those Addenda changed. The update script checks this by removing the link syntax again and
  comparing with the previous text.

**Eight references point to files that are not in the repository.**

- They are the workplace pre-registration documents of 2026-10-01 (`docs/findings/*-preregistration-2026-10-01.md`).
- Their text is in this Part as Addenda 272, 276, 278, 280, 282, 284, 286 and 288. The one original found at home
  (core-fix-c2) matches Addendum 272 apart from editing at merge time.
- Each of the eight references now carries a note saying so.

## 6. Data (`data/2026-10-02/ai6/`)

- `outputs/`: 90 job files and their logs, the prep log, `model_circuits.qpy`, `model_index.json`, `env.txt`,
  `run.log`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 314 (source: spare-qubit-cliff-addendum-314-2026-10-02.md) ===== -->

> **Note added when merging:** Exploratory diagnosis at home (commit 1e3f54b), not a test. It recompiles the scored F3 circuits with A5 and L3T only and reuses the measured infidelities of the ai6 run (Addendum 313).

## Addendum 314 -- Diagnosis: on FakeAuckland, a5 loses to Qiskit L3T on every F3 chain circuit on the same six qubits and with the same two-qubit count, and its own estimate knew: L3T's output was never among a5's candidates (2026-10-02)

**Status: exploratory diagnosis, not a test.**

- **Question:** it follows up Addendum 313. Over all GAP circuits on FakeAuckland, A5 trailed L3T (1.024), and the
  whole deficit sat in F3: A5/L3T was 1.111 on open chains and 1.227 on periodic ones, worse on 75 of 75 circuits
  each, with equal two-qubit counts. Every other cell favoured A5 (0.74-0.99).
- **Scripts and outputs:** [`data/2026-10-02/a7/diag/`](../../data/2026-10-02/a7/diag/) (`a5_f3_diag.py`, `run_a5_f3_diag.sh`, `outputs/`).
- **Where and when:** run at home at commit `1e3f54b`.

## 1. What was done

**Recompiling.** All 150 F3 circuits of the ai6 run were compiled again with A5 and L3T on each device. There was
no simulation:

- the measured infidelities were taken from the ai6 run;
- every recompiled two-qubit count matched the recorded one (0 mismatches on all three devices).

**Scoring.** Each output was scored three ways:

| score | what it is |
|---|---|
| est_sa | a5's own state-aware estimate (`state_aware_cost`), the score a5 selects with |
| S_eff | the summed -log(1 - e) of the errors Aer actually applies |
| S_rep | the same with the reported errors |

## 2. Findings

| device | chain | A5/L3T measured | est_sa: A5 better | S_eff: A5 better | measured: A5 better | est_sa agrees with measured | same qubit set |
|---|---|---|---|---|---|---|---|
| FakeAuckland | open | 1.111 | 0 of 75 | 3 | 0 | 75 of 75 | 75 of 75 |
| FakeAuckland | periodic | 1.227 | 0 of 75 | 27 | 0 | 75 of 75 | 75 of 75 |
| FakeTorino | open | 0.995 | 74 | 13 | 45 | 46 | 75 |
| FakeTorino | periodic | 1.007 | 74 | 75 | 14 | 15 | 75 |
| FakeKingston | open | 0.981 | 75 | 31 | 56 | 56 | 75 |
| FakeKingston | periodic | 1.003 | 75 | 75 | 28 | 28 | 75 |

**Means on FakeAuckland:**

| chain | est_sa A5 / L3T | S_eff A5 / L3T | measured infidelity A5 / L3T |
|---|---|---|---|
| open | 0.471 / 0.450 | 0.463 / 0.460 | 0.375 / 0.337 |
| periodic | 0.869 / 0.808 | 0.856 / 0.856 | 0.571 / 0.465 |

## 3. Reading

- **On FakeAuckland, a5's candidates never included what L3T produced.**
  - Both use the same six qubits ([2, 3, 5, 8, 11, 14]) in every circuit, and the same number of two-qubit gates.
  - a5's own estimate ranks L3T's output better in 150 of 150 circuits, in agreement with the measurement in 150 of
    150.
  - So a5 did not misjudge; it simply had no such candidate. Its re-placement can only relabel qubits, and the
    difference is not in which qubits are used.
- **What differs is the gate structure on those qubits** (cx direction, single-qubit gates, order), not the summed
  gate error.
  - S_eff is nearly equal for both (0.463 against 0.460; 0.856 against 0.856), yet the measured infidelity differs by
    11-23%.
  - On this cx device the effect depends on the state the circuit passes through, which is what a4's state-aware
    estimate models. This was not investigated further.
- **On the Heron devices the picture is different and the stakes small.**
  - A5/L3T is 0.98-1.01, and the qubit sets are again identical.
  - Here est_sa prefers A5 in 74-75 of 75 circuits but agrees with the measurement in only 15-56 of 75, so the
    estimate is weak at this level of difference. Any change that lets the estimate choose between near-equal outputs
    could therefore cost a little on these devices.
- **Consequence:** candidate a7 (Addendum 315) adds L3T's own output to a5's candidates. a5 already computes it to
  borrow its layout, so this costs nothing extra.

## 4. Data (`data/2026-10-02/a7/diag/outputs/`)

- Per device: the three scores, the qubits used and the measured infidelity, per circuit and arm (`diag_<device>.json`).
- `summary.md`, logs and `env.txt`.


---

<!-- ===== Addendum 315 (source: spare-qubit-cliff-addendum-315-2026-10-02.md) ===== -->

> **Note added when merging:** Home pre-registration of psf_ai_compile 2026-10-02.a7. Locked by the git commit that adds this Addendum, the candidate with its tests, and the evaluation scripts, pushed before the scored run. The predictions were written before the smoke run, which is disclosed in section 5.

## Addendum 315 -- Pre-registration: psf_ai_compile 2026-10-02.a7 (a5 plus Qiskit level 3's own output as a candidate). Does it remove a5's loss on the FakeAuckland F3 chains without costing anything elsewhere? (2026-10-02)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_ai_compile_a7_2026-10-02/psf_ai_compile.py`](../../patches/psf_ai_compile_a7_2026-10-02/psf_ai_compile.py), with its tests) and [`benchmarks/a7_eval.py`](../../benchmarks/a7_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before the smoke run** and were not changed after it.

## 1. The candidate (psf_ai_compile 2026-10-02.a7, item 12)

a5 already transpiles each circuit with Qiskit level 3 and the target, to borrow the layout it picks (a2). a7 also
offers that output itself as a candidate.

- **How it competes:** it is re-placed and scored by a5's state-aware estimate, like PSF-Zero's best candidates.
- **When it is scored:** whenever its two-qubit count is within REMAP_EXTRA_2Q (2) of PSF-Zero's best. It does not
  count against REMAP_TOP.
- **As returned:** it is used as Qiskit returns it, not polished.
- **Cost:** no extra compile.
- **Reporting:** `return_info` reports which kind of candidate won (`chosen`: "PSF" or "L3T").
- **Without a target:** a7 is a5, as checked by test.
- **Why:** Addendum 314.

## 2. Design (`benchmarks/a7_eval.py`)

- **Sets:** as in Addendum 312.
  - GAP: 693 circuits per device.
  - MODEL: the 153 model-written circuits. The smoke run uses mock circuits instead.
- **Arms:**

  | arm | what it is |
  |---|---|
  | A5 | [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) with the target |
  | A7 | the candidate with the target |
  | L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

  Release 2026-10-02.2 is underneath, without its opt-in arguments.
- **Devices and metric:** as in Addendum 312.
- **Sets seen before:** both sets were seen before (Addenda 285, 301-313). The F3 result that motivates a7 comes
  from the GAP set itself, so H1 and H3 test the fix on the circuits that revealed the problem. H2 and H4 test that
  nothing else gets worse.

## 3. Predictions (scored only by `a7_eval.py score`; written before the smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- 54 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% too wide;
- at least 150 model circuits converted.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the FakeAuckland F3 loss goes away | A7/L3T <= 1.02 on open and on periodic chains | either >= 1.10 |
| H2 | nothing gets worse on average | A7/A5 <= 1.00 on every device, GAP and MODEL | any > 1.02 |
| H3 | A7 is level with L3T on FakeAuckland GAP | A7/L3T <= 1.00 | > 1.02 |
| H4 | few circuits get worse | per circuit, A7 <= A5 in >= 95% of circuits on every device | < 90% on any device |
| H5 | no extra time | median compile time A7 <= 1.1 × A5 | > 1.5 × A5 |
| H6 | no failed element | 0 failed-edge or failed-qubit uses by A7 | any |

**Reported without prediction:**

- the cell table;
- how often A7 chose L3T's output, by set and device;
- compile times;
- off-target instructions.

## 4. What this will not establish

- Hardware.
- Held-out circuits.
- Why the gate structure matters on FakeAuckland (Addendum 314, section 3).

## 5. Development (disclosed)

### 5.1 Tests

`test_a7.py`, 5 tests, all pass at home:

- the version;
- without a target, a7 equals a5 gate for gate;
- on FakeAuckland, FakeTorino and FakeKingston, for the smoke F3 chains:
  - the output is exact and avoids failed elements;
  - `chosen` is reported;
  - a7's estimate is no worse than a5's or than L3T's output.

### 5.2 Smoke run (not a result)

The smoke run used 1 circuit per GAP sub-family and 3 mock model circuits, and took 78 s.

- **FakeAuckland F3:**
  - A7 chose L3T's output on both chains;
  - A7/L3T 0.887 (open) and 0.881 (periodic);
  - A5/L3T 1.150 and 1.297.
- **A7/A5 pooled:**
  - GAP: FakeAuckland 0.942, FakeTorino 0.999, FakeKingston 1.0003;
  - MODEL: 0.978, 1.000, 1.000.
- **Per circuit A7 <= A5:** FakeAuckland 0.842, FakeTorino 1.000, FakeKingston 0.947 (19 circuits per device).
  - FakeAuckland had 3 circuits worse and FakeKingston 1.
  - These include choices of L3T's output that the estimate preferred but the measurement did not.
- **Median compile time:** A5 0.682 s, A7 0.713 s.
- **Failed elements and off-target instructions:** none.
- **Its verdict lines:**

  | H1 | H2 | H3 | H4 | H5 | H6 |
  |---|---|---|---|---|---|
  | CONFIRMED | AMBIGUOUS | CONFIRMED | REFUTED | CONFIRMED | CONFIRMED |

**Expectations, stated after the smoke run; the predictions were not changed:**

- **H4 is at risk.** Addendum 314 found the estimate weak when two outputs are within a few percent (Heron F3).
  Letting it choose between PSF-Zero's and L3T's outputs will sometimes pick the slightly worse one.
- **H2 may be decided by small differences on the Heron devices.**

### 5.3 Other development

- The scorer was run on synthetic files built from the ai6 run.
- No scored circuit was compiled with a7 before the lock.
- The diagnosis of Addendum 314 recompiled the scored F3 circuits with a5 and L3T only. It motivated a7 and is
  disclosed there.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_ai_compile_a7_2026-10-02/psf_ai_compile.py`](../../patches/psf_ai_compile_a7_2026-10-02/psf_ai_compile.py) | `e2132e3c99b19ad3661cd757e9f787b50d738ea0cc180f350b4791c3fbdfa512` |
| [`patches/psf_ai_compile_a7_2026-10-02/test_a7.py`](../../patches/psf_ai_compile_a7_2026-10-02/test_a7.py) | `6d82b85e3d2fc8bdc95584204f67efcb1a606f0b1cdb693f34f536ca5358c14f` |
| [`benchmarks/a7_eval.py`](../../benchmarks/a7_eval.py) | `4d5359fe1e7e3af8ceed2a3051a9f3fa4fdcb99ed91d7d8ba5eff8d12779be74` |
| [`benchmarks/run_a7_2026-10-02.sh`](../../benchmarks/run_a7_2026-10-02.sh) | `abe60d4220c4cf47e59e51228bf65d55cd6b7a01ce45a1cfeae28ac5eb7b56fb` |


---

<!-- ===== Addendum 316 (source: spare-qubit-cliff-addendum-316-2026-10-02.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 315 (lock commit 3130e3e), scored by the locked script and re-checked by benchmarks/a7_verify.py, written after the locked score was seen and before the raw files were read.

## Addendum 316 -- Results: psf_ai_compile a7 (Addendum 315). All six confirmed: offering Qiskit level 3's own output as a candidate removes a5's FakeAuckland F3 loss (A7/L3T 0.932 open, 0.895 periodic, against A5's 1.111 and 1.227), makes A7 better than L3T on FakeAuckland GAP overall (0.949), and costs nothing on average elsewhere (A7/A5 0.93-1.00) (2026-10-02)

**Status: results of the pre-registered test in Addendum 315.**

- **Lock:** commit `3130e3e` (20:34 JST), pushed before the scored run.
- **Scoring:** by the locked `a7_eval.py score`, and re-checked by [`benchmarks/a7_verify.py`](../../benchmarks/a7_verify.py). That script was written
  after the locked score was seen and before the raw files were read.
- **Setting:** home (WSL2), 6 processes; 54 jobs, 7,614 circuit compilations; 153 model circuits converted.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 54 of 54 files; noiseless infidelity max 1.85e-8; 0 too wide; 153 model circuits |
| H1 | **CONFIRMED** | FakeAuckland F3 A7/L3T: open 0.932, periodic 0.895 |
| H2 | **CONFIRMED** | A7/A5: GAP 0.927 / 0.999 / 0.999, MODEL 0.979 / 0.983 / 0.983 (Auckland / Torino / Kingston) |
| H3 | **CONFIRMED** | FakeAuckland GAP A7/L3T 0.949 (A5 1.024) |
| H4 | **CONFIRMED** | per circuit A7 <= A5: 96.6% / 96.9% / 97.8% |
| H5 | **CONFIRMED** | median compile time A5 0.608 s, A7 0.649 s |
| H6 | **CONFIRMED** | 0 failed-element uses |

## 2. Numbers

**Cells** (A7/A5 | A7/L3T | A5/L3T; and how often A7 returned L3T's output):

| set | FakeAuckland | FakeTorino | FakeKingston |
|---|---|---|---|
| F3 open | 0.840, 0.932, 1.111 (75 of 75) | 0.999, 0.994, 0.995 (24) | 1.000, 0.981, 0.981 (15) |
| F3 periodic | 0.729, 0.895, 1.227 (75 of 75) | 1.000, 1.007, 1.007 (1) | 1.000, 1.003, 1.003 (0) |
| F2 | 0.998, 0.950, 0.952 (34 of 120) | 0.996, 0.972, 0.976 (24) | 0.996, 0.970, 0.974 (24) |
| F4 | 0.999, 0.934, 0.935 (11 of 135) | 1.000, 0.947, 0.947 (2) | 1.000, 0.955, 0.955 (2) |
| F5 | 1.000, 0.932, 0.932 (0 of 72) | 1.000, 0.997, 0.997 (1) | 1.000, 0.999, 0.999 (0) |
| MODEL | 0.979, 0.725, 0.741 (58 of 153) | 0.983, 0.917, 0.933 (55) | 0.983, 0.918, 0.934 (48) |

**When A7 chose L3T's output:**

| device | chosen | better than A5 | worse than A5 | worst single ratio A7/A5 |
|---|---|---|---|---|
| FakeAuckland | 282 | 253 | 29 | 1.465 |
| FakeTorino | 129 | 102 | 26 | 1.158 |
| FakeKingston | 106 | 86 | 19 | 1.177 |

- L3T's output is usually re-placed by the state-aware search after it is chosen. A7's output equals the L3T arm's
  in 55, 82 and 73 of those cases.
- Mean infidelity gain over A5 on the chosen circuits: FakeAuckland 0.058, FakeTorino 0.0013, FakeKingston 0.0007.
- Of the circuits where A7 is worse than A5, the median ratio is 1.007-1.012.

**Reproducibility.** A5's outputs equal the ai6 run's A5 arm (Addendum 313), circuit by circuit: 2,538 of 2,538.

## 3. Reading

- **The fix works where it was aimed.**
  - On FakeAuckland, A7 returns L3T's output, usually re-placed, for every F3 circuit.
  - It then beats L3T itself by 7-10%: the state-aware re-placement improves on Qiskit's own placement.
  - The device's GAP total moves from 2.4% behind L3T to 5.1% ahead.
- **It helps model-written circuits too**, by about 2% on every device. That is where PSF-Zero's and level 3's
  compilations most often differ.
- **The cost.**
  - Where the two candidates are within a few percent, the estimate sometimes picks the worse one. This is what
    Addendum 314 found on the Heron F3 chains.
  - About 3% of circuits get worse, typically by about 1% (worst 1.47× on one FakeAuckland circuit). On average the
    gains outweigh this on every device and set.
  - Compile time rises by 7% (median), with no extra compile, from scoring one more candidate.
- **Adoption** is the owner's decision (Addendum 317). This test supports replacing a5 with a7 as the front end.

## 4. Data (`data/2026-10-02/a7/outputs/`)

- 54 job files and their logs, the prep log, the model circuits and index, `env.txt`, `run.log`, `score.md`,
  `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 317 (source: spare-qubit-cliff-addendum-317-2026-10-02.md) ===== -->

> **Note added when merging:** Adoption record: psf_ai_compile 2026-10-02.a7 becomes the AI front end (owner's decision, 2026-10-02).

## Addendum 317 -- Adoption: psf_ai_compile 2026-10-02.a7 becomes the AI front end (`benchmarks/psf_ai_compile.py`); a5 is kept as `benchmarks/psf_ai_compile_a5.py` (2026-10-02)

**Status: adoption record.**

- **Decision:** the owner adopted candidate a7 on 2026-10-02, after seeing the scored results of Addendum 316 (six of
  six confirmed).
- **Basis:** this Addendum records what changed; Addendum 316 is the basis.

## 1. What changed

- **[`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) is now a7.**
  - The code is the candidate's ([`patches/psf_ai_compile_a7_2026-10-02/psf_ai_compile.py`](../../patches/psf_ai_compile_a7_2026-10-02/psf_ai_compile.py), Addendum 315).
  - Only the comment on the `AI_COMPILE_VERSION` line changed. The version string stays "2026-10-02.a7".
- **[`benchmarks/psf_ai_compile_a5.py`](../../benchmarks/psf_ai_compile_a5.py) is new:** a byte-for-byte copy of the previous [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py)
  (a5, normalized SHA-256 as run in Addenda 290-316). This follows the frozen copies `_a0`, `_a1`, `_a2` and `_a4`.
- **Effect on reruns.** Several locked evaluation scripts load [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) as their "A5" arm:
  - `gap_eval.py`, `qml_home_eval.py`, `qml_home2_eval.py`, `ai6_eval.py`, `a7_eval.py` and the long-loop `lt_eval.py`.
  - Rerunning them at the current HEAD now runs a7 in that arm.
  - To reproduce their recorded results, check out their lock commits, or point them at `psf_ai_compile_a5.py`.
  - Their verifiers read the recorded files and are unaffected.
- **Interfaces:**
  - **Without a target:** a7 behaves exactly as a5 (checked by test).
  - **With a target:** `return_info=True` now also reports `chosen` ("PSF" or "L3T").
  - **The vLLM harness v11** (`--compiler ai --ai-module psf_ai_compile.py`) picks it up unchanged.
- **New test:** [`benchmarks/test_ai_compile_a7.py`](../../benchmarks/test_ai_compile_a7.py), adapted from the candidate's tests, with a5 taken from
  `psf_ai_compile_a5.py`.
- **`README.md`:**
  - an "Update (2026-10-02) -- AI front end a7" block;
  - the 2026-10-01 block now names `psf_ai_compile_a5.py` for a5.
- **Data:** [`data/2026-10-02/a7/outputs/`](../../data/2026-10-02/a7/outputs/) (Addendum 316) and [`benchmarks/a7_verify.py`](../../benchmarks/a7_verify.py).
- **Links:** code-formatted paths in Addenda 314-317 were turned into links in the same way as in Addendum 313,
  section 5, with the same line-by-line check.

## 2. Checks before the commit

These tests were run at home on the applied files, and the commit command was chained on them passing:

- [`benchmarks/test_ai_compile_a7.py`](../../benchmarks/test_ai_compile_a7.py);
- [`patches/psf_ai_compile_a7_2026-10-02/test_a7.py`](../../patches/psf_ai_compile_a7_2026-10-02/test_a7.py);
- [`patches/psf_ai_compile_a6_2026-10-02/test_ai6.py`](../../patches/psf_ai_compile_a6_2026-10-02/test_ai6.py).

## 3. What remains

- **Hardware:** a7 against L3T on a real device.
- **The estimate's weakness between near-equal outputs** (Addendum 314). A margin, keeping PSF-Zero's candidate
  unless L3T's is estimated better by more than some amount, could be tested.
- **In the vLLM loop:** whether a7, with or without a fast mode, changes pass rates or cost.


---

<!-- ===== Addendum 318 (source: spare-qubit-cliff-addendum-318-2026-10-03.md) ===== -->

> **Note added when merging:** Home pre-registration of HOLD (held-out circuits and devices for release 2026-10-02.2 and a7). Locked by the git commit that adds this Addendum and the evaluation scripts, pushed before the scored run. The predictions were written on the evening of 2026-10-02, before the smoke run, which is disclosed in section 5.

## Addendum 318 -- Pre-registration: HOLD. Do the two adoptions of 2026-10-02 (release 2026-10-02.2's placement_refine, AI front end a7) hold on new circuits and on six devices no compiler test has used? (2026-10-03)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document and [`benchmarks/hold_eval.py`](../../benchmarks/hold_eval.py) with its runner, pushed before the
  scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written on the evening of 2026-10-02, before the smoke run**, and were not
  changed after it.

## 1. Why

Both adoptions of 2026-10-02 were developed and judged on the same material:

- **c5 → release .2:** the GAP circuits, on FakeAuckland, FakeTorino and FakeKingston (Addenda 309-311).
- **a7:** the GAP and model-written circuits on the same three devices (Addenda 314-317).

Every claim made for them is therefore in-sample. This test asks whether those claims hold on held-out circuits and
on held-out devices.

## 2. Design (`benchmarks/hold_eval.py`)

**Circuits (held out):**

- **F1-F5:** the five GAP families, with the generator code copied verbatim from `gap_eval.family` (checked
  textually).
  - New seed base: 20,000,000 + ...; GAP used 1,000,000-5,500,000.
  - Twice GAP's per-cell sizes.
- **F6 (new):** the quantum Fourier transform (h, cp, final swaps) on a random product state, n = 4, 5, 6.
- **Size:** 1,506 circuits per device, 13,554 per arm.

**Arms** (release 2026-10-02.2 and AI front end 2026-10-02.a7, both as adopted):

| arm | what it is |
|---|---|
| C3 | `compile_for_hardware(..., target)`, placement_refine off |
| C5 | the same call with `placement_refine=True` |
| A7 | [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) (a7) with the target |
| L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

**Devices:**

| group | devices |
|---|---|
| seen | FakeAuckland (cx), FakeTorino, FakeKingston (cz) |
| new, cx | FakeHanoiV2, FakeAlgiers, FakeGeneva (27 qubits) |
| new, cz | FakeFez, FakeMarrakesh, FakeAachen (156 qubits) |

**Metric:** as in GAP. Ratios are pooled mean infidelities per device, or per cell (family, with F3 split into open
and periodic) and device.

## 3. Predictions (scored only by `hold_eval.py score`)

**P0, harness.** All of these must hold, or nothing below is scored:

- 216 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% of the circuits too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | placement_refine helps on every device | C5/C3 <= 1.00 on all 9 devices | any > 1.02 |
| H2 | ... and almost everywhere | C5/C3 <= 1.00 in >= 90% of the 63 cell-device pairs | < 75% |
| H3 | on cz devices it matches L3T on chains | chains (F3o + F5) C5/L3T <= 1.05 on all 5 cz devices | any >= 1.15 |
| H4 | a7 is level with or better than L3T | A7/L3T <= 1.00 on all 9 devices | any > 1.05 |
| H5 | a7 is at least as good as the release alone | A7/C5 <= 1.00 on all 9 devices | any > 1.03 |
| H6 | neither uses a failed element | 0 failed-edge or failed-qubit uses by C5 and A7 | any |
| H7 | a7 holds on the new family | F6: A7/L3T <= 1.05 on all 9 devices | any > 1.20 |

**Expectations, stated with the predictions:**

- **H1-H3:** placement_refine scores by reported errors. On cx devices those can fall below the T1/T2 floor (Addendum
  293), so the new cx devices are where it is most likely to fall short, as FakeAuckland did in Addendum 310.
- **H4 and H7:** a7's state-aware estimate includes the floor, so it should hold on the cx devices too.
- **H7 is the least certain.** QFT needs much routing, and level 3 may route it better than PSF-Zero.

**Reported without prediction:**

- the device and cell tables;
- how often A7 chose L3T's output;
- compile times;
- off-target instructions.

## 4. What this will not establish

- Hardware.
- ecr devices (the harness supports cx and cz only).
- The model-written circuits (none are held out).

## 5. Development (disclosed)

### 5.1 Smoke run (not a result)

The smoke run used 1 circuit per cell and its own seed base: 684 compilations, 216 jobs, 257 s, on the morning of
2026-10-03. Nothing was changed after it.

- **P0** passed (noiseless infidelity max 8.2e-15).
- **Its verdict lines:** all seven CONFIRMED.

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
  |---|---|---|---|---|---|---|
  | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

- **Device level:**
  - C5/C3 0.60-0.90;
  - A7/L3T 0.91-0.99;
  - A7/C5 0.89-0.97;
  - chains C5/L3T on cz devices 0.999-1.039.
- **On the new cx devices**, chains C5/L3T were 1.07-1.17, as expected for reported-error placement on cx devices.
- **One observation outside the predictions:** the C3 arm (release with `target`, placement_refine off) used a
  failed element 33 times. Item 31 is meant to prevent that, and it did so on the three seen devices in Addenda 304,
  307 and 310. The smoke output does not say on which device it happened.
  - It does not bear on H6, which concerns C5 and A7 (both 0).
  - The scored data will be examined for it, and the result reported with this test's results.
- **Median compile time:** C3 0.036 s, C5 0.039 s, A7 0.706 s, L3T 0.018 s.

### 5.2 Other development

- The scorer was run on synthetic files built from the c5 and a7 runs.
- The F1-F5 generator code was checked textually against `gap_eval.family`: identical.
- No scored circuit was compiled before the lock.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/hold_eval.py`](../../benchmarks/hold_eval.py) | `bd087fa5c922653baa2b3311792d91ce0405a9ceea9a4829870fe5a2253afaf8` |
| [`benchmarks/run_hold_2026-10-03.sh`](../../benchmarks/run_hold_2026-10-03.sh) | `45a2af760aa003bf81fd37fa5e77422fea03f7b0b8b3e6090e88004925019786` |


---

<!-- ===== Addendum 319 (source: spare-qubit-cliff-addendum-319-2026-10-03.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 318 (lock commit 35637f9), scored by the locked script and re-checked by benchmarks/hold_verify.py, written after the locked score was seen.

## Addendum 319 -- Results: HOLD (Addendum 318). All seven confirmed on held-out circuits and six new devices: release 2026-10-02.2's placement_refine improves on every device (C5/C3 0.62-0.90, stronger on the new devices than on the seen ones), matches Qiskit L3T on chains on all five cz devices, and the AI front end a7 beats L3T on all nine devices (0.94-0.99) and on the new QFT family. The C3 arm's 4,039 "failed" uses are one-way coupler failures on FakeHanoiV2, routed in the healthy direction (2026-10-03)

**Status: results of the pre-registered test in Addendum 318.**

- **Lock:** commit `35637f9` (07:53 JST), pushed before the scored run.
- **Scoring:** by the locked `hold_eval.py score`, and re-checked by [`benchmarks/hold_verify.py`](../../benchmarks/hold_verify.py). That script was
  written after the locked score was seen.
- **Setting:** home (WSL2), 6 processes; 216 jobs, 54,216 circuit compilations, 2,502 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 216 of 216 files; noiseless infidelity max 2.9e-9; 0 too wide |
| H1 | **CONFIRMED** | C5/C3 on all 9 devices: 0.617-0.901 |
| H2 | **CONFIRMED** | C5/C3 <= 1.00 in 62 of 63 cell-device pairs (98.4%; the exception is FakeAlgiers F3 periodic, 1.041) |
| H3 | **CONFIRMED** | chains C5/L3T on the cz devices: Torino 0.997, Kingston 1.000, Fez 0.993, Marrakesh 1.022, Aachen 0.995 |
| H4 | **CONFIRMED** | A7/L3T on all 9 devices: 0.943-0.987 |
| H5 | **CONFIRMED** | A7/C5 on all 9 devices: 0.880-0.962 |
| H6 | **CONFIRMED** | 0 failed-element uses by C5 and by A7 |
| H7 | **CONFIRMED** | F6 (QFT) A7/L3T on all 9 devices: 0.653-0.982 |

## 2. Numbers

**By device:**

| device | C5/C3 | C5/L3T | A7/L3T | A7/C5 | C3/L3T | chains C5/L3T | A7 chose L3T |
|---|---|---|---|---|---|---|---|
| FakeAuckland | 0.901 | 1.056 | 0.945 | 0.894 | 1.172 | 1.097 | 470 |
| FakeTorino | 0.761 | 1.024 | 0.985 | 0.962 | 1.347 | 0.997 | 155 |
| FakeKingston | 0.866 | 1.041 | 0.982 | 0.943 | 1.202 | 1.000 | 114 |
| FakeHanoiV2 (new) | 0.853 | 1.065 | 0.981 | 0.921 | 1.249 | 1.141 | 587 |
| FakeAlgiers (new) | 0.811 | 1.109 | 0.976 | 0.880 | 1.367 | 1.196 | 513 |
| FakeGeneva (new) | 0.721 | 1.010 | 0.943 | 0.934 | 1.402 | 1.078 | 515 |
| FakeFez (new) | 0.667 | 1.041 | 0.987 | 0.948 | 1.561 | 0.993 | 102 |
| FakeMarrakesh (new) | 0.617 | 1.026 | 0.963 | 0.939 | 1.663 | 1.022 | 112 |
| FakeAachen (new) | 0.689 | 1.056 | 0.983 | 0.931 | 1.533 | 0.995 | 130 |

**Seen against new devices** (pooled):

| group | C5/C3 | C5/L3T | A7/L3T | A7/C5 |
|---|---|---|---|---|
| seen (3 devices) | 0.849 | 1.044 | 0.963 | 0.922 |
| new (6 devices) | 0.747 | 1.056 | 0.970 | 0.919 |

**Per circuit:**

- A7 <= L3T in 77.5-92.4% of circuits, by device.
- C5 <= C3 in 94.3-100% of circuits, except FakeAlgiers (74.2%).

The full cell table is in `outputs/score.md`.

## 3. Reading

- **The claims made in-sample on 2026-10-02 hold out of sample.**
  - New seeds, a new family and six devices that no compiler test had used change none of the seven verdicts.
- **placement_refine helps more on the new devices than on the seen ones** (0.747 against 0.849).
  - On every cz device it brings chains to L3T's level (0.99-1.02).
  - On the cx devices a chain gap to L3T remains (1.08-1.20), as on FakeAuckland (Addendum 310). The release scores
    by reported errors, which on cx devices can lie below the T1/T2 floor (Addendum 293).
- **The AI front end a7 is ahead of L3T on every device**, by 1.3-5.7%, and ahead of the release alone by 3.8-12%.
  - The new QFT family is where it leads most, up to 0.65 × L3T on FakeGeneva.
  - It chose L3T's own output in 102-587 circuits per device, most often on the cx devices.
- **Where to look next:**
  - Per circuit, A7 still loses to L3T in 8-22% of circuits. On average those losses are outweighed.
  - FakeAlgiers is the one device where placement_refine makes a quarter of circuits slightly worse (C5 <= C3 in
    74.2%), although it helps there on average (0.811).

## 4. The C3 arm's flagged uses (not a prediction)

The smoke run showed the C3 arm (release with `target`, placement_refine off) touching a failed element 33 times
(Addendum 318, section 5.1); the scored run shows 4,039 gates in 191 circuits.

**Where.** All are on FakeHanoiV2, in F2 (2), F3 periodic (150) and F4 (39).

**What those couplers are.** FakeHanoiV2's snapshot reports two one-way failures (read from the A0 data, Addendum
293):

- **cx(5, 8) error 1.0, but cx(8, 5) 0.009;**
- **cx(19, 20) error 1.0, but cx(20, 19) 0.004.**

FakeGeneva has two more such one-way entries, (16, 14) and (20, 19). FakeAlgiers has a two-way failure on (15, 18).
The data do not say which of FakeHanoiV2's two couplers each circuit touched.

**Why C3 touched it:**

- Item 31's `prune_coupling_map` removes a directed edge only when that direction is reported failed. This is
  documented: "the error ... on (a, b) -- or on (b, a) if (a, b) is not listed".
- So each such coupler stays usable in its healthy direction.
- This harness counts a coupler as failed if either direction is reported failed, so it flagged those uses.
- The release's own check (`_uses_failed`) is also direction-agnostic. It therefore triggers a recompile that
  still uses the healthy direction: wasted work, not a wrong result.

**Evidence that only the healthy direction was used.**

- The flagged C3 circuits have infidelity in the normal range: F3 periodic mean 0.577, against 0.487 for C5 and 0.381
  for L3T on the same circuits.
- 24 uses of a gate with error 1.0 would leave the six qubits nearly fully mixed (infidelity about 0.98).

**Consequence.**

- No physically failed gate was used by any arm.
- Whether a one-way failure should retire the whole coupler is a policy question for a later candidate. Making
  `_uses_failed` direction-aware would remove the wasted recompile.
- C5 and A7 avoid the coupler entirely (0 uses under the stricter count).

## 5. Data (`data/2026-10-03/hold/outputs/`)

- 216 job files and their logs, `env.txt`, `run.log`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 320 (source: spare-qubit-cliff-addendum-320-2026-10-03.md) ===== -->

> **Note added when merging:** Home pre-registration of HOLD2: candidate psf_compile 2026-10-03.c6 (re-placement scored by max(reported error, T1/T2 floor)) on fresh held-out circuits and HOLD's nine devices. Locked by the git commit that adds this Addendum, the candidate patch with its tests and the evaluation scripts, pushed before the scored run. The predictions were written before the smoke run, which is disclosed in section 5.

## Addendum 320 -- Pre-registration: candidate psf_compile 2026-10-03.c6 (floor-aware re-placement score) on fresh held-out circuits (HOLD2). Does scoring by max(reported error, T1/T2 floor) close the release's chain gap on cx devices without costing anything on cz devices? (2026-10-03)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c6_2026-10-03/psf_compile.py`](../../patches/psf_compile_c6_2026-10-03/psf_compile.py), with its tests) and [`benchmarks/hold2_eval.py`](../../benchmarks/hold2_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before the smoke run.**

## 1. The candidate (changelog item 34)

`compile_for_hardware(..., target=..., placement_refine=True, placement_score="floor")`:

- **What changes.** The re-placement of item 33 is scored on a copy of the Target in which every instruction's error
  is max(reported error, decoherence floor).
- **The floor:** the average gate infidelity of thermal relaxation on the gate's qubits for its duration, with T2
  capped at 2 T1. It is the closed form the AI front end has used since a3.
- **What does not change:** the compile, the routing and item 31's backstop all use the original Target.
- **The copy** is built per call and never cached.
- **Default:** `placement_score="reported"` is identical to release 2026-10-02.2, as checked by test.

**Why:**

- On the cx devices the release still trails Qiskit L3T on chains (FakeAuckland 1.10 in Addendum 310; 1.08-1.20 on
  four cx devices in Addendum 319).
- On those devices reported errors often lie below the T1/T2 floor (Addendum 293).
- Addendum 308 found the floor-aware score ordering arms better than the reported one only on FakeAuckland.

## 2. Design (`benchmarks/hold2_eval.py`)

**Circuits (held out again):**

- `hold_eval`'s families F1-F6 with the same per-cell sizes: 1,506 per device. The generator code was checked
  textually against `hold_eval.py`.
- New seed base 30,000,000 + ... (HOLD used 20,000,000 + ..., GAP 1,000,000-5,500,000).

**Arms:**

| arm | what it is |
|---|---|
| C5 | release 2026-10-02.2, `target`, `placement_refine=True` |
| C6 | the candidate, the same call plus `placement_score="floor"` |
| A7 | the adopted AI front end |
| L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

**Devices:** HOLD's nine.

| type | devices |
|---|---|
| cx | FakeAuckland, FakeHanoiV2, FakeAlgiers, FakeGeneva |
| cz | FakeTorino, FakeKingston, FakeFez, FakeMarrakesh, FakeAachen |

**Metric:** as in GAP.

**Failed-element uses** are counted two ways:

- by coupler: either direction reported failed, as in HOLD;
- by direction: the gate's own direction reported failed, which is what matters physically (Addendum 319, section 4).

## 3. Predictions (scored only by `hold2_eval.py score`; written before the smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- 216 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% of the circuits too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the floor score never costs on average | C6/C5 <= 1.00 on all 9 devices | any > 1.02 |
| H2 | it helps on cx devices | C6/C5 <= 0.98 on at least 3 of the 4 cx devices | > 1.00 on 2 or more cx devices |
| H3 | it is neutral on cz devices | C6/C5 within 0.98-1.02 on all 5 cz devices | any outside 0.95-1.05 |
| H4 | it closes most of the cx chain gap | chains (F3o + F5) C6/L3T <= 1.05 on at least 3 of 4 cx devices | >= 1.10 on all 4 |
| H5 | it never uses a failed direction | 0 failed-direction or failed-qubit uses by C6 | any |
| H6 | it stays cheap | median compile time C6 <= 3 × C5 | > 10 × C5 |
| H7 | the cx gain is broad | on the cx devices, C6/C5 <= 1.00 in >= 80% of the 28 cell-device pairs | < 50% |

**Reported without prediction:**

- the device and cell tables, including A7 against C6 (how much of the AI front end's lead the release closes);
- compile times;
- failed uses counted both ways;
- off-target instructions.

**H7 restriction.** H7 was restricted to the cx devices before the smoke run. The scorer had been run on synthetic
files, where cz cells, expected to be near 1.00, would have made an all-device threshold depend on ties.

## 4. What this will not establish

- Hardware.
- ecr devices.
- Whether the floor is the right model for real devices. It is the model Aer uses, so this simulation favours it by
  construction, as for a4 (Addendum 287).

## 5. Development (disclosed)

### 5.1 How the candidate was built

- `psf_compile.py` was generated from release 2026-10-02.2 by a script that inserts the two new functions
  (`decoherence_floor`, `floor_aware_target`), the `placement_score` parameter and its check, and changelog item 34.
  Nothing else in the release was changed.
- The scorer was run on synthetic files before the smoke run. That is why H7 was restricted to the cx devices
  (section 3).

### 5.2 Tests (`test_c6_floor.py`, 10 cases)

All 10 passed at home in 4.8 s, before the smoke run. They check:

- the version strings;
- that the default (`placement_score="reported"`) gives the same output as the release, with and without `target`
  and `placement_refine`, on FakeTorino and FakeAuckland;
- that an unknown `placement_score` raises `ValueError`;
- that `floor_aware_target` sets each error to max(reported, floor), leaves measure/delay/reset/barrier alone, does
  not modify the original Target, and raises at least one error on FakeAuckland and FakeHanoiV2;
- that the floor re-placement is exact, keeps the same gate multiset, and places no gate on a failed qubit or in a
  direction reported failed, on FakeAuckland, FakeHanoiV2, FakeTorino and FakeKingston.

### 5.3 Smoke run (not a result)

The smoke run used 1 circuit per cell and its own seed base: 684 compilations, 216 jobs, 259 s, at home on the
morning of 2026-10-03. Nothing was changed after it.

- **P0** passed (noiseless infidelity max 6.8e-15; 0 too wide).
- **Its verdict lines:**

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
  |---|---|---|---|---|---|---|
  | CONFIRMED | AMBIGUOUS | CONFIRMED | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED |

- **Device level:**
  - C6/C5: 0.975-0.990 on the cx devices, 0.997-1.000 on the cz devices.
  - Chains C6/L3T on the cx devices: Auckland 1.057, HanoiV2 1.142, Algiers 1.033, Geneva 0.958 (C5/L3T: 1.077,
    1.141, 1.119, 1.055).
  - A7/C6 0.930-0.965.
- **What the smoke run suggests, with one circuit per cell:**
  - the gain on the cx devices is real but smaller than H2's 0.98 threshold on three of them;
  - on FakeHanoiV2 the floor score does not move the chain gap at all (1.142 against 1.141).
  - The predictions were not changed.
- **Median compile time:** C5 0.036 s, C6 0.063 s, A7 0.724 s, L3T 0.018 s.
  - The extra time is the per-call copy of the Target.
- **Failed uses:** 0 by every arm, counted by coupler and by direction.
- **Off-target instructions:** 0 by every arm.

### 5.4 Other

- No scored circuit was compiled before the lock.
- The candidate is a patch ([`patches/psf_compile_c6_2026-10-03/`](../../patches/psf_compile_c6_2026-10-03/)). The released `psf_compile.py` is not changed by
  this commit.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c6_2026-10-03/psf_compile.py`](../../patches/psf_compile_c6_2026-10-03/psf_compile.py) | `c40e1bf133e0d3e152e775bd7db68bf92161265b4ad77ba5ccd5344f4492f0bb` |
| [`patches/psf_compile_c6_2026-10-03/test_c6_floor.py`](../../patches/psf_compile_c6_2026-10-03/test_c6_floor.py) | `bdada060f30e3c195d5c011de172d434d26ce822748c51faf378468022c17a78` |
| [`benchmarks/hold2_eval.py`](../../benchmarks/hold2_eval.py) | `2802a86512bf72421cf1b7001fc7441415ff0f8b38fc883de13ac4598c22b062` |
| [`benchmarks/run_hold2_2026-10-03.sh`](../../benchmarks/run_hold2_2026-10-03.sh) | `a9584adb1abf8bcda548265d899d4bc35c3d1018bbe2c7ea80ba6f41d733eba7` |

Normalization: CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines joined
with "\n" and no final newline.


---

<!-- ===== Addendum 321 (source: spare-qubit-cliff-addendum-321-2026-10-03.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 320 (lock commit 2cf5ca8), scored by the locked script and re-checked by benchmarks/hold2_verify.py, written after the run finished and before any of its output was seen.

## Addendum 321 -- Results: HOLD2 (Addendum 320). The floor-aware re-placement score (candidate c6) is safe and helps on the cx devices, but less than predicted, and it does not close the chain gap to Qiskit L3T: four confirmed (H3, H5, H6, H7), three ambiguous (H1, H2, H4), none refuted. The chain gap on the cx devices lies in the open-boundary F3 cell, where the floor score changes nothing on three of four devices; on the GHZ-type F5 cell, where the release and L3T give identical circuits, the floor score alone improves FakeGeneva by 28% (2026-10-03)

**Status: results of the pre-registered test in Addendum 320.**

- **Lock:** commit `2cf5ca8` (11:30 JST), pushed before the scored run.
- **Scoring:** by the locked `hold2_eval.py score`, and re-checked by [`benchmarks/hold2_verify.py`](../../benchmarks/hold2_verify.py). That script was
  written after the run finished and before any of its output was seen.
- **Setting:** home (WSL2), 6 processes; 216 jobs, 54,216 circuit compilations, about 40 minutes (job times sum to
  14,358 s).

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 216 of 216 files; noiseless infidelity max 2.1e-9; 0 too wide |
| H1 | **AMBIGUOUS** | C6/C5 <= 1.00 on 8 of 9 devices; FakeTorino 1.0000145 (none > 1.02) |
| H2 | **AMBIGUOUS** | C6/C5 on the cx devices: Auckland 0.984, HanoiV2 0.996, Algiers 0.970, Geneva 0.994 (one <= 0.98; none > 1.00) |
| H3 | **CONFIRMED** | C6/C5 on the cz devices: Torino 1.000, Kingston 0.998, Fez 0.999, Marrakesh 0.992, Aachen 1.000 |
| H4 | **AMBIGUOUS** | chains C6/L3T on the cx devices: Auckland 1.089, HanoiV2 1.138, Algiers 1.101, Geneva 1.036 (one <= 1.05; not all >= 1.10) |
| H5 | **CONFIRMED** | 0 failed-direction or failed-qubit uses by C6 (0 by every arm, counted either way) |
| H6 | **CONFIRMED** | median compile time C5 0.026 s, C6 0.050 s (1.9 ×) |
| H7 | **CONFIRMED** | C6/C5 <= 1.00 in 26 of the 28 cx cell-device pairs (92.9%; the exceptions are FakeHanoiV2 F5 1.0008 and F6 1.0044) |

**On H1.** FakeTorino's ratio is 1.0000145: C6 differs from C5 in 21 of 1,506 circuits there (15 worse, 6 better).
The prediction's bound was "<= 1.00", so the locked scorer reports AMBIGUOUS. That verdict stands. In substance the
floor score cost nothing measurable on any device.

## 2. Numbers

**By device:**

| device | C6/C5 | C6/L3T | C5/L3T | chains C6/L3T | chains C5/L3T | A7/C6 | A7/L3T |
|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.984 | 1.039 | 1.056 | 1.089 | 1.097 | 0.909 | 0.944 |
| FakeTorino | 1.000 | 1.025 | 1.025 | 0.996 | 0.996 | 0.961 | 0.985 |
| FakeKingston | 0.998 | 1.037 | 1.039 | 0.982 | 0.998 | 0.945 | 0.981 |
| FakeHanoiV2 (cx) | 0.996 | 1.059 | 1.064 | 1.138 | 1.140 | 0.924 | 0.979 |
| FakeAlgiers (cx) | 0.970 | 1.077 | 1.111 | 1.101 | 1.194 | 0.908 | 0.978 |
| FakeGeneva (cx) | 0.994 | 1.006 | 1.011 | 1.036 | 1.077 | 0.940 | 0.945 |
| FakeFez | 0.999 | 1.040 | 1.040 | 0.991 | 0.991 | 0.948 | 0.986 |
| FakeMarrakesh | 0.992 | 1.017 | 1.026 | 0.963 | 1.022 | 0.943 | 0.960 |
| FakeAachen | 1.000 | 1.055 | 1.055 | 0.992 | 0.992 | 0.930 | 0.981 |

**Per circuit, C6 against C5:**

| device | same result | C6 better | C6 worse |
|---|---|---|---|
| FakeAuckland | 48.1% | 42.0% | 9.8% |
| FakeHanoiV2 | 61.2% | 25.0% | 13.9% |
| FakeAlgiers | 49.6% | 48.1% | 2.3% |
| FakeGeneva | 96.7% | 3.3% | 0.0% |
| FakeTorino | 98.6% | 0.4% | 1.0% |
| FakeKingston | 89.7% | 9.9% | 0.4% |
| FakeFez | 95.7% | 3.1% | 1.2% |
| FakeMarrakesh | 78.1% | 20.8% | 1.1% |
| FakeAachen | 100% | 0% | 0% |

**Chain cells on the cx devices:**

| device | F3 open: C6/C5 | F3 open: C6/L3T | F5: C6/C5 | F5: C5/L3T |
|---|---|---|---|---|
| FakeAuckland | 1.000 | 1.111 | 0.934 | 1.000 |
| FakeHanoiV2 | 0.999 | 1.158 | 1.001 | 1.000 |
| FakeAlgiers | 0.921 | 1.126 | 0.931 | 1.000 |
| FakeGeneva | 1.000 | 1.091 | 0.721 | 1.000 |

The full cell table is in `outputs/score.md`; the re-computation is in `outputs/verify.txt`.

## 3. Reading

**What c6 does well:**

- **It is safe.**
  - No device got measurably worse.
  - On the cz devices it changes 0-22% of circuits and the pooled ratio by at most 0.8%.
  - It uses no failed element.
  - It roughly doubles a 26 ms compile.
- **It helps on all four cx devices** (1.6% on FakeAuckland, 3.0% on FakeAlgiers), and in 26 of 28 cx cells.
- **F5 isolates the effect of the score.**
  - On the cx devices the release (C5) and L3T give identical results on every F5 circuit (C5/L3T exactly 1.000; the
    same holds in the HOLD data). Both end with an exact VF2PostLayout on reported errors, and the GHZ-type circuit
    needs no routing.
  - Scoring the same placement step by the floor instead improves F5 by 6.6% on FakeAuckland, 6.9% on FakeAlgiers and
    28% on FakeGeneva.
  - On these devices, where reported errors lie below the T1/T2 floor, the reported-error score is the wrong
    objective for Aer's noise model. This is a direct demonstration of that.

**What c6 does not do:**

- **H2's 2% gain was too optimistic** for three of the four cx devices.
- **It does not close the chain gap.**
  - The gap to L3T on the cx devices sits mainly in the open-boundary F3 cell (1.09-1.16).
  - There the floor score changes nothing on FakeAuckland, FakeGeneva and FakeHanoiV2 (C6/C5 0.999-1.000).
  - So that gap is not a placement-scoring problem. The likely suspects are routing or synthesis on F3's
    random-unitary chains; this is to be diagnosed, not assumed.
- **FakeHanoiV2 is the exception.**
  - The floor score helps there in a quarter of circuits but loses in 14%.
  - On F5 and F6 it is slightly worse (1.0008, 1.0044).
  - Why is not known. FakeHanoiV2 is the device with two one-way failed couplers (Addendum 319, section 4).

**Where this leaves the AI front end:** a7 is still ahead of C6 by 4-9% on every device (A7/C6 0.908-0.961).

## 4. Consequences

**Adoption.** Whether to adopt the candidate is the owner's decision. The data support:

- `placement_score="floor"` is safe to offer;
- it is worth using on cx devices, and neutral on cz devices.

The data do not support:

- the claim that it closes the chain gap;
- making it the default on the strength of H2 or H4.

**Next diagnosis.** The F3 open-boundary gap on the cx devices: compare C6 and L3T on the same circuits for
placement, routing (swap count) and synthesis (two-qubit gate count).

## 5. Data (`data/2026-10-03/hold2/outputs/`)

- 216 job files and their logs, `env.txt`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 322 (source: spare-qubit-cliff-addendum-322-2026-10-03.md) ===== -->

> **Note added when merging:** Exploratory diagnosis (not a test, nothing pre-registered) of the chain gap left by Addendum 321, run at home at commit ebc9280. Scripts and outputs are in data/2026-10-03/f3o/diag/.

## Addendum 322 -- Diagnosis (exploratory, not a test): the release's chain gap to Qiskit L3T on the cx devices is a thermal-relaxation effect of PSF-Zero's own two-qubit synthesis. On the F3 open-boundary circuits the release and L3T use the same qubits and the same 60 cx gates with the same errors; the release leaves qubits excited for longer during the long cx gates. Re-synthesising every block with Qiskit removes 70-90% of that gap (2026-10-03)

**Status: exploratory diagnosis.**

- **Nothing was pre-registered**, and nothing here is a verdict.
- **The scripts** were written after HOLD2's results (Addendum 321) had been seen. Each part's design followed from the
  previous part's output.
- **Setting:** run at home on 2026-10-03 at commit `ebc9280` (Addendum 321), fake devices and Aer only.
- **Circuits:** HOLD2's F3 circuits (same generator and seeds):
  - parts 1-2: the 150 open-boundary circuits per device;
  - part 3: all 300.
- **Devices:** FakeAuckland, FakeHanoiV2, FakeAlgiers and FakeGeneva (cx), with FakeTorino (cz) as a control.
- **Reproduction:** in parts 1 and 2, every C5, C6 and L3T row reproduced HOLD2's two-qubit count and noisy infidelity
  exactly (0 differences on every device).

## 1. Part 1: it is not placement (`f3o_diag.py`)

**Arms:** C5, C6 and L3T, plus two swaps of placement:

- **L3onC6:** Qiskit level 3 pinned to C6's layout.
- **RELonL3:** the release pinned to L3T's layout.

**Ratio to L3T on F3 open:**

| device | C5 | C6 | L3onC6 | RELonL3 |
|---|---|---|---|---|
| FakeAuckland | 1.111 | 1.111 | 1.001 | 1.114 |
| FakeHanoiV2 | 1.159 | 1.158 | 1.018 | 1.158 |
| FakeAlgiers | 1.223 | 1.126 | 1.046 | 1.226 |
| FakeGeneva | 1.091 | 1.091 | 1.001 | 1.092 |
| FakeTorino | 0.996 | 0.996 | 0.996 | 0.993 |

**Same qubits, same cx gates.**

- On FakeAuckland, FakeHanoiV2 and FakeGeneva, C5 and L3T use the same six qubits in all 150 circuits.
- On those three devices the 60 two-qubit gates have the same mean applied error on FakeAuckland and FakeGeneva,
  and nearly the same on FakeHanoiV2 (0.00558 against 0.00550).

**FakeAlgiers is the exception.** There C6 chooses another qubit set (150 of 150 circuits), which is why the floor
score helped there in HOLD2 (1.223 to 1.126).

**The gap follows the synthesis, not the placement.**

- With L3T's layout, the release is as far behind as before.
- With C6's layout, Qiskit is level with L3T, or within 5%.

**Where the outputs differ.** Only in one-qubit gates:

- sx + x: 99.3 against about 82;
- depth: 91.9 against about 73;
- the summed average gate infidelity: by only 0.003-0.006, against measured differences of 0.025-0.070.

## 2. Part 2: it is thermal relaxation (`f3o_diag2.py`)

**Method:**

- Each output was simulated under the full noise model, under its depolarizing part alone
  (`thermal_relaxation=False`), and under its thermal-relaxation part alone (`gate_error=False`).
- Each output was also scored by its excitation exposure: the sum over two-qubit gates of the gate's duration times
  P(1) on each of its qubits, with P(1) taken from the noiseless state just before the gate.

**Results, F3 open, ratio C5/L3T:**

| device | full | depolarizing only | thermal only | x gates C5 / L3T | 2q exposure C5 / L3T | correlation* |
|---|---|---|---|---|---|---|
| FakeAuckland | 1.111 | 1.004 | 1.346 | 11.3 / 1.8 | 1.49 | 0.719 |
| FakeHanoiV2 | 1.159 | 1.009 | 1.193 | 11.3 / 2.0 | 1.38 | 0.630 |
| FakeAlgiers | 1.223 | 1.009 | 1.281 | 11.3 / 1.7 | 1.42 | 0.821 |
| FakeGeneva | 1.091 | 1.003 | 1.406 | 11.3 / 2.1 | 1.47 | 0.775 |
| FakeTorino | 0.996 | 1.000 | 0.978 | 1.4 / 1.4 | 0.97 | 0.955 |

\* Per-circuit correlation of the exposure difference with the infidelity difference (C5 minus L3T).

**What this shows:**

- **Under the depolarizing part alone** the two compilers are level to within 1%.
- **The whole gap is thermal relaxation.** The release's circuits keep qubits in |1> for longer during the cx gates,
  which last hundreds of ns on these devices. During that time amplitude damping acts on the excited population.
- **An average-infidelity sum cannot see this** by construction. On the cz device (short cz gates) the effect is absent.
- **Neither of these changes anything** (ratios within 0.002):
  - a one-qubit clean-up after the release (`Optimize1qGatesDecomposition`);
  - re-synthesis of blocks only where it saves gates.

  The extra one-qubit gates are not redundant. They are the local frames the release's synthesis chose around each
  cx.

## 3. Part 3: which stage, and does Qiskit's synthesis remove it? (`f3o_diag3.py`)

**Arms on all 300 F3 circuits:**

- **C5force:** C5's output with every two-qubit block re-synthesised by Qiskit (`ConsolidateBlocks(force_consolidate=True)`,
  `UnitarySynthesis`, `Optimize1qGatesDecomposition`, all with the target).
- **C5noc2:** the release with items 29-30 off.
- **C5can:** the release with `entangling_basis="canonical"`.

**Ratio to L3T, full noise:**

| device | open: C5 | open: C5force | open: C5noc2 | open: C5can | periodic: C5 | periodic: C5force |
|---|---|---|---|---|---|---|
| FakeAuckland | 1.111 | 1.018 | 1.111 | 1.533 | 1.237 | 1.194 |
| FakeHanoiV2 | 1.159 | 1.041 | 1.159 | 1.460 | 1.268 | 1.174 |
| FakeAlgiers | 1.223 | 1.063 | 1.223 | 1.486 | 1.530 | 1.408 |
| FakeGeneva | 1.091 | 1.012 | 1.091 | 1.639 | 1.161 | 1.120 |
| FakeTorino | 0.996 | 0.996 | 0.996 | 1.554 | 1.012 | 1.002 |

**Per circuit:** C5force is better than C5 on the cx devices in 96-100% of the open and 97-100% of the periodic
circuits. On FakeTorino it is mixed (50% open, 81% periodic).

**Reading:**

- **Items 29-30 are not the cause.**
  - C5noc2 is identical to C5 on every open circuit.
  - On the periodic circuits it is slightly worse (117 against 114 two-qubit gates), so those items help there.
- **The canonical route is no fix.** It doubles the two-qubit count (120 and 186). Direct cx synthesis is right.
- **The excitation comes from PSF-Zero's own two-qubit synthesis**, through the local frames it chooses.
- **On open chains, re-synthesis by Qiskit removes 70-90% of the gap** (72-87%).
  - x gates fall from 11.3 to about 3.
  - The thermal-only ratio falls to 1.01-1.06.
  - FakeTorino is unchanged.
- **On periodic chains it removes only a fifth to a third of the gap** (18-35%).
  - Routing is involved there.
  - C5force has far more sx gates than L3T (144 against 96-122).
  - Something else in the routed circuits also matters.

## 4. Consequences

**For the open chains, the cause is found:**

- PSF-Zero's two-qubit synthesis on cx devices chooses local frames that leave qubits excited during long cx gates.
- Reported errors cannot penalize this; average gate infidelity hides it.
- The AI front end a7 often chooses L3T's output on the cx devices (447-593 circuits per device in HOLD2). Its
  state-aware estimate may already capture this effect. That reading is not verified.

**Candidate.** A candidate that re-synthesises every two-qubit block with Qiskit after the release on cx devices
(here, C5force) is worth a pre-registered test. That test needs:

- held-out circuits beyond F3;
- a check that it costs nothing on the families where PSF-Zero's synthesis is ahead.

**What would go further:**

- **Exposure-aware synthesis:** choosing among equivalent local frames to minimise excitation exposure.
- **Re-placing with an exposure-aware score.**

**Not established:**

- hardware (whether the effect matters as much as Aer's model says);
- ecr devices;
- circuits other than F3.

## 5. Data (`data/2026-10-03/f3o/diag/`)

- **Scripts:** `f3o_diag.py`, `f3o_diag2.py`, `f3o_diag3.py` and their runners.
- **Outputs:** `outputs1/`, `outputs2/`, `outputs3/`, each with the per-device json, logs, `env.txt` and
  `summary.md`.


---

<!-- ===== Addendum 323 (source: spare-qubit-cliff-addendum-323-2026-10-03.md) ===== -->

> **Note added when merging:** Home pre-registration of HOLD3: candidate psf_compile 2026-10-03.c8 (final two-qubit re-synthesis by Qiskit, kept per circuit by an excitation-aware estimate) on fresh held-out circuits and HOLD's nine devices. Locked by the git commit that adds this Addendum, the candidate patch with its tests and the evaluation scripts, pushed before the scored run. The predictions were written before c8's smoke run and after the smoke run of an earlier, never-locked candidate c7; both are disclosed in section 5.

## Addendum 323 -- Pre-registration: candidate psf_compile 2026-10-03.c8 (final two-qubit re-synthesis by Qiskit, kept only when an excitation-aware estimate says it helps) on fresh held-out circuits (HOLD3). Can the release keep the chain gain found in Addendum 322 without the losses that unconditional re-synthesis brought on other families? (2026-10-03)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c8_2026-10-03/psf_compile.py`](../../patches/psf_compile_c8_2026-10-03/psf_compile.py), with its tests) and [`benchmarks/hold3_eval.py`](../../benchmarks/hold3_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before c8's smoke run.** They were written after the smoke run of an
  earlier candidate, c7, which is disclosed in section 5.1 and is the reason c8 exists.

## 1. The candidate (changelog item 35)

`compile_for_hardware(..., target=..., placement_refine=True, final_resynthesis="select")`:

- **Re-synthesis.** The finished circuit (after item 31's backstop) is passed through:
  - `ConsolidateBlocks(force_consolidate=True)`;
  - `UnitarySynthesis`;
  - `Optimize1qGatesDecomposition`.

  All three use the target and are exact (approximation_degree 1.0). Every two-qubit block is therefore
  re-synthesised by Qiskit.
- **The layout** is carried over unchanged.
- **Backstop:** a result with an off-target instruction, or a two-qubit gate in a failed direction, is refused.
- **Selection (`"select"`).** Both circuits are scored by `excitation_cost`, and the lower one is kept. The score is:
  - the summed -log(1 - reported error) of the gates as placed;
  - plus, for every gate, duration / T1 times P(1) on each of its qubits. P(1) is taken from the noiseless state just
    before the gate: the population that amplitude damping acts on.
- **When the estimate cannot be made.** Above 16 touched qubits, or with an instruction that has no matrix, the
  release's circuit is kept.
- **Modes:**

  | value | behaviour |
  |---|---|
  | `True` | always re-synthesise (c7's behaviour) |
  | `False` (default) | identical to release 2026-10-02.2, as checked by test |

- **Base:** release 2026-10-02.2. It does not contain the held candidate c6 (item 34).

**Why.** Addendum 322 traced the release's chain gap on cx devices to thermal relaxation. With the same qubits and
the same cx gates, PSF-Zero's synthesis leaves qubits excited for longer during the long cx gates. Qiskit's
re-synthesis removed most of that gap on F3. Section 5.1 shows why "always" is not enough.

## 2. Design (`benchmarks/hold3_eval.py`)

**Circuits (held out again):**

- `hold_eval`'s families F1-F6 with the same per-cell sizes: 1,506 per device. The generator code is HOLD2's,
  unchanged (checked textually).
- New seed base 40,000,000 + ... (HOLD2 used 30,000,000, HOLD 20,000,000, GAP 1,000,000-5,500,000).

**Arms:**

| arm | what it is |
|---|---|
| C5 | release 2026-10-02.2, `target`, `placement_refine=True` |
| C7F | the candidate with `final_resynthesis=True`: c7's behaviour, kept as the counterfactual for the selection |
| C8 | the candidate with `final_resynthesis="select"`: the candidate as proposed |
| A7 | the adopted AI front end |
| L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

**Size:** 270 jobs.

**Devices:** HOLD's nine.

| type | devices |
|---|---|
| cx | FakeAuckland, FakeHanoiV2, FakeAlgiers, FakeGeneva |
| cz | FakeTorino, FakeKingston, FakeFez, FakeMarrakesh, FakeAachen |

**Metric:** as in GAP.

**Failed-element uses** are counted by coupler and by direction.

## 3. Predictions (scored only by `hold3_eval.py score`; written before c8's smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- 270 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% of the circuits too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | selection never costs on average | C8/C5 <= 1.00 on all 9 devices | any > 1.02 |
| H2 | it helps on cx devices | C8/C5 <= 0.99 on at least 3 of the 4 cx devices | > 1.00 on 2 or more cx devices |
| H3 | it is neutral on cz devices | C8/C5 within 0.98-1.02 on all 5 cz devices | any outside 0.95-1.05 |
| H4 | it closes the open-chain gap | F3 open C8/L3T <= 1.05 on at least 3 of 4 cx devices | >= 1.10 on 2 or more |
| H5 | ... and the chain gap HOLD2 could not | chains (F3 open + F5) C8/L3T <= 1.05 on at least 3 of 4 cx devices | >= 1.10 on all 4 |
| H6 | it stays on the target | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C8 | any |
| H7 | it stays cheap | median compile time C8 <= 3 × C5 | > 10 × C5 |
| H8 | the cx gain is broad | on the cx devices, C8/C5 <= 1.00 in >= 80% of the 28 cell-device pairs | < 50% |
| H9 | the estimate chooses well | where C7F and C5 differ in measured infidelity, C8 has the lower of the two in >= 75% of circuits | < 50% |

**Expectations, stated with the predictions:**

- **The bound for H2 comes from c7's smoke run.** A perfect selector, choosing per circuit the better of C5 and c7,
  would have given C8/C5 of 0.983, 0.976, 0.977 and 0.986 on the four cx devices. 0.99 on three of four is therefore
  reachable only if the estimate selects well. H2 and H9 rise and fall together.
- **H9 is the least certain.**
  - In Addendum 322 the excitation exposure ordered C5 against L3T correctly in 143-150 of 150 open-chain circuits on
    the cx devices.
  - On F1, F2 and F6 the estimate has not been tried. There, c7's losses came with a much deeper circuit, which the
    estimate sees only through the extra gates' reported errors and their exposure.
- **H1** may fail by a small margin on a cz device, where the gains are small and a few wrong choices weigh as much.

**Reported without prediction:**

- the device table and both cell tables (C8/C5, C8/L3T), including A7 against C8;
- C7F/C5 by device;
- how often C8 chose each circuit;
- compile times;
- failed uses counted both ways;
- off-target instructions for every arm.

## 4. What this will not establish

- Hardware. The effect is in Aer's thermal-relaxation model.
- ecr devices.
- Circuits wider than 16 touched qubits, where "select" keeps the release's circuit.
- Whether choosing local frames for low excitation inside PSF-Zero's own synthesis would do better.

## 5. Development (disclosed)

### 5.1 The earlier candidate c7, and its smoke run (not a result)

**c7** was `final_resynthesis=True` alone: always re-synthesise. A pre-registration for it was drafted (HOLD3 with
arms C5, C7, A7, L3T and predictions H1-H8) but never locked. Its tests (10 of 10) passed.

**Its smoke run** (1 circuit per cell, 216 jobs, 261 s, early afternoon of 2026-10-03):

- **Unchanged:** the two-qubit count, in every circuit.
- **Gains:** C7/C5 was 0.85-0.98 on F3 open and 0.94-0.99 on F3 periodic, on all nine devices.
- **Losses:** 1.00-1.04 on F1, 1.03-1.09 on F2 and 1.03-1.13 on F6, on all nine devices (cz devices included), and
  1.08 on F5 on three cx devices.
- **Depth:** much greater wherever c7 lost. For example, F5 depth went from 14 to 39 with the same 7 cx, and F1 from
  255 to 412.
- **Its verdict lines** (H1 and H8 refuted, H2 and H3 ambiguous):

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 |
  |---|---|---|---|---|---|---|---|
  | REFUTED | AMBIGUOUS | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | REFUTED |

**Decision.** At this point c7 was not locked. The owner chose to rebuild the candidate as c8, with a per-circuit
selection.

**What c7's smoke data informed:**

- the design of c8 ("select");
- keeping c7's behaviour as the counterfactual arm C7F;
- the thresholds of H2 (the oracle bound above) and H9.

**What was kept:**

- c7's smoke circuits used HOLD3's smoke seeds: each family's scored base + 500,000. c8's smoke run uses the same
  seeds.
- The scored circuits have not been compiled by anything.

### 5.2 c8's tests and smoke run

**How the candidate was built.** `psf_compile.py` was generated from release 2026-10-02.2 by a script. It inserts:

- the import of `UnitarySynthesis`;
- `excitation_cost`, `_final_resynthesis` and `_select_resynthesis`, with `RESYNTH_STATS`;
- the `final_resynthesis` parameter, its checks and its use at the end of the target path;
- changelog item 35.

Nothing else in the release was changed.

**Tests** (`test_c8_resynth.py`, 15 cases). All passed at home in 5.9 s, before c8's smoke run. They check:

- the version strings;
- that the default gives the same output as the release (with and without `target` and `placement_refine`, on
  FakeTorino and FakeAuckland);
- that `final_resynthesis` without a target, or with an unknown value, raises `ValueError`;
- on FakeAuckland, FakeHanoiV2, FakeGeneva, FakeTorino and FakeKingston, that re-synthesis:
  - is exact;
  - is on the target;
  - keeps the final layout;
  - adds no two-qubit gate;
  - uses no failed qubit or failed direction;
- that on open XXZ chains on FakeAuckland it uses fewer x gates than the release;
- that the backstop returns the original circuit;
- that `excitation_cost`'s numpy state matches Qiskit's `Statevector` (to 1e-9 relative);
- that "select" returns whichever circuit has the lower estimate, exactly, on FakeAuckland, FakeHanoiV2 and
  FakeTorino.

**Smoke run (not a result).** 1 circuit per cell, on the same smoke seeds as c7's: 855 compilations, 270 jobs, about
315 s, on the afternoon of 2026-10-03. Nothing was changed after it.

- **P0** passed (noiseless infidelity max 6.6e-15; 0 too wide).
- **Its verdict lines:** all nine CONFIRMED.

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 |
  |---|---|---|---|---|---|---|---|---|
  | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

- **C8/C5 by device:** 0.979-0.988 on the cx devices, 0.988-0.996 on the cz devices.
- **H8 and H9:** H8's share was 0.821; H9's selection accuracy was 145 of 171 circuits (0.848).
- **By cell, c7's losses are gone:** F1, F2 and F6 within 0.991-1.009, F5 1.000. F3 keeps its gain (open 0.85-0.98,
  periodic 0.94-0.99).
- **Choices of C8 by family:**

  | family | kept the release's circuit | chose the re-synthesis |
  |---|---|---|
  | F1 | 41 | 13 |
  | F2 | 16 | 2 |
  | F3 | 0 | 18 |
  | F4 | 13 | 14 |
  | F5 | 24 | 3 |
  | F6 | 24 | 3 |

- **Median compile time:** C5 0.038 s, C7F 0.040 s, C8 0.060 s, A7 0.738 s, L3T 0.018 s.
- **Failed uses and off-target instructions:** 0 by every arm.

**Caveat.** This smoke run is not independent of c8's design: it used the circuits whose c7 results motivated
"select". The scored run on new seeds is the test.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c8_2026-10-03/psf_compile.py`](../../patches/psf_compile_c8_2026-10-03/psf_compile.py) | `ae48dd7a9ea59aa5da5b7a3f2b42e3b97eefcda28af6f31368b746d2b065ab1a` |
| [`patches/psf_compile_c8_2026-10-03/test_c8_resynth.py`](../../patches/psf_compile_c8_2026-10-03/test_c8_resynth.py) | `af22032dfbdde7d96226dfc6f91b9782cd186e99e3a054ec8626a284d6e3fdac` |
| [`benchmarks/hold3_eval.py`](../../benchmarks/hold3_eval.py) | `95d0fc5fb60f7077d0147b6b66ccfe5ea709e8e1d4e16b534bf539a185c57da6` |
| [`benchmarks/run_hold3_2026-10-03.sh`](../../benchmarks/run_hold3_2026-10-03.sh) | `0527a5a9a305fb375042e5a2ee4bbfff3db050cac95dca3e1274c6ce6c48d750` |

Normalization: CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines joined
with "\n" and no final newline.


---

<!-- ===== Addendum 324 (source: spare-qubit-cliff-addendum-324-2026-10-03.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 323 (lock commit c480953), scored by the locked script and re-checked by benchmarks/hold3_verify.py, written after the run finished and before any of its output was seen.

## Addendum 324 -- Results: HOLD3 (Addendum 323). Candidate c8 (final two-qubit re-synthesis by Qiskit, kept per circuit by an excitation-aware estimate) improves on the release on all nine devices (C8/C5 0.969-0.998) and closes the cx open-chain gap to Qiskit L3T from 1.09-1.21 to 1.01-1.05. Eight predictions confirmed, one ambiguous (H8), none refuted. The estimate picks the better circuit in 88% of circuits and captures 87-92% of the oracle's gain on the cx devices (2026-10-03)

**Status: results of the pre-registered test in Addendum 323.**

- **Lock:** commit `c480953`, pushed before the scored run (which started at 15:08 JST).
- **Scoring:** by the locked `hold3_eval.py score`, and re-checked by [`benchmarks/hold3_verify.py`](../../benchmarks/hold3_verify.py). That script was
  written after the run finished and before any of its output was seen. It agrees on every verdict.
- **Setting:** home (WSL2), 6 processes; 270 jobs, 67,770 circuit compilations, 2,794 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 270 of 270 files; noiseless infidelity max 3.7e-9; 0 too wide |
| H1 | **CONFIRMED** | C8/C5 <= 1.00 on all 9 devices: 0.969-0.998 |
| H2 | **CONFIRMED** | C8/C5 on the cx devices: Auckland 0.983, HanoiV2 0.971, Algiers 0.969, Geneva 0.986 (all four <= 0.99) |
| H3 | **CONFIRMED** | C8/C5 on the cz devices: Torino 0.998, Kingston 0.987, Fez 0.995, Marrakesh 0.989, Aachen 0.993 |
| H4 | **CONFIRMED** | F3 open C8/L3T on the cx devices: Auckland 1.014, HanoiV2 1.042, Algiers 1.051, Geneva 1.011 (three <= 1.05) |
| H5 | **CONFIRMED** | chains (F3 open + F5) C8/L3T: Auckland 1.012, HanoiV2 1.037, Algiers 1.045, Geneva 1.009 |
| H6 | **CONFIRMED** | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C8 (0 by every arm) |
| H7 | **CONFIRMED** | median compile time C5 0.026 s, C8 0.042 s (1.6 ×) |
| H8 | **AMBIGUOUS** | C8/C5 <= 1.00 in 20 of the 28 cx cell-device pairs (71.4%; threshold 80%) |
| H9 | **CONFIRMED** | C8 has the lower measured infidelity of C5 and C7F in 11,977 of 13,554 circuits (88.4%) |

**On H8.** Every one of the eight cx cells above 1.00 is within 1.0005-1.0048. Four of them are F2 (QAOA). The
selection's errors there are many and small: it keeps the release's circuit in 94% of F2 circuits, and the cells are
close to 1.000 either way.

## 2. Numbers

**By device:**

| device | C8/C5 | C7F/C5 | oracle/C5 | C8/L3T | C5/L3T | F3 open C8/L3T | F3 open C5/L3T | A7/C8 | A7/L3T |
|---|---|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.983 | 0.999 | 0.981 | 1.037 | 1.055 | 1.014 | 1.107 | 0.913 | 0.947 |
| FakeHanoiV2 (cx) | 0.971 | 0.987 | 0.968 | 1.034 | 1.065 | 1.042 | 1.156 | 0.949 | 0.981 |
| FakeAlgiers (cx) | 0.969 | 0.985 | 0.967 | 1.075 | 1.109 | 1.051 | 1.214 | 0.910 | 0.978 |
| FakeGeneva (cx) | 0.986 | 1.013 | 0.984 | 0.992 | 1.006 | 1.011 | 1.089 | 0.949 | 0.941 |
| FakeTorino | 0.998 | 1.013 | 0.997 | 1.021 | 1.023 | 0.992 | 0.997 | 0.964 | 0.985 |
| FakeKingston | 0.987 | 1.012 | 0.985 | 1.026 | 1.039 | 0.987 | 0.999 | 0.956 | 0.981 |
| FakeFez | 0.995 | 1.015 | 0.994 | 1.034 | 1.040 | 0.987 | 0.993 | 0.953 | 0.985 |
| FakeMarrakesh | 0.989 | 1.013 | 0.988 | 1.015 | 1.026 | 0.989 | 1.017 | 0.946 | 0.960 |
| FakeAachen | 0.993 | 1.023 | 0.993 | 1.048 | 1.055 | 0.989 | 0.994 | 0.937 | 0.981 |

"oracle" takes, per circuit, the better of C5 and C7F as measured. It is a bound, not a compiler.

**Share of the oracle's gain over C5 that C8 captures:** 87-92% on the cx devices, 60% on FakeTorino.

**By cell, C8/C5:**

| cell | Auckland | Torino | Kingston | HanoiV2 | Algiers | Geneva | Fez | Marrakesh | Aachen |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.997 | 1.000 | 0.999 | 0.990 | 1.001 | 1.001 | 0.999 | 0.999 | 0.999 |
| F2 | 1.002 | 1.000 | 1.000 | 1.005 | 1.004 | 1.001 | 1.000 | 1.000 | 1.000 |
| F3 open | 0.916 | 0.995 | 0.988 | 0.901 | 0.866 | 0.928 | 0.994 | 0.972 | 0.995 |
| F3 periodic | 0.967 | 0.995 | 0.947 | 0.931 | 0.925 | 0.967 | 0.981 | 0.969 | 0.974 |
| F4 | 0.997 | 0.998 | 0.999 | 0.988 | 0.993 | 1.000 | 0.997 | 0.997 | 0.997 |
| F5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F6 | 0.993 | 1.000 | 1.001 | 0.993 | 1.003 | 1.000 | 0.999 | 1.000 | 0.999 |

**C8's choices** (all nine devices):

| family | chose the re-synthesis | kept the release's circuit |
|---|---|---|
| F1 | 856 | 3,032 |
| F2 | 125 | 2,035 |
| F3 | 1,994 | 706 |
| F4 | 907 | 1,523 |
| F5 | 44 | 1,252 |
| F6 | 200 | 880 |

The full cell tables are in `outputs/score.md`; the re-computation is in `outputs/verify.txt`.

## 3. Reading

**What c8 does:**

- **Unconditional re-synthesis (C7F) is a mixed bag on new seeds, as in c7's smoke run.**
  - It helps on three cx devices (0.985-0.999).
  - It costs 1.2-2.3% on all five cz devices and on FakeGeneva.
- **The excitation-aware estimate turns that into a gain everywhere.**
  - It picks the measured-better circuit in 88% of circuits.
  - It gets within 0.0003-0.003 of the oracle on every device.
- **It removes most of the open-chain gap on the cx devices** that Addenda 310, 319 and 321 left open and that
  Addendum 322 traced to thermal relaxation: from 1.09-1.21 down to 1.01-1.05.
- **It is safe and cheap:**
  - It never uses a failed element, and nothing goes off the target.
  - It costs 16 ms more per compile.

**What is left:**

- **F3 periodic on the cx devices is still 12-40% behind L3T.** The re-synthesis helps there (0.93-0.97) but does not
  close the gap. Addendum 322 already showed that routing is involved.
- **F1 on the cz devices** (1.06-1.08 against L3T on four of five) is untouched. It is not a thermal-relaxation effect.
- **The AI front end a7 is still ahead of C8** by 3.6-9.0% on every device.
- **The smoke run was optimistic for F3 open.** It put C8/L3T below 1.00 on all four cx devices, against 1.01-1.05
  here, which is the reason for Addendum 323's caveat.

## 4. Consequences

**Adoption.** Whether to adopt c8 is the owner's decision. The data support:

- offering `final_resynthesis="select"` with a target;
- recommending it on every device type tested.

The data do not support making it the default without a target, where it is not defined.

**Next:**

- F3 periodic (routing) on cx devices;
- F1 on cz devices;
- exposure-aware choice of frames inside PSF-Zero's own synthesis, which might beat Qiskit's re-synthesis.

## 5. Data (`data/2026-10-03/hold3/outputs/`)

- 270 job files and their logs, `env.txt`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 325 (source: spare-qubit-cliff-addendum-325-2026-10-03.md) ===== -->

> **Note added when merging:** Adoption record: psf_compile 2026-10-03.c8 becomes release 2026-10-03.1 (owner's decision, 2026-10-03).

## Addendum 325 -- Adoption record: candidate psf_compile 2026-10-03.c8 becomes release 2026-10-03.1 (opt-in `final_resynthesis="select"`, changelog item 35) (2026-10-03)

**Status: adoption record.**

- **Decision:** the owner's, on 2026-10-03, after the results in Addendum 324 (eight of nine predictions confirmed,
  one ambiguous, none refuted).
- **Scope:** fake devices and Aer noise only. Nothing here was run on hardware.

## 1. What the release is

`psf_compile.py` 2026-10-03.1 is the candidate file
[`patches/psf_compile_c8_2026-10-03/psf_compile.py`](../../patches/psf_compile_c8_2026-10-03/psf_compile.py) with
three lines changed:

- the `VERSION:` header line;
- the changelog heading of item 35;
- the `VERSION` constant.

The previous release, 2026-10-02.2, is unchanged as code except for item 35.

**What it adds:** `compile_for_hardware(..., target=..., final_resynthesis=False | True | "select")`.

| value | behaviour | recommended |
|---|---|---|
| `False` (default) | identical to 2026-10-02.2 | -- |
| `True` | always re-synthesise every two-qubit block with Qiskit, exactly, on the target | no: it cost 1-2% on the cz devices (Addendum 324) |
| `"select"` | do so, and keep whichever circuit has the lower `excitation_cost` | yes, with `placement_refine=True` |

**What it does not contain:** the held candidate c6 (item 34, Addendum 321).

`psf_smart_layout` (2026-10-01.1), the Rust core (`CORE_VERSION` 2026-09-29.1) and the AI front end
([`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py), 2026-10-02.a7) are unchanged.

## 2. Files

**Added:**

- [`benchmarks/test_release_2026_10_03.py`](../../benchmarks/test_release_2026_10_03.py). It is the candidate's 15 tests, adapted:
  - the release is compared with the previous release's code, represented by
    [`patches/psf_compile_c5_2026-10-02/psf_compile.py`](../../patches/psf_compile_c5_2026-10-02/psf_compile.py), which differs from 2026-10-02.2 only in the version lines.

**Changed:**

- `psf_compile.py`: the three version lines above.
- `README.md`:
  - a new block for the current version, with the known gaps;
  - the 2026-10-02.2 block retitled "Previous release";
  - its first known gap marked as largely closed.
- **Version assertions** in eight tests, which check that the root file is the current release.
  - Four of them are tests of earlier candidates (c4, c6, c8, a6) whose files were locked by pre-registrations. In
    each, only the assertion of the current release's version string changed. Their normalized SHA-256 before and
    after:

| file | before | after |
|---|---|---|
| [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py) | `5f296067e33b417c…` | `1c6bfee806408340…` |
| [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py) | `cab184e63b308d7f…` | `af9ed2c682ed3d6c…` |
| [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) | `71c145210b5de2c5…` | `f71ca8fc783f75f2…` |
| [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py) | `03ababfa56d670f0…` | `778e6407246ba076…` |
| [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py) (locked, Addendum 306) | `a42b5ea9b6abdaab…` | `3be3e6f6ca122233…` |
| [`patches/psf_compile_c6_2026-10-03/test_c6_floor.py`](../../patches/psf_compile_c6_2026-10-03/test_c6_floor.py) (locked, Addendum 320) | `bdada060f30e3c19…` | `1f968cde2486c478…` |
| [`patches/psf_compile_c8_2026-10-03/test_c8_resynth.py`](../../patches/psf_compile_c8_2026-10-03/test_c8_resynth.py) (locked, Addendum 323) | `af22032dfbdde7d9…` | `7cef8fe327b67d6c…` |
| [`patches/psf_ai_compile_a6_2026-10-02/test_ai6.py`](../../patches/psf_ai_compile_a6_2026-10-02/test_ai6.py) (locked, Addendum 312) | `1b4920cbc1a31b85…` | `f2f34751379e0ae4…` |

**Part 9:** from Addendum 318 on, code-formatted paths that exist in the repository were turned into relative links,
as was done from Addendum 314 on in Addendum 317. A line-by-line check confirmed that only link syntax changed.

## 3. How to use it

```python
out = compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
                           target=backend.target, placement_refine=True, final_resynthesis="select")
```

## 4. Known gaps (Addendum 324)

- **Periodic chains on cx devices** remain 12-40% behind Qiskit level 3. The cause is routing.
- **F1-type rings on cz devices** remain 6-8% behind level 3.
- **The AI front end a7** is still 4-9% ahead of the release with "select".
- **Above 16 touched qubits**, "select" keeps the release's circuit, because its estimate needs a statevector.
- **Hardware has not been tested.** The effect "select" exploits is in Aer's thermal-relaxation model.


---

<!-- ===== Addendum 326 (source: spare-qubit-cliff-addendum-326-2026-10-03.md) ===== -->

> **Note added when merging:** Exploratory diagnosis (not a test, nothing pre-registered) of release 2026-10-03.1's remaining gaps to Qiskit level 3, run at home at commit 6d251c1. Scripts and outputs are in data/2026-10-03/ring/diag/.

## Addendum 326 -- Diagnosis (exploratory, not a test): release 2026-10-03.1's remaining gaps to Qiskit L3T have three different causes. Periodic chains on cx devices: synthesis of the routed circuit (same qubits, same cx count, about 50% more sx gates). F1 rings on cz devices: placement (59 against 54 two-qubit gates). QFT: one or two more two-qubit gates after routing. Choosing per circuit between the release's circuit and level 3's by the release's own excitation_cost comes within 0.000-0.005 of the measured better of the two (2026-10-03)

**Status: exploratory diagnosis.**

- **Nothing was pre-registered**, and nothing here is a verdict.
- **Setting:** run at home on 2026-10-03 at commit `6d251c1` (release 2026-10-03.1).
- **Circuits:** HOLD3's scored circuits (in-sample for the release, since Addendum 324 scored them):
  - all 150 F3 periodic circuits;
  - every second F1 circuit (216);
  - all 120 F6 circuits.
- **Devices:** FakeAuckland and FakeAlgiers (cx; the largest periodic-chain gaps); FakeTorino, FakeKingston and
  FakeAachen (cz; the largest F1 gaps).
- **Reproduction:** every R3 and L3T row reproduced HOLD3's C8 and L3T rows exactly (two-qubit count and noisy
  infidelity; 0 differences on every device).
- **Script:** [`data/2026-10-03/ring/diag/ring_diag.py`](../../data/2026-10-03/ring/diag/ring_diag.py).

## 1. Arms

| arm | what it is |
|---|---|
| R3 | release 2026-10-03.1 as recommended (`layout_search`, `target`, `placement_refine`, `final_resynthesis="select"`) |
| L3T | Qiskit level 3 with the Target |
| R3r3 | R3 with `routing_optimization_level=3` |
| L3onR3 | level 3 pinned to R3's initial layout |
| R3onL3 | the release pinned to L3T's initial layout |
| PICK | per circuit, whichever of R3 and L3T has the lower `excitation_cost`. No new compile. |

## 2. Results (infidelity relative to L3T; full table in `outputs/summary.md`)

**F3 periodic:**

| device | R3 | R3r3 | L3onR3 | R3onL3 | PICK | estimate agrees | same qubits R3/L3T | two-qubit R3/L3T | sx R3/L3T |
|---|---|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 1.192 | 1.172 | 1.020 | 1.184 | 1.000 | 150 of 150 | 150 | 114/114 | 143/96 |
| FakeAlgiers (cx) | 1.402 | 1.417 | 1.063 | 1.415 | 1.000 | 150 of 150 | 150 | 114/114 | 144/96 |
| FakeTorino | 1.008 | 1.050 | 1.000 | 1.005 | 1.001 | 91 of 150 | 150 | 114/114 | 266/281 |
| FakeKingston | 1.032 | 1.018 | 1.052 | 1.011 | 1.000 | 140 of 150 | 0 | 114/114 | 269/281 |
| FakeAachen | 1.142 | 1.047 | 1.150 | 1.015 | 1.000 | 150 of 150 | 0 | 114/114 | 266/281 |

**F1 (rings of cz):**

| device | R3 | R3r3 | L3onR3 | R3onL3 | PICK | estimate agrees | same qubits R3/L3T | two-qubit R3/L3T |
|---|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 1.002 | 0.994 | 1.001 | 0.991 | 0.992 | 173 of 216 | 180 | 53.5/53.8 |
| FakeAlgiers (cx) | 1.030 | 1.007 | 1.029 | 1.003 | 0.998 | 151 of 216 | 180 | 53.5/53.8 |
| FakeTorino | 1.071 | 1.001 | 1.057 | 0.992 | 0.984 | 212 of 216 | 0 | 59.2/53.7 |
| FakeKingston | 1.059 | 1.006 | 1.061 | 0.991 | 0.996 | 207 of 216 | 108 | 57.3/53.8 |
| FakeAachen | 1.077 | 1.003 | 1.075 | 0.988 | 0.992 | 209 of 216 | 108 | 57.3/53.8 |

**F6 (QFT):**

| device | R3 | R3r3 | L3onR3 | R3onL3 | PICK | estimate agrees | two-qubit R3/L3T |
|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 1.054 | 0.972 | 1.004 | 1.046 | 1.003 | 88 of 120 | 31.7/29.7 |
| FakeAlgiers (cx) | 1.060 | 1.004 | 0.995 | 1.031 | 0.998 | 112 of 120 | 31.7/29.7 |
| FakeTorino | 1.001 | 0.999 | 0.983 | 0.984 | 0.998 | 94 of 120 | 31.3/31.3 |
| FakeKingston | 1.047 | 1.002 | 1.018 | 0.987 | 0.999 | 99 of 120 | 33.0/31.7 |
| FakeAachen | 1.003 | 0.993 | 0.970 | 0.980 | 0.996 | 102 of 120 | 33.0/31.7 |

"Estimate agrees" counts the circuits where the lower `excitation_cost` is also the lower measured infidelity.
PICK's distance from the measured better of R3 and L3T (the oracle) is 0.000-0.005.

## 3. Reading

**Periodic chains on the cx devices: synthesis of the routed circuit.**

- R3 and L3T use the same qubits and the same 114 cx in every circuit.
- R3 carries about 50% more sx gates and 30 more layers.
- Neither re-placement (R3onL3) nor routing at level 3 (R3r3) helps.
- Qiskit's own pipeline on R3's placement (L3onR3) comes within 2-6% of L3T.
- Item 35's re-synthesis is applied after PSF-Zero has routed and absorbed SWAPs. It does not reach what level 3
  obtains by synthesising before routing and optimising after.
- The excitation estimate orders R3 against L3T correctly in every circuit on both devices.

**F1 rings on the cz devices: placement.**

- R3 uses 3.5-5.5 more two-qubit gates (SWAPs) than L3T, on a different qubit set.
- Given L3T's placement, the release is slightly ahead of L3T (R3onL3 0.988-0.992).
- Routing at level 3 (R3r3) also closes the gap (1.001-1.006), through its own layout stage.
- The release's layout search (`psf_smart_layout`) finds no exact embedding for a ring. Its fallback places the ring
  worse than level 3's layout stage does.

**F6 (QFT): routing.**

- R3 has 0-2 more two-qubit gates.
- R3r3 and L3onR3 close most of the gap.

**On FakeKingston and FakeAachen**, part of the periodic-chain gap is placement as well (different qubit sets;
R3onL3 1.01-1.02).

## 4. Consequences

**One mechanism covers all three causes:** let the release compare its circuit with level 3's, by the estimate it
already uses for item 35.

- **On these circuits PICK** is:
  - 1.000-1.001 of L3T on the periodic chains;
  - 0.984-0.998 on F1;
  - 0.996-1.003 on F6.
- **The cost** is one level-3 compile, about 15 ms on these circuits.

This is candidate c9 (changelog item 36, `compare_level3=True`), pre-registered in Addendum 327.

**What PICK cannot do** is beat the better of its two inputs. The fixes that would remove the causes inside
PSF-Zero, rather than sidestepping them, remain open:

- synthesis before routing for routed chains;
- a ring-aware layout search.

## 5. Data (`data/2026-10-03/ring/diag/`)

- `ring_diag.py`, `run_ring_diag.sh`;
- `outputs/`, with the per-device json, logs, `env.txt` and `summary.md`.


---

<!-- ===== Addendum 327 (source: spare-qubit-cliff-addendum-327-2026-10-03.md) ===== -->

> **Note added when merging:** Home pre-registration of HOLD4: candidate psf_compile 2026-10-03.c9 (choice against Qiskit level 3 by excitation_cost) on fresh held-out circuits and HOLD's nine devices. Locked by the git commit that adds this Addendum, the candidate patch with its tests and the evaluation scripts, pushed before the scored run. The predictions were written before the smoke run, which is disclosed in section 5.

## Addendum 327 -- Pre-registration: candidate psf_compile 2026-10-03.c9 (choice against Qiskit level 3 by excitation_cost) on fresh held-out circuits (HOLD4). Does the release, comparing its circuit with level 3's, reach or pass level 3 on every family and close most of the distance to the AI front end? (2026-10-03)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c9_2026-10-03/psf_compile.py`](../../patches/psf_compile_c9_2026-10-03/psf_compile.py), with its tests) and [`benchmarks/hold4_eval.py`](../../benchmarks/hold4_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before c9's smoke run.**

## 1. The candidate (changelog item 36)

`compile_for_hardware(..., target=..., placement_refine=True, final_resynthesis="select", compare_level3=True)`:

- **What it does.** After the release's circuit is finished (items 31, 33 and 35), the input is also compiled with
  `transpile(qc, target=target, optimization_level=3, seed_transpiler=..., approximation_degree=1.0)`.
- **When level 3's circuit is kept.** Two conditions:
  - it has no instruction the target does not provide, no failed qubit, and no two-qubit gate in a direction the
    target reports failed;
  - its `excitation_cost` (item 35's estimate) is lower.
- **Otherwise** the release's circuit is returned. That includes the case where either estimate cannot be made
  (above 16 touched qubits).
- **Default:** `compare_level3=False` is identical to release 2026-10-03.1, as checked by test.
- **Base:** release 2026-10-03.1.

**Why:**

- Addendum 326 found three different causes behind the release's remaining gaps to level 3:
  - synthesis of routed periodic chains on cx devices;
  - placement of rings on cz devices;
  - routing of QFT.
- On those circuits a per-circuit choice by the release's own estimate came within 0.000-0.005 of the measured
  better of the two compilers.
- That was in-sample (HOLD3's circuits). This test asks whether it holds on new seeds, on all six families and on
  nine devices.

## 2. Design (`benchmarks/hold4_eval.py`)

**Circuits (held out again):**

- `hold_eval`'s families F1-F6 with the same per-cell sizes: 1,506 per device. The generator code is HOLD3's,
  unchanged (checked textually).
- New seed base 50,000,000 + ... (HOLD3 used 40,000,000, HOLD2 30,000,000, HOLD 20,000,000).

**Arms:**

| arm | what it is |
|---|---|
| R3 | release 2026-10-03.1 as its README recommends (`target`, `placement_refine=True`, `final_resynthesis="select"`) |
| C9 | the candidate, the same call plus `compare_level3=True` |
| A7 | the adopted AI front end |
| L3T | Qiskit level 3 with the Target, `approximation_degree=1.0` |

**Size:** 216 jobs.

**Devices:** HOLD's nine.

| type | devices |
|---|---|
| cx | FakeAuckland, FakeHanoiV2, FakeAlgiers, FakeGeneva |
| cz | FakeTorino, FakeKingston, FakeFez, FakeMarrakesh, FakeAachen |

**Metric:** as in GAP.

## 3. Predictions (scored only by `hold4_eval.py score`; written before c9's smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- 216 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% of the circuits too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the comparison never costs on average | C9/R3 <= 1.00 on all 9 devices | any > 1.02 |
| H2 | the release reaches level 3 overall | C9/L3T <= 1.00 on at least 7 of 9 devices | > 1.03 on 3 or more |
| H3 | ... and almost everywhere | C9/L3T <= 1.02 in >= 90% of the 63 cell-device pairs | < 70% |
| H4 | periodic chains on cx devices are closed | F3 periodic C9/L3T <= 1.03 on all 4 cx devices | any >= 1.10 |
| H5 | rings on cz devices are closed | F1 C9/L3T <= 1.02 on all 5 cz devices | any >= 1.05 |
| H6 | it stays on the target | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C9 | any |
| H7 | it stays cheap | median compile time C9 <= 4 × R3 | > 10 × R3 |
| H8 | it closes most of the distance to the AI front end | A7/C9 >= 0.96 on at least 7 of 9 devices | < 0.93 on 3 or more |
| H9 | the estimate chooses well | where R3 and L3T differ, C9 has the lower measured infidelity of the two in >= 75% of circuits | < 55% |

**How the thresholds were set** (disclosed):

- **Bounds.** Before writing these predictions, the scorer was run on HOLD3's data with an oracle C9 (per circuit, the
  measured better of R3 and L3T) as a plumbing check. On those circuits that bound gave:
  - C9/R3 0.918-0.970;
  - C9/L3T 0.962-0.992;
  - A7/C9 0.957-0.999.

  A real estimate can only do worse than the oracle. The thresholds of H2, H3 and H8 therefore leave room below
  that bound.
- **H9:** the estimate's agreement with measurement in Addendum 326 was 61-100% by device and family. 75% is a
  moderate expectation.

**Expectations, stated with the predictions:**

- **H8 is the least certain.** a7 also compares several candidates including level 3's, with a state-aware estimate
  of its own. C9 has only two candidates.
- **H2 and H3 rest on the estimate** choosing well where the two compilers differ by little (F2, F4, F5).

**Reported without prediction:**

- the device table;
- three cell tables (C9/R3, C9/L3T, A7/C9);
- how often C9 chose level 3;
- compile times;
- failed uses counted both ways;
- off-target instructions for every arm.

## 4. What this will not establish

- **Hardware.** The estimate relies on Aer's noise model, and real devices may differ.
- **ecr devices.**
- **Circuits wider than 16 touched qubits**, where C9 keeps the release's circuit.
- **Whether fixes inside PSF-Zero would do better.** C9 selects between two compilers. It does not remove the causes
  Addendum 326 found (synthesis before routing for routed chains; a ring-aware layout).

## 5. Development (disclosed)

### 5.1 How the candidate was built

`psf_compile.py` was generated from release 2026-10-03.1 by a script. It inserts:

- `COMPARE_STATS`, `_acceptable` and `_compare_level3`;
- the `compare_level3` parameter, its check and its use at the end of the target path;
- changelog item 36.

The return at the end of the target path was rewritten so that item 35's result is assigned before item 36 runs.
Nothing else in the release was changed.

### 5.2 Tests (`test_c9_compare.py`, 10 cases)

All 10 passed at home in 6.7 s, before the smoke run. They check:

- the version strings;
- that the default gives the same output as the release, with and without `target`, `placement_refine` and
  `final_resynthesis="select"`, on FakeTorino and FakeAuckland;
- that `compare_level3` without a target raises `ValueError`;
- on FakeAuckland, FakeHanoiV2, FakeGeneva, FakeTorino and FakeKingston:
  - the result is the release's circuit or level 3's, whichever has the lower estimate and is acceptable;
  - it is exact;
  - it is on the target;
  - it uses no failed qubit or failed direction;
- that level 3 is chosen on periodic XXZ rings on FakeAuckland (at least 2 of 3);
- that `_acceptable` is direction-aware: it rejects FakeHanoiV2's failed cx(5, 8) and accepts cx(8, 5).

### 5.3 Smoke run (not a result)

The smoke run used 1 circuit per cell and its own seeds: 684 compilations, 216 jobs, about 255 s, on the evening of
2026-10-03. Nothing was changed after it.

- **P0** passed (noiseless infidelity max 6.0e-15; 0 too wide).
- **Its verdict lines:**

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 |
  |---|---|---|---|---|---|---|---|---|
  | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

- **H1** was ambiguous because of FakeGeneva (C9/R3 1.011). The other eight devices were 0.926-0.966.
- **C9/L3T by device:** 0.931-0.999.
- **F3 periodic on the cx devices:** 1.000 on all four (R3 1.108-1.415).
- **F1 on the cz devices:** 0.937-0.995.
- **A7/C9:** 0.942-1.019.
- **H9:** 134 of 154 (0.870).
- **C9 chose level 3** in 75 of 171 circuits.
- **Median compile time:** R3 0.066 s, C9 0.086 s, A7 0.698 s, L3T 0.018 s.
- **Failed uses and off-target instructions:** 0 by every arm.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c9_2026-10-03/psf_compile.py`](../../patches/psf_compile_c9_2026-10-03/psf_compile.py) | `e27d241776e8f16f407b5a478084eea977d9493c9352ebbb3c13d432bcbc0e9d` |
| [`patches/psf_compile_c9_2026-10-03/test_c9_compare.py`](../../patches/psf_compile_c9_2026-10-03/test_c9_compare.py) | `897bc5965ea5973fb72596babca5f90def92b412645ef629a8ea0791babeb965` |
| [`benchmarks/hold4_eval.py`](../../benchmarks/hold4_eval.py) | `a834164831ee51a9cc0093933356f059a425158b67b56591cd51736bf63a4c42` |
| [`benchmarks/run_hold4_2026-10-03.sh`](../../benchmarks/run_hold4_2026-10-03.sh) | `488ccf2c0d26bb147ee534bdf903ecaae9d7b810b61394975cb94de299f2fa59` |

Normalization: CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines joined
with "\n" and no final newline.


---

<!-- ===== Addendum 328 (source: spare-qubit-cliff-addendum-328-2026-10-03.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 327 (lock commit 9c65231), scored by the locked script and re-checked by benchmarks/hold4_verify.py, written after the run finished and before any of its output was seen.

## Addendum 328 -- Results: HOLD4 (Addendum 327). With `compare_level3=True`, candidate c9 is at or ahead of Qiskit L3T on every device (C9/L3T 0.967-0.993) and in every one of the 63 cells (none above 1.003), within 0.02-1.7% of the oracle bound. It is ahead of release 2026-10-03.1 by 1.3-8.0%, and within 0.3-4.8% of the AI front end a7. All nine predictions confirmed (2026-10-03)

**Status: results of the pre-registered test in Addendum 327.**

- **Lock:** commit `9c65231`, pushed before the scored run (which started at 18:20 JST).
- **Scoring:** by the locked `hold4_eval.py score`, and re-checked by [`benchmarks/hold4_verify.py`](../../benchmarks/hold4_verify.py). That script was
  written after the run finished and before any of its output was seen. It agrees on every verdict.
- **Setting:** home (WSL2), 6 processes; 216 jobs, 54,216 circuit compilations, about 2,730 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 216 of 216 files; noiseless infidelity max 9.5e-8 (an A7 circuit); 0 too wide |
| H1 | **CONFIRMED** | C9/R3 on all 9 devices: 0.920-0.987 |
| H2 | **CONFIRMED** | C9/L3T <= 1.00 on all 9 devices: 0.967-0.993 |
| H3 | **CONFIRMED** | C9/L3T <= 1.02 in 63 of 63 cell-device pairs (highest 1.003) |
| H4 | **CONFIRMED** | F3 periodic C9/L3T on the cx devices: Auckland 1.000, HanoiV2 1.001, Algiers 1.000, Geneva 1.000 |
| H5 | **CONFIRMED** | F1 C9/L3T on the cz devices: Torino 0.984, Kingston 0.996, Fez 0.997, Marrakesh 0.947, Aachen 0.992 |
| H6 | **CONFIRMED** | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C9 (0 by every arm) |
| H7 | **CONFIRMED** | median compile time R3 0.042 s, C9 0.070 s (1.7 ×) |
| H8 | **CONFIRMED** | A7/C9 >= 0.96 on 7 of 9 devices: 0.987-0.997 on seven, Auckland 0.952, Geneva 0.959 |
| H9 | **CONFIRMED** | C9 has the lower measured infidelity of R3 and L3T in 11,318 of 12,876 circuits (87.9%) |

## 2. Numbers

**By device:**

| device | C9/R3 | C9/L3T | oracle/L3T | R3/L3T | A7/C9 | A7/L3T |
|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.954 | 0.992 | 0.990 | 1.039 | 0.952 | 0.945 |
| FakeHanoiV2 (cx) | 0.954 | 0.987 | 0.983 | 1.034 | 0.994 | 0.981 |
| FakeAlgiers (cx) | 0.920 | 0.991 | 0.987 | 1.077 | 0.987 | 0.978 |
| FakeGeneva (cx) | 0.987 | 0.983 | 0.966 | 0.996 | 0.959 | 0.942 |
| FakeTorino | 0.967 | 0.986 | 0.986 | 1.020 | 0.997 | 0.984 |
| FakeKingston | 0.966 | 0.991 | 0.990 | 1.025 | 0.990 | 0.981 |
| FakeFez | 0.959 | 0.993 | 0.992 | 1.035 | 0.993 | 0.986 |
| FakeMarrakesh | 0.954 | 0.967 | 0.967 | 1.014 | 0.993 | 0.960 |
| FakeAachen | 0.942 | 0.986 | 0.986 | 1.047 | 0.995 | 0.981 |

"oracle" takes, per circuit, the measured better of R3 and L3T. It is a bound, not a compiler.

**By cell, C9/L3T:**

| cell | Auckland | Torino | Kingston | HanoiV2 | Algiers | Geneva | Fez | Marrakesh | Aachen |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.992 | 0.984 | 0.996 | 0.983 | 0.997 | 0.987 | 0.997 | 0.947 | 0.992 |
| F2 | 0.992 | 0.992 | 0.990 | 0.987 | 0.982 | 0.974 | 0.994 | 0.980 | 0.989 |
| F3 open | 0.988 | 0.991 | 0.988 | 0.982 | 0.999 | 0.992 | 0.984 | 0.989 | 0.986 |
| F3 periodic | 1.000 | 0.999 | 1.000 | 1.001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F4 | 0.984 | 0.965 | 0.970 | 0.979 | 0.974 | 0.982 | 0.980 | 0.930 | 0.954 |
| F5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F6 | 1.003 | 0.996 | 0.997 | 0.997 | 0.998 | 0.928 | 0.999 | 0.994 | 0.993 |

**Per circuit:**

- C9 <= L3T in 90.0-95.5% of circuits, by device.
- C9 is better than R3 in 30-50% of circuits, and worse in 2-8%.

**C9's choices** (all nine devices):

| family | level 3's circuit | the release's |
|---|---|---|
| F1 | 1,890 | 1,998 |
| F2 | 1,301 | 859 |
| F3 | 1,887 | 813 |
| F4 | 662 | 1,768 |
| F5 | 17 | 1,279 |
| F6 | 608 | 472 |

The full tables are in `outputs/score.md`; the re-computation is in `outputs/verify.txt`.

## 3. Reading

**What c9 achieves:**

- **The release with `compare_level3=True` is at or ahead of Qiskit L3T everywhere tested.**
  - On every device it is ahead, by 0.7-3.3%.
  - In no cell is it more than 0.3% behind.
  - The three gaps that Addendum 324 left are closed:
    - periodic chains on cx devices: 1.08-1.42 → 1.000;
    - rings on cz devices: up to 1.08 → 0.947-0.997;
    - QFT: up to 1.06 → 0.93-1.003.
- **The choice is close to the best possible.**
  - It is within 0.02-1.7% of the oracle on every device, and within 0.5% on eight of nine.
  - It picks the better of the two compilers in 88% of circuits where they differ.
- **It keeps PSF-Zero's own advantage.**
  - Where PSF-Zero's circuit is better, it is kept: F4 (C9/L3T 0.930-0.984) and F2.
  - The release's circuit is kept in 53% of all circuits.
- **The gap to the AI front end a7 shrinks** from 3.6-9.0% (release 2026-10-03.1, Addendum 324) to 0.3-4.8%.
  - On seven devices it is within 1.3%.
  - In several cells, F3 periodic in particular, C9 is ahead of a7.
- **It is safe and cheap:** no failed element is used, and it costs 28 ms more per compile.

**What c9 does not do:**

- **It selects; it does not repair.** The causes found in Addendum 326 remain inside PSF-Zero:
  - synthesis of routed chains;
  - ring placement;
  - QFT routing.
- **a7 is still ahead**, notably on FakeAuckland and FakeGeneva (F5, F6). It weighs more candidates with its own
  state-aware estimate.
- **Hardware is untested.** The estimate relies on Aer's thermal-relaxation model.

## 4. Consequences

**Adoption** is the owner's decision. The data support:

- offering `compare_level3=True` together with `final_resynthesis="select"`;
- recommending it as the release's best setting with a target.

**Next:**

- the remaining a7 lead on F5 and F6 on two cx devices;
- removing the causes inside PSF-Zero (synthesis before routing, a ring-aware layout).

## 5. Data (`data/2026-10-03/hold4/outputs/`)

- 216 job files and their logs, `env.txt`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 329 (source: spare-qubit-cliff-addendum-329-2026-10-03.md) ===== -->

> **Note added when merging:** Adoption record: psf_compile 2026-10-03.c9 becomes release 2026-10-03.2 (owner's decision, 2026-10-03).

## Addendum 329 -- Adoption record: candidate psf_compile 2026-10-03.c9 becomes release 2026-10-03.2 (opt-in `compare_level3=True`, changelog item 36) (2026-10-03)

**Status: adoption record.**

- **Decision:** the owner's, on 2026-10-03, after the results in Addendum 328 (nine of nine predictions confirmed).
- **Scope:** fake devices and Aer noise only. Nothing here was run on hardware.

## 1. What the release is

`psf_compile.py` 2026-10-03.2 is the candidate file
[`patches/psf_compile_c9_2026-10-03/psf_compile.py`](../../patches/psf_compile_c9_2026-10-03/psf_compile.py) with
three lines changed: the `VERSION:` header line, the changelog heading of item 36, and the `VERSION` constant.

**What it adds:** `compile_for_hardware(..., target=..., compare_level3=True)`.

- The input is also compiled with Qiskit's level 3 on the target.
- Level 3's circuit is kept when two conditions hold:
  - it uses no failed qubit, no failed gate direction and nothing off the target;
  - its `excitation_cost` (item 35's estimate) is lower.
- `compare_level3=False` (the default) is identical to 2026-10-03.1.

**Recommended call with a target:**

```python
out = compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
                           target=backend.target, placement_refine=True, final_resynthesis="select",
                           compare_level3=True)
```

**Unchanged:**

- `psf_smart_layout` (2026-10-01.1);
- the Rust core (`CORE_VERSION` 2026-09-29.1);
- the AI front end ([`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py), 2026-10-02.a7);
- the held candidate c6 (item 34), which is still not included.

## 2. Files

**Added:**

- [`benchmarks/test_release_2026_10_03_2.py`](../../benchmarks/test_release_2026_10_03_2.py). It is the candidate's 10 tests, adapted.
  - The release is compared with the previous release's code, represented by
    [`patches/psf_compile_c8_2026-10-03/psf_compile.py`](../../patches/psf_compile_c8_2026-10-03/psf_compile.py).
  - That file differs from 2026-10-03.1 only in the version lines.

**Changed:**

- `psf_compile.py`: the three version lines above.
- `README.md`:
  - a new block for the current version, with the recommended call and the known limits;
  - the 2026-10-03.1 block retitled "Previous release";
  - its gaps marked as closed by `compare_level3`.
- **Current-release assertions** in ten tests. In each, only the expected version string changed.
  - Five of them are tests of earlier candidates whose files were locked by pre-registrations: c4, c6, c8, c9 and a6.

**Normalized SHA-256, before and after:**

| file | before | after |
|---|---|---|
| [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py) | `1c6bfee806408340…` | `a7262eb199a2049c…` |
| [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py) | `af9ed2c682ed3d6c…` | `e44b5ec8b97d7a3c…` |
| [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) | `f71ca8fc783f75f2…` | `0140dacaa0259015…` |
| [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py) | `778e6407246ba076…` | `2194f1f6d2aa6e76…` |
| [`benchmarks/test_release_2026_10_03.py`](../../benchmarks/test_release_2026_10_03.py) | `3c92b2551f88edf2…` | `5fb3507b6568ff81…` |
| [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py) (locked, Addendum 306) | `3be3e6f6ca122233…` | `b93a6d4ddb8e3a72…` |
| [`patches/psf_compile_c6_2026-10-03/test_c6_floor.py`](../../patches/psf_compile_c6_2026-10-03/test_c6_floor.py) (locked, Addendum 320) | `1f968cde2486c478…` | `f37add9647f59609…` |
| [`patches/psf_compile_c8_2026-10-03/test_c8_resynth.py`](../../patches/psf_compile_c8_2026-10-03/test_c8_resynth.py) (locked, Addendum 323) | `7cef8fe327b67d6c…` | `0b18a266d039702d…` |
| [`patches/psf_compile_c9_2026-10-03/test_c9_compare.py`](../../patches/psf_compile_c9_2026-10-03/test_c9_compare.py) (locked, Addendum 327) | `897bc5965ea5973f…` | `fdc4f0b34716f5aa…` |
| [`patches/psf_ai_compile_a6_2026-10-02/test_ai6.py`](../../patches/psf_ai_compile_a6_2026-10-02/test_ai6.py) (locked, Addendum 312) | `f2f34751379e0ae4…` | `c450acae90fb3649…` |

The "before" hashes of c4, c6, c8 and a6 are those recorded as "after" in Addendum 325.

**Part 9:** from Addendum 326 on, code-formatted paths that exist in the repository were turned into relative links.
A line-by-line check confirmed that only link syntax changed.

## 3. Known limits (Addendum 328)

- **It selects; it does not repair.** The causes found in Addendum 326 remain inside PSF-Zero:
  - synthesis of routed chains;
  - ring placement;
  - QFT routing.
- **The AI front end a7 is still 0.3-4.8% ahead**, most on FakeAuckland and FakeGeneva (F5, F6).
- **Above 16 touched qubits** the estimate is not made, and the release's own circuit is kept.
- **Hardware has not been tested.**


---

<!-- ===== Addendum 330 (source: spare-qubit-cliff-addendum-330-2026-10-03.md) ===== -->

> **Note added when merging:** Exploratory diagnosis (not a test, nothing pre-registered) of the remaining gap between release 2026-10-03.2 and the AI front end a7, run at home at commit 27a9769. Scripts and outputs are in data/2026-10-03/a7gap/diag/.

## Addendum 330 -- Diagnosis (exploratory, not a test): what is left between release 2026-10-03.2 and the AI front end a7. On GHZ-type chains (F5) on the cx devices it is placement: the floor-aware re-placement of the held candidate c6 matches a7 there. The release's `excitation_cost` cannot tell such placements apart; a state-aware Pauli estimate with dephasing can. Choosing among three candidates by that estimate comes within 0.1-0.5% of the measured best (2026-10-03)

**Status: exploratory diagnosis.**

- **Nothing was pre-registered**, and nothing here is a verdict.
- **Setting:** run at home on 2026-10-03 at commit `27a9769` (release 2026-10-03.2).
- **Circuits:** HOLD4's scored circuits (in-sample, since Addendum 328 scored them), 524 per device:
  - F5 and F6 all;
  - F2 every second;
  - F4 every third;
  - F3 open every third.
- **Devices:** FakeAuckland, FakeGeneva, FakeAlgiers, FakeHanoiV2 (cx); FakeTorino, FakeMarrakesh (cz).
- **Reproduction:** every C9 row reproduced HOLD4's exactly (0 differences on every device).
- **Script:** [`data/2026-10-03/a7gap/diag/a7gap_diag.py`](../../data/2026-10-03/a7gap/diag/a7gap_diag.py).

## 1. Starting point (HOLD4)

- **F5 (GHZ chains):** a7 led C9 by 7-29% on three cx devices (A7/C9 0.71-0.93). a7 used the same gates and the same
  depth, and chose its own PSF-Zero candidate, not level 3's. The difference is placement.
- **HOLD2 (Addendum 321) had already measured this.** The held candidate c6 (item 34: the re-placement of item 33
  scored on max(reported error, T1/T2 floor)) had improved F5 by the same amounts: FakeGeneva 0.721, FakeAuckland
  0.934, FakeAlgiers 0.931.
- **F2, F4 and F6:** a7's circuits had about one two-qubit gate fewer. That comes from its several seeds and its
  polish.

## 2. Arms

| arm | what it is |
|---|---|
| R3 | release 2026-10-03.2 with `final_resynthesis="select"`, no level-3 comparison |
| R3F | the same pipeline re-placed on c6's floor-aware Target, then "select" |
| L3T | Qiskit level 3 with the Target |
| C9 | the release as recommended (`compare_level3=True`) |

**Choices among {R3, R3F, L3T}:**

- by the release's `excitation_cost` (PICKe);
- by `pauli_cost` (PICKp), a simplified form of a4's state-aware estimate:
  - for each gate's qubits, Pauli-twirled thermal relaxation p_P costs p_P (1 - <P>^2) on the noiseless state;
  - plus the rest of the reported error, state-independent.

## 3. Results (infidelity relative to L3T, all 524 circuits per device)

| device | C9 | A7 | R3 | R3F | PICKe | PICKp | oracle | best of three ranked by exc / pauli |
|---|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.993 | 0.933 | 1.024 | 0.988 | 0.983 | 0.970 | 0.966 | 316 / 457 |
| FakeGeneva (cx) | 0.965 | 0.858 | 0.921 | 0.900 | 0.943 | 0.886 | 0.884 | 435 / 474 |
| FakeAlgiers (cx) | 0.988 | 0.954 | 1.030 | 1.006 | 0.978 | 0.971 | 0.967 | 454 / 460 |
| FakeHanoiV2 (cx) | 0.991 | 0.964 | 1.023 | 1.025 | 0.991 | 0.989 | 0.984 | 434 / 404 |
| FakeTorino | 0.988 | 0.972 | 0.998 | 0.998 | 0.988 | 0.989 | 0.988 | 371 / 313 |
| FakeMarrakesh | 0.974 | 0.935 | 0.986 | 0.978 | 0.974 | 0.971 | 0.966 | 307 / 377 |

**F5 alone:**

| device | C9 | A7 | R3F | PICKp |
|---|---|---|---|---|
| FakeAuckland | 1.000 | 0.932 | 0.934 | 0.934 |
| FakeGeneva | 1.000 | 0.712 | 0.721 | 0.721 |
| FakeAlgiers | 1.000 | 0.931 | 0.931 | 0.931 |
| FakeMarrakesh | 1.000 | 0.940 | 0.950 | 0.950 |

**F6 on FakeGeneva:**

- R3 0.712 against C9 0.928.
- `excitation_cost` chose level 3's circuit wrongly in about half the circuits (it ranked 58 of 120 correctly);
  `pauli_cost` ranked 114 of 120 correctly.

The full table is in `outputs/summary.md`.

## 4. Reading

- **The F5 gap to a7 is placement, and c6's floor-aware score finds the placement a7 finds.**
  - PICKp reaches a7 on F5 on FakeAuckland and FakeAlgiers, and is within 1% of it on FakeGeneva and FakeMarrakesh.
- **`excitation_cost` cannot rank the placements of a GHZ chain, because those circuits differ in dephasing.**
  - Each qubit of a GHZ state is maximally mixed, so a Z error always costs.
  - Amplitude damping on P(1) = 1/2 looks the same on any qubit with similar T1.
  - `pauli_cost` includes the Z component and ranks the best of three correctly in 77-90% of circuits on the cx
    devices, against 60-87% for `excitation_cost`.
- **On the cz devices the two estimates choose about equally well.** `pauli_cost` ranks fewer circuits correctly on
  FakeTorino, but the circuits there differ little, so the outcome is the same (0.989 against 0.988).
- **The choice by `pauli_cost` among {R3, R3F, L3T} is within 0.1-0.5% of the measured best** on every device.
  - It improves on C9 by 0.2-8.2% on five devices and is level on FakeTorino.
- **What it does not reach:** a7's remaining lead on F2, F4 and F6. That comes from a7's extra seeds and polish: one
  two-qubit gate fewer.

## 5. Consequences

Candidate c10 (changelog item 37) does two things:

- it adds the floor-placed circuit as a candidate (`compare_floor=True`);
- it chooses among the candidates by `pauli_cost` (`candidate_score="pauli"`).

It is pre-registered in Addendum 331.

## 6. Data (`data/2026-10-03/a7gap/diag/`)

- `a7gap_diag.py`, `run_a7gap_diag.sh`;
- `outputs/`, with the per-device json, logs, `env.txt` and `summary.md`.


---

<!-- ===== Addendum 331 (source: spare-qubit-cliff-addendum-331-2026-10-03.md) ===== -->

> **Note added when merging:** Home pre-registration of HOLD5: candidate psf_compile 2026-10-03.c10 (floor-placed candidate and choice by a state-aware Pauli estimate) on fresh held-out circuits and HOLD's nine devices. Locked by the git commit that adds this Addendum, the candidate patch with its tests and the evaluation scripts, pushed before the scored run. The predictions were written before the smoke run; the smoke run and one test corrected after it are disclosed in section 5.

## Addendum 331 -- Pre-registration: candidate psf_compile 2026-10-03.c10 (floor-placed candidate and choice by a state-aware Pauli estimate) on fresh held-out circuits (HOLD5). Does it close the GHZ-chain gap to the AI front end a7 on cx devices without costing anything elsewhere? (2026-10-03)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, the candidate
  ([`patches/psf_compile_c10_2026-10-03/psf_compile.py`](../../patches/psf_compile_c10_2026-10-03/psf_compile.py), with its tests) and [`benchmarks/hold5_eval.py`](../../benchmarks/hold5_eval.py) with its
  runner, pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 3) were written before c10's smoke run.**

## 1. The candidate (changelog item 37)

The release's recommended call, plus `compare_floor=True` and `candidate_score="pauli"`:

```python
compile_for_hardware(..., target=..., placement_refine=True, final_resynthesis="select", compare_level3=True,
                     compare_floor=True, candidate_score="pauli")
```

**Candidates:**

- the release's circuit;
- the same pipeline re-placed on `floor_aware_target(target)` (c6's functions, unchanged), with item 31's backstop
  and item 35's "select";
- Qiskit level 3's circuit.

The floor-placed and level-3 circuits are used only if `_acceptable`.

**Choice:** the candidate with the lowest `pauli_cost`.

- `pauli_cost` is a state-aware first-order estimate with Pauli-twirled thermal relaxation, including dephasing,
  plus the remaining reported error.
- Ties and any estimate that cannot be made keep the release's circuit.

**Defaults:** with the defaults, the result is identical to release 2026-10-03.2, as checked by test.

**Why** (Addendum 330, in-sample on HOLD4's circuits):

- On GHZ chains on cx devices the remaining gap to a7 is placement, and c6's floor-aware placement finds a7's.
- `excitation_cost` cannot rank those placements; `pauli_cost` can.
- The choice among the three by `pauli_cost` came within 0.1-0.5% of the measured best on six devices.

## 2. Design (`benchmarks/hold5_eval.py`)

**Circuits:**

- `hold_eval`'s families F1-F6 with the same per-cell sizes: 1,506 per device. The generator code is HOLD4's,
  unchanged (checked textually).
- New seed base 60,000,000 + ....

**Arms:**

| arm | what it is |
|---|---|
| C9 | release 2026-10-03.2 as recommended |
| C10 | the candidate as above |
| A7 | the adopted AI front end |
| L3T | Qiskit level 3 with the Target |

**Size:** 216 jobs.

**Devices:** HOLD's nine (cx: FakeAuckland, FakeHanoiV2, FakeAlgiers, FakeGeneva; cz: FakeTorino, FakeKingston,
FakeFez, FakeMarrakesh, FakeAachen).

**Metric:** as in GAP.

## 3. Predictions (scored only by `hold5_eval.py score`; written before c10's smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- 216 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% of the circuits too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | it never costs on average | C10/C9 <= 1.00 on all 9 devices | any > 1.02 |
| H2 | it helps on cx devices | C10/C9 <= 0.99 on at least 3 of the 4 cx devices | > 1.00 on 2 or more cx devices |
| H3 | it is neutral or better on cz devices | C10/C9 <= 1.01 on all 5 cz devices | any > 1.03 |
| H4 | it closes the GHZ-chain gap | F5 C10/C9 <= 0.97 on at least 3 of 4 cx devices | > 1.00 on 2 or more |
| H5 | it brings the release within 4% of a7 everywhere | A7/C10 >= 0.96 on all 9 devices | < 0.94 on 2 or more |
| H6 | it stays on the target | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C10 | any |
| H7 | it stays cheap | median compile time C10 <= 3 × C9 | > 10 × C9 |
| H8 | it keeps the release ahead of level 3 | C10/L3T <= 1.00 on all 9 devices | any > 1.02 |

**How the thresholds were set** (disclosed):

- **In-sample figures.** Addendum 330's in-sample figures were known when these predictions were written. On a
  subsample weighted towards F5 and F6, PICKp/C9 was:
  - 0.977, 0.918, 0.983 and 0.998 on the four cx devices;
  - 1.001 and 0.997 on the two cz devices.
- **H2, H3 and H4** are set inside those figures, with room for new seeds.
- **The scorer** was run on HOLD4's data, with C10 set equal to C9, as a plumbing check.

**Expectations, stated with the predictions:**

- **H5 is the least certain.** In HOLD4, A7/C9 was 0.952 on FakeAuckland and 0.959 on FakeGeneva. The diagnosis
  closes F5 there, but a7's one-gate advantage on F2, F4 and F6 remains.
- **H2 depends on FakeHanoiV2**, where the diagnosis showed almost no gain (0.998). The other three cx devices carry
  it.

**Reported without prediction:**

- the device table, with F5 and F6;
- three cell tables;
- how often C10 chose each candidate;
- compile times;
- failed uses counted both ways;
- off-target instructions.

## 4. What this will not establish

- **Hardware.** `pauli_cost` relies on the same thermal model as Aer.
- **ecr devices.**
- **Circuits wider than 16 touched qubits.**
- **a7's remaining advantage**, which comes from its extra seeds and polish.

## 5. Development (disclosed)

### 5.1 How the candidate was built

`psf_compile.py` was generated from release 2026-10-03.2 by a script. It inserts:

- `decoherence_floor` and `floor_aware_target`, copied from candidate c6;
- `pauli_cost` and `_choose`, with two new counters in `COMPARE_STATS`;
- the parameters `compare_floor` and `candidate_score`, with their checks;
- the item-37 branch at the end of the target path, taken only when either parameter is not at its default;
- changelog item 37.

Nothing else in the release was changed.

### 5.2 Tests (`test_c10_floor_pauli.py`, 11 cases), and one test corrected after the smoke run

**First run** (at home, together with the smoke run): 10 passed, 1 failed. The failure was
`test_floor_candidate_used_on_ghz_geneva`, which expected the floor-placed circuit to be chosen for at least 2 of 3
six-qubit GHZ chains on FakeGeneva. It was chosen for none.

**The test was wrong, not the candidate:**

- In Addendum 330's data the floor placement on FakeGeneva differs from the release's only for 8-qubit chains (48 of
  48). For 4 and 6 qubits it is the same placement (96 of 96).
- A tie keeps the release's circuit, so the counter cannot move.
- The smoke run itself shows the floor candidate chosen 20 times, and F5 on FakeGeneva at C10/C9 0.721.

**The correction:** the test was changed to 8-qubit chains (`ghz(n=8, ...)`), with a note explaining why.

- The candidate (`psf_compile.py`) was not changed. Its normalized SHA-256 is the one the smoke run recorded.
- Before the lock commit, the tests are run again on the corrected file. The lock is committed only if all 11 pass.

**The 11 cases check:**

- the version strings;
- that the defaults give the same output as release 2026-10-03.2, with and without the recommended options;
- that bad arguments raise `ValueError`;
- that `pauli_cost` matches a reference computed with Qiskit's `Statevector` and `Pauli` expectation values;
- on FakeAuckland, FakeHanoiV2, FakeGeneva, FakeTorino and FakeKingston, that the full choice:
  - is exact;
  - is on the target;
  - uses no failed qubit or failed direction;
  - has no higher `pauli_cost` than the release's own circuit or level 3's;
- that the floor candidate is chosen on GHZ chains on FakeGeneva;
- that `floor_aware_target` gives max(reported, floor) for every instruction.

### 5.3 Smoke run (not a result)

The smoke run used 1 circuit per cell and its own seeds: 684 compilations, 216 jobs, about 260 s, on the evening of
2026-10-03. Nothing in the candidate or the evaluation scripts was changed after it.

- **P0** passed (noiseless infidelity max 5.2e-15; 0 too wide).
- **Its verdict lines:**

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 |
  |---|---|---|---|---|---|---|---|
  | AMBIGUOUS | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

- **H1** was ambiguous because of FakeTorino 1.004, FakeFez 1.001 and FakeAachen 1.001.
- **H2** was ambiguous because only 2 of the 4 cx devices were <= 0.99 (FakeAuckland 0.980, FakeGeneva 0.966;
  FakeAlgiers 0.992, FakeHanoiV2 0.998).
- **C10/C9 by device:** 0.966-1.004.
- **F5 on the cx devices:** 0.934, 1.001, 0.931 and 0.721.
- **A7/C10:** 0.973-0.996 (A7/C9 0.951-0.996).
- **C10/L3T:** 0.936-0.991.
- **Choices:** level 3 66, the release's circuit 85, the floor candidate 20.
- **Median compile time:** C9 0.095 s, C10 0.179 s (1.9 ×).
- **Failed uses and off-target instructions:** 0 by every arm.

## 6. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c10_2026-10-03/psf_compile.py`](../../patches/psf_compile_c10_2026-10-03/psf_compile.py) | `ae24779cb2703a15e2ab970b942f780a2dcf16cc0142cc26563d4b7b88eff1c5` |
| [`patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py`](../../patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py) (corrected, section 5.2) | `b628a0eec7a13d708cae1029e66e1cd1a42d3ff8c456e8d8d83ac9ba3d06cc3c` |
| [`benchmarks/hold5_eval.py`](../../benchmarks/hold5_eval.py) | `39d1527f52dc61d1fce1b7a1f382fd158e181be2cfb7194e0b4321b927fd1a02` |
| [`benchmarks/run_hold5_2026-10-03.sh`](../../benchmarks/run_hold5_2026-10-03.sh) | `e55bb9d1247ff8b3c2f95a06ba6221328bf3de25a532badb28b21a27636ec343` |

The test file as first run had `d3e1ff9400aac3173ceda9513e49c200c7abccb7b2eb3afe00e802567c51a013`.

Normalization: CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines joined
with "\n" and no final newline.


---

<!-- ===== Addendum 332 (source: spare-qubit-cliff-addendum-332-2026-10-03.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 331 (lock commit fb06c95), scored by the locked script and re-checked by benchmarks/hold5_verify.py, written after the run started and before any of its output was seen.

## Addendum 332 -- Results: HOLD5 (Addendum 331). Candidate c10 (floor-placed candidate, choice by `pauli_cost`) closes the GHZ-chain gap to the AI front end a7 (F5 C10/C9 0.721-0.934 on three cx devices; a7 now 0.963-0.998 of C10 on every device) and keeps the release ahead of Qiskit L3T everywhere. It does not help on cz devices, where it costs 0.1-0.2%, and it slightly loses on F3 chains, where `pauli_cost` (Pauli-twirled) ranks amplitude-damping effects worse than `excitation_cost`. Six confirmed, two ambiguous (H1, H2), none refuted (2026-10-03)

**Status: results of the pre-registered test in Addendum 331.**

- **Lock:** commit `fb06c95`, pushed before the scored run (which started at 20:22 JST). Before the lock, all 11 tests
  passed on the corrected test file (Addendum 331, section 5.2).
- **Scoring:** by the locked `hold5_eval.py score`, and re-checked by [`benchmarks/hold5_verify.py`](../../benchmarks/hold5_verify.py). That script was
  written after the run started and before any of its output was seen. It agrees on every verdict.
- **Setting:** home (WSL2), 6 processes; 216 jobs, 54,216 circuit compilations, about 3,020 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 216 of 216 files; noiseless infidelity max 1.1e-7; 0 too wide |
| H1 | **AMBIGUOUS** | C10/C9 <= 1.00 on 6 of 9 devices; FakeTorino 1.0016, FakeFez 1.0009, FakeAachen 1.0011 (none > 1.02) |
| H2 | **AMBIGUOUS** | C10/C9 on the cx devices: Auckland 0.989, HanoiV2 0.998, Algiers 0.990 (0.9901), Geneva 0.980. Two are <= 0.99; none is > 1.00 |
| H3 | **CONFIRMED** | C10/C9 on the cz devices: Torino 1.002, Kingston 0.999, Fez 1.001, Marrakesh 0.999, Aachen 1.001 |
| H4 | **CONFIRMED** | F5 C10/C9 on the cx devices: Auckland 0.934, HanoiV2 1.001, Algiers 0.931, Geneva 0.721 |
| H5 | **CONFIRMED** | A7/C10 on all 9 devices: 0.963-0.998 |
| H6 | **CONFIRMED** | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C10 (0 by every arm) |
| H7 | **CONFIRMED** | median compile time C9 0.071 s, C10 0.149 s (2.1 ×) |
| H8 | **CONFIRMED** | C10/L3T on all 9 devices: 0.963-0.993 |

## 2. Numbers

**By device:**

| device | C10/C9 | C10/L3T | C9/L3T | A7/C10 | A7/C9 | F5 C10/C9 | F6 C10/C9 |
|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.989 | 0.980 | 0.991 | 0.963 | 0.953 | 0.934 | 0.985 |
| FakeHanoiV2 (cx) | 0.998 | 0.986 | 0.988 | 0.994 | 0.992 | 1.001 | 0.996 |
| FakeAlgiers (cx) | 0.990 | 0.980 | 0.990 | 0.998 | 0.988 | 0.931 | 1.001 |
| FakeGeneva (cx) | 0.980 | 0.963 | 0.983 | 0.981 | 0.961 | 0.721 | 0.782 |
| FakeTorino | 1.002 | 0.989 | 0.988 | 0.995 | 0.997 | 1.000 | 1.003 |
| FakeKingston | 0.999 | 0.989 | 0.990 | 0.992 | 0.991 | 1.000 | 1.002 |
| FakeFez | 1.001 | 0.993 | 0.992 | 0.993 | 0.994 | 1.000 | 1.000 |
| FakeMarrakesh | 0.999 | 0.967 | 0.968 | 0.994 | 0.993 | 0.950 | 1.005 |
| FakeAachen | 1.001 | 0.987 | 0.986 | 0.994 | 0.996 | 1.000 | 1.006 |

**By cell, C10/C9:**

| cell | Auckland | Torino | Kingston | HanoiV2 | Algiers | Geneva | Fez | Marrakesh | Aachen |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.998 | 1.000 | 1.001 | 0.992 | 0.999 | 0.999 | 1.001 | 1.001 | 1.001 |
| F2 | 0.980 | 0.999 | 1.000 | 0.995 | 0.994 | 0.993 | 0.999 | 0.998 | 1.000 |
| F3 open | 1.013 | 1.004 | 0.990 | 1.018 | 0.997 | 1.007 | 1.008 | 1.001 | 1.006 |
| F3 periodic | 1.000 | 1.007 | 0.999 | 1.010 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F4 | 0.962 | 0.999 | 1.000 | 0.992 | 0.962 | 1.000 | 0.999 | 0.997 | 1.000 |
| F5 | 0.934 | 1.000 | 1.000 | 1.001 | 0.931 | 0.721 | 1.000 | 0.950 | 1.000 |
| F6 | 0.985 | 1.003 | 1.002 | 0.996 | 1.001 | 0.782 | 1.000 | 1.005 | 1.006 |

**Per circuit, C10 against C9:**

| | better | worse |
|---|---|---|
| cx devices | 15-34% | 9-17% |
| cz devices | 2-14% | 8-14% |

**C10's choices** (all nine devices):

| family | the release's circuit | level 3's | the floor-placed |
|---|---|---|---|
| F1 | 1,926 | 1,788 | 174 |
| F2 | 892 | 1,109 | 159 |
| F3 | 601 | 1,985 | 114 |
| F4 | 1,409 | 689 | 332 |
| F5 | 856 | 8 | 432 |
| F6 | 534 | 504 | 42 |

The full tables are in `outputs/score.md`; the re-computation is in `outputs/verify.txt`.

## 3. Reading

**What worked:**

- **The GHZ-chain gap to a7 is closed.**
  - F5 on FakeAuckland, FakeAlgiers and FakeGeneva improves by 7-28%.
  - It now matches a7 within 0.0-1.2% (F5 A7/C10 0.988-1.000).
  - On FakeGeneva the floor candidate also improves F6 by 22%.
- **The distance to a7 shrinks on the cx devices:**

  | device | A7/C9 | A7/C10 |
  |---|---|---|
  | FakeAuckland | 0.953 | 0.963 |
  | FakeAlgiers | 0.988 | 0.998 |
  | FakeGeneva | 0.961 | 0.981 |

  On every device a7 is now within 3.7% of C10.
- **Ahead of level 3, safe, affordable:**
  - It stays ahead of level 3 on every device (0.963-0.993).
  - It never uses a failed element.
  - The compile time doubles but stays at about 0.15 s, a fifth of a7's.

**What did not work:**

- **No gain on the cz devices.** It loses 0.1-0.2% on three of them.
- **It loses on F3 (XXZ chains)** by up to 1.8% (FakeHanoiV2 F3 open 1.018).
- **Why this is so.** F3 is where Addendum 322 found amplitude damping to dominate, and where `excitation_cost`
  ranked R3 against L3T correctly in 143-150 of 150 circuits.
  - `pauli_cost` twirls relaxation into symmetric Pauli errors. It therefore loses the non-unital part, the decay of
    |1> to |0>, that `excitation_cost` sees.
  - `pauli_cost` is better where dephasing decides (GHZ states). `excitation_cost` is better where amplitude damping
    decides (excited populations during long cx gates).
- **H2 missed by a hair.** FakeAlgiers came in at 0.9901 against a threshold of 0.99.

## 4. Consequences

**Adoption** is the owner's decision. The data support:

- `compare_floor=True` with `candidate_score="pauli"` on cx devices, where it gains up to 2% overall and up to 28% on
  GHZ-type circuits;
- no change on cz devices.

The data do not support recommending it for every device.

**Next:** an estimate that keeps both effects.

- Amplitude damping as `excitation_cost` counts it: the excited population times duration / T1.
- Pure dephasing as `pauli_cost` counts it: the Z component from T2 beyond T1, times (1 - <Z>^2).
- Plus the reported error above the thermal floor.

Addendum 330's and this test's data would show, in-sample, whether such a combined score ranks better than either.

## 5. Data (`data/2026-10-03/hold5/outputs/`)

- 216 job files and their logs, `env.txt`, `score.md`, `score_log.txt`, `verify.txt`.
- Local paths were replaced.


---

<!-- ===== Addendum 333 (source: spare-qubit-cliff-addendum-333-2026-10-03.md) ===== -->

> **Note added when merging:** Adoption record: psf_compile 2026-10-03.c10 becomes release 2026-10-03.3, recommended on cx devices only (owner's decision, 2026-10-03).

## Addendum 333 -- Adoption record: candidate psf_compile 2026-10-03.c10 becomes release 2026-10-03.3 (opt-in `compare_floor=True`, `candidate_score="pauli"`, changelog item 37), recommended on cx devices only (2026-10-03)

**Status: adoption record.**

- **Decision:** the owner's, on 2026-10-03, after the results in Addendum 332 (six predictions confirmed, two
  ambiguous, none refuted).
- **Scope:** fake devices and Aer noise only. Nothing here was run on hardware.

## 1. What the release is

`psf_compile.py` 2026-10-03.3 is the candidate file
[`patches/psf_compile_c10_2026-10-03/psf_compile.py`](../../patches/psf_compile_c10_2026-10-03/psf_compile.py) with
three lines changed: the `VERSION:` header line, the changelog heading of item 37 (which now says that the options are
recommended on cx devices only), and the `VERSION` constant.

**What it adds:**

- `compare_floor=True`: the floor-placed candidate.
- `candidate_score="pauli"`: the choice by `pauli_cost`.
- With the defaults it is identical to 2026-10-03.2.

**Recommended calls:**

| device type | call |
|---|---|
| cx devices | 2026-10-03.2's recommended call plus `compare_floor=True, candidate_score="pauli"` |
| cz devices | 2026-10-03.2's recommended call, unchanged |

**Why only cx devices** (Addendum 332):

- On the cx devices the options gained 0.2-2.0% overall and 7-28% on GHZ chains on three of four.
- On the cz devices they gained nothing and cost 0.1-0.2% on three of five.
- On XXZ chains (F3) they cost up to 1.8%.

## 2. Files

**Added:**

- [`benchmarks/test_release_2026_10_03_3.py`](../../benchmarks/test_release_2026_10_03_3.py). It is the candidate's 11 tests, adapted.
  - The release is compared with the previous release's code, represented by
    [`patches/psf_compile_c9_2026-10-03/psf_compile.py`](../../patches/psf_compile_c9_2026-10-03/psf_compile.py).
  - That file differs from 2026-10-03.2 only in the version lines.

**Changed:**

- `psf_compile.py`: the three version lines above.
- `README.md`:
  - a new block for the current version, with the cx-only recommendation and the known limits;
  - the 2026-10-03.2 block retitled "Previous release".
- **Current-release assertions** in twelve tests. In each, only the expected version string changed.
  - Six of them are tests of earlier candidates whose files were locked by pre-registrations.

**Normalized SHA-256, before and after:**

| file | before | after |
|---|---|---|
| [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py) | `a7262eb199a2049c…` | `a01a54a9f820d50f…` |
| [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py) | `e44b5ec8b97d7a3c…` | `8799a6f7e75b5ac7…` |
| [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) | `0140dacaa0259015…` | `a7636d31e2d89473…` |
| [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py) | `2194f1f6d2aa6e76…` | `61b315174b2787eb…` |
| [`benchmarks/test_release_2026_10_03.py`](../../benchmarks/test_release_2026_10_03.py) | `5fb3507b6568ff81…` | `2bc57d69eb9419f3…` |
| [`benchmarks/test_release_2026_10_03_2.py`](../../benchmarks/test_release_2026_10_03_2.py) | `40af92a7d44c05eb…` | `a89fa11e39460963…` |
| [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py) (locked, Addendum 306) | `b93a6d4ddb8e3a72…` | `9d74bfe499d3709a…` |
| [`patches/psf_compile_c6_2026-10-03/test_c6_floor.py`](../../patches/psf_compile_c6_2026-10-03/test_c6_floor.py) (locked, Addendum 320) | `f37add9647f59609…` | `8017ec31ff4437e1…` |
| [`patches/psf_compile_c8_2026-10-03/test_c8_resynth.py`](../../patches/psf_compile_c8_2026-10-03/test_c8_resynth.py) (locked, Addendum 323) | `0b18a266d039702d…` | `03cc3b33b906ee66…` |
| [`patches/psf_compile_c9_2026-10-03/test_c9_compare.py`](../../patches/psf_compile_c9_2026-10-03/test_c9_compare.py) (locked, Addendum 327) | `fdc4f0b34716f5aa…` | `113291666a59677f…` |
| [`patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py`](../../patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py) (locked, Addendum 331) | `b628a0eec7a13d70…` | `67f160551a7f99a0…` |
| [`patches/psf_ai_compile_a6_2026-10-02/test_ai6.py`](../../patches/psf_ai_compile_a6_2026-10-02/test_ai6.py) (locked, Addendum 312) | `c450acae90fb3649…` | `88e38220498e073a…` |

**Part 9:** from Addendum 330 on, code-formatted paths that exist in the repository were turned into relative links.
A line-by-line check confirmed that only link syntax changed.

## 3. Known limits (Addendum 332)

- **cz devices:** no gain.
- **XXZ-type chains:** small losses. `pauli_cost` misses amplitude damping, which `excitation_cost` sees.
- **The AI front end a7** is still up to 3.7% ahead.
- **Not tested:** hardware, ecr devices, and more than 16 touched qubits.


---

<!-- ===== Addendum 334 (source: spare-qubit-cliff-addendum-334-2026-10-04.md) ===== -->

> **Note added when merging:** Pre-registration of the unattended home suite of 2026-10-04 (STALE and WIDE, with the exploratory HYBRID diagnosis), written before any run of its scripts; the lock is the commit that adds it.

## Addendum 334 -- Pre-registration: release psf_compile 2026-10-03.3 on wider circuits (WIDE, 8-10 qubits) and with a stale calibration (STALE); plus an exploratory diagnosis (HYBRID) of an estimate that keeps both amplitude damping and dephasing. One unattended home suite (2026-10-04)

**Status: pre-registration, written at home on the evening of 2026-10-03, before any run of these scripts.**

- **Lock:** the git commit that adds this document, [`benchmarks/stale_eval.py`](../../benchmarks/stale_eval.py), [`benchmarks/wide_eval.py`](../../benchmarks/wide_eval.py),
  [`data/2026-10-04/hybrid/diag/hybrid_diag.py`](../../data/2026-10-04/hybrid/diag/hybrid_diag.py) and the suite runner [`benchmarks/run_suite_2026-10-04.sh`](../../benchmarks/run_suite_2026-10-04.sh), pushed
  before the suite is started.
- **No hardware:** fake devices and Aer noise only.
- **Release under test:** 2026-10-03.3 (Addendum 333), unchanged. Nothing in the release is changed by this addendum.
- **Smoke runs come after the lock** (section 5). They are plumbing checks only and are not results.

## 1. Why

Addendum 332 tested the release on the HOLD families at 4-8 logical qubits, with the compiler given the same
calibration that the simulator uses. Two questions are left open by every test so far:

- **Width.** Do the recommended calls keep their lead over Qiskit level 3 at 8-10 qubits, where routing is longer,
  a7 takes its fast path (above 8 qubits) and the estimates (`excitation_cost`, `pauli_cost`) work on larger states?
- **Calibration mismatch.** Items 35-37 choose by estimates computed from the Target's errors and T1/T2. On hardware
  the calibration the compiler sees is hours old. Do these choices keep their advantage when the Target is not the
  truth, or do they overfit to it?

And one development question (exploratory, section 4): Addendum 332 found that `pauli_cost` chooses well where
dephasing decides (GHZ chains) and loses where amplitude damping decides (F3), and the reverse for `excitation_cost`.

## 2. STALE (`benchmarks/stale_eval.py`)

**The stale Target.** For each device, a deep copy of the Target in which:

- every instruction error is multiplied by exp(N(0, 0.3)), capped at 0.49; failed entries (error >= 0.5) are not
  changed, so the failed set is the true one;
- every qubit's T1 and T2 are multiplied by independent exp(N(0, 0.2)) factors, with T2 capped at 2 T1 (new
  `QubitProperties` objects; the script asserts that the true Target is unchanged);
- the random numbers come from a fixed seed per device (80,000,000 + CRC32 of the device name).

The widths (30% on errors, 20% on T1/T2) were chosen before any run, as a plausible size for calibration drift within
a day. They were not tuned.

**Every arm compiles against the stale Target.** The noisy simulation uses the device's true noise model. Failed-element
uses are counted on the true Target.

**Circuits:** the six HOLD families, with HOLD5's generator code unchanged (checked textually) and half its per-cell
sizes: 753 per device. Seed base 80,000,000 + ....

**Arms:**

| arm | what it is |
|---|---|
| R2 | release 2026-10-03.3 with 2026-10-03.2's recommended call (`target`, `placement_refine=True`, `final_resynthesis="select"`, `compare_level3=True`) |
| R3 | the same plus `compare_floor=True, candidate_score="pauli"` (the cx-device option of 2026-10-03.3), on every device |
| A7 | the AI front end 2026-10-02.a7 with the (stale) Target |
| L3T | Qiskit level 3 with the (stale) Target |

**Size:** 216 jobs (9 devices × 4 arms × 6 families). **Devices:** HOLD's nine. **Metric:** as in GAP.

## 3. WIDE (`benchmarks/wide_eval.py`)

**Circuits:** the six HOLD families, with the same generator code and wider sizes. Only the loops over n (and F1's
depths) differ from HOLD5's (checked textually):

| family | sizes |
|---|---|
| F1 rings of cz | n 8 and 10, L 2 and 4 |
| F2 QAOA, 3-regular | n 8 and 10, p 1 and 2 |
| F3 XXZ chains | n 8 and 10, open and periodic |
| F4 random brickwork | n 8, 9 and 10 |
| F5 GHZ chains | n 9 and 10 |
| F6 QFT | n 8 and 9 |

12 circuits per (family, n, variant): 228 per device. Seed base 70,000,000 + ....

- Circuits whose compiled form touches more than 12 qubits are not simulated (too wide).
- The noisy simulation is GAP's density-matrix simulation.
- The noiseless P0 check uses Aer's statevector method (the reduced density matrix of the same qubits), to save time.

**Arms:** R2, R3, A7 and L3T as in section 2, with the true Target.

**Size:** 216 jobs.

## 4. Predictions (scored only by each script's `score`)

**P0, harness, for each test.** All of these must hold, or nothing of that test is scored:

- 216 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% (STALE) or 10% (WIDE) of the circuits too wide.

### STALE

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the release keeps its lead over level 3 | R2/L3T <= 1.00 on at least 7 of 9 devices | > 1.03 on 3 or more |
| H2 | the cx-device option keeps its gain | R3/R2 <= 1.00 on at least 3 of the 4 cx devices | > 1.01 on 2 or more |
| H3 | no family breaks | R2/L3T <= 1.02 in at least 85% of the 63 cell-device pairs | in fewer than 65% |
| H4 | it stays on the target | 0 failed-direction or failed-qubit uses and 0 off-target instructions by R2 and R3 | any |
| H5 | a7 keeps its lead over level 3 | A7/L3T <= 1.00 on at least 7 of 9 devices | > 1.03 on 3 or more |

### WIDE

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | the release keeps its lead over level 3 | R2/L3T <= 1.00 on at least 7 of 9 devices | > 1.03 on 3 or more |
| H2 | the cx-device option keeps its gain | R3/R2 <= 1.00 on at least 3 of the 4 cx devices | > 1.01 on 2 or more |
| H3 | no family breaks | R2/L3T <= 1.02 in at least 85% of the 63 cell-device pairs | in fewer than 65% |
| H4 | it stays on the target | 0 failed-direction or failed-qubit uses and 0 off-target instructions by R2 and R3 | any |
| H5 | it stays cheap at this width | median compile time of R3 <= 1.0 s | > 5 s |

The cells are F1, F2, F3 open, F3 periodic, F4, F5 and F6.

**How the thresholds were set** (disclosed):

- **In-sample figures.** On HOLD5 (true calibration, 4-8 qubits):
  - C9/L3T, which is R2/L3T, was 0.968-0.992 on the nine devices;
  - C10/C9, which is R3/R2 on the cx devices, was 0.980-0.998;
  - A7/L3T was 0.944-0.986.
- **The thresholds are looser than those figures**, because both tests move away from the conditions under which
  the release was developed. "At least 7 of 9" allows two devices to fall behind without a refutation.
- **The scorers were run on HOLD5's data, renamed** (C9 as R2, C10 as R3) as a plumbing check. They reproduce
  Addendum 332's ratios.

**Expectations, stated with the predictions:**

- **STALE H2 is the least certain.** The floor candidate's gain on GHZ chains comes from T1/T2-aware placement. With
  T1/T2 off by 20%, it may pick worse qubits.
- **WIDE H5:** at 10 qubits R3 compiles the circuit three times and the estimates work on 2^10 states. The median
  should stay well under a second, but the tail will be longer.

**Reported without prediction:**

- the device tables (R2/L3T, R3/L3T, R3/R2, A7/R2, A7/L3T for STALE, and the recommended call against L3T: R3 on cx
  devices, R2 on cz devices);
- the cell tables for R2/L3T and R3/R2;
- R3's choices;
- compile times;
- failed uses counted both ways and off-target instructions for every arm;
- for STALE, the number of qubits whose T1 the stale Target changed.

## 5. HYBRID (exploratory diagnosis, not a test) (`data/2026-10-04/hybrid/diag/hybrid_diag.py`)

**Circuits:** every HOLD5 circuit on all nine devices (in-sample, since Addendum 332 scored them).

**Candidates:** for each circuit, the candidate set that release 2026-10-03.3 builds with `compare_floor=True,
compare_level3=True, candidate_score="pauli"`:

- the release's own circuit;
- the floor-placed one, if `_acceptable`;
- level 3's, if `_acceptable`.

The set is captured by wrapping `_choose`, so it is exactly the release's. Each distinct candidate is simulated as in
HOLD5.

**Estimates.** Each candidate is scored with four:

| estimate | what it is |
|---|---|
| `exc` | the release's `excitation_cost` |
| `pauli` | the release's `pauli_cost` |
| `hyb` | amplitude damping as `exc` counts it, plus pure dephasing as `pauli`'s Z part counts it, plus the reported error above the thermal floor (see below) |
| `excz` | `exc` plus the same pure-dephasing term |

`hyb` in detail, per gate and qubit:

- **amplitude damping:** duration / T1 × P(1) on the noiseless state just before the gate;
- **pure dephasing:** p_phi (1 - <Z>^2) just after the gate, with p_phi = (1 - exp(-t / T_phi)) / 2 and
  1 / T_phi = 1 / T2 - 1 / (2 T1);
- **the rest:** the reported error above the thermal floor, × (d + 1) / d.

**Choices** are made as `_choose` makes them.

**Checks:**

- the choice by `pauli` must reproduce HOLD5's C10 rows;
- the choice by `exc` between the release's circuit and level 3's must reproduce HOLD5's C9 rows.

**Reported:** per device and family, the choice by each estimate relative to L3T, the measured best, and how often
each estimate picks the measured best.

**Status of any finding:**

- The two combinations were defined before any run. Both are reported, whatever they show.
- Any candidate built on them (c11) would need its own pre-registered test on fresh circuits.

## 6. How the suite runs (`benchmarks/run_suite_2026-10-04.sh`)

**Order:**

1. STALE: smoke run, scored run, score.
2. HYBRID: smoke run on one device, then the nine devices, then the summary.
3. WIDE: smoke run, scored run, score.

**Smoke runs** use 1 circuit per cell (2 per family for HYBRID) and their own seeds.

- A part whose smoke run shows a Traceback or STOP, or lacks an output file, is skipped. The next part still runs.
- If a part is skipped, its scripts are not changed silently. A correction would be disclosed in a new addendum
  before any rerun.
- Nothing in the scripts is changed after the lock.

**Settings:** PAR 6, as in HOLD5. Nothing is committed during the run.

**Expected wall time:**

| part | expected |
|---|---|
| STALE | about 30 min |
| HYBRID | about 15-20 min |
| WIDE | not measured. The density-matrix simulation at 10 qubits costs about 2^8 times more per gate than at 6, so WIDE is expected to take 1-4 h |

## 7. What this will not establish

- **Hardware.** The stale Target imitates calibration drift with independent log-normal factors. Real drift is
  correlated, can come in jumps (TLS defects), and includes errors the Target does not report.
- **ecr devices.**
- **Circuits wider than 10 logical qubits.**

## 8. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/stale_eval.py`](../../benchmarks/stale_eval.py) | `f2dd16caf17d25232f2452b08a5654f83a8ae10459c773dfa3acfc74005a1cc0` |
| [`benchmarks/wide_eval.py`](../../benchmarks/wide_eval.py) | `47a32ac09648e8437970722e852520300a4d01ec435c79bffe0cc7ccfb0f9d47` |
| [`data/2026-10-04/hybrid/diag/hybrid_diag.py`](../../data/2026-10-04/hybrid/diag/hybrid_diag.py) | `ba584d0cd64bea0661f42441d33d952a9b294e08d49c2ee4f35767982f592541` |
| [`benchmarks/run_suite_2026-10-04.sh`](../../benchmarks/run_suite_2026-10-04.sh) | `d76b95890b4a120fd1c25f00ae0c9c2fbd01ac9dff2909cc42f9024fc710b7ef` |

Normalization: CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines joined
with "\n" and no final newline.


---

<!-- ===== Addendum 335 (source: spare-qubit-cliff-addendum-335-2026-10-04.md) ===== -->

> **Note added when merging:** Results of the pre-registered tests in Addendum 334 (lock commit a88df77), scored by the locked scripts and re-checked by benchmarks/suite_verify_2026-10-04.py, written after the suite finished and before its output files were read.

## Addendum 335 -- Results: the 2026-10-04 home suite (Addendum 334). Release 2026-10-03.3 keeps its lead over Qiskit L3T with a stale calibration (STALE, 5 of 5 confirmed) and at 8-10 qubits (WIDE, 5 of 5 confirmed). Two findings outside the predictions: the AI front end a7 is 1.1-2.8 times worse than the release above 8 qubits, and there it uses failed couplers; and, in-sample, the combined estimate `hyb` chooses better than both of the release's estimates on all nine devices (HYBRID, exploratory) (2026-10-04)

**Status: results of the pre-registered tests in Addendum 334, and of its exploratory diagnosis.**

- **Lock:** commit `a88df77`, pushed before the suite started (10:03 JST).
- **Same bytes:** the suite ran the locked files. The SHA-256 values in `env.txt` are those of the files in the lock,
  and every job file records the normalized SHA-256 of its script.
- **Scoring:** by each locked script's `score`, and re-checked by [`benchmarks/suite_verify_2026-10-04.py`](../../benchmarks/suite_verify_2026-10-04.py), which agrees
  on every verdict.
  - **When the verify script was written:** after the suite finished, before its output files were read. Its
    normalized SHA-256 is `09d4da6cea9be5dcc4f0ded41aa6a9bf60853452db00f9dbaf292d240864e934`.
  - **What had been seen by then:** the last five lines of the suite log, which show WIDE's H3-H5 verdicts.
- **Setting:** home (WSL2, Ryzen 5 5500), PAR 6. Total 159 min.

  | step | time |
  |---|---|
  | STALE smoke | 263 s |
  | STALE scored | 1,546 s |
  | HYBRID | 835 s |
  | WIDE smoke | 730 s |
  | WIDE scored | 6,199 s |

## 1. STALE (compilers given a perturbed Target; simulation with the true noise)

**P0: PASS.** 216 of 216 files, 27,108 rows, noiseless max 2.0e-8, none too wide.

The stale Target changed T1 on every qubit that has one: 27 on the 27-qubit devices, 133-156 on the Heron devices.

| ID | Verdict | Numbers |
|---|---|---|
| H1 | **CONFIRMED** | R2/L3T <= 1.00 on 9 of 9 devices: 0.982-0.999 |
| H2 | **CONFIRMED** | cx devices, R3/R2: Auckland 0.963, HanoiV2 0.982, Algiers 0.997, Geneva 0.994 |
| H3 | **CONFIRMED** | R2/L3T <= 1.02 in 61 of 63 cell-device pairs (0.968). The two above: Kingston F6 1.024, Fez F3 periodic 1.038 |
| H4 | **CONFIRMED** | 0 failed-direction or failed-qubit uses and 0 off-target instructions by R2 and R3 |
| H5 | **CONFIRMED** | A7/L3T <= 1.00 on 9 of 9 devices: 0.951-0.987 |

**By device:**

| device | R2/L3T | R3/L3T | R3/R2 | A7/L3T | F5 R3/R2 | F3 open R3/R2 |
|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.992 | 0.956 | 0.963 | 0.951 | 0.920 | 1.030 |
| FakeTorino | 0.997 | 0.998 | 1.001 | 0.986 | 1.000 | 1.000 |
| FakeKingston | 0.999 | 0.990 | 0.991 | 0.972 | 1.000 | 0.930 |
| FakeHanoiV2 (cx) | 0.996 | 0.978 | 0.982 | 0.964 | 0.957 | 1.000 |
| FakeAlgiers (cx) | 0.986 | 0.984 | 0.997 | 0.978 | 0.936 | 1.015 |
| FakeGeneva (cx) | 0.982 | 0.976 | 0.994 | 0.958 | 1.000 | 0.998 |
| FakeFez | 0.992 | 0.991 | 0.999 | 0.974 | 0.979 | 1.065 |
| FakeMarrakesh | 0.985 | 0.950 | 0.964 | 0.956 | 0.938 | 1.007 |
| FakeAachen | 0.995 | 0.996 | 1.001 | 0.987 | 1.000 | 1.000 |

**Reading:**

- **The release does not overfit to the calibration it sees.** With 30% errors on the reported errors and 20% on
  T1/T2, R2 stays ahead of L3T on every device (0.1-1.8%), and so does a7 (1.3-4.9%).
- **The lead is smaller than with the true calibration on most cz devices.** C9/L3T in HOLD5 was:
  - FakeTorino 0.988, FakeKingston 0.990, FakeMarrakesh 0.968, FakeAachen 0.986;
  - here R2/L3T is 0.997, 0.999, 0.985 and 0.995.

  These are different circuits (new seeds), so this is not a paired comparison.
- **The floor candidate is robust to stale T1/T2.** R3 gains 0.3-3.7% on the cx devices and 0.9-3.6% on FakeKingston
  and FakeMarrakesh.
- **It still loses on F3 open chains**, as in Addendum 332: up to 6.5% on FakeFez and 3.0% on FakeAuckland.
- **Failed couplers** (direction ignored): R2 6, R3 48, A7 24, L3T 51 uses. All are on FakeHanoiV2, in the allowed
  direction of a coupler that has failed one way only. No arm used a failed direction.

## 2. WIDE (8-10 logical qubits)

**P0: PASS, but only just.** 216 of 216 files, 8,208 rows, noiseless max 8.8e-10.

- **Too wide: 788 of 8,208 (9.6%) against a limit of 10%.**
  - They are almost all n = 10 circuits: R2 160, R3 161, L3T 172, A7 277.
  - Most are on the Heron devices: FakeAachen 144, FakeFez 140, FakeMarrakesh 140, FakeKingston 138. FakeTorino has
    25; each cx device about 50.
- **The ratios below use only circuits simulated in both arms compared.** So at n = 10 on the Heron devices they rest on
  about two thirds of the circuits. This is a selection that the pre-registration did not discuss.

| ID | Verdict | Numbers |
|---|---|---|
| H1 | **CONFIRMED** | R2/L3T <= 1.00 on 9 of 9 devices: 0.951-0.998 |
| H2 | **CONFIRMED** | cx devices, R3/R2: Auckland 1.0001, HanoiV2 0.9998, Algiers 0.991, Geneva 0.994 (three of four <= 1.00) |
| H3 | **CONFIRMED** | R2/L3T <= 1.02 in 63 of 63 cell-device pairs |
| H4 | **CONFIRMED** | 0 failed-direction or failed-qubit uses and 0 off-target instructions by R2 and R3 |
| H5 | **CONFIRMED** | median compile time R3 0.237 s (R2 0.122 s, L3T 0.021 s) |

**By device:**

| device | R2/L3T | R3/L3T | R3/R2 | A7/R2 at n = 8 | A7/R2 at n = 9 | A7/R2 at n = 10 |
|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.998 | 0.998 | 1.000 | 0.968 | 1.157 | 1.141 |
| FakeTorino | 0.985 | 0.985 | 1.000 | 0.991 | 1.266 | 2.115 |
| FakeKingston | 0.978 | 0.977 | 1.000 | 0.993 | 1.218 | 1.294 |
| FakeHanoiV2 (cx) | 0.990 | 0.989 | 1.000 | 0.966 | 1.160 | 1.328 |
| FakeAlgiers (cx) | 0.974 | 0.965 | 0.991 | 0.965 | 1.769 | 1.526 |
| FakeGeneva (cx) | 0.970 | 0.964 | 0.994 | 0.937 | 1.206 | 1.281 |
| FakeFez | 0.988 | 0.987 | 0.999 | 0.987 | 1.470 | 1.469 |
| FakeMarrakesh | 0.951 | 0.951 | 1.000 | 0.987 | 2.844 | 2.665 |
| FakeAachen | 0.960 | 0.962 | 1.001 | 0.993 | 1.583 | 2.264 |

**Reading:**

- **The release keeps its lead at this width**, 0.2-4.9% ahead of L3T, with no cell above 1.02.
- **The cx-device option gains less here.** It gains 0.6-0.9% on FakeAlgiers and FakeGeneva and nothing on
  FakeAuckland and FakeHanoiV2.
  - The GHZ-chain gain of HOLD5 (n 4-8) does not carry over to n 9-10, except on FakeHanoiV2 (F5 0.953).
  - The floor candidate was chosen less often (162 of 2,052 circuits).
  - F3 open loses again, by up to 2.1%.
  - Couplers failed one way only were used in their allowed direction, on FakeHanoiV2 and FakeGeneva, by R2 (366),
    R3 (378) and L3T (409) alike.
- **a7 above 8 qubits** (a finding outside the predictions):
  - a7 takes its fast path above 8 qubits (`SMALL_MAX_QUBITS` = 8; 1,188 circuits here).
  - There it is 1.14-2.84 times worse than R2.
  - In 302 of those circuits it used failed elements: 10,057 uses of a failed direction or a failed qubit, on seven of
    the nine devices.
  - At n = 8 (its full path) it is still 0.7-6.3% ahead of R2.
  - None of this touches a prediction, because H4 concerns R2 and R3 only. It does mean that a7 must not be used above
    8 qubits as it stands.

## 3. HYBRID (exploratory diagnosis, in-sample on HOLD5's circuits)

**Checks:**

- 1,506 circuits per device; 0 mismatches against HOLD5's C9 and C10 rows on every device;
- noiseless max <= 1e-6.

**Ratios** (mean infidelity relative to HOLD5's L3T):

| device | C9 (exc, 2 cand.) | C10 (pauli, 3 cand.) | PICK_hyb | ORC | best of the candidates picked: exc / pauli / hyb / excz | hyb vs C10 per circuit: better / worse |
|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.991 | 0.980 | 0.977 | 0.977 | 1040 / 1219 / 1365 / 1290 | 185 / 39 |
| FakeTorino | 0.988 | 0.989 | 0.987 | 0.987 | 1317 / 1190 / 1399 / 1264 | 240 / 31 |
| FakeKingston | 0.990 | 0.989 | 0.988 | 0.988 | 1288 / 1294 / 1399 / 1384 | 136 / 29 |
| FakeHanoiV2 (cx) | 0.988 | 0.986 | 0.981 | 0.980 | 1109 / 1188 / 1353 / 1273 | 259 / 92 |
| FakeAlgiers (cx) | 0.990 | 0.980 | 0.978 | 0.977 | 1154 / 1283 / 1331 / 1090 | 143 / 87 |
| FakeGeneva (cx) | 0.983 | 0.963 | 0.962 | 0.962 | 1311 / 1321 / 1433 / 1365 | 129 / 17 |
| FakeFez | 0.992 | 0.993 | 0.991 | 0.991 | 1344 / 1226 / 1425 / 1105 | 215 / 14 |
| FakeMarrakesh | 0.968 | 0.967 | 0.965 | 0.963 | 1151 / 1187 / 1341 / 1374 | 198 / 14 |
| FakeAachen | 0.986 | 0.987 | 0.986 | 0.986 | 1402 / 1262 / 1422 / 1395 | 173 / 13 |

**Reading:**

- **`hyb` beats both of the release's estimates on all nine devices.** Its choice is:
  - better than C10's by 0.06-0.51%;
  - better than C9's by 0.02-2.2%;
  - within 0.0-0.2% of the measured best.

  It picks the measured best in 88-95% of circuits, more than any other estimate on eight devices. On FakeMarrakesh
  `excz` picks it more often (1,374 against 1,341).
- **It repairs F3.** `pauli`'s loss on F3 open is mostly recovered (PICK_hyb / PICK_pauli 0.979-0.998).
- **It keeps the GHZ gains.** On F5 and F6 it matches `pauli`, with one exception: FakeAlgiers F5, 1.012.
- **`excz`, the cruder combination, is worse.** It loses up to 8.3% on FakeAlgiers F3 and 6.1% on FakeFez F1, because
  it probably counts the thermal part twice.
- **On cz devices the floor candidate is rarely distinct** (0-317 circuits per device). There `hyb` among the release's
  circuit and level 3's alone still improves on C9, by 0.02-0.07%.
- **This is in-sample:** the same circuits from which Addendum 332 drew the idea. Two combinations were examined, and
  both are reported.

## 4. Consequences

**Adoption decisions** are the owner's.

- **Release 2026-10-03.3:** these results support keeping the recommendation as it is. That is .2's call everywhere,
  plus `compare_floor=True, candidate_score="pauli"` on cx devices.
- **a7 above 8 qubits:** the documentation should say that a7 is for at most 8 logical qubits.
  - A fix (candidate a8): above `SMALL_MAX_QUBITS`, hand the circuit to the release's recommended call with the target.
    That call never used a failed element in any test so far.
- **Next candidate, c11:** `candidate_score="hybrid"` with `hyb` as defined in Addendum 334, to be pre-registered on
  fresh circuits. On this evidence it might be recommended on every device.

## 5. Data (`data/2026-10-04/`)

| folder | contents |
|---|---|
| `stale/outputs/` | 216 job files and their logs, `score.md`, `score_log.txt`, `verify.txt` |
| `wide/outputs/` | 216 job files and their logs, `score.md`, `score_log.txt`, `verify.txt` |
| `hybrid/diag/outputs/` | per-device json, logs, `summary.md` |
| `suite/` | `env.txt`, `suite_log.txt` |

**Not included:**

- The smoke outputs. They were plumbing checks: P0 passed, and their verdicts are not results.
- The untracked-file lines (`git status`) in `env.txt` and `suite_log.txt`, which were removed.

Local paths were replaced.


---

<!-- ===== Addendum 336 (source: spare-qubit-cliff-addendum-336-2026-10-04.md) ===== -->

> **Note added when merging:** Pre-registration of HOLD6 (candidates psf_compile 2026-10-04.c11 and psf_ai_compile 2026-10-04.a8); the predictions were written before the smoke run; the lock is the commit that adds it.

## Addendum 336 -- Pre-registration: candidate psf_compile 2026-10-04.c11 (choice by an estimate with both amplitude damping and pure dephasing) and candidate front end psf_ai_compile 2026-10-04.a8 (target-aware path above 8 qubits), on fresh circuits (HOLD6) (2026-10-04)

**Status: pre-registration, written at home before any scored run.**

- **Lock:** the git commit that adds this document, with:
  - [`patches/psf_compile_c11_2026-10-04/`](../../patches/psf_compile_c11_2026-10-04/) (the candidate and its tests);
  - [`patches/psf_ai_compile_a8_2026-10-04/`](../../patches/psf_ai_compile_a8_2026-10-04/) (the candidate and its tests);
  - [`benchmarks/hold6_eval.py`](../../benchmarks/hold6_eval.py) and its runner.

  The commit is pushed before the scored run.
- **No hardware:** fake devices and Aer noise only.
- **The predictions (section 4) were written before the smoke run.**

## 1. Candidate c11 (changelog item 38)

**What it adds:** `candidate_score="hybrid"`, which chooses among the candidates of items 36-37 by `hybrid_cost`.

**`hybrid_cost`, per gate with a reported duration and for each of its qubits:**

- **amplitude damping:** duration / T1 × P(1) on the noiseless state just before the gate, as `excitation_cost`
  counts it;
- **pure dephasing:** p_phi (1 - <Z>^2) just after the gate, with p_phi = (1 - exp(-t / T_phi)) / 2 and
  1 / T_phi = 1 / T2 - 1 / (2 T1);
- **the rest:** the reported error above the thermal floor × (d + 1) / d, as `pauli_cost` counts it.

It is the estimate `hyb` of Addendum 334's diagnosis. A test checks that the two agree to 1e-12.

**Call tested:**

```python
compile_for_hardware(..., target=..., placement_refine=True, final_resynthesis="select", compare_level3=True,
                     compare_floor=True, candidate_score="hybrid")
```

It is tested on every device.

**Unchanged:** any call without `candidate_score="hybrid"` gives release 2026-10-03.3's circuit (checked by test).

**Why** (Addendum 335, exploratory, in-sample on HOLD5's circuits):

- On all nine devices, the choice by `hybrid_cost` among the release's candidates was better than by `pauli_cost`
  (0.06-0.51%) and better than release 2026-10-03.2's choice (0.02-2.2%).
- It came within 0.2% of the measured best.

## 2. Candidate a8 (front end, its item 13)

**The defect.** Above `SMALL_MAX_QUBITS` (8), a7 compiled without the target: the `target` argument was consumed and
never forwarded.

**What it cost** (WIDE, Addendum 335):

- 1.14-2.84 times the release's infidelity;
- failed elements used in 302 of 1,188 circuits.

**The change.** With a target, a8 hands such circuits to the release's recommended call:

- `target`, `placement_refine=True`, `final_resynthesis="select"`, `compare_level3=True`;
- caller kwargs override these.

Without a target, and at or below 8 qubits, a8 is a7 (checked by test).

## 3. Design (`benchmarks/hold6_eval.py`)

**Circuits** (new seeds):

| set | what it is | per device | seeds | simulated if touched qubits <= |
|---|---|---|---|---|
| F1-F6 | HOLD5's generator code and per-cell sizes | 1,506 | base 90,000,000 + ... | 11 |
| W1-W6 | WIDE's generator code at 9-10 logical qubits | 72 | base 95,000,000 + ... | 12 |

**The W families:**

| family | sizes |
|---|---|
| W1 rings | n 10, L 2 and 4 |
| W2 QAOA | n 10, p 1 and 2 |
| W3 XXZ | n 10, open and periodic |
| W4 brickwork | n 9 and 10 |
| W5 GHZ | n 9 and 10 |
| W6 QFT | n 9 |

**Arms:**

| arm | what it is |
|---|---|
| R3 | release 2026-10-03.3 as recommended: on cx devices 2026-10-03.2's call + `compare_floor=True, candidate_score="pauli"`; on cz devices 2026-10-03.2's call |
| C11 | the candidate with the call above, on every device |
| A7 | the adopted front end with the target |
| A8 | the candidate front end with the target (W families only) |
| L3T | Qiskit level 3 with the Target |

**Size:** 486 jobs (216 F, 270 W).

**Devices:** HOLD's nine.

**Metric:** as in GAP. The noiseless P0 check uses Aer's statevector method.

## 4. Predictions (scored only by `hold6_eval.py score`; written before the smoke run)

**P0, harness.** All of these must hold, or nothing below is scored:

- 486 job files;
- every noiseless infidelity <= 1e-6;
- at most 5% of the F circuits and 30% of the W circuits too wide.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | c11 never costs on average (F) | C11/R3 <= 1.00 on at least 8 of 9 devices | > 1.01 on any |
| H2 | it gains on cx devices (F) | C11/R3 < 1.00 on all 4 cx devices | > 1.003 on 2 or more |
| H3 | it repairs F3 open chains (cx) | F3 open C11/R3 <= 1.00 on at least 3 of 4 cx devices | > 1.01 on 2 or more |
| H4 | it keeps the GHZ gains (cx) | F5 C11/R3 <= 1.01 on all 4 cx devices | > 1.03 on any |
| H5 | it does not cost on cz devices (F) | C11/R3 <= 1.00 on at least 4 of 5 cz devices | > 1.005 on 2 or more |
| H6 | it stays on the target | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C11 (F and W) and A8 (W) | any |
| H7 | it stays cheap (F) | median compile time C11 <= 3 × R3 | > 10 × R3 |
| H8 | it keeps the lead over level 3 (F) | C11/L3T <= 1.00 on all 9 devices | > 1.02 on any |
| H9 | a8 repairs the large-circuit path (W) | A8/A7 <= 0.95 on at least 7 of 9 devices | > 1.00 on 2 or more |
| H10 | a8 is ahead of level 3 there (W) | A8/L3T <= 1.00 on at least 7 of 9 devices | > 1.03 on 3 or more |
| H11 | c11 does not cost at width (W) | C11/R3 <= 1.00 on at least 7 of 9 devices | > 1.02 on 2 or more |

**How the thresholds were set** (disclosed):

- **c11's in-sample figures** (Addendum 335, HOLD5's circuits):
  - against C10, which is R3 on the cx devices: 0.9949-0.9989 on the cx devices;
  - against C9, which is R3 on the cz devices: 0.9971-0.9998 on the cz devices;
  - PICK_hyb / PICK_pauli on F3 open: 0.979-0.993 on the cx devices;
  - on F5: within 1.2%, the exception being FakeAlgiers at 1.012.
- **H1, H2 and H5 allow for new seeds.** The in-sample gains are small: 0.02-0.5%.
- **H4's threshold** (1.01) is set at FakeAlgiers' in-sample value. So H4 may well come out ambiguous.
- **a8's figures** (WIDE, Addendum 335): above 8 qubits, A7/R2 was 1.14-2.84, and R2/L3T was 0.951-0.998. a8's call
  is R2's.
- **The scorer** was run on HOLD5 and WIDE data, renamed, as a plumbing check: C9 and R2 as R3 and A8; C10 as C11.
  Those verdicts are not results.

**Expectations, stated with the predictions:**

- **H5 is the least certain.** On cz devices c11 adds the floor candidate and the new estimate, against R3 = .2's call.
  The in-sample gain there was only 0.02-0.3%.
- **H11:** at 9-10 qubits the floor candidate was rarely distinct in WIDE, and F3-type losses are small.
- **On cz devices, A8 and R3 make the same call**, so their rows should be identical. This is reported as a check.

**Reported without prediction:**

- the device and cell tables for both sets;
- the choices;
- compile times;
- failed uses counted both ways;
- off-target instructions;
- too-wide circuits by arm.

## 5. Development (disclosed)

### 5.1 How the candidates were built

Both files were generated by scripts.

**c11**, from release 2026-10-03.3. The script inserts:

- `hybrid_cost`;
- the "hybrid" value in `_choose` and in the argument check;
- changelog item 38;
- the version lines.

**a8**, from [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) (a7). The script inserts:

- item 13 in the docstring;
- two constants;
- the target-aware branch in the large-circuit path;
- the version line.

Nothing else was changed.

### 5.2 Checks run before the stage (no qiskit in this environment)

- **`hybrid_cost` against the diagnosis's estimate.** It was checked on 200 random mock circuits with random
  unitaries, errors, durations and T1/T2. The largest relative difference was 7.7e-16.
- **The scorer** was run on renamed HOLD5 and WIDE data (section 4).
- **The runner** was run with a stub.

### 5.3 Tests and smoke run (not a result)

**Tests** (at home, 2026-10-04 afternoon, on the staged files): `test_c11_hybrid.py` (10 cases) and `test_a8.py`
(7 cases), 17 of 17 passed in 16 s.

**The 17 cases check:**

- the version strings;
- that any call without "hybrid" gives the release's circuit;
- that bad arguments raise `ValueError`;
- that `hybrid_cost` matches a reference computed with Qiskit's `Statevector` (P(1) before each gate, <Z> after it);
- that `hybrid_cost` equals the diagnosis's estimate (to 1e-12);
- on five devices, that the full choice:
  - is exact;
  - is on the target;
  - uses no failed qubit or failed direction;
  - has no higher `hybrid_cost` than the release's own circuit or level 3's;
- that a8 is a7 below 9 qubits and without a target;
- that above 8 qubits with a target, a8 gives exactly the release's recommended call, exact and free of failed
  elements, on FakeHanoiV2, FakeAlgiers, FakeTorino and FakeAachen.

**Smoke run:** 1 circuit per cell and its own seeds; 486 jobs, 848 s.

- No job failed. The runner's check for a Traceback or STOP found none.
- Nothing in the candidates or the evaluation scripts was changed after it.
- The predictions above were written before it, and were not changed.
- **Its verdict lines:**

  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H9 | H10 | H11 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | CONFIRMED | AMBIGUOUS | CONFIRMED | AMBIGUOUS | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

- **H2** was ambiguous because FakeGeneva came out at 1.0000.
- **H4** was ambiguous because FakeAlgiers F5 came out at 1.0125, as in-sample.
- **C11/R3 (F):** 0.9922-1.0007.
- **A8/A7 (W):** 0.36-0.89.
- **A8/L3T (W):** 0.970-1.000.
- **Median compile time:** F R3 0.124 s, C11 0.187 s; W A7 0.114 s, A8 0.255 s.
- **Failed directions or qubits:** 0 by C11 and A8, 629 by A7.
- **Too wide (W):** 12-20 of 99 per arm.

## 6. What this will not establish

- **Hardware.**
- **ecr devices.**
- **Circuits wider than 10 logical qubits.**
- **c11 at width beyond W's 72 circuits per device.**

## 7. Locked files (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c11_2026-10-04/psf_compile.py`](../../patches/psf_compile_c11_2026-10-04/psf_compile.py) | `726defb75c0e911c38565da0c1e7b7ca20079c7c54a5be1a72cb9b78004c0463` |
| [`patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py`](../../patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py) | `33bebe2da77d0b870e34d2bc29d57397e39050c0208f8f0f753ca95c51f54318` |
| [`patches/psf_ai_compile_a8_2026-10-04/psf_ai_compile.py`](../../patches/psf_ai_compile_a8_2026-10-04/psf_ai_compile.py) | `3f17afbf0aafbbeede6fe6e9c0a91c41db0f56b2d57b6b69dbc1d6f654df719c` |
| [`patches/psf_ai_compile_a8_2026-10-04/test_a8.py`](../../patches/psf_ai_compile_a8_2026-10-04/test_a8.py) | `b28239c46b310bd97fc7ca69e0543e60163a3a2a2e80c861ec00027b99fefd20` |
| [`benchmarks/hold6_eval.py`](../../benchmarks/hold6_eval.py) | `8740a33225f24da12f1d07c235695950f03643d4160283eb1623ad9d7f05108a` |
| [`benchmarks/run_hold6_2026-10-04.sh`](../../benchmarks/run_hold6_2026-10-04.sh) | `a1e5ff74eafc29bff3f3614d96a75af00d55297f4a41fe5f1bd3dfead1dbf6c7` |

Normalization: CRLF to LF, trailing whitespace stripped from each line, trailing blank lines dropped, lines joined
with "\n" and no final newline.


---

<!-- ===== Addendum 337 (source: spare-qubit-cliff-addendum-337-2026-10-04.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 336 (lock commit ad737a6), scored by the locked script and re-checked by benchmarks/hold6_verify.py, written after the run finished and before its output files were read.

## Addendum 337 -- Results: HOLD6 (Addendum 336). Candidate c11 (choice by `hybrid_cost`) improves on release 2026-10-03.3's recommended call on all nine devices (0.02-0.70%), repairs the F3 open-chain loss on the cx devices (0.8-2.3%) and keeps the lead over Qiskit L3T; candidate front end a8 repairs the large-circuit path (A8/A7 0.375-0.871, no failed element, ahead of L3T on all nine devices). Ten confirmed, one ambiguous (H4: FakeAlgiers F5 1.0125, from 4-qubit GHZ chains only), none refuted (2026-10-04)

**Status: results of the pre-registered test in Addendum 336.**

- **Lock:** commit `ad737a6`, pushed before the scored run (started 15:04 JST).
- **Same bytes:** the run used the locked files. The SHA-256 values in `env.txt` are those checked against the stage
  before the lock, and every job file records the normalized SHA-256 of the script and of both candidates.
- **Scoring:** by the locked `hold6_eval.py score`, and re-checked by [`benchmarks/hold6_verify.py`](../../benchmarks/hold6_verify.py), which agrees on
  every verdict.
  - **When the verify script was written:** after the run finished, before its output files were read. Its
    normalized SHA-256 is `62f6efaf4deedb8d40481e2e6cd6038cda8eb6d53556fdee50786bc29de1813a`.
  - **What had been seen by then:** the last three lines of the run log, which hold the too-wide counts by arm.
  - **One fix, before any read:** the first version of the script stopped on the renamed plumbing data, which lacks the
    new meta fields. It was changed to read them with `.get`.
- **Setting:** home (WSL2), PAR 6; 486 jobs, 5,605 s.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| P0 | **PASS** | 486 of 486 files; noiseless max 9.3e-10; too wide 0 of 54,216 (F), 447 of 3,240 (W, 13.8%) |
| H1 | **CONFIRMED** | F, C11/R3 <= 1.00 on 9 of 9 devices: 0.9930-0.9998 |
| H2 | **CONFIRMED** | F, cx devices, C11/R3 < 1.00 on all 4: Auckland 0.9973, HanoiV2 0.9930, Algiers 0.9971, Geneva 0.9988 |
| H3 | **CONFIRMED** | F3 open, cx devices, C11/R3: Auckland 0.984, HanoiV2 0.978, Algiers 0.980, Geneva 0.992 |
| H4 | **AMBIGUOUS** | F5, cx devices, C11/R3: Auckland 1.000, HanoiV2 0.999, Algiers 1.0125, Geneva 1.000 (threshold 1.01; none > 1.03) |
| H5 | **CONFIRMED** | F, cz devices, C11/R3 <= 1.00 on 5 of 5: 0.9971-0.9998 |
| H6 | **CONFIRMED** | 0 failed-direction or failed-qubit uses and 0 off-target instructions by C11 and A8 |
| H7 | **CONFIRMED** | F, median compile time C11 0.155 s, R3 0.086 s (1.8 ×) |
| H8 | **CONFIRMED** | F, C11/L3T <= 1.00 on 9 of 9 devices: 0.959-0.991 |
| H9 | **CONFIRMED** | W, A8/A7 <= 0.95 on 9 of 9 devices: 0.375-0.871 |
| H10 | **CONFIRMED** | W, A8/L3T <= 1.00 on 9 of 9 devices: 0.975-0.997 |
| H11 | **CONFIRMED** | W, C11/R3 <= 1.00 on 8 of 9 devices: 0.9978-1.0001 (FakeAachen 1.0001) |

## 2. Numbers

**F (HOLD sizes):**

| device | C11/R3 | C11/L3T | R3/L3T | F3 open C11/R3 | F5 C11/R3 | per circuit C11 better / worse |
|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.9973 | 0.978 | 0.980 | 0.984 | 1.000 | 11.4% / 3.1% |
| FakeTorino | 0.9995 | 0.986 | 0.986 | 1.000 | 1.000 | 5.3% / 1.3% |
| FakeKingston | 0.9984 | 0.989 | 0.990 | 0.989 | 1.000 | 10.0% / 1.3% |
| FakeHanoiV2 (cx) | 0.9930 | 0.980 | 0.987 | 0.978 | 0.999 | 19.2% / 4.8% |
| FakeAlgiers (cx) | 0.9971 | 0.976 | 0.979 | 0.980 | 1.0125 | 11.2% / 5.2% |
| FakeGeneva (cx) | 0.9988 | 0.959 | 0.960 | 0.992 | 1.000 | 8.2% / 1.7% |
| FakeFez | 0.9992 | 0.991 | 0.992 | 1.000 | 1.000 | 7.4% / 1.3% |
| FakeMarrakesh | 0.9971 | 0.963 | 0.966 | 0.990 | 0.950 | 14.3% / 0.5% |
| FakeAachen | 0.9998 | 0.986 | 0.986 | 1.000 | 1.000 | 3.4% / 0.9% |

**W (9-10 qubits):**

| device | A8/A7 | A8/L3T | A7/L3T | C11/R3 | A8/R3 |
|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.871 | 0.997 | 1.149 | 0.998 | 0.999 |
| FakeTorino | 0.545 | 0.992 | 1.819 | 0.999 | 1.000 |
| FakeKingston | 0.809 | 0.993 | 1.225 | 0.999 | 1.000 |
| FakeHanoiV2 (cx) | 0.791 | 0.990 | 1.242 | 0.999 | 1.002 |
| FakeAlgiers (cx) | 0.648 | 0.993 | 1.546 | 0.998 | 1.002 |
| FakeGeneva (cx) | 0.829 | 0.975 | 1.184 | 0.999 | 1.010 |
| FakeFez | 0.665 | 0.995 | 1.497 | 0.999 | 1.000 |
| FakeMarrakesh | 0.375 | 0.979 | 2.660 | 1.000 | 1.000 |
| FakeAachen | 0.495 | 0.993 | 1.823 | 1.000 | 1.000 |

**Identity check.** On the cz devices A8 and R3 make the same call. Their rows are identical in 297 of 297 simulated
pairs.

**Failed elements by arm** (direction ignored; by direction):

| arm | direction ignored | by direction |
|---|---|---|
| R3 | 159 | 0 |
| C11 | 159 | 0 |
| A8 | 140 | 0 |
| L3T | 188 | 0 |
| A7 | 4,623 | 4,509 |

All of A7's are in W, its target-blind path.

**Too wide in W:**

| arm | too wide |
|---|---|
| R3 | 78 |
| C11 | 78 |
| A8 | 82 |
| L3T | 93 |
| A7 | 116 |

The W ratios use only circuits simulated in both arms compared.

The full tables are in `outputs/score.md`; the re-computation is in `outputs/verify.txt`.

## 3. Reading

**c11:**

- **It does what the diagnosis said it would, on fresh circuits.**
  - It is better than the release's recommended call on all nine devices: 0.1-0.7% on the cx devices, 0.02-0.3% on the
    cz devices.
  - Per circuit, it is better 2-29 times as often as worse.
- **The F3 loss of `pauli_cost` is repaired** on the cx devices (0.8-2.3%).
- **It brings the floor candidate to the cz devices**, where R3 has none. On FakeMarrakesh F5 that gains 5% (4-qubit
  chains 18%, 6-qubit chains 5%).
- **The one miss is narrow and systematic** (H4).
  - On FakeAlgiers, for 4-qubit GHZ chains, `hybrid_cost` keeps the release's circuit in all 48 cases, where
    `pauli_cost` takes the floor-placed one. That is 6.7% worse for those circuits; 6- and 8-qubit chains are equal.
  - The same cell was 1.012 in-sample and 1.0125 in the smoke run, so it is a fixed ranking error on one placement,
    not noise.
  - `hybrid_cost` counts damping from P(1) at 1/2 on every qubit of a GHZ chain. A plausible cause is that this
    damping term outweighs a dephasing difference that `pauli_cost` weights more. That has not been checked.
- **Cost:** 1.8 × the compile time of R3 (0.155 s median), because every device now compiles the floor candidate.
- **At 9-10 qubits it changes little** (0.998-1.000), as expected: the floor candidate is rarely distinct there.

**a8:**

- **The large-circuit path is repaired.**
  - A8 is 13-63% better than a7 and 0.3-2.5% ahead of L3T on every device.
  - It uses no failed element. a7 used failed directions or qubits 4,509 times.
- **It equals the release's call, as designed.** On cz devices this is exact (297 of 297). On cx devices A8 is .2's
  call, so it lacks the floor/pauli option of R3 (A8/R3 0.999-1.010).

## 4. Consequences

**Adoption** is the owner's decision. The data support:

- **c11 as release 2026-10-04.1**, with a single recommended call on every device:

  ```python
  compile_for_hardware(..., target=..., placement_refine=True, final_resynthesis="select", compare_level3=True,
                       compare_floor=True, candidate_score="hybrid")
  ```

  - This replaces the per-device recommendation of 2026-10-03.3.
  - The FakeAlgiers 4-qubit GHZ case should be listed as a known limit.
- **a8 as the adopted front end.**
  - When c11 is adopted, a8's large-circuit call could also take c11's call (`FAST_PATH_RECOMMENDED`). That would be
    a later change, with its own test.

## 5. Data (`data/2026-10-04/hold6/outputs/`)

- 486 job files and their logs;
- `env.txt` (local paths replaced), `progress.txt`, `score.md`, `score_log.txt`, `verify.txt`.


---

<!-- ===== Addendum 338 (source: spare-qubit-cliff-addendum-338-2026-10-04.md) ===== -->

> **Note added when merging:** Adoption record: psf_compile 2026-10-04.c11 becomes release 2026-10-04.1 (one recommended call on every device), and psf_ai_compile 2026-10-04.a8 becomes the AI front end (owner's decision, 2026-10-04).

## Addendum 338 -- Adoption record: candidate psf_compile 2026-10-04.c11 becomes release 2026-10-04.1 (one recommended call on every device, `candidate_score="hybrid"`, changelog item 38), and candidate psf_ai_compile 2026-10-04.a8 becomes the AI front end (2026-10-04)

**Status: adoption record.**

- **Decision:** the owner's, on 2026-10-04, after the results in Addendum 337 (ten of eleven predictions confirmed,
  one ambiguous, none refuted).
- **Scope:** fake devices and Aer noise only. Nothing here was run on hardware.

## 1. What the release is

`psf_compile.py` 2026-10-04.1 is the candidate file
[`patches/psf_compile_c11_2026-10-04/psf_compile.py`](../../patches/psf_compile_c11_2026-10-04/psf_compile.py) with
only its version lines changed: the `VERSION:` header line, the changelog heading of item 38, and the `VERSION` constant.

**Recommended call with a device target, the same on cx and cz devices:**

```python
out = compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
                           target=backend.target, placement_refine=True, final_resynthesis="select",
                           compare_level3=True, compare_floor=True, candidate_score="hybrid")
```

This replaces 2026-10-03.3's two recommendations, one for cx devices and one for cz devices. Any call without
`candidate_score="hybrid"` gives exactly 2026-10-03.3's circuit.

**Unchanged:**

- `psf_smart_layout` (2026-10-01.1);
- the Rust core (`CORE_VERSION` 2026-09-29.1).

## 2. What the front end is

**[`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) is now a8.** It is the candidate file
[`patches/psf_ai_compile_a8_2026-10-04/psf_ai_compile.py`](../../patches/psf_ai_compile_a8_2026-10-04/psf_ai_compile.py)
with two changes, and nothing else:

- its item-13 heading marked as adopted;
- its version comment.

**a7 is kept as [`benchmarks/psf_ai_compile_a7.py`](../../benchmarks/psf_ai_compile_a7.py).** That file is byte-identical to the a7 that was adopted on
2026-10-02.

**Above 8 qubits with a target, a8 makes 2026-10-03.2's recommended call.** That call was the call fixed when a8 was
tested. Switching that path to this release's recommended call would be a further change, with its own test.

## 3. Files

**Added:**

| file | what it is |
|---|---|
| [`benchmarks/test_release_2026_10_04_1.py`](../../benchmarks/test_release_2026_10_04_1.py) | the candidate's 10 tests, adapted. The previous release is represented by [`patches/psf_compile_c10_2026-10-03/psf_compile.py`](../../patches/psf_compile_c10_2026-10-03/psf_compile.py), which differs from 2026-10-03.3 only in its version lines |
| [`benchmarks/test_ai_compile_a8.py`](../../benchmarks/test_ai_compile_a8.py) | the candidate's 7 tests, adapted |
| [`benchmarks/psf_ai_compile_a7.py`](../../benchmarks/psf_ai_compile_a7.py) | the frozen copy of a7 |

**Changed:**

- `psf_compile.py` and [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py): as described above.
- [`benchmarks/test_ai_compile_a7.py`](../../benchmarks/test_ai_compile_a7.py): it now loads the frozen a7.
- **Current-release assertions in fourteen tests.** Only the expected version string changed (to "2026-10-04.1").
  - Seven of these tests are tests of earlier candidates whose files were locked by pre-registrations: c4, c6, c8, c9,
    c10, c11 and a6.
  - c11's test also had its comment reworded.
- **[`patches/psf_ai_compile_a8_2026-10-04/test_a8.py`](../../patches/psf_ai_compile_a8_2026-10-04/test_a8.py)** (locked, Addendum 336) had two changes:
  - it now loads a7 from the frozen file, so that it still compares a8 with a7;
  - its release assertion now expects 2026-10-04.1.
- **`README.md`:**
  - a new block for the current version;
  - the 2026-10-03.3 block retitled "Previous release";
  - a block for the AI front end a8.

**Normalized SHA-256, before and after:**

| file | before | after |
|---|---|---|
| `psf_compile.py` | `6b2b211ecd4815bb…` | `7230adf00f152592…` |
| [`benchmarks/psf_ai_compile.py`](../../benchmarks/psf_ai_compile.py) | `0064e8cf7efef311…` | `5dd7f3a4b2d0b2aa…` |
| [`benchmarks/test_ai_compile_a7.py`](../../benchmarks/test_ai_compile_a7.py) | `c071be742e7e8ee4…` | `f61c0f20d16e4959…` |
| [`patches/psf_ai_compile_a8_2026-10-04/test_a8.py`](../../patches/psf_ai_compile_a8_2026-10-04/test_a8.py) | `b28239c46b310bd9…` | `91ed5c69f89e6025…` |
| [`benchmarks/test_release_2026_10_02.py`](../../benchmarks/test_release_2026_10_02.py) | `a01a54a9f820d50f…` | `511e21b85fe0a7b9…` |
| [`benchmarks/test_release_2026_10_02_2.py`](../../benchmarks/test_release_2026_10_02_2.py) | `8799a6f7e75b5ac7…` | `8864758981e60b9f…` |
| [`benchmarks/test_core_fix_c2.py`](../../benchmarks/test_core_fix_c2.py) | `a7636d31e2d89473…` | `138fd4f759b9a9a9…` |
| [`benchmarks/test_release_2026_09_28.py`](../../benchmarks/test_release_2026_09_28.py) | `61b315174b2787eb…` | `5dfd12f99480a31c…` |
| [`benchmarks/test_release_2026_10_03.py`](../../benchmarks/test_release_2026_10_03.py) | `2bc57d69eb9419f3…` | `dc5ebfac5aa91ad3…` |
| [`benchmarks/test_release_2026_10_03_2.py`](../../benchmarks/test_release_2026_10_03_2.py) | `a89fa11e39460963…` | `7ab1bb5da6ab5b28…` |
| [`benchmarks/test_release_2026_10_03_3.py`](../../benchmarks/test_release_2026_10_03_3.py) | `803314e759431e41…` | `0a1afebd230bdd4b…` |
| [`patches/psf_compile_c4_2026-10-02/test_c4_layout.py`](../../patches/psf_compile_c4_2026-10-02/test_c4_layout.py) | `9d74bfe499d3709a…` | `e508c6c45a4026b5…` |
| [`patches/psf_compile_c6_2026-10-03/test_c6_floor.py`](../../patches/psf_compile_c6_2026-10-03/test_c6_floor.py) | `8017ec31ff4437e1…` | `980723c9168768d8…` |
| [`patches/psf_compile_c8_2026-10-03/test_c8_resynth.py`](../../patches/psf_compile_c8_2026-10-03/test_c8_resynth.py) | `03cc3b33b906ee66…` | `9dd1da5f6aa9e37d…` |
| [`patches/psf_compile_c9_2026-10-03/test_c9_compare.py`](../../patches/psf_compile_c9_2026-10-03/test_c9_compare.py) | `113291666a59677f…` | `c5b6c74b94f1e0be…` |
| [`patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py`](../../patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py) | `67f160551a7f99a0…` | `5152354e9d67cc1e…` |
| [`patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py`](../../patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py) | `33bebe2da77d0b87…` | `00648dc337d1c285…` |
| [`patches/psf_ai_compile_a6_2026-10-02/test_ai6.py`](../../patches/psf_ai_compile_a6_2026-10-02/test_ai6.py) | `88e38220498e073a…` | `69372876c421da9c…` |
| `README.md` | `7ec634f933b819a3…` | `aebc211d020557b4…` |

**Part 9:** from Addendum 334 on, code-formatted paths that exist in the repository were turned into relative links.
A line-by-line check confirmed that only link syntax changed.

## 4. Known limits (Addendum 337)

- **FakeAlgiers, 4-qubit GHZ chains:** 6.7% worse than with `candidate_score="pauli"`, in all 48 such circuits. It is a
  fixed ranking error of `hybrid_cost` on one placement. Its cause is being diagnosed.
- **Tested on fake devices only.** Not on hardware, not on ecr devices, and not above 16 touched qubits.
- **Stale calibration:** the release was not tested with one. 2026-10-03.3 was (Addendum 335).
- **The locked evaluation scripts of Addenda 334 and 336** stop unless the release is 2026-10-03.3. They reproduce
  their results only at their lock commits (`a88df77`, `ad737a6`).

---

---

**End of Part 9 of 9 (end of document, for now).** Back to [Part 8](spare-qubit-cliff-combined-135.md), [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
