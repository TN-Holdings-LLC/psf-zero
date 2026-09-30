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

---

**End of Part 9 of 9 (end of document, for now).** Back to [Part 8](spare-qubit-cliff-combined-135.md), [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
