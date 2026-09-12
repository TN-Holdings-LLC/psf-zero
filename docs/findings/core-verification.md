# Numerical rigor and core verification in PSF-Zero

**Status:** the Haar-random sweep is solid and unaffected by anything below. The
near-CNOT sweep had a bug in its test construction (not in `psf_zero_core`), found
and fixed 2026-09-12 — see the correction at the end of this file before quoting the
near-CNOT number.

## What this verifies

PSF-Zero's decomposition is closed-form, not search-based, so the failure mode to
guard against isn't search quality — it's numerical stability: around the CNOT/SWAP
degeneracies, and in the agreement between the Rust core's 4×4 matrix reconstruction
and Qiskit's own `Operator(qc)` builder. `benchmarks/verify_core_infidelity.py` locks
both with an automated three-tier check (`cargo test --release`, then two Python
sweeps) run before every measurement in this file.

## Results (2026-09-12, post-correction)

| Suite | Samples | Worst infidelity (core) | Worst infidelity (strict circuit) | Fallbacks |
| :--- | :---: | :---: | :---: | :---: |
| Haar-random SU(4) | 500 | 8.88e-16 | 6.66e-16 | **0 / 500** |
| Near-CNOT (ε = 1e-7, corrected) | 200 | 2.63e-14 | (covered by the strict loop) | **0 / 200** |

- **Haar space:** worst-case infidelity sits near machine epsilon, with zero fallback
  exceptions across 500 samples.
- **Near-CNOT:** with the perturbation bug below fixed, the worst case is ~2 orders of
  magnitude above the Haar figure — the direction you'd expect near a codimension-2
  degeneracy — and the candidate-scoring plus Givens-sweep handling of the degenerate
  point still produces zero rejections at that harder point.
- **Rust↔Python agreement:** the `strict` tier reconstructs the circuit via Qiskit's
  `Operator(qc)` and compares against the Rust core's own math, ruling out endian
  mismatches or ZYZ phase/sign drift between the two sides.

Reproduce: `maturin develop --release && python benchmarks/verify_core_infidelity.py`.
Raw data: [`data/core_verification_2026-09-12.csv`](../../data/core_verification_2026-09-12.csv).

## Correction (2026-09-12): `near_cnot()`'s perturbation was not near CNOT

The near-CNOT sweep is meant to stress-test the codimension-2 CNOT singularity by
perturbing it with `eps = 1e-7` and checking that the decomposition stays accurate as
`eps → 0`. The first version did not do that.

**What was wrong.** The perturbed matrix was built as

```python
mixed = (1 - eps) * cnot + eps * pert
uu, ss, vv = np.linalg.svd(mixed)
u = uu @ vv
u = u / np.linalg.det(u) ** 0.25          # <- global phase fix, the problem
```

`det(u)**0.25` has four branches, and which one NumPy's `**0.25` lands on isn't
controlled for. Measured directly: for `eps` from `1e-1` down to `1e-9`, the resulting
`u` sits at operator-norm distance **0.765 from CNOT**, unchanged regardless of how
small `eps` is made — the phase-fixing step, not the perturbation, was deciding the
outcome. Correcting only for that global phase (fixing it from the overlap
`tr(CNOT^dagger u)` instead of `det(u)**0.25`) recovers the intended behaviour exactly:
the same construction at `eps=1e-7` then sits `~1e-7` from CNOT, as it should.

**What this means for the number originally published (1.68e-13).** The infidelity
metric `check_core`/`check_strict` compute
(`1 - (|tr(U^dagger V)|^2 + d) / (d(d+1))`) is itself phase-invariant, so that number
was not wrong on its own terms. What it measured was different from what the section
claimed: with the phase bug in place, `pert`'s contribution to `mixed` after SVD
projection is essentially an independent Haar-random unitary at operator-norm distance
0.765 from CNOT — the sweep was a second Haar-random test, not a stress test of the
singularity. 0.765 is on the order of the diameter of SU(4) under this norm, so "near"
did not hold.

**Fix applied**, replacing the phase-fixing line with one anchored to the overlap with
CNOT itself:

```python
phase = np.trace(cnot.conj().T @ u) / 4
if abs(phase) > 1e-12:
    u = u / (phase / abs(phase))
```

**Re-run on the project's machine**, 200 samples, `eps=1e-7`:

| | before fix | after fix |
| :--- | ---: | ---: |
| Worst infidelity | 1.68e-13 | **2.631e-14** |
| Rejections | 0/200 | 0/200 |

The corrected run is genuinely close to the CNOT singularity (`~1e-7` in operator
norm, confirmed directly) and its worst-case infidelity is about 6x *smaller* than the
figure this file previously reported — not larger, so the corrected number does not
weaken the "robust near the singularity" claim; if anything the original number
overstated the difficulty by testing something else. It lands roughly two orders of
magnitude above the Haar-random worst case (8.88e-16 core / 6.66e-16 strict, same
run), which is the expected direction — a true neighbourhood of a codimension-2
degeneracy should be numerically harder than a generic point, and 0 rejections at
that harder point is the result worth having.

**What this does not touch.** The Haar-random sweep (500 samples) does not use
`near_cnot()` and is unaffected. The core decomposition math, the candidate-scoring
and Givens-sweep handling of the degenerate point, and `verify="strict"`'s
Python↔Rust agreement check are all unchanged — the bug was in the *test's*
perturbation construction, not in `psf_zero_core` or `psf_compile.py`.

The pre-fix `near-CNOT` figure (1.68e-13) is superseded by the corrected one above and
should not be quoted as a singularity-robustness result going forward. The results
table at the top of this file already reflects the corrected run.

## Files

- Harness: [`benchmarks/verify_core_infidelity.py`](../../benchmarks/verify_core_infidelity.py)
- Raw data (post-correction): [`data/core_verification_2026-09-12.csv`](../../data/core_verification_2026-09-12.csv)
