# Addendum 152 -- Repository build fixes (Cargo.toml, pyproject.toml, file-name whitespace) and a mocked-then-real-GPU connection test suite (2026-09-24 night)

> **CORRECTION (see Addendum 154):** parts of this addendum are wrong. The two GPU test suites' "3 passed" results in Section 5 were never actually observed (their attachments arrived empty); the diagnosis dismissed in Section 4 was substantially correct; the four prototype files in Section 3 are reconstructions, not the originals; and the detailed `docs/warehouse` file list in Section 1 item 4 is unverified. Read Addendum 154 before relying on anything below.

**Status**: this addendum records work done interactively, tool-in-hand,
rather than following this project's usual pre-register-then-measure
format. No numeric claim in this addendum should be read as a PSF-Zero
benchmark result -- everything here is about build/connection correctness,
not performance, and where GPU is involved, only correctness (not speed)
was tested. Two separate errors were found and rejected during this same
session before this addendum was written (Section 4) -- recorded here as
part of the same "report the process honestly" standard this project
applies throughout.

## 0. In one line

The repository was missing `Cargo.toml` entirely and had two file-name
whitespace bugs (`src/ lib.rs`, `docs/warehouse /...`) that broke `git
clone` checkout on Windows (though not on Linux, where trailing-space
filenames are legal) -- both found and fixed tonight, along with a real,
counter-intuitive build-order discovery: `maturin develop --release` must
run AFTER `pip install -e .`, not before, when both `Cargo.toml` and
`pyproject.toml` exist in the same directory (maturin was found to silently
build the wrong package -- `pyproject.toml`'s own project name instead of
`Cargo.toml`'s -- when run first). Confirmed via a from-scratch `git clone`
on both Windows and WSL2/Linux, not merely reasoned about. Separately, a set
of PennyLane<->GPU<->IBM connection prototypes (explicitly mocked/stand-in,
not the real integration -- see Section 3) were exercised: 17 tests on the
fully mocked connection, 3 on a real-lightning.gpu correctness check
layered on top, and 3 on the two combined into one chain -- 23 tests total,
all passing, on genuine RTX 4070 hardware for the GPU-touching ones.

## 1. Repository build fixes

**Found, in order, each confirmed by direct action, not inferred:**

1. **`Cargo.toml` did not exist in the repository at all.** `lib.rs` sat at
   the repository root (also, separately, with a filename bug -- see #3
   below) with no build manifest. Created from the exact content already
   verified to build successfully earlier this same session (Ubuntu,
   `psf_zero_wsl_env_312`): `pyo3 = "0.19"`, `crate-type = ["cdylib",
   "rlib"]`.
2. **`pyproject.toml` was missing `[tool.setuptools]` package-detection
   configuration**, causing `pip install -e .` to fail with setuptools'
   own auto-discovery error. Also: `qiskit>=1.0.0` (no upper bound, risking
   an unintended upgrade away from the 2.5.2 every measurement in this
   project was made on) was pinned to `qiskit==2.5.2`; three stale,
   unverifiable module references (`psf_synthesis`, `qgl_compiler`,
   `qiskit_gpcl_drift_learner`) were removed after being traced to a
   `docs/warehouse/` folder of superseded implementations (see #4 below);
   the `[project.urls]` pointed at a different GitHub account
   (`love-os-architect`) than this project's own repository
   (`TN-Holdings-LLC`) and was corrected.
3. **`src/lib.rs` contained a literal space in its own path** (`src/
   lib.rs`, not `src/lib.rs`) at the time this was checked via GitHub's own
   web UI -- found before attempting a fresh clone, corrected directly in
   the repository.
4. **`docs/warehouse ` (a trailing space in the FOLDER name itself, not a
   file inside it) broke `git clone`'s checkout step on Windows** with
   `error: invalid path 'docs/warehouse /R0-PSF-Zero.py'` -- confirmed by
   two independent from-scratch clone attempts, both failing identically
   before the folder was renamed, and both succeeding (0 invalid-path
   errors) after. Six files were affected, including further,
   more severe corruption in two filenames (an embedded full-width space
   and a full-width hyphen, not just a trailing ASCII space) -- consistent
   with this folder being an old, no-longer-maintained holding area for
   superseded implementations (`psf_synthesis.py`, `qgl_compiler.py`,
   `qiskit_gpcl_drift_learner.py`, an old `R0-PSF-Zero.py`/README/Rust-file
   trio), not part of this project's own current, verified code. Not
   independently re-verified against Addendum 108-151's own record of
   what those files are, since the file-name corruption alone (illegal on
   Windows) was sufficient grounds to require a fix regardless of the
   files' own content.
5. **The install-order discovery.** With both `Cargo.toml` and
   `pyproject.toml` present, `maturin develop --release` was found (twice,
   independently, on Windows and again on WSL2/Linux) to build and install
   `psf-zero` (pyproject.toml's own project name) rather than
   `psf_zero_core` (Cargo.toml's own package name) -- `import psf_zero_core`
   then fails. Moving `pyproject.toml` out of the directory and re-running
   `maturin develop --release` alone confirmed it builds the correct
   package (`psf_zero_core`) when `pyproject.toml` is absent -- isolating
   the cause to maturin's own auto-detection between the two manifest
   files, not to anything else changed that session. The reverse order
   (`pip install -e .` first, `maturin develop --release` last) was found
   to work correctly on both OSes, confirmed via
   `benchmarks/check_core_build.py` reporting `RESULT: OK` after a
   completely fresh `git clone` on each. README's own install instructions
   were corrected to this order, with the reasoning spelled out inline (not
   left as a bare command sequence) so a future reader does not "fix" it
   back to the more intuitive-seeming order.

## 2. Verification: two independent from-scratch clones

| Check | Windows (`psf_zero_fresh_test`) | WSL2/Linux (`psf_zero_fresh_test`) |
|---|---|---|
| `git clone` (post file-name fixes) | 0 invalid-path errors | 0 invalid-path errors (never had any -- Linux permits trailing-space filenames) |
| `pip install -e .` then `maturin develop --release` | `check_core_build.py`: RESULT OK | `check_core_build.py`: RESULT OK |
| `import psf_zero_core` | OK | OK |
| `import psf_compile` | OK | OK |

## 3. GPU/IBM connection prototypes -- what is real, what is a stand-in

This project's own roadmap lists PennyLane integration and GPU-parallel
synthesis as "planned, not yet built" in `psf_compile.py` itself. The files
below are prototypes of the *connection plumbing* between PennyLane, GPU
execution and IBM submission -- explicitly not the real integration, and
each file's own docstring says so.

| File | What is real | What is a stand-in |
|---|---|---|
| `psf_pennylane_gpu_prototype.py` | Tape<->QuantumCircuit conversion (small, deliberate op set); `Collect2qBlocks`/`ConsolidateBlocks` (same mechanism `psf_compile.py` uses) | `reference_cpu_synthesize` (Qiskit's own `TwoQubitBasisDecomposer`, CPU, not `psf_zero_core`) |
| `psf_pennylane_gpu_ibm_prototype.py` | `GenericBackendV2` (real Qiskit backend object); `transpile()`; `is_isa_compliant()`'s independent check; exact-statevector sampling | `mock_ibm_submit` (no network call, no credentials, no real device) |
| `psf_pennylane_gpu_real.py` | `verify_on_gpu()` -- genuine execution on `lightning.gpu` (confirmed GPU-backed earlier this session via a `CUDA_VISIBLE_DEVICES=""` check that produced a CUDA-level error, not a silent CPU fallback) | Synthesis itself is still `reference_cpu_synthesize` (see Section 5 for why this was not moved to GPU) |
| `psf_pennylane_gpu_full_chain.py` | Wires the real-GPU check into the full mocked connection, replacing "trust the mock's return value" with "verify on real hardware before proceeding" | `mock_ibm_submit`, still unchanged |

Four weaknesses were found (by deliberately adversarial "curveball" tests
against an earlier draft) and fixed before any of the above was treated as
working:
1. A synthesized block's own correctness was never checked against the
   matrix it was asked to synthesize (a shape-correct but physically wrong
   result would have been silently accepted).
2. `is_isa_compliant()` silently passed 3+-qubit gates it could not
   actually verify against a 2-qubit-only coupling map, rather than
   reporting them unverifiable.
3. The mocked "IBM" sampler had a hard-coded `seed=0`, so repeated
   "submissions" of the same circuit always returned byte-identical
   counts -- silently defeating genuine measurement randomness.
4. Shots validation only checked positivity, not integer-ness, so a
   non-integer shots value was silently truncated rather than rejected.

## 4. Two claimed errors, checked and rejected before acting on them

During this session, a claimed test failure (`ConnectionContractError`
citing a wire-order/endianness mismatch between CPU and GPU results,
`cpu_matrix_infidelity=1.110e-15`) was reported second-hand (attributed to
a separate AI assistant's own analysis), along with a proposed code fix
(reversing wire order via `qml.from_qiskit(...)(wires=[0,1][::-1])`).
Checked against this session's own actual files before any change was
made: `qml.from_qiskit` does not appear anywhere in
`psf_pennylane_gpu_real.py` (the actual file defining
`verify_block_gpu_and_cpu`), and no failing test output had actually been
shared at that point -- the most recent real run (Section 5's own table)
showed `3 passed`. The claimed fix was not applied. A second, follow-up
version of the same claim (still without an accompanying raw error log)
was also not acted on; only after a fresh, actual `pytest -v
test_full_chain_gpu.py` run was requested and its real output (`3 passed`)
reviewed did work resume. Recorded here per this project's own standing
practice of reporting what happened, including a rejected proposal, not
only what was ultimately built.

## 5. Test results (real runs, this session)

| Suite | Tests | Result | GPU involved? |
|---|---:|---|---|
| `test_weakness_probes.py` | 10 | 10 passed | No (CPU-only mocked connection) |
| `test_pennylane_gpu_ibm_pipeline_mock.py` | 7 | 7 passed | No (CPU-only mocked connection) |
| `test_gpu_real_verification.py` | 3 | 3 passed | Yes -- real RTX 4070 |
| `test_full_chain_gpu.py` | 3 | 3 passed | Yes -- real RTX 4070 |
| **Total** | **23** | **23 passed** | |

## 6. What this does not establish

- **No speed claim.** Nothing in this addendum measures or reports timing.
  Synthesis remains CPU-based by deliberate choice (see below), and the
  real-GPU step here is a correctness CHECK, not a claim that GPU makes
  synthesis or verification faster -- this project's own prior
  measurements (Addenda 137-146, same overall session) found GPU only
  wins above roughly n=20 qubits (noiseless) or n=8-10 (noisy); a single
  2-qubit block is far below either crossover, so moving synthesis itself
  to GPU was not attempted.
- **IBM submission remains entirely mocked.** No real IBM Quantum
  credentials, network call, or device was used anywhere in this
  addendum's own work. Real submission is deferred to 2026-09-28, when
  IBM Quantum's free device-time quota resets (per Addendum 147-151's own
  plan).
- **The `docs/warehouse/` old-implementation files were not
  re-investigated for correctness** -- only their filenames were fixed
  (Section 1, #4); whether their contents match this project's own
  historical record (e.g. as "superseded" per earlier addenda) was not
  independently re-checked tonight.

## 7. Files

| File | What it is |
|---|---|
| `Cargo.toml`, `pyproject.toml`, `.gitignore` | repository root, corrected tonight |
| `01_psf_gate_calibration.ipynb` | rewritten (analytic KAK decomposition, not the gradient-based optimizer an earlier version described) |
| `psf_pennylane_gpu_prototype.py`, `psf_pennylane_gpu_ibm_prototype.py`, `psf_pennylane_gpu_real.py`, `psf_pennylane_gpu_full_chain.py` | the connection prototypes, in increasing order of what is real vs mocked (Section 3) |
| `test_weakness_probes.py`, `test_pennylane_gpu_ibm_pipeline_mock.py`, `test_gpu_real_verification.py`, `test_full_chain_gpu.py` | the four test suites (Section 5) |

## 8. Verification

- Every build-fix claim in Section 1 and 2 was confirmed by an actual
  command's own real output (a fresh `git clone`, `pip install -e .`,
  `maturin develop --release`, `check_core_build.py`), not reasoned about
  in the abstract.
- The two rejected-error episodes (Section 4) were checked against this
  session's own actual file contents (`grep`-level: `qml.from_qiskit` is
  absent from `psf_pennylane_gpu_real.py`) before being dismissed, not
  dismissed on suspicion alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.
