# PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-purple.svg)](https://github.com/qiskit/ecosystem)
[![Rust Core](https://img.shields.io/badge/Core-Rust_Native-E34F26.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3 Binding](https://img.shields.io/badge/FFI-PyO3-blue.svg)](https://pyo3.rs/)
[![DOI (Paper 1)](https://zenodo.org/badge/DOI/10.5281/zenodo.22977930.svg)](https://doi.org/10.5281/zenodo.22977930)
[![DOI (Paper 2)](https://zenodo.org/badge/DOI/10.5281/zenodo.22978090.svg)](https://doi.org/10.5281/zenodo.22978090)

PSF-Zero is a Qiskit-compatible compiler with two layers.

- **`compile()`: fast, exact two-qubit synthesis.** It replaces heuristic two-qubit unitary synthesis with a
  closed-form Cartan (KAK) decomposition in a small Rust core. It is deterministic (the same circuit for the same
  input, with no seed) and several times faster than Qiskit's `optimization_level=3` on circuits made of two-qubit
  blocks.
- **`compile_for_hardware()`: compilation for a device.** It adds layout search, routing, and, given the device's
  `Target`, error-aware placement and a choice among candidate circuits by a noise estimate. It is slower than
  Qiskit level 3, and in exchange gave lower simulated infidelity than level 3 on every fake device tested.

Everything is pre-registered and self-audited: predictions are locked in git before a scored run, and results,
including the failures, are recorded in [`docs/findings/`](docs/findings/).

## Current version

**`psf_compile.py` 2026-10-06.1** and the AI front end **a11**, with `psf_smart_layout` 2026-10-01.1 and the Rust core
`CORE_VERSION` 2026-09-29.1 (Part 9, Addenda 357-362). Every release and dated notice:
[`docs/RELEASES.md`](docs/RELEASES.md).

> **New in 2026-10-06.1 and a11: readout is counted, and ecr devices are fixed.**
>
> - **Readout:** compile a circuit that will be sampled **with** its final measurements. The choice among candidates
>   and the AI front end's estimate now include each measured qubit's readout error; without measurements nothing
>   changes.
> - **ecr devices:** a9 could return ECR gates in a direction the device does not provide; a11 never does.
> - **Opt-in `candidate_score="kraus"`:** an estimate exact to first order for the simulator's noise model. It has no
>   readout term, so the recommended call stays `"hybrid"`.
> - **Pre-registered tests** (fake devices; Addenda 358-361):
>   - RECR (104 sampled circuits on each of 6 devices): compiling with the measurements cut the measured qubits'
>     readout error to 0.40 and 0.18 times on the cz devices (2026-10-05.1 could already do this); the release's
>     readout term adds a small gain on top (0.96-1.00 times); a11's classical infidelity was 0.46 and 0.16 times
>     a9's there; a9 returned 622 and 860 gates in an unsupported direction on two ecr devices, a11 none; 10 of 10
>     predictions confirmed.
>   - KRAUS (1,506 circuits on each of 9 devices): `kraus` at least as good as `hybrid` on every device, 6.3% better on
>     the one known weak case; 3-5% more compile time; 7 of 7 confirmed. It reproduces the simulator's own noise
>     model, so this test favours it by construction.

> **Correctness fix: if you use 2026-10-03.1, .2, .3 or 2026-10-04.1, update.** On cx devices those releases could
> return a circuit that is NOT equivalent to the input, when called with `final_resynthesis` or `compare_level3`. The
> same held for the AI front end (a7, a8) given a target.
>
> - **When:** the input contains two-qubit unitaries near the boundary of Qiskit issue #17057, for example explicit
>   `unitary` gates from numerical optimisation or written by a language model, or Trotter steps with very small
>   angles.
> - **Fix:** since 2026-10-05.1 and a9, every circuit that Qiskit makes as a whole is checked before it is returned.
>   A circuit that fails the check, or cannot be checked, is replaced by PSF-Zero's own, guarded circuit.
> - **Pre-registered test** (Addenda 342-343; 6 devices, 768 near-boundary and control circuits, plus 1,506 ordinary
>   circuits on each of 9 devices):
>   - 2026-10-04.1 was wrong on 101 circuits on the cx devices (infidelity up to 0.34), a8 on 108, Qiskit level 3
>     alone on 112;
>   - 2026-10-05.1 and a9 were wrong on none;
>   - on the ordinary circuits they returned exactly the circuits of 2026-10-04.1 and a8;
>   - every refusal checked was of a wrong circuit (an exploratory check after the run, on FakeAuckland);
>   - cost: median compile time 1.27-1.42 times 2026-10-04.1 on the nine devices (35-55 ms more).
> - cz devices were not affected.
> - Whether Qiskit's failure appears depends on floating-point rounding: it appeared in Linux environments
>   (WSL2 and a Linux sandbox), and not in a Windows environment tested on 2026-10-06 (Addendum 357). The fix
>   protects either way.

## Quick start

**Two-qubit synthesis** (fast, exact, deterministic):

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from psf_compile import compile as psf_compile

qc = QuantumCircuit(2)
qc.append(UnitaryGate(random_unitary(4)), [0, 1])
optimized = psf_compile(qc)          # add verify=False for the fastest path
```

**For a device** (the recommended call since 2026-10-04.1, the same on cx and cz devices):

```python
from qiskit_ibm_runtime.fake_provider import FakeTorino
from psf_compile import compile_for_hardware

backend = FakeTorino()
cm = backend.target.build_coupling_map()
basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in backend.target.operation_names]
out = compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
                           target=backend.target, placement_refine=True, final_resynthesis="select",
                           compare_level3=True, compare_floor=True, candidate_score="hybrid")
```

If the circuit will be sampled, compile it **with** its final measurements: since 2026-10-06.1 the placement and the
choice among candidates then count readout error (Addenda 358, 360).

## Install

```bash
git clone https://github.com/TN-Holdings-LLC/psf-zero.git
cd psf-zero
pip install -e .            # the Python package (psf_compile.py), numpy, scipy, networkx, qiskit==2.5.2
maturin develop --release   # the Rust core (src/lib.rs, psf_zero_core) -- must run LAST
python benchmarks/check_core_build.py   # prints RESULT: OK if the core has every function the Python code calls
```

The order matters. With both `Cargo.toml` and `pyproject.toml` in one folder, running `maturin develop` first builds
the wrong package, and running `pip install -e .` after it undoes the core. The second step needs a Rust toolchain
([rustup.rs](https://rustup.rs)) and `maturin` (`pip install maturin`).

Source: [`psf_compile.py`](psf_compile.py) is the compiler, and the one place it lives (its `VERSION:` line names the
revision) · [`lib.rs`](lib.rs) is the Rust core · [`benchmarks/psf_smart_layout.py`](benchmarks/psf_smart_layout.py)
is the layout search · [`benchmarks/psf_ai_compile.py`](benchmarks/psf_ai_compile.py) is the AI front end for
model-written circuits.

## What it does well, and what it costs

| | `compile()` | `compile_for_hardware()`, recommended call | Qiskit `optimization_level=3` |
| :--- | :--- | :--- | :--- |
| Compile time | 2.5-5x faster than Qiskit L3 on two-qubit-block circuits (15-1000 qubits) | median about 0.2 s on 4-8-qubit circuits, about 13x Qiskit L3 with the Target | reference |
| Output | same two-qubit count as Qiskit L3 at 156 qubits; deterministic | simulated infidelity 0.959-0.991 of Qiskit L3 with the Target, on 9 fake devices | reference |
| Against TKET | 150-270x faster; depth 9 against TKET's 7 | not compared | |

- **The depth gap to TKET is the price of not searching.** We know no way to close it without giving up determinism.
- **`compile()` only helps on circuits with deep same-pair two-qubit chains** (Trotter steps, layered entanglers,
  SU(4) blocks). On a generic `random_circuit()` it reports `0/0 blocks` and passes the circuit through.
- **The device numbers are noisy simulations** with Qiskit Aer and the fake devices' published calibration. The
  noise estimates PSF-Zero chooses by share the simulator's physics, so part of that advantage is built in. Hardware
  is the real test, and it has not been run for the device layer.

## Results in brief

**Compile time against Qiskit `optimization_level=3`** (`compile()`, dense two-qubit-block circuits, `verify=False`,
10 seeds per point):

| | 15q | 50q | 100q | 156q | 300q | 500q | 1000q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qiskit ÷ PSF-Zero** | **5.15x** | **3.98x** | **3.45x** | **2.86x** | **3.02x** | **2.78x** | **2.54x** |

**Against TKET** (`FullPeepholeOptimise`, same family):

| | 10q | 20q | 40q | 80q | 160q |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Speed-up** | **152x** | **237x** | **262x** | **258x** | **272x** |
| Depth (TKET / PSF-Zero) | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 |

**For a device** (pre-registered tests on 1,506 new circuits per device, 9 fake devices: four cx, five cz):

- Release 2026-10-04.1 against Qiskit level 3 with the Target: 0.959-0.991 (Addendum 337). Release 2026-10-05.1
  returns the same circuits on those circuits, on all nine devices (Addendum 343), and so does 2026-10-06.1 (it differs
  only on circuits compiled with measurements).
- Sampled circuits compiled with their measurements (6 devices): release 2026-10-06.1 at 0.66-1.01 times Qiskit level 3's
  classical infidelity, a11 at 0.63-0.98 (Addendum 360).
- With a stale calibration (errors off by 30%, T1/T2 by 20%) the previous release stayed ahead of level 3
  (Addendum 335).
- At 8-10 logical qubits it stayed ahead too (0.951-0.998).
- No failed coupler or qubit was used in any test.

**On real IBM hardware** (15 qubits, 10 jobs on `ibm_marrakesh` and `ibm_fez`, `compile()` path): fidelity was
indistinguishable from Qiskit L3 (0.0919 ± 0.0016 against 0.0925 ± 0.0016), and compile time 14-16x shorter.

**Numerical accuracy of the core:** worst infidelity 1.1e-15 over 500 Haar-random SU(4), 2.6e-14 near CNOT, no
fallbacks. Repeated recompilation drifts linearly (6.1e-11 after 1,000 laps). See
[`core-verification.md`](docs/findings/core-verification.md) and [`docs/RELEASES.md`](docs/RELEASES.md).

**Correctness checks.** Qiskit's CX-basis synthesis returns wrong circuits for two-qubit unitaries near a boundary
(reported as [Qiskit issue #17057](https://github.com/Qiskit/qiskit/issues/17057)). PSF-Zero checks every block it
takes from that decomposer (since 2026-09-26.4), and since 2026-10-05.1 every circuit Qiskit makes as a whole. The
fix above is for releases 2026-10-03.1 to 2026-10-04.1, which used such circuits unchecked.

The full tables, figures and caveats are in [`docs/findings/`](docs/findings/) and the README snapshot
[`docs/README_2026-10-05_before_restructure.md`](docs/README_2026-10-05_before_restructure.md).

## A finding that is not about PSF-Zero: the coupling-map cliff

**Qiskit's `optimization_level` 2 and 3 slow down by 40x-420x when a circuit nearly fills the coupling map.** On one
42-qubit grid, a 42-qubit circuit takes 6.8 s at `opt=3`, while a 38-qubit circuit takes 28 ms.

- **What we found:**
  - Reproduced in three environments.
  - Present in every Qiskit release from 1.4.6 through 2.5.2.
  - Also reachable through PennyLane.
  - Also present on IBM's square-lattice model (FakeNighthawk, 12.9 s at spare 0) and, with circuits that use 3-qubit
    paths, on heavy-hex devices.
- **The cost** sits in two VF2-family searches, `VF2Layout` and `VF2PostLayout`, which report "no solution" on
  instances where a zero-SWAP layout provably exists.
- **PSF-Zero's layout search** finds those layouts in milliseconds.
- **How often:** the cliff needs a circuit that is both fully embeddable and fully saturated. Generic dense circuits
  do not land on it (Addenda 135-136).

Full account: [`spare-qubit-cliff.md`](docs/findings/spare-qubit-cliff.md) (summary) and Paper 1.

## Papers

- [**Paper 1 — Ordering Sensitivity in Subgraph-Isomorphism Layout Search**](docs/papers/vf2_cliff_paper.pdf)
  ([DOI: 10.5281/zenodo.22977930](https://doi.org/10.5281/zenodo.22977930)). It characterizes the failure region of
  Qiskit's `VF2Layout`.
- [**Paper 2 — PSF-Zero: An Analytic Two-Qubit Gate Synthesizer Combined with a Verified, Ordering-Aware Layout
  Search**](docs/papers/psf_zero_paper.pdf) ([DOI: 10.5281/zenodo.22978090](https://doi.org/10.5281/zenodo.22978090)).
- [**Technical overview slides**](docs/papers/psf_zero_technical_overview.pptx) (8 slides).

**Versions.** Both papers are at version 2 (2026-09-26). Version 1 remains citable:
[10.5281/zenodo.22869976](https://doi.org/10.5281/zenodo.22869976) and
[10.5281/zenodo.22870141](https://doi.org/10.5281/zenodo.22870141). The device layer of `compile_for_hardware()`
(October 2026) came after both papers.

## Known limits and open questions

- **Hardware:** the device layer (error-aware placement, noise-estimated choice, AI front end) has been tested only in
  noisy simulation.
- **ecr devices:** the release has not been tested on them in a pre-registered test. A workplace exploration found it
  working there. It also found the AI front end defect noted above.
- **Above 16 touched qubits** the noise estimates and the equivalence checks are not made, and the release keeps its
  own circuit.
- **Tolerance of the equivalence check:** a Qiskit-made circuit is accepted up to a state infidelity of 1e-6. On
  near-boundary Trotter circuits the accepted ones were off by up to 5.8e-8, where PSF-Zero's own path is exact to
  1e-14 (Addendum 343).
- **Benchpress** integration is not done.
- **Two upstream findings:**
  - Qiskit #17057 (CX-basis synthesis) is open.
  - A qiskit-aer `save_expectation_value` defect with qubit truncation was found on 2026-10-05 and is not yet reported.
- The longer list of open questions as of 2026-10-05 is in the README snapshot
  [`docs/README_2026-10-05_before_restructure.md`](docs/README_2026-10-05_before_restructure.md#open-questions).

## How these numbers were produced

- **Timing:** warm-up outside the timer for every engine; repeated timed calls; several seeds; `spawn`;
  `seed_transpiler` pinned.
- **Every timing comes with an equivalence check.** Medians, not means; ranges, not peaks.
- **Device results** come from pre-registered tests. Predictions are locked by a git commit before the scored run,
  re-checked by an independent script, and adopted only by the owner's decision.
- **Three earlier headline claims were retracted** after re-measurement. The record, including every retraction, is
  kept verbatim in [`docs/log/`](docs/log/) and [`docs/findings/`](docs/findings/).
- The conventions are in [`record-keeping.md`](record-keeping.md).

## Where everything is

| | |
| :--- | :--- |
| [`docs/RELEASES.md`](docs/RELEASES.md) | Every release, update and correctness notice, newest first |
| [`docs/findings/spare-qubit-cliff-combined-248.md`](docs/findings/spare-qubit-cliff-combined-248.md) | Part 9 of the full record (Addenda 248 on); earlier parts are linked from it |
| [`docs/findings/compile-time.md`](docs/findings/compile-time.md) | The compile-time arc: three retractions, the `verify` split, and what survives |
| [`docs/findings/spare-qubit-cliff.md`](docs/findings/spare-qubit-cliff.md) | The Qiskit coupling-map cliff |
| [`docs/findings/core-verification.md`](docs/findings/core-verification.md) | The core's infidelity harness |
| [`docs/findings/entangling-basis.md`](docs/findings/entangling-basis.md) | Why `entangling_basis="cx"` exists |
| [`docs/findings/real-hardware.md`](docs/findings/real-hardware.md) | IBM hardware runs and job IDs |
| [`docs/log/`](docs/log/README.md) | The unedited early log, with a chronology of every claim this project got wrong |
| [`data/`](data/), [`benchmarks/`](benchmarks/), [`patches/`](patches/) | Raw data, harnesses, and the candidates each test locked |

## Working with us

This is AGPL-licensed research code, not a supported product. If you're evaluating PSF-Zero against your own circuits
and want a second opinion before investing further, send a representative circuit (or a sanitized equivalent) to
`love.os.architect@proton.me` (under NDA first, if the circuit itself is sensitive), and we'll run it and report
results directly. No commitment is implied on either side; the goal at that stage is reproducing a result, not a sales
conversation.

## Citation

```bibtex
@software{psf_zero_2026,
  author = {The Architect},
  title  = {PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis},
  year   = {2026},
  url    = {https://github.com/TN-Holdings-LLC/psf-zero},
  license = {AGPL-3.0}
}

@misc{vf2_cliff_2026,
  author = {{TN Holdings}},
  title  = {Ordering Sensitivity in Subgraph-Isomorphism Layout Search:
            A Characterization of Catastrophic Failure Regions in
            Quantum Circuit Transpilation},
  year   = {2026},
  doi    = {10.5281/zenodo.22977930},
  url    = {https://doi.org/10.5281/zenodo.22977930},
  note   = {Preprint, revised 26 September 2026; version 1: 10.5281/zenodo.22869976}
}

@misc{psf_zero_paper_2026,
  author = {{TN Holdings}},
  title  = {PSF-Zero: An Analytic Two-Qubit Gate Synthesizer Combined
            with a Verified, Ordering-Aware Layout Search for Quantum
            Circuit Transpilation},
  year   = {2026},
  doi    = {10.5281/zenodo.22978090},
  url    = {https://doi.org/10.5281/zenodo.22978090},
  note   = {Preprint, revised 26 September 2026; version 1: 10.5281/zenodo.22870141}
}
```

AGPL v3. See `LICENSE`. **Evaluating this for potential commercial use?** AGPL's copyleft terms may not fit a
closed-source integration; see [Working with us](#working-with-us) before assuming the license as published is the
final word.

[Previous repository.](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/Previous_repository.md)
