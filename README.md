# PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-purple.svg)](https://github.com/qiskit/ecosystem)
[![Rust Core](https://img.shields.io/badge/Core-Rust_Native-E34F26.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3 Binding](https://img.shields.io/badge/FFI-PyO3-blue.svg)](https://pyo3.rs/)

A Qiskit transpiler pass that replaces heuristic 2-qubit unitary synthesis with an
**exact, closed-form Cartan (KAK) decomposition**, implemented in a small Rust core
via PyO3. Because the decomposition is analytic rather than search-based, it runs in
constant time per block and returns **the same circuit every time** for the same
input unitary.

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from psf_compile import compile as psf_compile

qc = QuantumCircuit(2)
qc.append(UnitaryGate(random_unitary(4)), [0, 1])
optimized = psf_compile(qc)          # add verify=False for the fastest path
```

Install: `git clone … && cd psf-zero && pip install -e .`
(needs `numpy`, `scipy`, `qiskit`; the Rust core builds via `maturin`/`pyo3`.)

Source: [`psf_compile.py`](psf_compile.py) — the pass itself ·
[`lib.rs`](lib.rs) — the Rust core (`psf_zero_core`) it calls into.

---

## The trade-off, stated up front

- **Faster than Qiskit** `optimization_level=3` on circuits it is designed for —
  roughly **2.5x–5x**, largest at small circuits (see the table below and its caveats).
- **Much faster than TKET** — **150x–270x**, flat across 10–160 qubits.
- **Deterministic** — 300 random SU(4) unitaries produced 300 identical circuits.
  There is no seed to control for.
- **But TKET finds a shallower circuit**: depth 7 against PSF-Zero's 9, on every
  sample we measured. That gap is the price of not searching, and we are not aware
  of a way to close it without giving up the determinism.

PSF-Zero only helps on circuits that actually contain deep, same-qubit-pair 2-qubit
chains — Trotterized Hamiltonian simulation, QAOA-style layered entanglers, circuits
built from arbitrary SU(4) blocks. On a generic `random_circuit()` it correctly
reports `0/0 blocks` and passes the circuit through unchanged.

## Results

**Compile time vs. Qiskit `optimization_level=3`**, dense-pair-block circuits,
`verify=False`, 10 seeds per point, warm-up outside the timer, `spawn`:

| | 15q | 50q | 100q | 156q | 300q | 500q | 1000q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qiskit ÷ PSF-Zero** | **5.15x** | **3.98x** | **3.45x** | **2.86x** | **3.02x** | **2.78x** | **2.54x** |

<sub>`phase1.py` (15–156q) and `phase2.py` (156–1000q), Windows,
`AMD64 Family 25 Model 80`, Python 3.10.11, Qiskit 2.5.2. Raw data in
[`data/archive/`](data/archive/).</sub>

**With the default safety check on** (`verify=True`, a cheap Rust-core check since
2026-09-09), measured over a 50,000-iteration compile loop at 15 qubits:
**4.79x** on one machine and **5.54x** (median) on another; `verify=False` gives
**9.12x** / **6.84x** on the same two runs. The check costs **1.2x–1.9x depending
on the machine** — not a single number.

**Output quality is matched, not traded away.** At 156 qubits, PSF-Zero and Qiskit
`optimization_level=3` emit the *same* 234 two-qubit gates; depth is **9**
(`entangling_basis="canonical"`), **13** (`"cx"`, hardware-comparable) and **16**
(Qiskit). Equivalence checked at every point (< 4.5e-15).

**vs. TKET** (`FullPeepholeOptimise`), same circuit family:

| | 10q | 20q | 40q | 80q | 160q |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TKET | 1.06s | 2.14s | 4.19s | 8.24s | 16.84s |
| PSF-Zero | 0.007s | 0.009s | 0.016s | 0.032s | 0.062s |
| **Speed-up** | **152x** | **237x** | **262x** | **258x** | **272x** |
| Depth (TKET / PSF-Zero) | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 |

**On real IBM hardware** (15 qubits, 10 job submissions across `ibm_marrakesh` and
`ibm_fez`): fidelity is **indistinguishable** from Qiskit L3 (0.0919 ± 0.0016 vs
0.0925 ± 0.0016, paired t = −0.78, n.s.); compile time was 14–16x faster in every
one of the 10 runs.

### Numerical accuracy of the core

The decomposition is closed-form, so the thing that can go wrong is not search
quality but numerical stability — around the CNOT/SWAP degeneracies, and in the
agreement between the Rust core's 4×4 reconstruction and Qiskit's `Operator(qc)`.
[`benchmarks/verify_core_infidelity.py`](benchmarks/verify_core_infidelity.py) locks
both, measured 2026-09-12:

| Suite | Samples | Worst infidelity (core) | Worst infidelity (strict circuit) | Fallbacks |
| :--- | :---: | :---: | :---: | :---: |
| Haar-random SU(4) | 500 | 1.11e-15 | 6.66e-16 | **0 / 500** |
| Near-CNOT (ε = 1e-7) | 200 | 2.63e-14 | (covered by the strict loop) | **0 / 200** |

Across the Haar space the worst case sits near machine epsilon with no fallback
exceptions. Near the codimension-2 CNOT singularity — where a naive single-route
diagonalisation is least stable — scored candidate selection plus a Givens sweep
holds infidelity well below the 1e-12 tolerance, with zero rejections. (An earlier
run reported 1.68e-13 here; the perturbation that generates the near-CNOT samples
had a global-phase bug that put the test points at distance ~0.765 from CNOT
regardless of ε, not ~ε as intended — fixed and re-run, see
[`docs/findings/core-verification.md`](docs/findings/core-verification.md).) The
`strict` tier is what rules out endian mismatches and ZYZ phase/sign drift between
the two sides.

Raw data: [`data/core_verification_2026-09-12.csv`](data/core_verification_2026-09-12.csv).
Reproduce with `maturin develop --release && python benchmarks/verify_core_infidelity.py`.
Full account: [`docs/findings/core-verification.md`](docs/findings/core-verification.md).

## What this is not

- **Not a full transpiler.** PSF-Zero targets the 2-qubit synthesis step. Routing,
  layout and multi-qubit decomposition stay with Qiskit; PSF-Zero composes with them.
- **Not faster with `verify="strict"`.** That option restores the old
  `Operator()`-reconstruction check and costs **5.1x–6.6x** the current default,
  which makes PSF-Zero *slower* than Qiskit above 15 qubits. It is there for people
  who want it, not as a recommended setting.
- **Not yet tested under real routing pressure.** The coupling-map benchmarks place
  blocks on adjacent logical pairs, which land on adjacent physical qubits under a
  row-major grid — so the router had almost nothing to route. Genuine SWAP-insertion
  cost is unmeasured.
- **Not run through Benchpress.** IBM's suite is far broader than ours; integration
  is in progress, not done.

## A finding that is not about PSF-Zero

While benchmarking against coupling maps we found, and then confirmed with a
controlled experiment, that **Qiskit's `optimization_level` 2 and 3 slow down by
40x–275x when the circuit nearly fills the coupling map.** Holding the map fixed and
varying only how many qubits the circuit occupies: on one unchanged 42-qubit grid, a
42-qubit circuit takes 6.8 s at `opt=3` while a 38-qubit circuit takes 28 ms. A
*smaller* circuit on a saturated grid runs ~200x slower than a *larger* one with
spare qubits.

Reproduced in three independent environments (Linux sandbox, and two Windows
machines with different CPUs), twice back-to-back on one of them, agreeing to within
5%, and present in every Qiskit release from **1.4.6 through 2.5.2** unchanged. Two
further controls narrow it: the effect needs the dense adjacent-pair circuit structure
as well as the saturated map (a gate-count-matched `random_circuit()` workload shows
no cliff at all), and Qiskit 2.1 made the *unsaturated* case ~93x faster while the
saturated case has not improved since 1.4.6.

**Two independent costs, both measured inside Qiskit.** `VF2Layout` and
`VF2PostLayout` are 99.9% of the time. Qiskit implements VF2 itself
(`crates/transpiler/src/passes/vf2_layout.rs`), and both passes use the VF2++ node
ordering unconditionally. Permuting the coupling graph's node order via `shuffle_seed`
turns `NO_SOLUTION_FOUND` into `SOLUTION_FOUND` in **4 of 30 seeds** on an unchanged
saturated grid — the layout it reports as absent exists, and Qiskit finds it 13% of
the time. `VF2PostLayout` succeeds on the *same four seeds*, so the two passes are one
failure paid for twice. But finding it does not help: with the default trial budget a
seed that finds the layout in **3.5 ms still runs for 343 ms**, because `minimize_vf2`
keeps searching for a better score afterwards. Fixing the ordering alone leaves the
trial loop; fixing the trial loop alone leaves the 26 seeds that never find anything.
Through the preset it is deterministic: 30 `transpile()` calls on the same input
report `NO_SOLUTION_FOUND` thirty times, so retrying does not help and
`seed_transpiler` is not a lever. Padding the coupling map with a few spare qubits
removes the effect entirely.
Reported upstream and rejected, because the report said Qiskit calls
`rustworkx.vf2_mapping`, which it does not. Experiments:
[`phase3_v5_spare_qubits.py`](benchmarks/phase3_v5_spare_qubits.py),
[`phase3_v6_workload_control.py`](benchmarks/phase3_v6_workload_control.py),
[`verify_vf2_seed.py`](benchmarks/verify_vf2_seed.py),
[`verify_vf2_max_trials.py`](benchmarks/verify_vf2_max_trials.py),
[`verify_preset_stop_reason.py`](benchmarks/verify_preset_stop_reason.py).
Full account, source reading, and raw data:
[`docs/findings/spare-qubit-cliff.md`](docs/findings/spare-qubit-cliff.md).

## How these numbers were produced

Warm-up call outside the timer for **both** engines; repeated timed calls per point;
multiple independent seeds; `spawn` start method; `seed_transpiler` pinned wherever
Qiskit's randomised layout/routing search is involved; unitary equivalence checked
alongside every timing; every harness records `platform.processor()`, Python and
Qiskit versions **into its output CSV**. Medians, not means, on shared hardware.
Ranges, not peaks.

**Three earlier headline claims in this README were retracted after re-measurement**
— a "200x" that turned out to be a no-op `transpile()` call, a "615x–867x" produced
by circuits that never triggered the pass, and a decay-with-iteration-count effect
that turned out to be background load on one machine. Two hypotheses this project
proposed were later **refuted by their own pre-registered criteria**. A third — a
mechanism proposed for the finding above — was rejected upstream for naming a code
path Qiskit does not use; the drafts and the outcome are in the log. The complete
record, including every retraction and the raw data behind it, is kept verbatim in
[`docs/log/`](docs/log/) rather than quietly edited away.

## Where everything is

**Findings** — one settled topic each, self-contained, ~5 minutes:

| | |
| :--- | :--- |
| [`docs/findings/compile-time.md`](docs/findings/compile-time.md) | The full compile-time arc: three retractions, the `verify` split, and what survives |
| [`docs/findings/spare-qubit-cliff.md`](docs/findings/spare-qubit-cliff.md) | The Qiskit coupling-map result above — controlled experiment, three environments |
| [`docs/findings/core-verification.md`](docs/findings/core-verification.md) | The three-tier infidelity harness: Haar space, CNOT singularities, Rust↔Python agreement |
| [`docs/findings/entangling-basis.md`](docs/findings/entangling-basis.md) | Why `RXX/RYY/RZZ` costs 2x the native gates of `CX`, and the `entangling_basis="cx"` fix |
| [`docs/findings/real-hardware.md`](docs/findings/real-hardware.md) | IBM hardware runs, job IDs, and the noisy-simulator comparison |

**The unedited record** — [`docs/log/`](docs/log/README.md), 2,973 lines kept verbatim,
including a [chronology of every claim this project got wrong](docs/log/README.md#chronology-of-things-this-project-got-wrong):

[`01` intro & methodology](docs/log/01-intro-and-methodology.md) ·
[`02` synthesis vs. TKET](docs/log/02-synthesis-vs-tket.md) ·
[`03` compile-time scaling](docs/log/03-compile-time-scaling.md) ·
[`04` real-device topology](docs/log/04-real-device-topology.md) ·
[`05` fidelity](docs/log/05-fidelity.md) ·
[`06` open questions & roadmap](docs/log/06-open-questions-and-roadmap.md)

**Data** — [`data/`](data/) holds every CSV behind a published number;
[`data/archive/`](data/archive/) holds the superseded and retracted runs, so the
retractions can be re-checked, with a file-by-file map in
[`provenance-map.md`](data/archive/provenance-map.md).

**Benchmarks** — the harnesses, in the order the story needs them:
[`phase1_v2.py`](benchmarks/phase1_v2.py) /
[`phase2_v2.py`](benchmarks/phase2_v2.py) (scaling sweeps) ·
[`test1_v3.py`](benchmarks/test1_v3.py) (methodology-corrected harness) and its
[`verify="strict"` wrapper](benchmarks/test1_v3_verify_strict.py) ·
[`test_cumulative_compile_time.py`](benchmarks/test_cumulative_compile_time.py)
(50,000-iteration loop) ·
[`verify_core_infidelity.py`](benchmarks/verify_core_infidelity.py) (core accuracy) ·
[`phase3_v4_dense_pair_blocks.py`](benchmarks/phase3_v4_dense_pair_blocks.py) and
[`phase3_v5_spare_qubits.py`](benchmarks/phase3_v5_spare_qubits.py) /
[`phase3_v6_workload_control.py`](benchmarks/phase3_v6_workload_control.py) (coupling maps) ·
[`test_psf_vs_tket.py`](benchmarks/test_psf_vs_tket.py) /
[`test_scale_explosion_war2.py`](benchmarks/test_scale_explosion_war2.py) (TKET) ·
[`test_real_hardware_fidelity.py`](benchmarks/test_real_hardware_fidelity.py) and
[`real_device_15q_fidelity_v2`](benchmarks/real_device_15q_fidelity_v2) (fidelity).

**Rules** — [`record-keeping.md`](record-keeping.md): the conventions this project
follows when publishing a measurement, most of them adopted after being burned by
their absence.

## Open questions

- What burns the budget inside the two VF2 passes. The ordering and the trial loop
  are both measured costs, but nothing inside the passes is instrumented, and it is
  one Qiskit version (2.5.2) on one topology family.
- Why the preset never reaches a winning ordering — whether it disables shuffling
  outright or shuffles something that does not reach the VF2 node order. The outcome
  is measured; which of the two explains it is not.
- Whether the compile-time advantage reduces real-hardware calibration-drift
  exposure in a variational loop. Plausible, untested, and not planned without a
  reason to spend QPU time.
- A routing benchmark on non-adjacent logical pairs, so SWAP insertion is actually
  exercised.
- `compile_for_hardware()` does not yet expose `seed_transpiler`, so its internal
  routing call stays unpinned.
- Benchpress integration ([issue #114](https://github.com/Qiskit/benchpress/issues/114)),
  PennyLane transforms, and parallel per-block synthesis — all unbuilt.

## Citation

```bibtex
@software{psf_zero_2026,
  author = {The Architect},
  title  = {PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis},
  year   = {2026},
  url    = {https://github.com/TN-Holdings-LLC/psf-zero},
  license = {AGPL-3.0}
}
```

AGPL v3. See `LICENSE`.

[Previous repository.](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/Previous_repository.md)
