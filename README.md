
# PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-purple.svg)](https://github.com/qiskit/ecosystem)
[![Rust Core](https://img.shields.io/badge/Core-Rust_Native-E34F26.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3 Binding](https://img.shields.io/badge/FFI-PyO3-blue.svg)](https://pyo3.rs/)

**The honest one-line summary, before the details below:** across every
benchmark in this README, PSF-Zero's consistent, verified advantage is
*speed with guaranteed determinism* — not smaller circuits. On raw 2-qubit
unitary synthesis, TKET's search-based optimizer reliably finds a shallower
circuit than PSF-Zero (depth 7 vs. 9, every time we measured it), while
PSF-Zero is 150–270x faster than TKET and, depending on scale, up to ~200x
faster than Qiskit, and always returns the exact same canonical circuit for
the same input unitary (zero variance across 300 random samples). So the
trade-off being offered here is speed and reproducibility for a modest,
consistent depth cost relative to TKET's slower search — not "faster and
better on every axis." The one place PSF-Zero also won on circuit size
(fewer gates and lower depth than Qiskit, section 5) was after real
coupling-map routing was added, which looks like a side effect of feeding
the router pre-consolidated blocks rather than PSF-Zero's synthesis being
more compact in general — see the caveat there.

## What this is

PSF-Zero is a Qiskit transpiler pass that replaces heuristic 2-qubit unitary
synthesis with an **exact, closed-form Cartan (KAK) decomposition**, implemented
in a small Rust core (via PyO3) for speed.

The pass itself lives in [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py); the Rust core it calls into (`psf_zero_core`) is in [`src/lib.rs`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/src/lib.rs).

Concretely: the pass runs `Collect2qBlocks` to find runs of gates acting on the
same qubit pair, consolidates each run into a single `UnitaryGate` via Qiskit's
`ConsolidateBlocks`, and then — instead of searching for a good decomposition the
way `transpile(..., optimization_level=3)` or TKET's `FullPeepholeOptimise` do —
computes the canonical KAK form of that unitary directly and emits the
corresponding native single- and two-qubit gates. Because the decomposition is
analytic rather than search-based, it runs in constant time per block and always
returns the same circuit for the same input unitary (up to global phase and the
Weyl-chamber canonicalization it enforces).

This only helps when a circuit actually contains such blocks — i.e. deep,
same-qubit-pair 2-qubit interaction chains that a generic random circuit
(with lots of single- and multi-qubit gates interleaved) usually doesn't have
enough of to trigger. We ran into this directly while building the benchmarks
below: several of our early scripts used Qiskit's generic `random_circuit()`,
which caps effective block sizes at 3–4 gates regardless of qubit count, so the
pass never activated and appeared to be "free" — it wasn't compressing anything.
The corrected benchmark circuits below are built so that each qubit pair
receives a genuinely deep sequence of 2-qubit interactions, which is the regime
PSF-Zero is designed for (e.g. Trotterized Hamiltonian simulation, QAOA-style
layered entanglers, or any circuit synthesized from a sequence of arbitrary
SU(4) building blocks).

## Installation

```bash
git clone https://github.com/TN-Holdings-LLC/psf-zero.git
cd psf-zero
pip install -e .
```

Dependencies: `numpy`, `scipy`, `qiskit`. The Rust core is built via `maturin`/`pyo3`
as part of the package build.

## Quickstart

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from psf_compile import compile as psf_compile

qc = QuantumCircuit(2)
qc.append(UnitaryGate(random_unitary(4)), [0, 1])

optimized_qc = psf_compile(qc)
print(optimized_qc.draw())
```

`psf_compile.compile()` runs block collection, consolidation, and KAK synthesis
end-to-end and returns a standard `QuantumCircuit`. It logs how many 2-qubit
blocks it found and how many it actually synthesized (`[Debug] ... executed for
X/Y blocks`) — on a circuit with no qualifying blocks, `X/Y` will correctly be
`0/0`, and the circuit passes through unchanged.

## Benchmark methodology

All numbers below are from local runs on 2026-09-03, generated from scripts in
`benchmarks/`, using circuits deliberately constructed to contain deep,
same-pair 2-qubit interaction chains (as described above), so that PSF-Zero's
synthesis path is actually exercised. Every comparison that reports a resulting
circuit was checked for unitary equivalence against the original circuit
(`Operator(...).equiv()`, phase-corrected overlap check) before being counted as
a valid result — no timing or depth number below is reported without a passing
correctness check alongside it. We are actively re-running these on real IBM
hardware and will update this section with fresh backend results as they come
in; we are deliberately not including the two single-run "Real Device Benchmark
(15 Qubits)" results from an earlier draft of this README, since they used a
version of `psf_compile.py` with a since-fixed consolidation bug and should not
be trusted until repeated on the corrected code.

### 1. Correctness at scale (N=300)

300 randomly sampled 2-qubit unitaries, each synthesized independently by all
four pipelines and checked for unitary equivalence against the original block
(all 300/300 passed for every pipeline):

| Metric | Qiskit (L3) | TKET | PSF-Zero | Hybrid (PSF→TKET) |
| :--- | :---: | :---: | :---: | :---: |
| Circuit depth — every one of 300 samples | 15 | 7 | 9 | 7 |
| Compile time, median | 6.0ms | 153.5ms | 1.5ms | 45.1ms |

![N=300 statistical benchmark: depth is identical for all 300 samples, and compile-time distributions by compiler](./charts/n300_boxplot.png)

The depth numbers aren't averages with some spread rounded off — they are
*exactly* 15 / 7 / 9 / 7 for every single one of the 300 randomly sampled
unitaries, with zero variance. That's expected, not surprising: PSF-Zero's
synthesis of a generic SU(4) unitary always resolves to the same canonical
(Weyl-chamber) form, so depth doesn't depend on which random unitary you feed
it — this is a direct consequence of doing an exact decomposition rather than
a search, not a claim about optimality. TKET's search-based peephole optimizer
reliably finds a shallower circuit (7 vs. 9) on this circuit family; we're not
aware of a way to close that gap without giving up the determinism and the
constant-time guarantee, and we think that's a fair trade-off to state plainly
rather than paper over. On compile time, PSF-Zero was the fastest of the four
in every sample, and also the most consistent (tightest distribution) —
Qiskit's L3 pass had a long tail, including two outlier samples that took
15–17x longer than its own median.

Code: [`benchmarks/test_psf_vs_tket.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_psf_vs_tket.py)

### 2. Native synthesis vs. TKET, by scale

Same circuit family (dense, same-pair 2-qubit interaction chains, built to
avoid the TKET `Unitary2qBox` incompatibility by pre-decomposing each block into
standard gates), run at 10, 20, 40, 80, and 160 qubits:

| Qubits | TKET time | PSF-Zero time | Speedup | TKET depth | PSF-Zero depth |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 1.064s | 0.007s | 152x | 7 | 9 |
| 20 | 2.135s | 0.009s | 237x | 7 | 9 |
| 40 | 4.193s | 0.016s | 262x | 7 | 9 |
| 80 | 8.244s | 0.032s | 258x | 7 | 9 |
| 160 | 16.840s | 0.062s | 272x | 7 | 9 |

![Native synthesis vs. TKET by scale: compile time and output depth](./charts/native_scale_comparison.png)

The depth gap (TKET 7 vs. PSF-Zero 9) is flat across every scale we tested —
the same trade-off as the N=300 result above, on a different circuit family.
Note that the speedup factor here behaves differently from the Qiskit
comparison in section 4 below: against TKET's `FullPeepholeOptimise`, the
speedup holds roughly steady (150x–270x) rather than shrinking as qubit count
grows, because TKET's own compile time is scaling worse than linearly on this
circuit family over the range we tested. We're reporting both comparisons
because they don't tell the same story, and we'd rather show that than pick
whichever one looks better.

Code: [`benchmarks/test_scale_explosion_war2_v2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_scale_explosion_war2_v2.py)

### 3. Hamiltonian simulation (Trotter blocks)

Using the standard XX/YY/ZZ/exchange/full two-qubit interaction blocks used in
Trotterized time evolution (VQE, condensed-matter simulation):

![Trotter interaction blocks: output circuit depth by compiler, original vs. Qiskit L3 vs. TKET vs. PSF-Zero](./charts/hamiltonian_depth.png)

Across all five interaction types, Qiskit Level 3 produced circuits of depth
15, PSF-Zero produced circuits of depth 9, and TKET's peephole optimizer
produced circuits of depth 7 — every interaction type gave the identical
15/7/9 split, the same three-way signature as the two benchmarks above, now
confirmed on a third, independently-motivated circuit family. PSF-Zero's
compile time was consistently the fastest of the three in every interaction
type tested (sub-3ms vs. Qiskit's ~5–40ms and TKET's ~50–57ms).

Code: [`benchmarks/test_official_hamiltonians_war.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_official_hamiltonians_war.py)

### 4. Compile-time scaling

This is the benchmark we'd point a skeptical reader to first, because it's the
one that actually shows the trade-off honestly rather than a flat "always N
times faster" number. Using disjoint dense-pair circuits with no coupling-map
constraint, and measuring PSF-Zero's raw per-block synthesis cost against
Qiskit's `optimization_level=3` transpile:

- PSF-Zero's compile time scales roughly **linearly** with the number of
  2-qubit blocks in the circuit (~0.0006s/block).
- Qiskit's compile time for this circuit family stays roughly **flat**
  (~1.3s), independent of block count, over the range we tested.
- The result is a speedup factor that **shrinks predictably as circuit size
  grows**: from 203x at 15 qubits / 7 blocks down to 4.4x at 1000 qubits /
  500 blocks.

![Compile time scaling: Qiskit flat vs. PSF-Zero linear, and the resulting speedup curve](./charts/compile_time_scaling.png)

We think this declining curve is more credible — and more useful to anyone
deciding whether to adopt this — than a single headline multiplier, and it's
consistent with the mechanism: PSF-Zero's advantage comes from skipping search
entirely per block, so the constant-time-per-block cost eventually catches up
to Qiskit's roughly-fixed overhead as the number of blocks grows.

Code: [`benchmarks/phase1_v2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase1_v2.py) (15–156 qubits) and [`benchmarks/phase2_v2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase2_v2.py) (156–1000 qubits)

### 5. Real-device topology (coupling-map-constrained)

We ran a coupling-map-constrained comparison (50–500 qubits, grid topology)
across all three `routing_optimization_level` settings (0, 1, 2), twice each
in independent sweeps run in opposite order (0→1→2 and 2→1→0) to make sure the
setting — not run order — was what determined the outcome. In every run, the
number of blocks PSF-Zero processed matched the number of qubit pairs in the
circuit exactly: it correctly found and synthesized every qualifying block
once real hardware connectivity constraints were introduced, not just in the
unconstrained case above.

| Qubits | Qiskit gates / depth | PSF-Zero, level=0 | PSF-Zero, level=1 | PSF-Zero, level=2 |
| :---: | :---: | :---: | :---: | :---: |
| 50 | 500 / 20 | 75 / 9 | 75 / 5 | 75 / 5 |
| 100 | 1000 / 20 | 150 / 9 | 150 / 5 | 150 / 5 |
| 156 | 1562 / ~39 | 306 / ~24 | 236 / ~9 | 237 / 10 |
| 300 | 3000 / 20 | 450 / 9 | 450 / 5 | 450 / 5 |
| 500 | 5003 / 41 | 992 / ~30 | 753 / 10 | 753 / 10 |

![Real-device topology: Qiskit vs. PSF-Zero at routing_optimization_level 0, 1, and 2](./charts/topology_two_configs.png)

(Each cell above is a mean over 6 seeds — 2 sweeps × 3 seeds — except Qiskit,
pooled across all 18 runs per scale.) `routing_optimization_level=0` gives
Qiskit's router noticeably less work to do, and PSF-Zero's post-synthesis
routing pass inherits that: consistently more 2Q gates and higher depth than
levels 1 or 2. Levels 1 and 2 were statistically indistinguishable from each
other at every scale we tested — for this circuit family, the extra search
budget of level 2 bought nothing over level 1. (An earlier draft of this
benchmark mislabeled which run was which level, based on a single sweep; the
numbers above come from two independent, oppositely-ordered sweeps and we're
confident in this mapping.)

Code: [`benchmarks/test1_v3.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test1_v3.py)

### 6. Sanity check against Benchpress

We don't have our own results in [Benchpress](https://github.com/Qiskit/benchpress) — IBM's open-source SDK benchmarking
suite (Nation et al., *Benchmarking the performance of quantum computing
software for quantum circuit creation, manipulation and compilation*,
[Nat. Comput. Sci. 5, 427–435 (2025)](https://doi.org/10.1038/s43588-025-00792-y)) — but two of our numbers above line up
with what that paper independently reports for TKET against Qiskit, which is
worth stating plainly rather than leaving unmentioned:

- Benchpress reports TKET's transpilation is "over an order of magnitude
  slower than Qiskit" across its 1,066-test suite. Our own N=300 result
  (TKET median 153.5ms vs. Qiskit 6.0ms, ~26x) and native-scale comparison
  (TKET 150–270x slower than PSF-Zero, itself faster than Qiskit) point the
  same direction — our absolute TKET-vs-Qiskit timing gap isn't an artifact
  of our narrow test construction.
- Benchpress specifically calls out Hamiltonian-simulation circuits as the
  case where TKET's synthesis step yields "substantial 2Q depth reduction
  relative to Qiskit," and that this synthesis advantage matters most on
  well-connected topologies and fades as routing starts to dominate on
  sparser ones. That is exactly the pattern in our own Hamiltonian
  benchmark (section 3) and in our coupling-map-constrained result (section
  5), where PSF-Zero's and TKET's edge over Qiskit shrinks once routing
  becomes the bottleneck rather than synthesis.

This is corroboration of the general trend, not a substitute for the real
test: Benchpress's suite is far broader than ours (1,066 tests, up to 930
qubits and O(10⁶) 2Q gates, real device coupling maps, and circuit families
we haven't touched — quantum volume, QAOA, HamLib, Feynman, QASMBench —
versus our own narrowly-constructed dense-pair circuits). Running PSF-Zero
through Benchpress's own harness is the obvious, credible next step, and
we haven't done it yet — see Roadmap.

## What we haven't verified yet

In the interest of not overstating anything:

- **Real IBM hardware fidelity results.** The two single-run 15-qubit results
  that appeared in earlier drafts of this README were produced before we found
  and fixed a bug in `ConsolidateBlocks` configuration that caused the pass to
  silently no-op on some circuits. We're re-running on real backends with the
  corrected code and will publish results (with job IDs) once we have more than
  a single run to report.
- **GPU / massively parallel execution.** Because PSF-Zero decomposes each
  2-qubit block independently, the per-block synthesis is embarrassingly
  parallel in principle. We have not implemented or benchmarked a parallel
  execution path — this is a plausible direction, not a measured result.
- **The 1000-qubit "615x–867x" and 156–1000-qubit "Empirical Benchmark
  Dataset" figures from an earlier draft of this README have been removed.**
  Both were produced using circuit generators (`random_circuit()` /
  `generate_scalable_dense_circuit()`) that structurally never produced blocks
  large enough for PSF-Zero's `block_gate_floor` to activate — PSF-Zero was
  returning the input circuit essentially unchanged, and the reported speedup
  reflected doing no work rather than doing the work faster. We caught this by
  directly measuring block sizes in the generators and by observing that
  PSF-Zero's own reported output depth was, in the worst case, no better than
  the unoptimized input. We'd rather retract these than leave them up.

## Design notes

- **Deterministic by construction.** The decomposition is exact and
  closed-form, so the same input unitary always produces the same canonical
  circuit (up to global phase). There is no random seed to control for.
- **Weyl-chamber canonicalization.** Every synthesized 2-qubit unitary is
  projected into the canonical region ($0 \le c_3 \le c_2 \le c_1 \le \pi/2$),
  so results are directly comparable across runs.
- **No silent fallbacks.** Degeneracies and edge cases in the decomposition are
  surfaced as explicit Rust `Result` errors rather than approximated away.
- **Scope.** PSF-Zero targets the 2-qubit unitary synthesis step specifically.
  It is not a full replacement for a transpiler's routing, layout, or
  multi-qubit gate decomposition — it composes with those (as shown in the
  coupling-map benchmark above), it doesn't replace them.

## Roadmap

- Fresh real-hardware validation with the corrected `ConsolidateBlocks`
  configuration (in progress).
- Running PSF-Zero through [Benchpress](https://github.com/Qiskit/benchpress) (IBM's open-source SDK benchmark suite)
  for an apples-to-apples comparison against Qiskit, TKET, and the other SDKs
  it already covers, on its own broad, realistic circuit collection rather
  than our own narrower constructions.
- Exploring parallel (multi-core / GPU) execution of independent block
  synthesis — currently unimplemented.
- PennyLane integration (`qml.transforms`) — planned, not yet built.

## Citation

```bibtex
@software{psf_zero_2026,
  author = {The Architect},
  title = {PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis},
  year = {2026},
  url = {https://github.com/TN-Holdings-LLC/psf-zero},
  license = {AGPL-3.0}
}
```

## License

AGPL v3. See `LICENSE`.
