> **Archived record — part 1 of 6.**
> This file is one slice of the original PSF-Zero README, kept **verbatim**. It is a
> lab notebook: it contains claims that were later retracted, hypotheses that were
> later refuted, and corrections written directly underneath the text they correct.
> Nothing has been deleted. For what the project currently claims, read the
> [README](../../README.md); for a settled account of one topic, read
> [`docs/findings/`](../findings/).
>
> Header, the honest one-line summary and its revisions, what the pass actually is, installation, quickstart, and the benchmark-methodology preamble.
>
>
> **One edit was made to this file when it was archived:** file-relative image paths
> were rewritten from `./docs/...` to `../../docs/...` for the new directory depth.
> Root-relative paths (`/docs/...`) were left untouched — they resolve correctly at
> any depth. No wording, number, claim or correction was altered. The 13 substitutions
> are listed in [`README.md`](README.md#the-one-edit).
> Source: README.md lines 1-160, as of 2026-09-11.

---

# PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-purple.svg)](https://github.com/qiskit/ecosystem)
[![Rust Core](https://img.shields.io/badge/Core-Rust_Native-E34F26.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3 Binding](https://img.shields.io/badge/FFI-PyO3-blue.svg)](https://pyo3.rs/)

![Performance Benchmark](../../docs/11.png)

**The honest one-line summary, before the details below:** across every
benchmark in this README, PSF-Zero has a real, verified speed advantage over
both TKET and Qiskit, plus determinism neither of them offers — but the
Qiskit advantage only shows up with a non-default setting, and earlier
drafts of this README overstated it in a way we're not going to repeat. On
raw 2-qubit unitary synthesis, TKET's search-based optimizer reliably finds
a shallower circuit than PSF-Zero (depth 7 vs. 9, every time we measured
it), while PSF-Zero is 150–270x faster than TKET (section 2) and always
returns the exact same canonical circuit for the same input unitary (zero
variance across 300 random samples).

Against Qiskit, the story took three separate corrections to get right (see
section 4 for the full account). First, an earlier draft claimed "up to
~200x faster" — that number was almost entirely measurement artifacts (a
no-op transpile bug, a `ConsolidateBlocks` bug, an unwarmed per-process
cold-start cost) and is retracted. Second, once those were fixed, PSF-Zero's
*default* behavior (`compile()`/`compile_for_hardware()` with their current
default of `verify=True`) measured *slower* than a properly warmed-up
Qiskit beyond the smallest circuits tested — traced to an unconditional,
every-call self-verification step that turned out to cost far more than the
actual decomposition. Third, with that check made optional
(`verify=False`, keeping the separate degenerate-point fallback that's
actually load-bearing) and re-measured with the same 10-seed rigor across
15–1000 qubits: PSF-Zero is genuinely faster than Qiskit at every scale
tested, by roughly 2.4x–5.2x, largest at the smallest circuits and settling
to a stable ~2.4x–3x band at 150+ blocks — correctness confirmed unaffected
throughout.

**Fourth, and this is the current state as of 2026-09-09: the "opt-in"
caveat that the rest of this README is written around is now largely
obsolete.** `verify` was split into a cheap Rust-core check (`True`, still
the default) and the old, expensive `Operator`-based reconstruction
(`"strict"`). Re-measured on the project's own Windows machine over a
50,000-iteration loop, the *default* path now runs at 2.303ms/call against
Qiskit L3's 11.038ms — **4.79x faster with the safety net still on**, and
9.12x with `verify=False`. The check now costs about 1.9x rather than the
4–5x it used to, so a caller who changes nothing already gets most of the
advantage. The old framing is left standing below with its correction
directly underneath, in the same way every other retraction in this
document is handled — see section 4's 2026-09-09 update.

So the real, current state is: a genuine, mechanism-backed, real-hardware-
confirmed speed advantage over Qiskit exists, and as of 2026-09-09 it is
no longer gated behind a non-default setting — though the *size* of it
turns out to be more machine-dependent than earlier revisions of this
README implied (section 4's second 2026-09-09 update). The trade-off being offered is
determinism plus a real (if currently opt-in) speed edge, for a fixed depth
cost relative to TKET's slower search — not "faster and better on every
axis" without qualification, but a real advantage once you know which knob
to turn. The one place PSF-Zero also won on circuit size (fewer gates and
lower depth than Qiskit, section 5) was after real coupling-map routing was
added, which looks like a side effect of feeding the router pre-consolidated
blocks rather than PSF-Zero's synthesis being more compact in general — see
the caveat there.

## What this is

PSF-Zero is a Qiskit transpiler pass that replaces heuristic 2-qubit unitary
synthesis with an **exact, closed-form Cartan (KAK) decomposition**, implemented
in a small Rust core (via PyO3) for speed.

The pass itself lives in [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py); the Rust core it calls into (`psf_zero_core`) is in [`/lib.rs`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/lib.rs).

Concretely: the pass runs `Collect2qBlocks` to find runs of gates acting on
the same qubit pair, consolidates each run into a single `UnitaryGate` via Qiskit's
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
correctness check alongside it. The two single-run "Real Device Benchmark (15
Qubits)" results that appeared in an earlier draft of this README used a
version of `psf_compile.py` with a since-fixed `ConsolidateBlocks` bug and have
been removed; section 7 below replaces them with a 10-run result on real IBM
hardware using the corrected code. Section 8 adds a separate noisy-simulator
comparison across all four engines (Qiskit, TKET, PSF-Zero, Hybrid) that isn't
covered by sections 1–6.

> **Note on paths and links (2026-09-11).** Every filename written as a link
> in this README was checked on 2026-09-11 by requesting it directly and
> confirming a 200 rather than a 404. The handful still written as plain code
> spans are the ones that came back 404: they exist in this project's records
> but are not in the repository yet, and are deliberately left unlinked rather
> than pointed at a URL that does not resolve. The audit, and the files that
> would need to be uploaded or renamed to close the remaining gaps, are
> recorded in `publication-policy.md`.
>
> An earlier revision of this note claimed the repository had no `data/`
> directory at all and unlinked thirteen data references on that basis. That
> was wrong — it came from a file listing that had been silently mangled in
> transit — and the links have been restored. The method is now per-file
> verification, not directory listings.
