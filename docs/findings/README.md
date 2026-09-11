# `docs/log/` — the unedited record

This directory holds the PSF-Zero README as it stood on **2026-09-11**, before it was
split into a short front page plus [`docs/findings/`](../findings/). It is kept
**verbatim and complete**: 2,973 lines across six files, concatenating back to the
original with exactly one class of edit, listed below.

**Why keep it.** Almost every number this project publishes today replaced a number it
published earlier and then found to be wrong. The corrections were written *underneath*
the claims they correct rather than replacing them, so the reasoning that produced a
wrong answer stays visible next to the reasoning that caught it. Deleting that would
make the surviving numbers look more solid than they are — and would remove the only
real evidence that this project audits itself. If you want to know whether to trust the
front page, this is the directory to read.

**Nothing here is current.** Claims in these files may be superseded, retracted, or
refuted. For what the project claims now, read the [README](../../README.md); for a
settled account of one topic, read [`docs/findings/`](../findings/).

## The files

| File | Lines | Contents |
| :--- | ---: | :--- |
| [`01-intro-and-methodology.md`](01-intro-and-methodology.md) | 160 | Header, the honest one-line summary and its successive revisions, what the pass is, install, quickstart, methodology preamble |
| [`02-synthesis-vs-tket.md`](02-synthesis-vs-tket.md) | 112 | Sections 1–3: N=300 correctness, TKET comparison across 10–160 qubits (three independent runs), Trotter blocks |
| [`03-compile-time-scaling.md`](03-compile-time-scaling.md) | 1,197 | Section 4. The longest and most-corrected part of the record |
| [`04-real-device-topology.md`](04-real-device-topology.md) | 483 | Sections 5–6: coupling-map results, the `routing_optimization_level` 2 → 1 decision, the spare-qubit cliff, Benchpress sanity check |
| [`05-fidelity.md`](05-fidelity.md) | 655 | Sections 7–8: real IBM hardware, noisy-simulator comparison, both fidelity root-cause threads |
| [`06-open-questions-and-roadmap.md`](06-open-questions-and-roadmap.md) | 366 | Open questions, design notes, the full Roadmap with every DONE / RESOLVED / REFUTED / CLOSED annotation |

Split by **section, not by date** — the dated updates are nested inside sections and
refer to each other as "the update above" / "the correction below", so cutting by date
would have broken those references. Each file carries a short archival header; that
header is the only text added.

## Chronology of things this project got wrong

Every row below is documented in place, with the measurement that caught it.

| Claim | What it actually was | Where |
| :--- | :--- | :--- |
| "up to ~200x faster than Qiskit" | `transpile()` had no `basis_gates`, so it passed every `UnitaryGate` through untouched. Its time was flat at 1.30–1.34 s from 15 to 156 qubits | `03` |
| "615x–867x at 1000 qubits" | The circuit generators never produced blocks above `block_gate_floor`; PSF-Zero returned the input unchanged | `06` |
| Two single-run 15-qubit real-device results | Produced with a since-fixed `ConsolidateBlocks` bug; removed and replaced with a 10-run result | `01`, `05` |
| "the ratio compresses at 10,000+ iterations" | Episodic background load on one machine. The raw per-iteration arrays show bursts with full recovery, and the final 5,000 iterations were the fastest of the run | `03` |
| "a 0.2 ms RSS sampler thread contends for the GIL during measurement" | The sampler does not run during the timed calls at all. We proposed a mechanism for a script before reading its source | `03` |
| "the declining run used a pre-2026-09-09 `psf_compile.py`" | Stated with a numeric prediction, tested, and **refuted** — `verify="strict"` overshot by ~3.1x at every scale | `03` |
| "the router found a SWAP-free solution in every single case" | False; 4 sandbox and 1 Intel measurement inserted SWAPs. The conclusion survives in weaker form | `04` |
| "`verify=\"strict\"` overshoots the old default by 2.5x" | Machine-mismatched arithmetic — one figure came from a different machine. Actual overshoot ~1.3x, and equivalence is simply unconfirmed | `03` |
| "PSF-Zero was 2.5–2.7x slower at Qiskit's slow iterations" | Recoverable only by dividing a mean by a median. Matched statistics give 1.78x or 3.20x depending on which | `03` |
| "the threshold is zero spare qubits" | The boundary is map-dependent: 2 spare is slow on a 100-qubit grid and fast on a 72-qubit one | `04` |
| Machine attribution by account name | One account name spans at least two physical machines. Machines are now identified by CPU signature only — and the original declining run's machine is **permanently unrecoverable** | `03` |

Two more happened outside these files and are summarised in
[`record-keeping.md`](../../record-keeping.md): a link audit that was itself wrong (a
mangled directory listing produced a false "the `data/` directory does not exist"),
and the record-keeping failures — fixed output filenames, environment metadata printed
but not stored — that made the machine question unanswerable.

*A note on one reference.* Files in this archive mention a `publication-policy.md`.
That is an internal working document — it holds the mapping between CPU signatures and
the physical machines they belong to, which is exactly the kind of thing this project's
own rules say must not be published. It is deliberately **not** in this repository. The
conventions from it that are useful to a reader are in
[`record-keeping.md`](../../record-keeping.md) instead.

## The one edit

Moving these files two directories deep broke every **file-relative** image path.
Those were rewritten mechanically — `./docs/x.png` → `../../docs/x.png` — in 13 places
across five files. **No wording, number, claim or correction was altered.**

Two image paths in the original are **root-relative** (`/docs/090201.png`,
`/docs/section8_real_hw_vs_sim.png`). Those were left exactly as they are: GitHub
resolves a leading slash against the repository, so they render correctly from any
directory depth and needed no change.

## Reconstructing the original

```bash
cat 01-intro-and-methodology.md 02-synthesis-vs-tket.md \
    03-compile-time-scaling.md 04-real-device-topology.md \
    05-fidelity.md 06-open-questions-and-roadmap.md \
  | sed '/^> \*\*Archived record/,/^---$/d' \
  | sed 's|](\.\./\.\./docs/|](./docs/|g' > README-2026-09-11.md
```

(Strip the archival headers and undo the image-path rewrite; the remainder is the
original file.)
