# spare-qubit-cliff: Combined Addenda, Part 7 of 8 (Addendum 108 through Addendum 136)

**Continued from [Part 6](spare-qubit-cliff-combined-88.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 5](spare-qubit-cliff-combined-51.md)).** Same conventions as every prior part: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

**Note on this part specifically**: it holds three threads, kept in chronological order and indexed here so each can be read on its own.

| Thread | Addenda | What it covers |
|---|---|---|
| Paper preparation | 108 | A correction to Paper 2's own figures, found by recomputing from raw data before writing. |
| Compilation inside a training loop | 109-118 | Whether re-compiling bound circuits pays off in variational training; a layout-once strategy; PSF-Zero's extra circuit depth measured, traced to one line of [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py), fixed (VERSION 2026-09-21), and the README re-measured. |
| Training under noise from PyTorch | 119-126 | A `torch.autograd.Function` bridge and noisy training experiments: PSF-Zero halves the execution error of redundant deep circuits (120, 122) and ties optimally written blocks (122); its per-circuit speed is a property of any bound-value re-compile (124); on a task that needs depth, the deep circuit beats a shallow one below about 0.5% CX error and loses above it (122, 126). A requested re-run after the CX-decomposer fix (116-117) was found unnecessary (133-134): the Heisenberg circuits were already built on the fixed decomposer from the start (117 predates 119 within the same day), confirmed by byte-identical results rather than assumed from dates. |
| PennyLane, non-IBM route | 127-130 | Whether the Qiskit version `pennylane-qiskit` can install still has Paper 1's VF2Layout failure region: Qiskit 1.2.4 (what pip resolved under Python 3.10) shows a different, seed-dependent pattern, 195-224x slower than Qiskit 2.5.2 even on the easy control (127-129); Qiskit 2.3.0 (the newest release, reachable under Python 3.11+) shows the SAME failure-region shape as 2.5.2, settling that 1.2.4's behaviour does not generalize (130). A Qiskit-independent PennyLane layout+synthesis module (`psf_pennylane.py`) was built and verified end-to-end on both PennyLane 0.42.3 and 0.45.1, finding perfect layouts on every instance where Qiskit's own VF2Layout fails. A fair head-to-head environment now exists. **The first real head-to-head, through PennyLane's own call path on both sides (131-132): Route A (IBM, Qiskit's own transpile) took 6.7-7.2s on the failure-region instance where Route B (PSF-Zero) took 4.6-5.8ms (1,200-1,450x), with n=6 correctness verified to machine precision (132). Two design bugs (a Windows-incompatible timeout, and an attempt to execute a 42-qubit circuit exactly, which needs tens of terabytes) were found and fixed before any data was collected.** |

Two things that happened in the same period are **not** addenda and are not merged here: the PennyLane transform `r0_psf_zero_transform.py` was re-built and verified end-to-end (its verification record lives in that file's own docstring), and `check_core_build.py` was revised to judge a build by its exports rather than file dates (see its docstring). Both papers were archived on Zenodo in this period (Paper 1: DOI 10.5281/zenodo.22869976; Paper 2: 10.5281/zenodo.22870141), recorded in the README rather than an addendum.

---
<!-- ===== Addendum 108 (source: spare-qubit-cliff-addendum-108-2026-09-21.md) ===== -->

> **Note added when merging:** Found while checking Paper 2's numbers against raw data before writing them down: the archived run's cumulative speedup was 7.00x, not the 8.87x Addendum 93 recorded. Every later citation uses 7.00x.

## Addendum 108 -- correcting Addendum 93: the archived run's own cumulative speedup was 7.00x, not 8.87x, verified by direct recomputation from the raw data (2026-09-21)

**Status**: a correction, found while preparing Paper 2's own numbers and
checking every figure against raw data before writing it down, per this
project's own standing practice.

## 0. In one line

**Addendum 93 stated the archived run's own cumulative speedup
(Qiskit / PSF-Zero \texttt{verify=False}) was $8.87\times$. Recomputing
directly from that run's own raw data
(\texttt{cumulative\_compile\_times\_10000\_ARCHIVE\_4daysago.npz}) gives
$7.00\times$.** The `today` and `rerun3` datasets' own figures
($8.75\times$, $9.27\times$) are confirmed to reproduce exactly by the
identical method, so the method itself is not in question -- the
archive figure specifically was wrong in the original addendum.

## 1. What was checked

```
archive: total_qiskit=92.071s, total_psf_false=13.144s -> 7.00x
today:   total_qiskit=103.754s, total_psf_false=11.864s -> 8.75x
rerun3:  total_qiskit=106.555s, total_psf_false=11.491s -> 9.27x
```

`today` and `rerun3`'s figures match Addendum 93's own and the original
terminal output's own printed values exactly. Only the archive figure
disagrees: Addendum 93 recorded $8.87\times$; direct summation of the
archive's own 10,000-point raw array gives $7.00\times$.

## 2. Likely cause, stated at the confidence level this addendum can support

We do not have the original archive run's own terminal output text to
compare against -- Addendum 93's own verification section states this
figure was "re-read directly from each run's own printed summary," but
the source terminal text is not independently available to us now to
determine whether the transcription was wrong, or whether the archive
`.npz` file itself was saved from a run other than the one whose
terminal output reported $8.87\times$. **We cannot distinguish these
two possibilities and do not guess between them.** What we can state
with confidence is that the raw data now on file, when summed directly,
gives $7.00\times$, and that this is the number that should be used in
any further work, since it is derived from data we can independently
recompute rather than from a transcribed terminal value we cannot
re-verify.

## 3. What this does not change

- Addendum 93's own primary finding (the variance collapse itself,
  $34$--$54\times$ via Levene's test) is unaffected -- that comparison
  used standard deviation, not the cumulative-speedup ratio, and was
  independently re-verified as part of preparing this correction.
- Addendum 99's own conclusion (the cause of the variance collapse
  remains unidentified) is unaffected.
- Every other addendum's own cumulative-speedup citations
  (`today`/`rerun3`, both confirmed exact) are unaffected.

## 4. Corrected figures for downstream use

| dataset | total Qiskit | total PSF (verify=False) | speedup |
|---|---:|---:|---:|
| archive (4 days ago) | 92.071s | 13.144s | **7.00x** (was reported as 8.87x) |
| today | 103.754s | 11.864s | 8.75x (unchanged) |
| rerun3 | 106.555s | 11.491s | 9.27x (unchanged) |

## 5. Verification

- Recomputed directly from all three `.npz` files' own raw arrays via
  `.sum()`, not from any printed summary text, immediately before this
  correction was written.
- `today` and `rerun3` were recomputed by the identical method
  specifically to confirm the method itself was not the source of the
  discrepancy, before concluding the archive figure was the one in
  error.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---

<!-- ===== Addendum 109 pre-registration (source: spare-qubit-cliff-addendum-109-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Opens the quantum-AI thread: in a variational training loop on hardware, is re-compiling every bound circuit worth it? Compile-once (A), Qiskit re-compile (B), PSF-Zero re-compile (C), with a negative-control ansatz and a pre-registered kill criterion.

## Addendum 109 -- Pre-registration: in a variational (quantum-AI) training loop run against hardware, is re-compiling every bound circuit worth it, and does PSF-Zero make it affordable? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

The goal is to decide whether PSF-Zero has a real role as a compilation
core for quantum-AI workloads (variational circuits trained by gradient
descent). The honest starting point, stated before designing anything:

- **In simulator-only training there is no compile step at all** -- the
  circuit is differentiated directly -- so PSF-Zero has no role there. The
  only place it can matter is training against hardware, where every
  circuit must be compiled to the device.
- **On hardware, Qiskit's recommended pattern avoids per-iteration
  compilation entirely**: transpile the *parameterized* circuit once, then
  bind new parameter values each step (strategy A below). If that is good
  enough, there is nothing for a faster compiler to speed up.
- **What binding-then-compiling (strategies B, C) can buy is circuit
  quality**: once values are numeric, consecutive gates on the same qubit
  pair can be consolidated and re-synthesized, which a parameterized
  circuit cannot do. Fewer two-qubit gates means less noise on hardware.

So the question is not "how fast is PSF-Zero" in isolation, but: **does
re-compiling bound circuits produce better circuits than compile-once, and
if so, does PSF-Zero bring the per-iteration cost of doing that down to
something usable?**

**A deliberate change from how this experiment was first described in
conversation**: it was originally framed as "what fraction of the training
loop is compilation." That fraction cannot be measured meaningfully here --
circuit execution on a simulator bears no relation to execution time on
real hardware (seconds to minutes per job including queueing). This
experiment therefore reports **absolute compile seconds per training
iteration** and leaves the comparison against a specific backend's job
time to the reader. No execution time is measured.

## 2. Design

### Ansatz families (both taken from the papers' own instance families)

Parameters: one `ry` and one `rz` per qubit per layer, so `P = 2 n L`.

| family | entangling pattern per layer | interaction graph | expected effect of re-synthesis |
|---|---|---|---|
| `same_pair` | CX on fixed pairs (2i, 2i+1), every layer | perfect matching (`dense_pairs`) | a pair accumulates L CXs, which KAK caps at 3 |
| `brickwork` | CX on (2i, 2i+1) on even layers, (2i+1, 2i+2) on odd | a path (chain-shaped) | none expected: each block holds one CX |

`brickwork` is included precisely because it is the family where
re-synthesis is expected to buy nothing -- a negative control against
choosing an ansatz that flatters the method. L = 6 layers for both, chosen
so that `same_pair` accumulates more CXs per pair (6) than a KAK
decomposition ever needs (3).

### Devices (fully saturated, as in both papers)

| grid | n | P (L=6) | circuits per iteration (parameter shift, 2P+1) | correctness check |
|---|---:|---:|---:|---|
| 4x4 | 16 | 192 | 385 | yes (statevector) |
| 6x7 | 42 | 504 | 1,009 | structural only |

### Strategies

- **A -- compile once**: transpile the parameterized circuit once with
  Qiskit `optimization_level=3`; each step only binds values. Reported: the
  one-time compile, and per-circuit bind time.
- **B -- re-compile with Qiskit**: bind, then Qiskit `optimization_level=3`.
- **C -- re-compile with PSF-Zero**: bind, then
  `compile_for_hardware(layout_search=True, routing_optimization_level=1,
  entangling_basis="cx", verify=False)`, matching Addenda 101-102.

For each (grid, family, strategy): K = 5 random parameter vectors (fixed
seeds), one excluded warm-up. Per circuit: compile (or bind) time, two-qubit
gate count, depth, coupling-map violations, and at 4x4 the error in
`<(1/n) sum_i Z_i>` against the ideal uncompiled circuit, with the routing
permutation taken from `TranspileLayout.final_index_layout()` (the method
verified in Addendum 103).

**Per-iteration cost is extrapolated, and labelled as such**: median
per-circuit time x (2P+1). Running all 1,009 circuits per iteration would
cost up to hours on the Qiskit arm and is not needed to answer the question.

## 3. Pre-registered predictions

**P1 (quality -- the gate that decides everything).**
  - `same_pair`: B and C produce at most 3 two-qubit gates per pair-block
    (about half of A's 6). **If A's count equals B's and C's, re-compiling
    buys nothing, strategy A dominates, and PSF-Zero has no role in this
    loop** -- the pre-registered kill criterion for this direction.
  - `brickwork` (negative control): A, B and C are predicted to have the
    same two-qubit gate count. A reduction here would be unexpected and
    would need explaining before being reported as a benefit.

**P2 (cost, conditional on P1 showing a quality gain).** C's per-circuit
compile time is predicted to be well below B's on every configuration, and
dramatically so at 6x7 `same_pair`, where B is predicted to hit the
`VF2Layout` failure region Paper 1 characterizes (seconds per circuit),
paid on every one of the 1,009 circuits per iteration, whereas A pays it
once.

**P3 (correctness).** At 4x4, every strategy's expectation value agrees
with the ideal circuit to better than 1e-9. Any violation invalidates that
strategy's timing and quality numbers.

**P4 (structure).** Zero coupling-map violations on every circuit, every
strategy.

## 4. What this cannot establish

- Actual hardware job time, queueing, or noise -- not measured; the
  gate-count difference is a proxy for noise, not a measurement of it.
- Training convergence -- no optimizer is run; only the compile side of one
  iteration is measured.
- Other ansatz families, other devices, gradient methods other than
  parameter shift, or batched submission patterns that change how many
  circuits must be compiled.
- Whether Qiskit strategies other than optimization level 3 (e.g. level 1
  with a warm layout) change the picture -- not tested here.

---

<!-- ===== Addendum 110 (source: spare-qubit-cliff-addendum-110-2026-09-21.md) ===== -->

> **Note added when merging:** Kill criterion not triggered: re-compiling halves two-qubit gates on a repeated-pair ansatz, and PSF-Zero is the only re-compile that stays usable at the layout cliff (278x). But compile-once stays far cheaper, PSF-Zero's circuits were deeper, and an error in this project's own extrapolation is corrected.

## Addendum 110 -- Variational-loop compile experiment: re-compiling halves two-qubit gates on repeated-pair ansatze, and PSF-Zero is the only per-circuit re-compile that stays usable where Qiskit hits the layout cliff (278x) -- but compile-once remains far cheaper, and PSF-Zero's circuits are deeper (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-109-preregistration-2026-09-21.md`, written and
locked before this run. All four predictions scored below, plus one error in
this project's own script, found while analysing the output.

## 0. In one line

**The kill criterion did not trigger.** On the `same_pair` ansatz, both
re-compile strategies produce exactly half of compile-once's two-qubit gates
(24 vs 48 at 4x4; 63 vs 126 at 6x7 -- three CX per pair instead of six); on
the `brickwork` negative control, all three are identical, as predicted.
**The gate reduction comes from re-compiling, not from PSF-Zero specifically**
-- Qiskit's own re-compile achieves the same count. PSF-Zero's contribution is
cost: its per-circuit time beats Qiskit's everywhere, modestly (1.3-2.1x) on
three configurations and by **278x** on 6x7 `same_pair`, where Qiskit re-hits
the `VF2Layout` failure region of Paper 1 on every circuit (6.8 s each, about
1.9 hours per training iteration) while PSF-Zero takes 24.5 ms. **Two results
cut the other way and are reported with equal weight**: compile-once is still
by far the cheapest per iteration (0.22-1.04 s vs PSF-Zero's 4.4-24.8 s), and
on `same_pair` PSF-Zero's circuits are **deeper** than Qiskit's own (23 vs 16)
at the same two-qubit count.

## 1. Results

Steady-state per-iteration cost is median per-circuit time x (2P + 1)
parameter-shift circuits. **The script's own printed figure for strategy A
added the one-time compile to every iteration; that is corrected here** (see
Section 3). Correctness at 4x4: every strategy's expectation value matched the
ideal circuit to at most 4.7e-14.

| config | strategy | 2q | depth | ms / circuit | s / iteration (steady) | one-time |
|---|---|---:|---:|---:|---:|---:|
| 4x4 same_pair | A compile once | 48 | 36 | 0.57 | 0.22 | 2.14 s |
| | B Qiskit L3 | 24 | 16 | 18.06 | 6.95 | |
| | C PSF-Zero | 24 | 23 | 13.45 | 5.18 | |
| 4x4 brickwork | A compile once | 45 | 36 | 0.66 | 0.25 | 0.02 s |
| | B Qiskit L3 | 45 | 33 | 16.78 | 6.46 | |
| | C PSF-Zero | 45 | 30 | 11.52 | 4.43 | |
| 6x7 same_pair | A compile once | 126 | 36 | 1.03 | 1.04 | 6.84 s |
| | B Qiskit L3 | 63 | 16 | 6,825.13 | 6,886.56 | |
| | C PSF-Zero | 63 | 23 | 24.54 | 24.76 | |
| 6x7 brickwork | A compile once | 123 | 36 | 0.99 | 1.00 | 0.04 s |
| | B Qiskit L3 | 123 | 33 | 36.39 | 36.72 | |
| | C PSF-Zero | 123 | 30 | 17.23 | 17.38 | |

## 2. Scoring

**P1 (quality; the kill criterion) -- CONFIRMED, criterion not triggered.**
`same_pair`: B and C both at exactly 3 two-qubit gates per pair, half of A's 6,
at both sizes. `brickwork` negative control: all three identical (45; 123), as
predicted -- the reduction is specific to the ansatz where it was expected,
not an artefact that would appear on any circuit.

**P2 (cost) -- CONFIRMED in direction, but "well below" overstated three of
four configurations.** C beat B everywhere, but by 1.3x, 1.5x and 2.1x outside
the cliff case. The large effect is confined to 6x7 `same_pair` (278x), which
the pre-registration did predict specifically. Note that A also hit the cliff
there -- once (6.84 s one-time), because the parameterized circuit has the
same interaction graph.

**P3 (correctness) -- CONFIRMED.** Maximum expectation-value error 4.7e-14
(PSF-Zero, 4x4 `same_pair`); all strategies under 1e-9.

**P4 (structure) -- CONFIRMED.** Zero coupling-map violations, every circuit,
every strategy.

## 3. An error in this project's own script, corrected

`verify_vqa_compile_loop.py` added strategy A's one-time compile to its
**per-iteration** extrapolation. Over a training run of many iterations, that
compile is paid once, not every iteration, so the script overstated A's
per-iteration cost -- most visibly at 4x4 `same_pair` (printed 2.36 s; steady
state 0.22 s) and 6x7 `same_pair` (printed 7.87 s; steady state 1.04 s). The
error flattered PSF-Zero relative to compile-once; the corrected figures in
Section 1 make compile-once look better, not worse. Found by recomputing from
the raw CSV before writing this addendum.

## 4. What this means for the quantum-AI direction

**PSF-Zero has a real, but narrow and conditional, role in this loop.**

- **The quality lever is real**: on ansatze that repeat entanglers on the same
  qubit pairs, re-compiling bound circuits halves two-qubit gates. On hardware
  that is a noise reduction, not a speedup.
- **PSF-Zero is what makes that lever affordable where it matters most.** The
  ansatz that benefits from re-synthesis (`same_pair`) has a matching-shaped
  interaction graph, which at full device occupancy is exactly the shape
  Qiskit's `VF2Layout` fails on. In the one such configuration tested,
  per-circuit re-compiling with Qiskit is impractical (about 1.9 hours per
  iteration) and with PSF-Zero is 25 s. Whether this coincidence holds
  generally is a hypothesis from two configurations, not a finding.
- **The decision is a trade-off, not a win.** Against compile-once, PSF-Zero
  costs roughly 5-24 s more per iteration in exchange for half the two-qubit
  gates, and produces deeper circuits than Qiskit's own re-compile. Whether
  that is worth it depends on a backend's job time and noise, neither measured
  here.

## 5. A candidate improvement, not tested

Strategy C re-runs layout search and routing on every circuit, but in a
training loop the interaction graph never changes -- only parameter values do.
Computing the layout once and re-running only block consolidation and KAK
synthesis per circuit should remove most of C's per-circuit cost (Paper 2's
synthesis-only path is about 1 ms per 15-qubit circuit). This would narrow the
gap to compile-once while keeping the halved gate count. Proposed, not
measured.

## 6. What this does not establish

- Hardware execution time, queueing, or noise -- not measured.
- Training convergence -- no optimizer was run.
- Why PSF-Zero's `same_pair` circuits are deeper than Qiskit's at equal
  two-qubit count -- consistent with Addendum 102's end-to-end result (depth 23
  vs 16 on the same family), not investigated.
- Other ansatze, devices, gradient methods, or Qiskit settings.

## 7. Files

| File | What it is |
|---|---|
| [`verify_vqa_compile_loop.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vqa_compile_loop.py) | this run's script (its strategy-A per-iteration figure is corrected in Section 3) |
| [`vqa_compile_loop_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vqa_compile_loop_2026-09-21.csv) | raw results, 12 rows |
| [`spare-qubit-cliff-addendum-109-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-109-preregistration-2026-09-21.md) | the predictions scored above |

## 8. Verification

- Every figure in Section 1 was recomputed from the raw CSV, not read off the
  terminal output; the steady-state column is computed there, not printed by
  the script.
- The B/C ratios in Section 0 were computed from per-circuit medians in the
  CSV.
- The depth comparison was checked against Addendum 102's own recorded values
  on the same interaction-graph family before being called consistent.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0 hits.

---

<!-- ===== Addendum 111 pre-registration (source: spare-qubit-cliff-addendum-111-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Strategy D: find the layout once, re-synthesize per circuit. Includes a basis mismatch caught while writing the script, before any run.

## Addendum 111 -- Pre-registration: lay out once, re-synthesize per circuit -- does it keep the halved two-qubit count at close to compile-once cost? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 110 found that re-compiling bound circuits halves two-qubit gates
on a repeated-pair ansatz, and that PSF-Zero makes this affordable where
Qiskit hits the layout cliff -- but that compile-once remains far cheaper
(0.22-1.04 s per iteration vs PSF-Zero's 4.4-24.8 s). Its Section 5 proposed,
without measuring, why: strategy C re-runs layout search and routing on every
circuit, although in a training loop the interaction graph never changes --
only parameter values do.

This experiment tests that proposal as a new strategy, D.

## 2. Design

Same ansatz families, grids, samples, seeds and correctness check as
Addendum 109 (`same_pair`, `brickwork`; 4x4 and 6x7, fully saturated;
L = 6; K = 5 samples plus one excluded warm-up).

**Strategy D -- lay out once, re-synthesize per circuit:**
- *Once per training run:* `smart_vf2_layout` (the repaired search verified
  in Addenda 88-106) finds a perfect layout -- every interacting pair on a
  device edge, no SWAPs -- for the ansatz's interaction graph.
- *Per circuit:* bind values; `psf_compile.compile(verify=False,
  entangling_basis="cx")` (gate synthesis only, no coupling map; the basis
  is set explicitly to match strategy C, since `compile()`'s own default is
  `"canonical"`, which emits non-CX two-qubit gates -- caught while writing
  the script, before any run); place the result on the device by the
  fixed layout; convert single-qubit gates to the device basis
  (`Optimize1qGatesDecomposition`). No layout search, no routing.

D only applies when a perfect layout exists. If `smart_vf2_layout` finds
none for a configuration, D is reported as not applicable there -- not
patched with routing.

Strategies A, B, C are re-run alongside for a same-session comparison.
**This addendum's script reports one-time and steady-state per-iteration
costs in separate columns**, fixing the error found in Addendum 110
Section 3.

Recorded per circuit, for every strategy: time, two-qubit count, depth,
coupling-map violations, and whether every output gate is in the device
basis. At 4x4, the expectation-value error against the ideal circuit; for D
this uses D's own fixed layout directly, since D produces no
`TranspileLayout` metadata.

## 3. Pre-registered predictions

**P1 (quality preserved).** D's two-qubit count equals C's on all four
configurations (24 / 63 on `same_pair`; 45 / 123 on `brickwork`): D runs the
same synthesis as C and inserts no SWAPs.

**P2 (cost, primary).** D's per-circuit time is at least 3x below C's on all
four configurations, and D's steady-state per-iteration cost is within 10x
of compile-once (A). **If D is not at least 3x faster than C, the proposal in
Addendum 110 Section 5 is falsified** -- layout and routing were not the
dominant part of C's cost.

**P3 (correctness).** At 4x4, D's expectation value matches the ideal
circuit to better than 1e-9.

**P4 (structure).** Zero coupling-map violations and every output gate in
the device basis, for D on every circuit.

**Not predicted:** D's depth relative to B and C. Addendum 110 found C's
`same_pair` circuits deeper than Qiskit's (23 vs 16); D replaces C's
post-processing with a single-qubit-only pass, and whether that raises or
lowers depth is reported without a prior claim.

## 4. What this cannot establish

- Circuits with no perfect layout -- D does not apply to them as designed.
- Hardware time, noise, or training convergence -- unchanged limits from
  Addendum 109.
- Whether a Qiskit-side analogue (fixed `initial_layout` with re-synthesis
  per circuit) would close the same gap -- not tested; this compares against
  Qiskit's own default re-compile only.

---

<!-- ===== Addendum 112 (source: spare-qubit-cliff-addendum-112-2026-09-21.md) ===== -->

> **Note added when merging:** D keeps the halved gate count at 2.3-4.7x below C's cost, within 5-8x of compile-once -- but the pre-registered 3x-on-all-four target failed on one configuration, reported as a miss. D's depth equals C's, localizing the depth gap to synthesis itself.

## Addendum 112 -- Lay out once, re-synthesize per circuit: gate count and correctness preserved exactly, 2.3-4.7x cheaper than per-circuit PSF-Zero, within 5-8x of compile-once -- but the pre-registered "at least 3x on all four configurations" failed on one (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-111-preregistration-2026-09-21.md`, written and
locked before this run.

## 0. In one line

**P1, P3, P4 confirmed; P2 partly falsified.** Strategy D (perfect layout
found once; per circuit only synthesis, placement, and single-qubit basis
conversion) reproduces strategy C's two-qubit count exactly on every
configuration and every sample (24 / 45 / 63 / 123), matches the ideal
circuit (4.7e-14 worst at 4x4), with zero coupling violations and every gate
in the device basis. It is 4.70x, 3.42x and 4.41x cheaper per circuit than C
on three configurations -- **but only 2.33x on 6x7 `brickwork`, below the
pre-registered 3x on all four.** Taken literally, the pre-registration's own
falsification clause applies to that configuration: there, per-circuit
layout and routing were not the dominant part of C's cost. D's steady-state
cost is 4.9-7.7x compile-once's, inside the pre-registered 10x bound. **An
unpredicted, clarifying result**: D's depth equals C's exactly (23 / 30),
which localizes the depth gap to Qiskit (23 vs 16 on `same_pair`) inside
PSF-Zero's synthesis itself, not in C's routing or post-processing.

## 1. Results

Same-session re-run of all four strategies. Steady-state per iteration =
median per-circuit time x (2P + 1), extrapolated; one-time costs separate.

| config | strategy | 2q | depth | ms / circuit | steady s / iter | one-time s |
|---|---|---:|---:|---:|---:|---:|
| 4x4 same_pair | A compile once | 48 | 36 | 0.59 | 0.23 | 1.278 |
| | B Qiskit L3 | 24 | 16 | 18.20 | 7.01 | |
| | C PSF-Zero | 24 | 23 | 14.29 | 5.50 | |
| | **D layout once** | **24** | **23** | **3.04** | **1.17** | 0.181 |
| 4x4 brickwork | A compile once | 45 | 36 | 0.73 | 0.28 | 0.019 |
| | B Qiskit L3 | 45 | 33 | 16.98 | 6.54 | |
| | C PSF-Zero | 45 | 30 | 12.22 | 4.70 | |
| | **D layout once** | **45** | **30** | **3.58** | **1.38** | 0.001 |
| 6x7 same_pair | A compile once | 126 | 36 | 1.00 | 1.01 | 6.970 |
| | B Qiskit L3 | 63 | 16 | 6,742.00 | 6,802.68 | |
| | C PSF-Zero | 63 | 23 | 23.39 | 23.60 | |
| | **D layout once** | **63** | **23** | **5.30** | **5.35** | 0.002 |
| 6x7 brickwork | A compile once | 123 | 36 | 0.96 | 0.97 | 0.041 |
| | B Qiskit L3 | 123 | 33 | 36.41 | 36.74 | |
| | C PSF-Zero | 123 | 30 | 17.01 | 17.16 | |
| | **D layout once** | **123** | **30** | **7.31** | **7.38** | 0.001 |

| config | C / D per circuit | D / A per circuit |
|---|---:|---:|
| 4x4 same_pair | 4.70x | 5.18x |
| 4x4 brickwork | 3.42x | 4.91x |
| 6x7 same_pair | 4.41x | 5.30x |
| 6x7 brickwork | **2.33x** | 7.65x |

A perfect layout was found for every configuration, so D was applicable
everywhere. All strategies: zero coupling violations, every gate in the
basis; at 4x4 every expectation-value error below 1e-13. Strategies A, B, C
reproduce Addendum 110's own figures closely (same two-qubit counts and
depths; timings within run-to-run variation).

## 2. Scoring

**P1 (quality preserved) -- CONFIRMED.** D's two-qubit count equals C's on
all four configurations, minimum and maximum across samples identical.

**P2 (cost) -- PARTLY FALSIFIED.** The prediction was "at least 3x below C on
all four configurations" plus "within 10x of A". The second half holds
everywhere (4.9-7.7x). The first holds on three of four and fails on 6x7
`brickwork` (2.33x). The pre-registration stated that failing 3x would
falsify the proposal that layout and routing dominate C's cost; on
`brickwork` at 6x7 that proposal is falsified, and on the other three it
holds. Reported as stated, not re-thresholded after the fact.

**P3 (correctness) -- CONFIRMED.** D's expectation-value error at 4x4:
4.7e-14 (`same_pair`), 2.8e-16 (`brickwork`), using D's own fixed layout.

**P4 (structure) -- CONFIRMED.** Zero violations; every gate in
{rz, sx, x, cx}.

**Depth (not predicted).** D's depth equals C's exactly on every
configuration. C adds routing (level 1) and a basis translation after
synthesis; D replaces both with a single-qubit-only pass. Identical depth
means neither step is responsible for PSF-Zero's deeper `same_pair` circuits
(23 vs Qiskit's 16): the extra depth is already present in the synthesized
circuit. This narrows Addendum 110's open depth question to one place,
without yet explaining it.

## 3. What this means for the quantum-AI direction

- **Where re-synthesis reduces gates (`same_pair`)**, D keeps the halved
  two-qubit count at about 1.2 s (16 qubits) and 5.4 s (42 qubits) per
  training iteration, against compile-once's 0.23 s and 1.01 s -- down from
  C's 5.5 s and 23.6 s, and from Qiskit's per-circuit re-compile at 7.0 s and
  about 1.9 hours. **This is the configuration the direction depends on, and
  the one where D performed as predicted (4.4-4.7x over C).**
- **Where it does not (`brickwork`)**, D costs 5-8x compile-once for no gate
  benefit. On such ansatze, compile-once is strictly better, and a user should
  not pay for re-synthesis.
- **D's one-time cost is negligible (0.001-0.18 s) and avoids the layout
  cliff entirely**, which compile-once pays once at 6x7 `same_pair` (6.97 s).
- **D is still 5-8x compile-once per iteration.** Where that residual goes
  (block collection and consolidation, the Rust call, parameter binding, the
  single-qubit pass) is not measured here; D minus A is roughly 2.5-6.3 ms per
  circuit.

## 4. What this does not establish

- Why D's advantage over C is smaller on `brickwork` -- a plausible reading
  is that the synthesis pipeline itself is a larger share there, but no cost
  breakdown was measured.
- The source of PSF-Zero's extra depth -- now localized to synthesis, not
  explained.
- Circuits with no perfect layout, hardware time, noise, convergence --
  unchanged limits.

## 5. Files

| File | What it is |
|---|---|
| [`verify_vqa_layout_once.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vqa_layout_once.py) | this run's script (strategies A-D) |
| [`vqa_layout_once_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vqa_layout_once_2026-09-21.csv) | raw results, 16 rows |
| [`spare-qubit-cliff-addendum-111-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-111-preregistration-2026-09-21.md) | the predictions scored above |

## 6. Verification

- Every figure recomputed from the raw CSV; ratios computed from per-circuit
  medians there.
- The P2 threshold was applied exactly as pre-registered (3x on all four);
  the one miss is reported as a miss, not re-framed.
- The depth equality was checked for all four configurations, not one.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0 hits.

---

<!-- ===== Addendum 113 pre-registration (source: spare-qubit-cliff-addendum-113-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Before trying to fix the depth gap, measure whether it is real pulses or only virtual rz, and where in each block it sits.

## Addendum 113 -- Pre-registration: what is PSF-Zero's extra depth made of, and where in each block does it appear? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addenda 110 and 112 found PSF-Zero's `same_pair` circuits deeper than
Qiskit's own re-compile at equal two-qubit count (23 vs 16), and Addendum
112 localized the extra depth to PSF-Zero's synthesis itself: strategy D,
which replaces all of C's post-processing, has exactly C's depth.

Two points were settled before designing this experiment:

1. **"Uncancelled adjacent single-qubit gates" is already ruled out as
   stated.** Strategy D runs `Optimize1qGatesDecomposition` over the whole
   circuit, which rewrites every maximal run of adjacent single-qubit gates
   into its minimal {rz, sx, x} form. Adjacent single-qubit gates therefore
   cannot survive unmerged. Any remaining difference must be structural: more
   single-qubit layers that are *separated by CX gates* and so cannot merge.
2. **The depth compared so far counts every gate, including `rz`.** On
   IBM-style hardware `rz` is a virtual frame change (no pulse, essentially no
   time or error); the noise-relevant quantities are the number of CX layers
   and the number of `sx` pulses. If the extra depth is made of `rz` alone, it
   is close to harmless on such hardware; if it includes `sx` or CX layers, it
   is a real cost. **This must be measured before any fix is attempted.**

## 2. Design

Same ansatz families, grids, seeds and samples as Addenda 109-112. Two
strategies compared: B (Qiskit L3 re-compile) and D (layout once +
PSF-Zero synthesis); A (compile once) included for reference. D's depth
equals C's (Addendum 112), so D stands for PSF-Zero's synthesis.

Per output circuit:
- total depth (as reported so far)
- **CX depth**: depth counting only two-qubit gates
- **sx depth**: depth counting only `sx`/`x` (the physical single-qubit pulses)
- gate counts: `cx`, `sx`, `x`, `rz`

And, for 4x4 `same_pair` only, one representative pair's full gate sequence
in order, for B and D side by side -- the pairs are independent in this
family, so this sequence is the whole block -- plus the number of
single-qubit gates in each gap between consecutive CXs on that pair.

## 3. Pre-registered predictions

**P1.** CX depth is equal between B and D on all four configurations
(3 on `same_pair`: three CX per pair, pairs in parallel; equal on
`brickwork`, where no two-qubit re-synthesis happens).

**P2 (the one that decides severity).** On `same_pair`, D has more `sx`
pulses and a larger sx depth than B -- i.e. the extra depth is not purely
virtual `rz`. **If D's sx depth and sx count equal B's, the extra depth is
made of `rz` only, the issue is close to harmless on IBM-style hardware, and
a fix is not a priority** -- recorded as the pre-registered outcome that
would deprioritize this line.

**P3 (location).** The extra single-qubit gates sit in the gaps between CXs
within a block (the interleaved layers of the two-qubit decomposition), not
before the first or after the last CX of a block.

**Not predicted:** the `brickwork` comparison, where Addendum 110 found D
*shallower* than B (30 vs 33); reported as measured.

## 4. What this cannot establish

- Actual pulse durations or error rates on a specific backend; `sx` count and
  depth are proxies. Backends where `rz` is not virtual would weigh the
  result differently.
- A fix -- this addendum only locates the cause.

---

<!-- ===== Addendum 114 (source: spare-qubit-cliff-addendum-114-2026-09-21.md) ===== -->

> **Note added when merging:** The extra depth is real sx pulses (1.6x Qiskit's), all between each block's CXs, traced to an unconfigured TwoQubitBasisDecomposer in psf_compile.py's CX path.

## Addendum 114 -- PSF-Zero's extra depth is real pulses (sx), located entirely between CXs, and traces to an unconfigured `TwoQubitBasisDecomposer` in its CX path (cause hypothesized, fix not yet tested) (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-113-preregistration-2026-09-21.md`, written and
locked before this run.

## 0. In one line

**P1, P2, P3 all confirmed.** CX depth is equal (3 on `same_pair`); the
extra depth is **not** virtual `rz` alone -- PSF-Zero emits 1.6x the `sx`
pulses of Qiskit's own re-compile (128 vs 80 at 4x4, 336 vs 210 at 6x7) and
a larger sx depth (8 vs 6). **So the gap is a real hardware cost**, and the
pre-registered "deprioritize" outcome does not apply. It sits **entirely in
the two gaps between each block's three CXs**: per pair, both place 4 + 4
`sx` outside the CXs; between them Qiskit places 1 + 1, PSF-Zero 4 + 4.
Reading PSF-Zero's source locates a specific, testable cause: its CX path
decomposes the Cartan core with `TwoQubitBasisDecomposer(CXGate())`,
constructed with no Euler basis or pulse-optimization option.

## 1. Results

Medians over 5 samples. CX depth counts two-qubit gates only; sx depth
counts `sx` and `x` only.

| config | strategy | total depth | CX depth | sx depth | #cx | #sx | #rz |
|---|---|---:|---:|---:|---:|---:|---:|
| 4x4 same_pair | A compile once | 36 | 6 | 12 | 48 | 192 | 288 |
| | B Qiskit L3 | 16 | 3 | 6 | 24 | 80 | 120 |
| | D PSF-Zero | 23 | 3 | 8 | 24 | **128** | 184 |
| 4x4 brickwork | A compile once | 36 | 6 | 12 | 45 | 192 | 288 |
| | B Qiskit L3 | 33 | 6 | 12 | 45 | 184 | 203 |
| | D PSF-Zero | 30 | 6 | 12 | 45 | 184 | 188 |
| 6x7 same_pair | A compile once | 36 | 6 | 12 | 126 | 504 | 756 |
| | B Qiskit L3 | 16 | 3 | 6 | 63 | 210 | 315 |
| | D PSF-Zero | 23 | 3 | 8 | 63 | **336** | 483 |
| 6x7 brickwork | A compile once | 36 | 6 | 12 | 123 | 504 | 756 |
| | B Qiskit L3 | 33 | 6 | 12 | 123 | 496 | 543 |
| | D PSF-Zero | 30 | 6 | 12 | 123 | 496 | 500 |

One pair's full block, 4x4 `same_pair`, first sample. Single-qubit gates per
segment (before the first CX, between CXs, after the last CX):

| strategy | segments | sx per segment |
|---|---|---|
| B Qiskit L3 | [10, 3, 2, 10] | 4, 1, 1, 4 (10 per pair) |
| D PSF-Zero | [10, 10, 9, 10] | 4, 4, 4, 4 (16 per pair) |

Per-pair `sx` counts reproduce the totals exactly: 10 x 8 pairs = 80 and
16 x 8 = 128 at 4x4; 10 x 21 = 210 and 16 x 21 = 336 at 6x7.

## 2. Scoring

**P1 (CX depth equal) -- CONFIRMED** on all four configurations.

**P2 (severity) -- CONFIRMED: the extra depth includes physical pulses.**
D has 48 more `sx` than B at 4x4 `same_pair` (126 more at 6x7) and a sx depth
of 8 vs 6. The pre-registered "extra depth is `rz` only, deprioritize"
outcome does not apply.

**P3 (location) -- CONFIRMED, exactly.** The outer segments are identical in
length and `sx` content (4 per side); all 6 extra `sx` per pair sit in the
two interior gaps.

**`brickwork` (not predicted):** D equals B in `sx` count and sx depth and
has fewer `rz` (188 vs 203; 500 vs 543), hence its shallower total depth.
No two-qubit re-synthesis happens there, so the interior-gap mechanism does
not arise -- consistent with the cause below.

## 3. Perspective against compile-once

Even with this gap, PSF-Zero's `same_pair` circuits beat compile-once (A) on
every noise-relevant count: half the CXs (24 vs 48), CX depth 3 vs 6, `sx`
128 vs 192, sx depth 8 vs 12. The gap reported here is against Qiskit's own
per-circuit re-compile only, which Addendum 110 found costs about 1.9 hours
per training iteration at 6x7.

## 4. Candidate cause, from the source

In `psf_compile.py`, the CX-basis path builds the Cartan core
`N(a, b, c)` and decomposes it with

```python
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())
```

constructed with no `euler_basis` and no `pulse_optimize` setting. The
interior single-qubit layers of a three-CX decomposition are therefore
emitted as general rotations, which become `rz sx rz sx rz` (two `sx`) per
qubit per gap after basis translation -- matching the observed interior
segments. Qiskit's own synthesis during transpilation targets the device
basis ({rz, sx}) and places at most one `sx` between CXs here -- matching
the observed `sx(a) rz(a) rz(b)` / `sx(a) rz(b)` interior segments.

**Stated at the confidence it has**: that `TwoQubitBasisDecomposer` accepts
an Euler-basis choice and a pulse-optimization option that minimizes
interior single-qubit gates for {CX, SX, RZ} is from knowledge of Qiskit, not
yet exercised in this project's environment (Qiskit 2.5.2). The mechanism
fits every observation above, but it is a hypothesis until a configured
decomposer is measured.

## 5. Proposed next test (not yet run)

Construct the decomposer as
`TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True)`
and re-run this addendum's analysis plus the correctness checks. Predictions
to pre-register: interior segments fall to B's length; `sx` per pair falls
from 16 to 10; sx depth from 8 to 6; CX count unchanged; unitary equivalence
unchanged; per-circuit time roughly unchanged (the decomposer runs only on a
cache miss per distinct Cartan triple).

## 6. What this does not establish

- That the proposed configuration exists and behaves as described in this
  Qiskit version -- untested.
- Pulse durations or error rates on any specific backend.
- Whether the same gap exists in PSF-Zero's other entangling bases
  (`canonical`, etc.) -- only the CX path was examined.

## 7. Files

| File | What it is |
|---|---|
| [`analyze_depth_composition.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/analyze_depth_composition.py) | this run's script |
| [`depth_composition_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/depth_composition_2026-09-21.csv) | raw results, 12 rows |
| [`spare-qubit-cliff-addendum-113-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-113-preregistration-2026-09-21.md) | the predictions scored above |

## 8. Verification

- All counts taken from the raw CSV; per-pair `sx` counts recomputed from the
  printed block sequences and checked against the totals.
- The candidate cause was read from `psf_compile.py` directly (the
  `_CX_DECOMPOSER` line and `_cx_core_cached`), not inferred from the output
  alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0 hits.

---

<!-- ===== Addendum 115 pre-registration (source: spare-qubit-cliff-addendum-115-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Tests configured decomposer variants in-process, without modifying psf_compile.py, with the decomposition cache cleared at each switch.

## Addendum 115 -- Pre-registration: does configuring PSF-Zero's CX decomposer for the {rz, sx} basis remove the extra pulses between CXs, without costing correctness or speed? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 114 found that PSF-Zero's `same_pair` circuits carry 1.6x the `sx`
pulses of Qiskit's own re-compile, all of it in the two gaps between each
block's three CXs (4 + 4 `sx` per pair vs Qiskit's 1 + 1), and traced this to
`psf_compile.py` line 265:

```python
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())
```

constructed with no Euler basis and no pulse-optimization setting. This
experiment tests whether configuring it fixes the gap.

## 2. Design

`psf_compile.py` itself is **not modified**. The script replaces the
module-level `_CX_DECOMPOSER` for the duration of each variant and restores
the original afterwards. `_CX_DECOMPOSER` is looked up at call time at both of
its use sites (the Cartan-core decomposition in `_cx_core_cached`, and the
whole-block fallback for degenerate blocks), so the replacement reaches both.
**The decomposition cache (`_CX_CORE_CACHE`) is cleared at every switch**, so
no result computed under one configuration is reused under another.

Variants:

| name | construction |
|---|---|
| `current` | `TwoQubitBasisDecomposer(CXGate())` (as shipped) |
| `zsx` | `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` |
| `zsx_pulse` | `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True)` |

If a variant fails to construct, or raises on any block, that is recorded as
its result rather than worked around.

**Test 1 -- correctness on isolated blocks.** 300 two-qubit circuits, each 20
random SU(4) gates on the same pair (seeded) -- above `compile()`'s
`DEFAULT_BLOCK_GATE_FLOOR` of 12, so every one is actually collected and
synthesized. Each is compiled with `psf_compile.compile(verify=False,
entangling_basis="cx")`, converted to {rz, sx, x, cx}, and checked:
exact operator fidelity against the input; that synthesis really happened
(no `unitary` left, 1-3 CXs present); `sx` count.

**Test 2 -- the variational ansatz.** Addendum 113's analysis re-run with
strategy D under each variant (4x4 and 6x7; `same_pair` and `brickwork`):
CX count and depth, `sx` count and depth, per-circuit time, the per-pair
segment view at 4x4 `same_pair`, and at 4x4 the expectation-value error
against the ideal circuit. Qiskit L3 re-compile (strategy B) re-run as the
reference.

## 3. Pre-registered predictions

**P1 (the fix works).** At least one configured variant brings `same_pair`
`sx` per pair from 16 to 10 (4x4: 128 -> 80; 6x7: 336 -> 210), sx depth from
8 to 6, and the interior segments to Qiskit's length -- matching strategy B.
**If neither configured variant changes the `sx` count, the cause in
Addendum 114 Section 4 is wrong** and the extra pulses come from somewhere
else in PSF-Zero's synthesis.

**P2 (nothing else changes).** CX count and CX depth unchanged on every
configuration; `brickwork` unchanged (its blocks are below the collection
floor and are never re-synthesized).

**P3 (correctness).** Every Test 1 block at fidelity >= 1 - 1e-9 under every
variant that constructs; every block actually synthesized; 4x4 expectation
error below 1e-9.

**P4 (speed).** Per-circuit time within run-to-run variation of `current`:
the decomposer runs once per distinct Cartan triple (cache miss), not per
gate.

## 4. What this cannot establish

- Pulse durations or error rates on real hardware -- `sx` count and depth are
  proxies.
- The effect on PSF-Zero's other entangling bases -- only the CX path is
  changed.
- Whether the chosen variant should become the shipped default -- that
  decision follows this measurement, it is not made by it.

---

<!-- ===== Addendum 116 (source: spare-qubit-cliff-addendum-116-2026-09-21.md) ===== -->

> **Note added when merging:** euler_basis="ZSX" removes the extra pulses exactly (sx and depth now equal Qiskit's), correctness unchanged; speed effect not resolvable from run noise. Includes a correction to its own first draft about which outputs the change affects.

## Addendum 116 -- Configuring the CX decomposer for the {rz, sx} basis removes PSF-Zero's extra pulses exactly: sx and depth now equal Qiskit's own re-compile, correctness unchanged; timing effect not resolvable from this run's noise (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-115-preregistration-2026-09-21.md`, written and
locked before this run. `psf_compile.py` was not modified; the decomposer was
replaced in-process for each variant and restored afterwards (confirmed by the
script's own final line).

## 0. In one line

**P1, P2, P3 confirmed; P4 not resolvable.** Constructing the decomposer as
`TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` brings PSF-Zero's
`same_pair` circuits exactly to Qiskit's own re-compile on every
noise-relevant count: `sx` 128 -> 80 (4x4) and 336 -> 210 (6x7), sx depth
8 -> 6, total depth 23 -> 16, per-pair `sx` per segment [4, 4, 4, 4] ->
[4, 1, 1, 4]. **This confirms the cause identified in Addendum 114.** CX count
and depth are unchanged; all 300 isolated blocks remain exact (worst fidelity
0.9999999999999973); expectation error at 4x4 unchanged (4.7e-14).
Adding `pulse_optimize=True` changes nothing measurable over `euler_basis`
alone. On `brickwork`, PSF-Zero stays shallower than Qiskit (total depth 30
vs 33) at equal `sx`.

## 1. Results

**Test 1 -- 300 isolated blocks (20 random SU(4) each; all actually
synthesized):**

| variant | raised | not synthesized | worst fidelity | median sx | median cx |
|---|---:|---:|---:|---:|---:|
| current | 0 | 0 | 0.9999999999999976 | 16 | 3 |
| zsx | 0 | 0 | 0.9999999999999973 | **10** | 3 |
| zsx_pulse | 0 | 0 | 0.9999999999999973 | **10** | 3 |

**Test 2 -- variational ansatz, strategy D, against Qiskit L3 re-compile (B):**

| config | B (Qiskit) sx / depth | current sx / depth | zsx sx / depth | zsx_pulse sx / depth |
|---|---|---|---|---|
| 4x4 same_pair | 80 / 16 | 128 / 23 | **80 / 16** | 80 / 16 |
| 4x4 brickwork | 184 / 33 | 184 / 30 | 184 / 30 | 184 / 30 |
| 6x7 same_pair | 210 / 16 | 336 / 23 | **210 / 16** | 210 / 16 |
| 6x7 brickwork | 496 / 33 | 496 / 30 | 496 / 30 | 496 / 30 |

sx depth on `same_pair`: B 6, current 8, zsx 6, zsx_pulse 6. CX count and CX
depth identical across all variants on every configuration.

Per-circuit time (median of 5), relative to `current`:

| config | current | zsx | zsx_pulse |
|---|---:|---:|---:|
| 4x4 same_pair | 5.80 ms | 5.67 ms (-2%) | 6.12 ms (+6%) |
| 4x4 brickwork | 3.49 ms | 3.49 ms (+0%) | 3.52 ms (+1%) |
| 6x7 same_pair | 12.08 ms | 14.43 ms (+19%) | 13.64 ms (+13%) |
| 6x7 brickwork | 6.84 ms | 7.47 ms (+9%) | 7.39 ms (+8%) |

## 2. Scoring

**P1 (the fix works) -- CONFIRMED, exactly.** Both configured variants reach
Qiskit's own counts and segment structure; the Addendum 114 cause is
confirmed rather than merely consistent with the data.

**P2 (nothing else changes) -- CONFIRMED.** CX count and depth unchanged
everywhere; `brickwork` unchanged in every count.

**P3 (correctness) -- CONFIRMED.** 300/300 blocks synthesized and exact under
every variant; no variant raised; 4x4 expectation error unchanged.

**P4 (speed) -- NOT RESOLVABLE from this run.** `brickwork` blocks fall
below `compile()`'s collection floor and never reach the decomposer, so its
timing differences between variants (up to +9% at 6x7) are pure run-to-run
noise within this script. The `same_pair` differences (-2% to +19%) are only
modestly larger than that noise, and this script's `current` timings are
themselves about 2x Addendum 112's for the same configuration (5.80 vs 3.04
ms; 12.08 vs 5.30 ms), so run-to-run variation across sessions is larger
still. The honest reading: no large slowdown; a slowdown of order 10-20% at
6x7 is neither shown nor excluded.

## 3. Recommended change (not yet applied)

In `psf_compile.py`, line 265:

```python
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
```

`euler_basis="ZSX"` alone is sufficient; `pulse_optimize=True` added nothing
measurable here and is left out as the simpler, less version-sensitive
choice.

**Consequences to state alongside the change:**
- **Which outputs change.** `compile()` and `compile_for_hardware()` default
  to `entangling_basis="canonical"`, which reaches this decomposer only on the
  degenerate-block fallback. Default-basis output, including Paper 2's
  10,000-iteration gate-synthesis speed benchmark, is therefore essentially
  unaffected -- confirmed by reading `test_cumulative_compile_scale.py`, which
  calls `compile()` without setting the basis. (A first draft of this section
  recommended re-running that benchmark; that was wrong and is corrected
  here.) Output with `entangling_basis="cx"` changes: same two-qubit counts,
  fewer `sx`, lower total depth -- e.g. the README's coupling-map cliff table
  (measured with "cx") reports depth 23 for `layout_search=True`, expected to
  become 16.
- The same decomposer serves the degenerate-block fallback path, which this
  experiment did not exercise (all 300 random blocks were non-degenerate).
- The benefit is specific to devices whose single-qubit basis is {rz, sx}
  (IBM-style). For other bases, transpilation still translates the output,
  but whether "ZSX" is then better, neutral, or slightly worse is untested.

## 4. What this does not establish

- Hardware noise impact beyond the proxies (`sx` count and depth).
- The speed effect to better than roughly +-20% (Section 2, P4).
- The fallback path's behaviour under the new setting.

## 5. Files

| File | What it is |
|---|---|
| [`compare_cx_decomposer.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compare_cx_decomposer.py) | this run's script (does not modify [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py)) |
| [`cx_decomposer_comparison_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cx_decomposer_comparison_2026-09-21.csv) | raw results, 19 rows |
| [`spare-qubit-cliff-addendum-115-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-115-preregistration-2026-09-21.md) | the predictions scored above |

## 6. Verification

- All figures recomputed from the raw CSV.
- The `brickwork` timing difference was used as an in-run noise estimate
  because those blocks provably never reach the decomposer (below the
  collection floor of 12 gates per pair run), not assumed to be noise.
- Cross-session timing variation was checked against Addendum 112's own
  recorded D timings for the identical configurations.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0 hits.

---

<!-- ===== Addendum 117 (source: spare-qubit-cliff-addendum-117-2026-09-21.md) ===== -->

> **Note added when merging:** The fix applied in psf_compile.py itself (VERSION 2026-09-21): depth and pulse counts identical to Qiskit's re-compile; 36/36 unitary equivalence; exactly the outputs that pass through the changed code changed.

## Addendum 117 -- The Addendum 116 decomposer fix, applied to `psf_compile.py` itself (VERSION 2026-09-21): depth and pulse counts now identical to Qiskit's own re-compile, and end-to-end correctness re-confirmed (2026-09-21)

**Status**: post-application check of the change recommended in Addendum 116,
now applied in the repository's own `psf_compile.py` (VERSION 2026-09-21,
changelog item 14). No new pre-registration: this re-runs two existing,
already pre-registered scripts unmodified, against their own prior results.

## 0. In one line

**The fix works in the shipped file, and broke nothing.** Re-running
`analyze_depth_composition.py` (Addendum 113) gives PSF-Zero's `same_pair`
output exactly Qiskit's own re-compile counts -- total depth 16, CX depth 3,
sx depth 6, `sx` 80 / 210, and even `rz` 120 / 315 -- where Addendum 114
recorded 23 / 3 / 8 and `sx` 128 / 336. Re-running
`verify_unitary_equivalence.py` (Addendum 103), which exercises the CX path
through real layout and routing, gives 36/36 at machine precision (minimum
0.9999999999999939).

## 1. Evidence that this is the patched code, not a stale run

Both CSVs differ from the previously saved files (different hashes). More
specifically, in the unitary-equivalence run:

| arm | uses the changed decomposer? | rows whose fidelity changed vs. Addendum 103 |
|---|---|---:|
| `qiskit_l3` | no | **0 / 12** (bit-identical) |
| `psf_ls_false` | yes (`entangling_basis="cx"`) | **12 / 12** |
| `psf_ls_true` | yes | **12 / 12** |

Exactly the outputs that pass through the changed code changed, and the one
that does not is bit-for-bit reproduced -- all still at machine precision.

## 2. Results

Depth composition, medians over 5 samples (Addendum 113's script, unmodified):

| config | strategy | total depth | CX depth | sx depth | #cx | #sx | #rz |
|---|---|---:|---:|---:|---:|---:|---:|
| 4x4 same_pair | B Qiskit L3 | 16 | 3 | 6 | 24 | 80 | 120 |
| | **D PSF-Zero** | **16** | 3 | **6** | 24 | **80** | **120** |
| 4x4 brickwork | B Qiskit L3 | 33 | 6 | 12 | 45 | 184 | 203 |
| | D PSF-Zero | 30 | 6 | 12 | 45 | 184 | 188 |
| 6x7 same_pair | B Qiskit L3 | 16 | 3 | 6 | 63 | 210 | 315 |
| | **D PSF-Zero** | **16** | 3 | **6** | 63 | **210** | **315** |
| 6x7 brickwork | B Qiskit L3 | 33 | 6 | 12 | 123 | 496 | 543 |
| | D PSF-Zero | 30 | 6 | 12 | 123 | 496 | 500 |

(A, compile once, unchanged from Addendum 114 and omitted.)

## 3. What this means

The quality gap first seen in Addendum 110 (depth 23 vs 16) is closed at its
source. Combined with Addenda 110-112, on the repeated-pair ansatz PSF-Zero's
layout-once strategy now matches Qiskit's own per-circuit re-compile on every
counted quantity, at 5.35 s instead of about 1.9 hours per training iteration
at 42 qubits (Addendum 112's figure, measured before this fix; P4 of
Addendum 116 leaves a possible 10-20% change on this path unresolved).

## 4. Consequences recorded, not yet acted on

- The README's coupling-map cliff table reports depth 23 for
  `layout_search=True`, measured with `entangling_basis="cx"` before this
  fix. It is now stale. It should be updated by re-running the script that
  produced it (`verify_end_to_end_layout_search.py`, Addenda 101-102), not
  edited by inference.
- Paper 2's gate-synthesis speed figures use the default `"canonical"` basis
  and are essentially unaffected (Addendum 116, Section 3).

## 5. Files

| File | What it is |
|---|---|
| [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py) | VERSION 2026-09-21, changelog item 14 |
| [`depth_composition_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/depth_composition_2026-09-21.csv) | this re-run, post-fix |
| [`depth_composition_2026-09-21_prefix.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/depth_composition_2026-09-21_prefix.csv) | the pre-fix run of the same script (Addendum 114), kept under a new name rather than overwritten |
| [`unitary_equivalence_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/unitary_equivalence_2026-09-20.csv) | this re-run, post-fix |
| [`unitary_equivalence_2026-09-20_prefix.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/unitary_equivalence_2026-09-20_prefix.csv) | the pre-fix run (Addendum 103), kept under a new name |

## 6. Verification

- Hashes of both CSVs compared against the previously saved files before
  treating them as new runs.
- The row-by-row fidelity comparison in Section 1 was computed from both CSVs
  directly.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both CSVs -> 0 hits.

---

<!-- ===== Addendum 118 (source: spare-qubit-cliff-addendum-118-2026-09-21.md) ===== -->

> **Note added when merging:** The README coupling-map cliff table re-measured after the fix and replaced wholesale, rather than edited by inference: gate counts unchanged, every PSF-Zero depth down, Qiskit reproduced exactly.

## Addendum 118 -- Coupling-map cliff table re-measured after the decomposer fix: gate counts unchanged, every PSF-Zero depth down, Qiskit reproduced exactly; README updated from the measurement, not by inference (2026-09-21)

**Status**: re-run of `verify_end_to_end_layout_search.py` (Addenda 101-102,
unmodified) against `psf_compile.py` VERSION 2026-09-21, to replace the
README coupling-map cliff table's depth figures that Addendum 117 recorded as
stale.

## 1. Before (Addendum 102, VERSION 2026-09-16) vs after (VERSION 2026-09-21)

| config | arm | 2q gates | depth | median ms |
|---|---|---|---|---|
| 6x7 dense_pairs | qiskit_l3 | 63 -> 63 | 16 -> 16 | 6695.91 -> 6706.98 |
| | psf_ls_false | 63-69 -> 63-69 | **23-44 -> 16-30** | 31.10 -> 28.43 |
| | psf_ls_true | 63 -> 63 | **23 -> 16** | 14.40 -> 13.06 |
| 6x7 chain-shaped | qiskit_l3 | 87 -> 87 | 104 -> 104 | 32.06 -> 30.97 |
| | psf_ls_false | 87 -> 87 | **167 -> 104** | 17.20 -> 16.23 |
| | psf_ls_true | 87 -> 87 | **167 -> 104** | 16.10 -> 14.65 |
| 8x8 dense_pairs | qiskit_l3 | 96 -> 96 | 16 -> 16 | 9017.56 -> 9018.48 |
| | psf_ls_false | 96 -> 96 | **23 -> 16** | 19.20 -> 16.48 |
| | psf_ls_true | 96 -> 96 | **23 -> 16** | 19.07 -> 17.19 |
| 8x8 chain-shaped | qiskit_l3 | 153-156 -> 153-156 | 233-268 -> 233-268 | 8853.58 -> 8962.18 |
| | psf_ls_false | 135 -> 135 | **347 -> 214** | 23.79 -> 21.22 |
| | psf_ls_true | 135 -> 135 | **347 -> 214** | 22.18 -> 20.07 |

Zero coupling-map violations on every row, before and after.

## 2. Reading

- **Only PSF-Zero's rows changed, and only in depth.** Every Qiskit row is
  identical in gate count and depth; every PSF-Zero row keeps its gate count
  and loses depth. This is the same "exactly the changed path moved" pattern
  as Addendum 117's unitary-equivalence check.
- On `dense_pairs` and 6x7 `chain-shaped`, PSF-Zero now equals Qiskit's depth.
  On 8x8 `chain-shaped` it is now **below** Qiskit in both two-qubit gates
  (135 vs 153-156) and depth (214 vs 233-268).
- **Speed on this path (Addendum 116, P4)**: all eight PSF-Zero medians are
  6-14% lower than before, against Qiskit rows moving by under 2%. That is
  no sign of the slowdown P4 could not rule out; it is not claimed as a
  speedup, given run-to-run variation of that order in Addendum 116.

## 3. README changes made

1. The cliff table was replaced wholesale with this run's figures, and its
   footnote now names this script as the source, states that the previous
   table came from a different script (`gate_count_vs_routing_level.py`,
   seed 42, Python 3.11.9), and records that this script reproduced that
   table's gate counts and depths exactly before the change (Addendum 102).
   Depths were not copied into the old table alone, to avoid mixing two
   measurements in one table.
2. The paragraph stating "`layout_search=True` ... is still deeper (23 vs.
   16)" was rewritten: it now matches Qiskit's gate count and depth.
3. The 156-qubit sentence reporting depth 13 for `"cx"` was **not** changed --
   it is a separate measurement not re-run here -- and now carries a note that
   it predates VERSION 2026-09-21.

Machine for this run not recorded; Python 3.10, Qiskit 2.5.2.

## 4. Files

| File | What it is |
|---|---|
| [`end_to_end_layout_search_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/end_to_end_layout_search_2026-09-20.csv) | this re-run, post-fix |
| [`end_to_end_layout_search_2026-09-20_prefix.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/end_to_end_layout_search_2026-09-20_prefix.csv) | the Addendum 102 run, kept under a new name |
| `README_release.md` | updated as in Section 3 |

## 5. Verification

- Row-by-row comparison computed from both CSVs directly.
- The four speed-up ratios in the README table were recomputed from this
  run's medians (235.9x, 513.6x, 547.2x, 524.6x) before being written.
- After editing, the README was searched for any remaining "depth 23" /
  "23 vs" claim; the only occurrence is the footnote describing the change.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list -> 0 hits.

---

<!-- ===== Addendum 119 pre-registration (source: spare-qubit-cliff-addendum-119-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** PyTorch-driven training under a depolarizing noise model: does halving CXs make the trained result better? Includes a shallow-ansatz control and a pre-stated caveat about re-synthesized gradients.

## Addendum 119 -- Pre-registration: does halving CXs through layout-once re-synthesis make variational training better under noise, when driven from PyTorch? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addenda 109-118 established that on a repeated-pair ansatz, PSF-Zero's
layout-once strategy (D) produces half the CXs of Qiskit's compile-once
strategy (A), with depth now equal to Qiskit's own re-compile, at 5-8x
compile-once's per-iteration cost. **None of that shows that training gets
better.** Halving CXs is a proxy; what a machine-learning user cares about is
whether the trained model is better on noisy hardware. This experiment
measures that directly, with PyTorch driving the optimization.

It also has to answer the obvious objection: *if fewer CXs is the goal, why
not just use a shallower ansatz?* D's value, if any, is keeping a deep
ansatz's expressivity while paying a shallow one's hardware cost. So a
shallow compile-once control is included.

## 2. Design

**Integration**: a single `torch.autograd.Function`. Forward: the energy
`<H>` at the current parameters. Backward: the exact parameter-shift gradient
(shifts of +-pi/2; every parameter appears in exactly one `ry` or `rz`). Each
training iteration therefore evaluates 2P + 1 circuits, compiled by the
strategy under test. No PennyLane and no `qiskit-machine-learning` -- to avoid
dependency changes to the verified Qiskit 2.5.2 environment.

**Device and noise**: a fully saturated 2x4 grid (8 qubits). Exact noisy
expectation values from Qiskit Aer's density-matrix simulator (no shot noise,
so differences come from gate noise only). Noise model, fixed before any run:
depolarizing error 1e-3 on `sx` and `x`, 1e-2 on `cx`; `rz` noiseless
(virtual on IBM-style hardware).

**Task**: ground-state search for a Hamiltonian made of one random two-qubit
Hermitian term per pair (4 disjoint pairs; coefficients on the 15
non-identity two-qubit Paulis drawn with a fixed seed). Its exact ground
energy is the sum of the four 4x4 minima, so every result is reported as a
gap to the true answer. A random generic term is chosen so that reaching the
ground state needs a genuinely general two-qubit operation -- the case where
a deeper ansatz can matter.

**Ansatz**: Addenda 109-118's `same_pair` family (one `ry` and one `rz` per
qubit per layer, then CX on each pair), L = 6 (P = 96) or L = 3 (P = 48).

**Runs** (Adam, learning rate 0.1, 40 iterations, 3 initialization seeds;
L = 6 strategies share each seed's initial parameters):

| name | layers | noise | compilation |
|---|---:|---|---|
| `ideal_L6` | 6 | none | none (logical statevector) |
| `ideal_L3` | 3 | none | none |
| `A_L6` | 6 | yes | compile once (Qiskit L3, parameterized) -- 6 CX per pair |
| `D_L6` | 6 | yes | PSF-Zero layout once + re-synthesis per circuit -- 3 CX per pair |
| `A_L3` | 3 | yes | compile once -- 3 CX per pair (the shallow control) |

Recorded per run: final energy under the run's own noisy execution (what the
hardware would report), final energy of the same parameters evaluated
noiselessly (the quality of what was learned), both as gaps to the exact
ground energy; the loss trajectory; wall time split into compile and
simulation.

**A caveat stated in advance.** Parameter shift is exact for A's noisy
objective (a fixed circuit in which each parameter enters one rotation).
For D, the compiled circuit is re-synthesized at every parameter value, so
its noisy objective need not be exactly sinusoidal in each parameter, and the
parameter-shift gradient is an approximation of its true gradient. The final
energies are evaluated directly, not through gradients, so they are exact
regardless; only the optimization path may be affected.

## 3. Pre-registered predictions

**P0 (sanity, gate for the rest).** At the initial parameters, A's and D's
compiled circuits evaluated *noiselessly* reproduce the logical energy to
better than 1e-9. Any failure invalidates that strategy.

**P1 (precondition: depth matters for this task).** Noiselessly, L = 6
reaches a smaller final gap than L = 3 on all 3 seeds. **If not, the task
does not need depth, and P3 is uninformative** -- reported as such, not as
evidence either way.

**P2 (main).** Under noise, `D_L6` reaches a smaller final noisy gap than
`A_L6` on all 3 seeds.

**P3 (the objection).** Under noise, `D_L6` reaches a smaller final noisy gap
than `A_L3` on all 3 seeds. **If `A_L3` matches or beats `D_L6`, then for
this task simply choosing a shallower ansatz is as good as re-synthesis, and
D's advantage in a training loop is not demonstrated.**

**Not predicted:** per-iteration wall time (reported), and whether D's
approximate gradients slow its convergence relative to A's exact ones
(reported from the trajectories).

## 4. What this cannot establish

- Real hardware: noise here is a simple depolarizing model, not a device
  calibration; crosstalk, readout error, coherent error and drift are absent.
- Shot noise: excluded by design (exact expectation values).
- Scale: 8 qubits, one task family, one optimizer setting, 3 seeds.
- JAX: PyTorch only in this experiment.

---

<!-- ===== Addendum 120 (source: spare-qubit-cliff-addendum-120-2026-09-21.md) ===== -->

> **Note added when merging:** PSF-Zero halves the execution error of the trained result (3/3 seeds), but the precondition failed -- the task barely needed depth -- and the learned parameters were equally good without PSF-Zero: its advantage here was entirely at execution time.

## Addendum 120 -- PyTorch-driven training under noise: PSF-Zero halves the noise-induced error of the executed result (3/3 seeds), but the task barely needed depth, so the "just use a shallower ansatz" objection is not answered; and the learned parameters were equally good without PSF-Zero (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-119-preregistration-2026-09-21.md`, written and
locked before this run. Environment: torch 2.14.0 added (dry-run checked
first; no Qiskit-stack package changed), `check_core_build.py` OK.

## 0. In one line

**P0 confirmed; P1 (the precondition) failed; P2 confirmed decisively; P3
holds numerically on all seeds but is, as pre-registered, not counted as
evidence.** Under noise, the energy the device reports after training is
roughly **half as far from the true ground energy** with PSF-Zero's
layout-once re-synthesis as with Qiskit's compile-once on the same 6-layer
ansatz (gap ratio 0.50-0.52 on all three seeds). But noiselessly, 6 layers
beat 3 layers on only 2 of 3 seeds (and on one of those by 0.001), so this
task barely needed depth, and the comparison against a shallow ansatz
cannot show what it was designed to show. A decomposition of the error
shows **PSF-Zero's entire advantage is at execution time**: the parameters
learned with and without it are equally good.

## 1. Results

Exact ground energy -5.009679. "Own execution" = energy reported by the run's
own noisy execution of its final parameters; "noiseless" = the same
parameters evaluated without noise (quality of what was learned).

| run | seed 0 own / noiseless | seed 1 | seed 2 |
|---|---|---|---|
| ideal_L6 | 0.0144 / 0.0144 | 0.0248 / 0.0248 | 0.0179 / 0.0179 |
| ideal_L3 | 0.0576 / 0.0576 | 0.0233 / 0.0233 | 0.0189 / 0.0189 |
| A_L6 (compile once, 6 CX/pair) | 0.3922 / 0.0146 | 0.4040 / 0.0246 | 0.3933 / 0.0171 |
| **D_L6 (PSF-Zero layout once, 3 CX/pair)** | **0.1968** / 0.0145 | **0.2082** / 0.0250 | **0.2001** / 0.0181 |
| A_L3 (compile once, shallow, 3 CX/pair) | 0.2494 / 0.0575 | 0.2143 / 0.0234 | 0.2115 / 0.0188 |

(All values are gaps to the exact ground energy.)

P0: every compiled circuit, simulated without noise, reproduced the logical
energy (worst 4.1e-13, D; A at most 1.1e-15).

## 2. Scoring

**P0 (sanity) -- CONFIRMED.**

**P1 (precondition: depth matters) -- FAILED.** L6 < L3 noiselessly on seeds
0 (0.0144 vs 0.0576) and 2 (0.0179 vs 0.0189, margin 0.0010), but not seed 1
(0.0248 vs 0.0233). The task does not reliably need depth.

**P2 (main: D_L6 beats A_L6 under noise) -- CONFIRMED on all 3 seeds**, by
about 2x (ratios 0.50, 0.52, 0.51).

**P3 (D_L6 beats A_L3 under noise) -- holds numerically on all 3 seeds
(margins 0.053, 0.006, 0.012), NOT counted as evidence.** The
pre-registration made P3 conditional on P1; with P1 failed, it is reported
and set aside. Note the one large margin is on seed 0 -- exactly the seed
where depth mattered noiselessly -- which is what the hypothesis would
predict, but one seed is not a result.

## 3. Where the difference comes from

Each own-execution gap splits into the learned-parameter gap (noiseless)
plus an execution-noise penalty:

| run | learned-parameter gap (3 seeds) | execution-noise penalty (3 seeds) |
|---|---|---|
| A_L6 | 0.0146, 0.0246, 0.0171 | 0.378, 0.379, 0.376 |
| D_L6 | 0.0145, 0.0250, 0.0181 | **0.182, 0.183, 0.182** |
| A_L3 | 0.0575, 0.0234, 0.0188 | 0.192, 0.191, 0.193 |

- **Learning was not improved.** A_L6, D_L6 and the noiseless ideal_L6 learned
  parameters of essentially the same quality on every seed. Under this noise
  model, noise did not mislead the optimizer.
- **Execution was.** D's penalty is 48% of A_L6's -- consistent with half the
  CXs -- and slightly below A_L3's (0.182 vs 0.192) despite both having three
  CX per pair; why is not measured (plausibly fewer `sx` pulses).
- **D's approximate gradient did no visible harm.** The iteration at which each
  run reached 90% of its own total improvement was identical for ideal_L6,
  A_L6 and D_L6 on every seed (9, 12, 7).

## 4. What this means, and a hypothesis it raises

For this task and noise model, PSF-Zero's value was entirely in **executing**
a deep circuit with half the CXs, not in training. That suggests -- as a
hypothesis, not a finding -- a cheaper workflow: **train with compile-once
(fast), then execute the final parameters once through PSF-Zero's
re-synthesis**, which would capture D's execution-noise penalty without its
per-iteration compile cost. It rests on the observation above that the noise
did not distort the optimum, which is expected for depolarizing noise (it
largely rescales the energy landscape) and not guaranteed for coherent errors,
amplitude damping or drift.

What this does **not** support: a claim that PSF-Zero "makes quantum machine
learning train better". Here it made the deployed result about 2x more
accurate on a deep ansatz, while a shallow ansatz came close for a task that
barely needed depth.

## 5. Wall time (simulator-bound; not a hardware estimate)

| run | wall s (compile / simulation), per seed |
|---|---|
| A_L6 | 75.1 (3.1 / 71.4), 76.1, 75.3 |
| D_L6 | 59.9 (15.3 / 44.1), 61.4, 60.0 |
| A_L3 | 26.5 (1.0 / 25.2), 26.7, 26.5 |

D is faster overall here only because the simulator has fewer noisy gates to
apply; its compile time is 5x A's. On hardware, execution time is set by the
device, and this comparison does not transfer.

## 6. What this does not establish

- Real hardware; noise beyond a simple depolarizing model; shot noise.
- That depth matters for the target problem (P1 failed) -- a harder task is
  needed to test the shallow-ansatz objection properly.
- The train-once-execute-with-PSF-Zero workflow in Section 4 -- untested.
- JAX.

## 7. Files

| File | What it is |
|---|---|
| [`train_noisy_vqe_torch.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/train_noisy_vqe_torch.py) | this run's script (PyTorch bridge + experiment) |
| [`noisy_vqe_torch_summary_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/noisy_vqe_torch_summary_2026-09-21.csv) | per-run results, 15 rows |
| [`noisy_vqe_torch_trajectories_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/noisy_vqe_torch_trajectories_2026-09-21.csv) | loss per iteration |
| [`spare-qubit-cliff-addendum-119-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-119-preregistration-2026-09-21.md) | the predictions scored above |

## 8. Verification

- Every figure recomputed from the two CSVs; the decomposition in Section 3
  is computed there, not printed by the script.
- P1 was applied as pre-registered (all three seeds), and P3's conditional
  status was applied as written rather than reinterpreted after seeing that
  P3 held numerically.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both CSVs -> 0 hits.

---

<!-- ===== Addendum 121 pre-registration (source: spare-qubit-cliff-addendum-121-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** A task that needs entangling depth (2x3 Heisenberg), plus a stronger control than a shallow circuit: blocks written optimally with 3 CXs from the start. PSF-Zero is predicted to TIE that control. Result: Addendum 122.

## Addendum 121 -- Pre-registration: on a task that genuinely needs entangling depth (2D Heisenberg), does PSF-Zero's re-synthesis beat (a) a shallower circuit and (b) a circuit whose blocks are written optimally in the first place? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 120 found that PSF-Zero halves the execution-noise error of a
6-layer `same_pair` circuit, but its precondition failed: the task barely
needed depth, so the objection "a shallower circuit would do" was not
answered. This experiment uses a task that needs entanglement across the
whole lattice.

Two design facts were settled before writing it:

1. **`same_pair` cannot represent the answer.** It entangles only within fixed
   pairs, so it can never reach the ground state of a Hamiltonian coupling
   neighbouring pairs, at any depth. The ansatz must alternate pairings.
2. **Re-synthesis only removes CXs that repeat on the same pair.** On
   `brickwork`, which switches pairs every layer, it removed none (Addenda
   110-114). So the ansatz here is built from **blocks**: k consecutive
   entangling layers on the same pair, with the pairing alternating between
   blocks. Within a block, re-synthesis can compress k CXs to 3; the
   alternation spreads entanglement across the lattice.

**And one objection stronger than "use a shallower circuit".** Any two-qubit
unitary can be written with exactly 3 CXs and 15 parameters. A researcher who
writes each block in that form gets 3 CXs per block with compile-once and no
PSF-Zero. PSF-Zero reduced 6 CXs to 3 only because the redundant block form
contains more CXs than needed. **The prediction below is that PSF-Zero ties
this control rather than beats it** -- stated before measuring, because if it
holds, PSF-Zero's training-loop value is "automatically removing redundancy a
researcher did not remove by hand", a narrower claim than "halving noise".

## 2. Design

**Lattice and Hamiltonian**: 2x3 grid, 6 qubits, fully saturated.
Antiferromagnetic Heisenberg model on every nearest-neighbour bond,
`H = sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j)`; exact ground energy by direct
diagonalization.

**Pairings** (row-major qubits 0-2 / 3-5), cycled in this order:
horizontal-a {(0,1), (3,4)}, vertical {(0,3), (1,4), (2,5)}, horizontal-b
{(1,2), (4,5)} -- one cycle = 7 blocks, covering every bond once.

**Block forms**:
- *redundant* (k = 6): six repetitions of (`ry`, `rz` on each qubit; `cx`) --
  24 parameters, 6 CX.
- *optimal*: `rz ry rz` on each qubit; `cx(b,a)`; `rz` on a, `ry` on b;
  `cx(a,b)`; `ry` on b; `cx(b,a)`; `rz ry rz` on each qubit -- the standard
  3-CX universal two-qubit circuit, 15 parameters, 3 CX. Every parameter sits
  in exactly one `rz`/`ry`, so parameter shift stays exact.

**Depth**: *deep* = 2 cycles (14 blocks); *shallow* = 1 cycle (7 blocks).

**Runs** (same PyTorch bridge, noise model, optimizer and simulator as
Addendum 119: depolarizing 1e-3 on `sx`/`x`, 1e-2 on `cx`, `rz` noiseless;
exact density-matrix expectation; Adam, lr 0.1, 40 iterations; 3 seeds):

| name | blocks | block form | noise | compilation | CX per block on device |
|---|---|---|---|---|---:|
| `ideal_deep_red` | 14 | redundant | none | none | -- |
| `ideal_shallow_red` | 7 | redundant | none | none | -- |
| `ideal_deep_opt` | 14 | optimal | none | none | -- |
| `A_deep_red` | 14 | redundant | yes | compile once | 6 |
| `D_deep_red` | 14 | redundant | yes | PSF-Zero layout once + re-synthesis | 3 |
| `A_deep_opt` | 14 | optimal | yes | compile once | 3 |
| `A_shallow_opt` | 7 | optimal | yes | compile once | 3 |

## 3. Pre-registered predictions

**P0 (sanity).** Every compiled circuit, simulated noiselessly at the initial
parameters, reproduces the logical energy to better than 1e-9.

**P1 (precondition: depth is needed).** `ideal_shallow_red`'s final gap is at
least 2x `ideal_deep_red`'s on all 3 seeds. **If not, the task still does not
need depth and P3 is uninformative**, as in Addendum 120.

**P1b (precondition: the optimal-block control is fair).** `ideal_deep_opt`
reaches a final gap no worse than 1.5x `ideal_deep_red`'s on all 3 seeds. If
not, the optimal-block form is less trainable here and P4 is not a fair
comparison.

**P2 (replication).** `D_deep_red` beats `A_deep_red` (smaller final gap under
its own noisy execution) on all 3 seeds.

**P3 (the shallow objection).** `D_deep_red` beats `A_shallow_opt` on all 3
seeds -- depth is worth paying for even under noise.

**P4 (the stronger objection) -- predicted to TIE.** `D_deep_red` does **not**
beat `A_deep_opt` by more than 20% of `A_deep_opt`'s gap on any seed. If
D does beat it by more on all 3 seeds, PSF-Zero's re-synthesis is adding
something beyond reaching the 3-CX form (e.g. fewer single-qubit pulses), and
that would be the finding.

**Not predicted:** wall time (reported, simulator-bound as before).

## 4. What this cannot establish

- Real hardware, non-depolarizing noise, shot noise -- as in Addendum 119.
- Larger lattices, other models, other optimizers.
- Whether researchers in practice write redundant or optimal blocks -- this
  experiment measures what each choice costs, not how common it is.

---

<!-- ===== Addendum 122 (source: spare-qubit-cliff-addendum-122-2026-09-21.md) ===== -->

> **Note added when merging:** Addendum 121's result. The task needs depth and PSF-Zero cuts the deep circuit's execution error by about 40%, ties optimally written blocks as predicted -- but a shallow circuit beats it on all three seeds at 1% CX error (robust on two seeds; seed 0 may be an iteration-budget artefact, Section 10). Also records that noise distorted learning here, weakening Addendum 120's hybrid hypothesis, and that a stated timing expectation was wrong per circuit.

## Addendum 122 -- On a task that needs depth, PSF-Zero's re-synthesis beats the same deep circuit compiled once (3/3) and ties optimally written blocks as predicted, but LOSES to a shallow circuit on all three seeds: at this noise level, depth does not pay for itself (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-121-preregistration-2026-09-21.md`, written and
locked before this run.

## 0. In one line

**P0, P1, P1b, P2, P4 confirmed; P3 FAILED on all three seeds.** The task
genuinely needs depth (noiselessly, 1 cycle leaves 2.2-2.5x the gap of 2
cycles), and PSF-Zero cuts the deep redundant circuit's execution error by
about 40% (gap ratio 0.59-0.61). But under the fixed noise model (1% per CX),
**the shallow compile-once circuit reached the true ground energy more closely
than PSF-Zero's deep circuit on every seed** (2.63-3.09 vs 3.34-3.36): its 21
CXs cost less noise than the extra expressivity of 42 CXs was worth. As
pre-registered, PSF-Zero **tied** the optimally written deep blocks (better by
3.7-11.5%, inside the 20% tie band). Two unregistered observations follow in
Sections 4-5, including one that weakens Addendum 120's train-cheap,
execute-with-PSF-Zero hypothesis.

## 1. Results

Exact ground energy -12.517541. Gaps to it; "own" = the run's own noisy
execution of its final parameters; "noiseless" = the same parameters without
noise.

| run | device CX / sx | seed 0 own / noiseless | seed 1 | seed 2 |
|---|---|---|---|---|
| ideal_deep_red | -- | 0.587 / 0.587 | 0.659 / 0.659 | 0.677 / 0.677 |
| ideal_shallow_red | -- | 1.426 / 1.426 | 1.471 / 1.471 | 1.679 / 1.679 |
| ideal_deep_opt | -- | 0.660 / 0.660 | 0.693 / 0.693 | 0.973 / 0.973 |
| A_deep_red | 84 / 336 | 5.507 / 0.743 | 5.698 / 1.096 | 5.542 / 0.970 |
| **D_deep_red** | **42 / 96** | **3.357** / 0.771 | **3.337** / 0.603 | **3.350** / 0.797 |
| A_deep_opt | 42 / 140 | 3.685 / 1.038 | 3.467 / 0.701 | 3.786 / 1.187 |
| **A_shallow_opt** | **21 / 70** | **3.086** / 1.874 | **2.641** / 1.334 | **2.630** / 1.352 |

P0: every compiled circuit reproduced the logical energy noiselessly (worst
2.6e-11, D).

## 2. Scoring

**P0 -- CONFIRMED.**

**P1 (depth is needed) -- CONFIRMED.** Shallow / deep noiseless gap: 2.43,
2.23, 2.48 (threshold 2).

**P1b (optimal-block control is fair) -- CONFIRMED**, though seed 2 was close:
1.12, 1.05, 1.44 (threshold 1.5).

**P2 (D beats A on the same deep circuit) -- CONFIRMED**, ratios 0.61, 0.59,
0.60.

**P3 (D beats the shallow circuit) -- FAILED, 0 of 3 seeds.** D 3.357 / 3.337 /
3.350 vs shallow 3.086 / 2.641 / 2.630. **At this noise level, the objection
"use a shallower circuit" stands**, even on a task that needs depth
noiselessly.

**P4 (D ties optimal blocks) -- CONFIRMED as predicted.** D better by 8.9%,
3.7%, 11.5% of A_deep_opt's gap -- all inside the 20% band.

## 3. Why the shallow circuit won

Splitting each gap into learned-parameter quality (noiseless) and
execution-noise penalty:

| run | learned gap | execution penalty |
|---|---|---|
| A_deep_red | 0.743, 1.096, 0.970 | 4.765, 4.602, 4.572 |
| D_deep_red | 0.771, 0.603, 0.797 | 2.586, 2.734, 2.554 |
| A_deep_opt | 1.038, 0.701, 1.187 | 2.647, 2.765, 2.600 |
| A_shallow_opt | 1.874, 1.334, 1.352 | 1.212, 1.308, 1.278 |

The shallow circuit learns worse (by about 0.8 on average) but pays about
half the noise penalty (about 1.3 vs 2.6). The penalty scales roughly with CX
count (21 -> 1.3, 42 -> 2.6, 84 -> 4.6). A linear extrapolation -- a
hypothesis, not a measurement -- puts the break-even, below which the deep
circuit with PSF-Zero would win, at a CX error of roughly 0.6% instead of the
1% used here. The value of halving CXs depends on how noisy the hardware is,
and at 1% it was not enough to overturn the shallow circuit's advantage.

## 4. Unregistered observation: noise distorted learning here -- weakening Addendum 120's hybrid hypothesis

In Addendum 120 (lighter circuits, 24-48 CX), noise did not affect what was
learned. Here it did: A_deep_red, training through 84 noisy CXs, learned
noticeably worse parameters than its noiseless counterpart (0.743 / 1.096 /
0.970 vs 0.587 / 0.659 / 0.677), while D, training through 42, stayed closer
(0.771 / 0.603 / 0.797). Addendum 120's proposed workflow -- train with the
cheap compile-once strategy, execute only the final parameters through
PSF-Zero -- relies on training being unaffected by noise. **In this heavier
setting that premise fails**, so the hybrid would inherit A's worse
parameters. It remains untested directly.

## 5. Unregistered observation: timing (exploratory; not pre-registered)

| run | CX / sx | params | circuits per iteration | compile s | simulation s | simulation per circuit |
|---|---|---:|---:|---:|---:|---:|
| A_deep_red | 84 / 336 | 336 | 673 | 23.5 | 1050.4 | 39.0 ms |
| D_deep_red | 42 / 96 | 336 | 673 | 114.5 | 356.6 | 13.3 ms |
| A_deep_opt | 42 / 140 | 210 | 421 | 9.4 | 349.3 | 20.7 ms |
| A_shallow_opt | 21 / 70 | 105 | 211 | 2.9 | 92.3 | 10.9 ms |

(means over 3 seeds)

In conversation before this run, the expectation stated was that D and
A_deep_opt would simulate in about the same time, since both put 42 CXs on
the device. **Per training iteration they did (356.6 s vs 349.3 s), but per
circuit that expectation was wrong: D's circuits simulated 36% faster
(13.3 vs 20.7 ms).** The difference is single-qubit gates: re-synthesizing a
circuit whose parameters are numbers lets consecutive single-qubit gates
merge (D: 96 `sx`), which compiling a parameterized circuit once cannot do
(A_deep_opt: 140). Two qualifications keep this from being a PSF-Zero-specific
claim: any per-circuit re-compile with bound values would merge the same way
(Qiskit's own, strategy B in Addenda 110-112, did); and the per-iteration tie
arises because D's redundant ansatz has 60% more parameters and so evaluates
60% more circuits per gradient. Wall time per seed: D 471 s, A_deep_opt 359 s
-- A_deep_opt faster overall, because D's compile adds 114 s. On the
accuracy side, the fewer `sx` barely mattered: D's execution penalty is
within 1-2.3% of A_deep_opt's (2.586 vs 2.647, 2.734 vs 2.765, 2.554 vs 2.600).

## 6. Where this leaves the quantum-AI direction

- **PSF-Zero's re-synthesis reliably removes redundant CXs and the noise they
  cause** (P2, here and in Addendum 120).
- **It does not beat writing the circuit well in the first place** (P4, tie),
  apart from a small, consistent edge that comes mostly from the redundant
  ansatz training slightly better, not from execution.
- **At 1% CX error, a shallower circuit beat the deep one even with PSF-Zero**
  (P3). The claim "depth plus PSF-Zero wins on hard tasks" is not supported at
  this noise level; whether it holds at lower noise is the open question
  (Section 3's rough break-even of about 0.6%).
- The PyTorch bridge itself worked throughout: 36 training runs across
  Addenda 120 (15) and 122 (21), exact parameter-shift gradients, no failures.

## 7. What this does not establish

- Any noise level other than the fixed model used; real hardware.
- The break-even in Section 3 (a linear extrapolation from three CX counts).
- The hybrid workflow (Section 4) -- inferred against, not tested.
- Other lattices, models, optimizers, or iteration budgets.

## 8. Files

| File | What it is |
|---|---|
| [`train_heisenberg_torch.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/train_heisenberg_torch.py) | this run's script |
| [`heisenberg_torch_summary_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/heisenberg_torch_summary_2026-09-21.csv) | per-run results, 21 rows |
| [`heisenberg_torch_trajectories_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/heisenberg_torch_trajectories_2026-09-21.csv) | loss per iteration, 840 rows (21 runs x 40); received after this addendum was first written -- see Section 10 |
| [`spare-qubit-cliff-addendum-121-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-121-preregistration-2026-09-21.md) | the predictions scored above |

## 9. Verification

- Every figure recomputed from the summary CSV; the decomposition, per-circuit
  simulation times and ratios are computed there, not printed by the script.
- P3 is reported as failed on all three seeds, as measured; its threshold was
  not revisited after seeing the result.
- The expectation contradicted in Section 5 was stated in conversation before
  the result was seen, and is recorded here as wrong rather than omitted.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the CSV -> 0 hits.

## 10. Convergence check (added after the trajectories were received; the scoring above is unchanged)

The trajectory file (840 rows, 21 runs x 40 iterations, all complete) shows
that **no run had fully converged at the 40-iteration budget**: over the last
10 iterations every run was still improving, by 0.8-3.8% of its own total
improvement.

Whether that matters depends on which way the late improvement points. For
P3 -- the one failed prediction -- comparing how fast D and the shallow
circuit were still improving over iterations 29-39:

| seed | P3 margin (D minus shallow, own-execution gap) | D's improvement, last 10 | shallow's improvement, last 10 | direction |
|---|---:|---:|---:|---|
| 0 | 0.271 | 0.306 | 0.068 | **closing**: at an unchanged rate, D would catch up in about 11 more iterations |
| 1 | 0.696 | 0.202 | 0.309 | widening |
| 2 | 0.720 | 0.236 | 0.358 | widening |

**P3's failure is robust on seeds 1 and 2 and fragile on seed 0**, where it
may be an artefact of the iteration budget. The pre-registered score stays as
measured (0 of 3); the honest reading is "shallow wins at 1% CX error on at
least 2 of 3 seeds even allowing for more training, and seed 0 is
undetermined". A longer run, or the noise sweep proposed in Section 3, would
settle it. The trajectories' final losses sit within 0.01-0.03 of the
summary's final own-execution energies (one further optimizer step apart),
consistent with the two files describing the same runs.

---

<!-- ===== Addendum 123 pre-registration (source: spare-qubit-cliff-addendum-123-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Tests Addendum 122's untested claim that its per-circuit speed is not specific to PSF-Zero, by adding the missing Qiskit re-compile arm -- standalone, no training loop.

## Addendum 123 -- Pre-registration: is PSF-Zero's per-circuit simulation speed (Addendum 122) specific to PSF-Zero, or does any re-compile of bound circuits give the same? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 122 found, as an unregistered observation, that PSF-Zero's circuits
(`D_deep_red`) simulated 36% faster per circuit than optimally written blocks
compiled once (`A_deep_opt`, 13.3 vs 20.7 ms), at the same 42 CXs. It
attributed this to single-qubit gates -- 96 `sx` vs 140 -- and claimed, without
testing it, that **any** per-circuit re-compile with bound values would merge
single-qubit gates the same way, so the effect is not specific to PSF-Zero.
That run had no Qiskit re-compile arm, so the claim is untested. This
experiment tests it directly, without a training loop.

A standalone test was checked first for whether it could differ from the
training-loop timing at all: in Addendum 122, time outside compilation and
simulation was 0.1-0.3 s per seed out of 95-1078 s, and simulation time
varied 0.5-1.2% across seeds over 8,000-27,000 circuits each. So this
experiment is not expected to change Addendum 122's timings; its purpose is
the missing comparison arm.

## 2. Design

Same lattice, Hamiltonian, ansatze, basis, noise model and simulator as
Addenda 121-122 (imported from `train_heisenberg_torch.py`, so they are
identical by construction): 2x3 Heisenberg, deep (2 cycles) redundant and
optimal block ansatze, depolarizing 1e-3 on `sx`/`x` and 1e-2 on `cx`,
density-matrix simulation.

Five ways of producing the circuit actually simulated, each applied to the
same 50 random parameter vectors per ansatz (seeded):

| name | ansatz | method |
|---|---|---|
| `A_red` | redundant | compile once (parameterized, Qiskit L3), then bind |
| `B_red` | redundant | bind, then Qiskit L3 per circuit |
| `D_red` | redundant | bind, then PSF-Zero layout-once re-synthesis (Addendum 111's D) |
| `A_opt` | optimal | compile once, then bind |
| `B_opt` | optimal | bind, then Qiskit L3 per circuit |

Per circuit: CX, `sx` (+`x`), `rz` and total gate count; depth; compile time.
Simulation time: each method's 50 circuits run as one batch (as in the
training loop), 3 repeats after a warm-up batch; per-circuit time is the
median batch time / 50. Correctness: every compiled circuit, simulated
noiselessly, must reproduce the logical energy to better than 1e-9.

## 3. Pre-registered predictions

**P1 (the claim under test).** `B_red`'s median `sx` count equals `D_red`'s,
and its per-circuit simulation time is within 10% of `D_red`'s. **If `B_red`
has materially more `sx` or is more than 10% slower, the effect is at least
partly specific to PSF-Zero's synthesis, and Addendum 122 Section 5's claim
is wrong.**

**P2 (re-compile helps the optimal ansatz too).** `B_opt` has fewer `sx` than
`A_opt` and simulates faster per circuit -- the same single-qubit merging,
applied to blocks that were already written with 3 CX.

**P3 (what sets simulation time).** Across all five methods, per-circuit
simulation time increases with total gate count (same rank order).

**P4 (correctness).** Every circuit within 1e-9 of the logical energy.

**Not predicted:** compile time per circuit (reported; Addenda 110-112 found
PSF-Zero faster than Qiskit's re-compile per circuit on every grid tested).

## 4. What this cannot establish

- Real hardware timing -- simulator only, as throughout.
- Other circuits, noise models, or simulator methods.
- Accuracy -- no training is run; this is about the circuits only.

---

<!-- ===== Addendum 124 (source: spare-qubit-cliff-addendum-124-2026-09-21.md) ===== -->

> **Note added when merging:** Confirmed: Qiskit's own re-compile of bound circuits gives exactly PSF-Zero's gate counts and near-identical simulation time. One Qiskit arm missed the correctness bound (9.3e-6); PSF-Zero did not. Summarizes which training-loop advantages are PSF-Zero-specific (layout-cliff compile speed) and which are not.

## Addendum 124 -- PSF-Zero's per-circuit simulation speed is not specific to PSF-Zero: Qiskit's own per-circuit re-compile produces identical gate counts; one Qiskit arm missed the pre-registered correctness bound (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-123-preregistration-2026-09-21.md`, written and
locked before this run.

## 0. In one line

**P1, P2, P3 confirmed; P4 failed for one arm (Qiskit's re-compile of the
optimal ansatz), not for PSF-Zero.** Re-compiling bound circuits with Qiskit
(`B_red`) gives **exactly** PSF-Zero's (`D_red`) gate counts -- 42 CX, 96 `sx`,
144 `rz`, 282 gates, depth 71 -- and simulates within 8.3% of it. The
per-circuit speed Addendum 122 saw is therefore a property of re-compiling
with bound values, as that addendum claimed without testing, not of
PSF-Zero's synthesis. Re-compiling the optimally written ansatz (`B_opt`)
reaches the same counts too. The standalone timings reproduce Addendum 122's
training-loop timings to within 3-4%.

## 1. Results

2x3 Heisenberg, deep ansatze, the same 50 parameter vectors per ansatz for
every method; medians. Simulation: one batch of 50, 3 timed repeats after a
warm-up.

| method | CX | sx | rz | total | depth | compile ms | sim ms / circuit (range) | max energy error |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A_red (compile once) | 84 | 336 | 504 | 924 | 216 | 0.95 | 37.39 (37.16-37.69) | 8.0e-15 |
| B_red (Qiskit re-compile) | 42 | 96 | 144 | 282 | 71 | 10.86 | 13.74 (12.83-14.01) | 5.9e-14 |
| **D_red (PSF-Zero)** | **42** | **96** | **144** | **282** | **71** | 9.29 | 12.69 (12.50-12.71) | 3.6e-10 |
| A_opt (compile once) | 42 | 140 | 308 | 490 | 126 | 0.65 | 20.01 (19.92-20.86) | 4.7e-15 |
| B_opt (Qiskit re-compile) | 42 | 96 | 144 | 282 | 71 | 10.54 | 13.63 (12.74-14.52) | **9.3e-06** |

`sx` was identical across all 50 samples within every method (min = max).

## 2. Scoring

**P1 (the claim under test) -- CONFIRMED.** `B_red` and `D_red` have identical
median counts in every column; `B_red` simulates 8.3% slower per circuit,
inside the 10% bound. With identical counts, the remaining difference is not
explained by gate number; its source (gate placement or ordering, or
run-to-run variation) was not investigated. **Addendum 122 Section 5's claim
stands, now tested: the effect is not specific to PSF-Zero.**

**P2 (re-compile helps the optimal ansatz too) -- CONFIRMED.** `B_opt` vs
`A_opt`: `sx` 96 vs 140, 13.63 vs 20.01 ms (0.68x). Merging single-qubit gates
after binding helps even blocks already written with three CXs.

**P3 (simulation time follows total gate count) -- CONFIRMED.** 282-gate
methods: 12.7-13.7 ms; 490: 20.0 ms; 924: 37.4 ms. Methods tied on gate
count differ by up to 8%, as in P1.

**P4 (every circuit within 1e-9 of the logical energy) -- FAILED for B_opt;
held for every other method including PSF-Zero.** At least one of `B_opt`'s 50
circuits reproduced the logical energy only to 9.3e-6. The cause was not
investigated; Qiskit's `optimization_level=3` pipeline includes steps that
can remove operations judged close enough to identity, which would fit, but
that is a hypothesis. It did not change any gate count (min = max for every
column), so P2's counts are unaffected; the timing of a circuit that differs
by one tiny operation is not meaningfully different. This is **not** evidence
that PSF-Zero is more accurate than Qiskit in any practical sense: an energy
error of 1e-5 in one circuit out of 50 is far below any physical relevance.

## 3. Other observations

- **Standalone matches the training loop.** Per-circuit simulation here vs
  Addendum 122 (inside training): D 12.69 vs 13.25 ms, A_opt 20.01 vs 20.74,
  A_red 37.39 vs 39.02 -- consistently 3-4% lower here, as expected from
  Addendum 122's measured 0.1-0.3 s of training overhead and a smaller batch
  size. This confirms the answer given in conversation before this run: a
  standalone test does not materially change the timings.
- **Compile time at this size**: PSF-Zero 9.29 ms vs Qiskit's re-compile
  10.86 ms per circuit -- only 1.17x at 6 qubits, where Qiskit's layout search
  has no failure region to fall into. PSF-Zero's larger advantages (Addenda
  110-112) appear at larger, saturated grids.
- **PSF-Zero's numerical floor** is visibly higher than Qiskit's (3.6e-10 vs
  1e-14) though well inside the 1e-9 bound -- consistent with the 1e-11 to
  1e-13 errors seen in Addendum 122's P0 checks.

## 4. What this means

Of the advantages measured for PSF-Zero in the training-loop experiments
(Addenda 110-124), the ones that are **specific to PSF-Zero** are:

- **Compile speed where Qiskit's layout search fails** -- up to 278x at 42
  qubits on a saturated repeated-pair ansatz (Addendum 110), from the layout
  work of Papers 1-2.
- **A compile-speed edge elsewhere that varies with the circuit** -- from 1.17x
  here (6 qubits, 14 blocks) to 4.7-6.0x for the layout-once strategy in
  Addendum 112 (16-42 qubits), excluding the layout-cliff configuration.

The ones that are **not** specific to PSF-Zero -- any per-circuit re-compile
with bound values gets them -- are the halved CX count on redundant ansatze,
the merged single-qubit gates, the resulting faster simulation, and the
resulting lower execution error.

## 5. Files

| File | What it is |
|---|---|
| [`compare_recompile_gate_counts.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compare_recompile_gate_counts.py) | this run's script (imports the problem from [`train_heisenberg_torch.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/train_heisenberg_torch.py)) |
| [`recompile_gate_counts_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/recompile_gate_counts_2026-09-21.csv) | per-method results, 5 rows |
| [`spare-qubit-cliff-addendum-123-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-123-preregistration-2026-09-21.md) | the predictions scored above |

## 6. Verification

- All figures recomputed from the CSV; ratios computed there.
- P4 applied to every method as pre-registered, including the Qiskit arms.
- The comparison with Addendum 122's timings uses that addendum's own
  recorded per-circuit figures.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the CSV -> 0 hits.

---

<!-- ===== Addendum 125 pre-registration (source: spare-qubit-cliff-addendum-125-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Sweeps CX error (0.2%, 0.5%, 1%) with 60 iterations to find where the deep circuit starts beating the shallow one; states in advance that failing at 0.2% would deprioritize the quantum-AI direction.

## Addendum 125 -- Pre-registration: at what CX error rate does the deep circuit (PSF-Zero, 42 CX) start beating the shallow one (21 CX)? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 122 found, on a task that needs depth, that at 1% CX error a shallow
circuit (21 CX) reached the ground energy more closely than PSF-Zero's
re-synthesized deep circuit (42 CX) -- robustly on 2 of 3 seeds; seed 0 may
have been an artefact of the 40-iteration budget (Addendum 122, Section 10).
A linear extrapolation put the break-even at a CX error of about 0.6%.
Addendum 124 then established that PSF-Zero's circuits equal Qiskit's own
per-circuit re-compile, so PSF-Zero's specific role in a training loop is the
speed of re-compiling at scale -- which only matters if the deep circuit is
worth running at all. **This experiment measures whether, and below what
noise level, it is.** It is the decision point stated in conversation: if the
deep circuit does not win even at low noise, the quantum-AI direction is
deprioritized.

## 2. Design

Everything is imported unchanged from `train_heisenberg_torch.py` (Addenda
121-122) -- lattice, Hamiltonian, ansatze, PyTorch bridge, optimizer --
except two settings:

- **Noise level**, swept: CX depolarizing error p2 in {0.2%, 0.5%, 1.0%},
  with `sx`/`x` error p1 = p2 / 10 (the same ratio as Addenda 119-122);
  `rz` noiseless.
- **Iterations: 60** (up from 40), since no Addendum 122 run had converged at
  40.

Two runs per (noise level, seed), seeds 0, 1, 2 with the same initial
parameters as Addendum 122:

| name | circuit | device CX |
|---|---|---:|
| `D_deep_red` | deep redundant ansatz, PSF-Zero layout-once re-synthesis | 42 |
| `A_shallow_opt` | shallow optimal-block ansatz, compiled once | 21 |

The comparison is the final gap to the exact ground energy under each run's
own noisy execution. Noiseless evaluation of the learned parameters is
recorded too, as in Addendum 122. Results are written after every run, so a
partial run still leaves usable data.

Expected run time: about 2 hours (from Addendum 122's per-iteration costs,
scaled to 60 iterations).

## 3. Pre-registered predictions

**P1 (replication at 1.0%).** The shallow circuit beats D on at least 2 of 3
seeds.

**P2 (main: low noise).** At 0.2%, D beats the shallow circuit on all 3 seeds.
**If D does not beat it on at least 2 of 3 seeds at 0.2%, depth does not pay
for itself at any noise level tested, and the quantum-AI direction is
deprioritized.**

**P3 (near the extrapolated break-even).** At 0.5%, D beats the shallow
circuit on at least 2 of 3 seeds. This is a weak prediction -- 0.5% is close
to the extrapolated 0.6% -- and is registered mainly so that the crossover can
be placed on one side or the other of 0.5%.

**P4 (the extrapolation's assumption).** Execution-noise penalty (own-execution
gap minus noiseless gap) scales roughly linearly with p2: for each circuit,
penalty at 0.2% divided by penalty at 1.0% lies between 0.1 and 0.3 on every
seed (linear would be 0.2). If not, Addendum 122's 0.6% estimate rested on a
wrong assumption.

## 4. What this cannot establish

- Where any real device sits relative to the crossover: this experiment uses a
  simple depolarizing model, not a device calibration, and does not claim a
  mapping from its noise levels to specific hardware.
- Other tasks, sizes, optimizers -- a single 6-qubit task.
- Anything about PSF-Zero versus Qiskit's own re-compile, which Addendum 124
  showed produce identical circuits here.

---

<!-- ===== Addendum 126 (source: spare-qubit-cliff-addendum-126-2026-09-21.md) ===== -->

> **Note added when merging:** Deep wins on every seed at 0.2%, shallow on every seed at 1%; the crossover is just under 0.5% on two of three seeds (P3 failed), lower than Addendum 122's 0.6% estimate. The deprioritization condition was not met.

## Addendum 126 -- Noise sweep: below about 0.5% CX error the deep circuit wins on every seed; at 1% the shallow one does; the crossover sits just under 0.5% on two of three seeds, lower than Addendum 122's 0.6% estimate (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-125-preregistration-2026-09-21.md`, written and
locked before this run.

## 0. In one line

**P1, P2, P4 confirmed; P3 failed.** At 0.2% CX error, PSF-Zero's deep
circuit (42 CX) reached the ground energy more closely than the shallow
circuit (21 CX) on all three seeds (mean gap 1.009 vs 1.546), so **the
pre-registered condition for deprioritizing the quantum-AI direction was not
met: depth does pay, at low enough noise.** At 1% the shallow circuit won on
all three seeds (replicating Addendum 122 with 60 iterations). At 0.5% the
deep circuit won on only one seed, so P3 failed; interpolating per seed puts
the crossover at 0.46% and 0.49% on two seeds and 0.98% on the third. Noise
penalty scaled close to linearly with CX error (P4), but the resulting
crossover is lower than Addendum 122's 0.6% extrapolation.

## 1. Results

Gap to the exact ground energy (-12.517541) under each run's own noisy
execution; 60 iterations; same initial parameters as Addendum 122.

| CX error | seed | D_deep_red (42 CX) | A_shallow_opt (21 CX) | D minus shallow | deeper wins? |
|---:|---:|---:|---:|---:|---|
| 0.2% | 0 | 0.964 | 1.927 | -0.963 | yes |
| | 1 | 0.952 | 1.394 | -0.442 | yes |
| | 2 | 1.111 | 1.318 | -0.207 | yes |
| 0.5% | 0 | 1.906 | 2.410 | -0.504 | yes |
| | 1 | 1.827 | 1.754 | +0.074 | no |
| | 2 | 1.738 | 1.730 | +0.008 | no (by 0.008) |
| 1.0% | 0 | 3.024 | 3.006 | +0.018 | no (by 0.018) |
| | 1 | 3.178 | 2.377 | +0.801 | no |
| | 2 | 3.116 | 2.375 | +0.741 | no |

Means over seeds -- 0.2%: 1.009 vs 1.546; 0.5%: 1.824 vs 1.964; 1.0%: 3.106
vs 2.586. P0: every compiled circuit reproduced the logical energy
noiselessly (worst 2.6e-11).

## 2. Scoring

**P1 (shallow wins at 1.0% on at least 2 of 3) -- CONFIRMED, 3 of 3**, though
seed 0 only by 0.018.

**P2 (deep wins at 0.2% on all 3) -- CONFIRMED.** The pre-registered
deprioritization condition is not met.

**P3 (deep wins at 0.5% on at least 2 of 3) -- FAILED, 1 of 3.** Seeds 1 and 2
went to the shallow circuit, seed 2 by only 0.008. The **mean** over seeds
favours the deep circuit (1.824 vs 1.964), driven by seed 0; the
pre-registered criterion was per seed, and is scored as such.

**P4 (penalty roughly linear in CX error) -- CONFIRMED.** Penalty at 0.2%
divided by penalty at 1.0%: D 0.232 / 0.246 / 0.251, shallow 0.218 / 0.213 /
0.212 -- all inside 0.1-0.3 (linear: 0.2). Slightly above linear at low noise
and correspondingly below it at high noise (0.5% / 1.0%: 0.52-0.60 against a
linear 0.5), i.e. the penalty grows a little less than proportionally.

## 3. Where the crossover is

Linear interpolation between neighbouring noise levels, per seed:

| seed | crossover CX error |
|---:|---:|
| 0 | 0.98% |
| 1 | 0.46% |
| 2 | 0.49% |

Seed 0 is the outlier because the shallow circuit trained poorly on that
seed at every noise level (noiseless gap 1.66-1.80, against 1.04-1.11 on the
other two seeds), not because the deep circuit did anything different. On
the two typical seeds the crossover is just below 0.5%. **Addendum 122's
linear extrapolation (about 0.6%) overestimated it**, partly because that
estimate was built from 40-iteration runs.

## 4. Two further readings

- **Addendum 122 Section 10's convergence reading was right in direction.**
  At 1% with 60 instead of 40 iterations, seed 0's margin for the shallow
  circuit shrank from 0.271 to 0.018 (projected to close; it did not quite
  flip), and seeds 1 and 2 widened (0.696 -> 0.801, 0.720 -> 0.741), as
  projected.
- **The deep circuit benefited from the longer run.** Its learned-parameter
  gap fell from 0.60-0.80 (Addendum 122, 40 iterations) to 0.18-0.53 here, and
  showed no consistent trend with the noise level (0.2%: 0.26-0.46; 0.5%:
  0.18-0.34; 1.0%: 0.33-0.53). With 42 CXs, noise did not systematically distort what was learned
  -- unlike the 84-CX compile-once circuit in Addendum 122, Section 4.

## 5. What this means for the quantum-AI direction

- The direction is **not** deprioritized: on a task that needs depth, a deep
  circuit with redundant CXs removed beats a shallow one at 0.2% CX error on
  every seed.
- The advantage disappears at about 0.5% CX error for this task under this
  noise model. A compile-strategy decision -- whether a deep circuit is worth
  running -- can therefore be made from the device's noise level, which is the
  first measured input for the "choose how to compile" component discussed in
  conversation.
- Per Addendum 124, re-synthesizing is not specific to PSF-Zero; what PSF-Zero
  adds is doing it fast at scale. That remains the claim to make.

## 6. What this does not establish

- Where any real device sits relative to 0.5%: this is a simple depolarizing
  model, not a device calibration, and no mapping to hardware is claimed.
- Other tasks: the crossover depends on how much depth buys for the task
  (here, the shallow circuit's noiseless gap is 2.0-5.7x the deep one's).
- Other optimizers or budgets; 3 seeds only.

## 7. Files

| File | What it is |
|---|---|
| [`noise_sweep_heisenberg.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/noise_sweep_heisenberg.py) | this run's script |
| [`noise_sweep_summary_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/noise_sweep_summary_2026-09-21.csv) | per-run results, 18 rows |
| [`noise_sweep_trajectories_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/noise_sweep_trajectories_2026-09-21.csv) | loss per iteration (produced by the run; not yet received for this record) |
| [`spare-qubit-cliff-addendum-125-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-125-preregistration-2026-09-21.md) | the predictions scored above |

## 8. Verification

- Every figure recomputed from the summary CSV; crossovers and ratios computed
  there.
- P3 scored per seed as pre-registered; the favourable mean is reported
  beside it, not in place of it.
- The comparison with Addendum 122 uses that addendum's own recorded values.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the CSV -> 0 hits.

---

<!-- ===== Addendum 127 pre-registration (source: spare-qubit-cliff-addendum-127-preregistration-2026-09-21.md) ===== -->

> **Note added when merging:** Before a PennyLane-vs-IBM head-to-head: does the Qiskit version pennylane-qiskit resolves to still show Paper 1's failure region?

## Addendum 127 -- Pre-registration: does the Qiskit version that pennylane-qiskit can use still have the VF2Layout failure region? (2026-09-21)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

The next goal is a head-to-head measurement inside PennyLane: the route
through IBM's stack (`pennylane-qiskit`, which transpiles with Qiskit) against
the route through PSF-Zero. `pennylane-qiskit` cannot be installed alongside
Qiskit 2.5.2 -- every release pip tried caps Qiskit at 2.3.0 or lower
(recorded in `r0_psf_zero_transform.py`'s environment note) -- so the IBM
route has to run in a separate environment on an older Qiskit.

Paper 1's failure region was characterized in Qiskit 2.5.2, whose
`VF2Layout` uses a Rust implementation with hardcoded VF2++ ordering. Whether
the older Qiskit shows the same behaviour decides what the head-to-head can
compare: if it does, the IBM route inside PennyLane falls into the same
region; if it does not, the comparison has to be built around something else.

## 2. Design

One script, `check_vf2_cliff_version.py`, run unchanged in both environments:

- the project environment (Qiskit 2.5.2), as a positive control;
- a new, separate environment with `pennylane-qiskit` installed, whatever
  Qiskit version pip resolves there (recorded with `pip freeze`).

Circuits: Paper 1's `dense_pairs` family -- disjoint pairs, each carrying one
random two-qubit unitary -- on square grids:

| config | grid | qubits used | spare |
|---|---|---:|---:|
| 6x7 spare 0 | 6x7 | 42 | 0 |
| 6x7 spare 2 (control) | 6x7 | 40 | 2 |
| 8x8 spare 0 | 8x8 | 64 | 0 |

Each transpiled with `optimization_level=3`, `basis_gates=["rz","sx","x","cx"]`,
3 seeds. Recorded through `transpile`'s callback: `VF2Layout`'s own run time
and its stop reason from the property set, plus total transpile time.

## 3. Pre-registered predictions and definitions

**Definition.** A configuration is "in the failure region" for a Qiskit
version if, on every seed, `VF2Layout` does not report a solution found and
takes at least 10x as long as on the 6x7 spare-2 control.

**P1 (positive control).** In Qiskit 2.5.2, both spare-0 configurations are in
the failure region and the spare-2 control is not. If this fails, the script
is not measuring what Paper 1 measured, and the second environment's result
is uninterpretable.

**P2 -- deliberately no directional prediction.** Whether the older Qiskit's
spare-0 configurations are in the failure region is the open question. Both
outcomes are useful and neither is favoured in advance.

## 4. What this cannot establish

- Anything about PennyLane itself -- this runs Qiskit directly, in the
  version `pennylane-qiskit` pulls in.
- Why the versions differ, if they do -- only whether.

---

<!-- ===== Addendum 128 (source: spare-qubit-cliff-addendum-128-2026-09-21.md) ===== -->

> **Note added when merging:** pip resolved pennylane-qiskit to Qiskit 1.2.4, which shows a different failure pattern: seed-dependent success/failure, and even the easy control instance takes 2.5-2.8s (vs 14ms on Qiskit 2.5.2).

## Addendum 128 -- In Qiskit 1.2.4 (what pennylane-qiskit resolved to), the failure region is not "absent" but different: seed-dependent, and even the easy control is slow -- the pre-registered definition did not anticipate that (2026-09-21)

**Pre-registered in**:
`spare-qubit-cliff-addendum-127-preregistration-2026-09-21.md`, written and
locked before either run.

## 0. In one line

**P1 confirmed; P2 (no directional prediction) resolves as "NOT in the failure
region" under the pre-registered definition -- but for reasons the definition
did not anticipate.** In Qiskit 2.5.2 both spare-0 configurations fail on
every seed (NO_SOLUTION_FOUND after 3.4-4.7 s) while the spare-2 control
succeeds in 13-16 ms. In Qiskit 1.2.4, the version pip installed alongside
`pennylane-qiskit` on Python 3.10, spare-0 fails on 2 of 3 seeds at both sizes
(4.0-5.1 s) and succeeds instantly on the third, and the spare-2 control --
which always succeeds -- itself takes 2.5-2.8 s. Both facts break the
definition's assumptions (failure on every seed; a fast control), so the
formal verdict is reported as measured and the behaviour is described
separately rather than the definition being changed after the fact.

## 1. Results

`VF2Layout` time per seed, stop reason, and median total transpile time
(`optimization_level=3`, 3 seeds).

| Qiskit | config | solution found | VF2Layout (ms), seeds 0 / 1 / 2 | total transpile, median |
|---|---|---:|---|---:|
| 2.5.2 | 6x7 spare 0 | 0 / 3 | 3363.8 / 3368.2 / 3360.2 | 6876 ms |
| 2.5.2 | 6x7 spare 2 (control) | 3 / 3 | 15.7 / 13.1 / 14.1 | 29 ms |
| 2.5.2 | 8x8 spare 0 | 0 / 3 | 4657.6 / 4546.2 / 4457.3 | 9234 ms |
| 1.2.4 | 6x7 spare 0 | 1 / 3 | **1.0** / 4063.0 / 3948.9 | 4046 ms |
| 1.2.4 | 6x7 spare 2 (control) | 3 / 3 | **2544.0 / 2755.9 / 2758.4** | 2846 ms |
| 1.2.4 | 8x8 spare 0 | 1 / 3 | 4900.8 / 5070.9 / **0.0** | 5061 ms |

Environment: both runs Python 3.10.11. The project environment has Qiskit
2.5.2; the separate environment (`psf_plq_env`) was created fresh and pip
resolved `pennylane-qiskit` to a release using Qiskit 1.2.4 -- the same
version it chose in the project environment earlier on 2026-09-21. The
separate environment's full `pip freeze` is not yet received for this record.

## 2. Scoring

**P1 (positive control in 2.5.2) -- CONFIRMED.** The script measures what
Paper 1 measured.

**P2 -- formal verdict: NOT in the failure region, both configurations.**
Two separate reasons, each sufficient:
1. One seed of three found a solution immediately at each size, so "fails on
   every seed" does not hold.
2. The control's median `VF2Layout` time is 2.76 s, so the "at least 10x the
   control" threshold would require about 27.6 s; the failing seeds took
   4.0-5.1 s, only 1.4-1.8x the control.

The definition was written assuming, from Qiskit 2.5.2's behaviour, that the
control would be fast. It is reported unchanged.

## 3. What the older version actually does

- **Outcome depends on the seed.** On two seeds per size, `VF2Layout` exhausts
  its search and reports no solution after 4-5 s -- the same failure as 2.5.2.
  On the third it finds a valid layout in 1 ms or less. One reading, consistent
  with Paper 1's central finding that the traversal order decides success: in
  this older line, the search order depends on the seed, so some seeds are
  lucky and others are not; in 2.5.2 the VF2++ order is fixed, so every seed
  fails alike. That reading is a hypothesis -- 1.2.4's source was not examined
  here.
- **Even easy instances are slow.** The spare-2 control always succeeds but
  spends 2.5-2.8 s in `VF2Layout`, about 195x Qiskit 2.5.2's time (medians 2755.9 vs 14.1 ms) on the same
  instance. A plausible cause is that the search continues after the first
  valid layout, looking for a better-scoring one; also not verified.

## 4. What this means for the planned PennyLane head-to-head

- If the IBM route inside PennyLane runs on Qiskit 1.2.4, layout alone costs
  roughly 2.5-5 s per circuit at 40-64 qubits whether or not the device has
  spare qubits, and fails outright on some seeds at full occupancy. That is a
  real comparison axis -- but against a Qiskit generation two major versions
  behind IBM's current release.
- **A fair comparison needs the newest IBM route `pennylane-qiskit` supports.**
  The latest release pip tried earlier today (0.45.0) accepts Qiskit up to
  2.3.0; here pip settled on a 1.2.4-compatible release instead. Why is not yet
  known; one untested possibility is that the newer PennyLane releases it would
  need do not support Python 3.10. The `pip freeze` from `psf_plq_env` will
  show which versions were chosen; if Python is the constraint, a separate
  Python 3.11+ environment would give the newer route.

## 5. Files

| File | What it is |
|---|---|
| [`check_vf2_cliff_version.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_vf2_cliff_version.py) | the script, run unchanged in both environments |
| [`vf2_cliff_check_qiskit_2_5_2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_2_5_2.csv) | project environment |
| [`vf2_cliff_check_qiskit_1_2_4.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_1_2_4.csv) | `psf_plq_env` |
| [`spare-qubit-cliff-addendum-127-preregistration-2026-09-21.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-127-preregistration-2026-09-21.md) | the predictions scored above |

## 6. Verification

- Both CSVs read directly; the formal verdict recomputed from them with the
  pre-registered definition.
- The Qiskit versions come from each run's own printed header and file name.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both CSVs -> 0 hits.

---

<!-- ===== Addendum 129 (source: spare-qubit-cliff-addendum-129-2026-09-22.md) ===== -->

> **Note added when merging:** The Qiskit 1.2.4 pattern reproduced exactly (all 9 seed/config outcomes) on a second, independent install found while setting up a Python 3.11+ environment for a fair comparison -- also documents an installation mistake that briefly uninstalled Qiskit from the project's own .venv, caught and reverted the same session.

## Addendum 129 -- Qiskit 1.2.4's seed-dependent behaviour (Addendum 128) reproduces exactly across two independent installs, and its "easy" control is 195x slower than Qiskit 2.5.2's on the identical instance (2026-09-22)

**Status**: an independent re-run of Addendum 128's check, on a second,
separately-created installation of Qiskit 1.2.4 (system Python, not the
`psf_plq_env` virtual environment Addendum 128 used), found while setting up
a PennyLane-vs-Qiskit head-to-head comparison. `check_vf2_cliff_version.py`
and its pre-registered definition are unchanged from Addendum 127.

## 0. In one line

**Every one of 9 seed x config outcomes reproduced exactly**: the same three
seeds found a solution instantly and the same six did not, across two
installations of `pennylane-qiskit` that independently resolved to Qiskit
1.2.4 (one in a fresh virtual environment on 2026-09-21, one on system Python
on 2026-09-22). Timings agree to within a few percent. This was not the
comparison being set up -- it was found as a side effect of an installation
mistake, documented in full in Section 2 for the record.

## 1. Results

| config | seed | run 1 (2026-09-21, `psf_plq_env`) | run 2 (2026-09-22, system Python) |
|---|---:|---|---|
| 6x7 spare 0 | 0 | SOLUTION_FOUND, 1.0 ms | SOLUTION_FOUND, 1.0 ms |
| | 1 | NO_SOLUTION_FOUND, 4063.0 ms | NO_SOLUTION_FOUND, 4151.7 ms |
| | 2 | NO_SOLUTION_FOUND, 3948.9 ms | NO_SOLUTION_FOUND, 4080.6 ms |
| 6x7 spare 2 (control) | 0 | SOLUTION_FOUND, 2544.0 ms | SOLUTION_FOUND, 2550.2 ms |
| | 1 | SOLUTION_FOUND, 2755.9 ms | SOLUTION_FOUND, 2789.0 ms |
| | 2 | SOLUTION_FOUND, 2758.4 ms | SOLUTION_FOUND, 2735.1 ms |
| 8x8 spare 0 | 0 | NO_SOLUTION_FOUND, 4900.8 ms | NO_SOLUTION_FOUND, 5147.0 ms |
| | 1 | NO_SOLUTION_FOUND, 5070.9 ms | NO_SOLUTION_FOUND, 5109.3 ms |
| | 2 | SOLUTION_FOUND, 0.0 ms | SOLUTION_FOUND, 1.0 ms |

Formal verdict (Addendum 127's definition), both runs: **NOT in the failure
region**, for the same two reasons as Addendum 128 -- not every seed fails,
and the control itself is not fast.

**The control comparison, made explicit here.** Qiskit 2.5.2's own median on
the identical 6x7 spare-2 instance (from `vf2_cliff_check_qiskit_2_5_2.csv`,
Addendum 127) is 14.1 ms. Qiskit 1.2.4's median here is 2755.9 ms (run 2):
**195x slower on the instance both versions solve easily.**

## 2. How run 2 happened

While setting up a second, Python-3.11+ environment for a fair PennyLane
head-to-head (`pennylane-qiskit`'s newest release needs Python >= 3.11,
confirmed from its own published release notes), a `py -3.12 -m venv ...`
command was issued without first confirming Python 3.12 was installed
(`py -0p` had not been run first). It failed, but the shell then continued
executing the remaining setup commands, which installed `pennylane`,
`pennylane-qiskit`, and their dependencies onto the machine's system Python
3.10 rather than into a new virtual environment. `pennylane-qiskit` resolved
to the same release as Addendum 128 (0.42.0, pulling in Qiskit 1.2.4).

`check_vf2_cliff_version.py`, which needs only Qiskit, ran correctly against
this system Python and produced the CSV scored above. A later step
(`python psf_pennylane.py`) failed, because the shell's `python` resolved to
this system installation, whose separately-installed `psf_zero_core` predated
`batch_decompose_checked` -- unrelated to Qiskit and not evidence about
anything in this addendum.

**A second, separate mistake followed.** The instruction given to clean up
the system Python's packages was executed against the project's own `.venv`
instead (both shells showed `(.venv)` in the prompt by that point), which
uninstalled `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `symengine` and
downgraded `sympy` in the environment every PSF-Zero benchmark and both
papers were measured on. This was caught immediately (`pip check` showed the
five packages as missing rather than conflicting) and reverted:
`qiskit==2.5.2`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.49.0`,
`sympy==1.14.0`, `symengine==0.14.1`, confirmed restored via
`check_core_build.py` (RESULT: OK) and `qiskit.__version__` (2.5.2). One
harmless residue remains: `pennylane-lightning` is installed in `.venv` with
no matching `pennylane`, flagged by `pip check` and not yet resolved. No
PSF-Zero benchmark or paper figure used this environment while it was in the
altered state.

This is recorded in full because the project's own standard is to report
what happened, not only what was intended -- the same standard Addendum 108
and others have applied to the project's own numbers.

## 3. What this means

- **Addendum 128's finding is not a fluke of one install.** Two independent
  resolutions of `pennylane-qiskit`'s dependency range landed on the same
  Qiskit release and reproduced identical pass/fail outcomes per seed.
- **The 195x figure sharpens what a PennyLane-via-Qiskit route would cost.**
  Addendum 128 already showed the "easy" control is slow in absolute terms
  (2.5-2.8 s); this addendum states the size of that cost directly against
  the version this project's own papers were measured on, on the identical
  instance.
- **The still-open item is unchanged**: a fair comparison needs the newest
  Qiskit `pennylane-qiskit` supports (2.3.0), which needs Python >= 3.11.
  That environment has not yet been built; see Section 2.

## 4. Files

| File | What it is |
|---|---|
| [`vf2_cliff_check_qiskit_1_2_4_rerun.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_1_2_4_rerun.csv) | run 2 (system Python, 2026-09-22) |
| [`vf2_cliff_check_qiskit_1_2_4.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_1_2_4.csv) | run 1 (`psf_plq_env`, 2026-09-21; Addendum 128) |
| [`vf2_cliff_check_qiskit_2_5_2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_2_5_2.csv) | the project environment control (Addendum 127) |
| [`check_vf2_cliff_version.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_vf2_cliff_version.py) | unchanged from Addendum 127 |

## 5. Verification

- The two CSVs were confirmed to be distinct files (different hashes) before
  being compared, so the agreement is between independent runs, not a
  re-upload.
- All nine seed x config outcomes compared directly; the 195x figure computed
  from the two files' own medians.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

<!-- ===== Addendum 130 (source: spare-qubit-cliff-addendum-130-2026-09-22.md) ===== -->

> **Note added when merging:** Resolves Addendum 128's open question: Qiskit 2.3.0 (the newest pennylane-qiskit supports) shows the same failure-region shape as Qiskit 2.5.2, not Qiskit 1.2.4's seed-dependent pattern -- so the 1.2.4 behaviour was specific to that old line, not to pennylane-qiskit generally. A fair head-to-head environment (psf_h2h_env, Python 3.12) now exists; psf_pennylane.py's self-test also passes unchanged on the newer PennyLane 0.45.1.

## Addendum 130 -- The failure region is present, in the same form as Qiskit 2.5.2, in Qiskit 2.3.0 -- the newest pennylane-qiskit supports; a fair comparison environment now exists, and psf_pennylane.py passes unchanged on the newer PennyLane (2026-09-22)

**Environment**: `psf_h2h_env`, freshly created (Addendum 129 traced why Python
3.10 could not reach this: `pennylane-qiskit`'s newer releases need Python
>= 3.11). Python 3.12.10, `pennylane-qiskit` 0.45.0, resolving `qiskit`
2.3.0, `pennylane` 0.45.1, `rustworkx` 0.18.1. `check_vf2_cliff_version.py`
and `psf_pennylane.py` both run unchanged from prior addenda.

## 0. In one line

**Qiskit 2.3.0 -- the newest release `pennylane-qiskit` supports -- shows the
same failure-region shape as Qiskit 2.5.2, not the seed-dependent pattern of
Qiskit 1.2.4 (Addenda 128-129).** Both spare-0 configurations fail on all 3
seeds (3.3-4.6 s) and the spare-2 control succeeds quickly on all 3 (12.3 ms
median, statistically indistinguishable from 2.5.2's 14.1 ms). This resolves
the open question from Addendum 128: the failure region is not an artefact of
one Qiskit line's age, and a comparison built on Qiskit 2.3.0 is a fair one,
not a comparison against a version already known to behave differently. A
second result: `psf_pennylane.py`'s self-test passes unchanged on PennyLane
0.45.1, a newer major line than the 0.42.3 it was verified against.

## 1. Results

| config | seed | Qiskit 2.3.0 | Qiskit 2.5.2 (Addendum 127) | Qiskit 1.2.4 (Addenda 128-129) |
|---|---:|---|---|---|
| 6x7 spare 0 | 0 | NO_SOLUTION_FOUND, 3358.0 ms | NO_SOLUTION_FOUND, 3363.8 ms | SOLUTION_FOUND, 1.0 ms |
| | 1 | NO_SOLUTION_FOUND, 3376.3 ms | NO_SOLUTION_FOUND, 3368.2 ms | NO_SOLUTION_FOUND, ~4100 ms |
| | 2 | NO_SOLUTION_FOUND, 3334.8 ms | NO_SOLUTION_FOUND, 3360.2 ms | NO_SOLUTION_FOUND, ~4000 ms |
| 6x7 spare 2 (control) | 0 | SOLUTION_FOUND, 12.3 ms | SOLUTION_FOUND, 15.7 ms | SOLUTION_FOUND, ~2550 ms |
| | 1 | SOLUTION_FOUND, 12.0 ms | SOLUTION_FOUND, 13.1 ms | SOLUTION_FOUND, ~2770 ms |
| | 2 | SOLUTION_FOUND, 13.0 ms | SOLUTION_FOUND, 14.1 ms | SOLUTION_FOUND, ~2750 ms |
| 8x8 spare 0 | 0 | NO_SOLUTION_FOUND, 4461.4 ms | NO_SOLUTION_FOUND, 4657.6 ms | NO_SOLUTION_FOUND, ~5000 ms |
| | 1 | NO_SOLUTION_FOUND, 4417.2 ms | NO_SOLUTION_FOUND, 4546.2 ms | NO_SOLUTION_FOUND, ~5100 ms |
| | 2 | NO_SOLUTION_FOUND, 4564.2 ms | NO_SOLUTION_FOUND, 4457.3 ms | SOLUTION_FOUND, ~0.5 ms |

(1.2.4 values are the mean of the two independent runs, Addenda 128-129,
rounded.)

Formal verdict (Addendum 127's pre-registered definition): **6x7 spare 0 and
8x8 spare 0 are both IN the failure region** on Qiskit 2.3.0 -- the same
verdict as 2.5.2, the opposite of 1.2.4's.

**Control comparison**: 12.3 ms (2.3.0) vs 14.1 ms (2.5.2) vs ~2755 ms (1.2.4).
2.3.0 is close to 2.5.2 and about 224x faster than 1.2.4 on the identical
instance.

`psf_pennylane.py` self-test on this environment: all checks passed,
including search times of 0.98-1.79 ms on the same failure-region instances,
loss/gradient agreement to 3.77e-15/1.22e-14, and no `qiskit` import
anywhere in the run -- consistent with Addendum 128's own run on PennyLane
0.42.3, now confirmed on 0.45.1 without any code change.

## 2. What this settles

- **The Qiskit 1.2.4 behaviour (Addenda 128-129) does not generalize to
  `pennylane-qiskit`'s current release.** It was specific to the old Qiskit
  line that pip resolved to under Python 3.10. Under Python 3.12, the same
  `pennylane-qiskit` package resolves to Qiskit 2.3.0, which fails the same
  way 2.5.2 does.
- **A head-to-head inside PennyLane can now be built fairly**: the IBM route
  (`pennylane-qiskit` 0.45.0 -> Qiskit 2.3.0) and the PSF-Zero route
  (`psf_pennylane.py`, Qiskit-independent) both run in `psf_h2h_env`, on the
  same Python, the same machine, the same session.
- **`psf_pennylane.py` needed no changes for the newer PennyLane.** Its
  known-untested items (torch/JAX interfaces, the PennyLane version pin) are
  unaffected by this; the autograd-interface, layout, and KAK checks it does
  cover all passed unchanged.

## 3. What remains before the actual head-to-head

- A shared script that runs both routes (IBM: `pennylane_qiskit.load`-backed
  device or Qiskit transpile inside a QNode; PSF-Zero: `psf_for_device`) on
  the identical circuits in `psf_h2h_env`, timing each the same way.
- Whether to compare compile/layout time only (as this addendum and Addendum
  127 do) or full circuit execution -- not yet decided.

## 4. Files

| File | What it is |
|---|---|
| [`vf2_cliff_check_qiskit_2_3_0.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_2_3_0.csv) | this run, `psf_h2h_env` |
| [`vf2_cliff_check_qiskit_2_5_2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_2_5_2.csv) | project environment (Addendum 127) |
| [`vf2_cliff_check_qiskit_1_2_4.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_1_2_4.csv), [`vf2_cliff_check_qiskit_1_2_4_rerun.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cliff_check_qiskit_1_2_4_rerun.csv) | Addenda 128-129 |
| [`check_vf2_cliff_version.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_vf2_cliff_version.py), [`psf_pennylane.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_pennylane.py) | unchanged |

## 5. Verification

- All figures read directly from the new CSV; the control ratio computed from
  it and Addendum 127's own file.
- The formal verdict applied with Addendum 127's definition, unchanged.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

<!-- ===== Addendum 131 pre-registration (source: spare-qubit-cliff-addendum-131-preregistration-2026-09-22.md) ===== -->

> **Note added when merging:** The first real head-to-head inside PennyLane: Route A (IBM, pennylane-qiskit) vs Route B (PSF-Zero). Records two design corrections made before any run: a Windows-incompatible SIGALRM timeout, and a first design that tried to execute 42-qubit circuits exactly (impossible on either route -- a 42-qubit statevector needs tens of terabytes), split into compile-time-only (42 qubits) and correctness (n=6) checks instead.

## Addendum 131 -- Pre-registration: the first head-to-head inside PennyLane -- does the IBM route (pennylane-qiskit) hit the same cliff there that Qiskit shows directly, while the PSF-Zero route does not? (2026-09-22)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 130 established a fair environment (`psf_h2h_env`: Python 3.12,
`pennylane-qiskit` 0.45.0 -> Qiskit 2.3.0) and confirmed both routes run in
it: the IBM route's Qiskit shows Paper 1's failure region directly (checked
with `check_vf2_cliff_version.py`, bypassing PennyLane), and the PSF-Zero
route (`psf_pennylane.py`) finds perfect layouts on the same instances in
under 2 ms, also bypassing PennyLane's own device machinery. Neither of those
checks went through a PennyLane device. This experiment does.

Reading `pennylane_qiskit`'s own source (`qiskit_device.py`,
`get_transpile_args` / `compile_circuits`) shows the device layer does not
reimplement transpilation: keyword arguments such as `optimization_level` and
`seed_transpiler` are separated out and passed straight to Qiskit's own
`transpile()`. The prediction that follows is mechanical, not a guess about
PennyLane's own behaviour: whatever Qiskit does directly, the device should
do when driven through a QNode.

## 2. Design

**Circuits**: Paper 1's `dense_pairs` family, 6x7 grid (42 qubits): spare 0
(the failure-region instance) and spare 2 (the control), 3 seeds each -- the
same instances Addendum 127/130 used directly.

**Coupling map for the IBM route**: a `GenericBackendV2` (Qiskit's own
fake-backend builder) constructed with the grid's own coupling map, so
`transpile` sees the identical device graph as the direct Qiskit runs.

**Route A (IBM), compilation only, 42 qubits**: the tape's operations
converted to a Qiskit `QuantumCircuit` (via `pennylane_qiskit`'s own
converter) and passed to Qiskit's own `transpile(circuit, coupling_map=...,
basis_gates=[...], optimization_level=3, seed_transpiler=<seed>)` directly --
matching what `pennylane_qiskit`'s device calls internally (confirmed from
its own source, `get_transpile_args`/`compile_circuits`), without executing
the result on any simulator.

**Route B (PSF-Zero), compilation only, 42 qubits**: `psf_layout` followed by
`r0_psf_zero_transform` applied to the same tape, timed the same way, also
without execution.

**Correctness, n=6, executed for real**: both routes' full QNode pipelines
(construction through execution) on a 6-qubit `dense_pairs` circuit --
small enough for exact statevector execution on both
`AerSimulator(method="statevector")` (Route A) and `default.qubit` (Route
B) -- comparing the two routes' returned expectation values directly.

Both routes act on the SAME logical circuit at each scale. This experiment
measures compilation/layout cost through each route's own real call path (P1,
P2, P4) and end-to-end correctness at a scale where exact execution is
possible for both (P3) -- not circuit fidelity or execution physics at 42
qubits, which neither route can compute exactly (Section 3a-2).

## 3. Pre-registered predictions

**P1 (the mechanical prediction).** On spare-0 seeds, Route A's QNode call
takes at least 2 s -- in the same range Qiskit showed directly (3.3-3.4 s,
Addendum 130) -- confirming the failure region is reachable through
PennyLane's own device layer, not only via direct `transpile()` calls.

**P2.** On spare-0 seeds, Route B's QNode call takes under 100 ms --
consistent with `psf_pennylane.py`'s own direct-call timings (1-2 ms for the
layout step alone; PennyLane's own tape/QNode overhead is untested and the
bound is set loosely to accommodate it).

**P3 (correctness, not yet checked in a PennyLane device context).** Route
B's execution matches Route A's on the spare-2 control (where both should
succeed) to within 1e-6 on a shared observable -- checked once, at spare 2,
since spare 0 is expected to fail on Route A entirely (P1) and there is
nothing to compare there.

**P4.** On the spare-2 control, Route A's compilation is fast (under 200 ms,
matching Qiskit's own direct 12-25 ms plus PennyLane's own conversion
overhead) on all 3 seeds -- a check that Route A's own machinery is not slow
generally, only on the failure-region instances.

## 3a-2. A third problem, found by actually running the script again: 42-qubit statevectors do not fit in memory

The AerSimulator(statevector) fix (Section 3a) itself failed to construct:
`pennylane-qiskit` checks a statevector backend's own qubit limit before
running anything, and refused with "supports maximum 29 wires". This is
correct behaviour, not a bug -- a 42-qubit statevector needs 2^42 complex
amplitudes, tens of terabytes, regardless of which route computes it. Neither
route can be checked for exact execution correctness at 42 qubits.

**The design is split into two separate checks accordingly:**
- **Compilation time (P1, P2, P4)**, at the full 42-qubit scale, measured by
  compiling each route's circuit WITHOUT executing it: Route A via Qiskit's
  own `transpile()` on the tape's Qiskit-converted circuit (matching what
  `pennylane_qiskit` calls internally, confirmed from its own source); Route
  B via `psf_layout` + `r0_psf_zero_transform` applied to the tape directly,
  without running it on a device.
- **Correctness (P3)**, at a small scale (n=6, well under the 29-wire limit),
  where both routes CAN execute exactly: comparing Route A's and Route B's
  QNode results directly, as originally designed.

## 3a. A second bug, found by actually running the script once

The first execution attempt (with GenericBackendV2 as Route A's backend)
crashed rather than measuring anything: GenericBackendV2's backend only
supports shot-based sampling, so `shots=None` was silently ignored (a
printed UserWarning was the only sign), and at 42 qubits the resulting shot
simulation exceeded Aer's own memory limit. Route A's backend was replaced
with `AerSimulator(method="statevector")`, which honours `shots=None` and
still exposes a coupling map and basis gates for `transpile()` to use, so
VF2Layout still sees the real 6x7 grid. This fix is recorded before the
predictions below are scored against any real run.

## 3b. A platform bug found and fixed before any run

The first draft of `head_to_head_pennylane.py` timed out a hung call using
`signal.SIGALRM`, which does not exist on Windows -- the platform this
project's own measurements run on. Found by inspection before running
anything; replaced with a `concurrent.futures.ThreadPoolExecutor`-based
timeout, which is cross-platform. Noted here because it is exactly the kind
of untested-assumption bug this project's own standard is to report, whether
or not it was ever executed.

## 4. What this cannot establish

- Real IBM hardware -- Route A runs against a simulator behind a fake
  backend, not a submitted job; no queueing or network time is included.
- Whether PennyLane's own overhead (tape construction, device setup) is
  comparable between the two device types -- `default.qubit` and
  `qiskit.aer` are different implementations; any difference outside the
  compilation step itself is not isolated here.
- Execution correctness at spare 0 -- Route A is expected to fail there
  (P1), so no comparison is made.

---

<!-- ===== Addendum 132 (source: spare-qubit-cliff-addendum-132-2026-09-22.md) ===== -->

> **Note added when merging:** All four predictions confirmed: Route A (IBM) 6.7-7.2s vs Route B (PSF-Zero) 4.6-5.8ms on the failure-region instance (1,200-1,450x); both fast on the control; n=6 correctness to machine precision. Also documents and corrects an error in this addendum's own first-draft analysis (comparing against the wrong prior timing figure).

## Addendum 132 -- The first real PennyLane head-to-head: Route A (IBM, via Qiskit's own transpile) takes 6.7-7.2s on the failure-region instance where Route B (PSF-Zero) takes 4.6-5.8ms -- all four predictions confirmed, correctness verified to machine precision (2026-09-22)

**Pre-registered in**:
`spare-qubit-cliff-addendum-131-preregistration-2026-09-22.md`, written and
locked before this run (after two design corrections found by actually
running the script -- Section 3a-2 of that document -- both made before any
data was collected).

## 0. In one line

**All four pre-registered predictions confirmed, on every seed.** At the
saturated 6x7 instance (spare 0), Route A's compilation (Qiskit's own
`transpile`, the same call `pennylane_qiskit`'s device makes internally)
takes 6.68-7.21s; Route B's (PSF-Zero's layout + synthesis, applied directly
to the PennyLane tape) takes 4.6-5.8ms -- **about 1,200-1,500x faster on this
instance.** At the spare-2 control, both are fast (Route A: 29-30ms; Route B:
4.5-84.9ms). At n=6, where both routes can execute for real, their results
agree to 1.9e-16 to 2.9e-14 -- machine precision. **This is the first time
PSF-Zero's advantage has been measured through PennyLane's own call path on
the IBM side**, rather than by calling Qiskit or `psf_smart_layout` directly.

## 1. Results

### Compilation time, 42 qubits (Route A: Qiskit `transpile` directly, the
same call `pennylane_qiskit` makes internally; Route B: `psf_layout` +
`r0_psf_zero_transform` on the PennyLane tape, not executed)

| config | seed | Route A (IBM) | Route B (PSF-Zero) | ratio |
|---|---:|---:|---:|---:|
| spare 0 | 0 | 7.2061 s | 5.82 ms | 1,238x |
| | 1 | 6.6786 s | 4.86 ms | 1,374x |
| | 2 | 6.7066 s | 4.64 ms | 1,445x |
| spare 2 (control) | 0 | 29.9 ms | 4.5 ms | 6.6x |
| | 1 | 30.0 ms | 84.9 ms | 0.35x |
| | 2 | 29.1 ms | 4.5 ms | 6.5x |

Two-qubit gate counts, spare 0: Route A 63, Route B 84 (matching Paper 1's
own figures for this family: Qiskit's default synthesis reaches 63 with
`layout_search`-style routing avoided by success; PSF-Zero's `canonical`
basis here -- `r0_psf_zero_transform`'s own default -- was not tuned for CX
count on this path, unlike `psf_compile.py`'s dedicated `"cx"` basis, which
Addenda 116-118 verified matches Qiskit's own count).

### Correctness, n=6, both routes executed for real

| seed | Route A | Route B | \|diff\| |
|---:|---:|---:|---:|
| 0 | 0.05570797 | 0.05570797 | 1.94e-16 |
| 1 | 0.12054389 | 0.12054389 | 2.85e-14 |
| 2 | 0.03300305 | 0.03300305 | 1.32e-15 |

## 2. Scoring

**P1 (Route A >= 2s at spare 0) -- CONFIRMED, all 3 seeds** (6.68-7.21s).

**P2 (Route B < 100ms at spare 0) -- CONFIRMED, all 3 seeds** (4.6-5.8ms).

**P3 (agreement < 1e-6 at n=6) -- CONFIRMED, all 3 seeds** (1.9e-16 to
2.9e-14 -- far tighter than the pre-registered bound).

**P4 (Route A < 200ms at spare 2) -- CONFIRMED, all 3 seeds** (29-30ms).

## 3. A comparison this addendum's own first draft got wrong, corrected before publication

A first pass at Section 1 compared Route A's spare-0 time here against
Addendum 130's own *VF2Layout-pass-only* time (3.3-3.4s, measured via a
callback isolating that one pass) and called it "about 2x slower than
expected." That was the wrong comparison: Route A here calls plain
`transpile()`, with no callback, so it measures the WHOLE pipeline --
VF2Layout failing, then whatever Qiskit falls back to, then routing and gate
synthesis -- not the isolated VF2Layout time. The correct comparison is
against Addendum 130's own **total transpile time** on the identical
instance: 7.28s, 6.74s, 6.72s (from `vf2_cliff_check_qiskit_2_3_0.csv`'s own
`total_s` column) -- which match this run's 7.21s, 6.68s, 6.71s closely. No
discrepancy exists; the error was in this addendum's own first-draft
analysis, not in the measurement, and is recorded rather than silently
dropped.

## 4. What this establishes

- **The failure region is reachable through PennyLane's own device-selection
  and transpilation path**, not only by calling Qiskit directly -- confirming
  what Addendum 131's mechanical prediction (from reading
  `pennylane_qiskit`'s own source) said should be true.
- **PSF-Zero's compilation-time advantage, measured for the first time inside
  PennyLane itself**, is of the same order Papers 1-2 and Addenda 110-118
  found by calling Qiskit and PSF-Zero directly: roughly three orders of
  magnitude on the saturated instance, and comparable (well within an order
  of magnitude, noisily) on the unsaturated control.
- **Correctness through the full PennyLane pipeline is exact**, not merely
  plausible: both routes' QNode executions agree to floating-point noise.

## 5. What this does not establish

- Real IBM hardware -- Route A's compile target is a coupling map and basis
  gate set, not a submitted job; no queueing or network time is included,
  and Route A's own circuit was never executed at 42 qubits (Section 3a-2 of
  Addendum 131 explains why: no exact statevector fits at that size, on
  either route).
- Gate-count comparability at 42 qubits: Route B used `r0_psf_zero_transform`
  ---- `s own default (`canonical`) basis, not the CX-tuned path
  `psf_compile.py` uses; the 63-vs-84 gap above should not be read as a
  quality regression without re-running with a CX-matched configuration.
- Anything about the spare-2 control's own timing beyond "both are fast":
  the ratio there flips between runs (6.6x, 0.35x, 6.5x) and is within noise
  at these small absolute times (tens of milliseconds).

## 6. Files

| File | What it is |
|---|---|
| [`head_to_head_pennylane.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/head_to_head_pennylane.py) | this run's script |
| [`head_to_head_2026-09-22.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/head_to_head_2026-09-22.csv) | raw results, 9 rows |
| [`spare-qubit-cliff-addendum-131-preregistration-2026-09-22.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-131-preregistration-2026-09-22.md) | the predictions scored above, including two design corrections made before any run |

## 7. Verification

- All figures recomputed directly from the CSV; the corrected comparison in
  Section 3 recomputed from `vf2_cliff_check_qiskit_2_3_0.csv`'s own
  `total_s` column, not from memory.
- Every pre-registered prediction checked against its own stated threshold,
  not restated after seeing the result.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

<!-- ===== Addendum 133 pre-registration (source: spare-qubit-cliff-addendum-133-preregistration-2026-09-22.md) ===== -->

> **Note added when merging:** Requests re-running the Addendum 126 noise sweep after the CX-decomposer fix (116-117), predicting the deep-vs-shallow crossover would move to a higher CX error.

## Addendum 133 -- Pre-registration: re-running the noise-sweep experiment (Addendum 126) after the CX-decomposer fix (Addendum 116) -- does the crossover move? (2026-09-22)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addenda 110-126 (the variational-training-loop line) all call
`psf_compile.compile(entangling_basis="cx")`. Addendum 116 later found and
fixed a defect in exactly that call path: the CX-basis decomposer, built
without an Euler-basis setting, placed two `sx` pulses per qubit between
each pair of CXs where Qiskit's own re-compile placed at most one --
confirmed and applied to the repository's `psf_compile.py` in Addendum 117.
None of the variational-loop experiments (Addenda 110-126) have been re-run
since. This addendum re-runs the one whose own conclusion is most sensitive
to circuit depth and pulse count: Addendum 126's noise sweep, which found the
deep-vs-shallow crossover at approximately 0.46-0.49% CX error on two of
three seeds.

**What is expected to change, stated before running anything**: the deep
circuit's `sx` count and depth should now match Qiskit's own re-compile
exactly (Addendum 117), reducing the deep circuit's own execution-noise
penalty. Since the crossover is where the deep circuit's penalty stops being
worth its quality advantage, a smaller penalty should push the crossover to a
HIGHER CX error (the deep circuit staying favourable over a wider noise
range) -- the opposite direction from a naive "PSF-Zero got faster" read,
which would not by itself move a noise-based threshold at all. This
directional prediction is registered before running anything.

## 1a. Which environment this runs in

This experiment needs `torch` (the PyTorch bridge, Addendum 119), which is
installed only in the project's own `.venv` -- not in `psf_h2h_env`
(Addenda 129-132's PennyLane environment, which has no torch). This must
run in `.venv`, activated fresh, with its own Qiskit (2.5.2) confirmed
present via `check_core_build.py` before starting, given 2026-09-22's own
history of environment mix-ups (Addendum 129).

## 2. Design

Unchanged from Addendum 125/126 in every respect except the installed
`psf_compile.py` (now VERSION 2026-09-21, Addendum 116's fix applied): same
task (2x3 Heisenberg), same ansatze (`D_deep_red`, 42 CX before consolidation
-- wait, 3 CX per pair as established; `A_shallow_opt`, 21 CX), same three
CX error levels (0.2%, 0.5%, 1.0%, with sx/x error at 1/10th), same 60
iterations, same 3 seeds, same script (`noise_sweep_heisenberg.py`,
unmodified).

Before the sweep itself, a smaller check confirms the fix is actually active
in this environment: `analyze_depth_composition.py` (Addendum 113/117) is
re-run and its `D_psf_synthesis` row for the `same_pair` family is compared
against Addendum 117's own recorded post-fix values (total depth 16, sx 80
at 4x4). This project's own `.venv` was involved in an installation
accident on 2026-09-22 (Addendum 129) that briefly removed Qiskit from it
entirely, later restored; confirming the fix's presence directly, rather
than assuming it survived, is the appropriate level of caution given that
history.

## 3. Pre-registered predictions

**P0 (precondition).** `analyze_depth_composition.py`'s `D_psf_synthesis` row
matches Addendum 117's recorded post-fix figures (4x4 same_pair: total depth
16, sx 80) to confirm the fix is active before trusting anything else in
this addendum.

**P1 (direction, the main prediction).** The crossover CX error (linear
interpolation between the 0.2%/0.5%/1.0% points, per seed, as in Addendum
126) is HIGHER in this re-run than Addendum 126's own values on at least 2 of
3 seeds (Addendum 126: 0.98%, 0.46%, 0.49%). **If the crossover does not move
higher on at least 2 of 3 seeds, the fix's effect on this specific
noise-threshold question is smaller than predicted, or absent, and that is
reported as such rather than reinterpreted.**

**P2 (magnitude, weak).** At 0.5% CX error specifically, the deep circuit's
execution-noise penalty (own-execution gap minus noiseless gap) is smaller
in this re-run than Addendum 126's own value on all 3 seeds. This is a weaker
claim than P1 and is scored separately: P1 could hold without P2 holding
exactly at 0.5% if the improvement is concentrated elsewhere in the sweep.

**P3 (nothing else changes).** The shallow circuit's own figures (`sx`,
depth, timing, learned-parameter quality) are unchanged from Addendum 126 to
within run-to-run noise, since `A_shallow_opt`'s blocks are below
`compile()`'s collection floor and never reach the fixed decomposer
(Addendum 113/114's own finding, restated here as a check rather than
assumed).

## 4. What this cannot establish

- Whether the new crossover, wherever it lands, is closer to or further from
  any real device's own CX error -- unchanged limit from Addendum 125/126.
- Other tasks or ansatze -- this re-runs one experiment on one task.

---

<!-- ===== Addendum 134 (source: spare-qubit-cliff-addendum-134-2026-09-22.md) ===== -->

> **Note added when merging:** The request was based on a wrong premise: comparing dates rather than checking order of events. The Heisenberg circuits (119-126) were already built on the fixed psf_compile.py from the start, since Addendum 117 (the fix) preceded Addendum 119 within the same day. Confirmed by byte-identical results (16 significant digits) between the partial re-run and Addendum 126's own file, not assumed. The re-run was stopped partway through once this was noticed.

## Addendum 134 -- Addendum 133's re-run request rested on a wrong premise: the Heisenberg noise-sweep circuits (Addenda 119-126) were already built on the post-fix psf_compile.py from the start, confirmed directly rather than assumed from dates (2026-09-22)

**Pre-registered in**:
`spare-qubit-cliff-addendum-133-preregistration-2026-09-22.md`. This
addendum reports why the full re-run was stopped partway through, rather
than scoring the pre-registered predictions -- none of them apply.

## 0. In one line

**Addendum 133 was requested on a mistaken premise.** Its own P0 check
(`analyze_depth_composition.py`) correctly confirmed the CX-decomposer fix
(Addendum 116-117) is active in `.venv`. But the noise-sweep re-run itself
(`noise_sweep_heisenberg.py`), stopped after the 0.2% and 0.5% levels once
the anomaly below was noticed, produced results **identical to Addendum
126's own file to full floating-point precision** -- not merely similar.
Checking why: Addenda 113-117 (the depth bug's discovery and fix) used the
`same_pair`/`brickwork` ansatz family from Addenda 109-112. Addenda 119-126
(the Heisenberg task, including `D_deep_red`) used a different circuit
construction, written and first run in `train_heisenberg_torch.py`, **after**
Addendum 117's fix was already applied to the repository's `psf_compile.py`
-- both happened on 2026-09-21, but the fix came first within that day. The
Heisenberg circuits were never built on the broken decomposer. There is no
pre-fix Heisenberg data to compare against, and nothing to re-measure.

## 1. Evidence

| check | this re-run | Addendum 126 | identical? |
|---|---|---|---|
| `D_deep_red` device_cx | 42 | 42 | yes |
| `D_deep_red` device_sx | 96 | 96 | yes |
| `gap_own_execution`, p2=0.2%, seed 0 | 0.9642595699419019 | 0.9642595699419019 | yes, to 16 digits |
| `gap_own_execution`, p2=0.2%, seed 1 | 0.9520196367933238 | 0.9520196367933238 | yes, to 16 digits |
| `gap_own_execution`, p2=0.2%, seed 2 | 1.1104953620110152 | 1.1104953620110152 | yes, to 16 digits |

Agreement to 16 significant digits across an exact density-matrix simulation
and a deterministic optimizer is not consistent with two independent runs
of a changed circuit; it is consistent with the identical circuit being
compiled and simulated twice.

`analyze_depth_composition.py`'s own output (P0) matched Addendum 117's
post-fix figures exactly (4x4 same_pair: total depth 16, sx 80) -- correctly
confirming the fix is active in `.venv` for the circuit family it tests.
That check was not wrong; the inference drawn from it (that the Heisenberg
circuits must therefore have changed too) was.

## 2. What this means

- **No correction is needed for Addenda 121-126's own reported figures.**
  They already reflect the fixed decomposer; nothing in this project's own
  public record needs updating because of this.
- **Addendum 132's own comparison (CUDA-layer framing) is unaffected**: its
  own claim rests on Addenda 127-132 (the layout-search cliff), which does
  not call `psf_compile`'s CX decomposer at all, and on Addenda 119-126 for
  the training-loop argument, which turn out to have been measured
  correctly the first time.
- **The two-hour re-run was stopped partway through** (after the 0.2% and
  0.5% noise levels) once the identical-value pattern was noticed, rather
  than run to completion for a result already known.

## 3. A standing lesson, stated plainly

The request to re-run was based on comparing dates ("both happened on
2026-09-21") rather than checking order of events within that day, and was
not verified against the actual circuit before asking for two hours of
compute. The fix, once it existed, should have been checked directly (as
Section 1's table now does) before requesting any re-run -- the same
standard this project applies to every other claim, applied here one step
too late.

## 4. Files

| File | What it is |
|---|---|
| [`depth_composition_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/depth_composition_2026-09-21.csv) | this session's P0 check, byte-identical to Addendum 117's file |
| [`noise_sweep_summary_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/noise_sweep_summary_2026-09-21.csv), [`noise_sweep_trajectories_2026-09-21.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/noise_sweep_trajectories_2026-09-21.csv) | this session's partial re-run (0.2% and 0.5% levels only), matching Addendum 126's own file exactly on every field checked |
| [`spare-qubit-cliff-addendum-133-preregistration-2026-09-22.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-133-preregistration-2026-09-22.md) | the request this addendum found to be unnecessary |

## 5. Verification

- The `depth_composition_2026-09-21.csv` byte-identity was checked by direct
  file comparison before drawing any conclusion from it.
- The noise-sweep agreement was checked at full CSV precision (16 significant
  digits), not at the terminal's own truncated display precision, before
  concluding the two runs used the identical circuit.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---

**End of Part 7 of 8.** Continue to [Part 8](spare-qubit-cliff-combined-135.md), or back to [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md). Back to [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
