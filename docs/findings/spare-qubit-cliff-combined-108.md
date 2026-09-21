# spare-qubit-cliff: Combined Addenda, Part 7 of 7 (Addendum 108 through Addendum 121)

**Continued from [Part 6](spare-qubit-cliff-combined-88.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 5](spare-qubit-cliff-combined-51.md)).** Same conventions as every prior part: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

**Note on this part specifically**: it holds three threads, kept in chronological order and indexed here so each can be read on its own.

| Thread | Addenda | What it covers |
|---|---|---|
| Paper preparation | 108 | A correction to Paper 2's own figures, found by recomputing from raw data before writing. |
| Compilation inside a training loop | 109-118 | Whether re-compiling bound circuits pays off in variational training; a layout-once strategy; PSF-Zero's extra circuit depth measured, traced to one line of [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py), fixed (VERSION 2026-09-21), and the README re-measured. |
| Training under noise from PyTorch | 119-121 | A `torch.autograd.Function` bridge and noisy training experiments. **Addendum 121's result is pending** -- only its pre-registration is included; the result will be added to this part when recorded. |

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

> **Note added when merging:** A task that needs entangling depth (2x3 Heisenberg), plus a stronger control than a shallow circuit: blocks written optimally with 3 CXs from the start. PSF-Zero is predicted to TIE that control. RESULT NOT YET RECORDED at the time this part was assembled.

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

**End of Part 7 of 7 (end of document).** Back to [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
