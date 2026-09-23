# spare-qubit-cliff: Combined Addenda, Part 8 of 8 (Addendum 135 onward)

**Continued from [Part 7](spare-qubit-cliff-combined-108.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 6](spare-qubit-cliff-combined-88.md)).** Same conventions as every prior part.

**Note on this part specifically**: opens a new thread -- whether the cliff, characterized throughout this project on circuits built for that purpose, is reached by a named, standard algorithm (QAOA) that a practitioner would actually run, and whether "denser problem" and "hits the cliff" are the same thing (they turn out not to be; see Addendum 136).

---
<!-- ===== Addendum 135 pre-registration (source: spare-qubit-cliff-addendum-135-preregistration-2026-09-22.md) ===== -->

> **Note added when merging:** Tests, for the first time, whether a NAMED algorithm (QAOA MaxCut) -- not a circuit constructed for this project's own purposes -- reaches the cliff, and whether denser problem graphs get there sooner.

## Addendum 135 -- Pre-registration: does QAOA hit Qiskit's VF2Layout cliff, and does it get there sooner as the problem graph gets denser? (2026-09-22)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Every circuit family used in this project's own cliff measurements so far
(Papers 1-2, Addenda 88-132) was constructed directly for the purpose:
disjoint pairs, chain-shaped families, and the variational-training ansatze
of Addenda 109-126. None was a standard, named algorithm a practitioner
would actually run. This experiment uses QAOA -- one entangling layer per
edge of the problem's own graph, mapped directly onto that graph's own
structure -- to test two claims made only informally in conversation:

1. That real workloads people would actually run are more likely to be
   "heavy" (using most or all of a device's qubits) than the synthetic
   circuits tested so far -- untested until now.
2. That "heavy" and "hits the cliff" are related but not identical: the
   cliff is about qubit occupancy (spare=0) and interaction-graph structure,
   not circuit depth (Addendum 126's own deep-vs-shallow circuits used
   identical qubit counts throughout). QAOA is a natural case to test this
   distinction, because its own interaction graph is literally the problem's
   graph -- controlled directly by problem density, independent of circuit
   depth (the number of QAOA layers, p).

## 2. Design

**Circuit**: standard QAOA for MaxCut on a random Erdos-Renyi graph G(n, d)
-- n qubits, edge density d (probability each pair is connected) -- with p=1
layer (one `RZZ` per graph edge, one `RX` per qubit; p does not change the
interaction graph, only depth, so p=1 suffices for this experiment's own
question). The interaction graph fed to layout search is exactly the
problem graph.

**Devices**: 8x8 grid (64 qubits, matching this project's own established
grid instances) and IBM's own heavy-hex topology at a comparable qubit count
(this project's own `heavy_hex_d5`, 57 qubits, reused unchanged from
Addendum 104/106).

**Density sweep**: d in {0.1, 0.2, 0.3, 0.4, 0.5}, 3 random graph seeds each,
at each device's own qubit count. Graphs with more edges than the device can
embed without idle qubits are included -- they cannot achieve spare=0 in the
combinatorial sense this project's other addenda use, so "spare qubits" here
is redefined as the device's own qubit count minus the number of qubits the
QAOA graph actually uses (every qubit, since MaxCut QAOA is defined on all n
problem variables) -- i.e. spare=0 always, by construction, once n equals
the device's own qubit count. What varies with density is not occupancy but
the interaction graph's own edge count and structure, which is what this
experiment tests as the independent variable instead.

**Measured**, via `check_vf2_cliff_version.py`'s own callback method (Qiskit
2.5.2, this project's own paper-measurement environment): `VF2Layout`'s own
time and stop reason, and total transpile time, `optimization_level=3`.

## 3. Pre-registered predictions

**P1 (density effect, the main question).** `VF2Layout`'s own time increases
with graph density d on the 8x8 grid, and at d=0.5 falls in this project's
own established failure-region range (>= 2s, matching the 6x7/8x8
`dense_pairs` instances) on at least 2 of 3 seeds. **If VF2Layout stays fast
(under 100ms) even at d=0.5, QAOA at this scale does not reach the cliff,
and the "heavier problems hit the cliff more" intuition is not supported by
this experiment.**

**P2 (heavy-hex).** On the heavy-hex device, VF2Layout's own time does NOT
show the same sharp rise with density that P1 predicts for the grid, because
heavy-hex's own lower connectivity (degree <= 3) makes most graphs above
d=0.2 or so infeasible to embed as a perfect layout at all (the search
should fail FAST -- via the feasibility guard PSF-Zero's own layout search
uses, not tested here directly, but Qiskit's own VF2Layout is not known to
have an equivalent fast guard) -- reported as measured either way, since
this is the less certain of the two predictions.

**P3 (PSF-Zero side, for context, not a new claim).** Not measured directly
in this addendum; deferred to a follow-up only if P1 confirms QAOA reaches
the cliff, to avoid running the layout-search comparison on instances that
turn out not to need it.

## 4. What this cannot establish

- Whether QAOA at these densities is representative of what practitioners
  actually run (real MaxCut problem instances vary widely in structure).
- p > 1 (multiple QAOA layers) -- depth is deliberately held at p=1 since
  this experiment is about the interaction graph, not depth.
- Molecular/VUCCSD circuits -- a separate, not-yet-designed experiment.

---

<!-- ===== Addendum 136 (source: spare-qubit-cliff-addendum-136-2026-09-22.md) ===== -->

> **Note added when merging:** Falsified as predicted: QAOA never reaches the cliff at any density tested. The reason is more informative than 'too light' -- random graphs this dense have average degree far exceeding the device's own max degree, so a perfect layout is combinatorially impossible and VF2Layout recognizes this in under 1.1ms, every time. Clarifies that 'heavier' and 'hits the cliff' are different axes: the cliff needs feasible-AND-saturated instances, which generic dense QAOA graphs do not land on by chance.

## Addendum 136 -- QAOA does not reach the cliff at any density tested -- not because it is "light enough," but because a degree-infeasibility wall makes a perfect layout topologically impossible before density even matters, so VF2Layout returns in under 1.1ms every time, including at heavy-hex where a fast rejection was predicted for a different reason (2026-09-22)

**Pre-registered in**:
`spare-qubit-cliff-addendum-135-preregistration-2026-09-22.md`, written and
locked before this run.

## 0. In one line

**P1 falsified, clearly.** `VF2Layout` never showed density-dependent
slowdown: at every density from 0.1 to 0.5, on both devices, all 30 runs
returned `NO_SOLUTION_FOUND` in under 1.1ms -- none in this project's own
established failure-region range (seconds), and no trend with density at
all. **The reason is more informative than "QAOA is light":** a random
Erdos-Renyi graph's own average degree grows with density and with n
independent of any device, and already at the lowest density tested (0.1,
n=64) the average degree is 6.7 -- exceeding the 8x8 grid's own maximum
possible degree of 4, and the heavy-hex's own maximum of 3. Most nodes in
every generated graph cannot be placed on ANY physical qubit at all; a
perfect (zero-SWAP) layout is combinatorially impossible before the search
even starts, and Qiskit's own `VF2Layout` recognizes this essentially
instantly. **P2's predicted outcome (fast rejection on heavy-hex) also
holds, but for the identical reason as the grid's own result, not the
distinct one predicted** (a guard specific to heavy-hex's low connectivity)
-- so the intended contrast between P1 and P2 did not materialize; both
devices behaved the same way for the same reason.

## 1. Results

| device | density | mean edges (3 seeds) | mean degree | VF2Layout time (all runs) | stop reason |
|---|---:|---:|---:|---:|---|
| 8x8 grid (max degree 4) | 0.1 | 200.3 | 6.26 | 0.0-1.0 ms | NO_SOLUTION_FOUND |
| | 0.2 | 389.3 | 12.17 | 0.0 ms | NO_SOLUTION_FOUND |
| | 0.3 | 587.3 | 18.35 | 0.0-1.0 ms | NO_SOLUTION_FOUND |
| | 0.4 | 788.0 | 24.63 | 0.0 ms | NO_SOLUTION_FOUND |
| | 0.5 | 993.7 | 31.05 | 0.0-1.1 ms | NO_SOLUTION_FOUND |
| heavy_hex_d5 (max degree 3, n=57) | 0.1 | 158.3 | 5.55 | 0.0-1.0 ms | NO_SOLUTION_FOUND |
| | 0.2 | 309.7 | 10.87 | 0.0 ms | NO_SOLUTION_FOUND |
| | 0.3 | 464.3 | 16.29 | 0.0-1.0 ms | NO_SOLUTION_FOUND |
| | 0.4 | 622.3 | 21.83 | 0.0 ms | NO_SOLUTION_FOUND |
| | 0.5 | 783.7 | 27.50 | 0.0-1.0 ms | NO_SOLUTION_FOUND |

Total transpile time (not just `VF2Layout`) grew from 44-1355ms at d=0.1 to
~100-110ms at d=0.5 on the grid, and similarly on heavy-hex -- growth with
density is real, but stays two to four orders of magnitude below this
project's own established failure-region range (multi-second), and the
growth is consistent with routing/synthesis cost on a denser circuit, not
with `VF2Layout`'s own search struggling.

`heavy_hex_d5` construction confirmed correct before this table: n=57 qubits
(matching Addendum 104/106's own recorded figure exactly), 128 edges.

## 2. Scoring

**P1 (density -> cliff on the grid) -- FALSIFIED.** No run reached even
100ms of `VF2Layout` time, let alone the >= 2s threshold. As pre-registered,
this is reported as "QAOA at this scale does not reach the cliff" -- with
the mechanism now understood directly, rather than left unexplained.

**P2 (heavy-hex fails fast, for its own reason) -- outcome confirmed, stated
mechanism not distinguished from P1's.** Both devices fail near-instantly at
every density; the experiment as designed cannot tell whether heavy-hex's
own lower connectivity contributes anything beyond what the degree wall
already explains, since the grid also hits the same wall at the same
densities.

**P3**: not reached, as pre-registered (deferred, conditional on P1
confirming -- it did not).

## 3. What this means

- **"Heavier" and "hits the cliff" are not the same axis**, confirmed
  directly rather than only argued informally (as raised in conversation
  before this experiment): a dense random graph is not more likely to
  produce the multi-second cliff behaviour -- it is more likely to be
  instantly, trivially infeasible for a perfect layout at all, which
  `VF2Layout` handles fast and correctly. The cliff (Papers 1-2) is specific
  to instances that are simultaneously FEASIBLE (a perfect layout exists)
  and SATURATED (using the device's full matching capacity) -- a narrow
  target this project's own `dense_pairs`/`chain_shaped` families were
  deliberately constructed to hit, which generic QAOA MaxCut instances at
  these densities do not land on by chance.
- **This narrows, rather than broadens, where the cliff is a practical
  risk.** A practitioner running QAOA on a dense MaxCut instance is not
  thereby at risk of this project's own characterized failure mode; they
  are more likely to simply fail to find a perfect layout at all (correctly
  and quickly) and fall through to ordinary heuristic routing -- a
  different, already well-understood code path.
- **The application-level "does the gap widen" question from earlier in
  this conversation remains open for QAOA specifically**: since QAOA at
  these densities does not reach the cliff, PSF-Zero's layout-search
  advantage (Addenda 88-132) does not apply here either -- there is nothing
  for it to win on. Whether QAOA instances constructed to hit the narrower
  feasible-and-saturated condition exist, and whether they resemble
  anything a practitioner would actually run, is untested.

## 4. What this does not establish

- Molecular/VUCCSD circuits -- untested, as before.
- Low-density QAOA (below 0.1) or larger devices, where the degree wall
  might not bind -- not tested; the degree wall's own threshold (roughly
  device max degree / n) was not swept finely enough to locate.
- Whether any real MaxCut problem instance's own graph structure (as opposed
  to a uniformly random one) would behave differently -- Erdos-Renyi graphs
  are a convenient, not necessarily representative, choice.

## 5. Files

| File | What it is |
|---|---|
| [`check_qaoa_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_qaoa_cliff.py) | this run's script (one fix applied before running: `heavy_hex_graph()`'s `bidirectional` keyword does not exist in the installed rustworkx, found by the actual TypeError rather than assumed from documentation; edges are made symmetric explicitly in Python instead) |
| [`qaoa_cliff_2026-09-22.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/qaoa_cliff_2026-09-22.csv) | raw results, 30 rows |
| [`spare-qubit-cliff-addendum-135-preregistration-2026-09-22.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-135-preregistration-2026-09-22.md) | the predictions scored above |

## 6. Verification

- Mean degree computed directly from the CSV's own `n_edges` and `n`
  columns for every row, not estimated.
- `heavy_hex_d5`'s n=57 checked against Addendum 104/106's own recorded
  value before trusting any other figure in this table.
- Every one of the 30 rows individually confirmed as `NO_SOLUTION_FOUND`
  with `vf2_s` under 1.1ms, not sampled.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

<!-- ===== Addendum 137 pre-registration (source: spare-qubit-cliff-addendum-137-preregistration-2026-09-22.md) ===== -->

> **Note added when merging:** After WSL2/Ubuntu + lightning.gpu confirmed genuinely GPU-backed (CUDA_VISIBLE_DEVICES="" produces a CUDA-level error, not a silent CPU fallback): does the RTX 4070 actually beat CPU simulation, and at what qubit count? Caps the sweep at n=24 (not 32) after computing that a 32-qubit statevector alone needs about 69 GB, found before running anything.

## Addendum 137 -- Pre-registration: does PennyLane's lightning.gpu (RTX 4070, WSL2) actually speed up the circuits from Addenda 119-126, and at what qubit count does the crossover happen? (2026-09-22)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 136's own session confirmed, directly rather than assumed, that
`lightning.gpu` is genuinely using the RTX 4070 (not silently falling back to
CPU): disabling the GPU via `CUDA_VISIBLE_DEVICES=""` produces a CUDA-level
`RuntimeError`, not a silent success. This experiment measures whether that
genuine GPU execution is actually faster than CPU, and at what circuit size,
for the specific kind of circuit this project's own noisy-training
experiments (Addenda 119-126) used -- so the answer, if favourable, tells us
directly whether re-running that earlier work on this GPU would be worth the
time.

**A qualifier stated in advance, from general knowledge of GPU computing, not
yet tested here**: GPUs typically win on large, parallel workloads and lose
on small ones, where the fixed overhead of moving data to the GPU and back
dominates. Small circuits are therefore expected to run SLOWER on
`lightning.gpu` than on CPU -- this is treated as an expected outcome, not a
failure, and the experiment's actual purpose is finding the crossover point,
not confirming GPU is faster everywhere.

## 2. Design

**Environment**: `psf_zero_wsl_env_312` (WSL2, Ubuntu, Python 3.12,
`pennylane-lightning[gpu]` 0.45.0, RTX 4070), confirmed genuinely GPU-backed
in the prior session.

**Circuit**: the same redundant-block ansatz as Addendum 121/122
(`train_heisenberg_torch.py`'s `build_ansatz`, "redundant" form: `ry`, `rz`
per qubit per layer, then `cx` on disjoint pairs), at a fixed depth (2
layers), swept over qubit count n in {4, 8, 16, 20, 24} -- capped at 24 because a 32-qubit statevector alone needs about 69 GB, far beyond the RTX 4070's 12 GB VRAM, found by computing this before running anything rather than by a crash. No noise model in
this experiment -- noise is a separate question (Addendum 133-134 already
found it does not need re-measuring); this experiment is purely about raw
statevector simulation speed.

**Devices compared**: `default.qubit` (PennyLane's own reference simulator,
used throughout Addenda 119-126), `lightning.qubit` (PennyLane's own
optimized CPU simulator, not yet used in this project but the fairer CPU
comparison since it is the same codebase family as `lightning.gpu`), and
`lightning.gpu`.

**Measured**: wall time for 20 repeated evaluations of the same fixed-parameter
circuit (a `qml.QNode` call returning `qml.expval(qml.PauliZ(0))`), after one
untimed warm-up call (to exclude one-time device/CUDA-context setup cost from
the per-call timing) -- median, min, max.

## 3. Pre-registered predictions

**P1 (crossover exists).** At n=4, `lightning.gpu`'s median time is at least
2x slower than `lightning.qubit`'s. At n=24, `lightning.gpu`'s median time is
faster than `lightning.qubit`'s. **If GPU is not faster at n=24, GPU
acceleration is not worth pursuing further for circuits in this size range,
and this line of investigation is closed.**

**P2 (default.qubit is the slowest throughout).** `default.qubit`'s median
time is worse than `lightning.qubit`'s at every n tested -- PennyLane's own
documentation describes `lightning.qubit` as the optimized CPU backend, so
this is close to a sanity check, not a novel claim.

**P3 (where Addendum 122's own circuit lands).** Addendum 122's own
`D_deep_red` circuit uses n=6 (the 2x3 Heisenberg lattice). Interpolating
this sweep's own n=4 and n=8 results, GPU is predicted to still be slower
than CPU at that size -- i.e., **the GPU would not have helped Addendum
119-126's own actual experiments**, which used n=6 or n=8. This is registered
so the sweep's own relevance to this project's prior work is stated before,
not after, seeing the result.

## 4. What this cannot establish

- Noisy (density-matrix) simulation speed -- this experiment uses exact
  statevector simulation only; Addenda 119-126's own noisy runs used
  `AerSimulator`, not any PennyLane-native device, so this is not a direct
  substitute for that comparison.
- Gradient computation speed (parameter-shift or adjoint) -- only forward
  evaluation is timed.
- Larger qubit counts than 32, or other ansatz shapes.

---

<!-- ===== Addendum 138 (source: spare-qubit-cliff-addendum-138-2026-09-22.md) ===== -->

> **Note added when merging:** Crossover confirmed: GPU is slower at n<=16, faster from n=20 (27x at n=24). But this means Addenda 119-126's own circuits (n=6-8) would NOT have benefited -- confirmed both by this sweep (still CPU-favoured at n=16) and independently by those experiments using a different simulation method (AerSimulator noisy density-matrix, not exact statevector).

## Addendum 138 -- lightning.gpu does beat CPU, but only from n=20 onward, and gets dramatically better with size (27x faster than lightning.qubit at n=24) -- exactly the regime Addenda 119-126's own circuits (n=6-8) never reached, so the crossover was correctly predicted to miss that prior work (2026-09-22)

**Pre-registered in**:
`spare-qubit-cliff-addendum-137-preregistration-2026-09-22.md`, written and
locked before this run, revised once (n cap lowered from 32 to 24 for a
memory-budget reason found before running anything, recorded in that
document).

## 0. In one line

**P1 partly confirmed (the crossover exists and is real, but the specific
n=4 threshold in the prediction was not met); P2 mostly confirmed (one
exception at n=4); P3 confirmed.** `lightning.gpu` is slower than CPU
(`lightning.qubit`) at n=4, 8 and 16, crosses over between n=16 and n=20, and
by n=24 is **27x faster** than CPU (141.7ms vs 3,862.5ms) and **149x faster**
than `default.qubit` (21,093.2ms). Interpolating this sweep, GPU was
predicted to still be slower than CPU at n=6 -- confirmed by the sweep's own
shape: GPU is still 1.03-3.22x slower even at n=4, 8, 16, well above n=6. **So
Addenda 119-126's own noisy-training experiments (n=6-8) would not have been
sped up by this GPU**, even setting aside that those experiments used
`AerSimulator` (density-matrix, noisy), not any PennyLane-native statevector
device -- a second, independent reason the GPU path does not apply to that
prior work, stated in the pre-registration and confirmed here.

## 1. Results

Median of 20 repeats after 1 untimed warm-up call; redundant-block ansatz
(`ry`, `rz` per qubit per layer, `cx` on disjoint pairs), 2 layers, no noise.

| n | default.qubit | lightning.qubit | lightning.gpu | GPU / CPU (lightning.qubit) |
|---:|---:|---:|---:|---:|
| 4 | 1.587 ms | 4.688 ms | 4.807 ms | 1.03x (CPU faster) |
| 8 | 2.783 ms | 2.154 ms | 6.926 ms | 3.22x (CPU faster) |
| 16 | 13.275 ms | 7.993 ms | 8.468 ms | 1.06x (CPU faster) |
| 20 | 925.394 ms | 90.224 ms | 12.784 ms | **0.142x (GPU faster)** |
| 24 | 21,093.223 ms | 3,862.459 ms | 141.672 ms | **0.037x (GPU faster)** |

## 2. Scoring

**P1 (crossover exists) -- PARTLY CONFIRMED.** The crossover is real and
sharp: CPU-favoured at n<=16 (ratios 1.03-3.22x), GPU-favoured at n>=20
(ratios 0.142x, then 0.037x) -- getting more favourable to GPU as n grows,
exactly the shape expected of fixed per-call overhead being amortized over a
growing workload. The specific pre-registered threshold for the CPU side
("at least 2x slower at n=4") was not met (1.03x, not >=2x) -- reported as a
miss on that specific number, though the qualitative prediction (small
circuits favour CPU, large ones favour GPU) held.

**P2 (default.qubit slowest throughout) -- CONFIRMED except at n=4.** At n=4,
`default.qubit` (1.587ms) was faster than `lightning.qubit` (4.688ms) --
consistent with `lightning.qubit`'s own fixed per-call setup cost not yet
being amortized at this trivially small size. At every larger n,
`default.qubit` was the slowest, as predicted, and by a widening margin
(149x slower than `lightning.gpu` at n=24).

**P3 (Addendum 121/122's own scale would not have benefited) -- CONFIRMED.**
Log-linear interpolation between this sweep's n=4 and n=8 points estimates a
GPU/CPU ratio of about 1.82x at n=6 -- CPU still faster. Addendum 121/122's
`D_deep_red` circuit (n=6) sits inside the CPU-favoured region this sweep
measured directly (n=4, 8, 16 all CPU-favoured), not only by interpolation.

## 3. What this means

- **The GPU path is real and substantial, but only pays off above roughly
  n=16-20 for this ansatz shape** -- below that, the fixed cost of moving
  data to and from the GPU dominates, exactly the general pattern stated as
  expected before this experiment ran.
- **This project's own prior noisy-training work (Addenda 119-126) is
  unaffected**, for two independent reasons: it never reached the qubit
  count where GPU wins (n=6-8, confirmed CPU-favoured here), and it used a
  different simulation method entirely (`AerSimulator`'s noisy
  density-matrix simulation, not exact statevector simulation on any
  PennyLane-native device).
- **Where this GPU path WOULD matter**: any future experiment on this
  project's own larger grid instances (the 6x7/8x8 dense-pairs and
  chain-shaped families used throughout Papers 1-2 and Addenda 88-136, at
  n=42-64) sits far above the n=20 crossover found here -- if a noiseless,
  exact-statevector check at that scale were ever needed (none of this
  project's own addenda so far have needed one; Addendum 122's own n=6 cap
  was chosen specifically because exact statevector simulation is
  infeasible at 42+ qubits regardless of CPU or GPU -- 2^42 amplitudes is
  about 70 TB, far beyond what even this GPU's 12 GB affords), this GPU path
  would need revisiting at that scale, though whether it fits in 12 GB VRAM
  at all depends on n far more than on device choice.

## 4. What this does not establish

- Noisy (density-matrix) simulation speed on GPU -- untested; Addenda
  119-126's own `AerSimulator` noisy runs are a different code path than
  what this experiment measured (see the WSL/GPU handoff notes from this
  session for why `qiskit-aer-gpu` itself could not be made to work with this
  project's Qiskit version).
- Gradient computation (parameter-shift or adjoint) speed -- only forward
  evaluation was timed.
- n between 24 and where a 42-64 qubit exact statevector would fit (it does
  not, on 12 GB, regardless of device) -- the practical ceiling for exact
  simulation on this GPU was not located precisely, only shown to be above
  24 and (trivially, from the memory calculation in Addendum 137) below 33.
- Whether `lightning.gpu`'s own advantage generalizes to circuit shapes
  other than this repeated-block ansatz.

## 5. Files

| File | What it is |
|---|---|
| [`bench_lightning_gpu.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_lightning_gpu.py) | this run's script |
| [`lightning_gpu_bench_2026-09-22.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/lightning_gpu_bench_2026-09-22.csv) | raw results, 15 rows |
| [`spare-qubit-cliff-addendum-137-preregistration-2026-09-22.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-137-preregistration-2026-09-22.md) | the predictions scored above |

## 6. Verification

- All ratios recomputed directly from the printed medians, not read off the
  script's own printed ratio line uncritically (cross-checked and found to
  match).
- The n=6 interpolation computed log-linearly between the sweep's own n=4
  and n=8 points, stated as an interpolation, not a measurement.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

<!-- ===== Addendum 139 pre-registration (source: spare-qubit-cliff-addendum-139-preregistration-2026-09-23.md) ===== -->

> **Note added when merging:** Chains PSF-Zero's compilation with real lightning.gpu execution for the first time (Qiskit compile-once vs PSF-Zero, at n=20/24, the GPU-favoured sizes from Addendum 138). Explicitly not a claim about NVIDIA partnership -- uses the public pennylane-lightning[gpu] package.

## Addendum 139 -- Pre-registration: does chaining PSF-Zero's own compilation with lightning.gpu execution actually help, at a scale where both are real? (2026-09-23)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 138 measured PSF-Zero's compilation and GPU execution as two
separate questions. This experiment chains them for the first time:
compile a circuit with PSF-Zero, then actually execute the compiled result
on `lightning.gpu`, and compare the combined (compile + execute) time
against a compile-once-with-Qiskit-then-execute-on-CPU baseline -- the same
kind of comparison Addenda 110-126 made for the training loop, but now with
GPU execution as one of the arms, at a scale small enough for exact
statevector simulation to actually be possible (Addendum 137/138's own
24-qubit ceiling on 12 GB VRAM).

**This is deliberately NOT a claim about NVIDIA partnership or anything
beyond what is measured here.** It uses `pennylane-lightning[gpu]`, a
public package Xanadu (PennyLane's maintainer) has shipped for years; this
project has built no NVIDIA-specific integration.

## 2. Design

**Circuit**: the redundant-block ansatz from Addendum 137/138
(`ry`/`rz` per qubit per layer, `cx` on disjoint pairs), at n=20 and n=24 --
the two sizes Addendum 138 found GPU-favoured -- with enough layers (6, matching
Addendum 121's own "deep" depth) that PSF-Zero's own re-synthesis has
redundant CXs to remove (a `same_pair`-style block: several layers on the
same disjoint pairs, not alternating).

**Compilation, once per size** (not timed against each other here --
Addendum 110-118 already measured PSF-Zero-vs-Qiskit compile time
extensively; this experiment's own question is what happens AFTER
compilation):
- **A (compile once, Qiskit)**: `transpile()` on a grid coupling map
  matching n, `optimization_level=3`.
- **D (PSF-Zero)**: `psf_smart_layout` + `psf_compile.compile(entangling_basis="cx")`,
  the verified post-Addendum-116 path.

**Execution, the actual new measurement**: each compiled circuit's own
resulting PennyLane-equivalent operations, run as a QNode, 10 repeats after
1 warm-up, on two devices: `lightning.qubit` (CPU) and `lightning.gpu`
(GPU) -- four combinations total per n (A-CPU, A-GPU, D-CPU, D-GPU).

**Measured**: execution time only (compilation is a one-time cost already
characterized elsewhere); two-qubit gate count of each compiled circuit
(to confirm PSF-Zero's own CX reduction, as in Addenda 110-118); exact
agreement between all four combinations' own expectation values (since no
noise model is used here, all four should compute the identical answer).

## 3. Pre-registered predictions

**P1 (GPU still wins at these sizes, regardless of compiler).** At both
n=20 and n=24, `lightning.gpu` execution is faster than `lightning.qubit`
execution for BOTH compiled circuits (A and D) -- replicating Addendum
138's own crossover, now confirmed on PSF-Zero's own compiled output
specifically, not only on the hand-written ansatz Addendum 138 used
directly.

**P2 (PSF-Zero's CX reduction still holds at this circuit shape).** D's
two-qubit gate count is lower than A's at both sizes, consistent with
Addenda 110-118's own repeated finding on same-pair-style ansatze.

**P3 (does PSF-Zero's own reduction translate to faster GPU execution
too, not just fewer gates)?** D's `lightning.gpu` execution time is faster
than A's `lightning.gpu` execution time, at both sizes -- i.e., the CX
reduction that helps CPU execution (Addendum 122's own execution-noise
argument, though that used a noisy simulator) also helps exact GPU
execution here, where the mechanism would have to be raw gate count/depth
rather than noise. **This is the addendum's central, least-certain
prediction** -- registered honestly as uncertain, since Addendum 138 never
tested whether GPU execution time scales with gate count the same way CPU
time does.

**P4 (correctness).** All four (A-CPU, A-GPU, D-CPU, D-GPU) expectation
values agree to within 1e-9 at both sizes -- no noise model is used, so
they should be identical up to floating-point precision.

## 4. What this cannot establish

- Any noise model -- exact simulation only, as in Addendum 137/138.
- Scale beyond n=24 -- the same 12 GB VRAM ceiling as Addendum 137/138
  applies.
- Anything about NVIDIA, CUDA-Q, or any partnership -- this measures one
  public software combination on one local GPU.

---

<!-- ===== Addendum 140 (source: spare-qubit-cliff-addendum-140-2026-09-23.md) ===== -->

> **Note added when merging:** GPU wins 28-31x regardless of compiler (P1, P4 confirmed) -- but PSF-Zero produced the SAME gate count as Qiskit here, not fewer (P2 falsified, unlike Addenda 110-118), so the central question (does CX reduction help GPU execution) has nothing to test on this run. Documents two bugs found and fixed first: device sizing across a larger coupling map, and measuring the wrong physical wire after layout (same fix pattern as Addendum 103/122).

## Addendum 140 -- Chaining PSF-Zero's compilation with real GPU execution: GPU wins by 28-31x regardless of which compiler produced the circuit, but PSF-Zero produced no CX reduction on this ansatz shape -- so P3 (does PSF-Zero's reduction help GPU execution too) has no effect to test, contradicting this project's own repeated finding elsewhere (2026-09-23)

**Pre-registered in**:
`spare-qubit-cliff-addendum-139-preregistration-2026-09-23.md`, written and
locked before this run. Two bugs found and fixed after the first two runs,
both before scoring any prediction (Section 3).

## 0. In one line

**P1 and P4 confirmed cleanly; P2 falsified; P3 registered as this
addendum's least-certain prediction and, as a direct consequence of P2's
failure, not meaningfully testable here.** `lightning.gpu` beat
`lightning.qubit` by 28-31x on BOTH the Qiskit-compiled and the
PSF-Zero-compiled circuit, at both n=20 and n=24 -- confirming Addendum
138's crossover generalizes to real compiled output, not only the
hand-written ansatz that experiment used directly. All four
compile-x-device combinations agreed to machine precision (worst 6.84e-13)
after a routing bug was fixed. **But PSF-Zero produced the SAME two-qubit
gate count as Qiskit's own compile-once (30 and 36, both sizes) -- not
fewer**, contradicting Addenda 110-118's repeated finding on same-pair-style
ansatze. With no gate-count difference between A and D, P3's own question
(does PSF-Zero's reduction translate into faster GPU execution) has nothing
to act on in this run; the near-parity result (D/A ratios 0.998x and 1.026x)
is consistent with equal circuits executing at equal speed, not with a
tested-and-confirmed or tested-and-falsified claim about reduction
translating to speed.

## 1. Two bugs, found and fixed before this result

1. **Device sizing (found on the first run).** `smart_vf2_layout` and
   Qiskit's own layout pass do not guarantee physical qubit indices 0..n-1
   for an n-qubit circuit on a larger coupling map (a 5x5=25-qubit grid, used
   to fit n=20/24) -- both compiled circuits are sized to the FULL device
   (25 qubits), and the execution device must match. Fixed by sizing every
   PennyLane device to the coupling map's own qubit count, not the logical
   circuit's.
2. **Wrong measured wire (found on the second run).** Even after fix 1, route
   A returned exactly 1.0 at both n=20 and n=24 -- a value a random-angle
   circuit essentially never produces by chance. Cause: `qml.PauliZ(0)` was
   measuring physical wire 0, not logical qubit 0's actual physical location
   -- for a 20-qubit circuit on a 25-qubit device, physical wire 0 can be an
   idle ancilla that never receives a gate, whose Z expectation is trivially
   1.0. Fixed using the same layout-tracking method already established in
   this project (Addendum 103/122): `TranspileLayout.final_index_layout()`
   for route A, the layout search's own `perm` for route D, each giving the
   physical wire logical qubit 0 actually landed on.

## 2. Results

Median of 10 repeats after 1 untimed warm-up; redundant-block ansatz (6
layers, same-pair CXs, the shape Addenda 109-118 used).

| n | route | 2q gates | lightning.qubit | lightning.gpu | GPU/CPU |
|---:|---|---:|---:|---:|---:|
| 20 | A (Qiskit) | 30 | 19,172.0 ms | 678.4 ms | 0.0354x |
| | D (PSF-Zero) | 30 | 21,008.9 ms | 677.2 ms | 0.0322x |
| 24 | A (Qiskit) | 36 | 24,324.7 ms | 794.1 ms | 0.0326x |
| | D (PSF-Zero) | 36 | 24,650.9 ms | 814.5 ms | 0.0330x |

All four values at each n agreed to machine precision (n=20: 2.78e-15; n=24:
6.84e-13).

## 3. Scoring

**P1 (GPU wins regardless of compiler) -- CONFIRMED, all four
combinations.** GPU/CPU ratios 0.0322-0.0354x, i.e. 28-31x faster --
consistent with, and slightly stronger than, Addendum 138's own n=20 and
n=24 figures on the hand-written ansatz (there: 0.142x and 0.037x for
`lightning.qubit` vs `lightning.gpu` medians; the somewhat larger speedup
here may reflect this circuit's different gate mix, not investigated
further).

**P2 (PSF-Zero's own CX reduction) -- FALSIFIED.** 30 vs 30 at n=20, 36 vs 36
at n=24 -- identical, not fewer. This is a genuine miss against Addenda
110-118's own repeated finding of CX reduction on same-pair ansatze, and is
reported as a miss rather than reconciled after the fact. A plausible but
unverified reason: this script's `build_qiskit_ansatz` draws EACH layer's
`ry`/`rz` angles independently at random (`rng.uniform` called fresh inside
the layer loop), rather than repeating the identical single-qubit rotation
across layers the way Addenda 110-118's own ansatz construction did -- if
consolidation depends on the local single-qubit rotations combining in a
way that leaves few net parameters (not merely on the CX pattern), randomly
varying angles per layer could change how much a Cartan-based re-synthesis
has to remove. This is a hypothesis about this script's own construction,
not a re-measurement -- not confirmed here.

**P3 (does PSF-Zero's reduction help GPU execution) -- NOT MEANINGFULLY
TESTABLE this run, as a direct consequence of P2's failure.** With A and D
producing the identical gate count, the near-parity GPU timing (0.998x at
n=20, 1.026x at n=24) shows equal circuits run at equal speed on the GPU --
unsurprising and uninformative about whether a REAL reduction (had one
occurred) would have helped. Registered honestly as this addendum's
least-certain prediction in advance; that caution was warranted, though not
for the reason anticipated (the mechanism was never exercised, rather than
being exercised and failing).

**P4 (correctness) -- CONFIRMED**, to 2.78e-15 and 6.84e-13.

## 4. What this means

- **The GPU crossover from Addendum 138 is real and robust**: it shows up
  identically whether the executed circuit came from Qiskit's own compiler
  or from PSF-Zero's, at 28-31x here.
- **This experiment does not show PSF-Zero producing a better circuit than
  Qiskit's compile-once on this specific construction** -- unlike Addenda
  110-118, which used a different ansatz-building pattern (repeated
  identical rotations per pair across layers) and found real CX reduction
  there. Whether this run's own random-per-layer angle construction is the
  reason is a hypothesis, not yet checked.
- **The central question this addendum set out to answer (does PSF-Zero's
  reduction help GPU too) remains open**, pending a re-run using the
  construction Addenda 110-118 actually used, where a real CX reduction is
  expected to occur.

## 5. What this does not establish

- Whether P3 would hold with an ansatz that actually produces a CX
  reduction -- the open item for a follow-up.
- Why this run's construction produced no reduction -- a hypothesis
  (Section 3) stated but not verified.
- Anything about NVIDIA, CUDA-Q, or partnership -- unchanged from the
  pre-registration.

## 6. Files

| File | What it is |
|---|---|
| [`bench_compile_then_gpu.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_compile_then_gpu.py) | this run's script (two fixes applied, documented in its own comments) |
| [`compile_then_gpu_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/compile_then_gpu_2026-09-23.csv) | raw results, 8 rows |
| [`spare-qubit-cliff-addendum-139-preregistration-2026-09-23.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-139-preregistration-2026-09-23.md) | the predictions scored above |

## 7. Verification

- All ratios recomputed directly from the printed medians.
- The two bugs' root causes were confirmed from the actual error messages
  and printed diagnostic values (the exact 1.0, the specific wire indices
  6 and 0), not assumed.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

**End of Part 8 of 8 (end of document, for now).** Back to [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
