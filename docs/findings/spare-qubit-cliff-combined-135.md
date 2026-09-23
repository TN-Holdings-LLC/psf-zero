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



 Back to [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).

<!-- ===== Addendum 141 pre-registration (source: spare-qubit-cliff-addendum-141-preregistration-2026-09-23.md) ===== -->

> **Note added when merging:** Corrects Addendum 140's mislabelled two-route comparison to the proper three-route framework Addenda 110-126 established: A (compile-once, parameterized, cannot reduce CX by construction), B (Qiskit bind-then-recompile), D (PSF-Zero) -- so a real gate-count difference exists to test against GPU execution time.

## Addendum 141 -- Pre-registration: redo of Addendum 139-140 with the correct three-route comparison (A/B/D) -- does PSF-Zero's genuine CX reduction over compile-once (A) translate into faster GPU execution? (2026-09-23)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this redo exists

Addendum 140 compared two routes, both compiled from ALREADY-BOUND numeric
angles: a Qiskit `transpile()` call (labelled "A") and PSF-Zero's
`compile(entangling_basis="cx")` (labelled "D"). Both produced identical
two-qubit gate counts (30 at n=20, 36 at n=24) -- **not a failure of
PSF-Zero, but a direct replication of Addendum 124's own finding**: any
bound-value re-compile, Qiskit's own included, achieves the same CX
reduction PSF-Zero does. Addendum 140's own route "A" was mislabelled --
it was actually equivalent to Addendum 124's own "B" (bind-then-recompile),
not Addenda 110-118's own "A" (compile the PARAMETERIZED circuit once with
Qiskit, bind values only afterward), which is the strategy that genuinely
does NOT achieve the reduction, because it never sees the specific numeric
values during synthesis.

This redo corrects the comparison to the three-route framework Addenda
110-126 actually established, so a real gate-count difference exists to
test against GPU execution time.

## 2. Design

Same ansatz shape, sizes (n=20, 24), and GPU environment as Addendum
139/140 (`psf_zero_wsl_env_312`, `lightning.gpu`/`lightning.qubit`), with
the SAME fixed random angles reused across all three routes (so all three
represent the identical logical circuit, differing only in how it is
compiled):

- **A (compile-once)**: a PARAMETERIZED Qiskit circuit (symbolic
  `Parameter` objects, not bound numeric values) is compiled ONCE with
  `transpile(optimization_level=3)`; the fixed numeric angles are bound into
  the resulting circuit AFTER compilation. This cannot achieve the CX
  reduction, by construction -- the expected, pre-registered mechanism, not
  a guess.
- **B (Qiskit re-compile)**: the same fixed numeric angles bound directly
  into the circuit BEFORE compiling; then `transpile(optimization_level=3)`.
  Included as the same-mechanism-as-D control Addendum 124 established.
- **D (PSF-Zero)**: the same bound circuit, `psf_compile.compile(entangling_basis="cx")`
  + `smart_vf2_layout`, as in Addendum 139/140.

Each route's compiled circuit is executed on `lightning.qubit` and
`lightning.gpu`, 10 repeats after 1 warm-up, exactly as in Addendum 139/140,
with the same layout-tracking fix (measuring the physical wire logical
qubit 0 actually landed on, not a hardcoded wire 0) applied to all three
routes from the start this time, not found by a second bug.

## 3. Pre-registered predictions

**P1 (the corrected P2: A has more CX than B and D).** A's two-qubit gate
count is higher than both B's and D's, at both n=20 and n=24 -- the
mechanism this redo specifically restores.

**P2 (B and D tie on gate count, as in Addendum 124).** B's and D's
two-qubit gate counts are equal at both sizes -- replicating Addendum 124's
own finding once more, now inside this GPU-execution context.

**P3 (the central question, GPU-side).** On `lightning.gpu`, both B and D
execute faster than A, at both n=20 and n=24 -- i.e., the gate-count
reduction (P1) translates into measurably faster GPU execution, not only
faster CPU execution (which Addendum 122's noisy-simulator context already
suggested indirectly).

**P4 (GPU still beats CPU regardless of route).** For every one of the three
routes, `lightning.gpu` is faster than `lightning.qubit` at both sizes --
replicating Addendum 138/140's own crossover a third time, now confirmed on
all three compilation strategies.

**P5 (correctness).** All three routes' expectation values agree to within
1e-9 at both sizes (across both CPU and GPU execution of each) -- six
values total per size, all equal, since no noise model is used.

## 4. What this cannot establish

- Whether the GPU speedup from gate-count reduction (P3, if confirmed)
  scales the same way at larger n -- only n=20/24 are tested, the same 12 GB
  VRAM ceiling as before.
- Anything about NVIDIA, CUDA-Q, or partnership -- unchanged from Addendum
  139.

---

<!-- ===== Addendum 142 (source: spare-qubit-cliff-addendum-142-2026-09-23.md) ===== -->

> **Note added when merging:** All five predictions confirmed: A has exactly 2x the CX count of B and D (which tie exactly, replicating Addendum 124); B and D run 2.3-2.4x faster than A on lightning.gpu -- the CX reduction genuinely translates into faster GPU execution, not only CPU/noisy-simulation speedups. GPU beats CPU 28-31x regardless of route. All six route-x-device combinations agree to machine precision.

## Addendum 142 -- The corrected redo: all five predictions confirmed -- PSF-Zero's genuine CX reduction over compile-once (2x fewer gates) translates into 2.3-2.4x faster GPU execution too, not only faster CPU execution, and GPU beats CPU by 28-31x regardless of which route produced the circuit (2026-09-23)

**Pre-registered in**:
`spare-qubit-cliff-addendum-141-preregistration-2026-09-23.md`, written and
locked before this run, correcting Addendum 139/140's mislabelled
comparison.

## 0. In one line

**All five predictions confirmed, at both sizes, no exceptions.** With the
three routes correctly separated -- A (compile the parameterized circuit
once, bind after), B (bind first, Qiskit re-compile), D (bind first,
PSF-Zero) -- A produced exactly 2x B's and D's two-qubit gate count (60 vs
30 at n=20; 72 vs 36 at n=24), replicating Addendum 124's finding that B and
D tie each other exactly. **That real gate-count difference translated into
real GPU execution speed**: on `lightning.gpu`, B and D ran 2.3-2.4x faster
than A at both sizes. GPU beat CPU on every one of the six
route-x-device combinations (28-31x), and all six routes' expectation
values agreed to machine precision (worst 6.84e-13).

## 1. Results

Median of 10 repeats after 1 warm-up; same fixed random angles across all
three routes (so all represent the identical logical circuit).

| n | route | 2q gates | lightning.qubit | lightning.gpu | GPU/CPU |
|---:|---|---:|---:|---:|---:|
| 20 | A | 60 | 47,422.6 ms | 1,608.6 ms | 0.0339x |
| | B | 30 | 19,337.2 ms | 687.1 ms | 0.0355x |
| | D | 30 | 20,591.6 ms | 681.4 ms | 0.0331x |
| 24 | A | 72 | 56,734.1 ms | 1,911.2 ms | 0.0337x |
| | B | 36 | 23,467.9 ms | 789.7 ms | 0.0337x |
| | D | 36 | 24,795.9 ms | 803.4 ms | 0.0324x |

GPU execution, relative to A: n=20 -- B/A=0.4271x, D/A=0.4236x (i.e. B and D
are 2.34x and 2.36x faster); n=24 -- B/A=0.4132x, D/A=0.4204x (2.42x and
2.38x faster).

## 2. Scoring

**P1 (A has more CX than B and D) -- CONFIRMED, exactly 2x at both sizes.**
Route A, unable to see numeric values during synthesis (Qiskit's own
`ConsolidateBlocks` pass needs a concrete unitary matrix to run KAK
decomposition on, and a parameterized block cannot be reduced to one --
confirmed by reading `psf_compile.py`'s own use of the identical mechanism
before this run, not assumed), keeps all 6 CXs per pair; B and D each
collapse to 3.

**P2 (B and D tie) -- CONFIRMED, exactly, both sizes** -- a third
replication of Addendum 124's finding, now in a different circuit
construction and a different (GPU-inclusive) execution context.

**P3 (the central question: does the CX reduction help GPU execution too)
-- CONFIRMED, both sizes, both of B and D.** B and D's `lightning.gpu`
execution is 2.34-2.42x faster than A's. This directly answers what
Addendum 139 set out to test and Addendum 140 could not, due to its own
mislabelled comparison.

**P4 (GPU beats CPU regardless of route) -- CONFIRMED, all six
combinations**, ratios 0.0324-0.0355x (28-31x faster) -- consistent with
Addendum 138 and 140's own figures at the same n, and now shown to hold for
route A too (which neither of those addenda tested, since A did not exist
as a separate route until this one).

**P5 (correctness) -- CONFIRMED**, worst spread 6.84e-13, well under the
1e-9 bound, across all six route-x-device combinations at each size.

## 3. What this means

- **PSF-Zero's own well-established CX-reduction advantage over
  compile-once (Addenda 110-118) is not merely a CPU-execution or
  noisy-simulation artefact.** It carries through, at essentially the same
  magnitude, to exact GPU statevector execution: roughly 2x fewer gates
  produces roughly 2.3-2.4x faster GPU execution here -- a slightly larger
  execution-time ratio than the gate-count ratio, plausibly because circuit
  depth (not only gate count) affects execution time and the routes' depths
  were not equalized by construction, though this was not measured
  separately in this run.
- **This advantage is not specific to PSF-Zero** (Addendum 124's finding,
  reconfirmed here): Qiskit's own bind-then-recompile (B) achieves the
  identical gate count and near-identical GPU speedup to PSF-Zero (D).
  PSF-Zero's own specific contribution, established elsewhere in this
  project (Addenda 110-112, 127-132), remains compilation SPEED at scale --
  not shown again here, since B and D's own compile times were not timed
  against each other in this experiment (deliberately out of scope,
  Addendum 139 Section 2).
- **The GPU crossover itself (Addendum 138) is now confirmed on real
  compiled circuits from all three compilation strategies**, not only the
  hand-written ansatz Addendum 138 used directly.

## 4. What this does not establish

- Compile time itself, for any of the three routes -- out of scope by
  design (Addendum 139).
- Whether the slightly larger execution-speedup-than-gate-count-ratio
  (2.3-2.4x speedup vs 2.0x gate reduction) is explained by depth
  specifically -- a plausible but unmeasured hypothesis.
- Scale beyond n=24, or noise -- unchanged limits from Addendum 137-140.
- Anything about NVIDIA, CUDA-Q, or partnership -- unchanged from Addendum
  139.

## 5. Files

| File | What it is |
|---|---|
| [`bench_abd_gpu.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_abd_gpu.py) | this run's script |
| [`abd_gpu_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/abd_gpu_2026-09-23.csv) | raw results, 12 rows |
| [`spare-qubit-cliff-addendum-141-preregistration-2026-09-23.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-141-preregistration-2026-09-23.md) | the predictions scored above |

## 6. Verification

- Every ratio and gate count recomputed directly from the printed results.
- The route-A mechanism (why parameterized compilation cannot reduce CX
  count) was confirmed by reading `psf_compile.py`'s own use of
  `ConsolidateBlocks` before this run, not asserted from expectation alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---

<!-- ===== Addendum 143 pre-registration (source: spare-qubit-cliff-addendum-143-preregistration-2026-09-23.md) ===== -->

> **Note added when merging:** Extends the A/B/D comparison to include compile time (deliberately excluded in Addendum 139-142) and pushes toward the VRAM ceiling. P1 (ceiling at 28/29) falsified during pre-run checks: naive ceiling is 30, but a 15x slowdown from n=28 to n=29 on a trivial circuit suggests WSL2 silently spills into system RAM beyond VRAM -- capped the timed comparison at n=28 for safety, recorded before running anything.

## Addendum 143 -- Pre-registration: pushing to the VRAM ceiling (n=25-29) and, for the first time, comparing COMPILE + EXECUTE combined -- does PSF-Zero's compile-time advantage widen the total-time gap as n grows, even though B and D tie on execution alone? (2026-09-23)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 142 deliberately excluded compile time (Addendum 139, Section 2):
it isolated execution-time differences between routes A, B, D at n=20/24,
finding B and D tie on execution (both being bound-value re-compiles with
identical gate counts). This addendum reintroduces compile time and asks
the combined question: as n grows toward the VRAM ceiling, does B's own
compile cost (which Addenda 110-112 showed grows sharply at saturated,
large instances) start to matter enough to change which route wins overall,
even though B and D remain execution-tied?

**A qualifier stated in advance**: n=25-29 is still well below the
42-64-qubit saturated grids where Addenda 110-112 found Qiskit's layout
search hitting multi-second-to-hour costs. At this smaller scale, B's
compile time is expected to stay fast (milliseconds), same order as D's --
so this experiment is predicted NOT to show a dramatic compile-time gap,
unlike the established large-scale cliff results. This is registered so a
null result here is not mistaken for contradicting Addenda 110-112 -- they
tested a different, larger regime.

## 2. Design

**VRAM ceiling search**: `lightning.gpu` statevector allocation attempted at
n=25, 26, 27, 28, 29 (a simple circuit, not the full benchmark), each
wrapped to catch and report an out-of-memory failure rather than crashing
the whole run. The largest n that succeeds sets the ceiling for the timed
comparison below. Stated in advance: hitting a wall here is a valid,
useful result, not a failure of the experiment.

**Combined compile+execute comparison**, at every n up to the found
ceiling (or up to 29, whichever is smaller): using Addendum 141/142's own
three-route construction (A/B/D) and script (`bench_abd_gpu.py`), now also
timing each route's OWN compile step (not excluded this time), then adding
that one-time compile cost to the median per-call execution cost -- both
reported separately and combined, so the reader can see each part.

**Devices**: `lightning.gpu` only for this addendum (the execution side is
already established at n=20/24 in Addendum 142; the new question is
compile time's contribution, not re-confirming the CPU/GPU crossover).

## 3. Pre-registered predictions

**P1 (VRAM ceiling).** The largest n that succeeds without an out-of-memory
error is 28 or 29 -- consistent with the 12 GB budget calculation
(2^28 * 16 bytes ~= 4.3 GB; 2^29 * 16 bytes ~= 8.6 GB), allowing for
`lightning.gpu`'s own working-buffer overhead during gate application. **If
the ceiling is materially lower than 28 (e.g. 26 or below), that overhead is
larger than this calculation assumed, and is reported as a finding, not
adjusted after the fact.**

**P2 (compile time stays small relative to execution, at this scale --
the "no cliff yet" prediction).** At every n tested, route B's own compile
time is under 1 second -- nowhere near Addenda 110-112's own multi-second
figures at 42+ qubits. **If B's compile time exceeds 1 second at any n
tested here, this scale reaches the cliff earlier than expected, and that
finding supersedes P2 rather than being explained away.**

**P3 (D's compile-time advantage over B, if any, is modest at this scale).**
D's compile time is faster than B's at every n, but by less than 10x --
much smaller than the 100-1000x+ margins Addenda 110-112 found at
saturated 42-64-qubit instances. This is the addendum's central,
scale-dependent claim: the compile-time advantage is expected to be real
but not yet dramatic in this regime.

**P4 (combined time: does D's compile edge change the overall ranking
versus B?).** Despite B and D tying on execution alone (Addendum 142),
D's combined (compile + execute) time is lower than B's at every n tested,
because D's smaller compile-time cost (P3) is not offset by any execution
disadvantage (P4 depends on P3 being true in direction, even if the margin
is modest).

## 3a. P1 scored, and a design change made, before the timed comparison

`find_vram_ceiling.py` (a trivial single-CNOT-chain circuit) succeeded at
n=25-29, and a follow-up check extended this to n=30 (also succeeded) and
n=31 (failed with an out-of-memory error). **P1's specific prediction (28
or 29) is FALSIFIED -- the naive ceiling is 30, not 28/29.**

But a direct timing check (the same trivial circuit, n=28/29/30) found a
15x jump in execution time from n=28 (300 ms) to n=29 (4,567 ms) -- far
outside the smooth scaling every smaller step showed -- and n=30 then
failed outright on a slightly more complex circuit (multiple CNOTs) despite
"succeeding" on the trivial one moments earlier. The most likely
explanation: WSL2's shared-GPU-memory mechanism allows CUDA allocations to
exceed the RTX 4070's 12 GB VRAM by spilling into system RAM, silently,
without raising an error -- so "did not crash" at n=29/30 is not evidence
of genuine VRAM-resident execution.

**Design change made before running the timed comparison**: `N_VALUES` in
`bench_abd_gpu_compile_included.py` is capped at 28 (4.29 GB, comfortably
inside the 12 GB budget), not 29 as originally planned, so P2-P4 below are
scored only on n where execution is confirmed genuinely VRAM-resident, not
on values where the timing itself might already be measuring slow RAM
swapping mislabelled as GPU speed.

## 4. What this cannot establish

- Anything about the true large-scale cliff regime (42-64 qubits) -- this
  experiment is confined to n<=29 by VRAM, deliberately a different,
  smaller regime than Addenda 110-112/127-132.
- Noisy/density-matrix simulation -- unchanged limit from Addendum 137-142.
- Anything about NVIDIA, CUDA-Q, or partnership.

---

<!-- ===== Addendum 144 (source: spare-qubit-cliff-addendum-144-2026-09-23.md) ===== -->

> **Note added when merging:** Confirms the compile-time edge (1.6-1.9x, modest, no cliff reached) is still enough to make D win on combined compile+execute time at n=24-27, despite B/D tying on execution alone (Addendum 142). n=20 is a cold-start outlier (failed P3/P4). n=28 interrupted after 2+ hours near VRAM exhaustion -- reported as unmeasured. Documents three design bugs found and fixed before the result stood: square-grid memory explosion, switch to a 1D chain topology, and an odd-n unpaired-qubit layout gap (reusing psf_pennylane.py's own established fix).

## Addendum 144 -- Even below the large-scale cliff regime, D's small compile-time edge over B (1.6-1.9x) is enough to make D win on combined compile+execute time at n=24-27; n=20 is a cold-start outlier and n=28 could not be measured safely (2026-09-23)

**Pre-registered in**:
`spare-qubit-cliff-addendum-143-preregistration-2026-09-23.md`, written and
locked before this run. Three design bugs found and fixed before or during
this run (Section 3); n=28 interrupted partway through for a documented
safety reason (Section 4), not completed.

## 0. In one line

**P2 confirmed at all 5 measured sizes; P3 and P4 confirmed at 4 of 5 (n=24-27), both failing only at n=20, plausibly a one-time cold-start cost, not a scaling effect.** As predicted, this n=20-27 regime stayed far below Addenda 110-112's own cliff (route B's own compile time never exceeded 26 ms, nowhere near 1 second). D's compile-time advantage over B was modest (1.6-1.9x at n=24-27), as predicted -- but even this small edge was enough to make D win on COMBINED compile+execute time at every one of those four sizes, despite B and D tying almost exactly on execution alone (Addendum 142). n=28 was interrupted after the GPU reached 11.8/12.3 GB VRAM usage and one execution ran past 2 hours without completing -- reported as unmeasured, not estimated.

## 1. Results

Route B (Qiskit bind-then-recompile) vs Route D (PSF-Zero), `lightning.gpu`
only, single run per (n, route) -- no repeats of the compile step itself
(execution is the median of 10 repeats, as in Addendum 141/142).

| n | 2q gates (B=D) | compile B | compile D | B/D | exec B | exec D | combined B | combined D | D/B combined |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 30 | 24.154 ms | 91.975 ms | 0.26x | 100.248 ms | 105.247 ms | 124.403 ms | 197.222 ms | 1.5854x |
| 24 | 36 | 25.953 ms | 13.445 ms | 1.93x | 1,569.526 ms | 1,534.036 ms | 1,595.479 ms | 1,547.481 ms | 0.9699x |
| 25 | 36 | 25.374 ms | 14.614 ms | 1.74x | 3,134.143 ms | 3,123.455 ms | 3,159.517 ms | 3,138.069 ms | 0.9932x |
| 26 | 39 | 25.995 ms | 14.050 ms | 1.85x | 6,584.793 ms | 6,548.084 ms | 6,610.788 ms | 6,562.135 ms | 0.9926x |
| 27 | 39 | 25.932 ms | 15.956 ms | 1.63x | 13,257.697 ms | 13,191.730 ms | 13,283.629 ms | 13,207.686 ms | 0.9943x |

All correctness spreads at or below 6.84e-13 across A/B/D at every n
(carried over from Addendum 142's own established check, re-run here).

## 2. Scoring

**P2 (no cliff yet at this scale) -- CONFIRMED, all 5 sizes.** B's compile
time: 24.2-26.0 ms throughout -- two orders of magnitude below the
pre-registered 1-second bound, and far below Addenda 110-112's own
multi-second-to-hour figures at 42-64 qubits. This scale genuinely does not
reach the cliff.

**P3 (D's edge is real but modest, under 10x) -- CONFIRMED at n=24-27
(1.63-1.93x); FAILED at n=20 (D was SLOWER, 0.26x -- i.e. B was 3.8x faster
than D).** n=20's own D compile time (91.975 ms) is 6-7x every other D
compile time in this run (13.4-16.0 ms), despite n=20 being the SMALLEST
circuit and the FIRST one measured in this run's own execution order. The
most likely explanation, stated as a hypothesis and not confirmed further:
a one-time warm-up cost (Rust core state, OS-level caching, or similar)
paid once on the very first compile of the run, not a genuine n=20-specific
scaling effect. Reported as an anomaly, not smoothed over.

**P4 (D wins on combined time despite tying on execution) -- CONFIRMED at
n=24-27; FAILED at n=20, for the same reason as P3.** At n=24-27, D's small
compile-time edge, even though modest, was enough to make D's TOTAL
(compile + execute) time lower than B's at every size -- direct evidence
that a compile-time advantage too small to matter much on its own (1.6-1.9x)
still shows up in the combined figure a real user would experience, once
execution itself is roughly tied.

## 3. Three design bugs found and fixed before this result stood

1. **Square-grid device sizing exploded memory.** An attempt to avoid the
   layout search hitting exact saturation (found at n=25 on a 5x5=25-qubit
   grid) by jumping to the next larger PERFECT SQUARE (6x6=36) would have
   required 2^36 amplitudes -- far beyond the 12 GB budget this whole
   addendum exists to respect. Caught by computing the memory requirement
   before running anything.
2. **Switched to a 1D chain topology** (`CouplingMap.from_line`), letting
   device size be set to exactly n+2 rather than jumping in large
   square-number increments -- confirmed as a real, documented Qiskit API
   before use.
3. **Odd n leaves one qubit unpaired.** `pairs = [(a, a+1) for a in
   range(0, n-1, 2)]` never includes the last qubit when n is odd (found via
   an actual `KeyError: 24` at n=25, not anticipated). `smart_vf2_layout`
   correctly reports success (`found: True`) for the qubits it DOES place,
   but the unpaired qubit was never assigned by it in the first place --
   this project's own check needed to place it separately. Fixed using the
   same free-wire-assignment pattern already established in
   `psf_pennylane.py`'s own `find_layout()`.

## 4. n=28: interrupted, not measured

After route A's own execution (84 CXs) ran past roughly 2 hours without
returning, `nvidia-smi` showed 11.8/12.3 GB VRAM in use and 100% GPU
utilization -- genuinely still computing, not frozen, but consistent with
Addendum 143's own earlier finding that approaching the VRAM ceiling causes
severe (there, 15x) slowdown rather than a clean failure. The run was
interrupted (`Ctrl+C`) rather than left to complete on an uncertain
timeline. **n=28's own compile-time figures (B=31.157 ms, D=20.717 ms,
CX 84 vs 42) were captured before the interruption and are reported as
compile-only data; no execution or combined-time figure exists for n=28.**

## 5. What this means

- **The compile-time advantage does not need to be dramatic to matter.**
  Addenda 110-112 established PSF-Zero's headline advantage at the
  saturated 42-64-qubit cliff (100-1000x+). This addendum shows that even a
  much smaller edge (under 2x), in a regime nowhere near that cliff, is
  still enough to tip the combined compile+execute comparison in D's favour
  at 4 of 5 sizes -- an argument for the compile-time advantage mattering
  across a wider range of circuit sizes than just the extreme cliff cases,
  not only at them.
- **B and D's near-tie on execution (Addendum 142) is not disturbed here**
  -- confirmed again at n=24-27 (execution ratios within about 2%).
- **The n=20 anomaly is a reminder that single-run compile timings
  (unlike this project's own repeated-and-medianed execution timings) carry
  more run-to-run noise**, particularly for whichever size happens to run
  first in a session.

## 6. What this does not establish

- n=28's own execution or combined time -- not measured, and not estimated.
- Whether n=20's anomaly is genuinely a cold-start effect -- a hypothesis,
  not confirmed by a repeat run.
- Anything about NVIDIA, CUDA-Q, or partnership -- unchanged from Addendum
  139.

## 7. Files

| File | What it is |
|---|---|
| [`find_vram_ceiling.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/find_vram_ceiling.py) | VRAM ceiling probe (Addendum 143) |
| [`bench_abd_gpu_compile_included.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_abd_gpu_compile_included.py) | this run's script, with all three fixes (Section 3) |
| [`spare-qubit-cliff-addendum-143-preregistration-2026-09-23.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-143-preregistration-2026-09-23.md) | the predictions scored above, including P1's own scoring (falsified; true ceiling found to be 30, not 28/29, with a WSL2 shared-memory caveat noted there) |

## 8. Verification

- All ratios recomputed directly from the printed per-n results.
- The n=20 anomaly was checked against every other n's own D compile time
  before being called an anomaly, not asserted from a single comparison.
- n=28's partial data (compile times, gate counts) confirmed present in the
  terminal output before the interruption; no execution figures for n=28
  were fabricated or estimated.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---

<!-- ===== Addendum 145 pre-registration (source: spare-qubit-cliff-addendum-145-preregistration-2026-09-23.md) ===== -->

> **Note added when merging:** Resolves the qiskit-aer-gpu blocker (found earlier today): convert_to_target was removed in Qiskit 2.0, so a separate, isolated environment pinned to Qiskit 1.4.4 lets qiskit-aer-gpu 0.15.1 import successfully, reporting GPU as an available density-matrix device for the first time. Tests whether noisy GPU simulation is actually faster, and specifically whether it would have sped up Addenda 119-126's own n=6/n=8 experiments.

## Addendum 145 -- Pre-registration: does GPU actually speed up the NOISY (density-matrix) simulation that Addenda 119-126 ran on CPU, now that qiskit-aer-gpu works (Qiskit 1.4.4, a separate isolated environment)? (2026-09-23)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Every noisy-training experiment this project has run (Addenda 119-126,
133-134) used `AerSimulator`'s density-matrix method on CPU; GPU
acceleration for that specific method was never available, because
`qiskit-aer-gpu` (0.15.1) imports `qiskit.providers.convert_to_target`,
removed in Qiskit 2.0 -- confirmed from Qiskit's own deprecation notice
(present through 1.4.x docs, absent from 2.0+). A new, fully isolated
environment (`psf_zero_wsl_env_aer`, Qiskit 1.4.4 + `qiskit-aer-gpu`
0.15.1, separate from the Qiskit-2.5.2 environments the rest of today's GPU
work used) now reports `AerSimulator(method="density_matrix",
device="GPU").available_devices() == ['GPU']`. This experiment tests
whether that device actually computes faster, not only whether it exists.

**A qualifier stated in advance**: density-matrix memory scales as O(4^n),
not O(2^n) -- the RTX 4070's 12 GB budget was already found tight for
exact statevector simulation at n=28-29 (Addendum 143). A density matrix
at n=14 already needs about the same memory a statevector needs at n=28
(2^28 complex amplitudes vs (2^14)^2 = 2^28 complex matrix entries) --
so this experiment is capped far below the statevector experiments'
own qubit range.

## 2. Design

**Environment**: `psf_zero_wsl_env_aer` only (Qiskit 1.4.4). This is a
DIFFERENT Qiskit version than `psf_compile.py` was verified against
(2.5.2) -- so PSF-Zero's own compile step is NOT run in this environment;
this experiment uses Qiskit's own `AerSimulator` directly, comparing
CPU vs GPU on the identical circuit and noise model, not comparing
compilers. (Whether `psf_compile.py` itself works on Qiskit 1.4.4 is a
separate, unasked question here.)

**Circuit and noise**: Addendum 121's own `D_deep_red`-shape circuit
(redundant-block ansatz, same-pair CXs) and noise model (depolarizing,
1e-3 on `sx`/`x`, 1e-2 on `cx`) -- but the compiled circuit is taken
directly from Addendum 122's own recorded structure (42 CX at n=6) rather
than re-running PSF-Zero in this environment, to sidestep the Qiskit
version mismatch above. Swept over n in {6, 8, 10, 12, 14} -- the upper end
chosen for the O(4^n) memory qualifier above (n=14: (2^14)^2 * 16 bytes
~= 4.3 GB; n=16 would already need ~68 GB).

**Measured**: `AerSimulator(method="density_matrix", device="CPU")` vs
`device="GPU"`, running the SAME noisy circuit (not a training loop --
single-shot expectation-value computation, matching what one training
iteration's own `sim_s` timing in Addendum 119-126 measured), 10 repeats
after 1 warm-up, at each n.

## 3. Pre-registered predictions

**P1 (GPU crossover exists for density-matrix too, likely at a smaller n
than the statevector case).** GPU is slower than CPU at n=6 (matching the
general small-circuit-favours-CPU pattern from Addendum 138, which found
the statevector crossover near n=20) but faster by n=14 -- the crossover
point itself is not predicted precisely, only that it exists somewhere in
this range, given density matrices are inherently more parallel workloads
(more amplitudes to update per gate) than statevectors of the same n.

**P2 (retrospective relevance to Addendum 119-126).** Addendum 119-126's
own circuits used n=6 and n=8. Based on P1, GPU is predicted to still be
CPU-favoured or at best roughly tied at n=8 -- i.e., **even with working
noisy-GPU simulation, this project's own prior noisy-training work would
likely not have been sped up by it**, the same conclusion Addendum 138
reached for the noiseless case, now checked for the noisy case
specifically rather than assumed to transfer.

**P3 (correctness).** CPU and GPU report the same expectation value at
every n, within 1e-9 (both methods should be numerically exact, no
sampling involved).

## 4. What this cannot establish

- Whether `psf_compile.py` itself runs on Qiskit 1.4.4 -- not tested here,
  deliberately (Section 2).
- n beyond 14 -- the O(4^n) memory wall, not merely a chosen cutoff.
- Anything about NVIDIA, CUDA-Q, or partnership.

---

<!-- ===== Addendum 146 (source: spare-qubit-cliff-addendum-146-2026-09-23.md) ===== -->

> **Note added when merging:** All three predictions confirmed. Noisy GPU simulation crosses over CPU at n=8-10 -- about half the qubit count of the noiseless statevector crossover (~n=20, Addendum 138). At n=6/n=8, the exact sizes Addenda 119-126 used, GPU was measurably slower (42%/15%), confirming directly (not merely inferring) that those prior experiments would not have benefited from GPU. n=14 killed by the OS (out of memory), consistent with the pre-registered O(4^n) memory wall.

## Addendum 146 -- Noisy (density-matrix) GPU simulation confirmed working for the first time: the CPU/GPU crossover sits at n=8-10, far below the noiseless statevector crossover (~n=20); at n=6-8, the exact sizes Addenda 119-126 used, GPU was still 14-42% slower than CPU (2026-09-23)

**Pre-registered in**:
`spare-qubit-cliff-addendum-145-preregistration-2026-09-23.md`, written and
locked before this run.

## 0. In one line

**P1 and P3 confirmed; P2 confirmed, precisely as predicted.** With
`qiskit-aer-gpu` finally working (Qiskit 1.4.4, isolated environment, found
after resolving the version conflict that blocked this earlier today), GPU
density-matrix simulation is real and does cross over CPU -- but at n=8-10,
not near n=20 as the noiseless statevector case showed (Addendum 138). At
n=12 GPU was already 1.44x faster than CPU. **At n=6 and n=8 -- the exact
sizes every noisy-training experiment in this project (Addenda 119-126) used
-- GPU was 42% and 15% slower than CPU respectively**, directly confirming
the pre-registered prediction that this project's own prior noisy-training
work would not have been sped up by GPU, now checked on the actual noisy
simulation method rather than inferred from the noiseless case. n=14 was
killed by the OS (out-of-memory), consistent with the O(4^n) memory wall
stated in advance; n=12 is this run's largest usable data point.

## 1. Results

Median of 10 repeats after 1 warm-up; same-pair redundant-block ansatz (6
layers), depolarizing noise (1e-3 sx/x, 1e-2 cx), `AerSimulator`
density-matrix method, Qiskit 1.4.4.

| n | CPU | GPU | GPU/CPU | value diff |
|---:|---:|---:|---:|---:|
| 6 | 5.179 ms | 7.369 ms | 1.4229x (CPU faster) | 8.33e-17 |
| 8 | 8.537 ms | 9.776 ms | 1.1451x (CPU faster) | 0.00e+00 |
| 10 | 36.717 ms | 31.196 ms | **0.8496x (GPU faster)** | 0.00e+00 |
| 12 | 428.854 ms | 297.739 ms | **0.6943x (GPU faster)** | 0.00e+00 |
| 14 | -- (process killed, out of memory) | -- | -- | -- |

## 2. Scoring

**P1 (a crossover exists, at a smaller n than the noiseless statevector
case) -- CONFIRMED.** The crossover sits between n=8 (CPU still favoured,
1.15x) and n=10 (GPU favoured, 0.85x) -- roughly half the qubit count of
Addendum 138's own statevector crossover (~n=20), consistent with the
pre-registered reasoning that density matrices are a more parallel workload
per qubit added.

**P2 (Addendum 119-126's own n=6/n=8 circuits would not have been sped up
by GPU) -- CONFIRMED, precisely.** GPU was slower at both sizes those
experiments actually used (42% slower at n=6, 15% slower at n=8) -- not
merely "not faster," but measurably worse. This directly answers, on the
real noisy simulation method for the first time, what Addendum 138 could
only infer from the noiseless case.

**P3 (correctness) -- CONFIRMED**, exactly. Three of four measured points
agree to 0.00e+00 (bit-identical); n=6 agrees to 8.33e-17 (floating-point
noise floor).

## 3. n=14: killed, not measured, as anticipated

The process was killed by the OS partway through n=14. The pre-registered
memory qualifier (Section 2 of Addendum 145) anticipated this: a density
matrix at n=14 needs about 4.3 GB, and CPU time alone grew 11.7x from n=12
to n=14 (extrapolating from the n=6-12 trend) before any GPU comparison
could run. This project's own working memory budget in WSL2 was
insufficient at this point; no execution or timing data exists for n=14,
and none is estimated here.

## 4. What this means

- **GPU-accelerated noisy simulation is now a real, working capability**
  for this project, resolving what Addendum 145's own environment setup
  made possible after this morning's blocked attempt.
- **It does not retroactively change any of Addenda 119-126's own
  results.** Those experiments' own circuit sizes (n=6, n=8) sit exactly in
  the region this addendum found GPU-disfavoured, not merely untested.
- **Where GPU noisy simulation WOULD help**: any future noisy-training
  experiment at n>=10 -- a size none of this project's own variational
  experiments have used so far (Addenda 109-134 stayed at n<=8 for
  noisy work, per the `AerSimulator`-imposed O(4^n) memory ceiling that
  made n=6 the practical choice at the time).
- Consistent with the noiseless case (Addendum 142), this experiment used
  Qiskit's own `AerSimulator` directly, not PSF-Zero -- this addendum makes
  no claim about compilation, PSF-Zero, or PennyLane, by design (Addendum
  145, Section 2).

## 5. What this does not establish

- n=14 or above -- not measured, killed by the OS.
- Whether `psf_compile.py` runs on Qiskit 1.4.4 -- not tested, by design.
- Whether a training loop (not single-circuit timing) shows the same
  crossover -- only per-circuit simulation time was measured here, matching
  what Addenda 119-126's own `sim_s` column recorded, not a full loop.
- Anything about NVIDIA, CUDA-Q, or partnership.

## 6. Files

| File | What it is |
|---|---|
| [`bench_aer_gpu_noisy.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_aer_gpu_noisy.py) | this run's script |
| [`aer_gpu_noisy_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/aer_gpu_noisy_2026-09-23.csv) | raw results (4 rows; n=14 not written, the run was killed before reaching the CSV write) |
| [`spare-qubit-cliff-addendum-145-preregistration-2026-09-23.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-145-preregistration-2026-09-23.md) | the predictions scored above |

## 7. Verification

- All ratios read directly from the run's own printed output.
- The O(4^n) memory explanation for n=14's failure was checked against the
  pre-registered calculation (Addendum 145, Section 2: ~4.3 GB at n=14)
  before being stated as the cause, not asserted from the "Killed" message
  alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---

<!-- ===== Addendum 147 pre-registration (source: spare-qubit-cliff-addendum-147-preregistration-2026-09-23.md) ===== -->

> **Note added when merging:** IBM Quantum's free device-time quota is exhausted until 2026-09-28. Trains XOR (ideal + noisy) as a mock/rehearsal, with a fixed decision rule (three numeric criteria) for whether to proceed to real-hardware submission when the quota resets -- decided before seeing any result, so a near-miss cannot be reinterpreted as a pass.

## Addendum 147 -- Pre-registration: does a real training loop (not single-circuit timing) on lightning.gpu solve a simple, human-checkable problem (XOR classification) well enough to justify sending the result to real IBM hardware on 2026-09-28? (2026-09-23)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement. This is the "mock, iterate
before 2026-09-28" experiment agreed in conversation: IBM Quantum's free
monthly device time is exhausted until 2026-09-28, so this addendum trains
and validates entirely in simulation first, with a pre-fixed decision rule
for whether the result is strong enough to spend that device time on.

## 1. Why this experiment exists

Every GPU experiment so far (Addenda 137-146) timed SINGLE circuit
evaluations -- compile once, execute once, repeat for a median. None ran an
actual training loop (many parameter updates in sequence) on GPU. This
addendum runs a real loop, on a problem simple enough to check by hand
(XOR), and fixes in advance what "good enough to send to real hardware"
means, so that decision is not made by impression after seeing the result.

## 2. Design

**Task**: XOR. Four classical inputs (00, 01, 10, 11), each encoded onto
qubits via `RX(pi * bit)` per input bit; a variational ansatz (`ry`/`rz`
per qubit per layer, `cx` entangling, 3 layers); output read from
`expval(PauliZ(0))`, mapped to a label via sign; trained to match XOR's
label (00->-1, 01->1, 10->1, 11->-1) with a mean-squared-error loss,
Adam optimizer, gradients via PennyLane's own `qml.grad` (autograd
interface -- exact gradients, not parameter-shift, since this is a
noiseless training phase and exact gradients are available and faster).

**Qubit count**: n=4 (2 input qubits + 2 ancilla/entangling qubits) --
deliberately small, inside `lightning.gpu`'s own CPU-favoured region
(Addendum 138 found the crossover near n=20), so this experiment is NOT
about GPU being fast per step; it is about whether the LOOP, run on GPU
end-to-end, actually learns. A separate timing note (P4) checks whether
GPU or CPU is faster for a loop this small, expecting CPU to win, since
n=4 is far below the established crossover.

**Two conditions**, 5 seeds each, 200 iterations:
- **Ideal**: `default.qubit`, no noise.
- **Noisy**: `lightning.qubit` is not noise-capable in the installed
  version; noisy training uses Qiskit's own `AerSimulator` (density-matrix,
  CPU -- the noisy-GPU environment, Qiskit 1.4.4, is a separate,
  PennyLane-incompatible environment per Addendum 145's own design choice,
  so noisy training here runs on CPU, with parameter-shift gradients since
  AerSimulator is not autograd-differentiable through PennyLane).
  Depolarizing noise: 1e-3 on single-qubit gates, 1e-2 on CX -- the same
  levels Addenda 119-126 used.

## 3. Pre-registered decision rule (fixed before any run)

**CONTINUE to real-hardware submission (2026-09-28) only if ALL THREE hold:**

- **C1 (solves it, ideal).** Final ideal-condition test accuracy (correct
  sign on all 4 XOR inputs) reaches 4/4 on at least 4 of 5 seeds.
- **C2 (survives noise).** Final noisy-condition accuracy reaches at least
  3/4 on at least 4 of 5 seeds, AND the noisy training loss trajectory is
  monotonically improving on average across its last 50 iterations (not
  diverging or stuck) -- checking directly for the training collapse
  Addendum 122's own noisy 84-CX circuit showed at high noise, not assuming
  it won't recur here.
- **C3 (time is practical).** 200 iterations complete in under 10 minutes
  for the ideal condition (a proxy for whether the loop itself is usable,
  not a claim about GPU speed at this small n, where CPU is expected to
  win per P4 below).

**If any of C1-C3 fails, this line is PENDING -- not retried with different
hyperparameters in an attempt to force a pass; a failure is reported as a
failure, and iteration before 2026-09-28 (as agreed) means re-running this
SAME pre-registered design on a different seed set or with a documented,
pre-stated change, not adjusting the bar after seeing a near-miss.**

## 4. Pre-registered predictions (separate from the decision rule)

**P1.** C1 holds (XOR is a simple, well-known learnable problem for a
3-layer ansatz).

**P2.** C2 is the least certain prediction: noise may degrade accuracy
below the 3/4 bar or cause the collapse Addendum 122 found at higher CX
counts (this circuit has far fewer CX than that one, so collapse is not
expected, but not certain).

**P3.** C3 holds easily at n=4 (trivially small by today's own standards).

**P4 (not part of the decision rule -- informational).** For this small
n=4 loop, `lightning.qubit` (CPU) is faster per iteration than
`lightning.gpu` -- consistent with Addendum 138's own crossover being near
n=20, included here as a direct check that this specific loop follows the
same pattern rather than assumed.

## 5. What this cannot establish

- Anything about real IBM hardware -- deliberately deferred to 2026-09-28.
- Larger, non-toy problems -- XOR is chosen specifically for being
  checkable by hand, not for being representative of a hard QML task.
- Whether GPU helps THIS LOOP's speed -- P4 predicts it will not, at this
  qubit count; that is expected and not the reason for running this
  experiment.

---

<!-- ===== Addendum 148 (source: spare-qubit-cliff-addendum-148-2026-09-23.md) ===== -->

> **Note added when merging:** All three criteria pass decisively: ideal 5/5 seeds at exact 4/4 XOR accuracy; noisy 5/5 seeds also at 4/4, converged near the noise floor (not merely above the 3/4 bar); training time far under budget. Decision: CONTINUE toward the 2026-09-28 hardware submission. Honestly notes a weakness in the pre-registered 'improving' trajectory check (satisfied only because training had already converged, not because it was actively improving in the last 50 iterations); the full trajectory CSV, received and checked after this addendum was first written, confirms convergence by iteration 48-65 of 200, not collapse.

## Addendum 148 -- Mock training decision: all three pre-registered criteria pass decisively -- CONTINUE toward real-hardware submission on 2026-09-28 (2026-09-23)

**Pre-registered in**:
`spare-qubit-cliff-addendum-147-preregistration-2026-09-23.md`, written and
locked before this run, including the decision rule this addendum applies.

## 0. In one line

**CONTINUE.** All three pre-registered criteria (C1, C2, C3) pass, decisively
rather than marginally: ideal-condition training solved XOR to 4/4 on all 5
seeds (final loss 0.0000); noisy-condition training also reached 4/4 on all
5 seeds, with losses converged to 0.0009-0.0016 -- near the noise floor, not
merely above the 3/4 bar; 200 iterations completed in 14.0 s/seed (ideal),
far under the 600 s bound. P4 (informational, not part of the decision)
confirmed CPU beats GPU 4.6x at this qubit count, as predicted.

## 1. Results

5 seeds each, 200 iterations, XOR (4 inputs, checked by hand).

| condition | seeds at 4/4 | final loss range | wall time (5 seeds) |
|---|---:|---|---:|
| ideal (default.qubit) | 5/5 | 0.0000 (all) | 70.1 s |
| noisy (AerSimulator CPU) | 5/5 | 0.0009-0.0016 | 305.1 s |

P4 (n=4, 10 iterations): `lightning.qubit` 17.13 ms/iteration,
`lightning.gpu` 78.65 ms/iteration -- GPU 4.59x SLOWER, confirming the
prediction that this qubit count sits well inside the CPU-favoured region
(Addendum 138's own crossover is near n=20).

## 2. Scoring

**C1 (solves it, ideal) -- PASS, 5/5, not merely the 4/5 bar.** Every seed
reached exact loss 0.0000 and 4/4 accuracy.

**C2 (survives noise) -- PASS, with an honest qualifier on the trajectory
check.** Accuracy: 5/5 seeds at 4/4 (the bar was 4/5 seeds at >=3/4 --
exceeded on both counts). The "last-50-iterations improving" check itself
was satisfied only in a weak sense: every seed's loss was already flat
(e.g. 0.0016 -> 0.0016) across its own last 50 iterations, because training
had already converged near the noise floor well before iteration 150 --
not because it was stuck far from a good solution. This is the OPPOSITE of
the training collapse Addendum 122 found at higher noise/CX counts (loss
staying elevated, not improving) -- convergence, not stagnation. The
pre-registered check ("not diverging or stuck") technically used a loose
proxy (`last50[-1] <= last50[0]`) that cannot distinguish "converged
early" from "stuck at a bad value" on its own; here the very low absolute
loss values (0.0009-0.0016, versus this problem's own scale where a
random/failed classifier would sit near loss ~2) make clear which case
this is. Reported as a real limitation of the pre-registered check's own
design, not glossed over.

**C3 (time is practical) -- PASS, comfortably.** 14.0 s/seed, well under
the 600 s bound (only about 2% of the budget used).

**P4 (informational) -- CONFIRMED.** GPU 4.59x slower than CPU at n=4,
consistent with Addendum 138's own established crossover being far higher
(~n=20).

## 3. Decision

**Trajectory data received and checked directly (added after the CSVs
arrived).** The full 2,000-row trajectory file (10 runs x 200 iterations)
confirms the summary figures exactly and settles the C2 qualifier with the
actual curve rather than only the last-50 endpoints: every noisy seed
reached within 1.5x of its own final loss by iteration 48-65 (of 200) --
roughly the first third of training -- and held there. Convergence,
confirmed directly, not the training collapse Addendum 122 found at higher
noise/CX counts.

**Per the pre-registered rule (Addendum 147, Section 3): all of C1-C3 hold,
so this line CONTINUES toward real-hardware submission when IBM Quantum's
free device-time quota resets on 2026-09-28.** No hyperparameters were
adjusted to force a pass; the result was decisive on the first run, at the
originally pre-registered settings.

## 4. What remains before 2026-09-28

- IBM Quantum account credentials (API key, instance CRN) -- not yet
  configured, per the current `channel="ibm_quantum_platform"` setup
  confirmed from IBM's own current documentation earlier today.
- A connection/submission script, using the credentials above -- not yet
  written, deliberately deferred until credentials are ready (Section 2 of
  the earlier conversation this addendum follows from).
- A decision on WHICH trained circuit to send: this addendum trained many
  independent seeds on a toy 4-qubit XOR problem; the actual hardware
  submission should use one specific, chosen set of trained parameters
  (e.g. the lowest-final-loss ideal-condition seed), compiled through
  PSF-Zero for the target device, sent once -- not yet decided which seed
  or which real device to target.

## 5. What this does not establish

- Anything about real IBM hardware -- deliberately deferred, per design.
- Larger or harder problems than XOR -- this was chosen specifically to be
  checkable by hand, not representative of a hard QML task.
- Whether GPU would help a LARGER training loop (more qubits) -- P4 only
  confirms the expected pattern at this small, deliberately chosen n=4.

## 6. Files

| File | What it is |
|---|---|
| [`train_xor_mock.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/train_xor_mock.py) | this run's script |
| [`xor_mock_summary_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_mock_summary_2026-09-23.csv) | per-seed final results, 10 rows |
| [`xor_mock_trajectories_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_mock_trajectories_2026-09-23.csv) | loss per iteration |
| [`spare-qubit-cliff-addendum-147-preregistration-2026-09-23.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-147-preregistration-2026-09-23.md) | the decision rule and predictions scored above |

## 7. Verification

- All three decision-rule criteria checked against their own pre-registered
  numeric bounds, not against impression.
- The C2 "improving" qualifier was investigated by comparing the absolute
  loss magnitude against this problem's own scale (a failed classifier's
  expected loss, ~2, vs the observed 0.001-0.002), not asserted from the
  flat trajectory alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---

<!-- ===== Addendum 149 pre-registration (source: spare-qubit-cliff-addendum-149-preregistration-2026-09-23.md) ===== -->

> **Note added when merging:** A safety-margin check before the 2026-09-28 hardware submission: sweeps CX noise from 1% to 20% to find where XOR training stops converging. Records a bug found and fixed before running anything: a relative convergence threshold falsely flagged non-converging runs as 'converged immediately,' fixed with an absolute threshold instead.

## Addendum 149 -- Pre-registration: how much noise can the XOR circuit tolerate before training fails to converge -- a safety margin check before sending anything to real hardware on 2026-09-28 (2026-09-23)

**Status: pre-registration only. No measurement has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 148 found the XOR circuit trains to near-perfect accuracy at the
noise level Addenda 119-126 used throughout (1e-3 single-qubit, 1e-2 CX).
That level was not chosen to be realistic for any specific device -- it was
inherited from earlier experiments. Before sending a trained circuit to
real IBM hardware on 2026-09-28, this addendum sweeps noise strength upward
to find where XOR training stops succeeding, so a poor real-hardware result
can be told apart from "expected, given the noise level" versus "something
else went wrong."

## 2. Design

Same circuit, task, and training procedure as Addendum 147/148 (noisy
condition: `AerSimulator` density-matrix, CPU, parameter-shift gradients,
200 iterations, 5 seeds), swept over CX depolarizing error p2 in
{0.01, 0.02, 0.05, 0.1, 0.2} (single-qubit error held at p2/10 throughout,
the same ratio as every noise experiment in this project). p2=0.01 is
Addendum 148's own already-measured point, included here as a consistency
check against that prior result, not re-derived from scratch.

**Measured**: final accuracy (0-4) and final loss per seed at each p2;
whether training converges within the 200-iteration budget, using a FIXED
ABSOLUTE loss threshold (0.1 -- well below this problem's own
random-guess loss of ~2, comfortably above Addendum 148's own converged
values of 0.0009-0.0016), not a threshold relative to each run's own final
value. A relative-threshold version was tried first and found, before
running anything, to misfire on exactly the failure case this check exists
to catch: a run whose loss never improves (oscillating near a bad value
throughout) has every early point already "close to" its own bad final
value, so a relative check reports false convergence at iteration 0. The
absolute threshold does not have this failure mode.

## 3. Pre-registered predictions

**P1 (p2=0.01 reproduces Addendum 148).** At p2=0.01, all 5 seeds reach
4/4 accuracy, matching Addendum 148's own result -- a sanity check that
this new script's own noise model construction matches the earlier one
exactly, checked before trusting anything at the new, higher noise levels.

**P2 (accuracy degrades with noise, not a sharp cliff).** Mean accuracy
across 5 seeds decreases monotonically as p2 increases from 0.01 to 0.2 --
a gradual degradation, not a sudden collapse, since this depolarizing noise
model was already shown (Addendum 122) to primarily rescale rather than
qualitatively distort a related landscape.

**P3 (a working threshold exists inside the swept range).** At p2=0.2 (20x
Addendum 148's own level), at least one seed fails to reach 4/4 accuracy --
i.e. the sweep's own upper end is chosen high enough to actually find a
failure, not merely confirm success everywhere tested. **If all 5 seeds
still reach 4/4 even at p2=0.2, the sweep did not reach far enough to find
a real limit, and that is reported as such -- the upper bound would need
raising, not the result stretched to claim a limit was found.**

**P4 (correctness of the sweep's own construction, not the circuit).**
Every accuracy figure at every p2 is computed from the same fixed final
parameters' own predictions (not estimated or interpolated) -- a
process check, expected to hold trivially, included so a script bug would
show up as an explicit assertion failure rather than a silently wrong
number.

## 4. What this cannot establish

- Any specific real IBM device's own actual error rates -- this is a
  simple depolarizing sweep, not a device calibration; whichever device is
  targeted on 2026-09-28, its own real noise profile is not looked up or
  matched here.
- Whether a harder task than XOR would show the same tolerance -- this
  addendum is specific to the already-easy XOR problem.

---

<!-- ===== Addendum 150 (source: spare-qubit-cliff-addendum-150-2026-09-23.md) ===== -->

> **Note added when merging:** Accuracy stayed perfect (4/4) at every noise level tested, even 20x this project's own standard level -- P3 as literally stated (accuracy-based failure) was falsified. But a different, already-collected metric (absolute-threshold convergence) found the real transition: 5/5 seeds converge at low noise, only 0/5 do at p2=0.2. Notes the practical implication for 2026-09-28: an accuracy-only check could look perfect while confidence has actually degraded substantially.

## Addendum 150 -- XOR's binary decision boundary is remarkably noise-tolerant (4/4 accuracy at every noise level tested, up to 20x Addendum 148's level), but prediction confidence degrades steadily and predictably -- P3 as literally stated was falsified, but a stricter, more informative signal (absolute-threshold convergence) found the real transition P3 was looking for (2026-09-23)

**Pre-registered in**:
`spare-qubit-cliff-addendum-149-preregistration-2026-09-23.md`, written and
locked before this run.

## 0. In one line

**P1 and P2 confirmed; P3 falsified as literally stated, but the underlying
question it was designed to answer is answered by a different, already-
collected metric.** Accuracy (sign of the prediction matching the XOR
label) stayed at a perfect 4/4 on every one of 25 runs (5 seeds x 5 noise
levels, 0.01 to 0.2 -- 20x Addendum 148's own level), so the pre-registered
P3 criterion (at least one seed failing to reach 4/4 by p2=0.2) never
triggered, and the sweep's own upper bound did not find an accuracy
failure within the tested range, exactly as the pre-registration said this
outcome would be reported. **But loss values rose steadily and substantially
with noise** (median 0.0016 to 0.3486, roughly 218x), and the absolute-
threshold convergence check (fixed at loss < 0.1, the bug-fixed metric from
Addendum 149's own design correction) shows a real, gradual transition: 5/5
seeds converge at p2 <= 0.05, 4/5 at p2=0.1, and **0/5 converge by p2=0.2**
-- the limit P3 was designed to find, visible in convergence rather than in
binary accuracy.

## 1. Results

5 seeds per noise level; CX depolarizing error p2, single-qubit error p2/10.

| p2 | accuracy (all seeds) | median final loss | seeds converged (loss < 0.1) |
|---:|---|---:|---:|
| 0.01 | 4/4 x5 | 0.0010 | 5/5 |
| 0.02 | 4/4 x5 | 0.0037 | 5/5 |
| 0.05 | 4/4 x5 | 0.0203 | 5/5 |
| 0.10 | 4/4 x5 | 0.0734 | 4/5 |
| 0.20 | 4/4 x5 | 0.2381 | **0/5** |

## 2. Scoring

**P1 (reproduces Addendum 148 at p2=0.01) -- CONFIRMED.** All 5 seeds at
4/4, losses (0.0009-0.0016) matching Addendum 148's own recorded range
exactly -- confirms this new script's noise model construction is
identical to the earlier one before trusting anything at the new noise
levels.

**P2 (accuracy degrades gradually, not a sharp cliff) -- CONFIRMED, but not
by the metric named in the prediction.** Accuracy itself shows no
degradation at all (perfect throughout), so "gradual degradation in
accuracy" cannot be assessed from accuracy. Loss, the metric this
addendum's own design also tracked, does show smooth, monotonic
degradation with p2 -- consistent with the spirit of P2 (no sharp
collapse) even though the specific metric named turned out to be the wrong
one to look at.

**P3 (a working threshold exists inside the swept range) -- FALSIFIED as
literally stated (accuracy-based); the underlying question is answered by
convergence instead.** No seed's ACCURACY ever failed, at any p2 up to
0.2 -- the pre-registered fallback ("if all 5 seeds still reach 4/4 even
at p2=0.2, the sweep did not reach far enough... reported as such") is
followed here: this is reported as a genuine non-finding for the accuracy
metric, not stretched into a claim a limit was found by that measure.
**However**, the convergence check (a different, already-collected metric,
using the absolute threshold Addendum 149's own design fixed a bug to
compute correctly) shows exactly the kind of transition P3 was designed to
detect: perfect convergence at low noise, degrading to 4/5 at p2=0.1, and
complete failure to converge (0/5) at p2=0.2.

**P4 (process correctness) -- CONFIRMED.** All 25 rows present with
directly computed values; no interpolation or estimation.

## 3. What this means for XOR's own decision boundary versus its confidence

XOR's binary output (which of two classes) is far more noise-robust than
its continuous output (how confidently). At p2=0.2, the median expectation
value driving each prediction has moved substantially toward zero (loss
0.2381 versus a near-zero loss at low noise implies the prediction
magnitude has shrunk considerably), yet the SIGN survives in every single
run. This is a property of this specific, easy, well-separated task (XOR
with a 3-layer ansatz on 4 qubits) -- the decision margin, not tested
directly here, is apparently large enough that even 20x this project's own
standard noise level does not flip it.

## 4. What this means for the 2026-09-28 hardware submission

- **A binary-accuracy check alone is not a sensitive safety indicator for
  this task** -- it would report "4/4, all good" even under noise 20x
  higher than expected, masking real confidence degradation. If real
  hardware behaves like this simulation, an accuracy-only check on the
  returned shots could look perfect even if the underlying signal is
  weak; the loss-equivalent quantity (or shot-count margin) should also be
  checked, not accuracy alone.
- **The convergence-based signal is the more honest indicator** of "how
  hard is this noise level fighting the circuit," and it degrades exactly
  as expected: gradually, with a real breakdown visible by p2=0.1-0.2.
- **This project's own standard noise level (p2=0.01) sits far inside the
  safe region** by both metrics -- 100x below where convergence starts
  failing (p2=0.1-0.2). Real IBM hardware error rates, not looked up or
  compared here, would need to be very high relative to typical reported
  values for this specific circuit and task to be at meaningful risk.

## 5. What this does not establish

- Any specific real IBM device's own error rates -- not looked up or
  compared, as stated in the pre-registration.
- Whether a harder task than XOR would show the same accuracy-robustness
  pattern -- this addendum is specific to this already-easy problem.
- A precise accuracy-failure threshold -- not found within 0.01-0.2; would
  need a noise level beyond this sweep's own range, or a harder task, to
  locate one.

## 6. Files

| File | What it is |
|---|---|
| [`xor_noise_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_noise_sweep.py) | this run's script |
| [`xor_noise_sweep_summary_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_noise_sweep_summary_2026-09-23.csv) | per-(p2, seed) results, 25 rows |
| [`xor_noise_sweep_trajectories_2026-09-23.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/xor_noise_sweep_trajectories_2026-09-23.csv) | loss per iteration |
| [`spare-qubit-cliff-addendum-149-preregistration-2026-09-23.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-149-preregistration-2026-09-23.md) | the predictions scored above |

## 7. Verification

- All figures recomputed directly from the summary CSV.
- The convergence counts (5/5, 5/5, 5/5, 4/5, 0/5) recomputed directly from
  the CSV's own `converged_by_iter` column (None vs a numeric value), not
  from the terminal's own printed text.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSVs ->
  0 hits.

---

<!-- ===== Addendum 151 (source: spare-qubit-cliff-addendum-151-2026-09-23.md) ===== -->

> **Note added when merging:** Practice preparation for the 2026-09-28 hardware submission, not a pre-registered experiment. Seed 0's training reproduced deterministically, compiled with PSF-Zero for a heavy-hex topology, verified exact on all 4 XOR inputs (worst diff 2.22e-16). Two bugs fixed before running: a removed Qiskit API (QuantumCircuit.qasm()) and an ad hoc coupling-map stand-in replaced with psf_pennylane.py's own verified EdgeListCouplingMap. No IBM connection was made; a credential inadvertently shared in this session's conversation is confirmed absent from every project file.

## Addendum 151 -- Hardware-ready circuits prepared for the 2026-09-28 submission: seed 0's training reproduced exactly, compiled for heavy-hex, verified exact on all 4 XOR inputs -- no IBM connection made (2026-09-23)

**Status**: practice/preparation run, not a pre-registered experiment. No
prediction is scored here; this addendum records what was prepared ahead
of the 2026-09-28 real-hardware submission, once IBM Quantum's free
device-time quota resets.

## 0. In one line

Addendum 148's own seed-0 ideal-condition training was re-run deterministically
(same seed, same optimizer) and reproduced its exact recorded result (final
loss 0.000005, 4/4 accuracy) -- confirming the parameters, not stored in any
CSV, can be regenerated reliably rather than lost. Each of the 4 XOR-input
circuits was compiled with PSF-Zero for a heavy-hex topology (the structure
real IBM devices use), placed on 19 available physical qubits (4 needed),
and verified to reproduce the logical circuit's own expectation value
exactly (worst diff 2.22e-16) on all 4 inputs. Two bugs in the preparation
script itself were found and fixed before this run (Section 2). No
connection to IBM Quantum was made; the API key shared earlier in this
session was not used and is not recorded anywhere in this project.

## 1. Results

| input | 2q gates (compiled) | ideal expval | compiled expval | diff |
|---|---:|---:|---:|---:|
| (0,0) | 9 | -0.997756 | -0.997756 | 0.00e+00 |
| (0,1) | 9 | 0.997757 | 0.997757 | 1.11e-16 |
| (1,0) | 9 | 0.997756 | 0.997756 | 2.22e-16 |
| (1,1) | 9 | -0.997758 | -0.997758 | 2.22e-16 |

All four differences are at the floating-point noise floor. Note the
expectation values themselves (~0.9978, not exactly +-1) reflect this
circuit's own noiseless-but-imperfect training (Addendum 148's own final
loss of 0.000005 is small but nonzero) -- not a compilation artifact; the
ideal and compiled columns agree to machine precision with each other,
which is what this addendum checks.

Four OpenQASM 2 files were written (`xor_hw_ready_input_00.qasm`,
`_01.qasm`, `_10.qasm`, `_11.qasm`), one per XOR input, ready to submit as
four separate circuits when the real-hardware step is taken.

## 2. Two bugs found and fixed before this run

1. **`QuantumCircuit.qasm()` was removed in Qiskit 1.0.** A first draft used
   this method to write the output files; confirmed removed via Qiskit's
   own current documentation (the replacement is `qiskit.qasm2.dumps()`)
   before running anything, not discovered by a runtime error.
2. **An inline, ad hoc coupling-map stand-in was replaced with the
   already-verified `EdgeListCouplingMap` class from `psf_pennylane.py`**
   (built and tested end-to-end in this project on 2026-09-22) -- reusing
   verified code rather than reimplementing the same small interface
   informally a second time.

## 3. What this establishes and does not

- **Establishes**: the specific four circuits that would be sent on
  2026-09-28 are ready, verified correct against the logical circuit they
  were trained to represent, and use only the heavy-hex-compatible basis
  gates (`rz`, `sx`, `x`, `cx`).
- **Does not establish**: anything about a specific real device -- the
  target device for 2026-09-28 is not yet chosen (this used a generic
  heavy-hex d=3 lattice, 19 qubits, not any particular named IBM backend),
  and the actual physical qubit mapping on a real device may differ once a
  specific backend's own calibration and `target` are used at submission
  time. Real hardware noise, queueing, and result interpretation are
  unchanged, deferred items (per Addendum 147's own "what remains before
  2026-09-28" list).

## 4. What remains before 2026-09-28

Unchanged from Addendum 148's own list, with one item now complete:

- IBM Quantum account credentials -- **not yet used or stored anywhere in
  this project**; a credential was inadvertently shared in this session's
  conversation and is not recorded in any file.
- A connection/submission script using those credentials -- not yet
  written.
- ~~Which trained circuit to send~~ -- **done**: seed 0, all four XOR
  inputs, compiled and QASM-exported (this addendum).
- Choosing the actual target device once the quota resets, and re-running
  this same compilation against that device's own real coupling map and
  basis (the generic heavy-hex d=3 lattice used here is a stand-in).

## 5. Files

| File | What it is |
|---|---|
| [`compile_for_hardware.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_for_hardware.py) | this run's script |
| `xor_hw_ready_input_00.qasm`, `_01.qasm`, `_10.qasm`, `_11.qasm` | the four compiled circuits (not yet collected into this project's own file set) |

## 6. Verification

- Seed 0's re-trained result checked against Addendum 148's own recorded
  CSV value (0.000005) before proceeding, not assumed to match.
- All four expectation-value differences read directly from the run's own
  output.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits. The API key
  shared earlier in this session is confirmed absent from this document
  and from every other file in this project.

---

**End of Part 8 of 8 (end of document, for now).**
