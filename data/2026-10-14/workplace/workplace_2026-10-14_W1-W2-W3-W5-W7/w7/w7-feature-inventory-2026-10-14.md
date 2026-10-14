# W7 (2026-10-14): inventory of the release's features by what justifies them

**Status: a list only; nothing is proposed for removal here (home decides).** Read from the changelog in the docstring
of `psf_compile.py` (release 2026-10-11.4, items 1-76) and of `patches/psf_compile_c48_2026-10-12/psf_compile.py`
(items 77-83, the prototype due to become the release). Items absent from both (14, 32, 34, 51, 52, 54, 55, 65-67, 81)
are candidates that were not adopted or numbers that were skipped. Workplace, 2026-10-14 (times from
`TZ=Europe/Zurich date`).

## Classes

| class | meaning |
|---|---|
| **HW** | supported by a real-device result |
| **CALC** | decided by a quantity computed exactly and uniquely: correctness (unitary or outcome equality), CZ / two-qubit count, circuit duration, compile time, determinism; or a fixed rule on the device's reported failed flags |
| **EST** | justified only by a simulation or by ESP / a noise estimate (excitation, Pauli, hybrid, Kraus cost; summed -ln(1 - e)) |

A feature can do something CALC (build a circuit) and then choose by EST; such items are marked "CALC + EST choice"
and the part that would go with EST is named. "Default" means it acts in `compile_for_hardware(qc, backend=b)`.

## The list

| item | feature | default? | class | basis given in the changelog | note |
|---|---|---|---|---|---|
| 1 | the real Rust core is imported, no silent stub | yes | CALC | correctness | |
| 2 | verification by the core's own infidelity | yes | CALC | compile time, exactness | |
| 3-11 | logging, fallback counting, argument validation, registers kept, error separation, housekeeping | yes | CALC | engineering, correctness | |
| 12 | `layout_search` (VF2 exact embedding, packing) | opt-in | CALC | two-qubit count; a search over embeddings | a search |
| 13 | `callback` | opt-in | CALC | engineering | |
| 15, 18 | closed-form CX core, native middle gaps | yes (cx basis) | CALC | exact to 3e-15; sx count; compile time | |
| 16 | `layout_edge_errors` (matching weighted by log(1 - e)) | opt-in | EST | product of edge fidelities | |
| 17, 23, 24 | every synthesised block checked to operator distance 1e-13 | yes | CALC | correctness (Qiskit decomposer bug) | |
| 19 | single-qubit errors in the weighted layout | opt-in | EST | log-fidelity weights | |
| 20 | batched polish | yes | CALC | compile time | |
| 21, 22, 25 | `block_gate_floor` 8 | yes | CALC | two-qubit count (48 -> 33), exactness over 100,000 compiles | |
| 26 | `REFINE_THRESHOLD` 1e-14 | yes | CALC | drift under recompilation, exactness | |
| 27 | `CORE_VERSION` exported | yes | CALC | engineering | |
| 28 | cost-aware consolidation of short blocks | yes (cx) | CALC | CX count (Weyl optimum) | |
| 29 | `elide_permutations` | yes | CALC | gate count, exact relabelling | |
| 30 | SWAP absorption after routing | yes (cx) | CALC | CX count | |
| 31, 41, 58, 61 | failed elements (reported error >= 0.5, direction-aware) avoided; prune first; never returned silently (`FailedElementsError`) | yes with a device | CALC (rule) | a rule on reported flags; no estimate | SCALE1 (Addendum 475): QK3 on 3 failed elements still gave half of PSF's signal, so "failed" is not "dead"; the 144-qubit rejections come from 58 (failed qubits not cut before placement) |
| 33, 60 | `placement_refine`: Qiskit level 3's re-placement scored by the Target's errors; default with a Target | yes with a device | EST | ESP 0.66/0.84 of Qiskit level 2 (ESP-C31) | HW1/HW2: ESP does not rank PSF against QK3 reliably |
| 35 | `final_resynthesis` ("select" by `excitation_cost`) | opt-in | EST | noise simulation (thermal relaxation) | |
| 36 | `compare_level3` (by `excitation_cost`) | opt-in | EST | estimate | |
| 37 | `compare_floor`, `candidate_score="pauli"` | opt-in | EST | estimate, floor-aware placement | |
| 38, 40 | `candidate_score="hybrid"`, readout in it | opt-in | EST | estimate | |
| 39, 44, 59 | circuits made by Qiskit checked for equivalence; no wide matrices; infidelities recorded | yes where they apply | CALC | correctness (Qiskit #17057) | 39 exists because of 35-37 and 80 |
| 42 | `candidate_score="kraus"` | opt-in | EST | first-order Kraus estimate, in-sample | |
| 43 | recompile failure kept with a warning | superseded by 58 | CALC | engineering | |
| 45, 46, 47, 49, 53, 56, 57 | speed of the estimates and checks (skips, fused matrices, reduced states, gate cache, Rust loops in `psf_zero_core57`) | serve 35-42 | CALC (compile time) | compile time | exist only for the EST features; they go with them |
| 48 | instructions on >= 3 qubits expanded first | yes | CALC | two-qubit count (up to 6.3x -> 1.00-1.38 of level 2) | |
| 50 | commutative cancellation when it removes two-qubit gates | yes | CALC | two-qubit count, exact; the result with fewer two-qubit gates | the choice is by a count, not an estimate |
| 62, 63 | stage timers; SWAP absorption synthesised by Qiskit's decomposer | yes | CALC | compile time 0.89, same two-qubit count | the changelog also cites ESP 1.005, but the count and time decide |
| 64 | routing at level 3 when a device is given | yes with a device | EST | ESP +5-10% (ROUTE-X) | |
| 68 | STEINER6: a commuting Pauli evolution built on the device by construction | yes with a device | CALC + EST choice | construction exact; placements and SWAPs "chosen by the device's errors"; adoption on 12 of 13 cases | the construction is what won on hardware (HW2 R4/S, with 83); its error-based placement is EST |
| 69 | STEINER6's output kept only where its ESP is higher | yes | EST | ESP | a choice by ESP between two constructions |
| 70 | size limit 2000 terms | yes | CALC | compile time | |
| 71 | STEINER6 for terms that do not all commute, kept by ESP | yes | CALC + EST choice | exact construction; choice by ESP (69) | |
| 72, 73, 76 | more inputs read as Pauli rotations; circuits written as gates; final permutations | yes | CALC | exact reading; structure | |
| 74 | near-failed elements (0.1 <= e < 0.5) avoided when that lowers summed -ln(1 - e) | yes with a device | EST | ESP | |
| 75 | deterministic re-placement (rustworkx VF2 in fixed order) scored by summed -ln(1 - e) | yes with a device | CALC (determinism) + EST choice | reproducibility; what it picks is by ESP | determinism is the CALC part |
| 77 | no time budget in the layout search | yes | CALC | determinism | |
| 78 | re-placement at Qiskit's routing stage, for 80 | yes with a device | EST | ESP | serves 80 |
| 79 | basis from a Target given alone | yes | CALC | correctness (runnable output) | |
| 80 | Qiskit level 3 as a candidate, kept when its ESP is higher | yes with a device | EST | ESP; called "a stopgap, not a method" in its own text | W2/W3 (2026-10-14): it reported "level 3 kept" in 51 of 64 W3 compiles and in every W2 TFIM compile; on W2's TFIM the output equalled PSF-C's (it re-runs level 3 on PSF's own circuit) |
| 82 | repeated commuting layers mirrored (every layer costs the first's SWAPs) | yes in STEINER6 | CALC + EST choice | SWAP counts (construction); kept by ESP against STEINER6's own | W3's even/odd pattern is consistent with this (even layer counts end where they began) |
| 83 | STEINER6's blocks in a time-aware order (`ST_SCHEDULE`) | yes | **HW** + CALC | real device: TFIM F x1.30 and x1.17 with the same gates (Addendum 461, part S); kept when its ASAP schedule in CX units is shorter | the only item with direct hardware support |

## Counts

| class | items (by number) |
|---|---|
| HW | 83 (direct); 68 indirectly (the construction 83 orders) |
| CALC | 1-13, 15, 17, 18, 20-31, 39, 41, 43-50, 53, 56-59, 61-63, 70, 72, 73, 76, 77, 79 |
| EST only | 16, 19, 33/60, 35-38, 40, 42, 64, 69, 74, 78, 80 |
| CALC + EST choice | 68, 71, 75, 82 (the construction or determinism is CALC; the final choice is by ESP) |

## What the classification shows (no recommendation)

- Every feature that acts in the default call with a device and is justified by ESP alone sits in placement or
  candidate choice: 33/60, 64, 69, 74, 75's score, 78, 80. These are the "estimate as judge" parts of the policy.
- The opt-in recommended call (35-38, 40, 42) and the speed items built for it (45-47, 49, 53, 56, 57) form one
  block; 39 (equivalence checks of Qiskit-made circuits) is needed by them and by 80.
- The failed-element rule (31/41/58/61) is CALC, but SCALE1 showed its premise ("failed = unusable") is not exact, and
  its order (route, then reject) caused the 144-qubit rejections (W5 tests cutting first).
