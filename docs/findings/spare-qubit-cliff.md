# Qiskit's `optimization_level` 2/3 falls off a cliff when the circuit nearly fills the coupling map

**Status:** causally established by controlled intervention, reproduced in three
independent environments. Two follow-up controls (2026-09-11) establish that the dense
adjacent-pair circuit structure is **required**, and that the effect spans Qiskit 1.4.6
through 2.5.2 unchanged. `VF2Layout` and `VF2PostLayout` account for 99.9% of the time,
each consuming the level-3 call budget in full and reporting `NO_SOLUTION_FOUND` for a
layout that provably exists. **The failure is ordering-dependent inside Qiskit's own
implementation**: permuting the coupling graph's node indices via `shuffle_seed` turns
`NO_SOLUTION_FOUND` into `SOLUTION_FOUND` in 4 of 30 seeds on an unchanged saturated
grid, and `VF2PostLayout` succeeds on the *same four seeds* (2026-09-11). Through the
preset, though, 30 calls find a layout zero times, so the cliff is deterministic for a
given input. And finding it would not help: a seed that finds the layout in 3.5 ms
still runs for 343 ms. Ordering and trial loop are two independent costs and neither
fix alone removes the cliff. One topology family (`CouplingMap.from_grid`).

**This is not a finding about PSF-Zero.** It surfaced while benchmarking PSF-Zero
against coupling maps, but it is a property of Qiskit's transpiler and it affects
anyone using `optimization_level` 2 or 3 against a map their circuit nearly fills.

---

## The short version

Hold the coupling map fixed. Change only how many of its qubits the circuit uses.

| Coupling map | Circuit | Spare qubits | `opt=2` | `opt=3` |
| :--- | :---: | :---: | ---: | ---: |
| 6×7 grid (42 physical) | 38 qubits | 4 | **14 ms** | **28 ms** |
| 6×7 grid (42 physical) | 42 qubits | 0 | **579 ms** | **6,803 ms** |

Same map. Same circuit family. Four fewer logical qubits, and `optimization_level=3`
goes from 28 milliseconds to 6.8 seconds — a **246x** change.

The effect is not about size. A 42-qubit circuit on a saturated 42-qubit grid takes
6,803 ms at `opt=3`; a **larger** 50-qubit circuit on a 56-qubit grid with 6 spare
takes 34 ms. The smaller circuit is ~200x slower.

**Workaround:** pad the coupling map with a few spare qubits. The effect disappears
completely.

---

## How this was found, and why replication could not settle it

The pattern first appeared in an ordinary scaling sweep (`phase3_v4.py`, 50/100/156/300
qubits): `opt=3` took tens of milliseconds at 50 and 300 qubits but **10–31 seconds**
at 100 and 156. Non-monotonic in circuit size, which nothing about "bigger circuits
are harder" explains.

It correlated exactly with spare qubits, because `get_grid_cmap()` picks
`cols = ceil(sqrt(n))`, `rows = ceil(n/cols)`:

| n | grid | physical | spare |
| :---: | :---: | :---: | :---: |
| 50 | 7×8 | 56 | 6 |
| 100 | 10×10 | 100 | **0** |
| 156 | 13×13 | 169 | 13 → but the run used a saturated map |
| 300 | 18×18 | 324 | 24 |

The trouble is that **"has no spare qubits" and "is one of those two sizes" were
perfectly confounded.** Running the same four points again on more machines could
raise confidence that the observation was real, but could never separate the two
explanations. This project reproduced those four points in three environments before
recognising that replication is not a test of a confound.

## The controlled experiment

[`phase3_v5_spare_qubits.py`](../../benchmarks/phase3_v5_spare_qubits.py) breaks the confound by **holding the coupling map fixed
and varying only how much of it the circuit occupies** (axis C), and separately by
holding the circuit fixed and varying the map (axis B).

Design choices that matter:

- It reuses this project's own `get_grid_cmap()` and
  `build_dense_pair_blocks_circuit()` **verbatim**, so the fixture is provably the
  same one the original observation came from.
- It re-derives the n=100 saturated point as an **anchor**: `opt=2` 1,139 ms and
  `opt=3` 13,060 ms here, against 1,244 ms / 14,343 ms for the same point in the
  earlier sweep. The fixture matches.
- The predictions were **written down before the run**: *if only the zero-spare points
  are orders of magnitude slower, the hypothesis is supported; if pairs on the same map
  do not differ, it is refuted.*

## Results

### Axis C — same map, varying occupancy

Min-of-reps, median over seeds. Three independent environments; the AMD machine was
run twice back-to-back.

**`optimization_level=2` (ms)**

| Grid | Spare | Linux sandbox | Intel (Win) | AMD run 1 | AMD run 2 |
| :--- | :---: | ---: | ---: | ---: | ---: |
| 6×7 = 42 | 4 | 34.1 | 11.6 | 14.6 | 14.1 |
| 6×7 = 42 | **0** | **621.1** | **549.6** | **578.7** | **576.3** |
| 7×8 = 56 | 6 | 16.1 | 12.6 | 16.6 | 15.9 |
| 7×8 = 56 | **0** | **767.7** | **633.0** | **686.6** | **676.2** |
| 8×8 = 64 | 4 | 20.0 | 12.2 | 18.0 | 17.3 |
| 8×8 = 64 | **0** | **840.8** | **706.6** | **777.3** | **773.6** |
| 8×9 = 72 | 6 | 20.5 | 15.8 | 19.1 | 18.3 |
| 8×9 = 72 | **0** | **877.8** | **720.0** | **790.6** | **789.0** |

**`optimization_level=3` (ms)**

| Grid | Spare | Intel (Win) | AMD run 1 | AMD run 2 |
| :--- | :---: | ---: | ---: | ---: |
| 6×7 = 42 | 4 | 23.4 | 27.6 | 27.6 |
| 6×7 = 42 | **0** | **7,725** | **6,803** | **6,710** |
| 7×8 = 56 | 6 | 29.2 | 34.3 | 32.9 |
| 7×8 = 56 | **0** | **9,770** | **8,088** | **8,044** |
| 8×8 = 64 | 4 | 29.8 | 33.6 | 33.0 |
| 8×8 = 64 | **0** | **8,184** | **9,243** | **8,928** |
| 8×9 = 72 | 6 | 36.3 | 37.2 | 35.8 |
| 8×9 = 72 | **0** | **8,295** | **9,197** | **9,157** |

### The cliff, as ratios

| Grid | `opt=2` | `opt=3` |
| :--- | :---: | :---: |
| 6×7 = 42 | 39.5x / 40.8x | 246x / 243x |
| 7×8 = 56 | 41.3x / 42.5x | 236x / 245x |
| 8×8 = 64 | 43.2x / 44.6x | 275x / 270x |
| 8×9 = 72 | 41.4x / 43.1x | 247x / 256x |

<sub>(AMD run 1 / run 2. The zero-spare point divided by the spare point on the
*same* map.)</sub>

Run-to-run agreement on the same machine: **0.2–3.5%** at the zero-spare points and
0.1–4.7% at the spare points. The experiment itself is stable.

### The threshold is not zero — it moves with the map

The four-point data could never have shown this. On the 10×10 grid, **2 spare qubits
is still fully in the slow regime**; 4 spare is fully out of it:

| Grid | Spare | `opt=2` | `opt=3` |
| :--- | :---: | ---: | ---: |
| 10×10 = 100 | 4 | 24.4 ms | 53.4 ms |
| 10×10 = 100 | **2** | **1,128 ms** | **13,033 ms** |
| 10×10 = 100 | **0** | **1,139 ms** | **13,060 ms** |

On the 8×9 = 72 grid, by contrast, 2 spare is already **out** of the slow regime
(35.6 ms at `opt=3`, against 10,115 ms at zero spare). So it is not a fixed count of
spare qubits; the boundary sits higher on the larger map. Two grids is not enough to
say whether it tracks area, perimeter, or something else, and we have not tried to
find out.

### Padding the map removes it entirely

Same 100-qubit circuit, three different maps:

| Map | Spare | `opt=2` |
| :--- | :---: | ---: |
| 10×10 = 100 | 0 | 1,139 ms |
| 10×11 = 110 | 10 | **25.0 ms** |
| 11×11 = 121 | 21 | **34.7 ms** |

## The slow runs are not doing more work

Across all four datasets (120 measurements) the output is essentially always the
baseline 3 two-qubit gates per logical pair, at depth 16, with **zero coupling
violations** — including in the slow cases. The time is spent searching for a
solution the router eventually finds, not producing a bigger circuit.

Seven measurements departed from that baseline by inserting SWAPs. Two observations
about them:

**1. Every SWAP case is on an odd-column grid.** All seven are on 6×7 or 8×9. The
even-column grids (7×8, 8×8) were SWAP-free in every single measurement. That fits the
mechanical explanation: consecutive logical pairs (0,1), (2,3), … straddle a row
boundary when the grid width is odd, and `transpile()` is not seed-pinned here, so the
layout pass avoids it in some runs and not others.

**2. Whether SWAPs were inserted makes no difference to the time.** Within one run, on
one grid, one arm, two seeds:

| Case | Output | Time |
| :--- | :---: | ---: |
| n=42, 6×7, `opt=2`, seed 1 | 63 gates, depth 16 (SWAP-free) | 571.8 ms |
| n=42, 6×7, `opt=2`, seed 2 | 66 gates, depth 19 (one SWAP) | 580.9 ms |
| n=72, 8×9, `opt=3`, seed 1 | 111 gates, depth 19 (one SWAP) | 9,137 ms |
| n=72, 8×9, `opt=3`, seed 2 | 108 gates, depth 16 (SWAP-free) | 9,177 ms |

1.6% and 0.4% apart. Paired, same run, same seeds' worth of noise. This is the
cleanest available evidence that the cliff is search cost, not output cost.

## The two gaps are now closed — and both answers sharpen the finding

The two things this document previously listed as unknown were run on 2026-09-11 in a
cloud sandbox (Linux, Python 3.11.15, 2 cores). Both came back with answers that
change how the result should be stated.

### 1. The dense adjacent-pair structure is required

`benchmarks/phase3_v6_workload_control.py` runs the **identical axis-C design** —
same grids, same spare-qubit pairs, same measurement discipline — and changes only
the circuit. The passthrough workload is `random_circuit(n, depth=90, max_operands=2)`,
chosen so its two-qubit gate count matches the dense circuit's to within ~10%
(1,190 against 1,260 at n=42; 2,094 against 2,160 at n=72).

| Grid | Spare | dense `opt=2` | passthrough `opt=2` | dense `opt=3` | passthrough `opt=3` |
| :--- | :---: | ---: | ---: | ---: | ---: |
| 6×7 = 42 | 4 | 33.1 ms | 645.8 ms | 62.0 ms | 520.1 ms |
| 6×7 = 42 | **0** | **817.0 ms** | 754.3 ms | **8,782 ms** | 485.2 ms |
| 7×8 = 56 | 6 | 22.4 ms | 1,107.5 ms | 41.0 ms | 944.6 ms |
| 7×8 = 56 | **0** | **881.3 ms** | 1,307.6 ms | **10,202 ms** | 901.2 ms |
| 8×8 = 64 | 4 | 61.6 ms | 1,590.7 ms | 44.0 ms | 1,185.4 ms |
| 8×8 = 64 | **0** | **1,048 ms** | 1,639.7 ms | **11,834 ms** | 1,164.1 ms |
| 8×9 = 72 | 6 | 42.8 ms | 1,875.0 ms | 47.9 ms | 1,235.3 ms |
| 8×9 = 72 | **0** | **1,045 ms** | 2,119.0 ms | **12,046 ms** | 1,554.6 ms |

**There is no cliff in the passthrough column.** Saturating the map costs
**1.03x–1.18x** at `opt=2` and **0.93x–1.26x** at `opt=3` — at three of four grids
`opt=3` is *faster* on the saturated map. Against 17x–39x and 142x–269x for the dense
workload measured on the same box in the same session.

What is left of the passthrough increase tracks the input gate count, not the
saturation: the passthrough circuits get larger as n grows (1,712 → 3,268 input 2-qubit
gates), and the time grows with them.

**So the earlier framing was too broad.** A saturated coupling map is not sufficient.
The trigger needs the saturated map **and** the dense adjacent-pair structure
together. The 2026-09-08 note that guessed this from a different experiment was right,
and it is now a controlled result rather than an inference.

Read the other way round, the surprising number is not that the saturated dense case
is slow — at ~1 second for 1,260 two-qubit gates it is in the same range as
passthrough. It is that the **unsaturated** dense case is extraordinarily *fast*
(22–62 ms), and saturating the map destroys that. Adjacent logical pairs map onto
adjacent physical qubits under a row-major grid, so with even one spare qubit the
layout pass finds a trivial, routing-free solution immediately. Take the slack away
and it stops finding it.

### 2. It is not a regression — the fast path was *added*, and never reached the saturated case

Six Qiskit versions, same box, same script, on the 6×7 = 42 grid:

| Qiskit | n=38 (spare 4) `opt=3` | n=42 (spare 0) `opt=3` | ratio |
| :--- | ---: | ---: | ---: |
| 1.4.6 | 2,654 ms | 9,983 ms | 3.8x |
| 2.0.3 | 2,612 ms | 9,781 ms | 3.7x |
| **2.1.2** | **54.8 ms** | 9,898 ms | **181x** |
| 2.2.3 | 274 ms | 8,952 ms | 33x |
| 2.4.2 | 34.3 ms | 8,998 ms | 262x |
| 2.5.2 | 56.1 ms | 8,760 ms | 156x |

Two things jump out.

**The saturated column has not moved since 1.4.6.** 9,983 → 8,760 ms across six
releases spanning two major versions — no improvement at all, within measurement
noise of each other.

**The unsaturated column dropped ~47x between 2.0.3 and 2.1.2.** Re-measured with
3 seeds × 3 repetitions to rule out noise: 2.0.3 gives 2,551–2,574 ms (min, spread
under 1%) and 2.1.2 gives 26.6–27.9 ms — a **93x** step, with **identical output**
(57 two-qubit gates, depth 16) on both sides.

So the "cliff" is not something that broke. Something in Qiskit 2.1 made
`optimization_level=3` dramatically faster on this circuit family, and **that
improvement does not engage when the coupling map is saturated.** The saturated case
has been paying ~10 seconds since at least 1.4.6 and still is.

That is a more useful statement for a Qiskit maintainer than "level 3 is slow on full
maps": it points at a specific release boundary and a specific pair of inputs that
differ by four qubits.

<sub>Caveat on these two experiments: they ran on a 2-core cloud sandbox, so absolute
times are not comparable with the Windows runs above. Every comparison drawn here is
between points measured in the same session on the same box, which is what the
argument needs. The version sweep is one seed and one repetition per point except the
2.0.3 / 2.1.2 / 2.5.2 confirmation, which is 3 seeds × 3 reps; the 274 ms at 2.2.3 is
most likely single-measurement noise on a shared 2-core box rather than a real
intermediate step.</sub>

## Where the time goes

Measured 2026-09-11, same sandbox. Per-pass timing via
`generate_preset_pass_manager(...).run(qc, callback=...)`:

| | total | `VF2Layout` | `VF2PostLayout` | `VF2Layout_stop_reason` |
| :--- | ---: | ---: | ---: | :--- |
| n=38 on 6×7 (4 spare) | 37.5 ms | 26.2 ms | 0.3 ms | `SOLUTION_FOUND` |
| n=42 on 6×7 (0 spare) | **12,789 ms** | **6,387 ms** | **6,384 ms** | `NO_SOLUTION_FOUND` |

**99.9% of the time is those two passes.** Nothing else moves. The level dependence is
the VF2 call budget, which `get_vf2_limits` sets to 50,000 at levels 1–2 and
**30,000,000 at level 3** (in-source comment: "~60 sec with rustworkx 0.10.2"):
22.6 ms / 1,117 ms / 12,771 ms of VF2 time at levels 1 / 2 / 3. That is also why
`routing_optimization_level=1` never enters the regime.

**The layout it fails to find exists.** The interaction graph is 21 disjoint edges on
42 qubits; the 6×7 grid has a perfect matching of size 21
(`networkx.max_weight_matching(G, maxcardinality=True)`). We constructed an explicit
layout and verified all 42 physical qubits are used and all 21 pairs land on coupling
edges.

**More budget does not help.** `call_limit=30,000,000` → 6.68 s, nothing found;
`100,000,000` → 21.96 s, nothing found. Linear in the limit, so the budget is binding
and buying more of it buys proportionally more failure.

## What the Qiskit source says

Read 2026-09-11 from `crates/transpiler/src/passes/vf2_layout.rs` (Qiskit `main`).

**Qiskit implements VF2 itself, in `qiskit-circuit`, not via `rustworkx.vf2_mapping`.**
The pass file imports `qiskit_circuit::{..., vf2}`; `rustworkx-core` appears only as
the source of the `petgraph` data structures. So a Python-space claim that these passes
call `rustworkx.vf2_mapping` is false — see "Reported upstream, and rejected" below.

**Both passes use the VF2++ node ordering.** `vf2_layout_pass_average` and
`vf2_layout_pass_exact` each build their search the same way, ending with
`.with_vf2pp_ordering()`:

```rust
let vf2 = vf2::Vf2::new(&interactions.graph, &coupling_graph, vf2::Problem::Subgraph)
    .with_scoring(score, score)
    .with_restriction(vf2::Restriction::Decreasing(best_score))
    .with_vf2pp_ordering();
```

That is the same heuristic — a different implementation of it — whose failure on this
exact pattern is measured in the next section. **It makes the ordering a live
candidate for the mechanism rather than an excluded one**, which is the opposite of
what this document concluded when it only knew that `rustworkx.vf2_mapping` was not
called.

**There is no ordering knob.** `Vf2PassConfiguration` exposes `call_limit`,
`time_limit`, `max_trials`, `shuffle_seed` and `score_initial_layout`.
`with_vf2pp_ordering()` is unconditional. So the `id_order=True` escape measured
against rustworkx below is not available to a Qiskit user at any level.

**`shuffle_seed` is a way to test this from Python.** It permutes the coupling graph's
node indices before the search (`vf2::reorder_nodes`). VF2++ ordering depends on node
degree and index, so if the failure is ordering-dependent, some seeds should turn
`NO_SOLUTION_FOUND` into `SOLUTION_FOUND` on an unchanged saturated grid. That
experiment has not been run; it would be evidence about Qiskit's own implementation,
with no rustworkx involved.

## The ordering is confirmed as the failure mode, inside Qiskit itself

Measured 2026-09-11 on the AMD machine (Windows 10, Python 3.10.11, Qiskit 2.5.2),
[`benchmarks/verify_vf2_seed.py`](../../benchmarks/verify_vf2_seed.py).

`VF2Layout` exposes no `id_order` equivalent, but `shuffle_seed` permutes the coupling
graph's node indices before the search (`vf2::reorder_nodes`), and VF2++ ordering
depends on node degree and index. So varying the seed varies the ordering and nothing
else. The saturated case — 42-qubit dense-pair circuit on the 6×7 grid,
`call_limit=3,000,000`, `VF2Layout` called directly rather than through a preset:

| Seeds | Stop reason | Count |
| :--- | :--- | ---: |
| 1, 8, 25, 29 | `SOLUTION_FOUND` | **4 / 30** |
| all others | `NO_SOLUTION_FOUND` | 26 / 30 |

**Same map, same circuit, same budget. Only the node order changed, and in 13% of
orderings Qiskit finds the layout it reports as nonexistent in the other 87%.**

Two things follow.

**`NO_SOLUTION_FOUND` here means "not found within this budget, under this ordering",
not "no solution exists".** That was already known indirectly — the grid has a perfect
matching and an explicit layout was constructed by hand — but this is Qiskit's own
implementation finding it, with no external library and no hand-construction involved.

**The failure is ordering-dependent in Qiskit's implementation, not only in
rustworkx's.** The parallel observation below is no longer the only evidence that the
VF2++ ordering is implicated. Whether the two implementations fail on the same
instances for the same structural reason is still untested, but the ordering is now a
measured factor in Qiskit rather than an inference from a different codebase.

### A prediction that failed: success is not fast

Written before the run: *if the ordering is the mechanism, the successful seeds should
return in under a millisecond, giving a two-regime picture — hit the right order and
it is immediate, miss and the budget burns.* That is what
`rustworkx.vf2_mapping` does with `id_order=True`.

It is not what happened. Elapsed time is flat:

| | count | min | median | max | stdev |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `NO_SOLUTION_FOUND` | 26 | 325.1 ms | **335.6 ms** | 380.5 ms | 11.4 |
| `SOLUTION_FOUND` | 4 | 337.6 ms | **339.2 ms** | 368.4 ms | 14.9 |

The successful seeds are 1.01x the failing ones — indistinguishable. Finding the
layout saves nothing.

The source suggested why, and a follow-up measurement confirms it.
`minimize_vf2` does not stop at the first match: it takes the first mapping, then
continues with the second element of `call_limit` under a trial budget
(`max_trials`, defaulting to `15 + max(needle.edge_count(), haystack.edge_count())`)
looking for a better-scoring one, and keeps the last improvement found.

### Separating "finding a layout" from "finishing the search"

[`benchmarks/verify_vf2_max_trials.py`](../../benchmarks/verify_vf2_max_trials.py)
re-runs four successful and four failing seeds with `max_trials=1`, which stops after
the first match, against the default. Min of 3 reps. Predictions were written before
the run: successes get much faster, failures do not change, stop reasons are
unchanged.

| Seed | Outcome | default | `max_trials=1` | ratio |
| ---: | :--- | ---: | ---: | ---: |
| 1 | `SOLUTION_FOUND` | 342.8 ms | **3.5 ms** | **98.2x** |
| 25 | `SOLUTION_FOUND` | 341.0 ms | **34.5 ms** | **9.9x** |
| 29 | `SOLUTION_FOUND` | 335.8 ms | **34.2 ms** | **9.8x** |
| 8 | `SOLUTION_FOUND` | 338.3 ms | 336.1 ms | 1.01x |
| 0 | `NO_SOLUTION_FOUND` | 329.4 ms | 329.4 ms | 1.00x |
| 2 | `NO_SOLUTION_FOUND` | 333.4 ms | 329.5 ms | 1.01x |
| 3 | `NO_SOLUTION_FOUND` | 331.3 ms | 326.4 ms | 1.02x |
| 4 | `NO_SOLUTION_FOUND` | 336.9 ms | 340.4 ms | 0.99x |

Stop reasons were identical under both settings for every seed.

Because `max_trials=1` returns at the first match, its time *is* the time to find a
layout. Splitting the default run on that gives three regimes, not two:

| Seed | Time to first match | Time spent after it | Share after the match |
| ---: | ---: | ---: | ---: |
| 1 | 3.5 ms | 339.3 ms | **99.0%** |
| 25 | 34.5 ms | 306.6 ms | 89.9% |
| 29 | 34.2 ms | 301.6 ms | 89.8% |
| 8 | 336.1 ms | 2.3 ms | 0.7% |
| failures | never | — | 100% |

**Seed 1 finds the layout in 3.5 milliseconds and the pass still takes 343.** The
trial loop is the cost, measured, not inferred — 90–99% of the time in the fast-find
cases.

**Seed 8 is the exception that fits.** It succeeds, but `max_trials=1` saves nothing,
because its *first* match arrives at 336 ms — it spent nearly the whole first call
budget getting there. So "successful" seeds are not one population: some orderings
find the layout almost immediately, one finds it just before the budget runs out.

### What this means for a fix

The two costs are independent, and neither fix alone removes the cliff.

**Fixing the ordering alone does not.** Seeds 1, 25 and 29 already have a good
ordering — they find the layout in 3.5 to 34 ms — and still take 336 to 343 ms,
because the trial loop runs afterwards regardless.

**Fixing the trial loop alone does not.** 26 of 30 orderings never reach a first
match, and `max_trials=1` changes their time by 0–2%. There is nothing to stop early
when nothing is found.

For a user, this is why the practical advice does not change: pad the coupling map, or
stay at `routing_optimization_level=1`. Neither of the two mechanisms above is
reachable from Python — `VF2Layout` exposes `seed` and `call_limit`, and the preset
sets the budget.

**Scope.** One grid (6×7), one circuit, 30 seeds (8 of them in the `max_trials` run),
one machine, one Qiskit version. The budget used here (3,000,000) is a tenth of what
`optimization_level=3` sets, chosen so a full burn takes ~335 ms instead of seconds.
`VF2PostLayout` is covered below.

### `VF2PostLayout` fails on exactly the same seeds

Same scan, same grid, same budget, run against `VF2PostLayout`
([`benchmarks/verify_vf2post_seed.py`](../../benchmarks/verify_vf2post_seed.py)). Its
input is the routed circuit (`transpile(..., optimization_level=1,
seed_transpiler=0)`), and it needs a `Target`, built from the same grid and basis with
no error rates — so its scoring falls to the degree-based legacy path in
`build_average_error_map`.

| Stop reason | Count | Seeds |
| :--- | ---: | :--- |
| `SOLUTION_FOUND` | 4 / 30 | **1, 8, 25, 29** |
| `NO_BETTER_SOLUTION_FOUND` | 26 / 30 | all others |

**The successful seeds are identical to `VF2Layout`'s: 1, 8, 25, 29.** If the two
passes failed independently, the chance of picking the same four out of thirty is
1 in 27,405. They are not independent failures; they are the same search, on the same
coupling graph, gated by the same node ordering, counted twice.

That matters for reading the 12.8 s. The per-pass table above splits it evenly between
`VF2Layout` (6,387 ms) and `VF2PostLayout` (6,384 ms), which looks like two separate
problems. On this evidence it is one problem paid for twice: whatever ordering makes
the first pass miss also makes the second one miss, and the preset runs the second
pass anyway after the first has already exhausted its budget on the same graph pair.

One detail worth recording, because it constrains the mechanism: the two passes were
given **different input circuits** — the raw dense-pair circuit and the routed one —
and still succeeded on the same seeds. The seed permutes the *coupling* graph, not the
interaction graph, so this is consistent with the ordering of the coupling graph being
what decides the outcome. It is not proof; the routed circuit's interaction graph is
probably close to the original's on this workload, and that was not checked.

**`NO_SOLUTION_FOUND` never appears here.** `VF2PostLayout` scores the incoming layout
first (`score_initial_layout`), so a failed search reports
`NO_BETTER_SOLUTION_FOUND` — "we looked and found nothing better", not "there is no
solution". In the 26 failing seeds it spends the full budget arriving at that.

Timing is flat again, as with `VF2Layout`: failures median **335.7 ms**, successes
median **342.1 ms**, against an overall median of 336.1 ms. Finding a better layout
does not shorten the pass any more than finding any layout shortened the first one.

### A negative result: timing cannot tell whether the preset draws a lucky ordering

`Vf2PassConfiguration::from_legacy_api` treats a `None` seed as "seed with OS entropy"
and `-1` as "no shuffling". If the preset pass managers leave it unset, every
`transpile()` call would draw a fresh node ordering, and about 1 call in 7 should land
on one of the orderings that finds a layout. Dozens of measurements of the saturated
case have never produced a fast one, which did not obviously fit.

[`benchmarks/verify_preset_shuffle.py`](../../benchmarks/verify_preset_shuffle.py)
tested it the wrong way: 20 `transpile(optimization_level=3)` calls with no
`seed_transpiler`, 20 with it pinned, counting calls that finished quickly.

| Arm | median | min | max | calls under 1 s |
| :--- | ---: | ---: | ---: | ---: |
| unpinned | 6,784.0 ms | 6,660.1 | 6,959.8 | **0 / 20** |
| pinned (`seed_transpiler=0`) | 6,758.1 ms | 6,692.5 | 6,946.3 | **0 / 20** |

Median ratio 1.004x.

**The experiment cannot answer the question it was built for, and the reason is in
the measurement directly above it.** A lucky ordering does not make the pass fast:
seed 1 finds its layout in 3.5 ms and the pass still runs for 343 ms, because the
trial loop burns the budget afterwards. So a `transpile()` call that drew a winning
ordering would take about as long as one that did not, and counting fast calls
detects nothing either way. The 1-second threshold was chosen from an assumption this
project had already refuted two experiments earlier. Recorded here rather than
dropped, because the design error is the useful part.

Detection power was marginal too: at a 4-in-30 hit rate, 20 calls miss every time with
probability 5.7%.

Three things it does establish.

**Pinning `seed_transpiler` does not change the cliff.** 1.004x on the median, and
the two arms' ranges overlap almost exactly. Whatever the seed reaches, it is not a
lever on this.

**There is randomness in the unpinned path, and it is not in the VF2 outcome.** One
unpinned call out of twenty emitted 69 two-qubit gates at depth 35 (SWAPs inserted)
against 63 at depth 16 for the other nineteen and for all twenty pinned calls. That
is the routing variation this document records elsewhere on odd-column grids. It cost
4% of the median time.

**The open question needed a different instrument**, which the next section uses.

### The preset never draws a winning ordering: 0 of 30

[`benchmarks/verify_preset_stop_reason.py`](../../benchmarks/verify_preset_stop_reason.py)
reads `VF2Layout_stop_reason` and `VF2PostLayout_stop_reason` out of the property set
after each run, instead of inferring the outcome from elapsed time. 30 calls with no
`seed_transpiler`, 10 with it pinned to 0, same saturated case.

| Arm | Calls | `VF2Layout` | `VF2PostLayout` | median |
| :--- | ---: | :--- | :--- | ---: |
| unpinned | 30 | `NO_SOLUTION_FOUND` × 30 | `NO_BETTER_SOLUTION_FOUND` × 30 | 6,719.3 ms |
| pinned | 10 | `NO_SOLUTION_FOUND` × 10 | `NO_BETTER_SOLUTION_FOUND` × 10 | 6,699.3 ms |

Every one of the 40 calls emitted 63 two-qubit gates at depth 16 — no variation in the
output either.

**Through the preset, the cliff is deterministic.** The standalone scan finds a layout
on 4 of 30 shuffle seeds; thirty preset calls find one zero times. If the preset were
drawing fresh orderings at that rate, the probability of missing every time is
(26/30)^30 = **1.4%**. Either the preset does not shuffle, or whatever it shuffles
does not reach the VF2 node ordering. Either way, there is no luck to be had: the same
input takes the same path and the same seconds on every call, and `seed_transpiler` is
not a lever on it.

`VF2PostLayout` reporting `NO_BETTER_SOLUTION_FOUND` in all 40 is the same statement
from the other side: it scores the incoming layout, searches for a better one, spends
its budget, and concludes there was none — every time, identically.

**Why the standalone scan and the preset disagree is not established.** One candidate
was ruled out. The standalone scan passed `coupling_map=`, so `VF2Layout` built a
target through `_build_dummy_target` — basis `["u", "cx"]`, no error rates, which
sends `build_average_error_map` down its degree-based legacy branch. The preset passes
a real target, and scoring drives pruning, so the two might not have been solving the
same problem.
[`benchmarks/verify_vf2_target_scoring.py`](../../benchmarks/verify_vf2_target_scoring.py)
re-ran the scan both ways, changing only how the target is supplied:

| Target | Found | Seeds | median |
| :--- | ---: | :--- | ---: |
| dummy (`coupling_map=`) | 4 / 30 | 1, 8, 25, 29 | 337.4 ms |
| real (`target=`, basis `rz sx x cx`) | 4 / 30 | 1, 8, 25, 29 | 335.9 ms |

Identical. Scoring is not the difference. What is left unchecked: the preset also sets
`vf2_avg_error_map` in the property set, and passes its own `call_limit` from
`get_vf2_limits` — which the 2-tuple form makes meaningful, since the second element
governs the budget *after* the first match. The `seed` it passes is `None`, which the
docstring says "seeds using OS entropy (and so is non-deterministic)", so the preset
should be shuffling. It nevertheless missed thirty times.

Put beside the two measurements above, the picture closes:

| | measured |
| :--- | :--- |
| An ordering that reaches the layout exists | yes — 4 of 30 seeds, standalone |
| The preset reaches one | no — 0 of 30 calls |
| Reaching one would fix the time | no — found in 3.5 ms, pass still 343 ms |

## A parallel observation: `rustworkx.vf2_mapping`

A separate VF2++ implementation, on the same patterns. Qiskit does not call it, so this
is not evidence about Qiskit's passes — it is a second data point about the heuristic,
and the one place where changing the ordering is directly available as a switch.

| Grid | Spare | `id_order=False` (VF2++) | `id_order=True` (plain VF2) |
| :--- | :---: | :--- | :--- |
| 6×7 = 42 | 4 | found, <1 ms | found, <1 ms |
| 6×7 = 42 | **0** | **not found, 6.64 s** | **found, <1 ms** |
| 7×8 = 56 | 6 | found, <1 ms | found, <1 ms |
| 7×8 = 56 | **0** | **not found, 7.33 s** | **found, <1 ms** |
| 8×8 = 64 | 4 | found, <1 ms | found, <1 ms |
| 8×8 = 64 | **0** | **not found, 9.05 s** | **found, <1 ms** |
| 8×9 = 72 | 6 | found, <1 ms | found, <1 ms |
| 8×9 = 72 | **0** | **not found, 8.43 s** | **found, <1 ms** |

**The boundary is 24 nodes.** Scanning every grid with an even node count and a perfect
matching: below 24 nodes the VF2++ ordering finds a mapping (2×2 through 4×5, all
under 75 ms); from 24 nodes up (3×8 and 4×6 are the smallest) it never does, on any of
the 14 grids tested.

This also fits the workload control above. A `random_circuit` interaction graph is
connected and dense, so VF2 either embeds it at once or rejects it at once — there is
little for an ordering heuristic to get lost in. A perfect-matching pattern is
maximally disconnected, which is where this one does.

**What it does and does not establish.** It is a real failure of a VF2++ implementation
on the exact pattern that makes Qiskit slow, and changing the ordering removes it
there. It is not a measurement of Qiskit's implementation, which is a separate
codebase that happens to use the same heuristic. Whether they fail for the same reason
is open.

## Reported upstream, and rejected

Filed 2026-09-11 as a rustworkx issue (the ordering heuristic) and as comments on
Qiskit [#7705](https://github.com/Qiskit/qiskit/issues/7705) and
[#14855](https://github.com/Qiskit/qiskit/issues/14855).

The rustworkx issue was closed the same day. The objection was to one sentence —
*"Qiskit's `VF2Layout` and `VF2PostLayout` use this call"* — which is false: Qiskit 2.x
implements VF2 in `qiskit-circuit` and calls it from
`crates/transpiler/src/passes/vf2_layout.rs`. The #7705 comment was hidden as spam,
with a warning about posting unverified LLM output.

**The objection was correct about the sentence.** It invalidates the bridge from the
rustworkx measurements to Qiskit's behaviour, and it invalidates the framing of #7705
(which asks for `vf2_mapping()` scaling benchmarks — a function these passes do not
use) as the issue this answers. Writing that sentence without reading the
implementation is the same failure this project recorded when it retracted the
RSS-sampler/GIL hypothesis, except that this time it went to a third party.

**It does not establish that the ordering is the wrong suspect.** Reading the source
afterwards shows Qiskit's passes use the VF2++ ordering too, unconditionally. The
correct statement is narrower than either the original draft or the retraction that
followed it: the ordering is a candidate, supported by a measurement in a different
implementation and by Qiskit using the same heuristic, and it has not been tested
against Qiskit's own code.

Drafts as submitted, and the outcome, in [`docs/log/`](../log/):
`rustworkx-issue-draft.md`, `qiskit-issue-7705-comment-draft.md`,
`qiskit-issue-14855-comment-draft.md`. No further upstream contact is planned.

## What is still unknown

- **Why one successful ordering (seed 8) is slow to find its first match** while the
  other three are 10–100x faster. The ordering decides more than hit-or-miss, and
  nothing here explains the spread.
- **Whether the two implementations fail on the same instances.** Both use the VF2++
  ordering and both are ordering-dependent on this pattern; that is a family
  resemblance plus a shared symptom, not a shared cause.
- **Whether the coupling graph's ordering alone decides the outcome.** The two passes
  succeed on the same seeds with different input circuits, which points that way, but
  the two interaction graphs were not compared.
- **Why the preset never reaches a winning ordering.** `VF2Layout` passes its `seed`
  straight through as `shuffle_seed`, and a `None` seed is documented as OS entropy,
  so the preset should be drawing a fresh order each call. Scoring has been ruled out
  as the difference. The untested candidates are the property-set
  `vf2_avg_error_map` and the preset's own `call_limit` 2-tuple.
- **Which part of the ordering causes it, in either implementation.** Neither
  `qiskit-circuit`'s `vf2` module nor rustworkx's ordering code has been read at that
  level.
- **Whether the rustworkx result is specific to disconnected patterns.** Only the
  perfect-matching pattern was tested against `vf2_mapping` directly.
- **Which topologies.** Still only rectangular grids from `CouplingMap.from_grid`.
  Untested on heavy-hex, linear, or real backend maps.
- **Where in Qiskit 2.1 the `SOLUTION_FOUND` speed-up came from.** The release
  boundary is known; the specific change is not. A `git bisect` between 2.0.3 and
  2.1.0 would name it.
- **rustworkx `main`.** 0.18.1 is the latest release and is what was measured; a source
  build was not possible in the sandbox used.

The two gaps this section used to list — other Qiskit versions, and whether the dense
structure is required — are now closed; see the section above.

## Practical consequence

If you run `optimization_level` 2 or 3 against a coupling map your circuit almost
fills, pad the map by a few qubits. It costs nothing and removes a 40x–275x penalty.
Nothing else in reach works: the ordering cannot be changed from Python
(`Vf2PassConfiguration` has no equivalent of `id_order`), `seed_transpiler` does not
move it, and retrying does not either — 30 preset calls on the same input took the
same path every time.

For this project specifically, it turns an earlier correlational argument for
`routing_optimization_level=1` into a measured one: `rl=1` never enters the regime at
all (9.6–50.3 ms across every scale tested, against `rl=2`'s 866–1,145 ms at the
saturated points).

## Files

- Experiments: [`benchmarks/phase3_v5_spare_qubits.py`](../../benchmarks/phase3_v5_spare_qubits.py)
  (the spare-qubit intervention) and
  [`benchmarks/phase3_v6_workload_control.py`](../../benchmarks/phase3_v6_workload_control.py)
  (the dense-vs-passthrough control)
- Raw data: [`data/phase3_v5_spare_qubits_linux_2026-09-10.csv`](../../data/phase3_v5_spare_qubits_linux_2026-09-10.csv),
  [`data/phase3_v5_spare_qubits_intel_2026-09-10.csv`](../../data/phase3_v5_spare_qubits_intel_2026-09-10.csv),
  [`data/phase3_v5_spare_qubits_amd_2026-09-10_run1.csv`](../../data/phase3_v5_spare_qubits_amd_2026-09-10_run1.csv),
  [`data/phase3_v5_spare_qubits_amd_2026-09-10_run2.csv`](../../data/phase3_v5_spare_qubits_amd_2026-09-10_run2.csv)
- Ordering experiments inside Qiskit:
  [`benchmarks/verify_vf2_seed.py`](../../benchmarks/verify_vf2_seed.py) /
  [`data/vf2_seed_scan_2026-09-11.csv`](../../data/vf2_seed_scan_2026-09-11.csv) and
  [`benchmarks/verify_vf2post_seed.py`](../../benchmarks/verify_vf2post_seed.py) /
  [`data/vf2post_seed_scan_2026-09-11.csv`](../../data/vf2post_seed_scan_2026-09-11.csv),
  and [`benchmarks/verify_vf2_max_trials.py`](../../benchmarks/verify_vf2_max_trials.py) /
  [`data/vf2_max_trials_2026-09-11.csv`](../../data/vf2_max_trials_2026-09-11.csv)
- Negative result (timing cannot detect a lucky ordering):
  [`benchmarks/verify_preset_shuffle.py`](../../benchmarks/verify_preset_shuffle.py) /
  [`data/preset_shuffle_2026-09-11.csv`](../../data/preset_shuffle_2026-09-11.csv)
- Stop reasons through the preset:
  [`benchmarks/verify_preset_stop_reason.py`](../../benchmarks/verify_preset_stop_reason.py) /
  [`data/preset_stop_reason_2026-09-12.csv`](../../data/preset_stop_reason_2026-09-12.csv)
- Scoring ruled out as the difference:
  [`benchmarks/verify_vf2_target_scoring.py`](../../benchmarks/verify_vf2_target_scoring.py) /
  [`data/vf2_target_scoring_2026-09-12.csv`](../../data/vf2_target_scoring_2026-09-12.csv)
- Pass timing and ordering probes: `benchmarks/qiskit_pass_timing.py`,
  `benchmarks/vf2_id_order_probe.py`;
  [`data/qiskit_pass_timing_2026-09-11.csv`](../../data/qiskit_pass_timing_2026-09-11.csv),
  [`data/rustworkx_vf2_id_order_2026-09-11.csv`](../../data/rustworkx_vf2_id_order_2026-09-11.csv),
  [`data/rustworkx_vf2_min_case_scan_2026-09-11.csv`](../../data/rustworkx_vf2_min_case_scan_2026-09-11.csv)
- 2026-09-11 controls:
  [`data/phase3_v6_passthrough_control_2026-09-11.csv`](../../data/phase3_v6_passthrough_control_2026-09-11.csv),
  [`data/qiskit_version_sweep_2026-09-11.csv`](../../data/qiskit_version_sweep_2026-09-11.csv),
  [`data/qiskit_version_step_confirm_2026-09-11.csv`](../../data/qiskit_version_step_confirm_2026-09-11.csv)
- The sweep it came from: [`benchmarks/phase3_v4_dense_pair_blocks.py`](../../benchmarks/phase3_v4_dense_pair_blocks.py),
  [`data/phase3_v4_intel_machine_2026-09-10.csv`](../../data/phase3_v4_intel_machine_2026-09-10.csv)
- Upstream drafts as submitted, and the outcome: [`docs/log/`](../log/) —
  `rustworkx-issue-draft.md`, `qiskit-issue-7705-comment-draft.md`,
  `qiskit-issue-14855-comment-draft.md`
- Full chronological record, including the retracted correlational framing:
  [`docs/log/04-real-device-topology.md`](../log/04-real-device-topology.md) and [`phase3-hardware-routing-regression.md`](../../phase3-hardware-routing-regression.md)

Environments: Linux sandbox (Qiskit 2.5.2); Windows / Python 3.11.9 /
`Intel64 Family 6 Model 181` / Qiskit 2.5.2; Windows / Python 3.10.11 /
`AMD64 Family 25 Model 80` / Qiskit 2.5.2.
