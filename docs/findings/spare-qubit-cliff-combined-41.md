# spare-qubit-cliff: Combined Addenda, Part 4 of 5 (Addendum 41 through Addendum 50)

**Continued from [Part 3](spare-qubit-cliff-combined-27.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md)).** Same conventions as Part 1: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

**Note on this part specifically**: three of the addenda here revise or retract a claim made in an earlier one (47 revises 46; 48 retracts part of 43; 50 overturns 49's headline finding). The superseded text is kept in place, unedited, with the correcting addendum noted in the merge note above it -- so the sequence of what was believed when remains readable.

---
<!-- ===== Addendum 41 pre-registration (source: spare-qubit-cliff-addendum-41-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether heavy-hex's cliff immunity (Addendum 40) comes from low degree or simply from its occupancy ceiling -- isolated with a synthetic graph matched on degree but balanced enough to reach true 100% occupancy.

## Addendum 41 -- Pre-registration: is heavy-hex's immunity (Addendum 40) caused by low degree/sparsity, or simply because the topology can never reach true 100% device occupancy? (2026-09-18)

**Status: pre-registration only. No timing measurement has been run yet.**
A synthetic graph construction was prototyped and validated for feasibility
(it produces connected, perfect-matching-admitting bipartite graphs with
heavy-hex-like degree statistics) before this document was written, but
no `transpile()` call has been timed on it. That prototyping is
methodology work (does the intended control graph exist at all), not a
measurement, and is not treated as data. Predictions below are locked
before any timing run.

## 1. Why this experiment exists

Addendum 40 found no occupancy cliff on heavy-hex at d=5 or d=7, and
listed several un-distinguished candidate mechanisms (lower average
degree, cheaper VF2 constraint propagation, the bipartite structure
itself, or the fact that `spare_pairs=0` is comfortably feasible by
construction). Re-reading Addendum 40's own numbers surfaces a sharper,
previously unstated possibility that this addendum isolates directly:

**Heavy-hex's own bipartite imbalance caps the maximum achievable device
occupancy well below 100%, for the dense-pair-blocks circuit family used
throughout this project.** At d=7, `max_matching_pairs=48` means the
hardest circuit this family can pose uses `n=96` of the device's 115
qubits -- **83.5% device occupancy**, not 100%. At d=5, the hardest case
reaches 84.2% (48/57 qubits). By contrast, every grid measurement that
showed a cliff (Addenda 4-35) was measured with `spare=0` meaning
`n_circuit == n_device` exactly -- **100%** device occupancy -- and
Addendum 34's own finer sweep found the cliff is a single-step event
concentrated at exactly that 100% point (spare=1, 97.6% occupancy on a
6x7 grid, already showed no cliff at all; only spare=0 did).

**This means Addendum 40's heavy-hex sweep never actually tested the
regime where the grid cliff lives.** The "no cliff on heavy-hex" result
could be entirely explained by this occupancy ceiling, with heavy-hex's
low degree/sparsity playing no protective role whatsoever -- or heavy-hex's
low degree/sparsity could be doing real protective work on top of that
ceiling. Addendum 40 cannot distinguish these because it could not reach
100% device occupancy on heavy-hex at all (a structural impossibility for
this circuit family, established in Addendum 39). This addendum is
designed specifically to separate the two effects.

## 2. Design: a synthetic control graph that isolates degree from balance

To test true 100% device occupancy on a heavy-hex-*like* (low-degree,
sparse, bipartite) graph, a synthetic random bipartite graph is
constructed with:

- **Balanced parts** (so a perfect matching -- and therefore a true,
  zero-slack `spare=0` -- exists, unlike real heavy-hex).
- **A degree distribution matched to heavy-hex's own** (mostly degree-2,
  a smaller fraction of degree-1 and degree-3 nodes, maximum degree 3),
  built via `networkx.bipartite.configuration_model` with per-side degree
  sequences drawn to approximate heavy-hex d=7's own proportions
  (~2% degree-1, ~67% degree-2, ~31% degree-3), then filtered (retried
  with a new seed) until the result is simple (no multi-edges/self-loops),
  connected, and admits a perfect matching.
- **Comparable size** to the heavy-hex sizes already tested: 58 nodes
  (paired with d=5's 57) and 116 nodes (paired with d=7's 115).

This is **not** a modification of the real heavy-hex graph (no nodes
pruned or added to it) -- it is an independently generated control with
matched size and degree statistics but genuinely balanced bipartite parts,
so it can be pushed to true 100% occupancy, which real heavy-hex cannot.
Validated instances (seed=42): 58-node graph, 69 edges, average degree
2.379, degree distribution {1:1, 2:34, 3:23}, connected, perfect matching
confirmed; 116-node graph, 137 edges, average degree 2.362, degree
distribution {1:2, 2:70, 3:44}, connected, perfect matching confirmed --
both within ~0.1 of heavy-hex's own average degree (2.246 at d=5, 2.296 at
d=7) and structurally similar in shape.

The same dense-pair-blocks circuit family, `timed_transpile`
instrumentation, and `VF2Layout_stop_reason`/feasibility checks from
`occupancy_sweep.py` / `occupancy_sweep_heavy_hex.py` are reused verbatim
on this new graph, swept over `spare` (in ordinary qubit-count terms now,
since a perfect matching exists and `spare=0` genuinely means 100% device
occupancy here).

## 3. Pre-registered predictions

**P1 (primary).** At `spare=0` (true 100% device occupancy) vs. `spare=2`
on the 116-node synthetic graph (the d=7-scale control), at
`optimization_level=3`:
  - **Cliff appears / low-degree-is-not-protective** if the ratio exceeds
    **10x** -- this would mean heavy-hex's real-world immunity (Addendum
    40) is fully explained by never reaching 100% occupancy, and low
    degree/sparsity confers no protection on its own.
  - **Cliff absent / low-degree-is-protective** if the ratio stays under
    **2x** -- this would mean something about low-degree, sparse,
    bipartite graphs (independent of whether 100% occupancy is reached)
    genuinely resists the cliff mechanism, and heavy-hex's immunity is
    doubly protected (structurally capped below 100% occupancy, *and*
    low-degree-protected even if it could reach 100%).
  - **2x-10x**: reported as ambiguous, not rounded either way.

**P2 (consistency check against the grid).** As a sanity check that this
synthetic-graph harness reproduces the known grid result on a graph of
comparable size and degree that is *not* heavy-hex-like (i.e., a
square-grid control at a similar qubit count, already on record from
Addendum 34), the 116-node synthetic graph's `spare=0` behavior is
compared against Addendum 34's 6x7 grid (42 qubits, average degree ~3.7)
only qualitatively -- no specific ratio is predicted for this
cross-topology comparison, since size and degree both differ. This is
listed to make clear that a positive P1 result (cliff appears) would be
expected to land somewhere below the grid's own ~193-353x range, not
necessarily match it, since the synthetic graph's average degree (~2.36)
is still lower than the grid's (~3.7-4).

**P3 (stop-reason check).** At `spare=0` on both synthetic graph sizes,
`VF2Layout_stop_reason` is predicted to flip between "solution found" and
"no solution found" in a way that tracks whichever of P1's two outcomes
holds -- if P1 confirms a cliff, the stop reason should show "no solution
found" at spare=0 and "solution found" at spare>=1 (mirroring Addendum
34's grid result exactly); if P1 finds no cliff, "solution found" should
appear at spare=0 too (mirroring Addendum 40's heavy-hex result).

**P4 (both sizes agree in direction).** The 58-node and 116-node synthetic
graphs are predicted to agree in *direction* (both cliff, or both don't) --
a result where one size cliffs and the other does not, at the same
relative occupancy, would be a surprising and separately reportable
finding, not something this pre-registration expects.

## 4. What this does and does not test

This isolates degree/sparsity from the "never reached 100%" confound using
one synthetic control family (configuration-model bipartite graphs with a
heavy-hex-like degree distribution). It does not test whether the
*specific* heavy-hex construction (its exact local motifs -- hexagonal
plaquettes with flag qubits) matters beyond degree statistics; a positive
P1 result (cliff appears on the synthetic control) would still leave open
whether real heavy-hex's particular local structure, as opposed to just
its low degree, adds any further protection beyond the occupancy-ceiling
effect already established in Addendum 39. It also does not test real,
calibrated device topologies (`FakeTorino`) -- still a separate, listed
open item.

## 5. Files

| File | What it is |
|---|---|
| [`synthetic_sparse_balanced_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/synthetic_sparse_balanced_cliff.py) | the script (to be delivered alongside this document), building the synthetic control graphs and running the same sweep instrumentation as [`occupancy_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep.py) |
| this document | the pre-registered predictions |

## 6. Run command (full scale)

```
python synthetic_sparse_balanced_cliff.py --n-per-side 58 --degree-seed 42 \
    --spares 0,1,2,4,8,16 --levels 3 --seeds 5 --repeats 3
python synthetic_sparse_balanced_cliff.py --n-per-side 29 --degree-seed 42 \
    --spares 0,1,2,4,8 --levels 3 --seeds 5 --repeats 3
```

Pre-publication check: before pasting either run's CSV anywhere outside
this machine, grep it for a local file path or other machine-identifying
string beyond the CPU signature column, per this project's standing
record-keeping rules.

---


<!-- ===== Addendum 41 (source: spare-qubit-cliff-addendum-41-2026-09-18.md) ===== -->

> **Note added when merging:** **Settles Addendum 40's open question.** A heavy-hex-matched low-degree sparse graph cliffs at ~296-303x once pushed to genuine 100% occupancy -- so heavy-hex's immunity is the occupancy ceiling, not low degree. Also records a pre-registered comparison point that turned out to sit inside the slow region, reported as a miss.

## Addendum 41 -- the cliff DOES appear on a heavy-hex-matched low-degree sparse graph once true 100% device occupancy is reached: heavy-hex's immunity (Addendum 40) is explained by the occupancy ceiling, not by low degree (2026-09-18)

**Pre-registered in**: `spare-qubit-cliff-addendum-41-preregistration-2026-09-18.md`.
This document scores every prediction against the actual full-scale run
(5 seeds x 3 repeats per spare value, both graph sizes). One prediction's
literal wording turned out to pick a misleading comparison point --
reported below exactly as it happened, not adjusted after the fact, with
the corrected comparison reported alongside it per this project's
append-only, don't-hide-a-miss discipline.

## 0. In one line

**The cliff reproduces, decisively, on a synthetic graph matched to
heavy-hex's own degree statistics (average degree ~2.2-2.3, mostly
degree-2 nodes, max degree 3) but with balanced bipartite parts -- as soon
as it is pushed to genuine 100% device occupancy**, something real
heavy-hex can never be pushed to (Addendum 39). At the d7-scale graph
(116 device qubits), `spare=0` through `spare=4` (100% down to 96.6%
occupancy) all take ~18 seconds median with `VF2Layout_stop_reason` =
"nonexistent solution," then `spare=8` (93.1% occupancy) drops to 60 ms
with "solution found" -- a **~303x** cliff. The d5-scale graph (58 device
qubits) shows the identical shape at the identical occupancy landmarks:
slow (~10.8s, "nonexistent solution") through `spare=2` (96.6%), fast
(~36ms, "solution found") from `spare=4` (93.1%) on -- a **~295x** cliff.
**This settles Addendum 40's open question**: heavy-hex's real-world
immunity to the occupancy cliff is best explained by Addendum 39's
structural ceiling (it can never reach anywhere near 100% device
occupancy with this circuit family), not by its low average degree or
sparse, bipartite structure being inherently protective. A low-degree
sparse bipartite graph cliffs just as hard as a dense grid, given the
chance.

## 1. Results

### n=116 (d7-scale synthetic control), optimization_level=3, 5 seeds x 3 repeats (15 rows per spare)

| spare | n (circuit qubits) | occupancy | median time (ms) | `VF2Layout_stop_reason` |
|---:|---:|---:|---:|:---|
| 0 | 116 | 100.0% | 18,250.5 | nonexistent solution |
| 2 | 114 | 98.3% | 18,104.4 | nonexistent solution |
| 4 | 112 | 96.6% | 17,886.1 | nonexistent solution |
| 8 | 108 | 93.1% | 60.3 | solution found |
| 16 | 100 | 86.2% | 68.2 | solution found |

### n=58 (d5-scale synthetic control), optimization_level=3, 5 seeds x 3 repeats (15 rows per spare)

| spare | n (circuit qubits) | occupancy | median time (ms) | `VF2Layout_stop_reason` |
|---:|---:|---:|---:|:---|
| 0 | 58 | 100.0% | 10,784.2 | nonexistent solution |
| 2 | 56 | 96.6% | 10,718.3 | nonexistent solution |
| 4 | 54 | 93.1% | 36.4 | solution found |
| 8 | 50 | 86.2% | 39.9 | solution found |

All 135 rows (75 + 60) completed with `error=""` and zero coupling-map
violations. Graph properties (both constructed with `degree_seed=42`,
independent of the circuit seeds above): n=116 graph -- 137 edges, average
degree 2.259, degree distribution {1:3, 2:80, 3:33}, connected, perfect
matching confirmed at construction time; n=58 graph -- 69 edges, average
degree 2.207, degree distribution {1:5, 2:36, 3:17}, connected, perfect
matching confirmed at construction time. Both are close to heavy-hex's own
average degree (2.296 at d=7, 2.246 at d=5) and share its degree-1/2/3
mix, but -- unlike heavy-hex -- have exactly balanced bipartite parts, so
`spare=0` here is a genuine, zero-slack, 100%-of-device-qubits condition,
which heavy-hex cannot ever present (Addendum 39).

## 2. Scoring against the pre-registration

**P1 (primary), scored exactly as written -- and why the literal score is
misleading.** The pre-registration's comparison point was `spare=0` vs.
`spare=2`. **Measured ratio: 18,250.5 / 18,104.4 = 1.008x** (n=116) and
10,784.2 / 10,718.3 = 1.006x (n=58) -- both far under the pre-registered
2x threshold, which by the letter of P1 scores as **"cliff absent."**
**This is the wrong conclusion, and the data itself shows why**:
`spare=2` is not actually outside the slow region for either graph --
both `spare=0` and `spare=2` return `VF2Layout_stop_reason="nonexistent
solution"` and both take ~18 (or ~10.8) seconds. The pre-registration
chose `spare=2` as its "away from the cliff" reference point by analogy
with the square grid, where Addendum 34 found the cliff confined to a
single step (`spare=0` alone; `spare=1` already flat). **That analogy
does not hold on this graph**: the slow region here extends through
`spare=4` (n=116) or `spare=2` (n=58) before clearing. Scoring P1 as
specified would report a false negative. The corrected comparison --
`spare=0` against the smallest spare value that actually returned
"solution found" -- gives **18,250.5 / 60.3 = 302.6x** (n=116) and
10,784.2 / 36.4 = 296.3x (n=58). **By this corrected comparison, P1's
underlying question is answered unambiguously: yes, a large cliff is
present, decisively above the 10x confirmation bar.** This is recorded as
a miss in the pre-registration's specific choice of comparison point, not
a miss in the phenomenon being real -- the same distinction this project
drew for TKET's narrow 5x miss in Addendum 35, applied here to a
reference-point design flaw rather than a magnitude shortfall.

**P2 (qualitative comparison against the grid).** The corrected ~296-303x
cliff sits squarely inside the grid's own documented 193-353x range
(Addenda 34-35), despite this graph's average degree (~2.2-2.3) being
roughly half the grid's (~3.7-4 at 6x7/8x8). The pre-registration
explicitly did not commit to a specific ratio here and only expected the
result to land "somewhere below the grid's own range" if a cliff
appeared, reasoning from the lower degree. **That soft expectation was not
met** -- the cliff here is not smaller than the grid's, it is squarely
within it. This further undercuts "lower degree tempers the cliff's
severity" as well as "lower degree prevents the cliff outright."

**P3 (stop-reason check).** **CONFIRMED, exactly**, once the corrected
comparison is used: `VF2Layout_stop_reason` is "nonexistent solution" at
every spare value in the slow region ("nonexistent solution") and
"solution found" at every spare value in the fast region, at both graph
sizes, with a clean, coincident flip at the same point the timing drops --
exactly mirroring Addendum 34's grid result, and exactly the opposite of
Addendum 40's heavy-hex result (100% "solution found" throughout, because
heavy-hex never reaches the slow region's occupancy range at all).

**P4 (both sizes agree in direction).** **CONFIRMED.** Both the n=58 and
n=116 graphs show the identical qualitative shape: slow with "nonexistent
solution" from 100% down to 96.6% occupancy, fast with "solution found"
from 93.1% occupancy down. The occupancy percentages at which the flip
occurs are identical between the two independently-constructed graphs
(96.6% still slow, 93.1% already fast at both sizes) -- a striking
agreement given the two graphs have different node counts, different
random constructions, and only approximately matched degree statistics.
This suggests the relevant boundary here may be occupancy-*relative*
rather than tied to a fixed absolute spare-qubit count -- consistent with,
though not proof of, a genuine size-independent property of this
graph family. This exact coincidence was not predicted in the
pre-registration (which made no claim about *where* the boundary would
land, only that both sizes would agree in *direction*) and is reported as
a bonus observation, not a scored prediction.

## 3. A new, unregistered observation: the slow region is wider here than on the grid

Not part of any pre-registered prediction, but worth recording plainly:
the square grid's cliff (Addendum 34) was a single-step event -- 100%
occupancy cliffed, 97.6% (one spare qubit) was already flat. On both
graphs here, the slow region persists all the way to 96.6% occupancy
(`spare=2` on the 58-qubit graph, `spare=4` on the 116-qubit graph) before
clearing by 93.1%. The exact boundary was not finely sampled (only 0, 2,
4, 8, 16 were tested, not the single-step-resolution sweep Addendum 34
used on the grid), so the true width of the slow region is bracketed, not
pinned, here. If real, a wider slow region on a sparser graph would be a
second point of qualitative difference from the grid (in addition to the
occupancy-relative flip point noted in P4) -- but with only two sizes and
coarse spare steps tested, this is a candidate pattern, not an
established one, and a finer sweep (matching Addendum 34's own
methodology) would be needed to confirm it before treating it as
anything more than a lead.

## 4. What this addendum settles, and what it leaves open

**Settled**: heavy-hex's immunity to the occupancy cliff (Addendum 40) is
not explained by its low average degree or its sparse, bipartite
structure being inherently protective against the cliff mechanism. A
graph matched to heavy-hex's own degree profile, differing only in having
balanced (rather than imbalanced) bipartite parts, shows a cliff of
comparable magnitude to the square grid's, as soon as it is pushed to true
100% device occupancy -- something heavy-hex's own structural imbalance
(Addendum 39) permanently prevents. The most economical explanation for
Addendum 40's null result is therefore the occupancy ceiling alone, not
degree or sparsity.

**Not settled**: whether real heavy-hex's *specific* local structure
(hexagonal plaquettes, flag qubits) would add any further protection
beyond degree statistics, if it could somehow be pushed past its own
occupancy ceiling (it cannot, with this circuit family, per Addendum 39)
-- this addendum's synthetic control is a random configuration-model
graph, not heavy-hex itself, so it cannot speak to heavy-hex's particular
topology beyond matching its coarse degree statistics. Not settled:
whether the apparently wider slow region on this sparser graph (Section 3)
is a real, general property of low-degree graphs or an artifact of the
particular random instances tested here -- untested at finer spare-value
resolution. Not settled: whether a real, calibrated device topology
(`FakeTorino`) would behave like real heavy-hex (immune, via the occupancy
ceiling) or like this synthetic control (cliffs, if it could somehow be
pushed to 100% occupancy) -- still a separate, listed open item.

## 5. Files

| File | What it is |
|---|---|
| [`synthetic_sparse_balanced_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/synthetic_sparse_balanced_cliff.py) | the script |
| [`synthetic_sparse_balanced_cliff_n116_merged_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n116_merged_2026-09-18.csv) | full d7-scale results (75 rows, merged from 4 sub-runs split for wall-clock-time reasons) |
| [`synthetic_sparse_balanced_cliff_n58_merged_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n58_merged_2026-09-18.csv) | full d5-scale results (60 rows, merged from 2 sub-runs) |
| [`spare-qubit-cliff-addendum-41-preregistration-2026-09-18.md`](#addendum-41----pre-registration-is-heavy-hexs-immunity-addendum-40-caused-by-low-degreesparsity-or-simply-because-the-topology-can-never-reach-true-100-device-occupancy-2026-09-18) | predictions, written before this run |
| this document | the results write-up |

## 6. Verification

- All 135 rows (75 + 60) completed with `error=""` and zero coupling-map
  violations, confirmed directly from the merged CSVs with pandas, not
  from the script's own printed summaries alone.
- `VF2Layout_stop_reason` was checked to be internally consistent within
  each (graph, spare) cell -- all 15 repeats at a given spare value agreed
  on stop reason at both graph sizes, with no mixed cells.
- Each graph's construction (bipartite, connected, degree distribution,
  perfect-matching-confirmed) was verified independently of Qiskit, via
  `networkx.bipartite.maximum_matching`, at construction time, before any
  `transpile()` call -- printed in each run's own console header and
  re-confirmed by re-reading the construction parameters recorded in every
  CSV row (`avg_degree`, `degree_seed`, `n_construction_tries`).
- The full pre-registered sweep for each graph size was split across
  multiple process launches purely to stay under this session's per-command
  wall-clock limit (each slow spare value alone took ~6-7 minutes for its
  full 5-seed x 3-repeat allotment); this did not change the design
  (seeds, repeats, spare values, or `seed_transpiler`) from what the
  pre-registration specified, only the number of separate invocations
  used to collect it. The resulting CSVs were concatenated with pandas
  before any analysis in this document.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document, the script, and both
  merged CSVs -> 0 hits in all four.

---


<!-- ===== Addendum 42 pre-registration (source: spare-qubit-cliff-addendum-42-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for resolving the slow region's width at finer spare resolution.

## Addendum 42 -- Pre-registration: is the slow region genuinely wider on a low-degree sparse graph than on a square grid, and where exactly does it end? (2026-09-18)

**Status: pre-registration only. No measurement at the finer spare
resolution described below has been run.** The coarse data this
prediction is reasoning from is already on record (Addendum 41,
`spare` values 0, 2, 4, 8, 16 only). No new timing run has been executed
between that addendum and this document. Predictions are locked before
any finer sweep.

## 1. Why this experiment exists

Addendum 41 recorded, as an explicitly unregistered observation
(its Section 3), that the slow region on the synthetic low-degree
bipartite graphs appears **wider** than on the square grid:

- **Square grid** (Addendum 34, single-step resolution sweep): the cliff
  is a one-step event. `spare=0` (100% occupancy) is catastrophically
  slow; `spare=1` (97.6% occupancy on 6x7) is already completely flat.
- **Synthetic sparse balanced graphs** (Addendum 41, coarse sweep): slow
  at `spare=0` *and* `spare=2` at both sizes, and additionally at
  `spare=4` on the n=116 graph -- clearing only by `spare=8` (n=116) or
  `spare=4` (n=58).

Addendum 41 flagged this honestly as a candidate pattern rather than an
established one, for a specific reason: **its spare values were 0, 2, 4,
8, 16 -- so the boundary is bracketed, not located.** On the n=116 graph
all that is known is "still slow at 4, already fast at 8"; on n=58, "still
slow at 2, already fast at 4." The untested values in between (1, 3, 5, 6,
7 on n=116; 1, 3 on n=58) are exactly where the answer lives.

This matters beyond bookkeeping. If the slow region really is a *band*
several spare-qubits wide on sparse graphs, rather than the single point
it is on grids, then "cliff" is the wrong word for the sparse case and
the phenomenon's shape is topology-dependent in a second, previously
unnoticed way -- on top of the severity differences already documented
across SDKs (Addenda 35, 37) and the occupancy-ceiling effect (Addenda
40-41).

## 2. What is being measured

The same script (`synthetic_sparse_balanced_cliff.py`), the same two
graphs (`degree_seed=42`, n=58 and n=116 per side-pair counts as
constructed in Addendum 41 -- identical construction parameters, so the
same graph instances), the same circuit family, the same
`optimization_level=3`, the same 5 seeds x 3 repeats per cell. **The only
change is the spare value list**, filled in at single-step resolution
across the bracketed boundary:

- **n=116**: `spare` in {0, 1, 2, 3, 4, 5, 6, 7, 8}
- **n=58**: `spare` in {0, 1, 2, 3, 4, 5}

Values above the known-fast point are not re-swept (Addendum 41 already
has 8 and 16 for n=116, 8 for n=58, all fast); the point is to resolve
the transition, not re-confirm the flat region.

## 3. Pre-registered predictions

The operative signal, per Addendum 41's own finding that timing and
`VF2Layout_stop_reason` flip together and coincidentally, is the stop
reason: **"nonexistent solution" = in the slow region, "solution found" =
out of it.** Timing is reported alongside but the stop reason is the
cleaner boundary marker and is what P1-P3 are scored on.

**P1 (primary -- is the slow region actually wider than one step?).**
Define the slow-region width as the number of consecutive spare values
starting at 0 that return "nonexistent solution" in a majority (>= 8 of
15) of runs.
  - **Wider-than-grid confirmed** if width >= 3 on the n=116 graph (i.e.
    `spare=0,1,2` all slow at minimum), which is unambiguously more than
    the grid's single-step behaviour.
  - **Grid-like after all** if width == 1 on the n=116 graph (only
    `spare=0` slow) -- which would mean Addendum 41's `spare=2` slow
    reading was something other than a contiguous band, and would itself
    demand explanation.
  - **Intermediate** if width == 2 -- reported as "wider than the grid but
    narrower than the coarse data suggested," not rounded to either side.

**P2 (does the boundary land at the same occupancy fraction at both
sizes?).** Addendum 41 noted, as a bonus observation, that both graphs
flipped between 96.6% and 93.1% occupancy despite different node counts.
At single-step resolution, the last slow spare value and the first fast
one define an occupancy bracket at each size.
  - **Occupancy-relative confirmed** if the two sizes' brackets overlap
    (i.e. there is at least one occupancy percentage consistent with being
    the boundary at both sizes).
  - **Not occupancy-relative** if the brackets are disjoint, which would
    argue the boundary tracks something else (absolute spare count,
    absolute qubit count, or a graph-instance-specific property).

**P3 (stop-reason/timing coincidence holds at finer resolution).** At
every spare value tested, the stop reason and the timing are predicted to
agree on which side of the boundary that value falls -- i.e. no value
should show "nonexistent solution" with a fast (<200 ms) time, or
"solution found" with a slow (>1 s) time. A mismatch at any single spare
value would be a notable finding in its own right (it would mean the two
signals decouple near the boundary) and is worth reporting separately if
it occurs.

**P4 (no prediction is made about the boundary's exact location).** This
is stated explicitly so that no post-hoc "we predicted spare=N" claim can
be read into this document. The bracket is 0-8 (n=116) and 0-4 (n=58);
anywhere inside it is consistent with this pre-registration.

## 4. What this experiment cannot establish

- **Why** a sparser graph would have a wider slow region, if it does.
  This measures the shape, not the mechanism.
- Whether the width generalizes beyond these two specific random graph
  instances (`degree_seed=42`). Two instances at two sizes is not a
  family. A genuinely general claim would need several independently
  seeded graphs per size.
- Whether real heavy-hex would show the same width if it could reach this
  occupancy range -- it cannot (Addendum 39), so this remains permanently
  untestable on the real topology with this circuit family.
- Anything about other SDKs. This is Qiskit-only, as Addenda 40-41 were.

## 5. Cost note

Addendum 41's verification section recorded ~6-7 minutes per slow spare
value (5 seeds x 3 repeats). This sweep adds up to 5 new slow-region
values on n=116 and up to 3 on n=58, so **30-55 minutes of compute is a
realistic expectation** if most of the new values land in the slow
region. Values that land in the fast region cost seconds. Splitting the
run across multiple invocations (as Addendum 41 did, for the same reason)
does not change the design and is not a deviation from it.

---


<!-- ===== Addendum 42 (source: spare-qubit-cliff-addendum-42-2026-09-18.md) ===== -->

> **Note added when merging:** The sparse-graph slow region is a multi-step staircase, not a one-step cliff, and `VF2PostLayout` stops costing time one occupancy step before `VF2Layout` becomes cheap. Also records a pre-registration design flaw: the dense-pair circuit family requires even qubit counts, so single-step resolution is structurally impossible.

## Addendum 42 -- the slow region on a sparse balanced graph is a multi-step staircase, not a one-step cliff, and the two VF2 passes clear at different points (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-42-preregistration-2026-09-18.md`, written
and locked before this run. Every prediction is scored below against the
actual data. **One design flaw in that pre-registration is reported
first, because it limits what P1 and P2 can actually claim.**

## 0. In one line

At the finer spare resolution this addendum set out to measure, the
sparse-graph slow region is confirmed **wider than the square grid's
single-step cliff** -- three consecutive measured points (spare=0, 2, 4)
at n=116, two (spare=0, 2) at n=58, all returning `VF2Layout_stop_reason
= "nonexistent solution"` in 15/15 runs. But the shape is not a cliff at
all: n=116 descends **33.6 s -> 32.5 s -> 12.0 s -> 669 ms -> 39.7 ms**
across spare 0/2/4/6/8, a **staircase with at least three distinct
steps**, and the per-pass breakdown shows why -- **`VF2PostLayout` stops
running entirely one step before `VF2Layout` becomes cheap.** The two
searches Addendum 34 identified as the cliff's real cost are released at
*different* occupancy thresholds, something the square grid's single-step
transition made impossible to see.

## 1. A pre-registration design flaw, reported before the results

The pre-registration specified single-step resolution: `spare` in
{0,1,2,...,8} for n=116 and {0,...,5} for n=58. **Every odd spare value
was skipped by the script**, correctly and by design: the dense-pair
circuit family requires an even qubit count, and both device sizes are
even, so an odd spare yields an odd `n` and is rejected
(`spare=1 skipped (n=115 invalid -- must be even and >=2)`).

**This means "single-step resolution" is not achievable for this circuit
family at all; the minimum step is 2.** The pre-registration's P1
threshold ("width >= 3") was written assuming steps of 1 and therefore
counts *measured points*, not spare-qubits. This was a mistake in writing
the pre-registration, not in the script, and it is recorded here rather
than quietly re-scoped. Its practical effect: the boundary is located to
within 2 spare qubits, not 1, and P1's "width" is in units of measured
points.

## 2. Results

### n=116 (58 per side), optimization_level=3, 5 seeds x 3 repeats

| spare | n | occupancy | median total (ms) | `VF2Layout` (ms) | `VF2PostLayout` (ms) | stop reason |
|---:|---:|---:|---:|---:|---:|:---|
| 0 | 116 | 100.0% | 33,587.2 | 16,706.8 | 17,275.4 | nonexistent solution (15/15) |
| 2 | 114 | 98.3% | 32,534.5 | 15,550.5 | 15,749.7 | nonexistent solution (15/15) |
| 4 | 112 | 96.6% | 12,049.2 | 6,034.1 | 6,000.0 | nonexistent solution (15/15) |
| 6 | 110 | 94.8% | 669.2 | 651.1 | **0.0** | solution found (15/15) |
| 8 | 108 | 93.1% | 39.7 | 17.4 | 0.0 | solution found (15/15) |

### n=58 (29 per side), optimization_level=3, 5 seeds x 3 repeats

| spare | n | occupancy | median total (ms) | `VF2Layout` (ms) | `VF2PostLayout` (ms) | stop reason |
|---:|---:|---:|---:|---:|---:|:---|
| 0 | 58 | 100.0% | 15,618.8 | 6,422.1 | 9,385.7 | nonexistent solution (15/15) |
| 2 | 56 | 96.6% | 19,904.0 | 9,398.8 | 9,386.3 | nonexistent solution (15/15) |
| 4 | 54 | 93.1% | 73.3 | 49.9 | **0.0** | solution found (15/15) |

All 120 rows (75 + 45) completed with `error=""`. Graph construction
parameters identical to Addendum 41 (`degree_seed=42`): n=116 graph
avg degree 2.259, n=58 graph avg degree 2.207, both connected with a
verified perfect matching.

## 3. Scoring

**P1 (is the slow region wider than the grid's single step?).**
Pre-registered: width >= 3 measured points at n=116 confirms
"wider than grid." **Measured: 3 points (spare=0, 2, 4), all 15/15
"nonexistent solution." P1 CONFIRMED.** At n=58 the width is 2 points
(spare=0, 2). For comparison, the square grid (Addendum 34) was slow at
spare=0 and already fully flat at spare=1 -- one point, and a *smaller*
step than the 2 used here. The sparse-graph slow region is therefore
wider on both counts, though (per Section 1) its exact edge is bracketed
to within 2 spare qubits rather than pinned.

**P2 (does the boundary land at the same occupancy fraction at both
sizes?).** Pre-registered: confirmed if the two sizes' occupancy brackets
overlap. **Measured: n=116's boundary lies in [94.8%, 96.6%]; n=58's in
[93.1%, 96.6%]. These overlap on [94.8%, 96.6%]. P2 CONFIRMED**, in the
weak sense the prediction allowed -- the brackets are consistent with a
single occupancy-relative boundary, but are wide enough (thanks to the
step-of-2 limitation) that they would also be consistent with two
different boundaries a few percent apart. **This is weaker evidence than
Addendum 41's coincidence suggested**, and should not be cited as
"the boundary is at the same occupancy at both sizes" without that
caveat.

**P3 (stop reason and timing agree on which side of the boundary each
point falls).** Pre-registered thresholds: fast = <200 ms, slow = >1 s,
mismatch at any point is separately reportable. **Measured: every
"nonexistent solution" point is >1 s (12.0-33.6 s) -- consistent. Every
"solution found" point at n=58 is <200 ms (73.3 ms) -- consistent. But
n=116's `spare=6` is "solution found" at 669.2 ms**, which is neither
<200 ms nor >1 s. **P3 is therefore not cleanly confirmed**: there is one
intermediate point that falls in the gap the pre-registration left
unlabelled. Its per-pass breakdown (Section 4) explains it and makes it
more interesting than a failed threshold check, but the letter of P3 is
not met, and that is recorded as stated rather than adjusted.

**P4 (no prediction about the boundary's exact location).** Honoured --
no claim of having predicted spare=4 or spare=6 is being made.

## 4. The unregistered finding, and it is the important one

The pre-registration asked about the slow region's *width*. The data
answers a question it did not ask: **what the staircase's steps are made
of.**

At n=116, reading the per-pass columns across the transition:

- **spare=0 and 2** (100%, 98.3%): `VF2Layout` ~16 s **and**
  `VF2PostLayout` ~16-17 s. Both searches run, both are expensive.
- **spare=4** (96.6%): both drop to ~6 s. Still both running, both
  roughly halved.
- **spare=6** (94.8%): `VF2PostLayout` is **exactly 0.0 ms -- it stops
  running entirely** -- while `VF2Layout` remains at 651 ms, still two
  orders of magnitude above its floor.
- **spare=8** (93.1%): `VF2Layout` drops to 17.4 ms.

The same shape appears at n=58: `VF2PostLayout` goes 9,385.7 ms ->
9,386.3 ms -> **0.0 ms** across spare 0/2/4, while `VF2Layout` does not
reach its floor until the same step.

**The two VF2-family passes Addendum 34 identified as the cliff's joint
cost do not clear together.** `VF2PostLayout` switches off at a distinct
occupancy threshold, one step before `VF2Layout` becomes cheap. On the
square grid (Addendum 34) both cleared within the single step from
spare=0 to spare=1, so this separation was invisible there. **The
staircase is not one phenomenon with a wide boundary; it is at least two
thresholds at different occupancies, which the sparse graph spreads far
enough apart to resolve.**

## 5. A second unregistered observation: monotonicity fails at n=58

At n=58, `spare=2` (19,904 ms) is **slower** than `spare=0` (15,618 ms) --
a ratio of 0.78x in the wrong direction. Every prior measurement in this
project, on grids and at n=116 here, showed time decreasing monotonically
as occupancy falls. The per-pass breakdown localises it: `VF2PostLayout`
is essentially identical at both points (9,385.7 vs 9,386.3 ms), while
**`VF2Layout` alone rises from 6,422 ms to 9,399 ms** going from 100% to
96.6% occupancy.

No mechanism is proposed here. Two candidates worth distinguishing in a
follow-up: (a) a genuine property of this graph instance near the
boundary, or (b) noise -- but the 15-run medians and the clean per-pass
attribution argue against simple noise, and this project's own history
(Addenda 27-29) shows anomalies that looked like noise deserve a
dedicated check rather than a dismissal. **Untested either way.**

## 6. What this does and does not establish

**Established**: on these two synthetic sparse balanced graphs, the slow
region spans multiple measured points rather than the grid's single
point; `VF2PostLayout` and `VF2Layout` clear at different occupancy
thresholds; and the boundary occupancy brackets at the two sizes overlap.

**Not established**: the boundary's location to better than 2 spare
qubits (structurally impossible with this circuit family -- Section 1);
whether the staircase shape generalises beyond these two `degree_seed=42`
instances; why `VF2PostLayout` switches off where it does; whether the
n=58 non-monotonicity is real; whether any of this holds on other SDKs
(Qiskit only here, as in Addenda 40-41) or on a real device topology.

## 7. Files

| File | What it is |
|---|---|
| [`synthetic_sparse_balanced_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/synthetic_sparse_balanced_cliff.py) | the script (unchanged from Addendum 41) |
| [`synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | n=116 results, 75 rows |
| [`synthetic_sparse_balanced_cliff_n58_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n58_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | n=58 results, 45 rows |
| [`spare-qubit-cliff-addendum-42-preregistration-2026-09-18.md`](#addendum-42----pre-registration-is-the-slow-region-genuinely-wider-on-a-low-degree-sparse-graph-than-on-a-square-grid-and-where-exactly-does-it-end-2026-09-18) | the predictions scored above |

## 8. Verification

- All 120 rows checked for `error=""` directly from the CSVs.
- `vf2_stop_reason` verified unanimous (15/15) within every (size, spare)
  cell -- no mixed cells at either size.
- Per-pass timings (`vf2layout_ms`, `vf2postlayout_ms`) read from the
  CSVs' own columns, not inferred from totals; the `VF2PostLayout = 0.0`
  readings were confirmed to be exact zeros across all 15 runs in each
  affected cell, not small values rounded down.
- The odd-spare skip (Section 1) was confirmed by reading the script's
  own guard (`if n < 2 or n % 2 != 0`) rather than inferred from the
  missing rows alone.
- Environment: Intel machine (`Intel64 Family 6 Model 181 Stepping 0`),
  this project's historical Intel host -- unlike Addenda 40-41, which ran
  on a cloud sandbox with different hardware. Cross-machine comparison of
  absolute times between this addendum and those is therefore not valid;
  only the within-run shapes are compared here.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both CSVs -> 0 hits.

---


<!-- ===== Addendum 43 (source: spare-qubit-cliff-addendum-43-2026-09-18.md) ===== -->

> **Note added when merging:** Re-analyses every existing per-pass CSV column and reads Qiskit's own release notes. Catalogues three apparently distinct `VF2PostLayout` behaviours -- **the third of which Addendum 48 later retracts as a cross-machine artefact.**

## Addendum 43 -- re-analysis of existing per-pass data across all three topology families, and what Qiskit's own release notes say about it (2026-09-18)

**Status**: re-analysis of already-collected data plus a documentation
review. **No new measurement was run.** Every number below comes from
CSVs already on record (Addenda 34, 40, 42). Nothing here required
executing `transpile()` again.

## 0. In one line

Addendum 42 found that on a synthetic sparse graph, `VF2PostLayout` stops
running one occupancy step *before* `VF2Layout` becomes cheap -- the two
searches Addendum 34 identified as the cliff's joint cost clear at
different thresholds. **Re-reading the per-pass columns of every dataset
this project has collected shows three qualitatively different behaviours
across three topology families, and Qiskit's own 2.2 release notes
describe a change targeting precisely the workload this project has been
using.**

## 1. The three behaviours, from existing data

All at `optimization_level=3`, medians across all seeds/repeats, read
directly from the `vf2layout_ms` and `vf2postlayout_ms` columns.

### Square grid 6x7 (Addendum 34): the two passes clear together

| spare | n | occupancy | total (ms) | `VF2Layout` | `VF2PostLayout` | stop reason |
|---:|---:|---:|---:|---:|---:|:---|
| 0 | 42 | 100.0% | 15,583.9 | 8,450.8 | 6,712.3 | nonexistent solution |
| 1 | 41 | 97.6% | 80.7 | 47.6 | **0.0** | solution found |
| 2 | 40 | 95.2% | 65.3 | 34.7 | 0.0 | solution found |
| ... | | | (65-105 ms, flat out to spare=24) | | 0.0 | solution found |

**No separation.** Both passes are expensive at spare=0 and both are
released in the same single step to spare=1.

### Synthetic sparse balanced n=116 (Addendum 42): the two passes separate

| spare | occupancy | total (ms) | `VF2Layout` | `VF2PostLayout` | stop reason |
|---:|---:|---:|---:|---:|:---|
| 0 | 100.0% | 33,587.2 | 16,706.8 | 17,275.4 | nonexistent solution |
| 2 | 98.3% | 32,534.5 | 15,550.5 | 15,749.7 | nonexistent solution |
| 4 | 96.6% | 12,049.2 | 6,034.1 | 6,000.0 | nonexistent solution |
| 6 | 94.8% | 669.2 | 651.1 | **0.0** | solution found |
| 8 | 93.1% | 39.7 | 17.4 | 0.0 | solution found |

**Clear separation.** `VF2PostLayout` goes to exactly 0.0 at spare=6
while `VF2Layout` is still at 651 ms -- two orders of magnitude above its
own floor of 17.4 ms, reached only at spare=8.

### Heavy-hex d=7 (Addendum 40): neither pass is ever expensive, and `VF2PostLayout` is never zero

| spare_pairs | total (ms) | `VF2Layout` | `VF2PostLayout` | stop reason |
|---:|---:|---:|---:|:---|
| 0 | 69.2 | 42.8 | **0.4** | solution found |
| 1 | 70.6 | 44.3 | 0.4 | solution found |
| 2 | 72.4 | 46.0 | 0.4 | solution found |
| 4 | 74.5 | 48.6 | 0.4 | solution found |
| 8 | 75.2 | 49.9 | 0.4 | solution found |
| 16 | 69.2 | 46.8 | 0.4 | solution found |

**A third behaviour entirely**: `VF2PostLayout` is a small, constant,
**non-zero** 0.4 ms at every occupancy tested (0.2 ms at d=5).

### The zero/non-zero distinction is not a rounding artefact

Counting exact values across every row of each dataset:

| dataset | `VF2PostLayout` exactly 0.0 | 0 < x < 1 ms | >= 1 ms | total rows |
|---|---:|---:|---:|---:|
| grid 6x7 (opt3) | 92 | 3 | 22 | 117 |
| heavy-hex d7 | **0** | **90** | 0 | 90 |
| synthetic n116 | 27 | 0 | 48 | 75 |

Heavy-hex never once produces an exact zero; grid and synthetic produce
them in bulk. **An exact 0.0 and a consistent 0.2-0.4 ms are different
events**, and the most natural reading is that they distinguish "the pass
did not run at all" from "the pass ran and returned almost immediately" --
though this addendum does not confirm that reading from source (Section
3).

## 2. What Qiskit's own documentation and release notes say

Three findings from Qiskit's published documentation, all directly
relevant, none of which this project had previously consulted:

**(a) Qiskit 2.2's release notes describe a change aimed at exactly this
workload.** Verbatim: *"The maximum call and trial limits for the
exact-matching run of `VF2PostLayout` at `optimization_level=3` have been
reduced to avoid excessive runtimes for **highly symmetric trial circuits
being mapped to large coupling maps**."* The circuit family used
throughout this project -- a disjoint union of N/2 unconnected 2-node
edges -- is maximally symmetric, and the coupling maps here are large.
**Qiskit's developers have already identified and acted on this class of
problem.** All measurements in this project used qiskit 2.5.2, i.e. a
version where that mitigation is already in place.

**(b) The same release notes state `VF2PostLayout` can now skip
outright.** Verbatim: *"Instead in these cases `VF2PostLayout` will now
skip the search since the layout problem isn't viable for the pass."*
**What "these cases" refers to was not determined** -- see Section 3.

**(c) `VF2PostLayout` scores candidate layouts using error rates, which
none of this project's experiments provide.** Its documentation: *"By
default, this pass will construct a heuristic scoring map based on the
error rates in the provided target."* Every measurement in this project
passes a bare `CouplingMap` with no `Target` and no calibration data, so
there are no error rates to score against. `VF2PostLayout` also has a
distinct four-value stop reason enum (`SOLUTION_FOUND`,
`NO_BETTER_SOLUTION_FOUND`, `NO_SOLUTION_FOUND`, `MORE_THAN_2Q`) and,
unlike `VF2Layout`, imports **two** Rust entry points
(`vf2_layout_pass_average` *and* `vf2_layout_pass_exact`) -- consistent
with the "exact-matching run at `optimization_level=3`" the release note
names.

**A correction to this project's own instrumentation, implied by (c)**:
every addendum from 34 onward has recorded `VF2Layout_stop_reason` but
**never recorded `VF2PostLayout_stop_reason`**, which is a separate
property-set key with its own distinct values. That column would likely
distinguish the three behaviours in Section 1 directly, and its absence is
why they have to be inferred from timing alone here.

## 3. What this does NOT establish

- **The actual skip condition.** The source file
  (`qiskit/transpiler/passes/layout/vf2_post_layout.py`) could not be
  fetched directly in this session; the findings in Section 2 come from
  release notes and API documentation, which state *that* a skip exists
  without stating its precise trigger. **"These cases" remains
  unidentified.** Reading that file is the obvious next step and is not
  done here.
- **Whether the absence of error rates explains any of Section 1's three
  behaviours.** It is a plausible common factor (all three datasets lack
  them) but cannot by itself explain why the three differ from each
  other.
- **Why heavy-hex's 0.2-0.4 ms is constant** across every occupancy
  tested while grid and synthetic swing over four orders of magnitude.
- **Whether the grid's simultaneous clearing is genuinely simultaneous**
  or merely appears so because the grid's step size (1 spare qubit) was
  too coarse to resolve a separation that the sparse graph's wider slow
  region spreads far enough apart to see. **This is the single most
  important ambiguity in Section 1** and is directly testable: the grid's
  transition is a single step, so there is no finer resolution available
  there -- but a different grid size, or a differently-shaped circuit
  with an odd qubit count, might place the two thresholds on different
  steps.

## 4. Why this matters for the project's central claim

Addendum 34 established the cliff's cost is two VF2-family searches
rather than a Sabre fallback. This re-analysis shows those two searches
are **not a single unit**: on at least one topology family they respond
to occupancy at different thresholds, and on another (heavy-hex) one of
them appears never to engage at all. Any mechanistic account of the
cliff has to explain both passes separately, not "VF2 search" as a
single entity -- and the Qiskit release notes in Section 2 suggest at
least part of `VF2PostLayout`'s behaviour is governed by deliberately
tuned limits specific to `optimization_level=3` and to symmetric
circuits, not by the occupancy structure this project has been varying.

## 5. Files

| File | What it is |
|---|---|
| (no new data) | this addendum re-analyses existing CSVs only |
| [`occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | grid data (Addendum 34) |
| [`occupancy_sweep_heavy_hex_d5_IntelR_XeonR_Processor___2_80GHz_2026-09-17_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_heavy_hex_d5_IntelR_XeonR_Processor___2_80GHz_2026-09-17_run2.csv) / [`d7`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_heavy_hex_d7_IntelR_XeonR_Processor___2_80GHz_2026-09-17_run2.csv) | heavy-hex data (Addendum 40) |
| [`synthetic_sparse_balanced_cliff_n58_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n58_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) / [`n116`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | synthetic data (Addendum 42) |

## 6. Verification

- Every per-pass median in Section 1 was recomputed from the CSVs' own
  `vf2layout_ms` / `vf2postlayout_ms` columns; none was carried over from
  a previous addendum's prose.
- The zero/non-zero counts in Section 1 were computed by exact float
  comparison (`v == 0.0`) against every row of each dataset, specifically
  to distinguish a true zero from a small value that a rounded table
  would display identically.
- The qiskit version was read from each CSV's own `qiskit_version`
  column: **2.5.2 in all three datasets**, confirming the 2.2 release-note
  behaviour in Section 2(a)/(b) was already active during every
  measurement.
- Section 2's quotations are verbatim from Qiskit's published 2.2 release
  notes and current `VF2PostLayout` API documentation, retrieved this
  session. The source file itself was **not** read -- stated plainly in
  Section 3 rather than implied.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 44 (source: spare-qubit-cliff-addendum-44-2026-09-18.md) ===== -->

> **Note added when merging:** Reads `vf2_post_layout.py` in full: it dispatches to `vf2_layout_pass_exact` by default, requires a `Target`, has four distinct stop reasons, and contains **no skip logic** -- that lives in the Rust.

## Addendum 44 -- reading `VF2PostLayout`'s source: the exact code path, the four stop-reason conditions, and where the "skip" is not (2026-09-18)

**Status**: source reading. No measurement. Addendum 43 Section 3 listed
"the actual skip condition" as unidentified because the source file could
not be fetched that session. It was fetched this session
(`qiskit/transpiler/passes/layout/vf2_post_layout.py`, main branch, 168
lines) and read in full. This addendum records what it settles and what
it does not.

## 0. In one line

`VF2PostLayout.run()` is a thin dispatcher: it **requires** a `Target`
(raises otherwise), always sets `score_initial_layout=True`, and routes to
one of two Rust entry points based on `strict_direction` -- **which
defaults to `True`, selecting `vf2_layout_pass_exact`.** That is the same
"exact-matching run" Qiskit's 2.2 release notes say had its limits reduced
"to avoid excessive runtimes for highly symmetric trial circuits being
mapped to large coupling maps" -- i.e. **this project's exact workload runs
down the exact path, by default.** The four stop-reason conditions are now
known precisely. **The skip logic itself is not in this file** -- it is
inside the Rust, so Addendum 43's open question is narrowed, not closed.

## 1. The dispatch, verbatim

```python
def run(self, dag):
    if self.target is None:
        raise TranspilerError("A target must be specified")
    self.avg_error_map = self.property_set["vf2_avg_error_map"]
    config = VF2PassConfiguration.from_legacy_api(
        call_limit=self.call_limit,
        time_limit=self.time_limit,
        max_trials=self.max_trials,
        shuffle_seed=self.seed,
        score_initial_layout=True,
    )
    try:
        if self.strict_direction:
            output = vf2_layout_pass_exact(dag, self.target, config=config)
        else:
            output = vf2_layout_pass_average(
                dag, self.target, strict_direction=False,
                avg_error_map=self.avg_error_map, config=config,
            )
    except MultiQEncountered:
        ...
```

Four facts follow directly:

1. **`strict_direction` defaults to `True`** (from the `__init__`
   signature), so `vf2_layout_pass_exact` is the default path. `VF2Layout`
   by contrast imports only `vf2_layout_pass_average` -- **the two passes
   run different Rust entry points**, which is a concrete mechanism for
   the different behaviour Addendum 43 observed between them, rather than
   the two being "the same VF2 search run twice."
2. **A `Target` is mandatory** -- `run()` raises if it is `None`. When
   this project calls `transpile(qc, coupling_map=..., basis_gates=...)`,
   Qiskit builds a `Target` internally from those, so the pass does run --
   but on a target whose `InstructionProperties` carry **no error rates**,
   since none were supplied.
3. **`score_initial_layout=True` is always set**, unconditionally. The
   pass always scores the layout it was handed before looking for a better
   one.
4. `avg_error_map` is read from the property set but is **only forwarded
   on the `strict_direction=False` branch** -- on the default exact path it
   is computed and then not passed.

## 2. The four stop reasons, and their exact conditions

Read directly from the source, in evaluation order:

| stop reason | condition |
|---|---|
| `MORE_THAN_2Q` (`">2q gates in basis"`) | `MultiQEncountered` raised by the Rust call |
| `NO_SOLUTION_FOUND` (`"nonexistent solution"`) | `not output.has_solution` |
| `NO_BETTER_SOLUTION_FOUND` (`"no better solution found"`) | `dag.is_empty()` **or** `output.new_mapping() is None` |
| `SOLUTION_FOUND` (`"solution found"`) | everything else; also the only branch that sets `property_set["post_layout"]` |

**This is directly relevant to Addendum 43's instrumentation gap.**
`"no better solution found"` is a distinct outcome from `"solution
found"`, and this project has never recorded
`VF2PostLayout_stop_reason` at all -- only `VF2Layout_stop_reason`.
The three behaviours catalogued in Addendum 43 Section 1 (grid: both
passes clear together; synthetic sparse: they separate; heavy-hex:
`VF2PostLayout` constant at 0.2-0.4 ms) would very likely be
distinguished outright by this column. **Adding it is a one-line change
to the instrumentation and should be done before any further
interpretation of those three behaviours.**

## 3. What this does NOT settle

- **The skip.** Qiskit 2.2's release note ("`VF2PostLayout` will now skip
  the search since the layout problem isn't viable for the pass") has **no
  counterpart in this Python file** -- there is no early return, no
  viability check, nothing between the `Target` check and the Rust call.
  The skip must live inside `vf2_layout_pass_exact` (Rust, in
  `qiskit._accelerate.vf2_layout`). **Addendum 43's open question is
  narrowed to that Rust function, not answered.**
- **Why the observed times are exactly `0.0`.** Nothing here explains a
  hard zero rather than a small positive number. If the Rust returns
  immediately on a viability check, a zero-ish measurement is plausible,
  but that is inference, not something this file shows.
- **What `vf2_layout_pass_exact` does with a target carrying no error
  rates.** The pass always scores (`score_initial_layout=True`) and the
  exact path never receives `avg_error_map`, so what it scores against
  when the target has no `InstructionProperties` is not visible from
  Python.
- **Whether `strict_direction` is left at its default in the preset
  pipelines.** This file shows the default is `True`; whether
  `generate_routing_passmanager` or the level-3 preset overrides it was
  not checked.

## 4. Next step, now specific

Read `vf2_layout_pass_exact` in the Rust source
(`crates/`, `qiskit._accelerate.vf2_layout`) for: the viability check that
produces the 2.2 "skip"; the default `call_limit`/`max_trials` values at
`optimization_level=3` after the 2.2 reduction; and the scoring behaviour
when no error rates are present. That is the remaining gap between this
project's timing observations and a mechanistic account of them.

Separately and independently: **record
`VF2PostLayout_stop_reason` in the sweep scripts** (Section 2). That
requires no source reading and would likely resolve Addendum 43's
three-behaviour puzzle on its own.

## 5. Verification

- The code block in Section 1 and the stop-reason table in Section 2 were
  transcribed from the fetched file, not paraphrased from documentation.
- The claim that `VF2Layout` imports only `vf2_layout_pass_average` while
  `VF2PostLayout` imports both it and `vf2_layout_pass_exact` was checked
  against both files' import blocks.
- The absence of a skip in the Python file was confirmed by reading
  `run()` end to end (it is 30 lines), not by keyword search alone.
- File identified as 168 lines, main branch, retrieved this session.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 45 (source: spare-qubit-cliff-addendum-45-2026-09-18.md) ===== -->

> **Note added when merging:** Qiskit's own C API docs state the two VF2 modes use different search-ordering heuristics: VF2++ for 'average' (`VF2Layout`), identity-start for 'exact' (`VF2PostLayout`) -- a documented mechanism for why the two clear at different thresholds. **UPDATE (Addendum 85, Part 5): reading Qiskit's actual current Rust source directly showed both `vf2_layout_pass_average` and `vf2_layout_pass_exact` call the identical `.with_vf2pp_ordering()` -- this addendum's documentation-based reading of "different orderings" was incorrect. The "identity-start" language most likely refers to `score_initial_layout`, a scoring detail, not the traversal order VF2++ itself governs.**

## Addendum 45 -- the two VF2 passes use different search-ordering heuristics: "average" mode uses VF2++ node ordering, "exact" mode starts from the identity mapping (2026-09-18)

**Status**: documentation reading. No measurement. Follows Addendum 44,
which established that `VF2Layout` and `VF2PostLayout` call *different*
Rust entry points (`vf2_layout_pass_average` vs. `vf2_layout_pass_exact`)
but could not say what the two do differently.

## 0. In one line

Qiskit's own C API documentation for the VF2 passes states the two modes
use **different node-ordering strategies**: *"Qiskit uses the VF2++
ordering improvements when running in 'average' mode (corresponding to
initial layout search), and **starts from the identity mapping in
'exact' mode**."* Since Addendum 44 established `VF2Layout` takes the
average path and `VF2PostLayout` (default `strict_direction=True`) takes
the exact path, **the two passes do not merely run the same search twice
-- they begin their backtracking searches from different places.** This
is a concrete, documented mechanism for the different occupancy
thresholds observed in Addenda 42-43, and it is not something this
project had considered.

## 1. The quotation, and where it comes from

From Qiskit's C API reference for `QkVF2LayoutConfiguration`
(`qk_vf2_layout_configuration_set_shuffle_seed`), describing what node
shuffling does and why it is usually not wanted:

> This effectively drives a modification of the matching order of VF2,
> which in theory means that the space of a bounded search is not biased
> based on the node indices. In practice, Qiskit uses the VF2++ ordering
> improvements when running in "average" mode (corresponding to initial
> layout search), and starts from the identity mapping in "exact" mode.
> Both of these ordering heuristics are typically far more likely to find
> results for the given problem than randomization.

Mapping this onto what Addendum 44 read from the Python source:

| pass | Rust entry point | mode | ordering heuristic |
|---|---|---|---|
| `VF2Layout` | `vf2_layout_pass_average` | "average" | VF2++ node ordering |
| `VF2PostLayout` (default) | `vf2_layout_pass_exact` | "exact" | **starts from the identity mapping** |

## 2. Why this is a plausible mechanism for the observed split

Addendum 42 found, on a synthetic sparse balanced graph, that
`VF2PostLayout` stops consuming time one occupancy step *before*
`VF2Layout` becomes cheap. Addendum 43 catalogued three distinct
behaviours across topology families and could offer no mechanism.

The documented ordering difference gives one, and it is testable rather
than merely plausible-sounding. **"Starts from the identity mapping" is
an ordering that is very good when the circuit is already laid out
roughly where it will stay** -- which is exactly `VF2PostLayout`'s
situation, since it runs *after* routing, on a circuit already mapped to
physical qubits. **VF2++ ordering, by contrast, is a general-purpose
heuristic for finding an embedding from scratch**, which is
`VF2Layout`'s situation. Two searches with different starting points and
different pruning orders, run on the same graph at the same occupancy,
have no particular reason to succeed or fail at the same threshold.

**This does not yet explain the three behaviours**, and is not claimed
to. It identifies a documented difference that a mechanism could be built
on, replacing "the two passes behave differently for unknown reasons"
with "the two passes search differently, in a specific documented way."

## 3. What is now known, and what is still missing

**Known** (Addenda 44-45 combined):
- The two passes call different Rust functions, by a `strict_direction`
  flag defaulting to `True` on `VF2PostLayout`.
- Those functions implement different ordering heuristics: VF2++ for
  average, identity-start for exact.
- `VF2PostLayout` requires a `Target`, always scores the initial layout,
  and distinguishes four outcomes including `"no better solution found"`,
  which this project has never recorded.
- Qiskit 2.2 reduced the call/trial limits specifically for the **exact**
  run at `optimization_level=3`, citing "highly symmetric trial circuits
  being mapped to large coupling maps" -- this project's exact workload.

**Still missing**:
- The Rust source of `vf2_layout_pass_exact` itself. Multiple attempts to
  locate it this session returned Python bindings, C API docs, and
  unrelated crates, but not the implementation. **The "skip" behaviour
  named in the 2.2 release notes remains unlocated in source.**
- The post-2.2 numeric values of the reduced call/trial limits.
- What either mode scores against when the `Target` carries no error
  rates, as in every measurement this project has run.

## 4. The cheapest remaining step is still not source reading

Addendum 44 already identified it and it has not been done:
**record `VF2PostLayout_stop_reason` in the sweep scripts.** With the
four-value enum now known (Addendum 44 Section 2) and the
ordering difference now documented (this addendum), that single column
would show directly whether `VF2PostLayout`'s early clearing in Addendum
42 is `"solution found"`, `"no better solution found"`, or the pass
declining to search at all -- three very different explanations that the
current instrumentation cannot distinguish.

## 5. Verification

- The quotation in Section 1 is verbatim from Qiskit's published C API
  documentation for `qk_vf2_layout_configuration_set_shuffle_seed`,
  retrieved this session.
- The mapping in Section 1's table between pass, entry point, and mode
  relies on Addendum 44's reading of `vf2_post_layout.py`'s dispatch
  (`if self.strict_direction: vf2_layout_pass_exact(...)`) plus the
  parenthetical "(corresponding to initial layout search)" in the quoted
  documentation, which identifies average mode with `VF2Layout`. **The
  documentation does not name `VF2PostLayout` explicitly**; that half of
  the mapping is inference from the entry-point names and the default
  flag value, and is labelled as such rather than asserted.
- Section 3's "still missing" items reflect actual failed retrieval
  attempts this session, not an assumption that the source is
  unavailable.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 46 (source: spare-qubit-cliff-addendum-46-2026-09-18.md) ===== -->

> **Note added when merging:** With the new instrumentation: `VF2PostLayout` is **never skipped** and returns `"no better solution found"` in 75/75 runs -- including where it spends 21 seconds doing so. Also reports a reproducibility problem that **Addendum 47 then revises.**

## Addendum 46 -- `VF2PostLayout` is never skipped and never improves anything: it returns "no better solution found" at every occupancy, including where it burns 21 seconds doing so (2026-09-18)

**Status**: results from the instrumentation added this session
(`vf2post_stop_reason`, `vf2postlayout_ran`, `vf2layout_ran`), run on the
n=116 synthetic sparse balanced graph at `optimization_level=3`, 5 seeds
x 3 repeats -- the same configuration as Addendum 42, on the same machine
and the same graph instance (`degree_seed=42`).

## 0. In one line

Addenda 43-45 framed the open question as three-way: at the occupancies
where `vf2postlayout_ms` reads exactly `0.0`, was the pass **skipped**,
did it **find a solution**, or did it **find no better solution**? The
answer is none of the three as posed: **`VF2PostLayout` ran in 75 of 75
runs (`vf2postlayout_ran=True` everywhere, no skip at any occupancy), and
returned `"no better solution found"` in 75 of 75 runs -- at every spare
value, on both sides of the cliff, including the ones where it spent 8.8
to 21.1 seconds getting there.** The pass is not being skipped, and it is
also never once improving the layout it was handed.

## 1. Results

| spare | occupancy | total (ms) | `VF2PostLayout` (ms) | ran? | `VF2Layout_stop_reason` | `VF2PostLayout_stop_reason` |
|---:|---:|---:|---:|:---|:---|:---|
| 0 | 100.0% | 21,010.7 | 8,806.1 | **True** | nonexistent solution | **no better solution found** |
| 2 | 98.3% | 37,048.8 | 15,986.3 | **True** | nonexistent solution | **no better solution found** |
| 4 | 96.6% | 42,750.3 | 21,125.3 | **True** | nonexistent solution | **no better solution found** |
| 6 | 94.8% | 2,530.6 | **0.0** | **True** | solution found | **no better solution found** |
| 8 | 93.1% | 154.4 | **0.0** | **True** | solution found | **no better solution found** |

Medians over 15 runs per cell; the stop reasons were unanimous (15/15) in
every cell, at both keys.

## 2. What this settles

**The skip hypothesis is dead.** Qiskit 2.2's release note ("`VF2PostLayout`
will now skip the search since the layout problem isn't viable for the
pass") describes real behaviour, but **it is not what produces the exact
`0.0` readings in this project's data**. The pass appears in the
transpiler callback at every single spare value, including the ones
reading `0.0` ms. Addenda 43-45 each listed "the pass did not run at all"
as a live candidate for those zeros; it is now ruled out.

**The `0.0` readings mean the pass ran and returned faster than the
timer's resolution**, while still setting its stop reason -- the
mechanism is that at `spare>=6`, `VF2Layout` has already found a perfect
layout (`"solution found"`), so `VF2PostLayout` has a trivially good
starting point and, per Addendum 45, starts its exact-mode search from
the identity mapping, i.e. from that already-perfect layout. It confirms
immediately that nothing better exists and exits.

**And the expensive cases are equally unproductive.** At `spare=0` to
`4`, where `VF2Layout` returns `"nonexistent solution"`,
`VF2PostLayout` spends 8.8-21.1 seconds -- **roughly half the entire
compile time** -- and reaches exactly the same verdict it reaches
instantly on the other side: no better solution. **Every second of that
is spent confirming a negative.** This is a sharper statement of the
cliff's cost than Addendum 34's "two expensive VF2 searches": one of the
two searches is, in this regime, guaranteed to produce nothing usable,
and the data cannot distinguish that from it being *expected* to produce
nothing usable.

## 3. A serious reproducibility problem, reported in full

The same script, same machine (Intel `Family 6 Model 181`), same graph
(`degree_seed=42`), same seeds, same repeats, run twice:

| spare | Addendum 42 total (ms) | this run (ms) | ratio | Add. 42 `VF2PostLayout` | this run |
|---:|---:|---:|---:|---:|---:|
| 0 | 33,587.2 | 21,010.7 | **0.63x** | 17,275.4 | 8,806.1 |
| 2 | 32,534.5 | 37,048.8 | 1.14x | 15,749.7 | 15,986.3 |
| 4 | 12,049.2 | **42,750.3** | **3.55x** | 6,000.0 | 21,125.3 |
| 6 | 669.2 | 2,530.6 | **3.78x** | 0.0 | 0.0 |
| 8 | 39.7 | 154.4 | **3.89x** | 0.0 | 0.0 |

**Spare=4, 6 and 8 all landed within 3.55-3.89x of each other's
displacement, in the same direction** -- a near-uniform slowdown of the
later part of the run -- while `spare=0` went the *other* way (0.63x).
This is the same shape as the unexplained whole-round drift recorded in
Addendum 29 (~40-second stretches running uniformly hot on both engines),
and it is now reproduced in a completely different script. **It is not a
per-measurement noise term; something makes contiguous stretches of a run
systematically slower or faster.** No mechanism is proposed here, and
Addendum 29's remains open.

**What this does not damage**: the qualitative findings are untouched.
The `VF2Layout` stop-reason flip is at exactly the same place in both
runs (between spare=4 and spare=6), the `VF2PostLayout` timing collapse
to `0.0` is at exactly the same place, and the staircase shape survives.
**What it does damage**: every absolute ratio this project has quoted
from single runs on this machine. Addendum 42's "~303x cliff" and this
run's equivalent are not the same number, and the difference is larger
than the effect some of this project's smaller comparisons rest on.

## 4. What is still open

- **Why `VF2PostLayout` is allowed to run for 21 seconds to reach a
  verdict it can reach instantly elsewhere.** It is given a
  `call_limit`/`max_trials` budget by the preset pipeline; Qiskit 2.2
  reduced those specifically for this case (Addendum 43 Section 2a), and
  this run is on 2.5.2, i.e. *after* that reduction. Whatever the
  post-2.2 limits are, they still permit 21 seconds here.
- **Whether `"no better solution found"` is also universal on the square
  grid and on heavy-hex.** Only the synthetic graph has been re-measured
  with the new columns. If heavy-hex (where `vf2postlayout_ms` is a
  constant 0.2-0.4 ms, never exactly zero -- Addendum 43) shows a
  *different* stop reason, that would finally explain its third
  behaviour. **This is now a cheap, high-value check**: the
  instrumentation is already in `occupancy_sweep_heavy_hex.py`.
- The drift in Section 3.

## 5. Files

| File | What it is |
|---|---|
| [`synthetic_sparse_balanced_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/synthetic_sparse_balanced_cliff.py) | the script, with the three new columns added this session |
| [`synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | this run, 75 rows |

## 6. Verification

- `vf2postlayout_ran` was confirmed `True` in all 75 rows, and
  `vf2post_stop_reason` confirmed unanimous within every cell, by direct
  count rather than by reading the script's printed summary.
- The Section 3 comparison uses Addendum 42's recorded medians against
  this run's, both recomputed from their own CSVs; neither was carried
  over from prose.
- The stop-reason flip location (between spare=4 and spare=6) was checked
  to be identical in both runs before claiming the qualitative findings
  survive the drift.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 47 (source: spare-qubit-cliff-addendum-47-2026-09-18.md) ===== -->

> **Note added when merging:** A third run revises Addendum 46's drift claim: two of three runs agree to within 1%, and the variance is confined to the *failing* search, not the succeeding ones.

## Addendum 47 -- a third run corrects Addendum 46's drift claim: two of three runs agree to within 1%, and the variance is confined to the slow region (2026-09-18)

**Status**: a third run of the same configuration, restricted to the
boundary region (`spare` in {4, 6, 8}) that Addendum 46 flagged as
inconsistent. 2 seeds x 2 repeats per cell (12 rows) rather than the full
5 x 3, deliberately, since the question is reproducibility of the medians
rather than new coverage. **This addendum revises a claim made in
Addendum 46 Section 3 and should be read with it.**

## 0. In one line

Addendum 46 reported a "serious reproducibility problem" from two runs
disagreeing by up to 3.89x, and read the pattern as a near-uniform
slowdown of contiguous stretches -- the same shape as Addendum 29's
unexplained drift. **A third run undercuts that reading: at `spare=6` and
`spare=8`, run 3 agrees with Addendum 42 to within 1% (673.2 vs. 669.2
ms; 39.2 vs. 39.7 ms), while run 2 sits ~3.8x above both.** Run 2 looks
like the outlier, not the norm, and the "two runs disagree so the
measurement is unstable" framing was built on a sample of two.
**`spare=4` is a different matter and remains genuinely variable** (12.0
/ 42.8 / 35.1 s across the three runs).

## 1. The three runs side by side

| spare | Addendum 42 | run 2 (Addendum 46) | run 3 (this) | max/min |
|---:|---:|---:|---:|---:|
| 4 | 12,049.2 ms | 42,750.3 ms | 35,133.5 ms | **3.55x** |
| 6 | 669.2 ms | 2,530.6 ms | **673.2 ms** | 3.78x |
| 8 | 39.7 ms | 154.4 ms | **39.2 ms** | 3.94x |

Reading the rows rather than the max/min column:

- **`spare=6`: 669.2, 2530.6, 673.2.** Two values within 0.6% of each
  other, one 3.8x higher.
- **`spare=8`: 39.7, 154.4, 39.2.** Two values within 1.3% of each
  other, one 3.9x higher.
- **`spare=4`: 12049, 42750, 35134.** No two values close; the spread is
  real and not attributable to one outlier run.

## 2. What this revises

**Addendum 46 Section 3 said**: *"Spare=4, 6 and 8 all landed within
3.55-3.89x of each other's displacement, in the same direction -- a
near-uniform slowdown of the later part of the run... It is not a
per-measurement noise term; something makes contiguous stretches of a run
systematically slower or faster."*

**That inference was over-drawn from two runs.** With three, the more
economical reading is that run 2 as a whole was anomalously slow (some
transient condition on the machine during that invocation), and that the
fast-region values are in fact highly reproducible. The claimed
resemblance to Addendum 29's drift is weakened correspondingly -- not
disproven, since run 2 *is* still an unexplained ~3.8x uniform
displacement, but no longer supported as a general property of this
measurement.

**What Addendum 46 said about absolute ratios still stands, for a
narrower reason.** Its warning that "every absolute ratio this project
has quoted from single runs on this machine" is suspect remains
appropriate -- but the evidence for it is now `spare=4`'s genuine 3.55x
spread across three runs, not a claimed run-wide drift.

## 3. The variance is where the search fails, and nowhere else

The sharpest pattern in the three-run table is not the magnitude of the
spread but **where it lives**:

- `spare=6, 8` (`VF2Layout_stop_reason = "solution found"`): reproducible
  to within ~1% in two of three runs.
- `spare=4` (`VF2Layout_stop_reason = "nonexistent solution"`): 3.55x
  spread with no two runs agreeing.

**The reproducible cases are the ones where the search succeeds; the
variable case is the one where it exhausts its budget without finding
anything.** That is not surprising in hindsight -- a successful search
terminates when it finds a solution, which for a given graph and seed is
a deterministic amount of work, while a failed search terminates on a
budget (call limit, trial limit, or time limit), and how much work fits
in that budget depends on machine conditions. **But it was not predicted,
and it reframes the variance from "this measurement is noisy" to "the
failure mode is the noisy part."** Not tested: whether this holds at
`spare=0` and `2` (not re-run here) or on other topologies.

## 4. What did not change

Every qualitative finding is identical across all three runs:

- `VF2Layout_stop_reason` flips between `spare=4` and `spare=6`, in all
  three.
- `VF2PostLayout` **ran in every row of every run** (`ran=True`) -- never
  skipped, confirming Addendum 46's central result on a second dataset.
- `VF2PostLayout_stop_reason` was **`"no better solution found"` in
  every row of every run**, at every spare value, on both sides of the
  cliff.
- `vf2postlayout_ms` collapses to exactly `0.0` at `spare=6` in all
  three.

Addendum 46's main conclusion -- that `VF2PostLayout` is never skipped
and never improves anything, spending up to 21 seconds in the slow region
to reach the same verdict it reaches instantly in the fast region --
is unaffected and now rests on two independent runs.

## 5. Still open

- **Why `spare=4` specifically is variable** while its neighbours are
  not, beyond the budget-vs-solution argument in Section 3.
- Whether `spare=0` and `spare=2` share `spare=4`'s variability (not
  re-run in this addendum; they cost ~7 minutes per cell).
- **Whether run 2 had an identifiable cause.** No system monitoring was
  running during any of the three invocations, so "some transient
  condition" is a placeholder, not a finding.
- Addendum 29's original drift observation, which this addendum weakens
  as a corroborating data point but does not address on its own terms.
- The heavy-hex re-measurement with the new stop-reason columns, still
  the cheapest remaining check (Addendum 46 Section 4).

## 6. Files

| File | What it is |
|---|---|
| [`synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run3.csv) | this run, 12 rows (spare 4/6/8, 2 seeds x 2 repeats) |
| [`synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/synthetic_sparse_balanced_cliff_n116_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | Addendum 46's run, 75 rows |
| (Addendum 42's CSV) | the first run, 75 rows |

## 7. Verification

- All three runs' medians were recomputed from their own CSVs for this
  comparison; none was taken from a previous addendum's prose.
- Stop reasons in run 3 were checked to be unanimous within each cell
  (all 4 rows per spare value agreeing) before being reported as
  matching runs 1-2.
- The claim that run 3 matches Addendum 42 "to within 1%" was computed
  (`673.2/669.2 = 1.006`, `39.2/39.7 = 0.987`), not estimated by eye.
- Section 2 quotes Addendum 46's own wording verbatim before revising it,
  so the revision is checkable against what was actually claimed.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 48 (source: spare-qubit-cliff-addendum-48-2026-09-18.md) ===== -->

> **Note added when merging:** **Retracts Addendum 43's 'third behaviour'** -- it was a cross-machine timer-resolution artefact, caught by checking a `cpu` column that had been present in every CSV all along. There were never three behaviours, only two.

## Addendum 48 -- the "third behaviour" on heavy-hex does not exist: `VF2PostLayout` returns "no better solution found" on every topology tested, and the 0.2-0.4 ms readings were timer resolution, not a distinct mechanism (2026-09-18)

**Status**: results from the `vf2post_stop_reason` / `vf2postlayout_ran`
instrumentation, run on heavy-hex d=7 (115 physical qubits), two
independent runs, 3 seeds x 2 repeats each, `optimization_level=3`.
**This closes the open question raised in Addendum 43 and carried forward
in Addenda 44-47.**

## 0. In one line

Addendum 43 catalogued three apparently distinct behaviours of
`VF2PostLayout` across topology families and singled out heavy-hex as a
third case because its `vf2postlayout_ms` was *"a small, constant,
**non-zero** 0.4 ms at every occupancy tested"* while grid and synthetic
produced exact zeros in bulk -- a distinction Addendum 43 explicitly
argued was "not a rounding artefact." **It was a rounding artefact, or
more precisely a timer-resolution artefact.** Re-measured on this
project's Intel machine, heavy-hex's `vf2postlayout_ms` is **exactly
0.000 in all 72 rows across two runs**, and its `vf2post_stop_reason` is
**`"no better solution found"` in all 72 rows** -- identical to the
synthetic sparse graph (Addenda 46-47). **There is no third behaviour.**

## 1. Results

### heavy-hex d=7, run 1 (36 rows) and run 2 (36 rows)

| spare_pairs | occupancy of capacity | total (ms) run1 / run2 | `VF2PostLayout` (ms) | ran? | `VF2Layout_stop_reason` | `VF2PostLayout_stop_reason` |
|---:|---:|---:|---:|:---|:---|:---|
| 0 | 100.0% | 43.5 / 43.1 | **0.000** | True | solution found | **no better solution found** |
| 1 | 97.9% | 45.0 / 46.7 | **0.000** | True | solution found | **no better solution found** |
| 2 | 95.8% | 44.2 / 44.4 | **0.000** | True | solution found | **no better solution found** |
| 4 | 91.7% | 48.1 / 47.0 | **0.000** | True | solution found | **no better solution found** |
| 8 | 83.3% | 46.2 / 47.9 | **0.000** | True | solution found | **no better solution found** |
| 16 | 66.7% | 45.3 / 45.4 | **0.000** | True | solution found | **no better solution found** |

Unanimous within every cell, in both runs, at both stop-reason keys.
`vf2postlayout_ran = True` everywhere -- as on the synthetic graph, the
pass is never skipped.

## 2. Why Addendum 43's "third behaviour" was an artefact

Addendum 43 compared heavy-hex data collected on **this session's cloud
sandbox** (`cpu = x86_64`, Intel Xeon, 2 cores) against grid and
synthetic data collected on **this project's Intel machine**
(`Intel64 Family 6 Model 181 Stepping 0`). Both on qiskit 2.5.2, both
heavy-hex d=7, both `optimization_level=3` -- **the only difference that
mattered was the host.**

| measurement | machine | `vf2postlayout_ms` |
|---|---|---|
| Addendum 40/43 heavy-hex | cloud sandbox (`x86_64`) | 0.2-0.4 ms, never exactly 0 |
| this addendum, same topology | Intel machine | **exactly 0.000, always** |

The pass does the same negligible amount of work in both cases; one
machine's `time.perf_counter()` resolves it as a few hundred
microseconds, the other's floors it to zero. **Addendum 43's Section 1
explicitly counted exact-zero vs. sub-millisecond readings as evidence
of a mechanistic difference** ("*An exact 0.0 and a consistent 0.2-0.4 ms
are different events, and the most natural reading is that they
distinguish 'the pass did not run at all' from 'the pass ran and returned
almost immediately'"*). That reading was wrong on both halves: the pass
ran in every case, and the numeric difference was the host, not the
behaviour.

**The error was avoidable and its cause is worth recording.** Addendum
43 compared across datasets without checking that they came from the same
machine, despite this project having repeatedly flagged cross-machine
comparison as invalid elsewhere (Addendum 42's own verification section
says exactly this about its own data). The `cpu` column needed for the
check was present in every CSV involved.

## 3. What is now settled across all three topology families

With this measurement, `VF2PostLayout`'s behaviour has been directly
instrumented on square grid (pending -- see Section 4), synthetic sparse
balanced (Addenda 46-47), and heavy-hex (here):

- **It is never skipped.** `vf2postlayout_ran = True` in every row of
  every run on every topology measured so far, at every occupancy.
  Qiskit 2.2's documented skip behaviour, whatever triggers it, is not
  triggered by anything this project has tested.
- **It never improves a layout.** `"no better solution found"` in 147 of
  147 instrumented rows to date (75 synthetic + 72 heavy-hex). Not once
  has it returned `"solution found"`.
- **Its cost tracks `VF2Layout`'s outcome, not occupancy directly.**
  Where `VF2Layout` returns `"solution found"`, `VF2PostLayout` costs
  ~0 ms (heavy-hex everywhere; synthetic at spare>=6). Where `VF2Layout`
  returns `"nonexistent solution"`, `VF2PostLayout` costs seconds
  (synthetic at spare<=4). **Heavy-hex never enters the expensive
  regime because `VF2Layout` always succeeds there** -- which is the same
  reason it has no cliff (Addenda 40-41), not a separate phenomenon.

**This simplifies the picture considerably.** There were never three
behaviours; there are two, and they are the two sides of `VF2Layout`'s
own success/failure boundary.

## 4. Still open

- **The square grid has not been re-measured with the new columns.** Its
  instrumented data (Addendum 34's CSV) predates them. The prediction,
  now strongly constrained by Section 3, is that it will show
  `"no better solution found"` throughout and `ran=True` throughout,
  with the cost collapsing at the same step `VF2Layout`'s stop reason
  flips (spare 0 -> 1). **Untested**, and cheap to test.
- Why `VF2PostLayout` is permitted seconds of budget to confirm a
  negative it can confirm instantly when handed a good layout
  (Addendum 46 Section 4) -- unchanged.
- The `spare=4` variability on the synthetic graph (Addendum 47 Section
  3) -- unchanged.

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep_heavy_hex_d7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_heavy_hex_d7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | run 1, 36 rows |
| [`occupancy_sweep_heavy_hex_d7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_heavy_hex_d7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | run 2, 36 rows |
| [`occupancy_sweep_heavy_hex.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep_heavy_hex.py) | the script; the summary-print block for the two new columns was missing in the version used for these runs (data was written to CSV correctly regardless) and has since been added |

## 6. Verification

- `vf2postlayout_ran` confirmed `True` and `vf2post_stop_reason`
  confirmed unanimous in all 72 rows across both runs, by direct count.
- `vf2postlayout_ms` confirmed to be exactly `0.000` (not a rounded
  display) in both runs before claiming the contrast with Addendum 43.
- The machine difference in Section 2 was read from the `cpu` column of
  both datasets' own CSVs (`x86_64` vs.
  `Intel64 Family 6 Model 181 Stepping 0`), and the qiskit version
  confirmed identical (2.5.2) in both, to isolate the host as the only
  relevant difference.
- Addendum 43's claim is quoted verbatim in Section 2 before being
  corrected, so the correction is checkable against what was written.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both CSVs -> 0
  hits.

---


<!-- ===== Addendum 49 (source: spare-qubit-cliff-addendum-49-2026-09-18.md) ===== -->

> **Note added when merging:** The square grid completes the unified account across all three topologies. Its Section 4 names the finding's own weakest point -- which **Addendum 50 then confirms and overturns.**

## Addendum 49 -- the square grid completes the picture: one mechanism across all three topologies, and `VF2PostLayout` has now never improved a layout in 171 instrumented runs (2026-09-18)

**Status**: results from the `vf2post_stop_reason` / `vf2postlayout_ran`
instrumentation on the 6x7 square grid, `optimization_level=3`, 24 rows.
This was the last of the three topology families to be re-measured with
the new columns, and it was pre-committed in Addendum 48 Section 4 with a
stated prediction. **The prediction held exactly.**

## 0. In one line

Addendum 48 predicted the grid would show `"no better solution found"`
throughout, `ran=True` throughout, and its `VF2PostLayout` cost
collapsing at the same step `VF2Layout`'s stop reason flips. **All three
hold.** With this measurement, the same mechanism now accounts for
`VF2PostLayout`'s behaviour on square grid, heavy-hex, and synthetic
sparse balanced graphs, with **no topology-specific exceptions
remaining**: its cost is governed entirely by whether `VF2Layout`
succeeded, and it has returned `"no better solution found"` in **171 of
171** instrumented rows to date -- never once `"solution found"`.

## 1. Results

### 6x7 square grid (42 physical qubits), optimization_level=3

| spare | n | occupancy | total (ms) | `VF2PostLayout` (ms) | ran? | `VF2Layout_stop_reason` | `VF2PostLayout_stop_reason` |
|---:|---:|---:|---:|---:|:---|:---|:---|
| 0 | 42 | 100.0% | 6,434.2 | **3,220.808** | True | nonexistent solution | **no better solution found** |
| 1 | 41 | 97.6% | 21.3 | **0.000** | True | solution found | **no better solution found** |
| 2 | 40 | 95.2% | 21.3 | 0.000 | True | solution found | **no better solution found** |
| 4 | 38 | 90.5% | 23.2 | 0.000 | True | solution found | **no better solution found** |

Unanimous within every cell. `VF2PostLayout` accounts for 3.22 s of the
6.43 s total at `spare=0` -- **50.1% of the entire compile time, spent
confirming a negative.**

## 2. The prediction, and its scoring

Addendum 48 Section 4 stated, before this run:

> The prediction, now strongly constrained by Section 3, is that it will
> show `"no better solution found"` throughout and `ran=True`
> throughout, with the cost collapsing at the same step `VF2Layout`'s
> stop reason flips (spare 0 -> 1).

**All three components confirmed**: `"no better solution found"` in
24/24 rows; `ran=True` in 24/24 rows; cost collapsing from 3,220.8 ms to
0.000 ms across exactly the spare 0 -> 1 step, the same step at which
`VF2Layout_stop_reason` flips from `"nonexistent solution"` to
`"solution found"`.

## 3. The unified account, now complete across three topologies

| topology | `VF2Layout` outcome | `VF2PostLayout` cost | `VF2PostLayout` verdict |
|---|---|---|---|
| grid, spare=0 | nonexistent solution | 3,220.8 ms | no better solution found |
| grid, spare>=1 | solution found | 0.000 ms | no better solution found |
| synthetic n=116, spare<=4 | nonexistent solution | 8,806-21,125 ms | no better solution found |
| synthetic n=116, spare>=6 | solution found | 0.000 ms | no better solution found |
| heavy-hex d=7, all | solution found | 0.000 ms | no better solution found |

**One rule covers every row**: `VF2PostLayout` costs effectively nothing
when `VF2Layout` has already found a perfect layout, and costs seconds
when it has not -- and in *both* cases returns the same verdict. It is
never skipped, and it has never once improved a layout in any of the 171
instrumented rows (24 grid + 75 synthetic + 72 heavy-hex).

**This retires the "three behaviours" framing from Addendum 43
entirely.** What looked like three topology-specific phenomena was one
mechanism observed at two points of `VF2Layout`'s own success/failure
boundary, plus (per Addendum 48) a cross-machine timer-resolution
artefact that made heavy-hex look like a separate case.

**And it sharpens the cliff's cost accounting.** Addendum 34 established
the cliff is two VF2-family searches rather than a Sabre fallback. It can
now be stated more precisely: **at the cliff, roughly half the compile
time is spent by a pass that, by its own reported stop reason, finds
nothing to improve** -- and which reaches that identical conclusion in
under a millisecond whenever the first search succeeds.

## 4. What this does not establish

- **Why the budget permits it.** `VF2PostLayout` is given
  `call_limit`/`max_trials` by the preset pipeline; Qiskit 2.2 reduced
  those specifically for symmetric circuits on large coupling maps
  (Addendum 43), and these runs are on 2.5.2, i.e. after that reduction.
  Whatever the post-2.2 limits are, they still permit 3.2 s here and
  21 s on the synthetic graph. **The actual numeric limits have not been
  read from source.**
- **Whether `"no better solution found"` would ever flip to `"solution
  found"` on a target that carries real error rates.** Every run in this
  project passes a bare `CouplingMap`, so `VF2PostLayout` -- whose whole
  purpose is finding a *lower-error* layout (Addendum 44) -- has no error
  information to improve against. **171 consecutive "no better solution
  found" results may be a straightforward consequence of never giving it
  anything to optimise, not a property of the pass.** This is the single
  most likely explanation and it is untested: it would require a run
  against a calibrated target (e.g. `FakeTorino`), which this project has
  never done.
- Anything about optimization levels other than 3, or SDKs other than
  Qiskit.

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | this run, 24 rows at opt3 |
| [`occupancy_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep.py) | the script, with the new columns and (since the heavy-hex runs) the matching summary-print block |

## 6. Verification

- All 24 opt3 rows checked: `vf2postlayout_ran=True` and
  `vf2post_stop_reason="no better solution found"` unanimous per cell.
- The 171-row total was computed by summing the instrumented row counts
  actually collected (24 grid + 75 synthetic + 72 heavy-hex), not
  estimated.
- The 50.1% figure is `3220.808 / 6434.2` from this run's own medians.
- Addendum 48's prediction is quoted verbatim in Section 2 before being
  scored, so the scoring is checkable against what was actually
  predicted rather than a restatement of it.
- All three topologies compared in Section 3 were measured on the **same**
  Intel machine (`Intel64 Family 6 Model 181 Stepping 0`), checked via
  each CSV's `cpu` column -- the cross-machine error that produced
  Addendum 43's false "third behaviour" was specifically guarded against
  here.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 50 pre-registration (source: spare-qubit-cliff-addendum-50-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether `VF2PostLayout` ever improves a layout when the target carries real error rates. Sets the confirmation bar at a single counterexample, deliberately.

## Addendum 50 -- Pre-registration: does `VF2PostLayout` ever return "solution found" when the target actually carries error rates? (2026-09-18)

**Status: pre-registration only. No run against a calibrated target has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 49 closed the three-topology picture with a unified account, and
then named its own weakest point in Section 4:

> **171 consecutive "no better solution found" results may be a
> straightforward consequence of never giving it anything to optimise,
> not a property of the pass.** This is the single most likely
> explanation and it is untested.

`VF2PostLayout`'s stated purpose is to find a **lower-error** layout
(Addendum 44: *"By default, this pass will construct a heuristic scoring
map based on the error rates in the provided target"*). Every measurement
in this project -- all 171 instrumented rows across grid, heavy-hex and
synthetic graphs -- passes a bare `CouplingMap` with `basis_gates`, from
which Qiskit builds a `Target` whose `InstructionProperties` carry **no
error rates at all**. A pass whose job is to reduce error, given a device
where every qubit and every link is equally (and unknown-ly) good, has
nothing to prefer. **Reporting "it never improves anything" without
testing it on a device where improvement is even definable would be a
misleading result**, and this addendum exists to avoid publishing one.

## 2. Design

`FakeTorino` (`qiskit_ibm_runtime.fake_provider`), a 133-qubit IBM Heron
snapshot including T1/T2 and per-gate error rates -- **the same target
Benchpress used for its device-transpilation tests** (Addendum 38), so
this also closes that comparison's outstanding "never tested on a real
device topology" gap in one run.

Two arms, identical in every respect except what is passed to
`transpile()`:

- **`with_errors`**: `transpile(qc, target=backend.target, ...)` -- full
  calibration data.
- **`no_errors`** (control): `transpile(qc, coupling_map=<Torino's own
  coupling map>, basis_gates=<Torino's own basis>, ...)` -- the *same
  topology and basis*, with the error rates stripped. This isolates the
  error rates as the only variable; without it, any difference could be
  attributed to the topology being new.

`spare` is measured relative to maximum matching capacity, per Addendum
39's correction (heavy-hex admits no perfect matching). Everything else
-- circuit family, `optimization_level=3`, per-pass callback
instrumentation, `vf2post_stop_reason` / `vf2postlayout_ran` columns --
is unchanged from `occupancy_sweep_heavy_hex.py`.

## 3. Pre-registered predictions

**P1 (primary).** In the `with_errors` arm, across all spare values and
seeds, `VF2PostLayout_stop_reason` will be `"solution found"` in **at
least one row**.
  - **Confirmed**: at least one `"solution found"`. This would mean the
    171-row streak is an artefact of missing error rates, and Addendum
    49's "never improves a layout" finding must be restated as "never
    improves a layout *when given no error information*."
  - **Falsified**: zero `"solution found"` rows in the `with_errors` arm,
    i.e. `"no better solution found"` throughout even with full
    calibration data. This would substantially strengthen Addendum 49 --
    the pass would then be failing to improve layouts on a real device
    snapshot, not merely on information-free synthetic ones.

**P2 (control arm).** The `no_errors` arm will show `"no better solution
found"` in **every** row, reproducing the existing 171-row pattern on
this new topology. A deviation here would mean something about Torino's
topology, rather than its error rates, changes the outcome -- and would
invalidate P1's interpretation either way.

**P3 (cost).** The `with_errors` arm's `VF2PostLayout` time will be
**greater than or equal to** the `no_errors` arm's at matched spare
values. Rationale: with a non-trivial scoring function there is
something to optimise over, so an early exit is less likely. No specific
ratio is predicted.

**P4 (no cliff expected).** `VF2Layout_stop_reason` will be `"solution
found"` at every spare value in both arms, reproducing Addendum 40's
heavy-hex result (no occupancy cliff on this topology family, because the
matching ceiling prevents reaching the dangerous regime). **If a cliff
does appear on the real device snapshot, that is a separate and more
important finding than P1**, and would be reported as such.

## 4. What this cannot establish

- Whether other calibrated devices behave like Torino. One snapshot.
- Whether `"solution found"` -- if it appears -- corresponds to a
  *meaningfully* better layout. The stop reason says a lower-scoring
  layout was found, not by how much. Quantifying the improvement would
  need the layout scores themselves, which this instrumentation does not
  capture.
- Anything about optimization levels other than 3.
- **Whether `qiskit_ibm_runtime` is even installed** on the machine this
  will run on. It is a separate package from `qiskit`; if absent, this
  experiment cannot run as designed and the fallback (a
  `GenericBackendV2` with `noise_info=True`, which generates plausible
  random error rates rather than using a real snapshot) is **not**
  equivalent and would need its own pre-registration.

## 5. Scoring discipline

Score P1-P4 exactly as written. In particular: **P1 is confirmed by a
single `"solution found"` row**, not by a majority -- the claim under
test is "this pass never improves anything," and one counterexample
refutes it. Do not raise the bar after seeing the data.

---


<!-- ===== Addendum 50 (source: spare-qubit-cliff-addendum-50-2026-09-18.md) ===== -->

> **Note added when merging:** **Overturns the 171-row 'never improves anything' finding.** On a calibrated `FakeTorino` snapshot, `VF2PostLayout` returns `"solution found"` -- while an otherwise-identical control with error rates stripped does not. Also finds the pass costs ~130 ms at every occupancy once error rates are present, roughly tripling total compile time.

## Addendum 50 -- with real error rates, `VF2PostLayout` does find better layouts: the 171-row "never improves anything" streak was an artefact of never giving it anything to optimise (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-50-preregistration-2026-09-18.md`, written
and locked before this run. All four predictions are scored below.

## 0. In one line

**P1 confirmed.** On `FakeTorino` (133-qubit IBM Heron snapshot with real
calibration data), `VF2PostLayout_stop_reason` is **`"solution found"`**
at `spare_pairs=4` and `8` in the `with_errors` arm -- while the
`no_errors` control, on the **identical topology and basis gates with only
the error rates stripped**, returns `"no better solution found"` at every
single spare value. **Addendum 49's headline finding must be restated**:
`VF2PostLayout` does not "never improve a layout"; it never improves a
layout *when given no error information to improve against*, which is
what all 171 previously instrumented rows did.

## 1. Results

`FakeTorino`, 133 physical qubits, bipartite parts 56/77 (imbalance 21),
`max_matching_pairs=56`, no perfect matching (as expected for heavy-hex,
per Addendum 39). `optimization_level=3`, 3 seeds x 2 repeats per cell.

| spare_pairs | occupancy of capacity | `VF2Layout` (both arms) | `VF2PostLayout`, `no_errors` | `VF2PostLayout`, **`with_errors`** |
|---:|---:|:---|:---|:---|
| 0 | 100.0% | solution found | no better solution found | no better solution found |
| 1 | 98.2% | solution found | no better solution found | no better solution found |
| 2 | 96.4% | solution found | no better solution found | no better solution found |
| 4 | 92.9% | solution found | no better solution found | **solution found** |
| 8 | 85.7% | solution found | no better solution found | **solution found** |

`vf2postlayout_ran = True` in every cell of both arms -- still never
skipped, consistent with Addenda 46-49.

Median total compile time: 112-121 ms across all spare values, flat.

## 2. Scoring

**P1 (primary) -- CONFIRMED.** Predicted: at least one `"solution found"`
row in the `with_errors` arm. Measured: `"solution found"` unanimously at
two of five spare values. The pre-registration set the bar at a single
row on purpose ("the claim under test is 'this pass never improves
anything,' and one counterexample refutes it"); the result clears it by a
wide margin without the bar having been moved.

**P2 (control arm) -- CONFIRMED.** Predicted: `no_errors` shows `"no
better solution found"` in every row, reproducing the 171-row pattern on
this new topology. Measured exactly that, at all five spare values.
**This is what makes P1 interpretable**: the two arms differ only in
whether error rates are present, so the flip cannot be attributed to
Torino's topology being new.

**P3 (cost) -- CONFIRMED, scored from the CSV after the fact.** Predicted
the `with_errors` arm's `VF2PostLayout` time would be >= the `no_errors`
arm's at matched spare values, on the reasoning that a non-trivial
scoring function makes an early exit less likely. No specific ratio was
predicted. Measured (medians, ms):

| spare_pairs | `no_errors` | `with_errors` |
|---:|---:|---:|
| 0 | 0.000 | 130.139 |
| 1 | 0.000 | 127.027 |
| 2 | 0.000 | 125.924 |
| 4 | 0.000 | 128.558 |
| 8 | 0.000 | 137.364 |

**The gap is not marginal: 0 vs. ~130 ms at every spare value.** Without
error rates the pass exits below timer resolution; with them it spends
~126-137 ms regardless of occupancy. The knock-on effect on total compile
time is large -- `no_errors` totals 49-54 ms, `with_errors` 172-186 ms,
**roughly 3.4x**, and essentially the entire difference is
`VF2PostLayout` (`VF2Layout` itself is 23-32 ms in both arms).

**This reframes Section 3's "newly interesting" point.** `VF2PostLayout`
does the same ~130 ms of work at every occupancy tested, including the
three where it concludes "no better solution found." It is not cheap when
it fails and expensive when it succeeds; **with error rates present it is
uniformly expensive, and only sometimes productive.**

**P4 (no cliff) -- CONFIRMED.** Predicted `VF2Layout_stop_reason =
"solution found"` at every spare value in both arms. Measured exactly
that. **A real, calibrated IBM device snapshot shows no occupancy cliff**,
consistent with Addendum 40's synthetic heavy-hex result and with
Addendum 41's explanation (the matching ceiling prevents reaching the
dangerous regime). Compile time is flat at 112-121 ms throughout.

## 3. What this changes, and what it does not

**Changes**: Addendum 49 Section 3's claim that `VF2PostLayout` "has
never once improved a layout in any of the 171 instrumented rows" was
literally true but misleading as stated, and Addendum 49's own Section 4
flagged exactly this risk. The pass works as documented; this project had
simply never run it on a device where "better" was defined.

**Does not change**: everything Addenda 46-49 established about the
*cliff* itself. `VF2PostLayout` still burns seconds confirming a negative
when `VF2Layout` fails (grid, synthetic) -- and on those targets there
genuinely was nothing to find, which now looks less like a flaw in the
pass and more like a consequence of the experimental setup. **The
question "why is it allowed seconds of budget to confirm a negative"
remains open and is arguably sharper now**: on an uncalibrated target it
cannot possibly succeed, yet nothing in the pipeline shortcuts it.

**Newly interesting**: `"solution found"` appears only at `spare_pairs=4`
and `8`, not at `0`, `1` or `2`. No prediction was made about which spare
values would flip, and none is claimed retroactively. A plausible reading
is that at high occupancy there is no spare capacity to relocate onto, so
even with error rates there is no better placement available -- but this
is untested and the occupancy resolution here is coarse.

## 4. A bug in the script, and what it did not affect

The run ended in `TypeError: unsupported format string passed to
tuple.__format__` in the step-to-step-ratio print block. Cause: adding
`arm` to the `groupby` made the index a `(arm, spare_pairs)` tuple, while
the print statement still formatted it as a bare integer. **This is a
display-only defect in code added this session; every CSV row was written
before it triggered, and no measurement is affected** -- confirmed by P3
being scorable from that same CSV afterwards (Section 2). Fixed by iterating
within each arm separately (a step from one arm's last spare value to the
next arm's first was never a meaningful step anyway).

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep_calibrated.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep_calibrated.py) | the script (display bug fixed after this run) |
| [`occupancy_sweep_calibrated_torino_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_calibrated_torino_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | this run's data |
| [`spare-qubit-cliff-addendum-50-preregistration-2026-09-18.md`](#addendum-50----pre-registration-does-vf2postlayout-ever-return-solution-found-when-the-target-actually-carries-error-rates-2026-09-18) | the predictions scored above |

## 6. Verification

- Stop reasons were read from the script's own per-arm groupby summary,
  which reports the set of distinct values per cell -- every cell printed
  a single value, i.e. all 6 runs in each cell agreed.
- The two arms are confirmed to differ only in the `transpile()` call
  (`target=` vs. `coupling_map=`/`basis_gates=`): both use the coupling
  map and basis gate list extracted from the same `FakeTorino` target, in
  the same process, on the same circuits.
- P3 is recorded as unscored rather than inferred from the totals,
  because the totals include all passes and cannot isolate
  `VF2PostLayout`.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits. **The
  terminal output supplied for this run contained a local file path;
  it has not been reproduced here or in any saved file.**

---

---

**End of Part 4 of 5.** Continue to [Part 5](spare-qubit-cliff-combined-51.md) or [Part 6](spare-qubit-cliff-combined-88.md) (Addendum 51-87), or back to [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
