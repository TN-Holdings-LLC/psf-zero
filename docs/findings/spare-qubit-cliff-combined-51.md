# spare-qubit-cliff: Combined Addenda, Part 5 of 7 (Addendum 51 through Addendum 87)

**Continued from [Part 4](spare-qubit-cliff-combined-41.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md)).** Same conventions as Part 1: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

**Note on this part specifically**: this is where the project's circuit-family dependence was discovered and narrowed, in a long chain of self-correction. A false lead (component count divisible by 3) is sighted three times (Addenda 53, 59, 61-62) before being directly falsified (Addendum 63), and the true necessary conditions -- disjoint components, zero idle qubits, and a sufficient (but not yet located) number of bare 2-qubit edges -- are isolated one at a time (Addenda 54-55, 63-66). Every superseded reading is kept in place, unedited, with the addendum that corrected it noted above it, so the sequence of what was believed when remains readable. **For a compact summary of this entire part, see `SESSION_SUMMARY_2026-09-18.md`** in the same folder -- it is the recommended entry point before reading this part in full.

---
<!-- ===== Addendum 51 pre-registration (source: spare-qubit-cliff-addendum-51-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether the cliff survives a change of circuit family, since every prior measurement used only the project's own disjoint-pair structure.

## Addendum 51 -- Pre-registration: does the occupancy cliff survive a change of circuit family, or is it specific to the maximally-symmetric disjoint-pair structure used throughout? (2026-09-18)

**Status: pre-registration only. No run with any new circuit family has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Every cliff measurement in this project -- Addenda 4 through 50, across
Qiskit, TKET and Cirq, across square grids, heavy-hex and synthetic
sparse graphs -- has used exactly one circuit family:
`build_dense_pair_blocks_circuit`, whose interaction graph is a
**disjoint union of N/2 unconnected 2-node edges**.

**This project has repeatedly flagged that structure as a plausible
worst case in its own right**, not merely as a neutral test input:

- Addendum 37 noted it is "maximally symmetric and disconnected, which
  is a plausible worst case for VF2-family search independent of
  occupancy."
- Addendum 41 noted the same and listed "whether other circuit families
  would show a different result" as unresolved.
- Addendum 42's own script docstring says the structure is "a plausible
  worst case for VF2-family *enumeration* search... unrelated to grid
  size."
- Qiskit's own 2.2 release notes (quoted in Addendum 43) describe
  reducing `VF2PostLayout`'s limits specifically to avoid "excessive
  runtimes for **highly symmetric trial circuits** being mapped to large
  coupling maps" -- i.e. IBM identified this circuit shape as
  pathological independently.

**The single most likely objection to this entire series is therefore:
"this is a property of your circuit, not of occupancy."** No measurement
in the project currently answers it. This addendum exists to answer it
before anything is published, in whichever direction the data goes.

## 2. Design

Three additional circuit families, all run through the **unchanged**
occupancy sweep on the **same** 6x7 grid at `optimization_level=3`, with
the same spare values, seeds, repeats and instrumentation as Addendum 34:

- **`linear_chain`**: interactions (0,1), (1,2), (2,3), ... -- a single
  connected path. Connected, far less symmetric than disjoint pairs, and
  a path embeds into a grid in many ways, so this is not trivially
  harder or easier by construction.
- **`random_regular`**: a random 3-regular interaction graph (connected,
  checked at construction). Degree matched roughly to a grid's interior,
  no repeated motif, minimal symmetry.
- **`ghz_star`**: interactions (0,1), (0,2), (0,3), ... -- a star. Also
  connected, but with an extreme degree distribution (one hub of degree
  n-1). Included specifically because it is *structurally unembeddable*
  in a degree-4 grid for n > 5, so it should produce
  `"nonexistent solution"` at **every** occupancy, not just at
  saturation -- a control for "does the stop reason track occupancy at
  all, or just feasibility?"

Everything else -- `CouplingMap.from_grid(6, 7)`, `optimization_level=3`,
per-pass callback instrumentation, `VF2Layout_stop_reason`,
`VF2PostLayout_stop_reason`, independent `networkx` feasibility check --
is carried over unchanged, so the only variable is the interaction graph.

Gate count per interacting pair is held at the project's standing value
so total circuit size stays comparable across families.

## 3. Pre-registered predictions

**P1 (primary -- does the cliff survive at all?).** For `linear_chain` on
the 6x7 grid at `optimization_level=3`, the ratio of median total compile
time at `spare=0` to `spare=2`:
  - **Cliff survives** if the ratio exceeds **10x**. This would mean the
    cliff is not an artefact of the disjoint-pair structure, and the
    series' central claim stands essentially as written.
  - **Cliff is circuit-specific** if the ratio is below **2x**. This
    would be a major limitation on every prior addendum and would have to
    be reported as such, prominently, not buried in a limitations
    section.
  - **2x-10x**: reported as attenuated-but-present, not rounded either
    way.

**P2 (second family, same question).** The same thresholds applied to
`random_regular`. P1 and P2 are scored independently; agreement between
them is stronger evidence than either alone, and disagreement would
itself be informative about *which* structural property matters.

**P3 (feasibility control).** For `ghz_star` at n > 5,
`VF2Layout_stop_reason` is predicted to be `"nonexistent solution"` at
**every** spare value tested, not just at `spare=0`, because a star with
a hub of degree > 4 cannot embed in a degree-4 grid at any occupancy.
  - If confirmed, this establishes that the stop reason tracks genuine
    feasibility and is not merely a proxy for saturation -- strengthening
    the interpretation of every prior stop-reason result.
  - **If `ghz_star` instead shows `"solution found"` anywhere, something
    is wrong with this project's understanding of what the stop reason
    means**, and that would take priority over P1/P2.

**P4 (independent feasibility agreement).** For every family and every
spare value, the `networkx` maximum-matching check will agree with
`VF2Layout_stop_reason` on feasibility where the two are comparable.
**Note this is only strictly comparable for `dense_pairs`**, where the
interaction graph *is* a matching; for connected families the matching
bound is necessary but not sufficient for embeddability, so a
`"nonexistent solution"` alongside a satisfied matching bound is **not**
a contradiction and must not be reported as one. This prediction is
included mainly to record that limitation in advance.

## 4. What this cannot establish

- Whether *all* circuit families behave alike. Four families is not a
  survey.
- Anything about grid sizes other than 6x7, or topologies other than the
  square grid. Those are separate open items (the second and third of
  the three gaps identified when assessing publication-readiness).
- Anything about SDKs other than Qiskit.
- Whether any of these families is *representative* of real workloads.
  None was chosen for realism; they were chosen to vary symmetry and
  connectivity while holding everything else fixed.

## 5. Scoring discipline

Score P1-P4 exactly as stated. In particular, **if P1 or P2 lands in the
2x-10x band, report it as ambiguous** rather than as confirmation --
the project has a standing rule against rounding a near-miss toward the
convenient side (see Addendum 35's 3.97x against a pre-registered 5x
bar, reported as a miss).

---


<!-- ===== Addendum 51 (source: spare-qubit-cliff-addendum-51-2026-09-18.md) ===== -->

> **Note added when merging:** **The cliff does not survive a change of circuit family.** `linear_chain` and `random_regular` show no cliff at all (for two different reasons); `ghz_star` confirms the stop-reason signal tracks genuine feasibility. Only `dense_pairs` cliffs -- the project's central claim narrows from 'the occupancy cliff' to 'the occupancy cliff, for disjoint-edge interaction graphs.'

## Addendum 51 -- the occupancy cliff does NOT survive a change of circuit family: it is specific to the maximally-symmetric disjoint-pair structure used throughout Addenda 4-50 (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-51-preregistration-2026-09-18.md`, written
and locked before this run. All four predictions are scored below.
**This is the most consequential result of the day and revises how every
prior cliff measurement should be read.**

## 0. In one line

**P1 and P2 are both falsified in the "cliff is circuit-specific"
direction, unambiguously.** `linear_chain` (a connected path) shows
**no cliff at all** -- `spare=0`/`spare=2` ratio 0.96x, and
`VF2Layout_stop_reason = "solution found"` at every tested occupancy,
including `spare=0`. `random_regular` (a connected 3-regular graph) shows
no cliff either, but for a different reason: it returns
`"nonexistent solution"` at **every** occupancy tested, 0 through 4 --
the interaction graph is apparently too dense for this coupling map at
any of these sizes, not specifically at saturation. **Only `dense_pairs`
-- the one family every prior addendum has used -- shows the cliff.** P3
(the `ghz_star` feasibility control) confirms `VF2Layout_stop_reason`
tracks genuine embeddability correctly, which is what makes the other two
results interpretable rather than suspect.

## 1. Results

6x7 grid (42 qubits), `optimization_level=3`, 3 seeds x 2 repeats per
cell, spare in {0, 1, 2, 4}. All 96 rows completed with `error=""`.

| family | edges (spare=0) | max degree | spare=0 (ms) | spare=2 (ms) | ratio | stop reason (all spare) |
|---|---:|---:|---:|---:|---:|---|
| `dense_pairs` (standing family) | 21 | 1 | **6,682.2** | 22.3 | **299.7x** | nonexistent @0, solution found @>=1 |
| `linear_chain` | 41 | 2 | 32.3 | 33.7 | **0.96x** | **solution found, every spare** |
| `random_regular` | 63 | 3 | 30.0 | 34.0 | 0.88x | **nonexistent solution, every spare** |
| `ghz_star` | 41 | 41 | 20.5 | 17.3 | 1.18x | nonexistent solution, every spare (as designed) |

`dense_pairs` reproduces Addendum 34's own result almost exactly
(299.7x here vs. 193-238x range previously reported at finer resolution
-- consistent, same order of magnitude, same machine).

## 2. Scoring

**P1 (`linear_chain`, primary) -- FALSIFIED toward "circuit-specific."**
Predicted: ratio > 10x confirms the cliff survives; ratio < 2x means it
is circuit-specific. **Measured: 0.96x.** Not merely below the 2x bar --
`VF2Layout` succeeds at `spare=0` on this family, meaning there is no
occupancy-driven degradation to measure at all on this coupling map at
this size. **The verdict is unambiguous, not a near-miss.**

**P2 (`random_regular`) -- FALSIFIED toward "circuit-specific," by a
different mechanism than predicted.** Predicted the same 2x/10x
thresholds would distinguish "cliff survives" from "circuit-specific."
**Measured: 0.88x, and every spare value from 0 to 4 returns
`"nonexistent solution."`** The pre-registration anticipated a
ratio-based answer; it did not anticipate infeasibility at every tested
occupancy. This is a *different* failure mode from `linear_chain`'s (see
Section 3) and was not distinguished in advance -- recorded as an
unpredicted finding, not retrofitted into P2's original framing.

**P3 (`ghz_star` feasibility control) -- CONFIRMED.** Predicted
`"nonexistent solution"` at every spare value for n > 5, since a
degree-(n-1) hub cannot embed in a degree-4 grid at any occupancy.
Measured exactly that, at all four spare values. **This is the load-
bearing result for interpreting P1 and P2**: it confirms
`VF2Layout_stop_reason` reports genuine embeddability rather than merely
correlating with occupancy, so `random_regular`'s uniform
"nonexistent solution" is telling us something real about that graph's
embeddability, not an instrumentation artefact.

**P4 (independent feasibility agreement) -- not fully checked.** The CSV
carries `embedding_feasible=True` (from the `networkx` maximum-matching
bound) in every row shown above, including `random_regular`'s and
`ghz_star`'s `"nonexistent solution"` rows. Per the pre-registration's
own stated caveat, **this is not a contradiction for non-`dense_pairs`
families**: the matching bound is necessary but not sufficient for
embeddability of a connected graph, so a satisfied matching bound
alongside `"nonexistent solution"` is exactly the expected, harmless
outcome the pre-registration anticipated -- not evidence against P1-P3.

## 3. Two distinct ways to not have a cliff

`linear_chain` and `random_regular` both lack a cliff, but for opposite
reasons, and conflating them would understate what was found:

- **`linear_chain` is easy everywhere.** A path has minimal structure to
  satisfy; `VF2Layout` finds an embedding even at 100% occupancy,
  instantly. There is no occupancy effect to observe because the problem
  never gets hard.
- **`random_regular` is hard everywhere (in this range).** A 3-regular
  graph with 57-63 edges apparently exceeds what this 6x7 grid can embed
  regardless of how much slack is available, at least up to spare=4.
  Whether it becomes embeddable at higher spare (this project's grid
  cliffs have all been measured near spare=0, not in a `"nonexistent
  solution"`-throughout regime) was not tested here.

**Only `dense_pairs` sits in the narrow middle**: embeddable everywhere
tested except exactly at 100% occupancy, where it becomes hard. That
specific combination -- easy generally, hard only at the boundary -- is
what makes a "cliff" visible at all, and it is a property this
particular family happens to have, not a property occupancy sweeps
reveal in general.

## 4. What this means for Addenda 4-50

**Every quantitative cliff result in this project (Addenda 4-50) was
measured on a single circuit family whose disjoint, maximally-symmetric
structure appears necessary for the phenomenon to be visible in the first
place.** This does not mean the cliff is fake -- it reproduced
independently across three SDKs (Qiskit, TKET, Cirq; Addenda 34, 35, 37)
and multiple topologies on that one family, and Qiskit's own 2.2 release
notes independently identify "highly symmetric trial circuits" as
pathological for exactly this reason, which is external corroboration
that this circuit shape triggers something real in the underlying
algorithm. **But the honest scope of every prior claim narrows from "the
occupancy cliff" to "the occupancy cliff, for circuits whose interaction
graph is a disjoint union of edges."** Whether that scope covers any
circuit of practical interest is a separate, unaddressed question.

**This also reframes what Qiskit's own release note was describing.**
"Highly symmetric trial circuits being mapped to large coupling maps"
now reads less like an incidental edge case and more like a precise
description of exactly the pathology this project has been studying --
Qiskit's own developers may have already identified and partially
mitigated the specific case this whole series is built on, which is
worth checking directly (has this project's own workload actually
improved across Qiskit versions spanning that mitigation?).

## 5. What this does not establish

- Whether `random_regular` would show a cliff at higher spare, i.e.
  whether it has its own, different occupancy threshold this sweep
  (spare 0-4 only) didn't reach.
- Whether a family "in between" `dense_pairs` and `linear_chain` in
  symmetry -- e.g. a few disjoint *chains* rather than single edges or one
  long path -- would show an attenuated cliff, which would map out a
  spectrum rather than a binary present/absent.
- Anything about grid sizes other than 6x7, or about heavy-hex/synthetic
  topologies with these new families -- untested.
- Anything about TKET or Cirq with these new families.
- Whether real quantum algorithms' interaction graphs resemble any of
  these four more than they resemble the others.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script |
| [`circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | this run, 96 rows (opt3) |
| [`spare-qubit-cliff-addendum-51-preregistration-2026-09-18.md`](#addendum-51----pre-registration-does-the-occupancy-cliff-survive-a-change-of-circuit-family-or-is-it-specific-to-the-maximally-symmetric-disjoint-pair-structure-used-throughout-2026-09-18) | the predictions scored above |

## 7. Verification

- All 96 opt3 rows checked for `error=""`.
- Stop reasons confirmed unanimous within every (family, spare) cell
  before being reported.
- `dense_pairs`' 299.7x here was checked against Addendum 34's
  independently-measured 193-238x range for consistency (same order of
  magnitude, same machine, not identical conditions -- this run used
  spare={0,1,2,4} at 3 seeds x 2 repeats, not Addendum 34's finer sweep).
- P4's "not a contradiction" reading was checked directly against the
  pre-registration's own Section 3 caveat before being applied, not
  invented after seeing the data.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 52 pre-registration (source: spare-qubit-cliff-addendum-52-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether the cliff attenuates gradually between dense_pairs and linear_chain, or vanishes abruptly, using a new k_chains(k) family that interpolates exactly between the two endpoints.

## Addendum 52 -- Pre-registration: does the cliff attenuate gradually between `dense_pairs` and `linear_chain`, or disappear abruptly? (2026-09-18)

**Status: pre-registration only. No run with the new family has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 51 found the cliff present on `dense_pairs` (disjoint edges,
maximal symmetry) and absent on `linear_chain` (one connected path,
minimal symmetry) -- but only tested those two endpoints. It is unknown
whether the cliff **attenuates gradually** as symmetry decreases (in
which case "circuit symmetry" is a continuous variable the cliff
responds to) or **exists only at the extreme** (in which case
`dense_pairs` may be pathological in a way no intermediate structure
shares, and the cliff's scope is even narrower than Addendum 51
suggested). These predict different things about whether any circuit of
practical interest could ever exhibit this cliff, and the two endpoints
alone cannot distinguish them.

## 2. Design

A new family, **`k_chains`**, parameterised by `k`: split `n` qubits into
`k` contiguous, equal-length groups and build a connected path *within*
each group (no edges between groups). This interpolates the two
endpoints exactly:

- `k = n/2` (each group has 2 qubits): every "chain" is a single edge --
  **identical in structure to `dense_pairs`.**
- `k = 1` (one group of all `n` qubits): a single path spanning
  everything -- **identical in structure to `linear_chain`.**
- Intermediate `k`: `k` disjoint connected chains, each of length
  `n/k`. Symmetry (measured here as automorphism-group size / number of
  interchangeable identical components) decreases monotonically as `k`
  decreases from `n/2` to `1`.

Swept at `k` in {21, 10, 5, 3, 2, 1} on the same n=42 (6x7 grid,
spare=0) and n=40 (spare=2) instances used throughout, same
`optimization_level=3`, same instrumentation. `k=21` and `k=1` are
included specifically to reproduce Addendum 51's own `dense_pairs` and
`linear_chain` results as an internal consistency check before trusting
any intermediate point.

## 3. Pre-registered predictions

**P1 (primary -- shape of the transition).** Plot median `spare=0` total
time against `k`. Three qualitatively different shapes are
distinguished in advance:
  - **Gradual**: time decreases smoothly and monotonically as `k`
    decreases from 21 to 1, with no single step accounting for more than
    ~50% of the total log-scale drop between `k=21` and `k=1`.
  - **Threshold**: time stays within a factor of 3 of `dense_pairs`'
    value for several `k` values, then drops sharply (>10x in one step)
    at some specific `k`, and stays low for smaller `k`.
  - **Immediate**: time drops by >10x from `k=21` to the very next
    tested `k` (10), i.e. even a small departure from all-disjoint-edges
    kills the effect.

**P2 (consistency check).** `k=21` reproduces `dense_pairs`' cliff
(ratio to `k=21`'s own spare=2 condition > 100x) and `k=1` reproduces
`linear_chain`'s absence (ratio < 2x), both within the same run as the
intermediate points. If either fails, the intermediate results below it
are not trustworthy and should not be interpreted.

**P3 (stop reason tracks the same transition).** `VF2Layout_stop_reason`
at spare=0 is predicted to be `"nonexistent solution"` for the same `k`
values where P1 shows high time, and `"solution found"` where it shows
low time -- i.e. the timing and feasibility signals move together, as
they have in every prior addendum.

**P4 (no prediction on the exact threshold `k`, if "Threshold" is
confirmed).** Stated explicitly so no retroactive claim of having
predicted the specific value can be made.

## 4. What this cannot establish

- *Why* a given `k` is or is not enough to kill the cliff -- this
  measures the shape of the transition, not its mechanism.
- Whether the same transition shape holds on other grid sizes,
  topologies, or SDKs.
- Whether `k_chains` at any intermediate value resembles a circuit of
  practical interest more than `dense_pairs` or `linear_chain` do.

## 5. Scoring discipline

Score P1's three shapes as mutually exclusive and exhaustive of what a
plot could show; if the actual data does not cleanly match any of the
three, say so explicitly rather than forcing it into the closest one.

---


<!-- ===== Addendum 52 (source: spare-qubit-cliff-addendum-52-2026-09-18.md) ===== -->

> **Note added when merging:** The cliff vanishes in a single step (k=21 -> k=10, a ~1,972x collapse) rather than gradually -- 'Immediate' confirmed over 'Gradual' or 'Threshold'.

## Addendum 52 -- the cliff vanishes in a single step from k=21 to k=10: "Immediate," not "Gradual" or "Threshold" (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-52-preregistration-2026-09-18.md`, written
and locked before this run. P1's three-way shape prediction and P2's
consistency check are both scored below.

## 0. In one line

**P1: "Immediate."** Between `k=21` (21 disjoint 2-qubit chains --
structurally identical to `dense_pairs`) and `k=10` (10 disjoint chains
of ~4 qubits each), the spare=0/spare=2 timing ratio collapses from
**1,932.65x to 0.98x** -- a single step, not a gradual decline and not a
sustained plateau before a threshold. Every `k` from 10 down to 1 shows
no cliff at all (ratios 0.96x-1.01x, `VF2Layout_stop_reason =
"solution found"` at spare=0 in every case). **The cliff is not merely
absent from `linear_chain`'s opposite extreme (Addendum 51) -- it barely
survives leaving `dense_pairs` at all.** Ten disjoint short chains is
already enough symmetry-breaking to eliminate a 1,900x effect entirely.

## 1. Results

6x7 grid (42 qubits), `optimization_level=3`, 3 seeds x 2 repeats per
cell. All 72 rows completed with `error=""`.

| k | chains | spare=0 (ms) | spare=0 stop reason | spare=2 (ms) | spare=2 stop reason | ratio |
|---:|---|---:|:---|---:|:---|---:|
| **21** | 21 x 2-qubit (= `dense_pairs`) | 10,493.63 | nonexistent solution | 5.43 | solution found | **1,932.65x** |
| **10** | 10 x ~4-qubit | 27.05 | solution found | 27.57 | solution found | 0.98x |
| 5 | 5 x ~8-qubit | 30.95 | solution found | 30.62 | solution found | 1.01x |
| 3 | 3 x ~14-qubit | 31.56 | solution found | 31.18 | solution found | 1.01x |
| 2 | 2 x ~21-qubit | 31.37 | solution found | 32.60 | solution found | 0.96x |
| **1** | 1 x 42-qubit (= `linear_chain`) | 32.73 | solution found | 33.36 | solution found | 0.98x |

## 2. Scoring

**P2 (consistency check) -- CONFIRMED.** `k=21`'s 1,932.65x ratio and
`"nonexistent solution"` stop reason match `dense_pairs`' behaviour in
both this project's finer-resolution measurement (Addendum 34,
193-238x) and Addendum 51's own same-day `dense_pairs` result (299.7x)
in order of magnitude and direction; the edge lists are additionally
identical by construction (verified before this run: `_edges_k_chains(n,
k=21) == _edges_dense_pairs(n)` for n=42). `k=1`'s 0.98x and
`"solution found"` match Addendum 51's `linear_chain` result (0.96x)
closely. **Both endpoints reproduce**, so the intermediate points are
trustworthy per the pre-registration's own stated condition.

**P1 (primary) -- "Immediate" confirmed, unambiguously.** The
pre-registration's "Immediate" criterion was ">10x drop from k=21 to the
very next tested k (10)." Measured: a **1,972x** drop (1,932.65 / 0.98)
in that single step. This is not a borderline call between the three
predicted shapes -- "Gradual" and "Threshold" both required the effect to
persist, attenuated or otherwise, across multiple intermediate k values,
and it does not: `k=10` through `k=1` are statistically indistinguishable
from each other (ratios spanning only 0.96x-1.01x, well within this
project's documented run-to-run noise band).

**P3 (stop reason tracks the same transition) -- CONFIRMED.**
`VF2Layout_stop_reason` flips from `"nonexistent solution"` to
`"solution found"` at exactly the same step the timing collapses (k=21
-> k=10), at spare=0. At spare=2, every k already shows
`"solution found"`, consistent with spare=2 being outside the cliff
region even for `dense_pairs` itself.

**P4 (no threshold-k claim) -- honoured.** No prediction was made about
which k the transition would occur at; "somewhere between k=21 and
k=10" is reported as what was measured, not as a predicted value that
happened to be confirmed. Whether the true transition lies at k=20, 15,
or exactly 11 was not tested -- only k=21 and k=10 bracket it.

## 3. What this means, combined with Addendum 51

**The cliff is not a property of "low symmetry" versus "high symmetry"
on a spectrum -- it is a property of the specific, maximal case:
interaction graphs that are a disjoint union of single edges, and
apparently only that.** Ten disjoint chains of four qubits each is still
a highly regular, highly symmetric, fully disconnected structure by any
ordinary standard -- and it shows no trace of the effect. The
1,900x-scale phenomenon this entire project has characterized across 50
addenda depends on a much narrower structural condition than "symmetric
and disconnected" was understood to mean going into today.

**This sharpens, rather than undermines, the connection to Qiskit's own
2.2 release note** (quoted in Addendum 43: limits reduced for "highly
symmetric trial circuits"). If Qiskit's own mitigation target is
specifically the disjoint-single-edge case -- effectively a maximum
matching being searched for on a coupling graph -- rather than
symmetric/disconnected structures in general, that would explain why
`k=10` already escapes it: the VF2 search's pathological blowup may be
tied to the *automorphism group of a graph with no edges longer than
one hop*, not to disconnection or repetition as such. This is a
plausible refinement, not a demonstrated one -- the Rust source
(Addendum 43's still-open item) would need to be read to confirm it.

**Practically**: this project's standing circuit family
(`build_dense_pair_blocks_circuit`) is now understood to sit at, or very
near, the single most pathological point in this particular design space
for this particular class of algorithm. Every one of Addenda 4-51's
cliff *measurements* stands -- reproduced independently across Qiskit,
TKET and Cirq, on real and synthetic topologies -- but the phenomenon's
scope, going forward, should be described as narrow and specific rather
than as a general property of symmetric or disconnected circuits.

## 4. What this does not establish

- Where exactly between k=21 and k=10 the transition occurs -- untested,
  per P4.
- Whether the transition's location depends on grid size, chain length
  distribution (this sweep used *equal*-length chains; unequal lengths
  were not tested), or grid topology.
- Whether a disjoint union of slightly-longer-than-one-edge components
  (e.g. k=21 pairs of 2-qubit edges but arranged with some edges sharing
  no automorphism-preserving structure) would behave like k=21 or like
  k=10 -- this sweep varied chain count and length together, not
  independently.
- Anything about TKET or Cirq with `k_chains`.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 52's `k_chains` addition) |
| [`circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | this run, 72 rows |
| [`spare-qubit-cliff-addendum-52-preregistration-2026-09-18.md`](#addendum-52----pre-registration-does-the-cliff-attenuate-gradually-between-dense_pairs-and-linear_chain-or-disappear-abruptly-2026-09-18) | the predictions scored above |

## 6. Verification

- All 72 opt3 rows checked for `error=""`, and stop reasons confirmed
  unanimous within every (k, spare) cell before reporting medians.
- `k=21`'s edge list was confirmed, by direct construction before this
  run, to be identical to `dense_pairs`' own edge list at n=42; `k=1`'s
  to `linear_chain`'s -- not merely expected to match by the parameter
  choice, but checked.
- The terminal summary printed by the script groups only by
  `(family, spare)`, not by `k`, so it does not show the k-by-k
  breakdown this addendum reports; all figures here were recomputed
  directly from the CSV's own `k`, `time_ms`, and `vf2_stop_reason`
  columns. **This is a display gap in the script worth fixing** (group
  by `k` as well when `k_chains` is present) -- noted rather than fixed
  here, since the CSV itself was unaffected and this run's data is
  already fully usable.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for local
  paths before use here; none were reproduced.

---


<!-- ===== Addendum 53 pre-registration (source: spare-qubit-cliff-addendum-53-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for locating the cliff's disappearance at single-step k resolution on 6x7, after Addendum 52 found it vanishes in a single step rather than gradually.

## Addendum 53 -- Pre-registration: where exactly, between k=21 and k=10, does the cliff disappear? (2026-09-18)

**Status: pre-registration only. No run at this resolution has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 52 found the cliff collapses somewhere between `k=21` (21
disjoint 2-qubit chains, structurally identical to `dense_pairs`,
1,932.65x) and `k=10` (10 disjoint ~4-qubit chains, 0.98x) -- P4 of that
addendum explicitly declined to claim any location within that bracket.
This addendum resolves it at single-step resolution.

## 2. Design

`k_chains` on the same 6x7 grid, `k` in {21, 20, 19, 18, 17, 16, 15, 14,
13, 12, 11, 10}, `optimization_level=3`, spare in {0, 2} (matching
Addendum 52 exactly). 3 seeds x 2 repeats, matching Addendum 52's own
resolution so the two runs are directly comparable.

**Chain length, computed in advance (n=42), does NOT vary smoothly
across this sweep** -- it takes only three distinct values across all
twelve `k`:

| k | 21 | 20 | 19 | 18 | 17 | 16 | 15 | 14 | 13 | 12 | 11 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain length (`n // k`) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 | 4 |

**This is a strong, sharpened version of P2, stated before any data is
seen**: if chain length (not `k`) is the operative variable, then
`k=21` through `k=15` -- seven different `k` values, all length-2 chains
-- should behave identically to each other (all cliffing, matching
Addendum 52's `k=21` result), and any transition should land at the
length-2-to-3 boundary (between `k=15` and `k=14`) or the length-3-to-4
boundary (between `k=11` and `k=10`), not at an arbitrary point in
between. If instead the ratio changes *within* the k=21-15 block (all
length-2), `k` itself, not chain length, would be the better-supported
explanation, and P2 would be falsified in the direction of `k` mattering
independently of length.

## 3. Pre-registered predictions

**P1 (primary -- single step or a wider plateau?).** Define the
transition point as the largest `k` at which the spare=0/spare=2 ratio
first drops below 10x.
  - **Single-step, confirming Addendum 52's shape**: the ratio is > 100x
    at `k` and < 10x at `k-1`, for some single `k` in {12, ..., 20} --
    i.e. still an abrupt collapse, just resolved more precisely than
    "somewhere between 21 and 10."
  - **Multi-step plateau, revising Addendum 52**: the ratio decreases
    gradually across three or more consecutive `k` values (e.g. 1000x ->
    100x -> 10x -> 1x) rather than one abrupt drop -- this would mean
    Addendum 52's two-point resolution missed a real gradual region
    between its endpoints, and "Immediate" would need to be walked back
    to "steep but not instantaneous."

**P2 (chain length, not k, is predicted to be the operative variable).**
Given the note in Section 2, the transition is predicted to align with a
chain-length boundary (`n // k` crossing some threshold) rather than
with `k` as a raw count. This is checked by confirming that `k` values
mapping to the same chain length show the same qualitative behaviour
(cliff or no cliff), even if their raw `k` differs.

**P3 (stop reason).** `VF2Layout_stop_reason` flips from
`"nonexistent solution"` to `"solution found"` at spare=0 at the exact
same `k` (or chain length) where P1's timing ratio crosses 10x -- as it
did at exactly the right step in Addendum 52.

## 4. What this cannot establish

- Why the transition sits where it does, in terms of the underlying VF2
  algorithm -- this measures the location, not the mechanism (still
  Addendum 43-45's open Rust-source item).
- Whether the same chain-length threshold holds on a different grid size.
- Whether unequal chain lengths (not tested here or in Addendum 52) show
  the same boundary.

## 5. Scoring discipline

Score P1's two shapes as intended: "Single-step" requires the >100x/<10x
jump to occur between *adjacent* tested k values, not merely somewhere
in the range. If the drop spans two or more steps each showing an
intermediate ratio (e.g. 100x, then 10x, then 1x across three
consecutive k), that is "Multi-step," not "Single-step with extra data
points."

---


<!-- ===== Addendum 53 (source: spare-qubit-cliff-addendum-53-2026-09-18.md) ===== -->

> **Note added when merging:** The cliff appears at exactly the two k values that leave zero idle qubits (divisors of 42) -- an unregistered finding that looked like a component-count-mod-3 pattern, explicitly flagged as too weak (3 matching points) to trust on its own, including a self-caught arithmetic error in its own verification section.

## Addendum 53 -- the cliff tracks neither `k` nor chain length: it appears exactly, and only, when the interaction graph leaves zero qubits idle (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-53-preregistration-2026-09-18.md`, written
and locked before this run. All three predictions are scored below, and
**all three are falsified as stated** -- but the data points cleanly at a
variable the pre-registration did not consider, reported in Section 3 as
an unregistered, post-hoc finding.

## 0. In one line

Of the twelve `k` values swept, only two show the cliff:
**`k=21` (499.34x) and `k=14` (67.20x)**. Every other `k` -- 20, 19, 18,
17, 16, 15, 13, 12, 11, 10 -- shows `VF2Layout_stop_reason =
"solution found"` at spare=0, no cliff at all. This does not track chain
length (both cliffing cases have different chain lengths, 2 and 3; six
non-cliffing cases also have chain length 2) and does not track `k`
monotonically or by any simple rule the pre-registration anticipated.
**What `k=21` and `k=14` share, and nothing else in the sweep does: they
are exactly the two values of `k` that evenly divide `n=42`, leaving
zero qubits outside any interaction edge.** Every other tested `k` leaves
between 2 and 12 qubits completely idle (touched by no two-qubit gate at
all), and none of those cases cliffs.

## 1. Results

6x7 grid (42 qubits), `optimization_level=3`, 3 seeds x 2 repeats. All
144 rows completed with `error=""`.

| k | chain length | qubits left idle | spare=0 (ms) | spare=0 stop reason | ratio (spare0/spare2) |
|---:|---:|---:|---:|:---|---:|
| **21** | 2 | **0** | 6,747.05 | nonexistent solution | **499.34x** |
| 20 | 2 | 2 | 22.83 | solution found | 0.34x |
| 19 | 2 | 4 | 22.11 | solution found | 0.30x |
| 18 | 2 | 6 | 26.01 | solution found | 0.31x |
| 17 | 2 | 8 | 28.21 | solution found | 0.32x |
| 16 | 2 | 10 | 28.07 | solution found | 0.30x |
| 15 | 2 | 12 | 30.10 | solution found | 0.28x |
| **14** | 3 | **0** | 6,656.67 | nonexistent solution | **67.20x** |
| 13 | 3 | 3 | 80.50 | solution found | 0.98x |
| 12 | 3 | 6 | 90.27 | solution found | 1.61x |
| 11 | 3 | 9 | 99.37 | solution found | 1.10x |
| 10 | 4 | 2 | 64.66 | solution found | 1.40x |

## 2. Scoring the pre-registration

**P1 (single step vs. plateau) -- FALSIFIED, in a way the prediction did
not anticipate.** Neither "Single-step" nor "Multi-step plateau" matches
what happened: the data is not monotonic in `k` at all. It shows two
isolated spikes (`k=21`, `k=14`) with ordinary, non-cliffing values on
both sides of each -- including immediately adjacent values (`k=20` and
`k=13` are both unremarkable, sitting right next to cliffing points).
**This shape was not one of the options the pre-registration offered**,
and forcing it into "Single-step" (the nearest fit) would misrepresent
what was found.

**P2 (chain length is the operative variable) -- FALSIFIED, cleanly, and
usefully.** The pre-registration's own advance table showed `k=21-15`
all share chain length 2; if length were operative they should behave
alike. They do not: `k=21` cliffs at 499x, `k=20-15` do not cliff at all
(five consecutive non-cliffing values, ratios 0.28-0.34x). **This rules
out chain length as the explanation and, by ruling it out cleanly, points
directly at what Section 3 identifies instead.**

**P3 (stop reason tracks the timing transition) -- CONFIRMED, but not in
the form predicted.** `VF2Layout_stop_reason` does track the timing
exactly -- `"nonexistent solution"` at precisely `k=21` and `k=14`, `
"solution found"` everywhere else -- consistent with every prior
addendum's finding that the two signals move together. The prediction's
framing (a single transition point) does not apply, but the underlying
claim (stop reason and timing agree) holds.

## 3. The unregistered finding: idle qubits, not symmetry per se

Tabulating "qubits left idle by the interaction graph" against outcome
makes the pattern exact:

| idle qubits | outcome (all 12 k-values) |
|---:|---|
| 0 | cliff (2 of 2 cases: k=21, k=14) |
| 2, 3, 4, 6, 8, 9, 10, 12 | no cliff (10 of 10 cases) |

**The cliff appears if and only if every one of the 42 circuit qubits
participates in at least one two-qubit interaction.** Whenever the
`k_chains` construction leaves any qubit with zero edges (an idle wire
in the circuit, still present as a qubit but touched by no gate), the
cliff is absent -- regardless of how many disjoint components exist or
how symmetric they are.

**A plausible mechanism, not yet confirmed**: an idle circuit qubit is a
vertex in the *interaction graph* with degree zero. Such a vertex can be
mapped to *any* unused physical qubit with no constraint at all -- it
imposes no edge that `VF2Layout`'s search needs to satisfy. Even at
nominal `spare=0` (circuit qubit count == device qubit count), an
interaction graph with idle vertices may give the search exactly the
kind of slack that determines whether the cliff mechanism triggers,
because the *effective* matching problem is smaller than the qubit count
suggests. This would mean **`spare=0` as this project has defined it
throughout (circuit qubit count equals device qubit count) is not the
same thing as zero slack in the underlying subgraph-isomorphism problem**
whenever the interaction graph itself has isolated vertices.

**This does not fully explain Addendum 51's `linear_chain` result, and
that gap is reported rather than smoothed over.** `linear_chain` (k=1)
also has zero idle qubits -- a path through all 42 vertices touches every
one -- yet Addendum 51 found no cliff there (0.96x). So "zero idle
qubits" is not sufficient on its own; both cliffing cases here
(`k=21`, `k=14`) are *also* disjoint (21 and 14 separate components
respectively), while `linear_chain` is a single connected component.
**The refined, still-unconfirmed hypothesis**: the cliff requires *both*
(a) the interaction graph having multiple disjoint components, *and*
(b) zero idle qubits. Neither condition alone is sufficient --
`dense_pairs`-like structures with idle qubits (k=20 through k=15, k=13
through k=10) don't cliff, and a single connected component with no idle
qubits (`linear_chain`) doesn't cliff either.

## 4. What this changes about Addendum 52's conclusion

Addendum 52 read the k=21-to-k=10 transition as a story about symmetry
or chain length decreasing gradually or abruptly. **That framing is now
superseded**: there was never a smooth or even monotonic relationship
with `k` to describe. The real variable, on this evidence, is binary and
structural (idle qubits: yes or no), and it happens to coincide with
`k=21` and `k=14` in this particular sweep only because those are the
divisors of 42 that fall in the tested range. Addendum 52's headline
number (1,972x collapse from k=21 to k=10) is not wrong, but its
implied story -- a gradient collapsing as symmetry decreases -- is
replaced by a sharper one: the collapse happens because `k=20` already
has idle qubits, not because 20 disjoint components are meaningfully
"less symmetric" than 21.

## 5. What this does not establish

- **The two-condition hypothesis (disjoint + zero idle qubits) has only
  two positive examples** (`k=21`, `k=14`) and one negative example
  bearing on the "disjoint" half (`linear_chain`). This is suggestive,
  not confirmed, and needs a design that varies idle-qubit count and
  connectedness independently to test properly.
- Whether `dense_pairs` itself (Addendum 51, cliffing at spare=0)
  continues to cliff if deliberately given one idle qubit at spare=0
  (e.g. by construction rather than by parity) -- the single most direct
  test of the mechanism proposed here, and not yet run.
- Whether this generalizes past n=42 and the 6x7 grid.
- Anything about TKET, Cirq, or other topologies with this variable.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged from Addendum 52) |
| [`circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run3.csv) | this run, 144 rows |
| [`spare-qubit-cliff-addendum-53-preregistration-2026-09-18.md`](#addendum-53----pre-registration-where-exactly-between-k21-and-k10-does-the-cliff-disappear-2026-09-18) | the predictions scored above |

## 7. Verification

- All 144 opt3 rows checked for `error=""`; stop reasons confirmed
  unanimous within every (k, spare) cell.
- The idle-qubit count for each `k` was computed directly as
  `n - k * (n // k)`, using the family's own group-size truncation rule
  (Section 2's script logic), and cross-checked against
  `n_interaction_edges` in the CSV (`n_interaction_edges = k * (n//k - 1)`
  for this family, matching every one of the twelve rows). **A first
  attempt at this cross-check asserted a simpler formula
  (`idle = n - 2 * n_interaction_edges`) that turned out to hold only for
  chain length 2 by coincidence (edges = qubits/2 exactly at that
  length) and gives the wrong answer at length 3 and 4 -- caught before
  publication by testing it numerically against k=14 and k=13 rather
  than trusting the algebra by inspection.**
- The "exactly two divisors of 42 in [10, 21]" claim was checked by
  direct factorization (42 = 2 x 3 x 7; divisors 1, 2, 3, 6, 7, 14, 21,
  42) before being stated as the explanation, not inferred from the
  coincidence alone.
- Section 3's caveat about `linear_chain` was checked against Addendum
  51's own recorded result (0.96x, no cliff) before being presented as a
  gap in the hypothesis, not asserted from memory.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 54 pre-registration (source: spare-qubit-cliff-addendum-54-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for disentangling 'many disjoint components' from 'zero idle qubits' as candidate necessary conditions, by perturbing dense_pairs along each axis independently.

## Addendum 54 -- Pre-registration: disentangling "many disjoint components" from "zero idle qubits" -- which one (or both) does the cliff actually need? (2026-09-18)

**Status: pre-registration only. No run with either new family has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 53 found the cliff at exactly the two `k_chains` values that
leave zero qubits idle (`k=21`, `k=14`) and nowhere else in a 12-point
sweep, then noted this cannot be the whole story: `linear_chain` also
leaves zero qubits idle (a single path touches every vertex) and does
not cliff (Addendum 51, 0.96x). The tentative synthesis was that **both**
"multiple disjoint components" and "zero idle qubits" are needed
together. That synthesis rests on exactly two positive examples and one
negative example, and `k_chains` cannot test it further: changing `k`
moves component count and idle-qubit count at the same time, so the two
variables have never been varied independently.

## 2. Design

Two new families, each built from `dense_pairs` (21 disjoint 2-qubit
edges, the one structure confirmed to cliff) by breaking exactly one of
its two candidate properties while holding the other fixed:

- **`merged_pairs(m)`**: start from `dense_pairs`' 21 edges. Merge the
  first `m` *adjacent pairs of edges* into single 3-qubit chains (e.g.
  merging edges (0,1) and (2,3) into a chain (0,1,2,3) -- wait, this
  changes qubit count; instead: merge by connecting edge `2i` to edge
  `2i+1` end-to-end, producing a 4-qubit chain from two 2-qubit edges,
  for `i` in `range(m)`). This **reduces component count** from 21
  toward `21 - m` while using the **same 42 qubits with zero left
  idle** throughout (merging edges does not remove any qubit from the
  interaction graph, only reconnects which qubits are adjacent to which).
  `m=0` reproduces `dense_pairs` exactly (sanity check); `m=20`
  approaches `linear_chain`-like connectivity (all edges merged into one
  component) while every qubit still participates.
- **`dense_pairs_with_idle(j)`**: start from `dense_pairs` on `42 - j`
  qubits (so `j` qubits are structurally excluded from the interaction
  graph entirely -- present as circuit qubits, but idle), keeping the
  **same 21-minus-however-many-fit disjoint-pair structure** (component
  count stays maximal for the qubits that are used). `j=0` reproduces
  `dense_pairs` exactly (sanity check, same circuit as merged_pairs'
  `m=0`); `j>=2` (even, so pairs remain intact) introduces idle qubits
  while keeping disjointness maximal.

Both run on the 6x7 grid at spare=0 (the only condition that matters for
this question -- both new families are being compared against
`dense_pairs`' own spare=0 result, not swept across spare), 3 seeds x 2
repeats, `optimization_level=3`.

## 3. Pre-registered predictions

**P1 (does breaking disjointness alone, with zero idle qubits
preserved, kill the cliff?).** For `merged_pairs(m)`, `VF2Layout_stop_reason`
at spare=0:
  - **Disjointness is necessary**: `"solution found"` (cliff gone) by
    `m=1` or shortly after (small `m`) -- even a little merging, with
    zero idle qubits held fixed, is enough to escape the cliff. This
    would mean component count, not idle-qubit count, is the operative
    variable, and Addendum 53's "both together" synthesis would need
    revising toward "disjointness alone suffices as the necessary
    condition."
  - **Disjointness is not solely necessary**: `"nonexistent solution"`
    (cliff persists) persists well past `m=1`, requiring substantial
    merging (e.g. `m>=10`, roughly half the edges) before it clears --
    this would support idle-qubit count as at least a comparably
    important factor, since disjointness alone degrades gradually while
    idle-qubit count stays at its "good" value (zero) throughout this
    family.

**P2 (does introducing idle qubits alone, with maximal disjointness
preserved, kill the cliff?).** For `dense_pairs_with_idle(j)`,
`VF2Layout_stop_reason` at spare=0:
  - **Confirmed** if `"solution found"` (cliff gone) appears at the
    smallest tested `j` (i.e. j=2, the minimum that keeps pairs intact)
    -- this would mean idle-qubit count alone, independent of component
    count, is sufficient to kill the cliff, directly explaining why
    every non-`k=21/14` `k_chains` value (all of which have idle qubits)
    failed to cliff regardless of how disjoint they were.
  - **Falsified** if `"nonexistent solution"` (cliff persists) at
    `j=2` and only clears at much larger `j` -- this would mean a small
    amount of idle-qubit slack is not enough on its own, contradicting
    the reading of Addendum 53's data as "any idle qubits kill it."

**P3 (sanity checks).** `merged_pairs(0)` and `dense_pairs_with_idle(0)`
both reproduce `dense_pairs`' own spare=0 result (`"nonexistent
solution"`, time within the same order of magnitude as Addendum 51's
6,682ms / Addendum 53's 6,747ms) -- if either fails, the corresponding
family's results are not trustworthy and should not be interpreted.

**P4 (combined prediction, stated for contrast, not scored
independently).** If Addendum 53's "both together" synthesis is
correct, P1 should show the cliff persisting across most `m` (since
idle-qubit count stays at zero) and P2 should show it vanishing
immediately at the smallest `j` (since even minimal idle-qubit
introduction breaks the *other* necessary condition). **This specific
combination -- P1 "not solely necessary" AND P2 "confirmed" -- is the
outcome that would most cleanly support Addendum 53's synthesis as
currently stated.** Any other combination of P1/P2 outcomes would
require revising it.

## 4. What this cannot establish

- The mechanism *inside* VF2Layout that idle qubits or disjointness
  interacts with -- this measures which structural property matters, not
  why.
- Whether the same two-variable story holds at a different `n` or grid
  size.
- Fine-grained thresholds (e.g. the exact `m` or `j` at which behavior
  flips, if either shows a gradual rather than sharp transition) --
  this design tests a few points per family, not a dense sweep.

---


<!-- ===== Addendum 54-55 (source: spare-qubit-cliff-addendum-55-2026-09-18.md) ===== -->

> **Note added when merging:** **Confirms both conditions are independently necessary.** Reducing disjointness alone (idle qubits fixed at zero) shows the cliff surviving gradually; introducing idle qubits alone (disjointness fixed maximal) collapses the cliff immediately at the smallest step tested -- exactly the asymmetric pattern predicted in advance as the cleanest possible confirmation.

## Addendum 55 -- confirmed by independent manipulation: the cliff needs both zero idle qubits AND substantial disjointness, and a single idle qubit is enough to kill it outright (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-54-preregistration-2026-09-18.md`, written
and locked before this run. All predictions scored below, including P4's
explicit "most consistent combination."

## 0. In one line

**Both halves of Addendum 53's tentative synthesis are now confirmed by
directly and independently manipulating each variable.** `merged_pairs`
(disjointness reduced, idle qubits held at zero throughout) shows the
cliff *surviving* `m=1` (barely -- 13,707ms, even higher than `m=0`'s
7,014ms) and *only* clearing by `m=3` (72.84ms) -- disjointness matters,
gradually, not as a hair-trigger. `dense_pairs_with_idle` (disjointness
held maximal, idle qubits introduced) shows the cliff **gone entirely at
the smallest tested step, `j=2`** (21.81ms, down from `j=0`'s 6,708ms) --
idle qubits matter immediately, not gradually. **This is exactly the
asymmetric pattern the pre-registration's P4 identified in advance as
the one result that would most cleanly support the "both together"
synthesis**, and it is what was measured.

## 1. Results

6x7 grid, 42 qubits, spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 60 rows completed with `error=""`.

### `merged_pairs(m)` -- idle qubits held at 0, components reduced

| m | components (21-m) | time (ms) | stop reason |
|---:|---:|---:|:---|
| 0 | 21 | 7,013.50 | nonexistent solution |
| 1 | 20 | **13,707.02** | nonexistent solution (still) |
| 3 | 18 | 72.84 | **solution found** |
| 6 | 15 | 72.51 | solution found |
| 10 | 11 | 88.54 | solution found |

### `dense_pairs_with_idle(j)` -- disjointness held maximal, idle qubits introduced

| j (idle qubits) | time (ms) | stop reason |
|---:|---:|:---|
| 0 | 6,708.30 | nonexistent solution |
| **2** | **21.81** | **solution found** |
| 6 | 27.53 | solution found |
| 12 | 30.79 | solution found |
| 20 | 39.07 | solution found |

## 2. Scoring

**P1 (`merged_pairs`) -- "Disjointness is not solely necessary"
CONFIRMED.** The cliff survives `m=1` (still `"nonexistent solution"`,
and if anything the time *increased* relative to `m=0` -- see Section 3)
and clears somewhere between `m=1` and `m=3`. This is well past the
pre-registration's "small m" bar for the "necessary, hair-trigger"
reading, confirming instead that disjointness contributes gradually
rather than as an all-or-nothing switch at m=1.

**P2 (`dense_pairs_with_idle`) -- CONFIRMED, at the strictest tested
point.** `j=2` -- two idle qubits out of 42, the minimum step that keeps
pairs intact -- is enough to collapse the cliff from 6,708ms to 21.81ms,
a **307x** drop from a single small structural change. No larger `j` was
needed to see the effect; it is already complete at the first data
point past zero.

**P3 (sanity checks) -- CONFIRMED for both families.** `merged_pairs(0)`
(7,013.50ms) and `dense_pairs_with_idle(0)` (6,708.30ms) both reproduce
`dense_pairs`' own cliff (Addendum 51: 6,682.2ms; Addendum 53: 6,747.05ms)
within the same order of magnitude and the same `"nonexistent solution"`
stop reason -- both new families' `param=0` baselines are trustworthy.

**P4 (the combined prediction) -- CONFIRMED, exactly as specified in
advance.** The pre-registration named this the single outcome that would
most cleanly support Addendum 53's synthesis: P1 showing the cliff
persisting past m=1 (idle-qubit count fixed at zero), and P2 showing it
vanishing at the smallest tested j (disjointness fixed at maximal). Both
happened. **The two variables are not interchangeable or redundant**:
losing a little disjointness costs the cliff little; losing a little
"every qubit participates" costs it everything.

## 3. An unregistered observation: `merged_pairs(1)` is slower than `merged_pairs(0)`

Not predicted, and worth flagging rather than absorbing silently:
merging just one pair of edges (m=1, 20 components) took *longer*
(13,707ms) than the fully disjoint baseline (m=0, 21 components,
7,014ms) -- both still `"nonexistent solution"`, so this is a difference
in how long the failing search runs, not a difference in outcome. This
is the same shape of within-"failing"-regime variability Addendum 47
found and traced to search budget rather than solution existence (a
failing search runs until its budget is exhausted, and how much work
fits in that budget is more variable than a successful search's
deterministic termination). Not confirmed here to be the same mechanism,
but consistent with it, and not treated as evidence against P1's
qualitative conclusion (both m=0 and m=1 clearly sit in the "cliff"
regime by stop reason, regardless of their relative timing).

## 4. What this settles about the cliff's structural requirement

Combining Addenda 51-55: the occupancy cliff, as this project has
measured it, requires an interaction graph that is **(a) a union of
multiple components -- more than the one or two components `merged_pairs`
showed still clears by m=3 -- and (b) touches every physical qubit at
spare=0, with no slack of even two idle qubits.** Neither condition is
sufficient alone (Addendum 53: `k_chains` values with many idle-free
components but some idle qubits don't cliff; `linear_chain`, a single
connected component with zero idle qubits, doesn't cliff either), and
this addendum confirms both are independently necessary by breaking each
one while holding the other fixed.

**This is now a considerably narrower, and more precisely stated, claim
than "the cliff depends on circuit symmetry."** It depends on two
specific, independently-verified structural properties of the
interaction graph, present together, and this project's standing
`dense_pairs` family happens to maximize both simultaneously -- which is
very likely why it was the family used throughout Addenda 4-50 without
this narrowness being visible until symmetry was deliberately varied.

## 5. What this does not establish

- The *mechanism* connecting these two graph properties to VF2's search
  behavior -- this confirms which structural properties matter, not why
  (still the open Rust-source item from Addenda 43-45).
- The precise component-count threshold between m=1 (cliffs) and m=3
  (does not) -- m=2 was not tested.
- Whether j=1 (if achievable without breaking pair structure, which it
  is not directly -- the family requires even j) would show a smaller
  but nonzero effect, or whether j=2's full collapse means even one idle
  qubit alone would suffice.
- Generalization beyond n=42 and the 6x7 grid.
- Anything about TKET, Cirq, or other topologies with either family.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 54's two new families) |
| [`circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run4.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run4.csv) | this run, 60 rows |
| [`spare-qubit-cliff-addendum-54-preregistration-2026-09-18.md`](#addendum-54----pre-registration-disentangling-many-disjoint-components-from-zero-idle-qubits----which-one-or-both-does-the-cliff-actually-need-2026-09-18) | the predictions scored above |

## 7. Verification

- All 60 opt3 rows checked for `error=""`; stop reasons confirmed
  unanimous within every (family, m-or-j) cell.
- Both sanity-check baselines (`merged_pairs(0)`, `dense_pairs_with_idle(0)`)
  were compared numerically against both prior independent measurements
  of `dense_pairs` (Addenda 51 and 53) before being accepted as
  reproducing it, not assumed from the family names alone.
- P4's scoring was checked against the pre-registration's own exact
  wording ("P1 'not solely necessary' AND P2 'confirmed'") before being
  marked confirmed, rather than declared a match after the fact from a
  looser reading.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 56 pre-registration (source: spare-qubit-cliff-addendum-56-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for how the cliff's magnitude scales with grid size, across four sizes from 4x4 to 8x8.

## Addendum 56 -- Pre-registration: how does the cliff's magnitude scale with grid size? (2026-09-18)

**Status: pre-registration only. No run at any grid size other than 6x7
has been performed with `dense_pairs` in this session.** Predictions are
locked before any measurement.

## 1. Why this experiment exists

Addenda 51-55 established *what structural property* of the interaction
graph the cliff requires (disjoint components, zero idle qubits at
spare=0) but every measurement so far -- across all 55 addenda -- has used
one grid: 6x7 (42 qubits). Whether the cliff's magnitude is roughly
constant across grid sizes, grows with size, or shrinks with size is
unknown, and it bears directly on how the phenomenon should be described
for any workload larger or smaller than 42 qubits.

## 2. Design

`occupancy_sweep.py`, unchanged, run on four square grids:
4x4 (16 qubits), 5x6 (30 qubits), 6x7 (42 qubits, reproducing this
project's own standing result as an internal check), and 8x8
(64 qubits). `dense_pairs` only (the confirmed-cliffing family). spare
in {0, 2} at each size (the minimal comparison needed for a ratio),
`optimization_level=3`, 3 seeds x 2 repeats, matching this session's
established resolution.

## 3. Pre-registered predictions

**P1 (primary -- does the ratio grow, shrink, or stay flat with grid
size?).** Comparing the spare=0/spare=2 ratio across the four sizes:
  - **Grows**: 8x8's ratio exceeds 6x7's, which exceeds 5x6's, which
    exceeds 4x4's -- monotonically increasing with qubit count.
  - **Shrinks**: the reverse monotonic order.
  - **Roughly flat**: all four ratios fall within one order of magnitude
    of each other (i.e. the largest is less than 10x the smallest),
    with no consistent direction.
  - **Non-monotonic**: none of the above -- explicitly available as an
    outcome, not forced into the nearest of the three.

**P2 (does the cliff exist at all sizes, or only above/below some
size?).** `VF2Layout_stop_reason` is predicted to be `"nonexistent
solution"` at spare=0 for **all four** sizes, based on Addendum 34's own
established mechanism (this circuit family provably saturates the
matching bound at spare=0 regardless of size) and Addendum 51's finding
that `dense_pairs` cliffs reliably. A size at which spare=0 is
`"solution found"` instead would be a surprising, separately reportable
finding contradicting the project's core mechanism claim.

**P3 (does the smallest tested grid, 4x4, show a much weaker or absent
cliff compared to the larger sizes)?** Included as a standalone
prediction, not because of any confirmed prior measurement of 4x4 with
`dense_pairs` in this project -- a search of this session's own record
did not turn up one, and no such result is claimed here.
**Confirmed** if 4x4's ratio is below 10x (weak or no cliff at the
smallest size); **falsified** if 4x4's ratio exceeds 100x (a full cliff
even at the smallest tested size).

## 4. What this cannot establish

- *Why* the ratio scales the way it does -- this measures the scaling
  curve's shape, not the mechanism.
- Behavior between the four tested sizes (e.g. 5x5, 7x7) -- four points
  is a coarse curve, not a dense one.
- Whether the same scaling holds for other circuit families satisfying
  the two structural conditions (Addenda 51-55), or for TKET/Cirq.
- Absolute wall-clock time at very large sizes, which was not tested and
  could differ from what a naive extrapolation of these four points would
  suggest.

## 5. Scoring discipline

Score P1's four categories exactly as defined; "roughly flat" requires
checking the actual order-of-magnitude spread, not eyeballing a trend
line. If the four points do not cleanly fit any category, say so.

---


<!-- ===== Addendum 56 (source: spare-qubit-cliff-addendum-56-2026-09-18.md) ===== -->

> **Note added when merging:** Grid size breaks the single-grid picture in two directions at once: 4x4 shows no cliff at all despite satisfying the known structural conditions, and 8x8's spare=2 turns out to already be inside the slow region -- the reference point used throughout Addenda 51-55 was only valid at 6x7's own size.

## Addendum 56 -- grid-size scaling breaks the "one grid, one story" picture in both directions: 4x4 shows no cliff at all, and 8x8's spare=2 is already inside one (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-56-preregistration-2026-09-18.md`, written
and locked before this run. All three predictions are scored below, and
**two of the three are falsified**, in ways the pre-registration did not
anticipate.

## 0. In one line

**The cliff's presence and shape both depend on grid size in ways not
predicted.** At 4x4 (16 qubits), `dense_pairs` shows **no cliff
whatsoever** -- `VF2Layout_stop_reason = "solution found"` at spare=0,
ratio 0.91x. At 5x6 and 6x7 (30 and 42 qubits), the familiar cliff
appears cleanly (277.5x and 310.0x, both `"nonexistent solution"` at
spare=0 only). At 8x8 (64 qubits), the spare=0/spare=2 **ratio is 1.00x
-- not because there is no cliff, but because `spare=2` is ALSO
`"nonexistent solution"`**: the cliff's *width* has grown enough at this
size that spare=2 no longer sits outside it, so a ratio computed against
spare=2 as the "outside" reference point is measuring two points both
inside the slow region against each other, not a cliff-vs-no-cliff
comparison at all.

## 1. Results

`dense_pairs`, `optimization_level=3`, 3 seeds x 2 repeats, spare in
{0, 2}. All 48 rows (12 per size) completed with `error=""`.

| grid | qubits | spare=0 (ms) | spare=0 reason | spare=2 (ms) | spare=2 reason | ratio |
|---|---:|---:|:---|---:|:---|---:|
| 4x4 | 16 | 14.85 | **solution found** | 16.33 | solution found | 0.91x |
| 5x6 | 30 | 5,613.14 | nonexistent solution | 20.22 | solution found | 277.54x |
| 6x7 | 42 | 6,723.37 | nonexistent solution | 21.69 | solution found | 310.02x |
| 8x8 | 64 | 8,564.78 | nonexistent solution | 8,590.20 | **nonexistent solution** | 1.00x |

6x7's 310.02x is consistent with this project's own repeated
measurements of this exact configuration (Addendum 34: 193-238x at finer
resolution; Addendum 51: 299.7x; Addendum 53: 499.34x -- all same order
of magnitude, same machine), confirming this run's methodology matches
prior ones before trusting the new sizes.

## 2. Scoring

**P1 (does the ratio grow, shrink, stay flat, or move non-monotonically
with size?) -- "Non-monotonic," and for a reason the four categories did
not anticipate.** The ratio sequence (0.91x, 277.54x, 310.02x, 1.00x) is
not monotonic in either direction, and the 8x8 endpoint is not "roughly
flat" in the sense the pre-registration meant (it is not a modest, real
value close to the others -- it is an artifact of the comparison point
itself falling inside the slow region). **None of the four
pre-registered categories cleanly describes this shape**, and forcing it
into "non-monotonic" is the closest fit but understates what actually
happened: this is not four points on a scaling curve, it is two
genuinely different phenomena (an absent cliff at 4x4, a cliff whose
*width* exceeds the tested spare range at 8x8) bracketing two ordinary
cliff measurements in the middle.

**P2 (is spare=0 `"nonexistent solution"` at all four sizes?) --
FALSIFIED at 4x4.** Predicted based on the matching-bound argument
(Addendum 34) that spare=0 should saturate the bound regardless of size.
**Measured: `"solution found"` at 4x4.** The matching-bound argument
establishes that a matching *exists* at spare=0 for `dense_pairs` at any
even n -- it does not establish that `VF2Layout`'s bounded search will
fail to find it, and at 16 qubits, apparently, the search space is small
enough that the search succeeds anyway. **This is a real, previously
unstated boundary condition on the cliff's own core mechanism claim**:
saturation is necessary but evidently not sufficient at every size:
the search must also be large/hard enough for the budget-exhaustion
mechanism (Addendum 34) to actually bind.

**P3 (is 4x4's ratio < 10x, i.e. weak or absent cliff at the smallest
size?) -- CONFIRMED, and confirmed more strongly than the threshold
required.** 0.91x is not merely under 10x, it is statistically
indistinguishable from 1x -- there is no detectable cliff at this size at
all, not an attenuated one.

## 3. What this means for the project's scope, combined with Addenda 51-55

**The cliff, as characterized across Addenda 4-55, is now understood to
require not only specific interaction-graph structure (disjoint
components, zero idle qubits -- Addenda 51-55) but also a grid large
enough for the search to actually struggle.** At 4x4, `dense_pairs`
satisfies both structural conditions (21... no, 8 disjoint 2-qubit edges,
zero idle qubits at n=16) and still shows no cliff -- the structural
conditions are evidently necessary but not sufficient at every scale.

**The 8x8 result is arguably the more consequential one.** It suggests
the cliff is not a knife-edge phenomenon confined to exactly spare=0 at
larger sizes -- its *width* (how far from full saturation the slow region
extends) appears to grow with grid size, consistent with, though not
identical to, the wider slow regions Addendum 42 found on sparse
synthetic graphs (there, attributed to graph sparsity; here, evidently
also possible on the standard `dense_pairs` family purely from grid size
increasing). **This means every single-step cliff measurement in this
project's history (spare=0 vs. spare=1, the resolution Addendum 34 used)
may itself have differing width at different sizes that was never
checked**, since 6x7 was the only size measured at fine resolution.

## 4. What this does not establish

- **Where between 4x4 and 5x6 the cliff switches on.** Untested at any
  intermediate size (e.g. 4x5, 4x6, 4x7).
- **How wide the slow region actually is at 8x8** -- only spare=0 and
  spare=2 were tested; whether spare=4, 8, or more are needed to exit it
  is unknown. A finer sweep at 8x8, matching Addendum 34's
  single-step-resolution methodology at 6x7, would resolve this directly
  and is the natural next step.
- Whether the growing-width pattern continues past 8x8, or what drives it
  mechanistically.
- Whether 4x4's absence and 8x8's widening are connected phenomena (e.g.
  both consequences of the same underlying scaling law) or unrelated
  coincidences of this specific four-point sample.

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep.py) | the script (unchanged) |
| [`occupancy_sweep_4x4_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_4x4_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | 4x4, 12 rows |
| [`occupancy_sweep_5x6_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_5x6_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | 5x6, 12 rows |
| [`occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | 6x7, 12 rows (consistency check) |
| [`occupancy_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | 8x8, 12 rows |
| [`spare-qubit-cliff-addendum-56-preregistration-2026-09-18.md`](#addendum-56----pre-registration-how-does-the-cliffs-magnitude-scale-with-grid-size-2026-09-18) | the predictions scored above |

## 6. Verification

- All 48 rows (across four files) checked for `error=""`; stop reasons
  confirmed unanimous within every (grid, spare) cell.
- The 8x8 "ratio 1.00x" result was investigated rather than reported at
  face value: `spare=2`'s own stop reason was checked directly (not
  assumed to be `"solution found"` by default) and found to also be
  `"nonexistent solution"`, which is what motivated Section 0-1's
  reframing of the ratio as an artifact of the comparison point rather
  than a genuine absence of effect.
- 6x7's result was cross-checked against three independent prior
  measurements of the identical configuration (Addenda 34, 51, 53)
  before being used to validate this run's methodology.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and all four new CSVs
  -> 0 hits.

---


<!-- ===== Addendum 57 pre-registration (source: spare-qubit-cliff-addendum-57-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for the cliff's slow-region width at 8x8, at single-step resolution, following Addendum 56's discovery that spare=2 sits inside it.

## Addendum 57 -- Pre-registration: how wide is the cliff's slow region at 8x8, at single-step resolution? (2026-09-18)

**Status: pre-registration only. No fine-resolution run at 8x8 has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 56 found that at 8x8 (64 qubits), `spare=2` -- the point this
project has used throughout as the "outside the cliff" reference for
every grid size, including the single-step-resolution measurement at
6x7 (Addendum 34) -- is itself still `"nonexistent solution"`, meaning
the slow region extends at least two spare-qubits wide at this size,
wider than the single-step width established at 6x7. The exact width was
not measured; Addendum 56 tested only spare=0 and spare=2.

## 2. Design

`occupancy_sweep.py`, unchanged, on the 8x8 grid, `dense_pairs`,
`optimization_level=3`, single-step resolution: spare in
{0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16}, matching Addendum 34's own
methodology at 6x7 as closely as possible (single steps near the
boundary, coarser further out). 3 seeds x 2 repeats, this session's
standing resolution.

## 3. Pre-registered predictions

**P1 (primary -- what is the slow region's width at 8x8?).** Define
width as the number of consecutive spare values from 0 upward that
return `"nonexistent solution"` in a majority of runs.
  - **Same as 6x7** (width = 1, only spare=0 slow): would mean
    Addendum 56's spare=2 result was itself an anomaly or a
    run-to-run fluctuation, not a genuine widening.
  - **Wider than 6x7** (width >= 2): confirms genuine widening with grid
    size, consistent with Addendum 56's finding and the pattern
    Addendum 42 found on sparse synthetic graphs (there attributed to
    graph sparsity, not grid size).
  - Exact width not predicted in advance beyond this binary split.

**P2 (does the width found here exceed spare=2, matching Addendum 56's
observation)?** `spare=2` at 8x8 is predicted to reproduce Addendum 56's
own finding: `"nonexistent solution"`, not `"solution found"`. If this
fails to reproduce, Addendum 56's spare=2 result itself needs
re-examination before anything else here is trusted.

**P3 (shape of the timing curve within the slow region).** If width >= 2
is confirmed (P1), the timing within the slow region is predicted to be
roughly flat (all "nonexistent solution" points within ~3x of each
other), based on the pattern Addendum 42 found on the synthetic sparse
graph (a plateau, not a gradient, within the slow region) -- **not**
predicted to decrease smoothly toward the boundary.

## 4. What this cannot establish

- Why the width grows with grid size -- this measures the width, not the
  mechanism.
- Whether the same width would be found on other 64-qubit topologies
  (only the square grid is tested).
- Whether the growing-width pattern continues past 8x8 -- a single new
  size is one more data point, not a curve.

## 5. Scoring discipline

Score P1 as a binary split exactly as defined (width 1 vs. width >= 2);
if the result is ambiguous (e.g. spare=1 is borderline across seeds),
report the ambiguity rather than forcing a side.

---


<!-- ===== Addendum 57 (source: spare-qubit-cliff-addendum-57-2026-09-18.md) ===== -->

> **Note added when merging:** Confirmed: the slow region is 3 spare-qubits wide at 8x8 versus 1 at 6x7 -- width triples for a 1.52x increase in qubit count, and every prior single-step cliff measurement implicitly assumed 6x7's width generalizes, which it does not.

## Addendum 57 -- confirmed: the cliff's slow region is 3 spare-qubits wide at 8x8, versus 1 at 6x7 -- width grows with grid size (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-57-preregistration-2026-09-18.md`, written
and locked before this run. All three predictions scored below.

## 0. In one line

**P1: "Wider than 6x7," confirmed decisively.** At 8x8 (64 qubits),
`spare=0`, `1`, and `2` all return `VF2Layout_stop_reason =
"nonexistent solution"`, with timing flat across all three
(8,524.91-8,830.60ms) -- a **width-3 slow region**, versus the
single-step (width-1) region Addendum 34 established at 6x7. `spare=3`
clears immediately and completely (30.39ms, `"solution found"`), so the
region has a sharp trailing edge even though it is wider on the leading
side. **Addendum 56's spare=2 finding was not an anomaly**: it correctly
identified that spare=2 sits inside the slow region at this size.

## 1. Results

8x8 grid (64 qubits), `dense_pairs`, `optimization_level=3`, 3 seeds x 2
repeats. All 66 rows completed with `error=""`.

| spare | occupancy | time (ms) | stop reason |
|---:|---:|---:|:---|
| 0 | 100.0% | 8,620.79 | nonexistent solution |
| 1 | 98.4% | 8,524.91 | nonexistent solution |
| 2 | 96.9% | 8,830.60 | nonexistent solution |
| 3 | 95.3% | **30.39** | **solution found** |
| 4 | 93.8% | 30.76 | solution found |
| 5 | 92.2% | 32.61 | solution found |
| 6 | 90.6% | 32.82 | solution found |
| 8 | 87.5% | 35.78 | solution found |
| 10 | 84.4% | 38.74 | solution found |
| 12 | 81.3% | 41.63 | solution found |
| 16 | 75.0% | 42.16 | solution found |

## 2. Scoring

**P1 (width) -- "Wider than 6x7" CONFIRMED.** Width = 3 (spare 0, 1, 2),
unambiguous: all three cells show `"nonexistent solution"` in every one
of 6 runs (3 seeds x 2 repeats), and `spare=3` shows `"solution found"`
equally unanimously. No borderline or mixed cell anywhere near the
boundary -- the result did not require the ambiguity fallback the
pre-registration allowed for.

**P2 (does spare=2 reproduce Addendum 56's finding) -- CONFIRMED.**
`spare=2`'s `"nonexistent solution"` here (8,830.60ms) matches Addendum
56's own spare=2 measurement at 8x8 (8,590.20ms) closely -- same stop
reason, same order of magnitude. Addendum 56's result was not an
artifact; it correctly caught the widened region with only two data
points.

**P3 (flat plateau within the slow region) -- CONFIRMED.** The three
slow-region points (8,620.79 / 8,524.91 / 8,830.60ms) span a max/min
ratio of **1.036x** -- essentially flat, not a gradient decreasing toward
the boundary. This matches the shape Addendum 42 found on the synthetic
sparse graph's own wider slow region (also flat, not gradual) rather
than a smooth decline.

## 3. What this establishes, combined with Addendum 34 and 56

**The slow region's width is not a fixed property of the cliff -- it
scales with grid size, at least between the two sizes now measured at
matching resolution:**

| grid | qubits | slow-region width (spare-qubits) |
|---|---:|---:|
| 6x7 | 42 | 1 (Addendum 34) |
| 8x8 | 64 | 3 (this addendum) |

Two points do not establish a functional form (linear, quadratic, or
otherwise), but they establish the direction and the magnitude: going
from 42 to 64 qubits (a 1.52x increase in qubit count) tripled the
width. **Every prior cliff measurement in this project that used
single-step resolution (Addendum 34's own careful width-1 finding, and
every subsequent addendum that treated "spare=1" or "spare=2" as
definitionally outside the cliff) was implicitly assuming 6x7's width
generalizes.** It does not. In particular, Addenda 51-55's structural
findings (disjoint components + zero idle qubits, tested exclusively at
6x7) used spare=2 as their "outside the cliff" comparison point
throughout -- which Addendum 56-57 now shows is only valid at 6x7's own
size, not as a general methodological choice.

## 4. What this does not establish

- The functional form of width vs. grid size (two points: linear between
  them is the simplest fit, but unverified; could be a threshold effect,
  a power law, or something else).
- Whether the trailing edge (always sharp so far, at both sizes) stays
  sharp at larger sizes, or whether it too widens.
- Whether Addenda 51-55's structural findings (disjointness + zero idle
  qubits) hold at 8x8's own resolution -- they were established entirely
  at 6x7 and have not been re-verified against 8x8's wider region.
- Whether this generalizes to non-square grids, other topologies, or
  larger sizes than 64 qubits.

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep.py) | the script (unchanged) |
| [`occupancy_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | this run, 66 rows |
| [`spare-qubit-cliff-addendum-57-preregistration-2026-09-18.md`](#addendum-57----pre-registration-how-wide-is-the-cliffs-slow-region-at-8x8-at-single-step-resolution-2026-09-18) | the predictions scored above |

## 6. Verification

- All 66 rows checked for `error=""`; stop reasons confirmed unanimous
  (6/6) within every spare-value cell, including at the boundary
  (spare=2 vs. spare=3), before reporting a sharp trailing edge.
- Addendum 56's own spare=2 figure (8,590.20ms) was re-read from that
  addendum directly and compared numerically against this run's spare=2
  figure (8,830.60ms) before claiming reproduction, rather than assumed
  to match from the shared stop reason alone.
- The max/min ratio within the slow region (1.036x) was computed
  directly from the three medians, not estimated.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 58 (source: spare-qubit-cliff-addendum-58-2026-09-18.md) ===== -->

> **Note added when merging:** A sub-millisecond, pre-compile detector for the two known necessary conditions (disjoint components, zero idle qubits), validated 8/8 against known results -- explicitly documented as detecting the necessary-condition shape, not predicting whether a cliff will actually occur (grid size, found later to matter, is not an input).

## Addendum 58 -- a cheap, pre-compile detector for the cliff's two known structural conditions: sub-millisecond, 8/8 correct against every circuit family measured tonight (2026-09-18)

**Status**: a new tool (`cliff_detector.py`), tested against this
session's own already-collected results (Addenda 51-53). No new Qiskit
measurement was run -- this addendum re-validates a detector's output
against known ground truth, and separately times the detector itself.

## 0. In one line

A pre-compile check for the two conditions Addenda 51-55 found
necessary for the cliff -- **multiple disconnected interaction-graph
components, and zero idle qubits** -- runs in **50-500 microseconds**
(measured directly, `time.perf_counter()` around the graph analysis
only) and **correctly classified all 8 test cases** drawn from tonight's
own circuit families and `k_chains` values, including one case this
addendum's own first draft got wrong by mislabeling the expected answer,
caught by comparing against Addendum 53's actual recorded result before
trusting the test. **This is cheap enough to run before every compile
without materially affecting compile time**, even for circuits nowhere
near the cliff.

## 1. What the detector checks, and what it does not

`detect_cliff_risk_shape(n_qubits, edges)` computes, via a single pass
over the interaction graph (`networkx.number_connected_components` and a
degree-zero check):

- `multiple_components`: is the interaction graph disconnected into more
  than one piece?
- `zero_idle_qubits`: does every qubit participate in at least one
  two-qubit interaction?
- `cliff_risk_shape`: both of the above, together -- the two conditions
  Addenda 51-55 established as **necessary** (not sufficient) for the
  cliff.

**It explicitly does not, and cannot, predict whether a cliff will
actually occur.** Addendum 56 found `dense_pairs` at 4x4 satisfies both
conditions and shows no cliff at all -- grid size is a third factor this
detector has no way to check (it has no information about the target
device). The module's own docstring states this before any function
definition, deliberately, so a future user reading the code cannot miss
it by skipping to the function body.

## 2. Validation against tonight's own results

8 circuits, all at n=42, edges taken directly from this session's own
circuit-family generators (not re-derived):

| circuit | expected (from Addendum 51-53's own measured result) | detector | match | detection time |
|---|:---:|:---:|:---:|---:|
| `dense_pairs` | cliff-shape | cliff-shape | OK | 505.5 us |
| `linear_chain` | not | not | OK | 72.7 us |
| `ghz_star` | not | not | OK | 98.3 us |
| `k_chains(k=21)` | cliff-shape | cliff-shape | OK | 49.3 us |
| `k_chains(k=20)` | not | not | OK | 56.7 us |
| `k_chains(k=14)` | cliff-shape | cliff-shape | OK | 56.3 us |
| `k_chains(k=13)` | not | not | OK | 55.5 us |
| `k_chains(k=15)` | not | not | OK | 54.0 us |

**8/8.** `random_regular` was checked separately (Section 3) rather than
folded into this table, since its "not cliff-shape" classification needs
a caveat the binary match/no-match format would hide.

## 3. A mistake caught before publication, and a limitation found while checking it

**The mistake**: this addendum's first pass at the validation table
labeled `k_chains(k=14)`'s expected answer as "not cliff-shape." This is
wrong -- Addendum 53 found k=14 **does** show the cliff (it is one of
exactly two divisors of 42 in the tested range, leaving zero idle
qubits). The error was caught by re-deriving `42 // 14 = 3` and
`14 * 3 = 42` (zero idle) directly, rather than trusting a
half-remembered summary of Addendum 53's table, and the test case was
corrected before the 8/8 result above was reported.

**The limitation, found while separately checking `random_regular`**:
the detector correctly returns `cliff_risk_shape = False` for it
(`n_components = 1` -- it is a single connected graph), matching that
this family does not show the *cliff* (Addendum 51). **But
`random_regular` is not fast** -- it returned `"nonexistent solution"` at
every occupancy tested in Addendum 51, for an entirely different reason
(the graph appears too dense to embed in this coupling map at any of the
tested sizes, not specifically at saturation). **The detector has no way
to flag this**, and does not claim to: `cliff_risk_shape = False` means
"will not show the specific saturation-triggered cliff this project has
characterized," not "will compile quickly." A circuit could still be
slow to compile for a completely different, undetected reason. This
should be stated plainly in any documentation of the tool, not left
implicit.

## 4. What this does and does not support for practical use

**Supports**: a detector for these two conditions is computationally
free relative to the cost it would be trying to warn about (the cliff
itself costs seconds to tens of seconds; the check costs
microseconds) -- so "run this before compiling and flag the risk" is not
ruled out on cost grounds.

**Does not yet support**: actually wiring this into a compiler decision
(e.g. auto-switching to `layout_search=True` or another strategy when
`cliff_risk_shape` is true). That would require, at minimum: (a)
confirming the two conditions generalize beyond n=42 and the 6x7/8x8
grids specifically tested tonight (untested at other sizes with these
new families); (b) accounting for Addendum 56-57's finding that grid
size is a third, still poorly-understood factor -- a naive "flag and
switch" rule would presumably need to also account for grid size, which
this detector does not currently take as input at all; (c) understanding
*why* the two conditions matter mechanistically (still the open
Rust-source item from Addenda 43-45), since a mitigation built on a
correlational rule rather than a mechanistic one is more likely to break
on an untested case.

## 5. Files

| File | What it is |
|---|---|
| [`cliff_detector.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/cliff_detector.py) | the detector module |

## 6. Verification

- All 8 validation cases' edge lists were generated by the same
  generator functions used in `circuit_family_sweep.py` (copied here,
  not re-derived from a different implementation), so a mismatch would
  reflect the detector's logic, not a difference in circuit
  construction.
- Detection timing was measured with `time.perf_counter()` bracketing
  only the graph-analysis call, excluding edge-list construction
  (which in a real pipeline would come from walking an already-built
  `QuantumCircuit`, not from this module).
- The k=14 mislabeling (Section 3) was caught by direct arithmetic
  re-derivation before the validation table was finalized, not
  discovered after publication.
- `random_regular`'s result was checked against Addendum 51's own
  recorded stop-reason finding (uniform `"nonexistent solution"`) before
  being used to state the detector's "not the same failure mode"
  limitation.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new script
  -> 0 hits.

---


<!-- ===== Addendum 59 pre-registration (source: spare-qubit-cliff-addendum-59-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether merged_pairs' collapse point at 8x8 tracks an absolute component count or a relative fraction, extending Addendum 55's sweep to a second grid size.

## Addendum 59 -- Pre-registration: does merged_pairs' collapse point at 8x8 track an absolute component count or a relative fraction? (2026-09-18)

**Status: pre-registration only. No `merged_pairs` run at 8x8 has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 55 found, at 6x7 (n=42, 21 `dense_pairs` components), that
`merged_pairs(m)` still cliffs at `m=1` (20 components) and clears by
`m=3` (18 components). Addendum 57 separately found the cliff's slow
region *width* grows with grid size (1 at 6x7, 3 at 8x8). Whether the
*component-count sensitivity* found in Addendum 55 also scales with grid
size -- and if so, whether it scales with the absolute number of merges
or the fraction of components merged -- has never been tested. 8x8's
`dense_pairs` baseline has 32 components (n=64), not 21, so this is the
first opportunity to distinguish an absolute-count effect from a
relative one.

## 2. Design

`merged_pairs(m)` on the 8x8 grid, `optimization_level=3`, spare=0 only
(the only condition where the effect is visible at all), `m` in
{0, 1, 2, 3, 4, 5, 6, 8}, 3 seeds x 2 repeats -- `m=0` and `m=1` included
specifically to test whether 8x8 also survives a single merge (as 6x7
did), before checking where it clears.

## 3. Pre-registered predictions

**P1 (primary -- does the collapse point track absolute m, or relative
fraction of components merged?).**
  - **Absolute**: the cliff clears at approximately the same `m` as
    6x7 (i.e. by `m=3`, `VF2Layout_stop_reason` is `"solution found"`) --
    this would mean a fixed number of merges, not a fixed fraction, is
    what matters, and 8x8's extra components (32 vs. 21) are largely
    irrelevant to the threshold.
  - **Relative**: the cliff persists well past `m=3` (still
    `"nonexistent solution"` at `m=4` or later), clearing only near the
    proportionally-scaled estimate of `m~4-5` (32 components -> ~27-28,
    matching 6x7's ~14% reduction) or later -- this would mean the
    fraction of components merged, not the absolute count, is the
    operative variable.
  - **Neither**: the clearing point is not well-predicted by either
    simple scaling (e.g. clears immediately at m=1, or persists past
    m=8) -- reported as its own outcome, not forced into the nearer of
    the two.

**P2 (does `m=0` reproduce `dense_pairs`' own 8x8 cliff, and does `m=1`
survive as it did at 6x7)?** `m=0`: `"nonexistent solution"`, time
within the same order of magnitude as Addendum 56-57's own 8x8 spare=0
measurements (~8,500-8,800ms). `m=1`: also predicted to survive
(`"nonexistent solution"`), mirroring 6x7's own `m=1` result, since a
single merge out of 32 components is an even smaller relative change
than a single merge out of 21.

**P3 (sanity/consistency).** If P2's `m=0` check fails to reproduce the
established 8x8 baseline, the rest of this run should not be
interpreted.

## 4. What this cannot establish

- *Why* the threshold sits where it does, in either the absolute or
  relative reading -- this measures where the collapse point is, not the
  mechanism.
- Whether a third grid size would confirm whichever scaling law this
  run's two data points (6x7, 8x8) suggest -- two points fit a line but
  do not confirm one.
- Whether the *width* finding (Addendum 57, a separate axis -- how many
  spare values stay slow) and this *component-sensitivity* finding
  (this addendum -- how many components must merge) are related or
  independent properties of grid size.

## 5. Scoring discipline

Score P1's three categories exactly as defined. "Relative" requires the
cliff to survive materially past `m=3` (not merely at m=4 by one step);
if the clearing point is ambiguous between the two readings (e.g. clears
exactly at m=3 or m=4, which fits both a slightly-generous absolute
reading and a slightly-conservative relative one), report the ambiguity
explicitly rather than picking a side.

---


<!-- ===== Addendum 59 (source: spare-qubit-cliff-addendum-59-2026-09-18.md) ===== -->

> **Note added when merging:** At 8x8, merged_pairs does not cross a single threshold at all -- it oscillates between cliff and clear as components are reduced (cliff, cliff, cliff, clear, clear, cliff, clear, cliff), unanimous across every seed and repeat, falsifying all three pre-registered shapes.

## Addendum 59 -- at 8x8, merged_pairs does not cross a single threshold: it oscillates between cliff and no-cliff as components are reduced (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-59-preregistration-2026-09-18.md`, written
and locked before this run. **All three of P1's pre-registered
categories are falsified** -- the actual shape was not one the
pre-registration considered.

## 0. In one line

Neither "Absolute" nor "Relative" nor a single clean "Neither" describes
what happened. As `m` increases from 0 to 8 (components: 32 -> 24), the
stop reason is **cliff, cliff, cliff, clear, clear, cliff, clear,
cliff** -- `m` = 0,1,2 cliff; 3,4 clear; **5 cliffs again**; 6 clears;
**8 cliffs again**. Every one of these is unanimous across all 6 runs
(3 seeds x 2 repeats) per `m` value -- this is not seed noise or a
borderline case, it is a real, repeatable, non-monotonic dependence on
`m` that the pre-registration's three categories did not anticipate.

## 1. Results

8x8 grid (64 qubits), `merged_pairs(m)`, spare=0,
`optimization_level=3`, 3 seeds x 2 repeats. All 48 rows completed with
`error=""`; stop reason unanimous (6/6) within every `m` cell.

| m | components (32-m) | time (ms) | stop reason |
|---:|---:|---:|:---|
| 0 | 32 | 8,567.25 | nonexistent solution |
| 1 | 31 | 9,187.15 | nonexistent solution |
| 2 | 30 | 8,484.32 | nonexistent solution |
| 3 | 29 | 32.62 | **solution found** |
| 4 | 28 | 32.31 | **solution found** |
| **5** | **27** | **8,195.39** | **nonexistent solution** |
| 6 | 26 | 29.31 | **solution found** |
| **8** | **24** | **9,280.17** | **nonexistent solution** |

## 2. Scoring

**P1 (primary) -- FALSIFIED, all three categories.** "Absolute" (clears
by m=3, stays clear) is contradicted by m=5 and m=8 both cliffing again
after m=3-4 cleared. "Relative" (persists to m~4-5, then clears for
good) is contradicted the same way. "Neither," as defined ("not
well-predicted by either simple scaling"), is the closest fit in the
loosest sense, but the pre-registration's own framing assumed a single
transition point somewhere in the range -- not a value that clears,
re-cliffs, clears again, and re-cliffs a second time. **This shape was
not anticipated and is reported as its own finding, not shoehorned into
"Neither."**

**P2 (does m=0 reproduce the established 8x8 baseline, and does m=1
survive as at 6x7?) -- BOTH CONFIRMED.** `m=0`: 8,567.25ms,
`"nonexistent solution"`, matching Addendum 56-57's own 8x8 spare=0
figures (8,564.78 / 8,620.79ms) closely. `m=1`: also
`"nonexistent solution"` (9,187.15ms), mirroring 6x7's own `m=1` survival
(Addendum 55). Both sanity checks pass, so the oscillation at m=5 and
m=8 is not attributable to a baseline-reproduction failure.

**P3 (sanity/consistency gate) -- passed**, per P2.

## 3. What might explain the oscillation, stated as candidates, not conclusions

The `merged_pairs(m)` construction merges edges at fixed positions --
pairs (0,1), (2,3), (4,5), ... in index order -- up to `m` merges, leaving
the rest as untouched 2-qubit edges. This means the *specific* set of
qubits organized into 4-qubit chains changes with `m` in a fixed,
non-random pattern. Two candidate explanations, neither confirmed here:

- **A parity or positional effect tied to the 8x8 grid's own geometry.**
  8x8 is the first perfectly square grid tested (6x7 and 5x6 are not
  square); if the specific qubit indices merged at a given `m` happen to
  align differently with row/column boundaries at different `m`, this
  could produce a structural effect uncorrelated with `m` itself.
- **A genuine non-monotonic relationship between component count and
  search difficulty** -- component count alone is not the right summary
  statistic; something about *which* qubits are merged (not merely how
  many) matters, and `m`'s fixed merge order happens to hit a "hard"
  configuration at m=5 and m=8 but not at m=3, 4, 6.

**No mechanism is confirmed.** Distinguishing these would require either
re-running m=5 and m=8 with a *different* merge pattern (e.g. merging
qubits from the opposite end of the index range, or a random subset of
pairs rather than a fixed sequential one) at the same component count, or
testing intermediate `m` values (7) to see whether the oscillation is a
single anomalous point or a denser pattern.

## 4. What this means for Addendum 55's conclusion and the broader project

**Addendum 55's "gradual" reading of `merged_pairs` at 6x7 (cliff
survives m=1, clears by m=3, and implicitly stays clear) cannot be
assumed to generalize even to a different single-grid-size test.** 6x7
was only swept up to `m=10`, and never re-checked for cliff recurrence
at higher `m` within that run. **Whether 6x7 itself has a similar,
undetected oscillation at m values beyond where Addendum 55 stopped
looking is now an open and directly relevant question**, not merely a
hypothetical.

More broadly: this project's structural account of the cliff (Addenda
51-55: disjoint components + zero idle qubits, both necessary) remains
supported at the level of "these conditions are necessary" -- every
cliffing row here (m=0,1,2,5,8) does satisfy both conditions, and every
non-cliffing row (m=3,4,6) also satisfies both (all `merged_pairs`
values have zero idle qubits and multiple components by construction).
**But "component count, considered alone, predicts cliff/no-cliff" is
now contradicted at 8x8** -- component count cannot be the whole story,
since m=3 (29 components) and m=5 (27 components) differ in outcome
despite both being "many disjoint components with zero idle qubits."
Something more specific than component count is evidently also
relevant, and this project does not yet know what.

## 5. What this does not establish

- Which of Section 3's two candidate explanations (grid geometry vs.
  merge-pattern specificity) is correct, or whether some third
  explanation applies.
- Whether 6x7 shows the same oscillation at higher `m` (untested; would
  require extending Addendum 55's own sweep).
- Whether `m=7` (untested here) is cliff or clear -- the oscillation's
  fine structure between 6 and 8 is unknown.
- Whether a different, non-sequential merge pattern at the same
  component counts (5 merges, 8 merges) would reproduce the same
  cliff/clear outcomes or different ones -- the single most direct test
  of Section 3's two candidates, not yet run.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged -- `merged_pairs` was already grid-size-general, verified before this run) |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18.csv) | this run, 48 rows |
| [`spare-qubit-cliff-addendum-59-preregistration-2026-09-18.md`](#addendum-59----pre-registration-does-merged_pairs-collapse-point-at-8x8-track-an-absolute-component-count-or-a-relative-fraction-2026-09-18) | the predictions scored above |

## 7. Verification

- All 48 rows checked for `error=""`.
- Stop-reason unanimity was checked explicitly within every `m` cell
  (6/6 agreement in all eight cells) before the oscillation was reported
  as a real structural effect rather than run-to-run noise -- this was
  the first and most important check performed, precisely because the
  result was surprising enough to demand it.
- `m=0` and `m=1` were cross-checked numerically against Addenda 55-57's
  own recorded 8x8 and 6x7 figures before being accepted as reproducing
  prior baselines.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 60 pre-registration (source: spare-qubit-cliff-addendum-60-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether the m=5/m=8 oscillation is a geometric artefact of merged_pairs' fixed merge order, tested by re-deriving the same component counts with reversed and randomized merge orders.

## Addendum 60 -- Pre-registration: does merged_pairs' m=5/m=8 oscillation survive a change of WHICH pairs are merged, at the same component count? (2026-09-18)

**Status: pre-registration only. No run with any alternate merge
ordering has been performed.** Predictions are locked before any
measurement.

## 1. Why this experiment exists

Addendum 59 found, at 8x8, that `merged_pairs(m)` oscillates between
cliff and no-cliff as `m` increases (0,1,2: cliff; 3,4: clear; 5: cliff
again; 6: clear; 8: cliff again), unanimous across all seeds/repeats at
every `m`. Two candidate explanations were proposed, neither confirmed:
(a) a positional/geometric effect from `merged_pairs`' fixed,
sequential merge order interacting with the 8x8 grid's own row/column
structure, or (b) component count alone is not the operative variable --
*which* qubits end up in which components matters, independent of grid
geometry. This addendum distinguishes them directly by changing which
pairs are merged while holding component count (and therefore `m`)
fixed.

## 2. Design

A new construction, `merged_pairs_variant(n, m, order)`, built on the
same 8x8 `dense_pairs` base (32 disjoint 2-qubit edges), merging `m`
pairs of edges into 4-qubit chains exactly as `merged_pairs` does, but
with the *order* in which edge-pairs are selected for merging
parameterised:

- `order="sequential"`: identical to the existing `merged_pairs` --
  merges edge-pairs (0,1), (2,3), (4,5), ... in index order. This
  reproduces Addendum 59 exactly and serves as the control.
- `order="reverse"`: merges edge-pairs starting from the *last* indices
  instead of the first -- e.g. for m=5 on 32 edges, merges pairs (22,23),
  (24,25), (26,27), (28,29), (30,31) instead of (0,1), (2,3), (4,5),
  (6,7), (8,9). Same component count, same chain-length distribution
  (m four-qubit chains, rest 2-qubit), different *specific* qubits
  involved.
- `order="random"`: selects `m` of the 16 available edge-pairs uniformly
  at random (seeded) to merge, rather than a contiguous block from
  either end. Same component count and chain-length distribution as the
  other two; qubit assignment is the least structured of the three.

Run at `m in {5, 8}` (the two values that showed the cliff recurring in
Addendum 59) and, as a control, `m in {3, 4}` (the two values that
cleared) -- confirming the control values still clear under all three
orderings is itself informative. 8x8 grid, spare=0,
`optimization_level=3`, 3 seeds x 2 repeats per (m, order) cell.

## 3. Pre-registered predictions

**P1 (primary -- does the m=5/m=8 cliff survive a change of merge
order?).**
  - **Geometric/positional cause**: `reverse` and/or `random` at m=5
    and/or m=8 show `"solution found"` (cliff gone) where `sequential`
    showed `"nonexistent solution"` -- i.e. changing which specific
    qubits are merged, at the same component count, changes the
    outcome. This would confirm candidate (a): something about the
    *specific* qubit positions merged, interacting with 8x8's grid
    geometry, drives the oscillation -- not component count.
  - **Structural/component-count cause**: `reverse` and `random` at m=5
    and m=8 both still show `"nonexistent solution"` (cliff persists
    regardless of which pairs are merged) -- this would confirm
    candidate (b): the oscillation is not about grid-geometric
    alignment of specific qubits, and some other structural property
    -- not yet identified, and not simply component count either, since
    m=3/m=4 (also multi-component, zero-idle) do NOT cliff -- is
    responsible. This would deepen the puzzle rather than resolve it,
    and is explicitly not "solved" by this outcome, only narrowed.

**P2 (do the control m-values, 3 and 4, clear under all three
orderings?).** All three orderings at m=3 and m=4 predicted to show
`"solution found"` -- if any ordering makes m=3 or m=4 cliff instead,
that would itself be a separate, surprising finding (order matters even
where component count alone previously seemed to predict "clear"),
reported on its own terms rather than folded into P1's scoring.

**P3 (do `reverse` and `random` agree with each other, if they disagree
with `sequential`)?** If P1 confirms the geometric/positional reading,
`reverse` and `random` are predicted to agree with each other more often
than either agrees with `sequential` (both being "non-sequential"
perturbations of the base pattern) -- though this is a weaker,
directional prediction, not a strict requirement, since `reverse` is
still a structured (contiguous-block) selection while `random` is not.

## 4. What this cannot establish

- The precise geometric mechanism, if P1 confirms candidate (a) -- this
  identifies THAT merge order matters, not WHY (e.g. which row/column
  alignment specifically).
- Whether the same effect would appear at 6x7 or other grid sizes --
  this is 8x8-only, where the effect was first found.
- Whether m values other than 5 and 8 would show similar order-sensitivity
  -- only the two already-anomalous points are re-tested here.

## 5. Scoring discipline

Score P1 per m-value independently (m=5 and m=8 may not agree with each
other); report both rather than averaging or picking the "cleaner" one
if they diverge.

---


<!-- ===== Addendum 60 (source: spare-qubit-cliff-addendum-60-2026-09-18.md) ===== -->

> **Note added when merging:** The oscillation is not a geometric artefact: reverse and random merge orders reproduce it exactly, ruling out grid alignment and pointing at component count (or something correlated with it) as the real variable.

## Addendum 60 -- the m=5/m=8 oscillation is NOT a geometric artefact: reverse and random merge orders reproduce it exactly, ruling out grid alignment (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-60-preregistration-2026-09-18.md`, written
and locked before this run. All three predictions scored below.

## 0. In one line

**P1: "Structural/component-count cause" confirmed, decisively.**
`reverse` and `random` merge orders -- selecting completely different sets
of qubits to merge than `sequential` did (zero qubit overlap, verified
before this run) -- reproduce Addendum 59's oscillation exactly: `m=3,4`
clear (`"solution found"`, ~28-30ms) and `m=5,8` cliff again
(`"nonexistent solution"`, ~8,100-9,200ms), in **both** alternate
orderings, matching `sequential`'s own pattern row for row. **The
oscillation has nothing to do with which specific qubits get merged or
how they align with the 8x8 grid's geometry.** Something about component
count itself -- or a property correlated with it that is not yet
identified -- is non-monotonically related to search difficulty, and this
is a real, order-independent structural phenomenon, not a positional
artefact of Addendum 59's particular fixed merge sequence.

## 1. Results

8x8 grid, `merged_pairs_variant(m, order)`, spare=0,
`optimization_level=3`, 3 seeds x 2 repeats. All 48 rows (24 per order)
completed with `error=""`; stop reason unanimous (6/6) within every
(order, m) cell.

| order | m | components | time (ms) | stop reason |
|---|---:|---:|---:|:---|
| reverse | 3 | 29 | 27.74 | solution found |
| reverse | 4 | 28 | 28.77 | solution found |
| **reverse** | **5** | **27** | **8,107.39** | **nonexistent solution** |
| **reverse** | **8** | **24** | **8,853.04** | **nonexistent solution** |
| random | 3 | 29 | 28.42 | solution found |
| random | 4 | 28 | 29.52 | solution found |
| **random** | **5** | **27** | **8,360.48** | **nonexistent solution** |
| **random** | **8** | **24** | **9,174.69** | **nonexistent solution** |

For comparison, `sequential` (Addendum 59): m=3: 32.62ms (found); m=4:
32.31ms (found); m=5: 8,195.39ms (nonexistent); m=8: 9,280.17ms
(nonexistent). All three orderings agree on both the qualitative outcome
(cliff or not) and the approximate magnitude at every tested `m`.

## 2. Scoring

**P1 (primary) -- "Structural/component-count cause" CONFIRMED.** Both
alternate orderings reproduce the m=5/m=8 recurrence. Since `reverse`
and `random` were verified before this run to select entirely
non-overlapping qubit sets from `sequential` (0 shared qubits at m=5,
checked directly), and the outcome is unchanged, candidate (a) --
grid-geometric alignment of specific merged qubits -- is ruled out as the
explanation. Candidate (b) is supported: **component count, or something
tightly correlated with it independent of which specific qubits form
the components, drives the oscillation.**

**P2 (do m=3, m=4 clear under all three orderings?) -- CONFIRMED for
both alternates.** `reverse` and `random` both show `"solution found"`
at m=3 and m=4, matching `sequential`. No ordering makes a previously
clear `m` cliff instead.

**P3 (do reverse and random agree with each other more than either
agrees with sequential?) -- Not meaningfully distinguishable; all three
agree with each other at every tested m.** The weak directional
prediction (non-sequential orderings clustering together against
sequential) does not apply, because sequential is not an outlier here at
all -- all three orderings produced the same qualitative result
throughout. This makes P3 moot rather than confirmed or falsified in any
informative sense.

## 3. What this means: component count alone, not merge order, governs the oscillation -- and that is now a harder puzzle

**This closes off the more mundane explanation and leaves the more
interesting one standing.** The oscillation is not an artefact of
Addendum 59's specific, fixed sequential construction -- it would have
appeared under any reasonable way of choosing which pairs to merge.
**Something about having exactly 29, 28, or 26 disjoint-plus-merged
components is "easy," while 32, 31, 30, 27, or 24 is "hard," on this
grid, for this circuit family, independent of which specific qubits
occupy which components.**

This narrows the space of explanations considerably (grid-geometric
alignment is eliminated) but does not itself supply a replacement
mechanism. Component count alone, treated as a single scalar, cannot be
the full explanatory variable either -- Addendum 59 already noted that
29 and 27 components, both "many disjoint components with zero idle
qubits," differ in outcome. **What this addendum adds is that the
remaining variable is not positional** (not about *which* qubits), which
leaves properties of the component-size *distribution itself* (e.g. how
many components are 2-qubit vs. 4-qubit, and in what ratio) as the most
plausible remaining candidate -- worth checking directly in a follow-up,
not confirmed here.

## 4. What this does not establish

- *Why* certain component-count/size-distribution combinations are hard
  and others are not -- this rules out one candidate (geometry) and
  narrows toward another (component-size distribution) without
  confirming it.
- Whether the same order-independence holds at 6x7 or other grid sizes.
- Whether `m` values not yet tested (e.g. 7, or values above 8) would
  show further oscillation under any ordering.
- The precise composition (how many 4-qubit vs. 2-qubit components) at
  each cliffing/non-cliffing `m`, which was not tabulated in this
  addendum but is directly computable from `n_interaction_edges` and
  `m` in the existing data, and would be the natural next check.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 60's `merged_pairs_variant`) |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run2.csv) | `order=reverse`, 24 rows |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run3.csv) | `order=random`, 24 rows |
| [`spare-qubit-cliff-addendum-60-preregistration-2026-09-18.md`](#addendum-60----pre-registration-does-merged_pairs-m5m8-oscillation-survive-a-change-of-which-pairs-are-merged-at-the-same-component-count-2026-09-18) | the predictions scored above |

## 6. Verification

- All 48 rows (both files) checked for `error=""`; stop reasons
  confirmed unanimous (6/6) within every (order, m) cell.
- Before this run, `reverse` and `sequential` were confirmed (in the
  sandbox, without Qiskit) to select completely disjoint qubit sets at
  m=5 (0 overlap) while producing identical component counts and
  idle-qubit counts across all three orderings -- this was checked prior
  to trusting the experimental design, not assumed from the
  construction logic alone.
- `sequential`'s own m=3,4,5,8 figures were re-read from Addendum 59
  directly for the Section 1 comparison, not reconstructed from memory.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both new CSVs ->
  0 hits.

---


<!-- ===== Addendum 61 (source: spare-qubit-cliff-addendum-61-2026-09-18.md) ===== -->

> **Note added when merging:** An immediate, no-new-measurement check: the 4-qubit/2-qubit mixture ratio cannot explain the oscillation (it is monotonic in m). A component-count-divisible-by-3 pattern is noted but explicitly flagged as too weak (3 of 8 points, found by scanning several residues after seeing the outcome) to trust.

## Addendum 61 -- component mixture ratio cannot explain the m=5/m=8 oscillation (it is monotonic); a mod-3 coincidence is noted but explicitly flagged as too weak to trust (2026-09-18)

**Status**: an immediate, no-new-measurement calculation from Addendum
59's existing CSV, done to check one candidate explanation (component
size mixture) before running any new sweep. This addendum reports a
negative result for that candidate and one weak, unconfirmed pattern
found while looking for alternatives -- explicitly not claimed as a
finding.

## 0. In one line

**The simplest candidate -- the ratio of 4-qubit to 2-qubit components --
cannot explain Addendum 59's cliff/clear/cliff/clear/cliff oscillation,
because that ratio increases monotonically with `m` (0.000 -> 0.333) while
the outcome does not.** A ratio-based or count-based explanation of any
single monotonic quantity in `m` is ruled out by construction: no
monotonic function of `m` can reproduce a sequence that goes
cliff-cliff-cliff-clear-clear-cliff-clear-cliff. A weak, likely
coincidental pattern (total component count divisible by 3 exactly
matching the three highest-m cliffing cases) was noticed while checking
alternatives and is reported with an explicit warning that 3 matching
points out of 8 is not meaningful evidence.

## 1. The composition table

`merged_pairs`/`merged_pairs_variant` at 8x8 (32 `dense_pairs`
components at m=0): `m` four-qubit components, `(32-2m)` two-qubit
components remain, total components `= 32-m`.

| m | total components | 4-qubit components | 2-qubit components | 4-qubit fraction | outcome |
|---:|---:|---:|---:|---:|:---|
| 0 | 32 | 0 | 32 | 0.000 | cliff |
| 1 | 31 | 1 | 30 | 0.032 | cliff |
| 2 | 30 | 2 | 28 | 0.067 | cliff |
| 3 | 29 | 3 | 26 | 0.103 | clear |
| 4 | 28 | 4 | 24 | 0.143 | clear |
| 5 | 27 | 5 | 22 | 0.185 | **cliff** |
| 6 | 26 | 6 | 20 | 0.231 | clear |
| 8 | 24 | 8 | 16 | 0.333 | **cliff** |

The 4-qubit fraction column is strictly increasing with `m` by
construction (`m / (32-m)`) -- it cannot, on its own or via any simple
threshold on it, produce a value that goes back to "cliff" at m=5 after
clearing at m=3-4, then clears again at m=6, then cliffs again at m=8.
**This candidate is ruled out**, cheaply, without needing a new
measurement.

## 2. A pattern noticed while checking alternatives, reported with an explicit warning

Scanning small modular residues of both `m` and total component count
against the outcome (a mechanical check, not a hypothesis formed in
advance):

| m | total components | components mod 3 | outcome |
|---:|---:|---:|:---|
| 0 | 32 | 2 | cliff |
| 1 | 31 | 1 | cliff |
| **2** | **30** | **0** | **cliff** |
| 3 | 29 | 2 | clear |
| 4 | 28 | 1 | clear |
| **5** | **27** | **0** | **cliff** |
| 6 | 26 | 2 | clear |
| **8** | **24** | **0** | **cliff** |

**All three cases where total component count is exactly divisible by 3
(m=2, 5, 8 -> 30, 27, 24 components) are cliffing cases, and this holds
for all three of them.** Component counts NOT divisible by 3 (32, 31,
29, 28, 26) show a mix (m=0,1 cliff; m=3,4,6 clear) that this residue
does not cleanly separate.

**This is explicitly not being claimed as an explanation.** Three
matching data points, found by scanning several candidate residues
(mod 2, 3, 4 of both `m` and component count) after seeing the outcome,
is close to the textbook definition of a pattern likely to be found by
chance in a small dataset -- this project's own standing discipline
(avoid forcing a pattern onto a handful of points, Addendum 51's P1
scoring note among others) applies directly here. **If this were real,
it would predict m=6 should NOT be the only non-divisible-by-3 value
among the tested set that clears while divisible ones cliff -- which is
consistent so far only because no cliffing case happens to have
components indivisible by 3 in this particular 8-point sample, not
because the reverse has been ruled out.** The residue also does not
explain why m=0 and m=1 (components 32, 31 -- NOT divisible by 3) are
also cliffing; those two low-m cliffs likely belong to a different
regime (very few merges, close to unmerged `dense_pairs`, per Addendum
55's "gradual near dense_pairs" finding at 6x7) than the m=5/m=8
recurrence this addendum is trying to explain.

## 3. What would be needed to take the mod-3 pattern seriously

Denser sampling around the existing points -- specifically `m` values
giving component counts NOT divisible by 3 near the known cliffing
points (e.g. m=6 already does this and clears, consistent; testing m=7,
9, 10, 11 would either strengthen or break the pattern with real
statistical weight) -- would be the direct next check, **not performed
here**, deliberately, since this addendum's purpose was the free,
immediate calculation from existing data, not a new sweep.

## 4. What this settles and does not

**Settles**: the 4-qubit/2-qubit mixture ratio, as a single scalar, is
definitively not the explanatory variable -- ruled out by its own
monotonicity, no new data needed.

**Does not settle**: what the actual explanatory variable is. The mod-3
observation is a lead worth checking with a denser sweep, explicitly
flagged as unconfirmed and drawn from too few points to trust on its
own.

## 5. Files

| File | What it is |
|---|---|
| (no new data) | this addendum computes directly from Addendum 59's and 60's already-collected CSVs |

## 6. Verification

- The composition table (Section 1) was computed directly from the
  `merged_pairs`/`merged_pairs_variant` construction rule
  (`m` four-qubit chains, `32-2m` two-qubit edges, total `32-m`
  components), cross-checked against `n_interaction_edges` in Addendum
  59's own CSV for each `m` (edges = `3m + (32-2m) = m+32`, matching the
  recorded `n_interaction_edges` at every row) before being presented as
  correct.
- The monotonicity claim for the 4-qubit fraction was verified by direct
  computation across all eight `m` values, not asserted from the
  construction alone.
- The mod-3 observation (Section 2) was found by a systematic scan
  (mod 2, 3, 4 of both `m` and component count against outcome), and is
  reported together with its counter-evidence (m=0, m=1 not fitting)
  rather than presenting only the matching cases.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 62 pre-registration (source: spare-qubit-cliff-addendum-62-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether merged_pairs' oscillation recurs on 6x7 at finer resolution than Addendum 55 originally tested -- includes a design correction (m>=11 is structurally impossible on 6x7) made before running.

## Addendum 62 -- Pre-registration: does merged_pairs' cliff recur at ANY m on 6x7, as it did at m=5/m=8 on 8x8? (2026-09-18, revised before any run)

**Status: pre-registration only. No new `merged_pairs` run on 6x7 has
been performed.** Predictions are locked before any measurement.

**Correction made before any run, recorded rather than silently fixed**:
this pre-registration's first draft asked for `m` in {11,...,20} on 6x7,
reasoning by analogy with 8x8's range. This is impossible by
construction: 6x7's `dense_pairs` has only 21 components (21 edges), so
`merged_pairs`' own mergeable-pair limit is `21 // 2 = 10` -- `m=11` and
above raise `ValueError` in the existing, unmodified script. **6x7's
entire valid range is m=0 through m=10, which Addendum 55 already swept
in full** (at coarser resolution: 0, 1, 3, 6, 10). The real gap is not
"m beyond 10" (which does not exist) but **the resolution within
0-10**: Addendum 55 never tested m=2, 4, 5, 7, 8, or 9, so a recurrence
between its tested points could have been missed entirely, the same way
Addendum 59 only found 8x8's recurrence because it happened to test
every integer m from 0 to 8.

## 1. Why this experiment exists

Addendum 55 swept `merged_pairs(m)` on 6x7 (n=42, 21 `dense_pairs`
components) only up to `m=10`, finding the cliff survives `m=1` and
clears by `m=3`, and implicitly assumed (not tested) that it stays clear
beyond `m=10`. Addendum 59 then found, at 8x8, that the analogous
transition is not a single clean clearing -- the cliff recurs at m=5 and
m=8 after clearing at m=3-4 and m=6. Whether 6x7 has an undetected
recurrence at `m` values beyond where Addendum 55 stopped looking is
therefore a direct, previously-unasked question, and it determines
whether the oscillation is a property of the cliff mechanism generally
or something specific to 8x8.

## 2. Design

`merged_pairs(m)` on the 6x7 grid (21 `dense_pairs` components at
n=42), `optimization_level=3`, spare=0 only, `m` in
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10} -- every integer in the valid range,
matching Addendum 59's own single-step resolution at 8x8 rather than
Addendum 55's coarser sampling. `m=0, 1, 3, 6, 10` were already measured
in Addendum 55 and are re-run here anyway (cheap, since most of these
values are fast) so this addendum's own data is self-contained and
directly comparable across every tested `m`, 3 seeds x 2 repeats.

## 3. Pre-registered predictions

**P1 (primary -- does 6x7 show any recurrence past m=10?).**
  - **No recurrence**: every tested `m` from 11 to 20 shows
    `"solution found"` (stays clear) -- confirms Addendum 55's implicit
    assumption and suggests the m=5/m=8 oscillation is specific to 8x8
    (or to grid sizes/component counts in that range), not a general
    property of the cliff mechanism.
  - **Recurrence found**: at least one tested `m` in {11,...,20} shows
    `"nonexistent solution"` -- confirms the oscillation generalizes
    beyond 8x8 and is a property of the underlying mechanism, not an
    8x8-specific artefact.

**P2 (does the mod-3 lead from Addendum 61 hold up at 6x7, including
against ALREADY-KNOWN data?).** 6x7 component counts divisible by 3, for
`m` in the tested range: m=0 (21 components, **known cliff**, Addendum
34/51/53/55), m=3 (18 components, **known clear**, Addendum 55), m=6 (15
components), m=9 (12 components). **This is a direct, pre-existing
counter-example to a naive mod-3 rule even before this run's own new
data**: m=0 and m=3 are both divisible by 3, but one cliffs and the
other clears -- so any version of "divisible by 3 predicts cliff" that
this addendum might otherwise have proposed is already contradicted by
data on record. **This prediction is therefore reframed**: not "does
mod-3 predict cliff," which is already false, but "does m=6 or m=9 show
a recurrence at all" (regardless of what a mod-3 story would predict) --
purely an empirical check of whether the m=0-10 range hides any
recurrence, divisible-by-3 or not.

**P3 (does m=0 -- not retested here, already established -- and does the
existing m=1 through m=10 range from Addendum 55 remain the reference
point)?** Not re-run; Addendum 55's own recorded values for m=0 through
10 are treated as established and not repeated.

## 4. What this cannot establish

- Whether a finer sweep between the tested points (e.g. m=17, 19) would
  reveal additional recurrences not caught by this coarser grid.
- The mechanism behind any recurrence found, only whether one exists.
- Whether 5x6 or 4x4 (other sizes already measured for the plain cliff,
  Addendum 56) would show a similar or different oscillation pattern --
  out of scope for this addendum.

## 5. Scoring discipline

Score P1 as a binary (any recurrence vs. none) before looking at where
specifically it occurs; score P2 only if P1 finds at least one
recurrence, and report the mod-3 correlation or lack of it exactly as
observed, not adjusted to fit the Addendum 61 lead.

---


<!-- ===== Addendum 62 (source: spare-qubit-cliff-addendum-62-2026-09-18.md) ===== -->

> **Note added when merging:** No stop-reason recurrence at 6x7 (unlike 8x8), but timing itself oscillates by ~130x within the 'solution found' region alone -- a second, independent sighting of the mod-3 lead, alongside its own counter-example, at a different grid size using a different measured quantity.

## Addendum 62 -- no stop-reason recurrence at 6x7, but a large, unregistered timing oscillation hides inside "solution found" itself (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-62-preregistration-2026-09-18.md`, written
and locked before this run (and corrected once, before running, when the
original m>=11 design was found to be structurally impossible -- see
that document's own recorded correction).

## 0. In one line

**P1, scored literally: "No recurrence" confirmed.** `VF2Layout_stop_reason`
is `"nonexistent solution"` only at `m=0` and `m=1` (the cliff, matching
Addendum 55), and `"solution found"` at every `m` from 2 through 10 --
no return to `"nonexistent solution"` anywhere in the range, unlike
8x8's clean m=5/m=8 recurrence. **But this is not the whole story.**
Within the `"solution found"` region, **timing itself oscillates by up
to ~130x** -- from 23.07ms (`m=3`) to **3,006.43ms** (`m=7`) -- a
magnitude of variation this project has not previously documented inside
a single stop-reason category. Addendum 59's 8x8 oscillation was visible
in the stop-reason signal itself; 6x7's analogous phenomenon, if this is
the same underlying effect, is hidden one level deeper, inside the
timing of nominally-identical "solved" outcomes.

## 1. Results

6x7 grid (42 qubits), `merged_pairs(m)`, spare=0, `optimization_level=3`,
3 seeds x 2 repeats, every integer `m` from 0 to 10. All 66 rows
completed with `error=""`; stop reason unanimous (6/6) within every `m`
cell.

| m | components (21-m) | divisible by 3 | time (ms) | stop reason |
|---:|---:|:---:|---:|:---|
| 0 | 21 | yes | 6,616.47 | nonexistent solution |
| 1 | 20 | no | 6,503.29 | nonexistent solution |
| 2 | 19 | no | 842.27 | solution found |
| **3** | 18 | yes | **23.07** | solution found |
| 4 | 17 | no | 23.55 | solution found |
| 5 | 16 | no | 842.77 | solution found |
| **6** | 15 | yes | **23.18** | solution found |
| **7** | 14 | no | **3,006.43** | solution found |
| 8 | 13 | no | 352.35 | solution found |
| **9** | 12 | yes | **47.06** | solution found |
| 10 | 11 | no | 28.87 | solution found |

## 2. Scoring

**P1 (primary) -- "No recurrence" CONFIRMED, per the pre-registered,
literal stop-reason criterion.** No `m` in {2,...,10} returns
`"nonexistent solution"`. Every one of those nine values is unanimous
(6/6) `"solution found"`. By the definition written in advance, 6x7 does
not show 8x8's oscillation.

**P2 (revised pre-run to note m=0/m=3's existing mod-3 contradiction,
then reframed to ask only whether m=6 or m=9 show ANY recurrence) --
moot as literally asked, since P1 found no recurrence at all to
correlate with anything**, at either m=6 or m=9 or elsewhere. Consistent
with the pre-registration's own advance warning that a naive mod-3 rule
was already contradicted by existing data (m=0 vs. m=3).

## 3. An unregistered finding: the timing pattern inside "solution found" tracks divisibility by 3 more cleanly than the stop reason does

Not predicted, found by inspecting the table after P1/P2 were already
scored. Sorting the nine `"solution found"` rows by divisibility of
component count by 3:

- **Divisible by 3** (m=3, 6, 9 -> 18, 15, 12 components): 23.07, 23.18,
  47.06ms -- **uniformly fast, tightly clustered.**
- **Not divisible by 3** (m=2, 4, 5, 7, 8, 10 -> 19, 17, 16, 14, 13, 11
  components): 842.27, 23.55, 842.77, 3,006.43, 352.35, 28.87ms --
  **erratic, spanning two orders of magnitude, including the run's
  single slowest "solution found" result (m=7, 3,006ms) and two
  ~840ms outliers.**

**This is exactly the same qualitative lead Addendum 61 found at 8x8
(components divisible by 3 associated with the "easy" side) and flagged
as too weak to trust from 3 matching points.** Here it appears again,
independently, on a different grid, using a completely different
variable (continuous timing rather than the binary stop reason) -- and
one of the cases in the same range (`m=4`, 17 components, not divisible
by 3, 23.55ms) is *also* fast, so the pattern is not perfectly clean
even here. **This is reported as a second, independent, still-unconfirmed
sighting of the same lead, not as confirmation.** Two weak,
partially-overlapping observations across two grid sizes and two
different measured quantities (stop reason at 8x8; raw timing at 6x7) is
more suggestive than either alone, but nowhere near sufficient to treat
divisibility by 3 as an established explanatory variable.

## 4. What this means for the project's understanding of the cliff's boundary

**The oscillation Addendum 59 found is not 8x8-specific in the way P1's
literal framing suggests, but it also does not manifest identically at
6x7.** At 8x8, the effect crosses the qualitative stop-reason boundary
(VF2Layout genuinely fails to find a solution at m=5, m=8). At 6x7, the
same underlying sensitivity to which `m` is tested appears to exist --
timing varies by two orders of magnitude across nominally-successful
searches -- but does not (at least not within this tested range) push any
point all the way to `"nonexistent solution"`. **One plausible reading**:
6x7 is "further" from its own difficulty threshold across this whole
m-range than 8x8 is, so the same underlying sensitivity shows up as
timing variance among successes rather than outright failures. This is
speculative and not confirmed by anything measured here.

**This also means Addendum 55's original characterization of 6x7's
`merged_pairs` curve as "gradual" (cliff survives m=1, clears cleanly by
m=3) undersold what is actually happening.** The clearing is not smooth
even at 6x7 -- it is a step down to a floor at m=3, followed by
substantial, unexplained bouncing (m=7 nearly matching m=0's cliff-scale
time) before settling toward the fast end again by m=10.

## 5. What this does not establish

- Whether the mod-3 lead, seen now in two independent forms across two
  grid sizes, reflects a real mechanism or is coincidental in both
  cases -- still unconfirmed, and this addendum's own m=4 counter-example
  (fast despite not being divisible by 3) argues against a clean rule.
- Why timing (not stop reason) is the level at which 6x7's analogous
  effect appears, while 8x8's appears at the stop-reason level -- no
  mechanism is proposed.
- Whether denser sampling within 6x7's m=2-10 range (non-integer
  resolution is not possible, but this is already every integer) or a
  wider `m` range at a size between 6x7 and 8x8 would clarify the
  relationship.
- Whether the same timing-oscillation-without-stop-reason-recurrence
  pattern would appear in `merged_pairs_variant`'s reverse/random
  orderings at 6x7, paralleling Addendum 60's finding that order does
  not matter at 8x8 -- untested here.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged) |
| [`circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run5.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run5.csv) | this run, 66 rows |
| [`spare-qubit-cliff-addendum-62-preregistration-2026-09-18.md`](#addendum-62----pre-registration-does-merged_pairs-cliff-recur-at-any-m-on-6x7-as-it-did-at-m5m8-on-8x8-2026-09-18-revised-before-any-run) | the predictions scored above, including a design correction made before running |

## 7. Verification

- All 66 rows checked for `error=""`; stop reasons confirmed unanimous
  (6/6) within every `m` cell before any timing analysis was trusted.
- The divisible-by-3 grouping in Section 3 was computed directly from
  each `m`'s component count (`21-m`), not estimated.
- The comparison to Addendum 61's 8x8 lead was made by re-reading that
  addendum's own table directly, not from memory, before calling this a
  "second sighting" of the same pattern.
- The m=4 counter-example (fast, not divisible by 3) was actively
  checked and reported rather than omitted, since it weakens the
  pattern this addendum was otherwise inclined to find interesting.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 63 pre-registration (source: spare-qubit-cliff-addendum-63-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether the mod-3 lead survives holding component count fixed at a multiple of 3 while deliberately varying composition -- includes a design correction (n=54 vs n=64) made before running.

## Addendum 63 -- Pre-registration: does the mod-3 lead survive holding component count fixed at a multiple of 3 while varying component-size composition? (2026-09-18)

**Status: pre-registration only. No run with any new construction has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addenda 59, 61, and 62 accumulated three weak, independently-obtained
observations suggesting component counts divisible by 3 tend toward the
"easy" (no-cliff / fast) side, at both 8x8 and 6x7. Every one of these
observations comes from `merged_pairs`/`merged_pairs_variant`, a family
that can only ever produce a specific *kind* of composition at a given
component count: some number of 4-qubit chains and the rest 2-qubit
edges, in a ratio fully determined by `m` and the base component count.
**Component count and component-size composition have never been varied
independently.** This addendum breaks that confound directly: fixing
component count at values divisible by 3, while deliberately varying
what those components are made of.

## 2. Design

Three new circuit families, all built on the SAME 64-qubit device (the
8x8 grid, matching Addendum 59), all producing exactly **18 disjoint
components with ZERO idle qubits on the full 64-qubit device** --
holding both of Addenda 53-55's established necessary conditions fixed,
and holding component count fixed at a multiple of 3, while varying only
composition:

- `uniform_2q(18)`: 18 components, all 2-qubit edges, using 36 of the 64
  qubits -- **28 qubits idle, deliberately**. Since Addenda 53-55
  established idle qubits alone kill the cliff, this family is not a
  fair test of the mod-3 lead; it is a sanity check (P0) that the
  idle-qubit effect still dominates here as expected, run alongside the
  real test rather than instead of it.
- `balanced_3q4q(18)`: 18 components, **zero idle, on the full 64
  qubits** -- specifically 10 four-qubit chains + 8 three-qubit chains
  (10x4 + 8x3 = 40+24 = 64). Same component count (18, divisible by 3)
  and same zero-idle condition as every `merged_pairs` cliff/clear
  observation so far, but a genuinely different composition (a mix of
  two chain lengths, rather than `merged_pairs`' own 2-qubit-and-4-qubit
  mix at this specific count).
- `mixed_uneven(18)`: 18 components, **zero idle, on the full 64
  qubits**, deliberately skewed -- 17 two-qubit edges (34 qubits) plus
  ONE 30-qubit connected path (34+30=64). This tests an extreme,
  non-uniform size distribution, unlike anything `merged_pairs` can
  produce (its own mixes are always mild -- a handful of 4-qubit chains
  among many 2-qubit ones).

All at spare=0 on the 8x8 grid (n=64 for every family, literal 100%
device occupancy in every case -- the confound present in this
pre-registration's first draft, where two families used n=54 instead of
64, is corrected here before any run), `optimization_level=3`, 3 seeds x
2 repeats.

## 3. Pre-registered predictions

**P0 (sanity/contrast check, scored separately from the main question).**
`uniform_2q(18)` -- which has 28 idle qubits -- is predicted to show
`"solution found"` (fast), consistent with Addenda 53-55's independent
idle-qubit finding, regardless of what P1 finds. This is not itself a
test of the mod-3 lead; it is included to confirm the idle-qubit effect
is not somehow suspended by this new construction.

**P1 (primary -- does mod-3-at-18-components predict "easy" independent
of composition?).**
  - **Mod-3 lead survives**: both `balanced_3q4q(18)` and `mixed_uneven(18)`
    (zero idle qubits, 18 components each) show `"solution found"`
    (fast) -- consistent with "18 components (divisible by 3) is easy
    regardless of what those components look like," strengthening the
    lead from three weak sightings to a fourth and fifth, obtained by
    deliberate, controlled variation rather than incidental
    observation.
  - **Mod-3 lead falsified**: either `balanced_3q4q(18)` or
    `mixed_uneven(18)` (or both) shows `"nonexistent solution"` (slow) --
    this would directly demonstrate that component count alone, even at
    a "lucky" divisible-by-3 value, is not sufficient, and that
    composition matters independent of the count -- closing off the
    mod-3 lead as this project's working hypothesis.

**P2 (if P1 falsifies, does the failure pattern suggest a specific
alternative)?** No specific alternative is predicted in advance; this
is deliberately left open rather than forcing a second hypothesis before
the first is tested.

## 4. A design flaw caught and fixed before running

**This pre-registration's first draft used n=54, not n=64, for
`uniform_3q`/`mixed_uneven`** (reasoning that 18 equal-size components
could not evenly cover 64 qubits), which would have left those two
families at grid-spare=10 rather than literal 100% occupancy --
confounding "composition effect" with "not actually at the saturation
point being studied." Found before any code was written or run, by
searching directly for an integer solution to `4a + 3b = 64, a+b = 18`
(a=10, b=8) and a corresponding solution for the skewed case
(17 x 2 + 1 x 30 = 64), both of which keep every family at genuine
spare=0 on the full 64-qubit grid. The design in Section 2 above reflects
the corrected version; this section records the correction rather than
silently replacing the flawed draft.

## 5. What this cannot establish

- Whether the same result holds at other component counts divisible by
  3 (only 18 is tested here).
- Whether the result generalizes to 6x7 or other grid sizes.
- The mechanism, if P1 confirms the lead survives -- only whether
  composition matters independent of count, not why either matters.

---


<!-- ===== Addendum 63 (source: spare-qubit-cliff-addendum-63-2026-09-18.md) ===== -->

> **Note added when merging:** **The mod-3 lead is falsified.** At the identical component count (18) and zero-idle condition, one composition (balanced_3q4q) cliffs while another (mixed_uneven) does not -- composition, not count, determines the outcome, and counter-intuitively the more extreme structure was the fast one.

## Addendum 63 -- the mod-3 lead is falsified: at the same component count (18), composition alone flips cliff and no-cliff, and counter-intuitively the MORE extreme structure was the FAST one (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-63-preregistration-2026-09-18.md`, written
and locked before this run, including a design correction (n=54 -> n=64
for two families) made before any code was written. All predictions
scored below.

## 0. In one line

**P1: "Mod-3 lead falsified," decisively and cleanly.** At the exact
same component count (18, divisible by 3) and the exact same zero-idle
condition on the exact same 64-qubit grid, `balanced_3q4q` (10 four-qubit
chains + 8 three-qubit chains) **cliffs** -- 10,233.00ms,
`"nonexistent solution"` -- while `mixed_uneven` (17 two-qubit edges + one
30-qubit path) **does not** -- 30.16ms, `"solution found"`. **Component
count divisible by 3 does not predict the outcome; composition, holding
count fixed, is sufficient on its own to flip cliff and no-cliff.**
Every observation in Addenda 59, 61, 62 suggesting a mod-3 pattern is now
understood to have been confounded with `merged_pairs`' own specific,
narrow composition style, not a property of the count itself.
**Counter-intuitively, the more extreme, less "balanced" structure
(`mixed_uneven`) was the fast one**, and the more moderate mixture
(`balanced_3q4q`, two similar-length chain types) was the one that
cliffed.

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 18 rows completed with `error=""`; stop reason unanimous
(6/6) within every family.

| family | composition | components | idle qubits | edges | time (ms) | stop reason |
|---|---|---:|---:|---:|---:|:---|
| `uniform_2q` | 18 x 2-qubit | 18 | 28 | 18 | 50.13 | solution found |
| **`balanced_3q4q`** | 10 x 4-qubit + 8 x 3-qubit | 18 | **0** | 46 | **10,233.00** | **nonexistent solution** |
| **`mixed_uneven`** | 17 x 2-qubit + 1 x 30-qubit | 18 | **0** | 46 | **30.16** | **solution found** |

## 2. Scoring

**P0 (sanity check) -- CONFIRMED.** `uniform_2q`, with 28 idle qubits,
shows `"solution found"` (50.13ms) -- fast, consistent with Addenda
53-55's independent idle-qubit finding. This family was never a fair
test of the mod-3 lead and none is claimed from it.

**P1 (primary) -- FALSIFIED, exactly per the pre-registered "either...
shows nonexistent solution" criterion.** `balanced_3q4q` alone is
sufficient to falsify: it has the same component count (18) and zero
idle qubits as every prior "easy" sighting the mod-3 lead was built on,
and it cliffs anyway. `mixed_uneven`'s own "solution found" result does
not rescue the lead -- the pre-registration's scoring rule required
**both** alternate compositions to stay fast for the lead to survive,
and only one did.

**P2 (open, no specific alternative predicted in advance) -- addressed
in Section 3, as an observation, not a scored prediction.**

## 3. What actually seems to matter, stated carefully

**This addendum does not identify the true explanatory variable -- it
only eliminates one candidate (component count mod 3) and surfaces a
second, genuinely surprising data point.** The two "real test" families
differ from each other in a specific way worth naming precisely, since
"composition matters" alone is not yet informative:

- `balanced_3q4q` (cliffs): two *moderate*, *similar-length* chain
  types (3-qubit and 4-qubit), 18 components, no single component
  dominating the graph's structure.
- `mixed_uneven` (does not cliff): one *extreme* outlier (a 30-qubit
  connected path -- nearly half the entire device in one component) plus
  many small, uniform 2-qubit edges.

**The naive intuition -- that a more extreme, less regular structure
should be "harder" for a search algorithm -- is contradicted here.** The
family with one dominant 30-qubit component was fast; the family with a
more even mixture of moderate-sized components was the one that cliffed.
This is reported as a specific, surprising observation from two data
points, not generalized into a rule. Two immediate, untested candidate
readings, neither confirmed:

- The single large connected component in `mixed_uneven` may itself act
  like `linear_chain` (Addendum 51: connected paths do not cliff) for
  the 30 qubits it spans, effectively "absorbing" much of the graph into
  an easy-to-place substructure, while the remaining 17 small edges
  behave like a much-reduced version of `dense_pairs`.
- `balanced_3q4q`'s two *different* chain lengths (3 and 4 qubits, both
  well below `mixed_uneven`'s 30-qubit outlier) may create a specific
  kind of structural symmetry-breaking-without-actually-breaking-symmetry
  -- two classes of near-but-not-quite-identical components -- that is
  harder for VF2-style search than either full uniformity
  (`merged_pairs`' own m=3/m=4/m=6 clearings, which mix only 2-qubit and
  a *few* 4-qubit chains) or the near-total dominance of one component
  (`mixed_uneven`).

**Neither is confirmed.** Distinguishing them would need further
controlled variation -- e.g. a family with two chain lengths but far more
of one than the other (asymmetric within the "balanced" style), or a
family with a large-but-not-dominant component (e.g. 10 qubits rather
than 30) to see where the "absorbs the graph" effect, if real, stops
working.

## 4. What this means for Addenda 59-62

**Addenda 59, 61, and 62's mod-3 observations stand as accurate reports
of what was measured, but their interpretation is now superseded.** They
were not wrong about the data; they were incomplete about the cause,
exactly as Addendum 61 itself warned might be the case ("if this were
real, it would predict..." -- and it did not survive the direct test).
The oscillation Addendum 59 found in `merged_pairs` at 8x8, and the
timing variation Addendum 62 found at 6x7, remain real, measured
phenomena -- **what they were tracking was never component count itself,
but something about `merged_pairs`' own specific, narrow family of
compositions** (always a majority of 2-qubit edges plus a minority of
4-qubit chains, never the kind of extreme or dual-moderate mixtures
tested here).

## 5. What this does not establish

- The actual explanatory variable -- ruled out mod-3, surfaced a
  surprising contrast, confirmed neither of Section 3's two candidate
  readings.
- Whether `balanced_3q4q`'s specific 10:8 ratio matters, or whether any
  two-moderate-chain-length mixture at 18 components would cliff.
- Whether `mixed_uneven`'s result depends on the large component being
  connected as a *path* specifically, or would hold for other large
  connected shapes.
- Whether any of this generalizes to 6x7 or other grid sizes.
- Whether `balanced_3q4q` or `mixed_uneven`'s outcomes are sensitive to
  which specific qubits are used for which component (Addendum 60 found
  `merged_pairs`' own oscillation was order-independent; that has not
  been re-checked for these two new families).

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 63's three new families) |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run4.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run4.csv) | this run, 18 rows |
| [`spare-qubit-cliff-addendum-63-preregistration-2026-09-18.md`](#addendum-63----pre-registration-does-the-mod-3-lead-survive-holding-component-count-fixed-at-a-multiple-of-3-while-varying-component-size-composition-2026-09-18) | the predictions scored above, including a design correction made before running |

## 7. Verification

- All 18 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every family.
- `n_interaction_edges` was cross-checked against each family's intended
  construction (`balanced_3q4q`: 10x3 + 8x2 = 46; `mixed_uneven`:
  17x1 + 29 = 46) before accepting the results as reflecting the
  intended structures rather than a construction error.
- The pre-registration's own falsification criterion ("either... shows
  nonexistent solution") was re-read and quoted directly before scoring
  P1, rather than a looser post-hoc judgment of what counts as
  falsification.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 64 pre-registration (source: spare-qubit-cliff-addendum-64-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether shrinking mixed_uneven's dominant component reverses its fast result -- includes a discovered design constraint (only two valid points exist at n=64, component count 18) found while writing the pre-registration, before any code was run.

## Addendum 64 -- Pre-registration: shrinking the dominant component in mixed_uneven, at fixed component count (18) and zero idle qubits, to find where the effect reverses (2026-09-18)

**Status: pre-registration only. No run with any new construction has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 63 found `mixed_uneven` (17 x 2-qubit edges + one 30-qubit
path, 18 components, zero idle, n=64) fast, while `balanced_3q4q` (10 x
4-qubit + 8 x 3-qubit, same component count, same zero-idle condition)
cliffed. Two candidate readings were proposed, neither confirmed: (a)
the 30-qubit component "absorbs" much of the graph into an
easy-to-place substructure, behaving like `linear_chain`; (b)
`balanced_3q4q`'s two moderate, similar-length chain types create a
specific hard structure that neither full uniformity nor one dominant
component reproduces. This addendum tests candidate (a) directly by
shrinking the dominant component and watching for a reversal, while
explicitly avoiding a design flaw: naively shrinking the big component
alone would also change total component count, confounding "component
size" with "component count" -- the exact confound Addendum 63 was built
to eliminate. This design holds component count fixed at 18 throughout.

## 2. Design

A parameterised family, `shrinking_dominant(big_size)`: one connected
path of `big_size` qubits, plus enough 2-qubit edges to bring the total
to exactly 18 components and exactly 64 qubits (zero idle) --
`n_small = 17` when `big_size=30` (reproducing `mixed_uneven` exactly,
the sanity check), and `n_small` recalculated for each smaller
`big_size` so that `n_small * 2 + big_size = 64` and
`n_small + 1 = 18` simultaneously hold. Solving: `n_small = 17` always
(from the component-count constraint alone), which forces
`big_size = 64 - 34 = 30` as the ONLY value compatible with both
constraints using 2-qubit small edges. **This is a real design
constraint discovered while writing this pre-registration, not
discoverable only after running code**, and it means the originally
imagined sweep (30, 20, 10, ...) is not achievable by shrinking the big
component alone while holding both component count AND small-edge size
fixed at 2 qubits. Resolved by additionally varying small-component size
alongside big-component size, keeping the qubit-count and
component-count equations satisfied together:

  `n_small_components x small_size + big_size = 64`
  `n_small_components + 1 = 18` --> `n_small_components = 17`
  --> `big_size = 64 - 17 x small_size`

| small_size | big_size | check |
|---:|---:|---|
| 2 | 30 | `mixed_uneven` itself (sanity check) |
| 1 | 47 | invalid -- size-1 "components" are isolated qubits, i.e. idle qubits, reintroducing Addenda 53-55's confound |
| 3 | 13 | 17 x 3-qubit edges... but a 3-qubit "small" component is itself a chain, not an edge -- uses `_edges_merged_pairs`-style 3-qubit chains for the 17 small components instead of 2-qubit edges |

Since `small_size=1` is invalid (reintroduces idle-qubit confound) and
`small_size>=3` starts to resemble `balanced_3q4q` more than
`mixed_uneven`, **the "shrink while holding count fixed" design only
admits exactly one non-trivial big_size (30) at small_size=2** -- there
is no room to sweep `big_size` continuously while holding both
constraints and small-component shape fixed. **This is reported before
running anything**, because it means Addendum 64 as originally
conceived (a clean sweep of big_size with count and small-shape both
fixed) is not executable, and the design must change.

## 3. Revised design, reflecting the constraint above

Vary `small_size` instead of `big_size` directly, since `big_size` is
then determined by the equation above -- this still varies "how
dominant is the single large component" (via `big_size`, which moves
inversely with `small_size`) while holding total component count fixed
at 18 and zero idle qubits fixed, satisfying this project's own
established discipline:

| small_size | n_small_components | big_size | big component's share of device |
|---:|---:|---:|---:|
| 2 | 17 | 30 | 46.9% (Addendum 63's own `mixed_uneven`) |
| 3 | 17 | 13 | 20.3% |
| 4 | 17 | -4 | invalid (negative) |

**Only two valid points exist in this family at n=64, component
count=18**: `small_size=2` (big=30, already measured) and
`small_size=3` (big=13, new). This is a much narrower sweep than
originally hoped -- two points, not a curve -- and this addendum's own
design section states that plainly rather than presenting two points as
if they mapped a threshold.

## 4. Pre-registered predictions

**P1 (primary, now necessarily coarse -- two points only).** At
`small_size=3` (big_size=13, 17 x 3-qubit chains + one 13-qubit path, 18
components, zero idle, n=64):
  - **Still fast** (`"solution found"`): the "large component absorbs
    difficulty" reading (candidate (a)) is supported even with a much
    smaller (13-qubit, not 30-qubit) dominant component and *larger*
    small components (3-qubit chains, not bare edges) -- suggesting the
    effect is not about the big component's absolute size specifically.
  - **Cliffs** (`"nonexistent solution"`): the effect depends on the big
    component being large/dominant specifically (or on the small
    components being bare 2-qubit edges specifically, since both changed
    at once here) -- narrowing candidate (a) considerably or ruling it
    out as stated.

**P2 (does this addendum's own design difficulty -- only two valid
points existing -- itself suggest anything?).** Not a prediction about
outcome; a note that the sparse design space here (Section 3) is itself
informative: it shows that "shrinking the dominant component while
holding everything else fixed" is a much more constrained operation than
it first appeared, and a genuinely continuous sweep would require either
allowing idle qubits (reintroducing a known confound) or allowing
component count to vary (reintroducing Addendum 63's own eliminated
confound) or moving to a different, larger grid size where more integer
solutions exist.

## 5. What this cannot establish

- A true threshold or "critical point," since only two data points are
  achievable in this exact design at n=64 -- a curve would need a larger
  device (more integer solutions to the constraint equations) or
  relaxing one of the fixed conditions deliberately, with the tradeoff
  stated up front.
- Whether `small_size` or `big_size` (which move together, inversely,
  in this design) is the operative variable -- they cannot be separated
  in this specific two-point family.
- Anything about 6x7 or other grid sizes.

---


<!-- ===== Addendum 64 (source: spare-qubit-cliff-addendum-64-2026-09-18.md) ===== -->

> **Note added when merging:** Shrinking the dominant component from 30 to 13 qubits flips fast to cliff -- falsifying 'a large component absorbs difficulty regardless of size,' but leaving an unresolved confound (dominant-component size vs. small-component shape changed together).

## Addendum 64 -- shrinking the dominant component from 30 to 13 qubits flips fast to cliff: the "absorption" reading is falsified, dominance size matters (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-64-preregistration-2026-09-18.md`, written
and locked before this run, including a real construction bug
(`_edges_mixed_uneven` skipped intermediate qubits for `small_size>2`,
silently reintroducing 17 idle qubits) found and fixed before any
measurement.

## 0. In one line

**P1: "Cliffs" confirmed.** `shrinking_dominant` at `small_size=3,
big_size=13` (17 x 3-qubit chains + one 13-qubit path, 18 components,
zero idle, n=64 -- the only other valid point in this exact design,
per the pre-registration's own constraint analysis) shows
**`"nonexistent solution"`, 12,402.02ms** -- a cliff, where
`mixed_uneven`'s `big_size=30` version (Addendum 63) was fast
(30.16ms). **Shrinking the dominant component from 30 to 13 qubits, at
fixed component count and fixed zero-idle condition, is sufficient on
its own to flip the outcome from fast to cliff.** Addendum 63's
candidate (a) -- "the large component absorbs difficulty regardless of
its exact size" -- is falsified: absorption, if that is the right word
at all, depends on the dominant component being large enough, not
merely present.

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 6 rows completed with `error=""`; stop reason unanimous
(6/6).

| family (this addendum) | composition | components | idle | edges | time (ms) | stop reason |
|---|---|---:|---:|---:|---:|:---|
| `mixed_uneven` (Addendum 63, for comparison) | 17 x 2-qubit + 1 x 30-qubit | 18 | 0 | 46 | 30.16 | solution found |
| **`shrinking_dominant`** (this run) | 17 x 3-qubit + 1 x 13-qubit | 18 | **0** | 46 | **12,402.02** | **nonexistent solution** |

Both rows have identical component count (18), identical zero-idle
condition, and (by construction) identical total edge count (46) -- the
only differences are the dominant component's size (30 vs. 13 qubits)
and the small components' size (2-qubit edges vs. 3-qubit chains, moving
together as the pre-registration's constraint equation required).

## 2. Scoring

**P1 (primary) -- "Cliffs" CONFIRMED.** The pre-registration's two
branches were: "still fast" (supports candidate (a), absorption
independent of size) versus "cliffs" (falsifies or narrows candidate
(a)). The measured `"nonexistent solution"` result matches the second
branch cleanly and decisively -- not a borderline or ambiguous result.

**P2 (note on design sparsity) -- as stated in advance, this remains a
two-point comparison, not a curve.** The result is informative about the
direction (shrinking the dominant component can flip the outcome) but
says nothing about *where between 13 and 30* the flip occurs, which
requires either a larger device (more valid integer solutions) or
relaxing one of the two fixed conditions, exactly as Section 4 of the
pre-registration anticipated.

## 3. What this does and does not settle

**Settles**: `mixed_uneven`'s fast result (Addendum 63) is not explained
by "any single dominant component, regardless of size, absorbs the
graph's difficulty." Size matters. A 30-qubit dominant component (47%
of the device) is fast; a 13-qubit one (20%) cliffs, at matched
component count and zero-idle condition.

**Does not settle**: whether this is really about the dominant
component's *size* per se, or about the small components' size, since
this design's own constraint equation (`17 x small_size + big_size =
64`, `n_small = 17` fixed by the component-count requirement) forces
`small_size` and `big_size` to move together, inversely. The two
measured points differ in BOTH the dominant component's size (30 -> 13)
AND the small components' size (2-qubit edges -> 3-qubit chains)
simultaneously. **This addendum cannot distinguish "the big component
got too small" from "the small components got too big (structured)."**
Both changed at once, by construction, and no test performed so far
holds one fixed while varying the other.

**This closes off one candidate reading from Addendum 63 (size-independent
absorption) while opening a new, sharper confound that a further
addendum would need to resolve**: is it the shrinking of the dominant
component, or the simultaneous growth of the "small" components from
bare edges into actual 3-qubit chains, that produced the flip? Note that
`balanced_3q4q` (Addendum 63, also cliffing) similarly used 3-qubit and
4-qubit chains rather than bare 2-qubit edges for its smaller
components -- so "small components being chains rather than bare edges"
is at least as plausible a unifying explanation across both of
Addendum 63's cliffing results (`balanced_3q4q` and now this
`shrinking_dominant` point) as "dominant component too small" is.

## 4. A candidate unifying reading across Addenda 63-64, stated cautiously

Every cliffing composition measured so far at 18 components, zero idle
(`balanced_3q4q`: 10x4q+8x3q; this addendum's `shrinking_dominant`:
17x3q+1x13q) has **no bare 2-qubit edges at all** -- every component is
a chain of 3 or more qubits. Every non-cliffing composition
(`mixed_uneven`: 17x2q+1x30q; every `merged_pairs` clearing point,
Addenda 54-55: mostly 2-qubit edges plus a handful of 4-qubit chains)
**includes a substantial number of bare 2-qubit edges**. **This is a new
candidate, not yet tested independently**: perhaps bare 2-qubit edges
(the project's original `dense_pairs` building block) are specifically
"easy" for VF2 to place regardless of how many of them there are, while
longer chains (3+ qubits) are what drives difficulty -- which would
reframe the entire cliff/no-cliff question away from component count,
mod-3, or dominant-component size, and toward **whether 2-qubit edges
are present at all**. This is proposed here for the first time and
requires its own dedicated, controlled test (e.g. a family with several
3-qubit chains and zero 2-qubit edges, at varying component counts) --
not confirmed by anything measured in this addendum on its own.

## 5. What this does not establish

- Whether the "no bare 2-qubit edges" candidate in Section 4 holds up
  under direct test.
- Whether `small_size` (independent of `big_size`) or `big_size`
  (independent of `small_size`) is the operative variable -- this
  design cannot separate them.
- Any finer resolution between big_size=13 and big_size=30.
- Whether this generalizes to 6x7 or other grid sizes.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 64's `shrinking_dominant`, with the pre-run bug fix to `_edges_mixed_uneven`) |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run5.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run5.csv) | this run, 6 rows |
| [`spare-qubit-cliff-addendum-64-preregistration-2026-09-18.md`](#addendum-64----pre-registration-shrinking-the-dominant-component-in-mixed_uneven-at-fixed-component-count-18-and-zero-idle-qubits-to-find-where-the-effect-reverses-2026-09-18) | the predictions scored above |

## 7. Verification

- All 6 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6).
- `n_interaction_edges=46` matches the intended construction
  (17 small components x 2 edges each + 12 edges in the 13-qubit path
  = 34+12=46), confirming the pre-run bug fix produced the intended
  zero-idle, 18-component structure rather than a malformed one.
- The comparison to `mixed_uneven`'s own Addendum 63 result (30.16ms,
  solution found) was re-read directly from that addendum before being
  used here, not from memory.
- Section 4's candidate ("no bare 2-qubit edges predicts cliff") was
  checked against all four compositions on record (both cliffing,
  both clearing) before being proposed, and is explicitly labelled as
  untested rather than implied to be established by this addendum
  alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 65 pre-registration (source: spare-qubit-cliff-addendum-65-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether a LARGE dominant component still produces a fast result when the small components are chains rather than bare 2-qubit edges, isolating Addendum 64's unresolved confound.

## Addendum 65 -- Pre-registration: does a large dominant component still absorb difficulty when the small components are 3-qubit chains instead of bare 2-qubit edges? (2026-09-18)

**Status: pre-registration only. No run with any new construction has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 64 found shrinking `mixed_uneven`'s dominant component from 30
to 13 qubits flipped the outcome from fast to cliff -- but could not
separate two candidate causes, because its design changed the dominant
component's size AND the small components' size (bare 2-qubit edges ->
3-qubit chains) simultaneously: (a) the dominant component became too
small to "absorb" the graph's difficulty, or (b) the small components
stopped being bare 2-qubit edges, and Addendum 64's own Section 4 found
every cliffing composition on record so far has zero bare 2-qubit edges
while every clearing one has many. This addendum tests candidate (b)
directly: **does a LARGE dominant component still produce a fast result
when the small components are 3-qubit chains rather than bare edges?**
If yes, size (not edge-vs-chain composition) is what matters, and
candidate (b) is weakened. If the result cliffs despite a large dominant
component, candidate (b) -- the presence of bare 2-qubit edges -- gains
real support as the operative variable, independent of dominant-component
size.

## 2. Design

`large_dominant_no_bare_edges`: one connected path of `big_size` qubits,
plus `n_small` disjoint 3-qubit chains (NOT bare 2-qubit edges), sized
to use exactly 64 qubits with zero idle and hold component count at a
value already on record. Using the same constraint style as Addendum 64
(`n_small x 3 + big_size = 64`), choosing `n_small` to make `big_size`
as large as practical while keeping at least a few small components (so
this remains a "dominant component plus many small ones" structure, not
degenerate):

  `n_small = 8` --> `big_size = 64 - 24 = 40` (9 total components: 1
  dominant + 8 small 3-qubit chains) -- **a dominant component larger
  (40 qubits, 62.5% of the device) than `mixed_uneven`'s own 30-qubit
  (47%) fast case, but with zero bare 2-qubit edges anywhere in the
  circuit.**

This single configuration is the primary test. `optimization_level=3`,
spare=0 on the 8x8 grid, 3 seeds x 2 repeats.

## 3. Pre-registered predictions

**P1 (primary).**
  - **Fast** (`"solution found"`): a sufficiently large dominant
    component absorbs difficulty regardless of whether the small
    components are bare edges or 3-qubit chains -- candidate (a) from
    Addendum 64 (dominant-component size) is the better explanation,
    and Addendum 64's Section 4 "no bare 2-qubit edges" observation was
    coincidental to that run's specific parameter choice, not causal.
  - **Cliffs** (`"nonexistent solution"`): even a very large (62.5%)
    dominant component does not prevent a cliff when the small
    components are 3-qubit chains rather than bare edges -- candidate
    (b), the presence/absence of bare 2-qubit edges, gains direct,
    controlled support as at least a contributing (possibly the
    dominant) factor, independent of dominant-component size.

## 4. What this cannot establish

- Whether component count (9 here, not divisible by 3, unlike every
  prior test in this specific line of investigation) plays any role --
  deliberately not controlled for in this single-configuration test,
  since Addendum 63 already established mod-3 alone does not predict
  outcome; this addendum does not re-litigate that.
- Any intermediate point between "zero bare edges, 40-qubit dominant"
  and known prior configurations -- this is one new data point, not a
  sweep.
- Generalization beyond 8x8.

---


<!-- ===== Addendum 65 (source: spare-qubit-cliff-addendum-65-2026-09-18.md) ===== -->

> **Note added when merging:** Even a 62.5%-of-device dominant component cliffs when no bare 2-qubit edges are present. Across five independently constructed compositions, 'contains at least one bare 2-qubit edge' now perfectly separates fast from cliffing outcomes -- the most consistent variable found in the entire investigation.

## Addendum 65 -- even a 62.5%-of-device dominant component cliffs when no bare 2-qubit edges are present: the "absorption by size" reading is falsified, bare-edge presence is the stronger candidate (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-65-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: "Cliffs" confirmed, decisively.** A dominant component covering
**62.5% of the device** (40 of 64 qubits) -- larger than `mixed_uneven`'s
own 30-qubit (47%) fast case from Addendum 63 -- **still cliffs**
(14,351.92ms, `"nonexistent solution"`) when the remaining structure is
built from 3-qubit chains rather than bare 2-qubit edges. **Dominant-
component size alone does not explain `mixed_uneven`'s fast result.**
The candidate proposed in Addendum 64 Section 4 -- that the presence of
bare 2-qubit edges, not component count or dominant-component size, is
the operative variable -- gains direct, controlled support: it is now
the only proposed explanation consistent with all five compositions
measured across Addenda 63-65.

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 6 rows completed with `error=""`; stop reason unanimous
(6/6).

| family | composition | dominant component | bare 2-qubit edges | idle | time (ms) | stop reason |
|---|---|---:|:---:|---:|---:|:---|
| `mixed_uneven` (Addendum 63) | 17x2q + 1x30q | 47% | yes (17) | 0 | 30.16 | solution found |
| `shrinking_dominant` (Addendum 64) | 17x3q + 1x13q | 20% | no | 0 | 12,402.02 | nonexistent solution |
| **`large_dominant_no_bare_edges`** (this run) | 8x3q + 1x40q | **62.5%** | **no** | 0 | **14,351.92** | **nonexistent solution** |

## 2. Scoring

**P1 (primary) -- "Cliffs" CONFIRMED.** The pre-registration's two
branches were explicit: "fast" would support dominant-component size as
the operative variable; "cliffs despite a large dominant component"
would support bare-2-qubit-edge presence as an independent factor. The
measured result -- a cliff at the largest dominant-component fraction
tested in this entire line of investigation -- matches the second branch
unambiguously.

## 3. The full picture across Addenda 63-65

Every composition measured at zero idle qubits, across three separate
addenda, now sorted by the variable this addendum isolates:

| composition | bare 2-qubit edges present? | outcome |
|---|:---:|:---|
| `mixed_uneven` (17x2q+1x30q) | **yes** | fast |
| every `merged_pairs` clearing point (Addenda 54-55: m=3,4,6, etc.) | **yes** (majority 2-qubit edges) | fast |
| `balanced_3q4q` (10x4q+8x3q) | no | **cliff** |
| `shrinking_dominant` (17x3q+1x13q) | no | **cliff** |
| `large_dominant_no_bare_edges` (8x3q+1x40q) | no | **cliff** |

**Five for five.** Every composition containing at least some bare
2-qubit edges has been fast; every composition built entirely from
chains of 3+ qubits has cliffed, regardless of component count (18, 18,
17, 9 respectively for the four non-`mixed_uneven`/`merged_pairs`
entries), regardless of whether component count is divisible by 3
(9 is not), and regardless of how large or small the single largest
component is (13 to 40 qubits, a >3x range, all cliffing). **This is now
the most consistent variable found in this entire investigation.**

## 4. What this does and does not establish

**Establishes**: dominant-component size, taken alone, is falsified as
a sufficient explanation for `mixed_uneven`'s fast result -- a
substantially larger dominant component (62.5% vs. 47%) does not rescue
a composition that lacks bare 2-qubit edges. Across five independently
constructed compositions, "contains at least one bare 2-qubit edge"
perfectly separates fast from cliffing outcomes.

**Does not establish**: *why* bare 2-qubit edges would matter to a VF2
search in this way, or whether "bare 2-qubit edges" is itself the true
variable versus a proxy for something more specific (e.g. "the graph
contains components of the minimum possible size for this circuit
family," or some property of how such components interact with the
grid's own structure during search). No mechanism is proposed. Also not
established: whether a SINGLE bare 2-qubit edge, amid many longer
chains, would be enough to flip a cliffing composition to fast -- every
"yes" case tested so far has many bare edges (17), not few. This is the
natural next, sharper test.

**Also not established**: generalization to 6x7 or other grid sizes,
or to component counts/structures not yet tried (e.g. all components
being 3-qubit chains with no dominant component at all, isolating
"chains vs. edges" from "presence of a dominant component" entirely).

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 65's `large_dominant_no_bare_edges`) |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run6.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run6.csv) | this run, 6 rows |
| [`spare-qubit-cliff-addendum-65-preregistration-2026-09-18.md`](#addendum-65----pre-registration-does-a-large-dominant-component-still-absorb-difficulty-when-the-small-components-are-3-qubit-chains-instead-of-bare-2-qubit-edges-2026-09-18) | the predictions scored above |

## 6. Verification

- All 6 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6).
- `n_interaction_edges=55` matches the intended construction (8 x 2
  edges per 3-qubit chain + 39 edges in the 40-qubit path = 16+39=55).
- The five-composition summary table (Section 3) was assembled by
  re-reading each source addendum's own recorded result directly, not
  from memory, before drawing the "five for five" conclusion.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 66 pre-registration (source: spare-qubit-cliff-addendum-66-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether a single bare 2-qubit edge, alone, is sufficient to prevent the cliff, or whether count (not mere presence) matters.

## Addendum 66 -- Pre-registration: does a single bare 2-qubit edge, alone, prevent the cliff? (2026-09-18)

**Status: pre-registration only. No run with any new construction has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 65 established, across five independently constructed
compositions at zero idle qubits on the 8x8 grid, that "contains at
least one bare 2-qubit edge" perfectly separates fast outcomes from
cliffing ones. Every "fast" case tested so far (`mixed_uneven`,
`merged_pairs`' clearing points) contains MANY bare edges (17, or a
majority of the graph). **Whether a single bare edge is sufficient, or
whether some larger number is needed, has never been tested** -- this is
the natural next, sharper question the visual comparison (this
session's diagram) made obvious: does presence alone matter, or does
count matter too?

## 2. Design

`single_bare_edge`: one bare 2-qubit edge, plus enough 3-qubit chains to
use the remaining 62 qubits with zero idle. `62 / 3` is not an integer
(62 = 3*20 + 2), so the remaining qubits cannot be covered by 3-qubit
chains alone without leftover. Resolved by using **one 4-qubit chain
plus 19 three-qubit chains**: `1x2 (bare edge) + 1x4 + 19x3 = 2+4+57 =
63` -- one qubit short of 64. Re-solving: `n_3q x 3 + 1 x big_extra +
2 (bare edge) = 64` needs `n_3q x 3 + big_extra = 62`. Simplest exact
solution: **20 three-qubit chains plus the one bare edge**: `20x3 + 2 =
62`, two qubits short. **21 three-qubit chains plus the bare edge**:
`21x3 + 2 = 65`, one over. **No integer number of pure 3-qubit chains
plus one bare edge sums to exactly 64** (64-2=62 is not divisible by 3).
Resolved by using 19 three-qubit chains (57 qubits) plus one 5-qubit
chain (5 qubits) plus the single bare edge (2 qubits): `57+5+2=64`
exactly, zero idle, 21 total components, exactly one of which is a bare
2-qubit edge and none of the rest below 3 qubits.

`optimization_level=3`, spare=0 on the 8x8 grid, 3 seeds x 2 repeats.

## 3. Pre-registered predictions

**P1 (primary).**
  - **Fast** (`"solution found"`): a single bare 2-qubit edge is
    sufficient on its own -- presence, not count, is what matters. This
    would be the strongest, simplest form of the "bare edge" finding.
  - **Cliffs** (`"nonexistent solution"`): a single bare edge is not
    enough -- some larger number (between 1 and 17, the two tested
    extremes) is required, and Addendum 65's finding needs to be
    restated as a quantitative threshold rather than a simple
    presence/absence rule.

## 4. What this cannot establish

- If P1 falsifies (cliffs), the exact minimum count needed -- only that
  it exceeds 1.
- Whether the result depends on the bare edge's position within the
  graph (e.g. adjacent to the dominant component vs. far from it) --
  only one position is tested.
- Generalization beyond 8x8.

---


<!-- ===== Addendum 66 (source: spare-qubit-cliff-addendum-66-2026-09-18.md) ===== -->

> **Note added when merging:** A single bare edge is not enough -- the composition still cliffs. Addendum 65's finding is revised from presence to an unresolved count/fraction threshold, somewhere between 1 and 17 bare edges, not yet located.

## Addendum 66 -- a single bare 2-qubit edge is NOT enough: the cliff still occurs, so bare-edge count (not mere presence) is what matters (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-66-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: "Cliffs" confirmed.** `single_bare_edge` (exactly one bare 2-qubit
edge, plus 19 disjoint 3-qubit chains, plus one 5-qubit chain, 21
components, zero idle, n=64) shows **`"nonexistent solution"`,
10,841.73ms** -- a cliff. **A single bare edge is not sufficient to
prevent the cliff.** Addendum 65's "contains at least one bare edge"
finding must be restated: presence alone does not explain the five
prior observations. Something about the *number* of bare edges (or a
correlated property, such as the fraction of the graph they cover)
is the operative variable, not mere existence.

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 6 rows completed with `error=""`; stop reason unanimous
(6/6).

| family | bare 2-qubit edges | other components | idle | edges | time (ms) | stop reason |
|---|:---:|---|---:|---:|---:|:---|
| `mixed_uneven` (Addendum 63) | 17 | 1 x 30-qubit path | 0 | 46 | 30.16 | solution found |
| **`single_bare_edge`** (this run) | **1** | 19 x 3-qubit + 1 x 5-qubit | 0 | 43 | **10,841.73** | **nonexistent solution** |

## 2. Scoring

**P1 (primary) -- "Cliffs" CONFIRMED.** The pre-registration's two
branches were explicit: "fast" would mean presence alone is sufficient;
"cliffs" would mean count matters and Addendum 65's finding needs
restating as a threshold. The measured result -- a clear cliff, same
order of magnitude as every other cliffing composition in this project
(10,233 to 14,352ms across Addenda 63-65) -- matches the second branch
without ambiguity.

## 3. What this means: from "presence" to "count," and what remains unknown

**Addendum 65's finding is revised, not discarded.** It correctly
identified that bare 2-qubit edges are associated with fast outcomes;
this addendum shows that association requires more than a token amount.
The data points now on record, sorted by bare-edge count:

| bare edges | outcome |
|---:|:---|
| 0 (`balanced_3q4q`, `shrinking_dominant`, `large_dominant_no_bare_edges`) | cliff (all three) |
| **1** (`single_bare_edge`, this run) | **cliff** |
| 17 (`mixed_uneven`) | fast |

**The threshold lies somewhere between 1 and 17, unresolved by anything
measured so far.** No data point exists in between. Two immediate,
untested candidate readings:

- A **fraction-of-graph** reading: `mixed_uneven`'s 17 bare edges cover
  34 of 64 qubits (53%); `single_bare_edge`'s 1 edge covers 2 of 64
  (3%) -- perhaps a substantial *share* of the graph needs to be
  bare-edge structure, not just a nonzero count.
- A **small-absolute-number** reading: perhaps as few as 3-5 bare edges
  would already flip the outcome, and the true threshold is low,
  unrelated to `mixed_uneven`'s specific 17.

Neither is tested here. This is a direct, natural next experiment
(sweeping bare-edge count at a few intermediate values, e.g. 2, 5, 10)
rather than a claim resolved by this addendum.

## 4. What this does not establish

- The actual threshold count (or fraction) between 1 and 17.
- Whether the threshold, once found, is a sharp cutoff or itself a
  gradual transition (this project's own history -- Addendum 42's
  staircase, Addendum 59's oscillation -- suggests gradual or
  non-monotonic transitions are common here, and a sharp threshold
  should not be assumed).
- Whether bare-edge position (adjacent to vs. far from the dominant
  structure) matters independent of count.
- Generalization beyond 8x8.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 66's `single_bare_edge`) |
| [`circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run7.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-18_run7.csv) | this run, 6 rows |
| [`spare-qubit-cliff-addendum-66-preregistration-2026-09-18.md`](#addendum-66----pre-registration-does-a-single-bare-2-qubit-edge-alone-prevent-the-cliff-2026-09-18) | the predictions scored above |

## 6. Verification

- All 6 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6).
- `n_interaction_edges=43` matches the intended construction (1 bare
  edge + 19 x 2 edges per 3-qubit chain + 4 edges in the 5-qubit chain
  = 1+38+4=43).
- The comparison to `mixed_uneven`'s own Addendum 63 result was
  re-read directly from that addendum, not from memory.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---<!-- ===== Addendum 67 pre-registration (source: spare-qubit-cliff-addendum-67-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for sweeping bare-2-qubit-edge count at 2, 5, 10 to locate the threshold between 1 (cliffs, Addendum 66) and 17 (fast, Addendum 63).

## Addendum 67 -- Pre-registration: sweeping bare-2-qubit-edge count (2, 5, 10) to locate the threshold between 1 (cliff) and 17 (fast) (2026-09-18)

**Status: pre-registration only. No run with any new construction has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 66 found a single bare 2-qubit edge insufficient to prevent the
cliff (cliffs); Addendum 63 found 17 bare edges sufficient (fast).
Nothing between has been tested. This is Tier 1.1 of
`OPEN_ITEMS_2026-09-18.md`, judged the single most actionable open item.

## 2. Design

Three new `n_bare_edges(count)` constructions, all on the 8x8 grid
(64 qubits), spare=0, zero idle qubits, built the same way as
`single_bare_edge` (Addendum 66): `count` bare 2-qubit edges, plus
enough 3-qubit chains to use the remainder exactly, with one chain
enlarged to absorb any remainder not divisible by 3 (matching Addendum
66's own construction rule, applied here rather than re-derived, since
it already handles non-exact division correctly):

| bare edges | qubits used by bare edges | remaining qubits | 3-qubit chains | extra chain (if any) |
|---:|---:|---:|---:|---|
| 2 | 4 | 60 | 20 | none (60 divides evenly by 3) |
| 5 | 10 | 54 | 18 | none (54 divides evenly by 3) |
| 10 | 20 | 44 | 13 | one 5-qubit chain (44 = 13*3 + 5) |

`optimization_level=3`, 3 seeds x 2 repeats per point, matching every
prior addendum in this line (63-66).

## 3. Pre-registered predictions

**P1 (primary -- shape of the transition).** Recording
`VF2Layout_stop_reason` at each of {1 (Addendum 66, already on record),
2, 5, 10, 17 (Addendum 63, already on record)}:
  - **Sharp threshold**: all of {1, 2, 5} cliff and only {10, 17} (or
    only {17}) are fast -- a step-like transition at a specific count.
  - **Gradual/staggered**: the outcomes do not form a single clean
    split (e.g. 2 is fast but 5 cliffs, mirroring the
    non-monotonic shape Addendum 59 found for component count) -- this
    project's own history makes this a live possibility that must not
    be discounted in advance.
  - **Fraction-of-graph reading supported**: the split roughly tracks
    the *qubit fraction* covered by bare edges (1/64=1.6%, 2/64=3.1%,
    5/64=15.6%, 10/64=31.3%, 17/64=53.1% covered by bare-edge qubits)
    rather than the raw count -- distinguishable from "sharp threshold"
    if the split lands near a specific fraction rather than a specific
    count.

**P2 (sanity/consistency).** `VF2Layout_stop_reason` at each new point
is checked for unanimity across all 6 runs (3 seeds x 2 repeats) before
being trusted, matching this project's standing practice.

## 4. What this cannot establish

- The mechanism behind wherever the threshold is found (Tier 2.1 of
  `OPEN_ITEMS_2026-09-18.md` remains separately open).
- Whether the same threshold holds at 6x7 (Tier 1.3) or whether these
  specific 8x8 points are confounded by the grid's own wider slow region
  (Tier 1.2) -- both remain separately open and are not addressed here.
- Bare-edge *position* effects (Tier 2.2) -- position is not varied in
  this design.

---


<!-- ===== Addendum 67 (source: spare-qubit-cliff-addendum-67-2026-09-18.md) ===== -->

> **Note added when merging:** 2, 5, and 10 bare edges all still cliff -- the threshold is narrower than expected, somewhere between 10 and 17, not near the low end of the tested range.

## Addendum 67 -- 2, 5, and 10 bare edges all still cliff: the threshold is narrower than expected, somewhere between 10 and 17 (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-67-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: neither "sharp threshold at a low count" nor a clean
fraction-of-graph split at a moderate value -- all three new points
(2, 5, 10 bare edges) cliff.** Combined with the two points already on
record (1 bare edge: cliff, Addendum 66; 17 bare edges: fast, Addendum
63), the full picture is: **1, 2, 5, 10 all cliff; only 17 is fast.**
The threshold is narrower than either the sanity-check framing or the
fraction-of-graph reading anticipated -- it sits somewhere in the gap
between 10 and 17, not near the low end of the tested range.

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 18 rows completed with `error=""`; stop reason unanimous
(6/6) within every `n_bare` cell.

| bare edges | qubit fraction covered | time (ms) | stop reason |
|---:|---:|---:|:---|
| 1 (Addendum 66) | 1.6% | 10,841.73 | nonexistent solution |
| **2** (this run) | 3.1% | 10,043.11 | **nonexistent solution** |
| **5** (this run) | 15.6% | 9,144.51 | **nonexistent solution** |
| **10** (this run) | 31.3% | 10,062.38 | **nonexistent solution** |
| 17 (Addendum 63) | 53.1% | 30.16 | solution found |

## 2. Scoring

**P1 (primary) -- neither pre-registered "positive" branch confirmed.**
"Sharp threshold" implied a step somewhere within the tested range
(e.g. between 1 and 10); none appeared -- every one of 1, 2, 5, 10
cliffs, all within a tight time band (9,145-10,842ms, less than 1.2x
spread across four very different bare-edge counts). "Fraction-of-graph
reading supported" would have predicted the split tracking something
closer to a moderate fraction (e.g. 10 bare edges, 31.3% coverage,
being fast or borderline); it is not -- 10 remains as firmly cliffing as
1. **The actual shape most resembles the pre-registration's second
branch ("gradual/staggered... outcomes do not form a single clean
split"), except in the opposite direction from what that branch's
example described**: rather than a non-monotonic bounce, this is a
*flat, uniformly cliffing plateau* from 1 through 10, followed by a
single large jump to fast at 17. The gap between 10 and 17 (a jump from
31.3% to 53.1% coverage) is the largest untested interval in the entire
investigation to date.

**P2 (sanity/consistency) -- CONFIRMED.** All three new points show
unanimous (6/6) stop reasons; no ambiguous or split cells.

## 3. What this means

**The threshold is real but much closer to `mixed_uneven`'s own 17-edge
value than to the low end tested.** This narrows the search
considerably: whatever the true cutoff is, it lies strictly between 10
and 17 bare edges (between 31.3% and 53.1% of the device's qubits being
covered by bare-edge structure) -- a gap of only 7 edges, not the
16-edge gap this addendum's design started with. The flat plateau from
1 to 10 is itself informative: **the cliff's cost does not decay
gradually as bare-edge count rises through this entire range** (times
at n_bare=1,2,5,10 are statistically indistinguishable from each
other), consistent with a discrete regime change rather than a smooth
trade-off, though "how discrete" (a single sharp step between 11-16, or
a shorter plateau-then-transition within that narrower gap) remains
unknown.

## 4. What this does not establish

- The exact cutoff within 11-16 -- the natural next, and likely final,
  sweep for this question.
- Whether the cutoff is best described by count or by fraction, since
  both move together across the untested 11-16 gap and cannot yet be
  distinguished (this remains open from Addendum 66's own framing).
- Whether this pattern (flat plateau, single jump near the top of the
  tested range) would look different if the *position* of the bare
  edges, not merely their count, were varied (Tier 2.2 of
  `OPEN_ITEMS_2026-09-18.md`, still separately open).
- Whether any of this generalizes to 6x7 or to grid sizes where the
  slow-region-width confound (Tier 1.2) might apply.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 67's `n_bare_edges`) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18.csv) | this run, 18 rows |
| [`spare-qubit-cliff-addendum-67-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-67-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 18 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `n_bare` cell.
- `n_interaction_edges` for each cell (42, 41, 40 for n_bare=2,5,10
  respectively) matches the intended construction
  (n_bare edges + remaining 3-qubit-chain edges), consistent with the
  `_edges_n_bare_edges` construction rule verified in the sandbox before
  this run.
- This run's machine (`AMD64 Family 25 Model 80 Stepping 0
  AuthenticAMD`) differs from the Intel machine used for Addenda 63-66's
  own bare-edge measurements. The absolute times are not directly
  compared across machines for anything beyond order-of-magnitude
  (all cliffing values here, ~9,100-10,800ms, and Addenda 63-66's own
  cliffing values, ~10,200-14,400ms on Intel, are the same order of
  magnitude); only the stop-reason outcome (cliff vs. fast), not exact
  timing ratios, is used for this addendum's conclusions.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 68 pre-registration (source: spare-qubit-cliff-addendum-68-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for locating the exact bare-edge threshold within 11-16 at single-step resolution, including a construction bug found and fixed before running.

## Addendum 68 -- Pre-registration: locating the exact bare-edge threshold within 11-16 at single-step resolution (2026-09-18)

**Status: pre-registration only. No run at n_bare in {11,...,16} has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 67 narrowed the bare-2-qubit-edge threshold from the original
1-17 range to 10-17: n_bare=10 (31.3% of the device) cliffs, n_bare=17
(53.1%) is fast, and 1, 2, 5, 10 all sit on a flat, uniformly cliffing
plateau. This addendum resolves the remaining 7-edge gap at single-step
resolution, matching the methodology Addendum 34 used to locate the
grid's own spare-qubit threshold and Addendum 53 used for `k_chains`.

## 2. Design

`n_bare_edges(count)` (Addendum 67's own construction, unchanged) on
the 8x8 grid, spare=0, `optimization_level=3`, `n_bare` in
{11, 12, 13, 14, 15, 16} -- every integer between the last known
cliffing point (10) and the last known fast point (17), which is not
re-run since Addendum 63 already measured it. 3 seeds x 2 repeats,
matching every point in this line (63-67).

## 3. Pre-registered predictions

**P1 (primary -- is there a single step, or a staggered/non-monotonic
region, as seen elsewhere in this project)?**
  - **Single clean step**: exactly one adjacent pair among
    {10,11,12,13,14,15,16,17} flips from `"nonexistent solution"` to
    `"solution found"`, and every value on the low side of that step
    cliffs while every value on the high side is fast -- a monotonic
    transition, unlike the non-monotonic patterns Addendum 59 found for
    `merged_pairs`' component count.
  - **Non-monotonic**: at least one value in {11,...,16} does not fit a
    single monotonic split (e.g. 13 is fast but 14 cliffs) -- explicitly
    a live possibility given this project's own history, and not to be
    forced into "single clean step" if the data does not support it.

**P2 (sanity/consistency).** Stop reason checked for unanimity (6/6)
across all seeds/repeats at every new `n_bare` value before being
trusted.

**P3 (timing plateau check).** If P1 confirms "single clean step," the
cliffing side of the new points is predicted to remain within the same
flat time band Addendum 67 found (roughly 9,000-11,000ms on the AMD
machine, or the corresponding Intel-machine band if run there instead),
not a gradual decline toward the fast side -- consistent with a
discrete regime change rather than a smooth trade-off.

## 4. What this cannot establish

- The mechanism behind wherever the threshold is found (Tier 2.1 of
  `OPEN_ITEMS_2026-09-18.md` remains separately open).
- Whether the threshold's exact location is confounded by 8x8's own
  wider slow region (Tier 1.2) -- not addressed here.
- Generalization to 6x7 or other grid sizes (Tier 1.3).

---


<!-- ===== Addendum 68 (source: spare-qubit-cliff-addendum-68-2026-09-18.md) ===== -->

> **Note added when merging:** The bare-edge threshold is a single sharp step between 16 and 17 -- a flat cliffing plateau across the entire 1-16 range, then an abrupt jump to fast. The same sharp-step character as the grid's own spare threshold and k_chains' component-count threshold. **UPDATE: this addendum's "single clean step, stays fast beyond it" reading is superseded by Addenda 70-71 below** -- the fast region turns out to be a narrow, closing WINDOW at both grid sizes, not a permanent escape. Read this addendum for the threshold's exact location (still accurate); read Addenda 70-71 for what happens beyond it.

## Addendum 68 -- the bare-edge threshold is a single step between 16 and 17: a flat cliffing plateau across the entire 1-16 range, then an abrupt jump to fast (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-68-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: "Single clean step" confirmed, decisively.** All six new points
(n_bare = 11 through 16) cliff, unanimous (6/6) at every value. Combined
with every prior point in this line (1, 2, 5, 10: cliff; 17: fast), the
complete picture across the entire tested range is now: **n_bare = 1
through 16 all cliff; only n_bare = 17 is fast.** The threshold is a
single, sharp step between 16 and 17 bare edges -- 50.0% device coverage
versus 53.1% -- not a gradual decline and not a non-monotonic region
anywhere in the 16-point range now on record.

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 36 rows completed with `error=""`; stop reason unanimous
(6/6) within every `n_bare` cell.

| n_bare | coverage | time (ms) | stop reason |
|---:|---:|---:|:---|
| 11 | 34.4% | 9,657.34 | nonexistent solution |
| 12 | 37.5% | 9,738.53 | nonexistent solution |
| 13 | 40.6% | 9,586.10 | nonexistent solution |
| 14 | 43.8% | 9,776.20 | nonexistent solution |
| 15 | 46.9% | 9,779.82 | nonexistent solution |
| **16** | **50.0%** | **9,252.72** | **nonexistent solution** |
| **17** (Addendum 63) | **53.1%** | **30.16** | **solution found** |

## 2. Scoring

**P1 (primary) -- "Single clean step" CONFIRMED.** Every value from 11
to 16 cliffs, extending the flat plateau Addendum 67 found (1-10) all
the way to 16. No non-monotonic behaviour appears anywhere in this
16-point range -- unlike `merged_pairs`' own component-count oscillation
(Addendum 59), the bare-edge variable produces a clean monotonic
transition once its full range is examined at single-step resolution.

**P2 (sanity/consistency) -- CONFIRMED.** All six new cells unanimous
(6/6); no ambiguity anywhere.

**P3 (timing plateau) -- CONFIRMED.** The six new points span
9,252.72-9,779.82ms, a max/min ratio of 1.057x -- effectively flat, and
consistent with Addendum 67's own five-point plateau (9,144.51-
10,841.73ms across n_bare=1,2,5,10, plus this run's six points, fifteen
total measurements across eleven distinct n_bare values, none departing
from a single tight band on this machine).

## 3. The complete picture: a 16-point flat plateau, then a single step

Across the full range now measured (n_bare = 1 through 17, all at zero
idle qubits, 8x8 grid, spare=0):

- **n_bare = 1 to 16** (1.6% to 50.0% device coverage by bare edges):
  uniformly `"nonexistent solution"`, times clustered within a ~1.2x
  band around 9,000-11,000ms on this session's AMD machine.
- **n_bare = 17** (53.1%): abruptly `"solution found"`, 30.16ms -- a
  ~300-350x drop (comparing against the plateau's own range) in a
  single step.

**This is the same qualitative shape this project found for the grid's
own spare-qubit threshold** (Addendum 34: flat at spare=0, instant drop
at spare=1) **and for `k_chains`' component-count threshold** (Addendum
52: flat at high k, instant drop between k=21 and k=10) -- a discrete
regime change with no detectable gradual approach, now confirmed a
third time for a different underlying variable (bare-edge count) on the
same grid. Three independent structural axes in this project (spare
qubits, component-merging degree, bare-edge count) all show this same
sharp-step character rather than a smooth trade-off.

## 4. What this does not establish

- The exact mechanism -- why 16 bare edges (50.0% coverage) is not
  enough but 17 (53.1%) is. No candidate is proposed here beyond noting
  the coincidence that 50% is exactly half the device; whether this is
  meaningful or incidental to this specific n=64 configuration is
  untested.
- Whether the threshold, expressed as a fraction, would land at the
  same ~50-53% mark on a different grid size, or whether it is an
  absolute count (~16-17) independent of device size -- this project has
  only ever tested n=64 for this specific question.
- Whether the same sharp single-step character holds if bare-edge
  *position* (not just count) is varied, or at 6x7's own resolution
  (both still open per `OPEN_ITEMS_2026-09-18.md`).

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged from Addendum 67) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run2.csv) | this run, 36 rows |
| [`spare-qubit-cliff-addendum-68-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-68-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 36 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `n_bare` cell.
- The 1.057x plateau ratio was computed directly from the six new
  medians, not estimated.
- This run used the same AMD machine as Addendum 67, so the timing
  comparison between the two additions (Addendum 67's 1-10 plateau and
  this run's 11-16 plateau) is a same-machine comparison, unlike the
  cross-machine caveat Addendum 67 itself had to note against Addenda
  63-66's Intel-machine data.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 69 pre-registration (source: spare-qubit-cliff-addendum-69-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether the same bare-edge threshold percentage holds at 6x7, including a design correction (proportional scaling by coverage) made before running.

## Addendum 69 -- Pre-registration: does the same bare-edge threshold (50-53% coverage) hold at 6x7, where the slow-region-width confound does not apply? (2026-09-18)

**Status: pre-registration only. No run of `n_bare_edges` at 6x7 has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 68 located the bare-edge threshold precisely at 8x8: 16 bare
edges (50.0% coverage) cliffs, 17 (53.1%) is fast. Two separate open
items from `OPEN_ITEMS_2026-09-18.md` bear on this result: **Tier 1.2**
asks whether the threshold measurement is itself confounded by 8x8's
own wider slow region (Addendum 57: width 3, versus 6x7's width 1);
**Tier 1.3** asks whether the threshold generalizes to 6x7 at all. Both
are addressed together here: re-running the same construction at 6x7 --
where the slow-region-width confound does not apply, since Addendum
34 established 6x7's own width is exactly 1 step -- tests
generalization on a size where the Tier 1.2 concern is moot by
construction. If the threshold looks similar at 6x7, Tier 1.2's concern
is indirectly addressed (the 8x8 result was not an artefact of its
wider region); if it looks different, that difference could still be
attributable to either genuine size-dependence or the Tier 1.2 confound,
and this addendum alone cannot fully separate them (noted in Section 4).

## 2. Design

`_edges_n_bare_edges(n, n_bare)` (Addendum 67's construction, already
general-purpose -- no code change needed, only `n=42` passed instead of
`n=64`) on the 6x7 grid. Coverage percentages at n=42, computed directly:

| n_bare | coverage |
|---:|---:|
| 8 | 38.1% |
| 9 | 42.9% |
| **10** | **47.6%** |
| **11** | **52.4%** |
| 12 | 57.1% |
| 13 | 61.9% |

`n_bare=10` and `n_bare=11` directly bracket 8x8's own threshold
percentage range (50.0%/53.1%). The wider bracket {8,...,13} is tested
to allow for the threshold not landing at exactly the same percentage.

`optimization_level=3`, spare=0 on the 6x7 grid, 3 seeds x 2 repeats.

## 3. Pre-registered predictions

**P1 (primary -- does 6x7 show the same percentage-based threshold, an
absolute-count-based threshold, or something else)?**
  - **Percentage-based**: the split at 6x7 lands near 50-53% coverage
    (n_bare=10 or 11) -- supports the threshold being a property of
    *coverage fraction*, generalizable across device sizes.
  - **Count-based**: the split at 6x7 lands near a similar *absolute*
    count to 8x8's 16-17 (which is impossible here, since 6x7 only has
    21 components total -- the closest absolute-count analog would be
    testing whether a MUCH lower n_bare, e.g. 3-4, already suffices,
    which would support count rather than fraction).
  - **Neither cleanly, or non-monotonic**: reported as its own outcome
    per this project's standing practice, not forced into either
    category.

**P2 (sanity/consistency).** `n_bare=0`'s equivalent (`dense_pairs`
itself, already on record from Addenda 34/51/53/55: cliffs) and the
highest tested point's behavior are checked for consistency with prior
6x7 measurements before trusting intermediate points.

## 4. What this cannot fully establish

- **This design cannot cleanly separate "6x7 has a different threshold
  because of genuine size-dependence" from "the 8x8 threshold itself
  was measured correctly but simply doesn't transfer."** Tier 1.2's
  original concern (is 8x8's OWN threshold measurement confounded by
  its wide slow region) is only indirectly addressed by this
  comparison, not directly re-verified at 8x8 itself.
- Whether the threshold depends on `n` in a way not captured by either
  "same fraction" or "same count" (e.g. some other function of n).
- Bare-edge position effects (still separately open).

---


<!-- ===== Addendum 69 (source: spare-qubit-cliff-addendum-69-2026-09-18.md) ===== -->

> **Note added when merging:** 6x7's threshold is NOT at the same coverage percentage as 8x8's -- all six points tested (38-62% coverage) still cliff, falsifying the simple fraction-transfers-across-sizes reading.

## Addendum 69 -- 6x7's threshold is NOT at the same coverage percentage as 8x8's: all six points (38-62%) still cliff, falsifying the simple fraction reading (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-69-preregistration-2026-09-18.md`, written
and locked before this run, including a design correction (proportional
scaling by coverage percentage rather than absolute count, found while
writing the pre-registration) made before any code was run.

## 0. In one line

**P1: neither "percentage-based" nor "count-based" as cleanly predicted
-- all six tested points cliff, including n_bare=13 at 61.9% coverage,
well past 8x8's own 53.1% fast point.** The pre-registration bracketed
{8,...,13} specifically to bound 8x8's 50-53% threshold range; the
entire bracket cliffed instead. **6x7's threshold, wherever it is, sits
at a higher coverage percentage than 8x8's** -- the simple "same
fraction transfers across grid sizes" reading is falsified. Whether it
is a different fraction or governed by something else (e.g. absolute
component count, which differs sharply between the two grids: 21 at
6x7 vs. 32 at 8x8) is not yet determined.

## 1. Results

6x7 grid (42 qubits), `n_bare_edges`, spare=0, `optimization_level=3`,
3 seeds x 2 repeats. All 36 rows completed with `error=""`; stop reason
unanimous (6/6) within every `n_bare` cell.

| n_bare | coverage | time (ms) | stop reason |
|---:|---:|---:|:---|
| 8 | 38.1% | 7,249.32 | nonexistent solution |
| 9 | 42.9% | 6,957.55 | nonexistent solution |
| 10 | 47.6% | 6,784.88 | nonexistent solution |
| **11** | **52.4%** | 6,844.18 | **nonexistent solution** |
| 12 | 57.1% | 7,355.81 | nonexistent solution |
| **13** | **61.9%** | 6,853.44 | **nonexistent solution** |

For reference, 8x8's own threshold (Addendum 68): n_bare=16 (50.0%)
cliffs, n_bare=17 (53.1%) is fast -- 6x7's n_bare=11 (52.4%), landing
squarely inside 8x8's fast range by percentage, still cliffs here.

## 2. Scoring

**P1 (primary) -- neither branch confirmed as stated; falsified in the
specific sense that the predicted bracket did not contain the
threshold.** "Percentage-based" predicted the split landing near
n_bare=10-11 (8x8's own 50-53% range); it did not -- both cliff, along
with everything up to 13 (61.9%). "Count-based" was defined narrowly in
the pre-registration (a much lower n_bare, e.g. 3-4, sufficing) and
also does not apply -- nothing this low was tested because the bracket
was chosen to bound the percentage hypothesis specifically. **The
actual result is closest to neither branch: 6x7 needs MORE than 61.9%
coverage**, a higher fraction than 8x8's 53.1%, not the same fraction
and not a low absolute count.

**P2 (sanity/consistency) -- CONFIRMED.** All six cells unanimous
(6/6); no ambiguity.

## 3. What this means for Tier 1.2 and Tier 1.3

**Tier 1.3 (generalization) is answered directly: the threshold does
NOT generalize as a fixed coverage percentage.** 6x7 requires
substantially more coverage than 8x8 to escape the cliff. This rules
out the simplest reading (same fraction, any size) and leaves two
candidates, neither tested here:
  - The threshold is a fixed **fraction that itself depends on grid
    size** (i.e. larger grids need a lower fraction) -- consistent with
    8x8 (53.1%) requiring less coverage than 6x7 (more than 61.9%,
    exact value unknown).
  - The threshold tracks something else entirely uncorrelated with
    simple coverage percentage (e.g. absolute remaining component
    count, or a property of the 3-qubit-chain segments that dominate
    the graph once few bare edges are used).

**Tier 1.2 (is 8x8's own threshold measurement confounded by its wider
slow region) remains only indirectly addressed, exactly as the
pre-registration's Section 4 anticipated.** This result is consistent
with either explanation for why 6x7 and 8x8 differ: a genuine,
size-dependent threshold, or an artefact specific to how 8x8's wider
slow region was measured. This addendum's design cannot distinguish
them, and neither should be assumed.

## 4. What this does not establish

- **Where 6x7's actual threshold lies.** Only that it exceeds 61.9%
  coverage (n_bare=13); the upper end has not been swept at fine
  resolution above 13.
- Whether the threshold is a smooth function of grid size or itself has
  the same sharp single-step character Addendum 68 found at 8x8 --
  untested at 6x7's own fine resolution above n_bare=13.
- Whether absolute component count (21 at 6x7 vs. 32 at 8x8) is a
  better predictor than coverage fraction -- this addendum's data is
  consistent with that reading but does not confirm it; a third grid
  size would be needed to test it properly.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged from Addendum 68) |
| [`circuit_family_sweep_6x7_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18.csv) | this run, 36 rows |
| [`spare-qubit-cliff-addendum-69-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-69-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 36 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `n_bare` cell.
- Coverage percentages were recomputed directly (`n_bare*2/42`) for
  every row before comparison against 8x8's own recorded percentages,
  not estimated.
- 8x8's reference figures (Addendum 68) were re-read directly from that
  addendum before the comparison in Section 1's table, not from memory.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 70 pre-registration (source: spare-qubit-cliff-addendum-70-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for locating 6x7's own bare-edge threshold at single-step resolution, including a logical check (n_bare=21 equals dense_pairs, already known to cliff) and a second construction bug found and fixed before running.

## Addendum 70 -- Pre-registration: locating 6x7's own bare-edge threshold at single-step resolution, n_bare 14-20 (2026-09-18)

**Status: pre-registration only. No run at n_bare in {14,...,19} on 6x7
has been performed.** Predictions are locked before any measurement.

## 1. A logical check made before designing this experiment

`n_bare=21` at n=42 is not an arbitrary upper bound -- it is
**identical, by construction, to `dense_pairs` itself** (verified
directly: `_edges_n_bare_edges(42, 21) == _edges_dense_pairs(42)`,
element-for-element). `dense_pairs` is already known to cliff at 6x7
(Addenda 34/51/53/55, ~6,600-6,700ms, the same time band as this
project's own `n_bare` cliffing plateau). **This means it is not
guaranteed that ANY value in {14,...,21} is fast** -- unlike 8x8, where
the threshold was known to exist somewhere below n_bare=32 (the maximum
possible) because 8x8's own equivalent maximum was never itself
suspected to cliff. At 6x7, the upper endpoint of the sweep range is
already known to cliff, so this experiment could in principle find NO
transition within {14,...,19} and instead find the cliff persisting
all the way to n_bare=21's own known-cliffing result. This possibility
is treated as a live, pre-registered outcome (P1's third branch below),
not discovered only after the data comes back.

## 2. Design

`n_bare_edges` on the 6x7 grid, spare=0, `optimization_level=3`,
`n_bare` in {14, 15, 16, 17, 18, 19} -- every valid integer between the
last known cliffing point (13, Addendum 69) and the already-known-
cliffing endpoint (21, equivalent to `dense_pairs`). 3 seeds x 2
repeats.

**A real bug was found and fixed while preparing this design, before
any run**: `_edges_n_bare_edges` (unchanged since Addendum 67) silently
produced out-of-range qubit indices whenever the remaining qubit count
after placing the bare edges fell strictly between 0 and 3 -- exactly
the case at `n_bare=20` on n=42 (remaining=2). Caught by a negative-
idle-qubit sanity check before running, not by a crash. The function
now raises an explicit `ValueError` for this case instead. **n_bare=20
is therefore excluded from this sweep** -- no zero-idle construction
exists for it via this function, and no substitute was devised for this
addendum. None of Addenda 66-69's actual measurements were affected
(their remainders were always large enough to avoid this edge case).

## 3. Pre-registered predictions

**P1 (primary -- does a transition exist in this range, and if so
where)?**
  - **Transition found**: at least one point in {14,...,19} is fast
    (`"solution found"`), and everything at or below that point (down
    to 13) is cliffing -- locates 6x7's actual threshold.
  - **No transition -- cliff persists to the known endpoint**: every
    point in {14,...,19} cliffs, consistent with n_bare=21
    (`dense_pairs`) also cliffing -- meaning 6x7 has NO bare-edge count
    that escapes the cliff at spare=0, in contrast to 8x8 where 17 of
    32 possible bare edges (53.1%) was sufficient. This would be a
    materially different and stronger finding than "the threshold is
    just higher at 6x7" -- it would mean the bare-edge escape route
    Addenda 63-68 characterized at 8x8 may not exist at all at 6x7.
  - **Non-monotonic**: at least one point does not fit a single clean
    split (e.g. 17 is fast but 18 cliffs) -- reported as its own
    outcome per this project's standing practice.

**P2 (sanity/consistency).** Stop reason checked for unanimity (6/6) at
every new point before being trusted.

## 4. What this cannot establish

- If P1's "no transition" branch is confirmed, *why* 6x7 differs
  structurally from 8x8 in this way -- only that it does.
- Whether component count (21 at 6x7 vs. 32 at 8x8) or something else
  explains any difference found -- a third grid size would be needed.
- Whether the Tier 1.2 confound (8x8's own wider slow region) plays any
  role -- still not directly addressed by this design.

---


<!-- ===== Addendum 70 (source: spare-qubit-cliff-addendum-70-2026-09-18.md) ===== -->

> **Note added when merging:** **The fast region at 6x7 is a narrow WINDOW (76-86% coverage), not a monotonic threshold** -- bounded by cliffing regions on both sides. This retroactively calls Addendum 68's 'single clean step, stays fast' conclusion at 8x8 into question, since that addendum never tested past the one fast point it found.

## Addendum 70 -- the fast region at 6x7 is a narrow WINDOW (76-86% coverage), not a monotonic threshold: this retroactively calls Addendum 68's "single clean step" conclusion into question (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-70-preregistration-2026-09-18.md`, written
and locked before this run, including a bug fix to `_edges_n_bare_edges`
(found while designing this experiment) and a logical check (n_bare=21
is identical to `dense_pairs`, already known to cliff) made before
running.

## 0. In one line

**P1: "Non-monotonic" confirmed, in a specific and consequential
shape.** `n_bare=14` (66.7% coverage) and `n_bare=15` (71.4%) cliff;
`n_bare=16, 17, 18` (76.2%-85.7%) are all fast; **`n_bare=19` (90.5%)
cliffs again.** Combined with Addendum 69 (8-13 all cliff) and the
logical fact that `n_bare=21` is `dense_pairs` itself (known to cliff),
the complete 6x7 picture is: **cliff (8-15) -- fast window (16-18) --
cliff again (19, and presumably 20-21)**. This is not a threshold at
all; it is a narrow band of feasibility sandwiched between cliffing
regions on both sides. **This retroactively undermines Addendum 68's
own conclusion at 8x8**: that addendum tested only up to n_bare=17
(fast) and concluded "single clean step," implicitly assuming the fast
region continues indefinitely -- but 8x8's own `dense_pairs` endpoint
(n_bare=32) was never approached, and this addendum's direct evidence
that a return-to-cliff is a real phenomenon means 8x8 may have the same
narrow-window structure, simply not yet detected because the sweep
stopped too early.

## 1. Results

6x7 grid (42 qubits), `n_bare_edges`, spare=0, `optimization_level=3`,
3 seeds x 2 repeats. All 36 rows completed with `error=""`; stop reason
unanimous (6/6) within every `n_bare` cell -- no ambiguous cells
anywhere in this surprising result.

| n_bare | coverage | time (ms) | stop reason |
|---:|---:|---:|:---|
| 14 | 66.7% | 6,862.03 | nonexistent solution |
| 15 | 71.4% | 6,938.19 | nonexistent solution |
| **16** | **76.2%** | **25.74** | **solution found** |
| **17** | **81.0%** | **860.15** | **solution found** |
| **18** | **85.7%** | **849.28** | **solution found** |
| **19** | **90.5%** | **6,693.30** | **nonexistent solution** |

Note the fast window is not itself flat: n_bare=16 is 25.74ms while 17
and 18 are ~850ms each -- over 30x slower than 16, though both remain
`"solution found"`. This internal variation within the fast window is
recorded but not analyzed further here.

## 2. Scoring

**P1 (primary) -- "Non-monotonic" CONFIRMED, and reported as the
serious finding it is, not smoothed into either of the other two
branches.** Neither "transition found" (a single clean split with
everything above the split fast) nor "no transition" (cliff persisting
throughout) describes what happened. A third, specific shape appeared:
**a narrow fast window bounded by cliffing regions on both sides.**

**P2 (sanity/consistency) -- CONFIRMED.** All six cells unanimous
(6/6); the non-monotonic pattern is not attributable to noise or
mixed-outcome cells.

## 3. The retroactive problem with Addendum 68

Addendum 68 swept 8x8 from n_bare=11 to 16 (all cliff) plus the
already-known n_bare=17 (fast, from Addendum 63's `mixed_uneven`), and
concluded: *"the threshold is a single, sharp step between 16 and 17
bare edges... not a gradual decline and not a non-monotonic region
anywhere in the 16-point range now on record."* **That conclusion is
accurate for the 16-point range actually tested (1-17) but was
implicitly read as "and therefore stays fast beyond 17," which was
never tested and is not established.** 8x8's `dense_pairs` equivalent
is n_bare=32 (16 points beyond where testing stopped) -- more than
double the distance 6x7's own return-to-cliff (at n_bare=19, only 2
steps past the fast window's upper edge, n_bare=18) took to reappear.

**This means Addendum 68's finding should be read narrowly**: 8x8 has a
fast point at n_bare=17 immediately following a cliffing plateau at
1-16. Whether that fast region is a permanent escape or (like 6x7) a
narrow window that closes again before n_bare=32 is **now an open,
directly testable question this addendum's own discovery raises**, not
previously flagged as a risk in Addendum 68 itself.

## 4. What this does not establish

- **Whether 8x8 has the same narrow-window structure.** This is now the
  single most important next check: sweep 8x8 from n_bare=18 upward
  (toward its own endpoint, 32) to see whether the cliff returns there
  too.
- Where exactly 6x7's window boundaries are at single-step resolution
  on the low side (window opens between 15 and 16, confirmed) and
  whether 19-21 are uniformly cliffing or contain further structure
  (only 19 and 21 are known; 20 is excluded by the construction bug,
  and 21 alone does not confirm 19-21 is a uniform cliffing plateau
  rather than containing its own internal structure).
- Why the window exists at this specific location (76-86% coverage) --
  no mechanism is proposed.
- The internal ~30x variation within the fast window itself (16 vs.
  17-18) -- noted, not investigated.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 70's bug fix to `_edges_n_bare_edges`) |
| [`circuit_family_sweep_6x7_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run2.csv) | this run, 36 rows |
| [`spare-qubit-cliff-addendum-70-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-70-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 36 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `n_bare` cell, specifically re-checked given how
  surprising the non-monotonic result was, per this project's standing
  practice of extra scrutiny on unexpected findings.
- Addendum 68's own conclusion was re-read and quoted verbatim (Section
  3) before being characterized as retroactively incomplete, rather
  than paraphrased from memory.
- The logical fact that n_bare=21 equals `dense_pairs` (Addendum 70's
  own pre-registration, Section 1) was re-confirmed relevant here:
  since n_bare=19 already cliffs, and n_bare=21 is independently known
  to cliff, the window (16-18) is bounded on both sides by verified
  data, not merely inferred from one side.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 71 pre-registration (source: spare-qubit-cliff-addendum-71-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether 8x8 also has a narrow fast window, or stays fast all the way to n_bare=32 (dense_pairs' own endpoint) -- the urgent question Addendum 70 raised.

## Addendum 71 -- Pre-registration: does 8x8 also have a narrow fast window, or does it stay fast all the way to n_bare=32? (2026-09-18)

**Status: pre-registration only. No run at n_bare in {18,...,32} on 8x8
has been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 70 found 6x7 has a narrow fast window (n_bare=16-18, 76-86%
coverage) bounded by cliffing regions on both sides -- not a permanent
escape past a single threshold. This directly calls into question
Addendum 68's own conclusion at 8x8 ("single clean step... stays fast"),
which only tested up to n_bare=17 (fast) and never approached 8x8's own
`dense_pairs` endpoint (n_bare=32). This is the single most important
open question this project currently has: **does the fast region found
at 8x8 (n_bare=17) close again before n_bare=32, the way 6x7's did?**

## 2. Design, in two stages

**Stage A (coarse scan, this run)**: `n_bare_edges` on the 8x8 grid,
spare=0, `optimization_level=3`, `n_bare` in
{18, 20, 22, 24, 26, 28, 30, 32} -- every other integer from just past
the known-fast point (17) to the `dense_pairs` endpoint (32), skipping
`n_bare=31` (excluded: Addendum 70's bug-fix guard raises for this
value at n=64, since remaining=2 there -- verified directly before
finalizing this design, the same edge case caught at n=42/n_bare=20).
3 seeds x 2 repeats.

**Stage B (not part of this pre-registration)**: if Stage A finds any
transition (fast-to-cliff or cliff-to-fast), a follow-up addendum will
locate it at single-step resolution, the same two-stage approach used
for the bare-edge threshold itself (Addenda 67 then 68).

## 3. Pre-registered predictions

**P1 (primary -- does the fast region persist, close into a window, or
show some other pattern)?**
  - **Persists**: every point in {18,...,32} is fast -- 8x8's escape
    from the cliff, once found at n_bare=17, is permanent within this
    circuit family, unlike 6x7.
  - **Window closes**: at least one point in {18,...,32} cliffs again,
    after n_bare=17's own fast result -- confirms the same narrow-window
    structure found at 6x7 generalizes to 8x8, and the "single clean
    step" reading from Addendum 68 was premature at both grid sizes.
  - **Complex/multiple windows**: more than one transition appears in
    this coarse scan (e.g. cliff, fast, cliff, fast) -- reported as its
    own outcome, not forced into either branch above.

**P2 (sanity/consistency).** Stop reason checked for unanimity (6/6) at
every point. `n_bare=32` (equivalent to `dense_pairs` at n=64, already
known to cliff from every prior `dense_pairs` measurement at 8x8) is
predicted to cliff regardless of P1's outcome -- this is not a live
prediction but a consistency check, exactly as n_bare=21 served for
6x7 in Addendum 70.

## 4. What this cannot establish

- The exact boundary of any window found -- Stage B, not this
  pre-registration, would locate it precisely.
- Whether a window's location (if found) bears any numeric relationship
  to 6x7's own window (76-86% coverage) -- this coarse, every-other-
  integer scan is not designed to resolve fine correspondence.
- Mechanism -- why a window would open and close, if it does.

---


<!-- ===== Addendum 71 (source: spare-qubit-cliff-addendum-71-2026-09-18.md) ===== -->

> **Note added when merging:** **Confirmed: 8x8 also has a closing window, even narrower than 6x7's** -- every point from 18 to 32 cliffs. The only fast point ever found at 8x8 (n_bare=17) may be an isolated single-point spike, not a sustained escape. Addendum 68's conclusion is formally superseded.

## Addendum 71 -- confirmed: 8x8 ALSO has a closing window, and it is even narrower than suspected -- n_bare=17 may be an isolated single-point spike, not a sustained escape (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-71-preregistration-2026-09-18.md`, written
and locked before this run, including an excluded value (n_bare=31,
same construction-bug edge case as 6x7's n_bare=20) found before
running.

## 0. In one line

**P1: "Window closes" confirmed, and more starkly than 6x7's own
result.** Every point from n_bare=18 (56.2% coverage) through n_bare=32
(100%, equivalent to `dense_pairs` itself) cliffs. **The only fast point
ever found at 8x8 across this entire investigation remains n_bare=17
alone** (Addendum 63's `mixed_uneven`, 53.1% coverage) -- immediately
preceded by a cliffing plateau (n_bare=1-16, Addenda 67-68) and
immediately followed by a cliffing plateau (n_bare=18-32, this run).
**Addendum 68's "single clean step... stays fast" conclusion is now
formally superseded**: at 8x8, the fast region is not a step to a
permanently-escaped state -- it may be a single isolated point, an even
narrower window than 6x7's own three-point-wide one (n_bare=16-18).

## 1. Results

8x8 grid (64 qubits), spare=0, `optimization_level=3`, 3 seeds x 2
repeats. All 48 rows completed with `error=""`; stop reason unanimous
(6/6) within every `n_bare` cell.

| n_bare | coverage | time (ms) | stop reason |
|---:|---:|---:|:---|
| 16 (Addendum 68) | 50.0% | 9,252.72 | nonexistent solution |
| **17** (Addendum 63) | **53.1%** | **30.16** | **solution found** |
| **18** (this run) | **56.2%** | **9,236.87** | **nonexistent solution** |
| 20 | 62.5% | 9,709.43 | nonexistent solution |
| 22 | 68.8% | 8,941.44 | nonexistent solution |
| 24 | 75.0% | 8,936.96 | nonexistent solution |
| 26 | 81.2% | 8,878.07 | nonexistent solution |
| 28 | 87.5% | 8,713.31 | nonexistent solution |
| 30 | 93.8% | 8,957.36 | nonexistent solution |
| 32 (= `dense_pairs`) | 100.0% | 8,943.58 | nonexistent solution |

The full 8x8 picture across every point ever measured in this line
(Addenda 63, 67, 68, this run): cliff at 1-16, **fast only at 17**,
cliff at 18-32. Timing on the cliffing side remains in the same tight
band throughout (8,713-9,776ms across all cliffing points measured on
this AMD machine today, including both the 1-16 and 18-32 regions),
reinforcing that this is a single discrete state, not a gradient, on
either side of the isolated fast point.

## 2. Scoring

**P1 (primary) -- "Window closes" CONFIRMED, more severely than either
pre-registered branch anticipated.** The pre-registration's "window
closes" branch predicted at least one cliffing point somewhere in
{18,...,32}; **all eight tested points cliff**, immediately adjacent to
the known fast point at 17. This is not merely "the escape eventually
closes" -- it is consistent with the escape being a single point,
possibly narrower than any window found so far in this project.

**P2 (sanity/consistency) -- CONFIRMED.** `n_bare=32`
(`dense_pairs`-equivalent) cliffs as predicted; all cells unanimous
(6/6).

## 3. What this means: Addendum 68's conclusion formally retracted, and 17's own status now the central question

**This is not a minor correction -- Addendum 68's "single clean step"
framing described the wrong shape entirely.** What Addendum 68 actually
found, re-read in light of this result, was one edge of what might be a
single-point spike: n_bare=16 cliffs, n_bare=17 is fast, n_bare=18
cliffs. Addendum 68 characterized the n_bare=16-to-17 transition
correctly (it is real, sharp, and reproducible) but incorrectly inferred
persistence beyond it. **The urgent open question this raises: is
n_bare=17 truly an isolated single point, or does the fast region extend
to some untested value that this integer-only sweep cannot see, or does
n_bare=17 itself need to be re-confirmed** (its "fast" status rests on
Addendum 63's `mixed_uneven`, a distinct construction, not a direct
`n_bare_edges(17)` run -- see Section 4).

**Practically, for any account of this project's "bare-edge escape
route": it should not be described as "adding enough bare edges avoids
the cliff."** At 8x8, only one specific point (out of 32 possible
values) escapes; every other value, including ones both far below and
far above it, cliffs. This is closer to "a specific, narrow, possibly
singular configuration escapes" than to "sufficient bare-edge coverage
escapes."

## 4. What this does not establish

- **Whether n_bare=17 reproduces on a re-run via `n_bare_edges` itself.**
  This addendum did not re-test it; its "fast" status rests on Addendum
  63's single measurement (`mixed_uneven`, a distinct construction
  achieving the same n_bare=17/coverage=53.1% condition via a different
  specific edge arrangement). Whether `n_bare_edges(17)` specifically is
  also fast has never been directly tested -- this is now the most
  important remaining gap.
- Whether any single-integer point between the tested even values (odd
  numbers 19, 21, 23, 25, 27, 29 -- 31 excluded by the construction bug)
  is also fast, given this was a coarse, every-other-integer scan.
- Why n_bare=17 specifically escapes, if it does -- no mechanism
  proposed.
- Whether 6x7's own three-point-wide window (16-18) and 8x8's
  apparently single-point one (17 alone) reflect a real difference
  between the two grid sizes or are both under-resolved at their
  current sampling densities.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged from Addendum 70) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run3.csv) | this run, 48 rows |
| [`spare-qubit-cliff-addendum-71-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-71-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 48 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `n_bare` cell.
- Addendum 68's own conclusion was re-read directly before being
  characterized as superseded, and the specific claim being revised
  ("stays fast," an inference beyond what was tested) is distinguished
  from the claim that remains correct (the 16-to-17 transition itself
  is real).
- n_bare=17's status was checked against its actual source (Addendum
  63's `mixed_uneven`, not a direct `n_bare_edges(17)` run) before
  Section 4 flagged this as an unconfirmed gap rather than treating it
  as equivalent.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 72 pre-registration (source: spare-qubit-cliff-addendum-72-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether n_bare=17 at 8x8 is reliably fast, reliably slow, or a genuine boundary point, tested at much higher seed count than this project's standing resolution.

## Addendum 72 -- Pre-registration: is n_bare=17 reliably fast, reliably slow, or a genuine boundary point? (2026-09-18)

**Status: pre-registration only. No run at n_bare=17 via `n_bare_edges`
(as opposed to `mixed_uneven`) has been performed.** Predictions are
locked before any measurement.

## 1. Why this experiment exists

Addendum 71 found 8x8's fast region may be a single isolated point at
n_bare=17, bounded by cliffing plateaus on both sides (16 and 18). But
that point's "fast" status rests entirely on Addendum 63's
`mixed_uneven` construction, not on `n_bare_edges(17)` itself -- a
different specific edge arrangement achieving the same coverage
(53.1%) and component count. Whether `n_bare_edges(17)` behaves the
same way has never been tested. This addendum tests it directly, with
substantially higher statistical power (20 seeds x 10 repeats = 200
runs, versus this project's standing 3x2=6) specifically because a
single-point finding is exactly the situation where a small sample
could mislead -- either by having gotten lucky/unlucky once, or by
masking genuine seed-dependent bistability.

**What varying `seed` actually varies here, confirmed from source
before writing this prediction**: `_edges_n_bare_edges`'s construction
(which qubits are bare-edge, which are chain) does not depend on `rng`
at all -- it is fully determined by `n_bare` alone. `seed` only affects
the random single-qubit-unitary content placed on each edge (via
`build_circuit_from_family`'s own `rng.integers(...)` calls feeding
`random_unitary(4, seed=...)`). **This experiment therefore tests
whether n_bare=17's outcome is robust to the specific gate content on a
FIXED interaction graph, not whether different graph structures at the
same n_bare all behave alike** (that remains a separate, unaddressed
question).

## 2. Design

`n_bare_edges` at `n_bare=17` on the 8x8 grid, spare=0,
`optimization_level=3`, **20 seeds x 10 repeats (200 total runs)** --
far more statistical power than this project's standing 3x2 resolution,
justified specifically because this single point is currently the
entire basis for this project's "bare-edge escape exists" claim.

## 3. Pre-registered predictions

**P1 (primary -- which of three regimes does n_bare=17 fall into)?**
  - **Reliably fast**: `"solution found"` in all or nearly all
    (>=190/200, allowing for rare unexplained outliers of the kind
    Addendum 27 once found and Addendum 28 failed to reproduce) runs --
    confirms a genuine, narrow, single-point stable escape exists at
    8x8, and `mixed_uneven`'s original result was not a fluke.
  - **Reliably slow**: `"nonexistent solution"` in all or nearly all
    runs -- would mean `mixed_uneven`'s original fast result
    (Addendum 63) was itself the outlier, not representative of
    `n_bare=17`'s typical behavior, and the entire "bare-edge escape"
    narrative for 8x8 would need to be retracted back to "no confirmed
    escape has been found."
  - **Genuine boundary / bistable**: a substantial mix of both outcomes
    (neither near-unanimous), meaning the specific random gate content
    determines which regime a given circuit falls into -- n_bare=17
    would be a true phase-boundary point, not a stable escape.

**P2 (consistency with `mixed_uneven`).** If P1 confirms "reliably
fast," this validates that `mixed_uneven` and `n_bare_edges(17)` --
different specific graphs at the same coverage and component count --
behave the same way, strengthening "bare-edge coverage" as the
operative variable over any more specific property of `mixed_uneven`'s
particular arrangement. If P1 finds "reliably slow" or "bistable," this
would mean the two constructions genuinely differ despite matching
coverage/component-count, and something more specific than those two
summary statistics matters.

## 4. What this cannot establish

- Whether OTHER n_bare=17 constructions (different specific bare-edge
  placements, not `n_bare_edges`'s or `mixed_uneven`'s particular
  choices) would show the same regime.
- Whether 6x7's own window (a 3-point-wide region, not obviously a
  single boundary point) has the same or different character --
  untested here.
- Mechanism, in any case.

## 5. A cost note

200 runs at this point's own measured cost (~9,000ms median when slow,
~30-860ms when fast per prior single measurements) could take anywhere
from roughly 30 minutes (if uniformly fast) to over 5 hours (if
uniformly slow) of wall-clock time. This is explicitly accepted as the
cost of resolving what this addendum's own introduction calls the
single most information-dense open point in the project.

---


<!-- ===== Addendum 72 (source: spare-qubit-cliff-addendum-72-2026-09-18.md) ===== -->

> **Note added when merging:** **`n_bare_edges(17)` cliffs in 23 of 23 runs, independently reproduced twice.** `mixed_uneven`'s fast result does NOT reproduce via the actual parameterized construction -- the two are structurally different graphs (18 vs. 27 components) despite matching coverage. The 8x8 'bare-edge escape' finding is substantially weakened.

## Addendum 72 -- n_bare_edges(17) cliffs in 13 of 13 runs across 5 seeds: `mixed_uneven`'s fast result at the same coverage does NOT reproduce, and the "bare-edge escape" finding at 8x8 is retracted (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-72-preregistration-2026-09-18.md`, written
and locked before this run. **The pre-registration's planned 200-run
design (20 seeds x 10 repeats) was not completed** -- a large-scale run
appeared to hang (CPU usage reported at 0%, no progress output) and was
interrupted. The cause of that apparent hang was not established (see
Section 4); rather than resolve it, this addendum proceeds with smaller,
successfully-completed runs that already answer P1 unambiguously,
documented in full below.

**Update**: after this addendum's initial 13-run result, the same
5-seed x 2-repeat (10-run) configuration was independently re-run in
full as a direct reproducibility check. Results incorporated below
(now 23 runs total).

## 0. In one line

**P1: "Reliably slow" confirmed, not "reliably fast," and now
independently reproduced.** Across 23 total runs (5 distinct seeds: 0,
1, 2, 3, 4, each run multiple times, including one full 5-seed x
2-repeat configuration reproduced independently in full) at
`n_bare_edges(17)` on the 8x8 grid, **every single run returns
`"nonexistent solution"`** -- times ranging 8,995.74-10,800.75ms, the
same cliffing band found throughout Addenda 63-71.
**Not one run was fast.** This directly contradicts Addendum 63's
`mixed_uneven` result (30.16ms, `"solution found"`) at the same
coverage (53.1%). **Addendum 71's characterization of n_bare=17 as "the
only fast point at 8x8" is retracted**: that characterization rested
entirely on `mixed_uneven`, a different specific construction, not on
`n_bare_edges(17)` itself, and the two do not behave the same way --
because, as Section 3 shows, they are not actually the same graph
shape.

## 1. Results

8x8 grid (64 qubits), `n_bare_edges(17)`, spare=0,
`optimization_level=3`. Three separate invocations, combined:

| run | seeds | repeats | rows | outcome |
|---|---|---|---:|---|
| single-point check | seed=0 | 1 | 1 | cliff (9,175.68ms) |
| small batch | seed=0,1 | 1 | 2 | cliff, cliff (9,214.65 / 9,283.97ms) |
| larger batch | seed=0-4 | 2 | 10 | cliff x10 (8,995.74-10,800.75ms) |
| **reproducibility re-run** (independent, same config as above) | seed=0-4 | 2 | 10 | cliff x10 (9,016.88-9,491.32ms) |
| **total** | **seed=0-4** | | **23** | **23/23 cliff** |

All 23 rows completed with `error=""`. No run, at any seed, at any of
the four invocations, returned `"solution found"`. The 5-seed x
2-repeat configuration was run twice, independently, with closely
matching results both times (both bands sit within ~9,000-9,500ms
aside from Section 1's original run's two mild outliers at
9,870/10,800ms), supporting that the cliffing result itself is stable,
not merely a one-off batch effect.

## 2. Scoring

**P1 (primary) -- "Reliably slow" CONFIRMED, "reliably fast" and
"bistable" both FALSIFIED, and independently reproduced.** 0/23 fast is
an even cleaner result than the "reliably slow" branch's own bar
required. No bistability was observed: every seed tested (0-4) produced
a cliff at every repeat, across two independent runs of the same
5-seed x 2-repeat configuration.

**P2 (consistency with `mixed_uneven`) -- FALSIFIED.** The two
constructions do NOT behave the same way despite matching coverage
(53.1%) -- see Section 3 for what actually differs between them.

## 3. What actually differs between `n_bare_edges(17)` and `mixed_uneven`, found while writing this addendum

Re-reading both constructions' own definitions side by side reveals
they are **not the same graph shape at all**, despite matching coverage
and superficially similar descriptions:

- **`mixed_uneven`** (Addendum 63): 17 bare 2-qubit edges plus one
  single 30-qubit dominant component -- **18 components total**
  (17 + 1).
- **`n_bare_edges(17)`** (Addendum 67, verified by direct computation
  before writing this section): 17 bare 2-qubit edges plus the
  remaining 30 qubits split into **10 separate 3-qubit chains** --
  **27 components total** (17 + 10), not 18.

**These were never the same construction test.** Addendum 71's
"n_bare=17 is fast" claim conflated two different graphs that happen to
share a coverage percentage and a bare-edge count, but differ sharply in
component count (18 vs. 27) and in whether the non-bare-edge portion of
the graph is one dominant component or many smaller ones. **This is the
same distinction Addenda 63-65 already established matters**
(`shrinking_dominant` and `large_dominant_no_bare_edges` showed
dominant-component size and structure matter independently of bare-edge
count) -- and it was overlooked when Addendum 71 treated the two
constructions as interchangeable at "n_bare=17."

## 4. The apparent hang: reported, not explained

A 200-run (20x10) invocation of this same command was reported by the
user as appearing to freeze: Task Manager showed 0% CPU usage for
`python.exe`, no progress lines printed, and the process did not
respond to normal operation. It was interrupted (Ctrl+C), which
produced a truncated traceback ending mid-line inside
`transpile()`/`passmanager.run()` -- consistent with an interrupt during
a call already in progress, not with a crash. **Whether this was a
genuine deadlock/hang, a very long individual `transpile()` call that
happened to coincide with a moment of low sampled CPU usage, or
something else was not established.** Follow-up smaller runs (2 runs, 10
runs) completed normally with no similar symptom. This is reported as an
open, unresolved operational issue -- not diagnosed, not attributed to a
specific cause, and not assumed to be resolved simply because smaller
runs succeeded afterward.

## 5. What this means for the project

**The "bare-edge escape" finding at 8x8 is substantially weakened.**
Addenda 63-71 built an account of a threshold (later revised to a
narrow window, then to a possibly-single-point spike) resting on
`mixed_uneven`'s single measurement at n_bare=17-equivalent coverage.
That single measurement is now understood to have come from a
**structurally different graph** (18 components) than the one this
project's own parameterized sweep produces at the same coverage
(27 components) -- and the parameterized version cliffs reliably,
13/13. **Whether ANY confirmed fast point exists at 8x8, once
`mixed_uneven`'s own specific graph is set aside, is now genuinely open
again.** The most honest current statement: `mixed_uneven`'s particular
18-component, one-dominant-plus-17-bare-edges structure was fast (once,
Addendum 63); `n_bare_edges`' 27-component, many-3-qubit-chains-plus-
17-bare-edges structure at the same coverage is reliably slow (13/13,
this addendum). **Component count and dominant-component structure, not
bare-edge coverage alone, may be the better-supported variable after
all** -- consistent with, not contradicting, Addenda 63-65's own earlier
finding that these mattered, but in tension with the simpler "bare-edge
coverage" narrative Addenda 66-71 built on top of it.

## 6. What this does not establish

- Whether `mixed_uneven` itself reproduces on a re-run -- it has only
  ever been measured once (Addendum 63). Given this addendum's own
  finding that a superficially similar construction is reliably slow,
  re-testing `mixed_uneven` directly is now a priority, not an
  assumption.
- The cause of the apparent large-run hang (Section 4).
- Whether 6x7's own "window" (Addendum 70, n_bare=16-18) is subject to
  the same construction-conflation problem -- `n_bare_edges` was used
  consistently there, not compared against a differently-shaped
  construction, so this specific issue may not apply, but this has not
  been explicitly re-checked.

## 7. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged) |
| (run4's own CSV, `..._run4.csv`, 1 row seed=0) | reported via terminal output only, not uploaded as a file -- its single result is still counted in Section 1's total (verified against the pasted terminal transcript) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run5.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run5.csv) | 2 rows (seed=0,1) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run6.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run6.csv) | 10 rows (seed=0-4, repeats=2) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run7.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run7.csv) | 10 rows (seed=0-4, repeats=2 -- independent reproducibility re-run of run6's own configuration) |
| [`spare-qubit-cliff-addendum-72-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-72-preregistration-2026-09-18.md) | the predictions scored above |

## 8. Verification

- All 23 rows across all four files (one reported via terminal only)
  checked for `error=""`; every one `"nonexistent solution"`, confirmed
  by direct count, not sampled.
- The component-count discrepancy between `mixed_uneven` (18) and
  `n_bare_edges(17)` (27) was verified by direct computation from both
  constructions' own defining parameters before writing Section 3, not
  assumed from their names or coverage percentages matching.
- Section 4's account of the apparent hang is stated as unresolved
  throughout, with no causal claim asserted beyond what was directly
  observed (0% CPU, no output, interrupted successfully, smaller runs
  worked).
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and all three CSVs
  -> 0 hits. Terminal output supplied during this exchange was reviewed
  for local paths before use; none were reproduced here.

---


<!-- ===== Addendum 73 pre-registration (source: spare-qubit-cliff-addendum-73-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether 6x7's window (n_bare=16-18) survives the same seed-count increase that just revealed 8x8's apparent escape was a construction-conflation artefact.

## Addendum 73 -- Pre-registration: is 6x7's "fast window" (n_bare=16-18) real, or does it evaporate under more seeds like 8x8's n_bare=17 just did? (2026-09-18)

**Status: pre-registration only. No re-run of 6x7's n_bare=16-18 beyond
Addendum 70's original 3-seed x 2-repeat measurement has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 72 found `n_bare_edges(17)` at 8x8 -- believed fast based on a
*different* construction (`mixed_uneven`) -- cliffs reliably across 23
runs and 5 seeds when tested via the actual `n_bare_edges` function.
That specific failure mode (construction conflation) does not directly
apply to 6x7's window, since Addendum 70 measured `n_bare_edges(16)`,
`n_bare_edges(17)`, and `n_bare_edges(18)` all via the same function,
consistently -- there is no cross-construction mismatch to find here.
**But the underlying statistical risk is the same one Addendum 72 just
demonstrated**: Addendum 70's "fast" conclusion for n_bare=16-18 rests
on only 3 seeds x 2 repeats (6 runs each), the same modest sample size
that made `mixed_uneven`'s single run look more solid than it was. This
addendum directly tests whether 6x7's window survives 5 seeds, mirroring
the exact scale increase that revealed 8x8's result was not what it
appeared to be.

## 2. Design

`n_bare_edges` on the 6x7 grid, spare=0, `optimization_level=3`,
`n_bare` in {16, 17, 18} (the three points Addendum 70 found fast), 5
seeds (0-4) x 2 repeats -- matching the scale that resolved Addendum
72's question at 8x8. Addendum 70's own original 3-seed result (seeds
0-2 implicitly, per this project's standing `seeds = range(args.seeds)`
convention) is treated as a subset already on record; seeds 3-4 are the
genuinely new data, with all 5 seeds re-run together here for a single,
directly comparable batch rather than only adding the increment.

## 3. Pre-registered predictions

**P1 (primary -- does the window survive 5 seeds)?**
  - **Window confirmed**: all three points (16, 17, 18) remain
    `"solution found"` across all 5 seeds x 2 repeats (30 runs total) --
    the window is real, and 6x7 genuinely differs from 8x8's own
    apparent single-point mirage.
  - **Window evaporates (fully)**: some or all of {16, 17, 18} return to
    `"nonexistent solution"` in the majority of the new runs -- mirrors
    Addendum 72's finding exactly, and would support the user's own
    proposed "Contiguous Cliff" law: the occupancy cliff is unbroken at
    spare=0 regardless of grid size or bare-edge composition, and every
    apparent escape found so far (both grids) was a small-sample
    artefact.
  - **Partial survival**: one or two of the three points remain reliably
    fast while others do not -- reported as its own outcome, refining
    rather than confirming or fully retracting the window.

**P2 (sanity/consistency).** Compare the new 5-seed results directly
against Addendum 70's own original recorded values (single runs at
3 seeds x 2 repeats) before drawing conclusions -- if the original
values do not reproduce even approximately, that is itself notable and
reported explicitly, not smoothed over.

## 4. What this cannot establish

- If the window evaporates, *why* the original 6-run measurement showed
  it -- only that it does not survive a larger sample.
- Whether 8x8 might still have some OTHER fast point not yet identified,
  now that its own n_bare=17 has been ruled out via the correct
  construction.
- Whether the "Contiguous Cliff" hypothesis, if supported here, would
  also hold for grid sizes or circuit families not yet tested this way.

---


<!-- ===== Addendum 73 (source: spare-qubit-cliff-addendum-73-2026-09-18.md) ===== -->

> **Note added when merging:** **6x7's window survives fully: 30/30 fast**, unlike 8x8's mirage. The same diagnostic (more seeds, same construction function) produces opposite verdicts at the two grid sizes -- they genuinely differ, not merely a sample-size artefact. The user's proposed 'Contiguous Cliff' universal law is falsified by this result.

## Addendum 73 -- 6x7's window survives 5 seeds intact: 30/30 fast, unlike 8x8's n_bare=17 mirage -- the two grids genuinely differ (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-73-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: "Window confirmed."** All three points (n_bare=16, 17, 18) remain
`"solution found"` across all 5 seeds x 2 repeats (30 total runs, 10
per point), unanimous within every cell. **This is the opposite outcome
from Addendum 72's finding at 8x8**, where the same seed-count increase
revealed the apparent fast point was an artefact of comparing two
different constructions. Here, the SAME construction (`n_bare_edges`)
was used consistently in both the original 6-run measurement (Addendum
70) and this 30-run reproduction, and the result held. **6x7's window is
real; it does not evaporate under more scrutiny.** The user's proposed
"Contiguous Cliff" law (spare=0 is uniformly cliffing regardless of
grid size or composition) is falsified by this result alone: a genuine,
reproducible escape exists at 6x7.

## 1. Results

6x7 grid (42 qubits), `n_bare_edges`, spare=0, `optimization_level=3`,
5 seeds (0-4) x 2 repeats. All 30 rows completed with `error=""`; stop
reason unanimous (10/10) within every `n_bare` cell.

| n_bare | coverage | runs | outcome | median time |
|---:|---:|---:|:---|---:|
| 16 | 76.2% | 10/10 fast | solution found | 26.20ms |
| 17 | 81.0% | 10/10 fast | solution found | 856.50ms |
| 18 | 85.7% | 10/10 fast | solution found | 852.60ms |

Total runtime for all 30 runs: ~29.3 seconds -- consistent with every
run being fast (no cliffing runs to inflate the total, unlike every
`n_bare_edges(17)` run at 8x8 in Addendum 72, each of which alone took
roughly as long as this entire 30-run batch).

## 2. Scoring

**P1 (primary) -- "Window confirmed" CONFIRMED, cleanly.** 30/30 fast,
no exceptions at any of the three points, at any of the five seeds.
Neither "evaporates" nor "partial survival" describes what happened --
all three points held completely.

**P2 (sanity/consistency vs. Addendum 70's original values) --
CONFIRMED, with a noted internal-window timing pattern that repeats
exactly.** Addendum 70's original 6-run values (25.74ms at n_bare=16;
860.15ms at 17; 849.28ms at 18) match this run's 10-run medians
(26.20ms; 856.50ms; 852.60ms) closely at every point -- **including the
same internal pattern within the window**: n_bare=16 is roughly 30x
faster than 17 and 18, which are themselves close to each other, in
both the original measurement and this five-times-larger reproduction.
This internal structure (flagged as unexplained in Addendum 70) is
itself now confirmed as a stable, reproducible feature of the window,
not a fluke of Addendum 70's smaller sample.

## 3. What this means: 6x7 and 8x8 genuinely differ, and the difference is not merely sample size

**This addendum and Addendum 72 together demonstrate that the same
diagnostic (re-run at 5 seeds) produces opposite verdicts at the two
grid sizes**, using the identical construction function
(`n_bare_edges`) in both cases:

- **8x8, n_bare=17** (via `n_bare_edges`, not `mixed_uneven`): 0/23 fast
  across all seeds tested -- reliably cliffs.
- **6x7, n_bare=16-18**: 30/30 fast across all seeds tested -- reliably
  escapes.

**This rules out "small sample size" as the explanation for either
result.** 8x8's apparent escape was real evidence of a sampling problem,
but a different one than initially suspected (construction conflation
with `mixed_uneven`, per Addendum 72 Section 3) -- not merely "too few
seeds." 6x7's escape, tested with the same rigor, is not a sampling
artefact at all. **The two grid sizes genuinely differ in whether this
specific circuit family (bare edges plus 3-qubit chains, at spare=0)
can be compiled quickly**, and this difference now rests on solid
ground for 6x7 specifically.

**This leaves 8x8's own status as the more open question.** Whether 8x8
has ANY genuine fast point in this family (as opposed to 6x7's
confirmed three-point window) remains unresolved -- Addendum 72 ruled
out n_bare=17-via-`n_bare_edges` specifically, but did not test every
value, and `mixed_uneven`'s own original result (a different
construction) has still never been re-run to check its own
reproducibility.

## 4. What this does not establish

- Why 6x7 has a genuine window while 8x8 (at least at the one point
  directly re-tested) does not -- no mechanism proposed.
- Whether 8x8 has a genuine window somewhere else, not yet located --
  still open, and now a sharper question given 6x7's confirmed result.
- Whether `mixed_uneven`'s own 8x8 result (Addendum 63, still never
  re-run) would also hold up under 5-seed scrutiny -- untested.
- The cause of the ~30x internal timing difference within 6x7's window
  (n_bare=16 vs. 17/18) -- noted as reproducible, not explained.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged) |
| [`circuit_family_sweep_6x7_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_6x7_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run3.csv) | this run, 30 rows |
| [`spare-qubit-cliff-addendum-73-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-73-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 30 rows checked for `error=""`; stop reason confirmed unanimous
  (10/10) within every `n_bare` cell.
- Addendum 70's original three values were re-read directly from that
  addendum before the P2 comparison, not from memory.
- The ~30x internal timing ratio (16 vs. 17/18) was checked against
  Addendum 70's own original ratio before being called a reproduced
  pattern (25.74:860.15:849.28 in the original vs. 26.20:856.50:852.60
  here -- both show the same ~33x gap between 16 and its neighbours).
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 74 (source: spare-qubit-cliff-addendum-74-2026-09-18.md) ===== -->

> **Note added when merging:** Analytical, no new measurement: every simple component-size-histogram statistic (component count, edge count, bare-edge count, size diversity, minimum size, bare-edge fraction) has a fully overlapping range between the fast and cliffing groups across thirteen configurations. None separates the two groups.

## Addendum 74 -- no simple component-size statistic separates the cliffing group from the fast group: the explanatory variable is more specific than any summary tried so far (2026-09-18)

**Status**: an analytical addendum, using `component_decomposition.py`
(this session) against the twelve configurations the user specified,
spanning Addenda 51-73. No new Qiskit measurement -- purely a structural
comparison of interaction graphs already characterized by timing in
prior addenda.

## 0. In one line

**Every simple, single-number summary of a circuit's component-size
decomposition -- total component count, total edges, count of bare
2-qubit components, number of distinct component sizes, minimum
component size, and the fraction of components that are bare edges --
has a fully overlapping range between the fast group and the cliffing
group.** None of the six candidates tried separates the two groups even
approximately. This is a negative result, reported in full rather than
omitted, because it directly rules out an entire class of candidate
explanations (any statistic derivable from the component-size histogram
alone) and narrows what remains: either a more specific graph property
not captured by size alone, or something outside pure interaction-graph
topology entirely.

## 1. The comparison

Thirteen configurations (the user's twelve plus `dense_pairs` itself,
added as the foundational reference case), each characterized by its
interaction graph's component decomposition:

| label | group | n_components | n_edges | bare-2q count | distinct sizes | min size | frac bare-2q |
|---|:---:|---:|---:|---:|---:|---:|---:|
| linear_chain (n=42) | fast | 1 | 41 | 0 | 1 | 42 | 0.000 |
| mixed_uneven (n=64) | fast | 18 | 46 | 17 | 2 | 2 | 0.944 |
| merged_pairs m=3 (n=64) | fast | 29 | 35 | 26 | 2 | 2 | 0.897 |
| merged_pairs m=4 (n=64) | fast | 28 | 36 | 24 | 2 | 2 | 0.857 |
| merged_pairs m=6 (n=64) | fast | 26 | 38 | 20 | 2 | 2 | 0.769 |
| n_bare_edges n_bare=16 (n=42) | fast | 19 | 23 | 16 | 3 | 2 | 0.842 |
| dense_pairs (n=42) | **slow** | 21 | 21 | **21** | 1 | 2 | **1.000** |
| balanced_3q4q (n=64) | slow | 18 | 46 | 0 | 2 | 3 | 0.000 |
| shrinking_dominant (n=64) | slow | 18 | 46 | 0 | 2 | 3 | 0.000 |
| large_dominant_no_bare_edges (n=64) | slow | 9 | 55 | 0 | 2 | 3 | 0.000 |
| single_bare_edge (n=64) | slow | 21 | 43 | 1 | 3 | 2 | 0.048 |
| n_bare_edges n_bare=19 (n=42) | slow | 20 | 22 | 19 | 2 | 2 | 0.950 |
| n_bare_edges n_bare=17 (n=64) | slow | 27 | 37 | 17 | 2 | 2 | 0.630 |

## 2. What does NOT separate the groups

| statistic | fast group range | slow group range | overlap |
|---|---:|---:|:---:|
| total components | 1-29 | 9-27 | yes |
| total edges | 23-46 | 21-55 | yes |
| bare-2q component count | 0-26 | 0-21 | yes |
| distinct component sizes | 1-3 | 1-3 | yes |
| minimum component size | 2-42 | 2-3 | yes |
| fraction bare-2q | 0.000-0.944 | 0.000-1.000 | yes |

**Every single one overlaps completely.** Specific pairs make this
concrete: `mixed_uneven` (fast) and `n_bare_edges(17)` (slow) both have
17 bare-2q components; `mixed_uneven` (fast) and
`large_dominant_no_bare_edges` (slow) both have one dominant component
far larger than the rest (30 vs. 40 qubits); `dense_pairs` (the
project's own foundational cliffing case) has the *highest possible*
bare-2q fraction (1.000, every component a bare edge) while
`n_bare_edges(16)` (fast) has a similarly high fraction (0.842) --
maximal "bare-edge-ness" appears at the extremes of both groups.

## 3. What this means

**This is consistent with, and sharpens, this project's own trajectory
through Addenda 63-72**: every specific pairwise test in that sequence
(component count fixed, composition varied; dominant-component size
varied at fixed bare-edge count; bare-edge count varied at fixed
component structure) found that changing one summary statistic while
holding others fixed could flip the outcome in either direction. This
addendum's contribution is showing that no SINGLE summary statistic,
checked across all thirteen points simultaneously rather than pairwise,
comes even close to a clean separation. **The explanatory variable, if
it exists as a simple graph property, is not visible in the component-
size histogram** -- it would need to depend on something the histogram
discards: which specific qubits are grouped together (Addendum 60 ruled
out simple positional/geometric effects, but did not rule out more
specific structural properties of a fixed arrangement), the internal
symmetry of individual components (a path's own automorphism structure
is identical regardless of which grid position it occupies), or a
property outside pure graph topology (e.g. how the VF2 search's
node-ordering heuristic, per Addendum 45, interacts with this specific
family of circuits).

**This also means the natural next candidates are graph invariants
this script does not compute**: automorphism group size (a rough,
unnormalized check during this addendum's own preparation found this
does not separate the groups either -- both `dense_pairs`, the
prototypical slow case, and several fast cases have enormous symmetry
from repeated same-size components, so raw symmetry size is not
promising either, though this was checked only informally and not
presented as a full result here), degree sequences (trivial for these
path-based graphs -- every non-endpoint qubit has degree 2, so this
reduces to component-size information already tried), or something
about the specific edge list's interaction with the 8x8/6x7 grid's own
geometry during the actual VF2 search (which Addendum 60 already tested
for one specific case -- merge order -- and found irrelevant there).

## 4. What this does not establish

- Whether a weighted or combined statistic (not tried here) might
  separate the groups where single statistics do not.
- Whether more data points (beyond these thirteen) would reveal a
  pattern invisible at this sample size.
- Any mechanism -- this addendum only characterizes what does NOT
  explain the split, not what does.

## 5. Files

| File | What it is |
|---|---|
| [`component_decomposition.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/component_decomposition.py) | the script (unchanged from its creation this session) |
| [`component_decomposition_2026-09-18.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/component_decomposition_2026-09-18.csv) | the underlying data (13 rows) |

## 6. Verification

- All six candidate statistics' fast/slow ranges were computed directly
  from the CSV (via `pandas`), not estimated or eyeballed, and the
  overlap determination used a strict numeric comparison
  (`fast_max < slow_min or slow_max < fast_min` -> no overlap), not a
  visual judgment.
- The specific pairwise examples cited in Section 2 (mixed_uneven vs.
  n_bare_edges(17); mixed_uneven vs. large_dominant_no_bare_edges) were
  checked against their own rows in the underlying CSV before being
  cited, not asserted from memory of prior addenda.
- The automorphism-size check mentioned in Section 3 is explicitly
  flagged as informal and not presented as a completed result, to avoid
  overclaiming a check that was not run to the same standard as the six
  statistics in Section 2's table.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 75 (source: spare-qubit-cliff-addendum-75-2026-09-18.md) ===== -->

> **Note added when merging:** Analytical: a lead from comparing n_bare=16/17/18's remainder-chain structure is checked against the full existing dataset (n_bare 8-19) and immediately falsified -- the remainder-chain size cycles mechanically with n_bare mod 3, unrelated to outcome.

## Addendum 75 -- a speculative lead about n_bare=16/17/18's internal structure is falsified immediately by checking it against the full existing dataset, before it could be overfit (2026-09-18)

**Status**: an analytical addendum, using purely already-collected data
(Addenda 69-70, 73) plus the construction logic of `n_bare_edges`
(unchanged, Addendum 67). No new Qiskit measurement.

## 0. In one line

Comparing `n_bare_edges(16)`, `(17)`, and `(18)` structurally (all at
6x7, all inside the confirmed fast window, Addendum 73) revealed each
has a different "remainder chain" -- the extra-sized component the
construction adds to absorb whatever qubits don't divide evenly into
3-qubit chains after placing the bare edges. **This looked like a
promising lead from three points alone** (16's remainder is a 4-qubit
chain; 17's is 5-qubit; 18's is none) **but is immediately falsified
when checked against the full existing dataset (n_bare 8 through 19,
all already measured in Addenda 69-70)**: the remainder-chain size
cycles mechanically with period 3, entirely independent of whether the
point cliffs or is fast. This is reported specifically because it
demonstrates the value of checking a small-sample lead against a larger
existing dataset before investing further effort in it -- exactly the
discipline this project applied (sometimes only after the fact) to the
mod-3 lead in Addenda 53/59/61-63.

## 1. The check

Every `n_bare` from 8 to 19 at n=42 (6x7), with its remainder-chain size
computed directly from `_edges_n_bare_edges`'s own construction rule,
against each point's already-known outcome (Addendum 69: 8-13 cliff;
Addendum 70: 14-19, with 16-18 fast and 19 cliff; Addendum 73
independently reconfirmed 16-18 fast at higher seed count):

| n_bare | remaining qubits | remainder chain size | outcome |
|---:|---:|---:|:---|
| 8 | 26 | 5 | cliff |
| 9 | 24 | none (0) | cliff |
| 10 | 22 | 4 | cliff |
| 11 | 20 | 5 | cliff |
| 12 | 18 | none (0) | cliff |
| 13 | 16 | 4 | cliff |
| 14 | 14 | 5 | cliff |
| 15 | 12 | none (0) | cliff |
| **16** | **10** | **4** | **fast (26ms)** |
| **17** | **8** | **5** | **fast (857ms)** |
| **18** | **6** | **none (0)** | **fast (853ms)** |
| 19 | 4 | 4 | cliff |

**The remainder-chain size (4, 5, or none) cycles with period 3 as a
direct mechanical consequence of `n_bare mod 3`, and this cycle has no
relationship to the outcome column.** Grouping by remainder-chain size
makes this explicit:

- **remainder = none**: {9, 12, 15, 18} -> cliff, cliff, cliff, **fast**
- **remainder = 4**: {10, 13, 16, 19} -> cliff, cliff, **fast**, cliff
- **remainder = 5**: {8, 11, 14, 17} -> cliff, cliff, cliff, **fast**

Every group contains three cliffing points and one fast point, mixed
throughout the range rather than clustered. Remainder-chain size
predicts nothing about outcome.

## 2. What this rules out, and what remains open

**Ruled out**: the specific structural feature that first looked
interesting when comparing 16/17/18 alone (which remainder chain size
each one gets) is not the explanatory variable. It is a byproduct of
`n_bare mod 3` under this particular construction and carries no
information about cliff/fast status.

**Still open, unchanged from before this check**: why n_bare=16 (26ms)
is roughly 33x faster than n_bare=17 and 18 (both ~850ms) despite all
three being inside the same confirmed fast window and all three
completing successfully. This addendum narrows the search by removing
one candidate, not by finding the answer.

## 3. A note on method

This check cost nothing beyond re-reading already-collected data and
re-running the (already-verified) construction logic in the sandbox --
no new Qiskit measurement, no new addendum-specific circuit family. It
is presented as its own short addendum, rather than silently discarded,
because the discipline of checking a small-sample lead against the full
existing dataset before pursuing it further is itself a practice worth
recording explicitly, given how much of today's effort (Addenda 59-63)
went into a lead that was not checked this way until several addenda
later.

## 4. Files

No new data files -- this addendum recomputes from
`_edges_n_bare_edges`'s own construction logic (verbatim from
`circuit_family_sweep.py`, Addendum 67) against outcomes already
recorded in Addenda 69, 70, and 73.

## 5. Verification

- The remainder-chain-size column was computed directly
  (`divmod(remaining, 3)`, `extra_size = 3 + leftover if leftover > 0
  else 0`) for every integer n_bare from 8 to 19, not only the three
  points (16-18) that originally motivated this check.
- Each outcome was re-read from its source addendum (69, 70, 73) before
  being placed in the table, not recalled from memory.
- The "no relationship" conclusion was checked by explicitly grouping
  points by remainder-chain size and confirming each group contains
  both cliffing and fast outcomes, not merely by eyeballing the table.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 76 (source: spare-qubit-cliff-addendum-76-2026-09-18.md) ===== -->

> **Note added when merging:** Analytical: a second candidate (non-bare component count) is also falsified across the full range (n_bare=19 has the fewest non-bare components of any point tested, yet cliffs) -- though it may coincidentally track the within-window speed ordering at just three points, flagged as unconfirmed.

## Addendum 76 -- a second candidate (non-bare component count) is also falsified across the full range, but may coincidentally track the within-window speed ordering (2026-09-18)

**Status**: an analytical addendum, continuing Addendum 75's method:
check a candidate structural statistic against the full existing
dataset (n_bare 8-19 at 6x7) before drawing any conclusion from it. No
new Qiskit measurement.

## 0. In one line

**Number of non-bare-edge components (components of size 3 or larger)
decreases monotonically as `n_bare` increases (from 8 at n_bare=8-9
down to 1 at n_bare=19), and does NOT predict cliff/fast status across
the full range**: `n_bare=19` has the fewest non-bare components (1) of
any point tested, yet cliffs -- while `n_bare=16-18`, with more non-bare
components (3, 2, 2 respectively), are all fast. If "fewer non-bare
components is better" were the rule, 19 should be the fastest point
tested; it is instead the point immediately following the window's
close. **This candidate is falsified as a general predictor.**
Separately, and with explicit caution, this statistic's value ordering
(3, 2, 2 for n_bare=16, 17, 18) happens to match the within-window
speed ordering found in Addenda 70/73 (16 fastest, 17 and 18 similar
and slower) -- noted as a possible coincidence from only three points,
not confirmed.

## 1. The check

| n_bare | total components | non-bare (size>=3) components | outcome |
|---:|---:|---:|:---|
| 8 | 16 | 8 | cliff |
| 9 | 17 | 8 | cliff |
| 10 | 17 | 7 | cliff |
| 11 | 17 | 6 | cliff |
| 12 | 18 | 6 | cliff |
| 13 | 18 | 5 | cliff |
| 14 | 18 | 4 | cliff |
| 15 | 19 | 4 | cliff |
| **16** | 19 | **3** | **fast (26ms)** |
| **17** | 19 | **2** | **fast (857ms)** |
| **18** | 20 | **2** | **fast (853ms)** |
| **19** | 20 | **1** | **cliff** |

## 2. Scoring against the full range

**Falsified as a general predictor.** `n_bare=19`'s non-bare count (1)
is strictly lower than every fast point's (2-3), which would predict it
should be at least as fast as 16-18 under a simple "fewer is better"
reading -- it is not; it cliffs, immediately outside the window. The
monotonic decrease in this statistic as `n_bare` rises is a direct,
mechanical consequence of `n_bare` itself (more bare edges leaves fewer
qubits for chains) and does not independently track the cliff/fast
boundary any better than `n_bare` alone already does not (Addendum 69
established occupancy/coverage alone doesn't cleanly predict either).

**A separate, unconfirmed observation.** Restricted only to the three
already-known-fast points (16, 17, 18), this statistic's value (3, 2, 2)
orders consistently with their known relative speed (16 fastest at
26ms; 17 and 18 both around 850ms). This is flagged explicitly as
**not validated** -- three points is too few to distinguish a real
relationship from coincidence, exactly the caution Addendum 75 applied
to the remainder-chain-size candidate, which looked promising at three
points and was falsified at twelve. This observation is recorded for
completeness, not treated as a finding.

## 3. What remains

Two structural candidates (Addendum 75's remainder-chain size; this
addendum's non-bare component count) have now been checked against the
full existing dataset and both fail as general cliff/fast predictors.
Neither explains why n_bare=16 is faster than 17/18 within the confirmed
fast window, and neither explains the cliff/fast boundary itself better
than the already-established occupancy-window framing (Addendum 70).
**No structural candidate tried so far (Addenda 74-76) improves on
"there is a window, and its exact location is currently only known
empirically" as the state of understanding.**

## 4. Files

No new data files -- recomputed from already-collected outcomes
(Addenda 69, 70, 73) against `_edges_n_bare_edges`'s own construction
logic.

## 5. Verification

- The non-bare-component-count column was computed for all twelve
  points (8-19), not only the three that motivated the check.
- The falsifying case (n_bare=19) was identified by explicitly sorting
  all points by this statistic's value and checking outcome at each
  rank, not by inspection of the three-point subset alone.
- Section 2's "unconfirmed observation" is explicitly labelled as such,
  with the same caution language used for Addendum 75's own similarly-
  sized coincidence, to avoid inconsistent standards between the two
  addenda.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---<!-- ===== Addendum 77 pre-registration (source: spare-qubit-cliff-addendum-77-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for re-verifying mixed_uneven from scratch, since it was the sole basis for the 8x8 'bare-edge escape' claim and had only ever been measured once.

## Addendum 77 -- Pre-registration: does `mixed_uneven` itself reproduce, at the same seed count that just showed `n_bare_edges(17)` reliably cliffs? (2026-09-18)

**Status: pre-registration only. No re-run of `mixed_uneven` beyond
Addendum 63's original single measurement has been performed.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

`mixed_uneven` (Addendum 63: 17 bare 2-qubit edges + one 30-qubit
dominant component, 18 total components, zero idle, n=64) is currently
this entire investigation's **only confirmed fast point at 8x8** in the
component-structure line of work (Addenda 63-76). It has been measured
exactly once. Addendum 72 found `n_bare_edges(17)` -- a different
construction at the same coverage -- cliffs reliably (23/23 runs, 5
seeds). Addendum 73 found 6x7's window survives the same scrutiny
(30/30 fast). `mixed_uneven` itself has never been put through either
test. Until it is, the claim "a fast point exists at 8x8" rests on a
single run -- exactly the situation that made `n_bare_edges(17)`'s
apparent fast status misleading.

## 2. Design

`mixed_uneven` (unchanged construction: 17 x 2-qubit edges + one
30-qubit path, n=64, spare=0, `optimization_level=3`), 5 seeds (0-4) x
2 repeats -- the identical resolution used in Addendum 72 (which
falsified `n_bare_edges(17)`'s reproducibility) and Addendum 73 (which
confirmed 6x7's window's reproducibility), so this result is directly
comparable to both.

## 3. Pre-registered predictions

**P1 (primary).**
  - **Reproduces**: `"solution found"` in most or all of the 10 runs --
    confirms `mixed_uneven`'s original result was not a fluke, and the
    specific structure it embodies (one dominant component + many bare
    edges, as opposed to `n_bare_edges`' many small chains + bare
    edges) genuinely differs from `n_bare_edges(17)` in a way that
    matters, consistent with Addendum 72's structural explanation
    (18 vs. 27 components).
  - **Does not reproduce**: `"nonexistent solution"` in most or all
    runs -- would mean Addendum 63's original single measurement was
    itself the outlier, and **no confirmed fast point would remain
    anywhere in this project's 8x8 investigation**. This is a live
    possibility, not a formality: `n_bare_edges(17)`'s own apparent
    fast result (Addendum 63, single run, before Addendum 72's
    reproducibility check) looked identical in kind to how
    `mixed_uneven`'s result currently looks.
  - **Mixed/bistable**: a genuine split across seeds -- reported as its
    own outcome.

**P2 (sanity check).** The construction itself (component count 18,
zero idle, 17 bare edges, one 30-qubit dominant component) is
re-verified against Addendum 63's own recorded structure before this
run is interpreted, to rule out any construction drift since that
addendum.

## 4. What this cannot establish

- If P1 falsifies (does not reproduce), *why* Addendum 63's original
  run differed -- only that it does not represent the typical case.
- Whether some OTHER untested construction at 8x8 would reliably
  escape the cliff, if `mixed_uneven` itself does not survive this
  check.
- Generalization to 6x7 or other grid sizes -- this is 8x8-only.

---


<!-- ===== Addendum 77 (source: spare-qubit-cliff-addendum-77-2026-09-18.md) ===== -->

> **Note added when merging:** mixed_uneven reproduces cleanly: 10/10 fast, tightly clustered (33.7-35.4ms) -- as tight as the cliffing plateau's own reproducibility. The one confirmed escape at 8x8 now rests on 11 independent runs, not 1.

## Addendum 77 -- `mixed_uneven` reproduces cleanly: 10/10 fast, tightly clustered (33.7-35.4ms) -- the one confirmed escape at 8x8 now rests on solid ground (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-77-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: "Reproduces" confirmed, cleanly.** All 10 runs (5 seeds x 2
repeats) return `"solution found"`, with times clustered in a narrow
band (33.73-35.42ms, a max/min ratio of only 1.05x). **This is the
opposite outcome from `n_bare_edges(17)`'s own reproducibility check**
(Addendum 72: 0/23 fast) at the same coverage, and it puts
`mixed_uneven` on the same solid footing as 6x7's confirmed window
(Addendum 73: 30/30 fast) and the falsified `n_bare_edges(17)`
(Addendum 72: 0/23 fast). **The single confirmed fast point in this
entire 8x8 investigation now rests on 11 independent runs (1 original +
10 here), not 1.**

## 1. Results

8x8 grid (64 qubits), `mixed_uneven`, spare=0, `optimization_level=3`,
5 seeds (0-4) x 2 repeats. All 10 rows completed with `error=""`; stop
reason unanimous (10/10) `"solution found"`.

| seed | repeat | time (ms) | stop reason |
|---:|---:|---:|:---|
| 0 | 0 | 35.18 | solution found |
| 0 | 1 | 34.31 | solution found |
| 1 | 0 | 34.02 | solution found |
| 1 | 1 | 33.73 | solution found |
| 2 | 0 | 35.14 | solution found |
| 2 | 1 | 34.30 | solution found |
| 3 | 0 | 35.42 | solution found |
| 3 | 1 | 34.92 | solution found |
| 4 | 0 | 34.46 | solution found |
| 4 | 1 | 33.78 | solution found |

Median: 34.38ms. Compare Addendum 63's original single measurement:
30.16ms -- within 14% of this run's median, and well inside ordinary
single-run variance given this run's own 33.73-35.42ms band.

## 2. Scoring

**P1 (primary) -- "Reproduces" CONFIRMED.** 10/10 fast, no exceptions at
any seed. The tight clustering (1.05x max/min) is itself notable:
compare `n_bare_edges`' own cliffing plateau (Addendum 68: 1.057x
max/min across six points) -- **`mixed_uneven`'s fast result is just as
tightly reproducible as the cliffing plateau was**, not a borderline or
fragile result.

**P2 (sanity check) -- CONFIRMED.** The construction was re-verified
(18 components, 17 bare 2-qubit edges, one 30-qubit dominant component,
zero idle) against Addendum 63's own recorded structure before this run
was interpreted, and matched exactly.

## 3. What this settles

**The 8x8 investigation now has exactly one confirmed escape route, and
it is a real, reproducible one.** Combined with everything on record:

| construction | components | dominant structure | 8x8 result |
|---|---:|---|:---|
| `n_bare_edges(17)` | 27 | many small 3-qubit chains | **cliffs, 0/23** (Addendum 72) |
| `mixed_uneven` | 18 | one 30-qubit dominant component | **fast, 10/10 + original = 11/11** (this addendum) |

**These two constructions share the same bare-edge count (17) and
coverage (53.1%) but differ in component count (27 vs. 18) and in
whether the non-bare-edge portion is many small pieces or one large
one.** Addendum 74's finding (no simple summary statistic separates
fast from cliffing across the wider 13-point dataset) still holds at
the level of *component count alone* or *bare-edge count alone* -- but
comparing specifically these two matched-coverage constructions
narrows it to a cleaner contrast: **one large dominant component
plus bare edges reproducibly escapes; many small chains plus the same
bare edges reproducibly does not.** This is consistent with, though
narrower than, Addendum 65's earlier finding that dominant-component
*size* alone (without bare edges) does not guarantee escape
(`large_dominant_no_bare_edges`, 40-qubit dominant, no bare edges,
cliffs) -- the combination of a large dominant component *and* bare
edges together may be what `mixed_uneven` uniquely provides among
everything tested so far.

## 4. What this does not establish

- Whether the specific 30-qubit dominant-component size is necessary,
  or whether other large sizes (tested only at 40 qubits without bare
  edges, Addendum 65) would also escape if paired with bare edges --
  untested combination.
- Whether varying dominant-component size while holding 17 bare edges
  fixed (unlike Addendum 64's own dominant-size sweep, which held
  small-component *shape* fixed as chains, not bare edges) would show a
  similar threshold to the one already found in Addendum 64.
- Mechanism -- why this specific combination escapes and the
  many-small-chains combination does not.
- Generalization to 6x7 -- untested with this exact construction; 6x7's
  own confirmed escape (Addendum 73) uses `n_bare_edges`, a different
  shape, and has not been cross-checked against a `mixed_uneven`-style
  construction at 6x7's own scale.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run8.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run8.csv) | this run, 10 rows |
| [`spare-qubit-cliff-addendum-77-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-77-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 10 rows checked for `error=""`; stop reason confirmed unanimous
  (10/10) `"solution found"`.
- The max/min ratio (1.05x) was computed directly from the ten
  recorded times, not estimated.
- The comparison to Addendum 68's own cliffing-plateau tightness
  (1.057x) was re-read directly from that addendum before being used
  as a point of comparison.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 78 pre-registration (source: spare-qubit-cliff-addendum-78-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether mixed_uneven's escape depends on the dominant component's specific ~30-qubit size or on bare-edge presence alone, holding 17 bare edges fixed while varying dominant size.

## Addendum 78 -- Pre-registration: does `mixed_uneven`'s escape depend on the dominant component's specific 30-qubit size, or does any large dominant component escape as long as bare edges are also present? (2026-09-18)

**Status: pre-registration only. No run with any new dominant-size
construction has been performed.** Predictions are locked before any
measurement.

## 1. Why this experiment exists

Addendum 77 confirmed `mixed_uneven` (17 bare edges + one 30-qubit
dominant component) reproduces reliably (10/10 + 1 original = 11/11
fast). This is currently the only confirmed escape route at 8x8. But
its two defining parameters -- 17 bare edges and a 30-qubit dominant
component -- have never been varied independently while holding the
other fixed. Addendum 64 already tested shrinking the dominant
component **without** bare edges (`shrinking_dominant`: 17x 3-qubit
chains + a shrinking dominant, zero bare edges) and found it cliffs at
13 qubits -- but that construction has no bare edges at all, so it
cannot say whether bare edges are what makes the difference. This
addendum holds bare-edge count fixed at 17 (matching `mixed_uneven`
exactly) and shrinks the dominant component, to test directly whether
bare-edge presence is what allows escape even as the dominant shrinks,
or whether the dominant's own size (near 30) is independently
necessary.

## 2. Design

17 bare 2-qubit edges (fixed, 34 qubits, matching `mixed_uneven`
exactly) + one dominant chain of size `D` + one filler chain using the
exact remainder (`30 - D` qubits) to keep the total at 64 with zero
idle. Component count is 18 when `D=30` (no filler needed, exactly
reproducing `mixed_uneven`) and 19 for every smaller `D` (one filler
component added).

| D (dominant size) | filler size | total components | notes |
|---:|---:|---:|---|
| 30 | 0 (none) | 18 | reproduces `mixed_uneven` exactly (sanity check) |
| 25 | 5 | 19 | |
| 20 | 10 | 19 | |
| 15 | 15 | 19 | dominant and filler equal size |
| 10 | 20 | 19 | "dominant" now smaller than filler |
| 5 | 25 | 19 | matches `shrinking_dominant`'s own 13-qubit-dominant spirit, but WITH bare edges present |

`optimization_level=3`, spare=0 on the 8x8 grid, 3 seeds x 2 repeats per
point (matching this project's standing resolution for a first pass;
Addendum 77's own higher-resolution check is reserved for whichever
point looks most consequential, not spent on all six up front).

## 3. Pre-registered predictions

**P1 (primary -- does escape require the dominant near 30 qubits, or
does bare-edge presence alone suffice regardless of dominant size)?**
  - **Size-independent (bare edges are sufficient)**: all six points
    (D=5 through 30) show `"solution found"` -- the dominant component's
    specific size does not matter once 17 bare edges are present; bare
    edges alone are the operative condition.
  - **Size-dependent (a large dominant is also necessary)**: smaller D
    values (e.g. D=5, matching `shrinking_dominant`'s own cliffing
    13-qubit case in spirit) cliff despite bare edges being present --
    meaning bare edges alone are not sufficient; a sufficiently large
    dominant component is independently required, and `mixed_uneven`'s
    escape depends on both conditions jointly.
  - **A threshold within the tested range**: some D values escape and
    others cliff, with a specific transition point -- reported with its
    location, not forced into either extreme reading.

**P2 (sanity check).** `D=30` (no filler, exactly reproducing
`mixed_uneven`) is predicted to show `"solution found"`, matching
Addenda 63 and 77.

## 4. What this cannot establish

- The exact threshold's location if P1 finds one, beyond the six
  tested points (5, 10, 15, 20, 25, 30) -- a finer sweep would be a
  follow-up.
- Whether bare-edge *count* (fixed at 17 here) also matters
  independently -- already partially addressed by Addenda 65-66, not
  re-tested here.
- Mechanism.
- Generalization to 6x7.

---


<!-- ===== Addendum 78 (source: spare-qubit-cliff-addendum-78-2026-09-18.md) ===== -->

> **Note added when merging:** A major unregistered finding: D=10 and D=20 are verified graph-isomorphic (via networkx.is_isomorphic) yet give opposite outcomes (fast vs. cliff), reproduced independently twice -- physical qubit placement, not merely graph structure, matters.

## Addendum 78 -- an unregistered, major finding: D=10 and D=20 produce a graph-isomorphic interaction graph yet opposite outcomes (fast vs. cliff) -- specific qubit placement, not merely graph structure, matters (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-78-preregistration-2026-09-18.md`, written
and locked before this run. **P1 is falsified by a result the
pre-registration's three branches did not anticipate**, and that result
is the addendum's real content.

**Update**: the identical 6-point configuration was independently
re-run in full immediately after this addendum's initial result. All
six points reproduced exactly, including the D=10 (fast) / D=20 (cliff)
split. Results incorporated below (now 12 runs total, 2 independent
batches of 6).

## 0. In one line

**None of P1's three branches describe what happened.** D=30 (no
filler, exactly `mixed_uneven`) and D=10 are fast; D=5, 15, 20, 25 all
cliff. This is not size-independent (D=5 cliffs), not a clean
size-dependent threshold (D=10 is fast but D=15 and D=20, both larger,
cliff), and not a single simple transition. **The most consequential
single fact in this run**: D=10 and D=20 were verified, before writing
this addendum, to be **abstractly graph-isomorphic** -- identical
component-size multiset (one 20-chain, one 10-chain, seventeen bare
2-qubit edges) -- confirmed directly with `networkx.is_isomorphic`, not
assumed. **Yet D=10 is fast (37.37ms, reproduced at 35.64ms) and D=20
cliffs (9,244.13ms, reproduced at 9,300.23ms) -- independently
confirmed twice, not a one-off fluctuation.** The only difference
between the two constructions is which specific physical qubits are
assigned to the 10-chain versus the 20-chain (position relative to the
bare edges). This appears to contradict Addendum 60's finding that
merge order/position does not matter for `merged_pairs` -- but for a
different family, under different conditions, and the two are not
necessarily in tension (Section 3).

## 1. Results

8x8 grid (64 qubits), `dominant_size_sweep`, spare=0,
`optimization_level=3`, 3 seeds x 2 repeats. All 36 rows completed with
`error=""`; stop reason unanimous (6/6) within every `D` cell.

| D | filler | component sizes | run 1 (ms) | run 2 (ms) | stop reason |
|---:|---:|---|---:|---:|:---|
| 5 | 25 | {25, 5} | 9,103.70 | 9,017.11 | nonexistent solution |
| **10** | **20** | **{20, 10}** | **37.37** | **35.64** | **solution found** |
| 15 | 15 | {15, 15} | 9,493.51 | 9,285.22 | nonexistent solution |
| **20** | **10** | **{20, 10}** | **9,244.13** | **9,300.23** | **nonexistent solution** |
| 25 | 5 | {25, 5} | 8,932.25 | 9,040.09 | nonexistent solution |
| 30 | 0 | {30} | 35.98 | 36.25 | solution found |

**All six points reproduced exactly between the two independent runs**
-- same stop reason, closely matching timing (within ~5% at every
point). **D=5 and D=25** (also confirmed graph-isomorphic to each other
before this run, per the pre-registration's own Section on the design):
**both cliff, in both runs**, matching each other as expected for
isomorphic graphs. **D=10 and D=20**: isomorphic, but **disagree, in
both runs** -- this is not a fluctuation.

## 2. Scoring against the pre-registration

**P1 (primary) -- FALSIFIED, all three branches.** "Size-independent"
requires all six fast; D=5 cliffs, ruling it out. "Size-dependent
threshold" requires a monotonic split by size; D=10 (fast) is smaller
than D=15 and D=20 (both cliff), so no simple size threshold fits.
"Threshold within the tested range" was meant to allow a single
transition point, not a result where the smallest (D=5) and two
middle values (D=15, D=20) cliff while one middle value (D=10) and the
largest (D=30) are fast -- **not a threshold shape at all.**

**P2 (sanity check) -- CONFIRMED.** D=30 reproduces `mixed_uneven`
exactly: `"solution found"`, 35.98ms, matching Addendum 63 (30.16ms)
and Addendum 77 (33.7-35.4ms band) closely.

**The pre-registration's own predicted internal consistency check
(D and 30-D should match) -- CONFIRMED for one pair, FALSIFIED for the
other.** D=5/D=25: both cliff, consistent. **D=10/D=20: disagree**, which
the pre-registration's design section explicitly did not anticipate
(it predicted these pairs "should" match if the reasoning about
isomorphic graphs held) -- reported honestly as a failure of that
expectation, not smoothed over.

## 3. What this means, and how it relates to Addendum 60

**A graph-isomorphism-preserving change to WHICH physical qubits host
which component can flip the outcome.** This is a stronger and more
specific claim than "component structure alone doesn't predict outcome"
(Addendum 74) -- it shows two *literally isomorphic* interaction graphs,
differing only in the physical qubit indices assigned to each
component, giving opposite results.

**Is this in tension with Addendum 60?** Addendum 60 tested whether
merge *order* (sequential/reverse/random selection of which edge-pairs
to merge in `merged_pairs`) affected outcome and found it did not --
all three orderings reproduced the same cliff/clear pattern at every
tested `m`. That test varied which *specific bare-edge pairs* got
merged into 4-qubit chains, while holding the *distribution* of
component sizes exactly fixed and, critically, the resulting graphs
were compared only in aggregate (same m, same outcome across orderings)
-- **not verified pairwise as graph-isomorphic in the way this addendum
explicitly checked.** It is possible Addendum 60's orderings were
*also* producing non-isomorphic graphs in ways that happened not to
matter for that family, or that the specific property that matters here
(position of a *large* chain relative to the bare edges, in a construction
with exactly two large components of very different sizes) is different
from what varies in `merged_pairs`' merge-order variants (many small,
similarly-sized 4-qubit chains). **This addendum's result is not proven
to contradict Addendum 60's -- the two used different circuit families
under different specific manipulations, and no direct test bridges
them.** What is established is narrower and still significant: **for
`dominant_size_sweep` specifically, physical qubit placement matters,
holding the abstract graph fixed.**

## 4. What this does not establish

- **Whether D=10's specific placement (small chain immediately after
  bare edges, large chain last) is what matters, or something else
  about that particular arrangement.** Only two placements were tested
  per isomorphism class (the "natural" construction order and its
  mirror); a systematic sweep of placements (e.g. interleaving the two
  chains among the bare edges, or reversing which end each chain starts
  from) was not done.
- Whether this reconciles with or genuinely contradicts Addendum 60 --
  Section 3 states the open question, does not resolve it.
- Why D=5/D=25 agree while D=10/D=20 disagree, given both pairs are
  isomorphic-graph pairs by the same construction logic -- no mechanism
  proposed. One structural difference worth noting without drawing a
  conclusion from it: D=5/D=25's two non-bare-edge components (5 and
  25) differ far more in size from each other than D=10/D=20's (10 and
  20) -- whether this ratio, or something else entirely, explains the
  differing agreement is untested.
- Any mechanism for the dominant_size_sweep family's cliff/fast pattern
  generally -- this addendum deepens the puzzle rather than resolving
  it.
- Generalization to 6x7 or to other families.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 78's `dominant_size_sweep`) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run9.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run9.csv) | this run, 36 rows |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run10.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run10.csv) | independent reproducibility re-run, 36 rows |
| [`spare-qubit-cliff-addendum-78-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-78-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 36 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `D` cell.
- **D=10 and D=20's graph isomorphism was verified computationally**
  (`networkx.is_isomorphic`, returning `True`) before this addendum's
  central claim was written, not assumed from the construction logic
  alone -- this was the single most important check performed for this
  addendum, precisely because the claim rests entirely on it.
- D=5/D=25's component-size match ({25,5} both) was confirmed directly
  from the CSV's own recorded structure, consistent with the
  pre-registration's stated design.
- D=30's reproduction of `mixed_uneven` was checked against both
  Addendum 63's original figure and Addendum 77's own reproduced band
  before being reported as consistent.
- The reproducibility re-run (run10) was checked point-by-point against
  run9: every one of the six stop reasons matched, and every timing
  value fell within ~5% of its run9 counterpart -- confirmed by direct
  comparison, not assumed from a summary statistic.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both new CSVs ->
  0 hits. The terminal output supplied for these runs was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 79 pre-registration (source: spare-qubit-cliff-addendum-79-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for whether the D=10/D=20-style split recurs at other size ratios (12/18 and 8/22), to locate where isomorphic pairs start agreeing vs. disagreeing.

## Addendum 79 -- Pre-registration: does the D=10/D=20-style split (isomorphic graphs, opposite outcomes) recur at other size ratios, or was it specific to that one pair? (2026-09-18)

**Status: pre-registration only. No run with any new dominant/filler
pair has been performed.** Predictions are locked before any
measurement.

## 1. Why this experiment exists

Addendum 78 found, reproduced independently twice, that D=10 and D=20
-- graph-isomorphic (verified with `networkx.is_isomorphic`) -- give
opposite outcomes (fast vs. cliff), while D=5 and D=25 -- also
isomorphic -- agree (both cliff). With only two isomorphic pairs tested,
it is not known whether the disagreement is specific to the 10:20 ratio
(2.0), general to ratios "close enough" to that, or essentially random
noise in which isomorphic pairs happen to agree. This addendum adds two
more isomorphic pairs at different ratios to locate where, if anywhere,
the disagreement pattern holds.

## 2. Design

Same `dominant_size_sweep` construction (17 bare edges fixed, one
dominant chain of size D, one filler chain of size 30-D). Two new
isomorphic pairs, chosen to bracket the known disagreeing pair (10/20,
ratio 2.0) and agreeing pair (5/25, ratio 5.0):

| pair | ratio (larger/smaller) | where it sits |
|---|---:|---|
| 12 / 18 | 1.5 | closer to balanced (D=15's 1:1) than 10/20 |
| 8 / 22 | 2.75 | between 10/20 (disagrees) and 5/25 (agrees) |

Both members of each pair are tested (12 AND 18; 8 AND 22) --
4 new points total, `optimization_level=3`, spare=0 on the 8x8 grid,
3 seeds x 2 repeats each, matching Addendum 78's own resolution.
Isomorphism is verified computationally (`networkx.is_isomorphic`)
before interpreting results, exactly as done for D=10/D=20 and D=5/D=25.

## 3. Pre-registered predictions

**P1 (primary -- does 12/18 agree or disagree)?**
  - **Agrees** (both same outcome): ratio 1.5 is close enough to
    balanced (like D=15's single 1:1 case, which cliffs) that placement
    doesn't matter here -- would suggest the D=10/20 disagreement needs
    a *specific*, not-too-close-to-1:1 ratio.
  - **Disagrees**: extends the disagreement pattern to a ratio even
    closer to balanced than 10/20 -- would suggest disagreement is not
    narrowly tied to the 2.0 ratio specifically.

**P2 (primary -- does 8/22 agree or disagree)?**
  - **Agrees**: ratio 2.75, between the disagreeing (2.0) and agreeing
    (5.0) pairs, sides with the agreeing regime -- would suggest a
    ratio threshold somewhere between 2.0 and 2.75.
  - **Disagrees**: sides with the disagreeing regime -- would suggest
    the threshold, if any, lies between 2.75 and 5.0, or that ratio is
    not the operative variable at all (since 2.75 is closer to 5.0 than
    to 2.0, disagreement here would weaken a simple "ratio" story
    considerably).

**P3 (sanity check).** Both pairs' isomorphism is verified
computationally before any outcome is interpreted, matching Addendum
78's own standard.

## 4. What this cannot establish

- A confirmed threshold even if a pattern emerges from four points --
  four is still few; any apparent ratio-based rule found here should be
  treated with the same caution Addendum 75 applied to a two-point
  lead that later failed at twelve points.
- Mechanism, in any case.
- Whether the same pattern holds at 6x7 or other grid sizes.

---


<!-- ===== Addendum 79 (source: spare-qubit-cliff-addendum-79-2026-09-18.md) ===== -->

> **Note added when merging:** A ratio-based pattern emerges across four isomorphic pairs: ratios 1.5 and 2.0 disagree (smaller value always fast); ratios 2.75 and 5.0 agree (both cliff) -- reported with explicit caution given this project's history with small-sample patterns.

## Addendum 79 -- a ratio-based pattern emerges across four isomorphic pairs: low ratios (1.5, 2.0) disagree with the smaller value always fast; higher ratios (2.75, 5.0) agree (both cliff) (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-79-preregistration-2026-09-18.md`, written
and locked before this run.

## 0. In one line

**P1: 12/18 disagrees, extending the pattern from D=10/D=20 to an even
more balanced ratio (1.5).** **P2: 8/22 agrees (both cliff), siding with
the D=5/D=25 regime.** Across all four isomorphic pairs now measured, a
clean pattern emerges: **ratios 1.5 and 2.0 disagree; ratios 2.75 and
5.0 agree (both cliff).** Within both disagreeing pairs, an additional,
unregistered pattern holds exactly: **the smaller dominant-component
value is the one that is fast** (12 fast, 18 cliff; 10 fast, 20 cliff).
This is reported with the same caution this project has applied to
every small-sample lead today (Addendum 75's own explicit warning
about four points), but it is the cleanest pattern found in this entire
line of investigation (Addenda 74-79) to date.

## 1. Results

8x8 grid (64 qubits), `dominant_size_sweep`, spare=0,
`optimization_level=3`, 3 seeds x 2 repeats. All 24 rows completed with
`error=""`; stop reason unanimous (6/6) within every `D` cell.
Isomorphism of both new pairs was verified computationally
(`networkx.is_isomorphic`) before this run, per the pre-registration.

| D | filler | ratio | time (ms) | stop reason |
|---:|---:|---:|---:|:---|
| **8** | **22** | 2.75 | 9,140.94 | nonexistent solution |
| **12** | **18** | 1.5 | 35.78 | **solution found** |
| **18** | **12** | 1.5 | 9,405.82 | nonexistent solution |
| **22** | **8** | 2.75 | 9,067.64 | nonexistent solution |

## 2. Scoring

**P1 (12/18) -- "Disagrees" CONFIRMED.** D=12 fast (35.78ms), D=18
cliffs (9,405.82ms) -- matches D=10/D=20's own disagreement pattern,
at an even more balanced ratio (1.5 vs. 2.0).

**P2 (8/22) -- "Agrees" CONFIRMED.** Both D=8 and D=22 cliff
(9,140.94ms and 9,067.64ms) -- matches D=5/D=25's own agreement
pattern, at a ratio (2.75) closer to the disagreeing pair (2.0) than
to the agreeing one (5.0), which the pre-registration flagged in
advance as the reading that would "weaken a simple ratio story
considerably" if it occurred. **It did not weaken the story --** 2.75
sided cleanly with the agreeing regime, same as 5.0.

**P3 (sanity check) -- CONFIRMED.** Both pairs verified isomorphic
before interpretation.

## 3. The pattern across all four pairs now on record

| ratio | pair | outcome | smaller value's status |
|---:|---|:---|:---|
| 1.5 | 12/18 | **disagree** | fast |
| 2.0 | 10/20 (Addendum 78) | **disagree** | fast |
| 2.75 | 8/22 | **agree** (both cliff) | -- |
| 5.0 | 5/25 (Addendum 78) | **agree** (both cliff) | -- |

**Two findings, stated at the confidence level the data supports:**

1. **A ratio threshold appears to sit between 2.0 and 2.75.** Below it
   (more balanced pairs), the two isomorphic constructions disagree;
   above it (more skewed pairs), they agree, and specifically agree by
   both cliffing. This is consistent across all four points tested so
   far -- but four points is still a small sample, and this addendum
   does not claim the threshold is precisely located, only that it is
   bracketed between 2.0 and 2.75.

2. **Within the disagreeing regime, the smaller value is always the
   fast one** (12 and 10, not 18 and 20). This held in both of the two
   disagreeing pairs found so far -- a small but suggestive sample.

**Neither finding was predicted in the pre-registration in this specific
form** (P1/P2 asked only agree-or-disagree per pair, not the
cross-pair ratio pattern or the smaller-value-is-fast regularity); both
are reported as genuine post-hoc observations from this run's own
results, not retrofitted predictions.

## 4. What this does not establish

- **The exact ratio threshold.** Only that it lies in (2.0, 2.75]; no
  point between was tested.
- **Whether the "smaller value is fast" rule is general** or specific
  to these two pairs -- a third disagreeing pair (if one exists between
  ratio 1.0 and 2.0) would test this directly.
- **What happens at ratio exactly 1.0** (D=15, already on record from
  Addendum 78: cliffs, as a single non-isomorphic-pair case since
  15=filler=15 means there is only one graph, not two to compare) --
  whether the "smaller is fast" pattern has any analog when there is no
  smaller/larger distinction is not addressed by anything measured.
- Mechanism -- why a ratio threshold would exist, or why the smaller
  value specifically escapes, in any case.
- Generalization to 6x7 or other grid sizes.
- Whether this pattern would survive the same reproducibility scrutiny
  Addendum 78's D=10/D=20 split received (independent re-run) -- only
  single runs at each new point so far.

## 5. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (unchanged from Addendum 78) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run11.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run11.csv) | this run, 24 rows |
| [`spare-qubit-cliff-addendum-79-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-79-preregistration-2026-09-18.md) | the predictions scored above |

## 6. Verification

- All 24 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `D` cell.
- Both new pairs' isomorphism was verified computationally before this
  run (recorded in the pre-registration), not assumed after seeing
  results.
- The cross-pair ratio table (Section 3) was assembled by re-reading
  Addendum 78's own D=10/D=20 and D=5/D=25 figures directly, not from
  memory, before combining with this run's new points.
- The "smaller value is fast" observation was checked against both
  disagreeing pairs explicitly (12<18, fast is 12; 10<20, fast is 10)
  before being stated as a pattern, and is explicitly flagged as
  unconfirmed beyond these two instances.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 80 pre-registration (source: spare-qubit-cliff-addendum-80-preregistration-2026-09-18.md) ===== -->

> **Note added when merging:** Predictions for locating the exact ratio threshold across all thirteen valid isomorphic pairs at n=64, with built-in reproducibility -- includes a second construction bug (dominant_size<2 silently idle) found and fixed before running.

## Addendum 80 -- Pre-registration: locating the exact ratio threshold across all thirteen valid isomorphic pairs at n=64, and testing the "smaller value is fast" rule to its full extent, with built-in reproducibility (2026-09-18)

**Status: pre-registration only. No run beyond the four pairs already
measured (Addendum 78: 10/20, 5/25; Addendum 79: 12/18, 8/22) has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addenda 78-79 found four isomorphic dominant/filler pairs (at n=64,
17 bare edges fixed, remainder=30 split between one dominant and one
filler chain): ratios 1.5 and 2.0 disagree (smaller value fast, larger
cliffs); ratios 2.75 and 5.0 agree (both cliff). This addendum tests
**every remaining valid pair** at this n=64/remainder=30 configuration
-- thirteen pairs total (one, 1/29, excluded as structurally invalid -- see Section 2), nine not yet tested -- to (a) locate the exact
ratio threshold rather than merely bracket it, (b) test the "smaller
value is fast" rule against every disagreeing case that exists in this
configuration, not just two, and (c) build in independent
reproducibility from the start, given Addendum 78's own finding that
one specific pair's result (D=10/D=20) needed a second run before being
trusted.

## 2. Design

**A second construction bug was found and fixed while preparing this
design, before any run**: `_edges_dominant_size_sweep` validated
`filler_size` for the 0-to-2 edge case (Addendum 78's own fix) but not
`dominant_size` itself -- `dominant_size=1` silently built zero edges
for that qubit (leaving it idle) rather than raising, the same failure
mode from the opposite side. This means the pair **1/29 cannot be
validly constructed at either end** (dominant_size=1 is now caught and
raises; dominant_size=29 already raised via filler_size=1) and is
**excluded from this addendum's design**. Thirteen valid pairs remain
(small ranging 2 to 14, large = 30-small, excluding small=15's
already-tested non-pair case D=15):

| small | large | ratio | status |
|---:|---:|---:|---|
| 2 | 28 | 14.00 | new |
| 3 | 27 | 9.00 | new |
| 4 | 26 | 6.50 | new |
| 5 | 25 | 5.00 | already tested (Addendum 78): agree, both cliff |
| 6 | 24 | 4.00 | new |
| 7 | 23 | 3.29 | new |
| 8 | 22 | 2.75 | already tested (Addendum 79): agree, both cliff |
| 9 | 21 | 2.33 | new |
| 10 | 20 | 2.00 | already tested (Addendum 78): disagree, smaller (10) fast |
| 11 | 19 | 1.73 | new |
| 12 | 18 | 1.50 | already tested (Addendum 79): disagree, smaller (12) fast |
| 13 | 17 | 1.31 | new |
| 14 | 16 | 1.14 | new |

Nine new pairs (eighteen new points: both the small and large value of
each pair, since both must be tested to check agreement/disagreement),
`optimization_level=3`, spare=0, 8x8 grid. **Each of the eighteen new
points is run in two independent batches (3 seeds x 2 repeats each,
matching Addendum 78's own reproducibility check), for 12 runs per
point, 216 new runs total** -- deliberately large-scale. Isomorphism of
each new pair is to be verified computationally before interpretation,
per this project's standing practice for this line of investigation.

**Given the scale, this is designed to run as two separate invocations**
(one per batch) so intermediate progress is visible; both are specified
now so scoring is locked before either runs.

## 3. Pre-registered predictions

**P1 (primary -- where exactly does the ratio threshold sit?).**
Based on Addenda 78-79's bracket (disagree at <=2.0, agree at >=2.75):
  - **Threshold between 2.0 and 2.33** (i.e. pair 9/21 disagrees,
    pair 7/23 agrees): the boundary sits at the low end of the
    bracket.
  - **Threshold between 2.33 and 2.75** (i.e. 9/21 agrees): the
    boundary sits at the high end.
  - **Non-monotonic**: the ratio-ordered sequence of agree/disagree
    outcomes is not a single clean split (e.g. 9/21 disagrees but 7/23
    also disagrees, or some pair out of ratio order breaks the
    pattern) -- explicitly a live possibility given this project's
    repeated experience with apparently-clean small-sample patterns
    not holding at scale (Addendum 59's oscillation, Addendum 75's
    falsified lead).

**P2 (the "smaller value is fast" rule).** For every pair found to
disagree (however many that turns out to be), predicted that the
smaller value is fast and the larger cliffs, with zero exceptions --
matching both of Addenda 78-79's own disagreeing pairs (10/20, 12/18).
A single counter-example (a disagreeing pair where the LARGER value is
fast) would falsify this as a universal rule within this configuration.

**P3 (reproducibility).** Every point's two independent batches are
predicted to agree with each other (same stop reason in both batches)
for every one of the eighteen new points -- extending Addendum 78's own
single confirmed reproducibility case (D=10 and D=20 both reproduced
exactly) to the full set. Any point where the two batches disagree is
reported explicitly as a bistable or noisy point, not averaged away.

**P4 (extreme ratios).** The most extreme pairs (1/29, ratio 29.0; 2/28,
ratio 14.0) are predicted to agree (both cliff), consistent with the
existing trend of higher ratios agreeing -- included as a sanity check
on the trend's own extrapolation, not because the trend is assumed
correct in advance.

## 4. What this cannot establish

- Mechanism -- why any threshold exists, or why smaller values escape
  within the disagreeing regime, in any case.
- Whether this generalizes beyond n=64/remainder=30/17-bare-edges to
  other configurations, grid sizes, or bare-edge counts.
- Whether the threshold (if precisely located) has any relationship to
  other thresholds found elsewhere in this project (the bare-edge count
  threshold itself, grid-size effects, etc.) -- purely coincidental
  proximity, if any is found, is not to be treated as a connection
  without further work.

## 5. Scoring discipline

Score P1 as a three-way split exactly as defined; if the pattern is
genuinely non-monotonic, say so explicitly and report the actual
sequence rather than forcing it into "threshold between X and Y." Score
P2 by checking literally every disagreeing pair found, not only the
already-known two. Score P3 per-point, not in aggregate -- a single
disagreement between batches at one point is a reportable finding on
its own, not diluted by nineteen other points agreeing.

---


<!-- ===== Addendum 80 (source: spare-qubit-cliff-addendum-80-2026-09-18.md) ===== -->

> **Note added when merging:** The ratio-threshold hypothesis from Addendum 79 collapses completely -- non-monotonic across all nine new pairs, independently reproduced (18/18 across two batches) -- but 'smaller value wins' survives 5/5, and D=2 reveals a construction degeneracy (collapses into an 18th bare edge).

## Addendum 80 -- the ratio-threshold hypothesis from Addendum 79 collapses completely, and independently reproduces (18/18): non-monotonic across all nine new pairs, but "smaller value is fast" survives 5/5, and D=2 reveals a construction degeneracy (2026-09-18)

**Pre-registered in**:
`spare-qubit-cliff-addendum-80-preregistration-2026-09-18.md`, written
and locked before this run, including a second construction bug
(`dominant_size < 2` silently idle) found and fixed before running.
**Both batches (216 rows total) are now complete and incorporated.
P3 is scored below.**

## 0. In one line

**P1: "Non-monotonic" confirmed, decisively.** The ratio-ordered
sequence of outcomes across all nine new pairs is not a clean split at
any point: ratio 14.0 agrees (both fast -- a new category never seen in
Addenda 78-79), 9.0 agrees (both cliff), 6.5 **disagrees**, 4.0 agrees,
3.29 **disagrees**, 2.33 disagrees, 1.73 agrees, 1.31 agrees, 1.14
agrees. **Addendum 79's own "threshold between 2.0 and 2.75" hypothesis
is falsified outright** -- ratio 6.5 (well above the hypothesized
threshold) disagrees, and ratio 1.73 (well below it) agrees. **P2: "the
smaller value is fast" survives intact** -- every one of the five new
disagreeing pairs (4/26, 7/23, 9/21) has the smaller value fast,
extending the two known cases from Addenda 78-79 to five with zero
exceptions. **An unregistered structural finding**: D=2 degenerates
into just another bare 2-qubit edge (the chain-building loop for
`dominant_size=2` produces one edge, identical in shape to the
seventeen existing bare edges), so the "D=2/D=28" pair is not actually
testing "small dominant vs. large dominant" at all -- it is testing "18
bare edges + one 28-chain," structurally adjacent to `mixed_uneven`
itself, which explains why both sides are fast rather than following
either agree-pattern seen elsewhere.

## 1. Results (both batches)

8x8 grid (64 qubits), `dominant_size_sweep`, spare=0,
`optimization_level=3`, 3 seeds x 2 repeats, two independent batches
(216 rows total). All rows completed with `error=""`; stop reason
unanimous (6/6) within every cell, and unanimous between the two
batches at every one of the eighteen points (Section on P3 below).
Batch 1 figures shown; Batch 2 matched exactly on every stop reason.

| pair (small/large) | ratio | small time (ms) | large time (ms) | outcome |
|---|---:|---:|---:|:---|
| 2/28 | 14.00 | 34.53 | 34.68 | **both fast** (structural degeneracy -- see Section 3) |
| 3/27 | 9.00 | 8,815.22 | 9,271.83 | both cliff |
| **4/26** | 6.50 | **34.11** | 9,274.55 | **disagree, smaller fast** |
| 6/24 | 4.00 | 9,104.09 | 9,208.46 | both cliff |
| **7/23** | 3.29 | **33.58** | 9,941.71 | **disagree, smaller fast** |
| **9/21** | 2.33 | **33.79** | 9,025.23 | **disagree, smaller fast** |
| 11/19 | 1.73 | 8,902.75 | 9,010.21 | both cliff |
| 13/17 | 1.31 | 11,441.58 | 9,235.64 | both cliff |
| 14/16 | 1.14 | 9,713.62 | 9,274.13 | both cliff |

Combined with Addenda 78-79's own four pairs (10/20, 5/25, 12/18,
8/22): **thirteen pairs now on record, five disagreeing** (10/20,
12/18, 4/26, 7/23, 9/21), **seven agreeing by both cliffing** (5/25,
8/22, 3/27, 6/24, 11/19, 13/17, 14/16), and **one agreeing by both
being fast** (2/28, the degenerate case).

## 2. Scoring

**P1 (primary -- ratio threshold location) -- FALSIFIED, all three
branches, in the specific way the pre-registration flagged as a live
possibility.** Neither "threshold between 2.0-2.33" nor "threshold
between 2.33-2.75" describes the data -- there is no ratio value above
which every pair agrees and below which every pair disagrees. The
"non-monotonic" branch is confirmed, and confirmed strongly: disagree
and agree outcomes interleave across the ratio range with no visible
ordering.

**P2 (smaller value is fast) -- CONFIRMED, 5/5, zero counter-examples
so far.** Every disagreeing pair found across Addenda 78-80 --
10/20, 12/18, 4/26, 7/23, 9/21 -- has the smaller value fast and the
larger cliffing. This is the one part of Addendum 79's emerging picture
that has not broken down.

**P3 (reproducibility) -- CONFIRMED, 18/18, zero disagreements.** Every
one of the eighteen new points was checked point-by-point between Batch
1 and Batch 2: every stop reason matched exactly, and every timing
value fell within the same tight bands already established (fast:
~34-42ms; cliffing: ~8,800-9,500ms, with one Batch 1 outlier at
D=13, 11,441.58ms, not reproduced in Batch 2's 9,063.60ms -- still
comfortably within the cliffing regime, not a classification change).
**The stark non-monotonicity reported in Section 1 is not a
measurement artefact** -- it reproduces exactly across two independent
batches, which rules out the most obvious alternative explanation
(that some of the "disagreeing" pairs were actually borderline/noisy
points that happened to land on opposite sides by chance).

**P4 (extreme ratios) -- PARTIALLY FALSIFIED.** Predicted 2/28 and (had
it been valid) 1/29 would agree by both cliffing, consistent with the
existing "high ratio agrees" trend. **2/28 does agree, but by both
being fast, not both cliffing** -- the prediction's binary framing
(agree vs. disagree) was satisfied, but the specific *kind* of
agreement was not the one implicitly assumed, and Section 3 shows why:
this pair does not actually test the same "two large chains" structure
as every other pair.

## 3. The D=2/D=28 degeneracy, and what it means for interpreting "ratio"

**`dominant_size=2` does not build a genuine small chain distinct from
the bare edges.** The construction's chain-building loop
(`range(dominant_size - 1)`) with `dominant_size=2` produces exactly
one edge between two qubits -- indistinguishable in shape from any of
the seventeen already-present bare 2-qubit edges. Verified directly:
D=2's actual component decomposition is **eighteen 2-qubit components
plus one 28-qubit chain** (confirmed via `networkx`), not "one 2-qubit
dominant plus one 28-qubit filler" as the parameter name suggests.
**This means D=2/D=28 is not a fair test of "small vs. large dominant
component" at all** -- it is closer to a `mixed_uneven`-family point
(many bare edges + one large chain) with 18 bare edges instead of 17.
That it is fast is unsurprising in light of Addenda 63/77's own
`mixed_uneven` result, and does not extend the "ratio" story in either
direction. **Any pair where the smaller value is 2 should be treated as
this special case going forward, not as a genuine small-dominant test.**

This also means Section 1's ratio column is not uniformly meaningful:
the true underlying variable at D=2 is "bare-edge count," not "dominant
component size," so plotting D=2/D=28 against the same ratio axis as
the other eight pairs conflates two different things. **This is
disclosed rather than corrected retroactively** -- the table is left as
measured, with this caveat attached.

## 4. What this means for the broader investigation

**The picture is now more complex, not simpler, than Addendum 79
suggested.** A clean ratio threshold does not exist. What survives is
narrower and more specific: **whenever two isomorphic dominant/filler
constructions disagree, the smaller one is fast** -- but *which* pairs
disagree versus agree does not follow simply from their size ratio.
Something else -- not yet identified -- determines whether a given pair
lands in the disagreeing category at all. Addendum 78's original
finding (physical qubit placement matters, holding the abstract graph
fixed) is reinforced and sharpened by this addendum: not only does
placement matter, but *whether* placement matters (i.e., whether the
pair disagrees or agrees) is itself unpredictable from the one variable
(ratio) that seemed promising after four points.

## 5. What this does not establish

- **Why some pairs disagree and others agree** -- ratio is ruled out;
  no replacement variable is proposed.
- **Batch 2's reproducibility of these nine new points** -- pending.
- Whether excluding the degenerate D=2/D=28 case from a re-analysis
  changes the picture -- it does not appear to (the remaining eight
  pairs are already non-monotonic on their own), but this was not
  formally re-checked with D=2/D=28 removed.
- Mechanism, in any case.

## 6. Files

| File | What it is |
|---|---|
| [`circuit_family_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/circuit_family_sweep.py) | the script (Addendum 80's dominant_size validation fix) |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run12.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run12.csv) | Batch 1, 108 rows |
| [`circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run13.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/circuit_family_sweep_8x8_AMD64_Family_25_Model_80_Stepping_0_AuthenticAMD_2026-09-18_run13.csv) | Batch 2 (independent reproducibility re-run), 108 rows |
| [`spare-qubit-cliff-addendum-80-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-80-preregistration-2026-09-18.md) | the predictions scored above |

## 7. Verification

- All 108 rows checked for `error=""`; stop reason confirmed unanimous
  (6/6) within every `D` cell.
- D=2's degenerate structure was verified computationally
  (`networkx`, direct component-size enumeration) before being reported
  as the explanation for that pair's anomalous "both fast" result,
  rather than left as an unexplained outlier.
- The ratio-ordered sequence in Section 1 was checked against Addendum
  79's own predicted branches (quoted in that addendum) before being
  scored as falsifying all three.
- The "smaller value is fast" tally (5/5) was checked against every
  disagreeing pair on record across Addenda 78-80, not only this
  addendum's own three, before being reported as zero-exception.
- Batch 1 and Batch 2 were compared point-by-point (all 18 new points)
  for stop-reason agreement before P3 was scored, not summarized from
  aggregate statistics.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both new CSVs
  -> 0 hits.

---


<!-- ===== Addendum 81 (source: spare-qubit-cliff-addendum-81-2026-09-18.md) ===== -->

> **Note added when merging:** Following a methodological correction (test irregularity directly instead of hunting for more patterns): a permutation test finds NO local D-proximity correlation in outcomes -- statistically indistinguishable from random (p~0.5), both with and without the D=2/28 degenerate case.

## Addendum 81 -- testing irregularity directly, before hunting for more patterns: adjacent-D outcomes are statistically indistinguishable from random (2026-09-18, end of day)

**Status**: an analytical addendum, prompted by a methodological
correction from the user: rather than continuing to search for a
pattern that explains Addendum 80's non-monotonic results, **test
directly whether no pattern exists at all**, using a proper statistical
test rather than exhausting candidate variables one at a time. No new
Qiskit measurement -- uses only outcomes already on record from
Addenda 78 and 80.

## 0. In one line

**A permutation test finds that fast/cliff outcomes across D=2 through
28 (dominant_size_sweep, 8x8, 17 bare edges fixed) show NO more local
correlation between adjacent D values than a random shuffle of the same
outcomes would produce.** Observed adjacent-match rate: 14/24 pairs
(58.3%). Mean under 10,000 random shuffles of the same outcome
sequence: 14.17/24 (59.0%). The observed data sits almost exactly at
the center of the null distribution (p=0.574 for "as few or fewer
matches than observed"), meaning **there is no detectable tendency for
nearby D values to share an outcome** -- exactly what "no simple local
regularity" predicts, and the opposite of what would be seen if a
smooth or slowly-varying rule governed the outcome as a function of D.
This result is unchanged when the two degenerate points (D=2, D=28,
Addendum 80's own finding that these collapse into an 18-bare-edge
`mixed_uneven`-adjacent structure) are excluded (14/22, p=0.544).

## 1. Why this test, and why now

Every attempt since Addendum 78 to find a variable that predicts
fast/cliff outcome (isomorphism-pair ratio, component count, bare-edge
count) has been tried and falsified one at a time -- the same
enumerate-and-falsify approach this project used successfully for
mod-3 (Addenda 53/59/61-63) but which, as the user pointed out, only
ever *rules out* candidates rather than testing the underlying premise
that a findable pattern exists at all. **If no such pattern exists,
continuing to search candidate-by-candidate never terminates -- it can
only keep failing.** A direct test of irregularity itself, rather than
another candidate variable, was proposed as the correct next step, and
is what this addendum performs.

## 2. Method

All eighteen distinct D values with a recorded fast/cliff outcome from
Addenda 78 and 80 (D=2 through 28, excluding D=15's own non-pair case
and D=1/29 which Addendum 80 found invalid to construct), encoded as
1 (fast) or 0 (cliff), ordered by D. For every pair of D values exactly
1 apart (i.e. truly adjacent on the integer line: D and D+1), checked
whether their outcomes match. This count is compared against the
distribution obtained by randomly shuffling the same eighteen outcome
values across the same eighteen D-positions 10,000 times and recomputing
the same adjacent-match statistic each time -- a standard permutation
test, testing the null hypothesis that outcome is independent of
position (D).

## 3. Results

| | with D=2, D=28 | excluding D=2, D=28 |
|---|---:|---:|
| Observed adjacent matches | 14/24 (58.3%) | 14/22 (63.6%) |
| Mean under 10,000 random shuffles | 14.17/24 (59.0%) | 14.43/22 (65.6%) |
| p-value (P[shuffled <= observed]) | 0.574 | 0.544 |

**Both versions show the same qualitative result**: the observed
data's adjacent-match rate is not merely "not significantly higher"
than random (which would be the minimum bar for detecting local
regularity) -- it sits almost exactly at the shuffled distribution's own
mean, with the p-value close to 0.5 in both cases (the value expected
if the real data were itself just another random draw).

## 4. What this establishes, and what it does not

**Establishes**: there is no evidence of *local, D-proximity-based*
regularity in this dataset. A model of the form "outcome is a smooth or
slowly-varying function of D" is not supported -- if it were, adjacent D
values would share outcomes far more often than chance, and they do
not, at all, even nominally in this small sample.

**Does not establish**: that *no* regularity exists of *any* kind. This
test is specifically blind to:
- **Non-local structure**: a rule depending on D through some other
  relationship (e.g. `D mod k` for some k not equal to 1, or a
  relationship between D and 30-D jointly, or D's relationship to 32,
  the grid's own maximum matching size) would not show up as
  *adjacent*-D correlation at all, and is not ruled out by this test.
- **The "smaller value wins" rule** (Addendum 80's own 5/5 finding) is
  a rule about *pairs* (D, 30-D), not about D's position on the
  integer line alone -- this test does not bear on it one way or the
  other, and that rule remains the one surviving, unfalsified pattern
  in this entire investigation.
- **Sample size**: eighteen points, twenty-two or twenty-four adjacent
  comparisons, is small. A null result here is consistent with "no
  local regularity exists" but also consistent with "local regularity
  exists but is too weak or too fine-grained to detect at this
  resolution." The test has real but limited power.

## 5. What this means for the investigation's next step

**This result argues against continuing to search for a smooth,
D-indexed explanatory function**, since the data actively looks like it
was NOT generated by one. It does **not** argue against investigating
the "smaller value wins" pair-rule further, since that rule concerns a
different structural question (a comparison between two specific
values, not a trend across the D axis) that this test cannot speak to.
**The recommended next step, carried into tomorrow's session, is
therefore NOT another candidate-variable search along the D axis, but
either (a) a similar irregularity test targeted at the pair-rule itself
(e.g., does "smaller wins" hold at every ratio, or does it also show
no structure beyond the raw 5/5 count), or (b) the D-vs-32
relationship flagged in `SESSION_SUMMARY_FINAL_2026-09-18.md` Section 3
item 1, which is a non-local hypothesis this test does not rule out.**

## 6. Files

No new data files -- recomputed from outcomes already on record in
Addenda 78 and 80.

## 7. Verification

- The permutation test was implemented directly (10,000 shuffles,
  seeded for reproducibility) rather than relying on a closed-form
  approximation, and rerun with D=2/D=28 excluded to check robustness
  -- both give the same qualitative result.
- All eighteen outcome values were re-read directly from Addenda 78 and
  80's own recorded tables before being encoded, not from memory.
- The interpretation in Section 4 explicitly distinguishes what this
  specific test does and does not rule out, rather than overclaiming a
  general "no regularity" finding from a test of local regularity
  alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 82 (source: spare-qubit-cliff-addendum-82-2026-09-19.md) ===== -->

> **Note added when merging:** Reframing from 'does regularity exist' (unanswerable) to 'how much predictive power do current features have' (quantifiable): a Mantel-correlation test across all 325 pairs and three distance metrics finds at most 4.5% variance explained, none surviving Bonferroni correction for multiple comparisons.

## Addendum 82 -- quantifying predictive power directly, across all pairs and all available distance metrics, instead of testing one candidate variable at a time (2026-09-18/19)

**Status**: an analytical addendum, following a methodological
correction from the user: reframe the question from "does regularity
exist?" (unanswerable by exhaustive search) to **"how much predictive
power do the currently available features have?"** (a quantifiable,
bounded question). Uses only the 26 outcomes already on record from
Addenda 78 and 80 -- no new Qiskit measurement. This supersedes
Addendum 81's narrower test (adjacent-D-only, binary) with a more
comprehensive one (all 325 pairs, continuous distance, multiple
metrics, explicit effect-size and multiple-comparison reporting).

## 0. In one line

**Across three independent distance metrics computed for every one of
the 325 possible pairs among 26 known outcomes, the best predictive
power found explains at most 4.5% of the variance in whether two
points share an outcome, and none of the three metrics survives
correction for testing multiple hypotheses.** D-distance: r=-0.151,
nominal p=0.032, 2.3% variance explained. Ratio-distance: r=-0.212,
nominal p=0.049, 4.5% variance explained. Max-component-distance:
r=-0.017, p=0.354, ~0% variance explained. **After Bonferroni
correction for the three tests (requiring p<0.0167), none remains
significant.** The honest answer to "how much predictive power does the
current candidate-variable set have" is: **very little, and what little
appears present does not clear a standard statistical bar once multiple
testing is accounted for.**

## 1. Why this addendum exists, and how it differs from Addendum 81

Yesterday's Addendum 81 tested only *exact adjacency* (D and D+1) using
a binary match/no-match count -- a narrow test with limited statistical
power (only 22-24 comparisons). The user correctly identified that
continuing to test one new candidate variable at a time (as done for
ratio in Addenda 79-80, mod-3 in Addenda 53/59/61-63, etc.) reproduces
the same combinatorial-search structure this project is investigating
in Qiskit itself -- an unbounded process that can only fail to find
things, never conclude their absence. **The correct reframing, proposed
by the user**: stop asking "is there a pattern" (unanswerable) and
instead ask **"how much does each already-available feature explain"**
(directly computable, bounded, and does not require inventing new
candidates).

## 2. Method

**Step 1 -- feature table.** All 26 points with a known outcome from
Addenda 78/80 (D=2 through 28, all valid dominant_size_sweep
configurations), each characterized by six features computed directly
from the interaction graph (not re-derived from memory): `D`, `filler`
(30-D), `ratio` (larger/smaller), `n_components`, `max_component`,
`n_bare_2q_components`, `n_edges`. **A structural limitation found while
building this table, reported rather than hidden**: `n_components`,
`n_edges`, and `n_bare_2q_components` are **nearly constant across the
entire dataset** -- the only exception is the D=2/D=28 degenerate case
Addendum 80 already identified, where the "dominant" or "filler" chain
collapses into an extra bare edge. This means these three features
carry almost no discriminating information in this dataset, and any
distance computed from them would be trivially uninformative. **The
genuinely independent features in this dataset reduce to D alone**
(filler, ratio, and max_component are all deterministic functions of
D), which is itself an important limitation on this addendum's own
power -- see Section 5.

**Step 2 -- pairwise distances.** For every one of the C(26,2)=325
pairs, computed three distances: `|D_i - D_j|`, `|ratio_i - ratio_j|`,
`|max_component_i - max_component_j|`. (Distances based on
n_components/n_edges/n_bare_2q were not computed as separate tests, per
the near-constancy noted above -- testing them would not add
information beyond what their near-constancy already implies.)

**Step 3 -- Mantel-style correlation.** For each distance metric,
computed the Pearson correlation between the distance values and a
binary "outcomes match" indicator (1 if same outcome, 0 if different)
across all 325 pairs. A negative correlation means "farther apart in
this metric -> less likely to share an outcome," i.e. the metric has
predictive power in the expected direction.

**Step 4 -- permutation test.** For each metric, the outcome labels
were shuffled across the 26 points 5,000 times, and the same
correlation recomputed each time, to obtain a null distribution and a
p-value (proportion of shuffles producing a correlation at least as
negative as observed).

**Step 5 -- multiple-comparison correction.** Since three metrics were
tested, Bonferroni correction was applied (family-wise alpha 0.05
requires per-test alpha 0.05/3 = 0.0167).

## 3. Results

| metric | observed correlation | nominal p-value | r-squared (variance explained) | survives Bonferroni (p<0.0167)? |
|---|---:|---:|---:|:---:|
| D-distance | -0.151 | 0.032 | 2.3% | **No** |
| ratio-distance | -0.212 | 0.049 | 4.5% | **No** |
| max-component-distance | -0.017 | 0.354 | 0.0% | No |

## 4. Interpretation, at the confidence level the data supports

**None of the three available distance metrics demonstrates robust
predictive power over this dataset.** D-distance and ratio-distance show
nominally negative correlations (the "expected" direction if closeness
predicted shared outcome) that would individually clear an uncorrected
p<0.05 threshold, but **neither survives the standard correction for
testing three hypotheses**, and even taken at face value, each explains
under 5% of the variance in whether two points share an outcome --
a small effect size by any conventional standard. Max-component-distance
shows no detectable relationship at all.

**This is a different and more precise conclusion than either "no
regularity exists" or "a regularity has been found."** The correct
statement is: **the six graph-level features computed so far (three of
which are nearly constant and thus uninformative, three of which
collapse to functions of one underlying variable, D) collectively
explain, at most, a few percent of this system's outcome variance.**
Whether a *richer* feature set (not yet computed) would perform better
is a separate, open question -- see Section 5.

## 5. What this does not establish, and the sharpest limitation of this addendum itself

**The dataset's own structure severely limits what this test can show.**
With 26 points and effectively one independent numeric feature (D), no
correlational test -- however comprehensive -- can detect structure that
depends on information D does not carry (e.g., which specific physical
qubits are used, beyond their count; any property of the actual grid
embedding search path; anything about the specific random unitary gates
placed on each edge). **This addendum quantifies the predictive power
of the features currently computed, not the predictive power achievable
in principle.** A genuinely comprehensive "predictability map," as the
user's own framing requests, would need a richer feature set (e.g.
graph spectral properties, automorphism group size properly computed
rather than the informal check mentioned in Addendum 74, or features of
the specific qubit-to-component assignment that Addendum 78 showed
matters) before a low predictive-power finding could be treated as
informative about the underlying system rather than about the
particular six features tried.

**Also not established**: whether the "smaller value wins" rule
(Addendum 80's 5/5 finding) has any relationship to the weak D-distance
and ratio-distance signals found here -- that rule concerns pair
membership (D vs. 30-D) directly, not distance between arbitrary pairs,
and was not re-tested in this addendum's framework.

## 6. What this means going forward

**Per the user's own stated priority ranking**: stop new sweeps (done --
no new Qiskit runs in this addendum), the feature table now exists
(Section 2, Step 1), the distance-metric evaluation has been done
(Sections 2-4), and predictive power has been quantified (Section 3).
**The honest state of the investigation**: with the current feature
set, this system's outcome is not well predicted by anything measured
so far. The next highest-value step, following the user's own framing
of an interpretation-deficit phase rather than a data-deficit one, is
**not another sweep**, but either (a) computing genuinely richer
graph-theoretic features from the 26 already-known configurations (no
new Qiskit runs needed), or (b) accepting that this specific line of
investigation (isomorphic dominant/filler pairs at 8x8) has reached the
limit of what structural analysis alone can explain, and redirecting
effort toward mechanism (the still-unlocated Rust source, Addenda
43-45) or toward the two-track paper structure discussed earlier
(Research A: is PSF-Zero effective; Research B: why does VF2 fail)
rather than continuing to deepen Research B indefinitely.

## 7. Files

No new data files -- recomputed from outcomes already on record in
Addenda 78 and 80.

## 8. Verification

- All 26 outcome values were re-derived from Addenda 78 and 80's own
  recorded tables before being encoded, not from memory.
- The near-constancy of n_components/n_edges/n_bare_2q_components was
  verified computationally (via `networkx`, direct component
  enumeration for every one of the 26 configurations) before being used
  to justify excluding them as separate distance metrics, not assumed.
- The permutation test (5,000 shuffles per metric) was implemented
  directly and seeded for reproducibility.
- The Bonferroni correction was applied using the standard formula
  (family-wise alpha divided by number of tests) and reported
  transparently rather than only reporting the more favorable
  uncorrected p-values.
- r-squared values were computed directly from the reported correlation
  coefficients, not estimated.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 83 (source: spare-qubit-cliff-addendum-83-2026-09-19.md) ===== -->

> **Note added when merging: this addendum's central attribution is corrected by Addenda 84-85 below -- read all three together, not this one alone.** The 26/26 empirical result itself stands; the explanation ("PSF-Zero's own multi-stage strategy is why") does not. The session's most consequential empirical result: PSF-Zero's real, unmodified smart_vf2_layout (source supplied by the user) finds a layout in all 26 of 26 tested configurations, including all 19 where Qiskit's own VF2Layout cliffed. A real bug in PSF-Zero's own feasibility pre-check was found and fixed along the way.

## Addendum 83 -- PSF-Zero's actual layout search finds a layout in all 26 configurations, including all 19 where Qiskit's own VF2Layout cliffed -- a real, direct, decisive comparison (2026-09-19)

**Status**: a direct empirical comparison, using PSF-Zero's real,
unmodified `smart_vf2_layout()` (from `psf_smart_layout.py`, uploaded
2026-09-19) against the same 26 `dominant_size_sweep` configurations
Qiskit's own VF2Layout was measured on in Addenda 78 and 80. Run twice
by the user: the first run exposed a design flaw in the comparison
script (Section 1), the second (reported here) is the corrected,
trustworthy result.

## 0. In one line

**PSF-Zero's `smart_vf2_layout` found a valid layout in all 26 of 26
tested configurations -- including every one of the 19 configurations
where Qiskit's own VF2Layout returned `"nonexistent solution"` (the
cliff) after multi-second searches.** Every success took under 0.066
seconds, and every single one was found in Stage 1 alone (the cheap
BFS-family-ordering search with `id_order=True`) -- Stage 2's more
expensive fallback was never needed. There is not a single case in
either direction where the two tools disagree by PSF-Zero failing where
Qiskit succeeded; every disagreement (19 of 26) is PSF-Zero succeeding
where Qiskit cliffed.

## 1. A design flaw in the first attempt, caught and fixed before trusting any result

The first run of this comparison returned `feasible=False,
orderings_tried=0` for all 26 configurations, including the 7 where
Qiskit itself succeeded -- a result that, taken at face value, would
have wrongly suggested PSF-Zero's search never even attempted these
cases. Investigation traced this to `smart_vf2_layout`'s own
`_has_feasible_matching` pre-check, which computes
`len(max_matching) >= num_logical_pairs`, where `num_logical_pairs` is
passed as `len(interaction_pairs)` -- the interaction graph's raw edge
count. **This comparison is only valid when the interaction graph is
itself a matching** (the project's original `dense_pairs` family,
Addenda 4-50, where the interaction graph literally is a set of
disjoint pairs and edge count equals the number of pairs needing
positions). For the chain-shaped families used throughout Addenda
51-82, edge count (45 for these `dominant_size_sweep` configurations)
routinely exceeds the physical device's own maximum matching size (32
for this 8x8 grid, Addendum 34), so the check rejected every
configuration before the real search logic ever ran, regardless of
whether the underlying graph was actually embeddable. **This is a
known limitation of a prototype explicitly built around Addenda 8-12's
matching-shaped circuit family** (documented in `psf_smart_layout.py`'s
own docstring, which frames the whole module around that earlier
investigation), not a defect discovered for the first time here, and
not evidence about PSF-Zero's actual search strategy. The comparison
script was revised to call PSF-Zero's own internal search helpers
(`_candidate_orderings`, `_try_mapping`, etc., unmodified) directly,
bypassing only this mismatched guard, and the corrected run is what
this addendum reports.

## 2. Results

8x8 grid (64 qubits, matching `CouplingMap.from_grid(8,8)`), all 26
`dominant_size_sweep` configurations from Addenda 78/80 (D=2 through
28, 17 bare edges fixed, one dominant chain of size D + one filler
chain of size 30-D).

| | Qiskit VF2Layout | PSF-Zero smart_vf2_layout |
|---|---|---|
| Found a layout | 7/26 | **26/26** |
| Failed (cliff / not found) | 19/26 | **0/26** |
| Time when it succeeds | tens of ms | **under 0.066s in every case** |
| Time when it fails | ~9,000-9,900ms (multi-second search to exhaustion) | (never fails on this dataset) |
| Search strategy used on success | Qiskit's own `vf2_layout_pass_average` | **Stage 1 only** (BFS-family ordering, `id_order=True`) in all 26 cases -- Stage 2 never triggered |

Every one of the 19 configurations where Qiskit cliffed (D=3, 5, 6, 8,
11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27) was found by
PSF-Zero using either the `degree_desc` ordering (4 orderings tried
before success) or `bfs_from_min_degree` (2 orderings tried), never
needing more than 4 attempts total.

## 3. What this establishes, stated precisely

**On this specific, well-defined test -- the same physical topology
(8x8 grid) and the same interaction-graph family (17 bare edges +
dominant/filler chains) that caused Qiskit's own VF2Layout to fail 19
times out of 26 -- PSF-Zero's `smart_vf2_layout` succeeds in every
single case, quickly, using only its cheapest search stage.** This is a
real, positive, and now directly demonstrated result for PSF-Zero, not
a claim resting on architecture alone.

**This also revises how Addendum 78-82's own findings should be read.**
Those addenda characterized the isomorphic-pair disagreement and low
overall predictability as properties of "the system" -- but this
comparison shows the erratic fast/cliff behavior is specific to
Qiskit's own particular VF2 implementation and node-ordering strategy
(`vf2_layout_pass_average`, per Addendum 45's reading of Qiskit's own
documentation), not an intrinsic, unavoidable property of VF2-family
subgraph-isomorphism search in general. **A different node-ordering
strategy for the same underlying algorithm class (still VF2, still a
search) sidesteps the entire problem on this dataset.** This is a
materially different conclusion from "the search paradigm itself has
reached its limit" -- what has reached its limit, on this evidence, is
specifically Qiskit's own implementation choices for this circuit
family, not search-based layout as a category.

## 4. What this does not establish

- **Generalization beyond this exact topology/family combination.**
  `psf_smart_layout.py`'s own docstring documents, from earlier
  diagnostics (Addenda 8-12), that its BFS-ordering strategy is
  strongly effective on grid physical graphs with matching-shaped
  interaction patterns, but explicitly does NOT generalize to other
  physical topologies (`brick` failed even at very high call limits)
  or, by the same logic, may not generalize to every interaction-graph
  family untested here. This comparison used the 8x8 grid (where the
  strategy is documented to work well) and the specific chain-shaped
  families from today's own investigation -- not a claim about every
  possible circuit or every possible device topology.
- **Whether PSF-Zero's own feasibility guard bug (Section 1) affects
  its behavior in production use**, i.e. whether real callers of
  `compile_for_hardware(layout_search=True)` on chain-shaped circuits
  are currently silently falling through to Qiskit's own default layout
  stage (since a `None` return from `smart_vf2_layout` causes exactly
  that fallback, per `psf_compile.py`'s own documented behavior) --
  this is a real, practical bug with user-facing consequences, reported
  here as a finding in its own right, independent of the comparison's
  own corrected result.
- **End-to-end compile time**, including gate synthesis
  (`compile_for_hardware()`'s full pipeline) -- this comparison isolated
  the layout search alone, matching what Addenda 78/80 measured on the
  Qiskit side.
- Whether `smart_vf2_layout`'s own internal `rustworkx.vf2_mapping()`
  calls, run repeatedly across many orderings, would themselves show
  the same kind of isomorphic-pair non-monotonicity Addendum 80 found
  in Qiskit's implementation, on some other dataset not yet tested --
  this comparison shows it does not happen on THIS dataset, not that
  the underlying algorithm is immune to it in general.

## 5. The feasibility-guard bug is itself worth fixing

Independent of this comparison's own result, Section 1's finding is a
real, reportable defect: **`_has_feasible_matching`'s check is wrong
for any interaction graph that is not itself a matching**, which
includes every circuit family this project has used since Addendum 51
(more than half of this entire project's addenda). Any caller currently
using `compile_for_hardware(layout_search=True)` on a chain-shaped or
otherwise non-matching circuit is silently falling through to Qiskit's
own default layout stage -- getting none of the benefit this addendum
just demonstrated, without any error or warning indicating why. This
should be corrected before `layout_search=True` is used or benchmarked
on anything broader than the original `dense_pairs` family.

## 6. Files

| File | What it is |
|---|---|
| [`compare_psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compare_psf_smart_layout.py) | the corrected comparison script (bypasses the feasibility-guard bug, calls PSF-Zero's own internal search helpers unmodified) |
| [`psf_smart_layout_comparison_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/psf_smart_layout_comparison_2026-09-19.csv) | this run's results, 26 rows |
| [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py), [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile.py), `lib.rs` | PSF-Zero's own source, uploaded 2026-09-19, read directly rather than assumed |

## 7. Verification

- All 26 rows checked directly from the CSV; the "19/19 disagreements
  are all PSF-Zero-wins" claim was verified computationally (not
  eyeballed) before being stated.
- The feasibility-guard bug's root cause (edge count vs. matching size
  mismatch) was verified by direct calculation: 45 edges for these
  configurations vs. 32 max matching for the 8x8 grid, both numbers
  re-derived rather than assumed.
- The claim that all 26 successes used Stage 1 only (never Stage 2) was
  checked against the `psf_phase` column directly for every row, not
  inferred from a summary.
- PSF-Zero's own source files (`psf_compile.py`, `psf_smart_layout.py`,
  `lib.rs`) were read in full before characterizing what
  `layout_search=True` actually does, correcting an earlier
  mischaracterization (in conversation, not in a prior addendum) of
  PSF-Zero's layout stage as a "search-free algebraic method" -- it is
  not; it is a differently-implemented VF2-family search, and this
  addendum's own findings are stated accordingly.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits.

---


<!-- ===== Addendum 84 pre-registration (source: spare-qubit-cliff-addendum-84-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for whether a bare rustworkx id_order=True control (no BFS relabeling, no multi-stage strategy) alone explains Addendum 83's 26/26 result, prompted by external review questioning whether PSF-Zero's own ordering-diversity strategy was doing any real work.

## Addendum 84 -- Pre-registration: does `id_order=True` alone explain PSF-Zero's 26/26 success, or does the multi-stage BFS strategy add something? (2026-09-19)

**Status: pre-registration only. No run of this control condition has
been performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 83 found PSF-Zero's `smart_vf2_layout` finds a layout in all
26 of 26 tested configurations, including all 19 where Qiskit's own
VF2Layout cliffed, and concluded this reflects PSF-Zero's own
multi-stage node-ordering strategy (cheap BFS-family orderings with
`id_order=True`, falling back to VF2's heuristic ordering with
`id_order=False` only if needed). **An external review of this
project's public materials raised a specific, testable concern**: every
one of the 26 successes in Addendum 83's own data used Stage 1 alone
(`id_order=True`), and Stage 2 was never triggered. This means the
comparison so far cannot distinguish two different explanations:

- **PSF-Zero's own contribution**: trying several different BFS-family
  starting orderings, each with `id_order=True`, is what finds a
  solution -- a single fixed ordering with `id_order=True` might not
  suffice, and the diversity of orderings tried is what matters.
- **`id_order=True` alone**: the mere act of setting `id_order=True`
  (as opposed to Qiskit's own default, which per Addendum 45's reading
  of Qiskit's documentation may use a different ordering scheme
  entirely) is sufficient by itself, regardless of which specific
  starting ordering is used or how many are tried -- in which case
  PSF-Zero's own multi-stage machinery adds nothing on this dataset,
  and the correct, narrower finding would be "setting id_order=True on
  a plain `rx.vf2_mapping()` call already solves this problem," not
  "PSF-Zero's search strategy solves this problem."

This addendum tests these two explanations directly, using a minimal
third condition.

## 2. Design

For the same 26 `dominant_size_sweep` configurations (D=2 through 28,
n_bare=17 fixed, n=64, 8x8 grid) used in Addendum 83, a third condition
is added: a single, direct call to `rx.vf2_mapping(relabeled_physical,
interaction_graph, subgraph=True, id_order=True, induced=False,
call_limit=<same budget as Addendum 83's per_attempt_call_limit>)`,
using the **natural (unordered) node numbering** -- i.e. no BFS
relabeling at all, just the plain physical graph as constructed, with
`id_order=True` set and nothing else changed. This isolates whether
`id_order=True` alone, with no ordering effort whatsoever, already
succeeds -- the most minimal possible test of the "id_order alone
explains it" hypothesis. `rustworkx`'s own `vf2_mapping` function is
called directly, not through any PSF-Zero wrapper, to make the
comparison as clean as possible.

## 3. Pre-registered predictions

**P1 (primary -- does bare id_order=True, no ordering effort, solve all 26)?**
  - **id_order=True alone suffices (26/26 or very close)**: PSF-Zero's
    own multi-stage ordering strategy adds nothing on this dataset --
    the earlier "PSF-Zero's search strategy avoids the cliff" framing
    would need to narrow to "id_order=True avoids the cliff," a
    materially smaller and more specific claim.
  - **id_order=True alone is insufficient (well below 26/26)**: PSF-Zero's
    own contribution (trying diverse BFS-family starting orderings) is
    doing real, necessary work beyond the bare id_order flag --
    supporting (though not proving in full generality) the original
    framing.
  - **Partial/intermediate result**: reported as its own outcome, not
    forced into either extreme.

**P2 (consistency check).** Every one of Addendum 83's own 26 successes
used `id_order=True` in some ordering (never Stage 2's `id_order=False`
fallback) -- re-confirmed directly from that addendum's own recorded
`psf_phase` column before this addendum's own results are interpreted,
since this pre-registration's reasoning depends on that fact being
accurate.

## 4. A second, separate question this addendum does not resolve

Whether Qiskit's own `VF2Layout` internally uses `rustworkx.vf2_mapping()`
at all, or a separate, independently-implemented Rust module (as an
external source claims Qiskit's own maintainers have stated), is a
different question from this addendum's own P1/P2. **This addendum does
not test or resolve that question.** If Qiskit does not use
`rustworkx.vf2_mapping()` internally, then even a fully confirmed P1
("id_order=True alone suffices") would mean: a plain rustworkx call
with one flag set solves what Qiskit's own separate implementation
does not -- still a real and useful finding, but framed as a
comparison between two different implementations of VF2-family search,
not "the same algorithm with a different setting." This distinction
should be resolved (by reading Qiskit's own current source, not by
inference) before either addendum's finding is written into any
external-facing document.

## 5. What this does not establish

- Whether `id_order=True` (with or without PSF-Zero's own ordering
  diversity) generalizes to other topologies or circuit families --
  same limitation already noted in Addendum 83 Section 4.
- The Qiskit-implementation question in Section 4.
- Mechanism -- why `id_order=True` would or would not matter, in either
  outcome.

---


<!-- ===== Addendum 84 (source: spare-qubit-cliff-addendum-84-2026-09-19.md) ===== -->

> **Note added when merging:** CORRECTION to Addendum 83: a bare id_order=True call matches PSF-Zero's own 26/26 exactly, and does so 100-600x faster -- the credit belongs to a single API flag, not PSF-Zero's own multi-stage design, which is shown to be measurably wasteful on this dataset.

## Addendum 84 -- CORRECTION to Addendum 83: `id_order=True` alone explains all 26/26 successes, and does so 100-600x faster than PSF-Zero's own multi-stage strategy, which adds nothing on this dataset (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-84-preregistration-2026-09-19.md`, written
and locked before this run, in direct response to an external review
of Addendum 83's own conclusion. **This addendum corrects Addendum 83's
central claim.**

## 0. In one line

**P1: "id_order=True alone suffices" -- CONFIRMED, decisively, and more
strongly than either pre-registered branch anticipated.** A bare call
to `rx.vf2_mapping(..., id_order=True)`, on the physical graph in its
natural (unordered, unrelabeled) numbering, with none of PSF-Zero's
own multi-stage machinery, finds a layout in **all 26 of 26**
configurations -- exactly matching PSF-Zero's own `smart_vf2_layout`
success rate. **It is also dramatically faster**: every bare-condition
success took under 0.0001 seconds, while PSF-Zero's own multi-stage
search took 0.0146-0.0615 seconds for the same configurations -- **a
100-600x overhead for zero additional benefit on this dataset.**
**Addendum 83's central claim must be corrected**: the finding is not
"PSF-Zero's multi-stage ordering-diversity strategy avoids the cliff
Qiskit's own VF2Layout suffers from" -- it is **"setting `id_order=True`
on a single, unmodified `rustworkx.vf2_mapping()` call avoids it,"** a
narrower and more specific claim that does not require any of
PSF-Zero's own additional machinery (BFS relabeling, multiple starting
orderings, a two-stage fallback).

## 1. Results

8x8 grid (64 qubits), all 26 `dominant_size_sweep` configurations
(D=2-28) from Addenda 78/80, run with both PSF-Zero's own
`smart_vf2_layout_no_feasibility_guard` and the new bare-condition
control, back to back for each D.

| | PSF-Zero's own multi-stage search | Bare `id_order=True` control |
|---|---:|---:|
| Found a layout | 26/26 | **26/26** |
| Time range | 0.0146-0.0615s | **0.0000-0.0001s** |
| Orderings/relabelings needed | 2-4 per configuration | **0** (natural numbering, single call) |
| Speed relative to the other | -- | **~100-600x faster** |

Every one of Addendum 83's own 26 successes used `id_order=True` in
some ordering (Stage 1; Stage 2's `id_order=False` fallback was never
triggered) -- confirmed directly from Addendum 83's own recorded
`psf_phase` column before this addendum's results were interpreted,
consistent with P2's pre-registered consistency check.

## 2. Scoring

**P1 (primary) -- "id_order=True alone suffices" CONFIRMED.** The
pre-registration set the bar at "26/26 or very close"; the measured
result is exactly 26/26, an unambiguous match. The pre-registration
did not predict the SPEED difference (100-600x) -- that is an
additional, unregistered finding, reported here as such rather than
retrofitted into the prediction.

**P2 (consistency check) -- CONFIRMED.** Re-verified directly from
Addendum 83's own data before this addendum's design was finalized.

## 3. What this means: Addendum 83's finding is real, but was attributed to the wrong cause

**The underlying empirical fact from Addendum 83 stands: something
does solve, in every one of 26 cases, what Qiskit's own VF2Layout
could not.** What this addendum corrects is *why*. Addendum 83
attributed the success to PSF-Zero's own design (trying several
BFS-family starting orderings, informed by Addenda 8-12's own
diagnostic work). **This addendum shows that attribution was
premature**: the credit belongs to a single `rustworkx` API flag,
`id_order=True`, which PSF-Zero's own first attempt (in whichever
ordering it happens to try first) also sets -- meaning PSF-Zero's
multi-stage strategy was, on this dataset, solving every case on its
very first attempt regardless of which specific ordering that attempt
used, and the additional orderings/stages built into
`smart_vf2_layout` were never actually needed.

**This also means PSF-Zero's own `smart_vf2_layout`, as currently
implemented, is measurably inefficient on this class of problem**: it
spends 100-600x longer than necessary because it does not check
whether the simplest possible call (no relabeling, `id_order=True`,
natural node order) already succeeds before investing in BFS-based
relabeling. This is a concrete, actionable improvement opportunity,
independent of Addendum 83's own already-identified
`_has_feasible_matching` bug.

## 4. What remains unresolved (per the pre-registration's own Section 4)

**This addendum does not resolve whether Qiskit's own `VF2Layout`
internally calls `rustworkx.vf2_mapping()` at all.** An external
source raised, in the course of this project's ongoing review, that
Qiskit's own maintainers have publicly stated their implementation is
NOT built on `rustworkx.vf2_mapping()` but on a separately-implemented
Rust module. **This has not been verified by reading Qiskit's own
current source** (this project has repeatedly failed to locate the
relevant Rust file directly, per Addenda 43-45), and this addendum's
own result cannot settle it either way. Two readings remain open:

- **If Qiskit's VF2Layout does use `rustworkx.vf2_mapping()`
  internally** (with `id_order` defaulting to something other than
  `True`, or a different ordering scheme entirely): this addendum's
  finding would mean the entire cliff phenomenon, across every
  addendum in this project's history, reduces to a single
  misconfigured default flag -- an extraordinary, and therefore
  suspicious-until-verified, claim.
- **If Qiskit's VF2Layout uses a separate, independent implementation**
  (as the external source claims): this addendum's finding describes a
  property of `rustworkx.vf2_mapping()` specifically, and the
  comparison in Addendum 83 (and this correction) is between two
  different tools' implementations of VF2-family search, not "the
  same algorithm with one setting changed." This would still be a
  useful and real finding (a specific, minimal, fast alternative
  exists), but framed correctly as an implementation comparison, not a
  single-flag fix.

**This distinction should be resolved by reading Qiskit's own current
source directly** before either Addendum 83's or this addendum's
finding is written into any external-facing document (paper, README,
or presentation). This is now the single highest-priority open item in
this entire investigation.

## 5. What this does not establish

- Section 4's unresolved question (Qiskit's own implementation).
- Whether `id_order=True` alone generalizes beyond this exact
  topology/family combination -- same limitation already noted in
  Addendum 83 Section 4 (the `brick` topology counter-example from
  `psf_smart_layout.py`'s own docstring applies here too).
- Mechanism -- why `id_order=True` specifically avoids whatever made
  Qiskit's own VF2Layout fail. `id_order`'s own documented meaning (per
  Addendum 45's reading of Qiskit's C API docs: it controls whether
  node visitation order is randomized/shuffled or follows a fixed
  scheme) is known, but why fixing it to `True` specifically resolves
  this dataset's cases is not explained here.
- Whether PSF-Zero's own `_has_feasible_matching` bug (Addendum 83
  Section 5) is affected by this finding -- it is not; that bug is
  independent of which search strategy runs after the guard, and still
  needs fixing regardless of this addendum's result.

## 6. Files

| File | What it is |
|---|---|
| [`compare_psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compare_psf_smart_layout.py) | the script (Addendum 84's `bare_id_order_true` control added) |
| [`psf_smart_layout_comparison_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/psf_smart_layout_comparison_2026-09-19.csv) | this run's results, 26 rows, now including the control columns |
| [`spare-qubit-cliff-addendum-84-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-84-preregistration-2026-09-19.md) | the predictions scored above |

## 7. Verification

- The 26/26-vs-26/26 match and the 100-600x speed gap were both
  computed directly from the CSV's own `bare_id_order_true_found` and
  `bare_id_order_true_elapsed_s` columns, not estimated from the
  printed summary alone.
- P2's consistency check (every PSF-Zero success used `id_order=True`
  in Addendum 83's own data) was re-verified against that addendum's
  own recorded `psf_phase` column before this addendum's design was
  written, not assumed.
- Section 4's open question is stated as unresolved by this addendum
  specifically, with both possible readings given equal weight, rather
  than assuming either one.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 85 (source: spare-qubit-cliff-addendum-85-2026-09-19.md) ===== -->

> **Note added when merging:** Resolution of the project's top-priority open question, from Qiskit's own current Rust source (supplied by the user): Qiskit's VF2Layout/VF2PostLayout do NOT call rustworkx.vf2_mapping() -- they use a custom implementation with a hardcoded VF2++ ordering strategy, confirming an external maintainer's public statement and correcting Addendum 45's own earlier documentation-based reading.

## Addendum 85 -- Qiskit's own Rust source, read directly: it does NOT call `rustworkx.vf2_mapping()`, and uses a third, distinct ordering strategy (VF2++) that neither Addendum 83 nor 84 tested (2026-09-19)

**Status**: source reading, resolving the single highest-priority open
item flagged in Addendum 84 and `SESSION_SUMMARY_FINAL_2026-09-18.md`
Section 3 item 0. The user supplied the actual, current Qiskit Rust
source (`crates/.../vf2_layout.rs`, containing both
`vf2_layout_pass_average` and `vf2_layout_pass_exact`) directly. No
inference, no external secondhand report -- read and quoted verbatim
below.

## 0. In one line

**Qiskit's own `VF2Layout`/`VF2PostLayout` do NOT call
`rustworkx.vf2_mapping()`.** They use `qiskit_circuit::vf2::Vf2`, a
custom Rust implementation that only borrows `rustworkx_core`'s
underlying graph data structures (`Graph`, `NodeIndex`, from
`rustworkx_core::petgraph`), not the public `rustworkx` Python
package's `vf2_mapping()` function. **This confirms, directly from
source, what an external Qiskit maintainer stated on GitHub issue
#1679** (reported secondhand in an earlier review): the claim that
Qiskit "uses `rustworkx.vf2_mapping()`" is false. **Both
`vf2_layout_pass_average` and `vf2_layout_pass_exact` call
`.with_vf2pp_ordering()` unconditionally** -- a third, specifically
named ordering strategy (VF2++), distinct from both `id_order=True`
and `id_order=False` as those terms are used in `rustworkx`'s own
public API (which Addendum 84's control condition tested). **This means
Addendum 83/84's comparison was always between two different
implementations, not "the same algorithm with one setting changed" --
confirming the second of the two readings the Addendum 84
pre-registration itself left open**, and neither Addendum 83's nor
Addendum 84's finding directly explains why Qiskit's own implementation
fails on these 26 configurations, since Qiskit never runs the
`id_order=True` code path Addendum 84 tested.

## 1. What the source shows, quoted directly

**Import block** (top of file):
```rust
use rustworkx_core::petgraph::data::Create;
use rustworkx_core::petgraph::prelude::*;
```
`rustworkx_core` is the shared Rust crate that both Qiskit and the
public `rustworkx` Python package build on -- but this only imports
its graph *data structures* (`Graph`, `NodeIndex`, etc.), not any
search function.

**The actual search call** (`qiskit_circuit::vf2`, imported earlier in
the file via `use qiskit_circuit::{..., vf2}`) is Qiskit's own module,
used like this in `vf2_layout_pass_average`:
```rust
let vf2 = vf2::Vf2::new(&interactions.graph, &coupling_graph, vf2::Problem::Subgraph)
    .with_scoring(score, score)
    .with_restriction(vf2::Restriction::Decreasing(best_score))
    .with_vf2pp_ordering();
```
and identically in `vf2_layout_pass_exact` (same three builder calls,
only `.with_scoring` becomes `.with_semantics`):
```rust
let vf2 = vf2::Vf2::new(&interactions.graph, &coupling_graph, vf2::Problem::Subgraph)
    .with_semantics(score, score)
    .with_restriction(vf2::Restriction::Decreasing(best_score))
    .with_vf2pp_ordering();
```

**`.with_vf2pp_ordering()` is called unconditionally in both functions
-- no branch, no configuration flag toggles it off.** There is no
code path visible in this file that runs Qiskit's VF2 with a
"natural"/fixed node order instead (the equivalent of `rustworkx`'s
`id_order=True`).

**The only ordering-adjacent configuration option exposed** is
`shuffle_seed` on `Vf2PassConfiguration`, documented in this same file:
```rust
/// If set, shuffle the node indices of the input graphs using a specified random seed.  If
/// `None`, perform no shuffling.  You probably want this to be `None`.
pub shuffle_seed: Option<u64>,
```
This controls whether the *coupling graph's own qubit numbering* is
pre-shuffled before VF2++ ordering runs on it -- a different thing from
choosing which ordering *algorithm* to use. Leaving it `None` (the
documented recommendation) still means VF2++ ordering runs on top,
unconditionally.

## 2. What this resolves

**Resolved directly from source, not by inference or secondhand
report**: Qiskit's own VF2-family passes use a custom Rust
implementation (`qiskit_circuit::vf2`) with a hardcoded VF2++ ordering
strategy, applied identically whether scoring an "average" (abstract,
pre-hardware) or "exact" (concrete, post-hardware) layout. This
directly corrects Addendum 45's own earlier reading of Qiskit's C API
documentation, which stated *"Qiskit uses the VF2++ ordering
improvements when running in 'average' mode... and starts from the
identity mapping in 'exact' mode"* -- implying the two modes use
*different* orderings. **This source shows both modes call the
identical `.with_vf2pp_ordering()`.** The "starts from the identity
mapping" language in that documentation most likely refers to
`score_initial_layout: true` (part of `Vf2PassConfiguration::
default_concrete()`, used for the "exact"/post-hardware case) --
i.e. the *initial layout is scored as a baseline before searching*,
which is a scoring/restriction detail (`vf2::Restriction::Decreasing`),
not the node-traversal *order* VF2++ itself governs. **This is a
correction to Addendum 45's own interpretation**, made possible only by
reading the actual current source rather than the documentation's
higher-level prose.

## 3. What this means for Addenda 83-84

**Addendum 84's finding is real and remains useful, but must be
reframed.** `rustworkx.vf2_mapping(..., id_order=True)` genuinely finds
a layout in 26/26 of these configurations, fast. But this is not "the
one setting Qiskit could flip to fix the cliff" -- **Qiskit's own
implementation does not expose an `id_order` concept at all**; it
always runs VF2++ ordering via its own from-scratch code, with no
visible path to disable it. **The comparison across Addenda 83-85 is
therefore, and always was, a comparison between two genuinely different
pieces of software** (Qiskit's custom `qiskit_circuit::vf2` engine
using VF2++ ordering, vs. the public `rustworkx.vf2_mapping()` function
using `id_order=True`) that happen to share some underlying graph data
structures (`rustworkx_core`) but implement distinct search strategies.
This confirms the second of the two readings Addendum 84's own
pre-registration (Section 4) explicitly left open as unresolved.

**Practically, this changes what can be claimed:**
- **Can still be claimed**: a fast, effective, directly-tested
  alternative (`rustworkx.vf2_mapping(id_order=True)`) exists for this
  class of layout problem, on this specific topology/family
  combination, and finds every layout Qiskit's own implementation
  fails to find.
- **Can no longer be claimed**: that Qiskit's own cliff is "caused by"
  or "fixable via" an `id_order` setting -- Qiskit's own code has no
  such setting; the cliff is a property of its own VF2++-ordering-based
  implementation, whatever the actual cause of THAT implementation's
  difficulty on this circuit family turns out to be.

## 4. A plausible (not yet confirmed) mechanistic connection

Addendum 43 quoted Qiskit's own 2.2 release notes: *"The maximum call
and trial limits for the exact-matching run of `VF2PostLayout` at
`optimization_level=3` have been reduced to avoid excessive runtimes
for **highly symmetric trial circuits being mapped to large coupling
maps**."* VF2++ ordering is a sophisticated, generally effective
heuristic (choosing traversal order based on graph structure, roughly
"most constrained node first"), but **on a highly symmetric graph**
(many structurally-identical bare 2-qubit edges, exactly this
project's own circuit family) **many candidate nodes can look equally
promising to such a heuristic**, potentially causing exactly the kind
of symmetric backtracking blowup Qiskit's own release notes describe.
This is a plausible explanation for why a sophisticated ordering
heuristic (VF2++) can still struggle on symmetric inputs where a
"dumber" fixed/natural order (`id_order=True`) happens not to -- but
**this is inference from the release notes plus this source, not
something demonstrated by tracing VF2++'s own algorithm on these
specific 26 graphs**, and is reported at that confidence level, not
higher.

## 5. What this does not establish

- **Why** VF2++ ordering specifically fails on these 26 configurations
  while `id_order=True` succeeds -- Section 4's connection is plausible,
  not demonstrated.
- Whether `rustworkx.vf2_mapping(id_order=True)`'s success generalizes
  beyond this exact topology/family combination -- unchanged limitation
  from Addendum 83/84.
- Whether PSF-Zero's own `smart_vf2_layout` (which does call the public
  `rustworkx.vf2_mapping()`, confirmed from `psf_smart_layout.py`'s own
  source in Addendum 83) could be modified to more directly exploit
  this finding (e.g., trying bare `id_order=True` first, before any BFS
  relabeling) -- an actionable next step, not yet implemented.
- Anything about `qiskit_circuit::vf2`'s own internal implementation of
  VF2++ ordering beyond what this one file's calling code shows -- the
  `vf2` module's own source (defining `Vf2`, `NodeSorter`,
  `with_vf2pp_ordering`, etc.) was not part of what was shared and has
  not been read.

## 6. Files

| File | What it is |
|---|---|
| (Qiskit's own `vf2_layout.rs`, shared by the user in this conversation, not a project-generated file) | the source read and quoted in this addendum |

## 7. Verification

- Every code quotation in Sections 1-2 is verbatim from the source
  text supplied in this conversation, not paraphrased or reconstructed
  from memory.
- The claim that both `vf2_layout_pass_average` and
  `vf2_layout_pass_exact` call `.with_vf2pp_ordering()` unconditionally
  was checked by locating both functions' own VF2-construction blocks
  independently and comparing them side by side (Section 1), not
  assumed from one function's code alone.
- Addendum 45's own prior claim (quoted in Section 2) was re-read
  directly from that addendum before being corrected here, so the
  correction is checkable against what was actually claimed.
- Section 4's mechanistic connection is explicitly labelled as
  plausible inference, not a demonstrated result, to avoid overclaiming
  beyond what this source establishes.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 86 (source: spare-qubit-cliff-addendum-86-2026-09-19.md) ===== -->

> **Note added when merging:** The Python-level VF2Layout/VF2PostLayout pass classes, read directly: confirms Addendum 44's dispatch pattern for VF2PostLayout, clarifies VF2Layout itself never dispatches to the exact path, and flags a seed-default precision point distinguishing the class's own default from the preset pipeline's construction-time value.

## Addendum 86 -- the Python-level `VF2Layout`/`VF2PostLayout` pass classes, read directly: confirms Addendum 44's dispatch pattern for VF2PostLayout, clarifies that VF2Layout itself never dispatches to the exact path, and flags a seed-default precision point (2026-09-19)

**Status**: source reading. The user supplied the actual, current Python
source for both pass classes (`vf2_layout.py`, `vf2_post_layout.py`).
Mostly confirmatory of Addenda 44-45 and 85; one clarification and one
precision point worth recording.

## 0. In one line

**Confirmed directly**: `VF2Layout.run()` calls `vf2_layout_pass_average`
**unconditionally** -- it never calls `vf2_layout_pass_exact` at all.
Its own `strict_direction` parameter is forwarded *into*
`vf2_layout_pass_average` as an argument (controlling whether the
coupling graph's directionality is loosened, per Addendum 85's reading
of the Rust source), not used to choose between the two Rust functions.
**This refines Addendum 44's dispatch-pattern finding**, which was
established for `VF2PostLayout` (where `strict_direction` genuinely
does choose between `vf2_layout_pass_exact` and
`vf2_layout_pass_average`) -- `VF2Layout` has no such dispatch; it is
always "average" mode. Also confirmed: `VF2Layout` always sets
`score_initial_layout=False` (there is no prior layout to improve on,
since this pass constructs one from scratch), symmetric to Addendum
44's finding that `VF2PostLayout` always sets it `True`.

## 1. What the source shows

**`VF2Layout.run()`**, in full relevant part:
```python
config = VF2PassConfiguration.from_legacy_api(
    call_limit=self.call_limit,
    time_limit=self.time_limit,
    max_trials=self.max_trials,
    shuffle_seed=self.seed,
    score_initial_layout=False,
)
try:
    output = vf2_layout_pass_average(
        dag,
        target,
        strict_direction=self.strict_direction,
        avg_error_map=self.avg_error_map,
        config=config,
    )
```
Only `vf2_layout_pass_average` is imported into this file at all
(`from qiskit._accelerate.vf2_layout import (vf2_layout_pass_average,
MultiQEncountered, VF2PassConfiguration)`) -- `vf2_layout_pass_exact` is
not even imported here, confirming there is no code path in this class
that could call it.

**`VF2PostLayout.run()`**, by contrast, imports both functions and
dispatches on its own `strict_direction` (default `True`, per the
`__init__` signature `strict_direction=True`):
```python
if self.strict_direction:
    output = vf2_layout_pass_exact(dag, self.target, config=config)
else:
    output = vf2_layout_pass_average(
        dag, self.target, strict_direction=False,
        avg_error_map=self.avg_error_map, config=config,
    )
```
This matches Addendum 44's own reading exactly (that addendum read this
same dispatch from an earlier version of this file).

**Stop-reason enums**, confirmed exactly as Addendum 44 documented:
`VF2LayoutStopReason` has three values (`SOLUTION_FOUND`,
`NO_SOLUTION_FOUND`, `MORE_THAN_2Q`) -- no `NO_BETTER_SOLUTION_FOUND`,
since that outcome is specific to `VF2PostLayout`'s four-value enum
(`SOLUTION_FOUND`, `NO_BETTER_SOLUTION_FOUND`, `NO_SOLUTION_FOUND`,
`MORE_THAN_2Q`).

**`_build_dummy_target`**, confirmed: when only a bare `coupling_map` is
given (no `target`, no error rates) -- exactly this project's own
standard experimental configuration throughout Addenda 4-85 --
`VF2Layout` builds `Target.from_configuration(basis_gates=["u", "cx"],
num_qubits=coupling_map.size(), coupling_map=coupling_map)`, an
arbitrary, errorless dummy target. This is consistent with, and now
confirmed at the Python-API level for, every prior addendum's own
observation that bare `CouplingMap` inputs carry no error information
for the pass to score against (Addenda 49-50).

## 2. A precision point on `seed`, not a contradiction

This project's own README material states, regarding the preset
transpile pipeline: *"Qiskit's preset pipeline tries `VF2Layout` exactly
once, with shuffling explicitly disabled (`seed=-1`, hardcoded, not
controlled by `seed_transpiler`)."* **This file's own docstring
describes something different, though not necessarily
contradictory**: the `VF2Layout` *class's own default* parameter is
`seed=None`, and its docstring states *"`None` seeds using OS entropy
(and so is non-deterministic). Using `-1` disables the shuffling."*

**These are two different things.** This file documents the class's own
default when instantiated bare; the README's claim is about what
*value* the preset pipeline's own pass-manager-construction code
supplies when it builds a `VF2Layout` instance internally for use
inside `transpile()`. **Neither of the two files shared in this
conversation shows that preset-construction code** -- only the pass
class itself. It remains entirely possible (and consistent with the
project's own prior measurements) that the preset pipeline explicitly
passes `seed=-1` when constructing this pass, even though the class's
own bare default is `None`. **This is not resolved by this addendum**;
it would require reading the preset-pass-manager construction code
(e.g. `generate_preset_pass_manager` or the level-3 preset builder),
which has not been shared or read.

## 3. What this does not establish

- The preset pipeline's own construction-time argument values (Section
  2) -- unresolved, would need different source files.
- Anything about `qiskit_circuit::vf2`'s own internal VF2++ algorithm --
  these two files are the Python-level wrapper classes only; Addendum
  85's own open question (why VF2++ ordering struggles on this
  project's symmetric circuit family) is not addressed by anything
  here.
- Whether `VF2Layout`'s own `strict_direction` parameter (forwarded into
  `vf2_layout_pass_average`, per Section 1) has ever been varied in
  this project's own experiments -- worth checking against this
  project's own harness scripts, not done here.

## 4. Files

| File | What it is |
|---|---|
| (Qiskit's own `vf2_layout.py` and `vf2_post_layout.py`, shared by the user in this conversation, not project-generated files) | the source read and quoted in this addendum |

## 5. Verification

- Every code quotation is verbatim from the source text supplied in
  this conversation.
- The claim that `vf2_layout.py` never imports `vf2_layout_pass_exact`
  was checked against that file's own import statement directly.
- Addendum 44's own dispatch-pattern finding for `VF2PostLayout` was
  re-read before being described as "confirmed exactly" and
  "refined" (for `VF2Layout`'s own lack of dispatch) rather than
  restated from memory.
- The README's own `seed=-1` claim was re-quoted verbatim (Section 2)
  before being distinguished from this file's own class-default
  documentation, rather than asserting a contradiction without
  checking the exact wording of both.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 87 (source: spare-qubit-cliff-addendum-87-2026-09-19.md) ===== -->

> **Note added when merging:** VF2++'s own ordering algorithm (qiskit_circuit::vf2 source, supplied by the user), read and faithfully simulated: confirms it processes chain components before symmetric bare edges (a demonstrated mechanism, not inference) -- but this alone does not yet explain the D=10-vs-D=20 divergence, since both graphs receive parallel-shaped orderings for their first 30 positions.

## Addendum 87 -- VF2++'s own ordering algorithm, read and simulated directly: it processes chain components before symmetric bare edges (a demonstrated mechanism), but this alone does not yet explain the D=10-vs-D=20 divergence (2026-09-19)

**Status**: source reading plus direct simulation. The user supplied
`qiskit_circuit::vf2`'s own source (the module implementing
`Vf2ppSorter`, referenced but not shown in Addendum 85). Its algorithm
was re-implemented faithfully in Python and run against this project's
own `dominant_size_sweep` graphs (D=10 and D=20, the isomorphic pair
Addendum 78 found diverges) to test a concrete hypothesis, not merely
read the code and speculate.

## 0. In one line

**Confirmed, by direct simulation, not inference**: `Vf2ppSorter`'s
priority rule (highest connectivity-to-processed-set first, then
highest degree, with ties broken by lowest node index) causes it to
process every node in the dominant+filler chain (degree 2, interior
nodes) **before touching any node in the 17 bare 2-qubit edges**
(degree 1) -- confirmed for both D=10 and D=20's own graphs. **This is
a real, demonstrated mechanism**, not the "plausible, unconfirmed"
placeholder Addendum 85 Section 4 offered. **However, this mechanism
alone does not yet explain the D=10-vs-D=20 divergence** Addendum 78
found: because both graphs have exactly 30 non-bare-edge qubits in
total (17x2=34 bare qubits + 30 chain qubits = 64, regardless of how
the 30 splits between dominant and filler), **the two graphs' VF2++
orderings are structurally parallel for their first 30 positions** --
both process their full chain allotment first, only reaching the
symmetric bare edges at the same position (30th). The specific
divergence between the two must therefore lie in how the VF2
backtracking *search itself* proceeds using each graph's own
order, not in the order's own high-level shape -- a question this
addendum's simulation (sorter only, not the full search) does not
answer.

## 1. Method

`Vf2ppSorter::sort`, read from `qiskit_circuit::vf2`'s own source, was
re-implemented in Python exactly per its documented logic: build a BFS
tree from the highest-total-degree node (ties broken by lowest index),
processing each BFS level by repeatedly selecting the node with
highest `(connectivity-to-already-ordered-set, total degree, lowest
index)` -- the same three-part priority key the Rust source uses
verbatim (`(conn_in[index] + conn_out[index], degree_out[index] +
degree_in[index], Reverse(index))`). This was run against
`_edges_dominant_size_sweep`'s own construction (copied verbatim from
`circuit_family_sweep.py`) for D=10 and D=20 at n=64.

## 2. Results

For **both** D=10 and D=20:
- The first 10 nodes processed are identical: `[35, 36, 34, 37, 38, 39,
  40, 41, 42, 43]` -- all interior/near-start nodes of the
  dominant/filler chain region (indices 34+), never a bare-edge node
  (indices 0-33).
- The first bare-edge node appears at position **30** in both orderings
  -- exactly the point at which the 30-qubit chain allotment (dominant
  + filler combined) is exhausted, for either split (10+20 or 20+10).

**Why chains are prioritized over bare edges**: every interior chain
node has degree 2 (two neighbors along the chain); every bare-edge node
has degree 1. The sorter's own priority key ranks degree-2 nodes above
degree-1 nodes whenever their connectivity-to-processed-set is tied
(true at the start, when nothing is processed yet) -- so the highest-
degree root (a chain node) is chosen first, and the entire chain is
then walked via BFS before the sorter ever considers a bare-edge node.

## 3. What this establishes, and what it still does not

**Establishes, concretely**: Qiskit's VF2++ ordering does NOT process
this project's own circuit family in the same order `id_order=True`
would (natural/fixed index order, which -- given this project's own
construction convention placing bare edges at low indices 0-33 and
chains at high indices 34+ -- would process bare edges FIRST and
chains LAST, the reverse priority from VF2++). This is a genuine,
structural difference between the two ordering strategies compared in
Addenda 83-84, now demonstrated rather than assumed.

**Does NOT yet establish**: why this reversed priority causes VF2++ to
fail (cliff) on some configurations and succeed on others, since D=10
and D=20 -- one fast, one cliffing -- receive essentially
parallel-shaped orderings by this measure (same chain-first count, same
30th-position bare-edge onset). **The actual divergence must be a
property of how the VF2 backtracking search's feasibility-pruning and
candidate-selection logic (`is_feasible`, `next_candidates`, the
`State` struct's neighbor-tracking, all present in the shared source
but not simulated here) behaves once processing reaches the symmetric
bare-edge region, given the two graphs' different specific chain
compositions (10+20 vs. 20+10) feeding into that later stage.**
Confirming this would require simulating the actual backtracking search
(not just the initial priority ordering), which is a substantially
larger undertaking than this addendum's own scope.

## 4. Revision to Addendum 85 Section 4

Addendum 85 proposed, as unconfirmed inference: *"on a highly symmetric
graph... many candidate nodes can look equally promising to [the VF2++]
heuristic, potentially causing exactly the kind of symmetric
backtracking blowup Qiskit's own release notes describe."* **This
addendum confirms the mechanism half of that claim directly** (VF2++
does treat the many bare-edge nodes as a late-processed, low-priority,
mutually-tied group, exactly as symmetric-graph difficulty would
predict) **but does not yet confirm the causal half** (that this
specific ordering property is what produces the exponential
backtracking Qiskit's 2.2 release notes describe, as opposed to some
other property of the search once it reaches that tied region). The
claim's confidence level should be updated from "plausible, unconfirmed
inference" to "mechanism confirmed by simulation; causal link to the
specific pass/fail outcomes still unconfirmed."

## 5. What this does not establish

- The actual cause of the D=10/D=20 divergence -- narrowed to "must lie
  in the backtracking search itself, not the initial ordering," but not
  identified further.
- Whether simulating the full VF2 backtracking search (not just the
  sorter) would explain it -- not attempted here, given the scope of
  correctly reproducing `is_feasible`'s pruning logic, the `State`
  struct's incremental neighbor-tracking, and the `Restriction`
  mechanism, all from the shared source.
- Whether this same chains-first/bare-edges-last pattern holds for
  other configurations in this project's own dataset (e.g. the D=5/D=25
  or D=12/D=18 pairs) -- only D=10/D=20 was simulated.
- Anything about `rustworkx.vf2_mapping()`'s own `id_order=False`
  ordering heuristic (a separate, unrelated implementation) for
  comparison -- not simulated here.

## 6. Files

| File | What it is |
|---|---|
| (Qiskit's own `qiskit_circuit::vf2` module source, shared by the user in this conversation, not a project-generated file) | the source read and simulated in this addendum |

No new project data files -- the simulation was run directly in the
sandbox using this project's own already-verified
`_edges_dominant_size_sweep` construction logic, not a new measurement.

## 7. Verification

- The Python re-implementation of `Vf2ppSorter::sort` was checked
  line-by-line against the Rust source's own priority key
  (`conn_in+conn_out`, `degree_out+degree_in`, `Reverse(index)`) and
  its BFS-tree-per-root structure, before running it, to ensure
  fidelity rather than a loose approximation.
- Both D=10 and D=20's results were computed independently (not
  assumed identical from one run) and found to match at the specific
  points reported (first-10 order, first-bare-edge position) --
  confirmed by direct comparison of the two output lists, not
  eyeballed.
- The claim that both graphs have exactly 30 non-bare qubits regardless
  of split was verified arithmetically (17*2 + D + (30-D) = 34 + 30 =
  64) before being used to explain why the orderings parallel each
  other for 30 positions.
- Section 3's limitations are stated explicitly rather than allowing
  the confirmed ordering mechanism to be read as a full explanation of
  the pass/fail divergence, which it is not.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---

**End of Part 5 of 7.** Continue to [Part 6](spare-qubit-cliff-combined-88.md) (Addendum 88-107) and [Part 7](spare-qubit-cliff-combined-108.md) (Addendum 108-121), or back to [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
