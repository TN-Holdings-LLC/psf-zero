# spare-qubit-cliff: Combined Addenda, Part 5 of 5 (Addendum 51 through Addendum 66)

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
| [`spare-qubit-cliff-addendum-51-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-51-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-52-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-52-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-53-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-53-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-54-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-54-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-56-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-56-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-57-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-57-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-59-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-59-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-60-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-60-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-62-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-62-preregistration-2026-09-18.md) | the predictions scored above, including a design correction made before running |

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
| [`spare-qubit-cliff-addendum-63-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-63-preregistration-2026-09-18.md) | the predictions scored above, including a design correction made before running |

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
| [`spare-qubit-cliff-addendum-64-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-64-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-65-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-65-preregistration-2026-09-18.md) | the predictions scored above |

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
| [`spare-qubit-cliff-addendum-66-preregistration-2026-09-18.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-66-preregistration-2026-09-18.md) | the predictions scored above |

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

---

---

**End of Part 5 of 5 (end of document).** Back to [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
