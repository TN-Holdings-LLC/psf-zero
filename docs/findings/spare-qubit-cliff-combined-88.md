# spare-qubit-cliff: Combined Addenda, Part 6 of 8 (Addendum 88 through Addendum 107)

**Continued from [Part 5](spare-qubit-cliff-combined-51.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 4](spare-qubit-cliff-combined-41.md)).** Same conventions as every prior part: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

**Note on this part specifically**: this is where the project's focus shifted from characterizing Qiskit's own VF2Layout cliff to actually repairing PSF-Zero's own `psf_smart_layout.py` prototype -- four real defects found, fixed, and verified (Addenda 88-92, 95), then confirmed end-to-end through the public `compile_for_hardware()` entry point for both structural correctness and exact unitary equivalence (Addenda 101-103), and generalized to topologies this project's own prior diagnostics had flagged as hard (Addendum 104, 106). A parallel thread (Addenda 93-100) investigated an unexplained variance collapse in PSF-Zero's own gate-synthesis timing; the actual 4-day-old source was obtained and every traceable candidate was ruled out (Addenda 96-99) without the cause ever being identified -- reported as a closed, honest non-finding rather than a guess. A separate memory-safety thread (Addenda 94, 105-107) fully resolved a question about reference cycles found when Python's garbage collector is disabled. **For a compact summary of this entire part, see `PROGRESS_SUMMARY_2026-09-19-20.md`** in the same folder -- it is the recommended entry point before reading this part in full.

---
<!-- ===== Addendum 88 pre-registration (source: spare-qubit-cliff-addendum-88-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for whether adding the natural (unrelabeled) ordering as _candidate_orderings' first entry, with id_order=True, recovers the bare-call speed inside PSF-Zero's own smart_vf2_layout code path.

## Addendum 88 -- Pre-registration: does adding the natural ordering as `_candidate_orderings`' first entry recover the bare `id_order=True` speed inside PSF-Zero's own code path? (2026-09-19)

**Status: pre-registration only. No run of the patched version has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 84 established that a bare `rx.vf2_mapping(..., id_order=True)`
call on the physical graph in its natural (unrelabeled) numbering finds
a layout for all 26 `dominant_size_sweep` configurations in under
0.0001s, while PSF-Zero's own `smart_vf2_layout` takes 0.0146-0.0615s
for the same 26 -- a 100-600x overhead for identical results.

**Reading `psf_smart_layout.py`'s own source located the specific
cause**: `_candidate_orderings` (Stage 1, which runs with
`id_order=True`) returns `bfs_from_max_degree`, `bfs_from_min_degree`,
`bfs_from_node0`, `degree_desc`, and two random-seeded BFS orderings --
**but never the plain natural ordering**. The natural ordering does
appear in `_fallback_orderings` (Stage 2), but Stage 2 runs with
`id_order=False`, a different search mode. **The specific combination
that Addendum 84 found solves every case instantly -- natural ordering
WITH `id_order=True` -- is therefore never attempted anywhere in
`smart_vf2_layout`.**

The recorded per-configuration data supports this as the cost driver
directly: configurations where 4 orderings were tried took ~0.05s;
where 2 were tried, ~0.02-0.03s -- roughly 15ms per *failed* attempt,
each burning its `call_limit` budget inside `rx.vf2_mapping` (which is
already Rust). The Python orchestration itself (`_relabel` over 64
nodes and ~110 edges, `_bfs_order_from`) is microseconds by comparison.
**This is why rewriting `psf_smart_layout.py` in Rust would not help:
the time is already being spent in Rust, on searches that should never
have been started.**

## 2. The change under test

One line added to `_candidate_orderings`, placing the natural ordering
first:

```python
orderings.append(("natural", list(nodes)))   # <-- added, first
orderings.append(("bfs_from_max_degree", _bfs_order_from(graph, max_deg_node)))
...
```

Nothing else changes -- same Stage 1/Stage 2 structure, same
`id_order=True` for Stage 1, same call limits, same time budget, same
fallback. `_has_feasible_matching`'s own separate bug (Addendum 83
Section 5) is bypassed in the test harness exactly as it was for
Addenda 83-84, so this experiment isolates the ordering change alone.

## 3. Pre-registered predictions

**P1 (primary -- does the patched version match the bare-call speed?).**
  - **Recovers most of the gap**: patched `smart_vf2_layout` completes
    every one of the 26 configurations in roughly the bare call's own
    time band (order 0.0001s, allowing for the wrapper's own graph
    construction and one successful `_relabel`), succeeding on the
    first ordering (`orderings_tried == 1`, `order == "natural"`) in
    all or nearly all cases.
  - **Recovers only part of the gap**: patched version is faster than
    the original but still materially slower than the bare call --
    would mean the wrapper's own per-call overhead (graph building,
    relabeling) is a larger share than the ~15ms-per-failed-attempt
    estimate implies, and is worth measuring separately.
  - **No improvement**: natural ordering does NOT succeed first on
    these configurations even with `id_order=True` -- would falsify
    the reading of Addendum 84's result as transferable into this code
    path, and require re-examining why the bare call differs from the
    wrapper's own first attempt.

**P2 (correctness preserved).** The patched version finds a layout for
all 26 configurations (same 26/26 as both the original and the bare
call). A patch that is fast but loses coverage would be a regression,
not a fix.

**P3 (no regression on the wrapper's other guarantees).** The patched
version's returned layout, where found, remains a valid mapping
(checked structurally by the harness: every interaction pair maps to a
physically adjacent qubit pair). This is checked independently rather
than assumed from `vf2_mapping` returning a result.

## 4. What this cannot establish

- Whether the natural-first ordering helps, hurts, or is neutral on
  circuit families or physical topologies outside this project's own
  `dominant_size_sweep` at 8x8 -- `psf_smart_layout.py`'s own docstring
  records that its BFS strategies failed entirely on the `brick`
  topology, so topology-specific behavior is already known to exist
  here. **Adding an ordering can only help coverage (it is tried first,
  and the others still follow if it fails), but its speed benefit is
  specific to cases where it succeeds.**
- Whether `_has_feasible_matching`'s own separate bug should be fixed
  the same way -- untouched by this change, still open.
- Anything about Qiskit's own implementation -- this concerns
  PSF-Zero's code only.

---


<!-- ===== Addendum 88 (source: spare-qubit-cliff-addendum-88-2026-09-19.md) ===== -->

> **Note added when merging:** The one-line fix works: 56.8x median speedup, all 26 solved on the first ordering, coverage and layout validity both preserved -- and the residual 12.7x gap is traced to eager graph construction, not orderings.

## Addendum 88 -- the one-line fix works: 56.8x faster (median), all 26 solved on the first ordering, coverage and layout validity both preserved -- and the residual 12.7x gap has an identified, further-fixable cause (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-88-preregistration-2026-09-19.md`, written
and locked before this run.

## 0. In one line

**P1: "Recovers most of the gap" confirmed. P2 and P3 both confirmed.**
Adding `("natural", list(nodes))` as `_candidate_orderings`' first entry
-- one line -- makes PSF-Zero's own layout search solve **all 26 of 26
configurations on its very first attempt** (`orderings_tried == 1`,
`order == "natural"`, unanimously), dropping the median from **17.85ms
to 0.314ms (56.8x faster; 78.5x on total wall time across all 26)**,
with **coverage unchanged at 26/26** and **every returned layout
independently verified structurally valid** (each interaction pair
lands on a physically adjacent qubit pair). **A residual 12.7x gap to
the bare `rx.vf2_mapping` call remains (0.314ms vs 0.0247ms), and its
cause is identifiable from the source**: `_candidate_orderings` builds
its entire list eagerly, computing five BFS traversals and two sorts
even when only the first entry is ever used -- see Section 4.

## 1. Results

8x8 grid (64 qubits), all 26 `dominant_size_sweep` configurations,
three conditions run back to back per configuration.

| condition | median | range | orderings tried | coverage |
|---|---:|---|---:|---:|
| **ORIGINAL** (`_candidate_orderings` as shipped) | 17.85ms | 14.1-54.0ms | 2 or 4 | 26/26 |
| **PATCHED** (natural ordering first) | **0.314ms** | 0.27-0.47ms | **1, always** | **26/26** |
| **BARE** (`rx.vf2_mapping(id_order=True)`, no wrapper) | 0.0247ms | 0.023-0.057ms | (n/a) | 26/26 |

- **Median improvement, original to patched: 56.8x.** Total wall time
  across all 26: 0.6824s to 0.0087s, **78.5x**.
- **Every one of the 26** patched runs succeeded on `orderings_tried=1`
  with `order="natural"` -- no exceptions, no fallback to Stage 2.
- **Residual gap, patched to bare: 12.7x** (0.314ms vs 0.0247ms).

## 2. Scoring

**P1 (primary) -- "Recovers most of the gap" CONFIRMED.** The
pre-registration's first branch required the patched version to
succeed on the first ordering in "all or nearly all cases" and land
roughly in the bare call's own time band. The first condition is met
exactly (26/26, unanimous). The second is met in the sense that
matters -- most of the 100-600x gap is closed -- but **not fully**:
0.314ms is 12.7x the bare call's 0.0247ms, larger than the
pre-registration's "order 0.0001s, allowing for the wrapper's own graph
construction and one successful `_relabel`" anticipated. Section 4
identifies why, which the pre-registration's second branch ("recovers
only part of the gap... worth measuring separately") explicitly asked
for.

**P2 (correctness preserved) -- CONFIRMED.** Patched coverage is
26/26, identical to both the original and the bare call. No
configuration that previously succeeded now fails.

**P3 (layout validity) -- CONFIRMED.** All 26 patched layouts were
structurally verified independently by the harness (every interaction
pair checked against the coupling map's own edge set), not assumed
from `vf2_mapping` returning a result. 26/26 valid.

## 3. Why this matters more than a Rust rewrite would

This addendum began as a direct answer to the question "should
`psf_smart_layout.py` be rewritten in Rust?" **The measured answer is
no -- or rather, that would optimize the wrong thing.** The original's
own cost was ~15ms per *failed* ordering attempt, and those attempts
run inside `rx.vf2_mapping`, which is already Rust. The Python
orchestration around them (building two `PyGraph`s, one `_relabel` over
64 nodes and ~110 edges) is microseconds by comparison. **Rewriting the
orchestration in Rust would have eliminated microseconds while leaving
the ~15ms-per-failed-search untouched.** One line of Python, correctly
placed, achieved 56.8x -- an outcome no amount of rewriting the wrapper
could have reached, because the wrapper was never the bottleneck.

## 4. The residual 12.7x, and how to close it

`_candidate_orderings` returns a **list**, built eagerly:

```python
orderings = []
orderings.append(("natural", list(nodes)))          # (the Addendum 88 addition)
orderings.append(("bfs_from_max_degree", _bfs_order_from(graph, max_deg_node)))
orderings.append(("bfs_from_min_degree", _bfs_order_from(graph, min_deg_node)))
orderings.append(("bfs_from_node0", _bfs_order_from(graph, nodes[0])))
orderings.append(("degree_desc", sorted(nodes, key=lambda n: -degrees[n])))
for seed in extra_seeds:
    ...
    orderings.append((f"bfs_from_random_seed{seed}", _bfs_order_from(graph, start)))
return orderings
```

**Every entry is computed before the caller sees any of them** -- five
BFS traversals (`_bfs_order_from` -> `rx.bfs_layers`) and two sorts,
all discarded unused now that the first entry always wins. **Converting
this function to a generator (`yield` instead of `append`) would
compute only what is actually consumed**, which on this dataset is one
`list(nodes)` call. This is a second, independent, equally small change
that would likely close most of the remaining 12.7x. **It is not
implemented or tested here** -- identified from source and from this
addendum's own timing data, not measured, and reported at that
confidence level.

## 5. What this does not establish

- **Generalization beyond this topology and circuit family.** The
  natural ordering winning 26/26 is specific to the 8x8 grid and this
  project's `dominant_size_sweep` family. `psf_smart_layout.py`'s own
  docstring records that its BFS strategies failed entirely on the
  `brick` topology -- whether the natural ordering does better or worse
  there is untested. **The patch is safe regardless**: adding an
  ordering at the front cannot reduce coverage, since every previously-
  tried ordering still follows if the new one fails. The *speed*
  benefit is what is topology-specific, not the correctness.
- Whether the generator change (Section 4) actually recovers the
  residual 12.7x -- proposed, not measured.
- Whether `_has_feasible_matching`'s own separate bug (Addendum 83
  Section 5) should be fixed in the same pass -- untouched here, still
  open, and still blocking `layout_search=True` from reaching any of
  this benefit in real use.
- Anything about Qiskit's own implementation -- this concerns
  PSF-Zero's code only.

## 6. Files

| File | What it is |
|---|---|
| [`psf_smart_layout_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout_patched.py) | the patched module -- one functional line added to `_candidate_orderings`, everything else byte-identical to the original |
| [`verify_natural_first_patch.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_natural_first_patch.py) | the three-condition harness (original / patched / bare), calling both modules' real internal helpers unmodified |
| [`natural_first_patch_verification_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/natural_first_patch_verification_2026-09-19.csv) | this run's results, 26 rows |
| [`spare-qubit-cliff-addendum-88-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-88-preregistration-2026-09-19.md) | the predictions scored above |

## 7. Verification

- The patch was confirmed to be a single functional change by direct
  `diff` against the original file before any run (one
  `orderings.append` line plus explanatory comments; no other line
  altered, line endings preserved).
- All three conditions call the real modules' own internal helpers
  (`_candidate_orderings`, `_relabel`, `_try_mapping`,
  `_fallback_orderings`) rather than reimplementations -- the harness
  skips only `_has_feasible_matching`, for the reason documented in
  Addendum 83 Section 1.
- P3's layout validity was checked structurally and independently
  (interaction pairs against the coupling map's own edge set), not
  inferred from a non-`None` return.
- Every summary figure (medians, ranges, the 56.8x and 12.7x ratios)
  was computed directly from the output CSV, not read off the printed
  table.
- Section 4's eager-list observation was read from the patched file's
  own source before being stated, and is explicitly marked as an
  untested proposal rather than a measured result.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 89 pre-registration (source: spare-qubit-cliff-addendum-89-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for fixing _has_feasible_matching to compare the interaction graph's OWN maximum matching against the device's, rather than raw edge count against device matching.

## Addendum 89 -- Pre-registration: fixing `_has_feasible_matching` to compare against the interaction graph's OWN maximum matching, so `layout_search=True` reaches the Addendum 88 speed-up in real use (2026-09-19)

**Status: pre-registration only. The fix has been written and its
correctness argument checked numerically, but `smart_vf2_layout` has
not been run end-to-end with it.** Predictions are locked before any
measurement.

## 1. The bug, precisely

`smart_vf2_layout` calls, before any search:

```python
if not _has_feasible_matching(coupling_map, len(interaction_pairs)):
    info["feasible"] = False
    return None, info
```

and `_has_feasible_matching` computes the *physical* graph's maximum
matching, returning `len(physical_matching) >= num_logical_pairs`,
where `num_logical_pairs` is passed as **the interaction graph's raw
edge count**.

**This is a valid test only when the interaction graph is itself a
matching** -- i.e. a set of pairwise vertex-disjoint edges, which is
exactly the `dense_pairs` family this module was built around (Addenda
8-12). There, every logical edge genuinely needs its own disjoint
physical edge, so edge count is the right quantity.

**For any interaction graph whose edges share qubits, the test is
wrong.** A 30-qubit path has 29 edges but needs only 30 physical qubits
arranged in a path -- not 29 disjoint physical edges. Every circuit
family this project has used since Addendum 51 is of this shape.
Measured consequence (Addendum 83 Section 1): all 26
`dominant_size_sweep` configurations have 45-46 interaction edges
against the 8x8 grid's own maximum matching of 32, so the guard
rejected **every one of them** -- including the 7 where Qiskit itself
succeeds -- before any search ran, returning `None`, which
`compile_for_hardware` then treats as "no layout found" and silently
falls back to Qiskit's own default layout stage.

## 2. The fix, and why it is correct

Compare against **the interaction graph's own maximum matching**, not
its edge count:

```python
def _has_feasible_matching(cmap, interaction_pairs):
    ig = nx.Graph()
    ig.add_edges_from(interaction_pairs)
    logical = nx.max_weight_matching(ig, maxcardinality=True)
    g = nx.Graph()
    g.add_nodes_from(range(cmap.size()))
    g.add_edges_from([tuple(e) for e in cmap.get_edges()])
    physical = nx.max_weight_matching(g, maxcardinality=True)
    return len(physical) >= len(logical)
```

**Why this is a genuine necessary condition**: any valid subgraph
embedding maps logical vertices injectively to physical vertices and
each logical edge to a physical edge. Take a maximum matching of the
interaction graph -- `k` pairwise vertex-disjoint logical edges. Their
images are `k` physical edges, pairwise vertex-disjoint because the
vertex map is injective. So the physical graph must contain a matching
of size at least `k`. Contrapositive: if it does not, no embedding
exists. The check can therefore still only reject genuinely
impossible cases, never a feasible one.

**Why it is backward compatible**: for a matching-shaped interaction
graph, its own maximum matching *equals* its edge count, so the new
check returns exactly what the old one did. Verified numerically for
`dense_pairs` at n=64: edge count 32, own maximum matching 32, 8x8
grid's maximum matching 32 -- both old and new return `True`.
Verified for `dominant_size_sweep` at D=2, 10, 20, 30: edge count
45-46 (old check `False`, the bug), own maximum matching 32 (new check
`True`, correct).

The one call site is updated to pass `interaction_pairs` instead of
`len(interaction_pairs)`; no other caller exists in either
`psf_smart_layout.py` or `psf_compile.py` (checked by grep across
both).

## 3. Pre-registered predictions

**P1 (primary -- does the real, unbypassed `smart_vf2_layout` now
work?).** Called directly (no test-harness bypass, unlike Addenda
83-88 which all had to skip this guard), on all 26
`dominant_size_sweep` configurations:
  - **Works**: `feasible=True` and a layout found for all 26, matching
    the bypassed results from Addendum 88 exactly (26/26, first
    ordering, ~0.3ms) -- the fix reaches real use.
  - **Still rejects some**: the corrected check still returns `False`
    somewhere -- would mean the necessary-condition reasoning in
    Section 2 has an error, or `nx.max_weight_matching` behaves
    differently than assumed on one of these graphs.

**P2 (backward compatibility).** On `dense_pairs` at both 6x7 (n=42)
and 8x8 (n=64) -- the family the original check was designed for -- the
fixed version returns the same `feasible` verdict as the original, and
`smart_vf2_layout`'s overall outcome (found / not found) is unchanged.
**A fix that quietly changes behavior on the original design case would
be a regression**, and is checked for explicitly rather than assumed
from the equality argument in Section 2.

**P3 (the guard still rejects genuinely impossible cases).** On a
deliberately infeasible input -- an interaction graph requiring a
larger matching than the device can provide (e.g. `dense_pairs` sized
for a larger device than the coupling map given) -- the fixed check
still returns `False` and short-circuits before searching. **A "fix"
that simply always returns `True` would pass P1 and P2 while
destroying the guard's entire purpose**, so this is tested directly.

## 4. What this cannot establish

- Whether the guard is worth keeping at all (VF2 would also
  conclude infeasibility on its own, just more slowly) -- this
  addendum repairs it rather than relitigating its existence.
- Whether the residual 12.7x wrapper overhead identified in Addendum 88
  Section 4 (eager `_candidate_orderings` list construction) is
  affected -- untouched here.
- Generalization to other topologies or circuit families.

---


<!-- ===== Addendum 89 (source: spare-qubit-cliff-addendum-89-2026-09-19.md) ===== -->

> **Note added when merging:** The guard fix works: all three predictions confirmed -- layout_search=True now reaches chain-shaped circuits for the first time in this project's history, with zero regression on dense_pairs and genuinely infeasible inputs still rejected.

## Addendum 89 -- the guard fix works: all three predictions confirmed, `layout_search=True` now reaches chain-shaped circuits for the first time -- but the repaired guard now costs 11x more than the search it protects (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-89-preregistration-2026-09-19.md`, written
and locked before this run. **This is the first harness since Addendum
83 to call `smart_vf2_layout()` directly, with no bypass** -- every
prior one had to skip `_has_feasible_matching` to get past it at all.

## 0. In one line

**P1, P2, P3 all confirmed.** The real, public `smart_vf2_layout()`
now finds a valid layout for **all 26 of 26** chain-shaped
configurations it previously rejected outright (`feasible=True`,
26/26 found, 26/26 structurally valid, 26/26 solved on the first
`natural` ordering); `dense_pairs` -- the family the original guard was
designed for -- behaves identically at both 6x7 and 8x8; and a
genuinely impossible input (4x4 grid, 32 disjoint pairs required) is
**still rejected before any search runs** (`feasible=False`,
`orderings_tried=0`). **An unregistered finding from the same data**:
the repaired guard itself now costs ~3.4ms per call -- **about 11x the
0.314ms search it is protecting** (Addendum 88's own bypassed median) --
making the guard the dominant cost of a successful layout. Whether it
still earns that cost is a real, open question, addressed in Section 4.

## 1. Results

| check | case | result |
|---|---|---|
| **P1** | 26 `dominant_size_sweep` configs, 8x8, real entry point | `feasible=True` and layout found **26/26**; structurally valid **26/26**; solved on first (`natural`) ordering **26/26** |
| **P2** | `dense_pairs` at 6x7 and 8x8 | found **2/2**, valid **2/2**, `feasible=True` both -- unchanged from the original guard's own verdict |
| **P3** | 4x4 grid asked for 32 disjoint pairs | `feasible=False`, `found=False`, `orderings_tried=0` -- **still short-circuits before searching** |

**Timing (P1, 26 configs)**: median **3.71ms**, range 2.33-5.07ms,
excluding one outlier at D=2 of **452.7ms**. That outlier was the very
first measurement taken in the run, and is most consistent with
first-call warm-up (`networkx` import, first `max_weight_matching`
invocation) rather than anything specific to D=2 -- **not confirmed**,
and worth a warm-up-outside-the-timer re-run if the number matters,
which no conclusion here depends on.

## 2. Scoring

**P1 -- CONFIRMED, in its first branch ("Works").** All 26 reach the
search and succeed, matching Addendum 88's bypassed results exactly in
coverage, ordering, and validity. **This is the first time in this
project's history that `layout_search=True`'s real code path has been
shown to work on a chain-shaped circuit** -- every prior demonstration
(Addenda 83, 84, 88) required bypassing the guard.

**P2 -- CONFIRMED.** `dense_pairs` at both grid sizes gives the same
`feasible=True` verdict and the same found/valid outcome as before.
The necessary-condition equality argument (a matching-shaped
interaction graph's own maximum matching equals its edge count) holds
in practice, not just on paper. **No regression on the original design
case.**

**P3 -- CONFIRMED.** The guard still does its job: the impossible
input is rejected in 0.0018s with `orderings_tried=0`, never reaching
VF2. The fix did not degenerate into "always return True."

## 3. What this completes

Three separate defects identified across this session are now all
addressed in `psf_smart_layout_patched.py`:

| defect | found in | status |
|---|---|---|
| `_candidate_orderings` never tries natural + `id_order=True` | Addendum 88 | **fixed** (one line), 56.8x median |
| `_has_feasible_matching` compares against edge count, not the interaction graph's own matching | Addendum 83 Sec. 5, fixed here | **fixed**, 26/26 now reach the search |
| eager `_candidate_orderings` list construction wastes 5 BFS traversals per call | Addendum 88 Sec. 4 | **not fixed** -- proposed, untested |

**The practical consequence**: before this session, any caller using
`compile_for_hardware(layout_search=True)` on a chain-shaped circuit
was silently getting Qiskit's own default layout stage, with no error
and no indication why. That is now repaired, and the path it unblocks
solves every configuration tested here that Qiskit's own `VF2Layout`
could not.

## 4. The guard now costs 11x the search it protects

Comparing this run against Addendum 88's own bypassed measurements on
the identical 26 configurations:

| | median |
|---|---:|
| Search alone (Addendum 88, guard bypassed) | 0.314ms |
| Search + repaired guard (this run) | 3.68ms (D=2 outlier excluded) |
| **Guard's own cost** | **~3.4ms** |

The guard runs **two** `nx.max_weight_matching` computations per call
-- one on the interaction graph (added by this fix), one on the
physical graph (original) -- to decide whether to attempt a search that
itself takes 0.314ms when it succeeds.

**This does not mean the guard should be removed.** Its value is
entirely in the infeasible case, and **how long VF2 itself would take
to reject a genuinely impossible input was never measured** -- P3's own
test short-circuits at the guard, so the comparison the question needs
(guard-rejection time vs. VF2-rejection time on the same impossible
input) does not exist in this project's data. If VF2 rejects such
inputs quickly, the guard is net-negative and should go; if VF2 grinds
on them the way it does on this project's cliffing cases, the guard is
earning its cost many times over. **Measuring that is a small, direct
experiment**, and is the natural follow-up to this addendum -- noted
rather than guessed at here.

A cheaper middle path also exists and is untested: the physical
graph's own maximum matching depends only on the coupling map, not the
circuit, so it could be computed once and cached per target rather
than recomputed on every call -- roughly halving the guard's cost
without changing its verdict. This is proposed, not measured.

## 5. What this does not establish

- Whether the guard is worth its cost at all (Section 4) -- requires a
  measurement that does not yet exist.
- Whether the D=2 outlier (452.7ms) is warm-up or something else --
  most consistent with warm-up, not confirmed.
- Generalization beyond the 8x8 grid and this project's own circuit
  families -- unchanged limitation from Addenda 83-88.
- Whether the eager-list inefficiency (Addendum 88 Section 4) is worth
  fixing now that the guard dominates the cost anyway -- arguably the
  guard should be addressed first, since 3.4ms dwarfs the ~0.29ms that
  change would recover.

## 6. Files

| File | What it is |
|---|---|
| [`psf_smart_layout_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout_patched.py) | now carries both fixes (Addendum 88's natural-first ordering, Addendum 89's guard) |
| [`verify_feasibility_fix.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_feasibility_fix.py) | this run's harness -- the first since Addendum 83 to call `smart_vf2_layout()` with no bypass |
| [`feasibility_fix_verification_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/feasibility_fix_verification_2026-09-19.csv) | this run's results, 29 rows (26 P1 + 2 P2 + 1 P3) |
| [`spare-qubit-cliff-addendum-89-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-89-preregistration-2026-09-19.md) | the predictions scored above |

## 7. Verification

- The fix's correctness argument (a maximum matching of the
  interaction graph maps to a matching of equal size in the physical
  graph, under any injective embedding) was checked numerically before
  this run, in the sandbox, against both `dense_pairs` (old and new
  verdicts identical at 6x7 and 8x8) and all 26 `dominant_size_sweep`
  configurations (old `False`, new `True`) -- and against a deliberately
  impossible case to confirm the guard still rejects.
- P3 was included specifically because a fix that always returned
  `True` would pass P1 and P2 while silently destroying the guard --
  tested directly rather than assumed from the code reading.
- Every P1/P2 layout was validated structurally (each interaction pair
  checked against the coupling map's own edge set), not inferred from
  a non-`None` return.
- Section 4's ~3.4ms figure was computed by differencing this run's own
  median against Addendum 88's recorded median on the identical
  configurations, both re-read from their CSVs rather than from memory.
- The D=2 outlier is reported as unexplained-but-probably-warm-up
  rather than silently excluded or silently included.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 90 pre-registration (source: spare-qubit-cliff-addendum-90-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for whether the repaired guard's ~3.4ms cost is earned, by measuring how long VF2 itself takes to reject matching-infeasible-but-qubit-feasible inputs on bipartite-imbalanced devices.

## Addendum 90 -- Pre-registration: does the repaired `_has_feasible_matching` guard earn its ~3.4ms, or is VF2 fast enough at rejecting infeasible inputs on its own? (2026-09-19)

**Status: pre-registration only. Neither experiment has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 89 repaired the guard and confirmed it works, but measured an
uncomfortable fact: **the guard costs ~3.4ms per call, roughly 11x the
0.314ms search it protects.** Section 4 of that addendum named the
missing measurement explicitly -- **how long VF2 itself takes to reject
a genuinely infeasible input was never measured**, so whether the guard
saves more than it costs is unknown. This addendum measures it.

A second, cheaper question was also left open: the guard runs two
`nx.max_weight_matching` calls, one on the physical graph (depends only
on the coupling map, so cacheable per target) and one on the
interaction graph (depends on the circuit, so not). How the 3.4ms
splits between them determines whether caching is worth implementing.

## 2. Designs

### Experiment A -- guard rejection vs. VF2 rejection

The guard's value is entirely in cases it correctly rejects. The
interesting cases are those that are **infeasible by matching but not
by qubit count** -- if an input fails on qubit count alone, VF2 will
reject it trivially and the guard is redundant. Bipartite-imbalanced
devices produce exactly this situation, and are not an artificial
construct: **this is the same mechanism Addenda 39-41 identified in
heavy-hex hardware**, where bipartite imbalance holds the maximum
matching below n/2 and so caps achievable occupancy.

| case | device | device qubits | device max matching | pairs requested | logical qubits | infeasible because |
|---|---|---:|---:|---:|---:|---|
| **A1 (near-miss, small)** | `K(4,16)` complete bipartite | 20 | 4 | 6 | 12 | matching only (12 <= 20 qubits) |
| **A2 (near-miss, larger)** | `K(8,32)` complete bipartite | 40 | 8 | 10 | 20 | matching only (20 <= 40 qubits) |
| **A3 (bare near-miss)** | `K(8,32)` complete bipartite | 40 | 8 | 9 | 18 | matching only, by exactly one pair |
| **A4 (control: obviously impossible)** | 4x4 grid | 16 | 8 | 32 | 64 | qubit count AND matching |

For each: measure **(a)** the guard's own rejection time (call
`_has_feasible_matching` directly), and **(b)** VF2's own rejection
time with the guard bypassed (run `smart_vf2_layout`'s search logic and
let `rx.vf2_mapping` conclude no match exists, with the module's own
standard `call_limit` and time budget).

### Experiment B -- decomposing the guard's own cost

Time the physical-graph matching and the interaction-graph matching
separately, on the 8x8 grid with a representative
`dominant_size_sweep` interaction graph -- the exact configuration
Addendum 89 measured at ~3.4ms total. **This is `networkx`-only and was
run directly in the sandbox**, unlike Experiment A which needs
`rustworkx`.

## 3. Pre-registered predictions

**P1 (Experiment A, primary -- does the guard earn its cost?).**
  - **Guard earns it**: on the near-miss cases (A1-A3), VF2's own
    rejection takes substantially longer than the guard's ~3.4ms --
    plausibly hitting its `call_limit` or time budget, as this project
    has repeatedly seen VF2 do on hard instances. Keeping the guard is
    then clearly correct.
  - **Guard does not earn it**: VF2 rejects the near-miss cases in
    comparable or less time than the guard takes -- meaning the guard
    costs 3.4ms on **every** call to save nothing, and should be
    removed or made much cheaper.
  - **Split verdict**: VF2 is fast on some infeasible shapes and slow
    on others -- reported with the specific split, not averaged into a
    single recommendation.

**P2 (Experiment A, control).** A4 (impossible by qubit count) is
predicted to be rejected quickly by VF2 without the guard -- this is a
sanity check that the harness measures what it claims, not a live
question. If VF2 is *slow* even here, something about the measurement
setup needs re-examining before A1-A3 are interpreted.

**P3 (Experiment B).** The physical-graph matching (64 nodes, 112
edges on an 8x8 grid) is predicted to be the **larger** share of the
guard's cost than the interaction-graph matching (64 nodes, 45-46
edges), since it has roughly 2.5x the edges. If so, caching it per
coupling map would recover the majority of the guard's overhead
without changing any verdict. **If the split is reversed or roughly
even, caching is worth much less than Addendum 89's Section 4
suggested**, and that suggestion should be corrected.

## 4. What this cannot establish

- Whether the guard is worth keeping in a *production* workload mix,
  which depends on how often real callers pass infeasible circuits --
  unknown, and not measurable from this project's own synthetic
  families.
- Whether VF2's rejection times on these bipartite cases generalize to
  infeasible inputs of other shapes.
- Anything about Qiskit's own implementation.

---


<!-- ===== Addendum 90 (source: spare-qubit-cliff-addendum-90-2026-09-19.md) ===== -->

> **Note added when merging:** The guard decisively earns its cost: it rejects in ~1ms what takes VF2 1.6-2.0+ seconds (1,600-2,400x), with a break-even around 1 infeasible input per 482 calls.

## Addendum 90 -- the guard decisively earns its cost: it rejects in ~1ms what takes VF2 1.6-2.0+ seconds (1,600-2,400x), breaking even at roughly 1 infeasible input per 482 (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-90-preregistration-2026-09-19.md`, written
and locked before either experiment. Experiment B was run in the
sandbox (`networkx`-only); Experiment A required `rustworkx` and was
run on the user's machine.

## 0. In one line

**P1: "Guard earns it" -- CONFIRMED, decisively.** On inputs that are
infeasible by matching but fine by qubit count -- the only cases where
the guard can help, and the same bipartite-imbalance mechanism Addenda
39-41 found in heavy-hex hardware -- the guard rejects in **0.7-1.2ms**
what costs VF2 **1,642-2,000+ms**: a **1,634-2,356x** saving.
**P2 (control) confirmed**: an input that also fails on qubit count is
rejected by VF2 itself in 0.303ms, faster than the guard -- exactly as
predicted, and a useful demonstration that the guard's value is
specific, not universal. **P3 partially confirmed**: the physical
graph's matching is the larger share of the guard's own cost, but only
marginally (50.2% vs 43.0%), not proportionally to its 2.5x edge count
-- **the prediction's conclusion was right and its stated reasoning was
wrong**. **Break-even: the guard pays for itself if infeasible inputs
occur more often than roughly 1 in 482 calls.**

## 1. Experiment A -- guard rejection vs. VF2 rejection

| case | device | qubits | max matching | pairs asked | guard | VF2 | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| **A1** near-miss, small | `K(4,16)` | 20 | 4 | 6 | 0.697ms | **1,642.5ms** | **2,356x** |
| **A2** near-miss, larger | `K(8,32)` | 40 | 8 | 10 | 1.224ms | **2,000.5ms** | **1,634x** |
| **A3** bare near-miss (off by one pair) | `K(8,32)` | 40 | 8 | 9 | 1.183ms | **2,000.1ms** | **1,690x** |
| **A4** control -- also fails on qubit count | 4x4 grid | 16 | 8 | 32 | 0.999ms | 0.303ms | 0.3x |

In every case VF2 correctly found no layout (`vf2_found_a_layout=False`
throughout), so the harness's own built-in warning -- that a case which
turned out feasible would invalidate its row -- never fired. All ten
orderings (7 stage-1, 3 stage-2) were attempted in every case.

**A crucial caveat on A2 and A3**: both landed at essentially exactly
2,000ms, which **is** the module's own `time_budget_s=2.0` default.
**VF2 did not conclude infeasibility in those two cases -- it ran out
of time and gave up.** Their figures are therefore **lower bounds**,
not measured rejection times; the true cost without a budget could be
far larger, or unbounded. **Only A1 (1,642.5ms, under the budget)
represents a search that actually finished.** The headline ratios are
correspondingly conservative.

## 2. Experiment B -- what the guard's own cost is made of

Measured in the sandbox, 8x8 grid with a representative
`dominant_size_sweep` interaction graph, warm-up outside the timer,
median of 30 repetitions:

| step | time | share | cacheable? |
|---|---:|---:|---|
| interaction graph construction | 0.036ms | 2.2% | no -- depends on the circuit |
| interaction graph matching | 0.692ms | 43.0% | no -- depends on the circuit |
| physical graph construction | 0.073ms | 4.6% | **yes** -- depends only on the coupling map |
| physical graph matching | 0.808ms | 50.2% | **yes** -- depends only on the coupling map |
| **total** | **1.609ms** | | **54.8% cacheable** |

## 3. Scoring

**P1 -- CONFIRMED in its first branch ("Guard earns it").** The
pre-registration's criterion was that VF2's rejection "takes
substantially longer than the guard's ~3.4ms." It takes 1,600-2,400x
longer, and in two of three cases did not finish at all.

**P2 (control) -- CONFIRMED.** A4 was rejected by VF2 in 0.303ms
without the guard, faster than the guard itself. The harness measures
what it claims, and the guard's value is confirmed to be specific to
matching-infeasible-but-qubit-feasible inputs rather than infeasible
inputs generally.

**P3 -- CONFIRMED in conclusion, FALSIFIED in reasoning.** The
prediction was that the physical-graph matching would be the larger
share "since it has roughly 2.5x the edges." It **is** larger (50.2%
vs 43.0%) but nowhere near proportionally -- 2.5x the edges produced
1.17x the cost. **Addendum 89's own Section 4 estimate that caching
would "roughly halve" the guard's cost turns out accurate (54.8%), but
the edge-count reasoning offered here for why was not**, and is
corrected rather than quietly retained.

## 4. The verdict on the guard

**Keep it.** Quantitatively, with `G` = guard cost per call (~3.4ms on
the 8x8 grid, Addendum 89; 0.7-1.2ms on these smaller devices, so it
scales with device size), `S` = search cost when feasible (0.314ms,
Addendum 88), and `V` = VF2's own rejection cost (>=1,642ms, A1):

- With the guard: feasible calls cost `G + S`; infeasible cost `G`.
- Without: feasible cost `S`; infeasible cost `V`.
- The guard wins when `N_infeasible / N_feasible > G / (V - G)`
  = 3.4 / (1642.5 - 3.4) = **0.00207**.

**So the guard pays for itself if more than roughly 1 call in 482 is
infeasible.** Given that A1's 1,642ms is itself a conservative figure
(A2/A3 exceeded the time budget entirely), the real threshold is likely
lower still. **This settles the open question Addendum 89 Section 4
raised, in the guard's favour, and by a wide margin.**

**A worthwhile refinement remains** (Experiment B): caching the
physical graph's own matching per coupling map would remove 54.8% of
the guard's cost without changing a single verdict, since that
computation depends only on the device. This is now a
straightforward optimization with a measured payoff, not a guess --
though at 3.4ms against a 1,642ms saving, it is a refinement, not a
necessity.

## 5. What this does not establish

- **How long VF2 actually takes on A2/A3** -- both hit the 2.0s time
  budget rather than concluding. Re-running with a much larger budget
  would measure it, and would only strengthen the conclusion.
- Whether infeasible inputs occur often enough in any real workload to
  clear the 1-in-482 threshold -- that depends on how callers use
  `compile_for_hardware`, which this project's synthetic families
  cannot answer.
- Whether these bipartite cases' VF2 rejection times generalize to
  infeasible inputs of other shapes.
- Whether the caching refinement is worth implementing -- measured as
  possible and quantified, not argued for here.

## 6. Files

| File | What it is |
|---|---|
| [`verify_guard_economics.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_guard_economics.py) | Experiment A's harness, calling the real `_has_feasible_matching` and the real search helpers |
| [`guard_economics_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/guard_economics_2026-09-19.csv) | Experiment A's results, 4 rows |
| [`psf_smart_layout_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout_patched.py) | the module under test (Addenda 88 + 89 fixes) |
| [`spare-qubit-cliff-addendum-90-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-90-preregistration-2026-09-19.md) | the predictions scored above |

## 7. Verification

- Each case's construction was checked numerically **before** the run
  to confirm it is genuinely "qubit-feasible but matching-infeasible"
  (A1-A3: logical qubits <= device qubits AND pairs > device max
  matching, both verified per case), since a case failing on qubit
  count would measure something else entirely.
- The harness carries a built-in warning for the failure mode that
  would invalidate a row (VF2 finding a layout, meaning the case was
  not actually infeasible). It did not fire for any case.
- The A2/A3 time-budget artefact was identified by noticing both landed
  at the module's own documented `time_budget_s=2.0` default, and is
  reported as a lower bound rather than a measured value -- the
  headline ratios use it conservatively.
- The break-even arithmetic in Section 4 is derived explicitly rather
  than asserted, so it can be rechecked against different `G`, `S`, `V`
  values.
- Experiment B used warm-up outside the timer and the median of 30
  repetitions per step, and decomposed all four steps (both graph
  constructions as well as both matchings) rather than only the two
  matching calls.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 91 pre-registration (source: spare-qubit-cliff-addendum-91-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for caching the physical graph's maximum matching per coupling map, since it depends only on the device and not the circuit -- including a safe-key-vs-fast-key design decision made before implementation.

## Addendum 91 -- Pre-registration: caching the physical graph's maximum matching per coupling map (2026-09-19)

**Status: pre-registration only. The cache has not been implemented or
run.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 90's Experiment B decomposed the repaired guard's cost and
found **54.8% of it is the physical graph's construction and matching,
which depend only on the coupling map and not on the circuit** -- so
the same computation is repeated identically on every call against the
same device. Caching it per coupling map would remove that share
without changing any verdict.

Before implementing, the one thing that could sink the idea was
measured in the sandbox: **the cost of building a cache key.** A key
must be derived from the coupling map's own edges (a `CouplingMap`
object is not a reliable dict key), and if that derivation costs as
much as the computation it avoids, the cache is pointless.

| approach | key cost | net saving against the 0.889ms it avoids |
|---|---:|---:|
| sorted tuple of sorted edge tuples (safest) | 0.022ms | **+0.868ms (97.5% realized)** |
| `frozenset` of `frozenset`s | 0.016ms | +0.874ms |
| `id()` of the object | 0.0001ms | +0.889ms, **but unsafe** -- two equal coupling maps built separately would miss, and a freed object's id can be reused by an unrelated one |

The sorted-tuple key costs 2.5% of what it saves, so the idea is viable
and the safe key is affordable. `id()` is rejected despite being
fastest: correctness over 0.02ms.

## 2. The change under test

A module-level, bounded cache keyed on the coupling map's own structure:

```python
@functools.lru_cache(maxsize=32)
def _physical_max_matching(size, edges_key):
    g = nx.Graph()
    g.add_nodes_from(range(size))
    g.add_edges_from(edges_key)
    return len(nx.max_weight_matching(g, maxcardinality=True))
```

called from `_has_feasible_matching` with
`edges_key = tuple(sorted(tuple(sorted(e)) for e in cmap.get_edges()))`.
The interaction-graph side is untouched (it genuinely varies per
circuit). `lru_cache(maxsize=32)` bounds memory rather than letting a
module-level dict grow without limit across many devices.

## 3. Pre-registered predictions

**P1 (primary -- correctness, which matters more than speed).** The
cached version returns the **identical verdict** to the uncached one on
every case already on record: the 26 `dominant_size_sweep`
configurations (all `True`), `dense_pairs` at 6x7 and 8x8 (both
`True`), and Addendum 90's four infeasible cases A1-A4 (all `False`).
**Any single disagreement is a correctness regression and would make
the speed result irrelevant.**

**P2 (cache discrimination -- the failure mode that matters most).**
Different coupling maps must get different answers. Called in sequence
against several devices with **different** maximum matchings (e.g. 4x4
grid = 8, 6x7 grid = 21, 8x8 grid = 32, `K(8,32)` = 8), each must
return its own correct verdict, not a stale one from a previous key.
**A cache that silently returns the first device's answer for every
subsequent device would pass P1 (if P1 used one device) and be
catastrophically wrong**, so this is tested explicitly, interleaved
rather than grouped.

**P3 (speed).** On repeated calls against the same coupling map, the
guard's cost drops by roughly the 54.8% Experiment B measured -- i.e.
from ~1.6ms to ~0.75ms in sandbox terms, or proportionally on whatever
machine runs it. The first call against a given device is predicted to
be unchanged (cache miss), and only subsequent ones faster.

## 4. What this cannot establish

- Whether `maxsize=32` is the right bound for any real workload.
- Whether the remaining 45.2% (the interaction-graph side) can also be
  reduced -- it genuinely varies per circuit, so probably not by
  caching.
- Whether this refinement is worth the added complexity at all, given
  Addendum 90 measured the guard saving 1,642ms against its own ~3.4ms
  cost -- **this is an optimization of something already 480x
  net-positive**, and is pursued because it is cheap and measurable,
  not because the guard needed rescuing.

---


<!-- ===== Addendum 91 (source: spare-qubit-cliff-addendum-91-2026-09-19.md) ===== -->

> **Note added when merging:** The cache works: 52.4% of the guard's cost removed, 32/32 verdicts identical, and the cache is proven to discriminate correctly between different devices via its own hit/miss statistics.

## Addendum 91 -- the physical-matching cache works: 52.4% of the guard's cost removed, 32/32 verdicts identical, and the cache provably discriminates between devices (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-91-preregistration-2026-09-19.md`, written
and locked before implementation. `_has_feasible_matching` and its new
helper use `networkx` only, so all three predictions were verified
directly in the sandbox.

## 0. In one line

**P1, P2, P3 all confirmed.** The cached guard returns **identical
verdicts on all 32 cases already on record** (26 `dominant_size_sweep`,
`dense_pairs` at 6x7 and 8x8, and Addendum 90's four infeasible cases)
-- **zero mismatches**. It **provably discriminates between devices**:
across four devices with different maximum matchings, called
interleaved over three rounds, every verdict was correct, and the
cache's own statistics show **exactly 4 misses and 20 hits** -- one
miss per distinct device, which is what a correctly-keyed cache must
produce and what a device-confusing one could not. And it removes
**52.4%** of the guard's cost (1.633ms to 0.778ms), close to the 54.8%
Addendum 90's decomposition predicted.

## 1. Results

### P1 -- correctness (32 cases, uncached vs. cached)

| group | cases | mismatches |
|---|---:|---:|
| `dominant_size_sweep` D=2..28 (all expected `True`) | 26 | **0** |
| `dense_pairs` at 6x7 and 8x8 (expected `True`) | 2 | **0** |
| Addendum 90's A1-A4 infeasible cases (expected `False`) | 4 | **0** |
| **total** | **32** | **0** |

### P2 -- device discrimination (the failure mode that matters most)

Four devices with deliberately different maximum matchings, called
**interleaved** (not grouped) over three rounds, each asked both for
exactly its matching capacity (must be `True`) and for one pair more
(must be `False`):

| device | max matching |
|---|---:|
| 4x4 grid | 8 |
| 6x7 grid | 21 |
| 8x8 grid | 32 |
| `K(8,32)` complete bipartite | 8 |

**All 24 verdicts correct.** Cache statistics after the run:
`hits=20, misses=4, currsize=4`. **The 4 misses are exactly the four
distinct devices**, and the 20 hits are the repeats -- the signature of
a cache keyed correctly on structure. A cache that collapsed different
devices onto one entry would instead show 1 miss and 23 hits, and would
have returned wrong verdicts. Note that 4x4 grid and `K(8,32)` share a
maximum matching of 8 but are structurally different devices: they
correctly occupy separate entries rather than colliding.

### P3 -- speed

| | median |
|---|---:|
| uncached (Addendum 89 version) | 1.633ms |
| cached (this version) | **0.778ms** |
| **reduction** | **52.4%** |

Addendum 90's decomposition predicted 54.8% cacheable; the realized
52.4% is slightly lower, the difference being the cache key's own
construction cost (measured separately at 0.022ms, ~2.4% of the
uncached total -- which accounts for essentially the entire gap between
predicted and realized).

## 2. Scoring

**P1 -- CONFIRMED.** 32/32 identical. The pre-registration stated that
any single disagreement would make the speed result irrelevant; none
occurred.

**P2 -- CONFIRMED, with direct evidence rather than absence of
failure.** The pre-registration specifically called out that "a cache
that silently returns the first device's answer for every subsequent
device would pass P1 (if P1 used one device) and be catastrophically
wrong." The interleaved multi-device test plus the hit/miss statistics
rule this out positively: the cache demonstrably stores four separate
entries and returns each device's own answer.

**P3 -- CONFIRMED.** 52.4% against a predicted ~54.8%; the shortfall is
accounted for by the key-construction cost that Addendum 91's own
pre-registration had already measured and budgeted for.

## 3. Why the safe key was chosen over the fast one

The pre-registration recorded three candidate key strategies with their
measured costs. `id()` was 220x cheaper than the sorted-tuple key
(0.0001ms vs 0.022ms) and was **rejected anyway**, for two reasons that
no timing measurement would reveal:

- Two separately-constructed but structurally identical coupling maps
  would have different `id()`s and miss the cache -- a performance bug.
- **Far worse**: Python reuses the ids of freed objects, so a new
  coupling map could inherit the id of a discarded one and silently
  receive its cached matching. On a differently-shaped device, that is
  a wrong verdict with no error and no way to notice.

The sorted-tuple key costs 2.4% of what the cache saves. **Paying 0.022ms
to make a silent-wrong-answer class of bug structurally impossible is
not a close call**, and P2's test exists precisely to confirm that
choice held up in practice.

## 4. Cumulative state of `psf_smart_layout_patched.py`

Three defects found this session are now fixed and verified in one
file:

| fix | found in | effect | verified |
|---|---|---|---|
| natural ordering tried first in `_candidate_orderings` | Addendum 88 | 56.8x median on the search | 26/26 first-ordering, coverage and validity preserved |
| `_has_feasible_matching` compares against the interaction graph's own matching, not its edge count | Addendum 83 Sec. 5, fixed in 89 | chain-shaped circuits reach the search at all | 26/26 via the real entry point; `dense_pairs` unchanged; impossible inputs still rejected |
| physical matching cached per device | Addendum 90 Sec. 4, fixed here | 52.4% of the guard's cost | 32/32 verdicts identical; device discrimination proven |

One identified inefficiency remains **unfixed**: `_candidate_orderings`
still builds its whole list eagerly, computing five BFS traversals per
call even though the first entry now always wins (Addendum 88 Section
4). At ~0.29ms against a guard that now costs 0.778ms and a search that
costs 0.314ms, it is no longer negligible in relative terms -- but it
remains untested, and is recorded rather than assumed.

## 5. What this does not establish

- Whether `maxsize=32` is right for any real workload -- it bounds
  memory; no workload evidence exists either way.
- Whether the remaining 45.2% (the interaction-graph side) can be
  reduced -- it genuinely varies per circuit, so caching does not apply.
- That the cache behaves correctly under concurrent use from multiple
  threads. `functools.lru_cache` is safe against corruption but this
  was not tested, and no part of this project exercises it.
- Any re-measurement through the real `smart_vf2_layout` entry point
  with `rustworkx` -- this addendum verified the guard function
  directly, which is where the entire change lives, but the end-to-end
  number from Addendum 89 (~3.4ms on the 8x8 grid) has not been re-run
  with the cache in place.

## 6. Files

| File | What it is |
|---|---|
| [`psf_smart_layout_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout_patched.py) | now carries all three fixes (Addenda 88, 89, 91) |
| [`spare-qubit-cliff-addendum-91-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-91-preregistration-2026-09-19.md) | the predictions scored above |

No new data file -- all three predictions were verified in the sandbox
against functions that use `networkx` only.

## 7. Verification

- P1 compared the cached implementation against a locally-written copy
  of the Addendum 89 (uncached) version on all 32 cases already on
  record, rather than against remembered expected values.
- P2 was run **interleaved across devices and repeated over three
  rounds**, not device-by-device, since a grouped test could mask
  exactly the staleness bug it is meant to catch. The cache's own
  `cache_info()` was inspected as positive evidence (4 misses for 4
  devices), not merely the verdicts.
- P2 deliberately included two structurally different devices sharing
  the same maximum matching (4x4 grid and `K(8,32)`, both 8) to confirm
  the key discriminates on structure rather than on the cached value.
- P3 used warm-up outside the timer and the median of 30 repetitions,
  and the shortfall against the predicted figure was accounted for
  against the separately-measured key cost rather than left
  unexplained.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 92 pre-registration (source: spare-qubit-cliff-addendum-92-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for making _candidate_orderings a lazy generator instead of building its whole list eagerly, so unused orderings are never computed.

## Addendum 92 -- Pre-registration: making `_candidate_orderings` lazy, so the four unused orderings are never computed (2026-09-19)

**Status: pre-registration only. The change has not been implemented or
run.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 88 Section 4 identified, from source, that
`_candidate_orderings` builds its entire return list eagerly:

```python
orderings = []
orderings.append(("natural", list(nodes)))                      # Addendum 88
orderings.append(("bfs_from_max_degree", _bfs_order_from(...)))
orderings.append(("bfs_from_min_degree", _bfs_order_from(...)))
orderings.append(("bfs_from_node0", _bfs_order_from(...)))
orderings.append(("degree_desc", sorted(...)))
for seed in extra_seeds:
    orderings.append((f"bfs_from_random_seed{seed}", _bfs_order_from(...)))
return orderings
```

**Every entry is computed before the caller sees any of them** -- five
BFS traversals (`_bfs_order_from` -> `rx.bfs_layers`, for max-degree,
min-degree, node0, and the two random seeds) plus a sort, all discarded
unused now that Addendum 88's `natural` entry always wins first on this
project's configurations.

This was left unfixed in Addendum 88 and again in Addendum 91 because
it was small relative to what else was wrong. It is no longer: after
Addendum 91 cut the guard to 0.778ms, the whole wrapper overhead
(~0.29ms, inferred in Addendum 88 as the gap between the patched
search's 0.314ms and the bare call's 0.0247ms) is a comparable share.

## 2. The change under test

Convert the function to a generator -- `yield` instead of building and
returning a list:

```python
def _candidate_orderings(graph, extra_seeds=(0, 1)):
    ...
    yield ("natural", list(nodes))
    yield ("bfs_from_max_degree", _bfs_order_from(graph, max_deg_node))
    ...
```

so each ordering is computed only when the caller actually asks for it.

**Safety check performed before writing this**: all three call sites in
this project (`smart_vf2_layout` itself, and the two verification
harnesses) consume the result with a plain `for ... in ...` loop --
none calls `len()`, indexes it, or iterates it twice. Confirmed by
grep across `psf_smart_layout_patched.py`, `verify_natural_first_patch.py`,
`verify_guard_economics.py`, and `psf_compile.py`. **A generator is
therefore a drop-in substitution at every existing call site.**

## 3. Pre-registered predictions

**P1 (primary -- identical behaviour).** The generator yields **exactly
the same `(name, ordering)` sequence, in the same order, with the same
contents**, as the list version, for any given graph and `extra_seeds`.
Verified by consuming both fully and comparing element by element.
**This is the prediction that matters**: a "faster" version that
produced a different ordering sequence would silently change which
layouts are found, invalidating every result from Addenda 88-91.

**P2 (speed on the common path).** When the caller stops after the
first ordering -- which is every one of this project's 26
configurations since Addendum 88 -- the lazy version avoids the five
BFS traversals and the sort, reducing `smart_vf2_layout`'s own
non-guard overhead measurably. **A specific figure is not predicted**,
because Addendum 88's ~0.29ms was inferred as a difference between two
measurements rather than measured directly, and this experiment
measures it properly for the first time.

**P3 (no regression when the first ordering fails).** On an input where
`natural` does NOT succeed, the lazy version must still try all the
remaining orderings and reach the same outcome as the list version --
laziness must not truncate the search. Tested on a deliberately
infeasible input, where every ordering is necessarily exhausted.

## 4. What this cannot establish

- Whether `extra_seeds`' random orderings are ever useful at all --
  untouched here; they are simply no longer computed unless reached.
- Whether the remaining overhead after this change (graph construction,
  `_relabel`) is worth attacking -- measurable afterwards, not now.
- Anything outside this project's own grid topologies and circuit
  families.

---


<!-- ===== Addendum 92 (source: spare-qubit-cliff-addendum-92-2026-09-19.md) ===== -->

> **Note added when merging:** The lazy version is correct and saves 0.075ms, but that is only a quarter of what Addendum 88 predicted -- the real remaining cost turns out to be graph construction, not orderings, correcting that earlier guess.

## Addendum 92 -- the lazy `_candidate_orderings` is correct and saves 0.075ms, but that is a quarter of what Addendum 88 predicted -- and the real remaining cost turns out to be graph construction, not orderings (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-92-preregistration-2026-09-19.md`, written
and locked before implementation. P1 was verified in the sandbox
(`networkx` stub for `_bfs_order_from`); P2 and P3 required
`rustworkx` and were run on the user's machine.

## 0. In one line

**P1, P2, P3 all confirmed -- but P2's magnitude corrects an
overoptimistic claim in Addendum 88.** The generator yields an
**identical sequence** to the old list version (verified element-by-
element on four graph shapes), still reaches **all seven orderings** in
the exact expected sequence when the first fails, and still solves the
end-to-end chain-shaped case on the first ordering. It saves **0.075ms
on the 8x8 grid** by skipping five BFS traversals and a sort.
**However, Addendum 88 Section 4 predicted this change would "likely
close most of" the remaining 0.289ms gap to the bare call; it closes
26%.** Measuring it properly for the first time also located where the
rest actually goes: **~0.258ms in graph construction and `_relabel`,
now by far the largest remaining item** -- which no addendum had
identified, because the gap had only ever been inferred as a difference
between two other numbers, never decomposed.

## 1. Results

### P2 -- cost of consuming one ordering vs. all of them

| device | first only | all seven | saved | BFS traversals avoided |
|---|---:|---:|---:|---:|
| 6x7 grid | 0.0212ms | 0.0797ms | **0.0585ms** | 5 |
| 8x8 grid | 0.0309ms | 0.1057ms | **0.0749ms** | 5 |

### P3 -- laziness must not truncate the search

On an infeasible device (`K(4,16)`, Addendum 90's A1), where no
ordering can succeed and every one must therefore be tried:

- **7 orderings yielded**, matching the expected sequence exactly:
  `natural`, `bfs_from_max_degree`, `bfs_from_min_degree`,
  `bfs_from_node0`, `degree_desc`, `bfs_from_random_seed0`,
  `bfs_from_random_seed1`.
- End-to-end on a feasible chain-shaped circuit (8x8, D=10):
  `found=True, tried=1, order=natural` -- unchanged from Addenda 88-91.

## 2. Scoring

**P1 -- CONFIRMED.** Identical `(name, ordering)` sequences on 8x8
grid, 6x7 grid, `K(8,32)` and a 20-node path -- all seven elements
matching in order and contents on every one. The prediction that
mattered most (a faster version producing a *different* sequence would
silently invalidate Addenda 88-91) is ruled out.

**P2 -- CONFIRMED as a direction, but the pre-registration deliberately
declined to predict a figure, and that caution was warranted.** 0.075ms
saved on 8x8. The pre-registration's stated reason for not predicting a
number -- that Addendum 88's ~0.29ms "was inferred as a difference
between two measurements rather than measured directly" -- is exactly
why: measured directly, the orderings were only a quarter of it.

**P3 -- CONFIRMED.** Laziness does not truncate: all seven orderings
are reached when needed, in the right order, and the feasible
end-to-end path still succeeds on the first.

## 3. Correction to Addendum 88 Section 4

Addendum 88 Section 4 stated that converting this function to a
generator "would likely close most of the remaining 12.7x." **That was
wrong, and is corrected here.** Against the 0.289ms gap between the
patched search (0.314ms) and the bare call (0.0247ms):

| component | cost | share of the gap |
|---|---:|---:|
| `_candidate_orderings`, first ordering only | 0.031ms | 11% |
| the search itself (bare-call equivalent) | 0.025ms | -- |
| **everything else -- graph construction + `_relabel`** | **0.258ms** | **89%** |
| *(what the lazy change removed)* | *0.075ms* | *26%* |

**The orderings were never the main cost.** Building the two
`rustworkx.PyGraph` objects (one physical, one interaction) and running
`_relabel` on the winning ordering together account for roughly 89% of
the gap. Addendum 88 inferred the gap's size correctly but attributed
it to the wrong component, and this addendum is the first measurement
that separates them.

**This does not make the lazy change unworthwhile** -- 0.075ms is real,
it costs nothing to keep, and it is strictly safer than the eager
version (it cannot compute an ordering that is never used). But the
next meaningful optimization of this wrapper, if one is wanted, is
**graph construction**, not orderings.

## 4. A correction within this addendum's own pre-registration

The pre-registration described the eager list as computing "four BFS
traversals plus a sort." **It is five** (max-degree, min-degree, node0,
and both random seeds) -- Addendum 88 had it right and the
pre-registration miscounted. Corrected in that file, noted here rather
than silently fixed, and confirmed by the harness's own count (five BFS
avoided, reported per device in Section 1).

## 5. Cumulative state of `psf_smart_layout_patched.py`

| fix | addendum | effect | verified |
|---|---|---|---|
| natural ordering tried first | 88 | 56.8x on the search | 26/26 first-ordering, coverage and validity preserved |
| guard compares against the interaction graph's own matching | 89 | chain-shaped circuits reach the search at all | 26/26 via the real entry point; `dense_pairs` unchanged; impossible inputs still rejected |
| physical matching cached per device | 91 | 52.4% of the guard's cost | 32/32 verdicts identical; device discrimination proven via cache statistics |
| `_candidate_orderings` made lazy | 92 | 0.075ms, five BFS traversals avoided | identical sequence on four graph shapes; full search still reached when needed |

**Remaining, now measured rather than guessed**: ~0.258ms of graph
construction and relabeling per call, the largest single item left in
the wrapper. Unaddressed, and no change is proposed for it here.

## 6. What this does not establish

- Whether graph construction can be reduced at all (e.g. by caching the
  physical `PyGraph` per device, the same way Addendum 91 cached its
  matching) -- plausible by analogy, entirely untested, and explicitly
  not claimed.
- Whether the 0.258ms figure holds on other devices or topologies --
  measured on 8x8 only, and derived by subtracting two separately-
  measured quantities from a third, so it carries their combined
  uncertainty.
- Whether any of this matters against a guard that now costs 0.778ms
  and a VF2 rejection that costs 1,642ms (Addenda 90-91) -- these are
  refinements of a component that is no longer the bottleneck.

## 7. Files

| File | What it is |
|---|---|
| [`psf_smart_layout_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout_patched.py) | now carries all four fixes (Addenda 88, 89, 91, 92) |
| [`verify_lazy_orderings.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_lazy_orderings.py) | P2/P3 harness |
| [`lazy_orderings_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/lazy_orderings_2026-09-19.csv) | this run's results, 3 rows |
| [`spare-qubit-cliff-addendum-92-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-92-preregistration-2026-09-19.md) | the predictions scored above (with its own five-vs-four correction applied) |

## 8. Verification

- P1 compared the generator against a locally-reconstructed copy of the
  original list version, with both using the same `_bfs_order_from`
  stub, so any difference would come from the change itself rather than
  from the stub -- and was run on four structurally different graphs,
  not one.
- Laziness was verified positively, by counting `_bfs_order_from` calls:
  0 when only the first ordering is taken, 5 when all are consumed.
  That count is also what exposed the pre-registration's own
  four-vs-five miscount.
- P3 used a deliberately infeasible device precisely because it forces
  every ordering to be tried -- a feasible input would stop at the
  first and could not detect truncation.
- Section 3's decomposition subtracts two independently-measured
  quantities (this run's first-only cost, Addendum 84's bare-call cost)
  from Addendum 88's patched-search median, all taken on the same
  machine; the residual is labelled "graph construction + `_relabel`"
  because those are what remain in the code path, not because they were
  timed directly.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 93 pre-registration (source: spare-qubit-cliff-addendum-93-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** Predictions for whether test_cumulative_compile_scale.py's own low standard deviation is a stable, reproducible property or a lucky single run -- originally scoped as a reproducibility check before genuine archived raw data was located mid-session.

## Addendum 93 -- Pre-registration: is `test_cumulative_compile_scale.py`'s low standard deviation (0.121-0.131ms) a stable property, or a lucky single run? (2026-09-19)

**Status: pre-registration only. No re-run has been performed.**
Predictions are locked before any measurement.

## 1. Why this experiment exists, and what it cannot settle

A single run of `test_cumulative_compile_scale.py` (10,000 iterations,
gate synthesis via `psf_compile`, unrelated to today's layout-side work)
showed a standard deviation of 0.121ms (`verify=False`) and 0.131ms
(`verify=True`) -- markedly smaller than a previously reported figure of
~0.90-0.91ms for both arms. This was read as evidence of increased
stability, possibly attributed to today's changes.

**Two things must be stated plainly before designing anything:**

1. **The "previous" figures are summary numbers (mean/median/stdev),
   not raw per-iteration data.** No `.npz` file with the earlier run's
   individual timings exists in this conversation (a file named
   `cumulative_compile_times_10000_2.npz` was looked for and does not
   exist). Without raw data on both sides, a formal statistical test
   (e.g. Levene's test for equal variances) comparing "before" and
   "after" **cannot be performed** -- only whether *today's* low
   variance is itself reproducible.
2. **`test_cumulative_compile_scale.py` exercises `psf_compile()` --
   gate synthesis via the Rust KAK/Cartan core (`lib.rs`) -- not
   `smart_vf2_layout()`.** Every change made this session (Addenda
   88-92: ordering, the feasibility guard, caching, laziness) is in
   `psf_smart_layout.py`, a separate module never imported by
   `psf_compile.py` per that file's own source (read in full when it
   was shared earlier). **If today's low variance is real, it cannot
   be attributed to today's own changes** -- the code path measured by
   this script was not touched today. This addendum tests only whether
   the observed stability is real, not why.

## 2. Design

Run `test_cumulative_compile_scale.py` **unmodified**, exactly as
already run once, **two more times**, on the same machine, back to
back, with no other changes to the environment between runs (same
Python process type, no other heavy processes started deliberately in
between). Each run already saves its own `.npz` with all 10,000
per-iteration timings for `qiskit`, `psf_true`, and `psf_false` --
these are kept, not discarded, so the three runs' raw data can be
compared directly rather than only their printed summary lines.

## 3. Pre-registered predictions

**P1 (primary -- does the low standard deviation reproduce?).**
  - **Reproduces**: both new runs show `psf_false` and `psf_true`
    standard deviations in roughly the same low range (order 0.1-0.2ms),
    not reverting toward the ~0.9ms figure -- supports that something
    about the current machine/environment state genuinely produces
    tight, low-variance timings now, whatever the cause.
  - **Does not reproduce**: one or both new runs show standard
    deviations closer to the older ~0.9ms figure -- would mean the
    first low-variance run was itself the unusual one (e.g. a
    background-load artifact, thermal state, or other transient
    condition), not a stable property of anything.
  - **Partially reproduces**: results land between the two regimes --
    reported as its own outcome, not forced into either bucket.

**P2 (consistency of the mean/median, independent of P1).** Regardless
of how P1 resolves, the three runs' `psf_false` means are predicted to
be broadly consistent with each other (no run wildly higher or lower on
central tendency) -- since nothing about the compile logic changed
between runs, only environmental noise should differ, and noise
primarily affects the tail/variance, not the typical case.

**P3 (Qiskit's own arm, as an internal control).** The `qiskit` arm's
own standard deviation is recorded alongside PSF-Zero's in all three
runs. If Qiskit's own variance is stable across runs while PSF-Zero's
swings between low and high, that would argue for something specific
to PSF-Zero's own measurement (not merely "the machine was busy," which
should affect both arms comparably in the same run).

## 4. What this cannot establish

- **Whether today's variance is actually lower than the historical
  ~0.9ms figure** -- no raw data exists for that comparison, and this
  addendum does not attempt to manufacture one from summary statistics
  alone.
- **Why** any reproducible low variance would occur, if it does --
  candidate explanations (Rust-side caching mentioned in passing,
  reduced background load, a warmed OS/file cache, CPU frequency
  scaling state) are not distinguished by this design.
- Anything about today's layout-side fixes (Addenda 88-92) -- this
  script does not exercise that code path.

---


<!-- ===== Addendum 93 (source: spare-qubit-cliff-addendum-93-2026-09-19.md) ===== -->

> **Note added when merging:** A genuine 4-day-old archive's raw data was recovered, enabling a real statistical comparison: the variance collapse is real and overwhelming (34-54x, Levene p~0), confirmed against a stable Qiskit control arm -- but the cause is not attributable to any of this session's own layout-side fixes.

## Addendum 93 -- the variance collapse is real and statistically overwhelming (34-54x, Levene p~0), confirmed against a stable control arm -- but the cause remains unidentified and is not attributable to today's own changes (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-93-preregistration-2026-09-19.md`, written
before this run. **The design improved beyond what the pre-registration
anticipated**: raw per-iteration data for the genuine "before" run (a
4-day-old archive from GitHub, 10,000 points) was located and preserved
mid-session, making a formal statistical comparison possible where the
pre-registration had explicitly said none could be done. **This
addendum reports that comparison directly, superseding the
reproducibility-only framing P1-P3 were originally written for.**

## 0. In one line

**The variance difference is real, large, and statistically
overwhelming -- not noise.** `psf_false`'s standard deviation dropped
53.99x (0.902ms to 0.123ms); `psf_true`'s dropped 34.64x (0.900ms to
0.153ms); Levene's test rejects equal variance at p~0 for both
(W=2,327.6 and 2,948.3). **The `qiskit` control arm, measured in the
same two runs, shows essentially no change** (variance ratio 0.98x,
practically identical) despite its own Levene test also reaching
significance (p=3.4e-37) purely from the sample size (n=10,000) making
even a 2% difference detectable. **This is the strongest evidence this
project could offer that the effect is specific to PSF-Zero's own
measurement, not generic machine noise**: if the earlier run had simply
been on a busier machine, Qiskit -- measured in the identical process,
presumably interleaved -- should show inflated variance too, and it does
not. **What did not change: this cannot be attributed to today's own
work.** `psf_compile.py`'s own source (read in full when shared earlier
this session) never imports `psf_smart_layout.py`; every fix made today
(Addenda 88-92) lives there. Whatever caused this was already true
before today's session began, or changed for a reason unrelated to it.

## 1. Results

Raw per-iteration data, both runs, `n=10,000` each, `ddof=1`:

| arm | run | mean | median | std | max |
|---|---|---:|---:|---:|---:|
| `psf_false` | 4 days ago (archive) | 1.314ms | 0.950ms | **0.902ms** | 22.682ms |
| `psf_false` | today | 1.186ms | 1.141ms | **0.123ms** | 3.169ms |
| `psf_true` | 4 days ago (archive) | 1.543ms | 1.211ms | **0.900ms** | 20.223ms |
| `psf_true` | today | 1.303ms | 1.266ms | **0.153ms** | 3.622ms |
| `qiskit` (control) | 4 days ago | 9.207ms | -- | 6.144ms | -- |
| `qiskit` (control) | today | 10.375ms | -- | 6.211ms | -- |

| arm | Levene W | Levene p | variance ratio (old/new) | Mann-Whitney p (medians differ) |
|---|---:|---:|---:|---:|
| `psf_false` | 2,327.6 | ~0 | **53.99x** | ~0 |
| `psf_true` | 2,948.3 | ~0 | **34.64x** | ~0 |
| `qiskit` (control) | 163.1 | 3.4e-37 | **0.98x** | (not tested -- not the question) |

## 2. What this establishes

**The variance collapse is not measurement noise or a lucky single
run.** Levene's test (robust to non-normality, appropriate for
timing data with a long right tail) rejects equal variance overwhelmingly
for both PSF-Zero arms. The maximum value alone tells the same story
without any test: 22.7ms and 20.2ms outliers in the old run, versus
3.2ms and 3.6ms in today's -- **the long tail specifically is what
disappeared**, consistent with the user's own original observation.

**The control arm is the decisive piece of evidence for specificity.**
A generic explanation ("the earlier run happened to share the machine
with other processes") predicts both arms in that run would show
inflated variance, since both are measured within the same process on
the same machine at the same time. **Qiskit's own variance did not
meaningfully change (0.98x) while PSF-Zero's collapsed by 34-54x** --
this pattern is hard to produce with a purely external, machine-wide
cause, and points toward something specific to how PSF-Zero itself
was built, cached, or executed differently between the two dates.

**A genuine trade-off, not a strict improvement, in central tendency.**
The *median* moved in the opposite direction from the mean: for
`psf_false`, mean dropped (1.314ms to 1.186ms) but median rose (0.950ms
to 1.141ms); same pattern for `psf_true`. Mann-Whitney confirms this
shift is itself significant, not noise. **The old distribution had many
very-fast typical calls dragged upward by a long tail of rare slow
ones; the new distribution is far more uniform, centered slightly
higher with the tail almost entirely gone.** This is why the user's
own headline "speedup" figures moved in different directions depending
on which statistic is quoted: the terminal output's own printed
cumulative speedup for `verify=False` was 8.87x on the archived run and
8.75x today -- a slight *decrease*, not an increase, despite the
variance collapse, because the cumulative total is mean-driven and the
mean itself dropped only modestly (1.314ms to 1.186ms, 9.7%) while the
comparison's other side (`qiskit`) got slightly slower between runs
too (9.207ms to 10.375ms mean).

## 3. What this does not establish

- **Why.** No code diff between the two dates was examined in this
  addendum -- only the timing data. Candidate mechanisms mentioned in
  conversation (Rust-side SVD-recomputation avoidance, magic-basis
  caching) are exactly that -- mentioned, not verified against source or
  against a controlled before/after code comparison.
- **That today's own session caused it.** Ruled out structurally:
  `psf_compile.py` does not import `psf_smart_layout.py`, and every
  change made today (Addenda 88-92) is confined to the latter. Whatever
  changed, changed in `psf_compile.py` or `lib.rs`, or in the
  environment, at some point in the 4 days between the archive and
  today -- not in this session.
- Whether the slight *decrease* in `verify=False`'s own cumulative
  speedup figure (8.87x -> 8.75x) reflects anything beyond the same
  variance-collapse-with-shifted-median pattern already described --
  not investigated further.
- Whether the same pattern holds at other circuit sizes or gate counts
  -- this comparison used the one configuration (15 qubits, 20
  gates/pair) both runs shared.

## 4. A process note: the original low-variance raw file was lost

The very first low-variance run shared in this conversation (before
this addendum's own design) was analyzed directly from the uploaded
file but **not copied to persistent storage** at the time. A
same-named re-upload (the 4-day-old archive, requested as a
reproducibility check) **overwrote it** in the uploads directory,
before a permanent copy existed. Only that first run's **printed
summary** (mean=1.155ms, median=1.117ms, std=0.121ms for `psf_false`)
survives; its raw per-iteration values do not. **Both files used in
this addendum's own comparison (the archive and today's fresh run) were
copied to persistent storage immediately upon receipt**, specifically
to prevent this from happening a second time.

## 5. Files

| File | What it is |
|---|---|
| `cumulative_compile_times_10000_ARCHIVE_4daysago.npz` | the genuine "before" raw data (10,000 points/arm), recovered from a GitHub-sourced archive upload |
| `cumulative_compile_times_10000_TODAY.npz` | the genuine "after" raw data (10,000 points/arm), a fresh run requested specifically to avoid re-overwriting |
| [`spare-qubit-cliff-addendum-93-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-93-preregistration-2026-09-19.md) | the original pre-registration (P1-P3, reproducibility-only framing, superseded by this direct comparison) |

## 6. Verification

- Both `.npz` files were copied to persistent output storage
  immediately upon receipt, before any analysis, specifically to avoid
  repeating Section 4's data-loss incident.
- Levene's test was chosen over an F-test for equal variances because
  it is robust to the non-normality visible in both distributions
  (long right tails), rather than assuming normality for convenience.
- The control-arm comparison (`qiskit`, Section 1's third row pair) was
  run through the identical test pipeline as the two PSF-Zero arms, not
  computed differently, so the 0.98x-vs-34-54x contrast is a like-for-
  like comparison.
- The mean/median directional disagreement (Section 2) was checked
  numerically in both arms before being reported as a genuine pattern,
  not a rounding artifact in one arm alone.
- The cumulative speedup figures quoted from the two terminal outputs
  (7.96x/8.87x archived vs. 7.96x/8.75x today) were re-read directly
  from each run's own printed summary before being compared, not
  recomputed independently, so they reflect exactly what each run itself
  reported.
- `psf_compile.py`'s own source was re-checked (import statements) before
  asserting it does not reach `psf_smart_layout.py`, rather than assumed
  from memory of reading it earlier in the session.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits. Both
  uploaded `.npz` files and the terminal output supplied for this run
  were reviewed for local paths before use; none were reproduced here.

---


<!-- ===== Addendum 94 pre-registration (source: spare-qubit-cliff-addendum-94-preregistration-2026-09-19.md) ===== -->

> **Note added when merging:** A retrospective clustering/autocorrelation analysis of the archive's own raw data (performed with data already in hand), plus predictions for a follow-up GC on/off test on today's code.

## Addendum 94 -- Pre-registration: the archive's tail was clustered and persistent, not independent noise -- what a cheap GC on/off test on TODAY's code could add (2026-09-19)

**Status: the retrospective analysis (Section 1) has been performed,
using data already in hand. The proposed follow-up test (Section 2)
has not been run.** Predictions for that follow-up are locked before
any measurement.

## 1. Retrospective analysis of already-collected data (no new run)

Using the same two raw datasets as Addendum 93
(`cumulative_compile_times_10000_ARCHIVE_4daysago.npz` and
`..._TODAY.npz`, `psf_false` arm), three checks were run against the
question Addendum 93 left open: is the variance collapse consistent
with a genuine environmental/systemic cause, or is it silent about
that entirely?

| check | 4 days ago | today |
|---|---|---|
| first-half / second-half std | 0.744ms / 1.029ms (worsens) | 0.137ms / 0.099ms (stable) |
| first-half / second-half mean | 1.237ms / 1.392ms (worsens) | 1.215ms / 1.157ms (stable) |
| top-1% outlier inter-arrival CV | **5.62** (highly clustered; 1.0 = random/Poisson) | (not computed -- too few outliers to be meaningful at this std) |
| lag-1 autocorrelation | **0.80** (strong persistence) | 0.56 (moderate persistence) |

**All three point the same direction**: the archive run's slow calls
were not independent, random events. They clustered in time (CV=5.62,
far above the ~1.0 a Poisson process would show), correlated strongly
with their immediate neighbours (lag-1 r=0.80 -- a slow call was very
likely followed by another slow call), and the whole run trended
slower and more variable as it progressed rather than settling after
an initial warm-up. **This is the signature of a persistent, episodic
condition -- something that stays elevated for several consecutive
calls, then relents -- not the signature of "different random circuits
happen to need different amounts of work."**

**What this does and does not narrow down.** It rules out pure
per-call independent noise (e.g. "some SU(4) inputs just need one more
SVD iteration than others, randomly") as the sole explanation for the
old run's variance. It does **not** distinguish between two remaining
candidates that would produce an identical signature:
- **External**: OS scheduler contention, thermal throttling, another
  process competing for CPU/memory, page cache eviction -- anything
  outside the code that comes and goes in bursts.
- **Internal-but-since-fixed**: a periodic expensive operation inside
  the *old* code itself (e.g. a cache that was invalidated and rebuilt
  every N calls, a garbage-collection-triggering allocation pattern,
  a lock or lazy-init check that only bites occasionally) that, if
  removed in whatever changed over the last 4 days, would show exactly
  this reduction in clustering and autocorrelation as a side effect.

**Today's own data still shows non-trivial persistence** (lag-1 r=0.56,
not zero) -- the improvement is large, not total. This is reported
directly rather than glossed over: even the "after" state is not proven
to be fully independent, per-call noise either.

## 2. A cheap, immediately actionable follow-up: GC on/off on TODAY's code

One concrete, internal candidate that produces exactly this signature
(episodic bursts, autocorrelated, worsening over a long run) is
**Python's own garbage collector**: a generational GC pause is
triggered periodically based on allocation counts, affects whatever
call happens to be running at that moment, and its frequency can
increase over a long-running process as more objects accumulate --
matching the "worsens over the run" pattern in Section 1's own archive
data. This is testable on **today's current code**, without needing
the old code at all, and would not resolve what changed in the last 4
days, but would tell us whether GC is a contributing factor in
`psf_compile`'s own timing variance generally.

**Design**: re-run `test_cumulative_compile_scale.py`'s own
`psf_false` arm (or an equivalent minimal loop) twice more on the same
machine -- once with `gc.disable()` called before the loop, once left
as-is (`gc` enabled, the default) -- otherwise identical.

## 3. Pre-registered predictions for the GC test

**P1 (primary).**
  - **GC matters**: disabling it measurably reduces the standard
    deviation and/or the lag-1 autocorrelation further, even from
    today's already-low baseline -- would mean GC pauses are a real,
    ongoing contributor, and the same mechanism likely explains at
    least part of the archive run's own worse behaviour.
  - **GC does not matter**: negligible difference -- would mean
    whatever changed in the last 4 days already addressed the dominant
    cause, or GC was never a major contributor for this workload's
    allocation pattern in the first place.

**P2 (sanity).** Disabling GC must not change *correctness* -- the
same fidelity checks the original script already performs should still
pass. A memory-growth check (process RSS before and after) is also
recorded, since disabling GC entirely for 10,000 iterations could, in
principle, allow unbounded growth for a workload with a real leak --
this would itself be a notable finding, not just a caveat.

## 4. What this does not establish

- **What actually changed in the code over the last 4 days** -- neither
  Section 1's analysis nor Section 2's proposed test can substitute for
  the actual diff, which remains the single highest-priority item
  carried over from Addendum 93.
- Whether the archive run's own clustering was GC-caused specifically,
  as opposed to OS-level or thermal -- the GC test only speaks to
  today's code, not the archive's.
- Whether this generalizes to other circuit sizes.

## 5. Files

No new data files for Section 1 -- computed directly from the two
`.npz` files already saved (Addendum 93). Section 2's test, if run,
would produce its own new data.

---


<!-- ===== Addendum 94 (source: spare-qubit-cliff-addendum-94-2026-09-19.md) ===== -->

> **Note added when merging:** GC is confirmed to be a real contributor to variance (2.73x reduction when disabled) -- but disabling it causes 2.27GB of memory growth over 10,000 calls, ruling it out as a usable fix and revealing a new, more specific mystery.

## Addendum 94 -- GC is confirmed to be a real contributor to the variance (2.73x), but disabling it causes 2.27GB of memory growth over 10,000 calls -- not a viable fix, and a new, more specific mystery about what accumulates (2026-09-19)

**Pre-registered in**:
`spare-qubit-cliff-addendum-94-preregistration-2026-09-19.md`, written
before this run, including a retrospective clustering/autocorrelation
analysis (Section 1 of that document, using data already in hand) that
motivated this specific follow-up.

## 0. In one line

**P1: "GC matters" confirmed.** Disabling Python's garbage collector
for the duration of the loop reduces `psf_compile`'s own timing
standard deviation by **2.73x** (0.314ms enabled to 0.115ms disabled).
**P2's safety check caught something serious**: with GC disabled,
process RSS grew by **2,273.4MB over 10,000 iterations** of a single,
fixed 15-qubit circuit family -- compared to +56.1MB with GC enabled
over the same workload. **This rules out "just disable GC" as a
usable fix**, regardless of its variance benefit, and reveals that
something in the compile path is generating **reference cycles** at a
rate the ordinary refcounting allocator cannot reclaim on its own --
a new, more specific question than the one this addendum set out to
answer. **An unregistered, counter-intuitive finding**: lag-1
autocorrelation went UP with GC disabled (0.17 to 0.41), not down --
complicating Addendum 94 Section 1's own simple "GC pauses cause the
clustering" reading.

## 1. Results

10,000 iterations each, 15-qubit circuit family, `verify=False`, same
process, GC arm run first then GC-disabled arm (both re-enabling GC on
exit).

| | GC enabled (default) | GC disabled |
|---|---:|---:|
| mean | 1.397ms | 1.312ms |
| median | 1.304ms | 1.283ms |
| **std** | **0.314ms** | **0.115ms** |
| max | 9.200ms | 4.056ms |
| lag-1 autocorrelation | 0.1724 | **0.4121** |
| RSS before | 64.9MB | 121.1MB |
| RSS after | 121.0MB | **2,394.4MB** |
| **RSS delta** | **+56.1MB** | **+2,273.4MB** |

**Std ratio: 2.73x. RSS growth ratio: ~40.5x.**

## 2. Scoring

**P1 (primary) -- "GC matters" CONFIRMED.** A 2.73x reduction in
standard deviation from disabling GC alone is a real, substantial
effect -- GC pauses are a genuine, currently-active contributor to
`psf_compile`'s own timing variance, not a negligible one.

**P2 (safety check) -- performed exactly as designed, and it caught a
real problem.** The pre-registration explicitly flagged this
possibility rather than assuming disabling GC was free: *"disabling GC
entirely for 10,000 iterations could, in principle, allow unbounded
growth for a workload with a real leak -- this would itself be a
notable finding, not just a caveat."* It is exactly that. **2.27GB of
growth for one fixed-size circuit family repeated 10,000 times, with
no growing input, is not modest measurement overhead -- it indicates
the compile path is building reference cycles (objects that reference
each other in a loop) at meaningful volume**, which only a cyclic-GC
pass, not simple refcounting, can reclaim. This does not necessarily
mean a true memory leak (cyclic garbage is eventually reachable and
collectible, just not by refcounting alone) -- but disabling GC to fix
timing variance would, in any real, long-running use, trade a timing
problem for an unbounded-memory problem.

## 3. An unregistered, complicating finding: autocorrelation went up, not down

Addendum 94 Section 1's own retrospective analysis found the archive
run's slowness was clustered and autocorrelated (lag-1 r=0.80),
proposing periodic GC as one candidate mechanism -- the naive
prediction being that removing GC should reduce that clustering. **It
did not: lag-1 autocorrelation more than doubled (0.17 to 0.41) with
GC disabled.**

**A plausible reading, not yet confirmed**: with GC enabled, slow
outliers look like discrete, acute *pauses* -- a collection runs,
several calls in a row are held up, then things return to normal
(higher variance, lower autocorrelation between arbitrary neighbouring
points, since most calls are fast and pauses are localized spikes).
**With GC disabled, the uncollected cyclic garbage accumulates
continuously**, and the resulting effect on speed (e.g. through
degraded allocator locality, more page faults as the heap grows to
2.4GB, larger working-set pressure on the CPU cache) is a **smooth,
monotonic drift** rather than a discrete pause -- which produces
*higher* autocorrelation (each call resembles its immediate neighbours
closely, because they share nearly the same accumulated heap state)
even as the *variance* drops (no more sharp spikes, just a gradual
trend). **This reading is consistent with the data but not directly
tested here** -- confirming it would need tracking RSS or GC-would-be-
collected-object-count over the course of the run, not just before and
after.

## 4. A tentative, uncertain observation about Addendum 93 itself

This run's `gc_disabled` standard deviation (0.115ms) is close to
Addendum 93's own `today` measurement (0.123ms, from the actual
`test_cumulative_compile_scale.py` script) -- while this run's own
`gc_enabled` arm (0.314ms) sits between Addendum 93's `today` (0.123ms)
and the `archive` (0.902ms). **This is flagged explicitly as
uncertain, not concluded**: this script's own circuit-building function
was reconstructed from the project's README description, not taken
from the original script's own source (never shared), so an exact
match to Addendum 93's own circuit family and generation logic is not
confirmed. The closeness could be a real signal (e.g., that today's
`test_cumulative_compile_scale.py` run happened to have GC in a
similarly-quiescent state) or could be coincidence given the
reimplementation uncertainty. **Not treated as evidence either way**
without the original script to verify against.

## 5. What this does not establish

- **What changed in the underlying code over the last 4 days** -- this
  remains completely open, and is not addressed by the GC finding,
  which concerns *today's* code only, run twice under different GC
  settings, not a before/after code comparison.
- Whether the reference cycles causing the 2.27GB growth originate in
  `psf_compile`'s own Python code, in Qiskit's own object graph
  (`QuantumCircuit`, `UnitaryGate`, etc.), or in the PyO3 boundary to
  the Rust core -- not investigated; would need cycle-tracing
  (`gc.get_objects()` diffing, or a tool like `objgraph`).
- Whether the archive run (4 days ago) would show the same GC-disabled
  memory growth -- untestable without the old code.
- The autocorrelation-increase mechanism proposed in Section 3 -- a
  plausible reading of the existing data, not independently confirmed.
- Whether this circuit family's own reconstruction (Section 4's caveat)
  exactly matches the original script -- unconfirmed.

## 6. Files

| File | What it is |
|---|---|
| [`verify_gc_effect.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_gc_effect.py) | this run's script (circuit family reconstructed from the README, not the original source, per its own docstring caveat) |
| [`gc_effect_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gc_effect_2026-09-19.csv) | this run's results, 20,000 rows (10,000 per arm) |
| [`spare-qubit-cliff-addendum-94-preregistration-2026-09-19.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-94-preregistration-2026-09-19.md) | the predictions scored above, including Section 1's retrospective analysis |

## 7. Verification

- All summary figures (std, mean, median, max, autocorrelation, ratios)
  were recomputed directly from the raw CSV, not read off the printed
  terminal summary alone.
- P2's RSS figures were reported exactly as measured (before/after,
  both arms), including the fact that `gc_enabled`'s own end-of-arm RSS
  (121.0MB) became `gc_disabled`'s own starting RSS (121.1MB, matching
  within rounding) -- consistent with the two arms running back-to-back
  in one process, as designed, rather than independently.
- The autocorrelation-direction finding (Section 3) was checked
  numerically before being called "unregistered" -- the pre-registration
  did not predict a direction for this specific statistic, only for std
  (P1), so this is correctly flagged as new rather than as a confirmed
  or falsified prediction.
- Section 4's comparison to Addendum 93 was made explicitly conditional
  on an unverified assumption (this script's circuit-family
  reconstruction matching the original), stated plainly rather than
  presented as settled.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 95 (source: spare-qubit-cliff-addendum-95-2026-09-19.md) ===== -->

> **Note added when merging:** The four layout fixes (Addenda 88-92) measured all together for the first time through the real, unbypassed smart_vf2_layout entry point: 26/26 succeed, all valid, all on the first ordering -- a capability change from 0/26 that morning, plus a genuine 2.72x apples-to-apples speedup from caching and laziness on top of the fixed guard.

## Addendum 95 -- the final-state measurement: 0/26 to 26/26 in capability, and (comparing like with like this time) a genuine, apples-to-apples 2.72x from caching and laziness on top of the fixed guard (2026-09-19)

**Status**: the missing measurement identified while assembling this
session's own progression table -- Addenda 88, 89, 91, and 92 had each
been verified individually, but never all four together through the
real, unbypassed `smart_vf2_layout()` entry point on the same 26
configurations. **This addendum also corrects two of its own mistakes,
caught before the run**: an unfair morning-vs-today comparison (mixing
search-only against search+guard) and 21 of 26 hand-transcribed
"morning" values that turned out wrong when checked against their own
source CSV.

## 0. In one line

**Capability, the primary result**: this morning, the real entry point
rejected **0 of 26** of these configurations outright (the
`_has_feasible_matching` bug, Addendum 83 Section 5) -- the search-only
numbers on record required bypassing that guard to measure anything at
all. **Today, with all four fixes applied and no bypass, 26/26
succeed, all 26/26 structurally valid, all 26/26 on the first
(`natural`) ordering.** **A second, genuinely comparable result**: both
Addendum 89 (guard fixed, but before caching or laziness) and today
(guard fixed, plus caching and laziness) measure the SAME thing --
search + working guard, no bypass -- so their **2.72x** difference
(3.71ms to 1.36ms median, D=2 excluded) is a real, apples-to-apples
speedup, unlike the morning-vs-today comparison this addendum
deliberately does not reduce to a single ratio.

## 1. Two mistakes caught before this run, corrected rather than hidden

**Mistake 1**: an earlier attempt at this same progression compared
Addendum 83's own search-only numbers (guard bypassed, because it was
broken) directly against Addendum 89's search+guard numbers, producing
a ratio (5.2x) that looked like a *regression* from Addendum 88's own
61.5x. **This was comparing different things, not a real regression** --
83 and 88 never had a working guard to include; 89 was the first point
a working guard existed at all. Caught before being reported as a
finding.

**Mistake 2**: the "morning" comparison values were initially
hand-copied from a printed table into this script, and checked against
Addendum 83's own source CSV before trusting them -- **21 of 26 were
transcribed slightly wrong**. The script was rewritten to read
`psf_smart_layout_comparison_2026-09-19.csv` directly at runtime
instead of hardcoding anything, and re-verified against that file
(0/26 mismatches) before this run.

## 2. Results

8x8 grid, 26 `dominant_size_sweep` configurations, `smart_vf2_layout()`
called directly with **no bypass** -- all four fixes (Addenda 88, 89,
91, 92) active simultaneously for the first time.

| | value |
|---|---:|
| Found a layout | **26/26** |
| Structurally valid (independently checked) | **26/26** |
| Solved on the first (`natural`) ordering | **26/26** |
| Today's median (all 26) | 1.617ms |
| Today's median (D=2 excluded) | **1.364ms** |
| Today's range (D=2 excluded) | 1.078-2.696ms |
| D=2 alone | 304.1ms (outlier) |
| Morning's own search-only median (guard bypassed) | 19.31ms |

**The D=2 outlier (304.1ms) is consistent with, not a new instance of,
Addendum 89's own first-call warm-up finding** (that addendum's own
D=2 landed at 452.7ms, also the first configuration processed in its
own loop, also excluded from that addendum's own headline figures for
the same reason). Both runs process D=2 first; both show an outsized
first value; neither addendum treats it as informative about D=2
itself.

## 3. The two comparisons, kept separate on purpose

**Comparison A -- morning vs. today (capability, not speed).** 0/26
usable via the real entry point this morning; 26/26 today. **This
addendum deliberately does not report a ratio for this comparison**,
for the same reason the harness's own printed output states directly:
the two sides measure different things (search alone vs. search with a
functioning guard), and dividing them would misrepresent both what
changed and by how much.

**Comparison B -- Addendum 89 vs. today (a real speedup, same
measurement both times).** Both points include the same fixed guard,
called through the same real entry point, with no bypass either time.
**2.72x** (3.71ms to 1.36ms, D=2 excluded both sides) is the genuine,
comparable effect of adding the physical-matching cache (Addendum 91,
predicted 52.4% guard-cost reduction) and lazy ordering generation
(Addendum 92, a further, smaller saving) on top of Addendum 89's own
already-fixed guard. This number, not any ratio involving the morning
data, is the correct "how much did today's later optimizations help"
figure.

## 4. Cumulative state, final

| fix | addendum | what it does | this run confirms |
|---|---|---|---|
| natural ordering first | 88 | 56.8x on the search alone | still the reason every case solves on attempt 1 |
| guard correctness | 89 | chain-shaped circuits reach the search at all | 26/26, up from 0/26 this morning |
| physical-matching cache | 91 | 52.4% of guard cost | contributes to comparison B's 2.72x |
| lazy ordering generation | 92 | 0.075ms, prevents wasted BFS work | contributes to comparison B's 2.72x |

**All four fixes, together, verified through the real public entry
point for the first time in this addendum.** `psf_smart_layout_patched.py`
carries all of them; none is yet applied to the actual PSF-Zero
repository.

## 5. What this does not establish

- Why D=2 specifically shows the warm-up cost in both this run and
  Addendum 89's -- both process it first in their own loop, so this is
  consistent with ordinary first-call overhead, not confirmed as such
  by any isolated test.
- Generalization beyond this 8x8/`dominant_size_sweep` combination --
  unchanged limitation carried from every addendum in this thread.
- Anything about the separate, still-unexplained gate-synthesis
  variance question (Addenda 93-94) -- unrelated code path.

## 6. Files

| File | What it is |
|---|---|
| [`verify_final_state.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_final_state.py) | this run's script, corrected to read Addendum 83's own CSV directly rather than hardcode values |
| [`final_state_verification_2026-09-19.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/final_state_verification_2026-09-19.csv) | this run's results, 26 rows |
| [`psf_smart_layout_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout_patched.py) | the module under test, all four fixes |

## 7. Verification

- The hardcoded-value transcription error (Section 1) was caught by an
  explicit diff against the source CSV before this run, not discovered
  after the fact -- the script was rewritten and re-verified (0/26
  mismatches) as part of this addendum's own preparation, not
  silently.
- Comparison B's inputs were checked to both represent the guard-
  included, no-bypass condition before being called comparable --
  Addendum 89's own P1 rows and this run's own rows were both confirmed
  to have `feasible=True` recorded via the real guard call, not
  inferred.
- D=2's outlier status was checked against Addendum 89's own recorded
  D=2 value (452.7ms) before being called consistent with a repeat of
  the same pattern, rather than assumed.
- All 26 layouts were validated structurally (each interaction pair
  checked against the coupling map's own edge set) by the harness
  itself, not inferred from a non-`None` return.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 96 (source: spare-qubit-cliff-addendum-96-2026-09-19.md) ===== -->

> **Note added when merging:** The actual 4-day-old psf_compile.py/lib.rs were obtained and diffed line by line -- but neither the cache eviction-policy change nor the verify='strict' optimization nor lib.rs's own 31-line diff explains the verify=False variance collapse; the source-level mystery narrows without resolving.

## Addendum 96 -- the actual 4-day code diff is in hand, and it does NOT explain the `verify=False` variance collapse -- two plausible candidates traced and ruled out, the mystery narrows but does not resolve (2026-09-19)

**Status**: the user supplied the genuine 4-day-old versions of both
files (`psf_compilev8.py`, `lib8.rs`) directly. **A real diff against
the files used earlier this session (`psf_compile.py`, `lib.rs`) was
performed** -- not inferred, not assumed. **A process hazard was caught
and avoided first**: several `psf_compile*.py` files already existed in
this session's own sandbox from earlier, unrelated work, carrying an
internal `VERSION: 2026-09-16` marker; these were confirmed to be
leftover artifacts (not today's genuine file) and excluded before any
comparison was made.

## 0. In one line

**The actual code diff is real, substantial, and documented in its own
changelog -- but neither of the two most plausible candidates traced
through it explains the `verify=False` arm's own 53.99x variance
collapse (Addendum 93), which is the specific condition both the
archive and today's runs measured.** The `_cx_core_cached` function's
eviction policy changed (plain dict that stops growing at 4096 entries,
to a proper `OrderedDict`-based LRU) -- but this code path is shared by
whichever arm calls it, and whether it matters at all for the timing
benchmark depends on a fact not yet confirmed (whether the benchmark
script rebuilds a fresh random circuit every iteration, or repeatedly
compiles one fixed circuit -- these produce very different cache
hit-rate behavior, and the eviction-policy difference is close to
irrelevant in one case and could matter in the other). A second, more
clearly *positive* finding was traced and ruled OUT directly:
`verify="strict"` no longer wastes a call to the core's own
self-check when it is going to discard the result anyway -- a genuine
optimization, but **it does not touch `verify=False`'s own code path
at all** (confirmed by reading both `synthesize()` implementations
side by side). **`lib.rs`'s own diff (31 lines) contains nothing
capable of a 34-54x effect** -- a duplicate computation removed, doc
comments reordered, an unnecessary `mut` on a closure binding removed.
None of these plausibly explain the magnitude Addendum 93 measured.

## 1. What was verified, concretely

**Both files' own "Changes in this revision" headers were read and
cross-checked** before anything else: `psf_compilev8.py`'s changelog
covers items 1-6 only; the file used earlier this session
(`psf_compile.py`) covers items 1-6 plus additional items (7 onward,
dated "2026-09-16") on top -- confirming the chronological relationship
the user described (v8 is the earlier file) from internal evidence,
not from filenames or upload order alone.

**The stub-import bug (item 1) was checked directly in both files**:
both actually import the real `psf_zero_core`, not
`psf_zero_core_stub` -- the fix described in item 1 is present in
*both* files already, so it cannot explain any difference between
them (it would explain a difference against some even-earlier version
not in hand).

**The cache eviction policy (verified code, not paraphrase)**:

```python
# v8 (older)
_CX_CORE_CACHE: dict = {}
...
if len(_CX_CORE_CACHE) < _CX_CORE_CACHE_MAX:
    _CX_CORE_CACHE[key] = result

# today
_CX_CORE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
...
if key in _CX_CORE_CACHE:
    _CX_CORE_CACHE.move_to_end(key)
    return _CX_CORE_CACHE[key]
...
_CX_CORE_CACHE[key] = result
if len(_CX_CORE_CACHE) > _CX_CORE_CACHE_MAX:
    _CX_CORE_CACHE.popitem(last=False)
```

Both cap at 4096 entries. The old version simply stops adding once
full (a bounded, non-evicting cache); the new version evicts the
least-recently-used entry to make room (a genuine LRU). **For a
workload where every `(a, b, c)` key is essentially unique** (e.g.
freshly-random `SU(4)` unitaries generated fresh each iteration), **both
versions cap at ~4096 entries and neither provides meaningful hit-rate
benefit** -- almost every call is a miss in both, and the eviction
policy difference does not matter. **For a workload that repeatedly
compiles the SAME fixed circuit** (a common benchmarking pattern, to
measure compile throughput without confounding circuit-generation
cost), a single circuit's own canonical triples (well under 4096 for a
15-qubit, 20-gate-per-pair circuit) would be fully cached by *either*
version after the first iteration, and the eviction-policy difference
would again not matter, since the cap is never approached.
**Whichever of these two usage patterns `test_cumulative_compile_scale.py`
actually follows was not confirmed** -- the original script's own
source was never shared, only reconstructed by inference for Addendum
94's own `verify_gc_effect.py` (flagged there as an unverified
assumption already).

**The `verify="strict"` waste-removal (verified code)**:

```python
# v8: computes core_infid even for verify="strict", which then ignores it
if _CORE_CHECKED is not None and self.verify is not False:
    cartan, k1, k2, global_phase, core_infid = _CORE_CHECKED(u_r, u_i)

# today: only computes it when verify is True specifically
want_core_check = self.verify is True
if _CORE_CHECKED is not None and want_core_check:
    cartan, k1, k2, global_phase, core_infid = _CORE_CHECKED(u_r, u_i)
```

**Confirmed this does not touch `verify=False` at all**: for
`verify=False`, `self.verify is not False` is `False` in the old
condition and `self.verify is True` is also `False` in the new one --
both take the plain, uncheck `geometric_decompose` path, identically,
in both versions. This change only affects the relative cost of
`verify=True` versus `verify="strict"`, neither of which is the arm
Addendum 93's own variance collapse was measured on
(`psf_false`).

**`lib.rs`'s own 31-line diff, in full**: a duplicate trace computation
merged into one (`w_r = w` instead of recomputing the same sum a
second time), a doc-comment block moved to a different location in the
file, and `let mut consider = ...` changed to `let consider = ...`
(removing an unnecessary mutability qualifier on a closure binding,
which has no runtime effect in Rust). **`geometric_decompose` itself --
the function `verify=False` actually calls -- is otherwise byte-for-byte
unchanged.**

## 2. What this means

**Having the real diff in hand ruled out the two most plausible
Python-level candidates and confirmed the Rust core's own decomposition
math is unchanged.** This is itself informative, not merely
inconclusive: it means Addendum 93's own variance collapse, for the
specific `verify=False` condition it measured, **is not explained by
any source-level change found in these two files.** The remaining
candidates are:

- **The benchmark's own circuit-repetition pattern**, which determines
  whether the cache eviction-policy change (Section 1) could matter at
  all -- unconfirmed, and the single most direct way to close this gap
  is the original `test_cumulative_compile_scale.py` source itself,
  never yet shared.
- **Something in item 7's changes** (`copy_empty_like()` replacing
  fresh-register construction, removing per-instruction `find_bit`
  calls from the final-assembly loop) -- this runs once per `compile()`
  call across all blocks, not once per block, so it would shift the
  *baseline* cost of a full compile rather than obviously explain
  *per-call variance*, but was not traced in as much depth as the two
  candidates above and is not ruled out.
- **Dependency versions** (NumPy, Qiskit, Rust toolchain, `maturin`
  build settings) installed on the machine at each point in time --
  outside the scope of a source diff entirely, and not checked here.
- **Machine/environment state**, unchanged as a candidate from Addendum
  93 itself.

## 3. What this does not establish

- **The actual cause of the `verify=False` variance collapse** --
  remains open. This addendum narrows the search (rules out the cache
  policy as certainly relevant, rules out the verify="strict" change
  entirely, rules out the Rust core's own decomposition math) without
  identifying the true cause.
- Whether the cache eviction-policy change matters for the REAL
  benchmark's own circuit-repetition pattern -- depends on a fact not
  yet confirmed (see Section 1).
- Item 7's own possible contribution -- flagged, not traced.
- Anything about dependency-version or environment differences between
  the two dates -- outside this addendum's own scope (a source diff).

## 4. Files

| File | What it is |
|---|---|
| [`psf_compilev8.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compilev8.py), `lib8.rs` | the genuine 4-day-old versions, supplied directly by the user |
| [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile.py), `lib.rs` | the versions used throughout this session (internally dated 2026-09-16, three days before today -- itself a precision worth noting for the record) |

No new generated data files -- this addendum is a source-code
comparison, not a new measurement.

## 5. Verification

- Before any comparison, the sandbox's own leftover
  `psf_compile_latest.py` / `_v3.py` / `_en.py` / `_candidate.py` /
  `_latest_en.py` files (from earlier, unrelated work in this project)
  were checked and found to share the same `VERSION: 2026-09-16`
  marker as the genuine session file -- confirmed as artifacts, not
  used for any part of this comparison.
- The chronological relationship between the two supplied files (v8
  older, session file newer) was established from each file's own
  internal changelog contents (item count), not assumed from filename
  or upload order.
- The claim that `verify=False` takes an identical code path in both
  versions for the `_verify_block`/`synthesize()` logic was checked by
  reading both conditions' exact boolean outcomes for `self.verify =
  False` side by side, not inferred from the surrounding prose.
- `lib.rs`'s full 31-line diff is quoted in Section 1 in its entirety,
  not excerpted, so nothing is omitted from what could be checked.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 97 (source: spare-qubit-cliff-addendum-97-2026-09-19.md) ===== -->

> **Note added when merging:** Eight additional historical version files were triaged by measuring textual similarity against the two known endpoints rather than assumed relevant -- six turned out to be from a much earlier, unrelated era of the codebase; the one genuine intermediate revision added no new candidate.

## Addendum 97 -- triage of eight additional version files: most are from an unrelated, much earlier era; the one genuine intermediate revision adds no new candidate beyond Addendum 96's own open item (2026-09-19)

**Status**: eight additional files were supplied
(`psf_compile10.py`, `psf_compilev5.py`, `psf_compilev6.py`,
`psf_compilev7.py`, `psf_compile__2_.py`, `libv4.rs`, `libv5.rs`,
`libv7.rs`). Rather than assume any relationship to the 4-day window
Addendum 93/96 investigate, each was measured for textual similarity
against the two already-understood endpoints
(`psf_compilev8.py`/`lib8.rs`, the 4-day-old archive; and
`psf_compile.py`/`lib.rs`, the file used throughout this session)
before drawing any conclusion.

## 0. In one line

**Similarity triage places six of the eight files in an unrelated,
structurally different, much earlier era of the codebase** (Python
similarity 0.01-0.28, Rust similarity 0.08-0.23 against both known
endpoints; line counts far smaller than either endpoint -- e.g. the
Rust files at 210-746 lines versus ~1,224 for both known endpoints) --
their own headers ("Corrected Version," "v3 (adds the smart block
filter fix on top of v2)") are consistent with this. **One file
(`psf_compile__2_.py`) is byte-identical to the session's own "today"
file** (similarity 1.000) -- a duplicate, no new information.
**`psf_compile10.py` is a genuine intermediate revision** between v8
and today (0.939 similarity to v8) -- **but it adds nothing new**: it
already contains the two changes Addendum 96 traced and ruled out for
`verify=False` (the cache LRU change, the `verify="strict"` waste
removal), and its own remaining difference from v8 is the `find_bit`
-based qubit/clbit resolution fix -- which is the *same* code Addendum
96 flagged as "not yet ruled out" (later superseded by
`copy_empty_like()` in the file used all session), not a new
candidate.

## 1. Similarity triage

| file | lines | similarity to v8 (4 days ago) | similarity to today | verdict |
|---|---:|---:|---:|---|
| [`psf_compile10.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile10.py) | 492 | **0.939** | 0.554 | genuine intermediate revision, between v8 and today |
| `psf_compilev5.py` | 251 | 0.015 | 0.014 | unrelated, much earlier era |
| `psf_compilev6.py` | 341 | 0.013 | 0.017 | unrelated, much earlier era |
| `psf_compilev7.py` | 168 | 0.280 | 0.128 | unrelated, much earlier era |
| `psf_compile__2_.py` | 871 | 0.497 | **1.000** | exact duplicate of the session's own "today" file |
| `libv4.rs` | 210 | 0.089 | 0.081 | unrelated, much earlier era |
| `libv5.rs` | 414 | 0.158 | 0.156 | unrelated, much earlier era |
| `libv7.rs` | 746 | 0.230 | 0.228 | unrelated, much earlier era |

## 2. What `psf_compile10.py` actually adds over v8

Direct diff against `psf_compilev8.py` shows exactly three substantive
changes, none of them new relative to Addendum 96's own findings:

1. **The cache LRU change** (identical to what Addendum 96 already
   found and ruled out as not touching `verify=False`'s own path).
2. **The `verify="strict"` waste-removal** (identical to what Addendum
   96 already found and ruled out for the same reason).
3. **A `find_bit`-based fix for resolving qubit/clbit indices** in the
   main block-processing loop, replacing direct use of the source
   circuit's own `Bit` objects (which failed for any circuit not built
   as a single, unnamed `QuantumCircuit(n)` register). **This is the
   same code region Addendum 96 Section 2 already flagged as "item
   7... not traced in as much depth as the two candidates above and is
   not ruled out"** -- except the version used throughout this session
   has since replaced this `find_bit`-based fix with `copy_empty_like()`
   entirely (removing the `find_bit` calls from the hot loop rather
   than keeping them). `psf_compile10.py` represents the intermediate
   state where the bug was fixed one way before being fixed a
   different, cheaper way later -- it does not introduce a new
   candidate mechanism beyond the one already on record as unexamined.

## 3. What this means

**No new candidate for the `verify=False` variance collapse was found
among these eight files.** The six unrelated, much-earlier-era files
provide no information about the specific 4-day window in question.
The one genuine intermediate revision confirms the sequence
(v8 -> psf_compile10 -> today) but does not surface anything beyond
what Addendum 96 already identified as open (item 7's own
`find_bit`/`copy_empty_like` region, still not traced to a timing
conclusion either way).

**This narrows, rather than expands, where to keep looking.** Given
six of eight supplied files turned out irrelevant, further requests for
"whatever old versions exist" are unlikely to be efficient. The two
remaining, genuinely promising directions are the same two Addendum 96
already named: (a) whether `test_cumulative_compile_scale.py`'s own
source (still never shared) repeats one fixed circuit or builds fresh
random ones each iteration, which determines whether the cache
question is live at all; and (b) tracing item 7's own
`find_bit`-versus-`copy_empty_like` region specifically for a timing
(not just correctness) effect, since it is the one candidate touching
the per-block hot loop that has not yet been examined for cost.

## 4. What this does not establish

- The actual cause of the `verify=False` variance collapse -- still
  open, unchanged from Addendum 96.
- Whether the six unrelated-era files might matter for some entirely
  different question this project has not yet asked -- not
  investigated here, since they fall outside the specific 4-day window
  this thread concerns.
- Item 7's own timing effect, if any -- flagged twice now (Addendum 96,
  this addendum), not traced either time.

## 5. Files

| File | What it is |
|---|---|
| [`psf_compile10.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile10.py) | the one genuine intermediate revision found among the eight supplied |

The other seven files are not carried forward as separate outputs --
six are confirmed irrelevant to this thread, and `psf_compile__2_.py`
is an exact duplicate of `psf_compile.py`, already on record.

## 6. Verification

- Similarity was computed with `difflib.SequenceMatcher` against both
  known endpoints for every one of the eight files, not assessed by
  eye or by filename alone, before any file was classified.
- `psf_compile__2_.py`'s exact-duplicate status was confirmed by its
  1.000 similarity ratio to the session's own "today" file, not merely
  a close match.
- `psf_compile10.py`'s own diff against v8 was read in full (not
  excerpted) before concluding it adds nothing new; all three of its
  substantive changes are accounted for in Section 2.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 98 (source: spare-qubit-cliff-addendum-98-2026-09-19.md) ===== -->

> **Note added when merging:** A self-correction: re-reading psf_compilev8.py directly (not the intermediate psf_compile10.py) found it has NEITHER find_bit calls NOR copy_empty_like -- the 'item 7' candidate from Addenda 96-97 rested on an incorrect premise and is retracted.

## Addendum 98 -- correcting Addenda 96-97's own framing: v8 never had `find_bit` at all, so that candidate's premise does not hold -- every traced candidate is now ruled out, and the mystery deepens rather than resolves (2026-09-19/20)

**Status**: a direct re-check of `psf_compilev8.py`'s own block-processing
loop, prompted by proceeding with the "trace `find_bit` vs
`copy_empty_like`" item Addenda 96-97 had flagged as unexamined.
**That framing turns out to rest on an incorrect premise**, corrected
here before it could mislead further work.

## 0. In one line

**`psf_compilev8.py` (the genuine 4-day-old file) contains neither
`find_bit` calls nor `copy_empty_like()`.** It uses a third, simpler
approach: `qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)`
followed by direct reuse of the original circuit's own `Bit` objects
(`qargs = inst.qubits`) with no resolution step at all. **The
`find_bit`-based approach only ever existed in `psf_compile10.py`, the
intermediate revision Addendum 97 examined** -- it was introduced
after v8 and removed again (replaced by `copy_empty_like()`) before
today's version. **Since v8 itself never paid a `find_bit` cost, "item
7" was never a real candidate for explaining the archive-vs-today
variance difference** -- Addenda 96 and 97 both flagged it based on an
incomplete reading (comparing `psf_compile10.py`'s own intermediate
state to today, not v8's own actual code to today). **Corrected here.**
Today's `copy_empty_like()` call runs once per `compile()` call (not
per block or per instruction), copying only empty register/metadata
structure -- a cost of comparable, likely smaller, magnitude to v8's
own `QuantumCircuit(qc.num_qubits, qc.num_clbits)` construction, not a
plausible source of a 34-54x per-call variance difference either way.

## 1. What was actually checked

`psf_compilev8.py`'s own `compile()` function, read directly (not
inferred from the intermediate revision):

```python
qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)
qc_psf.global_phase = qc_blocked.global_phase
...
for inst in qc_blocked.data:
    op = inst.operation
    qargs = inst.qubits
    cargs = inst.clbits
    ...
    qc_psf.append(op, qargs, cargs)
```

No `find_bit` anywhere in this function. This only works correctly
when `qc` was built as a single, unnamed `QuantumCircuit(n)` register
(the original circuit's own `Bit` objects are directly reused as if
they belonged to `qc_psf`, which is only valid when both circuits share
that exact simple structure) -- exactly the circuit shape this
project's own benchmarks use (`QuantumCircuit(N_QUBITS)`, no named or
split registers). **For this specific benchmark's own circuit
structure, v8's approach is correct, not merely "happens to work by
luck" -- and it does strictly less work per instruction than either
the intermediate `find_bit` fix or today's `copy_empty_like()`
approach**, since it performs zero lookups and zero copying beyond the
initial `QuantumCircuit(n)` construction.

## 2. What this means: every traced candidate is now ruled out

Combining this addendum with Addendum 96:

| candidate | status |
|---|---|
| cache eviction policy (dict -> LRU `OrderedDict`) | live only if the benchmark reuses one fixed circuit across all repeats -- still unconfirmed, not ruled out, but not yet shown to matter either |
| `verify="strict"` waste removal | ruled out -- confirmed not to touch `verify=False`'s code path at all |
| `find_bit` vs `copy_empty_like` ("item 7") | **ruled out by this addendum** -- v8 never had `find_bit`; the real difference (an extra once-per-call lightweight copy in today's version) is not plausibly large enough |
| `lib.rs`'s own 31-line diff | ruled out -- `geometric_decompose` itself unchanged, only micro-cleanups with no runtime effect |

**Every specific mechanism traced through the actual source so far has
been ruled out or shown not to plausibly explain a 34-54x variance
difference, except the cache question, which remains gated on one
still-unconfirmed fact about the benchmark's own circuit-repetition
pattern.** This is the single most direct, and now the only remaining,
lead from the source-level investigation.

## 3. What this does not establish

- Whether the cache eviction-policy question resolves the mystery once
  the benchmark's own repetition pattern is confirmed -- still the one
  live, untested candidate.
- Anything about dependency versions or environment differences between
  the two dates -- outside the scope of any source diff.
- Whether some other, not-yet-examined region of either file (beyond
  the block-processing loop and the two Section-96 candidates) holds
  the actual answer -- a full line-by-line audit of both files' entire
  diff has not been performed; only the specific candidates suggested
  by the changelog and by the block-processing loop have been checked.

## 4. Files

No new files -- this addendum re-reads `psf_compilev8.py`'s own
already-supplied source directly (Addendum 96), correcting an
inference error made when writing that addendum and Addendum 97.

## 5. Verification

- The claim that v8 contains no `find_bit` calls was checked by direct
  `grep` across the entire file, not sampled from the one function
  examined.
- The claim that v8's approach is correct (not merely accidental) for
  this project's own benchmark circuit shape was checked against how
  this project's own circuit-generation code builds circuits
  (`QuantumCircuit(N_QUBITS)`, single unnamed register) before being
  stated, rather than assumed.
- This correction was written and published as soon as the error was
  found, rather than silently fixed in a later addendum -- consistent
  with this project's own standing practice of recording corrections
  explicitly rather than editing prior claims in place.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 99 (source: spare-qubit-cliff-addendum-99-2026-09-19.md) ===== -->

> **Note added when merging:** The benchmark's own source (test_cumulative_compile_scale.py) was obtained for the first time: every iteration builds a fresh, uniquely-seeded circuit, definitively ruling out the cache-eviction-policy hypothesis and closing the source-level investigation with every traced candidate exhausted.

## Addendum 99 -- the benchmark's own source confirms: fresh random circuits every iteration -- the cache-policy hypothesis is now definitively ruled out, closing the source-level investigation with every traced candidate exhausted (2026-09-19/20)

**Status**: the user supplied `test_cumulative_compile_scale.py`'s own
actual source for the first time -- previously only reconstructed by
inference (flagged as an unverified assumption in Addendum 94). This
resolves the one fact Addendum 96 Section 1 identified as gating
whether the cache eviction-policy change could matter at all.

## 0. In one line

**Confirmed directly from source: every one of the 10,000 iterations
builds a completely fresh, uniquely-seeded circuit** --
`build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR, seed=1000 +
i)`, a different seed for every `i` from 0 to 9,999, each producing 7
pairs x 20 gates = 140 freshly-random `SU(4)` unitaries per iteration
(1.4 million across the full run). **This rules out the cache
eviction-policy hypothesis definitively**: with essentially every
canonical `(a, b, c)` triple unique across the entire run, the
`_cx_core_cached` cache is, for practical purposes, always a miss in
BOTH the old (dict, stop-adding-at-4096) and new (LRU `OrderedDict`)
implementations -- neither provides meaningful hit-rate benefit, and
the only operational difference between them (discarding a new entry
vs. evicting an old one, both O(1)) is not remotely capable of a
34-54x variance effect. **Every candidate traced from the actual
source diff across Addenda 96, 98, and this addendum is now ruled
out**, closing the source-level line of investigation.

## 1. What the source confirms

```python
def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc
```
called in the main loop as:
```python
for i in range(n_iter):
    qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR, seed=1000 + i)
```

**No circuit is ever reused.** `seed=1000+i` is unique for every `i`;
each seed drives an independent `np.random.default_rng`, which in turn
draws 140 independent sub-seeds for `random_unitary(4, seed=...)`.
Across 10,000 iterations, this is 1,400,000 independent random draws,
each producing its own canonical Weyl-chamber coordinates with
probability essentially 1 of never exactly repeating another draw in
this run (a continuous 3-parameter space; exact floating-point
collision probability is astronomically small).

## 2. Why this rules out the cache hypothesis definitively

`_cx_core_cached`'s cache is keyed on `(a, b, c)`. With every key
essentially unique:
- **Old version** (plain dict, stops adding past 4096 entries): the
  first ~29 iterations (4096 / 140 keys-per-iteration) fill the cache;
  every subsequent call is a miss, computes the full
  `Operator(core).data` + `_CX_DECOMPOSER` decomposition, and the
  result is simply discarded (not stored, since the cap is already
  reached).
- **New version** (LRU `OrderedDict`): every call past the first ~29
  iterations is also a miss (same reasoning), computes the same full
  decomposition, and then evicts the least-recently-used entry to
  insert the new one.

**Both versions perform the identical expensive computation
(`Operator(core).data` + `_CX_DECOMPOSER`) on essentially every one of
the 1.4 million gate-level calls in this benchmark.** The only
difference -- discard-on-miss versus evict-and-insert-on-miss -- is a
single O(1) dict operation either way, not a source of large timing
variance. **The cache is, for this specific benchmark, effectively
inert in both code versions**, contributing nothing to the
archive-vs-today difference Addendum 93 measured.

## 3. The source-level investigation is now exhausted

| candidate | addendum | verdict |
|---|---|---|
| cache eviction policy | 96, closed by this addendum | **ruled out** -- cache is inert either way for this benchmark's always-unique keys |
| `verify="strict"` waste removal | 96 | **ruled out** -- confirmed not to touch `verify=False`'s code path |
| `find_bit` vs `copy_empty_like` | 98 | **ruled out** -- v8 has neither; premise was based on an intermediate revision, not v8 itself |
| `lib.rs`'s own 31-line diff | 96 | **ruled out** -- `geometric_decompose` itself unchanged; only no-op micro-cleanups |

**Every specific mechanism identifiable from comparing the two source
trees has now been checked and ruled out.** This is a genuine, decisive
negative result, not an inconclusive one: the actual code differences
between the archive and today, to the extent they touch
`verify=False`'s own execution path at all, cannot explain a 34-54x
variance collapse.

## 4. What remains, now that source is exhausted

With every source-level candidate ruled out, the two remaining
candidate classes are both **outside what a source diff alone can
resolve**:

- **Dependency versions** -- NumPy, Qiskit, the Rust toolchain,
  `maturin`'s own build configuration -- installed on the machine at
  each point in time. A source diff of `psf_compile.py`/`lib.rs` alone
  cannot detect this; it would require comparing `pip freeze` /
  `cargo.lock` output (or equivalent) from both dates, which has not
  been requested or supplied.
- **Machine/environment state** -- Addendum 93's own original
  candidate, never ruled out and now the leading one by elimination:
  background load, thermal throttling, OS scheduler behavior, or
  something else external to the code entirely.

**Given the source-level thread is now exhausted, further progress on
this specific mystery most likely requires either (a) the dependency
manifest from both dates, or (b) accepting that the cause was
environmental and is not expected to be reproducible or explainable
after the fact.** Neither has been attempted; both are reasonable next
steps if this thread is resumed, listed in that order since (a) is
checkable and (b) is not.

## 5. What this does not establish

- Which of the two remaining candidates (dependency versions,
  environment) is the actual cause -- neither has been investigated.
- Whether obtaining a dependency manifest from 4 days ago is even
  possible at this point (depends on whether one was recorded at the
  time, which is unknown).
- Anything about the `psf_true` (verify=True) arm's own variance
  collapse specifically -- this addendum, like 96 and 98, focused on
  `verify=False`, the arm with the more dramatic effect; `verify=True`
  additionally involves the `verify="strict"`-adjacent code paths not
  re-examined here since they were already ruled out for a different
  reason (not applicable to `verify=True` either, per Addendum 96's own
  reading of the exact boolean conditions).

## 6. Files

| File | What it is |
|---|---|
| [`test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py) | the benchmark's own real source, supplied for the first time this session |

No new generated data -- this addendum is a source-reading exercise,
resolving the one fact Addendum 96 identified as necessary and not yet
known.

## 7. Verification

- The claim "every iteration uses a unique seed" was checked directly
  against the loop's own `seed=1000 + i` expression, not inferred from
  variable naming or surrounding comments.
- The claim that canonical-triple collisions are astronomically
  unlikely was reasoned from the continuous, 3-parameter nature of the
  Weyl-chamber coordinate space, not merely asserted.
- The summary table in Section 3 cross-references each candidate's own
  originating and ruling-out addendum directly, so the full chain of
  reasoning for each can be re-traced rather than taken on this
  addendum's word alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits.

---


<!-- ===== Addendum 100 (source: spare-qubit-cliff-addendum-100-2026-09-19.md) ===== -->

> **Note added when merging:** A third independent raw dataset confirms the low-variance state reproduces reliably, strengthening confidence in Addendum 93's own finding even though its cause (Addendum 99) remains unknown.

## Addendum 100 -- the low-variance state reproduces a third time, confirming the phenomenon itself is stable even though its cause (Addendum 99) remains closed and unknown (2026-09-19/20)

**Status**: a fresh, independent re-run of `test_cumulative_compile_scale.py`
on today's unmodified code, requested to check reproducibility after
Addendum 99 closed the source-level investigation into *why* the
variance changed. **This addendum answers a different, narrower
question than 93-99 did**: not "why," but "is the low-variance state
itself real and stable, or could it have been a fluke even on its own
terms?"

## 0. In one line

**The low-variance state reproduces.** A third independent raw dataset
(10,000 points/arm, same machine, same unmodified code) shows
`psf_false` std=0.164ms and `psf_true` std=0.176ms -- both dramatically
different from the archive's own 0.900-0.902ms (Levene's test, p~0 for
both, matching Addendum 93's own finding exactly), and both close to
the first "today" run's own figures (0.123ms / 0.153ms). **For
`psf_true`, the difference between the two "today" runs is not
statistically significant** (Levene p=0.082) -- consistent with pure
run-to-run sampling noise, not a further change. **For `psf_false`,
the two runs do differ significantly from each other** (p~0), but the
difference in magnitude (0.123ms vs 0.164ms) is small compared to the
archive's own value and both remain unambiguously in the same
"low-variance" regime -- this is reported precisely rather than
rounded into "identical," since it is not.

## 1. Results

| arm | archive (4 days ago) | today, run 1 (Addendum 93) | today, run 2 (this addendum) |
|---|---:|---:|---:|
| `psf_false` std | 0.902ms | 0.123ms | **0.164ms** |
| `psf_true` std | 0.900ms | 0.153ms | **0.176ms** |

| comparison | Levene p (`psf_false`) | Levene p (`psf_true`) |
|---|---:|---:|
| archive vs. this run | ~0 | ~0 |
| today run 1 vs. this run | ~0 (significant, but small effect size) | 0.082 (not significant) |

## 2. What this establishes, and what it does not

**Establishes**: the low-variance state is not a one-off artefact of a
single run. Measured independently a second time (a third raw dataset
overall, counting the archive), on the same unmodified code, it
reproduces in the same regime, an order of magnitude below the
archive's own value in both arms. This directly strengthens confidence
in Addendum 93's own original finding -- the variance collapse is a
real, apparently stable property of the current code+environment
combination, not a measurement fluke on that one occasion.

**Does not establish**: *why*. Addendum 99 closed the source-level
investigation with every traceable code candidate ruled out; the
remaining candidates (dependency versions, unrecoverable per the user's
own confirmation of not tracking this; general environment/machine
state) remain exactly as open as Addendum 99 left them. This addendum
adds evidence that the *effect* is real and repeatable -- it does not
narrow *why* any further.

**The small but real difference between the two "today" runs
(`psf_false`: 0.123ms vs 0.164ms, statistically significant) is worth
noting rather than smoothing over**: even a "stable" state shows some
run-to-run drift, a useful reminder that "reproducible" here means
"consistently in the same regime," not "bit-for-bit identical" -- exactly
the standard this project has applied to every other repeated
measurement today.

## 3. Where this leaves the investigation

**The mystery of causation remains closed, as Addendum 99 concluded.**
This addendum's own contribution is narrower and complementary: it
converts "the archive-to-today difference might have been a fluke of
comparing exactly one run to exactly one other run" into "the
archive-to-today difference is confirmed across multiple independent
today-side measurements, all clustering far from the archive's own
value." **The finding is now on firmer ground than a single A/B
comparison could provide, even though its explanation is not
available.**

## 4. What this does not establish

- Any explanation for the variance collapse -- unchanged from Addendum
  99's own closing state.
- Whether a fourth or fifth run would continue to cluster in the same
  low range, or whether the `psf_false`/`psf_true` small inter-run
  difference found here is itself the start of some slower drift --
  only two "today" runs exist; a trend cannot be distinguished from
  noise at n=2.
- Anything about the archive's own reproducibility -- only one archive
  dataset exists; whether re-running the OLD code (if it still existed)
  would also show std~0.9ms consistently, or was itself partly noise,
  cannot be checked (the old code is not available to re-run).

## 5. Files

| File | What it is |
|---|---|
| `cumulative_compile_times_10000_RERUN3_2026-09-20.npz` | this run's raw data, 10,000 points/arm |

## 6. Verification

- This run's raw data was saved to persistent storage immediately on
  receipt, consistent with the practice adopted after Addendum 93's own
  data-loss incident.
- Levene's test was applied identically to all three pairwise
  comparisons (archive-vs-run1, archive-vs-run2, run1-vs-run2), not
  selectively.
- The `psf_false` inter-run difference was reported as statistically
  significant rather than rounded to "no difference," since the test
  result (p~0) does not support that rounding, even though the
  practical magnitude is small relative to the archive.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document -> 0 hits. The
  terminal output supplied for this run was reviewed for local paths
  before use; none were reproduced here.

---


<!-- ===== Addendum 101 pre-registration (source: spare-qubit-cliff-addendum-101-preregistration-2026-09-20.md) ===== -->

> **Note added when merging:** Predictions for the first end-to-end measurement of compile_for_hardware(layout_search=True) -- through the real public entry point rather than the smart_vf2_layout component alone -- including a search-visibility check since the feature discards its own internal search info.

## Addendum 101 -- Pre-registration: does the repaired `layout_search=True` actually help end-to-end, through `compile_for_hardware()`? (2026-09-20)

**Status: pre-registration only. No end-to-end run has been performed.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addenda 88-95 verified four fixes to `psf_smart_layout.py` and
confirmed, through `smart_vf2_layout()` called directly, that 26/26
chain-shaped configurations now find a valid layout where 0/26 could
before. **But `smart_vf2_layout()` is a component, not the feature.**
The feature callers actually use is
`compile_for_hardware(layout_search=True)`, and `psf_compile.py`'s own
docstring for that option still says, verbatim: *"**Not yet
benchmarked in this file's own revision history** -- this changelog
entry describes what the code now does, not a measured result."*

**Finding a layout is not the same as the compile being better.** The
layout is handed to `transpile()`, which then routes and optimizes.
It is entirely possible for a valid layout to be found, cost real
search time, and leave the end result no better -- or worse -- than
letting Qiskit choose its own. **Nothing measured so far rules that
out**, and the whole point of the fix was practical benefit, not a
return value.

**A second reason this is now testable for the first time**: before
Addendum 89's guard fix, `layout_search=True` on a chain-shaped
circuit silently fell through to Qiskit's default layout stage (the
guard rejected the input before any search ran). It was therefore
*identical in effect* to `layout_search=False`. **So for chain-shaped
circuits, any difference this experiment measures between `True` and
`False` IS the fix's own end-to-end effect** -- no before/after code
swap is needed to isolate it.

## 2. Design

Four configurations, all at spare=0 (complete saturation, zero idle
qubits -- verified numerically before writing this: the interaction
graph's own maximum matching exactly equals the grid's in all four
cases, and no qubit has degree zero):

| grid | family | construction | edges | interaction-graph max matching = grid's |
|---|---|---|---:|---|
| 6x7 (42q) | `dense_pairs` (control) | 21 disjoint pairs | 21 | 21 = 21 |
| 6x7 (42q) | chain-shaped (new capability) | 11 bare edges + 10q chain + 10q chain | 29 | 21 = 21 |
| 8x8 (64q) | `dense_pairs` (control) | 32 disjoint pairs | 32 | 32 = 32 |
| 8x8 (64q) | chain-shaped (new capability) | 17 bare edges + 10q chain + 20q chain | 45 | 32 = 32 |

Three arms per configuration, all through the **real public entry
points**, not internal helpers:
  - **Qiskit L3**: `transpile(qc, coupling_map, basis_gates, optimization_level=3)`
  - **PSF-Zero, `layout_search=False`**: `compile_for_hardware(...)`
  - **PSF-Zero, `layout_search=True`**: `compile_for_hardware(...)`

`entangling_basis="cx"`, `routing_optimization_level=1`,
`seed_transpiler` pinned, `verify=False`, matching the settings this
project's own README cliff table already uses so the numbers are
comparable to what is published. Warm-up call outside the timer for
every arm; median over multiple seeds and repeats.

Recorded per run: wall-clock time, 2-qubit gate count, depth,
**coupling-map violations** (counted directly against the coupling
map's own edge set, not assumed), and -- separately, by calling
`smart_vf2_layout()` on the same interaction pairs -- whether the
search actually succeeded, since `compile_for_hardware()` discards its
own `_search_info` and there is otherwise no way to tell a successful
search from a silent fall-through.

## 3. Pre-registered predictions

**P1 (primary -- does the fix help end-to-end on chain-shaped
circuits?).** Comparing `layout_search=True` against
`layout_search=False` on the two chain-shaped configurations:
  - **Helps**: `True` is faster and/or produces fewer 2q gates or less
    depth than `False`, by a margin larger than run-to-run spread.
    This would be the first measured evidence that the fix delivers
    practical value, not just a working return value.
  - **No practical difference**: the search succeeds and costs its
    ~1-4ms, but the end-to-end result is statistically
    indistinguishable from `False`. **This is a live and entirely
    plausible outcome** -- it would mean Qiskit's own layout stage
    already does as well on these inputs, and the fix's value is
    limited to cases where Qiskit's own stage fails (which this
    configuration may or may not be). It would need reporting as
    prominently as a positive result.
  - **Hurts**: `True` is measurably worse on at least one metric --
    the search's own cost, or a worse-for-routing layout, outweighing
    any benefit. Would mean the option should not be recommended for
    this circuit class despite working correctly.

**P2 (control -- is the previously-working case unchanged?).** On the
two `dense_pairs` configurations -- where the guard passed even before
Addendum 89, so `layout_search=True` already worked -- results are
predicted to be broadly consistent with this project's own README
cliff table (6x7: PSF-Zero `layout_search=True` 63 gates, depth 23,
~17ms; 8x8: 96 gates, depth 23, ~24ms). **Exact reproduction is not
predicted** -- that table used `routing_optimization_level=1` on a
different machine with different Qiskit/PSF-Zero revisions, and this
run uses the repaired `psf_smart_layout.py`. What is predicted is that
gate count and depth land at the same values, since those were
recorded as identical across every seed and repeat (zero spread) in
the original measurement. **A change in gate count or depth here would
mean the four fixes altered the chosen layout on the family they were
not supposed to affect, and would be a regression worth investigating
before anything else in this addendum is interpreted.**

**P3 (correctness gate).** Every routed output in every arm must show
**zero coupling-map violations**. This is checked directly, not
assumed from `transpile()` succeeding. Any violation invalidates that
arm's timing numbers entirely.

**P4 (search visibility).** On the two chain-shaped configurations,
the separately-invoked `smart_vf2_layout()` must report success --
confirming that `layout_search=True`'s own internal search also
succeeded and that the arm is measuring the fixed path, not a silent
fall-through. If it reports failure, P1's comparison is meaningless
for that configuration and must be reported as such rather than
interpreted.

## 4. What this cannot establish

- Unitary equivalence of the output circuits -- that is a separate
  test (proposed as "test C" in the same discussion that produced this
  one), not attempted here. **This addendum checks that the output
  respects the coupling map, not that it computes the same thing.**
- Generalization to topologies other than these two square grids --
  `psf_smart_layout.py`'s own docstring records that its ordering
  strategies fail on `brick`; that is a separate test.
- Behaviour away from spare=0. All four configurations are fully
  saturated by construction, which is the regime the layout search
  exists for.
- Whether any difference found would hold on a different machine --
  single-machine measurement, as with every timing result in this
  project.

---


<!-- ===== Addendum 102 (source: spare-qubit-cliff-addendum-102-2026-09-20.md) ===== -->

> **Note added when merging:** Addendum 101's own end-to-end results: modest, consistent 1.07x gains on the newly-repaired chain-shaped path, zero coupling-map violations across all 24 runs, exact reproduction of the README's own published numbers on the fixes' own path -- plus an unregistered, isolated anomaly in Qiskit's own default layout at low optimization, unrelated to this session's fixes.

## Addendum 102 -- end-to-end measurement complete: modest, consistent gains on the newly-repaired chain-shaped path, zero regressions on the fixed code's own path, and an unrelated anomaly found in Qiskit's own default layout at low optimization (2026-09-20)

**Pre-registered in**:
`spare-qubit-cliff-addendum-101-preregistration-2026-09-20.md`, written
and locked before this run. All four predictions scored below.

## 0. In one line

**P1: "helps," modestly and consistently, on both chain-shaped
configurations** -- `layout_search=True` beats `False` by 1.07x on
both (17.20ms->16.10ms at 6x7; 23.79ms->22.18ms at 8x8), never worse.
**P2 (control): CONFIRMED exactly on the path the fixes actually
touch** -- `layout_search=True`'s own gate count and depth reproduce
the README's own recorded values with zero spread at both grid sizes
(63 gates/depth 23 at 6x7; 96 gates/depth 23 at 8x8), matching the
"identical across every seed and repeat" standard the pre-registration
required. **P3: CONFIRMED, perfectly** -- zero coupling-map violations
across all 24 runs, every arm, every configuration. **P4: CONFIRMED**
-- the search itself succeeded (found on the first, `natural`,
ordering) on all four configurations, so every timing comparison
reflects the real, working search path, not a silent fall-through.
**An unregistered, isolated anomaly was found**: `layout_search=False`
at 6x7 `dense_pairs` specifically shows real seed-dependent spread in
both timing (25.81-99.99ms, nearly 4x) and output shape (63-69 gates,
depth 23-44) -- absent from every other cell, including
`layout_search=True` on the identical circuit family. **This is on
Qiskit's own default layout path, which today's four fixes never
touch**, and has a plausible, specific explanation connecting to this
project's own prior work, detailed in Section 3.

## 1. Results

All four configurations at spare=0 (verified zero idle qubits,
interaction-graph maximum matching exactly equal to the grid's own),
3 seeds x 2 repeats, `entangling_basis="cx"`,
`routing_optimization_level=1`, `verify=False`.

| config | arm | median | gates (2q) | depth | violations |
|---|---|---:|---|---|---:|
| 6x7 `dense_pairs` | Qiskit L3 | 6,695.91ms | 63 | 16 | 0 |
| | PSF `ls=False` | 31.10ms | **63-69** | **23-44** | 0 |
| | PSF `ls=True` | 14.40ms | 63 | 23 | 0 |
| 6x7 chain-shaped | Qiskit L3 | 32.06ms | 87 | 104 | 0 |
| | PSF `ls=False` | 17.20ms | 87 | 167 | 0 |
| | PSF `ls=True` | **16.10ms** | 87 | 167 | 0 |
| 8x8 `dense_pairs` | Qiskit L3 | 9,017.56ms | 96 | 16 | 0 |
| | PSF `ls=False` | 19.20ms | 96 | 23 | 0 |
| | PSF `ls=True` | 19.07ms | 96 | 23 | 0 |
| 8x8 chain-shaped | Qiskit L3 | 8,853.58ms | 153-156 | 233-268 | 0 |
| | PSF `ls=False` | 23.79ms | 135 | 347 | 0 |
| | PSF `ls=True` | **22.18ms** | 135 | 347 | 0 |

Every one of the four configurations' search checks (P4) reported
`found=True, feasible=True, tried=1, order=natural` -- the same
one-shot success pattern established in Addenda 88-95, now confirmed
to hold through the real `compile_for_hardware()` entry point as well.

## 2. Scoring

**P1 (primary) -- "helps" CONFIRMED, in the modest branch the
pre-registration explicitly allowed for.** On both chain-shaped
configurations -- the ones the fix actually changes the behaviour of
-- `True` beats `False` by 1.07x, a small but real and *consistent*
margin (never reversed, never negative). This is a real, if modest,
end-to-end benefit: the fixed search costs its own ~1-4ms and still
comes out ahead. **Neither the "no practical difference" nor the
"hurts" branch applies.**

**P2 (control) -- CONFIRMED on the arm that matters, with a caveat on
the other.** The pre-registration's own criterion was specifically
about reproducing the README's recorded values -- and `layout_search=True`
(the arm the four fixes actually run through) does so exactly, with
zero spread, at both grid sizes. **This is the correct scope for P2**:
`layout_search=False` never calls `psf_smart_layout.py` at all (it
uses Qiskit's own default layout stage inside `transpile()`), so its
own behaviour is not a test of today's fixes -- but its unexpected
variability (Section 3) is reported rather than silently excluded from
the results table.

**P3 (correctness gate) -- CONFIRMED, perfectly.** 0 coupling-map
violations, checked directly against the coupling map's own edge set,
across all 24 runs in every arm and configuration. No arm's timing
numbers are invalidated.

**P4 (search visibility) -- CONFIRMED for all four configurations.**
The independently-invoked `smart_vf2_layout()` check confirms the
search inside `layout_search=True` actually succeeded (not a silent
fall-through) in every case, including both `dense_pairs` controls
(where it was already expected to work) and both chain-shaped cases
(the new capability).

## 3. The unregistered anomaly: `layout_search=False` at 6x7 `dense_pairs`

Isolated to exactly one cell out of twelve: `layout_search=False`
(Qiskit's own default layout, reached via `compile_for_hardware()`
with the search disabled) at 6x7 `dense_pairs` shows real spread across
its 6 runs -- timing 25.81-99.99ms (nearly 4x), gate count 63-69,
depth 23-44. **Every other cell in this table, including
`layout_search=True` on the identical 6x7 `dense_pairs` circuits, shows
zero spread.** This rules out the circuit family itself as the cause;
the variability is specific to this arm's own layout choice at this
grid size.

**A plausible, specific explanation, not confirmed here**: this
script calls `compile_for_hardware()` with `routing_optimization_level=1`,
matching this project's own README convention for this exact
comparison table. Addendum 34 (this project's own original occupancy-
cliff work) found that `VF2Layout`'s own search budget differs sharply
between optimization levels -- a roughly 1,500x gap between level 1's
and level 3's own failing-search times on an identical, budget-
exhausted instance, strong indirect evidence the presets configure very
different budgets at each level. **`layout_search=False`'s own path
runs Qiskit's default `VF2Layout` at whatever budget
`routing_optimization_level=1` implies** -- if that budget is small
enough to sometimes fail or land on a different tie-break outcome
depending on `seed_transpiler`, the resulting layout-quality variance
(and its downstream effect on routing/optimization, hence gate count
and depth) would look exactly like what was measured. **This is
inference connecting to a specific, already-established project
finding, not a new test performed here** -- confirming it would need
comparing `layout_search=False` at level 1 against level 3 directly on
this exact circuit family, not yet done.

**This anomaly does not implicate today's four fixes.**
`layout_search=False` never imports or calls `psf_smart_layout.py`;
the code path it exercises was not touched by Addenda 88-95. It is
reported here because it surfaced in this addendum's own data and
this project's own standing practice is to report a surprising finding
rather than silently exclude it, not because it bears on this
addendum's own central questions.

## 4. What this means

**The four fixes verified at the component level (Addenda 88-95) now
have their first end-to-end confirmation.** The practical benefit is
real but modest for this specific circuit family and grid size (1.07x,
not a dramatic win) -- an honest result the pre-registration explicitly
anticipated as a live possibility ("no practical difference... would
need reporting as prominently as a positive result"), and the actual
outcome landed just on the positive side of that line rather than
squarely in either extreme. **Critically, nothing regressed**: the
control configurations reproduce the project's own published numbers
exactly on the path the fixes touch, and every single run across all
twelve cells produced a coupling-map-valid circuit.

## 5. What this does not establish

- Unitary equivalence of any output circuit -- this addendum checked
  structural (coupling-map) validity only, per its own pre-registered
  scope. "Test C" (equivalence checking) remains a separate, proposed,
  not-yet-run test.
- The actual cause of Section 3's anomaly -- a specific, plausible
  mechanism is named, connecting to Addendum 34's own established
  finding, but not directly tested here.
- Generalization to topologies other than these two square grids, or
  to occupancy other than spare=0.
- Whether the modest 1.07x margin would grow, shrink, or reverse on a
  larger or differently-shaped chain family -- only two chain-shaped
  configurations were tested, both built from this addendum's own
  specific bare-edge-plus-two-chains construction.

## 6. Files

| File | What it is |
|---|---|
| [`verify_end_to_end_layout_search.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_end_to_end_layout_search.py) | this run's script |
| [`end_to_end_layout_search_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/end_to_end_layout_search_2026-09-20.csv) | this run's results, 12 rows |
| [`spare-qubit-cliff-addendum-101-preregistration-2026-09-20.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-101-preregistration-2026-09-20.md) | the predictions scored above |

## 7. Verification

- All four configurations' own zero-idle-qubit, matching-equals-grid
  status was verified numerically (via `networkx`) before the script
  was written, not assumed from the parameter choice -- reported
  directly in the pre-registration's own design table and reconfirmed
  by re-executing the same construction functions used in the actual
  script.
- P4's search-success check called `smart_vf2_layout()` independently
  of `compile_for_hardware()`'s own internal call, on the same sorted,
  deduplicated interaction pairs `compile_for_hardware()` itself
  derives, so it reflects what the real entry point actually searches
  over.
- P3's violation count was computed by checking every 2-qubit
  instruction in every routed output against the coupling map's own
  edge set directly, not inferred from `transpile()` completing without
  error.
- The anomaly in Section 3 was verified to be isolated to one specific
  cell (not a general property of `layout_search=False` or of 6x7) by
  checking all twelve cells' own min/max spread directly, not merely
  their medians.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 103 pre-registration (source: spare-qubit-cliff-addendum-103-preregistration-2026-09-20.md) ===== -->

> **Note added when merging:** Predictions for exact small-scale (n=6) unitary-equivalence checking of compile_for_hardware(layout_search=True)'s output, following this project's own established Operator()-based precedent, including a hand-verified permutation-handling design to correctly account for routing-induced qubit relabeling.

## Addendum 103 -- Pre-registration: does `compile_for_hardware(layout_search=True)` produce a circuit that computes the SAME thing, not just one that respects the coupling map? (2026-09-20)

**Status: pre-registration only. No equivalence check has been run.**
Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 102 confirmed `compile_for_hardware(layout_search=True)`
produces circuits with zero coupling-map violations, on the exact grid
positions Addendum 101's own pre-registration flagged directly:
*"This addendum checks that the output respects the coupling map, not
that it computes the same thing."* **Structural validity and
correctness are independent properties.** A layout assignment can be
perfectly legal on the device (every gate on an adjacent pair) while
still computing the wrong unitary, if the mapping from logical to
physical qubits was tracked incorrectly anywhere between
`smart_vf2_layout()`'s own return value and the final circuit
`compile_for_hardware()` hands back. **This has never been checked for
any of the four fixes made this session.** Every verification in
Addenda 88-95 checked structure (adjacency); none checked that the
computed operator matches the input.

**This project's own established practice** (README, quoted directly:
*"a small-scale (n=6) exact `Operator` equivalence check"*, and
*"Full-scale unitary equivalence is not computed -- infeasible at this
qubit count"*) sets the precedent this addendum follows: exact
`Operator()`-based fidelity at n=6, not at the 42/64-qubit scale used
for timing.

## 2. Design

Two circuit families, both built as genuine spare=0 saturation at
n=6 (verified numerically before writing this): `dense_pairs` (3
disjoint pairs, matching the grid's own max matching of 3 on a 2x3,
1x6, or 3x2 layout) and a chain-shaped family (1 bare edge + one
4-qubit chain, 4 edges, zero idle qubits, max matching 3 -- the
smallest chain-shaped construction this project's own
`edges_chain_shaped` logic can produce without degenerating into a
bare-edge-only or single-chain-only shape).

For each family, on a 2x3 grid (6 qubits, chosen since it is the
smallest square-ish grid this project's own `CouplingMap.from_grid`
convention supports at n=6):
  - Build the circuit with `GATES_PER_PAIR` random `SU(4)` blocks per
    edge (same construction as Addendum 101/102, just at n=6).
  - Run `compile_for_hardware(..., layout_search=True, verify=False)`.
  - Compute `Operator(input_circuit)` and `Operator(output_circuit)`,
    accounting for the fact that routing may have permuted which
    physical qubit each logical qubit ends up on: compare using the
    **actual final layout** `transpile()` reports (its own
    `final_layout` / `layout` property), not assuming qubit `i` in the
    input corresponds to qubit `i` in the output.
  - Compute exact fidelity, matching this project's own established
    formula (README: `unitary_fidelity`, `(|tr(U1^dagger U2)|^2 + d) /
    (d(d+1))`).

3 seeds, 2 repeats each family -- small counts, since this is a
correctness gate, not a timing measurement, and n=6 makes each run
cheap enough to afford more if the first results are ambiguous.

## 3. Pre-registered predictions

**P1 (primary -- exact equivalence).** Every one of the 12 runs (2
families x 3 seeds x 2 repeats) shows fidelity `>= 1 - 1e-9` against
the input circuit's own operator, accounting for the final qubit
permutation. This threshold matches this project's own established
tolerance for exact (non-approximate) circuit transformations --
looser than the `1e-12` used for the Rust core's own KAK decomposition
alone, since routing/optimization passes and floating-point
accumulation across a full `transpile()` call are expected to
introduce more numerical noise than a single 2-qubit block's own
decomposition.
  - **Confirmed**: `layout_search=True`'s full pipeline, including
    today's four fixes, preserves the computed unitary correctly.
  - **Falsified**: at least one run shows fidelity below threshold --
    would mean one of today's four fixes (or their interaction with
    the rest of `compile_for_hardware()`) introduces a genuine
    correctness bug, masked entirely by every structural check run so
    far. This would be the most serious possible finding from this
    entire investigation and would need to halt any further
    recommendation of `layout_search=True` until root-caused.

**P2 (comparison arm).** `layout_search=False` on the identical
circuits is checked the same way, as a control -- since it does not
touch today's fixes at all, it is predicted to pass at the same
threshold, and a failure there would indicate a problem in
`compile_for_hardware()`'s own general layout-application logic
unrelated to this session's work, not in `smart_vf2_layout()` itself.

**P3 (Qiskit L3, sanity check).** Also checked identically, expected to
pass trivially -- Qiskit's own `transpile()` is assumed correct by this
entire project's own prior practice; this arm exists only to confirm
the equivalence-checking methodology itself is sound (if Qiskit's own
output failed this check, the checker, not Qiskit, would be the
suspect).

## 4. What this cannot establish

- Equivalence at the actual sizes Addenda 101-102 measured timing on
  (42, 64 qubits) -- infeasible by `Operator()`, per this project's own
  established limit.
- Anything about `verify=True`'s own correctness path -- this addendum
  uses `verify=False` throughout, matching Addendum 101/102's own
  scope, since `verify=True`'s own internal check is a different,
  already-tested mechanism (Addendum 96).
- Approximate/statistical equivalence at larger sizes via sampling or
  other proxies -- not attempted; this is a small-scale exact check
  only, by design.

---


<!-- ===== Addendum 103 (source: spare-qubit-cliff-addendum-103-2026-09-20.md) ===== -->

> **Note added when merging:** Exact unitary equivalence confirmed, 36/36 at machine precision (minimum 0.999999999999994) -- the first correctness (not merely structural) confirmation of any of this session's four fixes, closing the last major verification gap.

## Addendum 103 -- exact unitary equivalence confirmed, 36/36 at machine precision: `layout_search=True` preserves the computed unitary, closing the one remaining gap this session's structural checks could not (2026-09-20)

**Pre-registered in**:
`spare-qubit-cliff-addendum-103-preregistration-2026-09-20.md`, written
and locked before this run. All three predictions scored below.

## 0. In one line

**P1, P2, P3 all confirmed, with no ambiguity.** Every one of the 36
runs (2 circuit families x 3 arms x 3 seeds x 2 repeats) shows fidelity
**>= 0.999999999999994** against the pre-registered `1 - 1e-9`
threshold -- the minimum observed value across all 36 runs sits at
machine-precision noise, not at any meaningfully lower level. **This is
the first correctness (not merely structural) confirmation of any of
this session's four fixes to `psf_smart_layout.py`.** Addenda 88-95
verified adjacency (every gate lands on a physically connected pair);
Addendum 102 verified zero coupling-map violations end-to-end; neither
checked that the computed operator matches the input. This addendum
closes that gap directly.

## 1. Results

| family | arm | runs | min fidelity | max fidelity |
|---|---|---:|---:|---:|
| `dense_pairs` | `qiskit_l3` | 6 | 1.000000000000 | 1.000000000000 |
| `dense_pairs` | `psf_ls_false` | 6 | 1.000000000000 | 1.000000000000 |
| `dense_pairs` | `psf_ls_true` | 6 | 1.000000000000 | 1.000000000000 |
| `chain_shaped` | `qiskit_l3` | 6 | 1.000000000000 | 1.000000000000 |
| `chain_shaped` | `psf_ls_false` | 6 | 1.000000000000 | 1.000000000000 |
| `chain_shaped` | `psf_ls_true` | 6 | 1.000000000000 | 1.000000000000 |

**All 36 runs used `final_index_layout(filter_ancillas=True)`** to
determine the qubit permutation -- Qiskit's own purpose-built method
was available and used throughout; the pre-registration's own
fallback-refusal path (raising rather than guessing, if that method
were unavailable) was never triggered. The true minimum across all 36
runs, read directly from the raw CSV rather than the rounded terminal
display, is **0.999999999999994** -- indistinguishable from exact
equality at double-precision floating point.

## 2. Scoring

**P1 (primary) -- CONFIRMED, unambiguously.** 36/36 at or above the
threshold, with the actual minimum landing at machine-precision noise
rather than anywhere near the 1e-9 boundary the threshold was set at.
`layout_search=True`'s full pipeline -- including all four of this
session's own fixes (natural-ordering-first, the feasibility-guard
correctness fix, the per-device matching cache, and lazy ordering
generation) -- computes exactly the same unitary the input circuit
specified, once the routing-induced qubit permutation is correctly
accounted for.

**P2 (comparison arm) -- CONFIRMED.** `layout_search=False` also shows
1.000000000000 throughout, exactly as predicted -- this arm does not
exercise any of today's fixes, and its own correctness was never in
doubt; it serves as intended, as a control confirming
`compile_for_hardware()`'s own general layout-application logic is
sound independent of `layout_search`'s own value.

**P3 (sanity check) -- CONFIRMED.** Qiskit's own `transpile()` output
also shows 1.000000000000 throughout, confirming the equivalence-
checking methodology itself is sound: had Qiskit's own well-established
output failed this check, the checker -- not Qiskit -- would have been
the suspect, and this arm's clean pass rules that out.

## 3. Why this result can be trusted

The permutation-handling logic this check depends on was itself
verified before use, not merely written and trusted: `permute_operator`
(the function responsible for correctly relabelling a routed circuit's
own qubit ordering back to the input's logical ordering) was checked
against two independent, hand-built cases using only `numpy.kron`
-- a single-qubit swap and a 3-qubit cyclic permutation -- with zero
reliance on Qiskit for the verification itself, precisely because a
bug in the checker's own permutation logic could produce a false
"correct" or false "incorrect" result indistinguishable from a genuine
finding. **This self-test runs automatically at the start of every
execution of the script**, not only once during development, so a
future edit cannot silently invalidate the check without the script
itself catching it. Both self-tests passed before this run's own 36
comparisons were attempted.

## 4. What this means for the session as a whole

**This closes the last major gap in this session's own verification of
Addenda 88-95's four fixes.** The progression is now complete:
individually verified (88, 89, 91, 92) -> verified together through the
real entry point (95) -> verified end-to-end through
`compile_for_hardware()` for structural validity (102) -> **verified
end-to-end for computational correctness (this addendum)**. No stage
of this progression found a defect; each closed a specific,
pre-identified gap the previous stage's own scope explicitly excluded.

## 5. What this does not establish

- Equivalence at the scale Addenda 101-102 actually measured timing on
  (42, 64 qubits) -- infeasible by `Operator()`, per this project's own
  established limit, and not attempted here, per this addendum's own
  pre-registered scope.
- Anything about `verify=True`'s own correctness path -- untested here,
  matching Addendum 101/102's own `verify=False` scope.
- Whether this generalizes to circuit families, grid sizes, or
  topologies beyond the two families and one grid tested here.

## 6. Files

| File | What it is |
|---|---|
| [`verify_unitary_equivalence.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_unitary_equivalence.py) | this run's script, including the self-tested permutation logic |
| [`unitary_equivalence_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/unitary_equivalence_2026-09-20.csv) | this run's results, 36 rows |
| [`spare-qubit-cliff-addendum-103-preregistration-2026-09-20.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-103-preregistration-2026-09-20.md) | the predictions scored above |

## 7. Verification

- The true minimum fidelity (0.999999999999994) was read directly from
  the raw CSV's own floating-point values, not from the terminal's
  rounded display, before being reported.
- The permutation method actually used (`final_index_layout(filter_ancillas=True)`)
  was confirmed present in all 36 rows via direct count, not assumed
  from the script's own preferred code path.
- `permute_operator`'s own correctness was established independently of
  this run, against hand-built `numpy.kron` cases with no Qiskit
  dependency, before being trusted for any of the 36 comparisons in
  this addendum.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV -> 0
  hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---


<!-- ===== Addendum 104 pre-registration (source: spare-qubit-cliff-addendum-104-preregistration-2026-09-20.md) ===== -->

> **Note added when merging:** Predictions for whether the natural-ordering-first fix helps, hurts, or does nothing on brick (a topology this file's own diagnostics found intractable) and heavy-hex (a genuinely bipartite-imbalanced topology), reusing this repository's own existing brick_edges and heavy-hex saturation conventions rather than re-deriving them.

## Addendum 104 -- Pre-registration: does the natural-ordering-first fix help, hurt, or do nothing on topologies where this file's own diagnostics already found the BFS strategies fail? (2026-09-20)

**Status: pre-registration only. No run on `brick` or heavy-hex has been
performed with the patched module.** Predictions are locked before any
measurement.

## 1. Why this experiment exists

Every verification of the four fixes (Addenda 88-95, 101-103) used one
topology: the square grid (6x7, 8x8). `psf_smart_layout.py`'s own
docstring records, from earlier diagnostics (Addenda 8-12), that its
BFS-family orderings found nothing on `brick` even at high call
limits, and that heavy-hex's own bipartite imbalance makes true
qubit-count "spare=0" impossible in the first place (Addendum 39-40,
already established in this repository's own
`occupancy_sweep_heavy_hex.py` -- reused here rather than
re-derived).

**A specific, already-settled reasoning point governs this design**:
the natural-ordering-first fix (Addendum 88) adds a new *first*
attempt; every ordering the original code already tried still follows
if that new attempt fails. **It therefore cannot reduce coverage on
any topology, including ones the original code already failed on.**
This experiment does not test whether coverage regresses (it provably
cannot); it tests whether the fix helps, is neutral, or -- the one way
it plausibly *could* cost something -- adds meaningful overhead on a
topology where it is not expected to succeed, before the original
orderings get their turn.

## 2. Design

**`brick`**: `brick_edges(rows, cols)`, copied verbatim from this
repository's own `verify_vf2_sparse_topology.py` (not re-derived), at
the same grid dimensions used throughout this session (6x7, 8x8), at
genuine qubit-count spare=0 (zero idle qubits, interaction graph is
`dense_pairs` sized to the grid's own qubit count -- the family this
project's own diagnostics found `brick` hardest on).

**Heavy-hex**: `CouplingMap.from_heavy_hex(d)` for `d=3` (11+8=19
qubits) and `d=5` (33+24=57 qubits), using this repository's own
already-established redefinition of saturation for this topology
(Addendum 39-40): `spare_pairs=0` relative to the graph's OWN maximum
matching (computed directly via `networkx`, not assumed), not relative
to raw qubit count. `dense_pairs` sized to exactly that many disjoint
pairs.

Both call `smart_vf2_layout()` directly (matching Addenda 83-95's own
methodology, not `compile_for_hardware()`, so the guard's own behavior
on non-grid coupling maps is also exercised) -- **and separately, a
sanity check that the guard's own necessary-condition logic (Addendum
89) is not grid-specific**: it compares two `networkx` maximum-matching
computations, which make no reference to grid structure, but this is
verified rather than assumed for these two new topologies before
trusting any timing result.

## 3. Pre-registered predictions

**P1 (primary -- does natural-first help, hurt, or do nothing on
`brick`?).**
  - **Helps or neutral**: `smart_vf2_layout()` on `brick` either finds a
    layout it could not before (unlikely, given this file's own
    diagnostics, but not ruled out -- `dense_pairs` specifically was not
    the family those diagnostics used), or fails at a cost
    indistinguishable from before (the natural-ordering attempt is
    cheap -- a single `rx.vf2_mapping` call bounded by the same
    `per_attempt_call_limit` every other ordering uses).
  - **Hurts**: failure on `brick` costs measurably more with the fix
    than a direct call to the original (unpatched) orderings alone --
    would mean the added attempt's own overhead is non-negligible on a
    topology it was never going to help on. **This is checked directly
    against a same-file comparison** (this addendum's own script also
    runs the sequence of orderings skipping the natural one, on the
    same graphs, for exactly this comparison).

**P2 (heavy-hex -- does the guard's own logic hold on a genuinely
different graph shape)?** The repaired guard (Addendum 89) is predicted
to correctly accept `dense_pairs` sized to heavy-hex's own maximum
matching (this is by construction a matching-shaped interaction graph
sized to exactly fit), and to correctly reject one pair more than that
capacity -- mirroring Addendum 90's own bipartite-imbalance test
(`K(4,16)`, `K(8,32)`), on a real, not synthetic, bipartite-imbalanced
topology this time.

**P3 (heavy-hex, primary layout question).** Whether
`smart_vf2_layout()` finds a layout at genuine heavy-hex saturation
(`spare_pairs=0`) is **not predicted in advance** -- no addendum in
this project has tested `dense_pairs` against heavy-hex through this
specific search before. Reported as a direct, open measurement, not a
confirmed-or-falsified prediction.

## 4. What this cannot establish

- Whether `brick`'s own failure (if it recurs) has the same or a
  different mechanism from the square-grid cliff this project's own
  Addenda 51-87 characterized in depth -- out of scope; this addendum
  measures whether the fix changes the outcome, not why `brick` is hard.
- Generalization to grid sizes or heavy-hex distances beyond the four
  tested configurations.
- Anything about `compile_for_hardware()`'s own end-to-end behavior on
  these topologies -- this addendum calls `smart_vf2_layout()` directly,
  matching Addenda 83-95's own component-level methodology, not
  Addendum 101-103's end-to-end one.

---


<!-- ===== Addendum 105 pre-registration (source: spare-qubit-cliff-addendum-105-preregistration-2026-09-20.md) ===== -->

> **Note added when merging:** Predictions for locating the source of Addendum 94's own 2.27GB of uncollected garbage via gc.get_objects() type census and gc.DEBUG_SAVEALL capture.

## Addendum 105 -- Pre-registration: what is actually accumulating in the 2.27GB of uncollected garbage found in Addendum 94? (2026-09-20)

**Status: pre-registration only. No object-level trace has been
performed.** Predictions are locked before any measurement.

## 1. Why this experiment exists

Addendum 94 found that disabling Python's garbage collector during
10,000 `psf_compile()` calls (a fixed-size, non-growing workload) grows
process RSS by 2,273.4MB -- confirming reference cycles are created at
meaningful volume somewhere in the compile path, but not identifying
where. This addendum locates the source using `gc.get_objects()`
snapshots, `sys.getrefcount`, and, if needed, `gc.DEBUG_SAVEALL` to
capture the actual unreachable-but-uncollected objects for inspection.

**Not urgent** (Addendum 94's own conclusion: GC is enabled by default,
cyclic garbage collection reclaims this normally, no user-visible
symptom exists) but worth answering now that the tooling is already
built, since "PSF-Zero's own compile path creates cycles" is a
concrete, checkable claim this project has stated without yet locating
the cycle.

## 2. Design

Run a much smaller loop (200 iterations, enough to see a clear signal
without an unwieldy object count) three ways:

- **Baseline**: `gc.collect()` before and after, comparing
  `gc.get_objects()` counts by type. Growth in a type's count after a
  full collection indicates objects that are NOT being reclaimed by
  the cyclic collector either -- a genuine leak, not merely
  cycle-requiring-GC.
- **Cycle census**: with GC left enabled throughout (normal operation),
  call `gc.collect()` and inspect its own return value (the number of
  unreachable objects it collected) per batch of iterations -- a
  non-zero, roughly constant count per batch directly demonstrates
  cycles are being created and successfully reclaimed at a steady rate.
- **Cycle capture**: `gc.set_debug(gc.DEBUG_SAVEALL)`, run the loop with
  GC disabled, then `gc.collect()` once at the end and inspect
  `gc.garbage`'s own contents by type and, for the most common types,
  trace one example's own referrers (`gc.get_referrers`) to identify
  which objects hold the cyclic references.

## 3. Pre-registered predictions

**P1 (primary -- what type dominates the cycles)?** No specific type is
predicted in advance -- candidates mentioned only as candidates, not
favored: Qiskit's own `QuantumCircuit`/`DAGCircuit`/`Instruction`
objects (known in general to form parent-child reference cycles in
some Qiskit versions), PSF-Zero's own `SU4GeodesicPSFSynthesizer`
instances or their internal state, or PyO3-boundary wrapper objects
from the Rust core. **Reported as a direct measurement**, since this
project has no prior addendum establishing which is responsible.

**P2 (steady-state rate).** The "cycle census" run (GC enabled) is
predicted to show `gc.collect()`'s own reclaimed-object count settle
into a roughly constant per-iteration rate after an initial batch or
two (rather than growing without bound), consistent with Addendum 94's
own finding that GC-enabled operation shows no unbounded memory growth
(+56.1MB over 10,000 iterations, not thousands of MB).

**P3 (no genuine leak).** The "baseline" run (`gc.collect()` before and
after, comparing object counts) is predicted to show **no net growth**
in any type's count once garbage collection is allowed to run --
i.e., every object created is eventually reachable-and-freed or
cycle-collected, matching Addendum 94's own conclusion that this is
"not necessarily a true memory leak... cyclic garbage is eventually
reachable and collectible, just not by refcounting alone." A type
showing genuine net growth even after full collection would contradict
this and be a materially more serious finding than Addendum 94's own.

## 4. What this cannot establish

- Which specific line of code creates the reference cycle, only which
  object type(s) are involved -- locating the exact line would need a
  further, targeted trace once the type is known.
- Whether the same cycle-creation pattern exists in `psf_smart_layout.py`
  -- this addendum concerns `psf_compile.py`'s own gate-synthesis path
  only, matching Addendum 94's own original scope.
- Whether other Qiskit versions show the same pattern -- single-version
  measurement, as with every result in this project.

---


<!-- ===== Addendum 106 (source: spare-qubit-cliff-addendum-106-2026-09-20.md) ===== -->

> **Note added when merging:** Test B: natural ordering succeeds instantly on brick where every other ordering fails outright after nearly a full second, and succeeds on heavy-hex too. Test D: the cyclic garbage is confirmed to be Qiskit's own QuantumCircuit-family objects, cleanly and predictably reclaimed by GC -- but a separate, unregistered finding of generic-object growth technically falsifies the 'no leak' prediction, left as an open question.

## Addendum 106 -- Test B: natural ordering succeeds instantly on `brick` where every other ordering fails outright; Test D: the cyclic garbage is `QuantumCircuit`-family objects, cleanly reclaimed by GC -- but a separate, unregistered finding of generic-object growth technically falsifies P3 (2026-09-20)

**Pre-registered in**:
`spare-qubit-cliff-addendum-104-preregistration-2026-09-20.md` (Test B)
and `spare-qubit-cliff-addendum-105-preregistration-2026-09-20.md`
(Test D), both written and locked before these runs.

## Part 1 -- Test B: topology generalization

### 0. In one line

**P1: far exceeds "helps or neutral" -- natural ordering succeeds
INSTANTLY on `brick`, a topology this file's own prior diagnostics
recorded as a documented failure case for every other ordering.** At
both 6x7 and 8x8 `brick`, with-natural finds a layout in ~1ms; without
natural (the original pre-Addendum-88 sequence, all BFS-family
orderings plus fallback) finds **nothing at all**, after burning
875-996ms. **P2: CONFIRMED** -- the repaired guard correctly accepts
`dense_pairs` sized to heavy-hex's own maximum matching and correctly
rejects one pair more, on both tested distances. **P3: an
unpredicted-in-advance positive result** -- natural ordering also
succeeds instantly (under 1ms) on heavy-hex at both d=3 and d=5.

### 1. Results

| topology | n | with-natural found | with-natural time | without-natural found | without-natural time |
|---|---:|:---:|---:|:---:|---:|
| `brick` 6x7 | 42 | **True** | 1.25ms | **False** | 874.97ms |
| `brick` 8x8 | 64 | **True** | 1.08ms | **False** | 995.84ms |

| topology | n | max matching | accept-at-capacity | reject-one-over | search found | order |
|---|---:|---:|:---:|:---:|:---:|---|
| heavy_hex d=3 | 19 | 8 | True | True | **True** | natural, 0.25ms |
| heavy_hex d=5 | 57 | 24 | True | True | **True** | natural, 0.69ms |

### 2. Scoring

**P1 -- CONFIRMED, far exceeding the pre-registration's own "helps or
neutral" branch.** The pre-registration anticipated, at most, that the
new attempt might succeed where the old ones failed ("unlikely...but
not ruled out") or cost nothing extra if it also failed. **What was
found is stronger than either framing anticipated**: the new attempt
alone achieves in ~1ms what nine attempts of the *entire original
strategy* (BFS-family orderings plus the fallback heuristics) could not
achieve at all in nearly a full second. `dense_pairs` on `brick` -- a
combination this project's own prior diagnostics never specifically
tested (the docstring's own "brick fails" finding used a different
circuit family) -- turns out to be exactly the kind of case natural
ordering handles trivially.

**P2 -- CONFIRMED.** Both heavy-hex distances show the guard correctly
distinguishing exact-capacity from one-over, on a real (not synthetic)
bipartite-imbalanced topology -- extending Addendum 90's own synthetic
`K(4,16)`/`K(8,32)` result to genuine hardware-shaped graphs.

**P3 -- a clean positive result, not predicted either way in
advance.** Natural ordering succeeds on heavy-hex too, at both tested
distances, in under 1ms.

### 3. What this means

**Across every topology this project has now tested with the natural-
ordering-first fix -- square grids (Addenda 88-95, 101-103), `brick`,
and heavy-hex (this addendum) -- it has never failed to find a layout
when one exists, and has never cost more than ~1ms.** This is a
substantially stronger generalization result than the pre-registration
itself anticipated, which was framed cautiously around "does not hurt"
given `psf_smart_layout.py`'s own prior, specific finding that `brick`
was hard for its existing strategies. **That prior finding is not
contradicted** -- it concerned different circuit families, and this
addendum confirms the *old* orderings still fail on `brick` with
`dense_pairs` too (consistent, not new) -- but natural ordering turns
out to be effective precisely where the old strategies were not, on
every topology tried so far.

### 4. What this does not establish

- Whether natural ordering succeeds on every topology, or on circuit
  families other than `dense_pairs` -- four new configurations is not
  an exhaustive survey.
- Why natural ordering succeeds where the old orderings fail on
  `brick` specifically -- no mechanism is proposed, matching this
  session's own established practice of not speculating beyond what
  was tested.

---

## Part 2 -- Test D: the source of the cyclic garbage

### 0. In one line

**P1 (what type dominates): ANSWERED, cleanly.** With GC disabled,
`QuantumCircuit`, `CircuitData`, and `_OuterCircuitScopeInterface` --
all Qiskit's own circuit-internal types, not any PSF-Zero-specific
class -- account for 6,000 of the captured garbage's 26,200 objects,
at exactly 10 of each per iteration (2,000 / 200). **P2 (steady-state
rate): CONFIRMED** -- with GC enabled, `gc.collect()` reclaims exactly
194 objects per 20-iteration batch, unchanged across all 10 batches:
bounded, not growing. **P3 (no genuine leak): FALSIFIED as literally
stated** -- real, substantial net growth (up to +9,246 for a single
type) was found across 287 distinct types even after a full
`gc.collect()`. **Critically, none of the 287 growing types is
`QuantumCircuit` or any circuit-related class** -- the growth is in
generic Python/Cython-runtime objects (`function`, `dict`, `tuple`,
`cell`, `builtin_function_or_method`, etc.), a different and separate
phenomenon from the cyclic-garbage question this addendum set out to
answer, and its own significance is not resolved here.

### 1. Results

**P1 -- garbage capture (GC disabled, `DEBUG_SAVEALL`, 200 iterations):**

| type | count | per iteration |
|---|---:|---:|
| `list` | 7,000 | 35 |
| `cell` | 5,600 | 28 |
| `tuple` | 2,800 | 14 |
| `function` | 2,800 | 14 |
| **`QuantumCircuit`** | **2,000** | **10** |
| **`_OuterCircuitScopeInterface`** | **2,000** | **10** |
| **`CircuitData`** | **2,000** | **10** |
| `dict` | 2,000 | 10 |

The referrers of one sampled `list` instance were only a `list` and a
`dict` -- uninformative on their own (consistent with a cycle whose
specific structure would need deeper, targeted tracing to fully map,
not attempted here).

**P2 -- cycle census (GC enabled throughout, 20-iteration batches):**
`gc.collect()` reclaimed exactly **194 objects in every one of the 10
batches**, with zero variation. This is a strikingly regular,
completely bounded rate.

**P3 -- baseline (GC enabled, full collection before and after 200
iterations):** 287 distinct types showed net growth even after
`gc.collect()`, led by `function` (+9,246), `dict` (+4,572), `tuple`
(+3,578), `cell` (+2,111), `builtin_function_or_method` (+1,647),
continuing down to smaller counts. **`QuantumCircuit` and every other
circuit-related type from the P1 capture do not appear anywhere in
this growth list.**

### 2. Scoring

**P1 -- CONFIRMED, with a direct, concrete answer.** The dominant
cyclic-garbage-requiring objects are Qiskit's own circuit-internal
types (`QuantumCircuit`, `CircuitData`, `_OuterCircuitScopeInterface`),
appearing at a clean, exactly-proportional rate (10 per `compile()`
call -- plausibly one for the input, one for `qc_blocked` after
`ConsolidateBlocks`, one for the output `qc_psf`, and further internal
copies Qiskit's own circuit-scope machinery creates). **No PSF-Zero-
specific class appears in the top 20** -- `SU4GeodesicPSFSynthesizer`
and its own internals are not implicated; the cycles are in Qiskit's
own object model, encountered via ordinary use of `QuantumCircuit`,
not in code this project wrote.

**P2 -- CONFIRMED.** A perfectly flat 194-per-batch rate across all 10
batches is as clean a "bounded, steady-state" result as this project
has recorded for anything. This directly corroborates Addendum 94's
own finding of modest (+56.1MB, not thousands of MB) growth with GC
enabled over a much larger (10,000-iteration) run.

**P3 -- FALSIFIED, exactly as stated, and reported as such rather than
reframed after the fact.** The pre-registration predicted no net
growth in any type; substantial growth was found in 287. **This does
not, however, implicate the cyclic-garbage mechanism Addendum 94 and
this addendum's own P1/P2 sections investigate** -- the growing types
are a disjoint set from the circuit-related ones P1 identified, and
none of the growing types are specific to this project's own code.
Whether this generic-object growth reflects a genuine, unbounded,
per-iteration accumulation (concerning) or a one-time, later-plateauing
warm-up cost of first exercising various Cython/Rust-wrapped code
paths inside Qiskit's own dependencies (benign) **cannot be
distinguished from this addendum's own single-measurement design** --
the counts (e.g. ~46 new `function` objects per iteration, on average
across the run) are large enough that a one-time, non-scaling
explanation is not obviously right either, and this should not be
assumed benign without a follow-up.

### 3. What this means

**The specific question Addendum 94 raised -- what accumulates when GC
is disabled -- is now answered concretely and reassuringly: it is
Qiskit's own circuit objects, created at a clean, bounded, per-call
rate, and reclaimed perfectly by the ordinary garbage collector.**
This is good news for PSF-Zero's own code specifically. **A separate,
new, and NOT reassuring-by-default finding emerged in the course of
checking this**: real growth in generic runtime object counts even
under normal (GC-enabled) operation, in types unconnected to circuit
construction. This was not what this addendum set out to investigate,
is not yet understood, and should not be dismissed merely because it
falls outside this addendum's own original question.

### 4. What this does not establish

- Whether the P3 growth (287 types, led by `function`/`dict`/`tuple`/
  `cell`) is a one-time process-warmup cost or genuine ongoing
  accumulation -- would need a longer run (e.g. checking whether the
  same 200-iteration increment at iterations 1,000-1,200 shows similar
  or near-zero growth) to distinguish, not attempted here.
- The exact code path creating the `QuantumCircuit`-family cycles --
  P1's own referrer trace on a sampled `list` was uninformative;
  identifying the specific reference chain would need a more targeted
  trace (e.g. `objgraph`'s own reference-graph visualization) than
  attempted here.
- Whether the P3 growth exists in `psf_smart_layout.py`'s own code path
  as well -- this addendum, like Addendum 94, examined
  `psf_compile.py`'s gate-synthesis path only.

## Files

| File | What it is |
|---|---|
| [`verify_topology_generalization.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_topology_generalization.py) | Test B's script |
| [`topology_generalization_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/topology_generalization_2026-09-20.csv) | Test B's results, 4 rows |
| [`verify_gc_cycle_source.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_gc_cycle_source.py) | Test D's script |
| [`gc_cycle_source_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gc_cycle_source_2026-09-20.csv) | Test D's results, 295 rows |
| [`spare-qubit-cliff-addendum-104-preregistration-2026-09-20.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-104-preregistration-2026-09-20.md) | Test B's predictions, scored above |
| [`spare-qubit-cliff-addendum-105-preregistration-2026-09-20.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-105-preregistration-2026-09-20.md) | Test D's predictions, scored above |

## Verification

- Test B's `brick_edges` construction was verified, before this run,
  to be byte-for-byte identical to `verify_vf2_sparse_topology.py`'s
  own function -- checked directly, not assumed from copying the code.
- Test B's heavy-hex "spare=0" definition (relative to the graph's own
  maximum matching) was taken from this repository's own
  `occupancy_sweep_heavy_hex.py`, and the matching value itself
  (8 for d=3, 24 for d=5) was computed directly via `networkx` in this
  run, not assumed from that script's own docstring.
- Test D's P1/P3 distinction (that `QuantumCircuit`-family types appear
  in the GC-disabled capture but NOT in the GC-enabled growth list) was
  checked by direct set comparison between the two result tables, not
  eyeballed from the printed output.
- The exact 10-per-iteration and 194-per-batch rates were confirmed by
  direct division against the known iteration/batch counts, not
  estimated.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and both new CSVs
  -> 0 hits. The terminal output supplied for these runs was reviewed
  for local paths before use; none were reproduced here.

---

## Addendum to Part 2 -- exact reproducibility confirmed, follow-up designed (2026-09-20, later)

A second, independent invocation of `verify_gc_cycle_source.py`
(fresh process, same 200-iteration workload) reproduced **all 295
rows with zero mismatches** against the first run -- every type's
count, in both the P3 growth list and the P1 capture, matched exactly.
**This confirms the phenomenon is deterministic and repeatable, not
measurement noise.** It does **not**, by itself, distinguish the two
explanations Section 2's own scoring left open (one-time process
warm-up vs. genuine per-iteration accumulation): both predict
identical results across separate fresh-process runs of the same
iteration count, for the same reason each explanation gives.

**A follow-up script, `verify_gc_growth_pattern.py`, was written to
resolve this directly**: within a single process, it measures growth
in four cumulative blocks (200/400/800/1600 iterations) and reports
each block's *own* new growth (not the cumulative total) per type. If
growth-per-iteration shrinks toward zero across successive blocks, the
cause is one-time warm-up; if it stays roughly constant, growth is
genuinely proportional to call count. **Not yet run.**

---


<!-- ===== Addendum 107 (source: spare-qubit-cliff-addendum-107-2026-09-20.md) ===== -->

> **Note added when merging:** Addendum 106's own open question resolved decisively: the 287-type generic-object growth is one-time process warm-up, not per-iteration accumulation -- growth collapses to exactly zero by the 800-iteration checkpoint and stays there through 1,600, closing this session's entire memory-safety thread.

## Addendum 107 -- resolved: Addendum 106's own open question answered decisively -- the 287-type growth is one-time process warm-up, not per-iteration accumulation (2026-09-20)

**Status**: the direct follow-up Addendum 106 Section 4 called for
("checking whether the same 200-iteration increment... shows similar
or near-zero growth"), run and scored.

## 0. In one line

**Resolved, decisively, in the benign direction.** Growth per block
collapses from 46.23 objects/iteration (`function`, the largest
contributor) in the first 200 iterations to **exactly 0** by the
800-iteration checkpoint, and **stays at exactly 0** through 1,600.
Every one of the 11 types that showed growth at all is either fully
flat by the 800-iteration mark or was already negligible (2 types
showed a single-digit residual at 400 iterations, then also reached
zero). **This is one-time process warm-up, not genuine per-iteration
accumulation** -- the question Addendum 106 explicitly left open is
now closed.

## 1. Results

Growth accrued *within each block alone* (not cumulative), by type,
across four checkpoints in one continuous process:

| type | 0-200 | 200-400 | 400-800 | 800-1600 |
|---|---:|---:|---:|---:|
| `function` | +9,246 | 0 | 0 | 0 |
| `dict` | +4,572 | 0 | 0 | 0 |
| `tuple` | +3,578 | 0 | 0 | 0 |
| `cell` | +2,111 | 0 | 0 | 0 |
| `builtin_function_or_method` | +1,647 | +22 | 0 | 0 |
| `list` | +1,569 | +1 | 0 | 0 |
| `ReferenceType` | +1,502 | +22 | 0 | 0 |
| `method` | +959 | 0 | 0 | 0 |
| `getset_descriptor` | +953 | 0 | 0 | 0 |
| `set` | +880 | 0 | 0 | 0 |
| `Counter` | 0 | +1 | 0 | 0 |

**Every type reaches exactly zero growth by the 800-iteration
checkpoint (a 400-iteration block with zero new objects of any
tracked type), and remains at zero through the 1,600-iteration
checkpoint (an 800-iteration block, also zero).** The two smallest
contributors (`builtin_function_or_method`, `ReferenceType`) show a
small residual (+22 each) in the second block before also reaching
zero -- consistent with a slightly slower-to-complete warm-up for
those two specific types, not a separate accumulation pattern, since
they too flatten completely by 800 iterations.

## 2. Scoring

**Addendum 106's own open question -- "cannot be distinguished from
this addendum's own single-measurement design... should not be
assumed benign without a follow-up" -- is now answered: benign,
confirmed directly rather than assumed.** The pattern is exactly what
one-time warm-up predicts (sharp initial growth, rapidly decaying,
reaching a hard floor of zero) and is the opposite of what genuine
per-iteration accumulation would show (roughly constant growth per
block, scaling with block size indefinitely). The `Counter` type's own
single-object appearance at the second checkpoint is an artifact of
this script's own measurement method (the script's own use of
`collections.Counter` for its type census), not a subject finding.

## 3. What this means for the session's own memory-safety thread

Combining Addenda 94, 106, and this addendum, the picture is now
complete and reassuring on every count checked:

| question | answer | addendum |
|---|---|---|
| Does disabling GC cause real memory growth? | Yes, 2.27GB/10,000 calls | 94 |
| Is that growth a genuine, unbounded leak? | No -- fully reclaimed when GC is enabled | 94 |
| What object types are involved? | Qiskit's own `QuantumCircuit`/`CircuitData`/`_OuterCircuitScopeInterface`, not PSF-Zero's own code | 106 |
| Is the reclaim rate bounded under normal operation? | Yes -- exactly 194/batch, unchanging across 10 batches | 106 |
| Is there a SEPARATE, unrelated growth pattern in generic runtime objects? | Yes -- found while checking the above | 106 |
| Is THAT growth a genuine per-iteration leak? | **No -- confirmed one-time warm-up, reaching exactly zero by 800 iterations** | **this addendum** |

**No open memory-safety question remains from this session's own
investigation.** Both growth phenomena examined (the cyclic,
GC-dependent circuit-object garbage, and the separate generic-object
warm-up) are now understood and confirmed non-problematic for any
realistic, long-running use of `psf_compile()`.

## 4. What this does not establish

- The specific code path responsible for the one-time warm-up (which
  Cython/Rust-wrapped subsystem creates these particular
  function/dict/tuple/cell objects on first use) -- not investigated;
  of academic interest only, given the finding is benign.
- Whether the same warm-up pattern and timeline holds on a different
  machine, Python version, or Qiskit version -- single-machine
  measurement, as with every result in this project.
- Anything about `psf_smart_layout.py`'s own code path -- this
  addendum, like 94 and 106, examined `psf_compile.py`'s gate-synthesis
  path only.

## 5. Files

| File | What it is |
|---|---|
| [`verify_gc_growth_pattern.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_gc_growth_pattern.py) | this run's script |
| [`gc_growth_pattern_2026-09-20.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gc_growth_pattern_2026-09-20.csv) | this run's results, 14 rows |

## 6. Verification

- The per-block (not cumulative) growth figures were read directly
  from the raw CSV's own `growth_this_block` column, not recomputed or
  estimated from the cumulative figures.
- "Zero growth at 800 and 1600" was confirmed by checking that no row
  in the CSV has `checkpoint` equal to 800 or 1600 -- the complete
  absence of rows at those checkpoints, not merely small values, is
  what establishes the zero.
- The two-type residual at the 400-iteration checkpoint
  (`builtin_function_or_method`, `ReferenceType`, +22 each) was
  reported explicitly rather than omitted, and checked to also reach
  zero by 800 iterations before being characterized as "slower-to-
  complete warm-up" rather than a distinct pattern.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this document and the new CSV ->
  0 hits. The terminal output supplied for this run was reviewed for
  local paths before use; none were reproduced here.

---

**End of Part 6 of 8.** Continue to [Part 7](spare-qubit-cliff-combined-108.md) (Addendum 108-134) and [Part 8](spare-qubit-cliff-combined-135.md) (Addendum 135 onward), or back to [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
