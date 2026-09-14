# spare-qubit-cliff: Combined Addenda (Addendum 2026-09-13 through Addendum 16)

**This is a merge of 14 separately-written addenda into one chronological
document, for convenience.** No wording in any individual addendum has been
changed -- each section's content is unedited. Two mechanical things were
done to make this readable as one document: (1) each addendum's own
top-level heading was normalized to the same heading level (H2), since the
originals ranged from H1 to H3 depending on which draft style was used when
it was written; (2) a one-line navigation note (in a blockquote) was added
above some headings to point forward to where a later addendum revises or
supersedes an earlier claim. These notes are new, are clearly marked, and
are not part of the original text. Nothing below either note was deleted or
rewritten.

Consistent with this project's standing rule ("corrections are appended
below the original, never overwriting it"), **nothing here has been deleted
or rewritten.** Where a later addendum changes the picture, the earlier
addendum's claim is left exactly as originally written, with a forward
pointer added above it.

## Reading guide -- if you only have five minutes

**What is now established, across the whole series:**
- The cliff's mechanism (Addendum 9, confirmed on real hardware in
  Addendum 10): Qiskit's preset pipeline tries `VF2Layout` once; if it
  fails, it falls back to `SabreLayout`. The "cliff" is this fallback and
  the extra work it causes, not merely "VF2 search taking a long time" in
  isolation.
- VF2's ordering dependence is real and reproducible inside Qiskit itself
  (Addendum 4), not only in the separate `rustworkx` package (the original
  report's mistake, corrected in the top addendum).
- A prototype layout-search tool (Addendum 13) can win by 15-420x when it
  finds a layout Qiskit's default search misses (Addendum 14 sandbox,
  Addendum 15/16 real hardware) -- but loses by roughly the time it spent
  searching when it does not (Addendum 15), and the margin of loss becomes
  **unmeasurable amid ~3x run-to-run variance at L3** (Addendum 16).
- Output circuit quality (gate count, depth) is identical whichever path is
  taken -- the cliff is a time cost, not a quality cost (Addendum 14/15).

**What is retracted or corrected along the way (kept, not deleted):**
- The original root-cause attribution to `rustworkx.vf2_mapping()` (top
  addendum -- Qiskit does not call that function).
- Addendum 8's framing of the trial-loop mechanism as the whole story
  (superseded by Addendum 9's fallback-to-Sabre finding).
- Addendum 12's initial read of "transpose symmetry" as a grid property
  (corrected, in the same addendum, to a necessity of the experimental
  design).

**What is still open at the end of Addendum 16:**
- Whether the ~3x variance found in Addendum 16 comes from `VF2Layout`'s
  `seed=-1` shuffle or from execution-environment drift -- an experiment
  with pre-registered predictions (P9-P11) is queued but not yet run.
- Whether the ordering effects found via the public `rustworkx` API
  (Addenda 4, 9, 12, 13) hold inside Qiskit's actual compiled implementation
  (`qiskit._accelerate.vf2_layout`) -- never directly tested throughout the
  whole series.
- The end-to-end PSF-Zero comparison this series was meant to answer is
  still blocked on `compile_for_hardware` gaining an `initial_layout`
  parameter (Addendum 15, section 4).

---



<!-- ===== Addendum (2026-09-13) (source: spare-qubit-cliff-addendum-2026-09-13.md) ===== -->

> **Note added when merging:** First addendum. Retracts the original upstream-report attribution; records that Qiskit's own VF2 implementation also uses VF2++ ordering, discovered by reading source.

## Addendum (2026-09-13): the objection traces to a specific, year-old commit

The claim rejected above — that `VF2Layout`/`VF2PostLayout` call
`rustworkx.vf2_mapping` — was true of an *older* Qiskit. Commit
[6421d77](https://github.com/Qiskit/qiskit/commit/6421d77) ("Handle VF2
coupling-map shuffling in Rust", [#14860](https://github.com/Qiskit/qiskit/pull/14860)),
authored by Jake Lishman and merged **2025-09-19** — a year before this project's
report — removed that call entirely. Before it, `qiskit/transpiler/passes/layout/vf2_layout.py`
branched on `self.seed`: `seed == -1` took a Rust fast path with no shuffling;
anything else (including the default `seed=None`) fell through to a pure-Python
path that built the interaction and coupling graphs, shuffled the coupling graph
in Python, and called `rustworkx.vf2_mapping` directly — 158 lines of scoring and
trial-loop logic that the PR deleted. After it, every path (shuffled or not) goes
through the single Rust function `vf2_layout_pass`, which now takes a
`shuffle_seed: Option<u64>` argument and does the reordering itself via
`vf2::reorder_nodes` — the same mechanism this project's [`verify_vf2_seed.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_seed.py)
exercises through the Python `seed=` parameter.

**So the rejection was correct about the code as it stood, by a year.** The report,
filed 2026-09-11, described a call path Lishman himself had deleted on 2025-09-19.
Whatever source informed the original claim — documentation, an older reading of the
code, or an unverified assumption — it predated this change and was not checked
against current `main` before posting. This is the same failure this document
already names elsewhere: proposing a mechanism without reading the implementation
it concerns.

**Two things in the same commit corroborate, rather than undercut, the ordering
investigation this document is built on.**

First, the commit message states plainly: *"The shuffling is, in general, not a
good idea."* Lishman's own assessment, a year before this project measured it,
matches what [`verify_vf2_max_trials.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_max_trials.py) found directly: a shuffled ordering that
finds a layout does not make the pass faster, because `minimize_vf2`'s trial loop
keeps searching afterward regardless of when the first match arrived. His caution
about the mechanism and this project's measurement of *why* it doesn't help point
the same direction.

Second, the PR is the origin of `_build_dummy_target` and the current
`vf2_layout_pass` signature — the exact functions read in "What the Qiskit source
says" above. The source this document analyzed is the post-#14860 version; the
analysis holds for the codebase as it exists now, independent of what the rejected
report got wrong about the codebase as it existed before.

**What remains unaffected.** Every measured result in this document — the cliff
itself, the per-pass timing, the shuffle-seed experiments on current `main`, the
trial-loop finding, the preset's 0/30 — was produced against the current Rust
implementation and is unaffected by which version's Python wrapper a since-deleted
code path belonged to. Only the historical accuracy of the original attribution is
resolved by this addendum, not any measurement.

---


<!-- ===== Addendum 4 (source: spare-qubit-cliff-addendum-4-shuffle-seed.md) ===== -->

> **Note added when merging:** Confirms the ordering-dependence hypothesis directly inside Qiskit (not just rustworkx) via `shuffle_seed`.

## Proposed addendum for `docs/findings/spare-qubit-cliff.md`

**Where it goes:** after the existing "Addendum (2026-09-13): the objection traces to
a specific, year-old commit".

**Sources read:** the diff of [6421d77](https://github.com/Qiskit/qiskit/commit/6421d77)
/ [#14860](https://github.com/Qiskit/qiskit/pull/14860) covering
`qiskit/transpiler/passes/layout/vf2_layout.py`,
`crates/transpiler/src/passes/vf2/vf2_layout.rs`,
`crates/transpiler/src/transpiler.rs`, `test/python/transpiler/test_vf2_layout.py`;
plus `qiskit/transpiler/preset_passmanagers/builtin_plugins.py` and
`qiskit/compiler/transpiler.py`.

**Caveat on version.** These files were read from Qiskit `main`. Every measurement in
this document was taken on 2.5.2. Where a number below can be checked against a
measurement, it is; where it cannot, it is marked. One check is given in section 3
and it agrees.

---

## Addendum (2026-09-14): the preset disables shuffling, explicitly

### 1. The retracted claim was exact about the code it described

The deleted Python block in #14860 is:

```python
from rustworkx import vf2_mapping
...
mappings = vf2_mapping(
    cm_graph, im_graph, subgraph=True, id_order=False, induced=False,
    call_limit=self.call_limit,
)
```

`subgraph=True, id_order=False, induced=False` is **the exact parameter set**
`benchmarks/vf2_id_order_probe.py` used when it measured `id_order=True` finding the
layout in under a millisecond on grids where `id_order=False` burned seconds. Before
#14860 that probe was not an analogy to Qiskit — it was a reproduction of Qiskit's
literal call.

The deleted code also shows when that call was reached:

```python
# Run rust fast path if we have no randomization
if self.seed == -1:
    ...                      # Rust
# We can't use the rust fast path because we have a seed set, or no target so continue
# with the python path
```

The Rust path required `seed == -1`; the constructor default is `seed=None`. So the
**default** path through `VF2Layout` went to `rustworkx.vf2_mapping` on every release
before 2025-09-19. The rejected sentence was precisely right about the default
behaviour of every Qiskit up to that date, and precisely wrong about the version this
project measured. Both halves matter: the first is why the reproducer was well
chosen, the second is why the report should not have been filed.

### 2. The open question is closed: the preset never shuffles, by construction

This document has listed as unknown *"Why the preset never reaches a winning ordering
— whether it disables shuffling outright or shuffles something that does not reach
the VF2 node order."* **It disables shuffling outright**, in the pass constructor.

**One path, not two.** `qiskit/compiler/transpiler.py` shows that `transpile()` is a
thin wrapper:

```python
pm = generate_preset_pass_manager(optimization_level, target=target, backend=backend, ...)
out_circuits = pm.run(circuits, callback=callback, num_processes=num_processes)
```

So [`verify_preset_shuffle.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_preset_shuffle.py) (which called `transpile()`) and
[`verify_preset_stop_reason.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_preset_stop_reason.py) (which built the pass manager directly) exercised the
**same** code path — the Python preset — not two independent ones. The Rust-native
`transpile` in `crates/transpiler/src/transpiler.rs` is a separate entry point that
`qiskit.transpile()` does not reach; it is quoted below as corroboration that the
same decision was made there too, not as the mechanism behind any measurement here.

**The Python preset.** `DefaultLayoutPassManager` in
`qiskit/transpiler/preset_passmanagers/builtin_plugins.py` constructs the pass with a
hardcoded seed at every optimization level:

```python
# level 1
choose_layout_1 = VF2Layout(coupling_map=..., seed=-1, call_limit=(50_000, 1_000), target=...)
# level 2
choose_layout_0 = VF2Layout(coupling_map=..., seed=-1, call_limit=(5_000_000, 10_000), target=...)
# level 3
choose_layout_0 = VF2Layout(coupling_map=..., seed=-1, call_limit=(30_000_000, 100_000), target=...)
```

`seed=-1` is documented as "disables the shuffling", and post-#14860 it maps to
`shuffle_seed=None`, which the Rust side skips:

```rust
if let Some(seed) = shuffle_seed {
    coupling_qubits.shuffle(&mut Pcg64Mcg::seed_from_u64(seed));
    coupling_graph = vf2::reorder_nodes(&coupling_graph, &order);
}
```

`VF2PostLayout` is constructed the same way in `OptimizationPassManager` at level 3
(`seed=-1`), and the routing plugins pass `seed_transpiler=-1` into
`generate_routing_passmanager`. **Both passes are deterministic on purpose.**

**The Rust-native transpile, for corroboration.** The separate `transpile` in
`crates/transpiler/src/transpiler.rs` — not the one `qiskit.transpile()` calls —
hardcodes the same argument at every level:

```rust
vf2_layout_pass(&dag, target, false, Some(5_000_000), None, Some(2500), None, None)?
//                                                                        ^^^^  shuffle_seed
```

**So 30 identical `NO_SOLUTION_FOUND` results were not thirty unlucky draws** (which
this document computed at 1.4%). There was never a draw. The same node order is used
every time, and on this input that order does not find the layout.

**`seed_transpiler` was never a lever on VF2, by construction.** Tracing it through
`builtin_plugins.py`, `pass_manager_config.seed_transpiler` reaches `SabreLayout` and
`SabreSwap` only. Every `VF2Layout` and `VF2PostLayout` gets the literal `-1`. This
document's observation that pinning it changed nothing (1.004x) was correct; the
reason is that it does not reach these passes at all.

**One prior result now reads as a consistency check rather than a coincidence.**
[`verify_preset_stop_reason.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_preset_stop_reason.py)'s pinned arm used `seed_transpiler=0`. That value never
reached `VF2Layout` either — so both arms ran the identical unshuffled search, which
is why all 40 calls agreed to the gate count and depth as well as the stop reason.

### 3. A correction: `get_vf2_limits` governs `VF2PostLayout`, not `VF2Layout`

This document says the level dependence is *"the VF2 call budget, which
`get_vf2_limits` sets to 50,000 at levels 1–2 and 30,000,000 at level 3"*. Reading
`builtin_plugins.py`, those are two different budgets:

- `VF2Layout`'s budget is the hardcoded 2-tuple in `DefaultLayoutPassManager`, listed
  above — **50,000 / 5,000,000 / 30,000,000** at levels 1 / 2 / 3.
- `get_vf2_limits` is called in the *routing* stage plugins and in
  `OptimizationPassManager`, and feeds `VF2PostLayout`.

**This resolves an inconsistency that has been sitting unnoticed in this document.**
The per-level VF2 timings recorded above are 22.6 ms / 1,117 ms / 12,771 ms at levels
1 / 2 / 3. If levels 1 and 2 shared a 50,000 budget, they should have been close;
they differ by 49x. The actual budgets differ by 100x (50,000 → 5,000,000), and by
600x from level 1 to level 3 against a measured 565x. **The measurement fits the
source, and the sentence describing the source did not.** It should be corrected
rather than left as the one number in the section that never added up.

### 4. A correction: the trial-loop cost was measured under a configuration the preset does not use

This document concludes that ordering and trial loop are *"two independent costs, and
neither fix alone removes the cliff"*, on the strength of seed 1 finding its layout in
3.5 ms and the pass still running 343 ms. That measurement passed `call_limit` as a
**scalar** (3,000,000), so the improvement search inherited the whole remaining
budget.

The preset passes a **2-tuple**, whose second element exists precisely to bound that
phase — 1,000 / 10,000 / 100,000 steps at levels 1 / 2 / 3, against first elements of
50,000 / 5,000,000 / 30,000,000. At level 3 the post-match budget is **0.33%** of the
pre-match one.

So for the preset's saturated case the two costs are not both live:

| | preset, saturated | standalone scan, scalar `call_limit` |
| :--- | :--- | :--- |
| Ordering | fixed, never shuffled, never finds a match | varies with `shuffle_seed`; 4/30 find one |
| Trial loop | **never runs** — there is no first match to improve on | runs with the full remaining budget |
| Where the time goes | entirely the pre-match budget burning | 90–99% after the match, on the seeds that find one |

**For the preset, the ordering is the whole cost.** The trial-loop finding remains a
correct and interesting measurement of `VF2Layout` called directly with a scalar
budget — and it is what made the `call_limit` 2-tuple legible when it appeared — but
it does not describe what the preset spends its 6.8 seconds on. The "two independent
costs" framing should be scoped to the standalone configuration.

**Unmeasured, and predicted before testing.** On the saturated 6x7 grid,
`call_limit=(3_000_000, 10_000)` should reproduce the `max_trials=1` result: seed 1
near 3.5 ms, seeds 25 and 29 near 34 ms, seed 8 and the 26 failing seeds unchanged
near 330 ms. If instead the successful seeds still take ~340 ms, the second element
does not do what its docstring says and this section is wrong.

### 5. Both remaining candidates are resolved

This document listed two untested candidates for the preset's behaviour: *"the
property-set `vf2_avg_error_map` and the preset's own `call_limit` 2-tuple."*

- The `call_limit` 2-tuple is now read directly (section 2).
- `vf2_avg_error_map` is **moot for this question**. It enters only as the scoring
  input, and the outcome is fixed before scoring matters: with shuffling disabled
  there is exactly one node order, so there is no "winning ordering to reach". This
  also agrees with [`verify_vf2_target_scoring.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_target_scoring.py), which found the same 4/30 and the
  same seeds under dummy and real targets.

### 6. The `max_trials` docstring does not describe this workload

> Since the scoring is done on-the-fly, the vast majority of candidate layouts are
> pruned out of the search before ever becoming complete, so **this option has little
> meaning**.

On this circuit family `max_trials=1` made seed 1 **98.2x** faster. Not a
contradiction so much as a workload outside the docstring's assumption: a
perfect-matching interaction graph on a saturated grid appears to produce complete
candidate layouts faster than scoring prunes them. Recorded as an observation about
where the documented intuition stops holding, not as a claim about Qiskit in general.

### 7. Qiskit's own tests show the ordering changes the answer

Two expected-layout assertions changed in #14860's
`test/python/transpiler/test_vf2_layout.py` — `{16, 24, 6, 7, 0}` became
`{26, 11, 14, 7, 10}`, and `{3, 1, 0}` became `{3, 2, 0}` — for unchanged inputs.
Same circuits, same targets, different node ordering, different chosen layout.
Independent confirmation from Qiskit's own suite that this is a real degree of
freedom.

### 8. Two incidental notes from `qiskit/compiler/transpiler.py`

**`optimization_level=None` means level 2, not level 1.** The default is read from
the user config with a fallback of 2. Nothing in this document depends on it, but any
experiment here that omits the argument is running level 2.

**"Unpinned" is not necessarily unpinned.** When `seed_transpiler` is `None`,
`transpile()` falls back to the `QISKIT_TRANSPILER_SEED` environment variable and
then to `transpiler_seed` in the user config file. This did not affect the VF2
measurements — the seed never reaches those passes — but it does reach `SabreLayout`
and `SabreSwap`, so it is a real consideration for the routing-variation observations
elsewhere in this document, and worth adding to
[`record-keeping.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/record-keeping.md): a run described as unpinned should
confirm that neither the environment variable nor the config file is setting a seed.

### What is still unknown after this

- Whether these files, read from `main`, match 2.5.2 in the details that matter. The
  budget check in section 3 agrees; `seed=-1` has not been checked against 2.5.2.
- The `call_limit` 2-tuple prediction in section 4, unmeasured.
- Why seed 8 is slow to its first match while seeds 1, 25 and 29 are 10–100x faster.
- Which part of the VF2++ ordering causes the failure, in either implementation.
- Whether the two implementations fail on the same instances.
- Topologies beyond rectangular grids.

**No measurement in this document changes.** Sections 1, 2, 5, 6 and 7 are source
reading; sections 3 and 4 correct statements *about* the source and about the scope
of a measurement, not the measurement itself.

### What the practical advice becomes

Unchanged in substance, sharper in reason. Padding the coupling map still removes the
effect. But *"seed_transpiler is not a lever"* is now explained rather than merely
observed — it never reaches these passes — and the honest summary of the preset's
behaviour is: **`VF2Layout` searches one fixed node ordering, that ordering fails on
a saturated grid with a perfect-matching interaction graph, and Qiskit spends the
level's full pre-match budget establishing it.** Whether a shuffled ordering *should*
be tried is a design question for Qiskit, and #14860's own commit message already
records a view on it: *"The shuffling is, in general, not a good idea."*

---

## Note on the upstream position

None of this re-opens contact upstream. The decision recorded above — no further
upstream contact — stands. Section 1 makes the original report's reasoning legible;
it does not make the report correct, and the objection to it was right.

---


<!-- ===== Addendum 5 (source: spare-qubit-cliff-addendum-5-2026-09-14.md) ===== -->

## Addendum (2026-09-14): six pre-registered predictions, tested on real hardware

All six verification scripts ([`vf2_probe_common.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/vf2_probe_common.py) + [`verify_vf2_call_limit_tuple.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_call_limit_tuple.py),
[`verify_vf2_steps_to_first_match.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_steps_to_first_match.py), [`verify_vf2_ordering_structure.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_structure.py),
[`verify_vf2_cross_implementation.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_cross_implementation.py), [`verify_vf2_topologies.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_topologies.py),
[`verify_qiskit_source_2_5_2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_qiskit_source_2_5_2.py)) were run on the Intel machine
(`Intel64 Family 6 Model 181`, Windows 10, Python 3.11.9, Qiskit 2.5.2, rustworkx
0.18.1, `psf_zero_core.cp311-win_amd64.pyd:418304`, rebuilt 2026-09-14T09:19:45 — the
same rebuilt core discussed in the [`compile-time.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/compile-time.md) addendum). Each script's
predictions were written before the run, in the script itself. Two came back clean
confirmations; three were refuted outright; one produced a genuine unresolved anomaly.
Per this project's own rule, a refuted prediction is not a failed experiment — it is
the result.

### Summary table

| Script | Predictions | Outcome |
| :--- | :--- | :--- |
| [`verify_qiskit_source_2_5_2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_qiskit_source_2_5_2.py) | P1–P4 (installed-2.5.2 source matches `main`-derived claims) | **All confirmed.** No change to prior addenda. |
| [`verify_vf2_cross_implementation.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_cross_implementation.py) | P1 (Qiskit and rustworkx fail at the same boundary), P2 (24 nodes is a property of the heuristic, not the wrapper) | **Both confirmed, cleanly.** 40/40 grids agree; both fail starting exactly at 24 nodes (3×8, 4×6). |
| [`verify_vf2_steps_to_first_match.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_steps_to_first_match.py) | P1 (step-count ratio ≈ time ratio for successful seeds), P2 (failing seeds never succeed under the 3,000,000 limit) | **P1 refuted, P2 confirmed.** Step ratio 4.3x against a time ratio of ~96x for the same seeds. |
| [`verify_vf2_call_limit_tuple.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_call_limit_tuple.py) | P1 (`tuple_10k` ≈ `max_trials1`), P2 (identical stop reasons across arms), P3 (`tuple_full` ≈ `scalar`) | **P3 confirmed. P1 refuted in a specific, informative way** (see below). **A new anomaly** in the failing seeds, unresolved. |
| [`verify_vf2_ordering_structure.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_structure.py) | P1 (locality-preserving orders succeed), P3 (bipartite is a dividing line) | **Both refuted.** Every hand-designed structured ordering failed, on every grid; only random draws ever succeeded, and rarely. |
| [`verify_vf2_topologies.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_topologies.py) | P1 (cliff only where a perfect matching exists), P3 (line/ring won't show a cliff since matching is trivial) | **P1 not falsified but insufficient; P3 refuted.** Line shows a cliff as large as the grids. Ring and full-graph — also with a trivial perfect matching — show none. |

---

### 1. Cross-implementation agreement: confirmed cleanly

[`verify_vf2_cross_implementation.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_cross_implementation.py) scanned every grid from 2×2 (4 nodes) to 8×9
(72 nodes) with a perfect matching, calling Qiskit's `VF2Layout` (`seed=-1`,
`call_limit=3,000,000`) and rustworkx's `vf2_mapping` (`id_order=False,
induced=False, call_limit=3,000,000`) side by side. Both succeed on every grid below
24 nodes and fail on every grid from 24 nodes up — **40 out of 40 grids agree**,
with the failure boundary landing exactly on 3×8 and 4×6 as in the original
sandbox measurement. This confirms, independently and on real (not sandboxed)
hardware, that the 24-node boundary is a property of the underlying VF2++ ordering
heuristic itself — not an artifact of the Qiskit wrapper, the sandbox's 2-core Linux
box, or the earlier low-`call_limit` smoke test (which had shown a spurious boundary
at 20 nodes, now understood as a call_limit-too-low artifact of that quick test, not
of the real behavior).

No open question remains here. This closes the "is the 24-node boundary a property
of the heuristic or of Qiskit's use of it" question that `docs/log/06-open-questions`
and this document's own "what is still unknown" section had left open.

### 2. Steps-to-first-match does not track wall-clock time — and the call_limit-tuple data explains why

[`verify_vf2_steps_to_first_match.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_steps_to_first_match.py) used a binary search (23 probes per seed) to find
the minimal `call_limit` — with `max_trials=1` — that still finds a mapping, as a
machine-independent proxy for search depth. Result, successful seeds only:

| Seed | Minimal sufficient `call_limit` | `max_trials1` wall time (median, this run) |
| :---: | ---: | ---: |
| 1 | 186 | 3.76 ms |
| 25 | 43 | 34.9 ms |
| 29 | 162 | 34.3 ms |
| 8 | 43 | **332 ms** |

**The predicted relationship (step ratio ≈ time ratio) is refuted outright.** The
step-count spread across these four seeds is 4.3x (186 / 43). The wall-time spread
for the same four seeds, at the same `call_limit=3,000,000, max_trials=1`
configuration, is ~96x (332 ms / 3.76 ms). Seed 8 needs the *fewest* calls of any
seed (43, tied with seed 25) to reach a solution, yet takes by far the *longest*
wall time to reach it when given a large budget.

This result only makes sense next to [`verify_vf2_call_limit_tuple.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_call_limit_tuple.py)'s data for the
same seeds. For seed 8: `max_trials1` (call_limit=3,000,000, stop after 1 solution) =
332 ms, but `tuple_10k` (call_limit=`(3,000,000, 10,000)`) = **1.7 ms** — nearly 200x
faster, for what should be the same search terminating at the same first solution.
Seeds 1, 25, 29 show the same pattern in miniature: `tuple_10k` is 2–20x faster than
`max_trials1` in every successful case.

**Working hypothesis (new, not yet independently confirmed): the magnitude of the
call-limit argument changes which part of the search tree is explored, not only when
the search is cut off.** A minimal sufficient `call_limit` (43, in seed 8's case)
forces the algorithm down a short, cheap path that happens to reach a solution
quickly. A large budget (3,000,000, or the effective cap implied by a 2-tuple's
second element) appears to let, or cause, the algorithm to take a much longer path
before landing on essentially the same kind of solution — plausibly because internal
batching, memory allocation, or heuristic re-evaluation inside `rustworkx.vf2_mapping`
scales with the requested budget rather than running identically regardless of it.
If true, this means **"steps to first match" and "wall time under an ample budget"
are measuring two different things**, and the earlier framing of the former as a
"machine-independent proxy" for the latter was wrong for seeds like 8, though it
still holds directionally for seeds where the two are closer (1, 25, 29 all show a
double-digit ms/step-poor correlation too, so this is not a clean binary).

This hypothesis is falsifiable and untested directly: the natural next experiment is
to sweep a single seed (8 is the sharpest case) across a range of scalar
`call_limit` values (e.g. 100, 1,000, 10,000, 100,000, 1,000,000, 3,000,000) with
`max_trials=1` fixed, and plot wall time against the limit. If the hypothesis is
right, time should jump non-monotonically or step-wise as the limit crosses some
internal threshold, rather than scaling smoothly with the number of calls actually
needed.

### 3. The call_limit-tuple anomaly in the failing seeds (unresolved)

For the four seeds with no solution (`NO_SOLUTION_FOUND`), the `scalar`,
`max_trials1`, `tuple_10k`, and `tuple_full` arms are expected to cost the same,
since there is no early exit to take advantage of — and seeds 0 and 2 confirm this
(all four arms land in the normal 320–330 ms band, consistently).

Seeds 3 and 4 do not:

| Seed | Arm | min (s) | median (s) |
| :---: | :--- | ---: | ---: |
| 3 | `scalar` | 0.358 | **1.013** |
| 3 | `max_trials1` | 1.013 | 1.023 |
| 3 | `tuple_10k` | 0.332 | 0.339 |
| 3 | `tuple_full` | 0.326 | 0.327 |
| 4 | `scalar` | 0.333 | 0.336 |
| 4 | `max_trials1` | 0.342 | 0.342 |
| 4 | `tuple_10k` | **0.969** | **1.148** |
| 4 | `tuple_full` | **0.860** | **1.157** |

Seed 3's `scalar` arm is bimodal across its own repetitions (0.358 s to 1.013 s — a
2.8x spread within one configuration), while its tuple arms are tight and normal.
Seed 4 shows the mirror image: `scalar`/`max_trials1` are normal, but *both* tuple
arms jump to nearly 3x the normal cost, consistently across repetitions (not
bimodal — 0.969–1.148 s is a tighter spread than seed 3's scalar arm, just centered
higher).

This does not fit the hypothesis in §2 cleanly: if a smaller effective budget (via
the tuple's second element) generally shortens the path taken, it should do so for
failing seeds too, not lengthen it. Two candidate explanations, neither confirmed:

- **External load.** The Intel machine's 2026-09-14 [`test_cumulative_compile_time.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_time.py)
  re-run (see [`compile-time.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/compile-time.md) addendum) showed heavy, uneven external contention
  during that session. If seeds 3/4 happened to run during a load spike, that alone
  could produce seed 3's bimodal `scalar` result — though it does not obviously
  explain why seed 4's tuple arms specifically, and consistently, were the slow ones.
- **A real interaction between the 2-tuple form and the "no solution" path**, where
  supplying a 2-tuple changes the internal work done even when no solution is ever
  found — for instance, if the second element causes periodic re-evaluation or
  restart behavior that is more expensive per unit time than the plain scalar path
  when nothing is ever found.

**This is flagged as unresolved, not folded into either the §2 hypothesis or a new
claim.** The proposed follow-up is a repeated, isolated run of seeds 3 and 4 alone
(20+ reps per arm, back to back, with system load logged if possible) to establish
whether the pattern is reproducible or was one-off contention.

### 4. Ordering structure: none of the hand-designed orderings behave differently from each other

[`verify_vf2_ordering_structure.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_structure.py) tested seven structural node orderings
(`row_major`, `col_major`, `snake`, `bfs_corner`, `dfs_corner`, `bipartite`,
`reverse`) plus five random draws, against `rustworkx.vf2_mapping(id_order=False, ...)`
directly, on 6×7, 7×8, 8×8, and 8×9 grids (28 structured attempts, 20 random
attempts, 48 total).

**Every one of the 28 structured attempts failed**, across all four grids, with times
clustered tightly (0.34–0.52 s) and no systematic difference between `row_major`,
`bipartite`, or any other ordering. **19 of 20 random attempts also failed**; the one
success (8×8, random seed 4) found a mapping in 0.1 ms — near-instant, unlike
anything the structured orderings produced.

Both P1 (locality-preserving orders succeed) and P3 (bipartite is a dividing line)
are refuted, and refuted in the same direction: **imposing a specific relabeling of
the graph's nodes had no measurable effect on outcome or timing**, whether the
relabeling was geometrically sensible (row-major, snake) or adversarial-looking
(reverse, random). Only 1 of 25 total attempts, structured or random, ever
succeeded.

**Working hypothesis (new): with `id_order=False`, relabeling the input graph's node
IDs does not control the search order the way `id_order=True` would.** `id_order=False`
tells `vf2_mapping` to compute its own traversal order from the graph's structure
(degree sequence and connectivity), which is invariant under relabeling — so feeding
it a "row-major" versus a "reverse" labeling of the identical grid graph should
produce the identical internal order, and did. The rare random success is then most
plausibly a tie-break effect: when several nodes tie on whatever criterion the
internal ordering uses, the node-ID order our relabeling supplies may break the tie,
and one draw in 20 happened to break it usefully. This is a small, weak effect
(5% observed rate on one grid, zero elsewhere) and should not be overstated as a
confirmed mechanism.

This hypothesis is directly testable and cheap: **repeat the same script with
`id_order=True`** on the same grids and orderings. If the hypothesis is right, the
orderings should now differ sharply from each other (since `id_order=True` uses the
supplied order directly, and we already know from the earlier root-cause work that
`id_order=True` finds a mapping in under a millisecond on these exact grids
regardless of ordering) — which would itself be a check that the harness is wired
correctly, since a total failure to differ under `id_order=True` would mean the
ordering argument to the harness isn't reaching the call at all.

### 5. Topology cliff: perfect-matching existence is not sufficient — vertex-transitivity looks like the real dividing line

[`verify_vf2_topologies.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_topologies.py) tested five topologies at 20–56 physical qubits, checking
`has_perfect_matching()` before interpreting any result (specifically to avoid
repeating the mistake that got the earlier upstream report rejected — reporting a
correct "no solution" as if it were a bug).

| Topology | Nodes | Perfect matching exists | `opt=2` ratio (0 spare ÷ 4 spare) | `opt=3` ratio |
| :--- | ---: | :---: | ---: | ---: |
| `grid_6x7` | 42 | True | 57x | 414x |
| `grid_7x8` | 56 | True | 57x | 284x |
| `line_42` | 42 | True | **61x** | **298x** |
| `ring_42` | 42 | True | 0.92x (no cliff) | 0.90x (no cliff) |
| `full_20` | 20 | True | 1.02x (no cliff) | 0.79x (no cliff, if anything reversed) |

All five topologies have a perfect matching in every configuration tested — so bare
matching existence, which P1 treated as the necessary condition, is confirmed as
necessary but is **not sufficient**, and P3's specific prediction (that line and ring
would both escape the cliff, since a matching is trivial to find on either) is
**refuted for line**: `line_42` shows a cliff (298x at level 3) fully comparable in
size to the grids, while `ring_42` — a line closed into a loop, one edge different,
with an equally trivial perfect matching — shows none at all.

**Working hypothesis (new): the cliff tracks structural inhomogeneity (the coupling
graph not being vertex-transitive), not graph family or matching existence.** A line
graph has two degree-1 endpoints and every other node at degree 2 — it is not
vertex-transitive; neither is a grid, which has corners (degree 2), edges (degree 3),
and interior nodes (degree 4). A ring is vertex-transitive (every node has degree 2
and the graph looks identical from any node); so is a complete graph (every node has
degree n−1). The two topologies that show no cliff are exactly the two that are
vertex-transitive; the two that show the full cliff are exactly the two that are not.
Removing spare qubits from an inhomogeneous graph removes specific low-degree
positions (a line's endpoints, a grid's boundary), which plausibly interacts badly
with whatever the VF2++ ordering heuristic does with degree information; removing
spare qubits from a vertex-transitive graph does not change which positions are
"special," because none are.

This reframes the open question from "is the effect specific to `CouplingMap.from_grid`"
to "is the effect specific to non-vertex-transitive graphs" — a cleaner, more testable
claim. The natural falsification test is a **periodic (toroidal) grid** — same local
connectivity as `grid_6x7`, but with wraparound edges so every node has degree 4 and
the graph is vertex-transitive. If the hypothesis holds, a saturated toroidal grid
should show *no* cliff, unlike the open grid tested here. Qiskit's `CouplingMap` does
not build this directly, but it is a few lines of `networkx.grid_2d_graph(..., periodic=True)`
converted to a `CouplingMap`.

### What is still unknown, updated

Superseded or answered by the runs above:

- ~~Whether the 24-node boundary is a property of the heuristic or of Qiskit's use of
  it~~ — confirmed as a property of the heuristic itself (§1).
- ~~Whether "steps to first match" is a valid machine-independent proxy for wall-clock
  cost~~ — it is not, for at least one seed (8) out of four tested (§2), and the
  call_limit-tuple data suggests why.
- ~~Whether line/ring topologies escape the cliff because a perfect matching is
  trivial to find on them~~ — refuted for line; matching existence is necessary but
  not sufficient (§5).

Newly open, in place of the above:

- **Does call_limit magnitude change the search path, not just where it is cut off?**
  (§2) — untested directly; proposed sweep given above.
- **Why do failing seeds 3 and 4 behave anomalously under the tuple forms of
  `call_limit`, in opposite directions?** (§3) — unresolved; proposed isolated re-run
  given above.
- **Does `id_order=False` ignore supplied node relabeling except as a tie-break?**
  (§4) — untested directly; proposed `id_order=True` re-run given above.
- **Is the cliff specific to non-vertex-transitive coupling graphs?** (§5) — untested
  directly; proposed toroidal-grid control given above.
- Seed 8's specific mystery (low step count, high wall time) is now understood as one
  instance of the broader §2 question rather than a seed-specific oddity, but the
  general mechanism is still a hypothesis, not a measurement.

**Note on the upstream position: none of this reopens or changes anything sent
upstream.** The reported claim — VF2++ ordering fails to find an existing mapping on
saturated grids, `id_order=True` fixes it, budget does not — is untouched by any
result here. These six experiments were run to sharpen this project's own
understanding of *why* the ordering fails and how far it generalizes, not to revisit
the report itself.

### Files

- Scripts: `benchmarks/verify_vf2_call_limit_tuple.py`,
  `benchmarks/verify_vf2_steps_to_first_match.py`,
  `benchmarks/verify_vf2_ordering_structure.py`,
  `benchmarks/verify_vf2_cross_implementation.py`,
  `benchmarks/verify_vf2_topologies.py`,
  `benchmarks/verify_qiskit_source_2_5_2.py`,
  `benchmarks/vf2_probe_common.py`
- Raw data (Intel, 2026-09-14): `data/vf2_call_limit_tuple_2026-09-14.csv`,
  `data/vf2_steps_to_first_match_2026-09-14.csv`,
  `data/vf2_ordering_structure_2026-09-14.csv`,
  `data/vf2_cross_implementation_2026-09-14.csv`,
  `data/vf2_topologies_2026-09-14.csv`,
  `data/qiskit_source_check_2026-09-14.csv`

---


<!-- ===== Addendum 6 (source: spare-qubit-cliff-addendum-6-2026-09-14.md) ===== -->

## Addendum (2026-09-14, second batch): four targeted follow-ups, tested on real hardware

Same machine as addendum-5 (`Intel64 Family 6 Model 181`, Windows 10, Python 3.11.9,
Qiskit 2.5.2, rustworkx 0.18.1, `psf_zero_core.cp311-win_amd64.pyd:418304`), same day.
These four scripts were written specifically to chase the open questions addendum-5
left behind. One anomaly turned out to be noise. The other three landed cleanly, and
one of them (§3) sharpens the whole call_limit-tuple story from "here is a puzzling
number" to "here is the mechanism."

### 1. The seed 3 / seed 4 anomaly does not reproduce — it was noise

[`verify_vf2_seed_anomaly_repro.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_seed_anomaly_repro.py) reran the four `call_limit` arms for seeds 3 and 4
at 20 repetitions instead of 3. Every arm for both seeds now lands in the same
320–360 ms band, with tight spreads (1.04–1.12x, all classified "single-peaked") and
no consistent CPU-load correlation.

**Both P1 and P2 are refuted, and the refutation is the good outcome here.** The
2026-09-14 (first-batch) result — seed 3's bimodal `scalar` arm (0.358 s vs 1.013 s),
seed 4's tuple arms running 3x slower than `scalar` — does not survive more
repetitions. At `reps=3`, a single slow outlier is enough to swing a reported min or
median by 3x; at `reps=20` the same configurations are indistinguishable from the
normal failing-seed cost. **This retracts addendum-5 §3's "unresolved anomaly"
entirely.** It was not a real property of these seeds or of the tuple form — it was
measurement noise from too few repetitions, most likely combined with transient
system load on that run. No further investigation of seeds 3/4 specifically is
warranted.

This is also a useful general lesson for this project's own discipline: `reps=3` is
too few when a single run can be perturbed by external load to this degree. The
existing six-script batch used `reps=3` throughout; any measurement in it whose
conclusion rests on a single seed's specific timing (rather than a pattern across
many seeds, like the cross-implementation or topology results) should be treated with
that in mind.

### 2. Ordering structure under `id_order=True`: confirmed, and sharper than predicted

[`verify_vf2_ordering_id_order_true.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_id_order_true.py) reran the seven structured orderings plus five
random draws under both `id_order=False` (reproducibility check) and `id_order=True`
(the new test), on all four grids.

**`id_order=False` reproduces addendum-5 exactly**, including the one specific
exception: 8×8's `random_4` succeeds instantly (0.000136 s here vs 0.0001 s in the
2026-09-14 original) — the same draw, the same result, to the same order of
magnitude. That this one exception is *reproducible* (same random seed, same
outcome) rather than a fluke supports the tie-break part of addendum-5 §4's
hypothesis: `id_order=False` mostly ignores supplied relabeling, but a tie-break
effect is real and deterministic for a given permutation, not noise.

**`id_order=True` confirms P1 and P2 cleanly, with a sharper structure than
predicted:**

| Order | 6×7 | 7×8 | 8×8 | 8×9 |
| :--- | :---: | :---: | :---: | :---: |
| `row_major`, `col_major`, `snake`, `bfs_corner`, `bipartite`, `reverse` | found | found | found | found |
| `dfs_corner` | **not found** | found | **not found** | **not found** |
| `random` (5 draws) | 2/5 found | 1/5 found | 0/5 found | 0/5 found |

Every structured ordering succeeds, fast (mostly under 0.2 ms), on every grid tested
— **except `dfs_corner`, which fails on 3 of 4 grids** (uniquely, and consistently:
it is the only structured order that ever fails under `id_order=True`). Random draws
mostly fail even under `id_order=True` (3 of 20 succeed), a real improvement over
`id_order=False`'s 1 of 20, but nowhere near the structured orders' near-total
success.

**P1 confirmed:** `id_order=True` makes structured orderings succeed where
`id_order=False` uniformly failed — direct evidence that supplied node relabeling is
actually used when `id_order=True`, unlike the `id_order=False` case.

**P2 confirmed, and refined:** orderings do differentiate sharply under
`id_order=True` — but the dividing line is not "locality-preserving vs not" (as
addendum-5's original ordering-structure predictions guessed) or "bipartite vs not"
(P3). It is **breadth-first vs depth-first from a corner.** `bfs_corner` succeeds on
every grid; `dfs_corner` fails on 3 of 4. Every other structured order tested —
including `reverse`, which is about as "non-local" a relabeling of `row_major` as one
can construct without literally becoming DFS — succeeds. This suggests depth-first
traversal specifically produces long, thin candidate chains that interact badly with
this circuit's interaction graph (disjoint adjacent pairs), plausibly because a long
DFS run crosses many pair-boundaries before backtracking, forcing the search deep
into a heavily constrained partial assignment before it can recover — while BFS
(and the geometric/algebraic orders that happen to resemble it locally) keeps the
frontier "wide," visiting whole rows or diagonals together and completing local pairs
before committing to distant ones.

**P3 (bipartite specifically fast) is not distinctly supported.** Bipartite does
succeed, but so does nearly everything except `dfs_corner` — it is not uniquely
fast or uniquely favored; the earlier framing overstated its role. The real
dividing line found here is narrower and more specific than "matching-structure vs
geometric-locality": both categories succeed under `id_order=True` (row_major is
geometric, bipartite is matching-structure-aligned, reverse is neither) except for
one specific traversal shape.

### 3. `call_limit` sweep: the search does not stop at the first match — and the tuple's second element does exactly what its docstring says

[`verify_vf2_call_limit_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_call_limit_sweep.py) swept scalar `call_limit` from 43 to 3,000,000 for
seed 8, seed 1, and failing seed 0, all with `max_trials=1`.

| `call_limit` | seed 8 (ms) | seed 1 (ms) | seed 0, failing (ms) |
| ---: | ---: | ---: | ---: |
| 43 | 0.61 | 0.55 (not found) | 0.56 |
| 1,000 | 0.71 | 0.72 | 0.70 |
| 10,000 | 1.69 | 1.78 | 1.67 |
| 100,000 | 11.69 | 3.74 | 11.71 |
| 300,000 | 33.91 | 3.86 | 33.73 |
| 1,000,000 | 113.71 | 3.73 | 112.06 |
| 3,000,000 | 344.55 | 3.80 | 330.62 |

**P1 (a discontinuous jump for seed 8) is refuted.** Time scales smoothly and
monotonically with `call_limit` — no jump, no threshold effect. But refuting P1
this way uncovers something more useful than a jump would have been.

**Seed 1 plateaus at ~3.7–4 ms once `call_limit` exceeds roughly 30,000 — spending
more budget buys it nothing further.** Seed 8, by contrast, **never plateaus**: its
time keeps climbing all the way to 3,000,000, and at every single `call_limit` value
in this table it lands within a few percent of **failing seed 0's time at the same
value** (100,000: 11.69 vs 11.71 ms; 1,000,000: 113.71 vs 112.06 ms; 3,000,000:
344.55 vs 330.62 ms). This is the real finding: **seed 8's search, once given a
budget of `call_limit`, consumes essentially all of it — indistinguishably from a
search that never finds a solution at all — and simply happens to have recorded a
valid mapping somewhere along the way, which it reports only once the full budget is
spent.** `max_trials=1` is not stopping the search early for this seed; it is only
bounding how many solutions get *recorded*, not how much of `call_limit` gets *used*.
Seed 1's search space beyond its own threshold is apparently small enough that it
runs out on its own well before `call_limit` is exhausted, which is why it plateaus
and seed 8 does not.

This directly and quantitatively explains the `call_limit`-tuple mystery from
addendum-5 §2, more precisely than the "budget changes the path" framing there:
**`tuple_10k = (3,000,000, 10,000)` gave seed 8 a time of 1.7 ms in the original
run — and this sweep's scalar `call_limit=10,000` gives seed 8 1.69 ms.** They match
to three significant figures. The 2-tuple's second element is exactly what its
docstring said all along: once the first match is found, the remaining budget swaps
to the second value — and seed 8's search, true to the pattern above, then consumes
*that* budget in full too, rather than stopping. `tuple_full = (3,000,000,
3,000,000)` giving seed 8 ~334–346 ms across the two runs (original vs this sweep's
scalar-3M value, 344.55 ms) is the same statement with the swapped budget equal to
the original one. Every number in the original tuple experiment is now accounted
for by one mechanism, not several.

**This also reframes what "steps to first match" (addendum-5 §2) actually measured.**
The binary search there found the smallest `call_limit` for which a solution is
*ever recorded* — and by the mechanism above, at any `call_limit`, the search runs to
completion of that budget regardless, so the binary search was really asking "how
small can the budget be and still have the solution appear before the budget runs
out," not "how many steps does the search take before it would naturally stop." That
these are different quantities is exactly why the step-count ratio (4.3x) did not
match the wall-time ratio (~96x) — wall time under an ample budget is governed by how
much of that budget gets consumed (which, per this section, is *all of it*, for
seed 8), not by how quickly a solution happens to be found within it.

**Open question this raises, not yet tested:** all of the `call_limit`-tuple and
steps-to-first-match scripts call Qiskit's `VF2Layout` pass (`p.run(dag)`), a Rust
routine that returns a finished result rather than a lazily-consumed Python
iterator. The ordering-structure and cross-implementation scripts, by contrast, call
`rustworkx.vf2_mapping()` directly and take `next(iter(it))` — ordinary Python
generator semantics, which should stop computing the moment one item is yielded. If
`VF2Layout`'s internal Rust call does not expose the same short-circuit-on-first-item
behavior that the raw iterator does, that alone would explain why seed 8 "burns the
full budget" through the Qiskit pass. The direct test: run the same seed-8 instance
through `rx.vf2_mapping(..., call_limit=L)` with `next(iter(it))` at a range of `L`
values matching this sweep, exactly as [`verify_vf2_ordering_structure.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_structure.py) already
does elsewhere. If the raw iterator stops immediately once a match is found
regardless of `L`, the "full budget consumed" behavior is specific to `VF2Layout`'s
wrapper, not to rustworkx's search itself — a meaningful distinction for anything
reported upstream in the future, though nothing here changes what has already been
reported.

### 4. Toroidal grid: the vertex-transitivity hypothesis holds

[`verify_vf2_toroidal_grid.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_toroidal_grid.py) built periodic (wraparound) 6×7 and 7×8 grids —
identical local connectivity to the open grids, but every node at degree 4 with no
boundary, i.e. vertex-transitive — and compared saturated vs. 4-spare timings
against the open (non-periodic) grids of the same shape, in the same run.

| Topology | Spare | `opt=2` ratio | `opt=3` ratio |
| :--- | :---: | ---: | ---: |
| `torus_6x7` | 0 vs 4 | **1.0x** | **0.8x** |
| `open_6x7` | 0 vs 4 | 33.7x | 249.7x |
| `torus_7x8` | 0 vs 4 | **1.1x** | **1.0x** |
| `open_7x8` | 0 vs 4 | 47.2x | 274.7x |

**P1 confirmed cleanly: the torus shows no cliff at all**, on both shapes and both
optimization levels — ratios of 0.8x–1.1x, the same "no cliff" signature already
seen for `ring_42` and `full_20` in addendum-5. **P2 confirmed: the open-grid cliff
reproduces in the same run** (33.7x–274.7x — same order of magnitude as the
2026-09-14 originals, 57x–414x; the exact numbers move between runs as this
project's other measurements do, but the qualifying "cliff" is unambiguous). **P3
confirmed:** a perfect matching exists in every row tested.

This is the strongest of the four follow-ups. The prediction was specific and
falsifiable — a graph with the identical local structure of a known-cliff grid, but
made vertex-transitive by adding wraparound edges, should behave like the other
vertex-transitive graphs (ring, full) rather than like the grid — and it held on both
shapes tested. **The dividing line for the spare-qubit cliff is now best stated as:
it appears on non-vertex-transitive coupling graphs (grid, line) and does not appear
on vertex-transitive ones (ring, full graph, and now the torus), independent of
"gridness" as such.** This is a real narrowing of the original finding, not just a
hypothesis anymore — though it rests on two shapes at one circuit family, and a
heavy-hex-shaped control (real quantum hardware's actual topology, which is neither a
simple grid nor vertex-transitive) remains untested.

### What is still unknown, updated again

Resolved by this batch:

- ~~Are seeds 3/4's call_limit-tuple timings anomalous?~~ — no; it was `reps=3`
  noise (§1).
- ~~Does `id_order=False` ignore supplied relabeling except as a tie-break?~~ —
  essentially yes, and `id_order=True` differentiates orderings sharply, on a
  breadth-first/depth-first axis rather than the originally-guessed
  locality/bipartite axis (§2).
- ~~Does call_limit magnitude change the search path, explaining the
  steps-vs-time mismatch?~~ — reframed and answered more precisely: for at least
  one seed (8), the search (via Qiskit's `VF2Layout`) consumes its entire assigned
  budget regardless of when a solution is found, and the 2-tuple's second element
  is exactly the post-match replacement budget the docstring describes (§3).
- ~~Is the cliff specific to non-vertex-transitive coupling graphs?~~ — supported
  directly on two shapes; not yet tested on a third topology family (§4).

Newly open:

- **Does the "full budget consumed" behavior belong to `VF2Layout`'s Rust wrapper
  specifically, or to `rustworkx.vf2_mapping()` itself?** (§3) — the direct
  `next(iter(it))`-at-varying-`call_limit` test is proposed there and not yet run.
- **Why does `dfs_corner` specifically fail while every other structured ordering
  succeeds under `id_order=True`?** (§2) — a breadth-first/depth-first mechanism is
  proposed but not independently verified; a natural next step is testing
  intermediate traversals (e.g. bounded-depth DFS, or DFS from the grid's center
  rather than a corner) to see whether the failure tracks "how thin and long the
  candidate chain gets" as a continuous variable.
- **Does the vertex-transitivity dividing line hold on a non-grid, non-vertex-
  transitive real topology** (heavy-hex) — the natural next control, not yet run
  with a perfect-matching-respecting circuit sized to actually saturate it.

**Note on the upstream position, unchanged:** none of this revisits or contradicts
what was reported upstream. These are this project's own follow-up investigations
into mechanism and generality, run well after the report was filed.

### Files

- Scripts: `benchmarks/verify_vf2_seed_anomaly_repro.py`,
  `benchmarks/verify_vf2_ordering_id_order_true.py`,
  `benchmarks/verify_vf2_call_limit_sweep.py`,
  `benchmarks/verify_vf2_toroidal_grid.py`
- Raw data (Intel, 2026-09-14): `data/vf2_seed_anomaly_repro_2026-09-14.csv`,
  `data/vf2_ordering_id_order_true_2026-09-14.csv`,
  `data/vf2_call_limit_sweep_2026-09-14.csv`,
  `data/vf2_toroidal_grid_2026-09-14.csv`

---


<!-- ===== Addendum 7 (source: spare-qubit-cliff-addendum-7-2026-09-14.md) ===== -->

> **Note added when merging:** Proposes the two-factor hypothesis (not vertex-transitive AND a perfect matching reachable to zero spare).

## Addendum (2026-09-14, third batch): heavy-hex was never actually tested, DFS has two causes, and the "burns the budget" behavior is Qiskit's, not rustworkx's

Same machine as addenda 5–6. Three more targeted follow-ups. One of them (§3) closes
the last open mechanism question from addendum-6; one (§1) corrects a real gap in
this project's own prior work rather than confirming or refuting a hypothesis; one
(§2) turns a clean binary finding into a more precise two-factor one.

### 1. Heavy-hex and the real backend map show no cliff — but the comparison with grid was not apples-to-apples, and that itself is the finding

[`verify_vf2_heavy_hex_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_heavy_hex_topology.py) fixed a real bug in [`verify_vf2_topologies.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_topologies.py)
(2026-09-14, first batch): heavy-hex (`from_heavy_hex(3)` = 19 nodes,
`from_heavy_hex(5)` = 57 nodes) and `FakeSherbrooke` (127 nodes) are all **odd**
node counts, and the original script only tried even `spare` values (0, 4) — so
`n = phys - spare` was odd every time and every row was silently skipped by the
`n % 2` guard. Addendum-5's P2 for these topologies was never actually tested,
despite reading as though it had been.

With odd `spare` values scanned properly:

| Topology | Tightest matching-having point | Loosest tested point | `opt=2` ratio | `opt=3` ratio |
| :--- | :--- | :--- | ---: | ---: |
| `grid_6x7` (control) | spare 0 (42q) | spare 4 (38q) | 49.4x | 239.0x |
| `heavy_hex_d3` | spare 3 (16q) | spare 13 (6q) | 0.9x | 1.3x |
| `heavy_hex_d5` | spare 9 (48q) | spare 39 (18q) | 1.0x | 0.9x |
| `fake_sherbrooke` | spare 19 (108q) | spare 59 (68q) | 1.1x | 1.0x |

**No cliff on any of the three real/near-real topologies, at either optimization
level — while the grid control reproduces its cliff in the same run.** Taken at
face value this refutes addendum-6 §4's "vertex-transitivity is the dividing line"
framing, because heavy-hex and `FakeSherbrooke` are *not* vertex-transitive (mixed
degree-2/3 nodes, and an irregular real device graph respectively) yet show no
cliff — the opposite of what that hypothesis predicts.

**But the comparison is not fair, and the reason why is itself informative.** For
`grid_6x7`, the tightest testable point is `spare=0` — every physical qubit used,
zero slack, and a perfect matching still exists there (the grid is dense enough).
For `heavy_hex_d3`, no perfect matching exists below `spare=3`; for `heavy_hex_d5`,
none below `spare=9`; for `fake_sherbrooke`, none below `spare=19`. **These sparse
topologies structurally cannot reach zero slack with this circuit** — the tightest
point we could test on any of them already has real breathing room, unlike the
grid's `spare=0`. So this experiment did not test "does a non-vertex-transitive,
*zero-slack* topology show a cliff" — it tested "does a non-vertex-transitive
topology with only *partial* slack show a cliff," which is a different and easier
question.

Revisiting all the topology data collected so far with this distinction in mind:

| Topology | Vertex-transitive? | Reaches zero slack w/ matching? | Cliff? |
| :--- | :---: | :---: | :---: |
| `grid_6x7`, `grid_7x8` | no | yes (spare=0) | **yes** |
| `line_42` | no | yes (spare=0) | **yes** |
| `ring_42`, `full_20` | yes | yes (spare=0) | no |
| `torus_6x7`, `torus_7x8` | yes | yes (spare=0) | no |
| `heavy_hex_d3`, `heavy_hex_d5`, `fake_sherbrooke` | no | **no** (needs real slack) | no |

**The better-supported account is now two conditions, not one:** the cliff appears
only where the coupling graph (a) is not vertex-transitive **and** (b) can actually
be driven to zero slack while a perfect matching still exists for the circuit's
interaction graph. Grid and line satisfy both and show the cliff. Ring, full-graph,
and the torus satisfy (b) but not (a), and show no cliff. Heavy-hex and the real
backend map satisfy (a) but not (b), and — consistent with this refined account —
show no cliff, for a mundane reason: they never actually get pushed to the knife's
edge that produces it. Addendum-6's single-factor "vertex-transitivity" statement is
retracted in favor of this two-factor one.

This has a practical upside worth stating plainly: **on real IBM-style heavy-hex
hardware, this specific cliff is unlikely to bite in practice for a dense-pairs-style
circuit**, not because heavy-hex is immune to the underlying VF2++ ordering issue,
but because its own sparsity means a circuit that exactly saturates the map's
matching capacity with zero spare is rarely achievable in the first place — the
device runs out of "matchable" configurations before it runs out of qubits. A truly
matched test of the vertex-transitivity claim under zero slack would need a
non-vertex-transitive topology sparse enough to resemble real hardware but still
dense enough to reach spare=0 with a matching intact — not yet found or tested.

### 2. `dfs_corner`'s failure has two independent causes, and one of them saturates with difficulty

[`verify_vf2_dfs_mechanism.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_dfs_mechanism.py) crossed 2 traversal modes (BFS, DFS) × 4 start points
(both corners, an edge midpoint, the center) × 4 neighbor-visit orders, on 6×7,
8×8, and 8×9 (96 probes total, all `id_order=True`).

Aggregated success rate by mode and start point (12 trials each):

| Start | BFS | DFS |
| :--- | ---: | ---: |
| corner (either) | 12/12 (100%) | 5/12 (42%) |
| edge midpoint | 12/12 (100%) | 2/12 (17%) |
| center | 7/12 (58%) | 0/12 (0%) |

**Both of addendum-6's candidate explanations turn out to matter, independently.**
At every fixed starting point, DFS is worse than BFS (100% → 42% at corners, 100% →
17% at the edge, 58% → 0% at the center) — so depth-first traversal itself does cost
something, as addendum-6 first proposed. But at both traversal modes, moving the
start point toward the grid's interior costs something too — BFS falls from a
perfect 100% at every non-center start to 58% at the center; DFS falls from 42% at
a corner to 0% at the center. Addendum-6's P1 (DFS fails regardless of start point)
is refuted by this — start point clearly changes DFS's success rate, from 42% down
to 0% — which is the specific outcome the script's own criterion flagged as
supporting the "corner is a special point" explanation instead. The honest reading
is that neither single-factor story is right on its own; the failure needs both
depth-first traversal *and* enough distance from a low-degree boundary point to show
up reliably.

**The per-grid breakdown adds a further wrinkle: the start-point effect shrinks as
the grid gets harder, and vanishes at 8×8.** Splitting the DFS row by grid:

| Grid | corner (either) | edge midpoint | center |
| :--- | ---: | ---: | ---: |
| 6×7 | 6/8 (75%) | 2/4 (50%) | 0/4 (0%) |
| 8×8 | 0/8 (0%) | 0/4 (0%) | 0/4 (0%) |
| 8×9 | 4/8 (50%) | 0/4 (0%) | 0/4 (0%) |

On 6×7 and 8×9, starting from a corner still rescues DFS in roughly half the
neighbor-order variants. On 8×8, **every DFS variant fails regardless of start
point** — the corner's advantage disappears entirely once the instance is hard
enough. This is consistent with corner-starting delaying, rather than preventing,
depth-first's failure: it buys some margin that a large-enough or already-difficult
grid simply consumes. It also means addendum-6's "dfs_corner uniquely fails, other
structured orders don't" framing was drawn from instances (6×7, 8×8, 8×9) that
happened to sit past that margin for the corner start already tested there (RDLU
specifically) — a different neighbor-order choice at the same corner sometimes
succeeds on the same grid (e.g. 6×7 corner_TL: `RDLU` fails, `DRUL`/`LURD`/`ULDR`
all succeed), so even "corner-DFS" is not a single well-defined outcome — the
specific first-direction choice matters too, a third factor not disentangled here.

### 3. Resolved: the "search burns its entire assigned budget" behavior belongs to Qiskit's `VF2Layout` pass, not to rustworkx's search algorithm

[`verify_vf2_rustworkx_raw_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_rustworkx_raw_sweep.py) scanned 200 random node relabelings of an 8×8
grid under `id_order=True`, using the raw lazy `rustworkx.vf2_mapping()` iterator
(`next(iter(it))`, exactly as the ordering-structure scripts already do) at
`call_limit=1,000,000` — 199 of 200 failed, one (seed 168) succeeded. That one
"medium difficulty" instance was then swept across `call_limit` from 43 to
3,000,000, alongside a known-failing control (`row_major`, `id_order=False`).

| `call_limit` | seed 168 (found once ≥1,000) | failing control (row_major, id_order=False) |
| ---: | ---: | ---: |
| 1,000 | 0.047 ms | 0.184 ms |
| 10,000 | 0.036 ms | 1.624 ms |
| 100,000 | 0.034 ms | 15.637 ms |
| 1,000,000 | 0.033 ms | 163.4 ms |
| 3,000,000 | 0.035 ms | 490.8 ms |

**P1 is confirmed cleanly.** Once past its own threshold (somewhere between 43 and
1,000), the raw rustworkx call's time is completely flat — 33–47 microseconds
regardless of whether `call_limit` is 1,000 or 3,000,000. It does not consume any of
the extra budget it's handed; ordinary Python generator semantics hold, and
`next()` stops computing the moment a match is yielded. **P2 is confirmed too:**
the failing control scales almost linearly with `call_limit` (roughly 10x limit →
roughly 10x time, throughout), exactly matching the failing-seed behavior already
seen through Qiskit's `VF2Layout` pass in addendum-6 §3.

Put next to addendum-6 §3's finding for Qiskit's `VF2Layout` pass — where seed 8,
despite needing very few calls to reach its own solution, tracked failing seed 0's
time almost exactly all the way to 3,000,000, never plateauing — **this resolves
the open question addendum-6 left standing.** The "consumes its entire assigned
budget regardless of when a solution is found" behavior is not a property of
rustworkx's underlying VF2++ search; the raw search plateaus exactly the way a
lazy iterator should. It is specific to Qiskit's `VF2Layout` pass — plausibly
because that pass does more than fetch one item from the generator (e.g. scoring
candidate mappings, building `vf2_avg_error_map`, or otherwise doing bookkeeping
tied to the requested budget rather than to how quickly a solution was found), work
this project has not read the source for closely enough to name precisely. What can
now be said with confidence: the mechanism explaining the `call_limit`-tuple
numbers throughout addenda 5–6 lives in Qiskit's own pass implementation, not in
rustworkx.

### What is still unknown, updated a third time

Resolved by this batch:

- ~~Is the spare-qubit cliff specific to non-vertex-transitive coupling graphs?~~ —
  narrowed to a two-factor account: non-vertex-transitive **and** capable of
  reaching zero slack with a matching intact. Heavy-hex and a real backend map
  satisfy the first condition but not the second, and show no cliff — consistent
  with, not contrary to, the refined account (§1).
- ~~Does `dfs_corner`'s failure come from depth-first traversal or from starting at
  a corner?~~ — both, independently, plus a third factor (the specific
  neighbor-visit order) not fully disentangled, plus the corner's protective effect
  vanishing as the grid gets harder (§2).
- ~~Does the "burn the whole budget" behavior belong to `VF2Layout` or to
  rustworkx's search itself?~~ — resolved: it is specific to Qiskit's `VF2Layout`
  pass. The raw rustworkx iterator behaves as an ordinary lazy generator (§3).

Newly open:

- **What inside `VF2Layout.run()` consumes the full budget even under
  `max_trials=1`?** (§3) — reading the pass's Rust/Python source directly (rather
  than inferring from timing) would name the specific mechanism; not yet done.
- **Is there a topology that is both non-vertex-transitive and capable of reaching
  zero slack with a matching, but sparser than a grid?** (§1) — the natural next
  control, still unfound. A candidate: a grid with a small fraction of edges removed
  at random (dilute the grid just enough to still admit a matching at spare=0, but
  break translational symmetry more than the current grids already do) or a
  "brick-wall" / offset-brick coupling pattern occasionally used for real hardware
  proposals.
- **What specifically about a given neighbor-visit order at a fixed corner start
  makes it succeed or fail?** (§2) — `RDLU` fails at 6×7's top-left corner while the
  other three rotations succeed; no mechanism proposed yet.

**Note on the upstream position, unchanged:** none of this revisits or contradicts
what was reported upstream.

### Files

- Scripts: `benchmarks/verify_vf2_heavy_hex_topology.py`,
  `benchmarks/verify_vf2_dfs_mechanism.py`,
  `benchmarks/verify_vf2_rustworkx_raw_sweep.py`
- Raw data (Intel, 2026-09-14): `data/vf2_heavy_hex_topology_2026-09-14.csv`,
  `data/vf2_dfs_mechanism_2026-09-14.csv`,
  `data/vf2_rustworkx_raw_sweep_2026-09-14.csv`

---


<!-- ===== Addendum 8 (source: spare-qubit-cliff-addendum-8-2026-09-14_1.md) ===== -->

> **Note added when merging:** **Root-cause claim in section 1 is superseded by Addendum 9**: Addendum 9 finds the cliff's actual mechanism is VF2Layout failing and switching to SabreLayout, not the VF2 search cost alone.

## spare-qubit-cliff addendum 8 (2026-09-14) -- VF2Layout's internal mechanism settled (as far as Python can reach), plus two new followups

## 0. In one line

Of addendum-7's three "still unknown" items, the first (what inside VF2Layout
actually burns the budget even with `max_trials=1`) has been settled **as far as
it can be reached from Python**, by reading the source. Two followup scripts
have been prepared for the remaining two items (whether sparsity is a third
factor, and the mechanism of neighbor-visit order) -- **neither has results
yet. The predictions below were written before any measurement.**

## 1. Solved: VF2Layout burns the budget because it is a separate, Qiskit-native implementation

Reading the source of `qiskit.transpiler.passes.layout.vf2_layout` (installed
Qiskit 2.5.2, read directly with `inspect.getsource`) shows that this pass
**does not call** the public `rustworkx.vf2_mapping()`.

```python
from qiskit._accelerate.vf2_layout import (
    vf2_layout_pass_average,
    MultiQEncountered,
    VF2PassConfiguration,
)
```

Summarized, here is everything `VF2Layout.run(dag)` does.

1. If only `coupling_map` was given, it builds one via
   `_build_dummy_target(coupling_map)` -- `Target.from_configuration(basis_gates=["u","cx"], ...)`
   -- so there is **always a concrete `Target` object** (never `None`). Every
   script in this project passes only `coupling_map`, so this branch is what
   applies.
2. It reads `self.avg_error_map = self.property_set["vf2_avg_error_map"]`.
   None of this project's scripts ever sets that key, so it is **always unset
   (effectively None)**. No error-rate-based scoring happens -- this can be
   ruled out as a cause of "burning the budget."
3. `VF2PassConfiguration.from_legacy_api(call_limit=..., time_limit=...,
   max_trials=..., shuffle_seed=..., score_initial_layout=False)` converts the
   legacy API (the arguments this project uses) into an internal configuration
   struct.
4. It calls `vf2_layout_pass_average(dag, target, strict_direction=...,
   avg_error_map=..., config=...)` **exactly once**. What comes back is the
   finished result; there is no visibility into what happens in between.

**So this was never a story about "Qiskit's wrapper misusing rustworkx."**
`qiskit._accelerate.vf2_layout` is a compiled Rust extension that Qiskit
itself builds and ships, and it is a **separate codebase** from the
`rustworkx` package. That is why the behaviour seen in followups 4 and 6/7 --
"seed 8 takes as long as a failing seed" on one side, and "the raw rustworkx
call plateaus past a threshold" on the other (followup 7) -- naturally
disagreed: two independent VF2++ implementations were being compared, not one
misusing the other.

### Why this is where it stops

`vf2_layout_pass_average` itself is compiled Rust and cannot be read with
`inspect.getsource` from the installed package. An attempt was made to locate
Qiskit's public Rust source on GitHub (presumably somewhere under
`crates/accelerate/src/`) via search, but this session's search results
returned no direct link to the file, and fetching an individual URL directly
was also blocked by provenance restrictions. **This goes beyond "what can be
reached from Python"**, so it stops here. If it is ever judged worth pursuing
this mechanism further, the option remains to `git clone` Qiskit's source
locally and read `crates/accelerate/src/vf2_layout.rs` (or whatever file
corresponds to it) directly.

## 2. Two new followups -- no results yet, only pre-registered predictions

These correspond to the remaining two items from addendum-7's "still unknown"
list.

### Followup 8: [`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py) -- is sparsity a third factor in the cliff?

Addendum-7's two-factor hypothesis was that the cliff appears when both
(a) not vertex-transitive **and** (b) a perfect matching can exist all the way
down to zero spare are satisfied -- but no sparse topology satisfying both
(a) and (b) (other than a line) had been tried. Heavy-hex is sparse but
cannot satisfy (b) (it structurally cannot reach zero spare), so heavy-hex
alone could not distinguish "sparsity itself" from "cannot satisfy (b)."

Starting from a grid (same physical qubit count), this builds a brick-laid
pattern (brick) and grids with vertical edges thinned probabilistically
(diluted_p0.25/0.5/0.75), sweeping average degree continuously between line
and grid.

**Pre-registered predictions**:
- P1: whichever candidate has a perfect matching near spare=0 shows the cliff
  regardless of average degree (supports the two-factor hypothesis --
  sparsity is not an independent factor).
- P2: the cliff's magnitude itself may shrink as average degree drops (not
  necessarily "the cliff vanishes," but a tendency toward "it gets
  shallower" is plausible).

**If this fails**: if a sparse topology is found where a perfect matching
exists but no cliff appears, the two-factor hypothesis is insufficient, and
average degree was an independent third factor.

### Followup 9: [`verify_vf2_neighbor_order_full.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_neighbor_order_full.py) -- pinning down the neighbor-visit-order mechanism with a number

Followup 6 tried only 4 neighbor-visit-order patterns (RDLU/DRUL/LURD/ULDR --
cyclic shifts of the 4 directions only). Here all 24 patterns are tried, and
additionally a "visit-order locality" metric is computed (the mean, over all
grid edges, of how close together the two endpoints land in the resulting
visit order -- `edge_locality`), to see whether it correlates with failure.

**Pre-registered predictions**:
- P1: even trying all 24 patterns, failure is skewed toward a minority (not
  all-fail or all-succeed).
- P2: orderings with higher `edge_locality` (worse locality) are more likely
  to fail and take longer.
- P3: this trend holds consistently regardless of grid shape (6x7, 8x8, 8x9, 9x9).

**If this fails**: if there is no relationship between `edge_locality` and
success rate, this hypothesis is wrong, and some other structural factor
(e.g. the parity of distance from the corner) needs to be sought instead.

**For reference only (a bare functionality check on this project's own
sandbox -- a weak 2-core Linux VM; the numbers are not meaningful)**: tried at
6x7, `call_limit=300,000`, 11 of the 24 patterns failed -- confirming that
more than a single RDLU pattern lands on the failing side (this only confirms
the code runs correctly; it is not used for the actual verdict).

## 3. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py) | followup 8's script |
| [`psf-zero/benchmarks/verify_vf2_neighbor_order_full.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_neighbor_order_full.py) | followup 9's script |

[`vf2_probe_common.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/vf2_probe_common.py) (existing, unchanged) is imported by both.

## 4. Commands to run

```
python verify_vf2_sparse_topology.py
python verify_vf2_neighbor_order_full.py
```

Run as-is with defaults, each writes a dated CSV.

## 5. Verification

- Both scripts were smoke-tested on small grids (5x5, 6x6) in the sandbox
  (2-core Linux VM) and completed without exceptions.
- [`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py): confirmed working through perfect-matching
  detection and the tight/loose ratio calculation at 6x6 (every candidate
  showed the cliff at this scale, but 6x6 is too small to use for the actual
  verdict).
- [`verify_vf2_neighbor_order_full.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_neighbor_order_full.py): confirmed 11 of 24 patterns fail at
  6x7 (see the "for reference" note above) -- confirming the code is
  detecting a real, meaningful difference rather than just returning an
  identity function.
- Pre-publication check: `grep` against this project's private personal-information pattern list, the two new
  files and this addendum -> 0 hits.

---


<!-- ===== Addendum 9 (source: spare-qubit-cliff-addendum-9-2026-09-14.md) ===== -->

> **Note added when merging:** **Key finding: identifies the cliff's real mechanism** (VF2Layout fails -> switches to SabreLayout). Confirmed on real hardware in Addendum 10.

## spare-qubit-cliff addendum 9 (2026-09-14) -- the cliff's real identity was the switch to SabreLayout; neighbor-visit order yields two new findings ("center start always fails," grid parity)

## 0. In one line

Right after followup 8's results turned up one exception with no cliff
(`diluted_p0.75`), reading the source led to **the cliff's actual identity**.
The `transpile()` cliff is not simply "slow because VF2's search is heavy" --
it is the result of a two-stage design: **"if VF2Layout fails, it switches to
a completely different algorithm called `SabreLayout`."**
`diluted_p0.75` showed no cliff not because it is "sparse," but because, for
that one particular instance, VF2Layout happened to succeed. **An urgent
followup 10 has been added to confirm this explanation (it already
reproduces cleanly in the sandbox; confirmation on real hardware is still
needed.)** Followup 9 (all 24 neighbor-visit-order patterns) produced a
robust new fact -- "a center start fails 100% of the time regardless of visit
order" -- and a new hypothesis -- "a grid fails more when both dimensions are
even, and succeeds more when both are odd."

## 1. Followup 8 ([`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py)) results

| topology | avg_deg | L2 ratio | L3 ratio | verdict |
|---|---|---|---|---|
| grid | 3.50 | 59.3x | 173.3x | cliff |
| brick | 2.62 | 64.5x | 254.7x | cliff |
| diluted_p0.25 | 2.25 | 79.9x | 261.0x | cliff |
| diluted_p0.5 | 2.50 | 67.2x | 248.7x | cliff |
| **diluted_p0.75** | **2.88** | **4.2x** | **1.3x** | **no cliff** |
| line | 1.97 | 74.4x | 321.5x | cliff |

**P1 (the cliff appears regardless of average degree) failed for
`diluted_p0.75` alone.** But this is not a simple threshold effect of "sparser
means the cliff vanishes" -- `diluted_p0.5` (degree 2.50), `diluted_p0.25`
(degree 2.25), and `line` (degree 1.97), all sparser than `diluted_p0.75`
(degree 2.88), all show the cliff cleanly. Plotting against degree shows no
monotonic trend; `diluted_p0.75` alone is an isolated exception.

### The cliff's real identity -- what reading the source revealed

Reading `qiskit.transpiler.preset_passmanagers.builtin_plugins.DefaultLayoutPassManager`
directly (installed Qiskit 2.5.2, via `inspect.getsource`) shows that the
layout decision for this project's `transpile(coupling_map=...,
optimization_level=2 or 3)` calls actually works like this.

```python
choose_layout_0 = VF2Layout(
    coupling_map=pass_manager_config.coupling_map,
    seed=-1,                              # not seed_transpiler -- always -1
    call_limit=(5_000_000, 10_000),       # optimization_level=2
    # optimization_level=3 uses (30_000_000, 100_000)
    target=pass_manager_config.target,
)
layout.append(ConditionalController(choose_layout_0, condition=_choose_layout_condition))

choose_layout_1 = SabreLayout(
    coupling_map, max_iterations=2, seed=pass_manager_config.seed_transpiler,
    swap_trials=trial_count, layout_trials=trial_count, ...
)
layout.append(ConditionalController(
    [BarrierBeforeFinalMeasurements(...), choose_layout_1],
    condition=_vf2_match_not_found,   # only runs if VF2Layout failed
))
```

In other words: **VF2Layout is tried once first, and if
`VF2Layout_stop_reason` is not `SOLUTION_FOUND`, it switches to
`SabreLayout`** (a completely different heuristic that runs an iterative
search over `swap_trials` x `layout_trials`). Every topology experiment in
this project (followups 5 through 8, the "cliffs" reported in addendum-5
through 8) was a measurement taken through this `transpile()` path.

### An important correction that had been missed -- `seed_transpiler` does not control VF2Layout

In the code above, **`VF2Layout` is hardcoded to `seed=-1`**, and
`seed_transpiler` is only passed to the `SabreLayout` side. Nearly every
script in this project implicitly assumed that writing
`transpile(..., seed_transpiler=0)` would make the result reproducible, but
**VF2Layout's internal shuffling is always random (`seed=-1`) and cannot be
fixed via `seed_transpiler`**. This is a fact that needs correcting, and it
is recorded here rather than silently edited away.

(Fortunately, every cliff verdict so far has come out robustly, at
order-of-magnitude ratios, so it is unlikely that an extremely hard or
extremely easy instance would flip on a single random draw. But for a
knife-edge-difficulty instance like this `diluted_p0.75`, this
non-determinism could sway the result.)

### Followup 10 (urgent) directly tested this explanation

A script was written to record, via `transpile(..., callback=...)`, which
passes actually ran and what `VF2Layout_stop_reason` was, and it was checked
**in this project's sandbox** (8x8, spare=0/40, optimization_level=2/3, all
12 combinations).

| topology | spare | VF2Layout | Sabre triggered |
|---|---|---|---|
| grid, brick, diluted_p0.25, diluted_p0.5, line | 0 (tight) | **fails** (`NO_SOLUTION_FOUND`, taking roughly 1.0-1.6s at L2, roughly 5-8s at L3) | **yes** |
| **diluted_p0.75** | 0 (tight) | **succeeds** (`SOLUTION_FOUND`, 72ms at L2, also fast at L3) | **no** |
| every topology | 40 (loose) | succeeds (a few ms) | no |

**It matched cleanly, as predicted.** Only `diluted_p0.75`'s tight case
succeeds immediately at VF2Layout, with no switch to Sabre -- this was the
direct reason "the cliff vanished." It succeeded independently at both L2
and L3 (two separate random shuffles, both `seed=-1`), which suggests this
graph instance is likely structurally easy to find rather than a matter of
one-off luck.

**Caveat**: the numbers above are from this project's sandbox (a weak 2-core
Linux VM), so absolute times are not meaningful. But whether
`VF2Layout_stop_reason` is success or failure is the search algorithm's own
resolved state -- a **qualitative** fact that should not depend on machine
speed -- so **confirmation on real hardware is strongly recommended**
(bundled as followup 10; not yet run on real hardware).

### Impact on the two-factor hypothesis

This finding **reinforces, rather than contradicts,** the explanation
established in addendum-6/7 that "VF2Layout burns the budget when it fails" --
the "burns the budget and fails" behaviour seen through direct calls turns
out to be exactly what is producing `transpile()`'s cliff, connecting the two
into a single line. The two-factor hypothesis itself (not vertex-transitive,
reachable down to zero spare) survives -- but the mechanism one level below
"why the cliff appears" needs refining: it is not "VF2's search cost grows
continuously," but rather the binary of "does VF2 succeed, or does it fail
and switch to Sabre."

## 2. Followup 9 ([`verify_vf2_neighbor_order_full.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_neighbor_order_full.py)) results

### New fact 1: a center start fails 100% of the time regardless of visit order (robust)

Across all 24 patterns and all 4 grids, `start=center` failed **100%, without
exception**.

| Grid | corner_TL | corner_BR | edge_mid | center |
|---|---|---|---|---|
| 6x7 | 11/24 failed | 11/24 failed | 20/24 failed | **24/24 failed** |
| 8x8 | 24/24 failed | 24/24 failed | 24/24 failed | **24/24 failed** |
| 8x9 | 15/24 failed | 15/24 failed | 22/24 failed | **24/24 failed** |
| 9x9 | **0/24 failed** | **0/24 failed** | 12/24 failed | **24/24 failed** |

Addendum-7 wrote, from only 4 patterns' worth of results, that "a center
start saturates to a 0% success rate on hard grids" -- but **even trying all
24 patterns, a center start never succeeded once, on any grid size**. This is
a stronger conclusion than "a coincidence of one particular visit order": it
is that **starting from the center under depth-first search is itself
fatally bad (regardless of visit order)**.

### New fact 2: whether a grid's two dimensions are both even or both odd flips a corner start's success rate

Looking at the failure counts for corner_TL / corner_BR:

| Grid | row x col parity | failures (of 24) |
|---|---|---|
| 8x8 | even x even | **24 (all failed)** |
| 6x7 | even x odd (mixed) | 11 |
| 8x9 | even x odd (mixed) | 15 |
| 9x9 | odd x odd | **0 (all succeeded)** |

This lines up cleanly: **both even (8x8) is worst, both odd (9x9) is best,
one of each (6x7, 8x9) is in between.** This was not among the
pre-registered predictions -- it is a new finding, and it shows this is not
a simple size effect of "bigger/harder grids fail more" (9x9 is bigger than
8x8, yet is actually easier from a corner). **However, there are only 4
grids so far, and until parity combinations like 6x6 (even-even), 7x7
(odd-odd), and 6x9 or 9x6 (a different even-odd pairing) are added, it is not
yet safe to say "parity is the operative factor."**

### P2 (higher `edge_locality` correlates with more failure) -- rejected in a naive pooled comparison, but supported within each grid

Splitting everything at the median `edge_locality` gave the **opposite** of
the prediction: the low-locality group's success rate (16.3%) was **lower**
than the high-locality group's (41.5%).

The reason is confounding -- different grids have entirely different ranges
of `edge_locality` values and different success-rate baselines (9x9 succeeds
100% even at high-locality values from a corner, while 8x8 fails 100% even
at relatively low-locality values). **Comparing raw distances pooled across
grids was itself invalid.** Comparing only within the same grid (verdict 3's
output):

| Grid | mean locality of failing orders | mean locality of successful orders |
|---|---|---|
| 6x7 | 10.66 | 9.69 (failing is higher) |
| 8x9 | 17.47 | 17.08 (failing is higher) |
| 9x9 | 20.86 | 18.76 (failing is higher) |

**All three grids go in the predicted direction** (failing orders have worse
locality). P2 is supported "within the same grid." Future cross-grid
comparisons should normalize the locality value per grid (e.g. dividing by
the grid's diameter or node count) rather than comparing raw values directly.

## 3. Updated list of what remains unknown

1. **[Top priority, new]** Followup 10's explanation (cliff = VF2 failure ->
   switch to Sabre; `diluted_p0.75` shows no cliff because VF2 succeeds) has
   only been confirmed in this project's sandbox. **Followup 10 needs to be
   run on real hardware to confirm.**
2. **[New, important]** How much can the fact that `seed_transpiler` does not
   control VF2Layout's random seed (fixed at `seed=-1`) sway the results
   reported for borderline (knife-edge-difficulty) instances among the
   cliffs reported in addendum-5 through 8? A followup is needed that repeats
   `transpile()` several times against the same topology and the same
   `seed_transpiler=0`, to check whether the tight-side cliff actually
   flickers on and off.
3. The relationship between grid row/column parity and the success rate of a
   depth-first, corner-started search -- 6x6 (even-even), 7x7 (odd-odd), and
   6x9 or 9x6 (a different mixed pairing) need to be added to fill out every
   parity combination.
4. [Carried over] A normalization method that makes `edge_locality`
   comparable across grids.

## 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/vf2_sparse_topology_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_sparse_topology_2026-09-14.csv) | followup 8's results (provided by the user) |
| [`psf-zero/data/vf2_neighbor_order_full_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_neighbor_order_full_2026-09-14.csv) | followup 9's results (provided by the user) |
| [`psf-zero/benchmarks/verify_vf2_pipeline_trace.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_pipeline_trace.py) | followup 10 (new, urgent) |

## 5. Command to run

```
python verify_vf2_pipeline_trace.py
```

Run as-is with defaults (grid=8x8, levels=2 3, spares=0 40 -- the same
combination followup 8 actually used to judge the cliff), it writes a dated
CSV.

## 6. Verification

- One bug was nearly introduced while writing the script: writing
  `"SOLUTION_FOUND" in row.VF2StopReason` in the verdict section would also
  match `NO_SOLUTION_FOUND` (the same pattern as a known trap this project
  has hit several times before). Fixed to `.endswith(".SOLUTION_FOUND")`
  before distributing it.
- Confirmed to complete on 6x6, 8x8, and both `optimization_level` values in
  the sandbox (2-core Linux VM). Results matched the P1/P2 predictions
  (only `diluted_p0.75`'s tight case succeeds at VF2Layout).
- Pre-publication check: `grep` against this project's private personal-information pattern list, the one new
  file and this addendum -> 0 hits (excluding existing cautionary strings).

---


<!-- ===== Addendum 10 (source: spare-qubit-cliff-addendum-10-2026-09-14.md) ===== -->

> **Note added when merging:** Confirms Addendum 9's mechanism on real hardware; finds a second cost layer (downstream routing/optimization after the Sabre fallback) at L3.

## spare-qubit-cliff addendum 10 (2026-09-14) -- followup 10 confirmed on real hardware: supports the "cliff = VF2 failure -> switch to Sabre" explanation, and additionally finds that, at L3, "the routing/optimization cost after the switch" is itself part of the cliff

## 0. In one line

The prediction made in addendum-9 (the cliff's real identity is VF2Layout
failing and switching to SabreLayout; `diluted_p0.75` alone shows no cliff
because VF2Layout succeeds) **matched completely on the user's real
hardware.** All 24 rows matched as predicted. On top of that, looking at the
real hardware data revealed a quantitative additional finding: at
`optimization_level=3`, there are cases where **"the time it takes for the
subsequent swap-insertion/optimization passes to handle the imperfect layout
Sabre chose" is larger than "the time VF2Layout burns by failing."**

## 1. Confirmed on real hardware -- P1/P2/P3, all 24 rows match

| topology | spare | VF2Layout (L2) | VF2Layout (L3) | Sabre triggered |
|---|---|---|---|---|
| grid | 0 (tight) | NO_SOLUTION_FOUND | NO_SOLUTION_FOUND | yes |
| brick | 0 (tight) | NO_SOLUTION_FOUND | NO_SOLUTION_FOUND | yes |
| diluted_p0.25 | 0 (tight) | NO_SOLUTION_FOUND | NO_SOLUTION_FOUND | yes |
| diluted_p0.5 | 0 (tight) | NO_SOLUTION_FOUND | NO_SOLUTION_FOUND | yes |
| **diluted_p0.75** | 0 (tight) | **SOLUTION_FOUND** | **SOLUTION_FOUND** | **no** |
| line | 0 (tight) | NO_SOLUTION_FOUND | NO_SOLUTION_FOUND | yes |
| every topology | 40 (loose) | SOLUTION_FOUND | SOLUTION_FOUND | no |

This matches the sandbox confirmation exactly. **Addendum-9's explanation
that "the cliff vanished not because it is sparse, but because VF2Layout
happened to succeed on this one instance" holds on real hardware too.**

`diluted_p0.75` succeeded on both of two independent `transpile()` calls
(L2 and L3 -- since `VF2Layout` is fixed at `seed=-1`, each is a separate
random shuffle), and combined with the sandbox's L2 and L3, that makes
**4 out of 4** landing on the same conclusion (this graph's particular
instance is easy for VF2, everything else is hard). This strengthens the
view that it is likely not one-off luck but that this specific graph
realization is structurally easy.

## 2. A new quantitative finding -- at L3, the "post-switch" cleanup cost is not negligible

Computing "downstream" (= Total - VF2 - Sabre, effectively the sum of the
subsequent swap-insertion, routing, and optimization passes) from the CSV's
`TotalTime_s` / `VF2Time_s` / `SabreTime_s`:

| topology | spare | level | VF2 (s) | Sabre (s) | downstream (s) | note |
|---|---|---|---|---|---|---|
| grid | 0 | 2 | 0.697 | 0.002 | 0.975 | downstream is larger |
| grid | 0 | 3 | 4.219 | 0.000 | 4.109 | roughly even |
| brick | 0 | 2 | 0.704 | 0.001 | 0.011 | small |
| **brick** | 0 | **3** | 4.141 | 0.000 | **12.064** | **downstream is about 3x the VF2-failure cost** |
| diluted_p0.25 | 0 | 2 | 2.676 | 0.004 | 0.058 | small |
| diluted_p0.25 | 0 | 3 | 14.496 | 0.006 | 9.302 | large (well over half the VF2-failure cost) |
| diluted_p0.5 | 0 | 2 | 0.716 | 0.000 | 0.020 | small |
| diluted_p0.5 | 0 | 3 | 4.267 | 0.003 | 4.305 | about the same |
| line | 0 | 2 | 0.614 | 0.000 | 0.029 | small |
| line | 0 | 3 | 3.815 | 0.015 | 3.765 | about the same |

**At `optimization_level=2` the downstream cost is essentially negligible
(under a few dozen ms, except grid), but at `optimization_level=3` there are
cases where the downstream cost equals or exceeds the time VF2Layout spent
burning its budget.** Notably, for `brick` at L3, the subsequent processing
(12.1s) is nearly 3x larger than VF2Layout's failure itself (4.1s).

### What this means

Up through addendum-9, the cliff has been explained as "VF2Layout fails and
burns the budget," but the real-hardware L3 data shows **there is another
layer of cost on top of that** -- because the layout SabreLayout chooses is
imperfect, the subsequent routing (swap-gate insertion) and the circuit
optimization passes that follow it end up doing heavier work than usual.
This is a **second contributing factor** to the cliff, and at a heavy
optimization level like L3 it can carry as much weight as, or more than, the
search cost of VF2 itself.

**Caveat**: this table is computed from a single run per condition; no
repeated measurement was taken ([`verify_vf2_pipeline_trace.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_pipeline_trace.py) does not use
`timed()`). The downstream-side numbers may be more susceptible to
measurement noise, so read this as a trend (small at L2, sometimes large at
L3) rather than placing too much confidence in any individual figure
(especially `brick`'s 3x ratio).

## 3. Updated list of what remains unknown

1. **[Solved]** The explanation "cliff = VF2 failure -> switch to Sabre,
   `diluted_p0.75` = VF2 success" has been confirmed on real hardware
   (supports addendum-9's P1/P2/P3).
2. **[New]** How the downstream (swap-insertion/optimization) cost varies by
   topology and optimization level. Whether `brick`'s outsized L3 figure
   (3x the VF2-failure cost) is a real effect or measurement noise is worth
   confirming with repeated measurements.
3. [Carried over] The grid row/column parity hypothesis (to be confirmed by
   adding 6x6, 7x7, 6x9/9x6).
4. [Carried over, still reassuring so far] `seed=-1`'s non-determinism --
   `diluted_p0.75` landing on the same conclusion 4 out of 4 times (sandbox
   L2/L3, real hardware L2/L3) is reassuring, but whether other
   knife-edge-difficulty instances are similarly stable remains unconfirmed.

## 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/vf2_pipeline_trace_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_pipeline_trace_2026-09-14.csv) | followup 10's real-hardware results (provided by the user) |

## 5. Verification

- The real-hardware data's `Downstream_s = TotalTime_s - VF2Time_s - SabreTime_s`
  was recomputed and confirmed with pandas (matches the table above).
- One row, `line` at spare=40, L2, has `Downstream_s` slightly negative at
  -0.0038 -- `VF2Time_s` (0.0152s) is marginally larger than `TotalTime_s`
  (0.0114s). This is noise from timing misalignment in the callback
  (`perf_counter` granularity, rounding of per-pass execution time) and is
  treated as effectively zero. It does not affect the interpretation of the
  other figures.
- Pre-publication check: `grep` against this project's private personal-information pattern list, the one new
  file and this addendum -> 0 hits.

---


<!-- ===== Addendum 11 (source: spare-qubit-cliff-addendum-11-2026-09-14.md) ===== -->

## spare-qubit-cliff addendum 11 (2026-09-14) -- code for the remaining two followups (seed=-1 non-determinism, grid parity hypothesis)

## 0. In one line

The cliff's "cause" itself (VF2Layout failing -> switching to SabreLayout) was
already confirmed on real hardware in addendum-10. After discussing with the
user, of the three remaining secondary questions, two were chosen for
followup: **"seed=-1's non-determinism"** and **"the grid-parity
hypothesis"** ("reproducibility of the downstream cost" is set aside for now).
Below are only the predictions written before measurement -- there are no
results yet.

## 1. Followup 11: [`verify_vf2_seed_nondeterminism.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_seed_nondeterminism.py) -- how much does VF2Layout's seed=-1 actually jitter?

Following the fact discovered in addendum-9 -- that
`transpile(..., seed_transpiler=0)` does not control `VF2Layout`'s own seed
(fixed at `seed=-1`) -- this **calls transpile() repeatedly against the same
topology and same circuit, and counts how often `VF2Layout_stop_reason`
actually flips.**

The target is spare=0 (tight) on the 6 topologies used in followups 8/10
(grid, brick, diluted_p0.25/0.5/0.75, line). By default only
optimization_level=2 is run, 15 repetitions each (an L3 failing case takes
roughly 20 seconds per call, so the design requires explicitly passing
`--levels 2 3` to include it).

**Pre-registered predictions**:
- P1: `grid` / `brick` / `diluted_p0.25` / `diluted_p0.5` / `line` fail
  almost every time on repetition (success rate ~=0%).
- P2: `diluted_p0.75` succeeds almost every time on repetition (success rate
  ~=100%).
- P3: if either fails and some candidate's success rate lands intermediate,
  that calls into question the robustness of the other topology experiments
  (heavy-hex, torus, ring, full, etc.), which were each measured only once.

**If this fails**: if neither P1 nor P2 fails (everything splits cleanly
into 0% or 100%), `seed=-1`'s non-determinism can be said not to be a
practical problem except at knife-edge difficulty.

**Sandbox functionality check** (8x8, L2, 3 repetitions only, not used for
the actual verdict): grid/brick/diluted_p0.25/diluted_p0.5/line all failed
3/3, `diluted_p0.75` all succeeded 3/3 -- in the predicted direction,
confirming the code runs correctly.

## 2. Followup 12: [`verify_vf2_grid_parity.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_grid_parity.py) -- filling out the grid-parity hypothesis with a 2x2

This confirms the pattern found in followup 9 -- "8x8 (even x even) fails
100% from a corner, 9x9 (odd x odd) succeeds 100%, 6x7/8x9 (one of each) are
intermediate" -- by filling out each of the 2x2 parity combinations with two
examples:

- even x even: `6x6`, `8x6`
- odd x odd: `7x7`, `9x7`
- even x odd: `6x7` (already covered in followup 9, repeated here as a control), `6x9`
- odd x even: `7x6`, `9x6`

To save time, only corner starts (corner_TL, corner_BR) are covered
(edge_mid and center already have adequate results from followup 9).

**Pre-registered predictions**:
- P1: even x even grids (6x6, 8x6) have a high corner-start failure count.
- P2: odd x odd grids (7x7, 9x7) have a low corner-start failure count.
- P3: even x odd / odd x even grids (6x7, 6x9, 7x6, 9x6) have an
  intermediate failure count.
- P4: behaviour is unchanged under transpose (6x7 and 7x6, 6x9 and 9x6, give
  similar results).

**If this fails**: if P1 or P2 fails (e.g. 8x6 turns out all-succeed like
9x9), the "parity" explanation is wrong, and some other factor (one side
having length 8, overall area, etc.) should be suspected instead. If P4
fails, that is a new clue that an asymmetric factor -- which of rows or
columns is "closer to the corner" -- is at work.

**Sandbox functionality check** (only 6x6 and 7x7, scaled down with
`call_limit=300,000`, not used for the actual verdict): 6x6 (even x even)
failed 22/24 from a corner, 7x7 (odd x odd) failed 0/24 from a corner --
already the same direction as the pattern seen in 8x8/9x9. Confirms the code
runs correctly and detects a meaningful difference.

## 3. Updated list of what remains unknown

1. [On hold, set aside for now] Reproducibility of the downstream cost
   (the case where `brick` at L3 was 3x). Agreed with the user to prioritize
   this below the two items above.
2. [In progress] `seed=-1`'s non-determinism (followup 11, awaiting results).
3. [In progress] The grid-parity hypothesis (followup 12, awaiting results).

## 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/verify_vf2_seed_nondeterminism.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_seed_nondeterminism.py) | followup 11 |
| [`psf-zero/benchmarks/verify_vf2_grid_parity.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_grid_parity.py) | followup 12 |

Both import [`vf2_probe_common.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/vf2_probe_common.py), [`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py), and
[`verify_vf2_neighbor_order_full.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_neighbor_order_full.py) (existing, unchanged). Keep them in the
same folder.

## 5. Commands to run

```
python verify_vf2_seed_nondeterminism.py
python verify_vf2_grid_parity.py
```

To also check L3 in followup 11: `python verify_vf2_seed_nondeterminism.py
--levels 2 3` (an L3 failing case takes roughly 20 seconds each, so even
with the default `reps-l3=5` this is expected to take a few minutes).

## 6. Verification

- Both scripts were confirmed to complete on scaled-down settings in the
  sandbox (2-core Linux VM) -- followup 11 at 8x8, reps=3, L2 only; followup
  12 at 6x6 and 7x7 only, `call_limit=300,000`. Both produced results in the
  predicted direction, confirming the code detects a meaningful difference
  rather than acting as a bare identity function.
- Pre-publication check: `grep` against this project's private personal-information pattern list, the two new
  files and this addendum -> 0 hits (excluding existing cautionary strings).

---


<!-- ===== Addendum 12 (source: spare-qubit-cliff-addendum-12-2026-09-14.md) ===== -->

> **Note added when merging:** Closes the seed=-1 non-determinism and grid-parity followups; corrects the transpose-symmetry (P4) claim to a design necessity.

## spare-qubit-cliff addendum 12 (2026-09-14) -- both followups 11 and 12 confirmed as predicted; these two items are closed here

## 0. In one line

**Both followup 11 (seed=-1's non-determinism) and followup 12 (grid parity)
had their pre-registered predictions confirmed cleanly.** Following the
cliff's real identity (addendum-9/10), these two secondary questions are now
also settled, so **it is recommended to close out these two items here**
(reasons in section 3).

## 1. Followup 11 results -- seed=-1's jitter does not sway the outcome at this scale of difficulty gap

| topology | successes/trials | success rate |
|---|---|---|
| grid | 0/15 | 0% |
| brick | 0/15 | 0% |
| diluted_p0.25 | 0/15 | 0% |
| diluted_p0.5 | 0/15 | 0% |
| **diluted_p0.75** | **15/15** | **100%** |
| line | 0/15 | 0% |

**90 out of 90 trials landed cleanly at 0% or 100%, exactly as predicted.**
Not a single candidate landed at an intermediate success rate. **Both P1 and
P2 are fully supported.** This means `diluted_p0.75`'s property of "VF2Layout
succeeds" has now agreed across every one of 32 total independent random
trials -- sandbox L2/L3, real hardware L2/L3, and 15 repetitions on real
hardware. This is close to conclusive that this is not one-off luck, and that
this specific graph realization is structurally easy.

**A remaining caveat**: the 6 topologies tried here all happened to be either
"extremely easy" or "extremely hard," with none being a "knife-edge
difficulty" instance. So whether stability holds even at knife-edge
difficulty has not been directly confirmed. However, what matters
practically is "whether the specific candidates judged cliff-present or
cliff-absent in addendum-5 through 10 are stable," and this round gives an
adequate answer to that.

## 2. Followup 12 results -- the parity hypothesis is confirmed. But transpose symmetry (P4) turned out to be a design necessity

| parity | failure rate (of 96) |
|---|---|
| even_even (6x6, 8x6) | **93.8%** (90/96) |
| even_odd (6x7, 6x9) | 47.9% (46/96) |
| odd_even (7x6, 9x6) | 47.9% (46/96) |
| odd_odd (7x7, 9x7) | **0.0%** (0/96) |

**P1, P2, and P3 are all fully supported.** The lineup -- even x even worst
(over 90% failing), odd x odd best (never fails), one of each in between --
was confirmed consistently across all 8 grids.

### A correction regarding P4 (transpose symmetry) -- this was a necessity of the experimental design, not new evidence

It was written that 6x7 and 7x6, and 6x9 and 9x6, matched exactly (identical
failure counts) "as predicted," but checking it by calculation as a
precaution found that **this was not a property of the grid, but a
mathematical necessity following directly from the experimental design of
trying all 24 patterns.**

Specifically: transposing a grid (swapping rows and columns) exactly swaps
the roles of the neighbor directions "right" and "down," and "left" and
"up." Since followups 9 and 12 in this project try **all 24 patterns** of
the neighbor-visit order (every permutation of the 4 directions, with none
omitted), applying this swap does not change the set of 24 patterns as a
set (it only reorders them). So the set of results for 6x7 and the set of
results for 7x6 are **guaranteed by design to match** -- this holds always,
regardless of how the grid actually behaves. (Confirmed in code by directly
comparing `traverse()`'s output: the corresponding permutation of 7x6, with
directions swapped, produces exactly the same visit order as 6x7's, just
with coordinates transposed -- confirmed with zero mismatches.)

**What this means**: of followup 12's 8 grids, `7x6` and `9x6` carry no
information independent of `6x7` and `6x9` (being transposes, they
automatically give the same result). That is, **the number of effectively
independent data points is not 8 but 6** (even_even: 6x6, 8x6 / odd_odd:
7x7, 9x7 / mixed: 6x7, 6x9). Even so, all 6 are consistent in supporting the
direction of P1/P2/P3, so the conclusion itself is unaffected -- but the
phrasing "also confirmed unchanged under transpose" is withdrawn, and
recorded correctly instead as "invariance under transpose is guaranteed by
the experimental design." (Nothing in addendum-5 through 11 contains an
incorrect conclusion needing correction; only P4's phrasing needed fixing.)

## 3. Recommendation to close out these two items here

Reasons:

- Followup 11: predictions matched completely, 90/90, with not a single
  intermediate result. Repeating the same design further is unlikely to
  yield new information.
- Followup 12: predictions matched completely on P1/P2/P3, with the
  direction consistent across the effectively 6 independent data points.
  Since P4 turned out to be a design necessity, there is no longer a reason
  to pursue "transpose symmetry" further.

Both are secondary questions relative to the cliff's main story (VF2Layout
failure -> switch to SabreLayout, settled in addendum-9/10), and this round
is judged to have answered them adequately. **If pursued further**, the next
questions would be more involved ones like these, though neither would
change the current understanding of the two-factor hypothesis or the
cliff's identity, so priority is considered low:

- The mathematical reason why grid row/column parity governs a
  corner-started DFS's success rate (its relationship to graph automorphisms
  or symmetry in the degree sequence).
- Deliberately searching for a "knife-edge difficulty" instance and checking
  whether `seed=-1`'s non-determinism actually sways the outcome there (the
  same idea as followup 7's "medium difficulty" search).

## 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/vf2_seed_nondeterminism_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_seed_nondeterminism_2026-09-14.csv) | followup 11's real-hardware results (provided by the user) |
| [`psf-zero/data/vf2_grid_parity_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_grid_parity_2026-09-14.csv) | followup 12's real-hardware results (provided by the user) |

## 5. Verification

- The claim that P4's "transpose symmetry is a design necessity" was
  confirmed with code that directly compares `traverse()`'s output (each
  permutation for 6x7 and the corresponding direction-swapped permutation
  for 7x6 produce exactly the same visit order once coordinates are
  transposed, with 0 mismatches found).
- That even_odd/odd_even's failure rates matched exactly (46/96 each) is
  also mathematically guaranteed by the same structural reason above -- it
  did not happen to coincide by measurement error.
- Pre-publication check: `grep` against this project's private personal-information pattern list, this addendum
  -> 0 hits (excluding existing cautionary strings).

---


<!-- ===== Addendum 13 (source: spare-qubit-cliff-addendum-13-2026-09-14.md) ===== -->

> **Note added when merging:** Introduces the [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) prototype. **Caveat carried forward through Addendum 16**: every finding is against the public `rustworkx.vf2_mapping()`, not Qiskit's internal implementation.

## spare-qubit-cliff addendum 13 (2026-09-14) -- the layout-search prototype [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py): what building it revealed (sandbox only, not yet confirmed on real hardware)

## 0. In one line

Following a suggestion that "today's results might let us design the best
possible search-based compiler," a layout-search prototype,
[`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py), was built with an eye toward integrating it into
PSF-Zero. **It works** (it can find a valid layout under conditions where
Qiskit's default pipeline fails, such as grid and line), **but its
effectiveness is more limited than hoped** -- the "BFS is robust" finding
from addenda 9 and 12 turns out to be specific to pure grid physical graphs,
and does not generalize as-is to sparse topologies like addendum-8's brick
and diluted_p. A hard-to-notice bug was also found and fixed during
implementation. **This is confirmed only in the sandbox so far; real
hardware testing and integration into Qiskit proper are still ahead.**

## 1. What was built

[`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) (implemented directly on top of the public
`rustworkx.vf2_mapping()`; Qiskit's internal implementation,
`qiskit._accelerate.vf2_layout`, was not touched -- see section 4 for why).
Design:

- Before calling VF2, a cheap feasibility pre-check using `networkx`'s
  maximum-cardinality matching (VF2 is never called at all if it is already
  known that no solution can exist).
- Stage 1: tries several cheap BFS-family node orderings (starting from the
  max-degree node, the min-degree node, node 0, degree-descending order, and
  two random-seed starting points), each with a small `call_limit` (default
  50,000) and `id_order=True`.
- Stage 2 (added during this implementation, see section 3): if stage 1
  finds nothing and time budget remains, tries `id_order=False` (VF2's
  built-in heuristic ordering) across several starting orderings, with a
  larger `call_limit` (default 2,000,000).
- Stops as soon as one succeeds. Returns `(layout_map, info)` (`info` holds
  diagnostics such as which orderings were tried, which stage, and time
  taken).

## 2. A bug found and fixed -- the mapping direction was mixed up

Per the official docstring, `rustworkx.vf2_mapping(first, second, ...)`
returns a mapping "**from first's node indices to second's node indices**."
The first version of the implementation misread this as the reverse
(`{im_idx: physical_idx}`).

When the number of physical qubits equals the number of logical qubits
(spare=0, which is exactly this test condition), this mix-up does not raise
a `KeyError` -- it **silently returns the wrong physical qubits**. This was
a hard-to-notice kind of bug: it looks like it's working, but the resulting
layout is invalid.

Past scripts in the `vf2_probe_common` family only ever checked "was a
mapping found or not" and never actually used the mapping's contents, so this
direction mistake surfacing in this project happened for the first time here.

**How it was discovered**: adding code to the smoke test that directly
validates whether the found layout actually places logical pairs on physical
edges revealed that 14 of the pairs, out of 64 qubits, were assigned to
invalid edges. Checking `vf2_mapping`'s input and output directly on a small
example from `rustworkx.generators.path_graph` identified the direction
mix-up, which was then fixed. After the fix, every test case showed 0
invalid pairs and 0 duplicate physical-qubit assignments.

**Lesson**: rather than only the true/false of "found / not found," **write
code that directly validates whether the contents of what was found are
correct** -- recorded here as an example of this project's recording and
verification discipline actually catching a bug.

## 3. An unexpected finding -- "BFS is robust" turned out to be specific to grid

Addenda 9 and 12 concluded, for **pure grid** physical graphs like 8x8 and
9x9, that "depth-first search (DFS) is fragile with respect to starting
point and row/column parity, while breadth-first search (BFS) is far more
robust." This prototype initially took that finding at face value and was
built using only BFS-family orderings + `id_order=True` (stage 1 alone).

Applying this to addendum-8's brick and diluted_p0.25/0.5/0.75 (8x8,
spare=0, tight) as well found:

| topology (avg_deg) | stage 1 (6 BFS orderings, id_order=True, limit 50K-10M) | Qiskit default pipeline (L2) result |
|---|---|---|
| grid (3.50) | succeeds instantly (under one step) | fails (as in addendum-9/10) |
| line (1.97) | succeeds instantly (under one step) | fails (as in addendum-9/10) |
| brick (2.62) | **all fail** (nothing found even at limit 10M) | fails (as in addendum-8 -- cliff present) |
| diluted_p0.25 (2.25) | **all fail** (nothing found even at limit 10M) | fails (as in addendum-8 -- cliff present) |
| diluted_p0.5 (2.50) | **all fail** (nothing found even at limit 10M) | fails (as in addendum-8 -- cliff present) |
| diluted_p0.75 (2.88) | **all fail** (nothing found even at limit 10M) | **succeeds** (as in addendum-8 -- no cliff) |

**The diluted_p0.75 row was the most surprising part**: even though
Qiskit's own VF2Layout (the internal implementation, a random `seed=-1`
shuffle, `call_limit=5,000,000`) succeeds 100% of the time (confirmed across
32 trials on real hardware and in the sandbox in addendum-10/11), against the
public rustworkx on the exact same physical graph and the same logical
interaction pattern, **a fixed BFS-family ordering with `id_order=True`
found nothing even with `call_limit` raised to 10 million**.

Trying `id_order=False` (VF2's built-in heuristic ordering, which dynamically
picks nodes based on degree and similar criteria) as a diagnostic found:

- diluted_p0.75: setting the starting ordering to "degree-descending" with
  `id_order=False` found a solution at `call_limit=1,000,000`, in 0.05
  seconds. But other starting orderings (BFS from the max/min-degree node,
  etc.) still found nothing under the same `id_order=False` -- so even in
  heuristic mode, the initial node numbering is not irrelevant.
- brick: raising `id_order=False`'s budget to roughly Qiskit's L3 level
  (`call_limit=30,000,000`) and trying several starting orderings **still
  found nothing**.

**Interpretation (tentative)**: when the logical interaction pattern is "a
set of disjoint edges" (exactly the structure this project's
`build_dense_pair_blocks_circuit` produces), a VF2 search under a fixed
node-visit order is likely prone to exponential backtracking, because a
greedy assignment for some components can conflict with the assignment of
others. VF2's built-in heuristic can work structurally in favour of this
kind of pattern, but it still cannot solve a candidate that is inherently
hard (brick) -- this **does not contradict, and rather independently
corroborates,** addendum-8's conclusion that "brick shows the cliff."

This finding is what led to adding stage 2 (the `id_order=False` fallback) to
the prototype. The results after adding it are already reflected in the
table above's columns (diluted_p0.75 is now caught by stage 2; brick and
diluted_p0.25/0.5 still fail across the board even with stage 2 added --
this is taken as confirmation of "inherently hard," not a "bug").

## 4. Important caveats (remaining unverified)

1. **Limited to the public rustworkx**: every finding here (both stage 1 and
   stage 2) was confirmed against the **public API**,
   `rustworkx.vf2_mapping()`. What Qiskit's `VF2Layout` pass actually calls
   is `qiskit._accelerate.vf2_layout.vf2_layout_pass_average`, a **separate
   compiled Rust module** (a discovery from addendum-9), and whether the
   ordering effects and stage-2 benefit found here reproduce there too is
   unverified.
2. **Sandbox only**: every experiment here was run on a 2-core Linux VM.
   Reproduction on real hardware (Windows, Intel machines) has not yet been
   confirmed.
3. **Connecting to `transpile(initial_layout=...)` is unimplemented and
   unverified**: `smart_vf2_layout()` only goes as far as returning
   `layout_map`; actually passing this into
   `transpile(qc, coupling_map=cmap, initial_layout=layout_map, ...)` and
   measuring the resulting final circuit-generation time and quality (the
   part corresponding to the "downstream cost" seen in addendum-10) has not
   been done yet.
4. **Integration into PSF-Zero proper has not started**: this is a
   standalone prototype only; integration into [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py) or the Rust
   core (`psf_zero_core`) has not been done.

## 5. How to read this result -- an honest assessment

The good part: the prototype was able to find a valid layout, under
conditions where Qiskit's default pipeline fails (grid, line, and
diluted_p0.75's tight spare), using a much smaller budget than default
VF2Layout (a few million calls at most, and often under tens of thousands).
A bug found during implementation was fixed, and a mechanism to directly
validate layout correctness was added.

The limited part: the design philosophy of "try several cheap orderings"
itself does not work on an inherently hard topology like brick -- this
result directly corroborates addendum-8's existence of the cliff, which is
not surprising, but against the original hope of "the best possible
search-based compiler," the honest assessment is that **this is not a
universal solution, but more a tool at the level of "catches cases that were
solvable given the right ordering but were not visible with the current
implementation."**

## 6. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/prototypes/psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) | the prototype itself |
| [`psf-zero/prototypes/smoke_test_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/smoke_test_smart_layout.py) | smoke test (grid/brick/diluted_p x3/line, plus the feasibility pre-check) |

## 7. Next steps (not yet started; priority to be discussed with the user)

1. Connect `smart_vf2_layout()`'s result to
   `transpile(initial_layout=...)` and write a benchmark comparing total
   time against Qiskit's default pipeline.
2. Confirm whether stage 2's `id_order=False` search shows the same trend
   on real hardware.
3. Check whether the same ordering effects and stage-2 benefit reproduce in
   Qiskit's internal implementation, `_accelerate.vf2_layout` (the single
   most important unverified item, which decides whether this prototype
   can genuinely function as a replacement for Qiskit's own `VF2Layout`
   pass).
4. Consider whether an approach other than VF2 (e.g. greedy matching plus
   local repair) should be tried for inherently hard cases like brick.

## 8. Verification

- In the smoke test, both the conditions already confirmed to make Qiskit's
  default pipeline fail (grid/brick/diluted_p0.25/0.5/line, 8x8, spare=0,
  L2) and the one condition that succeeds (diluted_p0.75) were reproduced
  first, and then `smart_vf2_layout()` was run under the same conditions.
- The layout found was validated by reading the mapping's contents directly
  to confirm that every logical pair lands on an edge of the physical
  coupling map, and that there are no duplicate physical-qubit assignments
  (the bug in section 2 was discovered through this check).
- Confirmed that the feasibility pre-check returns False immediately,
  without calling VF2, on a small example where no solution can obviously
  exist (a 3-qubit triangle asked to satisfy a 4-qubit requirement).
- Pre-publication check: `grep` against this project's private personal-information pattern list, this addendum,
  [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py), and [`smoke_test_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/smoke_test_smart_layout.py) -> 0 hits.

---


<!-- ===== Addendum 14 (source: spare-qubit-cliff-addendum-14-2026-09-14.md) ===== -->

> **Note added when merging:** Sandbox-only dry run of the prototype-vs-default benchmark; no real-hardware results yet.

## spare-qubit-cliff addendum 14 (2026-09-14) -- followup 14: prototype vs. default pipeline, a total-time contest including PSF-Zero (no results yet)

## 0. In one line

A benchmark was prepared that races addendum-13's layout-search prototype
against the default pipeline on total time, **in the actual usage form that
includes PSF-Zero (`compile_for_hardware`)**. **This is a contest the
prototype can lose**, and the time taken when the search fails is not
hidden -- it is included in the total without exception. The predictions
below were written before measuring, and **there are no real-hardware
results yet**.

## 1. What is raced against what

| Arm | Contents |
|---|---|
| `qiskit_opt2` | plain `transpile(optimization_level=2)` (the cliff's baseline) |
| `qiskit_opt2_smart` | `smart_vf2_layout()` -> if found, `transpile()` with `initial_layout` specified; if not found, plain `transpile()` |
| `psf_rl2` | `compile_for_hardware(routing_optimization_level=2, verify=False, entangling_basis="cx")` (PSF-Zero's current flow) |
| `psf_rl2_smart` | the same, but passing `smart_vf2_layout()`'s result as `initial_layout` |

**The smart arm includes the time spent searching in the total, even when
the search fails.** Hiding the failure cost would make this comparison
meaningless.

The Qiskit-only pair (rows 1 and 2) is also measured so that, if a
difference appears, it can be separated into "thanks to the layout search"
versus "something specific to PSF-Zero's side."

## 2. Pre-registered predictions (**written before measuring**)

- **(P1)** On tight `grid`, `line`, and `diluted_p0.75`, the smart arm is
  **faster** than the default arm (wins on total time).
- **(P2)** On tight `brick`, `diluted_p0.25`, and `diluted_p0.5`, the smart
  arm **loses**. The margin of loss should be roughly equal to the search's
  time budget (2.0 seconds by default) -- because the default pipeline ends
  up running anyway once the search fails entirely.
- **(P3)** At loose (spare=40, n=24), the difference is small across every
  topology. Since VF2Layout succeeds instantly even by default, the smart
  arm is only **slightly slower** due to the extra pre-check and first
  search attempt.
- **(P4)** Output circuit quality (2-qubit gate count, depth) is **equal to
  or better than** the default arm in cases where the smart arm wins.

**If this fails**:
- If P2 fails and smart wins even on `brick`, that means the failure-case
  cost is smaller than expected, and it would be worth raising
  `--time-budget` further.
- If P4 fails and quality drops, that would mean "faster but lower
  quality," and this whole direction would need reconsidering.

## 3. Sandbox functionality check (**not used for the actual verdict**)

A 2-core Linux VM, without the PSF arm (`psf_zero_core` is absent, so it was
automatically skipped), `--reps 1`, taking about 70 seconds. This confirms
the code runs and detects a meaningful difference:

| topology | spare | qiskit_opt2 | qiskit_opt2_smart | ratio | search succeeded |
|---|---|---|---|---|---|
| grid | 0 | 1067.5 ms | 64.1 ms | **16.66x** | True |
| line | 0 | 886.1 ms | 26.8 ms | **33.08x** | True |
| brick | 0 | 936.0 ms | 2268.9 ms | 0.41x | False |
| diluted_p0.25 | 0 | 1079.4 ms | 2892.9 ms | 0.37x | False |
| diluted_p0.5 | 0 | 1062.5 ms | 2916.3 ms | 0.36x | False |
| diluted_p0.75 | 0 | 66.8 ms | 144.5 ms | 0.46x | True |
| (loose spare=40, all 6 candidates) | 40 | 13-18 ms | 11-13 ms | 1.08-1.51x | True |

Things already visible at the sandbox stage (real hardware could still
overturn this):

1. **P1 looks likely to fail partially**: `diluted_p0.75` **lost** even
   though the search succeeded. The default pipeline already succeeds in
   66.8 ms on this candidate, while the prototype misses on all of stage 1
   (6 BFS orderings) and has to go all the way to stage 2, taking 124.7 ms.
   This is the obvious consequence of "you can't beat a default that's
   already fast" -- P1's wording (winning on all three candidates) was
   sloppy. If real hardware shows the same thing, P1 will be recorded as
   supported only for `grid` and `line`, and refuted for `diluted_p0.75`.
2. **P3 also looks likely to go in the opposite direction**: at loose,
   smart **won slightly** (1.08-1.51x). This is probably because passing
   `initial_layout` skips not just `VF2Layout` but `VF2PostLayout` entirely
   (see section 4-1), reducing the number of steps compared to the default.
3. **P4 looks likely to be supported (in the form of "no difference")**:
   2-qubit gate count and depth are **exactly identical across every
   condition and every arm** (tight: 96 gates / depth 16, loose: 36 gates /
   depth 16), with 0 coupling violations too. This would mean **"even when
   VF2Layout fails and falls back to Sabre, the final circuit's quality is
   unaffected,"** consistent with addendum-10's understanding that "the
   cliff is a time cost." This contest turns out to be purely about time.

## 4. Caveats (must be considered when reading the results)

1. **Passing `initial_layout` skips the entire layout stage.** Confirmed in
   the sandbox using `transpile(callback=...)`:
   - default: `SetLayout, VF2Layout, SabreLayout, VF2PostLayout`
   - with `initial_layout` specified: only `SetLayout, ApplyLayout`

   Note that `VF2PostLayout` is skipped too. This project's experiments only
   pass `coupling_map` and carry no error rates, so this should have no
   effect here, but **when using a real hardware `Target` (with error
   rates), skipping `VF2PostLayout` could lower fidelity**. This is an
   unverified item outside this measurement's scope.
2. **The PSF arm has never been run once.** Since `psf_zero_core` is absent
   in the sandbox, `psf_rl2` and `psf_rl2_smart` were automatically
   skipped. This will run for the first time on real hardware. In
   particular, whether `compile_for_hardware` can accept `initial_layout`
   is unconfirmed (see section 5).
3. `reps` defaults to 1 (to stay within a 5-minute budget). See addendum-10
   for the caveat about single-measurement variance.
4. The sandbox's absolute times are not representative of real hardware (a
   2-core VM).

## 5. How to run it, and where it might fail

```
python benchmark_smart_layout_vs_default.py
```

Place it in the **same folder** as [`vf2_probe_common.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/vf2_probe_common.py),
[`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py), and [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) (it imports all
of them).

**Two places this might not go smoothly** (both are designed to leave an
error message in the CSV and on screen):

- If `compile_for_hardware` does not accept `routing_optimization_level=2`
  -> re-run with `--psf-rl 1` (a value with a track record in phase3_v5).
- If `compile_for_hardware` does not accept `initial_layout` -> the
  `psf_rl2_smart` row becomes `TypeError: ...`. **In that case, if you send
  over the "PSF's initial_layout support: ... -- signature: ..." line that
  prints on screen at the start of the run, the calling convention will be
  fixed on this side.** Even when the signature check determines it cannot
  be accepted, it is **deliberately passed through anyway, letting a
  TypeError surface** -- silently dropping it would measure "an arm where
  the search was pointless," which would misrepresent the result.

To look at the Qiskit arms only first:
```
python benchmark_smart_layout_vs_default.py --arms qiskit_opt2,qiskit_opt2_smart
```

## 6. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) | followup 14 |

[`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) is unchanged from addendum-13.

## 7. Verification

- Ran all 24 conditions (6 topologies x spare 0/40 x 2 Qiskit arms) to
  completion in the sandbox (a 2-core Linux VM), confirming 0 coupling
  violations and valid output circuits.
- Directly confirmed, using `transpile(callback=...)`, which passes in the
  layout stage run when `initial_layout` is specified (section 4-1).
- Confirmed the PSF arm is automatically skipped in an environment where it
  is absent (including when explicitly specified via `--arms`).
- Pre-publication check: `grep` against this project's private personal-information pattern list, this addendum
  and [`benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) -> 0 hits.

---


<!-- ===== Addendum 15 (source: spare-qubit-cliff-addendum-15-2026-09-14.md) ===== -->

> **Note added when merging:** **First real-hardware run of the prototype benchmark (L2)**. Clean win/loss split; discovers the end-to-end PSF-Zero comparison could not be measured (`compile_for_hardware` lacks `initial_layout`).

## spare-qubit-cliff addendum 15 (2026-09-14) -- followup 14's real-hardware results. When it wins, 30x; when it loses, exactly the search time. The end-to-end PSF-Zero comparison **could not be measured**

## 0. In one line

Followup 14 was run on real hardware. **The Qiskit-side contest split cleanly
as predicted** -- 30.8x on `grid`, 35.4x on `line`, wins; losses of
0.38-0.47x on the three sparse candidates, and **the loss margin matched the
time spent searching almost exactly** (1-3% error). On the other hand, **the
"end-to-end with PSF-Zero included" comparison the user chose could not be
made** -- because `compile_for_hardware` does not accept `initial_layout`.
The cause and the needed fix are in section 4.

## 1. Real-hardware results -- tight (spare=0, 8x8=64 physical)

Environment: Windows 10, Python 3.11.9, `Intel64 Family 6 Model 181
Stepping 0, GenuineIntel`, 14 cores, Qiskit 2.5.2, rustworkx 0.18.1, numpy
2.4.6, `psf_zero_core.cp311-win_amd64.pyd:418304`. L2 only, reps=1.

| topology | qiskit_opt2 | qiskit_opt2_smart | ratio | search | search time |
|---|---|---|---|---|---|
| **grid** | 701.9 ms | **22.8 ms** | **30.78x win** | succeeded (stage 1) | 11.6 ms |
| **line** | 2165.1 ms | **61.2 ms** | **35.40x win** | succeeded (stage 1) | 27.2 ms |
| brick | 695.9 ms | 1632.7 ms | 0.43x loss | failed | 932.1 ms |
| diluted_p0.25 | 742.1 ms | 1970.2 ms | 0.38x loss | failed | 1196.1 ms |
| diluted_p0.5 | 2364.9 ms | 5029.6 ms | 0.47x loss | failed | 2657.7 ms |
| diluted_p0.75 | 119.9 ms | 364.2 ms | 0.33x loss | succeeded (stage 2) | 329.4 ms |

At loose (spare=40, n=24), all 6 candidates succeeded at stage 1 in
0.9-4.5 ms, with ratios of 0.98-1.40x (4 wins, 2 ties, 0 losses).

### Verdict on the predictions

- **(P1) partially failed.** `grid` and `line` are supported (30.8x, 35.4x).
  **`diluted_p0.75` is refuted** -- the search succeeded, yet still lost at
  0.33x. Against a default that already succeeds in 119.9 ms, going all the
  way to stage 2 cost 329 ms. This is exactly the kind of failure addendum-14
  section 3 had already flagged as possible from the sandbox.
- **(P2) supported -- and more precisely than expected.** Measuring the loss
  margin directly:

  | topology | increase | search time | difference |
  |---|---|---|---|
  | brick | +936.8 ms | 932.1 ms | +0.5% |
  | diluted_p0.25 | +1228.1 ms | 1196.1 ms | +2.7% |
  | diluted_p0.5 | +2664.7 ms | 2657.7 ms | +0.3% |

  **Loss margin = search time** holds to within 1-3% error. In other words,
  even when the search fails, the default pipeline afterward is not slowed
  down at all -- the loss is **entirely** the search time. This is about as
  clean an accounting as it gets, and the flip side is that **the loss can be
  eliminated simply by making the search cheaper or giving up on it
  sooner** (section 3).
- **(P3) the magnitude is supported, the direction is not.** "The difference
  is small" is supported (0.98-1.40x), but "smart is slightly slower" is
  wrong -- it was actually about the same or slightly faster. This is
  probably because passing `initial_layout` skips not just `VF2Layout` but
  `VF2PostLayout` too (addendum-14 section 4-1), reducing the total number
  of steps.
- **(P4) supported (in the form of "no difference").** The 2-qubit gate
  count and depth were exactly identical across all 24 conditions and every
  arm (tight: 96/16, loose: 36/16), with 0 coupling violations. **The cliff
  is a time cost, not a quality cost** -- addendum-10's understanding is
  corroborated by the real-hardware output circuits themselves.

## 2. What was found on the PSF-Zero side

Real measurements of `psf_rl2` (the default flow), tight:

| topology | qiskit_opt2 | psf_rl2 |
|---|---|---|
| grid | 701.9 ms | 719.2 ms |
| brick | 695.9 ms | 724.6 ms |
| diluted_p0.25 | 742.1 ms | 753.2 ms |
| diluted_p0.5 | 2364.9 ms | 2392.9 ms |
| diluted_p0.75 | 119.9 ms | 197.3 ms |
| line | 2165.1 ms | 1806.2 ms |

**Under conditions where the cliff is hit, PSF-Zero's and Qiskit's total
times come out nearly the same.** The difference is a few dozen
milliseconds, swallowed by the cliff's cost (a few hundred milliseconds to
2.4 seconds). This does not mean PSF-Zero's synthesis is slow -- it means
**under conditions where the cliff occurs, the layout stage dominates the
total compile time, and the speed of the synthesis side becomes invisible.**
Conversely, this is exactly where it would be worth putting a layout-stage
fix into PSF-Zero -- of the 719 ms `grid` tight takes in PSF-Zero, roughly
700 ms is the layout stage burning its budget; if that became 22.8 ms, the
felt experience of PSF-Zero overall would change.

(Note: this is a single measurement, at the specific condition of
`verify=False`, `entangling_basis="cx"`, `routing_optimization_level=2`.
It should not be read as a general speed comparison between PSF-Zero and
Qiskit.)

## 3. Two problems found

### 3.1 The search's time budget was not being respected (now fixed)

The search for `diluted_p0.5` took **2.658 seconds**. `--time-budget`
defaults to **2.0 seconds**, so this overran by 33%.

Cause: the budget was **only checked between attempts**. Once an attempt
starts, it does not stop until `call_limit` is exhausted, so the last
attempt overran in full. `SmartOrderingsTried` stopping at 8 (the others
stopped at 9) is the trace of this.

Fix: `call_limit` is now shrunk by estimating, from the "calls consumed per
second" observed so far, how many calls can be consumed in the remaining
time ([`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py)'s `_budgeted_call_limit()`). Since stage 1 and
stage 2 have different consumption rates, the rate is re-measured when the
stage changes.

### 3.2 Stage 2 (id_order=False) does not pay for itself

Stage 2 only caught 1 of the 6 candidates (`diluted_p0.75`), **and even on
that one it still lost** (0.33x). Meanwhile, the three candidates that
entered stage 2 (brick, diluted_p0.25, diluted_p0.5) threw away 0.9-2.7
seconds in full.

With stage 1 alone (6 BFS orderings x `id_order=True`), the loss margin
should be less than a tenth of stage 2's. An arm, `_smart1`, was added to
measure this directly. In a sandbox functionality check (**not used for the
actual verdict**), the loss shrank from 0.42x to **0.92x** (roughly a tie),
while `grid`'s and `line`'s wins (15-33x) remained intact.

## 4. Why the end-to-end comparison could not be measured, and what is needed

`compile_for_hardware`'s signature (obtained on real hardware):

```
compile_for_hardware(qc, coupling_map, basis_gates=None, block_gate_floor=12,
                     routing_optimization_level=1, verify=True,
                     entangling_basis='canonical', seed_transpiler=None,
                     on_unsupported='keep', tol=1e-05) -> QuantumCircuit
```

There is no `initial_layout`, and no `**kwargs` either. So there is
**currently no way** to pass a searched layout into PSF-Zero.

**Important warning -- this run's `psf_rl2_smart` rows must not be read.**
Of the 48 rows, the 3 where `psf_rl2_smart` shows `success` (brick,
diluted_p0.25, diluted_p0.5's tight cases) are rows where **the search
failed, `initial_layout` ended up `None`, so no `TypeError` was raised and
plain PSF-Zero just ran** -- rows where nothing but the search time was
lost, mixed in as if they were successes. So the `psf_rl2` vs
`psf_rl2_smart` win/loss table is **structurally guaranteed to show only
losses**. It is invalid as an end-to-end comparison. From now on, the PSF
smart arm is excluded from the start once `initial_layout` is determined to
be unsupported.

**What is needed (your decision required)**: adding an `initial_layout`
parameter to [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py)'s `compile_for_hardware` and passing it
straight through to the internal `transpile()` call -- probably about a
2-line change. Could you show me its contents so a patch can be written?

```
python -c "import inspect, psf_compile; print(inspect.getsource(psf_compile.compile_for_hardware))"
```

If you send that over, a diff will be prepared in the form matching this
project's existing `.patch` workflow (**the repository will not be edited
unilaterally from this side**).

## 5. Pre-registered predictions for the next run (**written before measuring**)

If `--reps 2` is run with the fixed versions
([`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) + [`benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py)):

- **(P5)** `_smart1` keeps roughly the same wins as `_smart` on `grid` and
  `line` (20-35x), since both are found at stage 1.
- **(P6)** `_smart1`'s loss margin on `brick`, `diluted_p0.25`, and
  `diluted_p0.5` **shrinks to roughly 0.85-0.98x** (since stage 1's cost is
  less than a tenth of stage 2's).
- **(P7)** after the fix, no row shows `SmartSearch_s` exceeding
  `--time-budget` (2.0 seconds). This run had a 2.658-second overrun on
  `diluted_p0.5`.
- **(P8)** on `diluted_p0.75`, `_smart1` **also loses** (since stage 1
  cannot find it), but the loss margin is smaller than `_smart`'s 0.33x.

**If this fails**: if P6 fails and `_smart1` also loses by a large margin,
that would mean stage 1's 6 attempts are themselves too heavy, requiring
fewer attempts or a lower `per_attempt_call_limit`. If P5 fails and
`_smart1` cannot win on `grid`, the record of which stage-1 ordering was
working (the `SmartOrder` column) needs to be reviewed.

## 6. On the pre-publication check (for the user)

The terminal output you sent included a path in the form `C:\Users\...`
(containing an account name). This is an item [`publication-policy.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/publication-policy.md)
section 4 specifies must not go into anything published. **It is not
included anywhere in the CSV saved to the Project, or in this addendum**
(confirmed on the CSV: `grep` against this project's private personal-information pattern list
-> 0 hits). Please strip the prompt portion before posting to GitHub.

## 7. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/smart_layout_vs_default_intel_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_intel_2026-09-14.csv) | followup 14's real-hardware results (48 rows) |
| [`psf-zero/prototypes/psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/prototypes/psf_smart_layout.py) | the time-budget fix (section 3.1) |
| [`psf-zero/benchmarks/benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) | added the `_smart1` arm, automatic exclusion of the PSF smart arm (sections 3.2, 4) |

## 8. Verification

- Re-aggregated the 48 rows of the real-hardware CSV and confirmed it
  matches the terminal output's summary.
- P2's "loss margin = search time" was confirmed by taking the difference,
  row by row, between the CSV's `Time_min_s` and `SmartSearch_s` (0.3-2.7%
  error).
- The time-budget overrun (2.658s > 2.0s) was confirmed directly from the
  `SmartSearch_s` column. `SmartOrderingsTried` stopping at 8 corroborates
  that the budget check was only operating at attempt boundaries.
- Ran the fixed version with the `_smart1` arm added through all 6
  topologies at tight in the sandbox (a 2-core Linux VM) to completion, with
  0 coupling violations, and confirmed the stage-1-only loss shrinks from
  0.42x to 0.92x (not used for the actual verdict).
- Pre-publication check: `grep` against this project's private personal-information pattern list
  against this addendum, the real-hardware CSV, and the two updated scripts
  -> 0 hits (excluding the cautionary strings mentioned in the text itself).

---


<!-- ===== Addendum 16 (source: spare-qubit-cliff-addendum-16-2026-09-14.md) ===== -->

> **Note added when merging:** **L3 real-hardware run. Discovers up to ~3x measurement variance** in the same condition within one run -- the sparse-topology win/loss verdicts from this round are left undecided pending variance measurement.

## spare-qubit-cliff addendum 16 (2026-09-14) -- L3 real-hardware results. Wins grew to 340-420x, but **the same measurement was found to swing by up to 3x**, so the loss side is left undecided

## 0. In one line

Ran on real hardware at L3 (`optimization_level=3`). **399.8x on `grid`,
339.2x on `line`** -- an order of magnitude up from L2's 30x. But at the
same time, a **serious measurement problem** was found -- **the time taken
by the exact same operation swung between 0.34x and 2.03x (roughly a 6x
range) within the same run.** So the win/loss on the sparse-topology side
**cannot be judged** with this round's reps=1. Addendum-15's claim that
"loss margin = search time (1-3% error)" also does not hold at L3. On top of
that, there was **one design mistake in the benchmark that ended up
measuring a combination that isn't a valid comparison** (section 4).

## 1. L3 real-hardware results -- tight (spare=0, 8x8=64 physical)

Same environment as addendum-15 (Windows 10, Python 3.11.9,
`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`, 14 cores, Qiskit
2.5.2). L3, reps=1.

| topology | qiskit_opt3 | _smart1 | _smart | search (smart1/smart) |
|---|---|---|---|---|
| **grid** | 8469.7 ms | **21.2 ms (399.8x)** | **20.1 ms (420.9x)** | succeeded/succeeded |
| **line** | 7240.1 ms | **21.3 ms (339.2x)** | **19.4 ms (373.7x)** | succeeded/succeeded |
| brick | 22567.0 ms | 24392.4 ms | 27109.7 ms | failed/failed |
| diluted_p0.25 | 28037.7 ms | 28417.0 ms | 11644.2 ms | failed/failed |
| diluted_p0.5 | 14851.4 ms | 30358.5 ms | 12247.8 ms | failed/failed |
| diluted_p0.75 | 47.4 ms | 96.6 ms | 102.1 ms | failed/succeeded |

At loose (spare=40), all 6 candidates and both arms succeeded at stage 1 in
1-2 ms, with **wins of 2.65-5.19x (12 wins out of 12)**. At L3, the default
pipeline's layout stage itself (`VF2Layout` + `VF2PostLayout`) becomes
heavier, so the gap widens even under conditions that don't hit the cliff.

## 2. A problem found -- the same operation swinging by up to 3x

When the search **fails**, what the smart arm is doing is "search (throwing
away time) + the default pipeline itself." That means **it is logically
impossible for the smart arm to beat the default arm when its search
fails.** Yet the verdict table showed the following two as "wins":

- `diluted_p0.25`: 28037.7 / 11644.2 = a **2.41x "win"** (search failed)
- `diluted_p0.5`: 14851.4 / 12247.8 = a **1.21x "win"** (search failed)

Since an impossible result appeared, this has to be variance on the default
side. Extracting "the body, with search time subtracted" from the CSV shows
**the exact same `transpile()` differing by this much within the same
run**:

| topology | default arm | smart1's body | smart's body | max/min |
|---|---|---|---|---|
| brick | 22.57 s | 24.19 s | 25.11 s | 1.11x |
| diluted_p0.25 | 28.04 s | 28.24 s | **9.62 s** | **2.94x** |
| diluted_p0.5 | 14.85 s | **30.12 s** | **10.24 s** | **2.94x** |

`diluted_p0.5` is **not monotonic** (30.12 s -> 14.85 s -> 10.24 s cannot be
explained by a simple drift of "gets faster/slower over time").

**What this means**:

1. **The win/loss on this round's three sparse candidates cannot be judged.**
   Anything with less than a 3x difference is entirely buried in noise.
2. **Addendum-15's "loss margin = search time (0.3-2.7% error)" also does
   not hold at L3.** At L2 it agreed to within 1-3% across all three
   instances, so that agreement was probably real, but **"the same absence
   of variance also holds at L2" has not been confirmed** -- even in the L2
   run, `psf_rl2`'s `brick` differs by **3.2x** between 724.6 ms (addendum-15's
   run) and 2321.2 ms (this run). Same settings, same operation. So **the L2
   results also need re-verification.**
3. `grid`'s and `line`'s wins (340-420x) and the loose side's wins
   (2.65-5.19x) are far larger than the width of the variance (~3x), so
   **these two conclusions are unaffected.**

### Two candidate causes, not yet separated

- **(a) Variance from `VF2Layout`'s `seed=-1`** (the shuffle discovered in
  addendum-9 that `seed_transpiler` cannot control). Addendum-11 confirmed
  that success/failure pins to 0% or 100%, but **the variance in the *time*
  taken to reach a failure has never been measured.**
- **(b) Drift in the execution environment** (thermal throttling, load from
  other processes). Though this alone is hard to reconcile with
  `diluted_p0.5` being non-monotonic.

In the sandbox (L2), looking at `Time_max_s / Time_min_s` across consecutive
reps shows a swing of only **1.00-1.08x**. **Back-to-back measurements don't
swing, yet measuring at a different point in the run gives a 3x
difference** -- this contrast is itself a clue. If it were (a), it should
also swing back-to-back, so this leans toward (b). The judgment is deferred
to the measurement in section 6.

## 3. Verdict on the pre-registered predictions (P5-P8)

- **(P5) supported.** `_smart1` kept roughly the same wins as `_smart` --
  399.8x/339.2x versus 420.9x/373.7x on `grid`/`line`. Dropping stage 2 does
  not lose the win in cases that can win.
- **(P6) undecidable.** The prediction was "the loss margin shrinks to
  0.85-0.98x." The measured values, brick 0.93x and diluted_p0.25 0.99x,
  fall within the predicted range, but diluted_p0.5 at 0.49x is outside it.
  However, as noted in section 2, **that difference is within the width of
  the noise**, so it can be called neither supported nor refuted. Note that
  `_smart1`'s own search cost is 175-239 ms, about 1% of the 15-28-second
  default -- **confirming it is "cheap" as the design intended.**
- **(P7) supported.** The time-budget fix worked. `_smart`'s search times
  were 2005.1 / 2029.7 / 2010.1 ms, overrunning the 2.0-second budget by
  0.25-1.5%. Before the fix it was 2657.7 ms (a 33% overrun).
- **(P8) roughly supported (a small difference).** On `diluted_p0.75`,
  `_smart1` is 0.49x and `_smart` is 0.46x. As predicted, `_smart1`'s loss
  is smaller, though the gap is slight. Note that `_smart1` fails the search
  on this candidate (it is only found at stage 2), and 47.4 ms + a 49.2 ms
  search = 96.6 ms -- **the arithmetic checks out exactly.**

## 4. A design mistake on the benchmark side -- this run's `psf_rl2` column cannot be used for comparison

`--level 3` was specified, but `--psf-rl` **defaulted to 2**, so only the
PSF arm ran at `routing_optimization_level=2`. That means the output's
`psf_rl2` column is **"opt3 vs rl2" -- a combination that was never a valid
comparison to begin with.** In fact, `psf_rl2`'s `grid` tight is nearly
identical between addendum-15's run (719.2 ms) and this one (724.3 ms),
which confirms it was never actually an L3 measurement.

At a glance, the numbers could be misread as "PSF-Zero is 10-38x faster than
Qiskit L3," but **most of that gap is the budget difference between L2 and
L3 (5 million calls vs. 30 million), not PSF-Zero's speed.** Recorded here
explicitly so it is not misread.

Fix: `--psf-rl`'s default now follows `--level`, and if they disagree, a
warning is printed at the start of the run saying "these two cannot be
compared directly."

## 5. Fixes applied this round

| Fix | Contents |
|---|---|
| `--psf-rl`'s default | now follows `--level` (section 4). Shows a warning if they disagree |
| Added `Time_max_s` and `Spread_max_over_min` columns | looking only at `min` cannot reveal variance, so the variance itself is now recorded |
| Added noise detection to the verdict table | a row that is a "win" despite the search having failed is now automatically flagged as **"undecidable / noise"** (the two cases in section 2 should have been caught by the code before a human noticed) |

## 6. What's next -- measuring the variance (with pre-registered predictions)

Before continuing the win/loss discussion, **the size of the variance needs
to be pinned down.** The "reproducibility of the downstream cost," set aside
in addendum-11 as "low priority," comes back here as the top-priority item.

```
python benchmark_smart_layout_vs_default.py --level 3 --spares 0 --reps 5 --arms qiskit_opt3
```

Measures the default arm alone, 5 times each, on the 6 tight candidates. The
newly added `Spread_max_over_min` column will show the back-to-back
variance. Expected to take roughly 15-20 minutes (L3's tight cases take
7-30 seconds each).

**Pre-registered predictions**:

- **(P9)** `grid`, `line`, and `diluted_p0.75` (candidates where the default
  finishes relatively fast) have `Spread_max_over_min` under 1.2x.
- **(P10)** `brick`, `diluted_p0.25`, and `diluted_p0.5` (candidates where
  the default takes 10+ seconds) have `Spread_max_over_min` **exceeding
  2x**. If it does, the cause leans toward (a) `seed=-1` variance -- since
  that means it swings even back-to-back.
- **(P11)** conversely, if P10 fails and the back-to-back variance stays
  under 1.2x, the cause leans toward (b) environmental drift, and an
  experiment checking **whether swapping the arms' execution order gives the
  same conclusion** (randomizing the order, or measuring each arm in a
  separate process) would be needed.

**Implications if this fails**: if P10 holds (seed=-1 is the cause), **every
time measurement this project has taken under tight conditions needs to be
redone as a median over multiple runs** -- including the cliff ratios in
addenda 5-10. The existence of the cliff itself (tens to hundreds of times)
is far larger than the variance, so that is unaffected, but the specific
"how many times" figures would need revisiting.

## 7. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/smart_layout_vs_default_L3_intel_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_L3_intel_2026-09-14.csv) | this round's L3 real-hardware results (48 rows) |
| [`psf-zero/benchmarks/benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) | section 5's fixed version |

## 8. Verification

- "The body, with search subtracted" (section 2's table) was computed by
  subtracting `SmartSearch_s` from `Time_min_s` in the CSV. This is applied
  only to search-failure rows (a success row uses `initial_layout`, so its
  body is doing different work).
- P7's budget compliance was confirmed directly from the `SmartSearch_s`
  column (2005.1 / 2029.7 / 2010.1 ms, all within 1.5% of 2.0 seconds).
- P8's arithmetic (47.4 + 49.2 = 96.6 ms) was confirmed against the
  corresponding CSV row.
- Section 4's finding that "the PSF arm was not actually at L3" was
  confirmed from `psf_rl2`'s `grid` tight being nearly identical between
  addendum-15's run (719.2 ms) and this run (724.3 ms).
- The fixed version (`--psf-rl` following `--level`, the
  `Spread_max_over_min` column, noise detection) was confirmed to complete
  in the sandbox (a 2-core Linux VM), with 0 coupling violations.
- Pre-publication check: `grep` against this project's private personal-information pattern list
  against this addendum, the real-hardware CSV, and the updated script ->
  0 hits (excluding the cautionary strings in the text itself). **Note that
  the terminal output you pasted this time also contained `C:\Users\...`.
  It has not been included in anything saved.**

---
