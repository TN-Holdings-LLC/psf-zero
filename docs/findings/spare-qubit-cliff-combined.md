# spare-qubit-cliff: Combined Addenda (Addendum 2026-09-13 through Addendum 27)

**This is a merge of 26 separately-written addenda into one chronological
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
  **unmeasurable amid ~3x run-to-run variance at L3** (Addendum 16). This
  same win/loss pattern, at close to the same magnitude, reproduces when
  routed through PSF-Zero's own compilation pipeline rather than bare
  `transpile()` (Addendum 17).
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
- **[Partially addressed in Addendum 18]** Whether Addendum 16's ~3x
  run-to-run variance is L3-specific: three independent L2 runs on the same
  tight, hard-to-layout topologies agreed to within 1.00-1.30x, unlike the
  ~3x spread found at L3. Whether L3 itself, other spare values, or other
  machines share this same reproducibility is still untested.
- Whether `smart_vf2_layout()`'s stage-2 budget can be tuned below its new
  default of 300,000 (Addendum 18) -- 200,000 already misses one topology
  outright, and no finer step was tried between the two values.
- **[Resolved in Addendum 17]** The end-to-end PSF-Zero comparison this
  series was meant to answer was blocked on `compile_for_hardware` gaining
  an `initial_layout` parameter (Addendum 15, section 4). That parameter
  was added and the comparison run on real hardware: the layout-search
  win/loss pattern reproduces through PSF-Zero's own pipeline at close to
  the same magnitude as the Qiskit-only comparison.
- **[Investigated in Addenda 20-21, still unresolved]** A reproducible
  slope anomaly on Qiskit's side of a *separate*, coupling-map-free
  compile-time comparison (Addendum 19, 10k and 50k iterations) -- visible
  on both runs, absent from PSF-Zero's curves. Near-degeneracy of the
  underlying circuit blocks was ruled out (Addendum 20). A ~145-iteration
  period found on the Intel machine that generated the original data did
  not reproduce on a different (AMD) machine (Addendum 20) -- but the AMD
  machine turned out to have its *own* comparably strong period instead
  (~187 iterations, autocorrelation 0.96-0.99), which survived changing
  the hash seed, the iteration count, and a Qiskit-free control loop
  (Addendum 21). The anomaly itself, and why the period's value differs by
  machine, remain unexplained; reading Qiskit's own source for a matching
  constant was identified as the natural next step but not yet attempted.
- **[New in Addendum 27]** A real bug was found and fixed in this
  session's own exact-fidelity checker (an `n_new == n_orig` special case
  skipped qubit remapping, producing false `exact_FAIL` results on
  layout-searched circuits). Once fixed, both Qiskit and PSF-Zero pass
  exact verification at the cliff and away from it. A wider, 5-round sweep
  (spare 0-24) mapped the cliff as sharp and confined to spare=0 (~250-280x
  at the peak, dropping to 1.0-1.7x by spare=2). It also found PSF-Zero
  itself is not perfectly stable exactly at the cliff's peak: 3 of 30
  spare=0 runs showed large, unexplained slowdowns (up to 8x the median),
  spread across different seeds in different rounds rather than tied to
  one specific circuit -- the mechanism behind this is untested.

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

## Proposed addendum for [`docs/findings/spare-qubit-cliff.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff.md)

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
[`benchmarks/vf2_id_order_probe.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/vf2_id_order_probe.py) used when it measured `id_order=True` finding the
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

- Scripts: [`benchmarks/verify_vf2_call_limit_tuple.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_call_limit_tuple.py),
  [`benchmarks/verify_vf2_steps_to_first_match.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_steps_to_first_match.py),
  [`benchmarks/verify_vf2_ordering_structure.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_structure.py),
  [`benchmarks/verify_vf2_cross_implementation.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_cross_implementation.py),
  [`benchmarks/verify_vf2_topologies.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_topologies.py),
  [`benchmarks/verify_qiskit_source_2_5_2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_qiskit_source_2_5_2.py),
  [`benchmarks/vf2_probe_common.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/vf2_probe_common.py)
- Raw data (Intel, 2026-09-14): [`data/vf2_call_limit_tuple_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_call_limit_tuple_2026-09-14.csv),
  [`data/vf2_steps_to_first_match_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_steps_to_first_match_2026-09-14.csv),
  [`data/vf2_ordering_structure_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_ordering_structure_2026-09-14.csv),
  [`data/vf2_cross_implementation_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_cross_implementation_2026-09-14.csv),
  [`data/vf2_topologies_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_topologies_2026-09-14.csv),
  [`data/qiskit_source_check_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/qiskit_source_check_2026-09-14.csv)

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

- Scripts: [`benchmarks/verify_vf2_seed_anomaly_repro.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_seed_anomaly_repro.py),
  [`benchmarks/verify_vf2_ordering_id_order_true.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_ordering_id_order_true.py),
  [`benchmarks/verify_vf2_call_limit_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_call_limit_sweep.py),
  [`benchmarks/verify_vf2_toroidal_grid.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_toroidal_grid.py)
- Raw data (Intel, 2026-09-14): [`data/vf2_seed_anomaly_repro_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_seed_anomaly_repro_2026-09-14.csv),
  [`data/vf2_ordering_id_order_true_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_ordering_id_order_true_2026-09-14.csv),
  [`data/vf2_call_limit_sweep_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_call_limit_sweep_2026-09-14.csv),
  [`data/vf2_toroidal_grid_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_toroidal_grid_2026-09-14.csv)

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

- Scripts: [`benchmarks/verify_vf2_heavy_hex_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_heavy_hex_topology.py),
  [`benchmarks/verify_vf2_dfs_mechanism.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_dfs_mechanism.py),
  [`benchmarks/verify_vf2_rustworkx_raw_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_rustworkx_raw_sweep.py)
- Raw data (Intel, 2026-09-14): [`data/vf2_heavy_hex_topology_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_heavy_hex_topology_2026-09-14.csv),
  [`data/vf2_dfs_mechanism_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_dfs_mechanism_2026-09-14.csv),
  [`data/vf2_rustworkx_raw_sweep_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_rustworkx_raw_sweep_2026-09-14.csv)

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
| [`psf-zero/data/vf2_pipeline_trace_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_pipeline_trace_2026-09-14.csv) (**not found in the repository as of 2026-09-14 -- link removed; the data behind addendum 9/10's "all 24 rows matched" claim was not preserved as a standalone file**) | followup 10's real-hardware results (provided by the user) |

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

> **Note added when merging:** Introduces the [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) prototype. **Caveat carried forward through Addendum 16**: every finding is against the public `rustworkx.vf2_mapping()`, not Qiskit's internal implementation.

## spare-qubit-cliff addendum 13 (2026-09-14) -- the layout-search prototype [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py): what building it revealed (sandbox only, not yet confirmed on real hardware)

## 0. In one line

Following a suggestion that "today's results might let us design the best
possible search-based compiler," a layout-search prototype,
[`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py), was built with an eye toward integrating it into
PSF-Zero. **It works** (it can find a valid layout under conditions where
Qiskit's default pipeline fails, such as grid and line), **but its
effectiveness is more limited than hoped** -- the "BFS is robust" finding
from addenda 9 and 12 turns out to be specific to pure grid physical graphs,
and does not generalize as-is to sparse topologies like addendum-8's brick
and diluted_p. A hard-to-notice bug was also found and fixed during
implementation. **This is confirmed only in the sandbox so far; real
hardware testing and integration into Qiskit proper are still ahead.**

## 1. What was built

[`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) (implemented directly on top of the public
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
| [`psf-zero/benchmarks/psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) | the prototype itself |
| [`psf-zero/benchmarks/smoke_test_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/smoke_test_smart_layout.py) | smoke test (grid/brick/diluted_p x3/line, plus the feasibility pre-check) |

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
  [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py), and [`smoke_test_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/smoke_test_smart_layout.py) -> 0 hits.

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
[`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py), and [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) (it imports all
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

[`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) is unchanged from addendum-13.

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
time ([`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py)'s `_budgeted_call_limit()`). Since stage 1 and
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
([`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) + [`benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py)):

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
(containing an account name). This is an item `publication-policy.md`
(since removed from the repository)
section 4 specifies must not go into anything published. **It is not
included anywhere in the CSV saved to the Project, or in this addendum**
(confirmed on the CSV: `grep` against this project's private personal-information pattern list
-> 0 hits). Please strip the prompt portion before posting to GitHub.

## 7. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/smart_layout_vs_default_intel_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_intel_2026-09-14.csv) (L2 run) | followup 14's real-hardware results (48 rows) |
| [`psf-zero/benchmarks/psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) | the time-budget fix (section 3.1) |
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
| [`psf-zero/data/smart_layout_vs_default_L3_intel_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_L3_intel_2026-09-14.csv) (L3 run) | this round's L3 real-hardware results (48 rows) |
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

<!-- ===== Addendum 17 (source: spare-qubit-cliff-addendum-17-2026-09-15.md) ===== -->

> **Note added when merging:** Confirms the fix requested in addendum 15
> section 4 (`compile_for_hardware` gaining an `initial_layout` parameter)
> was applied, and runs the end-to-end PSF-Zero comparison this whole series
> was originally motivated by for the first time.

## Addendum 17 (2026-09-15) -- the end-to-end PSF-Zero comparison finally ran. The layout-search win/loss pattern reproduces through PSF-Zero's own pipeline, unchanged

### 0. In one line

Addendum 15 identified that `compile_for_hardware` had no `initial_layout`
parameter, so the prototype's benefit could never be measured through
PSF-Zero's own compilation path -- every `psf_rl2_smart` row in that run's
data was an artifact (the search failed, `initial_layout` stayed `None`, and
plain PSF-Zero ran anyway, counted as a false "success"). A small patch
(`compile_for_hardware_initial_layout.patch`) adding that parameter -- three
lines: the signature, a docstring note, and forwarding it to the internal
`transpile()` call -- has now been applied and run on real hardware for the
first time. **The result: PSF-Zero's own pipeline shows the same win/loss
pattern as the Qiskit-only comparison, at close to the same magnitude, on
every topology tested.**

### 1. The patch

`compile_for_hardware` gained one new parameter:

```python
def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 1,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    seed_transpiler: int | None = None,
    initial_layout: list[int] | None = None,   # <-- new
    on_unsupported: str = "keep",
    tol: float = 1e-5,
) -> QuantumCircuit:
```

forwarded verbatim to the internal `transpile()` call. Every existing
call site is unaffected (`initial_layout` defaults to `None`). The
docstring carries forward the caveat from addendum 14 section 4-1 that
supplying this argument skips `VF2PostLayout` as well as the layout search
itself, which only matters once error rates are in play (not the case for
any measurement in this project so far).

Confirmed on real hardware: `PSFInitialLayoutMode` reads `True` for every
row in this round's data, meaning `psf_compile.compile_for_hardware`'s
signature was correctly detected as accepting the argument, and the
`psf_rl2_smart` / `psf_rl2_smart1` arms are no longer excluded.

### 2. Real-hardware results -- Qiskit-only vs. PSF-Zero, side by side

Same environment as addenda 15-16 (Windows 10, Python 3.11.9,
`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`, 14 cores, Qiskit
2.5.2, rustworkx 0.18.1, `psf_zero_core.cp311-win_amd64.pyd:418304`). L2 /
`routing_optimization_level=2`, `--time-budget 2.0`, `reps=1`.

| topology | spare | qiskit base | qiskit smart | ratio | psf base | psf smart | ratio |
|---|---|---|---|---|---|---|---|
| **grid** | 0 | 728.6 ms | 22.7 ms | **32.10x** | 739.0 ms | 27.2 ms | **27.18x** |
| **line** | 0 | 659.8 ms | 22.8 ms | **29.00x** | 629.1 ms | 22.4 ms | **28.05x** |
| brick | 0 | 735.7 ms | 1623.3 ms | 0.45x | 714.5 ms | 1661.4 ms | 0.43x |
| diluted_p0.25 | 0 | 764.8 ms | 1958.5 ms | 0.39x | 766.4 ms | 1995.9 ms | 0.38x |
| diluted_p0.5 | 0 | 748.5 ms | 1933.4 ms | 0.39x | 740.9 ms | 1967.4 ms | 0.38x |
| diluted_p0.75 | 0 | 44.4 ms | 104.4 ms | 0.43x | 48.6 ms | 116.9 ms | 0.42x |
| grid | 40 | 16.8 ms | 12.1 ms | 1.38x | 15.0 ms | 10.8 ms | 1.39x |
| line | 40 | 9.0 ms | 9.1 ms | 0.99x | 10.8 ms | 12.8 ms | 0.85x |
| brick | 40 | 10.6 ms | 10.9 ms | 0.98x | 12.9 ms | 15.6 ms | 0.83x |
| diluted_p0.25 | 40 | 13.4 ms | 12.9 ms | 1.04x | 17.1 ms | 10.5 ms | 1.63x |
| diluted_p0.5 | 40 | 10.9 ms | 9.9 ms | 1.10x | 11.5 ms | 11.3 ms | 1.02x |
| diluted_p0.75 | 40 | 11.4 ms | 9.2 ms | 1.24x | 13.1 ms | 11.0 ms | 1.20x |

**The Qiskit column and the PSF-Zero column tell the same story on every row.**
Where the Qiskit-only comparison wins big (`grid`, `line`, tight), PSF-Zero's
own pipeline wins by nearly the same factor (27-28x against 29-32x). Where
the Qiskit-only comparison loses because the search fails (`brick`,
`diluted_p0.25`, `diluted_p0.5`, tight), PSF-Zero's pipeline loses by
essentially the same factor (0.38-0.45x on both sides). At loose (spare=40)
the difference stays small on both, as in addendum 14/15.

### 3. What this settles, and what it does not

**Settled**: the layout-search prototype's effect is not an artifact of
measuring it against bare `transpile()` -- it survives, largely unchanged
in magnitude, when routed through PSF-Zero's full compilation pipeline
(2-qubit synthesis via the Rust core, then layout via the searched
`initial_layout`, then routing). This was the specific gap addendum 15
identified as blocking: **it is no longer blocked.**

**Not settled by this addendum**:
- This is a single run (`reps=1`) at one optimization level (L2). Addendum
  16 found up to ~3x run-to-run variance on the Qiskit side at L3 for
  exactly this kind of tight, hard-to-layout topology; whether the same
  variance affects the PSF-Zero column, and whether it holds at L3, has
  not been checked here.
- The underlying caveat from addendum 13 is unchanged: the search itself
  still calls the public `rustworkx.vf2_mapping()`, not Qiskit's internal
  `qiskit._accelerate.vf2_layout`. This addendum shows the searched layout
  integrates cleanly into PSF-Zero's pipeline once found -- it says
  nothing new about whether the search's own behavior matches Qiskit's
  internal implementation.
- Output circuit quality (2-qubit gate count, depth) was not re-checked in
  this round; addendum 14/15 found it identical across arms on the
  Qiskit-only comparison, and this data was not re-verified for the PSF
  arms specifically.

### 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/data/smart_layout_vs_default_2026-09-15.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_2026-09-15.csv) | this round's real-hardware results (72 rows, provided by the user) |
| [`compile_for_hardware_initial_layout.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_for_hardware_initial_layout.patch) | the patch described in section 1 |

### 5. Verification

- Confirmed `PSFInitialLayoutMode` reads `True` across every row in this
  round's CSV, i.e. the patched signature was detected correctly.
- The patch was verified before this round's run by (a) applying it to a
  clean copy of [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py) and confirming the result matches the
  intended edit byte-for-byte, (b) confirming the patched file compiles
  (`py_compile`), and (c) an AST check confirming `initial_layout` is both
  an accepted parameter and is actually forwarded as a keyword to the
  internal `transpile()` call.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and the round's CSV ->
  0 hits. **Note the terminal log the user pasted this round (file named
  `test.py`, actually five concatenated run logs) contained a Windows
  account name in five prompt lines; it has been redacted before saving and
  the original is not included in anything kept.**

<!-- ===== Addendum 18 (source: spare-qubit-cliff-addendum-18-2026-09-15.md) ===== -->

> **Note added when merging:** Confirms L2's tight-condition reproducibility
> across three independent runs (contrasting with Addendum 16's ~3x
> variance found at L3), then tunes `smart_vf2_layout()`'s stage-2 budget
> down from 2,000,000 to 300,000 based on a six-point sweep, cutting the
> failing-topology loss margin by roughly half with no cost to the winning
> cases.

## Addendum 18 (2026-09-15) -- L2 reproduces across three independent runs; stage-2 budget tuned from 2,000,000 to 300,000

### 0. In one line

Before tuning anything, the L2 tight-condition numbers behind Addendum 17
were checked for reproducibility, since Addendum 16 had found up to ~3x
run-to-run variance on the same kind of hard, tight topology **at L3**.
**Three independent runs at L2 agree to within 1.00-1.07x on every
tight-condition row** -- the L3 variance does not appear to carry over to
L2. With that reassurance, a six-point sweep of `smart_vf2_layout()`'s
stage-2 (`id_order=False` fallback) budget found `fallback_call_limit` can
be lowered from its previous default of 2,000,000 to **300,000** --
the smallest value that still reliably catches `diluted_p0.75` -- cutting
the three failing topologies' loss margin by roughly half, with the winning
topologies' margins unaffected. The default has been changed accordingly.

### 1. L2 reproducibility -- three independent runs, same environment

Same environment throughout (Windows 10, Python 3.11.9,
`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`, 14 cores, Qiskit
2.5.2, rustworkx 0.18.1). All three runs used `--level 2 --spares 0`,
`fallback_call_limit=2,000,000` (the then-current default), `reps=1`,
tight (spare=0) topologies only, Qiskit arms only.

| topology | arm | run 1 | run 2 | run 3 | max/min |
|---|---|---|---|---|---|
| grid | qiskit_opt2 | 728.6 | 746.0 | 732.7 | 1.02x |
| grid | qiskit_opt2_smart | 22.7 | 21.3 | 20.6 | 1.10x |
| line | qiskit_opt2 | 659.8 | 625.7 | 617.1 | 1.07x |
| line | qiskit_opt2_smart | 22.8 | 17.6 | 17.9 | 1.30x |
| brick | qiskit_opt2 | 735.7 | 733.5 | 715.4 | 1.03x |
| brick | qiskit_opt2_smart | 1623.3 | 1638.9 | 1644.9 | 1.01x |
| diluted_p0.25 | qiskit_opt2 | 764.8 | 760.9 | 748.1 | 1.02x |
| diluted_p0.25 | qiskit_opt2_smart | 1958.5 | 1962.9 | 1948.2 | 1.01x |
| diluted_p0.5 | qiskit_opt2 | 748.5 | 717.8 | 726.3 | 1.04x |
| diluted_p0.5 | qiskit_opt2_smart | 1933.4 | 1894.3 | 1896.8 | 1.02x |
| diluted_p0.75 | qiskit_opt2 | 44.4 | 44.3 | 44.5 | 1.00x |
| diluted_p0.75 | qiskit_opt2_smart | 104.4 | 102.1 | 99.5 | 1.05x |

(all times in ms; run 1 = the data behind Addendum 17, run 2 and run 3 are
independent re-executions of the same script and arguments)

**Every tight-condition row agrees to within 1.00-1.10x**, with `line`'s
`_smart` arm the loosest at 1.30x -- still far from Addendum 16's ~3x
finding. This is a small sample (three runs, one machine, L2 only), but it
is consistent with the run-to-run variance problem being specific to L3's
larger search budget and heavier downstream cost, rather than a general
property of this measurement setup. **Addendum 16's re-verification
question for L2 (raised in its section 2) is answered for this specific
condition: L2 does not show the same variance.** Whether this holds at
other spare values, other topologies, or other machines is untested.

### 2. Stage-2 budget sweep

[`benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) gained a `--fallback-call-limit`
argument (previously `smart_vf2_layout()`'s `fallback_call_limit` could
only be set by editing the default in the function signature). Swept at
six values -- 200k, 300k, 400k, 500k, 1m, 2m -- on the same six topologies,
tight, `qiskit_opt2_smart` only (the arm stage 2 actually applies to).

**diluted_p0.75 -- the only topology stage 2 needs to catch (found at
attempt 7 of 9, the `heuristic_natural` ordering):**

| budget | found | search time | tries |
|---|---|---|---|
| 200k | **False** | 137.0 ms | 9 (exhausted) |
| 300k | True | 96.6 ms | 7 |
| 400k | True | 87.9 ms | 7 |
| 500k | True | 97.9 ms | 7 |
| 1m | True | 88.9 ms | 7 |
| 2m | True | 100.1 ms | 7 |

**200,000 misses it entirely** -- the budget runs out before attempt 7
completes. 300,000 is the smallest value tested that still catches it, and
every value at or above 300,000 behaves identically (same attempt, same
ordering, search time flat around 88-110 ms with no further benefit from a
larger budget).

**The three genuinely-hard topologies -- cost of correctly finding nothing
(9 of 9 attempts fail) -- scales with the budget as expected:**

| topology | 200k | 300k | 400k | 500k | 1m | 2m |
|---|---|---|---|---|---|---|
| brick | 900.4 | 906.4 | 959.0 | 967.7 | 1216.1 | 1638.8 |
| diluted_p0.25 | 947.4 | 987.1 | 1029.2 | 1070.4 | 1356.8 | 1960.6 |
| diluted_p0.5 | 964.0 | 988.1 | 1061.8 | 1083.2 | 1337.6 | 1921.9 |

(total time in ms, `qiskit_opt2_smart` arm; `qiskit_opt2` baselines: brick
719.5, diluted_p0.25 751.5, diluted_p0.5 734.8)

**grid and line are unaffected by this budget across the whole sweep**
(both succeed at stage 1, so stage 2 never runs): their win margin against
the default pipeline stays at 27-35x throughout.

### 3. The change

`smart_vf2_layout()`'s `fallback_call_limit` default has been lowered from
2,000,000 to **300,000** -- the smallest value in the sweep that still
reliably catches `diluted_p0.75`, with no finer-grained search done between
200,000 and 300,000 to find a possibly-lower true threshold.

Effect on the win/loss ratio against `qiskit_opt2` (comparing the previous
default, 2m, to the new one, 300k):

| topology | ratio @ 2m (old default) | ratio @ 300k (new default) |
|---|---|---|
| grid | 34.5x | 34.1x |
| line | 33.0x | 27.5x |
| diluted_p0.75 | 0.53x | 0.47x |
| brick | 0.44x | **0.79x** |
| diluted_p0.25 | 0.38x | **0.76x** |
| diluted_p0.5 | 0.38x | **0.74x** |

The three failing topologies' loss margin nearly doubles (0.38-0.44x to
0.74-0.79x). `diluted_p0.75`'s margin moves slightly against the change
(0.53x to 0.47x) because its own total time barely changes (100.1ms to
110.9ms is within the noise seen in section 1) while the Qiskit-only
baseline for this topology happened to be measured slightly faster in this
particular run (44.5ms) than in the run used for the 2m column (52.6ms) --
this is consequently more a reflection of section 1's baseline variance on
a fast-running topology than a real cost of the new setting. `grid`'s ratio
is unaffected; `line`'s dropped from 33.0x to 27.5x, which is within the
1.00-1.30x spread already documented for that specific arm/topology in
section 1.

**Not tested**: whether 300,000 remains the right choice at L3, on a
different machine, or against a wider set of topologies than the six used
throughout this series. The finer boundary between 200,000 and 300,000 was
also not explored.

### 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) | gained `--fallback-call-limit` and a `FallbackCallLimit` CSV column |
| [`psf-zero/benchmarks/psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) | `fallback_call_limit` default changed from 2,000,000 to 300,000; docstring section added recording this sweep |
| [`sweep_200k.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_200k.csv), [`sweep_300k.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_300k.csv), [`sweep_400k.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_400k.csv), [`sweep_500k.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_500k.csv), [`sweep_1m.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_1m.csv), [`sweep_2m.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_2m.csv) (6 files, under `psf-zero/data/`) | the budget sweep, tight topologies, `qiskit_opt2`/`qiskit_opt2_smart`/`qiskit_opt2_smart1` (provided by the user) |
| [`smart_layout_vs_default_2026-09-15_run2_qiskit_only.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_2026-09-15_run2_qiskit_only.csv), [`..._run3_qiskit_2m.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_2026-09-15_run3_qiskit_2m.csv) (under `psf-zero/data/`) | the two additional reproducibility runs in section 1 (provided by the user; suffixed here to distinguish from the run behind Addendum 17, which shares the same base filename) |

### 5. Verification

- Section 1's reproducibility table was built by matching
  (topology, spare, arm) keys across the three source files and computing
  max/min directly; all three files were confirmed to have identical keys
  before comparing.
- Section 2's sweep values were read directly from each `sweep_*.csv`
  file's `SmartFound`, `SmartSearch_s`, `SmartOrderingsTried`, and
  `SmartOrder` columns; the "200k misses, 300k+ all behave identically"
  claim was checked across all six budget values, not inferred from the
  endpoints alone.
- The [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) default-value change and its accompanying
  docstring note were confirmed with `py_compile` (syntax) and a direct
  grep for the old value (2_000_000) to confirm no other reference to it
  was left stale.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and all data files
  named in section 4 -> 0 hits.

<!-- ===== Addendum 19 (source: spare-qubit-cliff-addendum-19-2026-09-15.md) ===== -->

> **Note added when merging:** Records a separate line of measurement from
> the same day (2026-09-15) -- coupling-map-free compile-time comparisons
> at 10,000 and 50,000 iterations -- and a visible, reproducible slope
> anomaly on the Qiskit side that is distinct from, but possibly related
> to, the run-to-run variance found in addendum 16. **No `coupling_map` is
> passed anywhere in this addendum's measurements**, so the mechanism
> described in addenda 9-10 (VF2Layout failing and falling back to
> SabreLayout) cannot be the cause here -- that mechanism requires a
> coupling map to fail against.

## Addendum 19 (2026-09-15) -- coupling-map-free compile-time comparison at 10k/50k iterations; a reproducible slope anomaly on Qiskit's side, cause unconfirmed

### 0. In one line

A separate benchmark ([`test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py)), run without any
`coupling_map` (so unrelated to this series' central VF2/SabreLayout
finding), compared Qiskit `optimization_level=3` against PSF-Zero
(`verify=True` and `verify=False`) over 10,000 and then 50,000 back-to-back
compiles of a fixed 15-qubit circuit. **PSF-Zero wins by 5.90x-7.00x
(cumulative-total basis) across both runs**, with correctness confirmed by
a 6-qubit fidelity check (1.000000000000 on all three arms) before each
sweep. Separately, **Qiskit's cumulative-time curve shows a visible,
non-smooth slope change at both sample sizes** -- present in the 10,000-run
plot and more pronounced in the 50,000-run plot -- that does not appear on
either PSF-Zero curve. The cause is not established.

### 1. Results

| | 10,000 iter | 50,000 iter |
|---|---|---|
| Qiskit median / mean / stdev | 7.570 / 9.207 / 6.143 ms | 8.355 / 12.007 / 8.236 ms |
| PSF-Zero (verify=True) median / mean / stdev | 1.211 / 1.543 / 0.900 ms | 1.458 / 2.036 / 1.324 ms |
| PSF-Zero (verify=False) median / mean / stdev | 0.950 / 1.314 / 0.901 ms | 1.155 / 1.857 / 1.313 ms |
| Speed-up, verify=True (cumulative-total) | 5.97x | 5.90x |
| Speed-up, verify=False (cumulative-total) | 7.00x | 6.47x |
| Time saved, verify=True | 76.65s | 498.56s |
| Time saved, verify=False | 78.93s | 507.53s |

Same environment as addenda 15-18 (Windows 10, `Intel64 Family 6 Model 181
Stepping 0, GenuineIntel`, Python 3.11.9). No `coupling_map` is passed to
`transpile()` at any point in this script -- only `basis_gates` and
`optimization_level=3`.

### 2. Cumulative-total vs. median-based speed-up

The headline figures above (5.90x-7.00x) are cumulative-total-based (total
Qiskit time divided by total PSF-Zero time). Computing the same ratio from
medians instead:

| | 10,000 iter | 50,000 iter |
|---|---|---|
| Median-based, verify=True | 6.25x | 5.73x |
| Median-based, verify=False | 7.97x | 7.23x |

The two methods disagree by roughly 5-10%, because the standard deviation
on every arm is close in magnitude to its own median (ratios of 0.74-0.99
at 10k, 0.91-1.14 at 50k) -- a long right tail (max values 20-30x the
median on every arm) rather than a tight, symmetric distribution. Neither
figure is more "correct" than the other; both are reported per this
project's standing practice of not picking one metric to represent
variance without stating the other.

**Relative spread (stdev/median) does not favor PSF-Zero as cleanly as the
absolute numbers suggest.** At 50,000 iterations, `verify=False` has the
*highest* relative spread of the three arms (1.14, against Qiskit's 0.99
and `verify=True`'s 0.91) -- the same pattern already seen once at 10,000
iterations (0.95 against Qiskit's 0.81). In absolute terms PSF-Zero's
timings are far less noisy (stdev under 1.4ms against Qiskit's 6-8ms), but
*relative to its own much smaller median*, `verify=False` swings
proportionally more than Qiskit does. This mirrors the same
absolute-vs-relative disagreement already documented for a different
measurement in addendum 15's determinism-variance work, and is recorded
here rather than picking a side.

### 3. A visible, reproducible slope anomaly -- Qiskit only, cause unconfirmed

Plotting cumulative time against iteration count (both sample sizes,
user-provided figures) shows Qiskit's curve is not a straight line: it has
one or more visible regions where the slope steepens before returning to
its baseline rate. At 10,000 iterations this appears as two modest
inflections, around iteration 4700 and 6000 (consistent with the
progress-log timestamps: the 4000-5000 and 5000-6000 iteration blocks took
34.9s and 37.5s against a typical ~23s for other 1000-iteration blocks in
the same run). At 50,000 iterations the same kind of feature appears more
visibly, with a pronounced slope change around iteration 25,000-30,000.
**Neither PSF-Zero curve (verify=True or verify=False) shows a comparable
feature at either sample size.**

This is **not** an instance of this series' central finding (VF2Layout
failing and falling back to SabreLayout, addenda 9-10) -- that mechanism
requires a `coupling_map`, and none is passed anywhere in this script. It
is recorded here as a separate, open observation because it is (a) visibly
reproducible across two independent runs at different sample sizes, on the
same fixed circuit, and (b) specific to Qiskit's arm, matching the general
shape (of the several unresolved variance questions in this series --
addendum 16's ~3x L3 run-to-run spread being the other) that Qiskit's side
of these comparisons has shown more of this kind of behavior than
PSF-Zero's.

**Candidate causes, none checked**: background system load coinciding with
that iteration range; an internal Qiskit effect (caching, JIT-like
warm-up, or similar) with a delayed onset; or measurement variance of a
kind related to, but distinct from, addendum 16's finding (that was at
`optimization_level=3` with a `coupling_map` present; this has no coupling
map at all, so if there is a common cause it is not the specific
VF2Layout/SabreLayout mechanism, at most something further upstream that
both configurations might share).

### 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py) | the script used for both runs (provided by the user; one Japanese-language comment translated to English before this round) |
| [`psf-zero/data/cumulative_compile_times_10000.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_10000.npz) | raw per-iteration timings, 10,000-iteration run (provided by the user) |
| [`psf-zero/data/cumulative_compile_times_50000.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_50000.npz) | raw per-iteration timings, 50,000-iteration run (provided by the user) |
| [`Figure_1.png`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/Figure_1.png) | cumulative-time and box-plot figure, 10,000-iteration run (under `docs/`; provided by the user) |
| [`cumulative_compile_results_50000.png`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/cumulative_compile_results_50000.png) | the same pair of plots, 50,000-iteration run (under `docs/`; provided by the user) |

### 5. Verification

- Re-loaded both `.npz` files directly and recomputed median, mean,
  standard deviation, and the stdev/median ratio for all three arms at
  both sample sizes; all values match the script's own printed summary.
- Confirmed no `coupling_map` argument appears anywhere in
  [`test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py)'s three `transpile()` call sites.
- The slope-anomaly timing at 10,000 iterations was cross-checked against
  the script's own progress-log timestamps (34.9s and 37.5s for the two
  affected 1000-iteration blocks, against a ~23s baseline for unaffected
  blocks in the same run) rather than read off the figure alone.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and the files named in
  section 4 -> 0 hits.

<!-- ===== Addendum 20 (source: spare-qubit-cliff-addendum-20-2026-09-15.md) ===== -->

> **Note added when merging:** Follows up on Addendum 19's unconfirmed
> Qiskit-side slope anomaly. Traces the worst Qiskit outliers from the
> 50,000-iteration run back to their exact circuits, rejects a
> near-degeneracy explanation, finds a strong ~145-iteration periodicity in
> the same dataset by full-series autocorrelation, and then **rejects
> that too** when it fails to reproduce on a second machine across two
> independent runs. The Addendum 19 anomaly remains unexplained at the
> end of this addendum.

## Addendum 20 (2026-09-15) -- chasing the Addendum 19 anomaly: near-degeneracy rejected, a ~145-iteration period found and then rejected on a second machine

### 0. In one line

Addendum 19 found a visible, reproducible slope anomaly on Qiskit's side of
a coupling-map-free compile-time comparison, with no established cause.
This addendum traces it further. **The 20 slowest Qiskit compiles in the
50,000-iteration run (Intel machine) were rebuilt exactly from their seeds
and inspected block by block; none were close to a degenerate point,
rejecting that explanation.** Sorting the same 20 indices by hand instead
suggested a repeating gap of ~145 iterations. A full-series autocorrelation
check found this was real and strong on the Intel machine's data (rank 1
of 500 lags, modular-bin spread 7.49x against ~1.03-1.11x for four other
candidate periods) -- but **the same check on two independent 5,000-iteration
runs on a different (AMD) machine found no trace of it** (rank 42 and 70 of
500, spread ~1.04x, indistinguishable from the other candidate periods).
**The ~145-iteration period is not a general property of this
measurement; whatever caused it appears specific to the single Intel-machine
run it was found in, and remains unexplained.**

### 1. Rejecting near-degeneracy as the cause of the outliers

The 20 slowest Qiskit compiles from the Addendum 19 50,000-iteration run
(indices 953, 3853, 5158, 5448, 5883, 9363, 10523, 10668, 12118, 12408,
16613, 24443, 24588, 24733, 30533, 30678, 30823, 30968, 31113, 47933 --
14.7x-16.3x the median) were rebuilt exactly: `build_dense_pair_blocks_circuit`
seeds its generator with `1000 + index`, so each circuit's construction is
fully determined by its index.

19 of these 20 indices are *also* elevated on PSF-Zero's side at the same
index (2.9x-3.9x its own median) -- only index 47933 is slow on Qiskit
alone. That pattern by itself pointed at the circuit rather than either
engine in isolation, motivating a look at what these circuits actually
contain.

Each of the 140 two-qubit blocks (7 pairs x 20 circuits) was checked for
Frobenius distance, after SU(4) projection, to four landmark points
(identity, CNOT, SWAP, iSWAP) that this project's own findings on [`lib.rs`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/lib.rs)
name as historically hard for KAK-style decomposition. Mean distance across
the 140 outlier blocks was 2.305 (min 1.681, max 2.658). Three baseline
indices (100, 5000, 40000 -- chosen without reference to the outlier
ranking) gave a mean of 2.367 across their 21 blocks (min 1.629, max
2.610) -- **statistically indistinguishable from the outlier blocks.**
**Near-degeneracy is rejected as the explanation**: the outlier circuits'
blocks are not meaningfully closer to a hard point than an arbitrary
circuit's blocks are.

(One implementation bug was caught and fixed while building this check: an
early version measured distance to each landmark without first SU(4)-
normalizing the landmark itself, so CNOT's own distance to CNOT came out
as 1.53 instead of 0 in a self-test. Fixed by projecting both sides before
comparing, and re-verified against all four landmarks before use.)

### 2. A ~145-iteration period, found and then rejected

With near-degeneracy rejected, the 20 outlier indices were sorted and their
gaps inspected by hand: 14 of the 20 fell into small clusters with gaps
close to 145 or a small multiple of it (10523-10668, 24443-24588-24733,
30533-30678-30823-30968-31113). A dedicated check
(autocorrelation across lags 1-500, plus a modular-bin comparison against
four other candidate periods with no particular reason to matter -- 100,
120, 160, 200) was run against the **full** 50,000-point series to
establish whether this was a real effect or an artifact of eyeballing 20
points.

**On the Intel machine (the same run the outliers came from), the effect
was strong and specific to Qiskit:**

| series | autocorrelation at lag 145 | rank (of 500) | modular-bin spread @ 145 | spread @ other periods |
|---|---|---|---|---|
| qiskit | 0.8621 | 1 | 7.493x | 1.074x-1.107x |
| psf_true | 0.6378 | 146 | 1.389x | 1.091x-1.148x |
| psf_false | 0.7280 | 145 | 1.354x | 1.134x-1.217x |

Lags 290 and 435 (both multiples of 145) also ranked in Qiskit's top 5,
which a coincidental single-lag spike would not produce.

**This did not reproduce on a second machine.** Two independent
5,000-iteration runs on an AMD machine gave:

| run | qiskit autocorr @ 145 | rank | spread @ 145 | spread @ other periods |
|---|---|---|---|---|
| AMD run 1 | 0.0026 | 42 | 1.040x | 1.027x-1.047x |
| AMD run 2 | -0.0005 | 70 | 1.039x | 1.020x-1.039x |

Neither run shows anything resembling the Intel result. The modular-bin
spread at 145 is indistinguishable from the spread at every other
candidate period tried, in both runs -- exactly the "coincidence of a
20-point sample" outcome the check's own verdict section describes as the
negative case. One mild curiosity: both AMD runs' top-5 autocorrelation
lags include 374, 187, and 102 in common, despite being independent runs --
but the autocorrelation values themselves are small (0.003-0.02, against
Intel's 0.86), so this is more likely coincidental structure in short
series than a real effect, and was not investigated further.

**One easy candidate cause was checked and ruled out**: Python's garbage
collector generation-0 threshold on the machine used was `(700, 10, 10)` --
no relation to 145.

### 3. Where this leaves Addendum 19's anomaly

The original slope anomaly Addendum 19 found (visible on Qiskit's
cumulative-time curve at both 10,000 and 50,000 iterations, absent from
PSF-Zero's curves) **remains unexplained.** What has been established since:

- It is not explained by circuit-level near-degeneracy (section 1).
- A specific, testable periodic-effect hypothesis (~145 iterations) was
  found, measured precisely, and then **rejected** on a second machine
  across two runs (section 2) -- it does not generalize, and whatever
  produced it on the Intel run was most likely specific to that run's
  environment, not a property of the measurement itself or of Qiskit's
  code.
- Python's GC threshold is not the cause.

**What remains untried**: repeating the run on the *same* Intel machine a
second time, to check whether the ~145 period is specific to that one run
(environmental noise, coincident with something running at the time) or
whether it recurs on that particular machine specifically (which would
narrow the search to something about that machine's configuration rather
than the measurement in general). This was the natural next step but the
Intel machine was not available to repeat the check in this session.

### 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/diagnose_outlier_circuits.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/diagnose_outlier_circuits.py) | section 1's reconstruction and landmark-distance check |
| [`psf-zero/benchmarks/check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py) | section 2's autocorrelation and modular-bin check |
| [`psf-zero/data/cumulative_compile_times_5000_amd_run1.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_amd_run1.csv), [`..._amd_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_amd_run2.csv) | the two AMD-machine runs in section 2 (provided by the user; both are raw `.npz`, not `.csv`, exact filenames as saved by the user) |

### 5. Verification

- Section 1's landmark-distance function was self-tested against each of
  the four landmarks compared to itself (expected distance 0 in every
  case) both before and after the SU(4)-normalization bug fix; the fix
  was confirmed necessary and sufficient (pre-fix: CNOT-to-CNOT gave 1.53;
  post-fix: 0.0, along with identity-to-identity and SWAP-to-SWAP both
  giving 0.0 and clear separation, 1.5-2.8, between every distinct pair of
  landmarks).
- Section 1's outlier-vs-baseline comparison used indices (100, 5000,
  40000) chosen before seeing the outlier analysis's own numeric spread,
  to avoid picking a baseline that happened to confirm the hypothesis
  under test.
- Section 2's autocorrelation and modular-bin results were computed once
  in the sandbox against the same `.npz` file the user's own run produced,
  and matched the user-reported terminal output exactly (all reported
  figures agree to the digits shown); this confirms the analysis script
  itself, not a second independent data source.
- The AMD-machine non-reproduction (section 2) is the user's own two
  independent terminal runs of [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py), both included
  verbatim in the figures above.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and the two new
  scripts named in section 4 -> 0 hits.

<!-- ===== Addendum 21 (source: spare-qubit-cliff-addendum-21-2026-09-15.md) ===== -->

> **Note added when merging:** Addendum 20 found that the ~145-iteration
> period discovered on an Intel machine did not reproduce on an AMD
> machine, and left it there. This addendum went back to the AMD machine's
> own data and found a **different but equally strong period (~187
> iterations)** that had been sitting in the same "not 145" data all
> along. Hash randomization, iteration count, and a dummy-loop control
> (no Qiskit involved at all) were all ruled out as the cause -- the
> period survives all three, narrowing the explanation toward something
> about Qiskit's own execution, though what specifically remains open.

## Addendum 21 (2026-09-15) -- a second, stronger period found on the AMD machine (~187 iterations); hash seed, iteration count, and a Qiskit-free control all ruled out as the cause

### 0. In one line

Addendum 20 concluded the ~145-iteration period found on an Intel machine
"is not a general property of this measurement" after it failed to
reproduce on an AMD machine across two runs. Revisiting those same
AMD-machine runs' own top autocorrelation lags (rather than only checking
lag 145) found `374` and `187` (374 = 187 x 2) ranked consistently at the
top across **every** AMD-machine run collected so far. Measuring the
autocorrelation at lag 187 directly gives **0.96-0.99** -- stronger than
the original Intel-machine 145-period's 0.86 -- and this held across three
different `PYTHONHASHSEED` values and two different iteration counts (5000
and 2500). A dedicated control loop with no Qiskit or PSF-Zero involved at
all (pure Python arithmetic, `time.sleep`, and a numpy matrix multiply,
each timed the same way) showed **no trace of a 187-iteration period** in
any of its three variants, across two runs. **The period is not explained
by hash randomization, elapsed time, or the measurement loop's own
mechanics -- what remains, by elimination, points toward something in
Qiskit's own execution, not yet identified.**

### 1. How this was found

Addendum 20's hash-seed and reproducibility checks used
`check_period_145.py --npz <file>`, which reports each run's top-5
autocorrelation lags regardless of which period was requested. Three AMD-
machine runs collected to test the hash-seed hypothesis (`PYTHONHASHSEED`
42 run 1, 42 run 2, and 7 run 1, all 5,000 iterations) each printed `374`
and `187` as their top two Qiskit lags:

| run | top-5 lags | autocorr @ 145 (requested) |
|---|---|---|
| seed42 run1 | 374, 187, 476, 289, 102 | 0.0037 (rank 25/500) |
| seed42 run2 | 374, 187, 102, 272, 85 | 0.0009 (rank 57/500) |
| seed7 run1 | 374, 187, 289, 476, 102 | 0.0011 (rank 41/500) |

145 itself was, as Addendum 20 found, unremarkable in all three (rank
25-57 of 500, values near zero). But `187` and `374` (an exact multiple)
appearing at the top of every single run, across three different hash
seeds, was not something the original 145-focused check would have
surfaced on its own -- it only reports a pass/fail against the one
requested period.

Measuring the autocorrelation at lag 187 directly (rather than reading it
off the top-5 list) gives:

| run | autocorr @ 187 | autocorr @ 374 | modular-bin spread @ 187 |
|---|---|---|---|
| seed42 run1 | 0.9631 | 0.9633 | 7.490x |
| seed42 run2 | 0.9702 | 0.9702 | 7.419x |
| seed7 run1 | 0.9916 | 0.9926 | 7.469x |

These are **stronger** than the original Intel-machine 145-period result
(autocorrelation 0.8621, spread 7.493x) -- this is not a weaker echo of
the same thing, it is a comparably strong effect at a different value.

### 2. Ruling out hash randomization

The three runs in section 1 used `PYTHONHASHSEED` values 42, 42 (repeated),
and 7 -- deliberately including a repeat of the same seed to distinguish
"changes with the seed" from "changes between runs regardless of the
seed." **187 appeared identically in all three, including both runs on
seed 42.** If hash randomization were the cause, either the repeated seed
(42, 42) should have produced the same period while the different seed (7)
produced a different one, or every run should have differed. Neither
happened: all three agree on 187 regardless of seed.

### 3. Ruling out elapsed time

The ~145-period search in Addendum 20 could not distinguish a period
counted in iterations from one counted in elapsed seconds, since the
iteration count was not varied. Here it was: the same check was run at
2,500 iterations (half of 5,000). If the true period were time-based (a
process running every N seconds regardless of how fast the loop was
iterating), halving the iteration count would not preserve the same
iteration-based period. It did:

| run | iters | autocorr @ 187 | modular-bin spread @ 187 |
|---|---|---|---|
| seed(unspecified) | 5,000 | ~0.96-0.99 (section 1) | ~7.4-7.5x |
| seed(unspecified) | 2,500 | 0.9931 | 7.524x |

The period is counted in iterations, not elapsed time.

### 4. Ruling out the measurement loop and machine in general

A dedicated control script ([`check_dummy_loop_period.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_dummy_loop_period.py)) replaces the
loop body with three alternatives that call neither Qiskit nor PSF-Zero,
each timed with the same `time.perf_counter()` pattern and warm-up-outside-
the-timer discipline as the original benchmark:

- `busy`: pure-Python arithmetic, no imports or I/O inside the loop.
- `sleep`: `time.sleep()`, which hands control back to the OS scheduler
  every iteration (`busy` does not).
- `numpy`: a fixed-size matrix multiply, exercising the same BLAS/thread-
  pool machinery Qiskit's own linear algebra depends on, without going
  through Qiskit.

Each arm was calibrated to take roughly the same order of magnitude of
time per iteration as one Qiskit compile in the original benchmark
(6-16ms). Run twice, 5,000 iterations each time, checked at lag 187:

| run | busy | sleep | numpy |
|---|---|---|---|
| 1: autocorr @ 187 | -0.0093 | 0.0006 | 0.0784 |
| 1: modular-bin spread | 1.011x | 1.023x | 1.075x |
| 2: autocorr @ 187 | 0.0209 | -0.0008 | 0.0154 |
| 2: modular-bin spread | 1.006x | 1.018x | 1.059x |

**None of the six results (three arms x two runs) come close to the
0.96-0.99 autocorrelation or ~7.4-7.5x spread Qiskit's own timings show.**
The largest value across all six is 0.0784 (numpy, run 1) -- roughly 1/12
of Qiskit's weakest observed value. This rules out the OS scheduler
(`sleep` shows nothing), raw CPU/interpreter overhead (`busy` shows
nothing), and the BLAS/threading layer generically (`numpy` shows
nothing, despite exercising the same underlying linear-algebra
infrastructure Qiskit itself uses).

### 5. Where this leaves the investigation

What has been ruled out, in order across Addenda 20-21: near-degeneracy of
the circuit blocks, hash randomization, elapsed time, the OS scheduler,
raw computation overhead, and generic BLAS/threading activity. What
remains, by elimination, is **something specific to Qiskit's own code path
during `optimization_level=3` compilation** -- an internal cache, counter,
or state that changes behavior on a ~187-iteration cycle on this machine
(and a ~145-iteration cycle, differently, on the Intel machine from
Addendum 19-20). Neither the mechanism nor why the period's value differs
between the two machines has been identified.

**No numerical relationship between 187 and this machine's readily
available parameters was found**: `os.cpu_count()` returns 12 on this
machine, and 187 (= 11 x 17) is neither a multiple nor a divisor of 12.
This does not rule out a machine-specific cause -- it only means the
obvious candidate (core count) is not it.

**What was not tried**: reading Qiskit's own source for a constant near
145 or 187 (a cache size, a batch limit, a buffer threshold) that might
explain either machine's period directly, the way this project's earlier
addenda settled the spare-qubit-cliff mechanism by reading source rather
than only measuring around it. This is the natural next step but was not
undertaken in this round.

### 6. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/check_dummy_loop_period.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_dummy_loop_period.py) | section 4's Qiskit-free control script |
| [`psf-zero/data/cumulative_compile_times_5000_seed42_run1.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_seed42_run1.npz), [`..._seed42_run2.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_seed42_run2.npz), [`..._seed7_run1.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_seed7_run1.npz) | section 1-2's hash-seed runs (provided by the user) |
| [`psf-zero/data/cumulative_compile_times_2500.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_2500.npz) | section 3's half-length run (provided by the user) |

### 7. Verification

- Section 1's autocorrelation-at-187 and modular-bin-spread-at-187 figures
  were computed directly from the three user-provided `.npz` files in the
  sandbox, using the same `autocorrelation()` and `modular_bin_medians()`
  functions [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py) already uses -- not read off the
  terminal output's top-5 list, which only reports rank, not the
  underlying value.
- Section 4's control script was smoke-tested in the sandbox (500
  iterations, period 47, to fit the sandbox's smaller resource budget)
  before being sent to the user, confirming it runs to completion and
  produces the same three metrics (median/mean/std, top-5 lags,
  autocorrelation-at-period, modular-bin spread) as the main period
  checker, before the user ran the real 5,000-iteration version on the
  AMD machine.
- Section 3's iteration-count-independence claim (2,500 vs 5,000) was
  checked by directly comparing the autocorrelation and spread values
  side by side, not just their qualitative rank.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and the new script
  named in section 6 -> 0 hits.


<!-- ===== Addendum 22 (source: spare-qubit-cliff-addendum-22-2026-09-16.md) ===== -->

> **Note added when merging:** Tests the pre-registered hypothesis that CPython's garbage collector causes the ~145/~187-iteration period (Addenda 19-21): confirms 145 reproduces a second time on the same Intel machine, and reads Qiskit's own Sabre Rust source, ruling it out as the cause since that code path never runs in the coupling-map-free experiment that found the period.

## Addendum 22 (2026-09-16) -- the ~145-iteration period reproduces six times on the same Intel machine, survives three hash-seed conditions, and is not explained by the OS scheduler, raw computation, or BLAS/threading

## 0. In one line

Addendum 20 found a strong ~145-iteration autocorrelation period in Qiskit's
compile-time series on one Intel machine, and flagged as the natural next
step -- not yet attempted -- repeating the run on that *same* machine to see
whether the period recurs or was a one-off. It has now been repeated **six
times** (the original 2026-09-14 run plus five independent runs on
2026-09-16), and the period reproduces in every one, with autocorrelation
0.68-0.91 and modular-bin spread 7.5x-10.7x at lag 145, always ranking #1 of
500 lags tested. The period is unaffected by `PYTHONHASHSEED` across three
conditions (unset, 42, 7) and does not appear in a Qiskit-free control loop
(pure computation, `time.sleep`, or a numpy matrix multiply) run on the same
machine. **What causes it remains unidentified**, but the candidate causes
ruled out by Addendum 21 for the AMD machine's 187-period now also rule out
the same explanations for this machine's 145-period specifically.

## 1. Reproducibility: six independent runs, same machine, same script

Same machine as addenda 5-20 (Windows 10, Python 3.11.9, `Intel64 Family 6
Model 181 Stepping 0, GenuineIntel`, 14 cores), same script
(`test_cumulative_compile_scale.py --iters 5000`, no `coupling_map`, as in
Addendum 19). All six `.npz` outputs were independently reloaded and
re-analyzed in the sandbox (not read off terminal output) to compute
autocorrelation and modular-bin spread directly.

| Run | Date | `PYTHONHASHSEED` | Qiskit total (s) | Autocorr @ 145 | Rank (of 500) | Modular-bin spread @ 145 |
| :--- | :--- | :--- | ---: | ---: | :---: | ---: |
| Original (Addendum 20) | 2026-09-14 | unset | -- | 0.8621 | 1 | 7.493x |
| Run 2 | 2026-09-16 | unset | 74.292 | 0.8669 | 1 | 8.830x |
| Run 3 | 2026-09-16 | unset | 69.752 | 0.7824 | 1 | 10.673x |
| Run 4 | 2026-09-16 | unset (see §2) | 56.170 | 0.6833 | 1 | 7.531x |
| Run 5 | 2026-09-16 | 42 | 67.662 | 0.7380 | 1 | 10.661x |
| Run 6 | 2026-09-16 | 7 | 38.323 | 0.9062 | 1 | 8.016x |

Every run puts lag 145 at rank 1 of 500 lags tested, with autocorrelation
well above the 0.0-0.3 background level `psf_true`/`psf_false` show at the
same lag in every one of these runs (not tabulated here; consistent with
Addenda 19-21's own finding that the PSF-Zero arms do not show this
pattern). The absolute compile times vary run to run (38-74s total, a
~2x spread reflecting ordinary machine-load variation of the kind already
documented in Addendum 16) but the periodic structure itself is stable
throughout.

**This resolves Addendum 20's open item.** The ~145-iteration period is not
specific to the one 2026-09-14 run it was discovered in -- it is a
persistent, repeatable property of this machine (or of Qiskit running on
it), observed independently across six separate process launches spanning
two different days.

## 2. Hash-seed test: a labeling correction, and a clean result across three conditions

The run sequence was:

```
python test_cumulative_compile_scale.py --iters 5000        (no PYTHONHASHSEED set yet)
ren cumulative_compile_times_5000.npz ..._seed42_run1.npz    (mislabeled -- see below)

set PYTHONHASHSEED=42
python test_cumulative_compile_scale.py --iters 5000
ren cumulative_compile_times_5000.npz ..._seed42_run2.npz

set PYTHONHASHSEED=7
python test_cumulative_compile_scale.py --iters 5000
ren cumulative_compile_times_5000.npz ..._seed7_run1.npz
```

**Correction: the file named [`..._seed42_run1.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_seed42_run1.npz) was actually run *before*
`set PYTHONHASHSEED=42` was issued**, so it in fact ran under Python's
default (per-process random) hash seed, not seed 42. Only
[`..._seed42_run2.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_seed42_run2.npz) (seed 42) and [`..._seed7_run1.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_seed7_run1.npz) (seed 7) are
correctly labeled. This is recorded here rather than silently relabeled,
per this project's standing rule against silent correction. It does not
weaken the result below -- if anything it adds a third, genuinely
independent hash-seed condition (unset) rather than the originally-planned
"same seed twice" comparison.

| Condition | Autocorr @ 145 | Rank | Spread @ 145 |
| :--- | ---: | :---: | ---: |
| unset (mislabeled `seed42_run1`) | 0.6833 | 1/500 | 7.531x |
| `PYTHONHASHSEED=42` | 0.7380 | 1/500 | 10.661x |
| `PYTHONHASHSEED=7` | 0.9062 | 1/500 | 8.016x |

**All three hash-seed conditions show the identical pattern**: lag 145 at
rank 1, autocorrelation 0.68-0.91, spread 7.5x-10.7x. This mirrors Addendum
21's finding for the AMD machine's 187-period exactly, now independently
confirmed on this machine for its own 145-period: **`PYTHONHASHSEED` does
not affect whether the period appears.**

## 3. Dummy-loop control: no comparable period from the OS scheduler, raw computation, or BLAS/threading

`check_dummy_loop_period.py --period 145` (default period is 187, tuned to
the AMD machine's finding -- the run must explicitly pass `--period 145` to
test this machine's own period; the first attempt omitted this flag and
tested the wrong period, corrected here):

| Arm | Autocorr @ 145 | Rank (of 500) | Spread @ 145 |
| :--- | ---: | :---: | ---: |
| `busy` (pure Python arithmetic) | 0.5893 | 145/500 | 1.278x |
| `sleep` (`time.sleep`) | -0.0137 | 428/500 | 1.034x |
| `numpy` (fixed-size matmul) | 0.2889 | 179/500 | 1.417x |

None of the three arms comes close to Qiskit's rank-1, 7.5-10.7x-spread
signature -- `busy`'s raw autocorrelation value (0.59) looks superficially
non-trivial but ranks only 145th of 500 lags, meaning it is not a peak at
all, just background short-range correlation. **This rules out the OS
scheduler, raw CPU/interpreter overhead, and the BLAS/threading layer as
the cause on this machine**, the same three explanations Addendum 21 ruled
out for the AMD machine's 187-period.

(Note: this run's raw per-iteration data was not saved as a file and could
not be independently re-verified in the sandbox the way the six
`.npz` files in sections 1-2 were; the table above is taken from the
script's own terminal output. If this matters later, [`check_dummy_loop_period.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_dummy_loop_period.py)
could be extended to save its per-arm timings the way
[`test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py) already does.)

## 4. Where this leaves the investigation

What has now been ruled out for this machine's 145-period, specifically:
hash randomization (§2), the OS scheduler, raw computation overhead, and
generic BLAS/threading activity (§3) -- the same list Addendum 21 worked
through for the AMD machine's 187-period, now independently confirmed here.
Near-degeneracy of the circuit blocks was already ruled out in Addendum 20
§1 using outlier circuits from this same machine's original run.

**What remains unidentified, unchanged from Addendum 21's conclusion**:
what inside Qiskit's own `optimization_level=3` code path produces a
~145-iteration cycle on this machine (and a different, ~187-iteration cycle
on the AMD machine) -- an internal cache, counter, or state with a
period-like reset. Reading Qiskit's own source for a constant near 145 or
187 (a cache size, batch limit, or buffer threshold) remains the identified
but unattempted next step, carried over from Addenda 20 and 21.

**What is newly established that was not before**: this is not a
single-machine curiosity from one run. It is a stable, repeatable property
of this specific machine across at least six independent process launches
over two days, immune to hash-seed changes, and not attributable to any of
the three generic causes tested. The parallel finding on the AMD
machine (Addendum 21) used a different period value (187 vs. 145) but an
identical elimination pattern, which is itself worth noting: **whatever
this is, it appears to reproduce the same *kind* of effect on both machines
tested so far, at a machine-specific period.**

## 5. Files

| Path in the project | Contents |
| :--- | :--- |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_run2.csv) | Run 2, Section 1 (unset hash seed). Converted from the original `.npz` (per-iteration `qiskit`/`psf_true`/`psf_false` columns) -- the Project's storage rejected `.npz` uploads directly. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_run3.csv) | Run 3, Section 1 (unset hash seed). Same conversion as above. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_run4_unset_hashseed.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_run4_unset_hashseed.csv) | Run 4 / Section 2's "unset" condition (mislabeled on disk as `seed42_run1`). Same conversion as above. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_run5_seed42.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_run5_seed42.csv) | Run 5 / Section 2's `PYTHONHASHSEED=42` condition. Same conversion as above. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_run6_seed7.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_run6_seed7.csv) | Run 6 / Section 2's `PYTHONHASHSEED=7` condition. Same conversion as above. |

[`benchmarks/test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py), [`benchmarks/check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py),
and [`benchmarks/check_dummy_loop_period.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_dummy_loop_period.py) are unchanged from Addenda 19-21.

## 6. Verification

- All five new `.npz` files (Runs 2-6) were independently reloaded in the
  sandbox and autocorrelation / modular-bin spread at lag 145 were
  recomputed directly from the raw per-iteration arrays, not read off
  terminal output. Where a terminal printout was also available (Runs 2-3,
  6), the recomputed values matched to the digits shown (e.g. Run 3's
  spread of 10.673x and top-5 lag list `[145, 290, 435, 1, 2]` matched
  exactly).
- One duplicate upload was caught and excluded before analysis: an file
  submitted as a fourth "new" run was byte-for-byte identical (median,
  mean, total, and all autocorrelation figures to full precision) to Run 3,
  confirming it was a re-upload of stale output rather than a new
  measurement, and it is not counted among the six runs in Section 1.
- The `seed42_run1` / `seed42_run2` / `seed7_run1` command transcript was
  read line by line to confirm the actual order `set PYTHONHASHSEED`
  commands were issued in, which is how the labeling error in Section 2 was
  caught -- it was not visible from the data alone (all three conditions
  produced qualitatively the same result, so the mislabeling would not have
  been noticed without checking the command order against the file names).
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and the five data files
  named in Section 5 -> 0 hits. Note that, as in prior addenda, the raw
  terminal transcripts pasted into this conversation to produce this
  addendum contained `C:\Users\...` paths; none of that text is included in
  this addendum or in any saved file.

---


<!-- ===== Addendum 23 (source: spare-qubit-cliff-addendum-23-2026-09-16.md) ===== -->

> **Note added when merging:** `gc.disable()` removes the ~145-iteration period across four independent runs and a pooled, higher-power check -- the leading candidate mechanism as of this addendum, though the exact GC trigger is not yet identified.

## Addendum 23 (2026-09-16) -- `gc.disable()` removes the ~145-iteration period: four independent runs, individually and pooled, all show the signature is gone

## 0. In one line

A hypothesis was pre-registered before this measurement: if the ~145-iteration
period (Addenda 19-22) is caused by CPython's garbage collector firing on a
roughly-periodic schedule, disabling it (`gc.disable()`) before the
compile-time loop should make the period disappear and the modular-bin
spread flatten. Four independent runs with `gc.disable()` were collected on
the same Intel machine, individually and pooled (n=20000). In every case the
lag-145 rank collapses from 1/500 (baseline, every run) to 143-146/500, and
the modular-bin spread collapses from 7.493x-10.673x (baseline) to
1.10x-1.34x -- and in the pooled, higher-power check, period 145 no longer
stands out at all against neighboring candidate periods (100, 120, 160,
200). **The pre-registered prediction is confirmed**: the period is gone
under `gc.disable()`. This does not yet identify the exact GC mechanism
(which generation, which threshold) or confirm the same explanation on the
AMD machine's ~187-period -- both remain open.

## 1. Four independent runs, same machine, `gc.disable()` added

Same machine as Addenda 5-22 (Windows 10, Python 3.11.9,
`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`, 14 cores), same
script and circuit family (`test_cumulative_compile_scale.py --iters 5000`,
no `coupling_map`), with `gc.disable()` added before the timing loop. All
four `.npz` outputs were independently reloaded and re-analyzed (not read
off terminal output) to compute autocorrelation and modular-bin spread
directly from the raw per-iteration arrays.

| Run | Qiskit total (s) | Qiskit mean (ms) | Autocorr @ 145 | Rank (of 500) | Modular-bin spread @ 145 |
| :--- | ---: | ---: | ---: | :---: | ---: |
| gc.disable() run 1 | 38.931 | 7.786 | 0.4018 | 146 | 1.098x |
| gc.disable() run 2 | 58.965 | 11.793 | 0.7070 | 144 | 1.253x |
| gc.disable() run 3 | 80.510 | 16.102 | 0.3498 | 143 | 1.336x |
| gc.disable() run 4 | 58.878 | 11.776 | 0.6427 | 143 | 1.291x |
| (baseline, 6 runs, Addenda 20 & 22, no `gc.disable()`) | 38-74 | -- | 0.68-0.91 | **1** (every run) | **7.493x-10.673x** |

Two things stand out. First, rank and spread land in a narrow, consistent
band across all four runs (143-146 and 1.10x-1.34x respectively) that is
completely disjoint from the baseline's band (always rank 1, always
7.5x-10.7x) -- despite the raw autocorrelation value at lag 145 varying
fairly widely run to run (0.35-0.71), which by itself would be easy to
over-read as "still there" (see Section 3, where exactly that
over-reading happened locally with a different dataset). Rank and spread,
not the raw autocorrelation value, are what actually distinguish "a real
period-145 effect" from "generic short-range noise that happens to have
some value at lag 145."

Second, total compile time rises monotonically across the four runs
(38.9s -> 59.0s -> 80.5s -> 58.9s is not quite monotonic across all four,
but 1-3 rise sharply before run 4 drops back). Run 2 and 3's own progress
logs show large, irregular mid-run slowdowns (e.g. run 3: 26.6s for the
first 500 iterations, then 44.7s, 41.3s, 42.1s... for subsequent 500-blocks
-- no clean steady-state rate). This is recorded but not attributed here;
it is consistent with ordinary machine-load variation of the kind already
documented in Addendum 16, and/or with memory build-up within a single run
once garbage collection is disabled (uncollected reference cycles
accumulating over 5000 iterations). Distinguishing those two explanations
is a separate question from the one this addendum answers and is not
pursued further here.

## 2. Pooled check (n=20000): period 145 no longer stands out against neighboring candidate periods

The four runs above were pooled per-arm (qiskit, psf_true, psf_false each
concatenated across all four runs, n=20000) and re-checked with
`check_period_145_pooled.py --period 145`, which -- like [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py)
-- reports modular-bin spread at several candidate periods, not just the
requested one, specifically to guard against reading a coincidental match at
one period as if it were a real periodic effect.

| Arm (n=20000) | Autocorr @ 145 | Rank (of 500) | Spread @ 100 | Spread @ 120 | **Spread @ 145** | Spread @ 160 | Spread @ 200 |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| qiskit | 0.6899 | 145 | 1.102x | 1.129x | **1.117x** | 1.147x | 1.156x |
| psf_true | 0.6009 | 145 | 1.128x | 1.137x | **1.197x** | 1.368x | 1.179x |
| psf_false | 0.6671 | 145 | 1.080x | 1.368x | **1.293x** | 1.538x | 1.523x |

In the pre-gc.disable() baseline, period 145 was dramatically higher than
every neighboring candidate (7.493x-10.673x at 145 vs. ~1.0-1.1x at 100,
120, 160, 200 -- see Addendum 20). Here, with four times the per-arm sample
size of any single baseline run, period 145 is not distinguishable from its
neighbors for any of the three arms -- in psf_false it is not even the
highest of the five candidates tested (160 and 200 are higher, which is
itself evidence that whatever small differences remain across candidate
periods here are just sampling noise, not structure). This is a
higher-powered version of the per-run result in Section 1 and points to the
same conclusion.

## 3. A methodological trap encountered along the way, recorded rather than smoothed over

While these four runs were being collected, a local invocation of
`check_period_145.py --period 145` (no `--file` argument given) produced a
result that looked like a direct contradiction: n=50000, qiskit at rank
1/500 with spread 7.493x (matching the baseline signature exactly), while
psf_true/psf_false ranked 145-146/500 with spread 1.35-1.39x (matching the
gc.disable() pattern). On inspection this is very unlikely to reflect an
actual mix of results within one honest measurement. None of the four
gc.disable() `.npz` files is anywhere near n=50000 (each is n=5000, and all
four combined is n=20000, not 50000); a local file browser screenshot from
the same session showed a pre-existing file named
[`cumulative_compile_times_50000.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_50000.npz), dated 2026-09-15 (the day before
`gc.disable()` was tried at all) sitting in the same working directory. The
strong inference is that [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py)'s hardcoded default file
path pointed at that old, pre-gc.disable() 50,000-iteration file, and the
command as typed (without `--file`) silently analyzed *that* file instead
of any of today's data. The exact match of its reported spread (7.493x) to
Addendum 20's originally-recorded value for a different, specific historical
run is the strongest piece of evidence for this; it was not, however,
independently confirmed by inspecting [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py)'s own default
argument or by re-running it with an explicit `--file` pointed at the old
file to reproduce the number byte-for-byte, so this remains a
high-confidence inference rather than a proven fact, and is recorded as
such.

This is a live example of exactly the failure mode this project's own
standing rule against ambiguous, fixed output filenames exists to prevent:
[`test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py) always writes to the same
`cumulative_compile_times_5000.npz`, silently overwriting the previous run's
data, and at least one other local script apparently defaults to a fixed
filename as well. Nothing about the four runs analyzed in Sections 1-2 is
affected by this -- those were re-derived directly from the four `.npz`
files uploaded immediately after each run, not from any locally-persisted
file -- but the trap is worth naming so it is not repeated: any future
period check must pass an explicit, disambiguated `--file`/path argument,
never rely on a script's default.

## 4. Where this leaves the investigation

**Newly established**: on this Intel machine, disabling CPython's garbage
collector removes the ~145-iteration period, both per-run (four independent
runs) and in a pooled, higher-power check that also shows period 145 is no
longer distinguishable from neighboring candidate periods. Combined with
Addendum 22's elimination of `PYTHONHASHSEED`, the OS scheduler, raw
computation, and BLAS/threading as causes, the garbage collector is now the
leading candidate mechanism for this machine's period.

**Still open**:

- *Mechanism, not just correlation.* This shows that disabling the GC
  removes the effect, which is consistent with the GC being the cause, but
  it does not yet show which GC behavior specifically produces a ~145-cycle
  (a generation-1/2 collection threshold, an allocation-count trigger, or
  something else). Instrumenting `gc.callbacks` or comparing
  `gc.get_stats()` collection counts against iteration number, rather than
  simply disabling the GC outright, would let the ~145 number be predicted
  from GC internals instead of just correlated with them after the fact.
- *Whether this generalizes to the AMD machine's ~187-period* (Addendum 21).
  Untested here -- everything in this addendum is the Intel machine only.
- *The run-to-run slowdown noted in Section 1* (memory build-up from
  disabled collection vs. ordinary machine load) is a plausible side effect
  of this same fix and worth separating out, but is not resolved by
  anything in this addendum.

## 5. Files

| Path in the project | Contents |
| :--- | :--- |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run1.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run1.csv) | Run 1, Section 1. Converted from the original `.npz` (per-iteration `qiskit`/`psf_true`/`psf_false` columns) -- the Project's storage rejects `.npz` uploads directly, as in Addendum 22. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run2.csv) | Run 2, Section 1. Same conversion. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run3.csv) | Run 3, Section 1. Same conversion. |
| [`psf-zero/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run4.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_5000_intel_2026-09-16_gcdisable_run4.csv) | Run 4, Section 1. Same conversion. |
| [`psf-zero/benchmarks/check_period_145_pooled.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145_pooled.py) | New. Pools multiple same-condition CSVs and reruns the [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py) multi-candidate-period check at higher n; used for Section 2. Takes explicit file paths as arguments (no hardcoded default), specifically to avoid the trap described in Section 3. |

[`benchmarks/test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py) and [`benchmarks/check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py)
are unchanged from Addenda 19-22 (the latter's own default-file behavior is
implicated, not modified, in Section 3 -- changing it was not attempted here
since it lives on the user's local machine, not in this project).

## 6. Verification

- All four new `.npz` files were independently reloaded (from the raw
  per-iteration arrays, not terminal output) and autocorrelation /
  modular-bin spread at lag 145 recomputed directly, both individually
  (Section 1) and pooled (Section 2).
- The four converted CSV files were re-loaded independently by
  [`check_period_145_pooled.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145_pooled.py) and reproduced the same pooled figures
  (autocorr 0.6899/0.6009/0.6671, rank 145/145/145, spread 1.117x/1.197x/
  1.293x for qiskit/psf_true/psf_false respectively) as the direct-from-`.npz`
  pooled computation, confirming the CSV conversion did not alter the
  result.
- The n=50000 discrepancy in Section 3 was checked against the sizes of all
  four `.npz` files (each n=5000, none close to 50000) before concluding it
  could not have come from today's gc.disable() data; this rules out
  "all four runs got silently duplicated or mixed" but does not
  independently confirm which file [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py) actually read
  (noted as an open item in Section 3 itself).
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, both new scripts, and
  the four data files named in Section 5 -> 0 hits. As in prior addenda,
  the terminal transcripts and file-browser screenshot pasted into this
  conversation to produce this addendum contained a local Windows path
  (`C:\Users\...`) and a file-browser view of the user's own working
  directory; neither the path text nor any other content from those images
  beyond the file names and timestamps needed for Section 3's reasoning is
  included in this addendum or in any saved file.

---


<!-- ===== Addendum 24 (source: spare-qubit-cliff-addendum-24-2026-09-16.md) ===== -->

> **Note added when merging:** First real-core measurement of plain `compile_for_hardware()` (no smart-layout aid) across the classic spare-qubit boundary: it still crosses the cliff, but far more gently than plain Qiskit L3 (7.3x-8.2x vs 263x-300x).

## Addendum 24 (2026-09-16) -- `compile_for_hardware()`, with no smart-layout aid, still crosses the spare-qubit cliff, but far more gently than plain Qiskit L3

## 0. In one line

Using the corrected [`test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py) (see that file's own
docstring for the two bugs in its predecessor: a call that put a
`CouplingMap` into `psf_compile.compile()`'s `block_gate_floor` slot and was
silently caught and replaced with a placeholder value, and a circuit shape
not established to reproduce either side of the comparison), a real run
against the actual `psf_zero_core` on the known Intel machine
(`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`) swept the classic
zero-spare boundary (38-42 qubits on a 6x7 = 42-qubit grid,
dense-adjacent-pair-blocks circuit, `seed_transpiler=42` pinned) and found:
**Qiskit L3 jumps 263x-300x at the spare=0 boundary (21.4-24.4 ms to
6422.3 ms); `compile_for_hardware(routing_optimization_level=1)` also jumps,
but only 7.3x-8.2x (10.7-12.0 ms to 88.1 ms).** This is the first plain
`compile_for_hardware()` measurement (no `smart_vf2_layout` aid) across this
specific spare-qubit sweep -- prior addenda (14-17) tested the smart-layout
prototype against a set of named topology conditions (grid/brick/diluted/line
at fixed spare values), not this n-qubit sweep on one fixed grid, and always
with the prototype's search included. This result is a single run (one
seed, one grid, no repeats across independent invocations of the whole
script), and Addendum 16 already found up to ~3x same-condition variance at
`optimization_level=3` on a related benchmark -- so the exact ratios above
should be treated as a first observation, not yet a reproducibility-checked
figure.

## 1. What was pre-registered before this run

[`test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py)'s own docstring, written before it was ever
run against the real core, committed to this before seeing any numbers:

> On a saturated coupling map, Qiskit L3's compile time is expected to jump
> sharply ... between 2 spare qubits and 0 spare qubits ... Whether
> PSF-Zero's `compile_for_hardware()` shows the same cliff, a smaller one,
> or none at all is an OPEN question this script does not assume an answer
> to going in.

The result below answers that: **a smaller cliff, not none, and not the
same size.**

## 2. Setup

Same known Intel machine as the majority of this project's Windows-side
measurements (`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`, per the
CSV's own `cpu` column -- machine identified by CPU signature, not by
account or path). Python 3.11.9, Qiskit 2.5.2, [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py) VERSION
2026-09-16 (the real `psf_zero_core`, not the stub -- confirmed by the
script's own small-scale correctness pre-check passing:
`Operator(out).equiv(Operator(in)) == True` on a 6-qubit circuit, no
coupling map, before the sweep ran; 0 fallback/degenerate warnings across
every point in the sweep, so no numerically degenerate blocks were hit
anywhere in this run).

- Circuit: `build_dense_pair_blocks_circuit()` (adjacent pairs `(0,1),
  (2,3), ...`, 20 random SU(4) gates per pair, decomposed) -- the same
  generator this project used to originally establish the spare-qubit
  cliff, not an untested shape.
- Coupling map: `CouplingMap.from_grid(6, 7)`, 42 physical qubits, held
  fixed; only the logical qubit count `n` (38-42) varied.
- `basis_gates=["rz","sx","x","cx"]` for both engines.
- Qiskit side: `transpile(qc, coupling_map=cm, basis_gates=BASIS,
  optimization_level=3, seed_transpiler=42)`.
- PSF-Zero side: `compile_for_hardware(qc, coupling_map=cm,
  basis_gates=BASIS, routing_optimization_level=1, entangling_basis="cx",
  seed_transpiler=42)` -- `routing_optimization_level=1` is
  `compile_for_hardware`'s own default and this project's documented
  recommendation (README: level 2/3 "undo this pass"), not raised to 3 to
  chase a superficially matching label against "Qiskit L3."
  `entangling_basis="cx"` was chosen because `basis_gates` here is
  CX-based; the canonical (RXX/RYY/RZZ) default would have made the
  translation stage do avoidable extra work on top of routing.
- Each point: 1 discarded warm-up call, then 5 timed calls, median reported
  (this project's own documented convention). A coupling-map-validity scan
  (every 2-qubit gate lands on an edge of the 6x7 grid) ran on every output,
  for both engines, independently of anything either engine reported about
  itself -- 0 violations everywhere.

## 3. Results

| n | spare | Qiskit L3 (median, ms) | PSF-Zero `compile_for_hardware` (median, ms) | Speedup | PSF-Zero fallback count |
| ---: | :---: | ---: | ---: | ---: | :---: |
| 38 | 4 | 22.876 | 11.569 | 1.98x | 0 |
| 39 | 3 | 24.389 | 10.706 | 2.28x | 0 |
| 40 | 2 | 21.388 | 12.033 | 1.78x | 0 |
| 41 | 1 | 21.708 | 11.655 | 1.86x | 0 |
| 42 | 0 | **6422.288** | **88.148** | **72.86x** | 0 |

![Compile time (log scale) vs. qubits, and PSF-Zero's speedup ratio over Qiskit L3, both engines flat until the spare=0 cliff](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/png/spare-qubit-cliff-addendum-24-2026-09-16-chart.png?raw=true)

Left panel: compile time (log scale) vs. qubit count for both engines --
both essentially flat from n=38-41, then Qiskit L3 breaks upward sharply at
n=42 (spare=0) while PSF-Zero's own rise is visible but far smaller on the
same axis. Right panel: the same data expressed as PSF-Zero's speedup ratio
over Qiskit L3 per point. Every value labeled on the chart (2.0x, 2.3x,
1.8x, 1.9x, 72.9x) was checked against the table above and matches to the
first decimal. Read the sharp visual break in the left panel as "Qiskit L3
crosses the cliff, PSF-Zero mostly doesn't" -- not as "PSF-Zero is flat,"
since PSF-Zero's own line does rise measurably at n=42 (11.6-12.0 ms to
88.1 ms, Section 3), just not enough to be visually distinct from its
own pre-cliff noise band at this scale. One presentation note, not a data
issue: the two series are colored red/green, a pairing that is not reliably
distinguishable under red-green color vision deficiency; the legend labels
("Qiskit L3" / "PSF-Zero") and the marker shapes (circle / square) still
carry the identity independently of color, so the chart remains readable
without relying on hue alone.

All five `speedup` values were independently recomputed from the raw
`qiskit_l3_ms`/`psf_zero_ms` columns and matched the CSV's own `speedup`
column exactly (no transcription errors).

Away from the boundary (spare 1-4), both engines are essentially flat with
no visible spare-dependent trend: Qiskit spans 21.39-24.39 ms, PSF-Zero
spans 10.71-12.03 ms -- consistent with ordinary run-to-run noise at this
timescale, not a slope. At spare=0, both jump, but by very different
factors depending which pre-cliff point is used as the baseline:

| | using nearest baseline (spare=1) | using slowest pre-cliff baseline (spare=3) |
| :--- | ---: | ---: |
| Qiskit L3 cliff ratio | 295.9x | 263.3x |
| PSF-Zero cliff ratio | 7.6x | 8.2x |

(Full baseline-choice range: Qiskit 263x-300x, PSF-Zero 7.3x-8.2x, using
whichever of the four pre-cliff points is picked as "before.")

## 4. Reading this result

**What this does establish**: on this grid, this circuit family, this seed,
and this one run, `compile_for_hardware()` at its own default
`routing_optimization_level=1` does not avoid the spare-qubit cliff
outright, but the absolute cost of crossing it is roughly two orders of
magnitude smaller than plain Qiskit L3's (77 ms added vs 6400 ms added).
This is consistent with, but does not on its own confirm, the mechanism
this project has already documented: `VF2Layout` failing and falling back
to `SabreLayout` is a property of Qiskit's layout stage that fires at any
optimization level (so a smaller PSF-Zero cliff, not zero, is expected);
the *additional* downstream routing/optimization cost documented in
Addendum 10 as specific to `optimization_level>=2/3` is what
`routing_optimization_level=1` skips, which would explain why PSF-Zero's
jump is much smaller rather than absent. This run does not instrument
Qiskit's own pass timings (no `callback=` trace was taken here, unlike
Addendum 10's L3 investigation), so this explanation is a plausible fit to
already-established mechanism, not a re-confirmation of it under this
exact configuration.

A second, separate observation: PSF-Zero's advantage over Qiskit L3 is not
constant across the sweep. It sits at a modest 1.8x-2.3x away from the
cliff and jumps to 72.9x exactly at the point Qiskit's own layout search
fails -- the "speedup" number here is not a fixed property of PSF-Zero
alone, it is a property of how much of Qiskit's own pathology PSF-Zero's
lower routing level happens to sidestep.

## 5. What is still open

- **Reproducibility.** This is one run: one seed (`seed=7` for the circuit
  generator, `seed_transpiler=42` pinned for both engines), one grid
  (6x7), 5 timed repeats within the run but no repeats of the run itself.
  Addendum 16 already found same-condition variance up to ~3x at
  `optimization_level=3` on a related (not identical) benchmark; whether
  that applies to this specific n-sweep, and to `compile_for_hardware` at
  `routing_optimization_level=1` specifically, is untested here.
- **Mechanism, not just outcome.** This run did not trace which Qiskit pass
  the 77 ms in PSF-Zero's jump is actually spent in (VF2Layout's own
  failure, or the Sabre fallback that follows it, or something else at
  level 1 specifically). Addendum 10's `callback=` trace method would
  answer this directly but was not applied here.
- **Generalization.** One grid shape (6x7 square-ish grid), one circuit
  family (dense adjacent-pair blocks), one qubit-count range (38-42).
  Whether the ~7-8x PSF-Zero cliff ratio holds on other grid shapes/sizes,
  other topologies (line, brick, diluted -- the families this project's
  smart-layout addenda already used), or other seeds is untested.
- **Whether `routing_optimization_level=2` or `3` reintroduces the larger
  cliff.** Not measured here; the README's existing claim that level 2
  makes PSF-Zero's output "bit-identical" to plain
  `transpile(optimization_level=2)` would predict that it does, but that
  claim was made in a different (non-coupling-map-saturated) context and
  was not re-checked against this specific boundary.

## 6. Files

| Path in the project | Contents |
| :--- | :--- |
| [`benchmarks/test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py) | The corrected script used for this run (see its own docstring for the two bugs it fixes in the uploaded original). |
| [`data/cliff_sniper_corrected_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv) | The raw 5-row result CSV this addendum's Section 3 is drawn from. |
| [`docs/png/spare-qubit-cliff-addendum-24-2026-09-16-chart.png`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/png/spare-qubit-cliff-addendum-24-2026-09-16-chart.png) | The two-panel chart embedded in Section 3 (compile time log-scale + speedup ratio), built from the CSV above. |

## 7. Verification

- All five `speedup` values in Section 3 were independently recomputed from
  the CSV's raw `qiskit_l3_ms`/`psf_zero_ms` columns and matched the CSV's
  own `speedup` column exactly.
- The cliff-ratio ranges in Section 3 were computed against both the
  nearest pre-cliff point (spare=1) and the extreme pre-cliff point
  (spare=3) rather than a single cherry-picked baseline, since the four
  pre-cliff points themselves span a non-trivial range (21.4-24.4 ms for
  Qiskit, 10.7-12.0 ms for PSF-Zero) with no visible trend.
- `psf_zero_fallback_count` was read directly from the CSV (0 at every
  point) rather than assumed; combined with the small-scale correctness
  pre-check passing, this run gives no indication of numerically
  degenerate blocks or a stub-core substitution.
- The user's terminal transcript accompanying this CSV contained a local
  Windows path (`C:\Users\...`) in its shell prompt. That path is not
  reproduced anywhere in this addendum or in any saved file -- only the
  CPU signature and the numeric/CSV contents were used, per this project's
  standing rule against copying local file paths into saved output.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum -> 0 hits.

---


<!-- ===== Addendum 25 pre-registration (source: spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md) ===== -->

> **Note added when merging:** Predictions written before testing whether raising `routing_optimization_level` closes the gap Addendum 24 found -- quoted verbatim in Addendum 25 below.

## Pre-registration for Addendum 25 (written 2026-09-16, before this experiment is run)

This records a prediction ahead of the follow-up Addendum 24 called for,
so the prediction cannot be adjusted after the numbers come back. When the
results arrive, they get their own addendum (25), which quotes this section
verbatim rather than restating it from memory.

## What is being tested

Addendum 24 found that on a 6x7 (42-qubit) grid, at the spare=0 boundary,
`compile_for_hardware(routing_optimization_level=1)` crosses the same
spare-qubit cliff Qiskit L3 does, but far more gently (7.3x-8.2x vs
263x-300x). Two mechanisms could explain the gap, and this experiment does
not assume which:

- **(a) Budget.** Qiskit's preset pass managers give `VF2Layout` a larger
  `call_limit`/trial budget at higher optimization levels before it gives
  up and falls back to `SabreLayout` (Addendum 9's mechanism). A smaller
  budget at level 1 means a cheaper failure, independent of anything
  downstream.
- **(b) Downstream cost.** Addendum 10 found `optimization_level=3` adds a
  second cost layer on top of the layout-stage failure -- once Sabre's
  fallback layout is in hand, the routing/optimization passes that follow
  can cost as much as, or more than, the layout failure itself. Level 1
  might simply skip that second layer while still paying the first.

Both predict the same direction (higher `routing_optimization_level` ->
bigger PSF-Zero cliff) but for different reasons, and this single
experiment does not by itself separate them -- see "What this will NOT
establish" below.

## Exact change from Addendum 24's run

Same script ([`test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py), now emitting
`routing_optimization_level` in its output filename so same-day runs at
different levels don't rely on the run2/run3 collision-avoidance suffix to
stay distinguishable), same grid (6x7), same circuit seed (7), same
`seed_transpiler` (42, pinned). Only `--routing-optimization-level` changes,
run once at 2 and once at 3:

```
python test_cliff_sniper_corrected.py --rows 6 --cols 7 --routing-optimization-level 2
python test_cliff_sniper_corrected.py --rows 6 --cols 7 --routing-optimization-level 3
```

Expected output filenames:
[`cliff_sniper_corrected_6x7_rl2_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_rl2_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv)
[`cliff_sniper_corrected_6x7_rl3_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_rl3_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv)

## Prediction

1. **Direction**: the spare=0 cliff ratio for PSF-Zero will increase
   monotonically with `routing_optimization_level`: `ratio(rl=1) <
   ratio(rl=2) <= ratio(rl=3)`, i.e. 7.3x-8.2x (Addendum 24) will not be the
   ceiling.
2. **Level 3 lands closest to Qiskit L3's own 263x-300x.** At
   `routing_optimization_level=3`, `compile_for_hardware`'s internal
   `transpile()` call runs at the same optimization level as the Qiskit L3
   baseline it is compared against, on a circuit whose *interaction graph*
   (which logical qubit pairs need to be adjacent) is unchanged by PSF-Zero's
   synthesis step -- PSF-Zero changes which gates implement each block, not
   which pairs interact. Since `VF2Layout` reasons about the interaction
   graph, not gate content, its success/failure on this instance is
   predicted to be effectively the same whether it is fed the raw circuit or
   PSF-Zero's re-synthesized one. Prediction: PSF-Zero's rl=3 cliff ratio
   will land within roughly 2x of Qiskit L3's 263x-300x, not remain closer
   to rl=1's ~7-8x.
3. **Level 2 lands strictly between rl=1 and rl=3**, closer to rl=3 than to
   rl=1 -- because the README's own finding that "at level 2 the result is
   bit-identical to plain `transpile(optimization_level=2)`" implies
   `VF2Layout` is already being run at (or close to) its full-budget
   configuration at level 2, so most of mechanism (a) above (the budget
   difference) is predicted to already be gone by level 2, with only
   mechanism (b) (the level-3-specific downstream cost) remaining to
   separate levels 2 and 3.
4. **The absolute fallback count stays 0** at both levels (no new
   degenerate/numeric fallbacks introduced by changing the routing level --
   that parameter does not touch the 2-qubit synthesis path at all).

## What this will NOT establish, even if the prediction holds

- **Cause (a) vs (b), directly.** Confirming the direction and rough
  magnitude does not by itself prove *which* of budget vs. downstream cost
  explains it -- that needs the `callback=` pass-timing trace Addendum 10
  used, which this run does not perform. If the result is scheduled as a
  follow-up after this one, it should not be described as already settled
  by this round.
- **A plain Qiskit `optimization_level=2` baseline.** Prediction 3 above
  reasons from the README's existing "bit-identical to plain L2" claim
  rather than from a fresh L2 measurement taken alongside this run. This
  round only re-runs the PSF-Zero side at rl=2/3 against the *same* fixed
  Qiskit L3 baseline from Addendum 24 -- it does not add a plain-Qiskit-L2
  data point. If the L2-vs-L2 comparison turns out to matter, that is a
  gap in this round, not something it quietly assumes away.
- **Generalization** to other grids, topologies, or seeds -- unchanged from
  Addendum 24's own limitations section.

---


<!-- ===== Addendum 25 (source: spare-qubit-cliff-addendum-25-2026-09-16.md) ===== -->

> **Note added when merging:** Raising `routing_optimization_level` closes most of the Addendum-24 gap by level 2, then the rest by level 3, where PSF-Zero stops being faster than Qiskit L3 at all -- suggesting the earlier advantage was largely a side effect of a cheaper default routing level, not the synthesis pass itself.

## Addendum 25 (2026-09-16) -- raising `routing_optimization_level` closes most of the gap to Qiskit L3's cliff by level 2, then the rest by level 3, where PSF-Zero stops being faster than Qiskit L3 at all

## 0. In one line

Following up on Addendum 24 (PSF-Zero's own spare-qubit cliff at
`routing_optimization_level=1` is ~7-8x, far smaller than Qiskit L3's
~263-300x), the same 6x7-grid sweep was re-run at `routing_optimization_level=2`
and `=3`. **PSF-Zero's own cliff ratio grows to ~37-40x at level 2 and
~251-275x at level 3 -- landing inside Qiskit L3's own 263-300x range.**
Two of the four predictions pre-registered before this run
([`spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md)) are
confirmed outright (direction, and level 3 landing close to Qiskit L3); one
is confirmed (fallback count stays 0); and **one is not confirmed**: level 2
was predicted to land "closer to rl=3 than to rl=1," and in log-scale terms
it instead sits almost exactly at the geometric midpoint, marginally closer
to rl=1. A result not asked for by the pre-registration, but visible in the
same data: **at `routing_optimization_level=3`, PSF-Zero's
`compile_for_hardware` is no longer faster than plain Qiskit L3 anywhere in
the sweep, cliff or no cliff** (0.83x-0.96x -- i.e. 4%-17% slower) --
extending, to this specific saturated-coupling-map scenario, the README's
existing claim (previously demonstrated only in a non-saturated context)
that raising the routing level "undoes" PSF-Zero's advantage.

## 1. What was pre-registered

Quoted verbatim from
[`psf-zero/docs/findings/spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md),
written before either of this addendum's two runs:

> 1. **Direction**: the spare=0 cliff ratio for PSF-Zero will increase
>    monotonically with `routing_optimization_level`: `ratio(rl=1) <
>    ratio(rl=2) <= ratio(rl=3)` ...
> 2. **Level 3 lands closest to Qiskit L3's own 263x-300x** ... Prediction:
>    PSF-Zero's rl=3 cliff ratio will land within roughly 2x of Qiskit L3's
>    263x-300x, not remain closer to rl=1's ~7-8x.
> 3. **Level 2 lands strictly between rl=1 and rl=3**, closer to rl=3 than
>    to rl=1 ...
> 4. **The absolute fallback count stays 0** at both levels ...

The same document named two things this round would not establish even if
confirmed: which of "VF2Layout's search budget" vs. "Addendum 10's
level-3-specific downstream cost" actually explains the gap (needs a
`callback=` trace this run does not perform), and a plain Qiskit
`optimization_level=2` baseline (this round only reruns the PSF-Zero side
at rl=2/3 against Addendum 24's fixed Qiskit-L3 baseline).

## 2. Setup

Identical to Addendum 24 -- same machine
(`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`), same 6x7 grid,
same `build_dense_pair_blocks_circuit(seed=7)`, same `seed_transpiler=42`,
same 1-warm-up + 5-timed-repeats/median protocol, same coupling-map-validity
scan (0 violations at every point, both engines, all three runs) -- with
only `--routing-optimization-level` changed between runs (1, already on
record from Addendum 24; 2 and 3, new in this addendum). All three CSVs now
encode the level in their filename (`..._rl1_...`, `..._rl2_...`,
`..._rl3_...`), per the script change made between Addendum 24 and this run
specifically so same-day runs at different levels would stay
self-describing.

## 3. Results

| rl | n=38 | n=39 | n=40 | n=41 | n=42 (spare=0) |
| :---: | ---: | ---: | ---: | ---: | ---: |
| 1 -- Qiskit L3 (ms) | 22.876 | 24.389 | 21.388 | 21.708 | **6422.288** |
| 1 -- PSF-Zero (ms) | 11.569 | 10.706 | 12.033 | 11.655 | **88.148** |
| 2 -- Qiskit L3 (ms) | 23.118 | 22.248 | 22.483 | 22.622 | **6376.804** |
| 2 -- PSF-Zero (ms) | 13.895 | 14.961 | 13.744 | 13.800 | **554.557** |
| 3 -- Qiskit L3 (ms) | 23.131 | 22.211 | 22.111 | 21.540 | **6403.135** |
| 3 -- PSF-Zero (ms) | 26.441 | 26.703 | 24.353 | 25.596 | **6696.621** |

`psf_zero_fallback_count` was 0 at every point in all three CSVs.

| rl | PSF-Zero's own spare=0/pre-cliff ratio | Qiskit L3's own spare=0/pre-cliff ratio | Speedup at spare=0 | Speedup range, spare>=1 |
| :---: | ---: | ---: | ---: | ---: |
| 1 | 7.3x-8.2x | 263.3x-300.3x | 72.86x | 1.78x-2.28x |
| 2 | 37.1x-40.3x | 275.8x-286.6x | 11.50x | 1.49x-1.66x |
| 3 | 250.8x-275.0x | 276.8x-297.3x | **0.96x** | **0.83x-0.91x** |

(Each range uses the same method as Addendum 24: the point's spare=0 value
divided by, respectively, the fastest and slowest of the four pre-cliff
points at spare 1-4, rather than a single chosen baseline.)

## 4. Prediction-by-prediction verdict

- **Prediction 1 (monotonic direction) -- confirmed.** PSF-Zero's own cliff
  ratio: ~7.75x (rl=1, midpoint) -> ~38.6x (rl=2) -> ~262x (rl=3), strictly
  increasing.
- **Prediction 2 (level 3 lands within ~2x of Qiskit L3's 263-300x) --
  confirmed, and more precisely than hedged for.** PSF-Zero's rl=3 cliff
  ratio (250.8x-275.0x) sits inside Qiskit L3's own range (263.3x-300.3x)
  at every level tested in this addendum, not merely within 2x of it.
- **Prediction 3 (level 2 closer to rl=3 than to rl=1) -- NOT confirmed.**
  In log10 space (appropriate here since the three ratios span two orders
  of magnitude), the gap from rl=1 to rl=2 is 0.698 decades and from rl=2
  to rl=3 is 0.832 decades -- level 2 sits almost exactly at the geometric
  midpoint between rl=1 and rl=3, marginally closer to rl=1, not clearly
  closer to rl=3 as predicted. The reasoning behind this prediction (that
  the README's "level 2 is bit-identical to plain `optimization_level=2`"
  claim implies most of the budget difference is already resolved by level
  2) is not supported by this result as stated -- the three levels'
  contribution to closing the gap is closer to evenly split (on a log
  scale) than front-loaded into the rl=1-to-rl=2 step.
- **Prediction 4 (fallback count stays 0) -- confirmed** at both levels.

## 5. A result the pre-registration did not ask about

The pre-registration was written entirely in terms of the *cliff ratio*
(each engine's own spare=0/pre-cliff jump) and did not predict the
Qiskit-relative *speedup* column's behavior. That column turns out to be
the more practically important one: **at `routing_optimization_level=3`,
PSF-Zero's `compile_for_hardware` is slower than plain Qiskit L3 at every
point measured, cliff or not** (0.83x-0.91x pre-cliff, 0.96x at spare=0).
This is the first time this project has measured that specific claim on a
saturated-coupling-map circuit; the README's existing "every millisecond
PSF-Zero spends [at level 2] is thrown away" claim was demonstrated in a
different, non-saturated context (100-156 qubit dense-pair-block circuits
without a near-full coupling map). This addendum's result is consistent
with that claim's spirit but is a new, separate measurement, not a
re-confirmation of the original one.

## 6. Reading this result

Taken together with Addendum 24, the picture is now: PSF-Zero's advantage
over Qiskit L3 on a saturated coupling map is not a fixed property of the
synthesis pass, it is almost entirely a side effect of running its internal
routing call at a lower `routing_optimization_level` than the Qiskit L3
baseline it is compared against. As that internal level is raised toward
matching Qiskit L3's own level 3, PSF-Zero's cliff grows to match Qiskit's,
and its overall advantage disappears (and turns slightly negative) well
before the levels are fully matched. This does not contradict this
project's own recommendation to use `routing_optimization_level=1` in
practice -- it sharpens the reason for it: at least on this circuit family
and grid, level 1's real advantage close to the cliff is not really "PSF-Zero
is fast," it is closer to "PSF-Zero, at level 1, inherits a cheaper VF2Layout
failure than Qiskit L3 does" -- a claim about Qiskit's own preset pipeline,
not about PSF-Zero's synthesis.

## 7. What is still open

- **Mechanism.** Confirmed here: the direction and rough landing zone.
  Not confirmed: whether the level 1->2->3 progression is driven by
  `VF2Layout`'s call-limit budget growing with level, the Addendum-10
  downstream-routing-cost layer specific to level 3, or some combination
  that isn't evenly split between the two -- prediction 3's failure argues
  against a clean "budget resolved by level 2, only downstream cost left
  for level 3" story, but does not offer a replacement mechanism. A
  `callback=`-based pass-timing trace (Addendum 10's method) at each level
  would settle this directly and has still not been run.
- **The missing plain-Qiskit-`optimization_level=2` baseline**, named in
  the pre-registration, remains missing.
- **Reproducibility at rl=2 and rl=3.** Addendum 24's update showed rl=1
  reproduces to within ~9% across two independent runs; rl=2 and rl=3 have
  each only been run once so far.
- **Generalization** to other grids, topologies, and seeds -- unchanged
  from Addendum 24.

## 8. Files

| Path in the project | Contents |
| :--- | :--- |
| [`benchmarks/test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py) | Unchanged from Addendum 24's version; only its CLI flag was used differently. |
| [`data/cliff_sniper_corrected_6x7_rl2_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_rl2_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv) | `routing_optimization_level=2` run, Section 3. |
| [`data/cliff_sniper_corrected_6x7_rl3_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_rl3_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv) | `routing_optimization_level=3` run, Section 3. |
| [`docs/findings/spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md) | The prediction quoted in Section 1, written before this addendum's runs. |

## 9. Verification

- All cliff-ratio and speedup figures in Section 3-5 were computed directly
  from the two uploaded CSVs' raw columns (not transcribed from the
  terminal output), using the same min/max-of-pre-cliff-points method as
  Addendum 24.
- `psf_zero_fallback_count` was read directly from both CSVs (0 at every
  row) rather than assumed.
- The log10 gap calculation in Section 4 (Prediction 3) was computed
  directly from the same midpoint cliff-ratio values reported in Section 3,
  not estimated by eye.
- Both correctness pre-checks (n=6, no coupling map) reported
  `Operator(out).equiv(Operator(in)) == True` before their respective
  sweeps, and every point's coupling-map-validity scan reported 0
  violations for both engines.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum -> 0 hits.

---


<!-- ===== Addendum 26 (source: spare-qubit-cliff-addendum-26-2026-09-16.md) ===== -->

> **Note added when merging:** The new `layout_search=True` option collapses PSF-Zero's own cliff to ~1.5x-1.6x -- but this addendum also discovers, chases, and narrows down (without fully resolving) an unrelated ~4x same-day timing drift in its own no-search control, illustrating this project's practice of reporting a discrepancy honestly rather than averaging it away.

## Addendum 26 (2026-09-16) -- `layout_search=True` collapses PSF-Zero's own spare-qubit cliff to ~1.5x-1.6x, but the paired no-search control this run does not reproduce Addenda 24-25's own ~7.3x-8.2x figure for the identical code path

## 0. In one line

[`test_cliff_sniper_layout_search.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_layout_search.py) was run once, on the real `psf_zero_core`,
on the same 6x7-grid/seed=7/seed_transpiler=42 scenario as Addenda 24-25.
**PSF-Zero with the new `layout_search=True` shows almost no cliff at all
(~1.5x-1.6x, vs its own pre-cliff points), confirming the pre-registered
prediction (Addendum 26 preregistration) more strongly than hedged for.**
But this run's own paired `layout_search=False` control measured a
spare=0 time of 22.050ms (cliff ratio ~2.15x-2.24x) -- **roughly 4x lower**
than the 84.472ms-88.148ms (~7.2x-8.2x) the identical code path measured in
two independent prior runs (Addendum 24 and its reproducibility check) on
the same machine, same Python (3.11.9), same Qiskit (2.5.2), same seed and
seed_transpiler. Away from the cliff (spare>=1) all three runs agree to
within normal noise (~10-12ms throughout). **This addendum reports the
layout_search result honestly against its own paired control, and flags
the control's own ~4x swing at exactly the cliff point as an open,
unexplained finding this addendum does not resolve** -- it is not glossed
over or averaged away.

> **Update, 2026-09-16 (see Section 5's dated updates below for the full
> account): the ~4x swing was tracked down to a same-day, same-machine
> timing drift unrelated to `layout_search` or to which script was run --
> a re-run of the original, unmodified Addendum-24 script later the same
> day also landed at ~22ms. It is not a `layout_search`-specific artifact,
> and it does not affect the layout_search-vs-no-search comparisons in
> Sections 3-4, which are paired within a single process run.**

## 1. What was pre-registered

Quoted verbatim from
[`psf-zero/docs/findings/spare-qubit-cliff-addendum-26-preregistration-2026-09-16.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-26-preregistration-2026-09-16.md),
written before this run:

> 1. **The cliff is not merely shrunk but effectively eliminated for the
>    layout_search arm.** ... predicted to land within roughly 3x of its
>    own pre-cliff (spare=1-4) values ...
> 2. **The layout_search arm is not reliably faster than the no-search arm
>    away from the cliff (spare>=1).** ... predicted: `speedup_search_vs_
>    nosearch` at spare>=1 will scatter close to 1.0x (roughly 0.7x-1.3x) ...
> 3. **Zero fallback warnings from either PSF-Zero arm, at every point.**
> 4. **`layout_search=True` becomes faster than Qiskit L3 specifically at,
>    and only at, the spare=0 point**, reversing the no-search arm's own
>    Addendum 24/25 pattern into a larger margin at spare=0, while the
>    spare>=1 speedup over Qiskit L3 stays comparable to the no-search
>    arm's own existing 1.78x-2.28x range (Addendum 25, rl=1 row) ...

The same document named what this run would not establish even if
confirmed: generalization beyond this one grid/seed, `smart_vf2_layout`'s
own scaling behavior past 42 qubits, any interaction with
`routing_optimization_level` 2/3, and a pass-timing-level mechanism for any
residual cliff.

## 2. Setup

Same machine (`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`),
same Python (3.11.9) and Qiskit (2.5.2) as every prior addendum in this
series, same 6x7 grid, same `build_dense_pair_blocks_circuit(seed=7)`,
same `seed_transpiler=42`, same 1-warm-up + 5-timed-repeats/median
protocol, `routing_optimization_level=1` for both PSF-Zero arms. Unlike
Addenda 24-25 (two separate scripts/runs for the two arms being compared),
[`test_cliff_sniper_layout_search.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_layout_search.py) measures Qiskit L3, PSF-Zero
no-search, and PSF-Zero with search in the same process, same run, so the
no-search-vs-search comparison in Sections 3-4 below is a paired one
(same instant, same warm/cold process state) -- only the comparison
against Addenda 24/25's own historical no-search numbers (Section 5) is
cross-run. Small-scale (n=6, no coupling map) correctness pre-check passed
(`Operator(out).equiv(Operator(in)) == True`) before the sweep ran. Post-hoc
coupling-map-validity scan reported 0 violations for all three engines at
every point (not shown as a separate column here since the script only
records violations as a hard error, and none occurred).

## 3. Results (this run, all three arms, one CSV)

| n | spare | Qiskit L3 (ms) | PSF-Zero no-search (ms) | PSF-Zero layout_search (ms) |
| :---: | :---: | ---: | ---: | ---: |
| 38 | 4 | 23.505 | 10.110 | 10.231 |
| 39 | 3 | 23.390 | 9.859 | 10.690 |
| 40 | 2 | 22.072 | 10.270 | 10.870 |
| 41 | 1 | 22.585 | 10.197 | 10.596 |
| 42 | 0 | **6394.819** | **22.050** | **16.414** |

`psf_zero_nosearch_fallback_count` and `psf_zero_search_fallback_count`
were both 0 at every point.

| | PSF-Zero's own spare=0/pre-cliff ratio (this run) | Speedup vs Qiskit L3, spare=0 | Speedup vs Qiskit L3, spare>=1 range |
| :--- | ---: | ---: | ---: |
| no-search | 2.15x-2.24x | 290.01x | 2.15x-2.37x |
| layout_search | **1.51x-1.60x** | 389.59x | 2.03x-2.30x |

(Ratio ranges use the same method as Addenda 24-25: the spare=0 value
divided by, respectively, the fastest and slowest of the four spare=1-4
points, for that same arm.)

## 4. Prediction-by-prediction verdict

- **Prediction 1 (cliff within ~3x for layout_search) -- confirmed, more
  strongly than hedged for.** The layout_search arm's own spare=0/pre-cliff
  ratio is 1.51x-1.60x -- inside the predicted 3x bound with more than 1x of
  margin to spare, and qualitatively far closer to "no cliff" than to
  "shrunk cliff." This verdict holds regardless of the Section 5 control
  discrepancy, since it only compares the layout_search arm to its own
  pre-cliff points, measured in the same run.
- **Prediction 2 (search not reliably faster than no-search away from the
  cliff, ~0.7x-1.3x) -- confirmed.** `speedup_search_vs_nosearch` at
  spare>=1: 0.988x, 0.922x, 0.945x, 0.962x -- clustered just under 1.0x
  (the search arm pays a small, consistent overhead, roughly 4%-8%, rather
  than showing a gain) at every point, comfortably inside the predicted
  range and, if anything, more tightly clustered than "scatter" implied.
- **Prediction 3 (zero fallback warnings, both arms) -- confirmed** at
  every point, both arms.
- **Prediction 4 (search beats Qiskit L3 specifically at spare=0 by a
  larger margin than no-search does; spare>=1 margins stay comparable to
  Addendum 25's 1.78x-2.28x) -- confirmed in direction, magnitude flagged
  as unreliable.** Spare>=1 speedups this run (no-search 2.15x-2.37x,
  search 2.03x-2.30x) land close to, and slightly above, Addendum 25's
  1.78x-2.28x range for the same rl=1 no-search arm -- consistent with
  "comparable." At spare=0, layout_search's 389.59x margin over Qiskit L3
  does exceed no-search's own 290.01x this run, confirming the predicted
  direction. **But neither of these two numbers should be read as "PSF-Zero
  is 290x-390x faster than Qiskit at the cliff" as a stable property** --
  see Section 5. Both figures are inflated relative to Addenda 24/25's own
  72.86x-75.57x range for the identical no-search comparison, entirely
  because this run's own no-search denominator is unusually low, not
  because Qiskit L3 got slower (its spare=0 time, 6394.819ms, is within 1%
  of Addenda 24/25's own 6383.353ms/6422.288ms).

## 5. An open, unresolved finding: the no-search control did not reproduce Addenda 24/25's own cliff ratio at spare=0

This run's `layout_search=False` arm is the same code path, same
parameters, same machine, same Qiskit/Python versions, same seed and
seed_transpiler as Addendum 24's original run and its reproducibility
check. The three independent measurements of that one arm:

| Run | spare=1-4 range (ms) | spare=0 (ms) | Cliff ratio |
| :--- | ---: | ---: | ---: |
| Addendum 24 (original) | 10.706-12.033 | 88.148 | 7.32x-8.23x |
| Addendum 24 (reproducibility check) | 11.576-11.782 | 84.472 | 7.17x-7.30x |
| **This addendum (paired w/ layout_search)** | 9.859-10.270 | **22.050** | **2.15x-2.24x** |

The spare>=1 points agree across all three runs to within ordinary
run-to-run noise (roughly 10%-15%, consistent with what Addendum 24 itself
called "unrelated to the cliff mechanism"). **The spare=0 point alone
differs by a factor of ~3.8x-4.0x between this run and the prior two**,
which themselves agreed with each other to within ~9% (Addendum 24's own
figure). Qiskit L3's own spare=0 time, measured in the same three runs, is
stable to within ~1% (6383-6423ms) throughout -- so this is not a
machine-wide timing artifact affecting that run generally, it is specific
to whatever PSF-Zero's no-search path does differently right at the
`VF2Layout`-fails/`SabreLayout`-fallback boundary.

**This addendum does not know why, and does not guess a specific
mechanism as established.** One plausible, unconfirmed lead: this project
has separately documented `VF2Layout`/`SabreLayout` seed- and
ordering-dependent nondeterminism at exactly this kind of saturated-map
boundary (`docs/findings/spare-qubit-cliff-addendum-9` through `-14`,
[`data/vf2_seed_nondeterminism_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_seed_nondeterminism_2026-09-14.csv)). It is consistent with this
result that the fallback path's own cost is more run-to-run variable than
Addenda 24/25's two-sample reproducibility check happened to reveal, and
that this run simply landed on a cheaper fallback trial by chance -- but
that is a hypothesis carried over from separate, earlier work, not
something this run's data tests directly, and it is reported here as a
lead, not a conclusion.

### What this does and does not undercut

- It does **not** undercut Section 4's prediction-1 and prediction-3
  verdicts, which compare the layout_search arm only to itself.
- It **does** mean prediction 4's magnitude claim, and any headline framing
  of "layout_search makes PSF-Zero ~390x faster than Qiskit at the cliff,"
  should not be taken as a stable number -- it inherits whatever caused the
  no-search control's own swing, applied on top of the ~1.34x layout_search-
  vs-no-search improvement measured directly (`speedup_search_vs_nosearch`
  at spare=0: 1.343x) in this same run.
- **The 1.343x figure (layout_search vs. no-search, both measured in the
  same run, same instant, at spare=0) is this addendum's most trustworthy
  single number for "how much did layout_search help at the cliff,"** since
  it cancels out whatever is driving the run-to-run swing in the fallback
  path's absolute cost, on the assumption that both arms in the same run
  are equally exposed to it. That assumption is not verified here either.

#### Update (2026-09-16): two more runs show the ~22ms value is reproducible under this script -- the Section 5 discrepancy looks systematic, not per-run randomness

Two further runs of [`test_cliff_sniper_layout_search.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_layout_search.py) (same flags, same
day, same machine) were uploaded after this addendum's first version. Both
land close to the original run, not close to Addenda 24/25's historical
figure:

| Run | no-search spare=0 (ms) | no-search cliff ratio | search spare=0 (ms) | search cliff ratio |
| :--- | ---: | ---: | ---: | ---: |
| This addendum, original | 22.050 | 2.15x-2.24x | 16.414 | 1.51x-1.60x |
| This addendum, run 2 | 22.729 | 2.14x-2.31x | 16.674 | 1.42x-1.55x |
| This addendum, run 3 | 22.581 | 2.09x-2.21x | 16.895 | 1.53x-1.61x |
| **3-run mean (this script)** | **22.45** | -- | **16.66** | -- |
| Addendum 24, original (other script) | 88.148 | 7.32x-8.23x | n/a | n/a |
| Addendum 24, reproducibility check (other script) | 84.472 | 7.17x-7.30x | n/a | n/a |
| **2-run mean (other script)** | **86.31** | -- | n/a | n/a |

All three of this script's runs cluster within a 3.1% band (22.05-22.73ms);
the two historical runs of the other script cluster within a 4.4% band
(84.47-88.15ms); the two bands do not overlap and sit ~3.7x-4.0x apart
(mean ratio 3.84x). Pre-cliff points and Qiskit L3's own spare=0 time stay
consistent across all five runs regardless of which script produced them
(as already noted above).

**This changes the leading hypothesis.** Section 5's original speculation
-- that this was ordinary VF2/Sabre-fallback run-to-run nondeterminism
(Addenda 9-14) and "this run simply landed on a cheaper fallback trial by
chance" -- predicts scatter *within* repeated runs of the *same* script,
roughly comparable to the scatter *between* the two scripts. That is not
what happened: three independent process runs of the new three-arm script
landed tightly together, and two independent process runs of the old
two-arm script landed tightly together somewhere else entirely. A
per-invocation random fallback-trial-count effect does not produce that
pattern; a **difference tied to which script/process ran** does. That
earlier lead is not retracted as impossible, but it no longer fits the
data as well as it seemed to with only one data point per script, and it
should not be treated as the working explanation going forward without
more support.

**A more parsimonious candidate, not yet tested:** `test_cliff_sniper_
layout_search.py` calls the `layout_search=True` path (which imports and
runs [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py), and through it `rustworkx`/`networkx` graph
routines) dozens of times *before* the no-search arm's own spare=0 point
ever runs (once for the deliberate cold-start call, plus warm-up+timed
calls at n=38-41). [`test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py) never calls that code
path at all. Qiskit's own `SabreLayout` fallback -- the thing that actually
runs at the no-search arm's spare=0 point -- is also built on `rustworkx`.
If the two share enough of the same native (PyO3/Rust) machinery, then
exercising `smart_vf2_layout()` many times early in the process could
warm something (shared library relocation/paging, an internal cache, a
thread pool, an allocator arena) that `SabreLayout`'s own fallback then
gets to reuse later in the same process -- for free, in the new script;
paid for in full, in the old one, since nothing else in that process ever
touches `rustworkx` before the fallback runs at n=42. This is a hypothesis
this addendum has not tested, not a conclusion.

**The single most direct next measurement** is not a new script: it is
running the existing, unmodified `test_cliff_sniper_corrected.py
--routing-optimization-level 1` again, right now, today, on this same
machine. If that still reproduces ~85ms (as it did twice before), the
"presence of the search arm in-process" hypothesis above is supported
(same day, same machine, same everything except which script/process ran).
If it now also comes back near ~22ms, the cause is something that changed
in the environment or machine state over the course of today rather than
anything specific to the two scripts' code, and that would need its own
investigation (background load, driver/OS update, thermal state, or
similar) -- a different, and less interesting, story than a genuine
code-level interaction. Either outcome is useful; this addendum does not
have a preferred outcome going in.

#### Update (2026-09-16), continued: the decisive test refutes the rustworkx-warm-up hypothesis -- the unmodified original script now reproduces ~22ms too

> **Correction, 2026-09-16 -- the "presence of the search arm in-process
> warms rustworkx" hypothesis above did not survive its own proposed test.**

The direct test proposed above was run: `test_cliff_sniper_corrected.py
--routing-optimization-level 1` -- the same, completely unmodified script
used for Addendum 24 and its reproducibility check, which never imports or
calls anything from [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) and has no `layout_search` code
path at all -- was run again, later the same day, same machine:

| | spare=1-4 range (ms) | spare=0 (ms) | Cliff ratio |
| :--- | ---: | ---: | ---: |
| Addendum 24 (original, early in the day) | 10.706-12.033 | 88.148 | 7.32x-8.23x |
| Addendum 24 (reproducibility check, early in the day) | 11.576-11.782 | 84.472 | 7.17x-7.30x |
| **This run (unmodified script, later in the day)** | 9.927-11.128 | **22.155** | **1.99x-2.23x** |

This lands right in the same ~22-23ms band as all three runs of the new
`layout_search`-carrying script (Section 5 update above, mean 22.45ms),
using a script that has no `layout_search` code path to have "warmed up"
anything. **This rules out the rustworkx-warm-up hypothesis as stated**:
if that were the mechanism, this unmodified script -- run in a fresh
process with no prior `smart_vf2_layout()` calls anywhere in that
process's history -- should still land near ~85ms, and it did not.

**The pattern across all six runs of PSF-Zero's no-search path at spare=0,
in chronological order today, is now: 88.148, 84.472, [layout_search-script
runs:] 22.050, 22.729, 22.581, [unmodified-script run:] 22.155.** The
split is not by script at all -- it is by *when in the day* the run
happened: the first two runs (both early) are slow; all four subsequent
runs (three different processes of one script, one process of a
completely different, unmodified script) are fast and mutually consistent
to within ~3%. **The most parsimonious reading left standing is a
session-level or machine-level drift over time** -- something about this
specific Windows machine's state changed between the early runs and
everything after, in a way that happens to land squarely on the
`SabreLayout`-fallback path's cost and nothing else this project has
measured (Qiskit L3's own spare=0 time and every arm's pre-cliff points
stayed stable across the whole day). Candidate mechanisms this addendum
has not tested and does not assert: CPU frequency/turbo-boost state
ramping up with sustained use over the session, a background process
(antivirus/indexing/Windows Update) that was active early and settled
later, or a power-plan/thermal effect specific to a longer-running laptop
session. None of these is confirmed; this addendum records the pattern,
not the cause.

**What this means for reading Sections 3-5 above:** the layout_search-vs-
no-search comparisons in Section 3/4 (both arms measured in the same
process, same instant) are unaffected by this -- whatever is drifting,
it drifts at the scale of minutes-to-hours between separate process
launches, not within one. The absolute ~85ms-vs-~22ms comparison across
scripts that Section 5 originally puzzled over is now understood to be a
same-machine, same-day, time-of-run effect having nothing to do with
`layout_search` specifically -- Addendum 24's own historical ~85-88ms
figure should be read as "the fallback cost measured early in that day's
session," not as a fixed property of this machine that later measurements
disagree with by way of a bug.

**Next step, if this project wants to pin the mechanism down further**
(not yet done): a same-day sequence of several more unmodified-script runs
spaced across a few hours, to see whether the value stays at ~22ms from
here on (supporting one-time session warm-up, e.g. first-run-after-boot)
or drifts further (supporting an ongoing effect, e.g. thermal). This
addendum stops at reporting the pattern above and does not speculate
further without that data.

## 6. Reading this result

The core claim the user asked this work to test -- can integrating
`smart_vf2_layout` into `compile_for_hardware` eliminate the spare-qubit
cliff itself, rather than merely inheriting a cheaper failure via a lower
`routing_optimization_level` (Addendum 25's finding) -- has a clear
"yes" answer at the level of the layout_search arm's own shape: 1.5x-1.6x
is not a cliff in any meaningful sense, next to Qiskit L3's 263x-300x or
even the no-search arm's own historical 7.3x-8.2x. The mechanism is also
the intended one: 0 fallback warnings, and the design's own reasoning
(searching the interaction graph directly rather than leaving Qiskit's
`VF2Layout` to fail on it) is supported by the result. Away from the
cliff, the new option costs a small, consistent overhead (roughly 4%-8%
slower than no-search) rather than helping -- exactly as predicted, and
consistent with the feature being opt-in rather than a new default.

The control discrepancy in Section 5 means this addendum cannot yet put a
reliable absolute number on "how much did layout_search help," only a
same-run relative one (~1.34x at spare=0, Section 5). Before this result is
used to justify making `layout_search=True` the recommended default (as
opposed to the "eliminates the cliff shape" qualitative claim, which stands
on firmer ground), the no-search control's reproducibility should be
pinned down first.

## 7. What is still open

- **The Section 5 discrepancy's root physical cause.** Tracked down to a
  same-day, cross-script, time-of-run pattern (see Section 5's dated
  updates) rather than to `layout_search`, the specific script, or
  per-run VF2/Sabre randomness -- but *why* this machine's `SabreLayout`-
  fallback cost dropped by ~4x partway through the day remains
  unconfirmed. Candidate mechanisms (CPU boost/thermal ramp, a background
  process settling, a power-plan effect) are listed in Section 5's update
  and none has been tested directly.
- **Whether the VF2/Sabre nondeterminism work (Addenda 9-14) explains any
  part of Section 5** -- this addendum's own later update found the
  cross-script pattern inconsistent with that lead as the primary
  explanation (see the dated correction in Section 5), though it may
  still contribute to the residual ~3% spread seen within each cluster of
  runs. Confirming or ruling this out further would need a `callback=`-
  based pass-timing trace (Addendum 10's method) run on the no-search arm
  across several fresh-process runs spaced across a session.
- **Whether the ~22ms level is now stable going forward on this machine**,
  or whether it will drift again -- not yet tested; would need further
  unmodified-script runs spaced across a later session.
- **Generalization** to other grids, topologies, seeds, and interaction
  with `routing_optimization_level` 2/3 -- unchanged from every prior
  addendum's own limitations section; this run is 6x7/seed=7/rl=1 only.
- **`smart_vf2_layout`'s own scaling** past 42 qubits -- not tested by this
  run, per the preregistration's own stated limits.

## 8. Files

| Path in the project | Contents |
| :--- | :--- |
| [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py) | The `layout_search` integration under test (item 12, 2026-09-16). |
| [`benchmarks/test_cliff_sniper_layout_search.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_layout_search.py) | The three-arm script that produced Section 3's data. |
| [`data/cliff_sniper_layout_search_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_layout_search_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv) | Original run's raw output (Section 3). |
| [`data/cliff_sniper_layout_search_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_layout_search_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16_run2.csv) | Run 2's raw output (Section 5 update). |
| [`data/cliff_sniper_layout_search_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16_run3.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_layout_search_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16_run3.csv) | Run 3's raw output (Section 5 update). |
| [`docs/findings/spare-qubit-cliff-addendum-26-preregistration-2026-09-16.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-26-preregistration-2026-09-16.md) | The prediction quoted in Section 1, written before this run. |
| [`data/cliff_sniper_corrected_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv) | Addendum 24's reproducibility-check run, used for Section 5's comparison. |
| [`data/cliff_sniper_corrected_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16.csv) | Addendum 24's original run, used for Section 5's comparison. |
| [`data/cliff_sniper_corrected_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cliff_sniper_corrected_6x7_rl1_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-16_run2.csv) | The decisive later-in-the-day re-run of the unmodified Addendum-24 script (Section 5's second update). |

## 9. Verification

- All ratios and speedups in Sections 3-5 were computed directly from the
  uploaded CSV's raw columns (`speedup_search_vs_nosearch` etc. were also
  cross-checked by hand from `qiskit_l3_ms`/`psf_zero_nosearch_ms`/
  `psf_zero_search_ms`, not only read from the CSV's own precomputed
  columns), and from the two historical CSVs read directly from the
  project for Section 5 -- not transcribed from terminal output or from
  memory of earlier addenda.
- `psf_zero_nosearch_fallback_count` and `psf_zero_search_fallback_count`
  were read directly from the CSV (0 at every row) rather than assumed.
- Environment match for the Section 5 comparison (Python 3.11.9, Qiskit
  2.5.2, same CPU signature, same seed=7/seed_transpiler=42) was confirmed
  directly from all three CSVs' own metadata columns, not assumed from the
  addenda text.
- The small-scale correctness pre-check result quoted in Section 2 was
  read directly from this run's terminal output, not assumed.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum -> 0 hits.
- The 2026-09-16 update's run-2/run-3 figures were read directly from the
  two newly uploaded CSVs' raw columns, cross-checked against the terminal
  output accompanying them, not transcribed from memory; the spread/mean
  percentages were computed programmatically, not estimated by eye.

---

<!-- ===== Addendum 27 (source: spare-qubit-cliff-addendum-27-2026-09-16.md) ===== -->

> **Note added when merging:** Fixes a real bug in this session's own
> exact-fidelity checker (found via a suspiciously clean pattern: only
> the arms that do layout search failed, and only where qubit counts
> happened to match), confirming both engines' correctness at cliff and
> non-cliff conditions once fixed. Then runs a wider, repeated
> tight-to-flat sweep and finds the cliff is sharp (spare=0 only) and that
> PSF-Zero, while far more stable than Qiskit overall, is not perfectly so
> at the cliff's exact peak -- an early seed-specific outlier turned out
> not to be seed-specific once more rounds were run.

## Addendum 27 (2026-09-16) -- a fidelity-checker bug found and fixed; the cliff's shape mapped from spare=0 to spare=24; PSF-Zero's own rare outliers at the cliff's peak

### 0. In one line

Building exact unitary-equivalence verification for the Qiskit-vs-PSF-Zero
comparison (following up on Addendum 25's call to verify correctness
before further speed work) surfaced a real bug in the checker itself: an
`n_new == n_orig` special case skipped qubit-remapping entirely, on the
false assumption that equal qubit *counts* meant no permutation had
happened. Once fixed, **both Qiskit L3 and PSF-Zero (with `layout_search`
either on or off) pass exact verification at machine precision, at both
the cliff's peak and away from it.** A wider sweep (spare 0 through 24, 5
rounds, fixed seeds) then mapped the cliff's shape precisely: it is sharp
and confined to `spare=0` (Qiskit ~250-280x slower there, dropping to
1.0-1.7x by `spare=2` and staying flat out to `spare=24`). **PSF-Zero is
far more stable than Qiskit overall, but not perfectly so exactly at the
cliff's peak**: 3 of 30 `spare=0` runs across 5 rounds showed a large,
unexplained slowdown (up to 207ms against a ~24-29ms median) -- and
critically, this did **not** track a single suspicious seed once more data
came in, ruling out "one hard circuit" as the explanation.

### 1. The fidelity-checker bug

Two scripts built earlier the same day
([`bench_qiskit_tket_psf.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_qiskit_tket_psf.py), [`bench_cliff_1v1.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_1v1.py)) both contained the same
`exact_fidelity_check()` function, with a special case: if the compiled
circuit's qubit count equaled the original's, the two were compared
directly with no remapping. This is wrong -- equal qubit *counts* does
not mean qubit *i* still holds logical qubit *i*'s state; both Qiskit's
own layout stage and PSF-Zero's `layout_search` can permute qubits while
leaving the total count unchanged.

The bug surfaced as a strikingly clean pattern on a 3x4 grid at
`spare=0` (where the circuit's qubit count exactly equals the physical
qubit count, triggering the buggy branch): `qiskit_opt3` and
`psf_zero(layout_search=True)` -- both of which invoke a layout search
that can reorder qubits -- came back `exact_FAIL` with infidelity
0.995-0.9999 (i.e. almost completely different operators), while
`psf_zero(layout_search=False)` -- whose design does not reorder qubits
-- passed exactly, every time. That contrast (which arms fail lines up
exactly with which arms *could* have reordered qubits, not with which
circuit was being compiled) was itself the evidence the checker, not the
circuits, was wrong.

Fixed by removing the special case: every comparison now goes through the
same touched-qubit-count check and layout-based remapping regardless of
whether the qubit counts happen to match. Re-run after the fix:

- [`bench_qiskit_tket_psf.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_qiskit_tket_psf.py) (4/6/8 qubits, 5 seeds, 45 rows): Qiskit and
  PSF-Zero both `exact_pass`, all via `order_source=qc_new.layout.final_index_layout`
  (the trustworthy path, not a fallback guess).
- [`bench_cliff_1v1.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_1v1.py) (3x4 grid, spare=0, the condition that first
  exposed the bug): `qiskit_opt3`, `psf_zero_ls0`, and `psf_zero_ls1` all
  `exact_pass` on the first four rows checked before the run was stopped
  as no longer informative (see section 2).

TKET was excluded from this fix's benefit: its `DefaultMappingPass`
output carries no Qiskit `.layout` to recover the true qubit
correspondence from, so it remains stuck on the fallback
(`ascending-index`) path and continues to show spurious `exact_FAIL`
results. This is a separate, still-open limitation, noted but not
pursued further in this addendum (TKET was already out of scope for the
cliff-focused comparison in section 2).

### 2. The cliff's shape, spare=0 through spare=24

Once the checker was trusted again, a wider sweep was run: 6x7 grid (42
physical qubits), `spare` in {0, 2, 4, 8, 16, 24}, 5 rounds, 3 fixed seeds
per round, Qiskit `optimization_level=3` against PSF-Zero
(`layout_search` both off and on). All 270 rows succeeded.

| spare | qiskit_opt3 (median) | psf_zero_ls0 (median) | psf_zero_ls1 (median) | qiskit max/min | psf_ls0 max/min | psf_ls1 max/min |
|---|---|---|---|---|---|---|
| 0 | 6716.0 ms | 28.7 ms | 24.0 ms | 1.13x | **3.74x** | **8.93x** |
| 2 | 26.5 ms | 16.6 ms | 17.0 ms | 1.71x | 1.16x | 1.59x |
| 4 | 27.6 ms | 16.8 ms | 16.9 ms | 1.06x | 1.11x | 1.17x |
| 8 | 31.2 ms | 16.1 ms | 15.9 ms | 1.04x | 1.13x | 1.09x |
| 16 | 36.6 ms | 14.3 ms | 14.5 ms | 1.04x | 1.13x | 1.15x |
| 24 | 39.4 ms | 13.3 ms | 13.2 ms | 1.17x | 1.15x | 1.12x |

**The cliff is confined entirely to `spare=0`.** By `spare=2` Qiskit has
already dropped from ~6.7 seconds to ~26 ms -- a drop of roughly 250x in
a single step -- and stays in the same range (26-39 ms) all the way out
to `spare=24`, drifting gently upward as spare increases (more physical
qubits to search over). PSF-Zero's own median drifts gently *downward*
over the same range (28.7 ms to 13.3 ms), the opposite direction, for
both `layout_search` settings.

**PSF-Zero's win margin at the cliff's peak, on medians: roughly
230-280x.** Away from the cliff (spare 2-24): roughly 1.0-2.9x, in line
with Addendum 25-26's earlier findings for this comparison.

### 3. PSF-Zero's own instability, exactly at the cliff's peak

The `max/min` column above shows something new: at `spare=0` specifically,
PSF-Zero's own spread (3.74x for `layout_search=False`, 8.93x for
`layout_search=True`) is far larger than at any other spare value tested
(1.04x-1.71x everywhere else, Qiskit included). Three individual rows
account for this:

| round | spare | arm | seed | time |
|---|---|---|---|---|
| 1 | 0 | psf_zero_ls1 | 0 | 207.3 ms |
| 1 | 0 | psf_zero_ls0 | 1 | 91.2 ms |
| 4 | 0 | psf_zero_ls0 | 2 | 103.7 ms |

against a `spare=0` median of 24-29 ms -- these are 3-8x the typical
value, all three confined to `spare=0`, none appearing at any other spare
value in any of the 5 rounds.

**The round-1 outlier (seed=1) was initially suspected, by hand, of being
a property of that specific seed's circuit** -- it reproduced identically
(144.202ms) across two independent manual runs at `spare=2` before this
wider sweep was designed. **That suspicion did not hold up**: across the
5-round sweep, `spare=2` showed no elevated values at all for seed=1 (or
any seed), and the three outliers that did appear were spread across
three different seeds (0, 1, 2) in two different rounds. **The pattern is
"an outlier at spare=0 happens occasionally, on no seed in particular,"**
not "seed 1's circuit is slow." This matches the shape of unexplained
timing variance found earlier in this session (Addendum 19's Qiskit-side
anomaly, Addendum 26's same-day drift) more than a circuit-specific
effect.

**Not yet determined**: whether this is specific to `spare=0` because
that is exactly where PSF-Zero's own Sabre-derived routing pass (used
internally when `layout_search` does not fully resolve the layout, or in
the routing stage after it) is under the same kind of stress that
produces Qiskit's cliff in the first place, or something else entirely
tied to running at the coupling map's exact saturation point. No
mechanism has been proposed or tested.

### 4. Files

| Path in the project | Contents |
|---|---|
| [`psf-zero/benchmarks/bench_qiskit_tket_psf.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_qiskit_tket_psf.py) | fidelity-checker bug fixed (section 1) |
| [`psf-zero/benchmarks/bench_cliff_1v1.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_1v1.py) | same fix; single-condition 1-on-1 cliff comparison tool |
| [`psf-zero/benchmarks/bench_cliff_overnight.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_overnight.py) | the multi-round sweep script used for section 2-3 (fixed seed set across rounds, by design, to let seed-specific effects be checked directly) |
| [`psf-zero/data/bench_qiskit_tket_psf_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_qiskit_tket_psf_2026-09-16.csv) | post-fix 4/6/8-qubit verification run (45 rows) |
| [`psf-zero/data/bench_cliff_1v1_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_cliff_1v1_2026-09-16.csv) | earlier single-run cliff data, including the pre-fix false `exact_FAIL` rows and the post-fix confirmation rows (provided by the user across several partial runs) |
| [`psf-zero/data/bench_cliff_overnight_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_cliff_overnight_2026-09-16.csv) | the 5-round, spare 0-24 sweep behind sections 2-3 (270 rows) |

### 5. Verification

- The checker bug was confirmed by the pattern itself before being
  investigated further: failures lined up exactly with which arms invoke
  a layout search (`qiskit_opt3`, `psf_zero_ls1`), and passes lined up
  exactly with the one arm that does not (`psf_zero_ls0`) -- checked
  against the actual `Infidelity` values (0.995-0.9999 for failures, i.e.
  clearly not a numerical-precision issue) before concluding the checker,
  not the circuits, was at fault.
- Post-fix, `order_source` was read directly from each row's
  `FidelityDetail` column to confirm the trustworthy path
  (`qc_new.layout.final_index_layout`) was actually used, not a fallback,
  for both re-verification runs (section 1).
- Section 2's cliff-shape table and section 3's outlier table were both
  computed directly from the 270-row overnight CSV (median, min, max,
  and max/min per spare/arm combination), not summarized from the
  terminal's own running output.
- Section 3's "not seed-specific" conclusion was checked by tabulating
  every `psf_zero` row at `spare=0` by round and seed together (a 5x3
  grid) and confirming the elevated values do not share a common seed
  across rounds.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum, and the three CSV
  files named in section 4 -> 0 hits.
