# Qiskit's `optimization_level` 2/3 falls off a cliff when the circuit nearly fills the coupling map

**Status:** mechanism identified and confirmed on real hardware. A saturated
coupling map causes Qiskit's `VF2Layout` pass to fail; the preset pipeline
then falls back to `SabreLayout`, and that fallback (plus, at higher
optimization levels, its downstream routing/optimization cost) is what
actually burns the seconds. This is a summary; the full investigation --
exact wording, exact tables, every pre-registered prediction as originally
written, and the complete history of what was tried and revised along the
way -- lives in
[`spare-qubit-cliff-combined.md`](spare-qubit-cliff-combined.md)
(same folder as this file). Reading all of that means reading the same
explanation of "VF2Layout fails -> falls back to SabreLayout" five or six
times, the same benchmark-arm definitions three times, and the same "a win
is impossible when the search failed" caveat three times -- this file states
each finding once, with a pointer to where it came from.

**An earlier version of this file attributed the cliff to
`rustworkx.vf2_mapping()`'s VF2++ node ordering and reported this upstream.
That attribution was wrong** (Qiskit's own `VF2Layout` does not call that
function -- see section 1) **and has been superseded by the mechanism
described below**, confirmed by reading Qiskit's source and reproduced on
real hardware.

---

## 1. Where this started

The original report to Qiskit's issue tracker attributed the spare-qubit
cliff to `rustworkx.vf2_mapping()`'s VF2++ node ordering. This was rejected:
Qiskit does not call that function. **[Top addendum, 2026-09-13]** traced
the rejection to a specific commit -- 6421d77 / PR #14860, merged by Jake
Lishman on 2025-09-19, a year before the report -- which had removed the
last call to `rustworkx.vf2_mapping()` from `VF2Layout` and moved everything
into a Qiskit-native Rust implementation. The rejection was correct, and
correct about code that predated the report by a year.

That addendum also read the *current* source and found something the
original report had gotten right in spirit if not in fact: **Qiskit's own
VF2 implementation still uses the VF2++ node ordering, unconditionally**
(`with_vf2pp_ordering()` in `crates/transpiler/src/passes/vf2_layout.rs`).
So the ordering-dependence hypothesis was not wrong -- only the claim about
which function embodied it.

## 2. Confirming ordering-dependence inside Qiskit itself

**[Addendum 4]** tested this directly, since a separate implementation
proving something about `rustworkx` says nothing about Qiskit's compiled
Rust. Using `VF2Layout`'s own `shuffle_seed` parameter (which the earlier
deleted code had exposed as `id_order`), 30 different orderings were tried
against the same saturated coupling graph. Roughly a third found a solution
Qiskit's default reported as impossible -- confirming the ordering-dependence
mechanism is real *inside Qiskit's own code*, not merely a property of the
separate `rustworkx` package.

## 3. Two batches of targeted follow-ups, real hardware

**[Addenda 5, 6, 7]** ran six pre-registered predictions on real hardware,
then four more targeted follow-ups, then a third batch. What they
established, taken together:

- **The 24-node boundary and ordering sensitivity reproduce on real
  hardware**, not just in the sandbox.
- **Heavy-hex topology had never actually been tested** -- an earlier claim
  that it "avoids the cliff" turned out to rest on an untested assumption.
  Testing it directly found the cliff *does* appear on heavy-hex under the
  right conditions.
- **DFS-vs-BFS node visitation has (at least) two separable causes** for why
  depth-first orderings tend to fail more often than breadth-first ones.
- **The "burns the entire call budget" behavior is Qiskit's, not
  rustworkx's** -- when a search fails, Qiskit's implementation consumes its
  whole configured budget before giving up, which is a property of Qiskit's
  wrapper logic, not something inherited from the underlying library.
- **[Addendum 7]** proposed the **two-factor hypothesis** that organizes
  everything found through addendum 12: the cliff appears when a topology is
  (a) not vertex-transitive **and** (b) has a perfect matching reachable all
  the way down to zero spare qubits. Grids and lines satisfy both and show
  the cliff; rings, tori, and complete graphs are vertex-transitive and do
  not; heavy-hex is not vertex-transitive but (in most configurations)
  cannot reach zero spare, so it mostly doesn't show the cliff either --
  except where it can, which is exactly where it does (per addendum 6).

## 4. The mechanism reframed: it isn't search cost, it's a fallback

**[Addendum 8]** got as far as Python could reach: reading
`qiskit.transpiler.passes.layout.vf2_layout`'s installed source confirmed
`VF2Layout.run(dag)` calls the compiled Rust function
`vf2_layout_pass_average` exactly once and gets back a finished result, with
no visibility into the search itself. Two follow-up experiments were queued
un-run.

**[Addendum 9] is the turning point of the whole series.** While reviewing
those follow-up results (one sparse topology, `diluted_p0.75`, unexpectedly
showed *no* cliff), reading
`qiskit.transpiler.preset_passmanagers.builtin_plugins.DefaultLayoutPassManager`
revealed the actual two-stage design of Qiskit's preset layout stage:

```
choose_layout_0 = VF2Layout(seed=-1, call_limit=(5_000_000, 10_000), ...)   # tried first
choose_layout_1 = SabreLayout(seed=pass_manager_config.seed_transpiler, ...)
                  # only runs if VF2Layout failed
```

**The cliff is not "VF2 search taking a long time" in isolation -- it is
VF2Layout failing, burning its full budget, and then falling back to
SabreLayout**, a completely different (and, for this circuit family,
comparatively slow) algorithm. `diluted_p0.75` showed no cliff not because
it is sparse, but because VF2Layout happens to succeed on that specific
graph. This reframing does not contradict the earlier "burns the budget"
finding (addendum 6) -- it explains *why* that budget-burning matters enough
to be visible as a multi-second cliff: it triggers an expensive fallback.

A second, easy-to-miss fact from the same addendum: `seed_transpiler` **does
not control `VF2Layout`'s shuffle** (it's hardcoded `seed=-1`). Every
topology experiment up to this point had implicitly assumed
`seed_transpiler=0` made results reproducible; it does not, for the layout
search specifically.

Also found in the same pass: node-visit order matters in a specific,
previously-unnoticed way -- **starting a depth-first search from the center
of a grid fails 100% of the time**, regardless of which of 24 neighbor-visit
orderings is used, and **a grid's row/column parity governs a corner-started
search's success rate** (both-even is worst, both-odd is best). The parity
finding was provisional here; **[Addendum 12]** later confirmed it with a
full 2x2 sweep of parity combinations, and in the same addendum corrected an
early claim that the result was "symmetric under transpose" -- that symmetry
turned out to be a mathematical necessity of testing all 24 permutations,
not a discovered property of the grids themselves.

**[Addendum 10]** confirmed addendum 9's fallback-to-Sabre mechanism **on
real hardware**: all 24 predicted rows matched exactly, including
`diluted_p0.75` succeeding at VF2Layout on both optimization levels tried.
It also found a **second cost layer specific to `optimization_level=3`**:
after Sabre's fallback layout is chosen, the routing and optimization passes
that run afterward can cost as much as, or more than, VF2Layout's original
failure -- in one case (`brick` at L3) nearly 3x as much. So at L3, the
cliff has (at least) two additive stages: VF2Layout failing, and Sabre's
imperfect layout being expensive to route around afterward.

**[Addendum 11]** ran the code for two remaining follow-ups (seed=-1's
jitter, grid parity); **[Addendum 12]** reports both as fully confirmed:

- Across 90 repeated trials on 6 topologies, success rate pinned cleanly at
  0% or 100% for every candidate -- no knife-edge cases were found in this
  set, so `seed=-1`'s non-determinism did not, in this test, sway any of the
  addendum-5-through-10 verdicts.
- The parity pattern held across a full 2x2 sweep (8 grids). As noted above,
  the "transpose symmetry" part of this was a design artifact, not a new
  empirical finding, and was corrected in the same addendum it was reported
  in.

## 5. Trying to do something about it: a layout-search prototype

**[Addendum 13]** built [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py), a two-stage prototype VF2
search: a cheap stage 1 (six BFS-family node orderings, Qiskit's default
`id_order=True` mode) followed, if that fails and time remains, by a more
expensive stage 2 (`id_order=False`, Qiskit's built-in heuristic, several
starting orderings). A hard-to-notice bug was found and fixed during
development: `rustworkx.vf2_mapping()`'s return direction had been misread,
which silently returned *wrong* physical qubits whenever the qubit count was
exact (spare=0) rather than raising an error -- caught only because the
smoke test validated the layout's correctness directly rather than just
checking "found or not."

**Important, and true through the rest of the series**: every finding about
this prototype's orderings comes from the **public** `rustworkx.vf2_mapping()`
API. Whether the same ordering effects hold inside Qiskit's actual compiled
implementation (`qiskit._accelerate.vf2_layout`, a separate codebase per
addendum 9) was never directly tested. The prototype is deliberately built
on the public API rather than trying to imitate or patch Qiskit's internals.

A second finding from building it: the "BFS is robust" result from addenda
4/9/12 turned out to be **specific to pure grid graphs**. Applied to
addendum-7's sparse topologies (`brick`, `diluted_p0.25/0.5/0.75`), BFS-only
search (stage 1) found nothing on any of them even at a 10-million-call
budget; only `diluted_p0.75` was eventually found, and only by stage 2's
heuristic ordering -- which is what motivated adding stage 2 to the design
in the first place.

## 6. Does the prototype actually help? Three rounds of measurement

**[Addendum 14]** designed the real test: race the prototype against
Qiskit's default pipeline **and** against PSF-Zero's own compile flow
(`compile_for_hardware`), on total time, counting the search time even when
it fails (since a failed search followed by the default pipeline anyway is
a real cost, not a null result). Four pre-registered predictions (P1-P4)
were written before any real-hardware run. A sandbox dry run (not used for
verdicts) already suggested P1 would only hold for `grid`/`line` --
`diluted_p0.75`, though found, lost, because the default pipeline was
already fast on that instance and the prototype's stage 2 wasn't.

**[Addendum 15], the first real-hardware run (L2), delivered a clean, almost
suspiciously exact result**: `grid` won 30.8x, `line` won 35.4x; the three
sparse losers lost by a margin matching their search time to within 0.3-2.7%
error -- confirming that a failed search costs *exactly* its own time and
nothing more. `diluted_p0.75` did lose as the sandbox had suggested. Circuit
quality (gate count, depth) was bit-for-bit identical across every arm and
condition -- the cliff is purely a time cost. But **the comparison this
series actually wanted -- PSF-Zero end to end -- could not be made**:
`compile_for_hardware`'s signature has no `initial_layout` parameter and no
`**kwargs`, so the search result had nowhere to go. Worse, the benchmark's
own PSF-smart rows were silently invalid (a failed search fell through to
plain PSF-Zero, which was then counted as a "success" despite having wasted
the search time for nothing) until this was caught and the arm was excluded
by design from then on. A stage-1-only arm (`_smart1`) was added, since
stage 2 had cost 0.9-2.7 seconds while only ever rescuing one candidate.

**[Addendum 16], the L3 run, is where the ground shifted.** Wins grew
another order of magnitude (`grid` 399.8x, `line` 339.2x). But **the same
measurement, repeated within the same run, was found to vary by up to
roughly 3x** -- two "wins" appeared on the loss side that are logically
impossible (a failed search cannot win), which is what exposed the variance
in the first place. Addendum 15's clean "loss = search time" accounting does
not hold at L3, and it is now unclear whether it ever really held at L2 --
the same `brick` condition varied 3.2x between the addendum-15 and
addendum-16 runs. A second, unrelated bug was found in the same round: the
benchmark's `--psf-rl` flag had not been following `--level`, so the L3 run
had accidentally measured PSF-Zero at L2 settings against Qiskit at L3 --
not a valid comparison, and now fixed with an explicit mismatch warning. Two
candidate causes for the variance -- `VF2Layout`'s `seed=-1` shuffle, or
environment drift -- were proposed with pre-registered predictions (P9-P11),
but the measurement to distinguish them has not been run.

**[Addendum 17] patched the blocker**: `compile_for_hardware` gained an
`initial_layout` parameter (three lines -- signature, docstring, and
forwarding to the internal `transpile()` call), and the end-to-end PSF-Zero
comparison ran for the first time. The result matched the Qiskit-only
comparison closely on every topology: `grid` won 27.2x through PSF-Zero's
pipeline against 32.1x through bare `transpile()`, `line` 28.1x against
29.0x, and the three losing topologies lost by essentially the same margin
(0.38-0.45x) on both sides. The prototype's benefit survives being routed
through PSF-Zero's actual compilation flow.

**[Addendum 18] first re-checked reproducibility before tuning anything**:
three independent L2 runs on the same tight, hard-to-layout topologies
agreed to within 1.00-1.30x -- unlike the ~3x spread Addendum 16 found at
L3, this specific L2 condition held up. With that reassurance, a six-point
sweep of the prototype's stage-2 search budget (`fallback_call_limit`,
200k/300k/400k/500k/1m/2m) found 300,000 is the smallest value that still
reliably catches the one topology stage 2 exists to catch
(`diluted_p0.75`, found at attempt 7 of 9; 200,000 misses it outright). The
default was lowered from 2,000,000 to 300,000 accordingly, nearly doubling
the three failing topologies' loss margin (0.38-0.44x to 0.74-0.79x) with
no measured cost to the winning topologies.

**[Addendum 19] recorded a separate line of measurement from the same day**:
a coupling-map-free compile-time comparison (`test_cumulative_compile_scale.py`,
no `coupling_map` passed anywhere, so unrelated to the VF2/SabreLayout
mechanism above) at 10,000 and 50,000 iterations. PSF-Zero won 5.90x-7.00x
(cumulative-total basis; 5.73x-7.97x median-based) with correctness
confirmed by a 6-qubit fidelity check before each run. Separately, Qiskit's
cumulative-time curve showed a visible, reproducible slope anomaly at both
sample sizes, absent from either PSF-Zero curve -- cause unconfirmed, and
distinct from Addendum 16's L3 variance since no coupling map is involved
here.

**[Addenda 20-21] chased that anomaly and only partly explained it.**
The 20 slowest Qiskit compiles from the 50k run were rebuilt exactly from
their seeds and checked for proximity to degenerate points (CNOT, SWAP,
iSWAP, identity) -- rejected, indistinguishable from a random baseline.
Sorting the same 20 indices by hand suggested a ~145-iteration gap; a
full-series autocorrelation check confirmed it was strong and real on the
Intel machine that produced the original data (autocorrelation 0.86, rank
1 of 500 lags), but it did not reproduce on an AMD machine across two
independent runs. Revisiting that AMD data's own top lags (rather than
only checking 145) turned up a **different, equally strong period at
~187 iterations** (autocorrelation 0.96-0.99) that had been there all
along. This 187-period survived three separate attempts to explain it
away: it was unchanged across three different `PYTHONHASHSEED` values
(ruling out hash randomization), unchanged when the iteration count was
halved (ruling out an elapsed-time-based cause), and **completely absent**
from a dedicated Qiskit-free control loop (pure Python arithmetic,
`time.sleep`, and a numpy matmul, each timed the same way -- none showed
anything above 0.08 autocorrelation, against Qiskit's 0.96-0.99). By
elimination, this points toward something in Qiskit's own execution
specifically, on a machine-dependent cycle length -- not yet identified,
and reading Qiskit's own source for a matching constant is the natural
next step that has not yet been tried.

**[Addenda 22-23] closed the "what causes it" question for the Intel
machine's 145-period, via two paths tried in parallel.** Reading Qiskit's
own Sabre routing/layout Rust source (`layout.rs`, `route.rs`) directly
ruled it out as the location: no `145`/`187` constant appears in either
file, and more fundamentally, the coupling-map-free experiment that found
the period never executes that code path at all (it returns immediately
on `TargetCouplingError::AllToAll`). Separately, the 145-period was first
confirmed to reproduce a second time on the same Intel machine (closing
addendum 20's open item), then a pre-registered hypothesis --
`gc.disable()` should remove the period if CPython's garbage collector is
the cause -- was tested directly and **confirmed**: four independent runs,
individually and pooled (n=20,000), all show the period collapse from
rank 1/500 to rank ~145/500 and modular-bin spread from 7.5x-10.7x to
~1.1x-1.3x once the GC is disabled. The exact GC trigger (which
generation, which threshold) remains unidentified, and whether this
explains the AMD machine's separate 187-period is untested.

**[Addenda 24-26] turned from explaining the cliff to acting on it,
using a corrected benchmark script** (the previous version had a bug that
silently replaced every failed PSF-Zero call with a fabricated placeholder
number, discovered by reading its own source before trusting its output).
Three findings, in order: plain `compile_for_hardware()` (no search aid)
still crosses the spare-qubit cliff, but far more gently than Qiskit L3
(7.3x-8.2x vs 263x-300x) -- the first real-core measurement of this
specific comparison. Raising `compile_for_hardware`'s own internal
`routing_optimization_level` toward Qiskit L3's level closes most of that
gap by level 2 and nearly all of it by level 3, where PSF-Zero stops being
faster than plain Qiskit L3 at all (0.83x-0.96x) -- suggesting the
original advantage near the cliff was largely a side effect of a cheaper
default routing level, not the synthesis pass itself. Finally, a new
opt-in `layout_search=True` option (built on this project's own
`smart_vf2_layout()` prototype, now callable directly from
`compile_for_hardware()`) collapses PSF-Zero's own cliff to roughly
1.5x-1.6x -- not fully eliminated, but no longer a cliff in any meaningful
sense. That same measurement also surfaced and chased down an unrelated,
unexplained ~4x same-day timing drift in its own control arm, eventually
traced to a session/machine-level effect rather than to the new code, and
reported as an open finding rather than smoothed over.

**[Addendum 27] found and fixed a real bug in the exact-fidelity checker
built alongside these speed comparisons**, then used the corrected checker
to confirm both engines' correctness and map the cliff's exact shape. The
bug: an `n_new == n_orig` special case skipped qubit remapping entirely,
wrongly assuming equal qubit counts meant no permutation had occurred --
caught by a suspiciously clean failure pattern (only layout-search-capable
arms failed) rather than by inspection. Fixed, both Qiskit L3 and
PSF-Zero (layout_search on or off) pass exact unitary verification at
machine precision, at the cliff and away from it. A wider sweep (6x7 grid,
spare 0 through 24, 5 rounds) then mapped the cliff precisely: it is sharp
and confined entirely to spare=0 (Qiskit ~250-280x slower there, dropping
to 1.0-1.7x by spare=2, staying flat out to spare=24). PSF-Zero is far
more stable than Qiskit overall, but not perfectly so exactly at the
cliff's peak: 3 of 30 spare=0 runs across the 5 rounds showed large,
unexplained slowdowns (up to 8x the median) -- and an early suspicion that
one specific seed's circuit was the cause did not hold up once more
rounds were run, since the outliers landed on different seeds in
different rounds.

## 7. Where this stands

**Solid:**
- The cliff's mechanism (VF2Layout fails -> Sabre fallback -> at L3,
  possibly-expensive downstream routing) -- established by reading source,
  confirmed by direct experiment, confirmed again on real hardware.
- Ordering-dependence is real inside Qiskit's own compiled code, not just in
  `rustworkx` -- confirmed directly via `shuffle_seed`.
- Output quality never differs by path taken -- the cliff costs time only.
- The prototype's wins on `grid`/`line` (15x-420x across every round
  measured) are large enough that the ~3x variance found in addendum 16
  cannot explain them away.
- The prototype's benefit survives integration into PSF-Zero's own
  compilation pipeline, not just bare `transpile()` -- confirmed on real
  hardware (Addendum 17).
- At L2 specifically, tight-condition measurements reproduce across three
  independent runs to within 1.00-1.30x (Addendum 18) -- unlike the ~3x
  spread found at L3.
- The Addendum 19 timing anomaly is not caused by circuit-level
  near-degeneracy, hash randomization, elapsed time, the OS scheduler, or
  generic computation/BLAS overhead (Addenda 20-21) -- each was tested
  directly and ruled out in turn.
- On the Intel machine, the ~145-period is caused by CPython's garbage
  collector: `gc.disable()` removes it cleanly across four independent
  runs (Addendum 23). Qiskit's own Sabre Rust source was read directly and
  ruled out as the location (Addendum 22).
- A corrected benchmark (the prior version silently fabricated failed
  PSF-Zero measurements) confirms `compile_for_hardware()` still crosses
  the cliff without help, but ~35x more gently than Qiskit L3 (Addendum
  24); raising its internal routing level toward Qiskit L3's own erodes
  that gap until PSF-Zero is no longer faster at all by level 3 (Addendum
  25); and the new `layout_search=True` option collapses PSF-Zero's own
  cliff to ~1.5x-1.6x (Addendum 26).
- A real bug in this session's own exact-fidelity checker (skipped qubit
  remapping when counts happened to match) was found and fixed; once
  fixed, both Qiskit and PSF-Zero pass exact verification at machine
  precision, at the cliff and away from it (Addendum 27).
- The cliff's shape is now mapped precisely: sharp and confined to
  spare=0 (~250-280x), dropping to 1.0-1.7x by spare=2 and staying flat
  out to spare=24 (Addendum 27).

**Open:**
- What causes the ~3x same-condition variance seen at L3 (seed=-1 shuffle
  vs. environment drift) -- experiment designed, not yet run.
- Whether L2's reproducibility (confirmed for tight conditions in Addendum
  18) extends to L3, other spare values, or other machines -- untested.
- Whether the ordering effects driving the prototype (found via public
  `rustworkx`) hold inside Qiskit's actual compiled VF2 implementation --
  never tested, through all 26 rounds.
- Whether the tuned stage-2 budget (300,000, Addendum 18) can go lower --
  200,000 already misses one topology outright and no finer step was tried
  between the two values.
- Which specific GC behavior produces a ~145-cycle on the Intel machine
  (a generation threshold, an allocation-count trigger) -- disabling the
  GC outright shows it is the cause, but not the exact mechanism
  (Addendum 23). Whether the same explanation applies to the AMD
  machine's separate ~187-period is untested.
- Which of "VF2Layout's search budget" vs. "Addendum 10's level-3-specific
  downstream cost" drives the routing-level progression found in Addendum
  25 -- a `callback=`-based pass-timing trace would settle this directly
  and has not been run. A plain Qiskit `optimization_level=2` baseline is
  also still missing from that comparison.
- What causes PSF-Zero's own rare, large slowdowns exactly at the cliff's
  peak (spare=0) -- 3 of 30 runs across 5 rounds, up to 8x the median,
  spread across different seeds in different rounds rather than tied to
  one circuit. No mechanism proposed or tested (Addendum 27).
- The ~4x same-day timing drift Addendum 26 found in its own no-search
  control -- narrowed to a session/machine-level effect rather than to
  `layout_search` itself, but the root physical cause (CPU boost/thermal
  state, a background process, a power-plan effect) remains unconfirmed.
- Generalization of Addenda 24-26's findings beyond the one grid
  (6x7), one seed, and one machine tested.

## 8. Files, by round

| Addendum | Scripts | Data |
|---|---|---|
| 4 | (source-reading only) | -- |
| 5-7 | `verify_vf2_*` batch (topologies, ordering structure, cross-implementation, steps-to-first-match, call-limit tuple/sweep, source check, seed anomaly, id_order=True, toroidal grid, rustworkx raw sweep, DFS mechanism, heavy-hex, neighbor-order-full, sparse-topology, grid-parity) | matching dated CSVs |
| 8-9 | [`verify_vf2_sparse_topology.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_sparse_topology.py), [`verify_vf2_neighbor_order_full.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_neighbor_order_full.py), [`verify_vf2_pipeline_trace.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_pipeline_trace.py) | [`vf2_sparse_topology_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_sparse_topology_2026-09-14.csv), [`vf2_neighbor_order_full_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_neighbor_order_full_2026-09-14.csv) |
| 10 | (pipeline trace, above) | `vf2_pipeline_trace_2026-09-14.csv` (not found in the repository as of 2026-09-14) (real hardware) |
| 11-12 | [`verify_vf2_seed_nondeterminism.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_seed_nondeterminism.py), [`verify_vf2_grid_parity.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_vf2_grid_parity.py) | [`vf2_seed_nondeterminism_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_seed_nondeterminism_2026-09-14.csv), [`vf2_grid_parity_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/vf2_grid_parity_2026-09-14.csv) (real hardware) |
| 13 | [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py), [`smoke_test_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/smoke_test_smart_layout.py) | -- (sandbox only) |
| 14-16 | [`benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) (evolving: `_smart1` arm added in 15, `--psf-rl`/`Spread_max_over_min`/noise detection added in 16) | [`smart_layout_vs_default_intel_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_intel_2026-09-14.csv) (L2), [`smart_layout_vs_default_L3_intel_2026-09-14.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_L3_intel_2026-09-14.csv) (L3) |
| 17 | [`compile_for_hardware_initial_layout.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_for_hardware_initial_layout.patch) (adds `initial_layout` to `psf_compile.compile_for_hardware`) | [`smart_layout_vs_default_2026-09-15.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_2026-09-15.csv) (the end-to-end PSF-Zero run) |
| 18 | [`benchmark_smart_layout_vs_default.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/benchmark_smart_layout_vs_default.py) (gained `--fallback-call-limit`), [`psf_smart_layout.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_smart_layout.py) (`fallback_call_limit` default 2,000,000 -> 300,000) | [`sweep_200k.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_200k.csv) through [`sweep_2m.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/sweep_2m.csv) (6 files), plus two further L2 reproducibility runs ([`run2`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_2026-09-15_run2_qiskit_only.csv), [`run3`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/smart_layout_vs_default_2026-09-15_run3_qiskit_2m.csv)) |
| 19 | [`test_cumulative_compile_scale.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_scale.py) (coupling-map-free comparison, unrelated to the VF2/SabreLayout mechanism) | [`cumulative_compile_times_10000.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_10000.npz), [`cumulative_compile_times_50000.npz`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_compile_times_50000.npz), [`Figure_1.png`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/Figure_1.png), [`cumulative_compile_results_50000.png`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/cumulative_compile_results_50000.png) |
| 20 | [`diagnose_outlier_circuits.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/diagnose_outlier_circuits.py), [`check_period_145.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145.py) | two AMD-machine reproducibility runs (`.npz`, filenames as saved by the user) |
| 21 | [`check_dummy_loop_period.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_dummy_loop_period.py) | three hash-seed runs and one half-length run (`.npz`, filenames as saved by the user) |
| 22 | [`layout.rs`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/layout.rs), [`route.rs`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/route.rs) (Qiskit's own Sabre source, read and ruled out) | second Intel-machine 145-period confirmation (real hardware) |
| 23 | [`check_period_145_pooled.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/check_period_145_pooled.py) | four `gc.disable()` runs, individually and pooled (n=20,000) |
| 24 | [`test_cliff_sniper_corrected.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_corrected.py) (fixes a predecessor script that silently fabricated failed PSF-Zero measurements) | `cliff_sniper_corrected_6x7_...` sweep, 38-42 qubits (real hardware) |
| 25 | (same script, `--routing-optimization-level` varied) | rl=2 and rl=3 sweeps, plus [pre-registration](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-25-preregistration-2026-09-16.md) |
| 26 | [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/psf_compile.py) (new `layout_search` option), [`test_cliff_sniper_layout_search.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cliff_sniper_layout_search.py) | three-arm sweep (Qiskit L3 / no-search / `layout_search=True`), 3 runs |
| 27 | [`bench_qiskit_tket_psf.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_qiskit_tket_psf.py), [`bench_cliff_1v1.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_1v1.py), [`bench_cliff_overnight.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_overnight.py) (fidelity-checker bug fixed in all three) | [`bench_qiskit_tket_psf_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_qiskit_tket_psf_2026-09-16.csv), [`bench_cliff_1v1_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_cliff_1v1_2026-09-16.csv), [`bench_cliff_overnight_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_cliff_overnight_2026-09-16.csv) (270 rows, spare 0-24, 5 rounds) |

Full text, exact tables, and every pre-registered prediction as originally
written:
[`spare-qubit-cliff-combined.md`](spare-qubit-cliff-combined.md).

---

## See also

- [`spare-qubit-cliff-combined.md`](spare-qubit-cliff-combined.md) --
  all 26 addenda, unedited, in chronological order (same folder). This is
  where the exact wording, exact tables, and every pre-registered prediction
  as originally written can be found.
