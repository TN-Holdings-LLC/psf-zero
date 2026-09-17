# spare-qubit-cliff: Combined Addenda, Part 3 of 3 (Addendum 27 through Addendum 36)

**Continued from [Part 2](spare-qubit-cliff-combined-17.md) (and [Part 1](spare-qubit-cliff-combined.md)).** Same conventions as Part 1: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

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


<!-- ===== Addendum 28 (source: spare-qubit-cliff-addendum-28-2026-09-17.md) ===== -->

> **Note added when merging:** A 400-round null-result hunt for Addendum 27's rare, unexplained PSF-Zero slowdowns at the cliff's peak: 800 calls, zero outliers, under both normal and gc.disable() conditions. The original finding is not confirmed by this larger sample -- but see Addendum 29, which found the anomaly may have simply moved to a different shape this script cannot detect.

## Addendum 28 -- spare=0 outlier hunt: a clean null result (2026-09-17)

**Scored against the pre-registration**:
`spare-qubit-cliff-addendum-28-preregistration-2026-09-16.md`, written
and saved to the project before this measurement was run. Every
prediction below is scored against what that document actually said,
not restated to fit the data after the fact.

## 1. What was run

Two invocations of `spare0_outlier_hunt.py --rows 6 --cols 7 --seeds 10
--repeats 20` -- the exact pre-registered command, once as-is and once
with `--disable-gc` added (the user ran both before this write-up, so
both are scored together rather than in two passes). 6x7 grid (42
qubits, spare=0), 10 distinct circuit seeds (100-109) x 20 repeats x 2
arms (`layout_search=False` / `layout_search=True`) = 400 timed PSF-Zero
`compile_for_hardware()` calls per invocation (800 total),
`routing_optimization_level=1`, `seed_transpiler=42` (pinned). Machine:
`Intel64 Family 6 Model 181 Stepping 0, GenuineIntel` (this project's
faster machine -- see `psf-zero/README.md` for the neutral
machine-naming convention). Python 3.11.9, Qiskit 2.5.2. Sweep wall
time: 14.4s (normal), 10.4s (`--disable-gc`). All 800 calls across both
runs completed with zero errors and zero coupling-map violations (each
call's own `check_routing_validity()` passed). The `--disable-gc` run's
own GC counters confirm collection really was suppressed throughout:
`gc_collections_gen0` shows zero net change from the first to the last
row of that run, while `gc_gen0_count` (allocations since the last
collection) climbs unboundedly across the run (up to 3900, vs. resetting
to single digits every few calls in the normal run) -- the flag did what
it was supposed to do.

## 2. Headline result: no outliers, in either run, by a wide margin

| Run | Arm | n | median | mean | std | max | max/median | outliers (>=3x) |
|---|---|---|---|---|---|---|---|---|
| normal | nosearch | 200 | 23.040 ms | 23.628 ms | 1.894 ms | 31.282 ms | 1.36x | 0 |
| normal | search | 200 | 17.176 ms | 17.322 ms | 0.955 ms | 20.966 ms | 1.22x | 0 |
| `--disable-gc` | nosearch | 200 | 23.282 ms | 23.762 ms | 1.954 ms | 32.144 ms | 1.38x | 0 |
| `--disable-gc` | search | 200 | 17.326 ms | 17.599 ms | 1.546 ms | 25.086 ms | 1.45x | 0 |

**Zero rows in either arm, in either run (0 of 800 calls total) cross
the pre-registered 3.0x-median outlier threshold.** The single slowest
call across all 800 (32.144 ms, nosearch, `--disable-gc` run) is only
1.38x that run's own median -- not even close to 3x, let alone the 8x
Addendum 27 saw in its worst case. Timing is tight and well-behaved in
both runs: coefficient of variation stays under 9% throughout, and
every one of the 10 seeds' per-seed medians and maxima sit inside a
narrow band with no seed standing out (normal-run nosearch per-seed
medians: 22.31-24.31 ms; per-seed maxima: 25.57-31.28 ms).

## 3. Predictions scored

**P1 (outlier rate -- expected >=1 outlier in 200 no-search calls):
FALSIFIED.** Zero outliers occurred in either arm at 200 calls per arm,
not just in the no-search arm the prediction focused on.

**P2 (seed independence -- outliers should not cluster on 1-2 seeds):
Not testable.** With zero outliers to distribute across seeds, this
prediction has no outlier population to check. As a secondary
observation: per-seed *medians* (the ordinary, non-outlier timing) also
show no seed standing out as unusually slow or fast, which is at least
consistent with the "nothing seed-specific is happening here" reading,
but this was not what P2 was designed to test.

**P3 (arm comparison -- outliers in both arms, testing the
shared-process-state hypothesis): Not testable.** Same reason as P2 --
no outliers occurred in either arm to compare.

**P4 (`--disable-gc` reduces but doesn't zero the outlier count):
Vacuously true, uninformatively.** The outlier count was already zero
in the normal run, so it could not go any lower in the `--disable-gc`
run -- it also came back zero, but that is not evidence for or against
a GC-driven mechanism when there was nothing to reduce. Looking at the
weaker signal this comparison can still offer (raw timing, per section
5's option (a)): median times shifted by under 1.1% between the two
runs in both arms (nosearch: +0.242 ms / +1.0%; search: +0.150 ms /
+0.9%) -- noise-level, not a directional effect. More notably, the
*search* arm's standard deviation actually **increased** with GC
disabled (0.955 ms -> 1.546 ms, a 62% increase) and its max rose from
20.966 ms to 25.086 ms, which is the opposite direction a
"GC-causes-slowdown" story would predict. Taken together, this is a
clean negative result for a GC-mediated mechanism at spare=0 -- unlike
Addendum 23's unrelated ~145-iteration-period finding, where
`gc.disable()` produced an unambiguous, repeated effect, here it
produced no effect worth calling a finding. This is exactly the outcome
the preregistration's P4 discussion warned against overreading: Addendum
23's mechanism does not transfer here, and this run confirms rather than
assumes that.

**P5 (an outlier's pass-timing trace should be dominated by one pass):
Not testable.** No outliers occurred. For reference only (not a
substitute for what P5 asked): among ordinary, non-outlier calls, the
dominant pass differs sharply and *systematically* by arm --
`VF2Layout`/`VF2PostLayout` dominate 157/200 no-search calls (this is
exactly the fallback mechanism Addendum 24 identified), while
`BasisTranslator`/`Optimize1qGatesDecomposition` dominate 38/200
search-arm calls (consistent with `layout_search=True` bypassing that
mechanism when it succeeds, as designed). This is expected, unsurprising
behavior, not a new finding -- it is included only to confirm the
`callback` instrumentation (item 13) is capturing real, arm-appropriate
pass data rather than something degenerate.

## 4. The pre-registered falsification condition applies

Section 3 of the preregistration stated, before this run: *"If, at 200+
calls per arm, this run produces ZERO outliers by the pre-registered
rule, the honest conclusion is that Addendum 27's 3 events (30 total
calls) may not represent a stable, reproducible phenomenon at this
sample size, and the 'rare unexplained slowdown' framing should be
walked back to 'not reproduced at higher N' rather than pursued further
with more elaborate instrumentation."**

That is exactly what happened here, so that is exactly the conclusion
being drawn: **Addendum 27's spare=0 outliers (3 of 30 calls, up to 8x
median) did not reproduce at nearly 7x the sample size (400 calls, this
run) on the same machine, same grid, same routing level.** This does
not prove Addendum 27's 3 events were measurement noise or an artifact
of that specific run -- 3 events in 30 calls (10%) versus 0 events in
400 calls is not automatically a contradiction if the true rate is,
say, well under 1% and Addendum 27 simply landed on an unlucky round --
but it does mean the working hypothesis changes from "PSF-Zero has a
rare, real slowdown mechanism at spare=0, mechanism unknown" to "not
reproduced at this sample size; if it is real, it is rarer than 30
calls could reliably show, and chasing its mechanism (GC, a specific
pass, a specific seed) via more instrumentation is not yet justified
by data dense enough to have a pattern to find."

Per the preregistration's own instruction, this is being reported as a
null result, not reframed as "still consistent with a low base rate."

## 5. What is still open, and a decision point

- **Both the normal and `--disable-gc` runs are now in hand, and neither
  shows anything.** The decision point raised in an earlier draft of
  this document (whether to bother running `--disable-gc` given the
  null in the normal run) is now moot -- it was run, and it also came
  back null, with no directional signal even in raw timing. This
  further weakens, rather than supports, a GC-mediated explanation for
  Addendum 27's original 3 outliers.
- **This is one machine, one routing level, one grid size, two runs.**
  Addendum 27's original 3 outliers were also all on this same faster
  Intel machine (per the historical record), so this null result is a
  same-machine comparison, not a different-machine check.
- **A much larger N (e.g. 2000+ calls) would be the natural next step
  if this thread is pursued further** -- but per the preregistration's
  own discipline, that decision should follow from whether the null
  result here is treated as a stopping point or not, not be launched
  reflexively. Given that BOTH the plain sweep and the GC-disabled
  sweep came back clean at 400 calls each (800 total, vs. Addendum 27's
  original 30), the more economical reading is that this specific
  thread (GC and simple repetition as candidate mechanisms) has been
  reasonably exhausted without support, and any further chase would need
  a different angle (e.g., much higher N specifically watching for a
  <<1% event, or instrumenting something other than GC) rather than
  repeating the same two knobs at larger scale.

## 6. Files

| File | What it is |
|---|---|
| [`spare0_outlier_hunt_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/spare0_outlier_hunt_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | Raw per-call data, all 400 rows, normal run |
| [`spare0_outlier_hunt_6x7_gcdisable_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/spare0_outlier_hunt_6x7_gcdisable_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | Raw per-call data, all 400 rows, `--disable-gc` run |
| [`spare-qubit-cliff-addendum-28-preregistration-2026-09-16.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-28-preregistration-2026-09-16.md) | Predictions, written before either run |
| [`spare-qubit-cliff-addendum-28-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-28-2026-09-17.md) | This document |

## In one line

Two 400-call sweeps at the spare-qubit cliff's peak (normal and
`--disable-gc`) found zero outliers in either PSF-Zero arm in either
run (max was only 1.36-1.45x median, far under the pre-registered 3x
bar), and GC-disable produced no directional timing effect either --
so Addendum 27's rare-slowdown finding did not reproduce at this sample
size and a GC-mediated mechanism has no support here, unlike the
unrelated GC finding in Addendum 23. Per the preregistration's own
stated rule, this is reported as a null result; this specific chase
(GC, simple repetition) is reasonably exhausted, and a different angle
would be needed to pursue the thread further.

---


<!-- ===== Addendum 29 (source: spare-qubit-cliff-addendum-29-2026-09-17.md) ===== -->

> **Note added when merging:** Revisits Addendum 28's null result: the same overnight sweep script, run again, shows whole ~40-second rounds running uniformly hot on *both* Qiskit and PSF-Zero, at several spare values -- not the single-call, PSF-Zero-only spikes Addendum 27 first reported. Also corrects an over-claim ("PSF-Zero has no cliff") and finds an unrelated fact: PSF-Zero's 2-qubit gate count is exactly double Qiskit's, independent of the cliff.

## Addendum 29 -- the spare=0 "cliff-peak" instability looks like a round-scoped session/machine effect, not a spare=0-specific or PSF-Zero-specific one (2026-09-17)

**Status note on process**: this addendum analyzes a CSV the user uploaded
unprompted (from re-running `bench_cliff_overnight.py` -- the exact same
script that produced Addendum 27's original spare 0-24, 5-round sweep,
now run again a day later). **No prediction was pre-registered before
this run**, unlike Addenda 25/26/28. What follows is therefore reported
explicitly as a post-hoc analysis of data that arrived without a
pre-registration, not as a confirmed, pre-tested result -- and section 6
below states what a proper pre-registered follow-up would need to look
like before this reframing is treated as settled.

## 0. In one line

A fresh, independent re-run of the identical `bench_cliff_overnight.py`
script that produced Addendum 27's "PSF-Zero has rare unexplained
slowdowns at spare=0" finding now shows that the elevated timings are
**not confined to spare=0** (they recur, in the same rounds, at spare=2,
4, and 8 too) **and are not confined to PSF-Zero** (Qiskit L3's own
timing is elevated in the same windows, whereas in the original run
Qiskit was essentially rock-solid at spare=0, 1.13x max/min). The
elevated timings instead line up with two specific **rounds** (Round 1
and Round 4 of 5, each a ~40-second window covering many spare values
and all three arms) -- a shape that matches this project's own earlier,
already-suspected "session/machine-level drift" explanation (Addenda 19
and 26) far better than a spare=0-specific or PSF-Zero-specific
mechanism. This also gives a plausible reason Addendum 28's much
larger, spare=0-only, Qiskit-free sweep found zero outliers: if
whatever causes these windows needs something a tight, ~14-second,
Qiskit-free loop never does (run for very long, or include the
multi-second Qiskit L3 calls), Addendum 28's design would not have been
able to see it even if the underlying phenomenon is real and unchanged.

## 1. What arrived, and what it is

The user uploaded `bench_cliff_overnight_2026-09-17.csv` (270 data rows)
together with a terminal excerpt confirming it came from running
`python bench_cliff_overnight.py > overnight_log_2026-09-17.txt 2>&1` on
the same machine as the rest of this project's Intel-side work. This is
**the same script**, unmodified, that produced
`bench_cliff_overnight_2026-09-16.csv` (Addendum 27's own data): same
column schema (`Round, Spare, Qubits, LayoutSearch, Seed, Arm, Time_s,
Status, Error, FidelityCheck, ..., Grid, Timestamp`), same design (5
rounds x 6 spare values [0, 2, 4, 8, 16, 24] x 3 seeds [0, 1, 2] x 3 arms
[`qiskit_opt3`, `psf_zero_ls0`, `psf_zero_ls1`] = 270 rows), same 6x7
grid. All 270 rows report `Status=success` with 0 errors and 0
`CouplingViolations`; `FidelityCheck` is `structural_only` for every row,
which is expected and not a red flag -- at 18-42 qubits, full unitary
verification is not computable, exactly as Addendum 27 itself noted for
this same script.

## 2. The headline table: elevated spread now extends well past spare=0

Median times (this run) and each arm's own max/min spread at each spare
value:

| spare | qiskit median | qiskit max/min | psf_ls0 median | psf_ls0 max/min | psf_ls1 median | psf_ls1 max/min |
|---|---|---|---|---|---|---|
| 0 | 6494.7 ms | **3.02x** | 23.63 ms | **8.95x** | 16.62 ms | **10.43x** |
| 2 | 25.03 ms | **3.22x** | 10.40 ms | **4.24x** | 11.05 ms | **4.13x** |
| 4 | 26.57 ms | **3.74x** | 9.84 ms | **4.68x** | 10.74 ms | **4.29x** |
| 8 | 28.78 ms | **3.77x** | 9.86 ms | **3.69x** | 9.61 ms | **3.99x** |
| 16 | 32.84 ms | 1.10x | 8.72 ms | 1.23x | 8.80 ms | 1.16x |
| 24 | 34.19 ms | 1.09x | 7.96 ms | 1.23x | 8.09 ms | 1.25x |

Compare against Addendum 27's own table for the 2026-09-16 run (quoted
directly from that addendum, unchanged):

| spare | qiskit max/min (2026-09-16) | psf_ls0 max/min (2026-09-16) | psf_ls1 max/min (2026-09-16) |
|---|---|---|---|
| 0 | 1.13x | 3.74x | 8.93x |
| 2 | 1.71x | 1.16x | 1.59x |
| 4 | 1.06x | 1.11x | 1.17x |
| 8 | 1.04x | 1.13x | 1.09x |
| 16 | 1.04x | 1.13x | 1.15x |
| 24 | 1.17x | 1.15x | 1.12x |

Two things stand out from this side-by-side comparison:

1. **This run's spread at spare=0 for `psf_zero_ls0` (8.95x) and
   `psf_zero_ls1` (10.43x) is even larger than the original 3.74x/8.93x**
   -- so whatever this is, it did not go away or shrink a day later.
2. **This run's spread also shows up at spare=2, 4, and 8** (3.2x-4.7x
   across all three arms), where the original run was essentially flat
   (1.04x-1.71x, indistinguishable from ordinary noise). **And this
   run's `qiskit_opt3` spread at spare=0 (3.02x) is nearly 3x larger
   than the original's 1.13x** -- in the original run Qiskit's own
   timing was, by contrast, remarkably stable at the cliff's peak.

The core cliff finding itself is unaffected and still holds: this run's
own spare=0 median speedup is 274.8x (`psf_zero_ls0`) / 390.9x
(`psf_zero_ls1`) over Qiskit L3 -- squarely inside this project's
established 230-300x range from prior addenda. It is only the *spread*
around that median, and *where else it appears*, that has changed.

## 3. The pattern is round-scoped, not spare-value-scoped

Grouping every row's time by its own (spare, arm) median and averaging
that ratio across all spare values and all three arms, per round:

| Round | Mean ratio-to-median (all spare values, all 3 arms) | Timestamp window |
|---|---|---|
| 1 | **2.44x** | 08:54:27.27 -- 08:55:09.71 |
| 2 | 0.97x | 08:55:09.79 -- 08:55:30.64 |
| 3 | 1.02x | 08:55:30.71 -- 08:56:03.90 |
| 4 | **1.73x** | 08:56:03.96 -- 08:56:41.75 |
| 5 | 1.00x | 08:56:41.82 -- 08:57:02.99 |

Rounds 2, 3, and 5 are essentially at their own baseline (0.97x-1.02x --
indistinguishable from noise). **Rounds 1 and 4 are elevated across the
board**, and within each of those rounds the elevation appears at
spare=0, 2, 4, and 8 (fading to baseline by spare=16 and 24 -- see
section 5). Within Round 1, all three of its seeds (0, 1, 2) trend
upward together at spare=0 (`qiskit_opt3`: 7.71s -> 10.75s -> 19.24s;
`psf_zero_ls0`: 24.3ms -> 84.5ms -> 192.6ms); within Round 4, seed 0 is
at baseline (6.36s / 22.2ms) while seeds 1 and 2 are both elevated and
*stay* elevated (9.66s and 18.37s; 81.4ms and 81.2ms) rather than
spiking on a single call and immediately reverting. This "elevated for
a stretch of consecutive calls, not a single-call spike" shape is
different from how Addendum 27 characterized its own three outliers
("an outlier at spare=0 happens occasionally, on no seed in
particular") -- that description fit isolated single points in the
2026-09-16 data; it does not fit this run's two multi-call, multi-arm,
multi-spare-value windows.

**One exception, reported for honesty rather than smoothed over:**
Round 3, seed 1 shows `qiskit_opt3` at 18.61s (2.87x that spare's
median) at spare=0, while both PSF-Zero arms for that exact same
(round, seed) point are completely unremarkable (23.2ms and 16.1ms,
within 2% of Round 3's own other two seeds). This one row does not fit
the "whole round elevated together" pattern -- it looks like an
isolated Qiskit-only event inside an otherwise clean round, closer to
Addendum 19's original single-outlier shape than to Rounds 1/4's
broader pattern. This is flagged as an unresolved wrinkle, not
explained away.

## 4. Why this matters for Addendum 28's null result

Addendum 28 (`spare0_outlier_hunt.py`, 2026-09-17) ran 800 PSF-Zero-only
calls at spare=0 and found zero outliers by a 3x-median rule, leading to
the conclusion that Addendum 27's finding "did not reproduce at this
sample size." That conclusion is not being retracted -- the 800-call
sweep genuinely found nothing, and that is an honest result of the test
it ran. But this new data suggests the two scripts may simply not be
comparable tests of the same thing:

- `spare0_outlier_hunt.py`'s entire 400-call sweep completes in **14.4
  seconds** (10.4s with `--disable-gc`). Neither of this new run's two
  elevated windows (Round 1: ~42s; Round 4: ~38s) would fit inside that
  short a runtime even once, let alone with enough margin to land inside
  one by chance.
- `spare0_outlier_hunt.py` deliberately drops the Qiskit L3 arm entirely
  (its own docstring calls Qiskit L3 "too slow to repeat this many
  times"). If whatever produces these windows is triggered by, or
  otherwise linked to, the multi-second Qiskit L3 compile itself (this
  is speculation, not established -- see section 6), a script that never
  calls it would have no way to encounter that trigger.

So Addendum 28's null result and this addendum's positive finding are
not necessarily in conflict: they may be measuring the same underlying
phenomenon with two instruments, one of which (this one) happens to run
long enough and do enough varied work to catch it, and one of which
(Addendum 28's) does not. This is offered as the most likely
reconciliation, not as a proven one.

## 5. What is not explained by this reframing

- **Why Round 1 and Round 4 specifically became slow this run**, and
  why Rounds 2, 3, and 5 did not -- no external cause (CPU load,
  background process, thermal state) was measured or is available to
  check. This is exactly the same gap Addendum 19's original anomaly and
  Addendum 26's ~4x same-day drift were left with.
- **Why the elevation fades out above spare=8** (clean again by spare=16
  and 24, in both this run and the original). This could mean the
  underlying effect is genuinely tied to heavier compiles (more qubits,
  more two-qubit gates -- spare=0's circuit has 132 two-qubit gates vs.
  spare=24's 27, per this run's own `TwoQubitGates` column), or it could
  simply mean the effect's absolute-time contribution is small and
  invisible against spare=16/24's much smaller baseline times. Not
  distinguished here.
- **Round 3 seed 1's isolated Qiskit-only spike** (section 3) does not
  fit the round-scoped story cleanly.
- This is **one new day's data point** against **one original day's data
  point** -- both from the same machine. Nothing here says whether this
  pattern is common, rare, or specific to this particular machine.

## 6. What a proper follow-up would need (not run yet -- no prediction pre-registered here)

Per this project's own discipline, the next actual measurement on this
question should be pre-registered before it is run, not analyzed
post-hoc like this addendum. A reasonable design, sketched here without
committing to running it: repeat `bench_cliff_overnight.py`-style sweeps
(same shape: multiple spare values, multiple rounds, Qiskit L3 included)
for many more rounds than 5, logging wall-clock elapsed time and, if
available on the target machine, some independent proxy for machine
load (e.g. `psutil.cpu_percent()`, memory pressure, or at minimum a
precise per-row timestamp already present in this script's output) so
that "was this round one of the elevated ones" can be tested against
something other than the compile times themselves. A pre-registered
prediction from this addendum's own reframing: elevated rounds should
correlate with each other **within** a round (i.e., if `qiskit_opt3` is
elevated at spare=0 in a round, `psf_zero_ls0`/`ls1` at spare=2 and 4 in
that same round should also tend to be elevated) more strongly than
elevated calls correlate with anything about which spare value or which
arm was running -- that is the falsifiable core of "round-scoped
session effect" as opposed to "spare=0/PSF-Zero-specific mechanism."
This addendum does not run that test; it only proposes it.

## 7. Files

| File | What it is |
|---|---|
| [`bench_cliff_overnight_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_cliff_overnight_2026-09-17.csv) | This run's raw data (270 rows), uploaded by the user |
| [`bench_cliff_overnight_2026-09-16.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/bench_cliff_overnight_2026-09-16.csv) | Addendum 27's original run (already in the project, referenced for comparison, not re-uploaded) |
| [`spare-qubit-cliff-addendum-29-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-29-2026-09-17.md) | This document |

## 9. Update (2026-09-17): gate-count/depth analysis, in response to a proposed "PSF-Zero has no cliff" / "peephole optimization" framing

A follow-up question asked whether PSF-Zero's spare=0 spikes are simply
Qiskit's freeze dragging PSF-Zero down as OS-level collateral damage
(implying PSF-Zero itself has no cliff at all), and proposed closing the
2-qubit-gate-count gap with a cheap peephole cancellation pass. Two
corrections to the record, plus one new measurement from the same CSV
already in hand:

**Correction 1 -- PSF-Zero does have its own cliff; it is real, just far
smaller than Qiskit's.** Addendum 24 established a ~7.3x-8.2x cliff ratio
for `compile_for_hardware()` at `routing_optimization_level=1` (this
project's default and what `bench_cliff_overnight.py` uses), and Addendum
25 showed that ratio grows to ~11.5x at level 2 and ~250-275x at level 3
(landing inside Qiskit's own 263-300x range -- PSF-Zero's cliff advantage
comes specifically from staying at level 1). "PSF-Zero has no cliff" is
not accurate; "PSF-Zero's cliff is roughly 30-100x smaller than Qiskit's,
depending on the level" is.

**Correction 2 -- a GC/CPU-throttling cause for PSF-Zero's own spikes is
not established, and GC specifically was already tested and found to have
no effect.** Addendum 28 ran a dedicated `--disable-gc` A/B comparison
(800 calls total) specifically to test whether CPython's garbage collector
explained PSF-Zero's own spare=0 spikes -- median times shifted by under
1.1% with GC off, and the search arm's spread actually *increased*
slightly with GC disabled. This addendum's own section 3-4 found a
correlation between elevated rounds and all three arms' timing, but
explicitly flagged that as unconfirmed causation, not a demonstrated
"Qiskit's freeze causes PSF-Zero's spike" mechanism -- and section 3 also
reported a counter-example (Round 3, seed 1) where Qiskit alone was slow
(18.61s) while both PSF-Zero arms stayed completely normal at that exact
point, which a simple "Qiskit freezes, PSF-Zero gets dragged down with
it" story does not explain.

**New measurement -- the 2-qubit-gate-count gap is exactly 2.00x, at
every spare value, not a cliff-specific effect:**

| Spare | Qiskit 2Q-gate median | PSF-Zero (either arm) 2Q-gate median | Ratio |
|---|---|---|---|
| 0 | 63 | 132 / 126 | 2.10x / 2.00x |
| 2 | 60 | 120 | 2.00x |
| 4 | 57 | 114 | 2.00x |
| 8 | 51 | 102 | 2.00x |
| 16 | 39 | 78 | 2.00x |
| 24 | 27 | 54 | 2.00x |

This ratio is exact and constant from spare=0 to spare=24 -- it is a
general property of PSF-Zero's synthesis on this circuit family (dense
adjacent-qubit-pair blocks), not something specific to the saturated,
cliff-triggering coupling map. This is a cleaner target for a
gate-reduction pass than the timing cliff is: it is stable and does not
depend on hitting the rare slow window discussed in sections 3-4 above.

**One nuance worth flagging before designing a fix**: at spare=0
specifically, `psf_zero_ls0`'s *output circuit itself* is not identical
across repeated calls on the same input. Round 1's three seeds produced
132, 129, and 126 two-qubit gates respectively (depth 45, 28, 22) for
what should be deterministic input given a pinned `seed_transpiler` --
consistent with Addendum 9's previously-documented Qiskit-internal
VF2Layout/routing nondeterminism specifically at a saturated coupling
map. Away from spare=0, both PSF-Zero arms' gate counts were completely
stable call to call. A gate-reduction pass measured only at spare=0 could
therefore appear to help or hurt by chance, depending on which of these
non-deterministic outputs it happened to run against; measuring away
from the cliff (where output is stable) or averaging many repeats at
spare=0 avoids that trap.

**A related existing claim in `psf_compile.py`'s own docstring does not
hold at this circuit's scale.** The `compile_for_hardware()` docstring
states that at `routing_optimization_level=1`, PSF-Zero already produces
"the same 2-qubit gate count as Qiskit's optimization_level 2 and 3" on a
100-156 qubit dense-pair-block benchmark, paying only ~30-40% extra
depth. That is not what this 42-qubit, 6x7-grid measurement shows: here
PSF-Zero pays a flat, exact 2.00x gate-count penalty against
`qiskit_opt3` (Qiskit at `optimization_level=3`) at every spare value,
well outside "the same." This is not necessarily a contradiction --
different qubit count, and critically a coupling-map-constrained
(routed) circuit here vs. an unconstrained one in the docstring's own
benchmark -- but it means that docstring claim should not be assumed to
transfer to this circuit family, and the gate-count gap here is real and
larger than that docstring would suggest going in.

**What this means for the peephole-pass proposal**: it is a reasonable
idea and the 2.00x gap is real, general, and worth closing. But two
things should be checked before writing new code: (a) whether
`routing_optimization_level=2` -- an existing, already-implemented knob,
zero new code -- already closes some or all of this gap on this specific
circuit family (Addendum 25 measured its *time* cost, ~11.5x cliff ratio,
but never checked its gate count/depth output); if it does, most of the
value is available today for free. (b) the cliff-independent framing above
means a peephole pass needs to stay cheap on *every* compile, not just at
spare=0 -- away from the cliff PSF-Zero's speed margin over Qiskit is only
~1-3x (Addendum 24's table), not the "6-19 seconds of slack" the
spare=0 case offers, so a pass that is "free" relative to a multi-second
Qiskit cliff compile is not automatically free relative to a 20-30ms
non-cliff one.

## 10. Verification

- All medians, max/min ratios, per-round mean ratios, and the
  spare=0/2/4/8/16/24 tables in sections 2-3 were computed directly from
  the uploaded CSV's `Time_s` column with pandas (grouped by `Spare` and
  `Arm`), not transcribed from the terminal excerpt or estimated by eye.
- Addendum 27's original comparison table (section 2) was taken
  verbatim from that addendum's own text (via project search), not
  recomputed from a re-read of last year's -- sorry, last day's -- raw
  CSV, since that file was not re-fetched for this addendum; a future
  check could re-derive it directly from
  `bench_cliff_overnight_2026-09-16.csv` in the project to confirm the
  transcription is exact.
- `Status`, `Error`, `CouplingViolations`, and `FidelityCheck` were
  checked directly (270/270 success, 0 errors, 0 violations, all
  `structural_only`) rather than assumed from the terminal excerpt's
  summary lines.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum and the uploaded
  CSV -> 0 hits in both. (The terminal excerpt accompanying the upload
  did contain the local path `C:\Users\...\psf_zero_test>` -- per this
  project's standing rule, that path was not transcribed anywhere in
  this document or copied into any saved file; only the aggregate
  numeric results and the already-known script name were used.)
- Section 9's gate-count/depth table and the "exactly 2.00x, at every
  spare value" claim were computed directly from the same uploaded CSV's
  `TwoQubitGates` and `Depth` columns with pandas (grouped by `Spare` and
  `Arm`, median per group), not estimated from the two example numbers
  (126/132) quoted in the message that prompted this update. The
  `psf_compile.py` docstring quote in Section 9 was read directly from
  the current project copy of that file, not from memory of an earlier
  addendum's summary of it.

---


<!-- ===== Addendum 30 pre-registration (source: spare-qubit-cliff-addendum-30-preregistration-2026-09-17.md) ===== -->

> **Note added when merging:** Predictions for whether raising routing_optimization_level shrinks the 2x gate-count gap Addendum 29 found -- quoted verbatim in Addendum 30 below.

## Addendum 30 -- Pre-registration: does `routing_optimization_level` close PSF-Zero's 2.00x gate-count gap? (2026-09-17)

**Status: pre-registration only. No measurement has been run yet.** This
document states falsifiable predictions BEFORE `gate_count_vs_routing_level.py`
is run against the real `psf_zero_core`, per this project's standing
discipline (Addenda 25/26/28 preregistrations follow the same pattern).

## 1. What this follows up on

Addendum 29's 2026-09-17 update found that on this project's canonical
6x7 dense-pair-blocks grid circuit, PSF-Zero's own 2-qubit gate count at
`routing_optimization_level=1` (the default) is **exactly 2.00x** Qiskit
L3's, uniformly across spare=0 through spare=24 -- a general property of
this circuit family, not a cliff-specific artifact. Separately,
`psf_compile.py`'s own `compile_for_hardware()` docstring claims that
raising `routing_optimization_level` to 2 already makes PSF-Zero's
output "the same 2-qubit gate count as Qiskit's optimization_level 2 and
3," but that claim comes from a different benchmark (100-156 qubit
dense-pair-blocks over an unconstrained topology, not this project's 6x7
saturated-map grid). This was raised in response to a proposal to write a
brand-new custom gate-cancellation ("peephole") pass to close the gap --
before doing that, this addendum tests whether the gap is already closed
by an existing, zero-new-code parameter.

`gate_count_vs_routing_level.py` (delivered alongside this document)
measures `TwoQubitGates` and `Depth` for PSF-Zero at
`routing_optimization_level` in {1, 2, 3}, at spare in {0, 2, 4, 8, 16,
24}, across 3 seeds x 3 repeats per point (repeats exist specifically to
catch the spare=0 output-circuit nondeterminism Addendum 29 also found),
against a Qiskit L3 baseline measured once per (spare, seed).

## 2. Predictions, stated before running

**P1 -- rl=1 baseline replication.** At `routing_optimization_level=1`,
this sweep should reproduce Addendum 29's exact 2.00x gate-count ratio
(psf/qiskit) at every spare value tested, within the small variation
`--repeats` may reveal at spare=0. If this does NOT reproduce, that is a
methodology red flag (different circuit, different environment, or a
coding mistake in the new script) and the rl=2/rl=3 results below should
not be trusted until it is resolved.

**P2 -- rl=2 gate count.** Based on `psf_compile.py`'s own docstring
claim (measured on a different, non-saturated benchmark), the
directionally expected result is that `routing_optimization_level=2`
reduces PSF-Zero's gate-count ratio toward 1.0x, but this is explicitly
NOT assumed to fully match that docstring's "the same" claim on this
different circuit family -- this addendum's own reason for existing is
that the docstring's claim has never been tested here. A ratio that
lands meaningfully above 1.3x-1.5x at rl=2 would mean the docstring's
claim does not transfer to this circuit family, and would need its own
follow-up (flagging the docstring itself as needing a scope caveat).

**P3 -- rl=3 gate count.** `routing_optimization_level=3` is expected to
close the gate-count gap essentially completely (ratio close to 1.0x) --
Addendum 25 already established that rl=3's *time* cost converges to
being indistinguishable from plain Qiskit L3 (0.83x-0.96x relative
speed, i.e., no longer faster) at this project's cliff scenario, which
that addendum's own text attributes to Qiskit's preset pipeline
"re-running ConsolidateBlocks and UnitarySynthesis over input it has no
reason to trust" -- i.e., discarding PSF-Zero's own synthesis and
effectively re-deriving something close to Qiskit's own answer. If gate
count at rl=3 is NOT close to 1.0x despite the time cost already being
Qiskit-equivalent, that would be a genuinely surprising result worth its
own investigation (paying Qiskit's full time cost without getting
Qiskit's own gate-count answer).

**P4 -- spare=0 nondeterminism, level dependence unknown.** Addendum 29
found call-to-call gate-count variation for `psf_zero_ls0` specifically
at spare=0, `routing_optimization_level=1` (132/129/126 two-qubit gates
across 3 calls on identical input). This addendum makes NO directional
prediction about whether that nondeterminism persists, grows, or
disappears at `routing_optimization_level=2` or `3` -- this is
explicitly an open question the repeats are designed to answer, not a
hedged guess. A finding either way (persists / changes / disappears) is
reportable.

**P5 -- away from spare=0, gate count is stable call-to-call at every
level.** Based on Addendum 29 finding stability (zero spread) at spare
2-24 for `routing_optimization_level=1`, the same stability is expected
to hold at rl=2 and rl=3 too -- i.e., the nondeterminism (if P4 confirms
it exists) is expected to be specific to the saturated coupling map,
not a general property of higher routing levels. This is falsifiable:
nonzero spread at spare>=2 at any level would contradict it.

## 3. What this does NOT test

This script does not re-measure time cost at each level (already on
record from Addendum 25 for spare=0; not re-derived here for other
spare values, which is a real gap this addendum does not fill). It also
does not test `layout_search=True`'s interaction with routing level in
depth -- both PSF-Zero arms are measured for completeness, but no
prediction above is specific to the `layout_search=True` arm; any
difference between the two PSF-Zero arms found in the data is reported
as an observation, not scored against a prediction.

## 4. Run instructions

On the machine with the real `psf_zero_core` built:

```
python gate_count_vs_routing_level.py --rows 6 --cols 7 --spares 0,2,4,8,16,24 --levels 1,2,3 --seeds 3 --repeats 3
```

**Expect several minutes, not seconds.** At spare=0,
`routing_optimization_level=3` alone costs several seconds per call
(Addendum 25: ~6.7s median) and this sweep calls it repeatedly (3 seeds
x 3 repeats x 2 arms, plus warm-ups, at that one cell); the Qiskit L3
baseline at spare=0 is also multi-second per call. Every other (spare,
level) cell is fast (tens to a few hundred ms). This is expected
behavior, not a runaway loop.

Output: one CSV named
`gate_count_vs_routing_level_6x7_{cpu}_2026-09-17[_runN].csv` in the
current directory.

Before pasting this file's contents anywhere outside that machine: check
it for a local file path or any other machine-identifying string beyond
the CPU signature column, per this project's standing record-keeping
rules.

---


<!-- ===== Addendum 30 (source: spare-qubit-cliff-addendum-30-2026-09-17.md) ===== -->

> **Note added when merging:** The baseline measurement itself failed to reproduce before the routing-level question could even be asked: repeated runs returned one of three different, internally-consistent values (1.00x, 1.10x, 2.00x) with no run-to-run variation within a single process launch -- as if fixed once per process start. Per its own pre-registration, P2/P3 were not scored once P1 (baseline reproducibility) failed.

## Addendum 30 -- P1 failed: this run's own rl=1 baseline does not match Addendum 29's, in a way that reveals at least three distinct layout outcomes (2026-09-17)

**Scored against the pre-registration**:
`spare-qubit-cliff-addendum-30-preregistration-2026-09-17.md`. Per that
document's own instruction ("If this does NOT reproduce, that is a
methodology red flag ... and the rl=2/rl=3 results should not be trusted
until it is resolved"), this addendum reports that failure and does NOT
attempt to answer the original routing_optimization_level question from
this run's data.

## 0. In one line

`gate_count_vs_routing_level.py` was run once, and its own
`routing_optimization_level=1` baseline does **not** reproduce Addendum
29's 2.00x gate-count ratio: `psf_zero_ls1` came back at exactly 1.00x
(63 gates, identical to Qiskit, across all 9 calls) and `psf_zero_ls0`
came back at ~1.10x (69 gates, also identical across all 9 calls) --
neither matches the previously-measured ~2.00x (126-132 gates). Because
every one of this run's 9 repeats per arm agreed exactly (zero spread),
while a side-by-side look at Addendum 29's own raw data shows real
call-to-call variation for `psf_zero_ls0` within its own single run
(seed=2 alone produced 126, 132, 126, 129, and 132 across its 5 rounds),
the emerging picture is: **the layout stage at this saturated coupling
map can land in at least three distinct, reproducible-within-a-run
outcomes (roughly 63, 69, and 126-132 two-qubit gates), and which one a
given process lands in is not controlled by the circuit seed or
`seed_transpiler`** -- both of which were held fixed and identical
between this run and Addendum 29's. P1 is scored FAILED, and P2/P3
(whether `routing_optimization_level` closes the gate-count gap) cannot
be answered from this run's data, because this run never landed in the
"~126-132" regime the original question was about.

## 1. Correcting an imprecise claim in Addendum 29

Addendum 29's update illustrated PSF-Zero's spare=0 output-circuit
nondeterminism with "Round 1's three seeds produced 132, 129, and 126
two-qubit gates" -- that specific example compares three *different*
seeds (0, 1, 2), which is weaker evidence than it was presented as
(different circuits can legitimately need different gate counts). The
correct, stronger evidence for genuine same-input nondeterminism, not
surfaced at the time, is in the same CSV: **seed=2 alone, across the 5
rounds, produced 126, 132, 126, 129, and 132 two-qubit gates** --
five calls, same circuit, same `seed_transpiler=42`, same everything,
three different results. Seed=0 was perfectly stable at 132 across all 5
rounds; seed=1 was stable at 132 except for one call (129, Round 1).
This is corrected here rather than silently fixed in Addendum 29's own
text, per this project's append-only convention.

## 2. Side-by-side: this run vs. Addendum 29's run, `routing_optimization_level=1`, spare=0

| Source | Arm | Seed(s) | Two-qubit gates | Depth | Within-run spread |
|---|---|---|---|---|---|
| Addendum 29 (2026-09-17, [`bench_cliff_overnight.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/bench_cliff_overnight.py)) | `psf_zero_ls0` | 0 | 132 (all 5 rounds) | 45 | none |
| Addendum 29 | `psf_zero_ls0` | 1 | 129, 132, 132, 132, 132 | 28/45 | **yes** |
| Addendum 29 | `psf_zero_ls0` | 2 | 126, 132, 126, 129, 132 | 22/45/28 | **yes** |
| Addendum 29 | `psf_zero_ls1` | 0, 1, 2 | 126 (all 15 calls) | 22 | none |
| This run ([`gate_count_vs_routing_level.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/gate_count_vs_routing_level.py)) | `psf_zero_ls0` | 0, 1, 2 | **69** (all 9 calls) | 44 | none |
| This run | `psf_zero_ls1` | 0, 1, 2 | **63** (all 9 calls) | 23 | none |
| Both runs | `qiskit_opt3` | 0, 1, 2 | 63 (all calls, both runs) | 16 | none |

Qiskit's own baseline is identical and rock-stable across both
independent script runs (63 gates, depth 16) -- the circuit itself and
the Qiskit call are confirmed consistent between the two scripts; the
discrepancy is isolated entirely to PSF-Zero's own layout stage,
in both arms.

## 3. A working hypothesis, explicitly unconfirmed

The pattern -- perfectly stable *within* a single process's run, but
landing on a different stable value *across* two independent process
launches (this addendum's run vs. Addendum 29's run) -- points toward
something that is fixed once per process rather than randomized on every
individual compile call. Two candidates, neither tested yet:

- **`PYTHONHASHSEED`**, which Python randomizes once per process at
  startup unless explicitly set, and which can change the iteration
  order of sets/dicts used inside a backtracking search (such as VF2).
  This project has direct precedent for checking this exact variable
  (Addendum 22, for the unrelated ~145-iteration period), including the
  finding there that PYTHONHASHSEED did NOT explain that particular
  phenomenon -- that result does not transfer here without its own test.
- **Some other process-level state** (library initialization order,
  a warm/cold cache inside `psf_zero_core`, thread pool sizing) not
  narrowed down further here.

Addendum 9's previously-documented "Qiskit's own `seed=-1`-driven
VF2Layout jitter" is the most likely proximate mechanism (both PSF-Zero
arms' layout stages ultimately rely on VF2-family search), but "jitter"
alone does not explain why NINE calls in a row, across three different
circuit seeds, all landed on the exact same outcome in this run -- that
part specifically suggests a per-process rather than per-call source of
variation, which `seed=-1` jitter by itself would not obviously produce.

## 4. What this means for the original question

**P1 (rl=1 baseline replication): FAILED**, for both arms, in a way that
is qualitatively informative (a third, previously unseen value) rather
than just numerically off.

**P2 (rl=2 closes some of the gap) and P3 (rl=3 closes it further): not
answered by this run.** This run's own internal rl=1 -> rl=2 -> rl=3
progression is real data (`psf_zero_ls0`: 69 -> 69 -> 63 gates, depth 44
-> 35 -> 16; `psf_zero_ls1`: 63 -> 63 -> 63 gates, depth 23 -> 16 -> 16
throughout) and is reported for the record, but it describes what
happens to THIS run's ~1.0x-1.1x starting point, not what would happen
to Addendum 29's ~2.0x starting point -- those may respond to
`routing_optimization_level` differently, and this run cannot say.

**P4/P5 (spare=0 nondeterminism, and whether it's confined to
spare=0):** this run showed zero spread anywhere, including at spare=0 --
consistent with "outcome is fixed per process," which would predict
exactly this (no spread within one run, at any spare value) regardless
of whether the underlying phenomenon is real. Not distinguishing evidence
either way on its own.

## 5. What is proposed next (not yet run -- see Addendum 31 preregistration)

The natural, minimal-new-code test: run the *same*
`gate_count_vs_routing_level.py` script several more times as **separate
process launches** (not more repeats within one run -- this run already
showed that doesn't reveal anything) and record each run's own `rl=1`
gate count for both arms. If each full run lands on one of a small set
of discrete values (63 / 69 / 126-132, or others not yet seen) and stays
internally consistent throughout, that supports the per-process-lottery
hypothesis. A second pass, run only if the first is inconclusive: repeat
with `PYTHONHASHSEED` explicitly pinned to the same value across several
launches vs. left to vary, mirroring Addendum 22's method, to test that
specific candidate mechanism directly.

## 6. Files

| File | What it is |
|---|---|
| [`gate_count_vs_routing_level_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gate_count_vs_routing_level_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | This run's raw data (378 rows) |
| [`spare-qubit-cliff-addendum-30-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-30-2026-09-17.md) | This document |

## 7. Verification

- All values in sections 1-2 were read directly from the two CSVs
  (`bench_cliff_overnight_2026-09-17.csv`, already in the project, and
  this run's own upload) with pandas, not from memory of Addendum 29's
  prose summary.
- Confirmed both scripts' Qiskit baseline (`two_qubit_gates`/`depth`
  columns) agree exactly (63/16) before treating the PSF-Zero-side
  discrepancy as isolated to PSF-Zero's own layout stage rather than a
  circuit-construction difference between the two scripts.
- Confirmed zero spread in this run's own data across all 378 rows
  (every (spare, arm, level) cell's 3 seeds x 3 repeats = 9 values were
  identical), not just eyeballed from the printed summary table.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum and the uploaded
  CSV -> 0 hits in both.

---


<!-- ===== Addendum 31 pre-registration (source: spare-qubit-cliff-addendum-31-preregistration-2026-09-17.md) ===== -->

> **Note added when merging:** Predictions for testing whether Addendum 30's process-launch-lottery pattern is hash-seed-driven -- explicitly re-testing rather than assuming Addendum 22's unrelated hash-seed-negative result transfers to this different phenomenon.

## Addendum 31 -- Pre-registration: is PSF-Zero's spare=0 layout outcome fixed per process? (2026-09-17)

**Status: pre-registration only. No measurement has been run yet.**
Follows directly from Addendum 30, which found that a single run of
`gate_count_vs_routing_level.py` reproduced Addendum 29's exact
2-qubit-gate-count value for neither PSF-Zero arm, while showing zero
internal spread across 9 calls per arm/level -- suggesting the layout
outcome may be fixed once per process rather than randomized per call.

## No new script needed

This reuses `gate_count_vs_routing_level.py` exactly as already
delivered, run multiple times as **separate process launches** rather
than with more repeats inside one run (Addendum 30 already showed
within-run repeats do not reveal anything -- they were identical every
time).

## Predictions

**P1.** Running `python gate_count_vs_routing_level.py --rows 6 --cols 7
--spares 0 --levels 1 --seeds 3 --repeats 1` five separate times (five
separate `python` process launches, not five repeats in one launch) will
NOT produce five identical `psf_zero_ls0`/`psf_zero_ls1` gate-count
values at spare=0 -- i.e., at least one of the five launches will differ
from the others. This is the core falsifiable claim: if all five
launches agree exactly, the "per-process lottery" hypothesis from
Addendum 30 is wrong and something else (a stable configuration
difference between this session's script and Addendum 29's, not
randomness) would need to be found instead.

**P2.** Whichever value a given launch lands on, it will match one of
the three values already seen (63, 69, or 126-132) more often than not,
rather than producing many distinct new values -- weakly suggesting a
small number of discrete stable outcomes rather than a continuous
range. This is a soft, exploratory prediction, not a strict one.

**P3 (only run if P1 is confirmed).** Repeating the same five-launch
experiment with `PYTHONHASHSEED` pinned to the same fixed value for
every launch (e.g. `PYTHONHASHSEED=0` on Windows: `set
PYTHONHASHSEED=0` before each run in the same terminal session) will
produce five *identical* results, in contrast to P1's expected spread
with `PYTHONHASHSEED` left unset (Python's default: randomized per
process). This directly tests the hash-seed hypothesis from Addendum
30 section 3. This project has direct precedent for this exact test
(Addendum 22), including the finding there that hash-seed did NOT
explain a different, unrelated periodic anomaly -- that prior null
result is not assumed to transfer to this different phenomenon.

## Run instructions

```
python gate_count_vs_routing_level.py --rows 6 --cols 7 --spares 0 --levels 1 --seeds 3 --repeats 1
```

Run this exact command **five separate times** (five separate terminal
invocations -- closing and reopening the terminal between runs is not
necessary, just run the command five times). Each produces its own CSV
(run2/run3/... suffixing is automatic). Note, for each run, the
`psf_zero_ls0` and `psf_zero_ls1` two-qubit gate counts at spare=0.

If P1 confirms (the five runs do not all agree), and only then, repeat
with `PYTHONHASHSEED` pinned:

```
set PYTHONHASHSEED=0
python gate_count_vs_routing_level.py --rows 6 --cols 7 --spares 0 --levels 1 --seeds 3 --repeats 1
python gate_count_vs_routing_level.py --rows 6 --cols 7 --spares 0 --levels 1 --seeds 3 --repeats 1
python gate_count_vs_routing_level.py --rows 6 --cols 7 --spares 0 --levels 1 --seeds 3 --repeats 1
```

(all three in the same terminal session, so the pinned environment
variable applies to all of them), and compare against three more runs
with `set PYTHONHASHSEED=` (cleared, back to random) for contrast.

Each run takes well under a minute at `--spares 0 --levels 1` (no rl=3
multi-second calls in this reduced sweep). Before pasting any resulting
CSV's contents anywhere outside that machine: check it for a local file
path or other machine-identifying string beyond the CPU signature
column, per this project's standing record-keeping rules.

---


<!-- ===== Addendum 31 (source: spare-qubit-cliff-addendum-31-2026-09-17.md) ===== -->

> **Note added when merging:** The process-launch-lottery hypothesis is falsified, and replaced by the actual cause found in source: two scripts called compile_for_hardware with different entangling_basis and seed_transpiler arguments. This single fact explains both the gate-count-doubling puzzle (Addendum 29) and the three-value instability (Addendum 30) at once, as already-known behavior (entangling-basis.md; Addendum 9) rather than a new phenomenon.

## Addendum 31 -- P1 falsified: the gap is two real, named configuration differences between the two scripts, not randomness (2026-09-17)

**Scored against the pre-registration**:
`spare-qubit-cliff-addendum-31-preregistration-2026-09-17.md`. That document's
own text anticipated exactly this outcome as the alternative to P1: "if all
five launches agree exactly, the 'per-process lottery' hypothesis from
Addendum 30 is wrong and something else (a stable configuration difference
between this session's script and Addendum 29's, not randomness) would need
to be found instead." That is what happened, and the configuration
difference has now been found and confirmed directly from `psf_compile.py`'s
own source and docstrings -- not inferred from timing patterns alone.

## 0. In one line

Five separate process launches of `gate_count_vs_routing_level.py` (Addendum
30/31's script) produced **69/63 two-qubit gates, five times out of five, with
zero exceptions** -- P1 (predicting at least one launch would differ) is
**FALSIFIED**. The reason is not `PYTHONHASHSEED` or any other per-process
randomness: `gate_count_vs_routing_level.py` and `bench_cliff_overnight.py`
(Addendum 29's script) call `psf_compile.compile_for_hardware()` with two
different arguments -- `entangling_basis` ("cx" vs "canonical") and
`seed_transpiler` (pinned to 42 vs never passed at all, defaulting to `None`)
-- and both differences are already independently documented elsewhere in
this project as real, causal effects on exactly this quantity.

## 1. The five launches: exact agreement

| Launch | `psf_zero_ls0` (2Q gates / depth) | `psf_zero_ls1` (2Q gates / depth) | `qiskit_opt3` (2Q gates / depth) |
|---|---|---|---|
| run2 | 69 / 44 (all 3 seeds) | 63 / 23 (all 3 seeds) | 63 / 16 (all 3 seeds) |
| run3 | 69 / 44 | 63 / 23 | 63 / 16 |
| run4 | 69 / 44 | 63 / 23 | 63 / 16 |
| run5 | 69 / 44 | 63 / 23 | 63 / 16 |
| run6 | 69 / 44 | 63 / 23 | 63 / 16 |

45/45 individual calls (5 launches x 3 seeds x 3 arms) agree exactly within
each arm. **P1 is FALSIFIED**: the pre-registration required at least one
launch to differ from the others, and none did.

**P2** (a launch will land on one of the three previously-seen values --
63, 69, or 126-132 -- more often than not): trivially holds (all five landed
on 69/63, two of the three previously-seen values), but see Section 3 below
for why this should not be read as support for "a small number of discrete
random outcomes" -- the real explanation is that both scripts are each
internally deterministic, for a specific, named reason, not that there is a
small random menu of outcomes any one process might draw from.

**P3** (`PYTHONHASHSEED` pinning test): **not run**, correctly, per its own
"only run if P1 is confirmed" condition. P1 was not confirmed -- it was
falsified in the informative direction the pre-registration described.
Pinning `PYTHONHASHSEED` would not have been informative here regardless: the
mechanism found below has nothing to do with hashing or dict/set iteration
order.

## 2. The two real differences, read directly from the two scripts

`bench_cliff_overnight.py`'s PSF-Zero call:

```python
def run_psf(qc, cmap, routing_optimization_level, layout_search):
    import psf_compile
    return psf_compile.compile_for_hardware(
        qc, coupling_map=cmap, basis_gates=BASIS_GATES,
        routing_optimization_level=routing_optimization_level,
        verify=True, entangling_basis="canonical",
        layout_search=layout_search,
    )
```

`gate_count_vs_routing_level.py`'s PSF-Zero call:

```python
out = compile_for_hardware(
    qc, coupling_map=cm, basis_gates=BASIS_GATES,
    routing_optimization_level=level,
    entangling_basis="cx", seed_transpiler=SEED_TRANSPILER,
    layout_search=layout_search,
)
```

Two arguments differ, and neither is a typo or a default carried over by
accident -- both are named, deliberate choices in each script:

1. **`entangling_basis`**: `"canonical"` (Addendum 29's script) vs. `"cx"`
   (Addendum 30/31's script). `psf_compile.py`'s own `compile()` docstring
   states the effect directly: *"entangling_basis: 'canonical' emits
   RXX/RYY/RZZ directly; 'cx' re-expresses the entangling core through
   Qiskit's exact CX-basis decomposer, which costs 2x fewer native gates on
   hardware whose native 2-qubit gate is CX-like (see
   docs/findings/entangling-basis.md)."* That referenced document
   (already in this project) measured exactly this ratio independently, on
   an unrelated fidelity investigation, months before this addendum:
   *"At optimization_level 0-1 the PSF-Zero-shaped [canonical] circuit costs
   exactly twice the native two-qubit gates for the identical unitary."*
   Both benchmark scripts here call the outer `transpile()` with
   `basis_gates=["rz","sx","x","cx"]` at `routing_optimization_level`
   (=`optimization_level`) 1 -- exactly the regime `entangling-basis.md`'s
   table shows the 2.00x ratio holding.

2. **`seed_transpiler`**: pinned to `42` in `gate_count_vs_routing_level.py`,
   never passed at all in `bench_cliff_overnight.py` (so it defaults to
   `compile_for_hardware`'s own default, `None`). `compile_for_hardware()`'s
   own docstring: *"seed_transpiler pins the internal routing search. Leaving
   it unset means an optimization_level >= 2 transpile returns a different
   circuit, and takes a different amount of time, on every call for
   identical input."* This project's own Addendum 9 separately established
   that `VF2Layout` itself is non-deterministic at a saturated coupling map
   without a pinned seed -- the mechanism this docstring is describing.

## 3. Why this explains BOTH open puzzles at once, not just the headline number

**Puzzle A (this addendum's main question): why 69/63 here vs. ~126-132/126
in Addendum 29.** `entangling_basis="canonical"` (Addendum 29) makes each
consolidated block emit up to 3 native rotation gates (RXX/RYY/RZZ), each of
which then costs 2 further CX gates once the outer `transpile()` translates
to the `cx`-only basis (6 CX-equivalent per block). `entangling_basis="cx"`
(this addendum's script) emits the block as CX gates directly via
`TwoQubitBasisDecomposer`, needing no further translation (3 CX per block).
That is the ~2x factor: `psf_zero_ls1`'s 63 (this script) vs. 126 (Addendum
29) is **exactly** 2.00x, matching `entangling-basis.md`'s own table exactly.

**Puzzle B (Addendum 29's own internal spread, never explained until now):
why did `psf_zero_ls0` vary call-to-call within Addendum 29's own single run
(126, 132, 126, 129, 132 for seed=2) while `psf_zero_ls1` stayed rock-stable
at 126 every time?** `psf_zero_ls0` is `layout_search=False`: it relies on
Qiskit's own default layout stage (`VF2Layout` -> `SabreLayout` ->
`VF2PostLayout`), which Addendum 9 already showed is non-deterministic at a
saturated coupling map -- and Addendum 29's script never pinned
`seed_transpiler` for this call, so nothing suppressed that non-determinism.
`psf_zero_ls1` is `layout_search=True`: `compile_for_hardware()`'s own
docstring confirms this **skips the whole layout stage, not just the layout
search** ("with initial_layout: SetLayout, ApplyLayout" -- no `VF2Layout`, no
`SabreLayout`, no `VF2PostLayout` at all), because `layout_search=True`
finds its own layout via `smart_vf2_layout()` and threads it through
`initial_layout`. With no `VF2Layout`/`SabreLayout` stage running at all,
there is no seed-dependent search left to vary -- hence `psf_zero_ls1`'s
perfect stability, even fully unseeded, even at the cliff's worst point.

**Corroborating evidence already sitting in Addendum 29's own raw data**,
re-read for this addendum: away from the cliff (spare != 0, e.g. spare=2/
4/8/16 in `bench_cliff_overnight_2026-09-17.csv`), `psf_zero_ls0` and
`psf_zero_ls1` agree with EACH OTHER exactly (120/120 at spare=2, 114/114 at
spare=4, 102/102 at spare=8, 78/78 at spare=16) and both are exactly 2.00x
`qiskit_opt3`'s value at that spare -- consistent with the pre-existing,
uniform ~2.00x factor `gate_count_vs_routing_level.py`'s own docstring cites
from Addendum 29's update. **Only at spare=0** -- the one point where
`VF2Layout` is known (Addendum 9) to struggle -- does `psf_zero_ls0` decouple
from `psf_zero_ls1` and start varying. That is exactly the pattern predicted
by "the layout-search stage is the only source of variance, and it only
misbehaves at the saturated point," not a generic per-process lottery.

## 4. What this addendum does and does not establish

**Established, from source + two independent prior measurements agreeing:**
the entire ~2x magnitude gap between this script's numbers and Addendum 29's
is `entangling_basis` (a known, already-published, ~2x effect), and the
within-run instability specific to Addendum 29's `psf_zero_ls0` arm is the
combination of an unpinned `seed_transpiler` and the already-documented
`VF2Layout` jitter at a saturated map (Addendum 9) -- not a new phenomenon.

**Not established, and not claimed:** that `PYTHONHASHSEED` plays no role in
anything (it was never tested here, correctly, since P1's condition for
testing it was not met); that every remaining digit of Addendum 29's spread
(126 vs 129 vs 132, specifically) is fully accounted for -- the *mechanism*
(unseeded VF2Layout/SabreLayout search) is identified, but which exact swap
count each specific unseeded call happens to land on was not re-derived
gate-by-gate here.

**Superseded, explicitly:** Addendum 30's Section 3 "working hypothesis"
(`PYTHONHASHSEED` or unspecified "process-level state") is superseded by
this addendum's finding. That hypothesis is not shown to be wrong in
principle -- it was simply never the actual cause here; the actual cause is
two named, intentional keyword arguments that differ between the two
scripts.

## 5. What is proposed next

This closes the immediate mystery that motivated Addenda 30-31. It does
**not** close Addendum 29's own original question (does raising
`routing_optimization_level` close PSF-Zero's 2x gate-count gap against
Qiskit L3) -- that requires re-running the `routing_optimization_level`
sweep with a configuration that matches the ~2.00x-gap regime this project
actually cares about (i.e. `entangling_basis="canonical"`, matching real
hardware whose native gate is not CX-like per `entangling-basis.md`, or
`entangling_basis="cx"` if the target hardware's native gate is CX-like --
the project's own README should be checked for which is the intended
target before choosing). `gate_count_vs_routing_level.py`'s existing rl=2/
rl=3 data (already collected, not yet reported here) describes the `"cx"`-
basis, seed-pinned regime only, and should be labeled as such rather than
presented as answering the original `"canonical"`-basis question.

## 6. Files

| File | What it is |
|---|---|
| [`gate_count_vs_routing_level_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17_run2.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gate_count_vs_routing_level_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17_run2.csv) through [`gate_count_vs_routing_level_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17_run6.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gate_count_vs_routing_level_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17_run6.csv) | The five separate process launches (9 rows each, 45 total) |
| [`spare-qubit-cliff-addendum-31-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-31-2026-09-17.md) | This document |

## 7. Verification

- All five CSVs read directly and compared cell-by-cell; the "45/45 calls
  agree" claim in Section 1 was checked exactly, not eyeballed from the
  printed summary tables.
- `entangling_basis` and `seed_transpiler` differences in Section 2 were
  found by reading both scripts' actual `run_psf`/PSF-Zero-call source side
  by side, not from memory or from either script's docstring alone.
- The "2x fewer native gates" claim was cross-checked against
  `docs/findings/entangling-basis.md`'s own independently-measured table
  (0.00x/1.00x ratios by `optimization_level`), not taken solely from
  `psf_compile.py`'s one-line docstring summary of it.
- The "layout_search=True skips VF2Layout/SabreLayout/VF2PostLayout
  entirely" claim was verified against `compile_for_hardware()`'s own
  docstring, which documents this from a direct `transpile(callback=...)`
  measurement (dated 2026-09-14).
- The "away from spare=0, ls0 and ls1 agree with each other and with 2x
  qiskit_opt3" corroborating pattern (Section 3) was re-read directly from
  `bench_cliff_overnight_2026-09-17.csv`'s non-spare-0 rows, not assumed.
- Pre-publication check: `grep` against this project's private personal-information pattern list, this document and all five uploaded CSVs -> 0 hits.

---


<!-- ===== Addendum 32 (source: spare-qubit-cliff-addendum-32-2026-09-17.md) ===== -->

> **Note added when merging:** Confirms the gate-count parity (Qiskit vs. PSF-Zero, matched entangling_basis) generalizes to an 8x8 grid, exactly as predicted. A secondary prediction about which layout_search mode would exceed Qiskit's count did not hold, traced to grid column parity -- recorded as a refined, not exciting, correction.

## Addendum 32 -- P1 confirmed exactly; P2's letter holds but its interesting part didn't: 8x8 needed zero extra swaps where 6x7 needed some (2026-09-17)

**Scored against the pre-registration**:
`spare-qubit-cliff-addendum-32-preregistration-2026-09-17.md`.

## 0. In one line

`qiskit_opt3` and `psf_zero_ls1` both came back at **exactly 96** two-qubit
gates (`3 x 32` blocks), with zero spread across all 3 seeds and all 3
repeats -- **P1 CONFIRMED exactly**, generalizing Addendum 31's "zero-swap,
3-CX-per-block" pattern from 6x7 to 8x8. `psf_zero_ls0` also came back at
**exactly 96** -- P2's literal claim (`>= 96`, stable) holds, but the
interesting part of the reasoning behind it (that `ls0` would show *some*
excess over the zero-swap baseline, mirroring 6x7's 69-vs-63) did **not**
happen: at 8x8, `ls0` needed **zero** extra swaps, matching `ls1` exactly,
including on depth (23 vs 23, not 44 vs 23 as at 6x7). This is a genuine,
useful negative result, and a plausible mechanism for it -- grid column
parity -- is proposed below as a new, separately falsifiable hypothesis,
not asserted as confirmed.

## 1. Results

| Arm | Two-qubit gates | Depth | Spread across 3 seeds x 3 repeats |
|---|---|---|---|
| `qiskit_opt3` | 96 | 16 | none |
| `psf_zero_ls0` | 96 | 23 | none |
| `psf_zero_ls1` | 96 | 23 | none |

All 27 rows (3 seeds x 3 repeats x 3 arms) matched this exactly -- no
failures, no spread anywhere.

## 2. Scoring against the pre-registration

**P1** ("`qiskit_opt3` and `psf_zero_ls1` will each come back at exactly 96
... identical across all 3 seeds and all 3 repeats"): **CONFIRMED
exactly**, no qualification needed.

**P2** ("`psf_zero_ls0` will be >= 96 ... and perfectly stable"):
**Technically confirmed** -- 96 >= 96, and it is perfectly stable. But the
pre-registration was explicit that it did *not* commit to a specific
excess and predicted one would exist, reasoning from the 6x7 result (69,
6 gates over the 63-gate zero-swap baseline). At 8x8 the excess is
**zero** -- `ls0` matched the zero-swap baseline exactly, something the
6x7 data gave no reason to expect. This should be read as **P2's
underlying expectation not materializing**, not as a clean confirmation.

**P3** ("`ls0`'s excess ... will be a small multiple of 3"): **Vacuous**.
Zero is a multiple of 3, but there was no excess to check the claim
against, so this pre-registered prediction was not meaningfully tested.

## 3. A candidate mechanism for why 8x8 needed zero extra swaps, proposed but NOT confirmed

`build_dense_pair_blocks_circuit()` always pairs *consecutive* logical
qubit indices: `(0,1), (2,3), (4,5), ...`. `CouplingMap.from_grid(rows,
cols)` numbers physical qubits row-major (`index = row*cols + col`), so a
pair `(i, i+1)` sits on a physically adjacent edge -- needing no routing
at all under the plain identity/trivial layout -- exactly when `i` and
`i+1` fall in the *same row*, i.e. whenever `i mod cols != cols - 1`.

Since every pair start `i` is even, this fails (the pair straddles a row
boundary and is not a native edge) only when `cols - 1` is itself even --
i.e. **when `cols` is odd**. At 6x7 (`cols=7`, odd), pair `(6,7)`, `(20,21)`,
`(34,35)` all straddle a row boundary this way -- three pairs per that
grid's 42-qubit circuit that the trivial layout cannot satisfy, which is
one candidate explanation for why `psf_zero_ls0` (relying on Qiskit's own
default layout search rather than PSF-Zero's own matching-aware search)
came in above the zero-swap baseline there. At 8x8 (`cols=8`, even),
`cols - 1 = 7` is odd, so no even pair-start `i` can ever equal it --
**every** pair sits inside one row, the trivial layout already satisfies
every interaction, and no routing swap is needed regardless of which
layout algorithm is used. That would explain both this addendum's `ls0
== ls1 == qiskit_opt3 == 96` result and 6x7's `ls0 > ls1` gap in one
stroke, with no per-process randomness involved either way.

**This is proposed, not confirmed.** It was reasoned out after seeing the
result, not pre-registered beforehand, and Qiskit's actual default layout
pipeline (`VF2Layout` -> `SabreLayout` -> `VF2PostLayout`) is not the
trivial/identity layout -- it could in principle reach a different
zero-swap layout by a different route, or the row-boundary count (3 for
6x7, per above) might not map cleanly onto the observed 6-gate (2-swap)
excess there. The clean way to test this is a new pre-registered
comparison that varies row/column parity independently (e.g. an
even-columns grid that still needs swaps for some other reason, or an
odd-columns grid to confirm the excess reappears), not to declare it
settled from a single data point.

## 4. What this addendum does and does not establish

**Established:** Addendum 31's "3-CX-per-block, zero-swap-when-possible"
gate-count model generalizes exactly from 6x7 to 8x8 for `qiskit_opt3`
and `psf_zero_ls1`, and `entangling_basis="cx"` + `seed_transpiler=42`
continue to produce fully deterministic output (zero spread) at a larger,
still-fully-saturated (spare=0) grid.

**Not established:** that PSF-Zero's default (`ls0`) layout path
generally matches the zero-swap baseline at larger grids -- this addendum
shows one grid size (8x8) where it happened to, and one (6x7, Addendum
31) where it didn't. The column-parity explanation in Section 3 is a
hypothesis for *why*, awaiting its own test.

## 5. Proposed next test (not yet run -- see Addendum 33 preregistration)

Run the same script on a grid that isolates column parity from the 6x7
vs. 8x8 comparison's other differences (grid size, aspect ratio): an
**odd-columns grid close in size to 8x8** (e.g. `--rows 9 --cols 7`, 63
qubits, spare=0 after dropping to an even qubit count, or `--rows 8 --cols
7`, 56 qubits) would test whether odd `cols` reproduces an `ls0` excess
at a grid this large, and an **even-columns grid close in size to 6x7**
(e.g. `--rows 6 --cols 8`, spare enough to reach an even n) would test
whether even `cols` removes the excess even at 6x7's scale. See Addendum
33's preregistration for the exact pre-committed predictions before
either is run.

## 6. Files

| File | What it is |
|---|---|
| [`gate_count_vs_routing_level_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/gate_count_vs_routing_level_8x8_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | This run's raw data (27 rows) |
| [`spare-qubit-cliff-addendum-32-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-32-2026-09-17.md) | This document |

## 7. Verification

- All values in Sections 1-2 read directly from the uploaded CSV (27
  rows), not from the printed summary table alone -- confirmed the
  summary table's `96.0/96.0` matched every individual row, including
  `qiskit_opt3` (which the printed summary does not show, since it prints
  only the two PSF-Zero arms' pivot table).
- The column-parity arithmetic in Section 3 was checked by hand for both
  grids (6x7's three straddling pairs at i=6,20,34; 8x8's zero straddling
  pairs) against `build_dense_pair_blocks_circuit()`'s actual pairing
  rule (`range(0, num_qubits - 1, 2)`) and `CouplingMap.from_grid`'s
  documented row-major indexing -- not assumed.
- Pre-publication check: `grep` against this project's private personal-information pattern list, this document and the uploaded CSV -> 0 hits in both. (The
  terminal transcript accompanying this run's upload contained the
  `C:\Users\...\psf_zero_test>` prompt path, as it always does -- per this
  project's standing rule, that raw path was not copied into this
  document or the CSV; only the aggregate numeric results were used.)

---


<!-- ===== Addendum 34 pre-registration (source: spare-qubit-cliff-addendum-34-preregistration-2026-09-17.md) ===== -->

> **Note added when merging:** Predictions for locating the cliff's exact occupancy threshold and testing whether VF2Layout's stop reason coincides with it -- cites an independent external profiling study (arXiv 2504.15141) that found the same VF2Layout cost pattern without varying occupancy.

## Addendum 34 -- pre-registration: where exactly is the occupancy threshold, and does VF2Layout's stop reason move with it? (2026-09-17)

**Written BEFORE `occupancy_sweep.py` is run at full scale on any
machine.** Reduced-scale smoke tests (3x4 grid, 2 gates per pair) were run
in a cloud container solely to confirm the script executes and that its
instrumentation captures what it claims to capture; those runs are code
validation, not measurement, and their numbers are not used anywhere. Do
not edit this document after seeing full-scale output.

## 0. Why this experiment exists

Every cliff measurement in Addenda 4-32 compares a saturated map
(`spare=0`) against a comfortably padded one (`spare=4`, `6`, `24`). That
establishes *that* there is a cliff. It does not say **where the edge
is**, and it does not separate two candidate explanations that both
predict the same coarse result:

- **(A) Budget exhaustion near a constraint-satisfaction threshold.** A
  valid embedding still exists at `spare=0`, but the space of valid
  embeddings has collapsed to so few that VF2's bounded search
  (`call_limit`) runs out before finding one, and Qiskit falls back to
  `SabreLayout`. Under (A), the timing cliff should coincide with the
  point where `VF2Layout_stop_reason` flips.
- **(B) Something that scales smoothly with occupancy** -- more qubits to
  route, longer circuits, more SWAPs -- with no distinguished threshold
  at all.

An external profiling study (arXiv 2504.15141) independently found
`VF2Layout` consuming >99% of compile time (61.6s) on a 100-qubit circuit,
so the phenomenon is not unique to this project's setup. That study used
one backend at one fixed qubit count and never varied device occupancy,
so it cannot distinguish (A) from (B) either. That is the gap this
addendum is aimed at.

## 1. What is being measured

`occupancy_sweep.py`, on a fixed coupling map, sweeping only the number of
spare (unused) physical qubits. Per run it records total transpile time,
**per-pass time via `transpile(callback=...)`**, the value of
`property_set["VF2Layout_stop_reason"]`, output 2-qubit gate count and
depth, and two feasibility facts computed independently of Qiskit: whether
the coupling graph admits a perfect matching, and the maximum matching
size against the number of disjoint qubit pairs the circuit requires.

No PSF-Zero code is involved in this experiment at all. It measures
Qiskit's own behaviour.

## 2. Pre-registered predictions

**P1 (threshold is sharp, not gradual).** On a 6x7 grid at
`optimization_level=3`, median compile time at `spare=0` will be **at
least 20x** the median at `spare=2`. The single largest step-to-step ratio
across the whole sweep will occur at one of the first two steps
(`spare=0 -> 1` or `1 -> 2`), not somewhere in the middle of the sweep.

**P2 (the mechanism is visible, and it coincides).** `VF2Layout_stop_reason`
will read `NO_SOLUTION_FOUND` at `spare=0` and `SOLUTION_FOUND` at the
largest spare value tested. The spare value at which it flips will lie
**within one step** of the spare value at which the largest timing drop
occurs. If the stop reason flips several steps away from the timing cliff,
P2 is falsified and explanation (A) is in trouble.

**P3 (the time is in the layout stage).** At `spare=0`,
`optimization_level=3`, the summed time of the layout-related passes
(`VF2Layout`, `SabreLayout`, `VF2PostLayout`) will be **more than 80%** of
total transpile time. At the largest spare tested, it will be **under
50%**.

**P4 (the instance is hard, not impossible).** At `spare=0` on a 6x7 grid,
a perfect matching **does exist** in the coupling graph (6x7 = 42 vertices;
pairing vertically inside each of the 7 columns of 6 gives 21 disjoint
edges), so the circuit's interaction graph -- 21 disjoint pairs -- is
embeddable. Therefore any `NO_SOLUTION_FOUND` observed at `spare=0` is a
**search-budget failure, not an infeasibility**. The script computes the
matching independently with `networkx` and records it, so this is checked
rather than asserted.

**P5 (soft, no commitment to magnitude).** `optimization_level=1` will
show no cliff worth the name -- under 3x across the entire sweep -- because
it does not run `VF2Layout` in the same configuration. This is the control
arm; if `optimization_level=1` also cliffs, the story is wrong.

## 3. What would falsify the addendum's thesis

- Timing declines smoothly with occupancy and no step exceeds ~5x (kills
  P1; supports explanation (B)).
- `VF2Layout_stop_reason` is `NO_SOLUTION_FOUND` across the entire sweep,
  including at large spare, while timing still drops (kills P2; the stop
  reason is then not the mechanism).
- The layout passes are a minority of runtime even at `spare=0` (kills
  P3; the time is going somewhere else and this project has been pointing
  at the wrong pass for 30 addenda).
- No perfect matching exists at `spare=0` (kills P4; the instance is
  infeasible, `NO_SOLUTION_FOUND` is simply correct, and there is no
  CSP-threshold story to tell).

## 4. Run command (full scale, on the measurement machine)

```
python occupancy_sweep.py --rows 6 --cols 7 \
    --spares 0,1,2,3,4,5,6,8,10,12,16,20,24 \
    --levels 1,3 --seeds 3 --repeats 3
```

Expect this to be slow: `spare=0` at `optimization_level=3` costs several
seconds per call on this project's machines, and the first few spare values
are the expensive ones. A second run at `--rows 8 --cols 8` is worth having
for the paper but is not required to score the predictions above.

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep.py) | the script (no PSF-Zero dependency) |
| `occupancy_sweep_<rows>x<cols>_<cpu>_<date>.csv` | its raw output |
| this document | the pre-registered predictions |

---


<!-- ===== Addendum 34 (source: spare-qubit-cliff-addendum-34-2026-09-17.md) ===== -->

> **Note added when merging:** **Corrects this series' own mechanism description.** The cliff is a single-step event (spare 0->1, 193-238x) coinciding exactly with VF2Layout's stop-reason flip, and a perfect matching provably exists at spare=0 (independent networkx check) -- confirming search-budget failure, not infeasibility. But `SabreLayout` measured exactly 0ms in all 234 rows: the real cost is two separate VF2-family searches, `VF2Layout` and `VF2PostLayout`, not a fallback to Sabre as every earlier addendum assumed.

## Addendum 34 -- the threshold is exactly at spare=0, VF2Layout's stop reason coincides with it exactly, and a perfect matching exists at spare=0 anyway: this is a search-budget failure, not an infeasible instance (2026-09-17)

**Scored against the pre-registration**:
`spare-qubit-cliff-addendum-34-preregistration-2026-09-17.md`.

**No PSF-Zero code was exercised anywhere in this addendum.** This measures
Qiskit's own `transpile()` on a 6x7 grid, `spares` swept finely from 0 to
24, `optimization_level` in {1, 3}, 3 seeds x 3 repeats (234 rows, 0
errors).

## 0. In one line

Every strong-form prediction was confirmed, several more precisely than
expected: the cliff is a single, one-step event between spare=0 and
spare=1 (193x), it coincides exactly (same step) with `VF2Layout`'s stop
reason flipping from `nonexistent solution` to `solution found`, and a
perfect matching **provably exists** in the spare=0 coupling graph
(`matching_slack=0`, computed independently with `networkx`) -- so the
`nonexistent solution` verdict at spare=0 is Qiskit's search giving up
inside its own budget, not the instance actually being unsolvable. One
prediction (P3's second half) was falsified in an informative way, and
digging into *why* surfaced a real correction to this addendum's own
pre-registered mechanism story: the dominant cost at spare=0 is not
"failed `VF2Layout` then slow `SabreLayout`" as the pre-registration's
motivation section assumed -- `SabreLayout` measured **0ms** in every
single row. The cost is two separate, expensive VF2-family searches:
`VF2Layout` itself (8,451ms median) and, afterward, `VF2PostLayout`
(6,712ms median) attempting its own post-hoc re-embedding.

## 1. Results

Median total compile time (ms), `optimization_level` 1 vs 3, by spare:

| spare | occupancy | L1 (ms) | L3 (ms) |
|---:|---:|---:|---:|
| 0 | 100.0% | 36.27 | **15,583.91** |
| 1 | 97.6% | 76.91 | 80.70 |
| 2 | 95.2% | 62.12 | 65.25 |
| 3 | 92.9% | 56.37 | 65.31 |
| 4 | 90.5% | 65.62 | 67.15 |
| 5 | 88.1% | 62.38 | 70.17 |
| 6 | 85.7% | 60.46 | 69.77 |
| 8 | 81.0% | 56.84 | 73.64 |
| 10 | 76.2% | 56.96 | 81.87 |
| 12 | 71.4% | 49.99 | 71.63 |
| 16 | 61.9% | 46.35 | 85.84 |
| 20 | 52.4% | 41.92 | 101.76 |
| 24 | 42.9% | 38.26 | 105.02 |

The entire cliff is one step: spare 0 -> 1 is a 193.1x drop at L3; every
other step-to-step ratio in the sweep is under 1.3x in either direction.
(Interesting secondary detail, not part of any prediction: L3's time
*rises* slowly as spare keeps increasing past the cliff, from 65-70ms
around spare=2-6 to 105ms at spare=24 -- consistent with a larger,
easier-to-search coupling graph costing slightly more per `VF2Layout`
call even when it succeeds immediately, not with anything cliff-like.)

`VF2Layout_stop_reason`, both optimization levels, every seed and repeat:
`nonexistent solution` at spare=0, `solution found` at every spare>=1
tested, with no mixed or inconsistent point anywhere in the sweep.

Independent feasibility check (`networkx` max-cardinality matching,
computed without touching Qiskit's own layout code) at spare=0: the 6x7
grid (42 physical qubits) admits a **perfect matching** of size 21, and
the circuit requires exactly 21 disjoint pairs -- `matching_slack=0`,
`perfect_matching_exists=True`, `embedding_feasible=True`. A valid,
zero-SWAP embedding is proven to exist at the exact point where
`VF2Layout` reports none can be found.

## 2. Scoring against the pre-registration

**P1** ("spare0/spare2 (L3) >= 20x, and the single largest step-to-step
ratio is at one of the first two steps"): **CONFIRMED, decisively.**
Measured ratio 238.83x (>= 20x). The largest step ratio in the entire
13-point sweep is 193.1x, and it occurs at the very first step
(spare 0->1).

**P2** ("`VF2Layout_stop_reason` is `NO_SOLUTION_FOUND` at spare=0 and
`SOLUTION_FOUND` at the largest spare tested, flipping within one step of
the timing cliff"): **CONFIRMED, exactly.** The stop reason flips at
precisely the same step (0->1) as the largest timing drop -- not merely
"within one step," but the identical step, at both optimization levels.

**P3** ("layout-stage passes >80% of total time at spare=0, and <50% at
the largest spare tested"): **Half confirmed, half falsified.** At
spare=0, L3, the broad layout-pass bucket is 99.8% of total time (comfortably
above 80%). At spare=24, L3, it is **79.3%**, not under 50% -- this half
of P3 is **FALSIFIED**. Digging into why (see Section 3) shows this
falsification is informative rather than a measurement problem: layout
passes remain the single largest cost category throughout the whole
pipeline at L3, at every spare value, simply because nothing else in this
circuit's compilation is more expensive -- not because layout search
keeps struggling once it succeeds instantly. The *absolute* time drops by
two orders of magnitude; the *share* does not, because there is no other
big pass for it to lose share to.

**P4** ("a perfect matching exists at spare=0, so any `NO_SOLUTION_FOUND`
there is a search-budget failure, not infeasibility"): **CONFIRMED,
exactly as predicted** -- `matching_slack=0`, `perfect_matching_exists=True`.
This is the single most load-bearing result in this addendum: it rules
out the possibility that spare=0 is simply an unsolvable instance that
`VF2Layout` is correctly rejecting.

**P5** ("`optimization_level=1` shows no cliff worth the name, under 3x
across the whole sweep"): **CONFIRMED.** L1 ranges from 36.27ms (spare=0)
to 76.91ms (spare=1, its *maximum* -- not spare=0), a 2.12x spread.
Spare=0 is not even the slowest point for L1.

## 3. Correction to this addendum's own pre-registered mechanism story

The pre-registration's motivation section (Section 0) described the
expected mechanism as "`VF2Layout` fails, Qiskit falls back to the slower
`SabreLayout`." **That specific description is wrong, and the data says
so directly**: `sabrelayout_ms` is exactly 0.0 in all 234 rows, at every
spare value and both optimization levels. `SabreLayout` is not what runs,
or if it runs it is not what the callback instrumentation is attributing
meaningful time to.

What the per-pass breakdown actually shows, at spare=0, `optimization_level=3`
(median of 9 runs): `VF2Layout` **8,450.8ms**, `VF2PostLayout` **6,712.3ms**,
everything else combined under 400ms. `slowest_pass` is `VF2PostLayout` in
6 of 9 runs and `VF2Layout` in the other 3, at spare=0 -- versus `VF2Layout`
alone in all 9 of 9 runs at every other spare value tested. In other
words, at saturation there are **two** separate, expensive
subgraph-isomorphism searches paying the cost, not one search followed by
a cheap deterministic fallback: `VF2Layout` itself exhausts its budget
trying to find an initial layout, and **`VF2PostLayout`** -- which runs
afterward to check whether a better final layout exists, using the
already-routed circuit's interaction graph -- pays a comparable or larger
cost doing the same kind of bounded search a second time. This is a
correction to the mechanism description, recorded here per this project's
standing rule against silently rewriting a prediction after seeing data,
not a change to any scored prediction above (P1, P2 and P4 stand as
measured; the motivation text was background reasoning, not itself a
pre-registered numeric claim).

`VF2Layout`'s own median time also differs sharply by optimization level
at spare=0 despite both reporting the identical stop reason: 8,450.8ms at
L3 versus 5.4ms at L1. Qiskit's preset pass managers evidently configure
`VF2Layout`'s search budget (`call_limit` and/or related parameters)
very differently between optimization levels 1 and 3; a failing search
at L1 gives up in single-digit milliseconds, while the L3 configuration
grinds for seconds before giving up on the *same* infeasible-by-budget
instance. This is a plausible complete explanation for P5 (why L1 never
cliffs) that was not part of the original prediction and should be
verified directly (comparing `call_limit` values across optimization
levels) before being treated as established.

## 4. What this addendum does and does not establish

**Established:** on this circuit family and grid, the compile-time cliff
is a single-step event tightly localized to full saturation; it coincides
exactly with `VF2Layout`'s own stop-reason flip; and the spare=0 instance
is provably solvable (perfect matching exists) at the exact point where
Qiskit's bounded search reports it cannot find a solution. Together these
three facts are strong, direct evidence for explanation (A) from the
pre-registration -- budget exhaustion near a feasibility threshold, not a
genuinely infeasible or gradually-scaling phenomenon.

**Also established, as a correction:** the expensive fallback is not
`SabreLayout` as this addendum originally assumed before running; it is
a second VF2-family search inside `VF2PostLayout`.

**Not established:** whether this generalizes beyond one grid size (6x7)
and one circuit family (dense pair blocks). Not established: the actual
`call_limit` (or other budget parameter) values Qiskit's preset pass
managers use at each optimization level -- inferred here only indirectly,
from the size of the time gap between L1's and L3's failing searches, not
read from source or logged directly. Not established: whether
`VF2PostLayout` exposes its own stop-reason-equivalent property that a
future version of this script should also capture (it was not
instrumented in this run).

## 5. Proposed next steps

1. Read Qiskit's preset pass manager source directly to confirm the
   `call_limit` (and any other VF2-related parameter) actually used at
   `optimization_level` 1 vs 3, rather than inferring it from timing.
2. Check whether `VF2PostLayout` exposes a comparable stop-reason
   property in its own `property_set`, and instrument it directly in a
   follow-up run rather than only inferring its cost from `slowest_pass`.
3. Repeat at a second grid size (8x8, following this project's existing
   convention) to check whether the single-step, exact-coincidence result
   here is specific to 6x7 or general.

## 6. Files

| File | What it is |
|---|---|
| [`occupancy_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep.py) | the script (no PSF-Zero dependency) |
| [`occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | this run's raw data (234 rows, 0 errors) |
| [`spare-qubit-cliff-addendum-34-preregistration-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-34-preregistration-2026-09-17.md) | predictions, written before this run |
| this document | the results write-up |

## 7. Verification

- All figures in Sections 1-3 were recomputed directly from the raw CSV
  with pandas (median/groupby, not read off the script's own printed
  summary), cross-checked against the script's own console output where
  both are available, and found to agree (e.g. the printed "layout-stage
  fraction" table and the CSV-derived one match to 3 decimal places).
- The `error` column was empty-string on write but reads back as `NaN`
  through `pandas.read_csv` (a CSV round-trip quirk, not a data problem);
  confirmed 234/234 rows have no error before treating any row as valid.
- Feasibility numbers (Section 1, Section 2 P4) were computed by this
  script's own `matching_feasibility()` using `networkx.max_weight_matching`,
  independent of any Qiskit layout code, and were re-verified here by
  reloading the CSV and reprinting the `matching_slack` column directly
  rather than trusting only the script's own console summary.
- Pre-publication check:
  `grep` against this project's private personal-information pattern list, this document and the CSV -> 0 hits in both. (The terminal
  transcript accompanying this run's upload contained the
  `C:\Users\...\psf_zero_test>` prompt path, as it always does -- per this
  project's standing rule, that raw path was not copied into this
  document or the CSV; only the aggregate numeric results were used.)

---


<!-- ===== Addendum 35 pre-registration (source: spare-qubit-cliff-addendum-35-preregistration-2026-09-17.md) ===== -->

> **Note added when merging:** Predictions for whether TKET's GraphPlacement degrades at the same saturation point as Qiskit's VF2Layout -- the question of whether this series describes a Qiskit bug or a property of the technique. Includes a same-day amendment (P4->P4') made before any full-scale run, after a smoke test exposed a design confound.

## Addendum 35 -- pre-registration: is the saturation cliff Qiskit-specific, or does TKET hit it too? (2026-09-17)

**Written BEFORE `cross_compiler_cliff.py` is run at full scale on any
machine.** Reduced-scale smoke tests in a cloud container were code
validation only; their numbers are not used. Do not edit this document
after seeing full-scale output.

## 0. Why this experiment exists

This is the single highest-value open question for whether Addenda 4-34
amount to a bug report or a finding.

- If only Qiskit cliffs at `spare=0`, the result is: *Qiskit's preset
  passmanager has a pathology*. That is a useful issue report (and is
  already partly filed upstream as #7705 / #8667), but it is a fact about
  one codebase.
- If TKET cliffs too, the result is: *placement by bounded subgraph
  isomorphism degrades as device occupancy approaches 100%*. That is a
  property of the technique, not of an implementation, and it applies to
  every compiler that places qubits this way.

The comparison is meaningful because the two tools use structurally the
same approach at this stage. Qiskit's `VF2Layout` runs a VF2 subgraph
isomorphism search bounded by `call_limit`. TKET's `GraphPlacement` runs a
subgraph-monomorphism search bounded by `maximum_matches` (default 1000)
**and a wall-clock `timeout` (default 1000 ms)**. Both are budgeted
searches for an embedding of the circuit's interaction graph into the
device graph. The budgets differ in kind, and that difference drives P3
below.

## 1. What is being measured

`cross_compiler_cliff.py`, on identical circuits and identical coupling
maps, sweeping spare qubits, with these arms:

- `qiskit_opt1` -- control, expected not to cliff
- `qiskit_opt3` -- the known cliff
- `tket_placement_routing` -- `PlacementPass(GraphPlacement(arch))` then
  `RoutingPass(arch)`, timed **separately** so the placement stage can be
  compared against Qiskit's layout stage rather than against Qiskit's
  whole pipeline
- `tket_default_mapping` -- `DefaultMappingPass(arch)`, TKET's
  out-of-the-box equivalent

Heavy peephole optimisation (`FullPeepholeOptimise`) is deliberately
**off** by default. This experiment is about the placement stage, and
including a full optimisation pass would compare Qiskit's layout search
against TKET's layout search plus unrelated circuit rewriting.

No PSF-Zero code is involved.

## 2. Pre-registered predictions

**P1 (TKET degrades too, in the same place).** TKET's **placement** time at
`spare=0` will be at least **5x** its placement time at `spare=4` on the
same 6x7 grid. The direction matters more than the magnitude here: the
prediction is that the degradation is present and located at saturation,
not that it is as large as Qiskit's.

**P2 (but TKET's total blow-up is far smaller than Qiskit's).** TKET's
total (placement + routing) time at `spare=0` will be **under 3 seconds**,
and Qiskit `optimization_level=3` at the same point will be **at least
20x** TKET's total.

**P3 (and the reason is the shape of the budget, not the quality of the
algorithm).** TKET's placement time at `spare=0` will be **bounded near
its configured timeout** -- with the default 1000 ms it will not exceed
roughly 2x that per placement call. Qiskit's `VF2Layout` has no wall-clock
cap, only a call count, so its cost at `spare=0` is not bounded the same
way. If TKET's placement time at `spare=0` runs to many seconds despite
the timeout, P3 is falsified.

**P4 (quality is not the story).** ~~Output 2-qubit gate counts at
`spare=0` will be within **2x** across all arms.~~ The cliff is a
compile-time phenomenon, not an output-quality one. (Prior addenda
support this for Qiskit; TKET is untested at saturation.)

> **Amended 2026-09-17, BEFORE any full-scale run, after a reduced-scale
> code-validation run exposed a design confound.** On a 3x4 smoke grid,
> `qiskit_opt3` returned 18 two-qubit gates while `qiskit_opt1`,
> `tket_placement_routing` and `tket_default_mapping` all returned 54.
> That 3x spread has nothing to do with saturation: `optimization_level=3`
> resynthesises the circuit, while `optimization_level=1` and a
> placement-plus-routing-only TKET arm do not optimise at all. P4 as
> originally written would therefore have been "falsified" by a fact
> about which arms run an optimiser, not by anything about the cliff.
>
> **P4 is replaced by P4'**: gate counts are compared only between arms
> doing comparable optimisation work -- `qiskit_opt1` against
> `tket_placement_routing` (neither optimises), and `qiskit_opt3` against
> `tket_placement_routing --with-peephole` (both do). Within each of
> those two pairings, 2-qubit gate counts at `spare=0` will be within
> **2x**. Across pairings no prediction is made, and raw cross-arm gate
> counts must not be quoted as a like-for-like quality comparison.
>
> This amendment is recorded rather than applied silently because it
> changes a pre-registered prediction. It was written before any
> full-scale measurement existed; the smoke run that prompted it used a
> 3x4 grid with 3 gates per pair, a size at which no cliff occurs at all
> (VF2Layout reported `solution found` at every point), so no
> cliff-relevant result was visible when this was written.

**P5 (soft).** `tket_default_mapping` will behave qualitatively like
`tket_placement_routing`, within 3x, since `DefaultMappingPass` wraps the
same placement machinery.

## 3. What would falsify the addendum's thesis

- TKET's placement time is flat across the whole sweep, within 2x
  (kills P1). The cliff is then Qiskit-specific, the finding is an issue
  report, and the paper's framing must shrink accordingly. **This outcome
  must be reported as prominently as the positive one.**
- TKET fails outright at `spare=0` (raises, or returns an invalid
  routing). That is a different and also-publishable result, but it is not
  the predicted one and must not be presented as if it were.
- Gate counts diverge wildly between tools at saturation (kills P4), in
  which case compile time is not comparable like-for-like and the
  experiment needs redesigning before it means anything.

## 4. Run command (full scale, on the measurement machine)

```
python cross_compiler_cliff.py --rows 6 --cols 7 \
    --spares 0,1,2,4,8,16 --seeds 3 --repeats 3
```

## 5. Known limitations, stated in advance

- Two tools is not "the field". A negative TKET result does not prove
  every compiler is immune, and a positive one does not prove every
  compiler suffers. BQSKit and Cirq are not covered here; adding them is
  the obvious follow-up.
- TKET's placement budget is configurable. Running only the default
  (`timeout=1000`, `maximum_matches=1000`) measures TKET-as-shipped, not
  TKET-at-its-best. A sweep over the budget would separate "the technique
  degrades" from "the default budget is too small", and should be the next
  experiment if P1 holds.
- Version-pinned: results are specific to the `qiskit` and `pytket`
  versions recorded in the output CSV, and both projects change their
  preset pipelines between releases.

## 6. Files

| File | What it is |
|---|---|
| [`cross_compiler_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/cross_compiler_cliff.py) | the script (no PSF-Zero dependency) |
| `cross_compiler_cliff_<rows>x<cols>_<cpu>_<date>.csv` | its raw output |
| this document | the pre-registered predictions |

---


<!-- ===== Addendum 35 (source: spare-qubit-cliff-addendum-35-2026-09-17.md) ===== -->

> **Note added when merging:** **Answers Addendum 35's motivating question.** TKET's placement stage does slow down at saturation, same direction, same location -- the cliff is not Qiskit-specific. But the severity gap is enormous: ~4x for TKET (narrowly missing the pre-registered 5x bar, reported as a miss) against ~353x for Qiskit at the same grid -- a 54x gap between the two tools' worst case.

## Addendum 35 -- TKET degrades at saturation too, but by ~4x where Qiskit degrades by ~350x: the cliff is not Qiskit-specific, but its severity is (2026-09-17)

**Scored against the pre-registration**:
`spare-qubit-cliff-addendum-35-preregistration-2026-09-17.md` (including
its 2026-09-17 amendment replacing P4 with P4', made before this run).

**No PSF-Zero code was exercised anywhere in this addendum.** Same 6x7
grid and circuit family as Addendum 34, spares 0/1/2/4/8/16, 3 seeds x 3
repeats, four arms (`qiskit_opt1`, `qiskit_opt3`, `tket_default_mapping`,
`tket_placement_routing`), 216 rows, 0 errors. `--with-peephole` was not
used this run, so P4's peephole-matched half is untested (see Section 2).

## 0. In one line

**This is the answer to "is the cliff a Qiskit bug or a property of the
technique."** TKET's placement stage does slow down at full saturation --
direction confirmed -- but the pre-registered 5x threshold was narrowly
missed (measured ~3.97-4.18x), and far more importantly, TKET's *absolute*
worst case at spare=0 is 210ms while Qiskit's is 11,416ms: a 54x gap
between the two tools at the exact point where Qiskit is at its worst.
The cliff is real for both tools -- placement-by-bounded-subgraph-search
degrades under saturation in general -- but it is nowhere near
catastrophic for TKET, which is consistent with the pre-registered
explanation (P3): TKET's `GraphPlacement` is bounded by a wall-clock
timeout as well as a call count, and its budget shape prevents the kind
of unbounded-in-practice blow-up Qiskit's `VF2Layout`/`VF2PostLayout`
pair shows in Addendum 34.

## 1. Results

Median total time (ms) by spare and arm:

| spare | occupancy | qiskit_opt1 | qiskit_opt3 | tket_default_mapping | tket_placement_routing |
|---:|---:|---:|---:|---:|---:|
| 0 | 100.0% | 36.88 | **11,415.71** | 293.04 | 210.19 |
| 1 | 97.6% | 21.06 | 22.47 | 96.55 | 54.93 |
| 2 | 95.2% | 21.67 | 21.61 | 92.62 | 52.27 |
| 4 | 90.5% | 20.22 | 22.11 | 84.68 | 50.25 |
| 8 | 81.0% | 18.31 | 27.23 | 77.07 | 45.35 |
| 16 | 61.9% | 15.56 | 32.34 | 57.05 | 33.50 |

TKET placement time alone (the direct analogue of Qiskit's `VF2Layout`
stage), median ms:

| spare | placement_ms | routing_ms |
|---:|---:|---:|
| 0 | 151.14 | 57.36 |
| 1 | 41.73 | 13.56 |
| 2 | 39.85 | 12.50 |
| 4 | 38.03 | 11.76 |
| 8 | 33.42 | 10.88 |
| 16 | 25.16 | 7.91 |

Coupling-map violations: 0 across all 216 rows, all four arms. 2-qubit
gate counts at spare=0: `qiskit_opt1` 1266, `tket_placement_routing` 1260,
`tket_default_mapping` 1260 (all within 0.5% of each other -- none of
these three arms optimises); `qiskit_opt3` 63 (this arm alone resynthesizes
the circuit, so it is not comparable to the other three on gate count,
per the P4 amendment).

## 2. Scoring against the pre-registration

**P1** ("TKET placement time at spare=0 >= 5x its time at spare=4"):
**NARROWLY FALSIFIED.** Measured ratio, placement time alone:
151.14 / 38.03 = **3.97x**. Using total (placement+routing) time instead:
4.18x (`tket_placement_routing` arm) and 3.46x (`tket_default_mapping`
arm). All three ways of computing it land in the high-3x to low-4x range,
consistently under the pre-registered 5x bar. The direction the
pre-registration cared about most -- "the prediction is that the
degradation is present and located at saturation, not that it is as large
as Qiskit's" -- **does hold**: placement time strictly increases as spare
decreases toward 0 in every arm, with spare=0 the single worst point
everywhere. But the specific falsifiable threshold was missed, and this
must be reported as a miss, not rounded up to a confirmation.

**P2** ("TKET's total time at spare=0 stays under 3 seconds, and Qiskit
`optimization_level=3` at spare=0 is at least 20x TKET's total"):
**CONFIRMED, comfortably.** `tket_placement_routing`: 210.19ms (<< 3,000ms);
`qiskit_opt3`/`tket_placement_routing` = 54.31x (>= 20x).
`tket_default_mapping`: 293.04ms (<< 3,000ms);
`qiskit_opt3`/`tket_default_mapping` = 38.96x (>= 20x). This is the
addendum's headline result: TKET simply never leaves millisecond
territory at this saturation level, on this grid and circuit family.

**P3** ("TKET placement time at spare=0 stays near its configured
timeout, not exceeding ~2x the 1000ms default"): **CONFIRMED, with much
more headroom than predicted.** Measured 151.14ms median (individual
runs ranged 67.8-186.2ms across all 9 seed/repeat combinations) --
nowhere near even the 1000ms timeout itself, let alone the 2000ms
ceiling the prediction allowed. TKET's placement search is not straining
against its budget at all at this problem size; it is simply doing more
work than at higher spare values, comfortably inside budget. This
somewhat undercuts the specific mechanism story (P3 predicted TKET would
be "bounded near its timeout" as the reason it stays fast) -- the data
shows TKET isn't approaching that bound at all here, so the wall-clock
cap's role as *the* protective mechanism is not directly demonstrated by
this run; it may simply be that 6x7/spare=0 is not yet hard enough to
make TKET's search struggle the way it makes `VF2Layout`'s search
struggle. This is worth testing at a larger grid before treating the
"timeout protects TKET" explanation as confirmed rather than merely
consistent with the data.

**P4' (amended 2026-09-17, before this run, after a smoke-test exposed
that raw P4 confounded optimisation level with saturation)**: **Half
confirmed, half untested.** The no-optimisation pairing
(`qiskit_opt1` vs `tket_placement_routing`) gives 1266 vs 1260 two-qubit
gates at spare=0 -- ratio 1.005x, comfortably within the pre-registered
2x. The optimisation-matched pairing (`qiskit_opt3` vs
`tket_placement_routing --with-peephole`) was **not run this session**
(`--with-peephole` was not passed), so that half of P4' is untested, not
falsified -- it should be run before this addendum is treated as
complete on gate-count comparability.

**P5** ("`tket_default_mapping` behaves like `tket_placement_routing`,
within 3x"): **CONFIRMED** at every spare value tested: ratios range
1.39x (spare=0) to 1.77x (spare=2), all comfortably inside 3x.

## 3. What this settles, and what it doesn't

**Settled:** the coupling-map saturation cliff is not unique to Qiskit's
implementation. TKET's placement stage measurably slows down under the
same condition, on the same circuits and coupling map, using a
structurally similar bounded-search technique (`GraphPlacement`). That
generalizes Addendum 34's finding from "a Qiskit pathology" to "a
property that shows up in at least two independent subgraph-isomorphism-based
placement implementations." This is the addendum's most important
contribution and directly answers the "is there novelty here" concern
this line of work was pushed on.

**Also settled:** the two tools' *severity* at saturation is wildly
different -- ~4x for TKET against ~353x for Qiskit (Addendum 34's L3
figure, and this run's own qiskit_opt3 spare0/spare16 = 352.95x from the
script's console summary) -- so "the technique degrades under saturation"
and "the technique catastrophically blows up under saturation" are
separate claims, and only the first generalizes across both tools tested
here. Framing this as "TKET has the same problem as Qiskit" would
overstate the finding; framing it as "bounded subgraph-isomorphism
placement is not saturation-proof in general, though implementations
differ enormously in how badly they cope" is what the data supports.

**Not settled:** whether TKET's relative immunity comes specifically from
its wall-clock timeout (P3's mechanism story), from a smaller default
search space (`maximum_matches=1000`), from implementation differences
unrelated to budget shape, or simply from this problem size not yet being
hard enough to expose a TKET-side cliff the way spare=0 exposes Qiskit's.
Not settled: whether a larger grid (this addendum used only 6x7) pushes
TKET's placement time closer to or past its timeout, which would test
the budget-shape explanation much more sharply than this run did. Not
settled: gate-count comparability under matched optimisation effort
(P4's untested half). Not settled: whether BQSKit or Cirq's placement
stages show the same pattern -- two tools is evidence of "not unique to
one implementation," not evidence of a field-wide law.

## 4. Proposed next steps

1. Re-run at a larger, more saturated grid (8x8 or larger, spare=0) to
   see whether TKET's placement time climbs toward its 1000ms timeout,
   which would directly test P3's mechanism claim rather than leaving it
   merely consistent with the data.
2. Run with `--with-peephole` to complete P4's optimisation-matched gate-count
   comparison.
3. Sweep `--tket-timeout` itself (the script already exposes this as a
   flag) to separate "the technique degrades" from "the default budget is
   too conservative for this circuit size" -- flagged as a follow-up in
   the original pre-registration's Section 5 and still open.
4. Add a third and fourth tool (BQSKit, Cirq) before generalizing beyond
   "not unique to Qiskit" to "a field-wide property."

## 5. Files

| File | What it is |
|---|---|
| [`cross_compiler_cliff.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/cross_compiler_cliff.py) | the script (no PSF-Zero dependency) |
| [`cross_compiler_cliff_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cross_compiler_cliff_6x7_Intel64_Family_6_Model_181_Stepping_0_GenuineIntel_2026-09-17.csv) | this run's raw data (216 rows, 0 errors) |
| [`spare-qubit-cliff-addendum-35-preregistration-2026-09-17.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-35-preregistration-2026-09-17.md) | predictions (including the pre-run P4->P4' amendment), written before this run |
| this document | the results write-up |

## 6. Verification

- All figures in Sections 1-2 were recomputed directly from the raw CSV
  with pandas, not read off the script's own printed summary alone;
  where both are available (e.g. the spare0/spare16 cliff ratios) they
  agree with the console output to 2 decimal places.
- The `error` column reads back as `NaN` rather than empty string through
  `pandas.read_csv` (the same round-trip quirk noted in Addendum 34);
  confirmed 216/216 rows have no error before treating any row as valid.
- `violations` was checked directly per-arm (min/max/sum all 0 across
  216 rows) rather than trusting only the script's own "0 rows with
  non-zero" summary line.
- Pre-publication check:
  `grep` against this project's private personal-information pattern list, this document and the CSV -> 0 hits in both. (As with Addendum
  34, the terminal transcript accompanying this run's upload contained
  the `C:\Users\...\psf_zero_test>` prompt path; that raw path was not
  copied into this document or the CSV.)

---


<!-- ===== Addendum 36 (source: spare-qubit-cliff-combined.md, extracted) ===== -->

> **Note added when merging:** Records an external event: the GitHub issue this project drafted and filed upstream (rustworkx#1679) was closed by a maintainer disputing a claim this project's own top addendum (2026-09-13) had already, independently, corrected before the review happened. No new measurement; a record-keeping entry.

## Addendum 36 (2026-09-17) -- the GitHub issue this project filed upstream (rustworkx#1679) was closed on a claim this project's own Addendum "2026-09-13" had already corrected

**Status note on process**: this is a record-keeping addendum, not a new
measurement. No code was run. It documents an external event (a
maintainer response on a public GitHub issue) and cross-references it
against this project's own existing text.

### 0. In one line

The GitHub issue drafted in `rustworkx-issue-draft.md` (this project's own
document, dated 2026-09-11) was filed upstream as rustworkx#1679 and
closed after rustworkx maintainer @jakelishman disputed one sentence in
it -- "Qiskit's `VF2Layout` and `VF2PostLayout` use this call [i.e.
`rustworkx.vf2_mapping()`]" -- calling it false. That specific sentence
was already, independently, corrected inside this project's own records
before this review happened: "Addendum (2026-09-13)" (the very first
entry in this combined document) and "Addendum (2026-09-14): the preset
disables shuffling, explicitly" both establish, from direct source
reading, that the disputed call path existed only **before** Qiskit PR
#14860 ("Handle VF2 coupling-map shuffling in Rust," merged 2025-09-19)
and does not describe current Qiskit. A dated correction has now been
added directly to `rustworkx-issue-draft.md` itself (2026-09-17), and a
follow-up comment citing PR #14860 and Addendum 4's mechanism has been
drafted for posting to the closed issue.

### 1. What happened, in order

1. 2026-09-11: `rustworkx-issue-draft.md` written, describing the
   `id_order=False` vs. `id_order=True` gap in `rustworkx.vf2_mapping()`
   and, in a "why it matters downstream" section, claiming Qiskit's
   `VF2Layout`/`VF2PostLayout` call this function directly.
2. 2026-09-13/2026-09-14 (already in this document, above): this
   project's own subsequent source-reading found that claim describes a
   Python code path Qiskit removed in PR #14860 (merged 2025-09-19);
   current Qiskit uses a separate Rust function, `vf2_layout_pass`, that
   is not a wrapper around the public `rustworkx.vf2_mapping()` API.
3. The draft was filed upstream as rustworkx#1679 (opened by the
   project's own GitHub persona). Maintainer @jakelishman replied,
   agreeing that VF2 is match-order dependent, but stating of the
   downstream claim: "This is false. Please don't post unverified LLM
   output." The issue was closed.
4. 2026-09-17: cross-referencing the closed issue against this project's
   own already-written correction confirmed the two are the same point --
   this project's Addendum 2026-09-13/4 had already reached, from its own
   source-reading, the same conclusion @jakelishman reached from his. A
   dated correction block was added to the top of `rustworkx-issue-draft.md`
   itself (not just to this combined summary), and a short follow-up
   comment was drafted, citing PR #14860 by number and Addendum 4's
   current-mechanism description, for the user to post to the closed
   issue under their own account.

### 2. What this does and does not establish

**Established:** the maintainer's specific objection was correct about
current Qiskit, and this project's own records already said so, from
independent source-reading, without needing the upstream rejection to
prompt it. The `rustworkx`-level measurement in the issue (the
`id_order=False`/`id_order=True` gap itself) is unaffected by this --
only the downstream "why it matters for Qiskit" claim was wrong, and only
because it described a version of Qiskit that had already been changed
by the time the report was filed.

**Not established, and stated explicitly rather than asserted:** that the
GitHub account which filed rustworkx#1679 is provably this project's own
account. The username pattern is a strong match to the contact persona
used elsewhere in this project's public-facing documents, but a username
match on a public platform is not cryptographic proof of identity, and is
reported here as a strong inference, not a verified fact.

**Not yet done as of this writing:** the drafted follow-up comment has
not been posted to the issue; that is the user's own action, to be taken
under their own GitHub account, not something this project's tooling
does on their behalf.

### 3. Files

| File | What it is |
|---|---|
| [`rustworkx-issue-draft.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/rustworkx-issue-draft.md) | the original draft, now carrying a 2026-09-17 correction block at its top and inline at its "why it matters downstream" section |
| this document | the record of the external event and the cross-reference |

### 4. Verification

- The claim-vs-correction cross-reference in Section 1 was checked by
  re-reading "Addendum (2026-09-13)" and "Addendum (2026-09-14): the
  preset disables shuffling, explicitly" directly (both are earlier in
  this same combined document) rather than from memory of their content.
- Pre-publication check:
  `grep` against this project's private personal-information pattern list, this document and the updated `rustworkx-issue-draft.md` -> 0
  hits in both.
---

# Addendum 37 -- Cirq collapses at exact saturation too, harder than either Qiskit or TKET: the cliff now spans three independently-implemented subgraph-isomorphism placers (2026-09-17)

**Status note on process**: this is a post-hoc analysis, not a
pre-registered one. No predictions document was written before these two
runs. Two predictions *were* stated inside the script's own docstring
before it ran (quoted verbatim in Section 2), and one of them was
falsified -- that is reported below as a miss, not quietly dropped, but
it does not carry the weight a separate pre-registration would.

## 0. In one line

Addendum 35 found the saturation cliff in TKET as well as Qiskit, and
framed it as "bounded subgraph-isomorphism placement is not
saturation-proof in general, but implementations differ enormously in
severity." Cirq -- whose own documentation states its placement tooling
uses `networkx` subgraph-monomorphism routines -- was the obvious third
data point, and it now has one. **At `max_placements=1` (the closest
this API gets to Qiskit's "find one layout" semantics), Cirq's
`cirq.get_placements()` runs in 0.65-2.2 ms at every spare value from 2
to 24, and then fails to return at all at spare=0, hitting a 10-second
hard timeout in 3 of 3 attempts** -- a ratio of **at least 4,493x**, and
in truth unbounded, since the timeout is a floor on the real time, not a
measurement of it. A perfect matching provably exists at that exact point
(independent `networkx` max-matching check, `max_matching_size=21 ==
required=21`, recorded in every row of both CSVs). The cliff is now
observed in three independently-written placers from three different
organizations.

## 1. Two runs, and why both were needed

**Run A -- Cirq's default-ish behaviour (`max_placements=2000`)**:

| spare | qubits | median elapsed | outcome |
|---|---|---|---|
| 0 | 42 | 10,029.6 ms | **timed out** (cap not reached) |
| 2 | 40 | 10,030.4 ms | **timed out** (cap not reached) |
| 4 | 38 | 10,026.5 ms | **timed out** (cap not reached) |
| 6 | 36 | 10,030.7 ms | **timed out** (cap not reached) |
| 8 | 34 | 10,022.4 ms | **timed out** (cap not reached) |
| 16 | 26 | 2,313.6 ms | hit 2000-placement cap |
| 24 | 18 | 1,400.9 ms | hit 2000-placement cap |

This run says almost nothing about the cliff, because
`cirq.get_placements()` does not stop at one solution: reading its source
([`cirq-core/cirq/devices/named_topologies.py`](https://github.com/quantumlib/Cirq/blob/main/cirq-core/cirq/devices/named_topologies.py), fetched directly from
`quantumlib/Cirq`) shows it iterates `subgraph_monomorphisms_iter()` and
**enumerates every distinct placement**, de-duplicating only exact
rotations/reflections that reuse the same device qubits. Comparing that
against Qiskit's `VF2Layout` -- which needs exactly one layout -- is not a
like-for-like comparison of the same task. Run A's five timeouts are
therefore reported as what they are: this API, at its own default-ish
setting, does not complete on this circuit family at these sizes, for
reasons that are mostly about enumeration cost, not about saturation.

**Run B -- one-solution semantics (`max_placements=1`)**, the comparison
that actually matters:

| spare | qubits | median elapsed | outcome |
|---|---|---|---|
| 0 | 42 | 10,026.4 ms | **timed out** (cap not reached) |
| 2 | 40 | 2.232 ms | hit cap |
| 4 | 38 | 2.080 ms | hit cap |
| 6 | 36 | 1.846 ms | hit cap |
| 8 | 34 | 1.626 ms | hit cap |
| 16 | 26 | 1.088 ms | hit cap |
| 24 | 18 | 0.648 ms | hit cap |

**spare=0 / spare=2 ratio: >= 4,493x** (a floor, not a measurement --
spare=0 never finished).

Note what `hit_cap` means here and what it does not. Because Cirq's check
is `if len(dedupe) > max_placements`, a cap of 1 raises only once a
*second* placement is found. So every spare>=2 row above means: **Cirq
found two valid placements in under 2.3 ms**. And the spare=0 row means:
**Cirq did not find even the first one within 10 seconds** (`hit_cap` is
False there -- the cap was never reached). That asymmetry is the finding.

## 2. The script's own stated predictions, and the miss

[`occupancy_sweep_cirq_real.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep_cirq_real.py)'s docstring, written before either run,
predicted:

> at spare=0, the interaction graph nearly fills the device, so there
> should be very few ways to place it (little room to shift or rotate the
> embedding) -- enumeration should be fast. At large spare, the same
> interaction graph can be placed in many more positions within the
> larger empty region -- enumeration could plausibly be *slower*, not
> faster. If this holds, Cirq would show the *opposite* direction of
> cliff from Qiskit's.

**The second half held: enumeration cost does rise with spare** -- Run A
shows spare=16 and 24 reaching the 2000-placement cap in 1.4-2.3 s, i.e.
placements are plentiful when the device is empty, exactly as predicted.

**The first half was wrong, and wrong in the most interesting possible
way.** "Few placements exist at spare=0, so it should be fast" conflated
*how many solutions exist* with *how hard they are to find*. At spare=0
the search does not finish at all -- not because it enumerates many
solutions, but because it cannot locate the first one inside 10 seconds,
on an instance where one provably exists. The prediction's own framing
("Cirq's cliff, if any, runs the opposite direction") is therefore
**falsified**: Cirq's cliff runs in the *same* direction as Qiskit's and
TKET's, at the same point, and is the steepest of the three.

## 3. Where this leaves the cross-compiler picture

| tool | placer | cliff at spare=0 | severity |
|---|---|---|---|
| Qiskit `optimization_level=3` | `VF2Layout` + `VF2PostLayout` | yes | ~193x (Addendum 34), ~353x vs spare=16 (Addendum 35) |
| TKET | `GraphPlacement` | yes | ~4x (Addendum 35) |
| Cirq | `get_placements` (`max_placements=1`) | yes | **>= 4,493x** (this addendum, a floor) |

Three placers, three organizations, three independent implementations of
bounded subgraph-isomorphism-style placement -- all three degrade at
**exactly** full occupancy, on an instance where a valid zero-SWAP
embedding provably exists. Addendum 35's framing survives and
strengthens: the degradation is a property of the technique, and the
severity is a property of the implementation, now with a 1,000x spread
between the mildest (TKET) and the harshest (Cirq) measured so far.

## 4. Limits of this result, stated plainly

- **The cliff's location is bracketed, not pinpointed, for Cirq.** The
  dense-pair circuit family needs an even qubit count, so spare=1 (n=41)
  cannot be tested with it. All that is established is that the collapse
  happens somewhere between spare=2 and spare=0. Qiskit's threshold was
  pinned to the single step spare=1 -> 0 (Addendum 34); Cirq's has not
  been.
- **`max_placements=1` is a proxy for one-solution semantics, not an
  exact match for it.** Cirq raises on the *second* placement, not the
  first. Qiskit's `VF2Layout` genuinely stops at one. The comparison is
  closer than Run A but still not identical, and no attempt was made to
  patch Cirq to stop at one.
- **The 10-second timeout is arbitrary** and was chosen for sweep
  tractability, not from any property of the problem. Every spare=0
  number in this addendum is a floor. Whether the real figure is 30
  seconds, 30 minutes, or non-terminating is unknown.
- **An earlier, weaker measurement of this same question gave the
  opposite impression, and is superseded rather than deleted.** A first
  script ([`occupancy_sweep_cirq.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep_cirq.py)) called
  `GraphMatcher.subgraph_is_monomorphic()` -- existence-only, stopping at
  the first match -- and found **no cliff at all**: 1.65 ms at spare=0
  versus 1.40 ms at spare=2, a ratio of 1.18x. That script did not use
  `cirq` at all (it was written before `cirq` was available in the
  environment) and called `networkx` directly. **Why the same underlying
  library is instant via `subgraph_is_monomorphic()` and does not finish
  in 10 s via `subgraph_monomorphisms_iter()` at the same spare=0 has not
  been investigated**, and is the single most important loose end here:
  until it is, "Cirq collapses at saturation" is a statement about
  `cirq.get_placements()` specifically, not about every way of asking
  `networkx` the same question.
- **Whether this circuit family is a realistic input for
  `cirq.get_placements()` is not established.** That function exists to
  build a candidate list for `RandomDevicePlacer` to sample from; feeding
  it a 42-qubit dense-pair interaction graph may be outside its intended
  use. This addendum shows what happens when you do, not that Cirq users
  routinely do it.
- **The interaction graph is a disjoint union of N/2 unconnected 2-node
  edges** -- maximally symmetric and disconnected, which is a plausible
  worst case for VF2-family search independent of occupancy. Run A's
  timeouts at spare=2 through 8 are consistent with that structure being
  expensive in its own right. Run B separates the two effects (at cap=1,
  spare>=2 is fast, so the structure alone is not the problem), but a
  differently-shaped interaction graph has not been tried.

## 5. Files

| File | What it is |
|---|---|
| [`occupancy_sweep_cirq_real.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep_cirq_real.py) | the script for both runs (real `cirq.get_placements()`, hard subprocess timeout) |
| [`occupancy_sweep_cirq_real_6x7_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_cirq_real_6x7_2026-09-17.csv) | Run A, `max_placements=2000` (21 rows) |
| [`occupancy_sweep_cirq_maxplace1_6x7_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_cirq_maxplace1_6x7_2026-09-17.csv) | Run B, `max_placements=1` (21 rows) |
| [`occupancy_sweep_cirq.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/occupancy_sweep_cirq.py) | the superseded existence-only script (Section 4) |
| [`occupancy_sweep_cirq_6x7_2026-09-17.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/occupancy_sweep_cirq_6x7_2026-09-17.csv) | its data, showing no cliff (63 rows) |

## 6. Verification

- Every figure in Sections 1-2 was recomputed directly from the two CSVs
  (median by spare), not read off the script's printed summary.
- `EmbeddingFeasible=True` and `MaxMatchingSize == RequiredPairs == 21`
  were confirmed present in **every** row of both runs, including every
  timed-out spare=0 row -- so "a solution exists there" is not inferred
  from Addendum 34 but re-established independently in this run's own
  data, by a different algorithm (`nx.max_weight_matching`) from the one
  being timed.
- `cirq.get_placements()`'s enumerate-don't-stop behaviour and its
  `> max_placements` off-by-one were read from Cirq's own source on
  GitHub ([`cirq-core/cirq/devices/named_topologies.py`](https://github.com/quantumlib/Cirq/blob/main/cirq-core/cirq/devices/named_topologies.py)), fetched
  directly, not inferred from documentation prose or from behaviour.
- The subprocess timeout mechanism was tested independently before these
  runs (a deliberately 100-second worker killed at a 2-second timeout,
  and a 0.1-second worker returning normally) to confirm timeouts are
  real kills rather than post-hoc elapsed-time checks -- the flaw that
  made an earlier version of this script hang indefinitely on a 4x4 grid
  and require Ctrl+C.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum and both CSVs -> 0 hits.
---

# Addendum 38 -- prior art check: the field's most comprehensive cross-SDK benchmark (Nature Comput. Sci., 2025) never varies device occupancy (2026-09-17)

**Status note on process**: this is a literature/prior-art addendum, not
a measurement. No code was run. It records what an existing, peer-reviewed
benchmark does and does not cover, so that the claims in Addenda 34-37
can be positioned honestly against published work rather than asserted as
novel on the basis of not having looked.

## 0. In one line

Addenda 34, 35 and 37 established that Qiskit, TKET and Cirq all degrade
sharply at exactly full coupling-map occupancy. Before treating that as a
new observation, the obvious prior-art question is whether the field's
existing benchmarking work already covers it. **Benchpress** (Nation et
al., *Nature Computational Science* 5, 427-435, 2025,
doi:10.1038/s43588-025-00792-y) is the most comprehensive published
cross-SDK benchmark available: 1,066 tests, seven SDKs, circuits up to
930 qubits and O(10^6) two-qubit gates, from the IBM Quantum team.
**Reading its full text: the words "saturation," "occupancy," "VF2" and
"layout" do not appear anywhere in the paper**, and its device
transpilation tests are structured in a way that systematically excludes
the saturated regime -- circuits larger than the 133-qubit target are
skipped rather than tested, and no test varies how full the target device
is while holding the circuit fixed. The occupancy axis this project has
been sweeping is not covered there.

## 1. What Benchpress actually measures

From the paper's own Methods and Results sections:

- **Target device for device-transpilation tests**: `FakeTorino`, a
  snapshot of a 133-qubit IBM Heron system including calibration data.
- **Abstract topologies also tested**: all-to-all, square, heavy-hex,
  linear (predefined `rustworkx` graphs).
- **Metrics**: 2Q gate count, 2Q gate depth, transpilation runtime, plus
  input qubit count, QASM load time, and output operation counts.
- **Timeout**: 3,600 s (1 hour), after which a test is marked FAILED.
- **SDKs**: Braket, BQSKit, Cirq, Qiskit, Qiskit Transpiler Service,
  Staq, Tket.
- **Hardware used for the runs**: AMD 7900, 128 GB, Linux Mint 21.3,
  Python 3.12.

The paper is explicitly framed around scaling in *circuit size* -- "as
quantum computers continue to grow in size, it is imperative that the
associated classical computing costs be evaluated for scalability."

## 2. The specific gap

Two sentences in the paper define the boundary precisely:

> "Twenty-two tests are universally skipped due to insufficient qubit
> count for the target used in device transpilation."

> "Out of 1,054 total tests, 22 are device transpilation tests larger
> than the target Heron device and are SKIPPED regardless of the SDK."

So the closest the suite comes to a full device is: a circuit either fits
comfortably inside 133 qubits, or it does not fit at all and is skipped.
**There is no test in which the same circuit is run against targets of
decreasing spare capacity**, which is the entire axis Addenda 34-37
sweep. A 133-qubit device and a 42-qubit circuit is a 32%-occupancy
instance -- comfortably in the flat region this project's own data shows
(Addendum 34: everything from spare=1 outward is within ~1.3x
step-to-step).

Three further absences, checked by full-text search rather than by
impression:

- **"saturation" / "occupancy" / "fully occupied" / "spare qubit":** zero
  occurrences.
- **"VF2":** zero occurrences. The paper reports transpilation runtime
  as a whole and never attributes it to a specific pass.
- **"layout":** zero occurrences. Routing is discussed (Sabre is named,
  and the paper notes its stochastic component makes gate count and depth
  vary run to run) but the layout stage -- where this project found
  ~99.8% of the time going at saturation (Addendum 34) -- is not broken
  out.

## 3. What this does and does not license

**It does support**: the occupancy axis is not covered by the field's
most comprehensive published cross-SDK benchmark. Addenda 34-37's
measurements are not a re-derivation of something Benchpress already
reported.

**It does not support** "nobody has studied this." A single paper, however
comprehensive, is not the literature. This project has already cited one
independent study (arXiv 2504.15141) that found VF2Layout consuming >99%
of compile time on a 100-qubit circuit -- the same pass, the same order of
dominance, found without varying occupancy. Others may exist. **No
systematic literature search has been performed**, and this addendum
should not be cited as if one had been. The honest statement is: the
obvious place for this to have been covered does not cover it.

**It also does not establish** that the saturated regime matters
practically. Benchpress's choice to test circuits that fit inside the
device is a reasonable reflection of how devices are used today, when
133-qubit hardware and 42-qubit workloads are a normal pairing. The
argument that full occupancy will matter more as workloads grow to fill
devices is plausible and is this project's motivation, but it is an
argument about the future, not a measurement.

## 4. What Benchpress offers this project going forward

Reading it suggests three concrete improvements to this project's own
experimental setup, none of them yet done:

1. **A realistic device target.** Every cliff measurement here has used
   `CouplingMap.from_grid(6, 7)` -- a plain rectangular grid. Benchpress
   uses `FakeTorino` (real 133-qubit Heron topology with calibration
   data) and heavy-hex among its abstract topologies. **Whether the
   cliff reproduces on a heavy-hex or real-device topology, rather than a
   square grid, has not been tested.** Addendum 34's own "proposed next
   steps" called for a second grid size; a second *topology class* is a
   stronger test.
2. **A far more generous timeout.** Benchpress allows 3,600 s before
   declaring failure. This project's Cirq runs (Addendum 37) used 10 s
   and hit it in every spare=0 attempt, making every spare=0 figure a
   floor rather than a measurement. A long-timeout re-run would turn
   ">= 4,493x" into an actual number.
3. **Its methodology note on subprocess timing.** The paper flags that
   "using subprocesses to enforce timing can have adverse effects when
   timing software uses parallel processing" -- relevant directly to the
   `multiprocessing`-based hard timeout added in Addendum 37, which has
   not been checked for this interaction.

## 5. Files

| File | What it is |
|---|---|
| Benchpress paper | Nation, P. D. et al. *Nat. Comput. Sci.* **5**, 427-435 (2025). doi:10.1038/s43588-025-00792-y. Read in full for this addendum; not redistributed here |
| this document | the prior-art record |

## 6. Verification

- All four absence claims ("saturation", "occupancy", "VF2", "layout")
  were checked by full-text extraction and case-insensitive search of the
  complete 10-page PDF, not by reading the abstract or skimming.
- Both quoted sentences in Section 2 were read in their surrounding
  context (Fig. 1 caption and the Results section's transpilation
  paragraph respectively), not lifted from a search-result snippet.
- The device target (`FakeTorino`, 133-qubit Heron), timeout (3,600 s),
  metrics list, and SDK list were all read from the paper's Methods
  section directly.
- Pre-publication check: `grep` against this project's private
  personal-information pattern list, this addendum -> 0 hits.




---

---

**End of Part 3 of 3 (end of document).** Back to [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
