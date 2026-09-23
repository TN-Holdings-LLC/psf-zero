# spare-qubit-cliff: Combined Addenda, Part 2 of 7 (Addendum 17 through Addendum 26)

**Continued from [Part 1](spare-qubit-cliff-combined.md).** Same conventions as Part 1: nothing has been deleted or rewritten; navigation notes added when merging are clearly marked and separate from the original text.

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

---

**End of Part 2 of 7.** Back to [Part 1](spare-qubit-cliff-combined.md), or continue to [Part 3](spare-qubit-cliff-combined-27.md) (Addendum 27-36), [Part 4](spare-qubit-cliff-combined-41.md) (Addendum 41-50), [Part 5](spare-qubit-cliff-combined-51.md) (Addendum 51-87), [Part 6](spare-qubit-cliff-combined-88.md) (Addendum 88-107), [Part 7](spare-qubit-cliff-combined-108.md) (Addendum 108-134) and [Part 8](spare-qubit-cliff-combined-135.md) (Addendum 135 onward).
