> **Archived record — part 3 of 6.**
> This file is one slice of the original PSF-Zero README, kept **verbatim**. It is a
> lab notebook: it contains claims that were later retracted, hypotheses that were
> later refuted, and corrections written directly underneath the text they correct.
> Nothing has been deleted. For what the project currently claims, read the
> [README](../../README.md); for a settled account of one topic, read
> [`docs/findings/`](../findings/).
>
> Section 4 in full. The longest and most-corrected part of the record: three retracted headline numbers, the profiling that found 87% of per-block cost was PSF-Zero verifying its own output, the `verify` split, five runs of `test1_v3.py`, the 50,000-iteration cumulative loops on two machines, the per-iteration npz analyses, the `verify="strict"` hypothesis and its refutation, and the archived-data provenance finding.
>
>
> **One edit was made to this file when it was archived:** file-relative image paths
> were rewritten from `./docs/...` to `../../docs/...` for the new directory depth.
> Root-relative paths (`/docs/...`) were left untouched — they resolve correctly at
> any depth. No wording, number, claim or correction was altered. The 13 substitutions
> are listed in [`README.md`](README.md#the-one-edit).
> Source: README.md lines 273-1469, as of 2026-09-11.

---

### 4. Compile-time scaling

This is the benchmark we'd point a skeptical reader to first — and it's also
the one that took the most rounds to get right. An earlier draft of this
README reported a speedup "from 203x at 15 qubits / 7 blocks down to 4.4x at
1000 qubits / 500 blocks," framed as PSF-Zero's constant-time-per-block
advantage gradually catching up with Qiskit's fixed overhead. That number is
retracted as of this revision: it was built almost entirely out of
measurement artifacts, not real compute-time differences, and once those are
removed the actual result is close to the opposite of what was claimed.

**What was wrong, found one bug at a time by re-running this benchmark on
real hardware after each fix:**

1. `worker_qiskit` called `transpile(circuit, backend=None,
   optimization_level=3)`. With no backend and no `basis_gates`,
   `transpile()` has no target basis, so every `UnitaryGate` in the circuit
   passed straight through completely untouched — confirmed directly
   (`count_ops()` identical before and after). Qiskit's reported time was
   therefore measuring almost no real work, at every scale.
2. Once `basis_gates` was supplied so `transpile()` had something to do, the
   real `compile()`'s own `ConsolidateBlocks(kak_basis_gate=None)` call
   turned out to default to `force_consolidate=False`, which silently fails
   to merge a candidate block made entirely of pre-existing `'unitary'`-named
   nodes — exactly the structure this benchmark's own circuit generator
   produces. This made PSF-Zero resynthesize once per original gate instead
   of once per qubit pair: a confirmed 20x gate-count blowup and 16–23x
   compile-time blowup, first spotted from a real-hardware run where
   PSF-Zero came out roughly 4x *slower* than Qiskit at 1000 qubits — the
   opposite direction from the original claim, and the finding that
   triggered this whole re-investigation.
3. Even with both of those fixed, both benchmark scripts spawn a brand-new
   `multiprocessing.Process` (Windows: `'spawn'`) for every single timed
   measurement, and `transpile()` pays a real, one-time cost the first time
   it runs in a fresh interpreter (building its internal preset
   `PassManager`, loading stage plugins) — confirmed directly at 2.41s
   (cold) vs. 0.07s (warm) for an identical 156-qubit circuit, a 32x
   difference from measurement methodology alone.
4. That warm-up cost isn't unique to Qiskit — `psf_compile.py`'s own first
   call in a fresh process also pays a smaller, but non-zero, cost. A fair
   comparison has to warm up both sides identically before starting the
   timer, not just the one side that happened to look slow.

**With all four fixed** — real `basis_gates`, `force_consolidate=True`, and a
symmetric warm-up call for both `worker_qiskit` and `worker_psf` outside the
timed interval — here is what real hardware, running the real
`psf_zero_core`, actually reports:

| Qubits | Blocks | Qiskit (mean ± sd) | PSF-Zero (mean ± sd) | Qiskit ms/block | PSF-Zero ms/block | Ratio (Qiskit ÷ PSF-Zero) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 15 | 7 | 0.0103s ± 0.0018s | 0.0068s ± 0.0006s | 1.47 | 0.98 | 1.5x — PSF-Zero faster |
| 50 | 25 | 0.0158s ± 0.0024s | 0.0257s ± 0.0070s | 0.63 | 1.03 | 0.61x — PSF-Zero ~1.6x slower |
| 100 | 50 | 0.0220s ± 0.0037s | 0.0360s ± 0.0025s | 0.44 | 0.72 | 0.61x — PSF-Zero ~1.6x slower |
| 156 | 78 | 0.0345s ± 0.0065s | 0.0636s ± 0.0087s | 0.44 | 0.82 | 0.54x — PSF-Zero ~1.9x slower |
| 300\* | 150 | 0.0450s | 0.0893s | 0.30 | 0.60 | 0.50x — PSF-Zero ~2.0x slower |
| 500\* | 250 | 0.0713s | 0.1593s | 0.29 | 0.64 | 0.45x — PSF-Zero ~2.2x slower |
| 1000\* | 500 | 0.1319s | 0.2957s | 0.26 | 0.59 | 0.45x — PSF-Zero ~2.2x slower |

(15–156 qubits: mean ± stdev over 10 seeds, real Windows machine, real
`psf_zero_core`. \*300–1000 qubits: single run each — the "dead zone" scaling
script doesn't loop over seeds the way the 15–156 qubit script does — same
machine and core, so treat these three rows as indicative of the trend
rather than statistically confirmed the way the top four rows are.)

![Compile time scaling, final: verify=False confirmed faster at every scale tested](../../docs/compile_time_scaling_3.png)

The honest picture: PSF-Zero's advantage at the smallest circuit we tested (7
blocks) is real but modest, about 1.5x. Past that, once the timer is
measuring real work on both sides, Qiskit's `optimization_level=3` transpile
is consistently *faster* than PSF-Zero's own Rust-core KAK synthesis, and the
gap **widens** with scale — roughly 1.6x at 25–50 blocks, up to roughly 2.2x
at 500–1000 blocks — which is the opposite trend from the original,
artifact-driven curve. Per-block cost makes the mechanism visible directly:
Qiskit's cost per block falls from ~1.47ms to ~0.26ms as scale grows (its
fixed per-call overhead amortizing over more blocks), while PSF-Zero's holds
roughly flat around 0.6–1.0ms per block and never catches up.

We don't have a confirmed root cause yet for why the constant-time,
no-search KAK path costs more per block than a full `optimization_level=3`
search-based transpile once process-level artifacts are removed, but we have
a strong lead. The "search vs. analytic" framing that motivated this whole
benchmark is itself questionable at the per-block level: Qiskit's own
2-qubit unitary synthesis is *also* an analytic Cartan/KAK decomposition
internally, not a combinatorial search — the search that
`optimization_level=3` actually does lives in circuit-level heuristics
(layout, routing, gate cancellation), not in synthesizing one already-
isolated 2-qubit block. So the premise that PSF-Zero should trivially win
at this specific step because it "skips search" doesn't hold up.

[`benchmarks/profile_synthesize_breakdown.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/profile_synthesize_breakdown.py)
breaks `SU4GeodesicPSFSynthesizer.synthesize()` into its four sub-phases and
times each over 2000 random SU(4) blocks (using `psf_zero_core_stub.py`
in-process, not the real Rust extension over PyO3 — see the script's own
caveat about what that under- and over-states):

| Phase | ms/block | Share |
| :--- | :---: | :---: |
| 1. matrix → list (`u_r`/`u_i`) | 0.003 | 0.3% |
| 2. `geometric_decompose()` itself | 0.038 | 3.3% |
| 3. circuit construction | 0.109 | 9.4% |
| 4. `Operator()` fidelity self-check | 1.002 | **87.0%** |
| **Total** | **1.153** | 100% |

The decomposition itself is a rounding error. 87% of the per-block cost is
phase 4: `synthesize()`'s own unconditional, production-path fidelity
self-check — reconstructing `Operator(qc)` from the freshly-synthesized
2-qubit circuit and comparing it against the target unitary, on *every*
block, every call — which is this project's own documented "no silent
fallback" policy, not the decomposition math. That total (1.15ms/block) also
lands close to section 4's real-machine range (0.6–1.0ms/block), which is
consistent with this being the same mechanism, though the stub's
in-process call likely somewhat understates whatever the real PyO3 FFI
round-trip costs. Since phase 2 (the actual decomposition) is only 3.3% of
the total either way, a higher real-FFI cost would have to be enormous to
change the top-line conclusion: **the per-block cost gap in section 4 looks
like it's coming from PSF-Zero verifying its own output, not from the
decomposition being slow** — still not confirmed against the real Rust
core, so treat this as a strong lead rather than a closed case (see
Roadmap).

What we can say with confidence is that the "PSF-Zero is up to 200x faster
than Qiskit" framing used in earlier drafts of this README does not hold —
that specific number was a measurement artifact, full stop. What follows
below is a different, later finding: a real, smaller, mechanism-backed
speed advantage that only shows up once a specific, currently-optional
production setting is changed.

#### A concrete, testable way to actually earn a real speed advantage back

If 87% of the per-block cost really is `synthesize()` re-verifying its own
output rather than the decomposition itself, the natural design question is:
should that check even run on every production call? The math it's
re-checking has already been extensively validated offline — worst-case
(1-fidelity) = 1.11e-15 over 1000 trials against the real core's math
(`test_geometric_decompose.py`) and 8.88e-16 over 200 trials against the
stub (`test_psf_zero_core_stub.py`), independently reproduced on a second
machine. Re-proving already-proven math on every single call, rather than
during development/CI, is a reasonable default while the math is still
earning trust, but not obviously the right trade-off once it has.

This does **not** mean touching the exception-based degenerate-point
fallback (CNOT, SWAP, iSWAP, identity, ...) — that's a different mechanism
(it's how `synthesize()` finds out a given input needs the CX-basis path at
all) and stays exactly as-is. It's specifically the unconditional
`Operator()` re-verification of every non-degenerate result that's on the
table.

[`benchmarks/profile_synthesize_fast_vs_verified.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/profile_synthesize_fast_vs_verified.py)
first measured dropping that check on the stub core (in-process, N=2000
blocks): an 8.11x speedup on `synthesize()` itself, with correctness checked
out-of-band rather than per-call (worst-case 1-fidelity = 8.88e-16,
identical to today's code). [`benchmarks/compile_optional_verify.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_optional_verify.patch)
turned that into an opt-in `verify: bool = True` flag (default unchanged) so
it could actually be tried against the real core, and
[`benchmarks/psf_compile_prototype_v4.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile_prototype_v4.py) /
[`benchmarks/test_prototype_v4_correctness_and_speed.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_prototype_v4_correctness_and_speed.py)
packaged it for exactly that.

**Confirmed, full 15–1000 qubit sweep, 10 seeds per point, real hardware,
real `psf_zero_core`, `verify=False`:**

This started as a single confirmed data point (156 qubits, one run: 4.98x)
and has now been fully re-measured with the same 10-seed rigor as the rest
of this section — `verify=False` integrated directly into `phase1.py`
(15–156 qubits) and `phase2.py` (156–1000 qubits, which now also has a seed
loop, closing the statistical gap the earlier revision of this table
flagged):

| Qubits | Blocks | Qiskit (mean ± sd) | PSF-Zero, `verify=False` (mean ± sd) | Ratio (Qiskit ÷ PSF-Zero) |
| :---: | :---: | :---: | :---: | :---: |
| 15 | 7 | 0.0091s ± 0.0005s | 0.0018s ± 0.0003s | **5.15x — PSF-Zero faster** |
| 50 | 25 | 0.0188s ± 0.0028s | 0.0047s ± 0.0009s | **3.98x — PSF-Zero faster** |
| 100 | 50 | 0.0229s ± 0.0025s | 0.0066s ± 0.0017s | **3.45x — PSF-Zero faster** |
| 156 (phase1.py) | 78 | 0.0342s ± 0.0054s | 0.0120s ± 0.0026s | **2.86x — PSF-Zero faster** |
| 156 (phase2.py) | 78 | 0.0341s ± 0.0045s | 0.0141s ± 0.0028s | **2.42x — PSF-Zero faster** |
| 300 | 150 | 0.0544s ± 0.0100s | 0.0180s ± 0.0025s | **3.02x — PSF-Zero faster** |
| 500 | 250 | 0.0848s ± 0.0110s | 0.0305s ± 0.0063s | **2.78x — PSF-Zero faster** |
| 1000 | 500 | 0.1670s ± 0.0212s | 0.0656s ± 0.0159s | **2.54x — PSF-Zero faster** |

(mean ± stdev over 10 seeds at every scale; the two 156-qubit rows are two
independent scripts/circuit generators measuring the same scale, kept
separate rather than pooled — they agree to within run-to-run noise, 2.4x
vs. 2.9x.)

![Compile time scaling, corrected: both engines warmed up, real Rust core](../../docs/compile_time_scaling_2.png)

**This is the real, final answer for this section.** PSF-Zero is
genuinely, robustly faster than a fully warmed-up Qiskit `optimization_level=3`
transpile across the entire 15–1000 qubit / 7–500 block range we tested —
by roughly 2.4x–5.2x, largest at the smallest scale and settling to a stable
~2.4x–3x band from 150 blocks up, rather than decaying toward parity (the
original retracted curve's shape) or staying negative (the intermediate
`verify=True` finding above). Correctness was confirmed unaffected at every
step this project checked it (`Operator`-equivalence to the original circuit
and to `verify=True`'s own output).

**The catch, and it matters:** this advantage exists only with `verify=False`
explicitly passed. `compile()`'s and `compile_for_hardware()`'s actual
current default is `verify=True`, which the table earlier in this section
shows is *slower* than Qiskit beyond the smallest circuits (0.5x–0.6x, i.e.
1.6x–2.2x slower). So, as shipped today, a caller who doesn't know to pass
`verify=False` gets the slower behavior — the real, mechanism-backed speed
advantage documented here is currently opt-in, not the out-of-the-box
experience. Whether to flip the *default* to `verify=False` (trusting the
now-extensively-validated decomposition math by default, verifying only in
tests/CI) is a real design decision worth making deliberately, not a change
this README is making on the project's behalf — see Roadmap.

> **Superseded 2026-09-09.** The paragraph above is kept for the record but
> no longer describes the shipped code. `verify` is now
> `Union[bool, str]`: `True` (the default) runs a cheap Rust-core check,
> and `"strict"` runs the old `Operator()` reconstruction that this
> section's profiling found was costing 87% of per-block time. The
> expensive thing the paragraph is warning about is now opt-*in* under a
> different name, not the default. Measured default-path performance is in
> the 2026-09-09 update below.

#### Independent cross-machine confirmation: sustained, repeated compile-time savings

Everything above measures compile time as a single call, averaged over 10
independently-seeded circuits per scale. A natural follow-up question: does
the same advantage hold up under *sustained, repeated* use — e.g. the
compile/execute loop an iterative algorithm (VQE, QAOA parameter search)
would actually run thousands of times — rather than once per seed?

[`benchmarks/test_cumulative_compile_time.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_time.py)
answers this directly: it builds a fresh, independently-seeded 15-qubit /
7-block circuit (the same dense-pair-blocks family as the rest of this
section) 3,000 times in a tight loop, compiling each one with Qiskit L3,
PSF-Zero (`verify=True`, the current default), and PSF-Zero
(`verify=False`), and accumulates the wall-clock time for each path.
Correctness was spot-checked on a 6-qubit version of the same circuit
family beforehand (`Operator`-equivalence on the full 15-qubit circuit
would need a 32768×32768 / 8GB matrix per check — infeasible to do 3,000
times, and not necessary given how extensively this exact code path has
already been checked elsewhere in this project); all three engines
reconstructed to fidelity 1.000000000000.

This was run independently on two separate machines — a cloud sandbox used
while building the script, and, separately, this project's own Windows
machine — with the following results:

| Environment | Qiskit mean | PSF-Zero `verify=True` mean | Ratio | PSF-Zero `verify=False` mean | Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Original 15-qubit point (this section, single-run-per-seed)\* | 9.1–10.3ms | 6.8ms | 1.5x | 1.8ms | 5.15x |
| Cloud sandbox (this script, 3,000-iteration loop) | 13.97ms | 12.66ms | 1.10x | 2.43ms | 5.74x |
| Project Windows machine (this script, 3,000-iteration loop) | 7.79ms | 5.13ms | 1.52x | 1.04ms | **7.50x** |

(\*the 9.1ms figure is this section's final `verify=False` table's own
Qiskit column at 15 qubits; 10.3ms is from the earlier intermediate
`verify=True`-era table above it — the two single-run tables used slightly
different Qiskit measurements at the same nominal scale, itself a small
reminder of run-to-run variance even at 10 seeds.)

![PSF-Zero speedup ratio across three independent environments, verify=True vs verify=False](../../docs/cumulative_compile_time_3000iter_1.png)

Absolute times differ across environments, as expected (different CPUs,
different background load) — but the *ratio* holds in the same range on
every machine tested: roughly 1.1x–1.5x for `verify=True` and roughly
5.1x–7.5x for `verify=False`, with the project's own real machine landing
at the high end of both ranges rather than being an outlier in either
direction. Cumulated over the full 3,000-iteration loop, the Windows run
saved 7.99s (`verify=True`) and 20.26s (`verify=False`) against Qiskit's
23.37s total for the same 3,000 calls. **This "grows linearly, doesn't
diminish" claim held at 3,000 iterations but turned out to be
overstated at higher iteration counts on real, shared hardware — see the
correction immediately below before relying on it.**

#### Correction: the ratio does not hold unconditionally at 10,000+ iterations — and here's why

The same Windows machine was later used to re-run this exact benchmark
(extended with an `--iters` flag,
[`benchmarks/test_cumulative_compile_time.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_time.py))
at 10,000 and 50,000 iterations. The raw numbers looked like a real,
concerning regression:

| Iterations | Qiskit mean | PSF `verify=True` mean | Ratio | PSF `verify=False` mean | Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3,000 | 7.729ms | 4.914ms | 1.57x | 1.045ms | 7.39x |
| 10,000 | 8.611ms | 6.347ms | 1.36x | 1.305ms | 6.60x |
| 50,000 | 10.403ms | 9.825ms | 1.06x | 1.903ms | 5.47x |

Taken at face value, this says the speed advantage shrinks — even
disappears for `verify=True` — the longer the loop runs, which would mean
the earlier "constant per-call difference" framing was wrong. Before
changing anything in `psf_compile.py` over this, we pulled the actual raw
per-iteration timing arrays that produced this table and looked directly
at *where* the time went, rather than just the aggregate:

- Binned into 10 equal segments, the slowdown is not a smooth drift. In
  the 50,000-iteration run, most bins sit at 8–10ms (Qiskit) / matching
  the 3,000-run's pace, one stretch spikes to 15.2ms, and the *final* bin
  (iterations 45,000–50,000) drops back to 8.1ms — as good as the very
  first bin. A genuine per-call cost inside the compiler (a leak, a
  growing cache) cannot recover like that; something external turning on
  and back off again can.
- At the exact iterations where Qiskit's call was anomalously slow (its
  slowest 1%), PSF-Zero's calls at those *same* iterations were also
  roughly 2.5–2.7x slower than PSF-Zero's own overall average — despite
  Qiskit and PSF-Zero being completely independent code paths measured
  back-to-back. That correlation (r≈0.64 between the two engines'
  per-call times) is the signature of a shared, external, per-iteration
  cause — something on the machine competing for the CPU/disk at that
  moment — not a property of either algorithm.
- We reproduced the identical benchmark, against the same compiled
  `psf_zero_core`, in a clean cloud sandbox with no antivirus or other
  background services, for 5,000 iterations, logging process RSS and the
  live Python object count every 1,000 calls. Result: the ratio was flat
  across all 10 bins (1.18x–1.24x for `verify=True`, 6.00x–6.33x for
  `verify=False`, no trend either direction), and RSS stayed at
  233–240MB and object counts at 181k–194k throughout — no growth in
  either.

> **Correction (2026-09-10): the raw arrays behind this bullet list have
> since been recovered, and two of its numbers need qualifying.** The
> "r≈0.64" figure is exact — the per-iteration Pearson correlation in that
> run is **+0.644** (Qiskit vs `verify=True`). The "roughly 2.5–2.7x"
> figure is not wrong but is not a matched comparison: it divides
> PSF-Zero's *mean* at Qiskit's slowest 1% (17.45ms / 3.28ms) by
> PSF-Zero's *median* over the whole run. Mean-against-mean gives
> 1.78x / 1.72x and median-against-median gives 3.20x / 3.27x. The
> qualitative claim survives under every pairing — PSF-Zero is markedly
> slower at exactly those iterations, by somewhere between 1.7x and 3.3x
> depending on the statistic — but the specific figure should be quoted
> with the statistic it came from. Full analysis in the 2026-09-10 update
> below.

**Conclusion:** the ratio compression seen at 10,000/50,000 iterations is
not a memory leak or an algorithmic scaling problem in `psf_compile.py` or
`psf_zero_core` — both stayed flat under controlled conditions. It tracks
with episodic background CPU/disk contention on that specific Windows
machine (real-time antivirus scanning, OS housekeeping, or thermal
throttling are the ordinary causes of a multi-minute, fully-loaded process
slowing down and recovering mid-run on consumer/laptop hardware — we did
not instrument which one specifically). Because PSF-Zero's own per-call
time is much smaller than Qiskit's, the same absolute external slowdown
erodes its *ratio* far more visibly than Qiskit's, even though both
engines are affected by the same underlying events at the same moments —
so short runs (3,000 iterations, ~1 minute) are far less exposed to this
than the 50,000-iteration run's ~40 minutes of continuous full-CPU
execution was. The corrected claim: the per-call speed advantage is
constant *in an uncontested environment*, confirmed directly; on shared
real-world hardware running for many minutes, expect the measured ratio
to be a noisy lower bound on that, not a fixed number — report medians
over single long runs, or repeat short runs, rather than trusting one
very long run's mean. No code change follows from this — it's a
measurement-environment finding, not a `psf_compile.py` bug.

**What this does not show:** faster compilation does not, by itself, mean
higher measured fidelity on real hardware for a given circuit — the two
circuits in section 7's real-device comparison, for instance, were
submitted together in one batched job, so both experienced identical
hardware conditions regardless of how fast either was compiled beforehand.
The place this compile-time advantage would actually matter is the total
wall-clock cost of a workflow that has to compile *repeatedly* — more
iterations completed per unit of session time, not a fidelity boost on any
single circuit. We have not tested that specific claim (an iterative
real-hardware session, where fewer total wall-clock minutes could plausibly
mean less exposure to calibration drift across the run) and are not
planning to spend real QPU time confirming it without a specific reason to
— see Roadmap.

#### Update (2026-09-09): the ratio holds flat at 50,000 iterations, and the default path got 2.2x faster

The correction above concluded that the ratio compression at 10,000/50,000
iterations was episodic background contention on that Windows machine, not
anything inside `psf_compile.py`. That conclusion was reached from binned
timings and a clean-sandbox reproduction; it had not been tested by simply
re-running the same 50,000-iteration loop on the same machine on a quieter
day. It has now been, and it holds:

| Iterations | Qiskit L3 mean | PSF `verify=True` mean | Ratio | PSF `verify=False` mean | Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3,000 (2026-09-09) | 10.503ms | 2.203ms | **4.77x** | 1.172ms | **8.96x** |
| 50,000 (2026-09-09) | 11.038ms | 2.303ms | **4.79x** | 1.211ms | **9.12x** |
| *3,000 (earlier run, for contrast)* | *7.729ms* | *4.914ms* | *1.57x* | *1.045ms* | *7.39x* |
| *50,000 (earlier run, for contrast)* | *10.403ms* | *9.825ms* | *1.06x* | *1.903ms* | *5.47x* |

Two separate things changed between the italicised earlier rows and the new
ones, and they should not be confused with each other.

**1. The decay is gone, and the raw progress log shows exactly why.** The
new 50,000-iteration run printed elapsed time every 500 iterations.
Differencing those: the loop holds 18.0–18.7s per 500 iterations for
essentially its entire length, rises to 22.9s / 26.1s / 20.9s / 21.9s /
21.0s / 20.9s across iterations 23,500–27,000, then returns to 18.5–18.7s
and stays there for the remaining 23,000 iterations. One contention window,
full recovery, no drift — the same signature the correction above inferred
indirectly, now visible directly in a single run's own progress output. The
`verify=False` mean moved 1.172ms → 1.211ms between the 3,000- and
50,000-iteration runs, a 3% difference, against 82% in the earlier
contended run. **The earlier "ratio compresses with iteration count"
observation is now confidently attributable to the environment, not to this
code.**

**2. `verify=True` is 2.2x faster than it was** (4.914ms → 2.203ms at
3,000 iterations), because it is no longer the same operation. It now runs
the cheap Rust-core check; the old `Operator()` reconstruction moved to
`verify="strict"`. This is the change that obsoletes this section's
"The catch, and it matters" paragraph above: the default is now 4.8x faster
than Qiskit rather than 1.1–1.6x, and the cost of keeping the safety net on
is about 1.9x rather than 4–5x.

Correctness was re-checked at the start of both runs on the 6-qubit version
of the same circuit family: Qiskit, `verify=True` and `verify=False` all
reconstructed to fidelity 1.000000000000.

#### Update (2026-09-09): under the corrected methodology the ratios are lower — and the cause is narrower than "Windows," not "which PC"

`benchmarks/test1_v3.py` is this project's methodology-corrected replacement
for the script that produced section 4's tables (warm-up outside the timer,
repeated timed calls per point, in-child memory sampling, `spawn` forced,
per-arm equivalence checking, and output quality recorded alongside every
timing — the five defects it fixes are catalogued in this project's
`benchmark-methodology-v3.md` note). Running it at 10 seeds per point on
**this project's faster machine** (the faster of the project's two
machines, not the slower one used for section 5's 2026-09-09 confirmation
below) gives materially lower ratios than either section 4's tables above
or the same script's own run in a Linux cloud sandbox:

| Qubits | qiskit opt=0 | qiskit opt=3 | psf canonical | psf cx | opt=3 ÷ canonical (fast PC) | *same ratio, Linux sandbox* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 15 | 3.62ms | 8.76ms | 2.02ms | 2.08ms | **4.35x** | *5.53x* |
| 50 | 7.88ms | 14.20ms | 6.44ms | 6.80ms | **2.21x** | *4.22x* |
| 100 | 12.69ms | 21.33ms | 12.84ms | 13.52ms | **1.66x** | *3.95x* |
| 156 | 17.97ms | 28.30ms | 19.69ms | 20.41ms | **1.44x** | *4.15x* |

(median of per-point medians over 10 seeds. Output quality, identical at
every scale in both environments: 2-qubit gate count 21/75/150/234 for
`opt=3`, `canonical` and `cx` alike — `opt=0` emits 20x more and is not a
quality-matched comparison — and depth **9** for canonical, **13** for cx,
**16** for Qiskit `opt=3`. Equivalence checked at 6 qubits, all arms
< 4.5e-15.)

Four things worth stating plainly about this table, one of which changed
after this section was first written.

**The obvious hypothesis — "the Windows number is just measuring a slower
machine" — is ruled out, and by evidence already in this README.** The
2026-09-09 cumulative-loop update directly above this one, run on this
*same* fast machine, showed the default-verify ratio holding flat at 4.79x
across 3,000 and 50,000 iterations — not the declining, sub-Linux-sandbox
pattern in the table here. Two runs, same hardware, same day: one gives a
stable ratio in the range this project considers healthy, the other gives a
declining one well below it. Raw machine speed cannot be the variable that
changed between them — something about the *scripts* differs.

**The one thing that does differ between those two scripts is the memory
sampler, which makes it the leading suspect rather than a speculative one.**
`test1_v3.py` runs a background thread inside the timed child process
sampling RSS at a requested 0.2ms interval; `test_cumulative_compile_time.py`
does no per-call sampling at all. Windows' default timer granularity is
15.6ms, so a 0.2ms sleep request is not honoured the way it is on Linux, and
the sample counts in `test1_v3.py`'s own run log are consistent with the
thread spinning rather than sleeping (28 samples inside a ~2ms measurement —
finer than the interval requested). A spinning Python thread contends for
the GIL, and PSF holds the GIL for a larger share of its work than Qiskit
does, so this would inflate the PSF arms specifically — which is exactly
the direction of the effect seen (Qiskit is *faster* on this machine than
on the Linux sandbox at 156 qubits, 17.97ms vs. 32.61ms; PSF canonical is
*slower*, 19.69ms vs. 11.47ms). Also consistent: `canonical` and `cx` are
1.75x apart on Linux (11.47 vs. 20.04ms, as expected — `cx` does strictly
more work) but essentially tied on this machine (19.69 vs. 20.41ms), which
is what sampler-driven GIL contention swamping the real difference would
look like.

**This is now a strong lead, not a confirmed cause — the confirming
experiment is cheap and has not been run.** Re-run `test1_v3.py` on the
same fast machine with the RSS sampler disabled (`--mem-calls 0` or
equivalent) and check whether the ratio recovers toward the Linux sandbox's
4.0–5.5x. If it does, the fix is a sampler that respects Windows' timer
granularity instead of a claim about "Windows being slower." If it doesn't,
the ratio really is machine- or OS-dependent for a reason not yet found, and
this README's speedup claims need an explicit range rather than a headline
number. **Until that one run happens, treat this table's ratio column as a
lower bound, not a representative number.**

> **Correction (2026-09-10): the paragraph above is retracted. We finally
> read `test1_v3.py`'s actual source, and the mechanism it describes does
> not exist in that script.** The claim was that a 0.2ms RSS-sampler thread
> spins during the timed calls and contends for the GIL, penalising the PSF
> arms specifically. The source shows otherwise: the timed repetitions
> (`for _ in range(reps): ... times.append(...)`) run with **no sampler
> thread active at all** — the script's own change-log comment says so
> directly ("v3 samples RSS from a thread inside the child ... during a
> **dedicated (untimed) pass**"). The `RssSampler` is only started *after*
> timing is finished, in a separate block whose entire purpose is measuring
> memory, not compile time. Whatever caused the declining ratio in the
> table above, it cannot be sampler-driven GIL contention during
> measurement, because no sampler runs during measurement. We should have
> read the script before proposing a mechanism for what it does — this
> project's own stated discipline ("check the fixture, not just the code")
> applies to reading our own diagnostic scripts too, and we didn't follow
> it here. **The real cause of the decline in the table above is open
> again, with no candidate mechanism.** The Update immediately below was
> written under the retracted hypothesis; read it as a data point only, not
> as evidence for a mechanism that isn't real.

**The depth result is not affected by any of this and is the most robust
thing in the table.** Depth 9 / 13 / 16 reproduced exactly, at every scale,
in both environments, on 10 independent seeds — as it should, given the
decomposition is deterministic. At hardware-comparable basis
(`entangling_basis="cx"`), PSF-Zero produces the same 2-qubit gate count as
Qiskit `optimization_level=3` at **depth 13 vs. 16**, and did so faster in
both environments. That claim does not depend on which timing column you
believe.

#### Update (2026-09-10): the exact scripts that produced this section's own tables, re-run on a second machine, reproduce cleanly — a data point only, since the correction above retracts the mechanism this was originally framed around

> **Note:** this update was drafted earlier the same day as the Correction
> above, under the sampler hypothesis that correction retracts. The data
> below is unchanged and still real; only the interpretation ("supporting
> evidence for the sampler hypothesis") no longer holds, since that
> mechanism doesn't exist in the script. Read this as: two scripts with
> different memory-measurement designs gave different ratios, on two
> different machines — a correlation with two things changing at once
> (script and machine), not an isolated variable.

The update directly above flagged that `test1_v3.py` gives declining,
below-sandbox ratios on the project's *faster* machine. A separate, useful
data point: `phase1.py` and
`phase2.py` — the exact fully-patched scripts (symmetric warm-up,
`verify=False`, 10-seed loop) that produced this section's own final
15–1000 qubit table above — were re-run end to end on the project's
*slower* machine, against a freshly-built, real `psf_zero_core` wheel.
Unlike `test1_v3.py`, neither script polls memory on a sub-millisecond
timer: `phase1.py`'s peak-RSS monitor polls the child process every 100ms
(`time.sleep(0.1)`, comfortably above Windows' 15.6ms timer granularity),
and `phase2.py` does no per-call memory sampling at all.

15–156 qubits (`phase1.py`, median of 10 seeds):

| Qubits | Qiskit median | PSF-Zero median | Ratio |
| :---: | :---: | :---: | :---: |
| 15 | 10.23ms | 1.08ms | **9.51x** |
| 50 | 16.05ms | 2.60ms | **6.16x** |
| 100 | 22.34ms | 5.10ms | **4.38x** |
| 156 | 29.59ms | 8.84ms | **3.35x** |

156–1000 qubits (`phase2.py`, mean ± sd of 10 seeds):

| Qubits | Qiskit mean ± sd | PSF-Zero mean ± sd | Ratio |
| :---: | :---: | :---: | :---: |
| 156 | 29.85 ± 2.54ms | 10.08 ± 1.72ms | **2.96x** |
| 300 | 46.46 ± 2.71ms | 15.94 ± 1.54ms | **2.91x** |
| 500 | 72.68 ± 3.21ms | 25.15 ± 1.52ms | **2.89x** |
| 1000 | 136.58 ± 2.53ms | 49.06 ± 2.83ms | **2.78x** |

The `phase2.py` ratios land within a few percent of this section's own
published table above (2.42x/3.02x/2.78x/2.54x at the same four scales) —
close reproduction of the existing claim, on different hardware. The
`phase1.py` ratios are, if anything, healthier than either the existing
table or the Linux-sandbox reference cited in the update above (9.51x down
to 3.35x, versus the sandbox's 5.53x down to 4.15x).

This was measured on the *slower*, noisier of the project's two machines —
the one that, on a different workload (section 5's `phase3_v4.py`, with its
saturated-grid Qiskit instability), has shown the *worst* variance of
anywhere in this project. If "the slower machine" by itself explained
`test1_v3.py`'s decay, this run should have shown the same pattern or
worse. It didn't. That observation stands on its own (it doesn't depend on
the sampler mechanism above, which is retracted) — but it doesn't identify
what *does* explain the original decline, either. It's ruling something
out, not confirming a replacement.

#### Update (2026-09-10): `test1_v3.py` re-run on a third machine (distinct from both machines named elsewhere in this section) — healthy ratios again, and the sampler's presence in the script is now known to be irrelevant either way

With the sampler mechanism retracted (see the Correction above), the
question of *why* the original `test1_v3.py` table declined has no
candidate cause left standing. A fresh run of the identical, unmodified
`test1_v3.py`, on a machine distinct from both the one that produced the
original declining table and the slower machine in the update directly
above, gave:

| Qubits | qiskit opt3 median | psf canonical median | Ratio (median) | Ratio (min-of-samples) |
| :---: | :---: | :---: | :---: | :---: |
| 15 | 6.15ms | 0.80ms | **7.65x** | 6.81x |
| 50 | 10.06ms | 2.37ms | **4.24x** | 4.09x |
| 100 | 15.90ms | 4.78ms | **3.33x** | 3.35x |
| 156 | 22.87ms | 8.16ms | **2.8x** | 2.9x |

(median of per-point medians over 10 seeds, 5 reps per point; equivalence
checked at 6 qubits, all four arms < 4.5e-15; depth reproduced exactly at
9/9/9/9 for `psf_canonical` across all four scales, matching every other
run of this circuit family in this README.)

These ratios are healthy — close to or above the Linux-sandbox reference
(5.53x/4.22x/3.95x/4.15x) — on a run of the exact same script whose RSS
sampler thread was, per the run log, still reporting high sample counts in
small time windows (e.g. 94 samples inside a 0.8ms `psf_canonical` call at
15 qubits). That the sampler was evidently still active and still shows the
same "more samples than the requested interval should allow" signature,
*and* the ratio came out healthy anyway, is a second, independent
confirmation (beyond reading the source) that the sampler's behavior does
not track with the ratio outcome — consistent with the Correction above,
which already explains why: the sampler doesn't run during the timed
calls, so nothing about it can affect them either way.

**What remains unresolved:** whether this machine is the same one that
produced the original declining table, a different machine, or possibly
the same machine as the "slower, noisier" one referenced elsewhere in this
section under a different account, is not yet established with certainty —
this project has already had to correct its own machine attributions more
than once (see `publication-policy.md`), and a shared account name across
physical machines was discovered to be part of the problem. Until the
provenance of the *original* declining-ratio run is pinned down, the honest
summary is: `test1_v3.py` has now been run three times across this
project's history, giving 4.35x→1.44x once and two independent
healthy runs (this one, and the Linux sandbox) — and no mechanism
explains the one outlier. Treat the 4.35x→1.44x table as an unexplained
single run, not as this script's typical behavior, until it either
reproduces again or a real cause is found.

#### Update (2026-09-10): a fourth run of `test1_v3.py`, and the original declining table now has a quantitative account

`test1_v3.py` was run again, unmodified, on the same machine as the update
directly above (CPU signature `Intel64 Family 6 Model 181`), 10 seeds × 5
reps per point:

| Qubits | qiskit opt=0 | qiskit opt=3 | psf canonical | psf cx | opt=3 ÷ canonical |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 15 | 2.62ms | 6.54ms | 0.81ms | 1.25ms | **8.07x** |
| 50 | 6.06ms | 9.52ms | 2.37ms | 4.00ms | **4.02x** |
| 100 | 9.23ms | 15.55ms | 4.65ms | 7.63ms | **3.34x** |
| 156 | 14.06ms | 23.06ms | 7.42ms | 12.33ms | **3.11x** |

(median of per-point medians over 10 seeds; min-of-samples gives
7.29x/4.03x/3.24x/2.91x. 2-qubit gate count 21/75/150/234 and depth
9/13/16 for canonical/cx/opt=3 reproduced exactly, as in every run of this
script. Equivalence at 6 qubits, all arms < 4.5e-15.)

All four runs of this script side by side:

| Run | 15q | 50q | 100q | 156q |
| :--- | :---: | :---: | :---: | :---: |
| original (the unexplained outlier) | 4.35x | 2.21x | 1.66x | 1.44x |
| Linux sandbox | 5.53x | 4.22x | 3.95x | 4.15x |
| Intel machine, run 1 | 7.65x | 4.24x | 3.33x | 2.80x |
| Intel machine, run 2 | 8.07x | 4.02x | 3.34x | 3.11x |
| Intel machine, run 3 (added 2026-09-10) | 8.03x | 4.10x | 3.42x | 3.05x |

**The script is stable on a given machine.** Runs 1 and 2 on the Intel
machine agree to within 4–11% at every scale. Whatever produced the
original table, it is not this script being erratic.

**And the "decline with scale" was never the anomaly.** Every run declines
with scale — 8.07→3.11 here, 7.65→2.80 in run 1, 5.53→4.15 in the sandbox.
Comparing the original run's *absolute* times against this one, arm by arm:

| | 15q | 50q | 100q | 156q |
| :--- | :---: | :---: | :---: | :---: |
| original ÷ this run, `qiskit opt=3` | 1.34x | 1.49x | 1.37x | 1.23x |
| original ÷ this run, `psf canonical` | 2.49x | 2.72x | 2.76x | 2.65x |

Both rows are flat. The original run was not degrading as circuits grew: it
was paying **two different constant penalties** — roughly 1.35x on the
Qiskit arm and roughly 2.65x on the PSF arm — at every scale alike. Its
ratio column is those two divided, so it has the same shape as every other
run, uniformly scaled by about 0.51. (That division is an identity, not a
prediction; what is *not* an identity, and is the actual finding, is that
each penalty is scale-independent.)

So the open question changes from "why did the ratio decline?" — it didn't,
any more than usual — to "why was the PSF arm penalised about twice as hard
as the Qiskit arm, uniformly?" There is a candidate that fits the size,
built entirely from measurements already in this README:

- The ~1.35x on the Qiskit arm is what a slower machine looks like, and it
  matches the measured gap between this project's two machines on the dense
  workload in section 5 (`qiskit opt=1`, the arm least entangled with
  anything PSF-specific: 1.25x–1.37x).
- The *extra* ~1.95x on the PSF arm is the size of the `verify` change made
  on 2026-09-09. Before it, `verify=True` ran the `Operator()`
  reconstruction this section profiled at 87% of per-block cost; after it,
  a cheap Rust-core check. The cumulative-loop update above measured that
  swap at 4.914ms → 2.203ms (2.23x) and describes the remaining check as
  costing "about 1.9x rather than 4–5x". 1.95x sits inside that.

**So the original declining table is quantitatively consistent with the
slower machine running a pre-2026-09-09 `psf_compile.py`** — an older
build, not a property of the script, of Windows, or of a mystery machine.
This is an account that fits, **not a confirmed diagnosis**: nobody
recorded which `psf_compile.py` that run used, and the arithmetic cannot
distinguish "the old verify path" from any other cause that costs the PSF
arm ~2x uniformly and the Qiskit arm nothing.

**The confirming experiment is to re-run `test1_v3.py` with the PSF arms at
`verify="strict"`.** If this account is right, the ratios should fall from
~8.0/4.1/3.4/3.1 to roughly the original's 4.35/2.21/1.66/1.44. If they
don't, the account is wrong and the original run goes back to being
unexplained.

> **Correction (2026-09-10): an earlier revision of this paragraph called
> that "one flag away". It is not — `test1_v3.py` has no `--verify`
> option** (its arguments are `--qubits`, `--seeds`, `--reps`, `--arms`,
> `--gates-per-pair`, `--check-qubits`, `--out`, `--quick`), and passing
> one is an argparse error. The experiment needs an arm added, not a flag
> set. `benchmarks/test1_v3_verify_strict.py` does that without modifying
> `test1_v3.py`: it imports the module and registers
> `psf_canonical_strict` / `psf_cx_strict` into its `ARMS` table, so the
> current `verify=True` arm and the `verify="strict"` arm are measured
> **in the same run, on the same seeds, in the same process conditions** —
> a paired comparison rather than a cross-run one, which is stronger than
> what the original wording proposed. The wrapper's plumbing was verified
> end-to-end against a stub core (arms register, propagate to the `spawn`
> child processes, pass the equivalence check, and reach the CSV); the
> ratios it will produce against the real Rust core are of course still
> unmeasured, which is the point of running it.

#### Update (2026-09-10): the confirming experiment has now been run, and it REFUTES the account above

`benchmarks/test1_v3_verify_strict.py` was run on the Intel machine, 10
seeds × 5 reps, with `qiskit_opt3`, `psf_canonical` (the current
`verify=True` cheap core check) and `psf_canonical_strict`
(`verify="strict"`, the old `Operator()` reconstruction) measured **in the
same run, on the same seeds** — the paired comparison the account needed.

Compile time, median of per-point medians (ms):

| Qubits | qiskit opt=3 | psf canonical | psf canonical **strict** | opt3 ÷ strict | *the account predicted* |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 15 | 6.12 | 0.84 | 4.32 | **1.42x** | *4.35x* |
| 50 | 9.81 | 2.42 | 14.70 | **0.67x** | *2.21x* |
| 100 | 15.69 | 4.94 | 32.36 | **0.48x** | *1.66x* |
| 156 | 23.49 | 7.96 | 50.10 | **0.47x** | *1.44x* |

(min-of-samples gives 1.21x/0.65x/0.51x/0.46x — same picture. Output
unchanged by the flag, as it must be: 2-qubit gates 21/75/150/234 and depth
9 for both PSF arms, equivalence at 6 qubits < 4.5e-15.)

**The prediction was stated in advance and it missed, in the same direction
at every scale.** The account required `verify="strict"` to slow the PSF
arm by about 2.5–2.8x relative to a current run, landing it at
2.02/6.45/12.83/19.66ms. It actually slows it by 5.3–7.0x, landing at
4.32/14.70/32.36/50.10ms — roughly 2.1x–2.5x too slow, which puts the
ratios about 3.1x below the original table rather than on top of it.
**So the original declining run was not a pre-2026-09-09 `psf_compile.py`
running the old verify path.** That was the only candidate mechanism on the
table, and combined with the provenance finding above — the original run's
own output file and environment record no longer exist — the honest
disposition is that the 4.35x→1.44x table is **unexplained and now
permanently unexplainable**, not merely unexplained-so-far. It should be
read as a single anomalous run and nothing more. Five later runs of this
script (one Linux sandbox, three Intel, and this one) all sit in the
2.8x–8.1x band.

**Two things worth keeping from the experiment even though its hypothesis
failed.**

*First, a number this README did not previously have: what the strict
safety net actually costs, across scale.* `verify="strict"` costs
**5.1x/6.1x/6.6x/6.3x** the current default at 15/50/100/156 qubits. That
is large enough to invert the headline comparison — with strict on,
PSF-Zero is **slower than Qiskit `optimization_level=3`** at every scale
above 15 qubits (0.67x/0.48x/0.47x). Anyone who wants the old
reconstruct-and-check behaviour should know they are trading away the
entire speed advantage and then some, not a fraction of it.

*Second, a discrepancy this raises about `verify="strict"` itself.* This
section's 2026-09-09 update measured the pre-change default at 4.914ms
against `verify=False`'s 1.045ms, i.e. the old path cost about 4.7x
`verify=False`, and the current default about 1.9x. If `verify="strict"`
were simply the old default restored, it should cost about 4.7/1.9 ≈ 2.5x
the current default. It costs 5.1x–6.6x. So either `verify="strict"` today
is doing more work than the pre-2026-09-09 default did, or one of those two
measurements is not comparable to the other (different scripts, different
circuit sizes, different machines). We have not chased this down, and it is
not load-bearing for anything published here — but it does mean
`verify="strict"` should not be described as "the old default, still
available" without checking that claim first.

> **Correction (2026-09-10): the arithmetic in the paragraph above used a
> machine-mismatched figure, and the gap it reports is roughly half what it
> says.** It took "the current default costs about 1.9x `verify=False`"
> from a run on a *different* machine and applied it to a `strict`
> measurement taken on the Intel machine. A 50,000-iteration cumulative-loop
> run on the Intel machine itself (see the update below) puts that ratio at
> **1.22x, not 1.9x** — the cost of the cheap check is itself
> machine-dependent. Redone with machine-matched numbers: on the Intel
> machine `strict ÷ verify=True` is 5.14x at 15 qubits and
> `verify=True ÷ verify=False` is 1.22x, so `strict ÷ verify=False` is
> about **6.3x** against the pre-change default's **4.7x**. The overshoot is
> therefore about **1.3x, not 2.5x**. That is small enough to be explained
> by the remaining mismatches (the 4.7x is still from the other machine, a
> different script, and a different measurement style), so the honest
> statement is weaker than the one above: **there is no established
> discrepancy here, only an unverified equivalence.** `verify="strict"` may
> well be the old default; nobody has measured the two side by side. The
> practical advice is unchanged — check before describing it as such.

Raw data: `psf-zero/data/phase1_v3_verify_strict_intel_2026-09-10.csv` — the
run's own 120-row output, with per-seed min/median/max/stdev, RSS and the
environment columns. (An earlier revision of this section cited a summary
table typed up from the console log instead, because the CSV had not been
transferred yet. The CSV has since been checked against that summary and
matches it exactly at every one of the twelve cells; the hand-typed
intermediate has been removed so there is one source of truth for this run.)

##### Two things the per-seed data shows that the summary did not

**1. `qiskit opt=3` at 156 qubits has one slow repetition in every single
seed — and this is the documented reason the harness looks the way it
does.** Across all 10 seeds, `opt=3`'s five timed repetitions at 156 qubits
span a factor of **3.6x** (min 20.4–29.1ms against max 71.8–96.7ms), with
the per-point standard deviation (20–33ms) *exceeding* the median. Every
other arm at every other scale sits at 1.05x–1.3x. The same pattern is in
the earlier run's CSV at 3.3x, so it is reproducible, not a one-off.

This is not a new discovery — it is `test1_v3.py`'s own stated reason for
existing. Defect [2] in the script's header reads: *"ONE SAMPLE PER POINT
against a 1–25 ms workload. Warm repeats at 156q measured a 198.7% spread
on the Qiskit side (min 22.3, max 96.3 ms). A single sample there is noise,
not a measurement."* Those numbers are within a couple of milliseconds of
what these two runs measured, 20 seeds later. What the new data adds is
that the effect is **universal at that point** (10/10 seeds in each run,
not an occasional spike) and **specific to the Qiskit arm at the largest
scale** — so any single-sample measurement of `opt=3` at 156 qubits has
roughly a one-in-five chance of landing on a number 3–4x too high, which
would silently inflate every speed-up quoted against it. The median-based
reporting this README uses is unaffected; a mean would not be.

**2. The `strict` arm's memory columns are not comparable to the others,
and the CSV will mislead anyone who reads them as-is.** `psf_canonical_strict`
shows a *lower* `RSS_Delta_MB` than `psf_canonical` at every scale (0.55MB
against 1.66MB at 156 qubits) — which reads as "strict uses less memory"
and is not what happened. `test1_v3.py` measures memory in a separate,
untimed pass that runs for a **fixed wall-clock window**
(`t_end = time.perf_counter() + MEM_WINDOW_S`), so a slower arm simply fits
fewer compilation calls into it. That shows directly in the sample counts:
strict is 5.2x–6.6x slower per call and records 2.9x–5.7x fewer samples.
Fewer calls sampled, lower observed peak. **Nothing about the strict path
allocates less** — it is the same synthesis plus an extra reconstruction.
Read `RSS_Delta_MB` as comparable only between arms of similar speed.

#### Update (2026-09-10): the 50,000-iteration cumulative loop, on the Intel machine — the ratio holds, and the quoted headline should be the median one

`test_cumulative_compile_time.py` was run at its full 50,000 iterations on
the Intel machine (15 qubits, 20 gates/pair — the same circuit family as
the rest of this section). Correctness was checked first on the 6-qubit
version: all three engines reconstructed to fidelity 1.000000000000.

| | Qiskit L3 | PSF `verify=True` | PSF `verify=False` |
| :--- | :---: | :---: | :---: |
| mean | 9.209ms | 1.494ms | 1.222ms |
| **median** | **8.261ms** | **1.491ms** | **1.208ms** |
| stdev | 5.241ms | 0.567ms | 0.556ms |
| total over 50,000 | 460.5s | 74.7s | 61.1s |

**The script's own headline is mean-based and therefore slightly too
generous; the median is the number to quote.** It printed 6.16x
(`verify=True`) and 7.54x (`verify=False`). By median those are **5.54x**
and **6.84x**. The reason is visible in the table: Qiskit's mean sits 11.5%
above its median while both PSF arms sit within 1.2% of theirs, so the
run's noise inflates the numerator far more than the denominator. This
project's own rule — established when the 10,000/50,000-iteration decay
turned out to be background contention — is to report medians on shared
hardware, and applying it to this run means quoting the smaller pair.

**The decay question is settled further, and this run shows the mechanism
directly.** The progress log prints elapsed time every 500 iterations.
Differencing it: the loop holds a median of **13.4s per 500 iterations**,
and six of the hundred intervals exceed 1.3x that — at iterations
3,500–4,000 (39.3s), 10,500–11,500 (27.3s then 41.5s), and 40,500–42,000
(27.7s, 31.0s, 28.5s). Everything else sits at 13.2s. Those six windows
account for 115s, 8% of the 1,449s run. Crucially they are **scattered
across the run rather than accumulating** — one early, one in the middle,
one at 82% through — and the final interval (13.3s) is as fast as the
opening ones. A leak or a growing cache cannot produce that shape;
intermittent external load can, which is what this section concluded from
indirect evidence in September and has now been watched happening three
times.

**And the cost of the safety net is machine-dependent, which this README
had not established.** `verify=True ÷ verify=False` is **1.22x** here,
against 1.88x measured on the other machine at the same iteration counts.
Both are real; the honest range for "what the default check costs you" is
**1.2x–1.9x depending on the machine**, not a single figure. The absolute
PSF numbers barely moved between machines (1.208ms here against 1.172ms /
1.211ms there) while Qiskit's did (8.261ms against 10.503ms / 11.038ms), so
most of the ratio difference is Qiskit's side, not PSF's.

##### The per-iteration data: the contention account is confirmed directly, and one of this README's numbers is corrected by it

The script's raw per-call array (`cumulative_compile_times.npz`, 3 × 50,000
timings) has since been transferred, so the analysis above no longer has to
work at the progress log's 500-iteration granularity. Its aggregates
reproduce the table above exactly. Four things follow that the coarse data
could not show.

**1. In the contention windows all three arms slow by comparable
multiples — which is why the ratio survives.** Taking the three windows the
progress log identified and comparing each arm against its own baseline
over the rest of the run:

| Iterations | qiskit | psf `verify=True` | psf `verify=False` |
| :--- | :---: | :---: | :---: |
| 3,500–4,000 | 19.17ms (2.19x) | 3.15ms (2.23x) | 2.98ms (2.63x) |
| 10,500–11,500 | 16.93ms (1.93x) | 2.91ms (2.06x) | 2.73ms (2.41x) |
| 40,500–42,000 | 14.69ms (1.68x) | 2.55ms (1.80x) | 2.38ms (2.09x) |
| everything else | 8.76ms | 1.41ms | 1.13ms |

Everything is slowed by roughly the same factor at the same moments. That
is what an external load does, and it is why a run can lose 8% of its
wall-clock to contention without the measured ratio moving.

**2. The correlation is real but weaker than this README states, and the
figure it quotes is machine-specific.** The 2026-09-09 correction above
reports "r≈0.64 between the two engines' per-call times". On this machine
the per-iteration Pearson correlation is **+0.354** (qiskit vs
`verify=True`) and **+0.380** (vs `verify=False`); Spearman agrees at
+0.349 / +0.372. The qualitative claim — the two engines are slowed at the
same moments by something outside both — holds and is now shown directly.
The specific coefficient does not transfer between machines and should be
quoted with its run attached. *(Added 2026-09-10: the original run's raw
arrays have since been recovered and give exactly +0.644, so "r≈0.64" was
not an error — the two figures are two machines under different amounts of
external load. See the update below.)*

**3. Most of Qiskit's slow calls are *not* shared, which the correlation
alone would hide.** Of the 500 iterations where Qiskit exceeds its own 99th
percentile, **416 (83%) are Qiskit-only** — neither PSF arm is above its
own p99 at that iteration. Simultaneous outliers do occur far above chance
(37 iterations with all three arms over p99, against 0.05 expected if
independent; 241 with both PSF arms, against 5.0 expected), and the
co-occurrence is strongest between the two arms that run adjacently in the
loop. So there appear to be two distinct effects: sustained external
windows that move everything together and preserve the ratio, and a
separate, much more frequent Qiskit-only tail (p99 24ms, p99.9 65ms, max
146ms against a median of 8.26ms) that PSF-Zero does not share. This
README has never distinguished the two; the second is the larger
contributor to Qiskit's mean sitting 11.5% above its median.

**4. No drift, and the ratio is tighter than the aggregate suggests.**
Per 1,000 iterations (50 windows, [`psf-zero/data/cumulative_50k_intel_2026-09-10_per1000.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_50k_intel_2026-09-10_per1000.csv)),
the median-based `qiskit ÷ verify=True` ratio has a median of 5.57x with
first-half 5.34x against second-half 5.59x — flat. Absolute times do drift
up about 11–15% from the first tenth of the run to the last (qiskit 7.77 →
8.95ms, `verify=True` 1.435 → 1.599ms), but they drift *together*, so the
ratio does not.

**One anomaly, unexplained.** Around iterations 12,000–14,000 both PSF arms
run about 30% *faster* than their own baseline (`verify=True` 1.04/0.99ms
against ~1.49ms; `verify=False` 0.79ms against ~1.21ms) while Qiskit does
not move (7.78/7.94ms against ~8.26ms). That is the opposite of contention
and it is why one of the ten bins reports `qiskit ÷ verify=False` at 8.66x
when the other nine sit in a 6.68–6.94x band. We have no account for it and
have not investigated; it is flagged here rather than smoothed away.

#### Update (2026-09-10): a fifth run, and what it shows about ratios vs. absolute times

A third run on the Intel machine (the fifth overall), same 10 seeds × 5
reps, is in the table above: 8.03x/4.10x/3.42x/3.05x by median, and
7.37x/3.98x/3.25x/2.91x by min-of-samples against run 2's
7.29x/4.03x/3.24x/2.91x — identical to two decimal places at 156 qubits.

The absolute times, though, are 14–16% slower than run 2 at the larger
scales (`qiskit opt=3` at 156 qubits: 26.35ms against 23.06ms;
`psf canonical`: 8.64ms against 7.42ms), with the first two seeds of the
156-qubit block clearly the worst of the run. **Both arms moved together**,
which is the same shared-contention signature this section documents at
length for the 10,000/50,000-iteration cumulative loop — and the ratio
barely moved (3.05x against 3.11x). It is a clean, small-scale
demonstration of the rule this project arrived at the hard way: on shared
hardware, report ratios and medians, not absolute times from a single run.

Output quality was again exact: 2-qubit gate count 21/75/150/234 and depth
9/13/16 across all three quality-matched arms, `opt0` at 420/1500/3000/4680
and depth 320, equivalence at 6 qubits < 4.5e-15.

Raw data: `psf-zero/data/phase1_v3_test1_v3_intel_run2_2026-09-10.csv`.

##### The archived raw data, and why the provenance can't be settled from it

The account above turns on which `psf_compile.py` the original run used, and
the earlier open item turned on which machine it ran on. Both were checked
against everything this project still holds: all thirteen accumulated raw
CSVs, now archived under `psf-zero/data/archive/` with a file-by-file
mapping in `provenance-map.md`. Neither question can be answered from them.

- **Ten of the thirteen record no environment metadata at all** — no CPU
  string, no platform, no Python or Qiskit version. Those columns were only
  added to the harnesses later, which is exactly why runs 3 and 4 above
  *can* be positively tied to one machine while the original cannot.
- **`test1_v3.py` writes to a fixed filename** ([`phase1_v3_benchmark_results.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/archive/phase1_v3_benchmark_results.csv)),
  so each run overwrites the last. The surviving copy is run 4's. The
  original declining run's own output no longer exists.

So the `verify="strict"` account above stays testable going forward but can
never be checked against the original artifact, and the machine question is
closed as unanswerable rather than answered. This is a record-keeping
failure, not a measurement one, and it is already fixed for everything
written since: every current harness records `platform.processor()` and its
library versions in its output.

The archive is worth having for a separate reason: several of these files
could be matched to the exact published table they produced, by their
numbers alone. [`phase1_v2_benchmark_results.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/archive/phase1_v2_benchmark_results.csv) reproduces this section's
`phase1.py` re-run table to the last digit (PSF 1.08/2.60/5.10/8.84ms
against Qiskit 10.23/16.05/22.34/29.59ms), and [`phase2_benchmark_results.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/archive/phase2_benchmark_results.csv)
does the same for the `phase2.py` table (10.09/15.94/25.15/49.06ms against
29.85/46.46/72.67/136.58ms). The two retracted-artifact runs survive too,
and are visibly wrong in precisely the way this section describes:
[`phase1_benchmark_results.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/archive/phase1_benchmark_results.csv) has Qiskit pinned at 1.30–1.34s at *every*
scale (the no-op `transpile()` bug that produced the retracted "200x"), and
[`phase2_v2_deadzone_results.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/archive/phase2_v2_deadzone_results.csv) has PSF-Zero 4.1x *slower* than Qiskit at
1000 qubits (the `force_consolidate` bug — the measurement that triggered
this section's entire re-investigation). Those two files are the primary
evidence for retractions this README currently supports with prose only.

#### Update (2026-09-10): the raw per-iteration data behind the original decay claim has been recovered — the decay question closes, and one published figure is qualified

The 2026-09-09 correction near the top of this section was written from
binned summaries of a 50,000-iteration run whose raw per-iteration arrays
were, at the time, described but not held. Those arrays have now been
recovered. They are identifiable beyond doubt: their means are
**10.403ms / 9.825ms / 1.903ms**, matching the 50,000 row of that
correction's table to the last digit.

**This run predates the `verify` change, which makes it the first
within-run measurement of the old default path.** Its
`verify=True ÷ verify=False` is 5.16x by mean and 5.05x by median. The
2026-09-10 Intel run of the same script gives 1.22x / 1.23x. A 4x gap in
the same quantity is not run-to-run noise; it is the `Operator()`-based
check that the 2026-09-09 update reports removing. So every number below
labelled "old verify" is that path, measured against `verify=False` in the
same process, on the same circuits, in the same run — which nothing else
in this section has been able to do.

**1. The correlation figure is exact; the "2.5–2.7x" figure mixes
statistics.** Pearson on the per-iteration times: **+0.644** (Qiskit vs
old-verify) and **+0.616** (vs `verify=False`); Spearman +0.761 / +0.730;
the two PSF arms against each other +0.906. The published "r≈0.64" is
confirmed to three digits. The published "2.5–2.7x slower than PSF-Zero's
own overall average" reproduces only as mean-at-Qiskit's-p99 ÷ overall
median (2.77x / 2.63x); mean ÷ mean is 1.78x / 1.72x and median ÷ median
is 3.20x / 3.27x. A correction to that effect is recorded inline above.
Note also that the two correlation figures in this section are not in
conflict: +0.644 here and +0.354 on the Intel machine are different runs
on different hardware, and the coefficient is a property of how much
external load a given machine happened to be under.

**2. The decay was never monotonic — it is seven separate bursts with full
recovery between each.** Per 1,000 iterations (50 windows,
[`psf-zero/data/cumulative_50k_preverifychange_per1000.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_50k_preverifychange_per1000.csv)), 14 windows sit
above 1.3x the run's median window and they fall into seven contiguous
bursts: iterations 15,000–18,000, 21,000–25,000, 26,000–28,000,
34,000–35,000, 38,000–39,000, 40,000–42,000 and 43,000–44,000. Between and
after them the machine returns to baseline every time — the last 6,000
iterations are clean, and the 45,000–50,000 window is the *fastest* of the
entire run (Qiskit 7.434ms, old-verify 5.574ms, `verify=False` 1.151ms,
against first-window 8.388 / 6.160 / 1.237ms). Inside the bursts all three
arms move together, exactly as the Intel run shows: at 21,000–22,000, for
instance, Qiskit 16.85ms / old-verify 20.10ms / `verify=False` 4.13ms
against a clean baseline of 7.77 / 5.66 / 1.17ms. The bursts cost about
21% of the run's wall-clock (1,106s actual against 873s at the clean rate).
This is the shape the 2026-09-09 correction inferred from ten coarse bins,
now visible at 1,000-iteration resolution and with the recovery explicit.

**3. Excluding the bursts, the ratios agree with the Intel machine to
within 3%.** Over the 72% of iterations outside them: `qiskit ÷
verify=False` = **6.67x** (Intel: 6.84x), old-verify ÷ `verify=False` =
4.86x, `qiskit ÷ old-verify` = 1.37x. Two different machines, two different
`verify` implementations, and the quantity that does not involve `verify`
at all lands within 2.5% of itself. The whole-run figures are 6.62x / 5.05x
/ 1.31x, so even including the bursts the ratio moves by under 1%.

**4. The Qiskit-only tail reproduces.** Of the 500 iterations above
Qiskit's own 99th percentile, **451 (90%)** have neither PSF arm above its
own p99. The Intel run gave 83%. So the two-distinct-effects reading —
shared external windows that preserve the ratio, plus a much more frequent
Qiskit-only tail that does not — now holds on both runs rather than one.

**5. The `verify="strict"` question is narrowed but still open.** With the
old default now measured within a single run at 4.86–5.05x `verify=False`,
and `strict` measured at about 6.3x `verify=False` on the Intel machine,
the gap is **~1.28x**. That is smaller than the 2.5x this section once
claimed and larger than zero. It remains a comparison across two machines
and two scripts, so it still does not establish that `strict` differs from
the old default — only that equivalence is unconfirmed. Settling it needs
one three-arm run (`False` / current `True` / `strict`) in a single
process; that has not been done.

One thing worth recording because it constrains future comparisons: the
Qiskit arm's *median* is essentially identical across the two runs (8.262ms
here, 8.261ms on Intel) and its p10/p25 agree within 5%, but the upper
tails diverge sharply (p75 12.161 against 9.439; p90 17.554 against
10.850). Qiskit's typical call is the same on both machines; what differs
is how often it is disturbed. That is a further reason this project reports
medians.

Raw data: [`psf-zero/data/cumulative_50k_preverifychange_per1000.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/cumulative_50k_preverifychange_per1000.csv)
(50 windows). The underlying `.npz` is 3 × 50,000 float64 and is not
checked in; the CSV is the reusable summary.

#### Where this matters in practice: VQE and other variational hybrid workflows

The Variational Quantum Eigensolver (VQE) is the leading current-generation
approach to running chemistry and materials-science problems on NISQ-era
hardware: a classical optimizer repeatedly proposes new parameters for a
fixed-structure parameterized circuit (the ansatz), the quantum device
evaluates the resulting energy expectation value, and the loop repeats —
typically thousands to tens of thousands of times per problem — until the
optimizer converges. QAOA-style parameter search follows the same pattern.
In both cases, the same circuit *structure* is recompiled on almost every
iteration with new parameter values, which puts the compile step directly
in the hot path of the algorithm rather than being a one-time setup cost —
exactly the workload this section's cumulative-loop benchmark models.

> **Correction, 2026-09-09 — the sentence immediately above overstates the
> case, and the overstatement is the kind a reviewer would find first.** In
> the standard Qiskit pattern, a VQE loop does **not** recompile per
> iteration. `Parameter` objects survive transpilation, so the accepted
> practice is to transpile the parameterised ansatz **once** and then call
> `assign_parameters()` each iteration — binding is orders of magnitude
> cheaper than compiling, and is what `qiskit-algorithms`' VQE and the
> Runtime primitives are built around. The cumulative-loop benchmark in
> this section therefore models a loop that recompiles from scratch every
> iteration, which is not the default shape of a textbook VQE run.
>
> Two narrower cases where the loop genuinely does recompile, and where this
> section's numbers apply directly: **adaptive ansätze** (ADAPT-VQE and
> relatives), where the circuit *structure* grows each iteration and must be
> re-synthesised; and **simulator-side development loops** — parameter
> sweeps, ansatz search, CI over circuit families — where there is no QPU
> queue and compile time really is a visible fraction of wall clock.
>
> The claim that does *not* survive is the hardware one. A 50,000-iteration
> loop against a real backend is dominated by queue time and QPU execution:
> even in a dedicated Runtime session at a few seconds per job, that is on
> the order of days, against which the 491s (8.2 min) of compile time saved
> here is roughly 0.1–0.2%. Faster compilation is a real classical-side win;
> it is not a route to more hardware iterations per session, and this README
> should not be read as claiming otherwise. The calibration-drift question
> below remains untested for exactly this reason.

Two of this project's now-confirmed properties apply directly:

- **Compile-time overhead, not fidelity.** A variational loop's total
  wall-clock time is dominated by however many (quantum execution +
  classical compile) round-trips it needs, so shaving milliseconds off
  every compile call compounds linearly across the loop. Over the
  50,000-iteration run analyzed above, PSF-Zero (`verify=False`) saved
  roughly 425 seconds of cumulative compile time against Qiskit L3 — real
  and measured, and, per the correction above, not an artifact of that
  same run's background-contention noise (the *absolute* time saved held
  up even in the noisiest windows, since the contention slowed Qiskit's
  own calls by more in absolute terms than it slowed PSF-Zero's). This
  doesn't mean an optimizer reaches a lower energy in fewer iterations —
  that depends on the classical optimizer, not the compiler — it means
  more iterations fit in the same wall-clock budget, which is the actual
  constraint a fully-automated variational loop runs into on shared or
  rate-limited hardware.
- **Fails safe, not silently, and not fatally.** A variational loop that
  crashes or silently miscompiles partway through a long optimization run
  is worse than a slow one. PSF-Zero's decomposition occasionally lands on
  a measure-zero degenerate point in the Weyl chamber — this section's own
  diagnostic run above hit exactly one such case in roughly 35,000 block
  syntheses — and the project's explicit "no silent fallback" policy means
  this is handled by emitting a warning and falling back to standard
  CX-basis synthesis for that one block, not by crashing or by silently
  producing a wrong circuit. That's a different, and more useful, claim
  than "0% failure rate": the design degrades gracefully on the rare input
  it can't handle via its closed-form path, which is the property that
  actually matters for unattended, long-running automation — not a
  guarantee that the rare case never occurs.

We have not run an actual VQE loop against real hardware end-to-end (that's
the open, not-yet-tested Roadmap item above, on whether this compile-time
advantage actually reduces real-hardware calibration-drift exposure) — but
the two properties above are why this specific workload is a natural,
well-motivated target for `compile_for_hardware(..., verify=False)`, and
the mechanism by which this section's numbers would actually pay off in
practice.

Code: [`benchmarks/test_cumulative_compile_time.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_cumulative_compile_time.py)

Code: the original, superseded scripts are
[`benchmarks/phase1_v2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase1_v2.py)
(15–156 qubits) and
[`benchmarks/phase2_v2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase2_v2.py)
(156–1000 qubits). The four fixes applied on top of them, in the order
discovered, each with the measurement that motivated it, are documented in
[`benchmarks/phase1_qiskit_worker.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase1_qiskit_worker.patch) /
[`benchmarks/phase2_qiskit_worker.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase2_qiskit_worker.patch)
(no-op transpile fix),
[`benchmarks/compile_force_consolidate.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_force_consolidate.patch)
(`force_consolidate` fix),
[`benchmarks/phase1_warmup_v2.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase1_warmup_v2.patch) /
[`benchmarks/phase2_warmup.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase2_warmup.patch)
(symmetric warm-up fix, the one that produced this section's intermediate,
`verify=True` table), and
[`benchmarks/phase1_verify_false.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase1_verify_false.patch) /
[`benchmarks/phase2_verify_false.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/phase2_verify_false.patch)
(the `verify=False` change, on top of `psf_compile.py`'s own
[`compile_optional_verify.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_optional_verify.patch),
which produced this section's final table above).
