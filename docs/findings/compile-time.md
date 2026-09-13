# Compile time: what survives, and the three retractions it took to get there

**Status:** settled for the claims below. This is the most-corrected part of the
project — three headline numbers were retracted and two hypotheses were refuted by
their own pre-registered criteria. The full chronological record, with every
intermediate wrong answer left standing, is in [`docs/log/03-compile-time-scaling.md`](../log/03-compile-time-scaling.md).

---

## The claim

On circuits containing deep, same-qubit-pair 2-qubit chains, PSF-Zero's analytic KAK
synthesis compiles **2.5x–5x faster than Qiskit `optimization_level=3`** at matched
output quality, and **150x–270x faster than TKET** — the latter at a fixed depth cost
(9 against TKET's 7).

## What was retracted, and what caught it

| Retracted claim | What it actually was | How it was caught |
| :--- | :--- | :--- |
| "up to ~200x faster than Qiskit" | `transpile()` was called with no `backend` and no `basis_gates`, so it had no target basis and passed every `UnitaryGate` through **untouched** | `count_ops()` identical before and after; Qiskit's time was flat at 1.30–1.34 s from 15 to 156 qubits — invariant to circuit size, which is impossible if it were compiling |
| "615x–867x at 1000 qubits" | The circuit generators never produced blocks above `block_gate_floor`, so PSF-Zero returned the input unchanged and the "speed-up" was the cost of doing nothing | Measured block sizes in the generators directly; PSF-Zero's own reported output depth was no better than the unoptimised input |
| "the ratio decays with iteration count" | Episodic background load on one machine | Raw per-iteration arrays: the slowdown came in bursts with **full recovery** between each, and the final 5,000 iterations were the fastest of the run |

A fourth measurement error worked in the opposite direction and is worth recording
because it is the one that triggered the whole re-investigation: `ConsolidateBlocks`
defaults to `force_consolidate=False`, which silently refuses to merge a block made
entirely of pre-existing `'unitary'`-named nodes — exactly what this project's own
circuit generator produces. PSF-Zero was resynthesising once per gate instead of once
per pair, and measured **4.1x slower** than Qiskit at 1000 qubits.

## Where the time actually goes

Once the process-level artifacts were removed, PSF-Zero's Rust-core synthesis was
still *slower* per block than Qiskit's. Profiling `synthesize()` into its four
sub-phases over 2,000 random SU(4) blocks found why:

| Phase | ms/block | Share |
| :--- | ---: | ---: |
| matrix → list (`u_r`/`u_i`) | 0.003 | 0.3% |
| `geometric_decompose()` — the actual decomposition | 0.038 | **3.3%** |
| circuit construction | 0.109 | 9.4% |
| `Operator()` fidelity self-check | 1.002 | **87.0%** |

The decomposition was a rounding error. 87% of per-block cost was PSF-Zero
**re-verifying its own output on every call** — reconstructing `Operator(qc)` from the
freshly-synthesised 2-qubit circuit and comparing it against the target.

This also killed the premise the benchmark started from. Qiskit's own 2-qubit
synthesis is *itself* an analytic Cartan/KAK decomposition, not a search — the search
in `optimization_level=3` lives in layout, routing and gate cancellation. "PSF-Zero
should win because it skips search" was never the right mechanism at this level.

## The fix: `verify` became a three-way choice

Rather than trading the safety net for speed, `verify` was split:

| Setting | What it does | Cost |
| :--- | :--- | :--- |
| `verify=True` (**default**) | cheap Rust-core check | **1.2x–1.9x** `verify=False`, machine-dependent |
| `verify=False` | no per-call check; correctness verified in tests/CI | baseline |
| `verify="strict"` | the old `Operator()` reconstruction | **5.1x–6.6x** the current default |

`verify="strict"` makes PSF-Zero **slower than Qiskit `optimization_level=3`** at
every scale above 15 qubits (0.67x / 0.48x / 0.47x at 50/100/156 qubits). It exists
for people who want it; it is not a recommended setting.

The decomposition math it re-checks has been validated offline at worst-case
(1 − fidelity) = 1.11e-15 over 1,000 trials against the real core and 8.88e-16 over
200 trials against the stub, reproduced on a second machine.

## What survives

### Against Qiskit `optimization_level=3`, `verify=False`

10 seeds per point, symmetric warm-up outside the timer, `spawn`, equivalence checked
at every point:

| | 15q | 50q | 100q | 156q | 300q | 500q | 1000q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qiskit ÷ PSF-Zero** | 5.15x | 3.98x | 3.45x | 2.86x | 3.02x | 2.78x | 2.54x |

Largest at the smallest circuits, settling into a stable ~2.5x–3x band from 150 blocks
up. An independent re-run of the same two scripts on a different machine gave
9.51 / 6.16 / 4.38 / 3.35 (15–156q) and 2.96 / 2.91 / 2.89 / 2.78 (156–1000q) — the
large-scale half reproduces within a few percent; the small-scale half is
machine-dependent and wider than a single number can express.

### With the default check on

A 50,000-iteration compile loop at 15 qubits, run on two machines:

| | Qiskit L3 | `verify=True` | ratio | `verify=False` | ratio |
| :--- | ---: | ---: | :---: | ---: | :---: |
| AMD machine (mean) | 11.038 ms | 2.303 ms | **4.79x** | 1.211 ms | **9.12x** |
| Intel machine (median) | 8.261 ms | 1.491 ms | **5.54x** | 1.208 ms | **6.84x** |

The Intel row is quoted by median deliberately: that run's script printed 6.16x/7.54x
from means, but Qiskit's mean sat 11.5% above its median while both PSF arms sat
within 1.2% of theirs, so the run's noise inflated only the numerator.

### Output quality is matched

At 156 qubits, `qiskit opt=3`, `psf canonical` and `psf cx` all emit the **same** 234
two-qubit gates (21/75/150/234 at 15/50/100/156). Depth: **9** canonical, **13** cx,
**16** Qiskit. Equivalence < 4.5e-15 at every point. `opt=0` emits 20x more gates and
is not a quality-matched comparison.

### Against TKET

| | 10q | 20q | 40q | 80q | 160q |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TKET | 1.06 s | 2.14 s | 4.19 s | 8.24 s | 16.84 s |
| PSF-Zero | 0.007 s | 0.009 s | 0.016 s | 0.032 s | 0.062 s |
| **Speed-up** | 152x | 237x | 262x | 258x | 272x |
| Depth (TKET / PSF) | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 |

Reproduced three times. The 7-vs-9 depth split held at every scale in every run. One
third-run 80-qubit point (0.081 s against 0.002–0.022 s neighbours) does not fit the
trend, is a single measurement, and is flagged rather than averaged away.

### Determinism

300 randomly sampled 2-qubit unitaries produced **300 identical circuits**, depth
exactly 9 with zero variance. That is a consequence of doing an exact decomposition
rather than a search, not a claim about optimality.

## The measurement discipline this cost us

Everything above is the product of rules the project adopted after being wrong:

- Warm-up call outside the timer for **both** engines. A fresh interpreter pays a
  one-time `transpile()` cost measured at 2.41 s cold against 0.07 s warm on an
  identical 156-qubit circuit — a 32x difference from methodology alone.
- Multiple seeds, repeated timed calls per point. A single sample of `opt=3` at 156
  qubits has roughly a one-in-five chance of landing 3–4x high: across 10 seeds its
  five repetitions span a factor of **3.6x**, with the standard deviation exceeding
  the median, in **every seed**.
- Medians, not means, on shared hardware.
- `seed_transpiler` pinned wherever Qiskit's randomised layout/routing search is
  involved. Unpinned, the same 500-qubit circuit measured at both 0.06 s and 0.23 s
  across runs of otherwise-identical code.
- Environment recorded **into the output CSV**, not just printed. Output filenames
  never fixed.

## One anomaly, permanently unexplained

A single early run of `test1_v3.py` gave 4.35x / 2.21x / 1.66x / 1.44x at
15/50/100/156 qubits — roughly 0.51x every other run of the same script. Comparing
absolute times arm by arm showed it was paying two **scale-independent** penalties:
~1.35x on the Qiskit arm and ~2.65x on the PSF arm. So its "decline with scale" was
the same shape every run has, uniformly scaled; the thing to explain was the arm
asymmetry.

The candidate — that it ran a pre-2026-09-09 `psf_compile.py` with the old expensive
`verify` — was stated with a numeric prediction and then **tested and refuted**:
`verify="strict"` in a paired same-run comparison produced 1.42x / 0.67x / 0.48x /
0.47x, overshooting the prediction by about 3.1x at every scale.

The run's own output file no longer exists (`test1_v3.py` wrote to a fixed filename
and overwrote itself) and ten of the thirteen archived CSVs record no environment
metadata at all, so no replacement hypothesis can ever be tested against it. **It is
a single anomalous run and should not be treated as evidence about anything.** Five
later runs of the same script all sit in a 2.8x–8.1x band.

## Limits

- Everything here is one circuit family: dense, same-pair 2-qubit blocks. On a generic
  `random_circuit()` PSF-Zero correctly reports `0/0 blocks` and contributes nothing.
- Faster compilation is a classical-side win. On real hardware a variational loop is
  dominated by queue and QPU time; the ~8 minutes saved over 50,000 compiles is
  roughly 0.1–0.2% of such a session. This README should not be read as claiming
  otherwise.
- The standard Qiskit VQE pattern transpiles the parameterised ansatz **once** and
  calls `assign_parameters()` per iteration, so a textbook VQE does not recompile per
  iteration at all. The workloads where this matters are adaptive ansätze (ADAPT-VQE
  and relatives, where the circuit structure grows) and simulator-side development
  loops.
- Not run through [Benchpress](https://github.com/Qiskit/benchpress).


## Repeated compilation of a single fixed unitary (2026-09-13)

The sections above measure compile time across many independently-seeded circuits.
A separate question: what happens when the *same* input is compiled 1,000 times in a
row — the shape of a VQE/QAOA loop that re-submits one ansatz structure repeatedly.
[`benchmarks/verify_determinism_variance.py`](../../benchmarks/verify_determinism_variance.py)
fixes one Haar-random SU(4) unitary (`seed=42`) and compiles it 1,000 times with
Qiskit L3, TKET (`FullPeepholeOptimise`, via `DecomposeBoxes`), and PSF-Zero
(`verify=False`), hashing each output circuit's gate sequence to check whether the
same input keeps producing the same output.

**All three engines were fully deterministic on this single input — not just
PSF-Zero.** Every one of the 1,000 runs, for all three engines, produced a circuit
whose signature hash matched the other 999: 1 unique pattern per engine, 3 total. This
is not evidence of a determinism advantage specific to PSF-Zero; it shows that
Qiskit's and TKET's search procedures, when applied to one fixed input over and over,
also converge to the same output every time on this circuit. The determinism claim
this project makes elsewhere is about PSF-Zero's decomposition being closed-form
by construction (no seed to control for even in principle) — this experiment
doesn't distinguish that from "converges reliably in practice," since the other two
engines did too, on this input.

**Compile time, median (not mean — see below for why):**

| Engine | Median | Mean | IQR / median | Qiskit or TKET ÷ PSF-Zero (median) |
| :--- | ---: | ---: | ---: | ---: |
| Qiskit L3 | 7.460 ms | 7.768 ms | 6.1% | **15.2x** |
| TKET | 42.487 ms | 42.846 ms | 1.8% | **86.3x** |
| PSF-Zero | 0.492 ms | 0.504 ms | 14.9% | — |

**Qiskit's mean sits above its median because of two outliers, not a trend.** Of
1,000 calls, 998 fall in a tight band (min 6.889 ms); two — iterations 210 and 979 —
took 102.7 ms and 104.0 ms, roughly 14x the rest. Excluding them barely moves the
median (7.460 ms either way) but pulls the mean down from 7.768 to about 7.67 ms.
Binning the run into first-50-calls vs remaining-950 shows no warm-up effect (median
7.464 ms vs 7.460 ms) — the two slow calls are scattered, isolated events, not a
cold start. Consistent with this project's standing finding that shared-hardware runs
show occasional external contention: report medians, not means, and don't read a
single long run's mean as the number.

**PSF-Zero's relative spread is the largest of the three, not the smallest.** An
earlier read of this run's raw std/mean ratios suggested PSF-Zero was the most
stable; that doesn't hold up under a scale-independent measure. IQR-to-median is
6.1% for Qiskit, 1.8% for TKET, and **14.9% for PSF-Zero** — PSF-Zero's absolute
timings are smallest (sub-millisecond), so a small absolute jitter is a
proportionally larger fraction of its own median. The earlier claim that PSF-Zero
showed "the tightest distribution" is corrected here: on a relative basis it does
not, though its absolute time and absolute spread are both still the smallest of
the three.

**What this experiment does and doesn't establish.** It's a real, if narrow, model of
the fixed-ansatz-repeated-compile shape a variational loop has, and on it PSF-Zero is
15x–86x faster at the median with correctness unaffected (all engines producing their
own single stable pattern throughout). It does not test the calibration-drift
question raised elsewhere in this file — no hardware execution is involved, only
repeated compilation of the same circuit — and it does not add anything to the
determinism argument beyond what the closed-form construction of the decomposition
already establishes analytically. Two methodological notes for future runs: this
script does not warm up either engine outside the timed loop (unlike the
`phase1.py`/`phase2.py`/`test_cumulative_compile_time.py` harnesses elsewhere in this
project, which do), and TKET is measured through an extra `DecomposeBoxes()` pass not
present in this project's other TKET comparisons (`test_psf_vs_tket.py`,
`test_scale_explosion_war2.py`), so its number here is not directly comparable to
those.

Raw data: [`data/determinism_variance_2026-09-13.csv`](../../data/determinism_variance_2026-09-13.csv)
(3,000 rows: 1,000 iterations × 3 engines, per-iteration time, depth, CX count, and
circuit signature hash).


## Files

- [`benchmarks/phase1_v2.py`](../../benchmarks/phase1_v2.py),
  [`benchmarks/phase2_v2.py`](../../benchmarks/phase2_v2.py) — the scaling sweeps
- [`benchmarks/test1_v3.py`](../../benchmarks/test1_v3.py) — methodology-corrected harness, and
  [`benchmarks/test1_v3_verify_strict.py`](../../benchmarks/test1_v3_verify_strict.py) — the paired `verify="strict"` wrapper
- [`benchmarks/test_cumulative_compile_time.py`](../../benchmarks/test_cumulative_compile_time.py) — the 50,000-iteration loop
- [`benchmarks/test_psf_vs_tket.py`](../../benchmarks/test_psf_vs_tket.py),
  [`benchmarks/test_scale_explosion_war2.py`](../../benchmarks/test_scale_explosion_war2.py) — TKET comparisons
- [`benchmarks/compile_optional_verify.patch`](../../benchmarks/compile_optional_verify.patch),
  [`benchmarks/compile_force_consolidate.patch`](../../benchmarks/compile_force_consolidate.patch) — the two fixes
- Raw data: [`data/`](../../data/) and [`data/archive/`](../../data/archive/), with a
  file-by-file provenance map in
  [`data/archive/provenance-map.md`](../../data/archive/provenance-map.md)
