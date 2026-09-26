# Addendum 174 -- Long-run stability results: every prediction confirmed; PSF-Zero is bit-for-bit deterministic over 200 iterations (2026-09-25)

> **Imported into the home series as Addendum 174.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `longrun-stability-results-2026-09-25.md`; body below unchanged. L1-L6 and the exploratory figures re-computed at home from `longrun_none_2026-09-25.csv` and `longrun_each_2026-09-25.csv`: all match. The exploratory gc observation points the same way as home Addendum 94 (Part 6).

**Scored against:** [`longrun-stability-preregistration-2026-09-25.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-addendum-173-preregistration-2026-09-25.md)
(locked in the Project before the run). Thresholds applied exactly as
written. Every verdict was recomputed from the two raw CSVs in the workplace
sandbox and matches the script's own scoring. Section 4 is exploratory.

**Run:** RunPod pod, `Linux-6.8.0-64-generic-x86_64`, Qiskit 2.5.2,
qiskit-aer 0.17.2, PennyLane 0.45.1, PSF-Zero Rust core, real
`lightning.gpu` block checks on the pod's RTX 4090. Two processes of 100
iterations each, run back to back (`--gc none` first, then `--gc each`,
about 505 s each, 11 s apart). **Script integrity:** the post-run check
returned the pre-registered normalized SHA-256
`ec8f9b77319d43d732a251d4a804116c3a48a1e5a2f84742d7af96fcddfc37dd`. (A first
attempt failed with a SyntaxError before producing any data: the
pre-registration text had been pasted into the script file by mistake. It was
replaced with the script and the hash then matched.)

**All times are from this RunPod pod only** and are not comparable with the
home or workplace machines.

## 1. Scoring

| Prediction | Verdict | Numbers |
|---|---|---|
| L1 W1 all inputs correct (operational check) | CONFIRMED | 0 wrong of 800 |
| L2 W1 routed circuit identical every time (operational check) | CONFIRMED | 1 fingerprint per input over 200 iterations; measured qubit 112 for all inputs; 9 routed 2-qubit gates |
| **L3 PSF-Zero bit-identical output, 0 fallbacks** | **CONFIRMED** | **1 synthesized-circuit fingerprint and 1 routed-circuit fingerprint over 200 iterations; 0 fallbacks; worst GPU difference 7.4e-13** |
| L4 shot noise binomial, pooled ratio in [0.85, 1.15] | CONFIRMED | 0.989 (none), 1.035 (each) |
| L5 timing CV < 0.10 (iterations 6-100) | CONFIRMED | W1 compile 0.090 / 0.016; W2 synthesize-and-verify 0.058 / 0.071 |
| L6a RSS it100 / it10 <= 1.05 | CONFIRMED | 1.0037 (none), 1.0017 (each) |
| L6b RSS it100, each / none within 5% | CONFIRMED | 0.9987 |

L1 and L2 were pre-registered as checks expected by construction (fixed
transpiler seed, a margin of about 13 standard errors). L3 is the
substantive result: PSF-Zero's Rust synthesis, rerun 200 times across two
processes, returned a circuit identical down to the last bit of every
parameter.

## 2. Numbers (iterations 6-100 unless stated)

| Quantity | gc none | gc each |
|---|---|---|
| W1 compile, median (IQR) | 78.1 ms (76.2-79.5) | 45.7 ms (45.4-46.2) |
| W1 simulate (4 circuits x 4,000 shots), median | 3,972 ms | 3,835 ms |
| W1 total per iteration, median | 4,054 ms | 3,884 ms |
| W2 synthesize + GPU check, median (IQR) | 205.9 ms (201.6-207.4) | 204.0 ms (200.3-206.6) |
| W2 compile, median | 15.0 ms | 15.2 ms |
| W2 total per iteration, median | 1,042 ms | 1,041 ms |
| First iteration / median: W1 compile, W2 synthesize | 0.90, 2.60 | 1.37, 2.51 |
| RSS at iterations 1 / 10 / 100 | 680.2 / 697.5 / 700.1 MB | 679.6 / 697.9 / 699.1 MB |
| RSS slope, iterations 10-100 | +0.004 MB/iteration | +0.012 MB/iteration |
| W1 mean <Z0> (00, 01, 10, 11) | -0.922, 0.916, 0.917, -0.921 | -0.921, 0.918, 0.917, -0.923 |
| W2 TVD over both processes | mean 0.0509, SD 0.0025 | |

Figures (made in the workplace sandbox from the raw CSVs; delivered with this
document as PNG files): Figure 1, timing box plots; Figure 2, W1 <Z0>
distributions with the binomial expectation; Figure 3, RSS versus iteration.

## 3. What this establishes

The 2026-09-28 pipeline (W1) and the PSF-Zero synthesis path (W2) ran 200
iterations with identical answers, 0 fallbacks, noise statistics that match
the binomial model, stable per-iteration times on this machine, and no
memory growth (under 0.5% after warm-up). This is evidence of software
reliability for a proof of concept. It says nothing about real-hardware
behaviour or about compression value (Stage 1c).

## 4. Post-hoc observations (exploratory, not scored)

- **Where the time goes.** In W1, noisy simulation takes about 3.9 s of about
  4.0 s per iteration; compilation is 1-2% of the pipeline. Compilation speed
  is not the bottleneck for this workload.
- **Explicit gc and compile time.** W1 compile was 78.1 ms (median) without
  explicit collection and 45.7 ms with `gc.collect()` after every iteration,
  and much steadier (CV 0.016 versus 0.090). In the gc-none process, 4 of
  100 iterations (4, 22, 42, 48) ran below 55 ms, like the gc-each process;
  in the gc-each process every iteration after the first did. One candidate:
  without explicit collection, Python's automatic collector runs during
  `transpile` and adds about 30 ms. This is **not established**: the two
  processes ran once each, in a fixed order, on a shared cloud machine, and
  W1 simulation was also 3.5% slower in the first process. An order-swapped,
  repeated comparison would be needed; if it holds, calling `gc.collect()`
  between jobs is a cheap operational setting.
- **Cold start.** W2's first iteration took about 2.5x the median in both
  processes (GPU device and synthesizer initialisation, not isolated here).
- **Consistency with Stage 1c.** Since PSF-Zero's output does not vary
  between runs, the small TVD differences between PSF-Zero and Qiskit ZSX in
  Stage 1c (identical gate counts) are consistent with the two producing
  different but equivalent parameter values, not with run-to-run variation.
  Not checked directly.
- **Layout.** The M4 layout placed logical qubit 0 on physical qubit 112 of
  FakeBrisbane for all inputs, the same qubit Stage 1b's M1/M4 chose.

## 5. Files

| File | What it is |
|---|---|
| [`psf-zero/data/longrun_none_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/longrun_none_2026-09-25.csv) | raw data, 100 rows (44,247 bytes as received) |
| [`psf-zero/data/longrun_each_2026-09-25.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/longrun_each_2026-09-25.csv) | raw data, 100 rows (44,502 bytes as received) |
| [`psf-zero/benchmarks/xor_prereg_longrun_stability.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_longrun_stability.py) | the locked script |
| [`psf-zero/benchmarks/make_longrun_figs.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/make_longrun_figs.py) | figure script (sandbox) |
| `longrun_fig1_timing.png`, `longrun_fig2_z0.png`, `longrun_fig3_rss.png` | the three figures |

Pre-publication grep of both CSVs for account names, local paths and host
names: 0 hits (the only machine string is the platform column).
