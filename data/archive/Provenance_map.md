# Provenance map for `data/archive/`

**What this is.** A file-by-file account of the raw CSVs kept under
`data/archive/` — which published table each one produced, and which of them
support claims this project has since retracted.

**Why the archive exists.** Most of the files here are *superseded or retracted*
measurements. The README says three headline numbers were withdrawn; keeping the
data that produced them is what lets anyone check that the withdrawals were
correct. The purpose is re-checking, not reuse. If you quote a number from here,
say which era it belongs to.

Filenames are left **exactly as the harness wrote them**, so they can be matched
against a working directory. Current (non-archived) data lives in `data/` and is
listed in the second half of this file.

---

## 1. What the archive established, and one thing it got wrong

This project's rule is that a number is published with its source. Some of the
README's earlier claims were backed only by prose, and the underlying runs were
later recovered. They are archived here, and each was matched to the published
table it produced by recomputing the numbers.

Two record-keeping failures are visible across the set:

- **Ten of the thirteen original files record no environment metadata at all** —
  no CPU string, platform, Python or Qiskit version. Those columns were added to
  the harnesses later.
- **`test1_v3.py` wrote to a fixed filename** (`phase1_v3_benchmark_results.csv`),
  so every run overwrote the last. Only the fourth run's output survives.

### Correction (2026-09-13): "unrecoverable" was wrong

This file previously concluded that the machine behind the first `test1_v3.py`
declining run (4.35x / 2.21x / 1.66x / 1.44x) was **permanently unrecoverable**,
and that no replacement hypothesis could ever be tested. That conclusion is
withdrawn. It was true that the artifact is gone; it did not follow that the
question was closed.

On 2026-09-10 the same script, with the current `psf_compile.py`, reproduced the
table at 4.23x / 2.11x / 1.63x / 1.39x on the Windows /
`AMD64 Family 25 Model 80` machine — within 3–5% at every point, output quality
identical. A layered diagnostic then located the cause: that machine's installed
`psf_zero_core` build predates `geometric_decompose_checked`, so its `verify=True`
default silently falls back to a numpy reconstruction costing ~0.111 ms per
synthesised block. Full account in
[`docs/findings/compile-time.md`](../../docs/findings/compile-time.md), addendum
of 2026-09-13.

**The lesson is narrower than "the archive was insufficient".** The variable that
explained the anomaly — which build of the Rust core was loaded — was recorded by
*none* of these files, and is still not recorded by most current harnesses. Every
harness logs platform, Python, Qiskit and numpy. Only
`diag_canonical_penalty.py` logs `core_file` / `core_bytes` / `core_ext`. The
others should.

---

## 2. Archived files

"Env" = the file carries CPU / Platform / Python / Qiskit / StartMethod columns.

| File | Rows | Env | What it is / which published claim it backs |
| :--- | ---: | :---: | :--- |
| `phase1_benchmark_results.csv` | 80 | — | **Source data for the retracted "200x".** Qiskit flat at 1.30–1.34 s across 15/50/100/156q — the no-op `transpile()` bug itself. PSF 0.005–0.039 s, making it look like ~250x. |
| `phase1_v2_benchmark_results.csv` | 80 | — | The corrected `phase1.py` re-run. PSF median 1.08/2.60/5.10/8.84 ms, Qiskit 10.23/16.05/22.34/29.59 ms → 9.5/6.2/4.4/3.4x. |
| `phase1_v3_benchmark_results.csv` | 160 | **yes** (Intel) | `test1_v3.py`, **run 4** (10 seeds × 5 reps, 4 arms). The fixed filename means no earlier run of this script survives. |
| `phase2_benchmark_results.csv` | 40 | — | The corrected `phase2.py` re-run. Mean Qiskit 29.85/46.46/72.67/136.58 ms, PSF 10.09/15.94/25.15/49.06 ms (156–1000q). |
| `phase2_deadzone_results.csv` | 24 | — | Deadzone v1. Qiskit 1.40–2.44 s (cold-start dominated), PSF 0.17–1.17 s. |
| `phase2_v2_deadzone_results.csv` | 24 | — | **Primary evidence for the `force_consolidate` bug.** PSF degrades with scale to 6.34 s at 1000q against Qiskit's 1.55 s — **4.1x slower**. The measurement that triggered the whole re-investigation. |
| `phase3_physical_topology_results.csv` | 30 | — | phase3 v1. Qiskit nearly flat at 1.45–3.44 s — a `random_circuit`-era plus cold-start artifact. |
| `phase3_v2_physical_topology_results.csv` | 30 | — | phase3 v2. Both engines 1.3–4.1 s; cold start dominates both sides. |
| `phase3_v3_physical_topology_results.csv` | 30 | — | phase3 v3. **PSF slower than Qiskit at every scale** (50q: 0.034 vs 0.023 s). The state `phase3-hardware-routing-regression.md` diagnosed as "measuring circuits PSF passes through untouched". |
| `phase3_v3_physical_topology_results_level0.csv` | 30 | — | The `level=0` variant of v3. Both engines 1.32–1.54 s. |
| `phase3_v4_physical_topology_results.csv` | 120 | **yes** (Intel) | phase3_v4 on the Intel machine. Same content as `data/phase3_v4_intel_machine_2026-09-10.csv`. |
| `phase3_v5_seeded_results.csv` | 100 | — | One of the two runs pooled into the seed-pinned 20-measurements-per-scale table. Mean Qiskit 19.6/32.3/38.7/153.4/199.7 ms, PSF 16.3/23.6/36.9/112.1/172.6 ms. Its 500q PSF maximum is 0.180 s, identifying it as the run **without** the 0.825 s outlier. |
| `phase3_v5_spare_qubits_results.csv` | 32 | **yes** (Intel) | Spare-qubit controlled experiment (axis C), Intel machine. Same content as `data/phase3_v5_spare_qubits_intel_2026-09-10.csv`. |
| `ibm-jobs-2026-09-04-batch-a/` | 10 jobs | n/a | Raw IBM Quantum Platform job records (`info.json` + `result.json`). **Not** section 7's data — decoding shows 100/156-qubit circuits. Section attribution open; see [`real-hardware.md`](../../docs/findings/real-hardware.md). |
| `ibm-jobs-2026-09-04-batch-b/` | 15 jobs | n/a | Raw job records matching section 8 quantitatively (P(all-zero) 0.996 at `num_bits=2`, 0.962 at `num_bits=8`). |

### The retraction is visible in the data

Section 4 diagnosed the "200x" as Qiskit doing nothing, because `transpile()` was
called without `basis_gates`. `phase1_benchmark_results.csv` shows it directly:

```
Qubits:      15      50      100     156
Qiskit:   1.339s  1.308s  1.303s  1.333s   <- independent of circuit size
PSF-Zero: 0.005s  0.016s  0.025s  0.039s   <- proportional to circuit size
```

A tenfold larger circuit costs Qiskit the same 1.33 s. That is impossible if it
were compiling, and is exactly what measuring a fixed cold-start cost looks like.
The retraction was correct.

### Cross-checks that were recomputed

| CSV | Published table | Published | Recomputed | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `phase1_v2_benchmark_results.csv` | §4, `phase1.py`, 15–156q, median of 10 seeds | 9.51 / 6.16 / 4.38 / 3.35 | 9.51 / 6.16 / 4.38 / 3.35 | **exact** |
| `phase2_benchmark_results.csv` | §4, `phase2.py`, 156–1000q, mean ± sd | 2.96 / 2.91 / 2.89 / 2.78 | 2.96 / 2.92 / 2.89 / 2.78 | match (0.01 rounding at 300q) |
| `phase2_v2_deadzone_results.csv` | §4, "roughly 4x slower at 1000q" (`force_consolidate` era) | roughly 4x | 4.14x (6.47 s vs 1.56 s) | **match; this is the source** |
| `phase3_v5_seeded_results.csv` | §5, seed-pinned, 20 measurements/scale | 1.23 / 1.28 / 1.03 / 1.37 / 1.21 | 1.20 / 1.37 / 1.05 / 1.37 / 1.16 | consistent (published pools two runs; this is one) |
| `phase3_v3_physical_topology_results_level0.csv` | §5, gate-count/depth table, `level=0` column | 75/9, 150/9, 306/~24, 450/9, 992/~30 | 75/9, 150/9, 306/22, 450/9, 992/31 | match, within the stated `~` |
| `phase1_benchmark_results.csv` | §4, the **retracted** "up to ~200x" | (retracted) | apparent 258 / 85 / 52 / 34x | see above |

---

## 3. Current data in `data/` (not archived)

These are the files behind numbers the project currently stands behind. Listed
here because there was no other index of them.

### Compile time

| File | What it is |
| :--- | :--- |
| `phase1_v3_test1_v3_intel_run2_2026-09-10.csv` | `test1_v3.py` run 5 (Intel) |
| `phase1_v3_test1_v3_amd_2026-09-10.csv` | `test1_v3.py` on the AMD machine — **reproduces the declining table** (4.23/2.11/1.63/1.39) |
| `phase1_v3_verify_strict_intel_2026-09-10.csv` | The paired `verify="strict"` comparison that refuted the old-`psf_compile.py` hypothesis |
| `cumulative_50k_intel_2026-09-10_summary.csv`, `..._per1000.csv` | The 50,000-iteration loop, Intel |
| `cumulative_50k_preverifychange_per1000.csv` | The same loop before the `verify` split |
| `diag_canonical_penalty_intel_v2_2026-09-11.csv` | Layered diagnostic, Intel — seven layers × 2 thread arms × 3 seeds |
| `diag_canonical_penalty_amd_v2_2026-09-13.csv` | The same on AMD. `core_raw_checked` is `skipped` in all 6 attempts here and `success` in all 6 on Intel — the observation that identified the stale core |
| `determinism_variance_2026-09-13.csv` | 1,000 repeated compiles of one fixed SU(4), 3 engines |
| `framework_overhead_2026-09-13.csv` | Per-pass / per-stage breakdown; ~47% unattributable on Windows timer resolution |

### The Qiskit coupling-map cliff

| File | What it is |
| :--- | :--- |
| `phase3_v4_intel_machine_2026-09-10.csv` | The sweep the effect first appeared in |
| `phase3_v5_spare_qubits_linux_2026-09-10.csv` | Controlled experiment, Linux sandbox |
| `phase3_v5_spare_qubits_intel_2026-09-10.csv` | Same, Intel |
| `phase3_v5_spare_qubits_amd_2026-09-10_run1.csv`, `_run2.csv` | Same, AMD, twice back to back |
| `phase3_v6_passthrough_control_2026-09-11.csv` | The dense-vs-`random_circuit` workload control — no cliff in the passthrough arm |
| `qiskit_version_sweep_2026-09-11.csv` | Six Qiskit releases, 1.4.6 → 2.5.2 |
| `qiskit_version_step_confirm_2026-09-11.csv` | The 2.0.3 → 2.1.2 step, 3 seeds × 3 reps |
| `qiskit_pass_timing_2026-09-11.csv` | Per-pass timing: 99.9% in `VF2Layout` + `VF2PostLayout` |
| `vf2_seed_scan_2026-09-11.csv` | `shuffle_seed` 0–29 against `VF2Layout`: 4/30 `SOLUTION_FOUND` |
| `vf2post_seed_scan_2026-09-11.csv` | The same against `VF2PostLayout`: the **same four seeds** |
| `vf2_max_trials_2026-09-11.csv` | `max_trials=1` vs default — separates finding a layout from finishing the search |
| `preset_shuffle_2026-09-11.csv` | The negative result: timing cannot detect a lucky ordering |
| `preset_stop_reason_2026-09-12.csv` | Stop reasons through the preset: 0/30 |
| `vf2_target_scoring_2026-09-12.csv` | Dummy vs real target — scoring ruled out |
| `rustworkx_vf2_id_order_2026-09-11.csv`, `rustworkx_vf2_min_case_scan_2026-09-11.csv` | The parallel `rustworkx.vf2_mapping` observation. **Qiskit does not call this function** — see the "Reported upstream, and rejected" section of [`spare-qubit-cliff.md`](../../docs/findings/spare-qubit-cliff.md) |

### Fidelity and core accuracy

| File | What it is |
| :--- | :--- |
| `core_verification_2026-09-12.csv` | Haar-random 500 + near-CNOT 200, after the `near_cnot()` global-phase fix |
| `hw_fidelity_raw_2026-09-10.csv` | Real-hardware fidelity, **per repeat** (60 rows) — the first data supporting a paired test |
| `hw_fidelity_summary_2026-09-10.csv` | The same, aggregated (12 rows) |

---

## 4. Personal-information check

Every file listed here was checked against the project's pre-publication pattern
set (user name, organisation domain, `C:\Users`, `AppData` and similar). Zero
hits. None of the CSVs contains a path column.

The pattern set itself is kept in the project's internal working document, not
here — see [`record-keeping.md`](../../record-keeping.md) for the conventions
that are public.
