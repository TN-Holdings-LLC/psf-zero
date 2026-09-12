# On real IBM hardware: fidelity is a wash, compile time is not

**Status:** measured, with one open caveat about how the compile-time number was
produced. Fidelity conclusion is a null result and is reported as one.

---

## The claim

On a 15-qubit QuantumVolume circuit submitted to real IBM backends, PSF-Zero's output
is **indistinguishable from Qiskit `optimization_level=3` in measured fidelity**, and
compiled **14–16x faster** in every run. The fidelity result is a null result, not a
win — and this document says so rather than burying it.

## Setup

`real_device_15q_fidelity_v2` — 15-qubit `QuantumVolume(depth=15, seed=42)`,
decomposed, compiled by both engines, both circuits submitted **together in one
batched job** so they experience identical hardware conditions. 4096 shots.
Fidelity is the Hellinger-style overlap against the classically computed ideal
distribution.

10 independent job submissions split across `ibm_marrakesh` and `ibm_fez`. Every run
logged `105/105` blocks synthesised with **0 fallbacks**, confirming the corrected
`ConsolidateBlocks` path was actually exercised rather than silently no-oping.

## Results (n=10)

| Metric | Qiskit L3 | PSF-Zero |
| :--- | :---: | :---: |
| Fidelity, mean ± SD | 0.0925 ± 0.0016 | 0.0919 ± 0.0016 |
| Circuit depth, mean ± SD | 730 ± 70 | 710 ± 69 |
| 2Q gate count, mean ± SD | 644 ± 14 | 641 ± 11 |
| Compile time, mean ± SD | 2.36 s ± 0.09 s | **0.153 s ± 0.007 s** |

**Fidelity: no difference.** PSF-Zero was higher in 3 of 10 runs; a paired comparison
gives **t = −0.78**, not significant. On this circuit and these two backends we cannot
say PSF-Zero's real-hardware output is either better or worse. Depth and 2Q gate count
were a similar wash (PSF-Zero shorter/fewer in 5 of 10 each).

**Compile time: consistent.** 14.4x–16.2x faster across all 10 jobs, with no overlap
between the two distributions.

A separately recovered 11-run capture of a different script against real hardware
agrees: fidelity 0.0916 ± 0.0021 vs 0.0909 ± 0.0017 (t = −0.68, n.s.), depth
738.5 ± 47.2 vs 685.0 ± 59.2, compile 2.107 s vs 0.159 s (13.3x).

### Job IDs

`daclrrjdd5gc73d68pcg`, `dacls9e42tqs73asccbg`, `daclsstnj4cs73acqm00`,
`daclu3bdd5gc73d68rs0`, `dacluq5nj4cs73acqo70`, `daclv3m42tqs73ascfeg`,
`daclvgrdd5gc73d68thg`, `daclvre42tqs73ascgbg`, `dacm0gtnj4cs73acqq6g`,
`dacm0r642tqs73aschqg`.

## The caveat on the compile-time number

This script calls `transpile()` / `compile_for_hardware()` **exactly once per process**
— one process per job submission — which is structurally the same pattern that
produced this project's now-retracted unrouted compile-time numbers. A fresh
interpreter pays a one-time `transpile()` cost (measured at 2.41 s cold against 0.07 s
warm on an identical 156-qubit circuit elsewhere in this project), and neither side
here benefits from a prior warm-up call.

Two things cut the other way: real-device transpilation at `optimization_level=3`
with full routing against a 127+-qubit backend is inherently much heavier than the
unrouted `compile()` call that was affected before, so the fixed cost is a smaller
share; and both engines pay their own cold start. The number may well hold up.

**It has not been re-measured with a warm-up patch**, because doing so means spending
real QPU time to re-verify a compile-time claim. This is flagged as open rather than
either assumed fine or spent on. See the roadmap.

## Fidelity across four engines under a noise model

Separately, on `fake_sherbrooke` (127-qubit noise snapshot) with mirror circuits —
which return all-zero with probability ~1.0 noiselessly, confirmed for all four
engines first — across three families of increasing 2-qubit depth per pair:

| Family (2Q gates/pair) | Qiskit L3 | TKET | PSF-Zero v6 | Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| `deep2q` (3) | 0.9056 | 0.9077 | **0.8638** | 0.9076 |
| `multi_deep2q` (12) | 0.0849 | 0.0863 | **0.0720** | 0.0839 |
| `wide` (42) | 0.0033 | 0.0039 | 0.0044 | 0.0037 |

<sub>mean P(all-zero) ± SE, n=5.</sub>

PSF-Zero was lowest on the two families where its synthesiser actually runs, by a
margin outside the standard errors. That deficit was chased down to two separate
root causes and fixed — see [`entangling-basis.md`](entangling-basis.md). On `wide`
all four are already near the noise floor and the ordering means nothing.

## The same script against real hardware — and a gap we cannot explain

The identical script was later run with a `--real` flag: same three families, same
block counts per family (`deep2q` 1/1, `multi_deep2q` 4/4, `wide` 0/0, zero fallbacks
throughout, so it is provably the same circuit construction), 5 repeats × 4 engines,
batched as one job per sweep. Four sweeps: three on `ibm_marrakesh`, one on `ibm_fez`.

| Family | Real hardware (4 sweeps) | `fake_sherbrooke` |
| :--- | :---: | :---: |
| `deep2q`, all four engines | 0.9951–0.9955 | 0.864–0.908 |
| `multi_deep2q`, all four engines | 0.9615–0.9618 | 0.069–0.086 |
| `wide`, all four engines | 0.9611–0.9619 | 0.003–0.004 |

Two findings, and at the time they looked contradictory.

**1. No PSF-Zero-specific deficit on real hardware.** In every family the four
engines' means sit within about 0.001 of each other, far tighter than the
sweep-to-sweep spread (0.003–0.008). PSF-Zero's rank bounces between 1st and 4th
across sweeps, which is what noise looks like. The `deep2q` deficit
`fake_sherbrooke` predicted does not appear.

**2. The real-hardware numbers are dramatically higher than the simulator predicted,
for identical circuits, and we do not know why.** On `deep2q` a simulator being
somewhat pessimistic would explain it. On `wide` the gap is close to **two orders of
magnitude** — ~0.96 measured against ~0.003 predicted, for a circuit
`fake_sherbrooke` itself put at the noise floor. Candidates considered: the noise
snapshot being far more pessimistic than either backend's current calibration; a
parameter difference between the local and real runs; or `P(all-zero)` being computed
differently along the two paths. None confirmed.

Individual job IDs were not retained for this batched-submission script (each sweep
submits one batched job of 20 circuits per family), unlike the per-run IDs above.
That is a record-keeping gap, and the reason every current harness now records its
environment and identifiers into its output file.

## What this does and does not support

- Faster compilation does **not** by itself mean higher measured fidelity. The two
  circuits in the 10-run comparison were submitted in one batched job, so both
  experienced identical hardware conditions regardless of how fast either was
  compiled beforehand.
- Where the compile-time advantage would actually pay is total wall-clock cost of a
  workflow that compiles repeatedly — more iterations per unit of session time, not a
  fidelity boost on any single circuit. Whether that reduces exposure to calibration
  drift across a long session is plausible, untested, and not planned without a
  specific reason to spend QPU time.
- One circuit (15-qubit QuantumVolume, seed 42) on two backends. Larger qubit counts
  and more backends are unmeasured.

## Files

- [`benchmarks/real_device_15q_fidelity_v2`](../../benchmarks/real_device_15q_fidelity_v2)
  — the 10-run hardware comparison. A provenance note applies: the listing in
  [`docs/log/05-fidelity.md`](../log/05-fidelity.md) was reconstructed from the captured run log after the original file
  could not be located; parameters, control flow and print statements match the log
  exactly, but it is not guaranteed byte-for-byte.
- [`benchmarks/test_real_hardware_fidelity.py`](../../benchmarks/test_real_hardware_fidelity.py)
  — the four-engine mirror-circuit script (`fake_sherbrooke` and `--real`).
- The five-engine variant used for the `entangling_basis="cx"` comparison is a
  separate file,
  [`benchmarks/test_real_hardware_fidelity_cx.py`](../../benchmarks/test_real_hardware_fidelity_cx.py).


  ## Job records recovered from the platform (2026-09-12)

Sections 7 and 8 both note that their original scripts, and in section 8's case the
per-sweep job IDs, could not be found in the working repository. Two batches of raw
job records (`info.json` + `result.json` per job, fetched from IBM Quantum Platform's
own job history) have since been recovered. Read directly rather than summarized:
`backend`, `created`, `status`, and circuit count come from each `info.json`'s own
fields; circuit count is the length of `result.json`'s `pub_results` array.

**Note added after decoding (below): batch A is not section 7's data — see the
update.** Both batches run on the same day, back to back, in a single 38-minute
window:

| Batch | Jobs | Time span | Backends | Circuits/job |
| :--- | ---: | :--- | :--- | ---: |
| A | 10 (9 Completed, 1 duplicate) | 2026-09-04 12:48–13:00 UTC | marrakesh, fez | 2 |
| B | 15 (14 Completed, 1 Queued) | 2026-09-04 13:11–13:27 UTC | marrakesh, fez | 20 |

**Batch A matches section 7's design** — 2 circuits per job (`[qc_qiskit, qc_psf]`, one
batched submission per run) is exactly what `real_device_15q_fidelity_v2.py`'s `main()`
submits. **Batch B matches section 8's** — 20 circuits per job is what the recovered
`test_real_hardware_fidelity.py` submits per family per sweep.

**The counts do not match what either section states, and that is recorded rather than
smoothed over.** Section 7 says 10 job submissions split across two backends; batch A
has 9 Completed jobs (2 on `ibm_marrakesh`, 7 on `ibm_fez`) plus one further
`ibm_marrakesh` job 8 minutes later — 10 jobs total if that later one belongs to the
same run, but the backend split (2/8, not the roughly even split the prose implies) and
the gap do not obviously fit a single continuous run of 10. Section 8 says 4 sweeps (3
marrakesh, 1 fez); batch B has 15 jobs (12 marrakesh, 3 fez), one of them still
`Queued` rather than `Completed`.

**Job IDs, exactly as recovered, in submission order:**

Batch A (candidate match for section 7):

```
dadbs83dd5gc73d74et0  ibm_marrakesh  2026-09-04T12:48:32Z
dadbsiu42tqs73at7t30  ibm_marrakesh  2026-09-04T12:49:15Z
dadbsujdd5gc73d74fn0  ibm_fez        2026-09-04T12:50:02Z
dadbt8e42tqs73at7u00  ibm_fez        2026-09-04T12:50:41Z
dadbth642tqs73at7u9g  ibm_fez        2026-09-04T12:51:16Z
dadbtqt1ierc738klp40  ibm_fez        2026-09-04T12:51:55Z
dadbu4rdd5gc73d74h30  ibm_fez        2026-09-04T12:52:35Z
dadbudtnj4cs73admb30  ibm_fez        2026-09-04T12:53:11Z
dadbumdnj4cs73admbgg  ibm_fez        2026-09-04T12:53:45Z
dadc21l1ierc738kluqg  ibm_marrakesh  2026-09-04T13:00:54Z
```

Batch B (candidate match for section 8):

```
dadc6rd1ierc738km4f0  ibm_marrakesh  2026-09-04T13:11:09Z  Completed
dadc74d1ierc738km4q0  ibm_marrakesh  2026-09-04T13:11:45Z  Completed
dadc8le42tqs73at8ceg  ibm_marrakesh  2026-09-04T13:15:01Z  Completed
dadc9n3dd5gc73d74vng  ibm_marrakesh  2026-09-04T13:17:16Z  Completed
dadca0e42tqs73at8dvg  ibm_marrakesh  2026-09-04T13:17:53Z  Completed
dadcaardd5gc73d750cg  ibm_marrakesh  2026-09-04T13:18:35Z  Completed
dadcave42tqs73at8f00  ibm_marrakesh  2026-09-04T13:19:57Z  Completed
dadcb8m42tqs73at8fbg  ibm_marrakesh  2026-09-04T13:20:34Z  Completed
dadcbjdnj4cs73admr7g  ibm_marrakesh  2026-09-04T13:21:17Z  Completed
dadccelnj4cs73adms6g  ibm_fez        2026-09-04T13:23:06Z  Completed
dadccnd1ierc738kmbeg  ibm_fez        2026-09-04T13:23:41Z  Completed
dadcd1u42tqs73at8hig  ibm_fez        2026-09-04T13:24:23Z  Completed
dadcdjm42tqs73at8ig0  ibm_marrakesh  2026-09-04T13:25:34Z  Completed
dadcdsl1ierc738kmd10  ibm_marrakesh  2026-09-04T13:26:10Z  Completed
dadce73dd5gc73d75550  ibm_marrakesh  2026-09-04T13:26:52Z  Queued (excluded from counts above)
```

**Update: the bit-array data has now been decoded, and it rules out the
"batch A = section 7" match while it strengthens "batch B = section 8".**
`BitArray.__value__.array` is a zlib-compressed `.npy` (uint8, shots x bytes,
1 bit per classical bit, standard Qiskit `BitArray` packing). Decoding it for every
job in both batches gives:

**Batch A is not section 7.** Section 7 is a 15-qubit circuit; batch A's circuits have
`num_bits` 100 and 156 in every one of its ten jobs. That is a coupling-map-scale
circuit pair, not the 15-qubit `QuantumVolume` circuit `real_device_15q_fidelity_v2.py`
builds. **Batch A does not belong to section 7 and the correspondence proposed above
is withdrawn.** It is most likely part of section 5's real-device compile-time or
gate-count comparison (which does run paired Qiskit/PSF-Zero circuits at 100- and
156-qubit scale), but that has not been checked either — recorded here as an open
question, not asserted.

**Batch B matches section 8, closely.** Decoding `field "c"` for all 14 completed jobs
and computing P(all-zero) gives two clean populations, never mixed within a job:

| `num_bits` | Jobs | P(all-zero) range | Mean of means |
| :---: | ---: | :--- | ---: |
| 2 | 5 | 0.989 – 0.998 | **0.996** |
| 8 | 9 | 0.951 – 0.970 | **0.962** |

Section 8's real-hardware table reports `deep2q` at 0.995 (mean of 4 sweeps) and
`multi_deep2q`/`wide` at 0.961/0.961. The `num_bits=2` jobs land on `deep2q` almost
exactly; the `num_bits=8` jobs land on `multi_deep2q`/`wide` almost exactly. This is
strong, quantitative support that batch B is real data from section 8's experiment —
stronger than the timing/circuit-count match alone.

**It does not resolve which named sweep each job is, or account for all of section
8's structure.** Section 8 describes three families (`deep2q`, `multi_deep2q`,
`wide`) and four engines per family per sweep; batch B's per-job counts show only two
populations by `num_bits` (5 jobs at 2 bits, 9 at 8 bits), not three, and each job's
20 circuits all share one `num_bits` rather than splitting across engines within a
job as section 8's "one batched job of 20 circuits per family" would suggest if a
job corresponds to one family. Whether `multi_deep2q` and `wide` are both foldedinto
the `num_bits=8` population (they report nearly identical hardware fidelity in the
published table, 0.9615 and 0.9611, which would make them hard to tell apart this
way) has not been checked, and no per-circuit metadata recovered here identifies
which engine or family a given one of the 20 circuits in a job is.

**Revised correction:** section 8 states 4 sweeps (3 `ibm_marrakesh`, 1 `ibm_fez`).
Batch B has 14 completed jobs (11 `ibm_marrakesh`, 3 `ibm_fez`) plus 1 `Queued`. If a
"sweep" in the original prose meant one job (20 circuits, matching "one batched job of
20 circuits per family"), the actual count is 14, not 4 — an undercount, not an
overcount, of what was actually run. If a "sweep" meant something coarser (e.g. one
pass through all three families), the mapping from these 14 jobs to "4 sweeps" needs
the family-per-job identification above to be resolved first.

Raw job records (`info.json` + `result.json` pairs, as fetched from the platform):
`data/archive/ibm-jobs-2026-09-04-batch-a/`, `data/archive/ibm-jobs-2026-09-04-batch-b/`.
