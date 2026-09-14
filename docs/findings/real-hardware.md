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
