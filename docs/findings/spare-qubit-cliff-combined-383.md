# spare-qubit-cliff: Combined Addenda, Part 10 of 10 (Addendum 383 onward)

**Continued from [Part 9](spare-qubit-cliff-combined-248.md) (and [Part 1](spare-qubit-cliff-combined.md), [Part 2](spare-qubit-cliff-combined-17.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 7](spare-qubit-cliff-combined-108.md), [Part 8](spare-qubit-cliff-combined-135.md)).** Same conventions as every prior part.

**Note on this part specifically** (2026-10-07): Part 9 had grown to about 950 KB (17,600 lines), too large to read
in one piece, so a new Part starts here at Addendum 383; Part 9 closes at Addendum 382 and nothing was moved.
The current state at the start of this Part: release `psf_compile.py` 2026-10-06.4 (Addendum 381), AI front end
a12, `psf_smart_layout` 2026-10-01.1, Rust core `CORE_VERSION` 2026-09-29.1. Addendum 212, reserved for the
Stage-2 results of Addendum 211 (Part 8) and not yet recorded, will be recorded in this Part when it is available,
out of numerical order.


---

<!-- ===== Addendum 383 (source: spare-qubit-cliff-addendum-383-2026-10-07.md) ===== -->

> **Note added when merging:** Pre-registration of FUSE; the predictions were written before the first smoke run; the candidate was changed after it (section 6); the lock is the commit that adds this document, pushed after the scored run (no GitHub login at the workplace).

## Addendum 383 -- Pre-registration: candidate psf_compile 2026-10-07.c19 (changelog item 46: fewer, larger matrices in the state-vector estimates and checks) against release 2026-10-06.4, with the recommended call (FUSE); with the profile that motivated it and a defect of the first version found in the smoke run (2026-10-07)

**Status: pre-registration.**

- **Lock:** the git commit that adds this document. It locks the candidate
  [`patches/psf_compile_c19_2026-10-07/psf_compile.py`](../../patches/psf_compile_c19_2026-10-07/psf_compile.py), its
  test [`test_c19.py`](../../patches/psf_compile_c19_2026-10-07/test_c19.py),
  [`benchmarks/fuse_eval.py`](../../benchmarks/fuse_eval.py) (run and score), the runner
  [`benchmarks/run_fuse_2026-10-07.py`](../../benchmarks/run_fuse_2026-10-07.py) and
  [`benchmarks/fuse_verify.py`](../../benchmarks/fuse_verify.py), an independent re-computation of every verdict
  written before any scored output exists (it does not import `fuse_eval.py`).
- **Where and when:** the scored run is made at the workplace (Windows), on 2026-10-07, at the lock commit. **The
  workplace has no GitHub login, so the lock commit is pushed after the scored run**, as in Addendum 366; the run's
  metadata records the commit's hash, and the result Addendum will give both times.
- **No hardware:** fake devices only.
- **The predictions (section 5) are in `fuse_eval.py score`. They were written before the first smoke run** and are
  not changed. The candidate was changed after the first smoke run (section 6); the second smoke run tested the
  locked candidate.

## 1. Why: the 16-qubit cost

SKIP (Addendum 380) found that up to 16 logical qubits the recommended call takes 20-60 s per compile for 16-qubit
Hamiltonian and QFT circuits, against about 0.3 s at 17. An exploratory profile at home (2026-10-06, about 14:10 UTC;
[`benchmarks/cliff16_profile.py`](../../benchmarks/cliff16_profile.py),
[`cliff16_profile.txt`](../../data/2026-10-06/cliff16_profile.txt); SKIP's first circuit of each cell, FakeTorino,
release 2026-10-06.4) located the time:

| | pauli, 16 qubits | qft, 16 qubits |
|---|---|---|
| wall | 53.3 s | 27.1 s |
| `_apply_ops` (item 39's checks: `_same_action` 18.7 s and `_implements` 7.5 s) | 25.9 s | 11.8 s |
| `excitation_cost` (item 35's "select") | 13.7 s | 8.0 s |
| `hybrid_cost` (the choice among candidates) | 12.7 s | 6.6 s |
| Qiskit level 3's transpile | 0.1 s | 0.1 s |

Almost all of it is the release's own state-vector simulation, gate by gate, on up to 16 qubits; at 17 qubits none of
it runs (item 45), and the same circuits take 0.11-0.14 s.

## 2. The candidate (changelog item 46)

[`psf_compile.py`](../../patches/psf_compile_c19_2026-10-07/psf_compile.py) is release 2026-10-06.4 with:

- **checks** (`_same_action`, `_implements`): each single-qubit gate is multiplied into the next gate on its qubit
  that acts on more qubits, and consecutive gates on the same qubits are multiplied into one (`_fuse_ops`, applied in
  `_ops_of`);
- **estimates** (`excitation_cost`, `hybrid_cost`): a single-qubit gate with a diagonal matrix, no reported error
  and no duration (an `rz` on IBM devices) is multiplied into the next gate on its qubit; it adds nothing to either
  estimate and changes no population they read; and a qubit's populations are read in one pass (`_populations`),
  `hybrid_cost` no longer building the reduced density matrix of which it used only the diagonal;
- **ties:** `_choose`, item 35's "select" and item 36's comparison treat two estimates within a relative 1e-12
  (`ESTIMATE_TIE_TOL`) as equal, and a tie keeps the earlier candidate, as an exact tie always did (section 6).

The circuits returned should be 2026-10-06.4's: the checks and estimates compute the same quantities, in a different
order of floating-point operations. `pauli_cost` and `kraus_cost` are unchanged. The name c19, item 46 and this
number were checked unused.

## 3. Development (disclosed)

- **Checked without Qiskit** (numpy only, against the release's own functions): fusion gave the same state on 50
  random 300-gate circuits on 10 qubits (largest infidelity 1.9e-14, 43% of the operations kept); on synthetic
  compiled circuits with mock device properties the two estimates agreed with the release's to 1e-16 relative, and on
  16 qubits and about 3,500 gates `excitation_cost` took 1.4 s instead of 3.3 s and `hybrid_cost` 1.8 s instead of
  4.3 s.
- **[`test_c19.py`](../../patches/psf_compile_c19_2026-10-07/test_c19.py)** (workplace, Windows;
  [`test_c19_log.txt`](../../data/2026-10-07/test_c19_log.txt)): first version 12 passed (fusion's largest infidelity
  2.0e-14; a 16-qubit Hamiltonian on FakeTorino 80.9 s with the release, 39.4 s with the candidate, the same
  circuit); locked version **14 passed** (two tests added for ties, section 6). The tests check: the same action after
  fusion; the populations equal the reduced density matrix's diagonal; on compiled circuits on four devices both
  estimates agree to 1e-12 and both checks give the release's answers, also on a corrupted circuit; the outputs on
  four devices are the release's; the 16-qubit case is faster.

## 4. Design ([`benchmarks/fuse_eval.py`](../../benchmarks/fuse_eval.py))

**Circuits** per device: SKIP's four families, built by SKIP's own generator (`skip_eval.family_circuit`, Addendum
379, unchanged): ring, brick, pauli (one `PauliEvolutionGate` of 3n Pauli strings) and qft, at n = 8, 12, 14, 16 and
20; 3 circuits per (family, n), the first two with `measure_all()`, the third without: **60 per device**, at seeds
82,000,000 + k (smoke: 82,500,000 + k, one per cell), none used before. At n = 20 c19 changes nothing that runs
(item 45 skips the simulations above 16 qubits): it is the control.

**Devices:** FakeTorino, FakeKingston (cz); FakeAuckland, FakeHanoiV2 (cx); FakeBrussels, FakeOsaka (ecr).

**Per device:** one warm-up call of each module on a 4-qubit ring, discarded. **Per circuit,** with the README's
recommended call: the release and c19 in alternating order, each timed; the outputs compared instruction by
instruction (with clbits, parameters, global phase and where the logical qubits start and end); c19's output checked
for instructions or couplings the target lacks and, for ring, brick and qft circuits of 8 qubits, for exactness (the
workplace probe's state infidelity, measurements removed).

**Size:** 6 jobs, one per device, in parallel ([`run_fuse_2026-10-07.py`](../../benchmarks/run_fuse_2026-10-07.py));
then `fuse_eval.py score` and `fuse_verify.py`.

## 5. Predictions (scored only by `fuse_eval.py score`; written before the first smoke run)

**P0:** all of these must hold, or nothing below is scored:

- 6 files of 60 circuits; not smoke; one `git_head`; no uncommitted change to a tracked file; versions 2026-10-06.4
  and 2026-10-07.c19;
- no error; no c19 output off the target;
- every exactness check made (54) and at most 1e-6.

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| F1 | c19 returns the release's circuit | identical on all 360 | more than 2 differ |
| F2 | faster at 16 qubits | median per-circuit time ratio c19 / release <= 0.6 on every device | > 0.85 on any |
| F3 | faster at 12-14 qubits | median ratio <= 0.9 on every device | > 1.05 on any |
| F4 | no slower at 20 qubits (nothing simulated) | median ratio <= 1.10 on every device | > 1.25 on any |

**How the thresholds were set (disclosed):** F1 from the construction, with room for one or two near-ties broken
differently by rounding (the reason section 6 had to deal with). F2 and F3 from the profile and the numpy checks
(section 3): roughly a fifth of the check time and half of the estimate time, less where Qiskit's own compile
dominates. F4: c19 runs no new code there.

**Reported without prediction:** the ratios per family and at n = 8; total compile time per size.

## 6. The smoke runs (seen after the predictions were written)

**First smoke run** (first version of c19, kept as
[`data/2026-10-07/c19_first_version/psf_compile.py`](../../data/2026-10-07/c19_first_version/psf_compile.py);
workplace, 20 circuits per device, 657 s; [`data/2026-10-07/fuse_smoke1/`](../../data/2026-10-07/fuse_smoke1/)): no
error, nothing off the target, every exactness check at most 3e-14; median ratios n = 16 0.33-0.42, n = 12-14
0.53-0.69, n = 20 0.82-1.09, n = 8 0.82-0.96. **One circuit of 120 differed:** FakeKingston, ring, 16 qubits,
measured (seed 82,500,003).

**Diagnosis** ([`benchmarks/fuse_diag.py`](../../benchmarks/fuse_diag.py),
[`fuse_diag_log.txt`](../../data/2026-10-07/fuse_diag_log.txt)): the release kept its own circuit and the candidate
took level 3's. The two have the same placement, the same 30 CZ, and differ by two `rz` gates, which add nothing to
the estimate. The release scored both 0.23233754846574337 (`hybrid_cost`), equal to the bit, and kept the first; the
first version of c19 scored them 0.23233754846574342 and 0.2323375484657434, and took level 3's. Both circuits
implement the input. **An exact tie had been broken by rounding.** Circuits that differ from the release's own only in
free `rz` gates are common at these sizes, so the scored run would likely have refuted F1 for that reason alone.

**The fix:** the tie band of section 2 (`_lower`, `ESTIMATE_TIE_TOL` = 1e-12 relative), in the three places a
candidate is chosen by its estimate. It changes a choice only where the release itself saw two estimates within 1e-12
of each other without their being equal. The predictions were not changed.

**Second smoke run** (the locked candidate; workplace; [`data/2026-10-07/fuse_smoke2/`](../../data/2026-10-07/fuse_smoke2/)):

| device | identical | exact (max) | n = 16 | n = 12-14 | n = 20 | n = 8 |
|---|---|---|---|---|---|---|
| FakeAuckland | 20/20 | 2e-15 | 0.40 | 0.60 | 0.87 | 0.90 |
| FakeBrussels | 20/20 | 4e-15 | 0.30 | 0.59 | **1.12** | 0.84 |
| FakeHanoiV2 | 20/20 | 4e-16 | 0.40 | 0.56 | 1.00 | 1.00 |
| FakeKingston | 20/20 | 2e-16 | 0.35 | 0.67 | 0.91 | 0.98 |
| FakeOsaka | 20/20 | 5e-15 | 0.30 | 0.69 | 0.99 | 0.86 |
| FakeTorino | 20/20 | 2e-16 | 0.35 | 0.61 | 0.86 | 0.86 |

No error, nothing off the target. **FakeBrussels's n = 20 median (1.12) is above F4's line;** it rests on 4 circuits
of 0.0-0.2 s each, where timing noise is large, and the scored run has 12 per device. F4 is not changed.

**Seen, not predicted:** on this machine both arms are slower than at home: the 16-qubit Hamiltonian circuits took
113-241 s with the release (at home in SKIP the same family took a median of 31 s, at most 61 s).

## 7. Files and normalized SHA-256

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c19_2026-10-07/psf_compile.py`](../../patches/psf_compile_c19_2026-10-07/psf_compile.py) | `efb4dac3df771d032ae16338200ff103fe0b7bf98e1c29255d3a6e3b42b561e8` |
| [`patches/psf_compile_c19_2026-10-07/test_c19.py`](../../patches/psf_compile_c19_2026-10-07/test_c19.py) | `b2cbda138d9131cfd6682d5f050300e4b2388515d18cfa82d8b5a65414117aad` |
| [`benchmarks/fuse_eval.py`](../../benchmarks/fuse_eval.py) | `f2e9bdadcb48a39b4d81154298e12a009664089b2a4c8609f2a42dca0361fb03` |
| [`benchmarks/fuse_verify.py`](../../benchmarks/fuse_verify.py) | `d7955aec1439c6026b2d500486447af2dcaf5e39331dac4947cebc9f1c5c6a51` |
| [`benchmarks/run_fuse_2026-10-07.py`](../../benchmarks/run_fuse_2026-10-07.py) | `99ffc6d918bfea1a12f5d97c47a1bc440de827385e491754e4370b447367ec79` |
| [`psf_compile.py`](../../psf_compile.py) (release 2026-10-06.4, unchanged) | `69fe51d2d503638ceb4d067a0d86a5b27c38586694ec84996dea5e7ab6dab7aa` |
| [`data/2026-10-07/c19_first_version/psf_compile.py`](../../data/2026-10-07/c19_first_version/psf_compile.py) (record) | `42ec979bc0480ac55a337076b32cc7c68c610424d7aa947cae652cb5d0c1e6bc` |

**Command** (repository root, at the lock commit): `python benchmarks/run_fuse_2026-10-07.py fuse <out>`.

## 8. Disclosures

- **The predictions were written before the first smoke run;** both smoke runs were seen before this document.
- **The candidate was changed after the first smoke run** (section 6), to keep exact ties as the release keeps them.
  `fuse_verify.py` was changed with it only in the candidate's hash it checks.
- **The lock is pushed after the scored run** (no GitHub login at the workplace).
- **Times are wall-clock times of one process per device, six in parallel,** on one machine.


---

<!-- ===== Addendum 384 (source: spare-qubit-cliff-addendum-384-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 383 (lock commit c4f943b, made before the run and pushed after it), scored by the locked script and re-checked by benchmarks/fuse_verify.py; written after the output was seen.

## Addendum 384 -- Results of FUSE (Addendum 383): candidate c19 takes 0.36-0.49 of release 2026-10-06.4's time at 16 qubits and 0.53-0.65 at 12-14 (F2, F3 CONFIRMED); it returns the release's circuit on 359 of 360, the one difference being a 1.35e-16 near-tie that c19 keeps as a tie by design (F1 AMBIGUOUS); at 20 qubits, where both run the same code, one device's median ratio is 1.104 (F4 AMBIGUOUS) (2026-10-07)

**Status: results of the pre-registered test in Addendum 383, scored by the locked script and re-checked by the
independent [`benchmarks/fuse_verify.py`](../../benchmarks/fuse_verify.py).** Written after the output was seen.

## 1. The run

- **Lock:** commit `c4f943b` (Addendum 383), made at the workplace before the run; every job's `git_head` is
  `c4f943b`, with no uncommitted change to a tracked file. **It was pushed to GitHub after the run** (no GitHub login
  at the workplace), as Addendum 383 said it would be.
- **Machine:** the workplace PC, Windows, Python 3.11.9 (versions and hashes in `env.txt`); six jobs in parallel.
- **Time:** 2,012 s for the slowest job (FakeOsaka), on 2026-10-07 between about 01:00 and 01:34 UTC (`env.txt`,
  `progress.txt`); scored and verified at once.
- **Output:** [`data/2026-10-07/fuse/`](../../data/2026-10-07/fuse/): one json per device (metadata and every
  circuit), the logs, `env.txt`, `progress.txt`, `score.md`, `score_log.txt`, `verify_log.txt`.

## 2. Results (`score.md`)

**P0: PASS.** Six files of 60 circuits; no error; no c19 output off the target; all 54 exactness checks made and at
most 1e-6.

| device | circuits | identical | n = 16: median c19 / release | n = 12-14 | n = 20 | n = 8 |
|---|---|---|---|---|---|---|
| FakeTorino | 60 | 60 | 0.397 | 0.649 | **1.104** | 0.987 |
| FakeKingston | 60 | **59** | 0.418 | 0.606 | 0.933 | 0.982 |
| FakeAuckland | 60 | 60 | 0.485 | 0.530 | 1.035 | 0.964 |
| FakeHanoiV2 | 60 | 60 | 0.460 | 0.530 | 1.035 | 0.950 |
| FakeBrussels | 60 | 60 | 0.357 | 0.607 | 1.004 | 0.981 |
| FakeOsaka | 60 | 60 | 0.359 | 0.646 | 0.991 | 0.948 |

| | prediction | result | verdict |
|---|---|---|---|
| F1 | c19 returns the release's circuit | 359 of 360 (1 differs; REFUTED needed more than 2) | **AMBIGUOUS** |
| F2 | n = 16: median ratio <= 0.6 on every device | 0.357-0.485 | **CONFIRMED** |
| F3 | n = 12-14: median ratio <= 0.9 on every device | 0.530-0.649 | **CONFIRMED** |
| F4 | n = 20: median ratio <= 1.10 on every device | 0.933-1.104 (REFUTED needed more than 1.25) | **AMBIGUOUS** |

`fuse_verify.py`: P0 PASS, the same four verdicts, "verdicts identical to score.md: True".

**Reported without prediction:**

- **Per family** (median ratio at n = 16 | n = 12-14): ring 0.455 | 0.735, brick 0.350 | 0.578, pauli 0.469 | 0.520,
  qft 0.410 | 0.531.
- **Total compile time per size** (release, c19): n = 8 47.1 s, 44.9 s (0.953); n = 12 363.8 s, 185.3 s (0.509);
  n = 14 686.6 s, 353.1 s (0.514); **n = 16 5,982.2 s, 2,495.1 s (0.417)**; n = 20 7.6 s, 7.8 s (1.029).

## 3. The two AMBIGUOUS verdicts (exploratory, after the run)

[`benchmarks/fuse_run_diag.py`](../../benchmarks/fuse_run_diag.py) rebuilt the differing circuit and printed the n = 20
ratios ([`fuse_run_diag_log.txt`](../../data/2026-10-07/fuse_run_diag_log.txt)).

**F1.** The one difference is FakeKingston, ring, 8 qubits, measured (seed 82,000,001), and it reproduces. The
release took level 3's circuit and c19 kept its own. The two have the same placement and the same 14 CZ and differ
only in `rz` gates (28 against 32), which add nothing to the estimates. The release scored level 3's lower by
1.35e-16 (relative), a rounding difference, and took it; c19 treats estimates within 1e-12 as a tie and keeps the
earlier candidate. Both circuits implement the input. This is the one case item 46 names in advance as able to
differ ("where 2026-10-06.4 saw two estimates differ by less than 1e-12 without being equal"), and the margin F1
allowed for. Compared with the release, c19 is the consistent one here: the release chose between two circuits of
equal estimated cost by the direction of a rounding error; in the smoke run (Addendum 383) the same situation went the
other way, to the release's own circuit, on an exact tie.

**F4.** At 20 qubits both arms run the same code (item 45 skips every simulation above 16 qubits, so nothing c19
changed is reached). The 12 circuits per device took 0.01-0.32 s each; FakeTorino's ratios ranged 0.81-2.12 and their
median was 1.104, just over the line; the other devices' medians were 0.933-1.035. This is timing noise on short
calls with six jobs in parallel, not a cost of c19.

## 4. What this shows

1. **c19 roughly halves the recommended call's time where it simulates:** 0.36-0.49 of the release's time per device
   at 16 qubits (42% of the total there), 0.53-0.65 at 12-14, and nothing changes at 8 or at 20.
2. **The circuits are the release's,** except where the release's choice was decided by rounding between candidates
   of equal estimated cost; there c19 keeps the release's own circuit.
3. **16 qubits remain expensive on this machine:** in the smoke runs a 16-qubit Hamiltonian still took about 40-85 s
   with c19 (113-241 s with the release). The remaining time is the state-vector work itself.

## 5. Disclosures

- **The lock was pushed after the run** (section 1).
- **The predictions were written before the first smoke run;** the candidate was changed after it (Addendum 383,
  section 6).
- **Section 3 is exploratory,** after the output was seen.

---

---

**End of Part 10 of 10 (end of document, for now).** Back to [Part 9](spare-qubit-cliff-combined-248.md), [Part 8](spare-qubit-cliff-combined-135.md), [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
