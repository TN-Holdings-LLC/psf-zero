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

<!-- ===== Addendum 385 (source: spare-qubit-cliff-addendum-385-2026-10-07.md) ===== -->

> **Note added when merging:** The owner's decision on the evidence of Addendum 384: c19 accepted, not released; release policy from 2026-10-07.

## Addendum 385 -- Decision: candidate c19 (changelog item 46) is accepted but not released; from now on, speed and quality improvements are released together, correctness fixes at once (2026-10-07)

**Status: the owner's decision, on the evidence of Addendum 384.**

## 1. The decision

- **c19 is accepted.** FUSE (Addendum 384) confirmed that it roughly halves the recommended call's time where it
  simulates (F2, F3), and found its one difference from 2026-10-06.4 to be the near-tie it keeps by design (F1); F4's
  excess was timing noise on code both run. [`patches/psf_compile_c19_2026-10-07/psf_compile.py`](../../patches/psf_compile_c19_2026-10-07/psf_compile.py)
  (normalized SHA-256 `efb4dac3...`) stays where it is.
- **It is not released now.** [`psf_compile.py`](../../psf_compile.py) stays 2026-10-06.4. c19 will be released
  together with the next accepted improvements, in one release.
- **The next candidates are built on c19,** not on 2026-10-06.4, and are tested against c19 as well as against the
  release, so that what is finally released has been tested as one file.

## 2. The release policy from now on

Four releases in one day (2026-10-06.1 to .4) were each tested, but are hard to follow from outside. From 2026-10-07:

- **Correctness fixes** (a wrong circuit, an exception, a crash, a hang) are released at once, as 2026-10-06.2 (item
  43) and 2026-10-06.3 (item 44) were.
- **Speed and quality improvements** are accepted as candidates, recorded in an Addendum, and released together,
  when a batch is complete or before a pre-registered comparison against other compilers (BP-MOCK).
- [`docs/RELEASES.md`](../../docs/RELEASES.md) lists the accepted, unreleased candidates above the current version, so
  a reader can see what the next release will contain.

## 3. What changes in the repository

- `docs/RELEASES.md`: a short block "Accepted, not yet released" naming c19 (item 46, Addenda 383-385) above the
  current version. Nothing else.

This follows advice from a separate review of the record (the workplace session of 2026-10-07).


---

<!-- ===== Addendum 386 (source: spare-qubit-cliff-addendum-386-2026-10-07.md) ===== -->

> **Note added when merging:** Exploratory work of 2026-10-07 at the workplace, after Addendum 385; written after the output was seen.

## Addendum 386 -- DISPATCH-PROBE and DISPATCH-PROBE 2 (exploratory): most of the default call's gap to Qiskit level 2 on general circuits comes from instructions on three or more qubits; candidates c20 (item 47), c21 (item 48) and c22 (item 49) and their exploratory tests (2026-10-07)

**Status: exploratory, not pre-registered. Written after all the output below was seen.** Nothing here is a test of
a prediction; Addendum 387 pre-registers the test of the speed items, and the quality item is to be tested in BP-MOCK.

## 1. Why

After Addendum 385 the owner asked for two things: (1) the default call's quality against Qiskit level 2 on general
circuits, where BP-PROBE (Addendum 377) found it behind; (2) a shorter time for the recommended call at 16 qubits,
which after c19 still took about 40-85 s on the workplace PC for a Hamiltonian (Addendum 384, section 4).

## 2. DISPATCH-PROBE: where the default call loses to level 2

[`benchmarks/dispatch_probe.py`](../../benchmarks/dispatch_probe.py) was written to find a rule by which the default
call could hand a circuit to level 2. It was run at the workplace from the project folder outside the repository and
is copied here unchanged. Circuits: BP-PROBE's 14 Benchpress tests (built as Benchpress builds them, Benchpress at
`b695f30`), SKIP's four families at 12 and 40 qubits on FakeTorino, and PL-GPU-REDO's family T at spare 0 and 4 on
FakeAuckland and spare 0 on FakeKingston: 23 circuits. Arms, all without a target: PSF (candidate c19's default
call), PSFH (the same after unrolling the input to the basis with level 0), L2 (`transpile(..., optimization_level=2)`)
and PSF2 (c19 with `routing_optimization_level=2`). Every output was valid (basis gates, two-qubit gates on
couplings). Output: [`data/2026-10-07/dispatch/`](../../data/2026-10-07/dispatch/).

Two-qubit gates, as a ratio to level 2's (below 1: PSF-Zero has fewer):

| circuit | instruction on 3+ qubits | PSF | PSFH | PSF2 |
|---|---|---|---|---|
| EfficientSU2, 100 qubits | EfficientSU2 | 6.29 | 1.00 | 6.29 |
| barenco_tof_10 | ccx | 1.99 | 0.99 | 1.99 |
| HamLib (two) | PauliEvolution | 1.68, 1.90 | 1.63, 1.10 | 1.08, 1.15 |
| SKIP pauli 12 / 40 | PauliEvolution | 1.62 / 1.31 | 0.97 / 1.09 | 1.56 / 1.29 |
| SKIP qft 12 / 40 | qft | 1.49 / 2.23 | 1.25 / 1.29 | 1.01 / 1.31 |
| QFT, 100 qubits (flat) | none | 1.38 | **1.79** | 1.07 |
| bv_n14, basis_trotter_n4 | none | 1.12, 1.11 | 1.12, 1.41 | 1.00, 1.00 |
| QV 100, QASMBench 32-linear | QV layer / none | 1.04, 1.01 | 1.02, 1.01 | 1.03, 1.02 |
| ring, brick (4), adder_n4 | none | 1.00 | 1.00 | 1.00 |
| family T: Auckland spare 0, Kingston spare 0 | none | **0.94, 0.84** | 0.94, 0.84 | 0.94, 0.84 |

Readings:

- **Most of the gap comes from instructions on more than two qubits.** The default call handed them to its pipeline
  as they were; unrolling the input first removed most of the gap.
- **Unrolling everything breaks two-qubit structure the pipeline uses:** on the flat 100-qubit QFT the ratio rose
  from 1.38 to 1.79.
- **`frac_deep`** (the share of CX in same-pair blocks of at least 4 CX) **does not separate the cases:** the
  four-qubit Trotter circuit has 0.97 and level 2 wins; family T has 1.0 and PSF-Zero wins. No dispatch rule by
  these features was found.
- **PSF-Zero's own case holds:** on family T at full occupancy level 2 meets the coupling-map cliff (FakeKingston:
  327 two-qubit gates and depth 15 against 276 and 6).

## 3. DISPATCH-PROBE 2: expanding only the wide instructions

[`benchmarks/dispatch_probe2.py`](../../benchmarks/dispatch_probe2.py) (same circuits, imported from the first
probe) measured PSFU (Qiskit's `Unroll3qOrMore` first: only instructions on three or more qubits are expanded),
PSFU2 (PSFU with routing level 2) and PSFH2 (PSFH with routing level 2), and repeated PSF and L2. It was run twice;
both runs gave the same two-qubit count and depth in every row, and the saved files are the second run.

| arm | geometric mean of the ratio to level 2 (23 circuits) | largest | fewer / more than level 2 |
|---|---|---|---|
| PSF | 1.341 | 6.29 | 2 / 15 |
| PSFU | 1.045 | 1.38 | 3 / 13 |
| PSFU2 | 0.991 | 1.07 | 5 / 2 |
| PSFH2 | 1.024 | 1.30 | 5 / 8 |

- **PSFU had no more two-qubit gates than PSF on all 23;** on circuits without wide instructions it changed nothing.
- **PSFU2 matched level 2 gate for gate on many rows** (same two-qubit count and depth). This is the effect the
  docstring of `routing_optimization_level` records: at level 2 Qiskit's preset re-runs block consolidation and
  synthesis and the result is level 2's. At full occupancy, raising the level also brings back the cliff (Addenda
  24-25); family T did not show it here only because the layout search placed it without routing. **Routing level 2
  as a default is therefore not proposed;** it would give up what the default call is for. If it is ever considered,
  it is a separate item to be timed on the cliff first.

## 4. Candidates (all with changelog entries in the file)

| candidate | item | what it changes | based on |
|---|---|---|---|
| [c20](../../patches/psf_compile_c20_2026-10-07/psf_compile.py) | 47 | item 39's checks are made only where their result can change the output (estimates first) | c19 |
| [c21](../../patches/psf_compile_c21_2026-10-07/psf_compile.py) | 48 | instructions on 3+ qubits expanded with `Unroll3qOrMore` before PSF-Zero's own pipeline | c20 |
| [c22](../../patches/psf_compile_c22_2026-10-07/psf_compile.py) | 49 | `excitation_cost` and `hybrid_cost` follow single-qubit gates on each qubit's 2x2 reduced state | c21 |

Exploratory checks, all at the workplace (Windows, Python 3.11.9, core 2026-09-29.1), files in
[`data/2026-10-07/c20_c22/`](../../data/2026-10-07/c20_c22/):

- **c20.** [`test_c20.py`](../../patches/psf_compile_c20_2026-10-07/test_c20.py): 12 passed, twice. Outputs equal
  c19's, including with every item 39 check forced to fail or to pass. The checks made fell from 24 to 7-13 per
  device. [`c20_timing.py`](../../benchmarks/c20_timing.py) (24 circuits, SKIP's families at 12-16 qubits, seeds
  47,500,000 + k): all identical in both runs, total time at 16 qubits 310 → 274 s and 381 → 367 s (median ratios
  0.879 and 0.946; the saved log is the second run). The gain was smaller than expected because after item 46 the
  estimates, not the checks, took most of the time: [`c20_profile.py`](../../benchmarks/c20_profile.py) put 67-74 s of
  78-87 s of a 16-qubit Hamiltonian (two runs) in `excitation_cost` and `hybrid_cost` (reading populations and applying
  single-qubit gates to the whole state).
- **c21.** [`test_c21.py`](../../patches/psf_compile_c21_2026-10-07/test_c21.py): 9 passed (a first version used
  30-qubit circuits on the 27-qubit FakeHanoiV2 and failed for that reason in both modules; fixed to 24 qubits).
  Circuits without wide instructions: c20's output, default and recommended call. With wide instructions, the default
  call's two-qubit count fell on 26 of 27 circuits (by 10-48%; for example qft at 24 qubits on FakeTorino 1,725 →
  926) and rose by one on one (FakeHanoiV2, pauli at 8 qubits: 295 → 296); every output valid and, up to 16 qubits,
  implementing the input wherever c20's did. An instruction that cannot be expanded gives the same exception as c20.
- **c22.** [`test_c22.py`](../../patches/psf_compile_c22_2026-10-07/test_c22.py): 12 passed, twice; both estimates
  agree with c21's to 3.1e-15 (relative) or better on four devices; outputs equal c21's.
  [`c22_timing.py`](../../benchmarks/c22_timing.py): all 24 identical in both runs; total at 16 qubits 89.0 → 39.6 s
  and 79.9 → 33.0 s (median ratios 0.454 and 0.512), at 14 qubits 0.645 and 0.711.

**Something to watch (from `c22_timing`):** two of the 16-qubit pauli circuits took 0.5 s with c21 and c22 and made
no check. After expansion their routed circuit touched more than 16 qubits, so the recommended call made no estimate
and returned its own circuit without comparing it with level 3's (items 36-38). With c20 the same circuits may have
been compared. Whether item 48 lowers the recommended call's quality on some circuits this way is one of the things
BP-MOCK must measure.

## 5. What this shows and what it does not

- The DISPATCH probes chose item 48, so they cannot test it. Its effect is to be measured on circuits not used here
  (BP-MOCK, to be pre-registered).
- The timings come from one workplace PC under varying load (the same 16-qubit call took 50 s in one run and 106 s in
  another); they set the predictions of Addendum 387 and are not results.

## 6. Disclosures

- The probes were run from the project folder outside the repository; the copies here are the files that ran.
- Runs that were repeated overwrote their logs; the numbers of the first runs quoted above are from the output pasted
  into the session.
- Local paths in the saved logs are replaced by `<windows-home>`.


---

<!-- ===== Addendum 387 (source: spare-qubit-cliff-addendum-387-2026-10-07.md) ===== -->

> **Note added when merging:** Pre-registration of TRACK, committed with its smoke run as the lock before the scored run.

## Addendum 387 -- Pre-registration of TRACK: does candidate c22 (items 47-49) return candidate c19's circuit with the recommended call on inputs without instructions on three or more qubits, in 0.6 of its time or less at 16 qubits? (2026-10-07)

**Status: pre-registration, written before TRACK's smoke run and before any of its output exists.** The lock is the
commit that adds this Addendum together with the smoke run's output; the scored run follows that commit.

## 1. Question

Candidate c22 carries three items on top of candidate c19 (accepted, not released; Addendum 385). Two of them are
meant to change only the time of the recommended call:

- item 47: item 39's checks only where their result can change the output (c20);
- item 49: the estimates follow single-qubit gates on each qubit's 2x2 reduced state (c22).

The third, item 48 (c21), changes what is returned for inputs with an instruction on more than two qubits, and is
not tested here (BP-MOCK). TRACK asks whether c22 returns c19's circuit with the recommended call on inputs that item
48 does not touch, and how much faster it is.

## 2. Design

[`benchmarks/track_eval.py`](../../benchmarks/track_eval.py), run by
[`benchmarks/run_track_2026-10-07.py`](../../benchmarks/run_track_2026-10-07.py), checked by the independent
[`benchmarks/track_verify.py`](../../benchmarks/track_verify.py) (reads the raw json only). It is FUSE's design
(Addendum 383) with c19 and c22 as the two arms, new seeds and one change to the inputs:

- **Circuits:** SKIP's four families (ring, brick, pauli, qft; `skip_eval.family_circuit`, unchanged) at 8, 12, 14,
  16 and 20 qubits, 3 per (family, size), the first two measured: 60 per device, seeds 85,000,000 + k (smoke:
  85,500,000 + k, one per cell). These seeds have not been used.
- **Expansion:** every circuit is first passed through Qiskit's `Unroll3qOrMore`, so that it has no instruction on
  more than two qubits (SKIP's pauli and qft circuits have them; ring and brick do not). Both arms compile the same
  expanded circuit, and c22's item 48 leaves it as it is. P0 checks that no input has such an instruction.
- **Devices:** FakeTorino, FakeKingston (cz), FakeAuckland, FakeHanoiV2 (cx), FakeBrussels, FakeOsaka (ecr).
- **Per circuit:** c19 and c22 in alternating order, each timed, with the README's recommended call; the outputs
  compared instruction by instruction (clbits, parameters, global phase, initial and final layout); c22's output
  checked for instructions or couplings the target lacks and, for ring, brick and qft at 8 qubits, for exactness (the
  workplace probe's state infidelity, at most 1e-6).
- **Run:** one job per device, in parallel, on one machine, from the lock commit with no uncommitted change to a
  tracked file.

## 3. Predictions

| | prediction | CONFIRMED | REFUTED |
|---|---|---|---|
| P0 | the run is valid | six files of 60, no error, no input with a wide instruction, no c22 output off the target, all 54 exactness checks made and at most 1e-6, one git head | otherwise nothing is scored |
| T1 | c22 returns c19's circuit | no circuit differs | more than 2 of 360 differ |
| T2 | n = 16: median per-circuit time ratio c22 / c19 | <= 0.6 on every device | above 0.8 on any device |
| T3 | n = 12-14 | <= 0.85 on every device | above 1.05 on any device |
| T4 | n = 20 (nothing changed runs) | <= 1.15 on every device | above 1.30 on any device |

Between the two columns the verdict is AMBIGUOUS.

**Where the numbers come from.** The exploratory timings of Addendum 386 (c20 against c19: median 0.88-0.95 at 16
qubits; c22 against c21: 0.45-0.51 at 16 qubits, 0.65-0.71 at 14, 0.81-0.89 at 12) suggest about 0.4-0.5 at 16
qubits and about 0.7 at 12-14. The thresholds leave room for the workplace PC's load and for circuits whose routed
form touches more than 16 qubits, where neither arm simulates and the ratio is about 1. T1 allows 2 differences, as
FUSE's F1 did, for estimates within rounding of the tie band. T4's line is wider than FUSE's F4 (1.10), whose
AMBIGUOUS verdict came from timing noise on calls of 0.01-0.3 s (Addendum 384); the change is made before any TRACK
output exists.

## 4. Files locked

Normalized SHA-256 (CRLF to LF, trailing spaces and trailing blank lines removed):

| file | normalized SHA-256 |
|---|---|
| `patches/psf_compile_c19_2026-10-07/psf_compile.py` | `efb4dac3df771d032ae16338200ff103fe0b7bf98e1c29255d3a6e3b42b561e8` (as locked in Addendum 383) |
| [`patches/psf_compile_c22_2026-10-07/psf_compile.py`](../../patches/psf_compile_c22_2026-10-07/psf_compile.py) | `30675c37e6c9400803ed4d53c8d146ecdebec24b9452037808c9e9acf150f858` |
| [`benchmarks/track_eval.py`](../../benchmarks/track_eval.py) | `3a574d5f87e77a18337eb77156b999b2b3ac64ad73b1d5dea29060ec5238d7bd` |
| [`benchmarks/track_verify.py`](../../benchmarks/track_verify.py) | `094070dffc274d306fa2845f96c6132129d9e51c4eab6ddc54b0d4804f3935a9` |
| [`benchmarks/run_track_2026-10-07.py`](../../benchmarks/run_track_2026-10-07.py) | `762d8b1ea4fb308bca7076abd1ca58a2d115d458e2529a4e601ddae0951eb7a7` |

c20 and c21 (Addendum 386) are not run by TRACK; c22 contains their items.

## 5. Smoke run, then the scored run

1. Smoke: `python benchmarks/run_track_2026-10-07.py track data/2026-10-07/track_smoke --smoke` (20 circuits per
   device, seeds 85,500,000 + k). It checks that the scripts run; its numbers are reported in the lock commit and are
   not scored. If the smoke run shows a fault in a script or in c22, the fault is fixed, disclosed here, and the smoke
   run repeated before the lock.
2. Lock: the commit with this Addendum and the smoke output.
3. Scored run: `python benchmarks/run_track_2026-10-07.py track data/2026-10-07/track`, then scoring and
   `track_verify.py` (both run by the runner).

## 6. What the verdicts decide

- T1 CONFIRMED with T2 and T3 CONFIRMED: items 47 and 49 are proposed for acceptance (not release) on top of c19,
  under Addendum 385's policy.
- T1 REFUTED: whatever made the circuits differ is found before anything else.
- T2 or T3 AMBIGUOUS or REFUTED: the items are not proposed on time grounds alone; the profile is repeated.
- Item 48 is decided by BP-MOCK, not here.


---

<!-- ===== Addendum 388 (source: spare-qubit-cliff-addendum-388-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 387, and the owner's decision of 2026-10-07.

## Addendum 388 -- Results of TRACK (Addendum 387): candidate c22 returns candidate c19's circuit on all 360 circuits and takes 0.37-0.51 of its time at 16 qubits (T1-T4 CONFIRMED); items 47 and 49 accepted, not released (2026-10-07)

**Status: results of the pre-registered test in Addendum 387, scored by the locked script and re-checked by the
independent [`benchmarks/track_verify.py`](../../benchmarks/track_verify.py); then the owner's decision.** Written
after the output was seen.

## 1. The run

- **Lock:** commit `6dbadfa` (Addenda 386-387, with the smoke run), made at the workplace before the scored run;
  every job's `git_head` is `6dbadfa`, with no uncommitted change to a tracked file. It is pushed after the run (no
  GitHub login at the workplace).
- **Machine:** the workplace PC, Windows, Python 3.11.9, Qiskit 2.5.2, 14 logical CPUs, six jobs in parallel
  (`env.txt`).
- **Time:** started 2026-10-07T04:23:11Z; the slowest job (FakeKingston) finished after 541 s.
- **Output:** [`data/2026-10-07/track/`](../../data/2026-10-07/track/) (one json per device, the logs, `env.txt`,
  `progress.txt`, `score.md`, `score_log.txt`, `verify_log.txt`); the smoke run is in
  [`data/2026-10-07/track_smoke/`](../../data/2026-10-07/track_smoke/).

## 2. Results

**P0: PASS.** Six files of 60, one git head and one script and c22 hash, no error, no input with an instruction on
more than two qubits, no c22 output off the target, 54 of 54 exactness checks made and at most 1e-6.

| device | identical | n = 16: median c22 / c19 | n = 12-14 | n = 20 |
|---|---|---|---|---|
| FakeTorino | 60 / 60 | 0.395 | 0.656 | 1.035 |
| FakeKingston | 60 / 60 | 0.396 | 0.634 | 1.006 |
| FakeAuckland | 60 / 60 | 0.499 | 0.642 | 1.027 |
| FakeHanoiV2 | 60 / 60 | 0.506 | 0.675 | 0.910 |
| FakeBrussels | 60 / 60 | 0.365 | 0.592 | 0.964 |
| FakeOsaka | 60 / 60 | 0.386 | 0.601 | 0.995 |

| | prediction | result | verdict |
|---|---|---|---|
| T1 | c22 returns c19's circuit | 360 of 360 | **CONFIRMED** |
| T2 | n = 16: median ratio <= 0.6 on every device | 0.365-0.506 | **CONFIRMED** |
| T3 | n = 12-14: median ratio <= 0.85 on every device | 0.592-0.675 | **CONFIRMED** |
| T4 | n = 20: median ratio <= 1.15 on every device | 0.910-1.035 | **CONFIRMED** |

`track_verify.py`: P0 PASS, the same four verdicts, "verdicts identical to score.md: True".

**Reported without prediction:** median c22 / c19 per family at n = 16 | n = 12-14: ring 0.402 | 0.678, brick 0.379 |
0.644, pauli 0.438 | 0.542, qft 0.407 | 0.627. Total compile time (c19, c22): n = 8 47.2 s, 35.7 s (0.757); n = 12
132.2 s, 81.8 s (0.619); n = 14 339.1 s, 166.7 s (0.492); **n = 16 1,519.9 s, 644.1 s (0.424)**; n = 20 9.3 s, 8.7 s
(0.940).

## 3. What this shows

1. **Items 47 and 49 change the recommended call's time, not its output,** on inputs without instructions on more
   than two qubits: the same circuit on 360 of 360, including six devices of three gate families.
2. **At 16 qubits the recommended call takes 0.37-0.51 of c19's time** (0.42 of the total). With FUSE (c19 against
   release 2026-10-06.4: 0.42 of the total at 16 qubits, Addendum 384), that is roughly a fifth of the release's time
   at 16 qubits. This product of two tests on different circuits is an estimate, not a measurement.
3. **The cx devices gained least** (FakeAuckland 0.499, FakeHanoiV2 0.506), as the smoke run had shown (0.539 and
   0.610 on four circuits each); the prediction was left as written.
4. **Not established:** item 48 (c21), which TRACK kept out by expanding every input first; inputs that are not
   SKIP's families; any machine but the workplace PC.

## 4. Decision (the owner, 2026-10-07)

**Items 47 and 49 are accepted, not released,** under the policy of Addendum 385: they are released together with the
next accepted improvements. Item 48 is decided by BP-MOCK (Addendum 389). Candidate c22's file contains items 46-49;
if item 48 is not accepted, the release is built from c22 without item 48 and checked against c22 on inputs without
wide instructions before it is released. `docs/RELEASES.md` lists items 47 and 49 under "Accepted, not yet released".

## 5. Erratum to Addenda 377 and 386

Addendum 377 calls BP-PROBE's tests "fourteen"; its list and its data
([`data/2026-10-06/bp_probe/`](../../data/2026-10-06/bp_probe/)) have twelve test ids. Addendum 386 repeats "14"
(section 2); DISPATCH-PROBE ran those twelve, with SKIP's eight and family T's three circuits (23 in all, as its
tables show). BP-MOCK excludes the twelve.


---

<!-- ===== Addendum 389 (source: spare-qubit-cliff-addendum-389-2026-10-07.md) ===== -->

> **Note added when merging:** Pre-registration of BP-MOCK, committed with its smoke run as the lock before the scored run.

## Addendum 389 -- Pre-registration of BP-MOCK: a mock exam on 92 Benchpress transpilation tests; does candidate c22 (item 48: instructions on three or more qubits expanded first) bring the default call to within 15% of Qiskit level 2's two-qubit count, and leave every other input's circuit unchanged? (2026-10-07)

**Status: pre-registration, written before BP-MOCK's smoke run and before any of its output exists.** The lock is the
commit that adds this Addendum with the smoke run's output; the scored run follows that commit.

## 1. Question

BP-PROBE (Addendum 377) found the default call behind Qiskit level 2 on Benchpress's circuits (two-qubit count,
geometric mean 1.53 on twelve tests). DISPATCH-PROBE (Addendum 386) traced most of the gap to instructions on more
than two qubits and led to item 48 (candidate c21, carried in c22). Those probes chose item 48, so they cannot test
it. BP-MOCK tests it on a sample of Benchpress that excludes every test they used, and records where the default
call stands against Qiskit level 2 as Benchpress calls it. It is the "mock exam" named in Addendum 377.

## 2. Design

[`benchmarks/bp_mock.py`](../../benchmarks/bp_mock.py), checked by the independent
[`benchmarks/bp_mock_verify.py`](../../benchmarks/bp_mock_verify.py) (reads the raw json and re-draws the sample by
itself).

- **Benchpress:** commit `b695f30`, the clone used by BP-PROBE and DISPATCH-PROBE. The published reference
  ([`published_ref.json`](../../data/2026-10-06/bp_probe/published_ref.json), 1,032 test ids) is BP-PROBE's,
  unchanged.
- **Sample (fixed by rule; the list below is what the rule gives):** Benchpress's transpilation tests are divided into
  19 strata (QASMBench small, medium and large on each of the four abstract topologies; HamLib on each topology;
  HamLib, Feynman and the 100-qubit tests on FakeTorino). In each stratum the published test ids, minus the twelve ids
  BP-PROBE ran, are ordered by SHA-256 of `"BP-MOCK|" + id` and the first K taken: K = 5 for QASMBench small and
  HamLib on a topology, 4 for QASMBench medium and large, 8 for HamLib on FakeTorino, 6 for Feynman, 6 for the
  100-qubit tests (all that remain). 92 tests.
- **Building:** as Benchpress's Qiskit gym builds them (BP-PROBE's builders in `bp_probe.py`, plus the 100-qubit
  "summit" circuits as `test_summit.py` builds them): the same input circuit and backend (FakeTorino, or Benchpress's
  `FlexibleBackend` with basis `id, sx, x, rz, cz`), the same metrics (count and depth of the backend's two-qubit gate)
  and Benchpress's structural validator.
- **Arms** (each test and arm in its own process, 600 s limit; jobs run six at a time, which changes only times):

| arm | call |
|---|---|
| QK | `generate_preset_pass_manager(2, backend).run(circuit)`: Benchpress's Qiskit call, not seeded, as in Benchpress |
| REL | release 2026-10-06.4, default call: `compile_for_hardware(circuit, coupling_map, basis_gates, entangling_basis="cx", layout_search=True, seed_transpiler=0)` |
| C22 | candidate c22 (items 46-49), the same default call |
| RELR, C22R | the README's recommended call of each, with the backend's target (FakeTorino tests only) |

- **Also recorded:** compile time; whether the input has an instruction on more than two qubits (barriers aside);
  a hash of each output (every instruction, parameters, global phase, layouts); for outputs on at most 10 qubits,
  equivalence with the input (`Operator.equiv`, with measurements removed while keeping the output's layout, which
  fixes BP-PROBE's check, Addendum 377 section 3); the recommended call's counters.

## 3. Predictions

Ratios are of (two-qubit gates + 1), so that a test with none (a BV-like circuit) counts; geometric means are over the
tests that QK, REL and C22 all finished.

| | prediction | CONFIRMED | REFUTED |
|---|---|---|---|
| P0 | the run is valid | the 92 tests of the rule; C22 never fails where REL finishes; every C22 and C22R output passes Benchpress's validator; versions as named; one git head, no uncommitted change to a tracked file | otherwise nothing is scored |
| M1 | inputs without an instruction on more than two qubits: C22's default call returns REL's circuit | none differs | any differs |
| M2 | inputs with one: geometric mean C22 / REL | <= 0.85 | > 1.00 |
| M3 | all tests: geometric mean C22 / QK | <= 1.15 | > 1.30 |
| M4 | FakeTorino tests that both finish: geometric mean C22R / RELR | <= 1.00 | > 1.05 |
| M5 | C22 outputs checked for equivalence | all equivalent | any not equivalent |

Between the two columns the verdict is AMBIGUOUS. **Where the numbers come from:** on DISPATCH-PROBE's 23 circuits the
expansion alone (PSFU) gave a geometric mean of 1.045 against level 2 and never more two-qubit gates than the default
call; on BP-PROBE's twelve tests the release's default call was at 1.53. M1 is the property item 48 was written to
have. M4's line allows for the recommended call's choice by estimate rather than by count. Nothing is predicted about
time; it is reported.

**Not tested here:** routing level 2 (not proposed, Addendum 386); anything at 133 qubits beyond FakeTorino; hardware.

## 4. The sample

| stratum | K | tests |
|---|---|---|
| QASMBench small, all-to-all | 5 | `basis_test_n4`, `bb84_n8`, `qaoa_n6`, `lpn_n5`, `cat_state_n4` |
| QASMBench small, square | 5 | `vqe_n4`, `qaoa_n6`, `fredkin_n3`, `error_correctiond3_n5`, `lpn_n5` |
| QASMBench small, heavy-hex | 5 | `vqe_uccsd_n4`, `adder_n10`, `teleportation_n3`, `qaoa_n3`, `iswap_n2` |
| QASMBench small, linear | 5 | `hs4_n4`, `qaoa_n3`, `bb84_n8`, `adder_n4`, `dnn_n8` |
| QASMBench medium, all-to-all | 4 | `ghz_state_n23`, `qec9xz_n17`, `bwt_n21`, `seca_n11` |
| QASMBench medium, square | 4 | `qram_n20`, `ghz_state_n23`, `cat_state_n22`, `factor247_n15` |
| QASMBench medium, heavy-hex | 4 | `multiply_n13`, `dnn_n16`, `swap_test_n25`, `bwt_n21` |
| QASMBench medium, linear | 4 | `square_root_n18`, `dnn_n16`, `qec9xz_n17`, `factor247_n15` |
| QASMBench large, all-to-all | 4 | `qugan_n111`, `square_root_n45`, `qugan_n395`, `qugan_n39` |
| QASMBench large, square | 4 | `swap_test_n41`, `bv_n30`, `knn_341`, `ghz_n78` |
| QASMBench large, heavy-hex | 4 | `qft_n160`, `square_root_n60`, `knn_129`, `multiplier_n400` |
| QASMBench large, linear | 4 | `ghz_n127`, `bv_n140`, `swap_test_n83`, `adder_n64` |
| HamLib, all-to-all | 5 | `reg-4_n-90_rinst-07`, `tsp_prob-lin105_Ncity-7_enc-unary`, `ham_parity-4`, `4-uf100-0246.cnf-70-res`, `mu_x_prime_enc_unary_dvalues_4-4-4` |
| HamLib, square | 5 | `graph-1D-grid-pbc-qubitnodes_Lx-16_h-2`, `fh-graph-1D-grid-pbc-qubitnodes_Lx-50_U-2_enc-jw`, `reg-5_n-10_rinst-07`, `bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_Ly-7_U-70_enc-gray_d-4`, `mu_y_prime_enc_stdbinary_dvalues_4-...-4` (18 fours) |
| HamLib, heavy-hex | 5 | `graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-186_h-3`, `tsp_prob-ulysses22_Ncity-8_enc-stdbinary`, `ham_JW12`, `ham_JW-14`, `reg-5_n-10_rinst-07` |
| HamLib, linear | 5 | `mu_x_prime_enc_stdbinary_dvalues_8-8-8-8-8-8-8-8-8-4-4-4-4-4-4`, `enc_unary_dvalues_4-4-4`, `ham_BK22`, `reg-5_n-10_rinst-07`, `bh_graph-2D-triag-nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4` |
| HamLib, FakeTorino | 8 | `ham_JW-18`, `bh_graph-2D-triag-pbc-qubitnodes_Lx-3_Ly-22_U-70_enc-unary_d-4`, `bh_graph-2D-triag-pbc-qubitnodes_Lx-10_Ly-10_U-30_enc-stdbinary_d-4`, `ham_JW-10`, `ham_parity10`, `ham_JW-22`, `mu_x_prime_enc_unary_dvalues_4-4-4`, `ham_JW-6` |
| Feynman, FakeTorino | 6 | `mod_red_21`, `qcla_com_7`, `gf2^6_mult`, `mod5_4`, `gf2^8_mult`, `barenco_tof_5` |
| 100-qubit, FakeTorino | 6 | `circSU2_89`, `BVlike_simplification`, `QAOA_100`, `BV_100`, `square_heisenberg_100`, `clifford_100` |

(HamLib names without their `ham_` test-id prefix; `python benchmarks/bp_mock.py sample --bp <clone>` prints the full
ids.)

## 5. Files locked

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/bp_mock.py`](../../benchmarks/bp_mock.py) | `7757e848c9848c9644a845ce97198966b2fb2f98fa9584c2d0050d45b573b4d0` |
| [`benchmarks/bp_mock_verify.py`](../../benchmarks/bp_mock_verify.py) | `a9d2840065b6769d2e688c728290e7cbda335f2b676b3ac391ab2f25099bb9b4` |
| [`benchmarks/bp_probe.py`](../../benchmarks/bp_probe.py) (builders, unchanged) | `73defde66868852db583d5e4fb055ec53c6ea864aae6466876c571f8d26cee66` |
| `psf_compile.py` (release 2026-10-06.4, unchanged) | `69fe51d2d503638ceb4d067a0d86a5b27c38586694ec84996dea5e7ab6dab7aa` |
| `patches/psf_compile_c22_2026-10-07/psf_compile.py` (as locked in Addendum 387) | `30675c37e6c9400803ed4d53c8d146ecdebec24b9452037808c9e9acf150f858` |

## 6. Smoke run, then the scored run

1. Smoke: `python benchmarks/bp_mock.py run --bp <clone> --out data/2026-10-07/bp_mock_smoke --smoke` (the first test
   of each stratum: 19 tests) and `score` on it. It checks that the scripts run; its numbers are reported in the lock
   commit and not scored. A fault in a script found there is fixed and disclosed here before the lock.
2. Lock: the commit with this Addendum and the smoke output.
3. Scored run: `run` into `data/2026-10-07/bp_mock`, then `score` and `bp_mock_verify.py`.

## 7. What the verdicts decide

- M1 and M5 CONFIRMED with M2 CONFIRMED: item 48 is proposed for acceptance (not release) with items 46, 47 and 49.
- M1 or M5 REFUTED: item 48 is not proposed; the cause is found first.
- M3 is the mock exam's grade, not a condition for item 48: whatever its verdict, it says where the default call stands
  against Qiskit level 2 on Benchpress, and the README's claims follow it.
- M4 REFUTED: the recommended call's loss on wide inputs (Addendum 386, section 4) is studied before item 48 is
  proposed for the recommended call.


---

<!-- ===== Addendum 390 (source: spare-qubit-cliff-addendum-390-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 389, with an exploratory diagnosis made after the run.

## Addendum 390 -- Results of BP-MOCK (Addendum 389): with item 48 the default call comes to 1.115 of Qiskit level 2's two-qubit count on 92 Benchpress tests (1.364 for the release; M2-M4 CONFIRMED); M1 and M5 REFUTED, the first by the release's own non-reproducible layout on two BV circuits, the second by a defect of the equivalence check (2026-10-07)

**Status: results of the pre-registered test in Addendum 389, scored by the locked script and re-checked by the
independent [`benchmarks/bp_mock_verify.py`](../../benchmarks/bp_mock_verify.py); section 4 is an exploratory
diagnosis made after the output was seen.**

## 1. The run

- **Lock:** commit `86f992c` (Addenda 388-389, with the smoke run); the run recorded `git_head` `86f992c` and no
  uncommitted change to a tracked file. It is pushed after the run.
- **Machine:** the workplace PC (Windows, Python 3.11.9, Qiskit 2.5.2), Benchpress `b695f30`, six jobs at a time.
- **Time:** started 2026-10-07T04:56:02Z; 316 jobs in 929 s.
- **Output:** [`data/2026-10-07/bp_mock/`](../../data/2026-10-07/bp_mock/): `bp_mock.json` (every job), `score.md`,
  `run_log.txt`, `score_log.txt`, `verify_log.txt`, and the diagnosis of section 4 (`diag_log.txt`).

## 2. Results

**P0: PASS.** The 92 tests of the rule (re-drawn by `bp_mock_verify.py`: same set); no error or timeout in any arm;
every C22 and C22R output passes Benchpress's validator; versions as named.

| | prediction | result | verdict |
|---|---|---|---|
| M1 | inputs without wide instructions (38): C22's default call returns REL's circuit | 2 of 38 differ | **REFUTED** |
| M2 | inputs with wide instructions (54): geometric mean C22 / REL two-qubit count <= 0.85 | 0.709 | **CONFIRMED** |
| M3 | all 92: geometric mean C22 / QK <= 1.15 | 1.115 (REL / QK: 1.364) | **CONFIRMED** |
| M4 | FakeTorino tests (20): geometric mean C22R / RELR <= 1.00 | 0.887 | **CONFIRMED** |
| M5 | C22 outputs checked for equivalence (14) are all equivalent | 5 of 14 reported not equivalent | **REFUTED** |

`bp_mock_verify.py`: the same five verdicts, "verdicts identical to score.md: True".

**Reported without prediction** (geometric means of (two-qubit gates + 1); full table in `score.md`):

- Two-qubit depth against QK: C22 1.117, REL 1.279.
- By stratum, C22 / QK: QASMBench 0.96-1.12 (REL 0.96-1.56); HamLib on abstract topologies 1.00-1.22 (REL
  1.30-1.87); HamLib on FakeTorino 1.12 (REL 1.58); Feynman 1.01 (REL 1.26); the six 100-qubit tests 2.77 (REL 3.63).
- The 100-qubit stratum is dominated by `BVlike_simplification`: Qiskit reduces it to no two-qubit gate (the CX
  gates cancel), REL and C22 keep 392 (a ratio of 393 with the +1). Without it the five others are 0.97-1.12.
- Largest C22 / QK otherwise: `basis_test_n4` 1.57 (10 against 6), `error_correctiond3_n5` 1.46, HamLib
  `enc_unary_dvalues_4-4-4` on linear 1.47, `qft_n160` on heavy-hex 1.41. Lowest: `qaoa_n3` on linear 0.80.
- **One wide input where C22 used more two-qubit gates than REL:** HamLib
  `bh_graph-2D-triag-nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4` on linear, 64,436 against 56,291 (+14.5%;
  QK 57,349). On the other 53 C22 had as many or fewer.
- **The recommended call against the default call on FakeTorino's 100-qubit tests:** C22R used more two-qubit
  gates than C22's default call on all six, and RELR more than REL's on five (for example `BV_100` 580 against 200,
  `circSU2_89` C22R 1,344 against C22's 336). This is the release's behaviour as much as c22's (RELR equals C22R on
  five of six). Its cause
  (most likely item 31's recompile around FakeTorino's failed couplers) was not measured here.
- Time: C22's default call took 0.95-10.7 times QK's (stratum means); C22R took 1.15 times RELR's.

## 3. What the two REFUTED verdicts were scored on

- **M1:** `bv_n30` on square and `bv_n140` on linear. In both REL and C22 have the same two-qubit count and depth (51
  / 51 and 352 / 352); only the circuits' signatures differ. Neither input has an instruction on more than two
  qubits, so item 48 hands both modules the same input object.
- **M5:** the five C22 outputs reported not equivalent are `qaoa_n6`, `fredkin_n3`, `error_correctiond3_n5` and
  `lpn_n5` on square, and HamLib `ham_parity-4` on all-to-all. REL's outputs were reported not equivalent on the same
  five, and QK's on seven (those five, `basis_test_n4` and `dnn_n8`).

## 4. Diagnosis (exploratory, after the run)

[`benchmarks/bp_mock_diag.py`](../../benchmarks/bp_mock_diag.py), one call at a time on the workplace PC
([`diag_log.txt`](../../data/2026-10-07/bp_mock/diag_log.txt)):

- **M1: the release's default call is not reproducible on these two circuits.** Compiled REL, C22, REL, C22 in one
  process, the four circuits of `bv_n30` were all different (same counts); on `bv_n140` REL differed from itself and
  C22 from itself (one REL-C22 pair was identical). Neither run's circuit was reproduced, and a layout-search budget
  of 60 s instead of 2 s did not make REL and C22 agree. The difference is in the release, not in item 48: on a BV
  circuit (one qubit coupled to all others) the layout search does not settle on one placement. Which step is not
  deterministic was not found; the layout search has time budgets (`layout_search_time_budget_s`, and a fixed 1 s
  for its packing search), which make it depend on the machine's load. `seed_transpiler` does not make the default
  call reproducible on such inputs.
- **M5: the check, not the circuits.** Checked again in two ways that allow for the extra qubits of an abstract
  square lattice (a 3-qubit input on a 4-qubit square, 5 and 6 on 9) and for a `PauliEvolutionGate` built as a
  product formula (the input expanded through its definitions first): item 39's `_implements` (two random product
  states) and the workplace probe's state check from |0...0>, **REL's and C22's outputs implement the input on all
  seven tests** (state infidelity at most 8e-14). BP-MOCK's check compared operators of different sizes on the
  square tests, and on `ham_parity-4` compared the product formula with the exact exponential, the trap Addendum 377
  (section 4) had already described.
- **QK** fails the state checks on three tests (`basis_test_n4`, `error_correctiond3_n5`, `dnn_n8`; infidelity up to
  0.99). These inputs end in measurements, and Qiskit's level 2 removes gates before final measurements and folds
  final swaps into the measurement map, which changes the state but not the measured distribution. The checks here
  compare states, so this is not evidence of a Qiskit error.

## 5. What this shows

1. **Item 48 does what DISPATCH-PROBE suggested, on circuits it did not see:** with it the default call goes from
   1.36 to 1.12 of Qiskit level 2's two-qubit count on a stratified Benchpress sample, and below 1.05 on most strata;
   on wide inputs it uses 0.71 of the release's count, more on one of 54.
2. **Both REFUTED verdicts trace to things other than item 48** (the release's non-reproducible layout on BV
   circuits; the equivalence check), but that was found after the run. By the rule of Addendum 389 (section 7), item
   48 is not proposed on this test.
3. **New findings, each for its own item:** the default call is not reproducible on some inputs; it does not cancel
   the CX gates of a BV-like circuit that Qiskit removes; the recommended call uses more two-qubit gates than the
   default call on FakeTorino's 100-qubit tests.
4. **Not established:** anything about Benchpress's other 940 tests beyond what a stratified sample of 92 says; QK is
   unseeded and ran once; hardware.


---

<!-- ===== Addendum 391 (source: spare-qubit-cliff-addendum-391-2026-10-07.md) ===== -->

> **Note added when merging:** Pre-registration of BP-MOCK2, chosen by the owner after Addendum 390; committed with its smoke run as the lock.

## Addendum 391 -- Pre-registration of BP-MOCK2: item 48 re-tested on 48 new Benchpress tests, with the release compiled twice and the equivalence check of Addendum 390's diagnosis (2026-10-07)

**Status: pre-registration, written before BP-MOCK2's smoke run and before any of its output exists.** The owner chose
this re-test after Addendum 390. The lock is the commit that adds this Addendum with the smoke run's output; the scored
run follows that commit.

## 1. Question

BP-MOCK (Addenda 389-390) refuted M1 and M5, and its diagnosis traced both to things other than item 48: the release's
default call does not reproduce its own circuit on some inputs (two BV circuits), and the equivalence check compared
operators of different sizes and a product formula with the exact exponential. Under BP-MOCK's rule item 48 was not
proposed. BP-MOCK2 asks the same questions again on tests none of the earlier probes used, with those two faults
removed from the design, not from the data:

- **M1** now counts a difference only where the release reproduces itself: REL is compiled twice, in separate
  processes (REL, REL2), and a test where REL and REL2 differ is reported, not scored.
- **M5** uses the checks of Addendum 390's diagnosis: item 39's `_implements` (two random product states; layout and
  ancilla qubits handled) and, reported alongside, the workplace probe's state check from |0...0>, both against the
  input with final measurements removed and expanded through its definitions (a `PauliEvolutionGate` becomes the
  product formula every compiler builds). QK is checked and reported but not scored: its level 2 changes states
  before final measurements (Addendum 390, section 4).

## 2. Design

[`benchmarks/bp_mock2.py`](../../benchmarks/bp_mock2.py) (imports BP-MOCK's locked `bp_mock.py` for the strata,
builders and metrics), checked by the independent [`benchmarks/bp_mock2_verify.py`](../../benchmarks/bp_mock2_verify.py)
(imports neither; re-draws both samples itself).

- **Sample:** BP-MOCK's strata without the 100-qubit stratum (all nine of its tests are used). In each, the published
  test ids minus the twelve BP-PROBE ran and the 92 BP-MOCK ran, ordered by SHA-256 of `"BP-MOCK2|" + id`, the first
  K: 3 for QASMBench small and for HamLib on each topology, 2 for QASMBench medium and large, 4 for HamLib on
  FakeTorino, 4 for Feynman. 48 tests (listed in section 5).
- **Arms:** QK, REL, REL2 and C22, as in BP-MOCK (default calls; Benchpress's Qiskit call unseeded). No recommended
  call (M4 is not repeated).
- **Equivalence:** on inputs of at most 10 qubits, every arm's output; "checkable" means `_implements` could be made
  (at most 16 touched qubits, every instruction with a matrix).
- Each test and arm in its own process, 600 s limit, six at a time; Benchpress `b695f30`.

## 3. Predictions

Ratios are of (two-qubit gates + 1); geometric means over the tests all four arms finished.

| | prediction | CONFIRMED | REFUTED |
|---|---|---|---|
| P0 | the run is valid | the 48 tests of the rule; C22 never fails where REL finishes; every C22 output passes Benchpress's validator; versions as named; no uncommitted change to a tracked file | otherwise nothing is scored |
| M1 | inputs without wide instructions, where REL = REL2: C22 returns REL's circuit | none differs | 2 or more differ |
| M2 | inputs with wide instructions: geometric mean C22 / REL | <= 0.85 | > 1.00 |
| M3 | all tests: geometric mean C22 / QK | <= 1.15 | > 1.30 |
| M5 | every checkable C22 output implements its input (`_implements`) | none fails, at least 5 checked | any fails |

Between the columns the verdict is AMBIGUOUS. M1 allows one difference as AMBIGUOUS because a release that does not
reproduce itself can still agree with itself twice by chance. M2 and M3 repeat BP-MOCK's lines (BP-MOCK: 0.709 and
1.115).

**Reported without prediction:** how often REL2 differs from REL (BP-MOCK's diagnosis suggests BV-like circuits);
wide inputs where C22 uses more two-qubit gates than REL; REL's and QK's checks; the state check; per-stratum ratios.

## 4. What the verdicts decide

- M1, M2 and M5 CONFIRMED: item 48 is proposed for acceptance (not release), with items 46, 47 and 49.
- M1 or M5 REFUTED: item 48 is not proposed; the failing tests are examined first.
- M2 AMBIGUOUS or REFUTED: item 48 is not proposed on quality grounds.
- M3 is the mock exam's grade, as in BP-MOCK.
- The release's non-reproducibility (Addendum 390) is a separate item, whatever the verdicts.

## 5. The sample

| stratum | tests |
|---|---|
| QASMBench small, all-to-all | `qec_sm_n5`, `grover_n2`, `qaoa_n3` |
| QASMBench small, square | `qec_en_n5`, `grover_n2`, `wstate_n3` |
| QASMBench small, heavy-hex | `qaoa_n6`, `vqe_uccsd_n6`, `variational_n4` |
| QASMBench small, linear | `pea_n5`, `inverseqft_n4`, `sat_n7` |
| QASMBench medium, all-to-all | `qf21_n15`, `multiplier_n15` |
| QASMBench medium, square | `knn_n25`, `wstate_n27` |
| QASMBench medium, heavy-hex | `bv_n19`, `qram_n20` |
| QASMBench medium, linear | `ghz_state_n23`, `multiply_n13` |
| QASMBench large, all-to-all | `knn_n41`, `bv_n30` |
| QASMBench large, square | `ising_n420`, `ising_n66` |
| QASMBench large, heavy-hex | `ising_n98`, `qugan_n39` |
| QASMBench large, linear | `square_root_n45`, `bwt_n37` |
| HamLib, all-to-all | `graph-2D-grid-pbc-qubitnodes_Lx-2_Ly-185_h-0.5`, `ham_parity-14`, `ash608gpia,n-160,rinst-1` |
| HamLib, square | `graph-1D-grid-pbc-qubitnodes_Lx-26_h-6`, `bh_graph-2D-triag-pbc-qubitnodes_Lx-3_Ly-22_U-70_enc-unary_d-4`, `graph-2D-triag-nonpbc-qubitnodes_Lx-3_Ly-160_h-0.1` |
| HamLib, heavy-hex | `ham_parity-4`, `fh-graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-72_U-0_enc-parity`, `graph-1D-grid-pbc-qubitnodes_Lx-16_h-2` |
| HamLib, linear | `4-uf100-0246.cnf-70-res`, `enc_gray_dvalues_4-4-4-4-4-4-4`, `reg-4_n-90_rinst-07` |
| HamLib, FakeTorino | `tsp_prob-ts225_Ncity-5_enc-unary`, `gnp-k_5_n-60_rinst-19`, `ham_JW-14`, `enc_gray_dvalues_8-8-8` |
| Feynman, FakeTorino | `rb`, `grover_5`, `hwb10`, `mod_mult_55` |

The rule excludes test ids, not circuits: many of these circuits were compiled in BP-PROBE or BP-MOCK on another
topology or device (for example `qaoa_n3`, `qram_n20`, `bv_n30`, `square_root_n45`, HamLib `ham_parity-4`,
`reg-4_n-90_rinst-07`, `ham_JW-14`, and `enc_gray_dvalues_4-4-4-4-4-4-4`, a BP-PROBE circuit that DISPATCH-PROBE also
used, here on the linear topology instead of FakeTorino). Disclosed, not changed.

## 6. Files locked

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/bp_mock2.py`](../../benchmarks/bp_mock2.py) | `e0c182acb99768a88bedb07c6f56aae84fc234b62a17bd2a046452de3bccf93c` |
| [`benchmarks/bp_mock2_verify.py`](../../benchmarks/bp_mock2_verify.py) | `ac86488b3f4949d92dde1192cc3a0660f36a147716fbeb912471bed13af8263b` |
| `benchmarks/bp_mock.py` (as locked in Addendum 389) | `7757e848c9848c9644a845ce97198966b2fb2f98fa9584c2d0050d45b573b4d0` |
| `psf_compile.py` (release 2026-10-06.4) | `69fe51d2d503638ceb4d067a0d86a5b27c38586694ec84996dea5e7ab6dab7aa` |
| `patches/psf_compile_c22_2026-10-07/psf_compile.py` (as locked in Addendum 387) | `30675c37e6c9400803ed4d53c8d146ecdebec24b9452037808c9e9acf150f858` |

## 7. Smoke run, then the scored run

1. Smoke: `python benchmarks/bp_mock2.py run --bp <clone> --out data/2026-10-07/bp_mock2_smoke --smoke` (the first
   test of each stratum: 18) and `score`. Not scored; a fault in a script found there is fixed and disclosed here
   before the lock.
2. Lock: the commit with this Addendum and the smoke output.
3. Scored run into `data/2026-10-07/bp_mock2`, then `score` and `bp_mock2_verify.py`.


---

<!-- ===== Addendum 392 (source: spare-qubit-cliff-addendum-392-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 391, and the owner's decision of 2026-10-07.

## Addendum 392 -- Results of BP-MOCK2 (Addendum 391): M1, M2, M3 and M5 CONFIRMED on 48 new Benchpress tests; with item 48 the default call comes to 1.039 of Qiskit level 2's two-qubit count (1.261 for the release); item 48 accepted, not released (2026-10-07)

**Status: results of the pre-registered test in Addendum 391, scored by the locked script and re-checked by the
independent [`benchmarks/bp_mock2_verify.py`](../../benchmarks/bp_mock2_verify.py); then the owner's decision.**

## 1. The run

- **Lock:** commit `fffedd5` (Addendum 391, with the smoke run); the run recorded `git_head` `fffedd5` and no
  uncommitted change to a tracked file. Pushed after the run.
- **Machine:** the workplace PC (Windows, Python 3.11.9, Qiskit 2.5.2), Benchpress `b695f30`, six jobs at a time.
- **Time:** started 2026-10-07T05:29:56Z; 192 jobs in 740 s.
- **Output:** [`data/2026-10-07/bp_mock2/`](../../data/2026-10-07/bp_mock2/): `bp_mock2.json`, `score.md`,
  `run_log.txt`, `score_log.txt`, `verify_log.txt`; the smoke run in
  [`data/2026-10-07/bp_mock2_smoke/`](../../data/2026-10-07/bp_mock2_smoke/).

## 2. Results

**P0: PASS.** The 48 tests of the rule (re-drawn by `bp_mock2_verify.py`: same set); C22 never failed where REL
finished; every C22 output passes Benchpress's validator; versions as named. One test, QASMBench large `bwt_n37` on
linear, timed out at 600 s in QK, REL and REL2; C22 finished it (3,281,757 two-qubit gates). It is outside the 47
tests all four arms finished, over which M1-M5 are computed.

| | prediction | result | verdict |
|---|---|---|---|
| M1 | flat inputs where REL = REL2 (16 of 17): C22 returns REL's circuit | 0 of 16 differ | **CONFIRMED** |
| M2 | wide inputs (30): geometric mean C22 / REL <= 0.85 | 0.738 | **CONFIRMED** |
| M3 | all 47: geometric mean C22 / QK <= 1.15 | 1.039 (REL / QK: 1.261) | **CONFIRMED** |
| M5 | checkable C22 outputs (15) implement their input | 0 of 15 fail | **CONFIRMED** |

`bp_mock2_verify.py`: the same four verdicts, "verdicts identical to score.md: True".

**Reported without prediction:**

- REL2 differed from REL on 2 of 47 tests (one flat, one wide), with the same two-qubit count both times: the
  release's non-reproducibility of Addendum 390, section 4.
- Wide inputs where C22 used more two-qubit gates than REL: 3 of 30 (`qugan_n39` on heavy-hex 571 against 560, depth
  303 against 254; Feynman `grover_5` 537 against 526; `hwb10` 114,203 against 113,609).
- REL's 15 checkable outputs all implement their input; QK's fail on two (state checks, not comparable after Qiskit's
  measurement-aware passes; Addendum 390).
- By stratum, C22 / QK: QASMBench 1.00-1.15 (REL 1.00-1.34); HamLib on abstract topologies 1.00-1.05 (REL
  1.27-2.34); HamLib on FakeTorino 1.10 (REL 1.47); Feynman 1.04 (REL 1.07).

## 3. What BP-MOCK and BP-MOCK2 show together

1. **On two independent stratified samples of Benchpress** (92 and 48 tests, none used by BP-PROBE), item 48 lowers
   the default call's two-qubit count on inputs with instructions on more than two qubits to 0.71 and 0.74 of the
   release's, and the default call's distance to Qiskit level 2 from 1.36 / 1.26 to 1.12 / 1.04 (geometric means).
2. **It changes nothing else:** on inputs without such instructions C22 returns the release's circuit wherever the
   release reproduces itself (38 + 16 tests), and every checkable output implements its input.
3. **It is not better everywhere:** on 4 of 84 wide inputs it used more two-qubit gates than the release (+0.5% to
   +14.5%).
4. **Not established:** Benchpress's other tests; hardware; the recommended call (not part of BP-MOCK2; BP-MOCK's M4
   was CONFIRMED on 20 FakeTorino tests).

## 4. Decision (the owner, 2026-10-07)

**Item 48 is accepted, not released,** with items 46, 47 and 49, under the policy of Addendum 385. Candidate c22
(`patches/psf_compile_c22_2026-10-07/psf_compile.py`, items 46-49, as locked in Addendum 387) carries all four and is
the basis of the next release. `docs/RELEASES.md` lists item 48 under "Accepted, not yet released".

Open, each for its own item: the default call's non-reproducible layout on some inputs (Addenda 390, 392); the CX
gates of BV-like circuits that Qiskit cancels and PSF-Zero keeps (Addendum 390); the recommended call's higher
two-qubit count than the default call on FakeTorino's 100-qubit tests (Addendum 390).


---

<!-- ===== Addendum 393 (source: spare-qubit-cliff-addendum-393-2026-10-07.md) ===== -->

> **Note added when merging:** Exploratory, after Addendum 392, at the owner's request; written after the output was seen.

## Addendum 393 -- Three questions after BP-MOCK (exploratory): the non-reproducible circuits come from Qiskit's level 1 itself; the BV-like test's CX gates are removed by commutative cancellation, which must not be applied blindly; the recommended call's higher count at 100 qubits is the price of avoiding FakeTorino's failed couplers, which Qiskit level 2 uses (2026-10-07)

**Status: exploratory, not pre-registered; written after the output was seen.** It answers the three open points of
Addendum 392 (section 4) and leads to candidate c23 (item 50), pre-registered in Addendum 394.

## 1. The runs

[`benchmarks/investigate_2026-10-07.py`](../../benchmarks/investigate_2026-10-07.py), on candidate c22, at the
workplace, in three runs (the script grew between them; the committed file is the last; parts A and B are as in the
first run apart from a fallback added to the gate counter, which they do not reach): run 1 parts A and B, after which
part E stopped on that counter (a backend without `two_q_gate_type`); run 2 parts D, E and C; run 3 parts F and G. Logs: [`data/2026-10-07/investigate/`](../../data/2026-10-07/investigate/).
The circuits are Benchpress tests built as in BP-MOCK, SKIP's families and PL family T.

## 2. Reproducibility (parts A and D)

- On `bv_n30` (square) and `bv_n140` (linear) five default calls in one process gave five different circuits with
  the same two-qubit count. On `bv_n19` (heavy-hex), `BV_100` (FakeTorino) and a 78-qubit GHZ circuit all five were
  the same.
- PSF-Zero's own `compile()` gave the same output all five times, and its layout search found no layout on these BV
  circuits (Qiskit's layout stage takes over). Turning off `elide_permutations` and `post_routing_resynthesis` did
  not make the result reproducible.
- **`qiskit.transpile(..., optimization_level=1, seed_transpiler=0)` of the input itself gave five different circuits
  on both tests.** The non-reproducibility is Qiskit's, which the default call routes with; PSF-Zero adds none.
- In passing: on `bv_n140` PSF-Zero's post-routing re-synthesis lowers the count from 486 (Qiskit's level 1) to 352.

## 3. BV-like circuits and commutative cancellation (parts B, E and G)

| test | QK | C22 | Qiskit's `CommutativeCancellation` first, then C22 | C22, routing level 2 |
|---|---|---|---|---|
| `BVlike_simplification` (198 CX in the input) | 0 | 392 | **0** (the pass removes all 198) | 382 |
| `BV_100` | 196 | 200 | 200 | 196 |
| `bv_n19`, heavy-hex | 63 | 71 | 71 | 68 |
| `bv_n30`, square | 53 | 51 | 51 | 42 |
| `bv_n140`, linear | 353 | 352 | 352 | 352 |

On inputs where nothing should cancel (part E, C22's default call): ring, brick and QFT at 12 and 40 qubits on
FakeTorino and family T at spare 0 on FakeAuckland and FakeKingston were unchanged (the same circuit); SKIP's
Hamiltonians changed, 497 to 521 at 12 qubits and 3,089 to 3,124 at 40. On ten SKIP Hamiltonians of 8-40 qubits
(part G) the pass removed **no** two-qubit gate from any input, yet the default call's count moved both ways (for
example 676 to 634 and 677 to 698 at 16 qubits): the pass also merges and moves single-qubit gates, which changes the
blocks PSF-Zero builds. So the pass helps where it removes two-qubit gates and is noise where it does not.

## 4. The recommended call at 100 qubits (parts C and F)

- FakeTorino reports 22 directed couplers with error at least 0.5 (item 31's "failed"), which isolate 4 qubits. On
  all six 100-qubit tests the default call's circuit uses one of them; the recommended call compiles again on the
  pruned map (`PRUNE_STATS`: recompiled 1) and uses none. All of its extra two-qubit gates come from that step;
  `placement_refine` and the comparisons change nothing at this size (above 16 qubits they are skipped).
- **Qiskit level 2 (QK) also uses a failed coupler on five of the six** (the sixth has no two-qubit gate). On the
  pruned map Qiskit pays about as much: QK 554 / 9,406 / 1,755 / 62,246 / 1,446 against C22's 580 / 9,461 / 1,758 /
  64,650 / 1,344 (BV_100, QAOA_100, square-Heisenberg, Clifford, circSU2_89; 0.93-1.04 of QK).
- The comparison of BP-MOCK (C22R against QK) was therefore not like for like: QK's circuits would run through couplers
  the device reports as failed.

## 5. What follows

1. Reproducibility: nothing to change in PSF-Zero; the README should say that `seed_transpiler` does not make the
   default call reproducible on inputs where Qiskit's own level 1 is not.
2. BV-like circuits: candidate c23 (item 50) uses the cancellation only where it removes two-qubit gates, and then
   keeps whichever of the two compiles has fewer (Addendum 394).
3. The recommended call: nothing to change; the README should say that it avoids failed couplers that Qiskit level 2
   uses, and what that costs.


---

<!-- ===== Addendum 394 (source: spare-qubit-cliff-addendum-394-2026-10-07.md) ===== -->

> **Note added when merging:** Candidate c23 and the pre-registration of CANCEL, committed with c23's test log and the smoke run as the lock.

## Addendum 394 -- Candidate c23 (item 50) and the pre-registration of CANCEL: commutative cancellation is tried only where it removes two-qubit gates; does c23 leave every other default call unchanged, never use more two-qubit gates, stay exact and cost little time? (2026-10-07)

**Status: candidate and pre-registration, written before CANCEL's smoke run and before any of its output exists.**
The lock is the commit that adds this Addendum with c23's tests and the smoke run's output; the scored run follows it.

## 1. Candidate c23 (item 50)

[`patches/psf_compile_c23_2026-10-07/psf_compile.py`](../../patches/psf_compile_c23_2026-10-07/psf_compile.py), based
on c22 (items 46-49), changelog item 50. At the start of the pipeline without a `target` (after item 48's expansion),
Qiskit's `CommutativeCancellation` is run on the input. If it removes no two-qubit gate nothing else happens. If it
removes at least one, the pipeline runs on the input and on the cancelled input and returns the result with fewer
two-qubit gates (the input's on a tie). The reasons are in Addendum 393, section 3.
[`test_c23.py`](../../patches/psf_compile_c23_2026-10-07/test_c23.py) checks it against c22 before the smoke run
(Benchpress's BV-like circuit at 8, 30 and 100 qubits; unchanged circuits where nothing cancels, default and
recommended call, on three devices; exactness on random circuits with cancelling CX pairs; the pass's time on a
100-qubit QFT).

## 2. Design of CANCEL

[`benchmarks/cancel_eval.py`](../../benchmarks/cancel_eval.py), checked by the independent
[`benchmarks/cancel_verify.py`](../../benchmarks/cancel_verify.py) (imports none of the test's scripts; takes the
expected tests from BP-MOCK's and BP-MOCK2's committed output).

- **Tests:** the 140 Benchpress tests of BP-MOCK (92) and BP-MOCK2 (48), built as there. Item 50 was written from one
  of them (BV-like) after both runs; it was not tuned on the others. They are re-used because they are the Benchpress
  tests whose builders and checks are already locked; this is disclosed, not hidden.
- **Arms** (default call, no target): C22; C22 again in its own process (C22B), because Qiskit's level 1 does not
  always reproduce itself (Addendum 393); C23. Each test and arm in its own process, 1,500 s limit (C23 may run the
  pipeline twice), six at a time.
- **Recorded:** two-qubit count and depth, the circuit's hash, Benchpress's validator, `CANCEL_STATS`, time, and on
  inputs of at most 10 qubits item 39's `_implements` against the input expanded through its definitions (as BP-MOCK2).

## 3. Predictions

| | prediction | CONFIRMED | REFUTED |
|---|---|---|---|
| P0 | the run is valid | the 140 tests; C23 never fails where C22 finishes; every C23 output passes the validator; versions as named; no uncommitted change to a tracked file | otherwise nothing is scored |
| K1 | where the cancellation removes nothing and C22 = C22B: C23 returns C22's circuit | none differs | 2 or more differ |
| K2 | where it removes some: C23 has no more two-qubit gates than C22 | none has more, and it was tried at least once | any has more |
| K4 | every checkable C23 output implements its input | none fails, at least 5 checked | any fails |
| K5 | where it removes nothing: median time C23 / C22 | <= 1.15 | > 1.5 |

Between the columns the verdict is AMBIGUOUS. K1 and K2 are what item 50 is built to guarantee; K5 is the price of
running the pass on every input. **Reported without prediction:** on which tests the cancellation was tried, the
two-qubit counts there, which result was kept, and the time.

## 4. Files locked

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c23_2026-10-07/psf_compile.py`](../../patches/psf_compile_c23_2026-10-07/psf_compile.py) | `568de9e691796e4efb330dff3f2ed61aa1cca4cc888286f9727854cf8c744420` |
| [`patches/psf_compile_c23_2026-10-07/test_c23.py`](../../patches/psf_compile_c23_2026-10-07/test_c23.py) | `57f74590d0b46c3c6a6fc52da4f489d587e70d8f7f8ff326a078c61ece620c4f` |
| [`benchmarks/cancel_eval.py`](../../benchmarks/cancel_eval.py) | `b1f76259d66110c8c4e736cb408c7a74e6c86c88df7588fd63b4bce4fe659785` |
| [`benchmarks/cancel_verify.py`](../../benchmarks/cancel_verify.py) | `6f7ded7c65e82781497f5bb2d4be8d9eadee05b32d9648f8fffff4ab6a6e2326` |
| `patches/psf_compile_c22_2026-10-07/psf_compile.py` (as locked in Addendum 387) | `30675c37e6c9400803ed4d53c8d146ecdebec24b9452037808c9e9acf150f858` |

## 5. Smoke, lock, scored run

1. `test_c23.py`, then the smoke run (`cancel_eval.py run ... --smoke`: one test per stratum and the BV-like test, 20
   tests) and `score` on it. A fault found there is fixed and disclosed here before the lock.
2. Lock: the commit with this Addendum, the test log and the smoke output.
3. Scored run into `data/2026-10-07/cancel`, then `score` and `cancel_verify.py`.

## 6. What the verdicts decide

- K1, K2 and K4 CONFIRMED and K5 not REFUTED: item 50 is proposed for acceptance (not release) with items 46-49.
- K1, K2 or K4 REFUTED: item 50 is not proposed; the failing tests are examined first.
- K5 REFUTED: item 50 is not proposed in this form (the pass would cost too much on every input).


---

<!-- ===== Addendum 395 (source: spare-qubit-cliff-addendum-395-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 394, and the owner's decision of 2026-10-07.

## Addendum 395 -- Results of CANCEL (Addendum 394): K1, K2, K4 and K5 CONFIRMED on 140 Benchpress tests; commutative cancellation was tried on 32 and kept on 20 (Benchpress's BV-like test 392 to 0, others up to 29% fewer two-qubit gates) and never gave more; item 50 accepted, not released (2026-10-07)

**Status: results of the pre-registered test in Addendum 394, scored by the locked script and re-checked by the
independent [`benchmarks/cancel_verify.py`](../../benchmarks/cancel_verify.py); then the owner's decision.**

## 1. The run

- **Lock:** commit `9c341a3` (Addenda 393-394, with c23, its test and the smoke run); the run recorded `git_head`
  `9c341a3` and no uncommitted change to a tracked file. The lock commit and this one reach GitHub together.
- **Machine:** the workplace PC (Windows, Python 3.11.9, Qiskit 2.5.2), Benchpress `b695f30`, six jobs at a time.
- **Time:** started 2026-10-07T06:41:11Z; 420 jobs (140 tests, three arms) in 1,647 s.
- **Output:** [`data/2026-10-07/cancel/`](../../data/2026-10-07/cancel/): `cancel.json`, `score.md`, `run_log.txt`,
  `score_log.txt`, `verify_log.txt`; the smoke run in
  [`data/2026-10-07/cancel_smoke/`](../../data/2026-10-07/cancel_smoke/).
- **Disclosed:** the log of [`test_c23.py`](../../patches/psf_compile_c23_2026-10-07/test_c23.py) (9 passed, run
  before the smoke run and the lock) was left out of the lock commit; it is committed here as
  [`data/2026-10-07/cancel/test_c23_log.txt`](../../data/2026-10-07/cancel/test_c23_log.txt), unchanged apart from
  the local path.

## 2. Results

**P0: PASS.** The 140 tests of BP-MOCK and BP-MOCK2 (`cancel_verify.py`: same set); no error or time-out in any arm;
every C23 output passes Benchpress's validator; versions as named. All three arms finished all 140 tests.

| | prediction | result | verdict |
|---|---|---|---|
| K1 | nothing cancels and C22 = C22B (102 tests): C23 returns C22's circuit | 0 of 102 differ | **CONFIRMED** |
| K2 | some two-qubit gates cancel (32 tests): C23 has no more than C22 | 0 of 32 have more | **CONFIRMED** |
| K4 | checkable C23 outputs (41) implement their input | 0 of 41 fail | **CONFIRMED** |
| K5 | nothing cancels (108 tests): median time C23 / C22 <= 1.15 | 1.023 | **CONFIRMED** |

`cancel_verify.py`: the same four verdicts, "verdicts identical to score.md: True". Of the 108 tests where nothing
cancels, 6 are outside K1 because C22 did not reproduce itself (C22 against C22B; Addendum 393, section 2).

**Reported without prediction** (the full table is in `score.md`):

- **Cancelled input kept on 20 of 32**, two-qubit count C22 to C23:

  | test | C22 | C23 | change | time C23 / C22 |
  |---|---|---|---|---|
  | `BVlike_simplification` | 392 | 0 | -100% | 1.39 |
  | HamLib `enc_unary_dvalues_4-4-4`, linear | 20,982 | 14,833 | -29.3% | 1.67 |
  | QASMBench `error_correctiond3_n5`, square | 56 | 41 | -26.8% | 1.11 |
  | QASMBench `qft_n160`, heavy-hex | 21,361 | 15,793 | -26.1% | 1.24 |
  | HamLib `BK22`, linear | 256,735 | 207,796 | -19.1% | 1.78 |
  | HamLib `JW12`, heavy-hex | 6,175 | 5,007 | -18.9% | 1.50 |
  | HamLib `JW-10`, FakeTorino | 2,854 | 2,484 | -13.0% | 1.46 |
  | HamLib `JW-14`, heavy-hex | 15,941 | 14,364 | -9.9% | 3.96 |
  | HamLib `JW-18`, FakeTorino | 43,491 | 39,717 | -8.7% | 3.09 |
  | HamLib `bh_graph` triag Lx-11, linear | 64,436 | 59,389 | -7.8% | 1.97 |
  | HamLib `enc_gray_dvalues_8-8-8`, FakeTorino | 14,298 | 13,261 | -7.3% | 1.17 |
  | HamLib `parity10`, FakeTorino | 2,723 | 2,543 | -6.6% | 1.38 |
  | HamLib `JW-22`, FakeTorino | 129,070 | 120,561 | -6.6% | 1.87 |
  | HamLib `JW-14`, FakeTorino | 13,644 | 12,927 | -5.3% | 1.78 |
  | HamLib `reg-5_n-10`, square | 5,367 | 5,275 | -1.7% | 3.70 |
  | five more (Feynman `hwb10`, HamLib `parity-14`, `bh_graph` grid Lx-7, QASMBench `factor247_n15` square and linear) | | | -0.4% to -0.8% | 1.41-2.21 |

  Over the 19 with a non-zero result, the geometric mean of C23 / C22 is 0.895.
- **Input kept on 12 of 32:** the two compiles gave the same two-qubit count on all 12 (a tie keeps the input's), so
  C23's count equals C22's there; the time was spent for nothing (0.96-2.05 of C22's).
- **Time where cancellation is tried:** C23 / C22 median 1.52 (0.96-3.96); 1.73 where the cancelled input was kept,
  1.39 where the input was. The two compiles run one after the other.
- `CANCEL_STATS`: `failed` 0 on all 140.

## 3. What CANCEL shows

1. **Item 50 does what it was built for:** where the cancellation removes no two-qubit gate the default call returns
   c22's circuit wherever c22 reproduces itself, at about the same time; where it removes some, the result never has
   more two-qubit gates than c22's; every checkable output implements its input.
2. **The gains reach beyond the BV-like test it was written from:** they are largest on Hamiltonian-simulation
   circuits (Jordan-Wigner, Bravyi-Kitaev and parity encodings, unary and Gray encodings) and on the 160-qubit QFT,
   where adjacent terms leave CX gates that commute and cancel.
3. **The price:** on the inputs where it is tried the default call takes about 1.5 times as long, up to 4 times.
4. **Limits:** item 50 was written from one of these 140 tests after BP-MOCK and BP-MOCK2 and not tuned on the
   others, but the 140 are not a fresh sample. **Not established:** Benchpress's other tests; the distance to Qiskit
   level 2 with item 50 (CANCEL has no Qiskit arm; it can be computed from BP-MOCK's and BP-MOCK2's QK rows); depth
   (recorded in `cancel.json`, not scored); the recommended call (not an arm of CANCEL); hardware.

## 4. Decision (the owner, 2026-10-07)

**Item 50 is accepted, not released,** with items 46-49, under the policy of Addendum 385. Candidate c23
([`patches/psf_compile_c23_2026-10-07/psf_compile.py`](../../patches/psf_compile_c23_2026-10-07/psf_compile.py),
items 46-50, as locked in Addendum 394) carries all five and replaces c22 as the basis of the next release.
[`docs/RELEASES.md`](../../docs/RELEASES.md) lists item 50 under "Accepted, not yet released".

The README is to say, with that release: `seed_transpiler` does not make the default call reproducible where Qiskit's
own level 1 is not (Addendum 393, section 2); the recommended call avoids FakeTorino's failed couplers, which Qiskit
level 2 uses, and what that costs (Addendum 393, section 4).


---

<!-- ===== Addendum 396 (source: spare-qubit-cliff-addendum-396-2026-10-07.md) ===== -->

> **Note added when merging:** Exploratory, after Addendum 395, at the owner's request; computed from committed output only.

## Addendum 396 -- Candidate c23's default call against Qiskit level 2 on 139 Benchpress tests (exploratory, from committed output): 1.027 of Qiskit level 2's two-qubit count (2026-10-06.4: 1.328), fewer on 18, as many on 59, more on 62; about 3.9 times its compile time (2026-10-07)

**Status: exploratory, not pre-registered; computed after CANCEL (Addendum 395) from output already committed. No
compile was run.** It gives the number the README needs for the release of c23 (Addendum 397).

## 1. Why and how

CANCEL compared c23 with c22 and had no Qiskit arm. BP-MOCK (Addendum 390) and BP-MOCK2 (Addendum 392) had one: Qiskit
level 2 as Benchpress calls it (QK), on the same 140 inputs, built the same way from the same Benchpress commit
(`b695f30`) on the same workplace PC. [`benchmarks/c23_vs_qk.py`](../../benchmarks/c23_vs_qk.py) puts CANCEL's C23
rows next to those QK rows. A test counts where every arm of its BP-MOCK or BP-MOCK2 run finished and C23 finished:
92 + 47 = 139 (BP-MOCK2's `bwt_n37` on linear, where QK timed out, is left out). Two-qubit counts are compared as
(count + 1) / (count + 1), as in those tests. Output: [`data/2026-10-07/c23_vs_qk/`](../../data/2026-10-07/c23_vs_qk/).

A consistency check: CANCEL's C22 arm gave the same two-qubit count as C22 in BP-MOCK and BP-MOCK2 on 139 of 139
tests, so the runs can be put side by side.

## 2. Results

| sample | tests | C23 / QK | C22 / QK | 2026-10-06.4 / QK |
|---|---|---|---|---|
| BP-MOCK | 92 | 1.023 | 1.115 | 1.364 |
| BP-MOCK2 | 47 | 1.036 | 1.039 | 1.261 |
| both | 139 | **1.027** | 1.089 | 1.328 |

- **Per test:** C23 used fewer two-qubit gates than QK on 18, as many on 59 and more on 62; more by over 10% on 15
  and by over 25% on 4.
- **Per stratum** (19 strata of 5-12 tests): C23 / QK 0.994-1.083; 2026-10-06.4 / QK was 1.005-3.630. The largest
  remaining gaps are QASMBench medium on heavy-hex (1.083), HamLib on FakeTorino (1.066) and on linear (1.062), and
  QASMBench small on all-to-all (1.058).
- **Where C23 uses most more:** small circuits by a few gates (`basis_test_n4` 10 against 6, `wstate_n3` 13 against
  10, `pea_n5` 36 against 29); `bv_n19` on heavy-hex 71 against 55; HamLib `bh_graph` triangular Lx-10 (100 qubits)
  on FakeTorino 25,535 against 21,619 (+18%).
- **Where it uses fewer:** for example HamLib `JW12` on heavy-hex 5,007 against 5,547 (-10%), a 324-qubit HamLib
  graph on square 6,280 against 6,594, Feynman `mod_red_21`, `grover_5`, `circSU2_89`, `bv_n140` on linear.
- **Compile time** (C23 and QK come from different runs; indicative only): C23 / QK median 3.87, geometric mean
  3.72, range 0.10-19.0.

## 3. What this shows, and what it does not

1. With items 48 and 50 the default call's two-qubit count on this Benchpress sample is close to Qiskit level 2's
   (2.7% more as a geometric mean), where 2026-10-06.4 used 33% more. It is not lower: it uses more on more tests
   than it uses fewer.
2. It costs compile time: about four times Qiskit level 2's.
3. Limits: exploratory; c23 was built after BP-MOCK and BP-MOCK2 had shown these inputs (item 48 from DISPATCH-PROBE,
   item 50 from the BV-like test), so this is not an independent sample; one run per arm; Benchpress's other tests,
   depth and hardware are not covered. On FakeTorino's 100-qubit tests both the default call and QK use couplers the
   device reports as failed (the default call does not read the target); the recommended call does not (Addendum 393,
   section 4).


---

<!-- ===== Addendum 397 (source: spare-qubit-cliff-addendum-397-2026-10-07.md) ===== -->

> **Note added when merging:** The owner's release decision of 2026-10-07, committed with the release's files after the release tests.

## Addendum 397 -- Release: psf_compile 2026-10-07.1 = candidate 2026-10-07.c23 of Addendum 394 (changelog items 46-50, accepted in Addenda 385, 388, 392 and 395); the AI front end stays a12 (2026-10-07)

**Status: the owner's decision of 2026-10-07 (home), under the release policy of Addendum 385: improvements are
accepted one by one and released together.**

## 1. The decision

The owner released candidate c23, which carries items 46-50, as **2026-10-07.1**. The evidence, item by item:

| item | what it does | test | result |
|---|---|---|---|
| 46 | the recommended call's checks and estimates apply fewer, larger matrices; estimates within 1e-12 are a tie | FUSE (Addenda 383-385) | 0.36-0.49 of 2026-10-06.4's time at 16 qubits; the same circuit on 359 of 360, the other a near-tie |
| 47, 49 | item 39's checks only where they can change the output; estimates on 2x2 reduced states | TRACK (Addenda 386-388) | 0.37-0.51 of c19's time at 16 qubits; c19's circuit on 360 of 360 |
| 48 | the default call expands instructions on three or more qubits first | BP-MOCK, BP-MOCK2 (Addenda 389-392) | 0.71 and 0.74 of 2026-10-06.4's two-qubit count on such inputs; nothing else changed |
| 50 | commutative cancellation tried where it removes two-qubit gates | CANCEL (Addenda 394-395) | kept on 20 of 32 tried, never more two-qubit gates; elsewhere c22's circuit |

Together, on 139 Benchpress tests, the default call's two-qubit count is 1.027 times Qiskit level 2's (2026-10-06.4:
1.328; Addendum 396, exploratory). The AI front end is unchanged (a12); it calls `psf_compile`, so it gets items
46-50.

## 2. What changes in the repository

- **[`psf_compile.py`](../../psf_compile.py)** is c23's file with the two version lines changed
  (`VERSION: 2026-10-07.1 -- release ...` and `VERSION = "2026-10-07.1"`). Nothing else differs;
  [`benchmarks/test_release_2026_10_07_1.py`](../../benchmarks/test_release_2026_10_07_1.py) checks this.
- **The outgoing release 2026-10-06.4** is kept unchanged as
  [`patches/psf_compile_release_2026-10-06.4/psf_compile.py`](../../patches/psf_compile_release_2026-10-06.4/psf_compile.py).
- **Tests:**
  - [`benchmarks/test_release_2026_10_07_1.py`](../../benchmarks/test_release_2026_10_07_1.py) (new): the version and
    the new counters; the file equals c23's except the version lines; c23 and the kept 2026-10-06.4 are the locked
    files; on four devices 2026-10-06.4's circuit with the recommended and the default call (6, 9 and 20 qubits,
    nothing to expand or cancel); Benchpress's BV-like circuit at 8 and 30 qubits compiles to no two-qubit gate
    (2026-10-06.4 keeps them); a circuit with `ccx` gates is expanded and the output is valid and exact; three
    random circuits with commuting CX pairs give valid, exact outputs with no more two-qubit gates than 2026-10-06.4.
  - [`benchmarks/test_release_2026_10_06_4.py`](../../benchmarks/test_release_2026_10_06_4.py) and
    [`patches/psf_compile_c19_2026-10-07/test_c19.py`](../../patches/psf_compile_c19_2026-10-07/test_c19.py) now load
    2026-10-06.4 from the kept copy, because they compare against it.
  - Every other test that asserted the current release's version (24 lines in `benchmarks/` and `patches/`) now
    asserts `"2026-10-07.1"`.
  - [`benchmarks/test_release_2026_10_03_2.py`](../../benchmarks/test_release_2026_10_03_2.py): one comparison now
    uses the release's own tie rule (section 3).
  - The locked scripts of the tests (`fuse_eval.py`, `track_eval.py`, `bp_mock.py`, `bp_mock2.py`,
    `cancel_eval.py` and their verifiers) are records and are not changed; they name the releases of their day.
- **README:** "Current version" names 2026-10-07.1 with items 46-50; "Where it is weaker" reports the Benchpress
  results (1.03 times Qiskit level 2's two-qubit count, about 4 times its compile time), the non-reproducibility of
  Qiskit's level 1 that the default call inherits, and the failed couplers (Addendum 393).
- **[`docs/RELEASES.md`](../../docs/RELEASES.md):** a new "Current version" block replaces the "Accepted, not yet
  released" block; the 2026-10-06.4 block becomes "Previous release".

## 3. Checks before the commit

[`benchmarks/run_release_tests_2026-10-07_1.py`](../../benchmarks/run_release_tests_2026-10-07_1.py) ran the 31 test
files that load the repository's `psf_compile.py` (the 27 this commit adds or changes, and four that load it
unchanged), one pytest session per file, at home (WSL2; Python 3.12.13, Qiskit 2.5.2, NumPy 2.5.3, `psf_compile`
2026-10-07.1, core 2026-09-29.1; on commit `038417d` with these changes uncommitted). Logs:
[`data/2026-10-07/release_2026-10-07.1/`](../../data/2026-10-07/release_2026-10-07.1/).

**Run 1** (10:23:31-10:29:52 UTC): 250 passed, 2 failed.

1. `test_release_2026_10_07_1.py::test_bvlike_cancels[30]`: **a defect of the new test.** It asked `_implements` to
   confirm a 30-qubit output; `_implements` returns False for any circuit of more than 16 qubits (it cannot check
   them; item 44). The output was valid and had no two-qubit gate. The test now checks exactness at 8 qubits only.
2. `test_release_2026_10_03_2.py::test_compare_returns_lower_estimate_exact_and_safe[FakeAuckland]`: **a comparison
   made stricter than the release.** The test expects level 3's circuit whenever its `excitation_cost` is lower at
   all. [`benchmarks/release_10071_diag.py`](../../benchmarks/release_10071_diag.py) repeated the test's four
   circuits with 2026-10-07.1, 2026-10-06.4 and c19 (output `diag_out.txt`): on `chain5` the two estimates are equal
   in 2026-10-06.4 and c19 and differ by 1.6e-16 (relative) in 2026-10-07.1, from the different order of
   floating-point operations of item 49; all three releases return the same circuit (their own) on `chain5` and the
   same choice on the other three circuits, where the estimates differ by 1.5-7.7%. Under item 46 such a difference is
   a tie that keeps the own circuit. The test now decides with the release's `_lower` (the tie band
   `ESTIMATE_TIE_TOL`); before item 46 the two rules agree.

**Run 2** (ended 10:43:17 UTC), after the two changes: **252 passed**, none failed or errored, on all 31 files.

| file | normalized SHA-256 |
|---|---|
| [`psf_compile.py`](../../psf_compile.py) (release 2026-10-07.1) | `73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc` |
| [`patches/psf_compile_release_2026-10-06.4/psf_compile.py`](../../patches/psf_compile_release_2026-10-06.4/psf_compile.py) (kept) | `69fe51d2d503638ceb4d067a0d86a5b27c38586694ec84996dea5e7ab6dab7aa` |
| [`patches/psf_compile_c23_2026-10-07/psf_compile.py`](../../patches/psf_compile_c23_2026-10-07/psf_compile.py) (candidate, unchanged) | `568de9e691796e4efb330dff3f2ed61aa1cca4cc888286f9727854cf8c744420` |
| [`benchmarks/test_release_2026_10_07_1.py`](../../benchmarks/test_release_2026_10_07_1.py) (new) | `082681a6524db803bec28a73d75e1ee3f0f0430d4f87dc4c2acd6bb949435894` |
| [`benchmarks/test_release_2026_10_03_2.py`](../../benchmarks/test_release_2026_10_03_2.py) (changed) | `8efcc598f259dd4a3ea793f0533a0678d0b048dc64edb817ee480c0762d209ef` |

## 4. What 2026-10-07.1 does not establish

- **The recommended call where items 48 and 50 act.** Both act at the start of the pipeline without a target, so
  also in the recommended call's first compile. BP-MOCK's M4 checked item 48 there on 20 FakeTorino tests; item 50
  there was not part of a pre-registered test.
- **Benchpress beyond the 139 tests**, depth, and hardware. The 139 tests also informed items 48 and 50.
- **Reproducibility:** where Qiskit's own level 1 is not reproducible, neither is the default call (Addendum 393).
- **Time:** the default call takes about 4 times Qiskit level 2's compile time on the Benchpress sample, and where
  item 50's cancellation removes something it compiles twice.


---

<!-- ===== Addendum 398 (source: spare-qubit-cliff-addendum-398-2026-10-07.md) ===== -->

> **Note added when merging:** Exploratory, after the release of Addendum 397, at the owner's request.

## Addendum 398 -- REC-PROBE (exploratory): release 2026-10-07.1's recommended call on BP-MOCK's 20 FakeTorino tests: never more two-qubit gates than 2026-10-06.4, fewer on 8 (geometric mean 0.62; BV-like 1,071 to 0); and 2026-10-06.4 placed gates on failed couplers on two inputs where they could be avoided, which 2026-10-07.1 does not (2026-10-07)

**Status: exploratory, not pre-registered; run at home after the release (Addendum 397), at the owner's request.** It
answers the first open point of Addendum 397, section 4: what items 48 and 50 do inside the recommended call.

## 1. The run

[`benchmarks/rec_probe.py`](../../benchmarks/rec_probe.py): the 20 FakeTorino tests of BP-MOCK (HamLib 8, Feynman 6,
100-qubit 6), built as there (`bp_mock.build`, Benchpress `b695f30`); three arms, each test and arm in its own
process: **R4** 2026-10-06.4's recommended call (the kept copy), **R1** 2026-10-07.1's recommended call, **D1**
2026-10-07.1's default call. Home (WSL2, Python 3.12.13, Qiskit 2.5.2), commit `724acbd`, no uncommitted change;
started 2026-10-07T11:42:04Z, 60 jobs in 85 s, four at a time. Output:
[`data/2026-10-07/rec_probe/`](../../data/2026-10-07/rec_probe/) (`rec_probe.json`, `score.md`); the smoke run
(BV-like and `mod5_4`) in [`data/2026-10-07/rec_probe_smoke/`](../../data/2026-10-07/rec_probe_smoke/).

## 2. Results

- **R4 reproduces BP-MOCK's RELR** (workplace PC) on all 20: the same two-qubit count.
- **No error; every output passes Benchpress's validator.** On the 7 inputs of at most 16 qubits that item 39 can
  check, R1's output implements the input.
- **Two-qubit count, R1 against R4:** fewer on 8, the same on 12, more on none; geometric mean 0.62 (0.87 without
  the BV-like test).

| test | R4 | R1 | what acted in R1 |
|---|---|---|---|
| `BVlike_simplification` | 1,071 | **0** | item 50 (cancelled input kept) |
| HamLib `JW-22` | 182,362 | 120,561 (-34%) | items 48 and 50 |
| HamLib `bh_graph` triangular Lx-3 (132 qubits) | 91,486 | 58,199 (-36%) | item 48 |
| HamLib `JW-18` | 58,085 | 39,717 (-32%) | items 48 and 50 |
| Feynman `qcla_com_7` | 803 | 362 | item 48 (cancellation tried, input kept) |
| Feynman `gf2^6_mult` | 710 | 561 | item 48 |
| `circSU2_89` | 1,719 | 1,344 | item 48 |
| HamLib `bh_graph` triangular Lx-10 (100 qubits) | 26,221 | 25,616 | item 48 |

  On the other 12 the circuits' two-qubit counts are equal; on HamLib `JW-10` and `parity10` the cancellation was
  tried twice (first compile and floor candidate) and Qiskit level 3's candidate was chosen, as with R4.
- **Failed couplers** (error at least 0.5; FakeTorino reports 22 directed couplers, isolating 4 qubits). R4 placed
  two-qubit gates on them in three tests, each time with `PRUNE_STATS["unavoidable"]` (it recompiled on the pruned map,
  got a circuit that still used them, and returned the first circuit with a warning, item 43):
  - HamLib `bh_graph` Lx-10 (100 qubits): 1,438 gates; R1 **0**;
  - `circSU2_89` (89 qubits): 140 gates; R1 **0**;
  - HamLib `bh_graph` Lx-3 (132 qubits): 6,542 gates; R1 3,908, also `unavoidable` (132 logical qubits do not fit
    in the 129 qubits left).
  In the first two the input is one wide instruction (`PauliEvolutionGate`, `EfficientSU2`) on fewer qubits than
  the device has without its failed elements; with item 48 the pruned recompile works on the expanded circuit and
  avoids them. Why the recompile of the unexpanded input could not was not examined further.
- **Default call (D1):** no target, so it does not look at failed couplers; it used them on 14 of 20 tests (16 to
  9,533 gates), as BP-MOCK's default calls and Qiskit level 2 did (Addendum 393).
- **Time:** total R4 57.9 s, R1 74.9 s, D1 48.8 s; R1 / R4 median 1.00 per test. R1 is faster on the small HamLib and
  Feynman tests that build level 3 (items 46, 47, 49; `JW-10` 12.1 to 6.4 s) and slower where item 48 or 50 adds a
  compile or makes the pruned recompile succeed (`circSU2_89` 0.21 to 4.43 s, `bh_graph` Lx-10 1.75 to 5.80 s,
  `JW-18` 4.67 to 8.63 s).

## 3. What this shows

1. Inside the recommended call items 48 and 50 do what they do in the default call: on these 20 tests they never
   added a two-qubit gate, removed a third of them on three large HamLib inputs, and removed all of them on the
   BV-like test.
2. **A defect of 2026-10-06.4 found here:** on inputs that are one instruction over many qubits, its recommended call
   could fail to avoid failed couplers even where the device had room, and returned the circuit with item 43's
   warning. 2026-10-07.1 avoids them on both such inputs here. It was not a silent error (the warning names it). The
   README's statement that no failed coupler was used refers to the pre-registered tests of 1,506 circuits per
   device; the README now qualifies it, and [`docs/RELEASES.md`](../../docs/RELEASES.md) notes the defect under
   2026-10-07.1.
3. Not established: other devices; inputs with wide instructions beyond these; that the recompile always succeeds
   once the input is expanded.


---

<!-- ===== Addendum 399 (source: spare-qubit-cliff-addendum-399-2026-10-07.md) ===== -->

> **Note added when merging:** Pre-registration of DEPTH-R, committed with its scripts and dry run as the lock, before the scored run.

## Addendum 399 -- Pre-registration: DEPTH-R. QML-2 "DEPTH" stage 1 (Addenda 345-346) again, now scorable: its exactness check stated as a state infidelity, new data (split seed 4), and release 2026-10-07.1's calls; predictions H1-H6 and the stage-2 gate word for word (2026-10-07)

**Status: pre-registration, written after DEPTH-R's dry run (development seed 2) and before any circuit of split seed
4 is trained, compiled or simulated.** The lock is the commit that adds this Addendum with the three scripts and the
dry run's output; the scored run follows it. Owner's go-ahead: 2026-10-07 ("1": the same question again).

## 1. Why

DEPTH asked how a data re-uploading classifier trained on real data behaves on a noisy fake device as it is made
deeper, how much compilers move that, and whether fine-tuning through the noise helps. Its P0 failed on a design error:
it required every compiled circuit's noiseless z within 1e-6 of the logical z, and two Qiskit level 3 circuits of
286-289 two-qubit gates were off by 1.03e-5 and 8.7e-6, a state infidelity of order 1e-10 (Addendum 346, section 1).
So nothing was scored, and its "NO-GO" for a GPU stage was not a result. DEPTH-R asks the same question with the
check stated as Addendum 346 says it should be, on data no run has used, with the current release.

## 2. Design

[`benchmarks/depth_r_eval.py`](../../benchmarks/depth_r_eval.py) imports DEPTH's
[`depth_eval.py`](../../data/2026-10-05/workplace/depth1/depth_eval.py) unchanged (normalized SHA-256 `82488d96...`,
checked at start) and uses its data, model, training, noise simulation, shots, readout and fine-tuning as they are
(Addendum 345, section 2). What differs:

| | DEPTH (Addendum 345) | DEPTH-R |
|---|---|---|
| data | split seed 1, init seed 1 | **split seed 4, init seed 4** (0 pilot, 1 DEPTH, 2 development, 3 READOUT) |
| arms | RPSF (c12, target + `placement_refine`), C12 (c12, recommended call), L3T | **RPSF** (release 2026-10-07.1, target + `placement_refine`), **REC** (release 2026-10-07.1, recommended call: + `final_resynthesis="select"`, `compare_level3`, `compare_floor`, `candidate_score="hybrid"`), **L3T** (Qiskit level 3 with the Target, `approximation_degree=1.0`; unchanged) |
| fine-tuning arm | C12 | REC |
| P0, exactness | \|z compiled - z logical\| <= 1e-6 | **state infidelity <= 1e-6**: the compiled circuit's noiseless state on the qubits it touches against the logical circuit's output placed at the compiled circuit's final layout (other touched qubits in \|0>); \|z difference\| recorded and reported, not gated |
| P0, other | 24 deployment and 4 fine-tuning files; reduced simulation = whole-device simulation within 1e-9 on the first two points of every (L, file) | the same, and: every output names release 2026-10-07.1 and the locked release file, and split and init seeds 4 |

Everything else as in DEPTH: BC (114 test points) and D38 (72); n in {4, 6}; L in {1, 2, 4, 8, 12, 16}; FakeAuckland
(cx) and FakeTorino (cz); noise restricted to the touched qubits, Aer density matrix; readout of the qubit carrying
logical 0; 4,000 shots x 20 repetitions with the same random numbers in every arm; fine-tuning BC, n = 6, FakeAuckland,
L in {4, 12}, seeds 1 and 2, 40 SPSA steps, FTN and FT0. Run by
[`benchmarks/run_depth_r.sh`](../../benchmarks/run_depth_r.sh) (4 training, 24 deployment and 4 fine-tuning jobs, six
at a time) at home (WSL2; Python 3.12.13, Qiskit 2.5.2, qiskit-aer as installed, scikit-learn 1.8.0, Rust core
2026-09-29.1); checked by the independent [`benchmarks/depth_r_verify.py`](../../benchmarks/depth_r_verify.py) (reads
the raw JSON only, imports neither harness).

## 3. Predictions (scored only by `depth_r_eval.py score`)

**P0** as in section 2; if it fails nothing below is scored.

H1-H6 and the gate are DEPTH's (Addendum 345, section 3) word for word, with C12 read as REC:

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | depth stops paying in margin (FakeAuckland, n = 6) | for both datasets and every arm, the deployed margin at L = 16 is below 0.8 x the best margin over L | for any dataset and arm, L = 16 has the largest margin |
| H2 | ... and in accuracy (FakeAuckland, n = 6, shot-based with readout) | for at least one dataset, in every arm, the shot accuracy at L = 16 is at least 2 test points below the best over L | for every dataset and arm, L = 16 is the best (or tied best) |
| H3 | the recommended call keeps more margin than the guarded call | pooled over datasets, n and L, REC - RPSF mean margin >= +0.005 on both devices | < 0 on either device |
| H4 | REC is level with error-aware Qiskit | pooled \|REC - L3T\| margin <= 0.01 on both devices | REC < L3T - 0.02 on either device |
| H5 | REC flips no more predictions than RPSF | pooled exact flip rate (noisy sign != noiseless sign) REC <= RPSF on both devices | REC > RPSF + 0.01 on either device |
| H6 | fine-tuning through the noise helps where noise binds | at L = 12, mean over seeds of FTN - DEP deployed margin >= +0.02, and larger than FT0 - DEP | FTN - DEP < -0.01 |

**Gate for stage 2 (GPU):** GO if (H1 or H2 CONFIRMED) and (H3 CONFIRMED, or some cell has |shot accuracy REC - RPSF|
of at least 2 test points); NO-GO otherwise.

**Reported without prediction:** the full table; FakeTorino for H1 and H2; n = 4; fine-tuning at L = 4; compile
times; the largest |z difference|; and **the noise's share of H2** (FakeAuckland, n = 6: ideal accuracy minus shot
accuracy per L, in test points). DEPTH's H2 line was confounded by the noiseless model's own variation between depths
(Addendum 346, section 2); this report separates the two. It is not a prediction, and H2 is scored as written.

## 4. What is known, and what is expected (disclosed)

- **DEPTH's unscored outcome** (Addendum 346; split seed 1, release candidate c12): H1 and H2 lines "CONFIRMED" (H2
  confounded), H3 AMBIGUOUS (FakeAuckland +0.0049 against the +0.005 bound; FakeTorino +0.0187), H4 CONFIRMED, H5
  AMBIGUOUS (FakeTorino flip 0.0039 against 0.0036), H6 AMBIGUOUS (FTN - DEP -0.002); gate NO-GO. Noise removed
  77-80% of the margin at L = 16 and cost at most 2 test points in any cell.
- **DEPTH-R's dry run** (development seed 2, 12 points, L in {1, 4, 12}, 60 training and 3 fine-tuning steps; 109 s;
  output in [`data/2026-10-07/depth_r_dry/`](../../data/2026-10-07/depth_r_dry/)): the harness ran end to end; P0
  passed (largest state infidelity 8.2e-15, |z difference| 2.5e-14, reduced against whole device 0 on 144
  circuits); its verdict lines, which are not results: H1 CONFIRMED, H2 AMBIGUOUS, H3-H5 CONFIRMED (REC - RPSF
  +0.0070 and +0.0150), H6 AMBIGUOUS, gate GO. Its table's "ideal" column is over the whole test set while the dry
  run deploys 12 points, so the dry run's noise-share line is not meaningful; in the scored run both use the whole
  test set.
- **Expectations:** H1 expected again. H2 open (DEPTH's evidence is that noise costs at most about 2 points). H3 is
  near its bound on FakeAuckland, so AMBIGUOUS is likely there; on FakeTorino the 4-qubit ring's routing difference
  (Addendum 346) should give REC the margin. H4 expected. H5 and H6 open. The gate depends mostly on H3. As in
  DEPTH, REC's estimate shares the simulator's physics, so H3 and H4 favour it by construction.
- **The release on these circuits:** items 48 and 50 are not expected to act (no instruction on three or more qubits; CZ layers
  separated by RY gates, so no two-qubit gate cancels); items 46, 47 and 49 change the recommended call's time, not
  its circuit (Addenda 385, 388); the circuits are compiled without measurements, so the readout terms of
  2026-10-06.1 do not act.

## 5. What this will not establish

Hardware; more than 6 logical qubits; other models, encodings or optimisers; reliability over training seeds (one
noiseless training per (dataset, n, L), two fine-tuning seeds); a GPU stage itself (the gate only says whether this
evidence supports proposing one).

## 6. Files locked (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/depth_r_eval.py`](../../benchmarks/depth_r_eval.py) | `6f636b677f1a481ef3492bf0de467893cde70359c717bc901b6cb5cd2838f3b6` |
| [`benchmarks/run_depth_r.sh`](../../benchmarks/run_depth_r.sh) | `76487e410d656467308f272d2609fd755511b4356c4e76c142e752ff1f361cc6` |
| [`benchmarks/depth_r_verify.py`](../../benchmarks/depth_r_verify.py) | `b378f17205cbd1c7351e4fac44f0e4a047e403f4ec7a018ef0395042a48d12c8` |
| [`data/2026-10-05/workplace/depth1/depth_eval.py`](../../data/2026-10-05/workplace/depth1/depth_eval.py) (DEPTH, unchanged) | `82488d96144bb1c88f69676a0e6a642c22d4756a47d8a1ad8b07be5327c796ac` |
| [`psf_compile.py`](../../psf_compile.py) (release 2026-10-07.1) | `73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc` |

**Scored run:** `PAR=6 bash benchmarks/run_depth_r.sh data/2026-10-07/depth_r`, then
`python benchmarks/depth_r_verify.py data/2026-10-07/depth_r`; results in Addendum 400.


---

<!-- ===== Addendum 400 (source: spare-qubit-cliff-addendum-400-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 399.

## Addendum 400 -- Results of DEPTH-R (Addendum 399): P0 passed; H1-H4 CONFIRMED, H5 and H6 AMBIGUOUS, gate GO. But the compilers moved the margin, never the accuracy by 2 test points; H2's drop at L = 16 is the noiseless model's, not the noise's; and H5's AMBIGUOUS is a floating-point tie (2026-10-07)

**Status: results of the pre-registered test in Addendum 399, scored by the locked harness and re-checked by the
independent [`benchmarks/depth_r_verify.py`](../../benchmarks/depth_r_verify.py).**

## 1. The run

- **Lock:** commit `49637e9` (Addendum 399 with the scripts and the dry run), pushed before the run; no uncommitted
  change to a tracked file at the start.
- **Machine:** home (WSL2; Python 3.12.13, Qiskit 2.5.2, qiskit-aer 0.17.2, scikit-learn 1.8.0, NumPy 2.5.3, Rust
  core 2026-09-29.1), six jobs at a time; 32 jobs in 1,060 s.
- **Output:** [`data/2026-10-07/depth_r/`](../../data/2026-10-07/depth_r/) (`score.md`, every job's JSON and log,
  `env.txt`, `progress.txt`, `verify.txt`). `depth_r_verify.py`: the same P0, the same six verdicts and the same
  gate ("identical to score.md: verdicts True, gate True, P0 True").

## 2. Results

**P0: PASS.** 24 of 24 deployment files and 4 fine-tuning files; largest state infidelity **1.13e-9** (bound 1e-6);
reduced against whole-device simulation 0 on 288 circuits; every output names release 2026-10-07.1, its locked file
and seeds 4. Recorded, not gated: largest |z compiled - z logical| 7.45e-7 (this time below DEPTH's old 1e-6 too).

| | result | verdict |
|---|---|---|
| H1 margin stops paying (FakeAuckland, n = 6) | best margin at L = 2 (0.548-0.591), at L = 16 0.160-0.216, in every arm and dataset | **CONFIRMED** |
| H2 shot accuracy stops paying | D38: 0.903-0.904 at L = 16 against best 0.933-0.936 (2.2-2.4 test points of 72), every arm; BC: 0.956-0.959 against 0.961-0.962 | **CONFIRMED** |
| H3 REC - RPSF pooled margin >= +0.005 on both devices | FakeAuckland **+0.0056**, FakeTorino +0.0178 | **CONFIRMED** |
| H4 \|REC - L3T\| pooled margin <= 0.01 | +0.0046, +0.0019 | **CONFIRMED** |
| H5 REC flip rate <= RPSF on both devices | FakeAuckland 0.00935 and 0.00935; FakeTorino 0.0042 and 0.0042 | **AMBIGUOUS** (section 3, point 4) |
| H6 FTN - DEP >= +0.02 at L = 12, and above FT0 - DEP | FTN - DEP **+0.0197**, FT0 - DEP -0.0120 | **AMBIGUOUS** |

**Gate: GO** (H1 CONFIRMED and H3 CONFIRMED). No cell had |shot accuracy REC - RPSF| of 2 test points.

## 3. Reading

1. **Noise eats the margin, not the answers.** On FakeAuckland, n = 6, the noise removes 7-10% of the noiseless
   margin at L = 2 and 70-75% at L = 16. Predictions flipped by the noise: pooled 0.94% (FakeAuckland) and 0.42%
   (FakeTorino), at most 4 of 72 points in one cell (D38, n = 6, L = 16).
2. **H2 is confirmed by the noiseless model, as in DEPTH.** The pre-registered report of the noise's share (ideal
   accuracy minus shot accuracy) is -0.1 test points for D38 at L = 16: the noiseless model itself scores 0.903
   there, against 0.944 at L = 4. For BC the noise costs +0.7 to +1.0 points at L = 16. As a statement about noise,
   H2 does not hold; the prediction, scored as written, is CONFIRMED.
3. **The recommended call keeps more margin; it does not change an answer.** FakeAuckland +0.0056 is just above the
   bound (DEPTH: +0.0049, just below). FakeTorino's +0.0178 comes mostly from the 4-qubit rings, which the guarded
   call routes with 188 two-qubit gates at L = 16 against 157 (margin 0.338 against 0.404 for BC, 0.350 against
   0.424 for D38: +20%), as in DEPTH. REC is level with Qiskit level 3 with the Target (H4).
4. **H5's AMBIGUOUS is a floating-point tie.** Counted from the table, REC and RPSF flip exactly as many predictions
   on FakeAuckland: REC one more in D38, n = 4, L = 16 and one fewer in D38, n = 6, L = 4. The scorer's mean of the
   24 cells put REC 1.7e-18 above RPSF, so "REC <= RPSF" read false. On the prediction's own words an exact tie
   satisfies it; the scored verdict stays AMBIGUOUS. Lesson (as in Addendum 397, section 3): a scorer that compares
   means of counts needs a tolerance or compares the counts.
5. **Fine-tuning through the noise did not hurt and the noiseless control did:** FTN - FT0 +0.032 at L = 12 (DEPTH:
   +0.035), but FTN - DEP stayed just below the +0.02 bound.

## 4. The gate

By the rule, GO: the evidence supports proposing a stage 2 (GPU). On the substance the case is weak: the compiler
difference that opens the gate is a margin difference at its bound on FakeAuckland, no prediction changed by 2 test
points between compilers, and noise flips few predictions at 4,000 shots. A GPU stage of the same design would most
likely measure the same thing at larger size. **Proposal (not run):** before any GPU, the question where a margin
difference becomes an answer or a cost, the shot budget (Addendum 346, proposal 1): shots needed to reach a target
accuracy and the device time that takes, first computed from DEPTH-R's recorded z (exploratory, no new run). The
owner decides.

## 5. What this does not establish

Hardware; more than 6 logical qubits; other models, encodings, optimisers; reliability over training seeds (one
noiseless training per cell); a noise model independent of the calibration the recommended call reads (both come
from the same fake device, which favours REC and L3T by construction, Addendum 399, section 4).


---

<!-- ===== Addendum 401 (source: spare-qubit-cliff-addendum-401-2026-10-07.md) ===== -->

> **Note added when merging:** Exploratory, after Addendum 400, at the owner's request.

## Addendum 401 -- Exploratory: DEPTH-R's shots and device time to a common accuracy target. The recommended call needs fewer shots than the guarded call in 8 and 11 of 24 cells, the same in 12 (geometric mean 0.86-0.88, median 1.00); with the device's 250 us repetition delay, device time follows shots, not circuit length; above about 255 shots the compilers' accuracies are equal (2026-10-07)

**Status: exploratory, not pre-registered; computed after Addendum 400 at the owner's request ("実機時間の計算"). No
classifier was trained or simulated again;** only the first 3 test points of each cell were compiled again, to
measure circuit durations.

## 1. Method

[`benchmarks/depth_r_qpu_time.py`](../../benchmarks/depth_r_qpu_time.py), on DEPTH-R's recorded output
([`data/2026-10-07/depth_r/`](../../data/2026-10-07/depth_r/)); output in
[`data/2026-10-07/depth_r_qpu_time/`](../../data/2026-10-07/depth_r_qpu_time/).

- **Shots.** For each test point the recorded noisy z and the readout error of the qubit carrying logical 0 give the
  probability q of reading 0. With S shots (S odd) the prediction is the majority outcome; the chance it is right is
  a binomial tail, computed exactly. A cell's expected accuracy at S shots is the mean over its points.
- **Common target:** the noiseless model's accuracy on the cell minus 1 test point, the same for every arm. Reported:
  the smallest S on a fixed grid reaching it.
- **Device time per shot:** the compiled circuit's as-soon-as-possible duration with the fake device's gate durations
  (rz virtual), plus the final measurement, median of 3 recompiled points per cell (the two-qubit count equalled the
  record on all 432); with and without the device's reported `default_rep_delay`, 250 us on both devices.

## 2. Results

| | FakeAuckland | FakeTorino |
|---|---|---|
| cells where all three arms reach the target | 23 of 24 | 24 of 24 |
| shots REC / RPSF: fewer / same / more (cells) | 8 / 12 / 3 | 11 / 12 / 1 |
| shots REC / RPSF, geometric mean (median) | 0.859 (1.00) | 0.881 (1.00) |
| shots REC / L3T, geometric mean (median) | 0.987 (1.00) | 0.824 (1.00) |
| device time REC / RPSF, circuit only / with 250 us delay | 0.882 / 0.860 | 0.945 / 0.883 |
| device time REC / L3T, circuit only / with 250 us delay | 0.991 / 0.988 | 0.866 / 0.827 |

**Expected accuracy at a fixed number of shots** (pooled over the 24 cells; REC - RPSF in percentage points):
FakeAuckland +0.12 at 15 shots, +0.03 at 63, +0.02 at 255, 0.00 at 1,023; FakeTorino +0.59, +0.10, +0.01, 0.00. At
15 shots on FakeTorino Qiskit level 3 is 0.42 above REC.

**Scale:** circuit durations at L = 16 are 14-34 us on FakeTorino and 71-111 us on FakeAuckland, below the 250 us
repetition delay. Example: FakeAuckland, BC, n = 6, L = 16 needs about 1,000 shots per prediction, 0.11 s of circuit
time and 0.37 s with the delay; for the 114 test points about 42 s.

## 3. Reading

1. **Where the compiler shows, it saves shots, not seconds per shot.** Fewer two-qubit gates raise the margin, and
   with it the chance that few shots give the right sign; on half the cells the saving is zero. With the device's
   repetition delay the per-shot time is nearly the same for every compiler, so device time follows the shot count.
   (On FakeTorino's 4-qubit rings the recommended call's circuit has fewer two-qubit gates but a longer schedule,
   17.0 against 14.2 us; why was not examined.)
2. **The effect is confined to small shot budgets.** At 255 shots or more the three compilers give the same expected
   accuracy to within 0.02 points; at 15 shots the recommended call is ahead of the guarded call by 0.1-0.6 points.
3. **"Shots to a target" is a fragile measure.** It depends on the one or two points nearest the decision boundary:
   one cell needs 20,353 shots with Qiskit level 3 and 187 with the others, another flips the order. Its geometric
   means move with a few cells; the medians are 1.00. Accuracy at fixed shot budgets is the stabler measure.
4. **For the benchmark design** ("Task-Oriented Quantum Benchmark", owner's draft of 2026-10-07): report task
   accuracy at fixed shot budgets as the main task-time measure, shots to a target only beside it; count the device's
   repetition delay; and expect compilers to matter for device time mainly when shots are scarce.

## 4. Limits

The same fake noise as DEPTH-R (the calibration the recommended call reads; Addendum 399, section 4); circuit
durations from 3 points per cell; no queueing, no compile time in the device time; one training per cell.


---

<!-- ===== Addendum 402 (source: spare-qubit-cliff-addendum-402-2026-10-07.md) ===== -->

> **Note added when merging:** Pre-registration of CALSPLIT, committed with its scripts and dry run as the lock, before the scored run.

## Addendum 402 -- Pre-registration: CALSPLIT. Does the recommended call's task-level advantage survive when the calibration it compiles with is not the device's? The first neutrality test of the task-oriented benchmark draft: compilers read a stale calibration, the score uses the device's true noise (2026-10-07)

**Status: pre-registration, written after CALSPLIT's dry run (development seed 2) and before any circuit of the scored
data is compiled for it.** The lock is the commit that adds this Addendum with the three scripts and the dry run; the
scored run follows it. Owner's go-ahead: 2026-10-07 ("最初の試験を設計").

## 1. Why

In DEPTH and DEPTH-R the recommended call's estimate and the simulator's noise come from the same fake-device
calibration, so the tests favour calibration-aware compilers by construction (Addenda 345, 399). The benchmark draft
(owner's document of 2026-10-07) makes separating the two its first neutrality rule. CALSPLIT keeps everything of
DEPTH-R except what the compilers see.

## 2. Design

[`benchmarks/calsplit_eval.py`](../../benchmarks/calsplit_eval.py), run by
[`benchmarks/run_calsplit.sh`](../../benchmarks/run_calsplit.sh), checked by the independent
[`benchmarks/calsplit_verify.py`](../../benchmarks/calsplit_verify.py).

- **Unchanged from DEPTH-R** (imported through [`depth_r_eval.py`](../../benchmarks/depth_r_eval.py) and DEPTH's
  [`depth_eval.py`](../../data/2026-10-05/workplace/depth1/depth_eval.py)): data (split seed 4), the trained
  parameters as committed in [`data/2026-10-07/depth_r/`](../../data/2026-10-07/depth_r/) (no training), the circuits,
  BC and D38, n in {4, 6}, L in {1, 2, 4, 8, 12, 16}, FakeAuckland and FakeTorino, the noisy simulation with the
  device's **true** noise model, readout, P0's checks (state infidelity <= 1e-6; reduced = whole-device simulation
  within 1e-9 on the first two points of every (L, file)).
- **What the calibration-aware compilers see:** a stale Target from STALE's `stale_target()`
  ([`benchmarks/stale_eval.py`](../../benchmarks/stale_eval.py), Addendum 334, unchanged): instruction errors x
  exp(N(0, 0.3)), T1 and T2 x exp(N(0, 0.2)), failed entries unchanged; **three draws per device** (`BASE` =
  91,000,000 + 1,000,000 k, k = 0, 1, 2; STALE used 80,000,000).
- **Arms:** calibration-aware, each with every draw: **REC** (release 2026-10-07.1's recommended call), **RPSF** (the
  release with target + `placement_refine`), **L3T** (Qiskit level 3 with the stale Target, `approximation_degree=1.0`);
  calibration-blind, once: **DEF** (the release's default call: coupling map and basis only) and **L3B** (Qiskit level
  3 with coupling map and basis only). 88 deployment jobs.
- **Pooling:** for each device, the mean over the 24 cells (dataset, n, L) of the cell's mean margin y x z (true
  noise), then the mean over draws. Flips (noisy sign != noiseless sign) are counted, not averaged per cell. Every
  comparison carries a tolerance of 1e-12 (Addenda 397, 400: two ties read as differences).

## 3. Predictions (scored only by `calsplit_eval.py score`)

**P0:** 88 files; every state infidelity <= 1e-6; reduced = whole-device within 1e-9; every output names release
2026-10-07.1 and its locked file; every stale Target differs from the true one. Otherwise nothing is scored.

| ID | Prediction | CONFIRMED (both devices) | REFUTED (either device) |
|---|---|---|---|
| K1 | stale calibration still helps | REC - (better blind arm) pooled margin >= +0.002 | < -0.002 |
| K2 | the advantage over the guarded call survives | REC - RPSF >= +0.003 | < 0 |
| K3 | it shrinks little | DEPTH-R's same-calibration REC - RPSF (Addendum 400: +0.0056, +0.0178) minus the stale value <= 0.005 | > 0.01 |
| K4 | REC stays level with Qiskit level 3 (both stale) | \|REC - L3T\| <= 0.01 | REC < L3T - 0.02 |
| K5 | no more flipped answers than without calibration | (flips REC - flips of the better blind arm) / points <= +0.002 | > +0.01 |

"Better blind arm": per device, the blind arm with the larger pooled margin (K1) or with fewer flips (K5).

**Reported without prediction:** each arm's pooled margin and per-draw values; expected accuracy at 15, 63, 255 and
1,023 shots (exact binomial, as Addendum 401); two-qubit gates on couplers the true device reports as failed; compile
times; the full per-cell data.

## 4. What is known (disclosed)

- **The dry run** (development seed 2, 12 points, L in {1, 4, 12}; 88 jobs in 230 s; output in
  [`data/2026-10-07/calsplit_dry/`](../../data/2026-10-07/calsplit_dry/)): the harness ran end to end; P0 passed
  (largest infidelity 6.8e-11); the independent check agreed. Its lines, which are not results: REC - RPSF -0.0029 on
  FakeAuckland (draws +0.0003, -0.0101, +0.0011) and +0.0140 on FakeTorino, so **K2 read REFUTED** and K3 AMBIGUOUS;
  K4 CONFIRMED (+0.0059, -0.0076).
- **Changed after the dry run, before the lock: K1 and K5 now compare with the better blind arm,** not with DEF alone.
  In the dry run DEF placed 1,080 two-qubit gates on FakeTorino's failed couplers (it does not read the target;
  Addenda 393, 398), giving it margin 0.30 against L3B's 0.44 and 32 flips against REC's 0; "calibration helps" would
  have held for that reason alone. Only the score changed; the dry run's deployment files were made by the earlier
  version of `calsplit_eval.py` (normalized SHA-256 `cab715e3...`, deployment code identical) and are re-scored by the
  locked one in the lock commit.
- **Expectations:** K1 expected (the blind arms lose the error-aware placement). K2 open, and the dry run suggests it
  may fail on FakeAuckland: with a wrong calibration the recommended call's estimate-driven choices can be worse than
  the guarded call's. K3 follows K2. K4 expected (L3T reads the same stale Target). K5 expected. STALE (Addendum 335)
  found the release's lead over L3T kept under a stale calibration on other circuits and devices.

## 5. What this will not establish

Real calibration drift (a perturbation model, not a later calibration of a real device); hardware; readout errors
in the stale Target (STALE does not perturb them; these circuits are compiled without measurements, so the readout
term does not act); other tasks than DEPTH-R's classifiers.

## 6. Files locked (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/calsplit_eval.py`](../../benchmarks/calsplit_eval.py) | `0b39ad2edf7249a19dff94d8e2784389a14d7f412af0902a06c16be978e3600d` |
| [`benchmarks/run_calsplit.sh`](../../benchmarks/run_calsplit.sh) | `9b03f924f6449f30dadfd48ee18c11d51ea5506065b363a727643fd1791a5aeb` |
| [`benchmarks/calsplit_verify.py`](../../benchmarks/calsplit_verify.py) | `19c1d3deeb6e85c5c711e50a21a63dc163b7a2d18a2ddcc584d68662bf8d4958` |
| [`benchmarks/stale_eval.py`](../../benchmarks/stale_eval.py) (STALE, unchanged) | `f2dd16caf17d25232f2452b08a5654f83a8ae10459c773dfa3acfc74005a1cc0` |
| [`benchmarks/depth_r_eval.py`](../../benchmarks/depth_r_eval.py) (DEPTH-R, unchanged) | `6f636b677f1a481ef3492bf0de467893cde70359c717bc901b6cb5cd2838f3b6` |
| [`psf_compile.py`](../../psf_compile.py) (release 2026-10-07.1) | `73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc` |

**Scored run:** `PAR=6 bash benchmarks/run_calsplit.sh data/2026-10-07/calsplit`, then
`python benchmarks/calsplit_verify.py data/2026-10-07/calsplit`; results in Addendum 403.


---

<!-- ===== Addendum 403 (source: spare-qubit-cliff-addendum-403-2026-10-07.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 402.

## Addendum 403 -- Results of CALSPLIT (Addendum 402): with a stale calibration, calibration-aware compilation still beats calibration-blind compilation (K1, K5 CONFIRMED) and the recommended call stays level with Qiskit level 3 (K4 CONFIRMED); but on FakeAuckland its lead over the guarded call is gone (K2 REFUTED: -0.0041, one draw -0.0142); on FakeTorino it holds (+0.0194). The blind default call on FakeTorino placed 21,204 gates on failed couplers and lost 17 points of accuracy (2026-10-07)

**Status: results of the pre-registered test in Addendum 402, scored by the locked harness and re-checked by the
independent [`benchmarks/calsplit_verify.py`](../../benchmarks/calsplit_verify.py).**

## 1. The run

- **Lock:** commit `9435659` (Addendum 402 with the scripts and the dry run), pushed before the run; no uncommitted
  change to a tracked file at the start.
- **Machine:** home (WSL2; Python 3.12.13, Qiskit 2.5.2, qiskit-aer 0.17.2, scikit-learn 1.8.0), six jobs at a time;
  88 jobs in 1,745 s.
- **Output:** [`data/2026-10-07/calsplit/`](../../data/2026-10-07/calsplit/) (`score.md`, every job's JSON and log,
  `env.txt`, `progress.txt`, `verify.txt`). The independent check gave the same P0 and the same five verdicts.

## 2. Results

**P0: PASS.** 88 of 88 files; largest state infidelity 3.3e-9; reduced = whole-device simulation on 1,056 circuits;
release 2026-10-07.1 as locked; every stale Target differs from the true one.

| | FakeAuckland | FakeTorino | verdict |
|---|---|---|---|
| K1 REC - better blind arm, pooled margin (>= +0.002) | +0.0308 (vs DEF) | +0.0860 (vs L3B) | **CONFIRMED** |
| K2 REC - RPSF (>= +0.003; REFUTED if < 0) | **-0.0041** (draws +0.0010, -0.0142, +0.0009) | +0.0194 (draws -0.0005, +0.0314, +0.0273) | **REFUTED** |
| K3 shrinkage against DEPTH-R (<= 0.005; REFUTED if > 0.01) | +0.0056 - (-0.0041) = 0.0097 | +0.0178 - 0.0194 = -0.0016 | **AMBIGUOUS** |
| K4 \|REC - L3T\| (<= 0.01) | +0.0049 | -0.0098 | **CONFIRMED** |
| K5 flip-rate difference to the better blind arm (<= +0.002) | -0.0024 (18.7 against 24 of 2,232) | -0.0048 (8.3 against 19) | **CONFIRMED** |

**Reported without prediction** (pooled; true noise; accuracy at a fixed number of shots, exact binomial):

| device | arm | margin | 15 shots | 63 | 255 | 1,023 | gates on failed couplers |
|---|---|---|---|---|---|---|---|
| FakeAuckland | REC | 0.4001 | 0.8666 | 0.9178 | 0.9315 | 0.9341 | 0 |
| | RPSF | 0.4042 | 0.8700 | 0.9190 | 0.9314 | 0.9337 | 0 |
| | L3T | 0.3952 | 0.8648 | 0.9171 | 0.9311 | 0.9338 | 0 |
| | DEF | 0.3693 | 0.8492 | 0.9091 | 0.9286 | 0.9332 | 0 |
| | L3B | 0.3680 | 0.8464 | 0.9068 | 0.9279 | 0.9327 | 0 |
| FakeTorino | REC | 0.4838 | 0.9031 | 0.9309 | 0.9351 | 0.9356 | 0 |
| | RPSF | 0.4645 | 0.8972 | 0.9299 | 0.9350 | 0.9357 | 0 |
| | L3T | 0.4936 | 0.9049 | 0.9311 | 0.9350 | 0.9355 | 0 |
| | DEF | 0.2839 | 0.7386 | 0.7611 | 0.7651 | 0.7656 | 21,204 |
| | L3B | 0.3978 | 0.8541 | 0.9036 | 0.9243 | 0.9318 | 0 |

## 3. Reading

1. **A calibration that is 30% wrong is still worth reading.** Every calibration-aware arm keeps more margin and
   flips fewer answers than the blind arms, on both devices. At 15 shots the recommended call is 1.7 points (FakeAuckland)
   and 4.9 points (FakeTorino) of accuracy ahead of the better blind arm; at 1,023 shots 0.1 and 0.4.
2. **The recommended call's own choices are not robust to a wrong calibration on FakeAuckland.** Its extra steps over
   the guarded call (re-synthesis, the floor candidate and Qiskit level 3, chosen by an estimate) gained +0.0056 in
   DEPTH-R with the true calibration and lost -0.0041 with stale ones, -0.0142 in one draw. At 15 shots that is 0.3
   points behind the guarded call. The estimate picks among candidates that differ little; when its inputs are wrong
   by 30% it picks the worse one about as often as the better one, and sometimes clearly worse.
3. **On FakeTorino the lead holds (+0.0194), because it is structural.** It comes from routing the 4-qubit rings with
   fewer two-qubit gates (DEPTH-R, Addendum 400), which does not depend on the calibration. One draw of three was level
   (-0.0005).
4. **The recommended call stays level with Qiskit level 3 given the same stale Target** (+0.0049, -0.0098; the second
   close to the bound).
5. **The blind default call on a device with failed couplers is a real hazard at the task level.** On FakeTorino it
   placed 21,204 two-qubit gates on couplers the device reports as failed (it does not read the target; Addenda 393,
   398) and its accuracy stayed at 0.766 even at 1,023 shots, against 0.932-0.936 for every other arm. Qiskit level 3
   without a target happened to avoid them here.

## 4. What follows

- **For users (README):** on a device that reports failed couplers, use a target-aware call; the default call can
  lose most of a classifier's accuracy there. With an up-to-date calibration the recommended call is level with Qiskit
  level 3 and ahead of the guarded call; with a calibration that is off by tens of per cent its estimate-driven
  choices are no better than the guarded call's on FakeAuckland.
- **A candidate, not proposed yet:** make the recommended call switch to an alternative only when the estimated gain
  exceeds what calibration error could explain (a margin on the estimate, as item 46's tie band but wider). It would
  need its own pre-registered test with both true and stale calibrations.
- **For the benchmark draft:** the calibration split changes a verdict (K2), so it belongs in the standard tier; the
  blind default call's collapse shows why "hardware fitness" (failed elements) must be its own layer.

## 5. What this does not establish

Real calibration drift (a perturbation model); hardware; why draw 1 hurt the recommended call on FakeAuckland (not
examined); other tasks than DEPTH-R's classifiers; readout errors in the stale calibration.


---

<!-- ===== Addendum 404 (source: spare-qubit-cliff-addendum-404-2026-10-07.md) ===== -->

> **Note added when merging:** A documentation change after Addendum 403: README and RELEASES only, no code and no run. Committed before the files of candidate 2026-10-07.c24 and its test.

## Addendum 404 -- README and RELEASES: what CALSPLIT (Addendum 403) means for users. On a device with failed couplers, pass `target`; the recommended call's estimate-driven choices need a fresh calibration (2026-10-07)

**Status: a documentation change. No code changed and nothing was run; every number is from Addendum 403 or
Addendum 400.**

## 1. What changed

Addendum 403 (section 4) named two things users should know. They are now in the
[`README.md`](../../README.md) and in [`docs/RELEASES.md`](../../docs/RELEASES.md):

- **README, introduction.** The sentence that the recommended call "gave lower simulated infidelity than level 3 on
  every fake device tested" now says that its choice among candidates is only as good as the calibration it reads.
  With a calibration off by tens of per cent it stayed level with level 3. On one of two devices it lost its lead
  over the simpler target-aware call.
- **README, Quick start.** A new paragraph: on a device that reports failed couplers, give the call the device's
  `target` (the recommended call, or at least `target` with `placement_refine=True`). The default call does not read
  the target. In CALSPLIT on FakeTorino it reached 0.77 accuracy even at 1,023 shots, against 0.94 for every
  target-aware call.
- **README, "Where it is weaker".** A new paragraph on CALSPLIT. It gives:
  - what was tested;
  - that reading a calibration that is 30% wrong still beats not reading one (1.7 and 4.9 points of accuracy at 15
    shots, 0.1 and 0.4 at 1,023);
  - that the recommended call's estimate-driven choices need a fresh calibration (FakeAuckland: +0.0056 with the
    true calibration, -0.0041 with stale ones, -0.0142 in one draw; FakeTorino +0.0194, from routing);
  - that it stays level with Qiskit level 3 given the same stale Target;
  - the default call's 21,204 gates on failed couplers;
  - what CALSPLIT did not cover.
- **README, "Results in brief".** The stale-calibration line adds CALSPLIT's result. It used to cite only
  Addendum 335.
- **RELEASES.** The current release's block has a new bullet, "Found after the release (CALSPLIT, Addenda
  402-403, pre-registered)". It follows the REC-PROBE bullet of Addendum 398.

## 2. What did not change

- The release (`psf_compile.py` 2026-10-07.1), the AI front end, the layout search and the core.
- The README's table "What it does well, and what it costs". Its numbers come from tests with the true calibration,
  and they stand.
- The README line "No failed coupler or qubit was used in any of these tests". It is about the 1,506-circuit device
  tests, which used only target-aware calls. CALSPLIT's default-call arm is reported separately.

## 3. What follows

Addendum 403 also proposed a candidate: the recommended call switches to an alternative only when the estimated
gain exceeds a margin. It is being prepared as candidate 2026-10-07.c24 (changelog item 51). It will get its own
pre-registered test, with both true and stale calibrations, in a later Addendum. The README will say nothing about
it until that test is scored.


---

<!-- ===== Addendum 405 (source: spare-qubit-cliff-addendum-405-2026-10-08.md) ===== -->

> **Note added when merging:** Pre-registration of MARGIN, committed with candidate 2026-10-07.c24, its test output, the scripts and the dry run as the lock, before the scored run.

## Addendum 405 -- Pre-registration: MARGIN. Candidate 2026-10-07.c24 (changelog item 51) makes the recommended call switch to an alternative only when the estimate is lower by more than 5%. Does that keep the call's gain with a fresh calibration and lose less with a stale one? True and stale calibrations, new data (seed 5) (2026-10-08)

**Status: pre-registration.** It was written after the candidate's tests and MARGIN's dry run (development seed 2),
and before any circuit of the scored data was compiled. The lock is the commit that adds this Addendum together with
the candidate, the four scripts, the candidate's test output and the dry run. The scored run follows the lock.
Owner's go-ahead: 2026-10-07 ("今日やってしまいましょう": do the README change and the candidate with its test
today), continued on 2026-10-08 at the workplace.

**Where and how it is locked (a departure, disclosed).** The tests, the dry run and the scored run are made on the
workplace machine (Windows; Python 3.11.9, Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.49.0, NumPy 2.4.6,
scikit-learn 1.8.0; 14 cores). That machine does not push to GitHub, so the lock cannot be made public before the
scored run, as it was for every earlier test. Instead:

1. the lock commit is made locally;
2. its hash is e-mailed by the owner to himself before the scored run starts (a third-party timestamp);
3. the scored run's `env.txt` records the commit it ran on;
4. the lock commit and the results are pushed later from home, the lock commit unchanged.

The home machine (Python 3.12.13, NumPy 2.5.3) was used for a first test run and dry run on 2026-10-07; those are
reported below for comparison, and are not the committed ones.

## 1. Why

CALSPLIT (Addenda 402-403) compiled DEPTH-R's classifiers with stale calibrations and scored them with the device's
true noise. On FakeAuckland, the recommended call kept less margin than the guarded call (`target` and
`placement_refine` only):

- -0.0041 on average over three draws, and -0.0142 in one of them;
- with the true calibration (DEPTH-R, Addendum 400) it had kept +0.0056 more.

The recommended call's extra steps are three choices made by a noise estimate:

- item 35's re-synthesis;
- item 36's comparison with Qiskit level 3;
- item 37's floor candidate and level 3.

Each takes an alternative whenever the alternative's estimate is lower by more than item 46's tie band (1e-12,
relative). Addendum 403 (section 4) proposed a margin wider than that tie band. MARGIN tests it, with fresh data,
under both true and stale calibrations.

## 2. The candidate: 2026-10-07.c24 (item 51)

[`patches/psf_compile_c24_2026-10-07/psf_compile.py`](../../patches/psf_compile_c24_2026-10-07/psf_compile.py) is
release 2026-10-07.1 with the following changes:

- **`SWITCH_MARGIN = 0.05` and `_better(b, a)`.** `_better` is true if `b < a - max(1e-12, SWITCH_MARGIN) *
  max(|a|, |b|)`.
- **Item 35 and item 36 switch only if `_better`.** That is item 35's re-synthesis under `"select"`
  (`_select_resynthesis`) and item 36's comparison with level 3 (`_compare_level3`). Otherwise they keep the
  release's circuit, without item 39's check.
- **Item 37's choice (`_choose_lazy`) first leaves out every candidate that is not `_better` than the release's
  circuit.** Among the rest it takes the lowest, as before.
- **A candidate the margin turns away is counted in `margin_kept`.** These are candidates that the tie band alone
  would have let through.
- **Nothing else changes.** That covers the estimates, the placement, the default call and the legacy `_choose`.

The value 0.05 was set before any test of it, without a sweep. It is round, and it is not derived from anything.
With `SWITCH_MARGIN` at the tie band, the choices are the release's. There is one exception, in item 37, where three
estimates lie within about 2e-12 of each other; the tie band is not transitive.

**Its tests.** [`test_c24.py`](../../patches/psf_compile_c24_2026-10-07/test_c24.py) passed 11 of 11 on both
machines (home 52 s; workplace 113 s, the output committed in
[`data/2026-10-07/c24_tests/`](../../data/2026-10-07/c24_tests/), re-run by the lock bundle). The tests check these
things:

- The version is c24.
- `_better` equals `_lower` when the margin is 0 or the tie band (22,000 pairs).
- The default call returns the release's circuit on 4 inputs and 3 devices.
- With the margin set to the tie band, the recommended call returns the release's circuit on 8 inputs, 3 devices and
  both a true and a stale Target.
- With the margin at 0.05:
  - every output implements its input and uses no failed element of the true device;
  - every output that differs from the release's comes from a call in which the margin turned an alternative away.

The outputs differed from the release's on 3, 1 and 5 of 16 at home and 2, 1 and 4 of 16 at the workplace
(FakeAuckland, FakeTorino, FakeHanoiV2); compile time was 0.92-0.96 and 0.87-0.94 of the release's. That the counts
differ between the machines means some outputs of the release or of c24 differ between the two environments
(Python 3.12 / NumPy 2.5.3 against 3.11 / 2.4.6); this was not examined. Each machine's test compares the two files
in the same environment, so the checks hold on each. Before that, the choice logic alone was checked in a sandbox, against the release's
on 200,000 random sets of estimates. With the margin at 0 the two differed only in the non-transitive case above
(64 sets, built with estimates crowded within 3e-12). With the margin at 0.05, c24 never switched where the release
did not.

## 3. Design

The test is run by [`benchmarks/margin_eval.py`](../../benchmarks/margin_eval.py) through
[`benchmarks/run_margin.sh`](../../benchmarks/run_margin.sh) or, on a machine without a POSIX shell,
[`benchmarks/run_margin.py`](../../benchmarks/run_margin.py) (the same jobs and the same score; it also records
qiskit-ibm-runtime's version, which sets the fake devices' calibrations). The independent
[`benchmarks/margin_verify.py`](../../benchmarks/margin_verify.py) reads only the raw JSON and does not import the
harness.

- **Unchanged.** These come from DEPTH-R, through [`depth_r_eval.py`](../../benchmarks/depth_r_eval.py) and DEPTH's
  [`depth_eval.py`](../../data/2026-10-05/workplace/depth1/depth_eval.py), and from CALSPLIT:
  - the model, the training (300 steps), the circuits;
  - BC and D38, n in {4, 6}, L in {1, 2, 4, 8, 12, 16};
  - FakeAuckland and FakeTorino;
  - the noisy simulation with the device's **true** noise model, and readout;
  - P0's checks.
- **New data.** The split seed and the init seed are both 5:
  - seed 0 was the pilot, 1 DEPTH, 2 development and dry runs, 3 READOUT, and 4 DEPTH-R and CALSPLIT;
  - the classifiers are trained in this run (4 jobs);
  - seed 4 is not used because item 51 was proposed after seeing it.
- **Calibrations the compilers read.** "t" is the true Target. The three stale draws come from STALE's
  `stale_target()` ([`stale_eval.py`](../../benchmarks/stale_eval.py), unchanged), with `BASE` = 97,000,000 +
  1,000,000 k, k = 0, 1, 2. STALE used 80,000,000; CALSPLIT used 91,000,000-93,000,000; c24's tests used
  96,000,000.
- **Arms.** Each arm is run with every calibration:
  - **REC**: release 2026-10-07.1, recommended call;
  - **C24**: candidate c24, recommended call;
  - **RPSF**: the release with `target` and `placement_refine`.
  
  That is 96 deployment jobs. Every row records a signature of the compiled circuit, and every file records C24's
  choice counters.
- **Pooling, as in CALSPLIT.** For each device, take the mean over the 24 cells (dataset, n, L) of the cell's mean
  margin y x z, under true noise. "Stale" is the mean over the three draws. Every comparison has a tolerance of
  1e-12.

## 4. Predictions (scored only by `margin_eval.py score`)

**P0:**

- 96 deployment files and 4 training files;
- every state infidelity <= 1e-6;
- reduced simulation = whole-device simulation within 1e-9;
- REC and RPSF name release 2026-10-07.1, and C24 names 2026-10-07.c24, each with its locked file;
- `SWITCH_MARGIN` is 0.05 in C24 only;
- every stale Target differs from the true one;
- the seeds are (5, 5).

If P0 fails, nothing is scored.

| ID | Prediction | CONFIRMED | REFUTED |
|---|---|---|---|
| M1 | with stale calibrations, C24 keeps at least the guarded call's margin | C24 - RPSF (stale) >= 0 on both devices | < -0.002 on either |
| M2 | with stale calibrations, the margin costs nothing | C24 - REC (stale) >= -0.001 on both devices | < -0.005 on either |
| M3 | with the true calibration, it costs little | REC - C24 (true) <= +0.002 on both devices | > +0.005 on either |
| M4 | FakeTorino's structural lead (Addendum 403, reading 3) is kept | C24 - RPSF >= +0.01 with the true and with stale calibrations | < +0.005 in either |
| M5 | no bad draw | FakeAuckland's worst stale draw, C24 - RPSF >= -0.005 | < -0.01 |

**Reading rule.** Suppose C24 returns REC's circuit on more than 95% of a device's stale rows. Then M1, M2 and M5 on
that device show only that the margin rarely bit; they do not show that it works.

**Decision rule for item 51.** Acceptance is proposed to the owner only if all of these hold:

- P0 passes;
- M1 is CONFIRMED;
- M2 and M3 are not REFUTED;
- the margin bit on at least one device.

Otherwise item 51 is not proposed. The score prints this as `ITEM51`. Acceptance would not mean release: under
Addendum 385, improvements are batched.

**Reported without prediction:**

- pooled margins, per draw too;
- accuracy at 15, 63, 255 and 1,023 shots (exact binomial);
- flips;
- gates on the true device's failed couplers;
- median compile times;
- the share of rows where C24's circuit differs from REC's;
- C24's counters;
- REC - RPSF with true and with stale calibrations. That is CALSPLIT's K2 question again, on new data.

## 5. What is known (disclosed)

**The dry runs.** Development seed 2, 12 points, L in {1, 4, 12}. The committed one is the workplace's (100 jobs in
519 s with `run_margin.py`, 12 at a time; output in
[`data/2026-10-07/margin_dry/`](../../data/2026-10-07/margin_dry/)): the harness ran end to end, P0 passed (largest
infidelity 5.1e-15), and the independent check agreed. The home machine's dry run (317 s, `run_margin.sh`) gave the
same verdicts with numbers that differ in the third or fourth decimal (training and compilation in the two
environments). Their numbers are not results:

| | FakeAuckland, workplace (home) | FakeTorino, workplace (home) |
|---|---|---|
| C24 - RPSF, stale (worst draw) | +0.0013 (-0.0001); home +0.0013 (-0.0005) | +0.0188 (+0.0178); home +0.0180 (+0.0162) |
| C24 - REC, stale | -0.0006; home -0.0001 | -0.0005; home -0.0007 |
| REC - C24, true | **+0.0073**; home **+0.0064** | +0.0001; home +0.0002 |
| C24 - RPSF, true | +0.0002; home +0.0006 | +0.0147; home +0.0148 |
| REC - RPSF, true / stale | +0.0075 / +0.0019; home +0.0069 / +0.0014 | +0.0148 / +0.0193; home +0.0149 / +0.0187 |
| rows where C24 differs from REC, true / stale | 0.32 / 0.32; home 0.31 / 0.30 | 0.03 / 0.05; home 0.06 / 0.10 |

In both dry runs, M1, M2, M4 and M5 read CONFIRMED. **M3 read REFUTED, so ITEM51 read "do not propose".** On
FakeAuckland, with the true calibration, most of REC's gain over the guarded call came from switches whose estimated
gain was below 5%, and the margin removed that gain. With stale calibrations, REC did not lose to the guarded call in
the dry run (+0.0014), unlike CALSPLIT (-0.0041).

**Nothing was changed after the dry run.** That covers `SWITCH_MARGIN`, the predictions, the thresholds and the
decision rule. Choosing a width from 12 points per cell would be tuning on development data too thin to tune on. The
scored run is meant to answer two questions on new data:

- whether the small-gain switches are worth keeping with a fresh calibration (M3);
- whether CALSPLIT's loss under stale calibrations reproduces at all (REC - RPSF, stale).

**Expectations:**

- M3 may well fail on FakeAuckland, and with it item 51.
- M2 and M4 are expected.
- M1 and M5 are open.

A failed M3 with a CONFIRMED M1 would say the following: on FakeAuckland, the recommended call's small switches are
worth about +0.007 with a fresh calibration, and they are what goes wrong with a stale one. That would point to a decision
for the user (calibration age), not for the compiler.

## 6. What this will not establish

- real calibration drift (this is a perturbation model);
- hardware;
- readout errors in a stale Target;
- other tasks than DEPTH-R's classifiers;
- other margin widths: one value is tested, chosen in advance.

## 7. Files locked (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`patches/psf_compile_c24_2026-10-07/psf_compile.py`](../../patches/psf_compile_c24_2026-10-07/psf_compile.py) (candidate c24) | `9174ce54ba923d5628abdeb8c050bf84358e60bf3ad3e3b865f6bff56b21710e` |
| [`patches/psf_compile_c24_2026-10-07/test_c24.py`](../../patches/psf_compile_c24_2026-10-07/test_c24.py) | `c4473dc78f9f8ac830df50585d8396b3239238463c5ed5f3d4f7129e371cc607` |
| [`benchmarks/margin_eval.py`](../../benchmarks/margin_eval.py) | `c155216d3e0f2811eda3f84bfd4fa9d54d44e9e7f7df17df117646a48a301416` |
| [`benchmarks/run_margin.sh`](../../benchmarks/run_margin.sh) | `c03ec97f48c09d2a587b9ed68e5c511ba2f80954331318d3f499aba96397c861` |
| [`benchmarks/run_margin.py`](../../benchmarks/run_margin.py) | `b8840072c88fa53a5ae1a528dc1402893c811d77245a964c06b2271fe70f84d2` |
| [`benchmarks/margin_verify.py`](../../benchmarks/margin_verify.py) | `5f0f65d8bf680f478a4542871f294e650be1566a7c3d0aa489fc08ba445c88a5` |
| [`benchmarks/stale_eval.py`](../../benchmarks/stale_eval.py) (STALE, unchanged) | `f2dd16caf17d25232f2452b08a5654f83a8ae10459c773dfa3acfc74005a1cc0` |
| [`benchmarks/depth_r_eval.py`](../../benchmarks/depth_r_eval.py) (DEPTH-R, unchanged) | `6f636b677f1a481ef3492bf0de467893cde70359c717bc901b6cb5cd2838f3b6` |
| [`psf_compile.py`](../../psf_compile.py) (release 2026-10-07.1) | `73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc` |

**Scored run** (workplace, after the lock commit's hash is e-mailed): `python benchmarks/run_margin.py
data/2026-10-07/margin --par 12`, then `python benchmarks/margin_verify.py data/2026-10-07/margin`. The results go in
the next Addendum.


---

<!-- ===== Addendum 406 (source: spare-qubit-cliff-addendum-406-2026-10-08.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 405. Merged on 2026-10-08 together with Addenda 407 and 408: the first apply of this text (bundle ck) stopped at its checks, and the data was committed without it (ba4f2d0). The text and the data are unchanged; `verify.txt` was produced at this merge.

## Addendum 406 -- Results of MARGIN (Addendum 405): P0 passed; M1, M4, M5 CONFIRMED, M2 and M3 AMBIGUOUS. The decision rule prints "propose", but candidate c24 was never better than the release: the margin gave up part of the recommended call's gain with a fresh calibration and gained nothing with a stale one. The problem it was meant to fix, CALSPLIT's loss on FakeAuckland, did not reproduce on new data. Item 51 is not recommended (2026-10-08)

**Status: results of the pre-registered test in Addendum 405, scored by the locked harness and re-checked by the
independent [`benchmarks/margin_verify.py`](../../benchmarks/margin_verify.py).**

## 1. The run

- **Lock.** The local commit of Addendum 405, made at the workplace. Its hash was e-mailed before the run, as
  Addendum 405 set out. The commit is pushed together with this Addendum.
- **Machine.** The workplace PC: Windows, Python 3.11.9, Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.49.0,
  NumPy 2.4.6, scikit-learn 1.8.0, 14 cores. `run_margin.py` ran 12 jobs at a time: 4 training jobs and 96
  deployment jobs, 6,857 s in all.
- **Output.** [`data/2026-10-07/margin/`](../../data/2026-10-07/margin/) holds `score.md`, every job's JSON and log,
  `env.txt`, `progress.txt` and `verify.txt`. The independent check gave the same P0 and the same verdicts.

## 2. Results

**P0: PASS.**

- 96 of 96 deployment files and 4 of 4 training files.
- Largest state infidelity 2.1e-9.
- The reduced simulation equals the whole-device simulation.
- Versions and files are as locked, and `SWITCH_MARGIN` is 0.05 in C24 only.
- Every stale Target differs from the true one, and the seeds are (5, 5).

Pooled margin, mean y x z over the 24 cells; "stale" is the mean of three draws:

| | FakeAuckland | FakeTorino |
|---|---|---|
| REC, true / stale | 0.4275 / 0.4091 | 0.5132 / 0.5056 |
| C24, true / stale | 0.4241 / 0.4089 | 0.5132 / 0.5031 |
| RPSF, true / stale | 0.4217 / 0.4080 | 0.4957 / 0.4810 |

| ID | prediction | FakeAuckland | FakeTorino | verdict |
|---|---|---|---|---|
| M1 | C24 - RPSF (stale) >= 0 | +0.0009 | +0.0221 | **CONFIRMED** |
| M2 | C24 - REC (stale) >= -0.001 (REFUTED < -0.005) | -0.0002 | **-0.0024** | **AMBIGUOUS** |
| M3 | REC - C24 (true) <= +0.002 (REFUTED > +0.005) | **+0.0034** | +0.0000 | **AMBIGUOUS** |
| M4 | FakeTorino C24 - RPSF >= +0.01, true and stale | | +0.0175 / +0.0221 | **CONFIRMED** |
| M5 | FakeAuckland worst stale draw C24 - RPSF >= -0.005 | -0.0017 | | **CONFIRMED** |

**The margin bit often.** C24's circuit differed from REC's on these shares of rows:

| | true calibration | stale calibrations |
|---|---|---|
| FakeAuckland | 40% | 31% |
| FakeTorino | 1.3% | 17% |

It turned away 3,191 and 1,449 candidates of item 37, and 1,401 and 1,160 re-syntheses. The reading rule of Addendum
405 therefore does not apply on either device. The decision rule's line reads **ITEM51: PROPOSE ACCEPTANCE**.

**Reported without prediction.** Accuracy at 15 shots:

| | REC | C24 | RPSF |
|---|---|---|---|
| FakeAuckland, true | 0.8845 | 0.8828 | 0.8825 |
| FakeAuckland, stale | 0.8742 | 0.8744 | 0.8731 |
| FakeTorino, true | 0.9194 | 0.9194 | 0.9128 |
| FakeTorino, stale | 0.9188 | 0.9172 | 0.9099 |

At 1,023 shots the three arms are within 0.0004 of each other in every condition. No arm placed a gate on a failed
coupler. Median compile times were similar for REC and C24 (0.74-1.14 s), and RPSF took 0.12-0.13 s.

## 3. Reading

1. **C24 was never better than the release.** Here is C24 - REC in each of the four conditions:

   | calibration | FakeAuckland | FakeTorino |
   |---|---|---|
   | true | -0.0034 | 0.0000 |
   | stale | -0.0002 | -0.0024 |

   With a fresh calibration on FakeAuckland, the margin gave up 0.0034 of the release's 0.0058 lead over the guarded
   call; it threw away well-founded small switches. With stale calibrations it saved nothing on FakeAuckland, and it
   cost 0.0024 on FakeTorino.
2. **The decision rule was too weak, and that is this project's error.** The rule required M1 (C24 at least the
   guarded call's level under stale calibrations), and M2 and M3 not REFUTED. M1 is satisfied whenever the release
   itself does not lose to the guarded call. Nothing in the rule asked for C24 to beat REC anywhere. Read by its
   letter, the rule proposes a change that the data show does not help. **Item 51 is therefore not recommended for
   acceptance.** The decision is the owner's. Candidate c24 stays in [`patches/`](../../patches/) as the record.
3. **CALSPLIT's FakeAuckland loss did not reproduce on new data.** On seed 5 the release kept more margin than the
   guarded call even with stale calibrations: +0.0011 on FakeAuckland and +0.0246 on FakeTorino. CALSPLIT, on seed 4,
   had -0.0041 on FakeAuckland, with -0.0142 in one draw. With the true calibration the lead on FakeAuckland is the
   same in both tests: +0.0056 in DEPTH-R and +0.0058 here. The estimate-driven choices hold up with a fresh
   calibration. With a stale one they range from a small loss to a small gain on FakeAuckland, depending on the data
   and the draw. The README's statement that they "need a fresh calibration" (Addendum 404) is stronger than these
   two tests support. It will be corrected with the next README change, together with BP-FINAL's.
4. **A width chosen in advance can be the wrong width.** 5% was round, chosen without a sweep, and Addendum 405
   disclosed that. These data say only that 5% is too wide on FakeAuckland with a fresh calibration. They do not
   suggest a better width, and none is proposed.

## 4. What this does not establish

- real calibration drift, since this is a perturbation model;
- hardware;
- other tasks;
- other widths of the margin.


---

<!-- ===== Addendum 407 (source: spare-qubit-cliff-addendum-407-2026-10-08.md) ===== -->

> **Note added when merging:** Pre-registration of BP-FINAL. This text was written before the scored run (bundle cl, 2026-10-08, about 04:12 CEST), but its apply stopped at its checks because Part 10 did not yet contain Addendum 406. So the lock commit `79b73ed`, whose hash was e-mailed before the scored run, carries the two scripts and the smoke run but not this text. The population rule and the six predictions, with their thresholds, are in the locked `benchmarks/bp_final.py` (its `population` and `score`); this text states them in words and was not changed. The smoke run's `verify.txt` was produced at this merge.

## Addendum 407 -- Pre-registration: BP-FINAL. Release 2026-10-07.1 against Qiskit level 2, as Benchpress calls it, on every published Benchpress transpilation test this project has not yet used (880 tests). The last unseen sample: run once (2026-10-08)

**Status: pre-registration.** It was written after BP-FINAL's smoke run, which compiled only tests already used, and
before any of the 880 tests was compiled. The lock is the commit that adds this Addendum together with the two
scripts and the smoke run. The scored run follows the lock. Owner's decisions, 2026-10-08: all 880 tests, this
project's own harness (not a Benchpress gym), run on the workplace machine ("全件で自前GYMでここで回す").

**Locked at the workplace, as MARGIN was (Addendum 405).** The workplace machine does not push. The lock commit is
made locally. Its hash is e-mailed by the owner to himself before the scored run. The scored run records the commit
it ran on. The lock commit and the results are pushed later from home, the lock commit unchanged.

## 1. Why

The README says the default call is "about level with Qiskit level 2" on general circuits. That rests on Addendum
396: 1.027 times level 2's two-qubit count on 139 Benchpress tests. Those 139 are the tests of BP-MOCK and BP-MOCK2,
which items 48 and 50 were written from or tested on. So the number is not from an independent sample, and Addendum
396 says so. The workplace review of 2026-10-08 named this as the gap to close before the number is used outside the
project. BP-FINAL closes it on every test that is left.

After this run no unseen Benchpress transpilation test remains. A later change to the default call cannot be tested
on unseen Benchpress inputs again, and this run should not be repeated on a new candidate.

## 2. Design

The test is run by [`benchmarks/bp_final.py`](../../benchmarks/bp_final.py). The independent
[`benchmarks/bp_final_verify.py`](../../benchmarks/bp_final_verify.py) reads only the raw JSON.

- **Benchpress.** Commit `b695f30`, as in every earlier Benchpress test. Published reference: BP-PROBE's
  `published_ref.json` (1,032 test ids), unchanged.
- **Population (no sampling).** BP-MOCK's strata (`bp_mock.strata`, imported unchanged) contribute their published
  test ids, minus:
  - the 12 BP-PROBE ran;
  - the 92 BP-MOCK ran;
  - the 48 BP-MOCK2 ran.

  That leaves 880 tests:

  | family | tests |
  |---|---|
  | QASMBench, small, medium and large on four topologies | 407 |
  | HamLib on four topologies | 367 |
  | HamLib on FakeTorino | 67 |
  | Feynman on FakeTorino | 39 |

  The 100-qubit stratum is empty (all 9 used).
- **Building, backends and metrics.** As BP-MOCK and BP-MOCK2 (`bp_mock.py`, `bp_mock2.py`, unchanged):
  - Benchpress's own builders, backends and validator;
  - the backend's two-qubit gate count and depth;
  - the equivalence check of BP-MOCK2: item 39's `_implements` against the input expanded through its definitions,
    for inputs of at most 10 qubits.
- **Arms.** Each test and arm runs in its own process, with a 1,500 s limit for every arm:
  - **QK**: `generate_preset_pass_manager(2, backend).run(circuit)`, Benchpress's call, not seeded;
  - **REL**: release 2026-10-07.1, default call;
  - **REL2**: REL again (reproducibility);
  - **RECR**: the recommended call with the backend's target, on the 106 FakeTorino tests only.

  That makes 2,746 jobs. They run 12 at a time on the workplace machine (Windows, Python 3.11.9, Qiskit 2.5.2,
  14 cores). The jobs record times, so times are comparable only within the run.
- **Ratios.** The ratio is (two-qubit gates + 1) / (QK's + 1), so that a test with none counts. Means are geometric,
  over the tests that QK and REL both finished. The 95% interval resamples tests within each stratum (10,000
  resamples, seed 20261008).

## 3. Predictions (scored only by `bp_final.py score`)

**P0:**

- 880 tests, with every arm present;
- every PSF-Zero output names release 2026-10-07.1;
- the release file, the published reference and Benchpress as locked;
- no uncommitted change to a tracked file.

If P0 fails, nothing is scored.

| ID | Prediction | CONFIRMED | REFUTED |
|---|---|---|---|
| F1 | all tests: geometric mean REL/QK | <= 1.06 | > 1.10 |
| F2 | no family far behind (QASMBench, HamLib on topologies, HamLib on FakeTorino, Feynman) | every family <= 1.15 | any > 1.25 |
| F3 | few large losses: share of tests with REL/QK > 1.10 | <= 0.15 | > 0.25 |
| F4 | correctness: every REL, REL2 and RECR output passes Benchpress's validator, and every checkable one implements its input | none fails, at least 20 checked | any fails |
| F5 | the recommended call places no two-qubit gate on FakeTorino's failed couplers or qubits | 0 tests | any |
| F6 | robustness: REL fails (error or 1,500 s) where QK finishes | <= 1% | > 3% |

**Where the lines come from.**

- F1: the seen tests gave 1.027 (139 tests, Addendum 396), and the 48 new tests of BP-MOCK2 gave 1.039 before item
  50. Unseen tests are expected to be a little worse than the ones the items were written from.
- F2: per stratum, the seen values were 0.99-1.08.
- F3: the seen tests had 15 of 139 (11%) above 1.10.

**Reported without prediction:**

- two-qubit depth;
- the equally weighted mean over families;
- how many tests have fewer, as many and more two-qubit gates than QK;
- compile time REL/QK (median and geometric mean);
- how often REL2 differs from REL;
- on FakeTorino, RECR/QK and the gates each arm places on failed elements;
- a table per stratum.

## 4. What is known (disclosed)

- **The smoke run** used the first BP-MOCK2 test of each stratum: 18 tests already used, none of the 880. It started
  2026-10-08T02:07:59Z and ran 56 jobs in 163 s, 12 at a time, on commit `ba4f2d0`. Its output is in
  [`data/2026-10-08/bp_final_smoke/`](../../data/2026-10-08/bp_final_smoke/). Every arm finished, and every output
  passed Benchpress's validator. The independent check agreed with the score. Its numbers are not results:
  - REL/QK was 1.016;
  - F4 read AMBIGUOUS only because 11 outputs could be checked, fewer than 20;
  - REL2 differed from REL on 2 of 18 tests, with the same two-qubit counts;
  - the median compile time REL/QK was 4.85.
- **Two things the smoke run showed, which the predictions already allow for:**
  - `qec_sm_n5` returned `implements: False` for REL and REL2, but was marked not checkable. Its input measures and
    resets mid-circuit, so item 39's check does not apply, and F4 counts only checkable outputs, as BP-MOCK2 did.
  - On the FakeTorino HamLib test, the default call (REL) placed 28 two-qubit gates on failed couplers; QK and the
    recommended call placed none. The default call does not read the target (Addenda 393, 403), and F5 is about the
    recommended call only.
- **The scoring was rehearsed on synthetic data** of the full size. The independent check agreed with the score.
- **Nothing about the 880 tests was looked at before the lock** beyond their ids and strata, which the population
  rule lists.
- **The README will report F1's value and interval**, whatever the verdict, in the same paragraph as the compile-time
  cost.

## 5. What this will not establish

- Benchpress's other test groups (construction, manipulation);
- Benchpress's own gym and pytest harness (not used here; a PSF-Zero gym is a separate task);
- depth as a target;
- hardware;
- other versions of Qiskit.

## 6. Files locked (normalized SHA-256)

| file | normalized SHA-256 |
|---|---|
| [`benchmarks/bp_final.py`](../../benchmarks/bp_final.py) | `bbcbc6ebf313d838259f9cb0c507fd0f3aa5fb1937f65dd78bf4176e31e5e7d0` |
| [`benchmarks/bp_final_verify.py`](../../benchmarks/bp_final_verify.py) | `5bfe840c83306afdaccb2b6c4634c0634078c893b481a062c1331b6b2804f943` |
| [`benchmarks/bp_mock.py`](../../benchmarks/bp_mock.py) (BP-MOCK, unchanged) | `7757e848c9848c9644a845ce97198966b2fb2f98fa9584c2d0050d45b573b4d0` |
| [`benchmarks/bp_mock2.py`](../../benchmarks/bp_mock2.py) (BP-MOCK2, unchanged) | `e0c182acb99768a88bedb07c6f56aae84fc234b62a17bd2a046452de3bccf93c` |
| [`benchmarks/bp_probe.py`](../../benchmarks/bp_probe.py) (BP-PROBE, unchanged) | `73defde66868852db583d5e4fb055ec53c6ea864aae6466876c571f8d26cee66` |
| [`data/2026-10-06/bp_probe/published_ref.json`](../../data/2026-10-06/bp_probe/published_ref.json) | `d05231a75695c4fa9cbe2f09891ae8a835db32699e381c4d8de4b2afaaf56877` |
| [`psf_compile.py`](../../psf_compile.py) (release 2026-10-07.1) | `73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc` |

**Scored run** (workplace, after the lock commit's hash is e-mailed): `python benchmarks/bp_final.py run --bp <clone>
--out data/2026-10-08/bp_final --par 12`, then `python benchmarks/bp_final.py score --out data/2026-10-08/bp_final`
and `python benchmarks/bp_final_verify.py data/2026-10-08/bp_final`. The results go in the next Addendum.


---

<!-- ===== Addendum 408 (source: spare-qubit-cliff-addendum-408-2026-10-08.md) ===== -->

> **Note added when merging:** Results of the pre-registered test in Addendum 407, with the README and RELEASES changes they and Addendum 406 call for. Merged with Addenda 406 and 407: the first apply (bundle cm) stopped at its checks, and the data was committed without the text (c727f12). `verify.txt` was produced at this merge.

## Addendum 408 -- Results of BP-FINAL (Addendum 407): all six predictions CONFIRMED. On 877 Benchpress transpilation tests never used before, release 2026-10-07.1's default call uses 1.046 times Qiskit level 2's two-qubit gates (95% 1.037-1.055), at a median 3.5 times its compile time; every output valid, every checkable one exact. README and RELEASES updated, with MARGIN's correction (2026-10-08)

**Status: results of the pre-registered test in Addendum 407. The locked harness scored them, and the independent
[`benchmarks/bp_final_verify.py`](../../benchmarks/bp_final_verify.py) re-checked them. The documentation changes
listed in section 4 come with them.**

## 1. The run

- **Lock.** Commit `79b73ed`, the local commit of Addendum 407 at the workplace. Its hash was e-mailed before the
  run, and it is pushed unchanged with this Addendum. The run recorded it as its head, with no uncommitted change.
- **Machine.**
  - The workplace PC: Windows, Python 3.11.9, Qiskit 2.5.2, qiskit-ibm-runtime 0.49.0, NumPy 2.4.6, 14 cores.
  - Windows power mode "best power efficiency", with sleep off. The screen turned itself off; the machine did not
    sleep.
  - 2,746 jobs, 12 at a time, took 13,355 s.
  - Progress lines were shown on screen only. Every job's record is in the JSON.
- **Output.** [`data/2026-10-08/bp_final/`](../../data/2026-10-08/bp_final/) holds:
  - `bp_final.json`;
  - `bp_final_first.json`, from before the re-run below;
  - `score.md` and `score_first.md`;
  - `verify.txt`.
- **Re-run of timed-out jobs.** At 12:19 JST, before any result was seen, a rule was fixed in case the machine slept
  during the run. Every job that timed out is re-run once, with the locked script ("one" mode), the same limit and the
  same parallelism, by [`benchmarks/bp_final_rerun.py`](../../benchmarks/bp_final_rerun.py). The re-run's record
  replaces the first one and keeps it under "first". The machine had not slept, but the rule was applied as fixed.
  - Nine jobs had timed out on four tests:
    - `bwt_n37` on square: REL and REL2;
    - `bwt_n37` on heavy-hex: QK, REL and REL2;
    - `square_root_n60` on linear: QK, REL and REL2;
    - Feynman `hwb11`: RECR.
  - All nine timed out again.
  - The score and the verdicts are identical before and after.

## 2. Results

**P0: PASS.** Every condition was met:

- 880 of 880 tests;
- every arm present;
- every PSF-Zero output names release 2026-10-07.1;
- the release file, the published reference and Benchpress (`b695f30`) are as locked;
- no uncommitted change.

QK and REL both finished on 877 tests.

| ID | prediction | value | verdict |
|---|---|---|---|
| F1 | geometric mean REL/QK <= 1.06 (REFUTED > 1.10) | **1.046** (95% 1.037-1.055) | **CONFIRMED** |
| F2 | every family <= 1.15 (REFUTED if any > 1.25) | QASMBench 1.020 (404), HamLib on four maps 1.069 (367), HamLib on FakeTorino 1.090 (67), Feynman 1.022 (39) | **CONFIRMED** |
| F3 | share of tests with REL/QK > 1.10 <= 0.15 (REFUTED > 0.25) | 0.149 (131 of 877) | **CONFIRMED** |
| F4 | every output valid; every checkable one implements its input | 0 invalid of 1,859; 0 wrong of 491 checked | **CONFIRMED** |
| F5 | the recommended call places no gate on FakeTorino's failed elements | 0 of 105 tests | **CONFIRMED** |
| F6 | REL fails where QK finishes <= 1% | 1 of 878 (0.1%) | **CONFIRMED** |

**Reported without prediction:**

- **Two-qubit gates.** REL used fewer than QK on 117 tests, as many on 314 and more on 446.
- **Two-qubit depth.** REL/QK is 1.040.
- **Families weighted equally.** The geometric mean is 1.050.
- **Per stratum.**
  - QASMBench: 0.987-1.056. The best is large circuits on linear maps, the worst large circuits on square maps.
  - HamLib: 1.056-1.090.
- **Compile time.** REL/QK has a median of 3.52 and a geometric mean of 2.99. Per stratum the median is 2.3-6.5;
  small circuits cost the most, relative to Qiskit.
- **Reproducibility.** REL2 differed from REL on 83 of 877 tests, never in the two-qubit count.
- **FakeTorino.** RECR/QK is 1.035 (105 tests). Tests with gates on failed couplers or qubits:
  - QK: 25;
  - REL (the default call, which does not read the target): 62, including 40,801 gates on `hwb11`;
  - RECR: 0.

## 3. Reading

1. **The README's "about level with Qiskit level 2" holds on an independent sample, now with an interval.**
   - On unseen tests the default call is 4.6% above Qiskit level 2 (3.7-5.5%).
   - That is a little worse than on the 139 development tests (1.027, Addendum 396). Items 48 and 50 were written
     from those tests.
   - It is not better than Qiskit on general circuits, and the README does not say it is.
2. **F3 was confirmed at the edge (0.149 against 0.15).** About one test in seven costs more than 10% extra two-qubit
   gates. A small change in the population could have made F3 AMBIGUOUS.
3. **Hamiltonian simulation is the weakest family.** HamLib is at 1.06-1.09 in every stratum. QASMBench (1.02) and
   Feynman (1.02) are close to level. Any further work on the default call should start there. No unseen Benchpress
   test remains to test such work.
4. **Correctness held on every test.** No output was invalid, and none of the 491 checkable outputs fails to
   implement its input.
5. **The default call's blindness to failed elements shows on 62 of 105 FakeTorino tests.** The recommended call
   avoided them every time, at 1.035 times Qiskit level 2's count. This supports the README's advice (Addendum 404)
   to pass `target` on such devices.
6. **The price is compile time.** A median of 3.5 times Qiskit level 2's.

## 4. Documentation changed with this Addendum

[`README.md`](../../README.md):

- **The introduction's general-circuits bullet.** It now gives BP-FINAL's 1.046 (interval, counts, compile time)
  instead of the development tests' 1.03.
- **"Where it is weaker", Benchpress.**
  - BP-FINAL's design and table. The development tests' 1.03 and 2026-10-06.4's 1.33 are kept as context.
  - HamLib as the weakest family.
  - Compile time from BP-FINAL.
  - The reproducibility count.
  - The failed-element counts on FakeTorino.
  - What is not covered.
- **The calibration paragraph (MARGIN's correction of Addendum 404).**
  - "Need a fresh calibration" became: they pay off with a fresh calibration, and with a stale one it depends on the
    data. The figures are -0.0041 in CALSPLIT and +0.0011 in MARGIN.
  - Item 51 is not recommended (Addendum 406).
  - The introduction and "Results in brief" changed to match.
- **Known limits.** Benchpress is now covered by BP-FINAL, and no unseen test is left.

[`docs/RELEASES.md`](../../docs/RELEASES.md):

- **A five-line "In brief (as of 2026-10-08)" at the top.** The workplace review of 2026-10-08 asked for it: one day
  produced over twenty Addenda.
- **Two bullets under the current release:** BP-FINAL and MARGIN.

## 5. What this does not establish

- Benchpress's other test groups;
- Benchpress's own gym;
- other Qiskit versions;
- depth as a target;
- hardware.

Times come from one machine running 12 jobs at a time.

---

<!-- ===== Addendum 409 (source: spare-qubit-cliff-addendum-409-2026-10-09.md) ===== -->

> **Note added when merging:** Pre-registration of C25-ID, committed with candidate 2026-10-09.c25, its tests and its script as the lock, before the run.

## Addendum 409 -- Pre-registration: C25-ID. Candidate 2026-10-09.c25 (changelog item 52) computes the layout search's two maximum matching sizes with rustworkx instead of networkx. Are the outputs unchanged, and is it faster? The 152 development tests of Benchpress (2026-10-09)

**Status: pre-registration.** It was written after the candidate's tests and before C25-ID's run. The lock is the
commit that adds this Addendum with the candidate and the script. It is made at the workplace as for Addendum 405:

1. the lock commit is made locally;
2. its hash is e-mailed before the run;
3. the run records the commit it ran on;
4. the lock commit and the results are pushed later from home, unchanged.

## 1. Why

TOQB's profile of 2026-10-08 (seven development tests, one warm compile each under cProfile) found the release's
default call spends 7-17% of its time in networkx's `max_weight_matching` on HamLib (FakeTorino), Feynman and Ising
tests. Two feasibility checks in [`benchmarks/psf_smart_layout.py`](../../benchmarks/psf_smart_layout.py) call it:

- `_interaction_matching_size`: the circuit's interaction graph;
- `_has_feasible_matching`: the coupling map, on every call, although it does not depend on the circuit.

Both read only the size of a maximum matching. That size is unique, so computing it another way cannot change any
decision. networkx is pure Python; rustworkx, already a dependency, is compiled. In TOQB's time budgets a few
hundredths of a second matter: the smallest budget is 0.05 s.

## 2. The candidate: 2026-10-09.c25 (item 52)

[`patches/psf_compile_c25_2026-10-09/`](../../patches/psf_compile_c25_2026-10-09/):

- `psf_smart_layout.py`: the release's layout file (2026-10-01.1) with both sizes from rustworkx's
  `max_weight_matching` (every weight 1, `max_cardinality=True`), and the coupling map's size kept for the last 32
  maps. `LAYOUT_VERSION` 2026-10-09.c25. The layout itself (Stage 0b's `short_path_layout`, which uses a matching,
  not only its size) is not changed.
- `psf_compile.py`: release 2026-10-07.1 with `VERSION` 2026-10-09.c25 and changelog item 52. Nothing else.
- `test_c25.py`: the sizes equal networkx's on 400 random graphs, 300 random interaction graphs with sparse qubit
  indices, and six coupling maps (FakeTorino's among them), with thresholds around each size, twice (the second time
  from the kept sizes); the kept sizes stay at 32.

## 3. The test

[`benchmarks/c25_identity.py`](../../benchmarks/c25_identity.py):

- **Tests:** the 152 Benchpress tests this project's development used (BP-PROBE's 12, BP-MOCK's 92, BP-MOCK2's 48),
  built as those tests built them (`bp_mock.build`, unchanged). None of BP-FINAL's 880 is used.
- **Calls, as BP-FINAL made them:** the default call on every test; the recommended call with the backend's target
  on the FakeTorino tests.
- **Arms:** REL (the release's two files) and C25 (the candidate's two files). Each test, call and arm runs in its
  own process; one compile, timed, with the module load untimed. The two arms of a test and call are next to each
  other in the queue, in an order set by the test's hash. 1,800 s limit.
- **Output identity:** the SHA-256 of every instruction, its qubits, clbits and parameters, the global phase and the
  initial and final layouts (`bp_mock.sig_hash`, unchanged).

## 4. Predictions

| ID | prediction | confirmed if | refuted if |
|---|---|---|---|
| I0 | the run is as locked | 152 tests; both arms on every job; every record names its arm's two versions; no uncommitted change | any fails |
| I1 | item 52 changes no output | every test and call both arms finish has the same signature | one differs |
| I2 | item 52 changes no failure | the tests and calls that fail (error or timeout) are the same for both arms | they differ |

Reported without prediction: the compile time C25/REL (median and geometric mean), overall and on the pairs REL
compiled in under 1 s.

**What follows.**

- If I0-I2 hold, item 52 can be proposed for the next release on speed alone. The owner decides. Its effect on time
  budgets is measured later in TOQB.
- If I1 or I2 fails, item 52 is not proposed, and the difference is examined.

## 5. What this does not establish

- Speed on other machines, or with the jobs not run in parallel. Times come from one compile per job, with several
  jobs at a time.
- Anything about quality: the outputs are meant to be the release's.

---

<!-- ===== Addendum 410 (source: spare-qubit-cliff-addendum-410-2026-10-09.md) ===== -->

> **Note added when merging:** Results of C25-ID (Addendum 409), with its data.

## Addendum 410 -- Results of C25-ID (Addendum 409): I0 PASS, I1 and I2 REFUTED. 24 of 111 outputs differ and the failures differ, for two reasons outside item 52: the workplace machine ran out of memory with 12 jobs at a time, and the layout search stops by wall-clock time, so its output depends on the machine's speed. The prediction should have allowed for the second (2026-10-09)

**Status: results of the pre-registered test in Addendum 409, scored by the locked script.** The run is
[`data/2026-10-09/c25_identity/`](../../data/2026-10-09/c25_identity/): `c25_identity.jsonl` and `compare.md`. In
the records, the workplace machine's home folder is replaced by `<windows-home>`.

## 1. The run

- **Lock.** Commit `af2e640`, made at the workplace. Its hash was e-mailed a few minutes after the run started,
  before any result was seen.
- **Machine.** The workplace PC (Windows, Python 3.11.9, 14 CPUs), 12 jobs at a time, 372 jobs in 2,589 s. The PC
  froze several times during the run.

## 2. Results

| ID | prediction | value | verdict |
|---|---|---|---|
| I0 | the run is as locked | 152 tests; both arms on every job; every record names its arm's versions; no uncommitted change | **PASS** |
| I1 | item 52 changes no output | 87 identical of the 111 tests and calls both arms finished; 24 differ | **REFUTED** |
| I2 | item 52 changes no failure | REL failed on 55, C25 on 59, not the same set | **REFUTED** |

Reported without prediction: the compile time C25/REL has a median of 0.410 and a geometric mean of 0.422 over the
111 pairs, and no REL compile took under 1 s.

## 3. Why

**Most failures were the machine running out of memory.**

- Of 114 failed jobs, one was a timeout (bwt_n37 on linear, C25, 1,800 s).
- The others ended with `MemoryError`, NumPy unable to allocate about 1 MiB, OpenBLAS unable to allocate memory,
  extension modules that could not be imported, or a process that ended with no message at all.
- They include the smallest circuits (adder_n4, wstate_n3, iswap_n2). BP-FINAL ran 12 jobs at a time on the same
  machine without this; why memory ran short this time was not established.
- Which job fails depends on what else is running at that moment, so the failures differ between the arms whatever
  the arms compute.

**The differing outputs are explained by a search that stops by time.**

- `psf_smart_layout.smart_vf2_layout` has a time budget (2 s), shrinks each attempt's call limit to fit the time
  left, and gives the packing search a time budget of its own. On a slower or busier machine it tries fewer
  candidates, and can return another layout.
- Item 52 makes the two feasibility checks faster. Most of all on all-to-all maps, whose complete coupling graphs
  were slow for networkx. That leaves more of the budget for the search itself.
- 11 of the 24 differences are on all-to-all maps, and 4 are the 100-qubit circSU2 tests on FakeTorino.
- This was known: in BP-FINAL, the release run twice in separate processes (REL and REL2) gave different outputs on
  83 of 877 tests (Addendum 408). Addendum 409's I1 required identity against that background. **That was a design
  error in the prediction, not a property of item 52.**

**The time ratio is not item 52's speed-up in ordinary use.**

- Each job made one compile in a fresh process, under memory pressure.
- The release's checks import networkx inside the function. Its first import (a fraction of a second) is therefore
  inside the timed compile; item 52 does not import networkx there.
- A warm comparison is still needed.

## 4. What this shows about PSF-Zero

**A weakness, recorded as found.** With `layout_search=True`, the same input, seed and version can give different
outputs depending on the machine's speed and load. BP-FINAL measured it at 83 of 877 tests between two runs. Today it
showed in 24 of 111 under heavy load. A search budgeted in calls instead of seconds would make the output a function
of the input; it is a candidate for a later item.

## 5. What follows

- C25-ID2 (Addendum 411) tests item 52 again with the layout module's clock made virtual, so that the search no longer
  depends on the machine's speed, with a control arm (the release twice) and 4 jobs at a time.
- Item 52 is not proposed on this run.

---

<!-- ===== Addendum 411 (source: spare-qubit-cliff-addendum-411-2026-10-09.md) ===== -->

> **Note added when merging:** Pre-registration of C25-ID2, committed with its script and Addendum 410 as the lock, before the run.

## Addendum 411 -- Pre-registration: C25-ID2. C25-ID again with the layout search made independent of the machine's speed (a virtual clock in the layout module), a control arm (the release run twice) and 4 jobs at a time. Does item 52 change any output when timing cannot? (2026-10-09)

**Status: pre-registration.** Written after C25-ID's results (Addendum 410) and before C25-ID2's run. Locked at the
workplace like Addendum 409: a local commit, its hash e-mailed before the run, pushed later from home unchanged.

## 1. The test

[`benchmarks/c25_identity2.py`](../../benchmarks/c25_identity2.py) is C25-ID's script
([`c25_identity.py`](../../benchmarks/c25_identity.py), unchanged) with four differences:

1. **A virtual clock in the layout module.**
   - In each job, after the layout module is loaded, its `time` is replaced by `_VirtualTime`. Every
     `perf_counter()` call advances that clock by 1e-4 s; everything else is the real `time` module.
   - The search's three time-based decisions (its 2 s budget, the call limits shrunk to fit the time left, the
     packing search's budget) then depend only on the sequence of calls. An arm that makes the same decisions makes
     the same calls.
   - Item 52's functions do not read the clock. If they return the same sizes, every arm makes the same decisions.
   - The clock is the same in every arm. `psf_compile.py` and Qiskit are not changed.
2. **A control arm, REL2.** The release again, in its own process: it shows whether the harness itself is
   deterministic.
3. **4 jobs at a time,** not 12. Other programs are closed before the run.
4. **3,600 s per job,** because a search that no longer stops at 2 s of wall time can take longer.

Tests, calls and output signature as Addendum 409: the 152 development tests; the default call on all; the
recommended call on the FakeTorino tests; 558 jobs. The three arms of a test and call are next to each other in the
queue, in an order set by the test's hash. Times are recorded, but they describe this virtual-clock mode, not the
ordinary call.

## 2. Predictions

| ID | prediction | confirmed if | refuted if |
|---|---|---|---|
| J0 | the run is as locked | 152 tests; all three arms on every job; every record names its arm's versions and the virtual clock; no uncommitted change | any fails |
| J1 | the harness is deterministic (REL2 = REL) | every test and call both finish has the same signature | one differs |
| J2 | item 52 changes no output (C25 = REL) | every test and call both finish has the same signature | one differs |
| J3 | the same failures in every arm | the tests and calls that fail are the same in all three arms | they differ |

**How the verdicts are read.**

- If J1 is refuted, something else in the pipeline is not deterministic. J2 and J3 then cannot separate item 52 from
  that, and the test is inconclusive.
- If J1 holds and J2 and J3 hold, item 52 can be proposed for the next release on identity. Its speed in ordinary use
  is measured separately (warm compiles, and TOQB).
- If J1 holds and J2 or J3 fails, item 52 changes something, and it is not proposed.

## 3. What this does not establish

- Speed in ordinary use.
- That the release's outputs are reproducible in ordinary use (Addendum 410 shows they are not under load).

---

<!-- ===== Addendum 412 (source: spare-qubit-cliff-addendum-412-2026-10-09.md) ===== -->

> **Note added when merging:** Results of C25-ID2 (Addendum 411), with its data and the two exploratory scripts.

## Addendum 412 -- Results of C25-ID2 (Addendum 411): J0 PASS, J1 REFUTED (the release differs from itself on 8 of 185 tests and calls), J2 REFUTED on the same 8, J3 CONFIRMED. Inconclusive for item 52, as pre-registered. Two exploratory follow-ups: the release's own output changes from process to process on these tests even with a fixed hash seed and one thread (2026-10-09)

**Status: results of the pre-registered test in Addendum 411, scored by the locked script, and two exploratory
follow-ups that were not pre-registered.** The run is
[`data/2026-10-09/c25_identity2/`](../../data/2026-10-09/c25_identity2/): `c25_identity2.jsonl` and `compare.md`. In
the records, the workplace machine's home folder is replaced by `<windows-home>`.

## 1. The run

- **Lock.** Commit `759f220`, made at the workplace.
- **Machine.** The workplace PC (Windows, Python 3.11.9, 14 CPUs), 4 jobs at a time, 558 jobs in 6,533 s. No job ran
  out of memory.

## 2. Results

| ID | prediction | value | verdict |
|---|---|---|---|
| J0 | the run is as locked | 152 tests; all three arms on every job; versions and the virtual clock in every record; no uncommitted change | **PASS** |
| J1 | the harness is deterministic (REL2 = REL) | 177 identical of 185; 8 differ | **REFUTED** |
| J2 | item 52 changes no output (C25 = REL) | 177 identical of 185; 8 differ | **REFUTED** |
| J3 | the same failures in every arm | one in every arm: hwb10, recommended call, the 3,600 s limit | **CONFIRMED** |

**Reading, as pre-registered.** J1 is refuted, so the test cannot separate item 52 from the pipeline's own
variation: it is inconclusive, and item 52 is not proposed on it.

**Reported without prediction.**

- The 8 where C25 differs from REL are exactly the 8 where REL2 differs from REL:
  - bv_n140 on linear, bv_n30 on square, inverseqft_n4 on linear, qec_sm_n5 on all-to-all (default call);
  - circSU2 with 100 and 89 qubits on FakeTorino, default and recommended calls.
- On the other 177, all three arms agree.
- On all 8, the three arms' two-qubit counts are equal.

## 3. Exploratory follow-ups (not pre-registered; scripts committed, outputs as printed)

**Hash randomization is not the cause** ([`benchmarks/c25_hashseed.py`](../../benchmarks/c25_hashseed.py)).

- The 8 were compiled again in each arm with `PYTHONHASHSEED=0`, and in REL with `PYTHONHASHSEED=1`.
- The three arms still differed on all 8 at seed 0.
- REL at seed 0 equalled REL at seed 1 on 1 of 8.
- Every run gave a new signature.

**What differs, and whether one thread removes it** ([`benchmarks/c25_nondet.py`](../../benchmarks/c25_nondet.py)).
REL was compiled three times on each of the 8, in two environments:

- as C25-ID2 ran;
- with OMP, OpenBLAS, MKL and Rayon on one thread, Qiskit's parallelism off and `PYTHONHASHSEED=0`.

Neither environment gave three identical outputs on any of the 8.

| test | what differs between runs |
|---|---|
| bv_n140, linear | the layouts, and 4-5 instructions in name or qubits. In one thread, two of the three runs agreed, and their signature is C25-ID2's REL signature. |
| bv_n30, square | the layouts, and 24-91 instructions in name or qubits |
| inverseqft_n4, linear; qec_sm_n5, all-to-all | nothing the diagnostic compares: instruction names, qubits, parameter values, global phase and layouts are equal, yet the signature differs. `bp_mock.sig_hash` records parameters with `repr()`, which also records their number type. Equal values held in different types would do this. **Not established.** |
| circSU2, 100 and 89 qubits (4) | not examined: the diagnostic converts parameters to floats, and circSU2's parameters are unbound, so every run of it failed |

## 4. What this shows about PSF-Zero

**A second reproducibility weakness, recorded as found.** On bv_n140 and bv_n30, the release chose different layouts
in different processes. This held with the layout search's clock made virtual, the hash seed fixed and every
numeric library on one thread. The cause is not found.

- A candidate is iteration over objects hashed by identity, whose order follows their memory addresses.
- This weakness is separate from the time dependence of Addenda 410-411.

**The test method also has a weakness.** The output signature can differ between outputs with equal values (third row
of the table). Comparing outputs by value would avoid this: instruction names, qubits, parameter values with their
type ignored, global phase and layouts.

## 5. What follows

- Item 52 is not proposed on C25-ID or C25-ID2.
- Its correctness rests, for now, on its unit tests. Those show the two sizes equal networkx's on 400 random graphs,
  300 random interaction graphs and six coupling maps.
- Any later identity test first shows that the release is deterministic on its tests. It then compares outputs by
  value.
- Making the release deterministic is a candidate item:
  - search budgets counted in calls, not seconds (Addenda 410-411);
  - deterministic iteration where the layout is chosen (this Addendum).

  Item 52 is better tested after that.

---

---

**End of Part 10 of 10 (end of document, for now).** Back to [Part 9](spare-qubit-cliff-combined-248.md), [Part 8](spare-qubit-cliff-combined-135.md), [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
