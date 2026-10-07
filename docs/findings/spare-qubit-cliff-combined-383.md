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

---

**End of Part 10 of 10 (end of document, for now).** Back to [Part 9](spare-qubit-cliff-combined-248.md), [Part 8](spare-qubit-cliff-combined-135.md), [Part 7](spare-qubit-cliff-combined-108.md), [Part 6](spare-qubit-cliff-combined-88.md), [Part 5](spare-qubit-cliff-combined-51.md), [Part 4](spare-qubit-cliff-combined-41.md), [Part 3](spare-qubit-cliff-combined-27.md), [Part 2](spare-qubit-cliff-combined-17.md) or [Part 1](spare-qubit-cliff-combined.md).
