# Addendum 173 -- Pre-registration: long-run stability of the 2026-09-28 pipeline and of PSF-Zero, 100 iterations x 2 processes (2026-09-25)

> **Imported into the home series as Addendum 173.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `longrun-stability-preregistration-2026-09-25.md`; body below unchanged. Script hash (`xor_prereg_longrun_stability.py`, `ec8f9b77...`) re-checked at home: matches. The "home Addenda 94-95" this record mentions are in Part 6 ([`spare-qubit-cliff-combined-88.md`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/docs/findings/spare-qubit-cliff-combined-88.md)): Addendum 94 found garbage collection contributing to compile-time variance (2.73x) and that disabling it grew memory by 2.27 GB over 10,000 calls. Note found at home when reading the script: W2 re-synthesizes the same tape every iteration, so from iteration 2 on PSF-Zero's CX-core cache (`_cx_core_cached`, keyed on the exact Cartan floats) is hit; L3 still tests whether the Rust core returns bit-identical floats each time.

**Status: pre-registration, locked at the Project save time of this
document**, before any run on the pod. A short sandbox dry run with stubs was
made before locking (section 6).

**Numbering:** no number (the proposal called it "Addendum 167"; the home
record may already use 167). Assigned when merged.

## 1. Why this experiment exists

The question has moved from "does it run?" to "can it be operated?": does
the pipeline give the same result, at a stable speed, without memory growth,
when run many times. A proposal for a 100-iteration audit was reviewed
before writing this; it was changed in five ways, each for a stated reason:

1. **A second workload where PSF-Zero actually acts.** On the XOR circuit
   PSF-Zero does nothing (0 blocks; Stage 1c's C7), so an XOR-only audit
   measures Qiskit, not PSF-Zero.
2. **Predictions that could fail.** With fixed seeds, "the same qubit and
   2-qubit count 100 times" and "100/100 correct" are guaranteed by
   construction. They are kept, but labelled as operational checks (L1, L2).
   The substantive new prediction is L3: **PSF-Zero returns a bit-identical
   circuit every time** (full-precision parameters). Stage 1c found PSF-Zero
   and Qiskit ZSX structurally identical yet with small TVD differences in 8
   cells, so bit-level determinism is not assumed.
3. **A different simulator seed every iteration**, so the 100 noisy values
   form a real distribution that can be checked against the binomial model.
4. **Warm-up excluded from timing by rule**: iterations 1-5 are reported
   separately (cold start is itself an operational number) and the CV is
   computed on iterations 6-100.
5. **Memory measured as current RSS (`/proc/self/status` VmRSS), compared at
   iteration 10 versus 100**, and explicit `gc.collect()` compared in
   **separate processes** rather than once at iteration 50. (The proposal
   linked this to Addenda 94-95 of the home record, which are not in this
   Project; this design does not depend on them.)

## 2. Fixed design

**Per iteration** (100 iterations per process):

- **W1, the 2026-09-28 path:** load the trained XOR parameters from
  `~/xor_params_seed0.npy` (created once by `retrain_seed0()`; a harness check
  confirms the loaded parameters reproduce the rehearsal's exact <Z0> =
  +-0.99776), build the four circuits with logical qubit 0 measured, choose
  one measure-aware layout (input 00, `optimization_level=3`, seed 0) and
  transpile all four with it (Stage 1b's M4) on FakeBrisbane, simulate each
  with `AerSimulator.from_backend`, 4,000 shots, simulator seed
  `offset + 10 x iteration + input` (offset 0 or 100,000 by process).
- **W2, PSF-Zero acting:** Addendum 156 tape 0 synthesized by PSF-Zero
  (Rust core, `on_unsupported="raise"`, one synthesizer instance reused for
  the whole process) through the repository's `build_synthesized_circuit`
  (real `lightning.gpu` block check and equivalence check), measured on all
  four qubits, transpiled to FakeBrisbane (`optimization_level=3`, seed 0),
  simulated with 20,000 shots, seed `offset + 10 x iteration`; TVD against the
  exact distribution.

**Recorded per iteration:** W1 load/build, compile, simulate and total
times; per input <Z0>, correctness, measured physical qubit, routed 2-qubit
count and an exact circuit fingerprint (SHA-256 of gate names, qubit indices
and parameters in full-precision hex); W2 synthesize-and-verify, compile,
simulate and total times, fallbacks, worst GPU difference, fingerprints of
the synthesized and routed circuits, 2-qubit count, TVD; RSS; timestamp.

**Two processes:** `--gc none` (no explicit collection; Python's automatic
gc stays on) and `--gc each` (`gc.collect()` after every iteration). Then
`--score` reads both CSVs and refuses to score unless each has 100 rows.

## 3. Pre-registered predictions

**L1 (operational check; expected by construction).** W1: all four inputs
correct in all 200 iterations (800 of 800).

**L2 (operational check; expected by construction).** W1: per input, the
routed-circuit fingerprint, measured qubit and 2-qubit count are identical
in all 200 iterations across both processes.

**L3 (the substantive prediction).** W2: exactly **one** distinct
synthesized-circuit fingerprint and one routed-circuit fingerprint over all
200 iterations across both processes, and 0 fallbacks. *Refuted* by a second
fingerprint or any fallback.

**L4 (statistics).** For each process, W1 pooled ratio R of the observed SD
of <Z0> (100 samples per input) to the binomial SD: *confirmed* if
0.85 <= R <= 1.15, *refuted* if R < 0.7 or R > 1.3.

**L5 (timing; RunPod pod only).** Coefficient of variation over iterations
6-100 of W1 compile time and of W2 synthesize-and-verify time, in both
processes: *confirmed* if all four CVs < 0.10, *refuted* if any > 0.25.
The first iteration's ratio to the median is reported without a prediction.

**L6a (memory).** In each process, RSS at iteration 100 / RSS at iteration
10 <= 1.05 (*confirmed*); *refuted* if > 1.20 in either.

**L6b (explicit gc).** RSS at iteration 100, gc-each / gc-none, within 5%
of 1 (*confirmed*); *refuted* if more than 15% away.

Otherwise each is *ambiguous*.

## 4. What this can and cannot establish

It can show that the pipeline, as it will be used on 2026-09-28 and as it
uses PSF-Zero elsewhere, gives the same answers repeatedly, keeps a stable
speed on this machine and does not leak memory over 100 iterations. That is
evidence of software reliability for a proof of concept. It says nothing
about real-hardware behaviour or about PSF-Zero's compression value (see
Stage 1c). Timing numbers come from a shared cloud pod and are not
comparable with the home or workplace machines.

## 5. Figures (made in the workplace sandbox from the downloaded CSVs)

1. W1 compile time and W2 synthesize-and-verify time, iterations 6-100,
   both processes (box plots).
2. W1 <Z0> distribution per input over 100 iterations, with the binomial
   expectation.
3. RSS versus iteration for both processes.

## 6. Dry run before locking (methodology, not data)

The script ran in the workplace sandbox for 12 iterations per process with
stubs (PSF-Zero replaced by Qiskit's ZSX decomposer, CPU instead of GPU
check, a hand-written stand-in for the XOR module), and scoring ran on those
files with the row count check relaxed. It showed that the two processes
used identical simulator seeds, making L4's two values copies of each
other; the per-process seed offset was added before locking. No threshold
was changed. The stub run's numbers are not reported (sandbox machine, stub
modules).

## 7. Files, integrity check and run commands

| File | What it is |
|---|---|
| [`xor_prereg_longrun_stability.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_longrun_stability.py) | the script (Project: `psf-zero/benchmarks/`; on the pod: `~/pennylane_gpu_mock_test/`, next to the Stage-1 script) |
| this document | the pre-registered predictions |

Normalized SHA-256 of the script:
`ec8f9b77319d43d732a251d4a804116c3a48a1e5a2f84742d7af96fcddfc37dd`.

```
cd ~/pennylane_gpu_mock_test
python -c "import hashlib;print(hashlib.sha256('\n'.join(l.rstrip() for l in open('xor_prereg_longrun_stability.py',encoding='utf-8').read().strip().splitlines()).encode()).hexdigest())"
python -u xor_prereg_longrun_stability.py --gc none 2>&1 | tee ~/longrun_none.txt
python -u xor_prereg_longrun_stability.py --gc each 2>&1 | tee ~/longrun_each.txt
python -u xor_prereg_longrun_stability.py --score   2>&1 | tee ~/longrun_score.txt
```

Outputs: `~/longrun_none_2026-09-25.csv`, `~/longrun_each_2026-09-25.csv`
(100 rows each) and the scoring log. Expected run time 15-25 minutes for
both processes (estimated from the dry run, not measured on the pod).
