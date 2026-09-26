# Addendum 175 -- Pre-registration: compound round-trip chain of the PennyLane <-> Qiskit conversion, 100 steps (2026-09-25)

> **Imported into the home series as Addendum 175.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `roundtrip-chain-preregistration-2026-09-25.md`; body below unchanged. Script hash (`roundtrip_compound_chain.py`, `8aa601aa...`) re-checked at home: matches.

**Status: pre-registration for the RunPod pod run, locked at the Project
save time of this document.** **This is not a blind prediction.** A sandbox
dry run made before locking used the same two converter files the pod will
use (section 6), and the conversion is deterministic, so the outcome of
A1-A5 is already known. The pod run tests whether that outcome replicates
on the pod (A6). This is stated here so nobody reads A1-A5 as forecasts.

**Numbering:** no number; assigned when merged (the home record has reached
Addendum 166).

## 1. Why this experiment exists

Earlier checks of the conversion (`tape_to_qiskit` / `qiskit_to_tape`) were
single-shot: convert once, compare once. The question here is whether errors
**compound** when the output is fed back in repeatedly:

    tape_k --tape_to_qiskit--> qc_k --qiskit_to_tape--> tape_{k+1},  k = 1..100

A second question matters more. The bit-order bug fixed in Addenda 164-166
(home record) is **symmetric**: the pre-fix `tape_to_qiskit` handed the
2-qubit matrix to Qiskit without reversing the qubits, and the pre-fix
`qiskit_to_tape` made the same omission on the way back. The two errors
cancel in a round trip. A round-trip test, however many times it is chained,
therefore **cannot** see that bug. This design adds a direct meaning check at
every step and runs the pre-fix converter as a positive control, to show
that the meaning check catches what the round trip misses.

## 2. Fixed design

**Converters, loaded side by side under different module names:**

- NEW: `/root/psf-zero/benchmarks/psf_pennylane_gpu_prototype.py`
  (repository HEAD 2501c7e, with the bit-order fix: `qc.unitary(mat,
  qubits[::-1])` and `qml.QubitUnitary(mat, wires=qubits[::-1])`).
- OLD: `~/pennylane_gpu_mock_test/psf_pennylane_gpu_prototype.py` (the
  workplace copy made before the fix; positive control).

**Tapes (19; both converters run on every tape):**

- F1: 10 random tapes, seeds 0-9, wires `[0, 1, 2, 3]`, 20 operations
  alternating a Haar-random 2-qubit `QubitUnitary` on a random ordered pair
  (often in reversed order) and a random named 1-qubit gate (H, X, Y, Z, RX,
  RY, RZ).
- F2: 5 random tapes, seeds 100-104, same construction on mixed wire labels
  `["q3", "a", 7, "b"]`.
- F3: the four 2026-09-28 XOR tapes (Addendum 148 seed-0 parameters from
  `retrain_seed0()`), with each CNOT written as a 2-qubit `QubitUnitary` of
  the CNOT matrix, 35 operations each. (The converter has no CNOT mapping and
  raises on it; the 2026-09-28 path builds its circuit gate by gate and does
  not call the converter. CNOT is not symmetric in its two qubits, so a
  bit-order error changes these tapes' meaning.)

**Recorded at every step k, against the ORIGINAL tape:**

- `fp_equal`: SHA-256 fingerprint of tape_{k+1} (operation names, wires,
  parameters as full-precision hex / complex bytes) equals the original's.
- `rt_infid`: average-gate infidelity between the original tape's matrix and
  tape_{k+1}'s matrix.
- `meaning_infid`: average-gate infidelity between the original tape's
  PennyLane matrix and `Operator(qc_k).reverse_qargs()`, i.e. whether the
  Qiskit circuit means what the original tape means, in PennyLane's qubit
  order.
- `n_ops`, and `fp0` (the original tape's fingerprint).

Output: 3,800 rows (19 tapes x 2 converters x 100 steps). No timing, GPU or
backend. Harness gate C0: the script stops without scoring if the XOR tapes
cannot be built or NEW cannot convert any tape.

The infidelity of a matrix with itself is about 1e-15 in floating point, not
0, so the meaning tolerance is 1e-12.

## 3. Pre-registered predictions

**A1.** NEW: tape_{k+1} is fingerprint-identical to the original at every
step of every tape (1,900 of 1,900).

**A2.** NEW: `meaning_infid` <= 1e-12 at every step of every tape.

**A3.** OLD: fingerprint-identical at every step of every tape too (the
symmetric bug is invisible to the round trip).

**A4.** OLD: `meaning_infid` > 1e-3 at step 1 for all 15 F1/F2 tapes (the
meaning check catches the pre-fix bit order).

**A5.** F3 (XOR): NEW `meaning_infid` <= 1e-12 at every step, and OLD
`meaning_infid` > 1e-3 at step 1 for all 4 tapes.

**A6 (replication).** The pod's CSV is byte-identical to the sandbox dry
run's: SHA-256 of `~/roundtrip_chain_2026-09-25.csv` =
`5bc5b94e2fe9a73da083ec33371834f893765daf14d4b3d8482d9c5078d5f9c8`.
*Refuted* by any other hash. If refuted, the 19 printed `fp0` values are
compared with section 6: a differing `fp0` means the input tapes themselves
differ in their last bits (random-unitary generation or XOR training on a
different numeric library), not a converter failure, and A1-A5 are then
read on the pod's own numbers.

Each of A1-A5 is *confirmed* if it holds exactly and *refuted* otherwise.

## 4. What this can and cannot establish

It can establish that 100 chained conversions neither drift nor change the
circuit's meaning, and it demonstrates on real code that a chained round
trip gives no protection against a symmetric convention error. It does not
test operations outside the converter's small op set, PSF-Zero synthesis,
the GPU, or hardware. It is not a new check of the 2026-09-28 path, which
does not call the converter.

## 5. Files, integrity check and run commands

| File | What it is |
|---|---|
| [`roundtrip_compound_chain.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/roundtrip_compound_chain.py) | the script (Project: `psf-zero/benchmarks/`; on the pod: `~/pennylane_gpu_mock_test/`) |
| this document | the pre-registered predictions |

Normalized SHA-256 of the script:
`8aa601aa11041b71d0b57067c5a3ca7812ef2c7d25264ac4c9c5103ecb640e44`.

```
cd ~/pennylane_gpu_mock_test
python -c "import hashlib;print(hashlib.sha256('\n'.join(l.rstrip() for l in open('roundtrip_compound_chain.py',encoding='utf-8').read().strip().splitlines()).encode()).hexdigest())"
python -u roundtrip_compound_chain.py 2>&1 | tee ~/roundtrip_chain_run.txt
sha256sum ~/roundtrip_chain_2026-09-25.csv
```

Expected run time: about 1 minute (about 45 s in the sandbox, most of it
the XOR training).

## 6. Dry run before locking (methodology; its outcome is known)

**Setup.** Workplace sandbox (not the pod): Python with PennyLane 0.45.1,
Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0, numpy 2.4.4,
scipy 1.17.1. NEW = the public repository cloned at HEAD 2501c7e (the pod's
commit), OLD = the workplace copy of the prototype. No stubs.

**Design changes made because of the dry run (before locking; no threshold
changed):**

1. The first version fed the XOR tapes to the converter as they are; NEW
   raised `NotImplementedError` on `CNOT`, and C0 stopped the run. F3 was
   changed to write each CNOT as a `QubitUnitary`.
2. OLD had been planned for F1/F2 only; it now runs on F3 as well, and A5
   was extended with the OLD clause.
3. The `fp0` column and printed fingerprints were added for A6.

**Outcome of the final script (run twice; both CSVs byte-identical, SHA-256
as in A6):**

| | NEW steps exact | NEW max meaning_infid | OLD steps exact | OLD min step-1 meaning_infid |
|---|---|---|---|---|
| F1 (10 tapes) | 1000/1000 | 4.2e-15 | 1000/1000 | 0.919 |
| F2 (5 tapes) | 500/500 | 3.3e-15 | 500/500 | 0.937 |
| F3 XOR (4 tapes) | 400/400 | 8.9e-16 | 400/400 | 0.931 |

A1-A5 all hold on this run. Within each chain `meaning_infid` took a single
value over all 100 steps: step 1 returns the original tape exactly, so every
later step repeats step 1 and nothing compounds, for either converter. OLD
passes the round trip at every step while its circuit means something
different (infidelity above 0.9).

**Original-tape fingerprints (`fp0`), for A6:**

| Tape | fp0 | Tape | fp0 |
|---|---|---|---|
| F1 seed0 | 2d619d91f0d06cf7 | F1 seed8 | ca33ccca6dd86bda |
| F1 seed1 | 96c7390341ad98fc | F1 seed9 | f8ef39cb4cee38e9 |
| F1 seed2 | f02a79e70626df0e | F2 seed0 | 4fedc0b2bfce8038 |
| F1 seed3 | b713113e1d989c93 | F2 seed1 | 78d671254c8810cf |
| F1 seed4 | 953c1192827c1d5c | F2 seed2 | f3373718a851d079 |
| F1 seed5 | 772b0475c30612f0 | F2 seed3 | ea47cab6004eac10 |
| F1 seed6 | 6750ab5ac35e889a | F2 seed4 | 2fac28a593274d42 |
| F1 seed7 | 4fafe37f300269af | F3 input0 | b705940c295395c1 |
| | | F3 input1 | f52287f7083d9101 |
| | | F3 input2 | 1ef20ecce1c83ef9 |
| | | F3 input3 | 10e4e7bdc4079f19 |
