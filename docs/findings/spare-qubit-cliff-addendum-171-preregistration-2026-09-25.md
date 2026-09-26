# Addendum 171 -- Pre-registration, Stage 1c: circuit size and noisy score, PSF-Zero versus Qiskit, on circuits where PSF-Zero actually acts; and the XOR null control (2026-09-25)

> **Imported into the home series as Addendum 171.** Written at the workplace, run on a RunPod pod (RTX 4090), original file `xor-real-device-stage1c-preregistration-2026-09-25.md`; body below unchanged. Script hash (`xor_prereg_stage1c_sweep.py`, `4d342727...`) re-checked at home: matches. After reading this pre-registration, the home assistant flagged that R0 would very likely fail: the reference CSV (Addendum 156) predates the tape<->Qiskit conversion fix of Addenda 165-166, which changes the logical circuits and therefore the TVDs. Whether that note reached the workplace before the run is not recorded; Addendum 172 reports R0 failing for exactly that reason.

**Status: pre-registration, locked at the Project save time of this
document**, before the locked run on the pod. A sandbox dry run was made
before locking with PSF-Zero **stubbed** (section 7); it included real
Qiskit arms, and what it showed is disclosed there.

**Numbering:** no number; assigned when merged with the home record.

## 1. Why this experiment exists

Two hypotheses were proposed for the record before 2026-09-28:

1. Left to itself, Qiskit (`transpile(optimization_level=3)`) inserts
   unnecessary SWAPs and bloats the circuit, while PSF-Zero keeps it minimal
   ("size 56, depth 16, 0 fallbacks").
2. Running PSF-Zero and standard Qiskit side by side on noisy simulators
   will show a clear score difference, so that a good real-device result on
   2026-09-28 can be attributed to PSF-Zero's compression.

What the existing record already says:

- "Size 56, depth 16, 0 fallbacks" is PSF-Zero's arm in Addendum 156
  ([`data/compare_with_without_psf_2026-09-24.csv`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/data/compare_with_without_psf_2026-09-24.csv), FakeManilaV2, 5 tapes),
  against **84 / 23** for Qiskit's `TwoQubitBasisDecomposer(CXGate())` with
  its default Euler basis. Routed 2-qubit gates were **6 in both arms**
  (no SWAPs either way); the difference is all single-qubit gates.
- Noisy TVD in the same file shows no consistent direction (PSF lower on 3
  of 5 tapes, higher on 2). Addendum 157: no difference in noisy accuracy.
- Addendum 159 (per the 2026-09-25 handover): Qiskit with
  `euler_basis="ZSX"` produces **exactly the same circuit** as PSF-Zero;
  PSF-Zero's synthesis is 5.6-7.6x slower.
- The XOR classifier sent on 2026-09-28 contains **no block PSF-Zero would
  re-synthesize** (`blocks` = 0 in every cell of Stages 1 and 1b and the
  rehearsal). The circuit reaching the device is the same with or without
  PSF-Zero.

So hypothesis 2 cannot be tested on the XOR circuit at all: there is nothing
to compare. Stage 1c instead (a) re-measures the compression claim on the
circuits where PSF-Zero does act, against the fair baseline (Qiskit ZSX) and
against Qiskit's full pipeline, on the 2026-09-28 device class; (b) measures
the noisy-score difference there; and (c) records, as a pre-registered null
control, that PSF-Zero leaves the XOR circuit untouched.

## 2. Fixed design

**Circuits.** The five tapes of Addendum 156 (`compare_with_without_psf.py`
`make_tape`, seeds 0-4: two qubit pairs, each a run of 15 random 2-qubit
unitaries consolidated into one block). Synthesized once per tape with the
repository's `build_synthesized_circuit` (each block passes the real
`lightning.gpu` check and the whole circuit passes an operator-equivalence
check).

**Arms.**
- **A:** Qiskit `TwoQubitBasisDecomposer(CXGate())`, default Euler basis
  (Addendum 156 arm A).
- **Z:** Qiskit `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")`.
- **P:** PSF-Zero `SU4GeodesicPSFSynthesizer` (Rust core,
  `entangling_basis="cx"`, `on_unsupported="raise"`, `verify=True`).
- **T:** no pre-synthesis; the unsynthesized, measured circuit goes to
  `transpile(..., optimization_level=3, seed_transpiler=0)` ("leave
  everything to Qiskit", with the measurement inside so the layout sees it,
  per Stage 1b).

**Backends.** FakeManilaV2 (Addendum 156's backend) plus the 21 backends of
Stage 1's rule; FakeKyoto run but not scored (all 2-qubit errors 1.0). 21
scored backends x 5 tapes = 105 scored cells per arm.

**Routing.** For A, Z and P, one layout per (backend, tape), taken from a
measure-aware `optimization_level=3` transpile of arm Z's circuit, then
`transpile(initial_layout=that, optimization_level=1, seed_transpiler=0)`,
so the three synthesized arms run on the same qubits and differ only in
synthesis. T chooses its own layout.

**Measurement and score.** The four logical qubits are measured (A, Z, P:
at their final physical positions). Noisy score = total variation distance
(TVD) between the sampled distribution and the exact distribution of the
logical circuit; `AerSimulator.from_backend`, 100,000 shots, simulator
seed 0. Recorded per cell: routed 2-qubit gates, single-qubit gates, size,
depth (before measurement), TVD, PSF fallbacks.

**R0 (must pass before scoring).** Addendum 156's own path (FakeManilaV2,
`route_for_backend`, SamplerV2 local mode, seed 42, 4,000 shots) must
reproduce the repository CSV exactly for arms A and P: routed 2-qubit
gates, depth, size, fallbacks identical; `tvd_noisy` and `tvd_ideal` within
1e-9. If R0 fails, predictions are not scored.

## 3. Pre-registered predictions

**C1 (no fallback).** PSF-Zero fallbacks = 0 on all five tapes.

**C2 (no 2-qubit compression by any synthesizer).** Routed 2-qubit gates =
6 (two blocks x 3) in arms A, Z and P in every scored cell. *Refuted* by any
other value.

**C3 (Qiskit's full pipeline does not bloat this circuit).** Routed 2-qubit
gates = 6 in arm T in every scored cell (no SWAP overhead). *Refuted* if
any cell has more than 6.

**C4 (PSF-Zero equals Qiskit ZSX).** In every scored cell, P and Z have
identical size and depth and |TVD_P - TVD_Z| <= 0.002. *Refuted* by any cell
that differs in size or depth or exceeds 0.002.

**C5 (PSF-Zero is smaller than Qiskit's default decomposer).** In every
scored cell, P has smaller size **and** smaller depth than A. *Refuted* by
any cell where it does not.

**C6 (no meaningful noisy-score advantage).** Mean over scored cells of
(TVD_A - TVD_P) <= 0.01 (*confirmed*). *Refuted* if > 0.02 (a real
advantage for PSF-Zero). Otherwise *ambiguous*.

**C7 (XOR null control).** For all four XOR circuits, PSF-Zero's block
collection finds 0 blocks and the circuit after the PSF path is identical,
instruction by instruction, to the input.

No prediction is made for arm T's size, depth or TVD relative to the other
arms; they are reported.

## 4. What this can and cannot establish

It can establish, on the 2026-09-28 device class under Aer noise models,
whether PSF-Zero's circuits are smaller than Qiskit's default decomposer
and Qiskit's full pipeline, whether that is identical to Qiskit's ZSX
setting, whether any of it changes the noisy score, and that the XOR
circuit is untouched by PSF-Zero.

It **cannot** attribute any 2026-09-28 result to PSF-Zero: by C7's premise
the submitted circuit is the same with or without PSF-Zero. What 2026-09-28
can test is the layout procedure (Stages 1 and 1b); a real-hardware
comparison of that needs a pinned-layout control run on the same device
(to be decided in Stage 2). No timing is measured.

## 5. Stage-2 relevance

C7 goes into the 2026-09-28 record as the reason no PSF-Zero claim is made
from that run. C1-C6 are the up-to-date answer, on current device models,
to the question "does PSF-Zero compress circuits beyond what Qiskit can do".

## 6. Files, integrity check and run command

| File | What it is |
|---|---|
| [`xor_prereg_stage1c_sweep.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/xor_prereg_stage1c_sweep.py) | Stage-1c script (Project: `psf-zero/benchmarks/`; on the pod: `~/pennylane_gpu_mock_test/`, next to the Stage-1 script) |
| [`compare_with_without_psf.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compare_with_without_psf.py), [`psf_compile.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile.py) and the prototypes | repository modules imported unchanged from `/root/psf-zero/benchmarks` |
| this document | the pre-registered predictions |

Normalized SHA-256 of the script:
`4d342727ffafddf635a30488bee618155431be077c75a42cac6b7eb36f7bf195`.

```
cd ~/pennylane_gpu_mock_test
python -c "import hashlib;print(hashlib.sha256('\n'.join(l.rstrip() for l in open('xor_prereg_stage1c_sweep.py',encoding='utf-8').read().strip().splitlines()).encode()).hexdigest())"
python -u xor_prereg_stage1c_sweep.py 2>&1 | tee ~/xor_prereg_stage1c_run.txt
```

Output: `~/xor_prereg_stage1c_2026-09-25.csv` (22 backends x 5 tapes x 4
arms = 440 rows) and the scoring at the end of the log. Expected run time
about 10-15 minutes (from the dry run, not measured on the pod).

## 7. Dry run before locking -- disclosure

The script was run in the workplace sandbox on FakeManilaV2, FakeBrisbane
and FakeTorino with **stubs** for PSF-Zero (replaced by Qiskit's ZSX
decomposer, so "P" was a copy of Z), for the real-GPU check (CPU check) and
for the SamplerV2 submit function. **Arms A, Z and T were real Qiskit** on
real fake-backend models, so the dry run produced genuine information about
them. The predictions in section 3 were written before the dry run and are
unchanged. What the dry run showed that bears on them:

- On FakeManilaV2, arms A and Z reproduced Addendum 156's structural numbers
  (A: 84 / 23; Z: 56 / 16), consistent with Addendum 159. R0's noisy TVD
  could not be checked (stubbed sampler).
- On the two heavy-hex backends, after translation to the device basis, Z was
  **not** always smaller than A (in depth on FakeTorino it was larger), and
  T was **no larger** than A or Z. If P equals Z (C4), **C5 is therefore
  likely to be refuted**, and the locked run is, for C5 and for the T
  comparison, a confirmation on 21 backends rather than a blind test.
- The XOR null control behaved as C7 predicts.
- The stub dry run's TVD values are not reported.
