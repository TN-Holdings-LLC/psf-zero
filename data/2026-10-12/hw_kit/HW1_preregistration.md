# HW1 and PSF-XOR -- fixed before any real-device step (local; to be recorded as Addendum 454, results as 455)

Written 2026-10-12 before the real-device session, kept on the owner's PC with its sha256 and CEST time, and checked
against them before `prepare`. Run on the same day as Stage 2's attempt 2 (Addendum 211, unchanged; results as
Addendum 212).

**Seen before this file:**

- a full rehearsal on FakeKingston with a noiseless local simulator (every circuit exact, every F and R near 1, XOR
  4 of 4);
- the circuits' ESP on FakeKingston and FakeFez (calibration snapshots of 2026-04);
- no real-device data of any kind since Addendum 211.

## 1. Why

Every gain PSF-Zero has claimed since Addendum 300 or so is an ESP computed from a fake device's calibration
(summed -ln(1 - error) of every operation, no idling term). HW1 asks whether that estimate ranks the compilers
correctly on a real device: where PSF-Zero's own pipeline predicts a higher ESP than Qiskit level 3, does the real
device give a higher fidelity?

PSF-XOR runs the PennyLane-trained XOR classifier (Stage 2's four circuits, built by the locked Stage-1 code) through
PSF-Zero to the real device: the whole PennyLane -> PSF-Zero -> IBM pipeline on hardware.

## 2. Design (`hw1_real.py`, sha256 in `SHA256SUMS`)

**Device rule:** `ibm_kingston` if operational when `prepare` runs, else `ibm_fez`, else nothing is prepared. (The
dashboard on 2026-10-12: kingston 2Q layered error 3.1e-3, readout 8.7e-3, 3 pending jobs; fez 4.8e-3, 1.07e-2, 35;
marrakesh paused.) The device's Target at that moment is used for compiling and for every predicted ESP.

**HW1 circuits** (all with ideal distributions computed exactly; at most 20 qubits):

| test | what |
|---|---|
| qaoa_n12_p2, qaoa_n20_p2 | QAOAAnsatz on a random 3-regular graph (seed "HW1|<test>"), linear ramp gamma_k = 0.55 (k + 1)/p, beta_k = 0.45 (1 - k/p) |
| ba_n14, ba_n16, ba_n18, ba_n20 | Benchpress `qaoa_barabasi_albert_N{n}_3reps.qasm` (copies in the kit's `qasm/`) |
| tfim_n12, tfim_n16 | open-chain transverse-field Ising, J = h = 1, t = 0.6, Suzuki-Trotter order 2, 2 steps |
| ghz_n12 | H and a cx chain (control: the same circuit expected from both compilers) |

**Arms:** QK3 = Qiskit level 3 (`generate_preset_pass_manager(3, backend, seed_transpiler=0)`); PSF = prototype c45
with `level3_candidate=False` (PSF-Zero's own pipeline, no Qiskit candidate); C47 = prototype c47 (item 82) with
`level3_candidate=False`, submitted only where its circuit differs from PSF's. Every circuit is checked before
submission: the device's gates only, no failed element (error >= 0.5), and its measured distribution equal to the
ideal one to 1e-6 by noiseless simulation; a circuit failing a check is not submitted and is reported.

**PSF-XOR circuits:** Stage 2's four logical circuits (measuring logical qubit 0) compiled by release 2026-10-11.4
(R4) and by c45 (C45, default call). Predicted z from an Aer noise model of the device now (20,000 shots, seed
1000), as in Stage 2.

**One job:** HW1 circuits twice (repeat 1, then the XOR circuits, then repeat 2 in reverse order), 6,000 shots each;
XOR circuits once, 4,000 shots each. SamplerV2 with dynamical decoupling and twirling off. `prepare` refuses a job
whose rough QPU estimate (0.5 ms per shot + 1 s per circuit) exceeds 240 s.

**Measures:**
- F, primary: normalised linear cross-entropy against the ideal distribution, (2^n mean p_ideal(x) - 1) / (2^n
  sum p_ideal^2 - 1): 1 for the ideal sampler, 0 for uniform noise. Mean of the two repeats.
- R, secondary: observable / ideal (QAOA: sum <Z_i Z_j> over the cost terms; TFIM: sum of nearest-neighbour
  <Z Z>; GHZ: P(all 0) + P(all 1)); not used where the ideal is below 0.1 per term.
- XOR: M = mean over the four inputs of label x <Z0>, as Stage 2.

## 3. Predictions

| ID | Prediction | CONFIRMED | REFUTED (otherwise AMBIGUOUS) |
|---|---|---|---|
| H1 | Where the device's ESP predicts PSF >= 1.2 x QK3, the device agrees | PSF's F >= QK3's on more than half of those tests | on fewer than half |
| H2 | ESP ranks the PSF/QK3 differences | rank correlation of the ESP ratio and the F ratio over the non-GHZ tests > 0 | < 0 |
| H3 | The control shows no difference that ESP does not predict | GHZ: \|F ratio - ESP ratio\| <= 0.10 | otherwise NOT CONFIRMED |
| H4 | (reported) F / ESP per arm: how much the idle-free ESP over- or under-states the real fidelity | -- | -- |
| X1 | PSF-XOR: each arm classifies all four inputs correctly | 4 of 4 for R4 and C45 | -- |
| X2 | PSF-XOR: observed M >= predicted M - 0.15 (Stage 2's S2 floor) | for each arm | -- |
| X3 | PSF-XOR: R4's M >= Stage 2's chosen arm's M - 0.02 (two jobs minutes apart; reported with that caveat) | -- | -- |
| P0 | Every step completes; the job uses at most 240 QPU seconds | -- | -- |

**Two numbers to report:** display, the F ratio PSF / QK3 per test and pooled; essential, whether the ESP ratio
predicted on the day ranks and sizes the real ratio (H2, and the pooled F ratio against the pooled ESP ratio).

**Reserve (decided now):** if H1 is AMBIGUOUS, or the largest repeat difference exceeds |F_PSF - F_QK3| on more than
half of H1's tests, the same circuit file (same QPY, same manifest) is submitted once more in a new job, if the
allowance left is at least 4 minutes; both jobs are reported separately and pooled. Nothing else is resubmitted.

## 4. What this will not establish

- One device, one day, one calibration. Circuits up to 20 qubits.
- XEB measures closeness to the ideal distribution, not a task's success; the problem observables are secondary.
- Prototypes c45 and c47 are not releases. C45-VAL (Addendum to come) validated c45 on fake devices; c47 is
  unvalidated beyond its tests (27 passed) and HW1's exactness checks.
