# SCALE1 -- fixed before any real-device step (Clifford mirror circuits at 12-144 qubits; no claim of quantum advantage)

Fixed: 2026-10-13 09:02:47 CEST (Europe/Zurich). `scale1_real.py` SHA-256 `c40ab5dfa8beee4699947344b1a9365112d6f72f1c448870d50dd469780c1fb3`; every kit file in `SHA256SUMS`.

**What this test is and is not.**

- The circuits are Clifford mirror circuits with a known answer. A laptop simulates every one of them in well under
  a second, and this kit does so to check them.
- The test measures how well a compiler carries a large circuit through a real device. It does not measure a
  computation that a classical computer cannot do. **No claim of quantum advantage or speed is made**, and any record
  of the results says so in its first paragraph.

**Origin.**

- Designed at the workplace on 2026-10-13: the draft `scale1-design-2026-10-13.md` and its circuit file
  `scale1_circuits.py`. The circuit file is in this kit unchanged (SHA-256 `645627d3...`).
- Revised here before any run, after an offline check on FakeFez. The changes and their reasons are in section 2.

**Seen before this file** (all offline, on FakeFez, compile only; no device data):

- the workplace's pilot;
- this kit's rehearsal, including the compile table in section 4.

## 1. Circuits, arms, job

**Families** (`scale1_circuits.py`; the ideal outcome comes from a stabilizer simulation of the logical circuit):

- **L:** a GHZ chain grown from the middle outwards, a random Pauli layer, the chain's inverse.
- **R2, R4:** d = 2 or 4 layers of random one-qubit Cliffords and CZs on random pairs, a random Pauli layer, the
  inverse.
- A barrier on each side of the Pauli layer.

**Cells:** L n = 12, 24, 48; R2 n = 24, 48, 96, 144; R4 n = 24, 48. Three instances each (seeds fixed in the file):
27 logical circuits.

**Arms**, on the device's Target at `prepare`:

| arm | call |
|---|---|
| PSF | `compile_for_hardware(qc, backend=backend, seed_transpiler=0)` of release 2026-10-11.4 (`psf_compile.py`, SHA-256 `0e97c2e2...`) |
| QK3 | `transpile(qc, target=backend.target, optimization_level=3, seed_transpiler=0)`: a typical user's call |
| QK3F | the same, given a copy of the Target without the elements whose reported error is >= 0.5 |

**Checks before submission**, every circuit:

- **ISA:** only the device's gates on the device's qubits.
- **Exactness at every width:** a noiseless matrix-product-state simulation of the compiled circuit (32 shots) must
  return only the ideal bitstring.
- **No collapse:** the two-qubit count is at least half the logical circuit's. Cancelling the mirror across the
  barrier would remove most two-qubit gates. CZ pairs cancelling inside one half are legitimate; that happened in
  the rehearsal.

**Jobs.**

- Circuits that use an element with error >= 0.5 go in a separate job (part F), so that a rejected circuit cannot
  take the main job with it.
- Shots: 4,000 for n >= 96, 2,000 otherwise. One repeat; arms interleaved (cell, instance, arm). The runtime's
  dynamical decoupling and twirling are off.
- 81 circuits, 198,000 shots. The rough estimate is 180 s; HW1's ratio of used to estimated time (0.40) gives about
  70 s. `prepare` refuses more than 220 s rough.
- The device rule is as HW2's: of ibm_fez and ibm_marrakesh, the operational one with fewer pending jobs.

## 2. Changes from the workplace's draft, and why

| change | why (offline, FakeFez) |
|---|---|
| **Arm QK3F added** | With the failed elements removed from the Target, Qiskit level 3 matched or beat PSF where QK3 had used them: predicted 0.187 against 0.144 at R2 n = 96; 0.048 against 0.040 at n = 144. Without this arm, a difference at 96-144 qubits shows only that Qiskit's default call uses elements reported as failed. |
| **R2 is not a routing test** | At d = 2 the pair graph is a set of disjoint chains, which mostly embed on heavy-hex without a SWAP: both arms had the logical two-qubit count, except one of three instances at n = 144. Routing is tested only by R4. |
| **Exactness checked at every width** | MPS simulation of a compiled 144-qubit circuit takes under 0.2 s. The draft checked only up to 24 qubits. |
| **PSF called with `seed_transpiler=0`** | Without a seed the release gave a different output on each of three calls. QK3 was already seeded. |
| **Collapse check at half the logical count** | The draft's "at least the logical count" rejected a correct instance (R2 n = 144, 130 logical, 128 compiled in all three arms). |

**Also found:** with a seed, the release still returns different placements at 144 qubits from call to call, with
equal ESP. The 2-second time budget of the placement search is the likely cause. Item 75's "same input, same output"
therefore does not hold at that size. This is recorded for a later fix. The run's own compiled circuits are saved in
QPY and are the record.

## 3. Measures

- **S:** the share of shots equal to the ideal bitstring, with standard error sqrt(S (1 - S) / shots). A circuit is
  **resolved** if S >= 5 standard errors. Chance level is 2^-n, negligible from n = 12.
- **Signal width:** per family (L, R2, R4), the largest n whose three instances are all resolved.
- **Fixed predictor** (C4): ESP, readout included, times exp(-sum over used qubits of I (1/(6 T1) + 1/(3 T2))). I is
  the qubit's idle time from its first gate to the end, as soon as possible, minus its own gate time. This is the
  physics-fixed idle term of the workplace's T2-placement draft.
- **Reported:** the readout-only product, the duration, the Hamming-distance profile of every circuit, and S
  against the predicted value.

## 4. Offline compile on FakeFez (rehearsal; means of three instances)

| cell | 2q PSF / QK3 / QK3F | predicted PSF / QK3 / QK3F | readout only PSF / QK3F | QK3 on failed elements |
|---|---|---|---|---|
| L_n12 | 22 / 22 / 22 | 0.750 / 0.795 / 0.795 | 0.89 / 0.92 | 0 of 3 |
| L_n24 | 46 / 46 / 46 | 0.444 / 0.472 / 0.472 | 0.80 / 0.83 | 0 of 3 |
| L_n48 | 94 / 94 / 94 | 0.078 / 0.086 / 0.086 | 0.58 / 0.60 | 0 of 3 |
| R2_n24 | 25 / 25 / 25 | 0.707 / 0.717 / 0.717 | 0.84 / 0.84 | 0 of 3 |
| R2_n48 | 41 / 41 / 41 | 0.520 / 0.346 / 0.518 | 0.71 / 0.70 | 1 of 3 |
| R2_n96 | 102 / 102 / 102 | 0.144 / 0.000 / 0.187 | 0.31 / 0.38 | 3 of 3 |
| R2_n144 | 169 / 176 / 172 | 0.040 / 0.000 / 0.048 | 0.17 / 0.20 | 3 of 3 |
| R4_n24 | 130 / 120 / 129 | 0.283 / 0.357 / 0.357 | 0.81 / 0.83 | 0 of 3 |
| R4_n48 | 306 / 282 / 301 | 0.026 / 0.031 / 0.025 | 0.52 / 0.55 | 0 of 3 |

Where no failed element is involved, PSF's placement is predicted to be slightly worse than Qiskit's. Much of the
gap is readout (PSF chooses qubits with higher readout error). In R4, PSF routes with more two-qubit gates than QK3.

## 5. Predictions

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| C1 | signal width: PSF >= QK3 in each of L, R2, R4 | all three | PSF < QK3 in any |
| C2 | where QK3 used a failed element (part F) and PSF's circuit is resolved: S_QK3 <= 0.2 S_PSF | on every such pair (needs at least two) | S_QK3 >= S_PSF on any |
| C3a | PSF against QK3 where QK3 used no failed element: PSF's S >= QK3's | on at least half of the pairs | PSF's S < 0.8 x QK3's on more than half |
| C3b | PSF against QK3F, every pair: the same | on at least half | the same |
| C4 | the fixed predictor picks the device's winner (PSF against QK3 and QK3F) | on at least 2 of 3 pairs it predicts outside 0.9-1.1 (needs at least three) | on fewer than half |
| C5 | reported: the largest resolved circuit per arm (qubits, two-qubit gates, duration, S); S / predicted; Hamming profiles | - | - |

- C3a and C3b count only pairs where at least one arm is resolved.
- Results between the confirmed and refuted bounds are ambiguous.
- **Expectation, from section 4:**
  - C1 is likely confirmed through the failed elements in R2.
  - C3a and C3b are likely ambiguous or refuted: PSF a little lower.
  - C2 depends on what an element reported at error 1 actually does on the device. That has not been measured
    before, and it is the question C2 answers.

## 6. Reading, fixed now

- **The README may say** that "PSF-Zero's compilation kept a measurable known-answer signal on N-qubit, K-CZ Clifford
  mirror circuits on <device, date> where Qiskit level 3's default call given the same Target did not" only if C1
  and C2 are confirmed.
  - The same sentence must give C3b's result: whether Qiskit given a Target without failed elements did as well.
  - It must say that the circuits are Clifford mirror circuits and that no advantage is claimed.
- **If C3a and C3b are refuted or ambiguous with PSF lower:**
  - PSF's placement is weaker than Qiskit's on clean placement problems. That goes on the development list: readout
    in the placement cost; R4 routing.
  - It is reported in the same place as any C1/C2 result, as the essential number beside the display number.
- **If C2 is refuted** (QK3's circuits on failed elements still resolved as well as PSF's): the elements reported at
  error 1 are not broken in practice for these circuits. Release item 59's benefit is then a guard against a
  reported state, not a measured gain.

## 7. Risks stated in advance

- The failed elements change daily. Part F may be empty, and then C2 is not testable.
- Mirror circuits with a Pauli middle are not randomized compiling. Coherent errors can cancel in the mirror and make
  S look better than a non-mirror circuit of the same size would.
- At 96-144 qubits readout dominates: the readout-only product is 0.17-0.38 in section 4.
- One device, one day, one repeat.

**QPU:** about 70 s expected. Runs only with the owner's go-ahead, after HW2. It is independent of HW3.

## Version 2, fixed before any SCALE1 job

Fixed: 2026-10-13 12:24:09 CEST (Europe/Zurich). `scale1_real.py` version 2, SHA-256 `c3bbd5dcbcd412053313987af3132d300763b3ef030441e7bbc717a389216b17`; every kit file in `SHA256SUMS`.

**Seen before this change:** HW2's results on ibm_marrakesh (Addendum 461, to be written), the same evening. SCALE1's
circuits, arms, checks and predictions C1-C5 are unchanged by them.

**Device fixed to ibm_marrakesh, the owner's choice.** It replaces the rule "of ibm_fez and ibm_marrakesh, the one
with fewer pending jobs".

- The run uses the same device and calibration day as HW2.
- If ibm_marrakesh is not operational at `prepare`, nothing is prepared.

**Part X added: the PennyLane-trained XOR classifier.**

- **Circuits:** Stage 2's four logical circuits, built by the locked Stage-1 code, logical qubit 0 measured.
- **Arms:** R4, the release with HW1's exact call (`entangling_basis="cx"`, `layout_search=True`,
  `seed_transpiler=0`), and QK3, Qiskit level 3 with HW1's call.
- **Shots and job:** 4,000 shots each, in the main job.
- **Check:** each circuit is checked as in HW1: |<Z0>| = 0.99776 x label within 6e-5 by statevector, ISA, no failed
  element.
- **Total:** 89 circuits, 230,000 shots. The rough estimate is 204 s, under the 220 s cap; about 85 s by HW1/HW2's
  ratio.

| ID | prediction (as HW1's X1, X2) |
|---|---|
| X1 | each arm classifies all four inputs correctly |
| X2 | each arm's observed M >= its predicted M (Aer, the device's calibration) - 0.15 |
| X3 | reported: R4's M here against ibm_kingston 2026-10-12 (HW1's R4 0.9237, Stage 2's chosen arm 0.937) |
| X4 | reported: R4's M against QK3's |

The rehearsal on FakeFez (noiseless) ran end to end: 8 XOR circuits exact; part X scored.

## Version 3, fixed before any SCALE1 job

Fixed: 2026-10-13 12:38:28 CEST (Europe/Zurich). `scale1_real.py` version 3, SHA-256 `4b18ab35f09bea8cd4e057cbd17adbf2f38c2c81953162945b799e51d6541a5b`; every kit file in `SHA256SUMS`.

**What happened.** The first `prepare` on ibm_marrakesh (12:30 CEST; 31 elements with error >= 0.5) compiled the
first 60 circuits. Then the release refused R2 n = 144, instance 0, with `FailedElementsError` ("the circuit compiled
on the map without the failed couplers still uses a failed qubit", changelog item 58). The script stopped; nothing was
submitted. No other result of that `prepare` was used to change anything.

**Change.**

- An exception from any arm's compiler is recorded as a **refusal** for that circuit and arm, and `prepare` goes on.
- A refused circuit is not submitted and counts as **not resolved** for its arm. This affects C1, the signal width,
  and X1, where a refused XOR input counts as not correctly classified.
- Refusals are listed in the score.
- Nothing else changes, and the device stays ibm_marrakesh. Moving to another device after seeing PSF refuse here
  would choose the device for PSF's sake.

**Reading, fixed now.** A refusal is the release's safety guard (item 58) working as designed. On a device whose failed
qubits leave too little room, it also means PSF-Zero returns no circuit where Qiskit returns one. Both facts are
reported. Whether the refusal could have been avoided (failed qubits are taken off the map only after failed couplers)
goes on the development list.

**Rehearsal** (FakeFez, refusals injected for PSF at n = 144 and for one R4 XOR input): `prepare` went on; the refused
circuits were listed, counted as not resolved, and C1 and X1 were scored accordingly.
