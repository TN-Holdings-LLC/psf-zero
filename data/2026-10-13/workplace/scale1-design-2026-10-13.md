# Design (draft pre-registration): SCALE1 -- do PSF-Zero's compiled circuits keep a measurable signal at 24-144 qubits on a real Heron device, where Qiskit level 3's may not? Known-answer mirror circuits, no claim of quantum advantage (2026-10-13)

**Status: design with an offline compile pilot on FakeFez; nothing submitted.** Runs only with the owner's go-ahead,
after HW2, HW3 and the T2*/ZZ measurement. Times CEST.

## 1. What it claims and what it does not

- **Claims it can support:** at widths where brute-force simulation of a general state is impossible (2^96 or 2^144
  amplitudes), a circuit compiled by PSF-Zero for the device still returns its known answer with a measurable
  probability, and how that compares with Qiskit level 3 given the same device Target.
- **Does not claim:** quantum advantage. The circuits are Clifford circuits, which a laptop simulates instantly; the
  test measures how well a compiler gets a large circuit through a real device, not a computation a supercomputer
  cannot do. The record must say this in its first paragraph.

## 2. Circuits (`scale1_circuits.py`, fixed with this design)

Two families, each with a deterministic ideal outcome computed from the logical circuit by a stabilizer simulation:

| family | logical circuit | what the compiler decides |
|---|---|---|
| L (placement) | a GHZ chain grown from the middle outwards, a random Pauli layer, the chain's inverse | where the chain sits on the device (errors, T1/T2, failed elements); routing is not needed |
| R (routing) | d layers of random one-qubit Cliffords and CZs on random, non-native pairs, a random Pauli layer, the inverse | routing and placement |

- A barrier on both sides of the Pauli layer stops compilers from cancelling the halves.
- **Cells:** L at n = 12, 24, 48; R with d = 2 at n = 24, 48, 96, 144; R with d = 4 at n = 24, 48. Three instances per
  cell (seeds fixed in the file): 27 logical circuits.
- **Arms:** PSF = `compile_for_hardware(qc, backend=backend)` of the release current at the run (2026-10-11.4 or a
  release adopted before the run, named in the manifest); QK3 = `transpile(qc, target=backend.target,
  optimization_level=3, seed_transpiler=0)`, as a typical user calls it.

## 3. Offline pilot (FakeFez, compile only; instance 0)

| cell | PSF: CZ, duration | QK3: CZ, duration | ESP PSF / QK3 | ESP x idle term, PSF / QK3 |
|---|---|---|---|---|
| L n=12 | 22, 1.4 us | 22, 1.4 us | 0.81 / 0.85 | 0.73 / 0.77 |
| L n=24 | 46, 2.6 us | 46, 2.6 us | 0.64 / 0.70 | 0.41 / 0.44 |
| L n=48 | 94, 5.2 us | 94, 5.2 us | 0.36 / 0.39 | 0.07 / 0.08 |
| R d=2 n=24 | 24, 0.5 us | 24, 0.5 us | 0.73 / 0.72 | 0.69 / 0.69 |
| R d=2 n=48 | 40, 0.5 us | 40, 0.5 us | 0.57 / 0.57 | 0.50 / 0.50 |
| R d=2 n=96 | 98, 0.5 us | 98, 0.5 us | 0.17 / **0.000** | 0.12 / **0.000** |
| R d=2 n=144 | 140, 0.5 us | 140, 0.5 us | 0.08 / **0.000** | 0.05 / **0.000** |
| R d=4 n=24 | 144, 5.7 us | 114, 3.2 us | 0.44 / 0.55 | 0.15 / 0.34 |
| R d=4 n=48 | 307, 5.8 us | 283, 5.0 us | 0.09 / 0.14 | 0.01 / 0.03 |

(The idle term is the physics-fixed one of the T2-placement draft. ESP 0.000: the circuit uses an element the Target
reports failed.)

What the pilot says for the design:
- At 96 and 144 qubits Qiskit level 3 placed gates on FakeFez's failed elements; PSF-Zero, given the device, does not
  (release item 59). This is the most likely large difference on hardware too (Addenda 246, 430, 434).
- Where both avoid failed elements, Qiskit level 3 is level or slightly better on these Clifford circuits (R d=4:
  fewer CZ). The test must not assume PSF-Zero wins there.
- R with d = 8 at 96 qubits needed about 3,000 CZ (signal far below measurable); deeper cells were dropped.

## 4. Run

- **Device rule:** as HW2 (ibm_fez or ibm_marrakesh, the operational one with fewer pending jobs); the live Target at
  `prepare` is used for compiling and for every estimate.
- **Checks before submission:** the device's gates only; at most 24 active qubits: the ideal outcome by noiseless
  simulation of the compiled circuit; every circuit: two-qubit count at least the logical circuit's (no cancellation
  across the barrier). A wrong compilation cannot pass unnoticed: it returns a different bitstring and scores near 0.
- **Failed elements:** circuits that use an element with error >= 0.5 go in a separate **part F** job (the main job is
  never put at risk by a rejected circuit). Their use is recorded at `prepare`.
- **Shots:** 4,000 for n >= 96, 2,000 otherwise; one repeat, arms interleaved; dynamical decoupling and twirling off.
- **QPU:** 54 circuits, about 132,000 shots. HW2's rough rule gives 120 s; HW1's ratio (0.40) gives about 50 s.
  `prepare` refuses more than 150 s rough.

## 5. Measures

- **S:** the share of shots equal to the ideal bitstring; standard error sqrt(S (1 - S) / shots). A circuit is
  **resolved** if S >= 5 standard errors (2^-n is negligible from n = 12).
- **Hamming profile:** the distribution of the Hamming distance to the ideal bitstring (reported).
- **Signal width:** per family and depth, the largest n whose three instances are all resolved.

## 6. Predictions (to be fixed with the code before any run)

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| C1 | signal width: PSF >= QK3 in each of L, R d=2, R d=4 | all three | PSF < QK3 in any |
| C2 | where QK3 uses a failed element (part F) and PSF's S is resolved: S_QK3 <= 0.2 x S_PSF | on every such pair (needs at least two) | S_QK3 >= S_PSF on any |
| C3 | where neither uses a failed element: PSF's S >= QK3's on at least half of the pairs | yes | PSF's S < 0.8 x QK3's on more than half |
| C4 | the fixed predictor (ESP x idle term, section 3) picks the device's winner on at least 2 of 3 pairs whose predicted ratio is outside 0.9-1.1 | yes (at least three such pairs) | fewer than half |
| C5 | reported: per arm, the largest resolved circuit (qubits, CZ count, duration); S / predicted per circuit; Hamming profiles | - | - |

**Reading, fixed now.** The README may say "PSF-Zero's compilation keeps a measurable signal on N-qubit,
K-CZ circuits on <device> where Qiskit level 3 given the same Target does not" only if C1 and C2 are confirmed, with
the device, date and the fact that the circuits are Clifford mirror circuits. C3 decides whether anything is said
about circuits that avoid failed elements on both sides. Nothing about speed or advantage is claimed.

## 7. Risks stated in advance

- The live Target's failed elements change daily; part F may be empty, and then C2 is not testable.
- GHZ chains at 48 qubits may already be below resolution without dynamical decoupling (pilot estimate about 0.07).
- Mirror circuits with a Pauli middle are not randomized compiling; coherent errors can cancel in the mirror and make
  S look better than a non-mirror circuit of the same size would be.
