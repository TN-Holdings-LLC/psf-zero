# Workplace exploration (not pre-registered, not a test): the candidates on ecr (Eagle) devices. The release candidate c13 works there and is level with or ahead of Qiskit level 3 (0.67-1.00). The AI front end a9/a10 returned ecr gates in the unsupported direction (224-346 per 56 circuits), which Aer's noise model leaves noiseless, so a10 looked 0.48-0.76 of level 3 when it was not. Candidate a11 fixes the direction (0 off-target, tests 7/7); the honest figure is 0.65-0.99 of level 3 (2026-10-05)

**Status: exploratory.** No predictions, nothing locked.

- **Setting:** workplace sandbox, Qiskit 2.5.2, Aer 0.17.2, core 2026-09-29.1.
- **Why:** Addendum 293 found reported gate errors below the T1/T2 floor most often on ecr devices (22% pooled), but
  no compiler test had used one.
- **Scripts:** `ecr_explore.py`, `summarize.py`, `probe1.py`, `offt.py`, `quick_a11.py`.

## 1. Design

**Devices:** FakeBrussels, FakeStrasbourg, FakeOsaka, FakeSherbrooke (127 qubits; ecr, one direction per coupler).

| device | ecr gates below the T1/T2 floor |
|---|---|
| FakeBrussels | 51 of 138 |
| FakeStrasbourg | 50 of 142 |
| FakeOsaka | 52 of 137 |
| FakeSherbrooke | 26 of 135 |

**Circuits:** 56 per device, all measured.

- 48 from MODEL-RO2's generator, with new seeds (72,000,000 +): 3 per (family, n).
- 8 classifier circuits (n = 4 and 6, L = 4).

**Arms:**

| arm | what it is |
|---|---|
| RPSF | c13, `target` + `placement_refine` |
| C13 | c13, the recommended call |
| A10 | first pass (`out_a10/`) |
| A11 | second pass |
| L3TM | Qiskit level 3 with the Target |

**Recorded:** exactness (state infidelity), off-target instructions, failed-direction uses, the summed measure error,
the classical infidelity of the sampled distribution (MODEL-RO2's metric), the two-qubit count and the compile time.

## 2. What was found

**1. The release candidate c13 works on ecr devices.**

- Exact (max state infidelity 2e-15), 0 off-target, 0 failed-direction uses.
- Classical infidelity C13 / L3TM:

  | device | C13 / L3TM |
  |---|---|
  | FakeBrussels | 0.986 |
  | FakeStrasbourg | 1.000 |
  | FakeOsaka | **0.666** |
  | FakeSherbrooke | 0.964 |

- **FakeOsaka:** L3TM is worse than C13 in 68% of circuits (family ratios down to 0.33 on Dicke states). L3TM places on
  the lowest *reported* errors. On FakeOsaka those are often below the T1/T2 floor, and Aer applies the floor. This is
  Addendum 293's pattern acting on placement, on the device class where it is most frequent.

**2. A defect in the AI front end (a9, a10; inherited from the state-aware re-placement of a4/a5).**

- It returned ecr gates in the direction the device does not provide: 224 (FakeBrussels), 346 (FakeStrasbourg) and
  345 (FakeOsaka) per 56 circuits.
- **The causes:**
  - the re-placement relabels qubits on an undirected coupling graph;
  - the estimate looks a gate up in either direction.
- **The consequences:**
  - such a circuit cannot run on the device as compiled;
  - Aer's noise model has no entry for the reversed ecr, so the simulation treats it as error-free;
  - a10 therefore looked 0.48-0.76 of level 3 on these devices, which is an artefact.
- cz devices and the cx devices tested so far are bidirectional in their Targets, which is why no earlier test
  (HOLD-HOLD6, MODEL-RO, MODEL-RO2) saw it.

**3. The fix, candidate a11 (item 16):**

- the estimate charges an unsupported direction as infinite;
- a final backstop: Qiskit's GateDirection with the target, kept only if it acts as the original (release item 39's
  `_same_action`) and is on target; otherwise the release's recommended call.

Tests (`test_a11_direction.py`, 7 cases, all passed in 17 s) check:

- the version;
- that the estimate rejects the reversed direction and is unchanged for the supported one;
- that outputs on FakeBrussels and FakeOsaka are on target and exact, with and without measurements;
- that on FakeTorino and FakeAuckland a11 equals a10 instruction by instruction;
- that the backstop fixes a reversed gate exactly without calling the release.

On 14 circuits per device, a11's backstop was never needed: the estimate alone avoided every reversed gate.

**4. The honest comparison, a11 against level 3:**

| device | A11 / L3TM | per circuit A11 <= L3TM | A11 / C13 |
|---|---|---|---|
| FakeBrussels | 0.970 | 82% | 0.984 |
| FakeStrasbourg | 0.993 | 75% | 0.992 |
| FakeOsaka | 0.653 | 88% | 0.981 |
| FakeSherbrooke | 0.917 | 86% | 0.951 |

- 0 off-target and exact on all four devices.
- The compile time is a median 0.74-0.80 s, against c13's 0.22 s and level 3's 0.02 s.

## 3. Reading

- **On ecr devices, the floor-aware choices pay** where reported errors are least trustworthy (FakeOsaka: a third less
  infidelity than level 3). Elsewhere they are level or a few per cent ahead. This is the first measurement of
  Addendum 293's prediction in a compiler comparison. It is in simulation, where Aer applies the floor by construction.
- **The AI front end must not be used on ecr devices before a11 or an equivalent fix:** a9/a10 outputs there are not
  executable as compiled.
- **Simulation alone can hide an off-target instruction,** because the noise model has no error for it. Every
  harness should count off-target instructions (HOLD-HOLD6 do; MODEL-RO/RO2 did not).

## 4. Proposals (adoption is the owner's)

1. **a11 as the AI front end candidate**, replacing a10 in the line a9 → a10 → a11. It is identical to a10 on
   bidirectional devices (tested).
2. **A pre-registered test on ecr devices** (c13, a11, level 3; fresh circuits; off-target counted; exactness as a state
   infidelity). This exploration is its pilot.
3. **The off-target count in MODEL-RO-type harnesses.**

## 5. Files (`improve/ecr/` in the handoff)

- the scripts;
- `out/` (a11 pass) and `out_a10/` (a10 pass), with one JSON per device;
- `summary_a11.txt`, `summary_a10.txt`;
- candidate a11 (`psf_ai_compile.py`) and `test_a11_direction.py`.
