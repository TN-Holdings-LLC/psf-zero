# Pre-registration: C67-VAL, the validation of prototype c67 (item 104: ECR devices by a CZ view) (2026-10-17)

**Status: pre-registration.** Fixed by its SHA-256, given to the owner in the conversation before the run, with the
time in CEST; committed before the run's results and pushed with them. Nothing runs on a device. The design was
proposed by the authors and confirmed by the owner before it was built (2026-10-17, about 07:44 CEST).

**What is tested:** [`patches/psf_compile_c67_2026-10-17/psf_compile.py`](../../../patches/psf_compile_c67_2026-10-17/)
(`2026-10-17.c67`) = release 2026-10-16.2 + item 104 (Addendum 515), its default call `compile_for_hardware` on
devices whose two-qubit gate is ECR. Its templates on ECR devices are not covered (refused); on CZ devices c67 was
found equal to 2026-10-16.2, bit for bit, on C63-VAL's inputs (Addendum 515), and is not tested again here.

**Where and how** (the authors' cloud, 2 CPUs; Qiskit 2.5.2, qiskit-ibm-runtime 0.49.0): `run_c67_val.sh`, from the
PSF-Zero root, two processes of two devices each; `c67_val.py` writes one line per input; `score_c67_val.py` scores.

- **Devices:** FakeSherbrooke, FakeBrisbane, FakeKyiv, FakeOsaka (IBM Eagle, 127 qubits, ECR).
- **Inputs:** C63-VAL's 19 tests (12 formula tests and 7 native-gate tests, as `c63_val.py` builds them); per device
  and test the build input, then 25 same-sign and 25 free draws by C63-VAL's `randomize` with
  `random.Random(f"C67|{device}|{test}")`: 19 x 4 x 51 = 3,876 inputs.
- **Per input:** c67's call twice; the output's checks; its two-qubit count and duration (as soon as possible with the
  device's durations, measurements included) and compile time; Qiskit level 3 (seed 0) on the same input.

## Criteria

| ID | criterion |
|---|---|
| E1 | Every one of the 3,876 inputs is compiled (no refusal, no error). |
| E2 | Every output is native to the device (every operation on qubits and couplers the device offers) and uses no failed element (item 31's test at error 0.5). |
| E3 | Every output with at most 20 active qubits is exact: its outcome distribution within 1e-6 (total variation) of the input's (a Pauli evolution's input taken as its Trotter circuit). tfim_n40_r2 is not checked. |
| E4 | The same input gives the same output twice, bit for bit (every operation, angle, global phase and layout). |
| E5 | No output's duration exceeds 1.10 x Qiskit level 3's on the same input. |

**Reported, not criteria:** the duration and two-qubit count over Qiskit level 3's (display: the geometric mean;
essential: in how many inputs c67 is longer, and the largest ratio), per test; the compile time of both.

**The authors' expectation, stated before the run** (from the exploratory measurement on the build inputs, Addendum
515: 76 of 76 compiled, exact, native, at most 1.072 x level 3): E1-E4 confirmed; E5 likely, but the draws are new and
GHZ came close to the line (1.072 against 1.10).

**Not checked:** exactness above 20 active qubits; real devices; templates on ECR devices.

**Interruption:** a run that stops is started again into a new folder; the first complete one is scored.
