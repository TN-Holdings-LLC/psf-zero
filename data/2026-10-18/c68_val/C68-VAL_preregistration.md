# Pre-registration: C68-VAL, the validation of prototype c68 (item 105: chains and rings by psfz r4) (2026-10-18)

**Status: pre-registration.** Fixed by its SHA-256, given to the owner in the conversation before the run, with the
time in CEST; committed before the run and pushed with its results. Nothing runs on a device. The design was proposed
by the authors and confirmed by the owner before it was built (2026-10-18, about 04:01 CEST): 16 devices, 25 tests,
4,400 inputs, criteria G1-G6, the authors' cloud. The owner's earlier decisions (about 03:59 CEST): psfz used
automatically for chains and rings, its output as it is, a prototype and a pre-registered validation before a release.

**What is tested:** [`patches/psf_compile_c68_2026-10-18/psf_compile.py`](../../../patches/psf_compile_c68_2026-10-18/)
(`2026-10-18.c68`, commit `254b3de`) = release 2026-10-17.1 + item 105: the default call hands the circuit to psfz r4
(`rust/psfz/` as of Addendum 521, validated in PSFZ-VAL2) and returns its output when psfz compiles it; every other
call is the release's path. The reference is release 2026-10-17.1 (`psf_compile.py`).

**Where and how:** the authors' cloud (2 CPUs; Qiskit 2.5.2, qiskit-ibm-runtime 0.49.0), `run_c68_val.sh` from the
PSF-Zero root: psfz is built from its sources and `PSFZ_BIN` set to it (SHA-256 of the sources, the program, both
compilers and these files recorded), then two processes.

- **Devices:** PSFZ-VAL2's 16 -- FakeAachen, FakeBoston, FakeKingston, FakePittsburgh, FakeFez, FakeMarrakesh,
  FakeTorino (Heron, CZ); FakeBrussels, FakeStrasbourg, FakeCusco, FakeKyiv, FakeOsaka, FakeKawasaki, FakeQuebec,
  FakeSherbrooke, FakeBrisbane (Eagle, ECR).
- **Tests:** C63-VAL's 19 (12 formula tests, among them three heavy-hex graphs that are not chains or rings, and 7
  native-gate tests, two of them heavy-hex graphs) and PSFZ-VAL's 6 of new sizes: 25.
- **Inputs:** per device and test the build input, 5 same-sign and 5 free draws (C63-VAL's `randomize`,
  `random.Random(f"C68-VAL|{device}|{test}")`): 16 x 25 x 11 = **4,400**. None is left out: an input refused by the
  release is kept (G2 asks c68 to refuse it the same way unless psfz compiles it; see G4).
- **Per input** (`c68_val.py`): c68's default call twice, and whether it used psfz (`PSFZ105_STATS`); psfz's program
  on the same input directly (as PSFZ-VAL2); the release's default call; c68's default call with `PSFZ_BIN` pointing
  to no program.

## Criteria

| ID | criterion |
|---|---|
| G1 | c68 uses psfz exactly on the inputs psfz's program compiles, and there its output is psfz's own (the same operations, angles and qubits, and a layout equal to psfz's). |
| G2 | On every other input c68's output is the release's, bit for bit (operations, angles as a float's hex, global phase, layouts), or c68 refuses with the release's exception and message. |
| G3 | Every output c68 took from psfz is native, uses no failed element (error 0.5 or more), and is exact where checked: outcome distribution within 1e-6 (total variation) of the input's (a Pauli evolution taken as its Trotter circuit); checked on every such output up to 16 active qubits, from 17 to 20 the build input and the first two draws. |
| G4 | **(the owner's bar)** No output of c68 is longer than the release's on the same input (duration as soon as possible with the device's durations, measurements included; equal within 1e-9 is not longer), and c68 never refuses an input the release compiles. An input the release refuses and c68 compiles is counted, not a failure. |
| G5 | With no psfz program, c68's output (or refusal) is the release's on every input, as in G2. |
| G6 | The same input gives the same output twice (G2's comparison), or the same refusal. |

**Reported, not criteria:** on how many inputs psfz was used; there the duration over the release's (display:
geometric mean; essential: inputs shorter and equal), CZ and ECR devices separately, per test; compile times (c68 also
pays for writing the circuit as OpenQASM 3 and starting psfz's program, and on inputs psfz refuses it then runs the
release's path as well).

**The authors' expectation, stated before the run:** G1-G6 confirmed. psfz should take the 20 chain and ring tests
(its PSFZ-VAL2 tests, except FakeCusco's two longest rings, which both compilers refuse) and refuse the five heavy-hex
graph tests, which then go to the release's path unchanged. A risk for G2/G5 would be a release output that depends
on something c68's wrapper changes (it changes nothing but the order of calls).

**Disclosure:** the harness was checked before the run on other draws (`C68_SMOKE=1`: salt `C68-SMOKE`, the first two
draws, three tests, on FakeFez and FakeSherbrooke; 12 inputs, every check passed); c68 itself was tried on the build
inputs of FakeTorino's Ising chain of 24, FakeQuebec's GHZ of 12, FakeFez's Heisenberg ring of 12 and a 4-qubit star
(Addendum 524 when written). No draw of this run was computed before it.

**Interruption:** a run that stops is started again into a new folder; the first complete one is scored.
