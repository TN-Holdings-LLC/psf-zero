# Pre-registration: PSFZ-VAL, the validation of psfz r3 (the Rust compiler for chains and rings) against release 2026-10-17.1 (2026-10-17)

**Status: pre-registration.** Fixed by its SHA-256, given to the owner in the conversation before the run, with the
time in CEST; committed before the run and pushed with its results. Nothing runs on a device. The design was
confirmed by the owner before it was built (2026-10-17 about 16:20 CEST): unseen and seen devices; the 14 chain and
ring tests plus new sizes; the strict bar; the authors' cloud tonight. After a dry run showed psfz r2 losing on a new
Heisenberg chain, the owner chose to build r3 first (about 16:25 CEST).

**What is tested:** `rust/psfz/` at the commit that adds this file, `psfz r3`: r2 (Addenda 518-519) plus blocks of
several two-qubit rotations on one pair, with the single-qubit operations between them, multiplied into one 4 x 4
matrix and written by a Weyl (KAK) decomposition in closed form (`src/kak.rs`, no outside crate; a block whose
decomposition does not multiply back is written term by term), and a 2-CZ form for blocks of two rotation kinds. The
reference is release 2026-10-17.1's default call (`psf_compile.py`).

**Where and how:** the authors' cloud (2 CPUs; Qiskit 2.5.2, qiskit-ibm-runtime 0.49.0), `run_psfz_val.sh` from the
PSF-Zero root: psfz is built from its sources (their SHA-256 and the binary's are recorded), then two processes.

- **Devices:** 8 never used in psfz's development -- FakeAachen, FakeBoston, FakeKingston, FakePittsburgh (Heron,
  CZ), FakeKyiv, FakeOsaka, FakeKawasaki, FakeQuebec (Eagle, ECR) -- and the 5 used in it (FakeFez, FakeMarrakesh,
  FakeTorino, FakeSherbrooke, FakeBrisbane), with new draws. Reported separately.
- **Tests:** C63-VAL's 9 chain and 5 ring tests, and 6 of new sizes built by `psfz_val.py` (`NEW`): an Ising chain
  of 24, an Ising ring of 28, a Heisenberg chain of 16, a CZ ring of 20, a CX chain of 20, a GHZ chain of 20.
- **Inputs:** per device and test the build input, 5 same-sign and 5 free draws (C63-VAL's `randomize`,
  `random.Random(f"PSFZ-VAL|{device}|{test}")`): 13 x 20 x 11 = 2,860.
- **Per input:** psfz twice on the input written as gate-level OpenQASM 3; the release on the circuit itself.

## Criteria

| ID | criterion |
|---|---|
| V1 | Every input is compiled by psfz (no refusal). |
| V2 | Every output is native and uses no failed element (error 0.5 or more). |
| V3 | Every checked output is exact: outcome distribution within 1e-6 (total variation) of the input's (a Pauli evolution taken as its Trotter circuit). Checked: every input up to 16 active qubits; from 17 to 20 the build input and the first two draws; above 20 none. |
| V4 | The same input gives the same output twice. |
| V5 | **(the owner's bar)** No output of psfz is longer than the release's on the same input (duration as soon as possible with the device's durations, measurements included; equal within 1e-9 counts as not longer). An input the release refuses is left out and counted. |
| V6 | psfz's median compile time (its own clock) is below the release's (the call). |

**Reported, not criteria:** the duration and two-qubit count over the release's (display: geometric mean; essential:
how many inputs are longer and by how much), per test, unseen and seen devices separately.

**The authors' expectation, stated before the run:** V1-V4 and V6 confirmed. **V5 is likely refuted:** on the
development devices r3 is still longer on some Heisenberg inputs on ECR devices (FakeSherbrooke: 2 of 3 dry-run
inputs of the Heisenberg chain, by 0.1-0.2%; FakeBrisbane: 2 of 35 exploratory inputs, by at most 0.23%), where the
release's Weyl decomposition sometimes saves a pulse for particular angles.

**Disclosure:** a dry run of this script with psfz r2 (FakeAachen, the 20 build inputs, no draws) was seen before r3
was written: it showed the Heisenberg chain longer (177 two-qubit gates against 171). r3 was then developed on other
seeds (`PSFZ-DEV`) and the development devices only.

**Interruption:** a run that stops is started again into a new folder; the first complete one is scored.
