# Pre-registration: PSFZ-VAL2, the validation of psfz r4 against release 2026-10-17.1 (2026-10-18)

**Status: pre-registration.** Fixed by its SHA-256, given to the owner in the conversation before the run, with the
time in CEST; committed before the run and pushed with its results. Nothing runs on a device. The design was
confirmed by the owner before it was built (2026-10-18, about 03:05 CEST): three unseen Eagles and new
draws on PSFZ-VAL's 13 devices; PSFZ-VAL's criteria and the strict bar unchanged; the authors' cloud.

**What is tested:** `rust/psfz/` at commit `4a5768f` (Addendum 521), `psfz r4`: r3 (Addendum 520) plus (1) chains
placed on the longest row-and-bridge path when the snake has no long enough piece on couplers of the most common
duration, and (2) on ECR devices each CZ written in the one of 64 CZ-on-ECR forms that lets its ECR start earliest.
The reference is release 2026-10-17.1's default call (`psf_compile.py`).

**Where and how:** the authors' cloud (2 CPUs; Qiskit 2.5.2, qiskit-ibm-runtime 0.49.0), `run_psfz_val2.sh` from the
PSF-Zero root: psfz is built from its sources (their SHA-256 and the binary's are recorded), then two processes.

- **Devices:** 3 never used in psfz's development or validation -- FakeBrussels, FakeStrasbourg, FakeCusco (Eagle,
  ECR) -- and PSFZ-VAL's 13 (FakeAachen, FakeBoston, FakeKingston, FakePittsburgh, FakeFez, FakeMarrakesh, FakeTorino:
  Heron, CZ; FakeKyiv, FakeOsaka, FakeKawasaki, FakeQuebec, FakeSherbrooke, FakeBrisbane: Eagle, ECR), all of which r4
  has now been developed on (Addendum 521). Reported separately. No unseen CZ device that psfz can read exists among
  the fake devices (FakeBerlin, FakeMiami, FakeNighthawk, FakePrague are not heavy-hex), so CZ devices are tested on
  new draws only.
- **Tests:** PSFZ-VAL's 20 (C63-VAL's 9 chain and 5 ring tests and 6 of new sizes), the same build inputs.
- **Inputs:** new draws, 5 same-sign and 5 free (C63-VAL's `randomize`, `random.Random(f"PSFZ-VAL2|{device}|{test}")`);
  on the 3 unseen devices also the build input; on the 13 known devices the draws only (their build inputs were seen
  in PSFZ-VAL). **Left out:** FakeCusco's Ising ring of 28 and Ising ring of 40, which psfz and the release both
  refuse on the build input (no working ring of that length; checked before the run, below). Total
  (20 + 20 + 18) x 11 + 13 x 20 x 10 = **3,238** inputs.
- **Per input:** psfz twice on the input written as gate-level OpenQASM 3; the release on the circuit itself.

## Criteria (PSFZ-VAL's, unchanged)

| ID | criterion |
|---|---|
| V1 | Every input is compiled by psfz (no refusal). |
| V2 | Every output is native and uses no failed element (error 0.5 or more). |
| V3 | Every checked output is exact: outcome distribution within 1e-6 (total variation) of the input's (a Pauli evolution taken as its Trotter circuit). Checked: every input up to 16 active qubits; from 17 to 20 the build input and the first two draws; above 20 none. |
| V4 | The same input gives the same output twice. |
| V5 | **(the owner's bar)** No output of psfz is longer than the release's on the same input (duration as soon as possible with the device's durations, measurements included; equal within 1e-9 counts as not longer). An input the release refuses is left out and counted. |
| V6 | psfz's median compile time (its own clock) is below the release's (the call). |

**Reported, not criteria:** the duration and two-qubit count over the release's (display: geometric mean; essential:
how many inputs are longer and by how much), per test, unseen and known devices separately.

**The authors' expectation, stated before the run:** V1-V4 and V6 confirmed. V5 likely confirmed on the 13 known
devices (on PSFZ-VAL's draws r4 was never longer, and the draws change angles, not graphs). On the 3 unseen Eagles V5
is less sure: the ECR form choice is greedy (one gate at a time), and it was tuned on no device in particular but
checked only on known ones; a loss there, if any, would most likely be one pulse on a Heisenberg input.

**Disclosure (what was seen of the unseen devices before the run):** psfz r4 was run on the 20 build inputs of
FakeBrussels, FakeStrasbourg, FakeCusco, FakeKyoto, FakeCairoV2 and FakePeekskill to see which it refuses (Addendum
521); the release was run on the refused ones. psfz's durations on FakeBrussels's and FakeCusco's Ising chain of 12
were printed; the release's durations on these devices were not looked at. No draw of this run was computed before
it.

**Interruption:** a run that stops is started again into a new folder; the first complete one is scored.
