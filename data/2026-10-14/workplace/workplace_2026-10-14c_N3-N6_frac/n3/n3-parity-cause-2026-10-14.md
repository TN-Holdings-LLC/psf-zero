# N3 (2026-10-14): the cause of W3's odd/even difference -- item 73's reader, not item 82

**Status: offline, done.** Script `n3_parity.py`, rows `n3/run/n3_rows.jsonl` (64 cells x c48 and c49, with the
change of `STEINER_STATS` during each call). Default call `compile_for_hardware(qc, backend=b, seed_transpiler=0)`.

## Cause

DEPTH's classifier layer is RY, RY, RZ on every qubit and then a ring of CZ. Item 73 reads such a circuit gate by gate,
carries the Clifford gates (the CZs) to the end and turns each rotation into a Pauli term. It applies only when the
Clifford gates left at the end act on each qubit alone (or permute the qubits, item 76).

- **Even L:** the CZ rings pair up and cancel (CZ squared is the identity), nothing is left at the end, and the
  circuit is read (`gate_level`, "used"). STEINER6 builds it, with item 82's mirror tried and item 83's order.
- **Odd L:** one CZ ring is left at the end. It is entangling, so item 73 refuses (`not_applicable` in all 32 odd
  cells, both devices, c48 and c49), and the circuit goes by the usual path, the same as Qiskit level 3.

So the parity pattern is item 73's acceptance rule; item 82's mirror only acts once the circuit is read.

## The 6-qubit step at L = 10 (c48)

At even L <= 8 c48 recorded "kept_usual": its ESP comparison (item 69) preferred the usual path to STEINER6. From
L = 10 it recorded "chosen". c49 chooses by duration (item 84) and takes STEINER6 from L = 2:

| FakeFez, n = 6 | L = 2 | 4 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| c48 (us) | 3.08 kept_usual | 6.97 kept_usual | 11.05 kept_usual | 15.05 kept_usual | 12.44 chosen | 14.63 chosen | 19.06 chosen |
| c49 (us) | **2.57** chosen | **4.74** chosen | **7.76** chosen | **10.00** chosen | 12.89 chosen | 15.06 chosen | 19.31 chosen |

c49 removes the step, but from L = 10 it is 2-4% longer than c48: STEINER6 picks its construction by its own schedule
in CX units (item 84; c48 by an error-weighted estimate), which does not track the device's gate lengths exactly.

## Odd L in c48 and c49

Unchanged on FakeFez: odd L is not read by either, and both return the usual path. On FakeTorino at n = 4, c49 is
shorter at odd L (for example L = 9: 6.80 against 8.68 us; L = 15: 12.41 against 14.36) because item 80 keeps Qiskit
level 3's circuit by duration. All values are in the rows file.

## What would remove the parity (not implemented)

- **The measurement rule:** a diagonal Clifford (a CZ ring) left at the end, followed only by measurements, does not
  change the outcome distribution. Item 73 could drop it when the circuit ends in measurements, as Qiskit's
  RemoveDiagonalGatesBeforeMeasure does. Odd L would then be read like even L.
- **N2's ring formula** is shorter than either at every L (7 L / 15 L CZ), so for this model it supersedes the
  question.
