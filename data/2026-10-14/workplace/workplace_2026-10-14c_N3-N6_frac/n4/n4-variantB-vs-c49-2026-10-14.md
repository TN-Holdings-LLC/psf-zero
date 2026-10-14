# N4 (2026-10-14): W5's variant B against c49's default call

**Status: offline, done.** Script `n4_variantB.py`, rows `n4/run/n4_rows.jsonl`.

- **Variant B:** cut the failed elements; take the first exact embedding from VF2 (no score), otherwise a
  breadth-first order along the snake; route with Qiskit level 1 on the cut lattice (coupling map and basis only).
- **c49's numbers:**
  - C49-VAL: read from C49-VAL's own rows (arm C49, the call `entangling_basis="cx", layout_search=True`).
  - SCALE1: compiled here with that same call on the marrakesh snapshot.

## C49-VAL (18 tests x 5 devices = 90 cells)

| check | result |
|---|---|
| refused | 0 (and c49 0) |
| deterministic (two calls) | 90 of 90 |
| exact (outcome distribution, 62 cells with <= 20 active qubits) | 62 of 62 (largest error 8e-14) |
| duration at or below c49's | **4 of 90** (median 2.7 x c49) |

- Only ghz_n12 has an exact embedding. There B is level with c49 or shorter (0.82-1.01).
- Every other test needs routing, and B loses:
  - QAOA and BA: 1.3-3.3x in duration.
  - Heisenberg and chemistry: 1.7-6x.
  - TFIM given as a PauliEvolutionGate: 12-130x, because Qiskit level 1 lays out the evolution in its own order. On
    these TFIM chains item 86 (N1) is the formula, not variant B.

## SCALE1 (27 circuits, snapshot)

| circuits | B against c49 (CZ, us) |
|---|---|
| L (chains) n = 12-48, 9 | equal CZ; duration equal or up to 1.5% shorter, one 1.9% longer (L_n48_i2 5.22 vs 5.12) |
| R d=2 n = 24-144 with an exact embedding, 11 | equal CZ except R2_n144_i2 (130 vs 128); duration equal or shorter on 9 (R2 n=96: 0.476/0.464/0.508 vs 0.524/0.488/0.548), longer on 2 (R2_n48_i1 0.500 vs 0.488, R2_n144_i0 0.508 vs 0.488) |
| R2_n144_i1 and R d=4 (need routing), 7 | B loses: 2.5-13x CZ (R2_n144_i1: 3018 vs 261) |

No refusal on either side: c49's item 85 compiles the 144-qubit circuits too, and every B output is correct (known
bitstring) and deterministic.

## Reading

Variant B is a placement rule for circuits that embed exactly; there it matches c49, sometimes 5-10% shorter (R2
n = 96). Where routing is needed it is far behind, because level-1 routing after a fixed order is weak. If B is
adopted, the safe form is:

1. "B when an exact embedding exists";
2. compared with c49's own output by item 84's rule (as item 86 does);
3. c49 everywhere else.

On these sets that would change only ghz_n12 and SCALE1's embeddable circuits.
