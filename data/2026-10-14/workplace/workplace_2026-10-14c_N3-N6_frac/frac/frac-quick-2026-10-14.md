# Quick check (2026-10-14): rx as one pulse, native rzz, and rzz signs in a Pauli frame -- effect on duration

Workplace, offline, a few minutes of computation (`frac_quick.py`). Circuits built by formula on a straight line of
the device (first row and bridge): TFIM brickwork (N1's form), a Heisenberg XXX chain, DEPTH's classifier with N2's
ring. Durations as soon as possible: cz and rzz the coupler's cz length (FakeFez 84 ns, FakeMarrakesh 68 ns; on
ibm_marrakesh rzz = cz = 68 ns), rx = sx length (one pulse), rz 0. rzz only with 0 < theta <= pi/2: angles reduced
mod 2 pi, |theta| > pi/2 folded with Z x Z (free rz), negative angles either with x on both sides, or with one x
before (merged into the preceding single-qubit run) and the other kept in a Pauli frame (undone by one x per qubit
before measurement, which a classical bit flip could replace). Every output checked exact (outcome distribution;
largest error 5e-14).

Ratio to today's form (CZ, single-qubit gates as two sx pulses):

| circuit | (1) rx one pulse | (1)+(3) rzz, x both sides | (1)+(2)+(3) rzz + Pauli frame |
|---|---|---|---|
| TFIM J = +1, n=12, 4 steps | 0.90 / 0.86 | 0.39 / 0.34 | 0.39 / 0.34 |
| TFIM J = -1, n=12, 4 steps | 0.90 / 0.86 | 0.46 / 0.44 | 0.41 / 0.37 |
| TFIM random-sign J, n=16, 10 steps | 0.90 / 0.87 | 0.45 / 0.41 | 0.42 / 0.38 |
| Heisenberg XXX, n=12, 6 steps | 0.90 / 0.85 | 0.46 / 0.45 | 0.46 / 0.45 |
| classifier (N2 ring), n=4, L=16 | 0.96 / 0.93 | 0.96 / 0.93 (CZ ring: rzz(pi/2) = cz time) | same |
| classifier (N2 ring), n=6, L=16 | 0.97 / 0.96 | same | same |

(FakeFez / FakeMarrakesh.) Absolute examples on FakeMarrakesh: TFIM n=12, 4 steps 2.60 -> 0.88 us; random-sign
n=16, 10 steps 5.93 -> 2.23 us; Heisenberg n=12, 6 steps 8.86 -> 3.96 us.

Reading: (3) native rzz is the large effect on chains (0.34-0.46x), including Heisenberg when it is built as a
brickwork (home's c50 measured 1.05x on c49's heis_n12 output: the slowdown there is not rzz itself). (2) matters
only when angles are negative: it recovers most of the loss (0.44 -> 0.37). (1) alone gives 10-15% on chains and
4-7% on the classifier; it is larger on FakeMarrakesh because its sx is 36 ns. On the classifier rzz adds nothing
(the ring is CZ); only (1) acts. Not checked: hardware accuracy of rzz and of arbitrary-angle rx pulses.
