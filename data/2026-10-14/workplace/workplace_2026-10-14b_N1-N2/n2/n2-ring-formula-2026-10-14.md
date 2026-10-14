# N2 (2026-10-14): CZ rings on heavy-hex by formula -- shorter than a14's search, a15, c48 and Qiskit level 3 on the 4- and 6-qubit classifiers

**Status: design and prototype (`n2_ring.py`); offline only.** Workplace, 2026-10-14 (times from
`TZ=Europe/Zurich date`). No search, no score: one fixed construction per ring size. Measured: correctness, CZ count,
duration.

## 1. Why the 4-qubit ring loses on heavy-hex

Heavy-hex has no cycle shorter than 12 (its girth is 12). Any 4 or 6 qubits therefore induce a forest, with at most
n - 1 couplers among them, so at least one edge of an n-ring is never adjacent in a fixed placement. Every ring layer
needs at least one SWAP, or a CZ fused with a SWAP (3 CZ). This holds for every placement, so no layout search can
avoid it; searches only differ in how they route. A 12-qubit ring embeds exactly in one hexagon (W5's variant B finds
it).

## 2. The formula

1. **Folded order on a line of n qubits:** positions hold 0, n-1, 1, n-2, 2, ... . The ring edges (0, n-1) and the
   middle edge are adjacent; every other ring edge spans distance 2.
2. **One ring layer = rounds.** Each round: a CZ on every pending ring edge whose qubits are adjacent (the CZs are
   diagonal and commute, so their order is free), then, while edges remain, one odd-even transposition round of
   SWAPs on the interior positions 1..n-2 ((1,2), (3,4), ... in odd rounds; (2,3), (4,5), ... in even rounds).
3. **Mirror.** The next layer runs the rounds backwards: it undoes each round's SWAPs, then does that round's CZs.
   Every layer costs the same, and the qubits are back in the folded order after every second layer, so odd and even
   layer counts behave alike.
4. The rotations of each layer act on the logical qubits wherever they are. The line is placed by item 86's rule
   (W5's cut, the snake, the common gate length first).

| n | per layer | CZ per layer | CZ-depth per layer | L layers |
|---|---|---|---|---|
| 4 | CZ(0,3) CZ(1,2) / SWAP(pos 1,2) / CZ(0,1) CZ(3,2) | 4 + 1 SWAP = **7** | 1 + 3 + 1 = 5 | 7 L |
| 6 | 2 CZ / 2 SWAPs / 2 CZ / 1 SWAP / 2 CZ | 6 + 3 SWAPs = **15** | 1 + 3 + 1 + 3 + 1 = 9 | 15 L |

Lower bound from section 1: at least n + 2 CZ per layer (one fused CZ-SWAP). The formula gives n + 3 for n = 4 and
n + 9 for n = 6. Whether the bound can be reached for n = 6 was not settled.

## 3. Results on W3's circuits (same angles as W3; rows in `run/n2_rows.jsonl` and `a15/a15_rows.jsonl`)

Durations in us (CZ in brackets). a15 marked * returned a14's searched circuit (its item 27).

| device, n | L | N2 formula | a15 | a14 (search) | c48 PSF-C (W3) | QK3 (W3) |
|---|---|---|---|---|---|---|
| FakeFez, 4 | 2 | **1.18** (14) | 1.34 (18)* | 1.34 (18) | 1.74 (15) | 1.84 (17) |
| FakeFez, 4 | 8 | **4.58** (56) | 6.38 (90)* | 6.38 (90) | 6.77 (60) | 8.20 (77) |
| FakeFez, 4 | 16 | **9.36** (112) | 13.10 (186)* | 13.10 (186) | 13.42 (120) | 16.76 (157) |
| FakeFez, 6 | 8 | **7.22** (120) | 9.54 (142) | 15.07 (142) | 15.10 (142) | 15.07 (142) |
| FakeFez, 6 | 10 | **9.07** (150) | 11.63 (176) | 18.24 (179) | 12.44 (178) | 18.84 (179) |
| FakeFez, 6 | 16 | **14.52** (240) | 18.30 (278) | 29.23 (288) | 19.06 (278) | 30.44 (286) |
| FakeTorino, 4 | 16 | **8.96** (112) | 12.99 (120) | 15.00 (157) | 15.67 (157) | 15.48 (157) |
| FakeTorino, 6 | 16 | **13.73** (240) | 15.78 (314)* | 15.78 (314) | 18.93 (274) | 27.81 (289) |

- **Over all 64 cells (2 devices x 2 sizes x 16 depths):**
  - N2 is the shortest in 61.
  - The 3 exceptions are all L = 1, where the ring layer is not repeated: by 0.07-0.08 us at n = 4 on both devices
    (a14's 6 CZ against the formula's 7) and by 0.02 us at n = 6 on FakeFez.
  - At L = 16 it is 0.71x (n = 4) and 0.79x (n = 6) of the best other method on FakeFez, and 0.69x and 0.87x on
    FakeTorino.
- **All outputs are exact** (total variation distance at most 7e-14) and ISA.
- **The CZ count is exactly 7L (n = 4) and 15L (n = 6),** as the formula says. The gap to every other method grows
  linearly with L.
- **No parity effect:** the formula is equally good at odd and even L. W3's even/odd difference belonged to c48 (N3).
- **a15 picked a14's search** on most 4-qubit cells, as home found. The formula beats that search by 12% at L = 2
  and 29% at L = 16 on FakeFez.

## 4. Explaining W3's 6-qubit step at L = 10

Not explained here. N2's per-layer cost is constant, so it shows no step. c48 PSF-C jumps from about QK3's duration
to 0.66 of it between L = 8 and L = 10, which is the size at which c48 switches its construction. That belongs to
N3: the per-layer log of item 82's mirror decision.

## 5. What is not covered

- Rings of other lengths (7-11) were not derived.
- The formula assumes the ring's CZs commute with each other, which holds for CZ and any diagonal gate, and that each
  layer's rotations sit between ring layers (DEPTH's model). A ring of non-commuting two-qubit blocks would need its
  own order.
- A placement other than a line (for example around a degree-3 qubit) was considered for n = 4. It costs the same
  7 CZ but has a deeper CZ schedule (7 instead of 5), so the line was kept.

## 6. Files

`n2_ring.py` (the formula and the check), `n2_a15.py` (a14/a15 on the same circuits), `run/` and `a15/` (rows).
