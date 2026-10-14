# R3 and R4 (2026-10-14): the Heisenberg gap is not fusion (already done) but the input's serial term order and placement; on FakeTorino item 91 loses because no embedding fits its most common coupler length

Workplace, 2026-10-14, finished 09:11 CEST (`TZ=Europe/Zurich date`). Base: c54. Offline only.

## R3: Heisenberg

**Fusion is already done** (`r3_fuse.py`).

- Each edge's consecutive XX, YY, ZZ terms were replaced by one exact two-qubit `unitary` (exp(-i(aXX + bYY + cZZ)))
  before c54. This changed nothing:

  | test | T | F |
  |---|---|---|
  | heis_n12_r2 | 16.57 → 16.50 µs (135 → 135 two-qubit gates) | 14.44 → 14.41 |
  | heis2d_n16_r1 | 22.27 → 22.15 (429 → 429) | 20.47 → 20.50 |

- Level 3's block consolidation (ConsolidateBlocks / peephole, inside c54's paths) already merges the three terms into
  one 3-CZ block.

**What makes these circuits long.** gens writes the terms edge by edge (for each edge: XX, YY, ZZ).

- XX(a,b) and YY(b,c) anticommute on b. So the product formula as written chains every edge to the next — a serial
  staircase, 16.6 µs for a 12-ring.
- Every exact compiler must keep that order: c54, item 91 and the user alike.
- C54-VAL's heavy-hex-tree Heisenberg (5.508 µs, equal to c53) is the same: item 91 embeds the tree, but the levels force
  the staircase, so the colour layers hold one edge each.
- The user's 19.69 µs on heis2d (T) is a seed-searched placement: QK3 at seed 0 gives 22.67 as written and 25.20
  hand-layered, against c54's 22.27.

**What would help (not an exact rewrite, so not done):** listing the terms by colour (all XX of one colour, …) or by
Pauli type gives a different product formula, with the same order and the same Trotter error bound but a different
circuit. That is the input writer's choice, not the compiler's.

**Within the rules:** nothing found. The torus has degree 4, so no embedding; a fixed SWAP schedule would need the same
SWAP count the router already pays.

## R4: item 91 on FakeTorino

**Why nothing changed** (`r4_torino.py`).

- FakeTorino's CZ lengths are mixed: 68 ns on 216 couplers; the rest 72-184 ns (80, 88, 104, 108, 116, 120, 160, 184 …).
- **No embedding of hhI_n16, hhI_n20 or ring12 exists on the 68 ns couplers alone** (`common_length` + 0). So item 91's
  VF2 falls back to all couplers. Its first match uses slow couplers:

  | output | 68 ns | 80 ns | 108 ns | 160 ns | 184 ns |
  |---|---|---|---|---|---|
  | item 91, hhI_n16 F | 57 | 5 | 4 | — | 5 |
  | item 91, ring12 F | 36 | 4 | 5 | 4 | 5 |
  | c53, hhI_n16 F | 70 | — | — | — | — (plus 5 at 88 ns) |
  | c53, ring12 F | 55 | — | 5 | — | — |

  c53's placement (item 90: item 75 scored by gate durations) avoids the slow couplers.
- **Item 91's output was built, but the comparison kept c53's:**

  | test | target | item 91 | c53 |
  |---|---|---|---|
  | hhI_n16 | F | 1.768 | 1.520 µs |
  | hhI_n16 | T | 4.400 | 3.280 µs |
  | ring12 | F | 1.548 | 1.136 µs |
  | ring12 | T | 3.440 | 1.856 µs |

  `GRAPH_STATS["used"]` counts built outputs, not returned ones; `CHAIN_STATS["kept_default"]` counts these.
- **hhI_n20 F is the exception:** item 91 at 1.468 against c53's 1.756, and item 91 is returned.
- **A fixed tier rule did not help** (`r4_tier.py`): couplers ≤ 1.25 × the common length before all couplers. On
  FakeTorino no embedding exists in that tier either. On the other devices it changed nothing, or made things slightly
  worse.
- **On T (CZ), item 91 loses on 12-rings on every device** (FakeFez 2.40 against 1.61). STEINER6 shares CX between
  commuting ZZ terms there, and item 87/91's rzz-to-2-CZ emission does not.

**Workplace 0.62-0.74 against home's 0.80-0.86.** These measure different cells, as home guessed.

- My means include patches of 80 qubits and the whole device, where c53 fell back to Qiskit's output (ratios
  0.23-0.32).
- On comparable sizes my F ratios match home's:

  | cell | FakeFez | FakeMarrakesh |
  |---|---|---|
  | n = 18, 2 steps | 0.79 | 0.76 |
  | n = 18, 4 steps | 0.91 | — |
  | n = 40, 2 steps | 0.80 | — |

- Home's set adds FakeTorino, where item 91 loses for the reason above, and a Heisenberg tree, which shows no effect
  (R3).

## Files

`r3_fuse.py`; `r4_torino.py`, `r4_tier.py`.
