# W5 (2026-10-14): placement decided by structure only -- design and offline prototype on ibm_marrakesh's calibration and SCALE1's circuits

**Status: design and prototype (`w5_place.py`); offline, no IBM connection.** Workplace, 2026-10-14; times from
`TZ=Europe/Zurich date`. Rules followed: no weights (no readout, T1/T2 or gate-error weighting), no search over
placements, no seed variation, no ESP or simulation used to choose; measured: compiles or not, failed elements used,
correctness, CZ count, duration, determinism.

## 1. The rule

1. **Cut.** Remove every qubit whose reported sx or readout error is >= 0.5, and every coupler whose two-qubit error
   is >= 0.5; keep the largest connected component K of what is left (ties: smallest qubit index). This happens
   before anything is placed, so no later step can see a failed element.
2. **Snake.** Heron r2's heavy-hex is read from the coupling map (8 rows of 16 consecutive qubits; 4 bridge qubits
   between each pair of rows, at columns 3/7/11/15 below even rows and 1/5/9/13 below odd rows). The snake is the
   fixed path: row 0 left to right, the bridge at the right end, row 1 right to left as far as the leftmost bridge,
   down, and so on. It has 129 qubits (the first qubit of rows 1-6 and the inner bridges are off it).
3. **Place.**
   - A chain circuit (its two-qubit gates form one path) goes on the longest snake piece that survives in K, centred
     (start = floor((length - n) / 2)), along the chain.
   - Any other circuit, or a chain longer than every piece: the qubits of K in breadth-first order from the first
     qubit of the longest piece, neighbours in snake order (variant **A**).
   - Variant **B** (one more fixed step): if the circuit's interaction graph embeds in K with every two-qubit gate on
     a coupler, the first embedding returned by rustworkx's VF2 with nodes in index order (`id_order=True`) is used,
     with no score; otherwise as A. This is an exact graph match, not a search scored by errors; whether it counts
     as "search" under the new rule is home's call.
4. **Route.** Qiskit's level-1 pass manager on K's coupling map only, the layout from step 3, `seed_transpiler=0`,
   once.

## 2. Prototype on the calibration of 2026-10-13 11:37 (the only snapshot saved; SCALE1 compiled at 12:22)

- Cut: failed qubits 82, 94, 113, 130; failed couplers (82,83), (95,99), (112,113), (113,114), (113,119),
  (129,130), (130,131). Components left: 148, 3 and 1 qubits; **K has 148 qubits**, so a 144-qubit circuit fits.
- Snake pieces surviving in K: 66, 22, 22, 11 qubits.
- All 27 SCALE1 logical circuits (L n = 12/24/48; R d=2 n = 24/48/96/144; R d=4 n = 24/48; 3 instances), both
  variants:

| check | result |
|---|---|
| compiled (no rejection) | **54 of 54**, including R2 n=144 i0 and i2, which PSF refused on the day (`FailedElementsError`, item 58) |
| failed elements used | 0 in every circuit (QK3 used 7 and 2 on those two circuits, 13 on i1) |
| correct (stabilizer simulation of the compiled circuit, read at the final layout, equals the ideal bitstring) | 54 of 54 |
| same output on a second call | 54 of 54 |

CZ count and duration against QK3's circuit of the day (ratio = this rule / QK3; QK3 numbers from SCALE1's manifest):

| family | variant A: CZ, duration | variant B: CZ, duration |
|---|---|---|
| L (GHZ chains) | 1.00, 0.98-1.01 | 1.00, 0.98-1.01 |
| R d=2 (random pairs, mostly embeddable) | **6.3-18x, 7.8-47x** | 1.00 on 11 of 12 (incl. both refused 144-qubit circuits: 140 CZ / 0.51 us and 130 CZ / 0.51 us); n=144 i1 has no exact embedding and falls back to A: 3,018 CZ against 261 |
| R d=4 (needs routing) | 2.5-3.7x, 1.9-4.0x | the same as A (no exact embedding) |

## 3. What this says

- **The rejections at 144 qubits are avoided by step 1 alone.** On that calibration 148 healthy, connected qubits
  remain after the cut; the release refused because it pruned failed couplers but still let the placement use failed
  qubits (item 58's message). Cutting failed qubits and couplers first, before placement, is a rule and needs no
  estimate (home's item 5).
- **Chains:** the snake interval is as good as Qiskit level 3 and PSF on the GHZ chains (CZ equal, duration within
  2%), deterministic and weight-free.
- **Non-chain circuits:** a fixed order with no weights and no matching (variant A) is unusable (up to 18x the CZ).
  With an exact-embedding step (variant B) the rule matches QK3 wherever an exact embedding exists, but where the
  circuit needs routing (R d=4, R2 n=144 i1) the fallback gives 2.5-13x the CZ of PSF's and QK3's own routing.
  Structure-only placement therefore covers chains and embeddable circuits; for circuits that need SWAPs it does not
  yet compete, and this prototype does not offer a constructive alternative there.
- **R2 n=96 i0 (the unexplained low signal):** variant B places it on other qubits (98 CZ, 0.48 us, no failed
  element); whether that changes the device's result cannot be decided offline (W4 was cancelled; not analysed).

## 4. Limits

- One calibration snapshot (11:37), 45 minutes before SCALE1's compile; SCALE1's log reports 31 elements with error
  >= 0.5 at 12:22, counted per instruction and direction, which the snapshot's 4 qubits and 7 couplers are consistent
  with but not proven equal to.
- Routing is Qiskit's level-1 SABRE with one fixed seed: deterministic, but a heuristic router.
- The snake is derived for Heron r2's layout (rows of 16); other lattices need their own derivation (the code reads
  rows from the coupling map and asserts that the path is connected).

## 5. Files

`w5_place.py` (the rule and the checks), `run/w5_rows.json` (every circuit, both variants, with SCALE1's PSF/QK3/QK3F
numbers beside them).
