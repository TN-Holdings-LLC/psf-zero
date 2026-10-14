# Q1 (2026-10-14): item 91 -- an interaction graph that embeds in the device, built by edge colouring (workplace proposal on c53)

**Status: patch proposal with tests; offline only.** Workplace, 2026-10-14, finished 07:36 CEST (`TZ=Europe/Zurich date`).
Request: 「次の指示4」 Q1.

Common rule followed:

- No search, no sweep, no seed variation, and no choice by ESP or simulation.
- The placement is the first VF2 embedding (nodes in index order, no score).
- Measured: correctness, two-qubit count (CZ + rzz), and duration (ASAP on the Target's gate lengths).

## 1. Build

`make_c53_91.py <c53 psf_compile.py> c53_91_code.py <out>` makes exact substitutions on home's c53 (bundle e,
sha256 of the built file in MANIFEST):

- the two VERSION strings ("2026-10-14.c53+91"; home sets its own when merging);
- changelog item 91, after item 90;
- two lines in the item-87 wrapper: if item 87 does not apply, item 91 is tried.

The code is appended at the end of the file. Nothing in c49's body or items 86-90 is touched.

## 2. What item 91 does

It generalises item 87 from a path to any graph heavy-hex contains. It runs under item 87's conditions (device given,
cz, none of the caller's coupling_map, initial_layout, final_resynthesis, compare_level3 or compare_floor) and when
item 87 does not apply.

1. **Read.** Items 72-73's reader, merged where exact. Every term must act on one or two qubits. The two-qubit
   supports form the interaction graph.
2. **Embed.**
   - The graph must have max degree ≤ 3 and be bipartite.
   - rustworkx's VF2 takes the first embedding into the cut device (W5's cut). It tries the couplers of the device's
     most common gate length first, then all couplers, as in item 87. No score is used.
3. **Colour.** A proper edge colouring with max-degree colours (König). It uses deterministic alternating-path
   recolouring, in sorted edge order. Each colour is a matching, at most 3 on heavy-hex.
4. **Order.** Each term keeps item 87's level: 1 + the largest level of an earlier term that shares a qubit and does
   not commute with it. This keeps the result exact. Within a level, two-qubit terms go by colour, one layer per
   colour, then the single-qubit terms.
5. **Emit.** As item 87 (CZ, Qiskit level 3 on the fixed layout) or item 88 (rzz / rx, Pauli frame undone inside the
   circuit).
6. **Compare.** The result is compared with c53's own output by item 87's rule; a tie keeps item 91's output.

**What does not embed (and is left to c53, unchanged).** Heavy-hex has three structural limits:

- **No adjacent degree-3 qubits.** Two degree-3 qubits are never side by side. So random trees of max degree 3
  usually do not embed: **0 of 20** random 16-node trees embedded on FakeFez or FakeMarrakesh, each because of
  adjacent degree-3 nodes.
- **No short cycles.** There is no cycle shorter than 12, so 4-11 cycles do not embed.
- **No degree 4.** C52-VAL's **heis2d_n16_r1** is a 4×4 periodic square lattice: degree 4, with 4-cycles. **It does not
  embed** (`GRAPH_STATS["degree"]`), and its output is identical to c53's.

## 3. Results (`eval/q1_rows_*.jsonl`, 84 cells)

**Setup.**

- Devices: FakeFez and FakeMarrakesh. T = the device's Target; F = with rzz and rx added (test_c51).
- Arms:
  - new = c53 + item 91;
  - c53 and c52, default calls;
  - QK3 / QK3F: Qiskit level 3 given the Pauli evolution (QK3F folded).
- Inputs, all with random signs:
  - kicked-Ising-type ZZ + X on heavy-hex patches of the cut device, n = 18, 40, 80 and the whole cut device (154
    and 153 qubits), 2 and 4 Suzuki steps;
  - the same on BFS spanning trees of patches, n = 16 and 40;
  - an XYZ model on patches, n = 18 and 40, with terms listed as all XX, then YY, then ZZ;
  - 12-rings (Ising and XYZ);
  - C52-VAL's heis2d_n16_r1, heis_n12_r2 and tfim_n12_r2.

| device | input | target | new / c53 (geometric mean, range) | new / Qiskit (QK3 or QK3F) |
|---|---|---|---|---|
| FakeFez | heavy-hex Ising (8) | T | 0.69 (0.26-1.00) | 0.43 |
| FakeFez | heavy-hex Ising (8) | F | 0.62 (0.23-0.91) | 0.33 |
| FakeFez | tree Ising (4) | T / F | 0.97 / 0.82 | 0.42 / 0.31 |
| FakeFez | heavy-hex XYZ (4) | T / F | 1.00 / 0.94 | 0.90 / 0.76 |
| FakeFez | 12-ring TFIM (C52-VAL tfim_n12_r2) | T / F | 1.00 / 0.84 | 0.10 / 0.12 |
| FakeMarrakesh | heavy-hex Ising (8) | T | 0.74 (0.32-1.00) | 0.36 |
| FakeMarrakesh | heavy-hex Ising (8) | F | 0.64 (0.29-0.94) | 0.30 |
| FakeMarrakesh | tree Ising (4) | T / F | 0.99 / 0.79 | 0.47 / 0.31 |
| FakeMarrakesh | heavy-hex XYZ (4) | T / F | 0.99 / 0.89 | 0.79 / 0.62 |
| both | heis2d_n16_r1 | T / F | 1.00 (not embeddable; c53's output) | — |

**Headline results.**

- Never longer than c53 in 84/84 cells, by construction of the comparison.
- c52 equals c53 in every cell.
- On the whole device the gain is large: FakeFez 154 qubits, 4 steps gives 3.67 µs against c53's 14.09 (T) and 1.51
  against 6.71 (F). c53's output there is Qiskit's level-3 candidate.
- The TFIM duration does not depend on n: 0.84-0.86 µs (F, 2 steps) for n = 18 to 154.

**Where c53 stays.** c53's own output was kept in 23 of the 84 cells, almost all on T:

- heavy-hex Ising n = 40 with 4 steps;
- some trees;
- most XYZ cells;
- the 12-rings on T.

On T the CZ count of item 91 is sometimes higher (up to 1.12× of c53's). STEINER6 and item 83 already order commuting
ZZ terms well there. With rzz (F) item 91 wins almost everywhere.

**Correctness.**

- **n ≤ 24:** exact outcome distribution, largest total variation distance 3.2e-14.
- **n = 40 (the first rows):** MPS correlations up to 1.1e-9, the same in T and F. This is MPS truncation, not the
  circuit.
- **n > 24, by Clifford-angle twins.** Every rotation is a multiple of π/2, same graph and steps, compiled by the same
  call. They are compared with the logical twin's stabilizer state on every <Z_i> and every <Z_a Z_b> on interaction
  edges.
  - **F path:** 0 difference in all 9 cells, up to 154 qubits.
  - **T path:** cannot be read as a stabilizer circuit, because Qiskit's two-qubit re-synthesis gives non-Clifford
    local gates. It was checked instead by Aer MPS on the same twins against the stabilizer values
    (`eval/t_path_check.jsonl`). All 2-step cells agree within 3e-11: n = 40 patch and tree, 80, and the whole device
    (154 and 153 qubits), on both devices. The 4-step cells (n = 40) did not finish in 400 s of MPS and are not
    checked on T. Their F twins are exact.
- Every output is ISA, and every rzz angle is in range.
- Compile time is at most 27 s (154 qubits, including c53's own compile for the comparison).

Selected cells, duration in µs (two-qubit count in brackets):

| device | input | n | steps | target | c53 + 91 | c53 | Qiskit |
|---|---|---|---|---|---|---|---|
| FakeFez | heavy-hex Ising | 18 | 2 | F | **0.84** (46) | 1.06 (51) | 1.74 (63) |
| FakeFez | heavy-hex Ising | 80 | 4 | F | **1.51** (396) | 1.68 (425) | 4.79 (617) |
| FakeFez | heavy-hex Ising | 154 | 4 | T | **3.67** (1578) | 14.09 (2216) | 14.09 (2216) |
| FakeFez | heavy-hex Ising | 154 | 4 | F | **1.51** (789) | 6.71 (1256) | 7.28 (1256) |
| FakeMarrakesh | heavy-hex Ising | 153 | 4 | F | **1.46** (764) | 4.94 (1214) | 5.52 (1214) |
| FakeMarrakesh | tree Ising | 40 | 4 | F | **1.46** (182) | 1.60 (195) | 3.88 (282) |
| FakeMarrakesh | heavy-hex XYZ | 40 | 2 | F | **1.66** (246) | 1.94 (246) | 3.04 (246) |
| FakeFez | tfim_n12_r2 (12-ring) | 12 | 2 | F | **0.56** (30) | 0.67 (36) | 4.68 (47) |

## 4. Tests

**`test_91.py`: 10 tests, all pass.**

- A heavy-hex patch Ising on T and F is exact, ISA, has rzz in range, and is not longer than c53.
- Unmeasured inputs are the same unitary **with the global phase**: patch Ising, patch XYZ, and a tree with a degree-3
  node.
- The colouring is proper and uses max-degree colours on 30 random bipartite graphs, deterministically. A triangle is
  rejected.
- A degree-4 star, adjacent degree-3 nodes and a 4-cycle are not handled (counted by reason), and the output is
  identical to c53's.
- A 12-ring is embedded in a hexagon with 2 colours.
- A chain still goes to item 87.
- The CZ path without rzz is exact and ISA.
- A failed qubit in the embedding is cut first.
- The output is deterministic.

**c53's own 86 tests on the patched file:** 80 pass. 5 of the 6 failures are `test_version`.

**The sixth failure is `test_c49_on_c53.py::test_ising_still_short`.** It is not a regression.

- The test checks that item 83 halves the duration of c48's `_tfim(12, 4)`. That circuit is a 12-ring, compiled on
  FakeTorino, with `ST_SCHEDULE` on and off.
- With item 91, the ring embeds in a hexagon. The "off" arm becomes shorter (8.27 → 4.94 µs), because item 91's output
  replaces c53's slower one there.
- The "on" arm is unchanged at 3.38 µs: c53's own output, which item 91 does not beat on FakeTorino.
- So the ratio test no longer measures item 83 alone. **Fix for home:** set `GRAPH_BRICKWORK = False` inside that
  test's "off" arm, or in both arms.

## 5. Files

- `patch/`:
  - `make_c53_91.py`, `c53_91_code.py`, the built `psf_compile.py` and `test_91.py`;
  - test outputs: `test_91_result.txt` and `c53_tests_on_patched_file.txt`.
- `eval/`:
  - scripts: `q1_common.py`, `q1_eval.py`, `cliff_check.py`, `t_path_check.py`, `embed_probe.py`;
  - rows and outputs: `q1_rows_FakeFez.jsonl`, `q1_rows_FakeMarrakesh.jsonl`, `summary.txt`, `t_path_check.jsonl` and
    the logs.
