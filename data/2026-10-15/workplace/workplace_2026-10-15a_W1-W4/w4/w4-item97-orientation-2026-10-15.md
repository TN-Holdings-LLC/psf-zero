# W4 / item 97 (2026-10-15): in scope, c57 depends on the written orientation in 18 of 100 cells (up to 9.7%), all through Qiskit level 3 inside it; the rule removes the dependence and is neutral on average

Workplace, 2026-10-15, finished 04:00 CEST (`TZ=Europe/Zurich date`). Base: prototype c57 (bundle n). Offline only.
In-scope circuits only. No search: the rule fixes one orientation and nothing is compared or chosen.

## 1. Item 97 (`make_c57_97.py`, exact substitutions on c57)

**The rule.**

- Applies only when item 96 puts the circuit in scope, just before c55's path compiles it.
- Every top-level two-qubit instruction that is symmetric under exchanging its qubits is rewritten with the smaller
  index first. The symmetric instructions are: cz, cp, rzz, rxx, ryy, swap, iswap, and a two-qubit PauliEvolutionGate
  whose operator equals its qubit-exchanged copy (ZZ, XX + YY + ZZ, ...).
- The operation, its parameters and the instruction order are unchanged, so the unitary is the same.
- Out-of-scope inputs are not touched. The hand-off therefore stays identical to Qiskit level 3's output for the input
  as written. A test checks this.
- `ORIENT97 = False` restores c57. Counts are in `ORIENT97_STATS`.

**Tests.**

- `test_97.py`: 7 passed. They cover:
  - the rule is exact, and it flips only symmetric instructions (cx, crz and a ZX evolution stay as written);
  - three orientations give one output;
  - the output equals c57's on the canonical input;
  - out of scope, the output is identical to c57 and to Qiskit level 3;
  - the switch restores c57.
- c57's own 122 tests on the patched file: **113 passed; the 9 failures are all version-string assertions**
  ("2026-10-14.c57+97" against "2026-10-14.c57").

## 2. Measurement (`w4_orient.py`, `w4_report.py`, rows in `run/`)

**Why new tests.** C57-VAL's in-scope tests carry no orientation: each is one all-qubit PauliEvolutionGate, or cx. So
I wrote the same kinds of in-scope Hamiltonians gate by gate:

- TFIM chains with rzz (n = 12, 16, 20);
- a 12-ring with rzz;
- heavy-hex patches with rzz (n = 16, 20; C57-VAL's edges);
- a heavy-hex tree Heisenberg model with rxx, ryy and rzz (n = 16);
- a CZ-chain ansatz (n = 12, 16);
- a chain of two-qubit ZZ PauliEvolutionGates (n = 12).

**Setup.**

- Every pair is written smaller index first ("as written").
- "reversed" flips every symmetric gate. "random" flips each with probability 1/2, with a fixed seed per test.
- C57-VAL's five devices, each on T and F: 100 cells, all in scope.
- C57-VAL's call, with seed 0.

**Checks.** Every output was native, used no failed element, and was exact (largest difference 2.1e-7, within
1e-6).

**Results.**

| measure | result |
|---|---|
| c57, spread (max/min duration) over the three orientations | > 1 in **18 of 100** cells; largest **1.097** (FakeTorino F, chain_cz_n12) |
| c57 + item 97, spread | 1 by construction. On FakeFez the three orientations gave one circuit in 20 of 20 cells. Everywhere, the output on "reversed" equals c57's on "as written" (100 of 100). |
| (cell, orientation) pairs where the rule is longer | **16 of 300**, in 10 cells; at most **+9.7%** (FakeTorino F chain_cz_n12, against "reversed") |
| ... where the rule is shorter | 15 of 300, in 8 cells; at most -4.7% |
| against "as written" | identical in 100 of 100 cells, since "as written" is already the canonical form here |
| geometric mean, rule / orientation | 1.0023 against "reversed", 1.0025 against "random" |
| rule among the three, in the 18 sensitive cells | best in 8, worst in 9, middle in 1 |
| two-qubit count | never changed by the rule |
| compile time (mean per call) | c57 14.9 s, c57 + item 97 15.5 s (same machine, shared) |

**Per family.** Insensitive in every cell: the rzz chains, the 12-ring, the 20-qubit patch and the PauliEvolution
chain. Sensitive:

- chain_cz: F in 9 of 10 cells, T in 2 of 10 (up to 1.097);
- hhtree_heis_n16: T in 5 of 5, F in 1 (up to 1.032);
- hh_rzz_n16: F in 1 (1.050).

## 3. Where the dependence comes from

**Traced** (`w4_where.py`, the c57 counters for each orientation).

- In every sensitive cell I traced, the output comes from **Qiskit level 3 inside c57**:
  - chain_cz is read neither by item 87 (`not_chain`) nor by item 91 (`not_read`). c57 returns item 80's level-3
    candidate, whose output depends on the orientation.
  - On hhtree_heis T (FakeFez), item 91 builds its output for every orientation (3 colours each time). Item 80's
    level-3 candidate is 2.976 µs on "as written" and longer on the others. So the comparison keeps level 3 once, and
    PSF-Zero's output (3.000 µs) twice.
- c57's own formula paths (items 87, 91, 96) build their interaction graphs from sorted (min, max) edges. They are
  insensitive to orientation.

**DAG order in Qiskit** (shown directly).

- `DAGCircuit.topological_op_nodes()` is a lexicographic topological sort keyed by the qubit indices.
  - cz(1,0) and cz(0,1) sort differently against their neighbours.
  - Independent gates written in another order come out in the same order.
- So orientation is the one written freedom that this sort does not remove. Every pass that walks the DAG in this
  order sees it.

**Other places that depend on DAG order, besides SabreLayout** (code reading; not each measured):

- VF2Layout and VF2PostLayout. The interaction graph is built in node order, which fixes the first match and how
  ties break.
- SabreSwap's front layer and its ties.
- Collect2qBlocks / ConsolidateBlocks. Block grouping follows the topological order.
- UnitarySynthesis. The block's qubit order fixes which qubit gets which single-qubit gates.
- CommutationAnalysis / CommutativeCancellation. They group in walk order.
- Inside c57:
  - STEINER6's reordering (`STEINER_STATS.reordered`);
  - item 83's time-aware block order;
  - item 75/90 placement ties.

  These work on the DAG as given.

**The written order of commuting gates also matters** (`w4_order.py`; not addressed by item 97). On FakeFez, with
each run of commuting ZZ-type gates written in reverse order:

| test | target | as written | reversed order | same circuit? |
|---|---|---|---|---|
| chain_rzz n12, n16 | T, F | — | equal | yes |
| chain_pe_n12 | T, F | — | equal | yes |
| ring_rzz_n12 | T, F | — | equal duration | no |
| hh_rzz_n16 | T | 2.760 µs | 2.712 µs | no |
| hh_rzz_n16 | F | — | equal | yes |
| chain_cz_n16 | F | 1.932 µs | 1.884 µs | no |

A second rule could sort each run of commuting diagonal gates, for example by (min, max). It would be exact. It is
not built.

## 4. Reading

- Item 97 does what was asked: one output per circuit, whatever the orientation. It never changed the two-qubit
  count, and it is neutral on average (+0.2%).
- In the worst (cell, orientation) pair it is 9.7% longer than another orientation would have been. It is shorter
  about as often.
- The dependence it removes is small in scope: 18 of 100 cells, and only where c57 falls back to Qiskit level 3
  (CZ chains are not read by items 87 and 91).
- R1's 19% was on 3-regular QAOA, which is now out of scope and handed to Qiskit level 3 as written. Item 97
  deliberately leaves that unchanged, to keep the hand-off identical.
- A larger in-scope gain would come from reading CZ chains by formula (item 87 reads ZZ-type terms only). That is
  outside this item.

## 5. Files

- `c57_97/`: `make_c57_97.py`, `psf_compile.py` (built from c57), `test_97.py`, `c57_tests_on_97.txt`.
- `w4_orient.py`, `w4_report.py`, `w4_report.txt`, `run/` (rows and logs), `w4_where.py`, `w4_order.py`,
  `w4_order.log`, `w4_inventory.py`.
