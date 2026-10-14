# N1 (2026-10-14): chain Hamiltonians built by formula as a brickwork -- a patch proposal on c49 (item 86)

**Status: patch proposal with tests; offline only.** Workplace, 2026-10-14, finished 03:38 CEST (`TZ=Europe/Zurich
date`). Common rule followed: no search, no sweep, no seed variation, no choice by ESP or simulation (a complete
embedding taken as the first match is allowed, as home ruled); measured: correctness, CZ count, duration.

## 1. What item 86 does

Built from c49 by `make_c49_n1.py` (exact substitutions; c49's file sha256 `5544d7c9...`) with the code in
`n1_code.py`. With the device given (`steiner` "auto" true), its two-qubit gate cz, and no `coupling_map`,
`initial_layout`, `final_resynthesis`, `compare_level3` or `compare_floor` of the caller's:

1. **Read.** Terms by c49's own reader (items 72-73), merged where exact (`_st_merge`). It applies when every term
   acts on one or two qubits and the two-qubit supports form one path through all the circuit's qubits (TFIM, XXZ,
   Heisenberg chains; QAOA on a path graph).
2. **Cut and place (W5 variant B).** Failed qubits and couplers removed, the largest connected component K kept. The
   chain goes, in path order, on the longest surviving piece of Heron's snake, centred; if none is long enough, on
   the first path of n qubits rustworkx's VF2 finds in K (`id_order=True`, no score). Both are tried first on the
   couplers whose cz has the device's most common length (gate lengths are fixed per device), then on all of K.
3. **Order (formula).** Each term's level = 1 + the largest level of an earlier term that shares a qubit and does
   not commute with it. Emitted by level: two-qubit terms on the chain's even edges, then odd edges, then
   single-qubit terms. Every non-commuting pair keeps its order, so the result is exact. No STEINER6, no item 70
   size limit.
4. **Emit.** XX/YY/ZZ as Qiskit's rxx/ryy/rzz, other two-qubit Paulis by basis change and cx-rz-cx; Qiskit's
   level-3 preset translates on the fixed layout with `routing_method="none"`, coupling map and basis only (no
   Target: no error value is read).
5. **Safety.** The brickwork is compared with c49's own output for the same call by item 84's rule (`_better`:
   duration, then CZ) and kept unless c49's is shorter (a tie keeps the brickwork). This matters only on devices
   whose gate lengths vary from qubit to qubit (it happened on c49's generic 7-qubit test backend). Not applicable,
   or no path of n qubits: c49 unchanged. `CHAIN_BRICKWORK = False` restores c49; counts in `CHAIN_STATS`.

## 2. W2's 48 cells through the patched default call (`compile_for_hardware(qc, backend=b, seed_transpiler=0)`)

Rows `e2e2/n1_e2e_rows.jsonl`, log `e2e2.log`. Item 86's brickwork was returned in all 48 cells.

| check | result |
|---|---|
| correct (n <= 24 exact outcome distribution; n > 24 all one- and two-point Z correlations, MPS) | 48 of 48, largest error 3.8e-9; every output ISA |
| CZ count <= QK3 given the hand-written brickwork (QK3-BW) | **48 of 48** (0.84-0.96 of it) |
| duration <= QK3-BW | **46 of 48** (0.80-1.02 of it) |
| against c48 (W2's PSF) | 0.011-0.96 of its duration |
| against QK3 given the Pauli evolution | 0.011-0.13 of its duration |
| compile time | at most 21.8 s (n = 100, including c49's own compile for the comparison) |

Selected cells (CZ, us):

| device | n | steps | c49 + item 86 | QK3-BW | c48 |
|---|---|---|---|---|---|
| FakeFez | 12 | 10 | 232, 5.30 | 242, 5.30 | 242, 5.52 |
| FakeFez | 100 | 10 | 2080, 5.30 | 2178, 5.74 | 3922, 471 |
| FakeMarrakesh | 100 | 2 | 496, 1.45 | 594, 1.66 | 696, 6.74 |
| FakeMarrakesh | 100 | 10 | 2080, **6.02** | 2178, 5.93 | 3922, 483 |

**The two cells not met:** FakeMarrakesh n = 100 at 8 and 10 steps, +0.2% and +1.6% in duration (CZ 5% lower).
FakeMarrakesh has cz of 68 ns on most couplers and 80 ns on 11; with its 13 failed couplers cut, no 100-qubit path
exists on the 68 ns couplers alone, and the first VF2 path crosses 80 ns couplers. Picking among paths by their gate
lengths would meet the target, but that is a placement chosen by duration; whether that is allowed under the common
rule is home's decision. It is not done here.

Why the CZ count is lower than QK3-BW's: the last layer of ZZ terms item 86 emits is diagonal and stands
directly before the measurements, so Qiskit's level 3 drops it (FakeFez n = 12: 56 against 66 CZ at 2 steps, 232
against 242 at 10). Without final measurements the counts are equal to QK3-BW's (66 and 242). The outcome
distribution is unchanged, and the correctness check confirms that.

## 3. Tests (`patches/psf_compile_c49n1_2026-10-14/`)

- `test_n1.py`, 9 tests, all pass:
  - TFIM n = 8 and 12 exact and not longer than QK3-BW;
  - an XXZ chain with fields exact;
  - n = 100 with 10 steps (no size limit);
  - a ring and a QAOA not applied and identical to c49;
  - deterministic;
  - the layout reads the logical qubits;
  - not applied on a non-cz device;
  - on the generic backend, never longer than c49's own output.
- c49's own 54 tests on the patched file: 52 pass. The 2 failures are `test_version`, which expects
  "2026-10-14.c49" (the patch is "2026-10-14.c49+n1"). The first version of the patch failed one more test (a
  QAOA on a path on the generic backend was longer than c49's output), which led to the comparison in step 5.

## 4. Files

- `patch/make_c49_n1.py` (sha256 `9685ec085d9e4d33...`), `patch/n1_code.py` (`ed490179a88e852f...`).
- The built file, `patches/psf_compile_c49n1_2026-10-14/psf_compile.py` (`c1ee4d4986f7c4b3...`).
- `test_n1.py` (`add9052160d64ce7...`).
- `n1_chain.py` and `n1_check.py`: the standalone prototype and its first check.
- `n1_e2e.py`: the end-to-end check above.
