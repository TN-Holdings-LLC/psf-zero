# Workplace delivery 2026-10-14f: R1-R4 of 「次の指示5」 (base: c54, bundle g)

Finished 09:11 CEST (`TZ=Europe/Zurich date`). Offline only: no IBM connection, no commit, no push. Check the kit with
`python check_intake.py`. 「今夜の手順」 (TIME1, FRAC1) was received; nothing was done for it here, as it asks.

## Contents

- **`r1/`: 3-regular QAOA.**
  - **Layering against seed.** The hand-layered circuit through QK3 at seed 0 alone beats c54 only on qaoa_n12_p2
    (5.53 against 5.98 µs). On the 20-qubit tests c54 is already at or below the user's seed 0, and the user's margin is
    the seed (placement) search.
  - **No formula placement came near:** first Hamiltonian path, and reverse Cuthill-McKee on a line or a patch, are
    1.4-3.3× c54.
  - **Optional item 94** (`make_c54_94.py`): re-layer commuting ZZ runs by edge colouring, compile both inputs, take the
    shorter exact one.
    - Never longer than c54; up to 0.83× on 6 of 22 cells.
    - 7 tests pass.
    - Caveat: c54 is sensitive to the qubit order of a symmetric rzz, by up to 19%.
- **`r2/`: the ~1e-7 differences.**
  - **Cause:** the Weyl snap in Qiskit's two-qubit synthesis (UnitarySynthesis and TwoQubitPeepholeOptimization), plus
    CommutativeCancellation, inside STEINER6's final level-3 transpile.
  - **Which circuits:** in C52-VAL's 240 cells on c54, only tfim_n20_r3 (6 cells, 1.5e-8 to 3.2e-7). It has 20 active
    qubits, so item 93 (limit 16) does not check it.
  - **Fix (prototype `exact2q.py`):** exact two-qubit synthesis that re-solves Qiskit's own choice, without
    CommutativeCancellation. tfim_n20_r3 becomes exact at +10%. c54's 1e-9 fallback (the usual path) is 17-21× longer.
  - **Proposal:** extend item 93 to 20 qubits, use the exact pass manager, then 1e-9.
- **`r3r4/`.**
  - **R3:** per-edge fusion is already done by level 3 (identical two-qubit counts). The Heisenberg length comes from
    the input's edge-by-edge term order (a serial staircase) and from placement.
  - **R4:** on FakeTorino no embedding fits the 68 ns couplers, so item 91's first VF2 match uses 108-184 ns couplers,
    and c53's duration-scored placement wins. The 0.62-0.74 against 0.80-0.86 difference comes from my large patches;
    on comparable cells the ratios agree.
