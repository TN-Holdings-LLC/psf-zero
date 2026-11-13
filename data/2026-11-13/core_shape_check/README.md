# Core 2026-11-13.1: shape check only (home, 2026-11-13)

Owner 20:09 JST ("やる"), after an outside review proposed a "2026-09-29.2" core (partial_cmp -> total_cmp, input
checks, eigen route with tolerance candidates and pencil eigenvalues, static 4x4 matrices).

## What was checked (cargo test --release, rustc 1.97.0)

`dump_harness.rs` (appended identically to each core) decomposes 2239 inputs and writes every output's bits:
identity, CNOT, SWAP, iSWAP; 2000 random SU(4); the 15 eigen-route blocks of item 12 (`src/test_data/eigen15.txt`);
60 |cos| ties; 60 compound ties (|cos| tie plus an exactly repeated phase); 100 exactly repeated phases.

| core | errors | worst 1 - fidelity | outputs bit-identical to 2026-09-29.1 |
|---|---|---|---|
| 2026-09-29.1 (current) | 0 | 1.4e-15 | (reference) |
| proposed 2026-09-29.2 | 0 | 1.4e-15 | 2161 of 2239; differ: {'eigen': 14, 'compound': 14, 'exact': 50} |
| 2026-11-13.1 (this commit) | 0 | 1.4e-15 | 2239 of 2239 (dump identical: True) |

- The current core panics on lists that are not 4 x 4 (index out of bounds, seen as PanicException in Python): fixed
  here. It already rejects NaN (NumericInstability): unchanged.
- The proposed eigen-route change fixed no failure (the current core decomposes all 60 compound ties) and changed the
  outputs of 14 of the 15 production eigen-route blocks; its new test does not check which route it exercised. The
  speed items touch the eigen route only, which runs when the SVD route has failed. Not taken.

dump sha256: current e2a1c0b0b3a20de7..., this commit e2a1c0b0b3a20de7...,
proposed 396b8801bfd2eb0a....

Python needs the rebuilt core to see the change: `maturin develop --release` (README, after `pip install -e .`).
