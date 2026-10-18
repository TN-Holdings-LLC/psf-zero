# Prototype c68 (2026-10-18): item 105, chains and rings by psfz r4

`make_c68.py` = release 2026-10-17.1 (`psf_compile.py`) + `item105_psfz_code.py` -> `psf_compile.py` (the version
line changed to 2026-10-18.c68). Validated in C68-VAL (Addendum 524); released as 2026-10-18.1 (Addendum 525): `make_release.py` changes the docstring, the version line and item 105's heading; `check_release.py` compares the installed release with this prototype on C68-VAL's 4,400 inputs, with and without the psfz program (`check_release_result.txt`).

**Why.** psfz r4 (`rust/psfz/`, Addenda 518-522) met the owner's bar in PSFZ-VAL2 (Addendum 522): on 3,238 chain and
ring inputs exact, native, deterministic and never longer than release 2026-10-17.1, 0.956 x its duration on the
geometric mean (0.93 x on Eagle). The owner chose (2026-10-18): use it from `psf_compile.py` automatically for chains
and rings, its output as it is, through a prototype and a pre-registered validation before a release.

**Item 105.** The default call (`compile_for_hardware(qc, target=...)` or `backend=`, no option other than
`seed_transpiler` and the default `prune_max_error`) first hands the circuit to psfz's program, a separate process
(`$PSFZ_BIN`, else `rust/psfz/target/release/psfz` in the file's folder or one or two folders up): the circuit as
gate-level OpenQASM 3 (Qiskit's transpile to a gate basis, no layout, no routing, as in PSFZ-VAL2) and the device as
psfz's JSON. psfz's output is returned as it is (with a TranspileLayout) when psfz compiles the circuit and reports
version r4. In every other case -- no program, another version, psfz refuses (not a chain or ring, a gate it does
not read, ...), an option given, classical bits outside registers -- the call is the release's path, unchanged.
`PSFZ105 = False` turns the item off. Counts in `PSFZ105_STATS`. Not carried over: the global phase (psfz's output
equals the input up to one).

**Needs:** psfz built (`cd rust/psfz && cargo build --release`, no dependencies, no network).
