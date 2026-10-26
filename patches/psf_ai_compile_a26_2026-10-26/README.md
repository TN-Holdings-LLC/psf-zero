# AI front end a26 (2026-10-26): the ring folded onto one of the device's own cycles, the default

`psf_ai_compile.py` here is a25 (`patches/psf_ai_compile_a25_2026-10-25/`) with one item, made by `make_a26.py` (a25's
text with `a26_section.py` appended). With psf_compile 2026-10-18.1 (unchanged) and `rust/psfbind` ring r3 (for a25's
path, still used where item 50 does not fold).

- **Item 50.** A ring classifier that does not embed in the device is laid on a heavy-hex cycle (12, 20, 24, 28, ...
  qubits; the outline of a fixed shape of cells read off the coupling graph, the first working one in a fixed order;
  no search, no error value decides; a coupler or qubit at error >= 0.5 is never used):
  - **pendants:** the longest cycle of m <= n qubits; the n - m extra ring nodes hung off cycle nodes with a working
    outside neighbour; each layer every pendant node trades places with its cycle node by one CZ + SWAP block:
    **n + (n - m) two-qubit gates per layer, on exactly n qubits** (n = 13: 14, against the zipper's 29);
  - **holes:** the shortest cycle of m >= n qubits; its m - n spare qubits start in |0> and are moved across by two
    CX each: n + 2 (m - n) per layer;
  - **the rule:** of pendants n - m, holes 2 (m - n) and the zipper's extra gates per layer, the fewest decides (ties:
    pendants, then holes). On the four Heron fake devices: holes at n = 8-11 and 18-19, pendants at 13-17; n = 4-7
    keep a25's path (the zipper's count is lower), embeddable rings (n = 12, 20) are psf_compile's as before; at n =
    20-30 pendants on the 20-, 24- and 28-cycles and holes at 23 and 27 (TOQB results/rule-big-2026-10-26, compile
    only). ECR devices (Eagle): a25's path.
  - Built once into a `FoldTemplate` (Qiskit's level-1 preset on the fixed layout, no routing, as item 28's
    translation); `bind(qc)` assigns the angles; the default call `compile_for_model_circuit` binds the same template.
    **`PSF_RING_FOLD=0` gives a25's output.**
- **Evidence (TOQB).** Flows E and F on the twin of a 2026-10-13 snapshot of ibm_marrakesh:
  - Flow E (results/flow-e-2026-10-26): at n = 13-17 the pendants kept the most signal of pendants, holes and zipper
    in every value; at L = 16 2.1-4.0 times the zipper's lam.
  - Flow F, a declared match (results/flow-f-2026-10-26): the rule against the zipper at n = 13-18, L = 16, four
    random models: **WIN at every n, 24 of 24 models**; the rule's lam 0.51-0.76, the zipper's 0.09-0.46 (smallest
    ratio 1.5).
  - **Flow G (added 2026-10-26 evening), the same match on the twin of ibm_kingston's 2026-10-24 calibration** (the
    kind that matched the device for the zipper): WIN at every n (23 of 24 models), but by **1.07-1.28 times** the
    zipper's lam at n = 13-17 and about even at 18. Flow F's larger ratios came mostly from the marrakesh snapshot's
    pessimism for the zipper.
  - **Device test D1** (2026-10-26, ibm_kingston, n = 13, 64 random inputs x 250 shots, 7 s): the folded ring kept
    lam 0.90 +- 0.12 against its twin's 0.77 (AGREE, declared); a coarse test, no zipper on the device.
  - One place per n; a poor coupler next to the read-out node can cost much (flow F's read-out move at n = 13: -0.29
    to -0.39), and the formula does not read errors.

**Checks.**
- `test_a26.py` (8 tests): exact folds at n = 10, 13, 16, 18 with the default call equal to the template; a cycle in
  any qubit order; `PSF_RING_FOLD=0` and n = 5, 6 equal a25; a failed coupler avoided; a non-ring circuit equals a25.
- `check_a26.py` with `check_a26_result.txt`: check_a24's 272 ring cases plus 68 on FakeSherbrooke (n = 4-20,
  L = 1, 2, 4, 8), two calls each, `PSF_RUST_TEMPLATE=check`:
  - A, `PSF_RING_FOLD=0`: a26 equals a25 on all 680 calls;
  - B, default: 176 folded cases, each exact on every clbit (statevector, 1e-9), the default call equal to the
    template, pendants at n + (n - m) per layer (holes: as n + 2 (m - n) in 88 cases, 2 more in 8 where two holes
    meet); 328 other calls equal a25. RESULT PASS.
- Against TOQB's analysis arm `fold:rule` (145f38b) on TOQB's `snapshot:marrakesh`: the same circuits, instruction for
  instruction, at n = 8-19, L = 2, 4.

Files: `make_a26.py`, `a26_section.py`, `psf_ai_compile.py` (generated), `test_a26.py`, `check_a26.py`,
`check_a26_result.txt`.
