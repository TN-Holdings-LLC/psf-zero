# psfbind (2026-10-21; ring r2, 2026-10-22)

PSF-Zero's ring template in Rust, through a C ABI (no dependencies; Python calls it with ctypes):

- **Binding** (`psfbind_angles`): `psf_ai_compile.RingTemplate.angles` (item 35) for a batch of value vectors. Each
  run's product of fixed matrices and rotations, written as ZYZ angles exactly as `_t_zyz` (the same special cases),
  the same order of products as the Python, and complex division and square root as CPython's (`_Py_c_quot`,
  `cmath.sqrt`), so that the angles are the Python's to the last bit.
- **Building** (`psfbind_ring_template`, `src/ring.rs`): psf_ai_compile a22's `model_template` for a ring classifier
  that does not embed in the device -- item 28's reading, logical circuit and N2 schedule, item 31's schedule, item
  87's placement on the device's snake (`_chain_lattice`, `_chain_snake`, `_chain_place`), item 35's blocks and runs,
  and item 84's choice between the two schedules (`_better`: a missing operation or error >= 1 loses, then the
  as-soon-as-possible duration, then the CZ count). JSON in (the device's Target entries, the input circuit, the fixed
  CZ / SWAP blocks as a22's `_block_ops` decomposes them), JSON out (the template; or "none", the Python's None; or
  "refused" with the reason, for a case not ported: a ring that embeds, a circuit that is not a CZ ring; or
  "out_of_scope" where the Python would place the ring by VF2's search: outside the scope by the owner's rule of
  2026-10-21, formula only, never built), freed with `psfbind_free`.

**Ring r2 (2026-10-22, for the AI front end a24, Addendum 526):**

- the zipper schedule (`ring_rounds_zip`, the workplace's prototype P1; "zip", even n >= 4) beside N2 ("n2") and item
  31's ("q4"); the request may name the schedules to build, in order (`"schedules": [...]`; the choice among them as
  before, a full tie keeping the earlier); without it, "n2" then "q4" as in r1;
- the placement (E5, owner 2026-10-22): the snake piece on the couplers of the most common gate length, else the
  snake piece on every working coupler; "out_of_scope" only where neither holds the ring (r1 stopped after the
  first);
- `psfbind_version()`: "psfbind ring r2 (2026-10-22): n2 q4 zip", which a24 checks before use.

    cargo build --release     # target/release/libpsfbind.so
    cargo test                # ZYZ, the principal square root, the special cases, the schedules

Used by TOQB: `toqb/rustbind.py` (the fast path of the PSF template arm) and `toqb/ringport.py` (the template built in
Rust; the arm uses it with `TOQB_RUST_TEMPLATE=1`, or `=check` to build the Python's beside it and stop on any
difference). Checked there, 2026-10-21:

- binding: bit-identical to the Python on random points (L 8; before the CPython division and square root: 1.8e-15);
  63 times faster per point (0.14 ms against 8.8 ms on the cloud CPU, L 8);
- building: bit-identical templates on the stand-in (L 2, 8, 16; both schedules alone too), 146 of 150 cases identical
  over five devices with random failed elements, none different (the other 4 would need VF2's search: out of scope); a device-style run gives
  the same record and the same QPY (circuit names aside); 0.04 s against 0.15 s per template (L 8, one per run).
