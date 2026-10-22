# AI front end a24 (2026-10-22): the ring template in Rust, formula-only placement, the zipper as an option

`psf_ai_compile.py` here is a23 (`patches/psf_ai_compile_a23_2026-10-16/`) with three items, made by `make_a24.py`
(edits to a23's text, plus `a24_tail.py` appended). Addendum 526. With psf_compile 2026-10-18.1 (unchanged).

- **Item 46, the ring template in Rust.** `model_template` / `ring_template` for a ring that does not embed is built
  by `rust/psfbind` (ring r2), through ctypes; the result is a23's RingTemplate class. No library, or one older than
  ring r2: `RuntimeError` (build: `cargo build --release --manifest-path rust/psfbind/Cargo.toml`, or set
  `PSF_BIND_LIB`). `PSF_RUST_TEMPLATE=check` builds the Python template beside it and stops on any difference.
- **Item 47, E5 (formula only).** The ring's line is the snake piece on the couplers of the most common gate length,
  else the snake piece on every working coupler; where neither holds the ring, `OutOfScopeError` (a23 searched with
  VF2). In the template and in the default call.
- **Item 48, the zipper.** With `PSF_RING_ZIP=1`, an even ring of n >= 4 also gets the workplace's zipper schedule
  (5n/2 - 4 CZ per layer) and item 84's rule chooses. Off by default: report-only evidence so far (TOQB, on its twin).

Files: `make_a24.py`, `a24_tail.py`, `psf_ai_compile.py` (generated), `check_a24.py` with `check_a24_result.txt`,
`test_a24.py`.
