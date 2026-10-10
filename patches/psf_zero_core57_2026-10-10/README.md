# psf_zero_core57 -- item 57b (candidate 2026-10-10.c30)

The per-gate loops of `excitation_cost`, `hybrid_cost` (its gate terms) and item 39's `_apply_ops`, in Rust, for
[`patches/psf_compile_c30_2026-10-10/psf_compile.py`](../psf_compile_c30_2026-10-10/psf_compile.py). A Python
module of its own, so that the release's core (`psf_zero_core`, `src/lib.rs`) is not rebuilt or changed while the
candidate is tested. Without it, c30 runs the Python code, as the release does.

- `src/statevec.rs`: the loops; standard library only. Its tests: `rustc --edition 2021 --test -O statevec.rs`.
- `src/lib.rs`: the pyo3 binding: `estimate57(buffer)` and `apply_ops57(buffer)`; the buffer's format is
  `statevec::parse`'s.

Build into the active virtual environment, from this folder:

```bash
maturin develop --release
python -c "import psf_zero_core57 as m; print(m.CORE57_VERSION)"   # 2026-10-10.c30
```

Design and prototype: [`docs/findings/psf-zero-rust-estimates-design-2026-10-10.md`](../../docs/findings/psf-zero-rust-estimates-design-2026-10-10.md);
candidate tests: [`test_c30.py`](../psf_compile_c30_2026-10-10/test_c30.py); identity test: Addendum 426.
