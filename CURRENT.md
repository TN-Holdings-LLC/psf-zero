# What is current (2026-11-13)

One page that names the version of each part in use today and where it is. Older versions stay where they were
built (`patches/`), because about 120 scripts and tests outside that folder load them by path; nothing there is moved.

| part | current version | where | how it is used |
|---|---|---|---|
| Compiler | `psf_compile` 2026-10-18.1 | `psf_compile.py` (repository root) | `import psf_compile`; `benchmarks/psf_compile.py` only redirects to this file |
| Two-qubit core (Rust, Python module) | `psf_zero_core` 2026-11-13.1 | `src/lib.rs` | `pip install -e .`, then `maturin develop --release` |
| Chains and even rings (Rust program) | psfz r4 compile; r5 template prototype; gate 1 P0 (IR r2); MPS twin prototype 2026-11-13 | `rust/psfz/` | `cargo build --release` there; `psf_compile` calls it (item 105) and falls back to its own path when it is missing |
| AI front end | a26 (2026-10-26) | `patches/psf_ai_compile_a26_2026-10-26/psf_ai_compile.py` | load by path (README, "A ring classifier that does not embed"); needs `rust/psfbind` |
| Ring templates (Rust library) | psfbind ring r3 (2026-10-25) | `rust/psfbind/` | `cargo build --release --manifest-path rust/psfbind/Cargo.toml` |

Kept for older scripts (not current):

- `benchmarks/_frozen_release.py`: gives `psf_compile` 2026-10-14.1 (`patches/psf_compile_release_2026-10-14.1/`) to
  tests written for it.
- `benchmarks/psf_ai_compile.py`: redirects to the AI front end a23 (2026-10-16, `patches/psf_ai_compile_a23_2026-10-16/`),
  the version two benchmark scripts were written against. The current front end is a26 above.

Where the rest is:

- `docs/RELEASES.md`: every release and notice, newest first. `docs/psf_compile_CHANGELOG.md`: the compiler's items.
- `docs/findings/`: the dated record (Addenda), in combined files.
- `docs/design/`: proposals; each says at its top what was built from it.
- `data/<date>/`: the inputs, scripts and outputs behind each dated result.

Related repositories: [psf-os](https://github.com/TN-Holdings-LLC/psf-os) (the operating layer: scan, list, avoid,
sentinel, twin), [psf-os-lab-notebook](https://github.com/TN-Holdings-LLC/psf-os-lab-notebook) (declared device runs and
their records), [readable-depth](https://github.com/TN-Holdings-LLC/readable-depth).
