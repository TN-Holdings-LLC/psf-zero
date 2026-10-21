# psfbind (2026-10-21)

The template binding of `psf_ai_compile.RingTemplate.angles` (item 35) in Rust, for a batch of value vectors, through
a C ABI (no dependencies; Python calls it with ctypes). Each run's product of fixed matrices and rotations, written as
ZYZ angles exactly as `_t_zyz` (the same special cases), the same order of products as the Python.

    cargo build --release     # target/release/libpsfbind.so
    cargo test                # ZYZ, the principal square root, the special cases

Used by TOQB's `toqb/rustbind.py` (the fast path of the PSF template arm: the model's parameters straight to the
simulator's slot angles). Checked there against the slow path: equal to 1.8e-15 on random points, the twin's <Z>
identical with the same seeds; 63 times faster per point (0.14 ms against 8.8 ms on the cloud CPU, L 8).
The compiler's decisions are unchanged: the template is still built by psf_ai_compile; only its binding moved.
