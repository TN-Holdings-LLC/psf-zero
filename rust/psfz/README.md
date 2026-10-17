# psfz -- PSF-Zero in Rust, prototype r1 (2026-10-17)

A new compiler with a different structure from `psf_compile.py`: one Rust program, no outside crate, no Qiskit. It reads
a gate-level OpenQASM 3 circuit and a device file (JSON), and writes OpenQASM 3 on the device's qubits with the
device's gates only (rz, sx, x and cz on IBM Heron; ecr on IBM Eagle). The owner's decisions (2026-10-17, Addendum 518):
chains and rings only; OpenQASM in and out; placement by the lattice's coordinates only (no VF2); judged by
exactness and by duration against release 2026-10-17.1.

**No search, no estimate.** Placement: the heavy-hex lattice is read as rows, column offsets and bridges; a chain goes
on the longest working piece of the snake (centred), a ring on the boundary of a band of hexagons; candidates come in
a fixed order and the first whose qubits and couplers all work (first among couplers of the most common duration) is
used. Nothing is scored, matched or simulated. Every rewrite (diagonal gates as ZZ and phases, ZZ on CZ, the 3-CZ
canonical form, CZ on ECR, single-qubit runs as rz/sx/x) is a closed form, checked once by the unit tests.

    cargo build --release            # rust/psfz; no network needed (no dependencies)
    cargo test                       # the closed forms, the readers
    target/release/psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]

Exit 0: compiled; 2: refused (out of scope; the reason on stderr and in the report); 1: an input error.

**Scope (r1):** every qubit in one chain or one even ring of two-qubit gates; gates id x y z h s sdg t tdg sx sxdg rx
ry rz p u1 u2 u3 u, cx cz cp crz rzz rxx ryy; measurements only at the end. Not yet: other graphs, fractional
gates (rzz/rx natives), templates, mid-circuit measurement.

**Checking (`harness/`, Python, not part of the compiler):** `compare.py` runs psfz and release 2026-10-17.1 on
C63-VAL's 14 chain and ring tests on five fake devices (exactness against Qiskit's state vector); `summarize.py`.
