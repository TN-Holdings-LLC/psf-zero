# PSF-Zero — POC kit

Try PSF-Zero against real numbers in about five minutes. This kit does not
reimplement anything — every script here calls the real, installed
`psf_compile` module directly, so what you see is what the papers report,
not a simplified stand-in.

## What this is, and isn't

- **Is**: a small, self-contained set of scripts for a first evaluation —
  install, run one command, see real numbers.
- **Isn't**: the full research record. For the complete, pre-registered
  measurements behind every number quoted here, see the two papers in
  `docs/papers/` and the main repository README.

## 1. Install (2 minutes)

From the main repository root (this kit assumes PSF-Zero is already
installed there):

```bash
git clone https://github.com/TN-Holdings-LLC/psf-zero
cd psf-zero
pip install -e .
maturin develop --release      # builds the Rust core
```

Needs `numpy`, `scipy`, `qiskit` (installed automatically), and a Rust
toolchain for the `maturin` step. See the main README if `maturin` itself
isn't installed (`pip install maturin`).

## 2. Quick start (1-3 minutes)

```bash
cd poc_kit
python quickstart.py
```

Two short demonstrations: gate synthesis on a single block (exact
correctness, both tools, side by side), and the layout-search gap on a
small saturated grid. No arguments, no configuration — just run it.

## 3. The fuller comparison (3-10 minutes, configurable)

```bash
python compare.py                              # fast defaults
python compare.py --iterations 2000 --instances 9   # closer to paper scale
```

Reproduces the *shape* of both papers' own headline tables at a reduced
scale (fewer iterations, a subset of instances) so it stays fast. The
script is explicit throughout about which paper table each number
corresponds to, and that a reduced run is not a substitute for the full,
pre-registered measurement.

## 4. Sample circuits

```bash
python sample_circuits.py
```

The three circuit families used throughout both papers, as importable
functions (`dense_pair_blocks`, `saturated_grid`, `chain_shaped`) — reuse
these directly if you want to point either tool at a circuit shaped like
the ones already measured, before bringing in your own.

## 5. Try your own circuit

```python
from qiskit import QuantumCircuit
import psf_compile

qc = QuantumCircuit(...)   # your own circuit
out = psf_compile.compile(qc, verify=True)   # verify=True checks correctness as it runs
```

For a circuit that also needs layout onto a real device's coupling map,
see `psf_compile.compile_for_hardware()` — `quickstart.py`'s Demo 2 and
`compare.py`'s Part 2 both show it in use.

## Questions, or want a second opinion on your own circuit?

`love.os.architect@proton.me` — see "Working with us" in the main
repository README for how that conversation usually goes (short version:
send a representative circuit, we run it, no commitment implied).

## What's authoritative

Every number this kit prints is a live measurement, not a copy of a
paper's own table — expect small run-to-run variation, and expect the
*reduced-scale* comparison numbers in `compare.py` to differ somewhat
from the papers' own full-scale figures (both papers discuss this kind
of run-to-run variation directly; it is not swept under the rug here
either). For the authoritative, pre-registered numbers, always defer to
the papers themselves and their own linked raw data.
