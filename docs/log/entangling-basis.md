# Why PSF-Zero lost fidelity on a noisy simulator, and the two bugs behind it

**Status:** two independent root causes found, both real, both fixed. Substantially
closed rather than fully closed — the scripts that produced the *original* numbers
are lost, so the exact path they took cannot be confirmed.

---

## The observation that started it

On a `fake_sherbrooke` noise model with mirror circuits, PSF-Zero came out **lowest of
four engines** on two of three circuit families, by a margin well outside the standard
errors:

| Family (2Q gates/pair) | Qiskit L3 | TKET | **PSF-Zero v6** | Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| `deep2q` (3) | 0.9056 | 0.9077 | **0.8638** | 0.9076 |
| `multi_deep2q` (12) | 0.0849 | 0.0863 | **0.0720** | 0.0839 |
| `wide` (42) | 0.0033 | 0.0039 | 0.0044 | 0.0037 |

<sub>mean P(all-zero) ± SE, n=5. On `wide` all four are at the noise floor and the
differences mean nothing.</sub>

Roughly 4 percentage points on `deep2q` and 1.3 on `multi_deep2q` — repeatable, not
run-to-run noise. Notably, `wide` is the family where PSF-Zero's `block_gate_floor`
logic leaves the circuit **completely untouched**, and there was no deficit there.
That pointed at PSF-Zero's synthesis output rather than at anything else.

## Root cause 1: `RXX`/`RYY`/`RZZ` costs 2x the native gates that `CX` does

PSF-Zero v6 builds each block from four local single-qubit triples plus up to three
entangling gates — but those are `RXX`/`RYY`/`RZZ`, not `CX`. Neither is native to IBM
hardware (`fake_sherbrooke`'s basis is `ecr`/`rz`/`sx`/`x`), so the question is what
each costs after translation.

`diagnose_native_gate_inflation.py` builds the same canonical KAK structure PSF-Zero
emits for 200 random SU(4) unitaries, plus a CX-basis decomposition of the same
unitaries, transpiles both to `fake_sherbrooke` at each optimization level, and counts
native `ecr` gates (0 correctness failures at any level):

| `optimization_level` | RXX/RYY/RZZ basis | CX basis | Ratio |
| :---: | :---: | :---: | :---: |
| 0 | 6.00 ECR | 3.00 ECR | **2.00x** |
| 1 | 6.00 ECR | 3.00 ECR | **2.00x** |
| 2 | 3.00 ECR | 3.00 ECR | 1.00x |
| 3 | 3.00 ECR | 3.00 ECR | 1.00x |

At `optimization_level` 0–1 the PSF-Zero-shaped circuit costs exactly **twice** the
native two-qubit gates for the identical unitary. At level ≥ 2 Qiskit resynthesises
2-qubit blocks from scratch regardless of input basis and the gap vanishes.

This is invisible to any benchmark that counts 2-qubit gates on the **pre-ISA** circuit
— which is exactly what the table above reports as `mean_two_qubit_gates`, showing 3.0
for every engine.

## Root cause 2: `compile_for_hardware()` never passed `basis_gates`

The function as written was:

```python
def compile_for_hardware(qc, coupling_map, block_gate_floor=..., routing_optimization_level=0):
    qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
    return transpile(qc_compressed, coupling_map=coupling_map,
                     optimization_level=routing_optimization_level)
```

Only `coupling_map` is given — no `basis_gates`, no `backend`. So `transpile()` does
layout and routing and **never translates to a target basis at any
`routing_optimization_level` from 0 to 3**, confirmed directly. Its output still
contains `RXX`/`RYY`/`RZZ` and IBM Runtime's `SamplerV2` would reject it as non-ISA,
which means there had to be one more transpile-to-ISA step somewhere downstream — and
*that* step is where the 2x penalty above would land, if it used a low optimization
level.

The function's own docstring gave exactly the reasoning that would lead someone to
pick a low level there too: *"`routing_optimization_level` defaults to 0 (routing only)
since `compile()` already did the 2-qubit optimization that a higher
`optimization_level` would otherwise redo."* That is true about **logical** 2-qubit
gate count and false about **physical** native-gate count. A higher level there does
not redo work; it does work that was never done.

## The fixes, and what they recovered

### Fix A — thread `basis_gates` through `compile_for_hardware()`

`verify_compile_for_hardware_fix.py`, N=50 random SU(4) blocks forced onto
non-adjacent qubits on a 6-qubit line (so every trial needs real routing), basis
`['ecr','rz','sx','x']`:

| `routing_optimization_level` | Correctness failures | Mean ECR |
| :---: | :---: | :---: |
| 0 | 0/50 | 12.00 |
| 1 | 0/50 | 6.00 |
| 2 | 0/50 | **3.00** |
| 3 | 0/50 | **3.00** |

Levels 0–1 are *worse* here than in the unrouted diagnostic (4x and 2x, against 2x and
2x) because once `basis_gates` is supplied the routing SWAPs also need decomposing,
and low levels do not do that efficiently either.

Run end-to-end through the **real** `compile_for_hardware()` code, before and after,
with the same naive `optimization_level=1` final step and `fake_sherbrooke` noise:

| Family | Before, P(all-zero) | After | ECR: before → after |
| :--- | :---: | :---: | :---: |
| `deep2q` | 0.8747 ± 0.0076 | **0.9190 ± 0.0039** | 12 → 6 |
| `multi_deep2q` | 0.1484 ± 0.0122 | **0.1909 ± 0.0044** | 48 → 18 |
| `wide` | 0.0448 ± 0.0007 | **0.1418 ± 0.0069** | 24 → 18 |

**The `wide` row is the important one.** On `wide`, `compile()` reports `0/0 blocks` —
PSF-Zero's synthesiser never runs — and the fix still recovers ~9.7 points. So this
bug is **not a PSF-Zero synthesis defect at all**. It is a generic ISA-translation gap
in the hardware-submission step that would affect any circuit
`compile_for_hardware()` is asked to prepare, whatever produced it.

### Fix B — `entangling_basis="cx"`

An opt-in `entangling_basis: str = "canonical" | "cx"` parameter on
`GeodesicPSFHyper` / `synthesize()` / `compile()` / `compile_for_hardware()`. `"cx"`
resynthesises the entangling core through Qiskit's own
`TwoQubitBasisDecomposer(CXGate())` — already imported for the existing
degenerate-point fallback, so no new trust surface — instead of emitting
`RXX`/`RYY`/`RZZ` directly.

Measured through the real, unmodified harness with a fifth engine added at the
`psf_compile(qc)` call site, so before and after are compared **in the same run**:

| Family | Qiskit L3 | TKET | PSF v6 (canonical) | **PSF v6 cx** | Hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deep2q` | 0.9047 | 0.9134 | 0.8652 | **0.9088** | 0.9089 |
| `multi_deep2q` | 0.0883 | 0.0852 | 0.0692 | **0.0837** | 0.0872 |
| `wide` | 0.0029–0.0048 | ← | ← | ← | ← |

<sub>On `wide` all five engines land in 0.0029–0.0048 with no PSF-specific effect
either way, as expected: PSF-Zero makes no changes to that family.</sub>

<sub>3 repeats, 3000 shots. Correctness re-verified unchanged at fidelity
1.000000000000 across CX/SWAP/iSWAP/Identity and 100 random SU(4) samples, before and
after the change.</sub>

`PSF_Zero_v6_cx` lands within noise of Qiskit L3, TKET and Hybrid on both families
where PSF-Zero's synthesis is active — closing essentially the whole deficit.

## Why `canonical` is still the default

`RXX`/`RYY`/`RZZ` is the **right** choice on hardware whose native two-qubit
interaction is itself an XX/YY/ZZ-type gate — trapped-ion and neutral-atom
Mølmer–Sørensen gates, for instance. This is a target-basis choice to make
deliberately per backend, not a universal default to flip. Pass
`entangling_basis="cx"` when targeting a CX/ECR-native machine.

## An independent reproduction, built from the source alone

Because the scripts that produced the original section-8 numbers are lost,
`experiment_fixed_compiler_fidelity.py` builds a from-scratch mirror-circuit fidelity
test directed only by the real `psf_compile.py` source, and applies one common final
backend-transpile at a naive level (1) and a good level (3):

| Family | Engine | `optimization_level=1` | `optimization_level=3` |
| :--- | :--- | :---: | :---: |
| `deep2q` | psf | 0.8725 ± 0.0093 | 0.9925 ± 0.0026 |
| `deep2q` | qiskit | 0.9193 ± 0.0050 | 0.9930 ± 0.0018 |
| `multi_deep2q` | psf | 0.1438 ± 0.0047 | 0.9641 ± 0.0047 |
| `multi_deep2q` | qiskit | 0.1834 ± 0.0052 | 0.9651 ± 0.0017 |
| `wide` | psf | 0.0405 ± 0.0042 | 0.9373 ± 0.0046 |
| `wide` | qiskit | 0.0438 ± 0.0076 | 0.9388 ± 0.0045 |

At the naive level `psf` trails by a real, stdev-exceeding margin on `deep2q`
(~4.5 points — closely matching the original 4.2–4.4) and `multi_deep2q`; at the good
level all three converge; on `wide` there is no psf-specific gap at either level. The
qualitative pattern and, for `deep2q`, the approximate quantitative size both
reproduce. Independently re-run unmodified on a second machine and Qiskit environment,
matching within run-to-run stdev.

## Limits

- The *original* scripts behind the first section-8 numbers were searched for across
  the working repository and not found. So it cannot be said with certainty that this
  exact mechanism, rather than some combination of it and something else, produced
  those specific numbers. What can be said: the mechanism is real, reproduces at
  matching scale on three independent fronts, and a validated fix exists and is
  confirmed against the real production code.
- The end-to-end test used a verified stand-in for the Rust core
  (`psf_zero_core_stub.py`, worst-case (1 − fidelity) = 8.88e-16 over 200 trials)
  because the supplied `.so` was a non-x86 binary.
- One noise model (`fake_sherbrooke`) for the simulator half. Real-hardware results
  are in [`real-hardware.md`](real-hardware.md).

## Files

- [`benchmarks/diagnose_native_gate_inflation.py`](../../benchmarks/diagnose_native_gate_inflation.py)
- [`benchmarks/diagnose_compile_for_hardware.py`](../../benchmarks/diagnose_compile_for_hardware.py)
- [`benchmarks/verify_compile_for_hardware_fix.py`](../../benchmarks/verify_compile_for_hardware_fix.py)
- [`benchmarks/test_improved_compiler_end_to_end.py`](../../benchmarks/test_improved_compiler_end_to_end.py)
- [`benchmarks/experiment_fixed_compiler_fidelity.py`](../../benchmarks/experiment_fixed_compiler_fidelity.py)
- [`benchmarks/psf_compile_patched.py`](../../benchmarks/psf_compile_patched.py),
  [`benchmarks/psf_zero_core_stub.py`](../../benchmarks/psf_zero_core_stub.py),
  [`benchmarks/test_psf_zero_core_stub.py`](../../benchmarks/test_psf_zero_core_stub.py)
- The patch: [`benchmarks/compile_for_hardware.patch`](../../benchmarks/compile_for_hardware.patch)
