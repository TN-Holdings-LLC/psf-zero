# C53-VAL: pre-registration (fixed 2026-10-14 05:21 CEST, before the run)

**What is tested.** Prototype `2026-10-14.c53` is c52 plus the workplace's items 87-88 (Addendum 483), merged at
home by `make_c53.py`. Only the version strings and the changelog position change; the workplace's code is unchanged.

- **Item 87.** A chain Hamiltonian is built by formula as a brickwork, on a path placed by structure.
- **Item 88.** With rzz and rx in the Target, its terms are emitted as rzz and rx, with negative angles in a Pauli X
  frame that is undone inside the circuit.

Offline only.

**Files fixed by SHA-256** (in this order: psf_compile.py, make_c53.py, c53_87_88_code.py, test_87_88.py,
benchmarks/c53_val.py):

    81ef9beca7e0c5a6f2ff90813505d27c0df22328bbae223c9bc7eb659b8af1dd  patches/psf_compile_c53_2026-10-14/psf_compile.py
    0260bd95d9723c9e071c6ba30d3f43b0c73a88136510cc6c72170e0cae5f258d  patches/psf_compile_c53_2026-10-14/make_c53.py
    adb015e5a3f4f4a0cc6aa08b11d416cf8be73e0d59680161bcb425fcfcc0f1ce  patches/psf_compile_c53_2026-10-14/c53_87_88_code.py
    f848bcc7c97e6e2d8aa9421d62a9aa47ff8fcc7b9ae06cc78c9137049f89b5c1  patches/psf_compile_c53_2026-10-14/test_87_88.py
    7181ea85eb77e7f60770574d20272dddb20d6bedab246095d50bbc647e813e5c  benchmarks/c53_val.py

**Checks before the run.** 86 tests passed:

- the workplace's 13 (test_87_88.py, with c52 as the inner compiler compared against);
- c52's, c51's, c49's and c48's on c53.

a16's 10 tests also pass with c53.

**Seen before the run.** The workplace's own evaluation on c51 + 87-88 (TFIM chains at 0.66 / 0.45 of c51 on FakeFez
/ FakeMarrakesh with fractional gates; Addendum 483). None of C53-VAL's cells has been run on c53.

**Run.** `python benchmarks/c53_val.py data/2026-10-14/c53_val/run`.

- **Scope.** C52-VAL's 24 tests and 5 devices.
- **Arms.**
  - QK3: Qiskit level 3.
  - C52 and C53: no fractional gates.
  - C52F and C53F: the fractional Target.

**Predictions.**

| ID | prediction |
|---|---|
| V1 | C53 and C53F exact (<= 20 active qubits, 1e-6), with no refusal. |
| V2 | C53 <= 1.02 x C52 duration on >= 95% of cells. |
| V3 | C53F <= 1.02 x C52F on >= 95% of cells. |
| V4 | Ising chains on the fractional Target: C53F / C52F geometric mean <= 0.85. |
| V5 | C53F gives the same output on a second call: qaoa_n20_p2, tfim_n16, ba_n20 and clf_n6_L5 on every device. |
| V6 | Every C53F output native, with every rzz angle in range. |
| V7 | Compile time: C53F <= 1.5 x C52F + 2 s on >= 90% of cells. |

**Rule.** If V1, V2, V3, V5 and V6 are confirmed, c53 becomes the base. TIME1 and FRAC1 tonight are unchanged (c49
and c51).
