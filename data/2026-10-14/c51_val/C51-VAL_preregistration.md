# C51-VAL: pre-registration (fixed 2026-10-14 03:51 CEST, before the run)

**Why.** C50-VAL (Addendum 480) refuted V6. 69 of 90 of c50's outputs carried rzz angles outside IBM's executable
range 0 < theta <= pi/2, because Qiskit's own stages inside the compiler emit rzz with any angle when the Target
offers it.

- Prototype `2026-10-14.c51` is c50 with item 86 revised: every circuit returned on a Target with rzz is folded
  (qiskit-ibm-runtime's FoldRzzAngle) and checked, and the re-synthesis is compared with the folded circuit.
- The predictions, script and scope are C50-VAL's, unchanged except that c51 takes c50's place. The arm keeps the
  name C50 in the script's output.

**Files fixed by SHA-256.**

| file | SHA-256 |
|---|---|
| `patches/psf_compile_c51_2026-10-14/psf_compile.py` | `556d312b493012eb310813ec8367e0c370ceb6df9316210b37d1473cc3e52f96` |
| `patches/psf_compile_c51_2026-10-14/make_c51.py` | `1d6deb612af7ff2646c6200d2c21caec0a2593767233d30894df919b1efe2a84` |
| `patches/psf_compile_c51_2026-10-14/c51_code.py` | `3afc7ba78caef644f0867ca420c5ae5138d91d1f3ae17bf586965c5a5d35523a` |
| `patches/psf_compile_c51_2026-10-14/test_c51.py` | `79cf7425401cbbe8a89c8a86cbfbfb0fbc8f357feb913182154a58a311e2f23e` |
| `benchmarks/c51_val.py` | `2d3b4d600011b7d49dbc23a59bd9bf2a21c90f62ca3f41c840912971cd272860` |

**Checks before the run.** 63 tests passed:

- c49's and c48's tests on c51 (54);
- `test_c51.py` (9), including `test_every_output_executable`, which guards V6 with the re-synthesis on and off.

**Seen before the run.** C50-VAL's numbers. Folding adds single-qubit pulses where angles were negative, so durations
can only stay or grow against c50's.

**Predictions:** V1-V7 exactly as in C50-VAL's prestatement.

| ID | prediction |
|---|---|
| V1 | Exact (<= 20 active qubits, 1e-6) and no refusal. |
| V2 | C51 <= 1.02 x C49 on >= 95% of cells. |
| V3 | Ising: geometric mean <= 0.70 and every cell <= 0.95. |
| V4 | C51 <= 1.02 x QK3F on >= 90% of cells. |
| V5 | Same output on a second call (15/15). |
| V6 | Every output native to the fractional Target, with every rzz angle in (0, pi/2]. |
| V7 | Compile time C51 <= 1.5 x C49 + 2 s on >= 90% of cells. |

**Rule.** If V1, V2, V3, V5 and V6 are confirmed, c51 is the C50 arm of FRAC1, the hardware test.
