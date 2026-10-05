# Workplace results: MODEL-RO (pre-registration `model-ro-preregistration-2026-10-05.md`). P0 FAILED on a threshold set on the wrong scale (one A10 output with noiseless total variation 4.8e-6 against 1e-6; its state infidelity is 3e-11), so nothing is scored. Described: a10 halves the classical infidelity of sampled model-written circuits on FakeTorino (0.062 → 0.032) and cuts it by 70% on FakeKingston (0.058 → 0.017), by moving the measured qubits to better readout, level with or slightly ahead of Qiskit level 3 (2026-10-05)

**Status: results of the workplace pre-registration of 2026-10-05.**

- **Lock:** Project save time, before the run (started 05:15 UTC).
- **Scoring:** by the locked `ai10_eval.py score`.
- **Arithmetic re-check:** `ai10_verify.py`, which recomputes the table from the raw JSON. It was written after the
  locked score had been seen, so it is not a blind check.
- **Setting:** workplace sandbox, 2 processes; 153 circuits × 3 devices × 3 arms; 7 minutes.

## 1. P0 failed: no prediction is scored

**What failed.** Every output's noiseless distribution was required to be within total variation 1e-6 of the ideal.
One output exceeded it: A10, FakeKingston, circuit 141 (3 qubits), with 4.8e-6. All other outputs of all arms were at
most 6.1e-8.

**The cause is the threshold's scale, not the output.**

- Total variation is first order in the amplitude error; a state infidelity is second order. The same output has a
  state infidelity of 3.0e-11 (computed after the run with the READOUT check).
- The deviation comes from the AI front end's polish, which re-synthesises blocks with the PSF core at its declared
  tolerance (`tol=1e-5`, since a1). a10 merely chose a different candidate there than a9 did (a9's output: total
  variation 6.1e-8, state infidelity 4e-15).
- This is the same design error as DEPTH's P0 (a z difference) and Addendum 301 (a threshold that did not admit an
  arm's own tolerance).
- **The lesson, now stated for all workplace harnesses:** exactness checks are state infidelities, with a bound that
  admits every arm's declared tolerance (1e-6 does). Linear quantities (z, total variation) are never compared with
  the same 1e-6.

**What follows from the pre-registration.** The verdict lines the locked score printed are not results:

| M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|
| CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |

## 2. Descriptive results

| device | arm | classical infidelity | summed measure error | 2q | median compile s |
|---|---|---|---|---|---|
| FakeTorino | A9 | 0.0625 | 0.0843 | 5.82 | 0.639 |
| FakeTorino | A10 | **0.0320** | 0.0316 | 5.82 | 0.622 |
| FakeTorino | L3TM | 0.0324 | 0.0315 | 6.28 | 0.020 |
| FakeKingston | A9 | 0.0576 | 0.0888 | 5.88 | 0.773 |
| FakeKingston | A10 | **0.0172** | 0.0179 | 5.89 | 0.760 |
| FakeKingston | L3TM | 0.0177 | 0.0176 | 6.30 | 0.021 |
| FakeAuckland | A9 | 0.0341 | 0.0289 | 5.80 | 0.215 |
| FakeAuckland | A10 | 0.0333 | 0.0265 | 5.80 | 0.218 |
| FakeAuckland | L3TM | 0.0356 | 0.0239 | 6.29 | 0.013 |

**Ratios:**

| ratio | FakeTorino | FakeKingston | FakeAuckland |
|---|---|---|---|
| A10/A9 classical infidelity | 0.512 | 0.299 | 0.975 |
| A10/L3TM | 0.987 | 0.974 | 0.936 |

**Per circuit:** A10 is at or below A9 in 87.6% (FakeTorino), 83.7% (FakeKingston) and 94.1% (FakeAuckland) of
circuits.

**Compile time:** unchanged (A10/A9 1.00).

## 3. Reading

- **For model-written circuits that are sampled, readout dominates on uneven-readout devices.**
  - a9's estimate did not see it, and placed the measured qubits where the summed measure error was 0.084-0.089.
  - a10 places them at 0.018-0.032, as level 3 does through its averaged error map.
  - The output distribution's infidelity halves on FakeTorino and falls by 70% on FakeKingston.
  - On FakeAuckland, where readout is even (p90 0.0125), the change is small (-2.5%).
- **a10 keeps a9's gate advantage over level 3** (5.8 against 6.3 two-qubit gates). It is level with level 3 or ahead
  of it on all three devices.
- **This is the same blind spot as the release's (READOUT, c13).** Both fixes change nothing for circuits without
  measurements (tested).
- **Caution:**
  - A10's estimate and the readout model are the simulator's own (Aer's asymmetric assignment errors from the same
    snapshot).
  - On hardware, readout errors drift and are correlated across qubits.

## 4. Consequences (proposals; adoption is the owner's)

1. **a10 (item 15) as a candidate AI front end** on top of a9, with c13 underneath.
2. **A re-test with the threshold corrected**, on fresh circuits. There is no other model-written set, so use the
   synthetic GHZ/W/Dicke/QFT families, or new pod output. **It should be pre-registered at home.**
3. **In the vLLM loop (harness v11):** the model's circuits are scored by sampling, so a10's gain should show directly
   in the scored fidelity there. Not tested.

## 5. Files (`model_ro/` in the handoff)

`env.txt`, `progress.txt`, logs, the three `ai10_<device>.json`, `score.md`, `score_log.txt`, `verify.txt`; `dry/`;
`diag141.py` and `diag141b.py` (the P0 diagnosis).
