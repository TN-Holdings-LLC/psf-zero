# ESP check of HW1 (request 6 of 2026-10-13): script, inputs and how they fit together

Reproduces `docs/findings/esp-random-check-2026-10-13.md` (within-test analysis and its "Addition" table).

## Files

| file | what |
|---|---|
| `esp_random_check.py` | the analysis (within-test correlation, exact permutation test, repeat-noise share, probabilities) |
| `req1_out.json` | a14's estimates for every HW1 circuit (written by `req1_run.py`, request 1) |
| `req1_run.py`, `est_lc.py`, `hw1_check.py` | how `req1_out.json` was made (Aer reduced states fed to a14's formulas; Target = FakeKingston structure + the day's CSV) |
| `req1_score.py` | the seven-pair table of request 1 from `req1_out.json` |
| `esp_random_check_output.txt` | the output of the run below |

## How `req1_out.json` maps onto HW1

- Key: `"<test>|<arm>"` exactly as in `data/2026-10-12/hw1/hw1_score.json` `F` (e.g. `ba_n14|PSF`).
- `aer`: [gate + readout term, idle term] of a14's estimate (exact reduced states); the estimate used is
  `exp(-(aer[0] + aer[1]))`.
- `rule`: the same with rule R6/16; `np`: a14's own state-vector code (circuits of at most 22 active qubits, null
  otherwise); `esp`, `F`: copied from HW1's manifest and score; `active`: active qubits; `rule_stats`: the rule's counts.
- Only repeat 1 of each circuit is used for the estimates (both repeats are the same circuit).

## Inputs from the repository (`data/2026-10-12/`)

`hw1/hw1_score.json` (F, its two repeats, ESP), `hw1/hw1_manifest.json` and `hw1/hw1_circuits.qpy` (the committed
circuits, for the exposure), `hw_analysis/ibm_kingston_calibrations_2026-10-12T11_15_24Z.csv` (T2 for the exposure).

## Run (from anywhere)

    python esp_random_check.py <repository root> req1_out.json

Exclusions, as in Addendum 455: tfim_n16 (F > 1) and ghz_n12 (identical arms). 18 points in 7 tests.
The probabilities are computed with unrounded correlations, so they can differ in the third digit from the table in
the findings note (which used r rounded to two digits).
