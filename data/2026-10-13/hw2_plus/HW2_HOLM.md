# HW2: one multiplicity-corrected reading of all nine predictors, fixed before HW2 runs (2026-10-13)

**Fixed:** 2026-10-13 09:52:30 CEST (Europe/Zurich), before HW2's job exists. `benchmarks/hw2_holm.py` SHA-256 `8b34aed5099d0a22d3acd0400ffeb0c83236be3cc2312ed1ab58a5e026bcb028`.

**Why.** Nine predictors will be scored on the same seven pairs:

- ESP and R3 (Addendum 460);
- NS2 (Addendum 462);
- P1-P4 (Addendum 466);
- P1c and P2c (Addendum 468).

With that many, one of them can look good by chance. The workplace was asked for a multiplicity rule (request 12),
but it was not delivered, so it is fixed here instead.

**Rule.**

- For each predictor: the Spearman correlation of the predicted and measured PSF / QK3 ratios on HW2's seven reading
  pairs, and its one-sided exact permutation p. Both are computed with the workplace's fixed
  `hw2_within_test.py` (Addendum 467), imported unchanged and hash-checked.
- Holm's step-down correction over all nine p values. A predictor with missing data counts with p = 1.
- **A predictor "works on HW2" only if its Holm-adjusted p is <= 0.05.**

**What that demands.**

- With seven pairs, the best of nine predictors needs a one-sided p <= 0.0056, which means a Spearman correlation of
  about +0.93 or more. Exact permutation p values: +0.93 gives 0.0014; +0.89 gives 0.0062.
- Seven pairs cannot prove a predictor unless it ranks them almost perfectly. Anything weaker is reported as a
  lead, not as evidence.

**Kept beside it, unchanged:** every tool's own verdicts (R1-R6, NS2a-d, Q1-Q4, Q2c, the workplace's "strong / weak /
none"). If this reading and one of them disagree, both are reported.

**Run after the other tools:**

    python benchmarks/hw2_holm.py data/2026-10-13/hw2 <ns2>/ns_hw2.jsonl <pred>/hw2_predictors.jsonl <predc>/hw2_predictors_c.jsonl --out <file>.json

**Checked:**

- on HW2's rehearsal outputs (nine rows; missing tools count with p = 1);
- Holm on a known case: [0.01, 0.04, 0.03, 0.5] gives [0.04, 0.09, 0.09, 0.5].
