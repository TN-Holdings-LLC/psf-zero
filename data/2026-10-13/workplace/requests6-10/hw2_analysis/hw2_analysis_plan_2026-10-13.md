# HW2 analysis plan of the workplace (request 8 of 2026-10-13), fixed before HW2's results exist

Fixed: see the time and SHA-256 at the end; written before HW2's job exists (HW2 is planned for about 20:00 JST).

**What it is.** `hw2_within_test.py` applies the framework of the workplace's ESP check of HW1 (within-test pairs,
exact permutation tests, Bayesian probabilities, a Bayes factor) to HW2. It does not replace any verdict of HW2
(R1-R6, S1-S4, `hw2_real.py score`) or HW2-PLUS (Q1-Q4, `hw2_predictors.py`); it states, in one pre-fixed way, how
strong each predictor's evidence is.

**Inputs.** `hw2_score.json` and `hw2_manifest.json` from `hw2_real.py` (keys "part|test|arm" in `F`, with `f`,
`reps`, `esp`, `idle`; `entries[].submitted`), and `hw2_predictors.jsonl` from `hw2_predictors.py` for P1-P4.

**Predictors.** ESP and R3 (the idle-aware estimate of HW2's section 2) from the score file; P1-P4 from the
predictors file (first element of each value).

**Scopes.** A (primary): HW2's seven reading tests, PSF vs QK3. B (reported): every part R pair with F > 0 except
ghz_n12 and tfim_n16, plus every part S pair (S1 vs S0).

**Statistics per predictor and scope.** Winner agreement and a Beta(1+h, 1+n-h) hit rate; Spearman correlation of
predicted and measured ratios (average ranks) with an exact permutation test; the within-test correlation of log
ratios through the origin with an exact sign-flip test; P(rho > 0), P(rho > 0.2) and BF+0 from a flat prior on the
Fisher z of rho (standard error 1.06 / sqrt(n - 3), uniform prior on rho > 0 for the Bayes factor); the share of the
F-ratio variance in scope A that is not repeat noise.

**Self-test.** Run on HW1's data rewritten in HW2's format (`fake_hw2/`, R3 and P3 set to a14's exact estimate):
scope A gives ESP 3 of 7, Spearman -0.18, and the a14 estimate 5 of 7, +0.75 (permutation p 0.033, BF+0 3.2), the
numbers already reported for HW1. The self-test files are included; they are not HW2 data.

**How to run after HW2.**

    python hw2_within_test.py <hw2 output dir> <hw2_predictors out dir>/hw2_predictors.jsonl --out hw2_within_test.json

**Reading, fixed now.** Several predictors are analysed at once; the record must say how many (six here: ESP, R3,
P1-P4). A predictor's evidence is called "strong" only if, in scope A, its one-sided permutation p is <= 0.05 and
BF+0 >= 3; "weak" if one of the two holds; otherwise "none". With seven pairs, a one-sided permutation p of 0.05 or less needs
a Spearman of +0.68 or more.

**Fixed:** 2026-10-13 07:54:18 CEST (14:54 JST). `hw2_within_test.py` SHA-256 `889e1fc3b60cf8aacea0ec892b007cb246a42491ed029bc04700e23995c5f28f`.
