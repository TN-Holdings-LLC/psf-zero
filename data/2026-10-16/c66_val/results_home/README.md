# C66-VAL results: the complete run (the owner's home machine, 2026-10-17)

The first complete run of [`../C66-VAL_preregistration.md`](../C66-VAL_preregistration.md) (SHA-256 `ef249a23...`),
made by `../run_c66_val.sh` on the owner's home machine (WSL), 2026-10-17 00:38:32-01:35:23 CEST. The three runs
started earlier in the authors' cloud died with the container and are not scored (Addendum 512). `times.txt` holds the
SHA-256 of every file the run used; they match the committed files.

- `R/`, `D/`, `R_*.log`, `D.log`: C63-VAL's parts R and D on c66 (C1-C4).
- `default_call.json`, `default_call.log`: c66's default call against release 2026-10-16.1 (C5).
- `walk.json`, `walk.log`: the 100,000-bind walk (C6).
- `c63a_score.txt`: `python data/2026-10-16/c66_val/c63a_score.py <run>/R <run>/D` (C1-C3: its A-V1 to A-V4; its
  A-S1 and A-S2 are reported only here).
- `score_c66.txt`: `python data/2026-10-16/c66_val/score_c66.py <run>` (C4-C6 and the reported durations and binds).

Scored in the authors' cloud from the zip of the run folder. Addendum 513 reads the results.
