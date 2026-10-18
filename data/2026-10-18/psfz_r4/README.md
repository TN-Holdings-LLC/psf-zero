# psfz r4: exploratory run on PSFZ-VAL's inputs (2026-10-18)

Not a pre-registered run: r4 was built after PSFZ-VAL (Addendum 520) by looking at its failures, so these inputs are
seen. Addendum 521 has the reading.

- `results/rows_a.jsonl` (CZ devices: FakeAachen, FakeBoston, FakeKingston, FakePittsburgh, FakeFez, FakeMarrakesh,
  FakeTorino): `data/2026-10-17/psfz_val/run_psfz_val.sh`'s process a, psfz built from the sources whose SHA-256 are
  in `results/times_a.txt` (r4's placement, before the ECR form choice; process b of that run was stopped and its
  rows are not kept). 2026-10-18 01:53-02:19 CEST, the authors' cloud.
- `results/rows_b.jsonl` (ECR devices: FakeKyiv, FakeOsaka, FakeKawasaki, FakeQuebec, FakeSherbrooke, FakeBrisbane):
  `psfz_val.py` with a build of r4 that also read two exploratory environment variables (unset during the run).
  2026-10-18 02:02-02:27 CEST. That build was rebuilt once during the run with the same code for ECR (no input was
  refused).
- **Both checked against the committed r4:** `scripts/same_out.py` compiled every one of the 2,860 inputs with the
  build used for the rows and with the committed r4 (`rust/psfz`); the reports (operations, layout, measurements;
  timings left out) were identical on 1,540 of 1,540 CZ inputs and 1,320 of 1,320 ECR inputs.
- `results/score.txt`: `data/2026-10-17/psfz_val/score_psfz_val.py` on both files.
- `scripts/`: `explore.py` (psfz against the release on chosen inputs, both placements), `windows.py` (psfz's
  duration for every window and orientation of the longest snake piece; needs a build that reads `PSFZ_WALK`, an
  exploratory patch not committed), `diag.py` (coupler durations along given paths), `same_out.py`.
