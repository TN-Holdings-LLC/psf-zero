# psf_compile "2026-10-18.2": review of an outside proposal (2026-11-13, home) -- not adopted

An outside AI proposed a patch to `psf_compile.py` 2026-10-18.1 (lattice cache, item 104's ECR rewrite in closed
form, and smaller changes), described as bit-for-bit compatible. `apply_patch_10_18_2_fixed.py` is its patch script
with one syntax error fixed at home (a triple quote inside a string, line 81); its diff form did not apply (wrong hunk
counts). The patched file's SHA-256 is in `patched_psf_compile.sha256`.

Comparison (`compare.py`; C68-VAL's 25 tests, each as built and with two random draws; FakeFez, FakeMarrakesh,
FakeSherbrooke, FakeBrisbane; with psfz and without it): 477 of 600 records (the run's time limit stopped it during
FakeSherbrooke without psfz; FakeBrisbane without psfz not reached). `cmp.jsonl`.

- CZ devices (FakeFez, FakeMarrakesh), with and without psfz: 300 of 300 outputs identical (C68-VAL's fingerprint).
- ECR devices: 48 outputs differ (15 + 15 with psfz, 18 without). "Bit-for-bit compatible" does not hold there.
  `eq2.py`, `eq2.jsonl`: every differing output has the same ECR, sx and x counts and 4,115 more rz in total (25,279
  -> 29,394; rz is virtual on IBM devices); outcome distribution from |0...0> equal to the old output within 5.5e-15
  and to the input within 2.2e-14 (45 cases up to 20 active qubits); state fidelity from three random product inputs
  1 within 4e-14 (45 cases); the three 40-qubit cases were not checked (too large for a state vector).
- Compile time: 0.89-0.99 x the old (sums over each device's calls).

Not adopted: on CZ devices nothing changes but about 5-10 % of compile time; on ECR devices the output changes (more
rz, same pulses) with nothing to fix.
