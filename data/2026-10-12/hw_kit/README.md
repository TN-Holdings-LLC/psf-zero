# The kit of the real-device session of 2026-10-12 (Addenda 212, 454, 455)

Run on the owner's home machine (WSL2) from the repository root, in three steps:

- `81_hw_prep.sh` (VPN on, no IBM job): checks the kit, places prototypes c45 and c47 in `patches/`, runs Stage 2's
  local checks and HW1's rehearsal on FakeKingston.
- `82_hw_run.sh` (VPN off; no git, no pip): Stage 2 attempt 2 (Addendum 211), then HW1 + PSF-XOR (one job), then the
  results.
- `83_hw_record.sh` (VPN on, no IBM connection): scores HW1 offline and hashes every result file.

`hw1_real.py` here is [`benchmarks/hw1_real.py`](../../../benchmarks/hw1_real.py). `qasm/` holds four QAOA files from
Benchpress (Apache License 2.0), `qaoa_barabasi_albert_N{14,16,18,20}_3reps.qasm`, unchanged. `SHA256SUMS` lists the
kit as delivered; this README was added when committing.
