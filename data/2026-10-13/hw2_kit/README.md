# The kit of HW2 (Addendum 460; results in Addendum 461)

Delivered as `hw2_kit_2026-10-13.zip` (sha256 `afaca9deb4a16c6ceb419d91deb5537a4380934ca5e5093b7be3dd72f277d957`)
and run from its own folder on the owner's home machine (WSL2), from the repository root, in three steps:

- `91_hw2_prep.sh` (VPN on, no IBM job):
  - checks the kit against `SHA256SUMS`;
  - places prototypes c44, c45, c47 and c48 in `patches/` (a file already there must be identical);
  - runs c48's tests and HW2's rehearsal on FakeFez.
- `92_hw2_run.sh` (VPN off; no git, no pip): prepare (device rule and manifest before submission), submit one job,
  fetch.
- `93_hw2_record.sh` (VPN on, no IBM connection): scores offline and hashes every result file.

The kit's `patches/` folder is not repeated here: its files are the repository's
[`patches/psf_compile_c44_2026-10-11/`](../../../patches/psf_compile_c44_2026-10-11/),
[`psf_compile_c45_2026-10-12/`](../../../patches/psf_compile_c45_2026-10-12/),
[`psf_compile_c47_2026-10-12/`](../../../patches/psf_compile_c47_2026-10-12/) and
[`psf_compile_c48_2026-10-12/`](../../../patches/psf_compile_c48_2026-10-12/), byte for byte (their hashes are in
`SHA256SUMS`). `qasm/` holds four QAOA files from Benchpress (Apache License 2.0), unchanged. This README was added
when committing.
