# PSF-Zero proof of concept: placing circuits without the calibration map

**In one sentence.** psf-os places a ring circuit on an IBM heavy-hex device without reading any calibration
quality number. It tests the device with its own short circuits, avoids the paths it found dead, and attaches to every
result a report of how far that result can be trusted.

This page is for teams considering a proof-of-concept evaluation. Researchers will find the full records in the
papers and in TOQB, the test bench (links below).

**Status.** This is a draft entry point; the software is pre-release.
[psf-os](https://github.com/TN-Holdings-LLC/psf-os) (AGPL-3.0) is not yet public. The kit of September 2026 (gate
synthesis and layout search, Papers 1 and 2) is kept unchanged in [`v1_2026-09/`](v1_2026-09/).

## Why

- **A device's quality varies along its paths.** On `ibm_kingston` (30 October to 1 November 2026), the same 18 qubits
  of a 40-qubit ring read near random or inverted in every run of the same circuit. Seventeen of those qubits were
  healthy on other rings, and the calibration data showed nothing unusual there (Paper 5).
- **A layout chosen from calibration numbers cannot avoid what the calibration does not show.** psf-os measures the
  device itself, with a 24-circuit mirror-echo scan of about 10 seconds, and keeps a dated list of the dead paths it
  found.

## What was measured on devices

All numbers come from declared tests (each fixed by SHA-256 and time before its run). Records:
[Paper 5](https://doi.org/10.5281/zenodo.23827536) and the TOQB `results/` folders named below. Paper 6 (draft)
collects them.

| device, day | ring of 40 qubits: usable sites, standard ring -> ring placed around the list | ring of 72 qubits |
|---|---|---|
| ibm_kingston, 31 Oct | 22 -> 40 | 38 -> 72 |
| ibm_kingston, 1 Nov | 22 -> 40 | 39 -> 72 |
| ibm_fez, 1 Nov | 29 -> 40 | 36 -> 63 |
| ibm_marrakesh, 1 Nov | 40 -> 40 (nothing to avoid) | 67 -> 72 |

<!-- src: TOQB results/coverage-days-2026-10-31-day1/coverage.json, results/coverage-days-2026-11-01-day2/coverage.json,
     results/two-devices-2026-11-01/{fez,marrakesh}/avoid/coverage.json -->

- **The headline number and the essential one.** The headline is "usable sites 22 -> 40". The essential number: on
  the sites that were already usable, the signal moved only from 0.41 to 0.44. The gain is in *width* (how much of the
  device can carry a computation), not in a better signal everywhere. In three of the eight rows the mean signal on
  usable sites fell (kingston 31 Oct, 72 qubits: 0.38 -> 0.35; fez, 72 qubits: 0.27 -> 0.22; marrakesh, 40 qubits:
  0.27 -> 0.24). On fez the 72-qubit ring placed around the list still had 9 unusable sites.
  <!-- src: coverage day 1 n40 mean_usable 0.410 -> 0.441; n72 0.384 -> 0.346 (day 1);
       two-devices fez n72 0.269 -> 0.217 (usable 63 of 72), marrakesh n40 0.270 -> 0.241;
       Paper 6 numbers.json width.* -->
- **A forward computation** (not a mirror) on a ring where nothing was dead (`ibm_marrakesh`, 72 qubits, 12 layers):
  the mean site signal was 0.776 on the standard ring and 0.816 on the avoiding ring (+0.040 ± 0.002). The headline is
  "per-bit right 88.8 % -> 90.8 %". The essential number: all of the gain sits in the 10 qubits that the avoiding ring
  replaced (0.615 -> 0.866). On the 62 shared qubits the two rings read the same (0.802 against 0.808).
  <!-- src: results/qca-m3b-2026-11-01/score_S.json, score_V.json, README.md -->
- **Limits.** Three devices over a few days; one family of circuits (rings placed by one formula). The listed defects
  are paths, and their cause is not claimed. A first forward test on `ibm_kingston` was too deep to read and is
  reported as inconclusive. A repair by frame flips helps only mirror-type circuits. Avoiding is the default.

## Try it in five minutes (a fake device, no IBM account)

```bash
git clone https://github.com/TN-Holdings-LLC/psf-os      # pre-release: not yet public
pip install -e "psf-os[sim]"
git clone https://github.com/TN-Holdings-LLC/psf-zero
python psf-zero/POC/quickstart_os.py                      # writes ./quickstart_out
```

**This is a simulation with defects planted by the script, not a device result.** The script says so on its first and
last lines. It plants a two-qubit error on two couplers of IBM's FakeFez, scans twice, lists the planted paths, and
solves two practice problems with exact answers on the standard ring and on the avoiding ring. The problems are a
cluster-state parity and a random Clifford circuit read through its stabilizers. The scan runs at the Clifford point of
the echo so that a laptop can simulate 72 qubits; the device scan uses the full echo.

One run on a 2-core machine (92 s; the same output at home in 88 s), output unedited:

```
SIMULATION with planted defects on FakeFez -- not a device result
1. device record: fake_fez, 176 couplers, 7 failed, 0 failed qubits (no quality number read)
2. planted (known to this script only): couplers [(15, 19), (75, 59)], two-qubit depolarizing 0.15
3. two scans (24 circuits each); LISTED paths: [[14, 15, 19, 35], [74, 75, 59, 55]]
4. practice problems (exact answers recorded at build time), standard ring against avoiding ring:
   P2 cluster parity, n = 40: fraction right 0.920 (standard ring) -> 1.000 (avoiding ring); sentinel on the avoiding ring: CLEAN; masked couplers [[15, 19], [59, 75]]
   P1 random Clifford (k = 4), read stabilizers, n = 40: fraction right 0.980 (standard ring) -> 1.000 (avoiding ring); sentinel on the avoiding ring: CLEAN; masked couplers [[15, 19], [59, 75]]
5. done in 92 s on this machine. SIMULATION with planted defects on FakeFez -- not a device result.
```

A perfect 1.000 here only means the planted errors were the only noise and were avoided; on a device every ring has
its own noise, as the numbers above show.

## Try it on a device (an IBM Quantum account)

1. `psf-os fetch` the device's coupling graph and failed elements.
2. `psf-os plan` your ring job. With no scan of the day, it builds the scan job first (24 circuits, about 10 s by
   the time model below; 9 s billed in our runs).
3. Submit with TOQB's `s2rc_device submit`. The job says "stay", so it is never moved to another device.
4. `psf-os scan-ingest` the scan result, then `psf-os plan` again: the job is placed around the listed paths, with
   a 4-circuit sentinel on the same ring.
5. `psf-os result` returns the score and the sentinel verdict: CLEAN, WEAK or DEAD, with the sites.

The time model is 0.036 s per circuit + 0.00027 s per shot + 3 s per job. A 40-qubit job of 8 circuits with its
sentinel at 1,000 shots comes to about 6.7 s.

## A proof-of-concept evaluation

The request form (`psf_zero_poc_request_form.docx`), the mutual NDA template (`psf_zero_nda_template.pdf`) and the
report template (`psf_zero_poc_report_template.docx`) are here unchanged. They were written for the September kit
(gate synthesis and layout search). A version for psf-os will follow. Contact: love.os.architect@proton.me.

## Papers

- Paper 5: Readable Depth Without the Calibration Map (doi:10.5281/zenodo.23827536).
- Paper 4: Folding a Ring onto Heavy-Hex (doi:10.5281/zenodo.23826764).
- Paper 3: When a Quantum Compiler Looks Like a Better Learner (doi:10.5281/zenodo.23826596).
- Paper 6 (draft): What Sets the Readable Depth of a Placed Ring, and How Far a Self-Checking Layer Pushes It.

AGPL v3 (psf-os, psf-zero). See `LICENSE` in each repository.
