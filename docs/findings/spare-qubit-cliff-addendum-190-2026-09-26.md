# Addendum 190 -- The faster polishing step keeps Qiskit-level precision and brings PSF-Zero's cliff compile time to 0.08 s (about 160x faster than Qiskit's default); removing the duplicate `psf_compile.py` exposed that scripts under `benchmarks/` depended on it, so it becomes a redirect to the root file (2026-09-26)

**Pre-registered in**:
`spare-qubit-cliff-addendum-189-preregistration-2026-09-26.md`. Run on WSL2
(home). Logs received as files.

## 0. In one line

W1-W5 hold: maximum per-call error 6.18e-14, 36 tests pass, drift after
10 laps 7.27e-13, median compile time at spare 0 0.0805 s (bound 0.10 s;
0.135 s with the first polishing version, 0.061 s before any polishing),
Qiskit's values unchanged. **W0 holds only in part**: the scripts load the
root `psf_compile.py` (VERSION 2026-09-26.2), but `import psf_compile` from
inside `benchmarks/` fails once the duplicate there is removed -- the
assumption in Addendum 189 that the editable install makes the module
importable from any directory was wrong in this environment.

## 1. A first attempt that did not test the new version

`git rm benchmarks/psf_compile.py` refused (the file had local
modifications: Addendum 188 had overwritten it with the first fixed
version), so that copy stayed and the loaded-file check printed
`benchmarks/psf_compile.py 2026-09-26`. The numbers from that run match
Addendum 188's (median 0.135 s) because they are Addendum 188's code. The
check caught the mismatch before any result was read; the run is recorded
(`fix_validation_v2.txt`) and not scored. Forced removal
(`git rm -f`) followed.

## 2. The valid run (`fix_validation_v2b.txt`)

Header: root `psf_compile.py` 49,442 bytes, SHA-256 `c6621f71...`; with the
scripts' own import order, `LOADED .../psf_compile.py 2026-09-26.2`; from
inside `benchmarks/`, `ModuleNotFoundError`.

| Prediction | Result |
|---|---|
| W0: one copy; scripts load the root file | **PARTLY** -- scripts load the root file; `benchmarks/` cannot import it at all |
| W1: max PSF-Zero error over 60 pairs <= 1e-13 | **CONFIRMED** -- 6.18e-14 (median 6.71e-15; Qiskit median 8.72e-15, max 1.81e-13) |
| W2: 36 tests pass | **CONFIRMED** |
| W3: PSF-Zero max per-pair distance <= 1e-12 at every lap | **CONFIRMED** -- 6.21e-14 at lap 1, 7.27e-13 at lap 10 |
| W4: spare 0, 10/10 within 1 s and median <= 0.10 s | **CONFIRMED** -- 10/10, median 0.0805 s |
| W5: Qiskit's per-lap values identical | **CONFIRMED** |

At spare 8 PSF-Zero's median is 0.046 s (0.095 s with the first polishing
version; 0.028 s before polishing; Qiskit's default 0.15 s).

## 3. The duplicate, resolved

The scripts under `benchmarks/` had relied on the second full copy there.
`benchmarks/psf_compile.py` is replaced by a redirect of about ten lines
that holds no code of its own: it loads the root `psf_compile.py` in its
place, so there is exactly one implementation and `psf_compile.__file__`
always names it. Checked in isolation: imported from inside `benchmarks/`,
with the scripts' import order, and with `from psf_compile import ...`, it
yields the root file in all three cases.

## 4. Where PSF-Zero stands after Addenda 185-190

| | before (VERSION 2026-09-21) | now (VERSION 2026-09-26.2) | Qiskit default |
|---|---|---|---|
| worst per-call error (60 pairs) | 1.76e-11 | 6.18e-14 | 1.81e-13 (its own decomposer) |
| drift after 10 laps | 6.47e-10 | 7.27e-13 | ~3-5e-13 |
| cliff compile time (median) | 0.061 s | 0.0805 s | 12.9 s |

The remaining drift grows about 7.4e-14 per lap, roughly ten times Qiskit's
rate; at that rate 1e-12 is crossed near lap 14. The cost of the precision
fix is about 1.3x in compile time.

## 5. Files

| File | What it is |
|---|---|
| `fix_validation_v2.txt` | the first attempt (loaded the old copy; not scored) |
| `fix_validation_v2b.txt` | the valid run |
| `psf_compile.py` | VERSION 2026-09-26.2 (SHA-256 `c6621f71...`) |
| `benchmarks/psf_compile.py` | the redirect to the root file |
