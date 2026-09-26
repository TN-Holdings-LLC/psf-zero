# Addendum 188 -- Correction to Addendum 187 (its validation never loaded the fix: the scripts imported an older `benchmarks/psf_compile.py`), and the valid validation: PSF-Zero's per-call error falls from 1.76e-11 to 6.2e-14 and its drift over 10 laps from 6.47e-10 to 7.2e-13, at a 2.2x cost in compile time (2026-09-26)

**Pre-registered in**:
`spare-qubit-cliff-addendum-186-preregistration-2026-09-26.md` (predictions
unchanged). Run on WSL2 (home). Log received as a file:
`fix_validation_run2.txt`.

## 0. In one line

Addendum 187's run did not test the fix. Every script used there puts
`benchmarks/` ahead of the working directory on its import path, and the
repository has its own `benchmarks/psf_compile.py` (the old VERSION
2026-09-21), so the scripts imported the old file while the one-line
version check -- which does not reorder the path -- reported the new one.
Its conclusions ("the core's parameters were already accurate; the error
enters at circuit construction") are withdrawn. Run again with the fixed
file in both places and the loaded file recorded, the fix works: V1, V2,
V3 and V5 hold; V4 (median compile time <= 0.12 s) fails at 0.135 s.

## 1. How the mistake was found

`diag_construction.py`, run after Addendum 187, showed the core's own
parameters, rebuilt by `_reconstruct`, off by up to 1.76e-11 (`d_model`),
while the middle CX part (`d_core_cx`, max 4.9e-14) and the local
rotations (`d_local`, max 3.6e-16) were accurate. A parameter error of
1.76e-11 would have triggered the polishing step, which contradicted
Addendum 187. Checking the import with the scripts' own path order printed
`benchmarks/psf_compile.py 2026-09-21`.

The same run confirmed that `psf_compile.py`, `benchmarks/psf_compile.py`
and the backup `psf_compile_2026-09-21.py` all had SHA-256 `66a705ea...`
before the fix was installed -- every earlier experiment ran the same code,
whichever copy it loaded; no earlier result is affected.

## 2. The valid run

Setup, recorded at the top of the log: the fixed file (47,606 bytes,
SHA-256 `a1a20781...`) in both `psf_compile.py` and
`benchmarks/psf_compile.py`; with the scripts' import order, `LOADED
.../benchmarks/psf_compile.py 2026-09-26`.

| Prediction | Result |
|---|---|
| V1: max PSF-Zero error over 60 pairs <= 1e-13 | **CONFIRMED** -- 6.18e-14 (was 1.76e-11); median 6.77e-15 (Qiskit on the same pairs: median 8.72e-15, max 1.81e-13) |
| V2: 36 tests pass | **CONFIRMED** -- 36 passed |
| V3: PSF-Zero max per-pair distance <= 1e-12 at every lap | **CONFIRMED** -- 6.21e-14 at lap 1 rising to 7.18e-13 at lap 10, both spares (was 1.76e-11 to 6.47e-10) |
| V4: 10/10 laps within 1 s at spare 0, median <= 0.12 s | **REFUTED** -- 10/10 within 1 s, but median 0.135 s (was 0.061 s); at spare 8, 0.095 s (was 0.028 s) |
| V5: Qiskit's per-lap values identical to Addendum 184 | **CONFIRMED** |

`diag_construction.py` in the same run still shows `d_model` = 1.76e-11:
it calls the core directly, bypassing the polishing step, so this is the
expected confirmation that the core itself is unchanged and the polishing
is what removes the error.

**Disclosed**: when this valid run was made, Addendum 187's follow-up had
already shown the error to be at the parameter level, which made V1 and V3
more likely than when they were registered. The predictions were not
changed.

## 3. What this means

- **Precision**: PSF-Zero's per-call error is now at Qiskit's level on these
  inputs -- better at the median, lower at the maximum (6.2e-14 vs 1.8e-13).
- **Drift**: still linear, at 7.3e-14 per lap against Qiskit's default at
  about 8e-15 (Addendum 185's control), so roughly ten times Qiskit's rate
  rather than a thousand. At that rate the 1e-12 bound would be crossed
  around lap 14.
- **Cost**: compile time 2.2x (spare 0) to 3.4x (spare 8) higher. On the
  cliff PSF-Zero remains about 96 times faster than Qiskit's default (0.135
  s vs 12.9 s) and meets the 1 s deadline on every lap. The polishing uses
  a 16-column forward-difference Jacobian; an analytic Jacobian, or a
  cheaper trigger, are the obvious ways to recover the speed.
- **Two copies of `psf_compile.py`** (root and `benchmarks/`) caused this
  error; they should become one, or every script should record which file
  it loaded.

## 4. Files

| File | What it is |
|---|---|
| `fix_validation_run2.txt` | raw log of the valid run, including the loaded-file check |
| `psf_compile_fix_2026-09-26/psf_compile.py` | the fixed file (SHA-256 `a1a20781...`) |
| `diag_construction.py` | the diagnostic that exposed the mistake |
