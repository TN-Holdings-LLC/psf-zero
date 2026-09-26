# Addendum 180 -- The layout cliff appears on IBM's square-lattice generation: on FakeNighthawk, Qiskit's default compilation takes ~12.9 s at spare = 0 and 2 and misses a 1 s deadline every time, while PSF-Zero produces the same circuit quality in ~0.15 s and meets it every time; away from the cliff the difference disappears (2026-09-25)

**Pre-registered in**:
`spare-qubit-cliff-addendum-178-preregistration-2026-09-25.md`, including
its Section 6 amendment (a failed first run and the harness fix, recorded
before this run). Run on WSL2 (home), Python 3.12.13, Qiskit 2.5.2. Script
checked on the machine: 8009 bytes, normalized SHA-256 `ce11be15...` --
matches the locked Section 6 version (the check was shown after the run;
the file was unchanged in between). Raw log and CSV received; every figure
below recomputed from the CSV, and the log's per-compile lines agree with
it.

## 0. In one line

All pre-registered predictions hold. N1: Qiskit's median compile time at
spare = 0 is 74 times its median at spare = 8. N2: at spare = 0, PSF-Zero
met the 1 s deadline in 5/5 seeds and Qiskit in 0/5. N3: at spare = 8 both
met it in 5/5. N4: two-qubit counts identical in 20/20 comparisons and
every pair exactly correct for both compilers (worst infidelity
2.6e-15). Same output, about 80 times faster, only on the cliff.

## 1. Results (5 seeds per cell; all 40 compiles finished, 0 DNF)

| spare | arm | median s | min s | max s | <=0.01 s | <=0.1 s | <=1 s | <=10 s | two-qubit | per-pair check (worst) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | Q3 Qiskit | 12.927 | 12.686 | 13.084 | 0/5 | 0/5 | 0/5 | 0/5 | 180 | exact (2.3e-15) |
| 0 | P PSF-Zero | 0.158 | 0.148 | 0.274 | 0/5 | 0/5 | 5/5 | 5/5 | 180 | exact (2.6e-15) |
| 2 | Q3 Qiskit | 12.911 | 12.818 | 12.961 | 0/5 | 0/5 | 0/5 | 0/5 | 177 | exact (2.3e-15) |
| 2 | P PSF-Zero | 0.155 | 0.150 | 0.161 | 0/5 | 0/5 | 5/5 | 5/5 | 177 | exact (2.6e-15) |
| 4 | Q3 Qiskit | 0.152 | 0.147 | 0.155 | 0/5 | 0/5 | 5/5 | 5/5 | 174 | exact (2.3e-15) |
| 4 | P PSF-Zero | 0.142 | 0.134 | 0.148 | 0/5 | 0/5 | 5/5 | 5/5 | 174 | exact (2.6e-15) |
| 8 | Q3 Qiskit | 0.174 | 0.170 | 0.177 | 0/5 | 0/5 | 5/5 | 5/5 | 168 | exact (2.3e-15) |
| 8 | P PSF-Zero | 0.142 | 0.141 | 0.144 | 0/5 | 0/5 | 5/5 | 5/5 | 168 | exact (2.6e-15) |

Two-qubit counts are 3 per logical pair (e.g. 60 pairs x 3 = 180 at
spare = 0): both compilers consolidate each pair's 20 unitaries into one
block. The per-pair check applied to every compile (no SWAP connected
different pairs).

## 2. Scoring (Addendum 178)

- **N0 -- CONFIRMED** (from the Stage 0 of both runs): 120 qubits, bipartite
  60/60, perfect matching of 60 pairs.
- **N1 -- CONFIRMED.** Q3 median 12.927 s at spare = 0 versus 0.174 s at
  spare = 8: ratio 74 (bar: 10).
- **N2 -- CONFIRMED.** At spare = 0, 1 s deadline: P 5/5, Q3 0/5.
- **N3 -- CONFIRMED.** At spare = 8, 1 s deadline: P 5/5, Q3 5/5. (Seed 0
  of this cell had been seen in the Section 6 diagnosis, as disclosed.)
- **N4 -- CONFIRMED.** Two-qubit counts equal in 20/20; per-pair exact check
  applicable and passed for every compile of both arms.

## 3. Observations (not pre-registered)

- **The cliff also covers spare = 2** (Q3 median 12.911 s), and is gone by
  spare = 4 (0.152 s). The prediction only scored spare = 0 and 8.
- **Speed ratio on the cliff**: about 82x at spare = 0 (12.927 / 0.158) and
  83x at spare = 2. Away from it, PSF-Zero is still slightly faster
  (0.142 s vs 0.152-0.174 s), but both are far inside every deadline above
  0.1 s.
- **Qiskit's cliff time is nearly constant** (12.69-13.08 s across 10
  compiles). That points to a fixed search limit being exhausted before a
  fallback, rather than a variable search -- a hypothesis only; the
  internal cause was not traced here.
- **Deadline ladder**: neither compiler meets 0.01 s or 0.1 s anywhere
  (PSF-Zero's time includes Qiskit's own routing at level 1 on a 120-qubit
  device, as expected in advance); the 1 s and 10 s deadlines separate
  them completely on the cliff and not at all away from it.

## 4. What this means

This is the first result in this project showing PSF-Zero's established
advantage on a topology of a current IBM device generation. Earlier, the
cliff was shown on square grids built for the purpose, and shown NOT to
arise on heavy-hex (workplace Addenda 39-40; Addenda 135-136). FakeNighthawk
is square-lattice, admits the saturated layout, and with Qiskit given the
full device target (not a bare coupling map), the cliff is there.

Together with Addendum 179 (no quality difference at any depth where the
cliff does not occur), the picture is consistent across both experiments:
**PSF-Zero produces the same circuits as Qiskit; where Qiskit's layout
search hits the cliff, PSF-Zero produces them about two orders of
magnitude faster, fast enough to meet a 1 s deadline Qiskit misses.**

## 5. What this does not establish

- Real Nighthawk hardware. FakeNighthawk's coupling map is the device's
  topology, but its error properties are, by its own warning, "not
  intended to represent typical nighthawk error values"; Qiskit's
  optimization level 3 uses those values in layout scoring, so its
  timing on the real device could differ.
- Circuits other than this project's dense disjoint-pair generator, which
  is built to reach the saturated condition; typical application circuits
  may or may not land on it (Addenda 135-136 found random dense circuits
  generally do not on heavy-hex).
- Noisy execution quality (compiles checked structurally and exactly, not
  simulated).
- Whether 1 s is the right deadline for a given user (all four levels
  reported).

## 6. Files

| File | What it is |
|---|---|
| `nighthawk_deadline_cliff.py` | the script (Section 6 version, hash `ce11be15...`) |
| `nighthawk_deadline_cliff_2026-09-25.csv` | raw results, 40 rows |
| `nighthawk_result_run2.txt` | raw log of this run |
| `nighthawk_result.txt` | log of the failed first run (Addendum 178, Section 6) -- not yet received |
