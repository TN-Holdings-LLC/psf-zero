# Record-keeping conventions

The rules this project follows when publishing a measurement. Most of them were
adopted after being burned by their absence — each one has an incident behind it,
noted where it applies.

## Identifying which machine a number came from

- **Identify machines by the CPU string the run itself printed**, never by anything
  else. An account name was used for this once and turned out to span more than one
  physical machine, which silently mis-attributed a set of results.
- **Do not label machines "fast" and "slow".** Measured across two machines on two
  workloads, the ordering reversed: one was uniformly faster on the dense
  coupling-map workload (0.65x–0.89x the other's time on every arm and scale) and
  uniformly slower on the large passthrough workload (1.4x–1.9x at 300 qubits). The
  two also ran different Python versions, so hardware and interpreter are confounded.
  Where such labels appear in [`docs/log/`](docs/log/), read them as identifiers for
  *which run* a number came from, not as claims about hardware speed.

## What every harness must record

- The environment row — `platform.platform()`, `platform.processor()`, Python
  version, library versions, multiprocessing start method — as **columns in the
  output CSV**, not printed to stdout. Printed output cannot be matched back to a
  file later.
- **Never write to a fixed output filename.** Use `*_YYYY-MM-DD.csv` or similar. A
  harness that overwrote its own previous output is the direct reason one anomalous
  run in this project's history can never be attributed to a machine or a code
  version — the evidence was silently destroyed by the next run. That question is
  closed as permanently unanswerable, not as answered.
- **Never include a filesystem path** that contains a user or host name.

## Reporting numbers

- Medians, not means, on shared hardware. A mean gets dragged by episodic background
  load; the median does not.
- Ranges, not peaks. Where a ratio is machine-dependent, say so and give the range.
- Warm-up outside the timer, for **both** engines being compared.
- Multiple seeds and repeated timed calls per point. One sample of Qiskit
  `optimization_level=3` at 156 qubits has roughly a one-in-five chance of landing
  3–4x high.
- Pin `seed_transpiler` wherever Qiskit's randomised layout/routing search is
  involved.
- Quote a statistic with the statistic it came from. One figure in this project's
  history reproduced only as a mean divided by a median; matched statistics gave
  1.78x or 3.20x for the same quantity depending on which pairing was used.

## Links

- **Verify a link resolves before writing it.** A 404 link is a false claim about
  where something is, and gets the same standard as a number.
- Verify by requesting the file directly
  (`https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>`) and checking
  for a 200. **Do not use a directory listing** — one summarised listing arrived with
  filenames mangled and entries missing, which produced a confident and completely
  wrong audit, including a false claim that a whole directory did not exist.
- When re-checking after a fix, **bust any cache** (`?v=2`). A stale 404 returned
  after a rename once produced a second wrong report.
- Do not put spaces in filenames. Several uploads to this repository arrived with a
  space inserted mid-name, breaking otherwise-correct links.

## Corrections

- Corrections go **underneath** the text they correct, never in place of it. The
  reasoning that produced a wrong answer stays visible next to the reasoning that
  caught it.
- A hypothesis gets a **numeric prediction written down before the test**. Two
  hypotheses in this project were refuted by their own pre-registered criteria; that
  only works if the criterion is recorded first.
- Read the source before proposing a mechanism for what a script does. One published
  explanation described a thread contending for the GIL during measurement; the
  script's own change-log says that thread runs in a separate untimed pass.

The full record of what this project got wrong, and the measurement that caught each
one, is in [`docs/log/README.md`](docs/log/README.md).

## What never goes into a published artifact

- Account names, host names, or any path containing them.
- Email addresses or organisation names.
- API keys, quantum-service tokens, or any other credential.
- Anything from which the physical location or ownership of a machine could be read.

## What is fine to publish

- IBM Quantum job IDs — reproducibility identifiers, not credentials.
- `platform.platform()` / `platform.processor()` output, which identifies a CPU model
  and not an individual machine.
- Python, library and OS versions; multiprocessing start method; seeds.

## Pre-publication check

Every file is scanned for the categories above before it is published. As of
2026-09-11 the scan covers the README, all four `docs/findings/` documents, all seven
`docs/log/` files and every CSV in `data/`, with no hits. The scan has caught real
leaks twice: a location-revealing word that slipped into an edit, and an internal
document that was almost published because it was linked from the README — the
project's own machine-identity notes, which are kept out of this repository entirely.
