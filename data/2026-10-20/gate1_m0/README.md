# Gate 1 (P0): milestone M0 on the fixed N2 (2026-10-20, about 06:40 CEST)

`psfz gate RECORD.json CONFIG.json OUT.json` (psf-zero `rust/psfz/src/gate.rs`), built at home from IR r2
(`docs/design/IR_v0_spec_r2_2026-10-20.md`) after N2 was fixed (06:35 CEST, `n2.json` SHA-256 `458e53df…81ef`; the
test asserts that hash before it reads the set). Pure Rust, no outside crate (SHA-256 and a strict JSON parser are
in the crate). Cloud workspace, one core.

**M0: passed.** `cargo test --release n2_milestone_m0` (the unit test reads `n2.json` and its fixtures):

- 23 of 23 refusal records give exactly their code and check (R01-R18 with R04b, R08b, R09b; R20, R21);
- R19 (R10's record with check R3 disabled, test-only) gives `BUILD_REFUSED` at check 9 (psfz: no working ring);
- K1-K5 pass, each with the built output's counts equal to the gate's formula; K6 is declined.

| control | built (psfz) |
|---|---|
| K1 ring n 12, L 8, template, F1 | CZ 96, sx 216, 300 slots, ring on 12 qubits |
| K2 ring n 12, L 16 | CZ 192, sx 408, 588 slots |
| K3 TFIM n 16, 4 steps, template | CZ 120 = 15 × 4 × 2, sx 256, 64 slots |
| K4 TFIM n 10, circuit, F2 (ECR) | ECR 72 = 9 × 4 × 2, sx 140 |
| K5 TFIM n 24, 8 steps, twin requested | CZ 368, bond bound 2^8 = 256 = χ_max (inclusive) |

K1 and K2 equal the template check of 2026-10-19 (96 / 216, 192 / 408).

**Time.** The whole M0 test (30 records) runs in 0.06 s. The command line on K1 and K2, including process start,
reading the device file and psfz's template build: 4.8-11.1 ms per record (10 runs). The budget for gate 1 alone
(ARCHITECTURE §7) is 10 ms; the build is part of these figures.

**Schema cross-check.** The schema is written out as code in `gate.rs` (no JSON Schema crate). `schema_fuzz.py`
compares it with Python `jsonschema` 4.26 on the r2 schema file, with strict JSON on both sides: 566 random mutations
of N2's controls (548 invalid, 11 wrong version, 7 valid) and 420 boundary variants (217 invalid, 203 valid);
**0 disagreements in 986.** Not a proof: it covers what the mutations reach.

**Conventions gate 1 fixes where the IR left them open** (delegated by the owner on 2026-10-20; stated in `gate.rs`):

- TFIM: rzz angle 2 J dt and rx angle 2 h dt per first-order step; second order as Strang's splitting
  rx(h dt) rzz rx(h dt), so k = 1 rzz layer per step at both orders (the ZZ terms commute); start in |0…0⟩, measure
  every qubit.
- Ring classifier: the template check's structure (per layer ry encoding, ry and rz, CZ ring; a final ry), the
  readout qubit measured into clbit 0.
- An rzz layer that passes Z1-Z6 is refused at check 9 (`BUILD_REFUSED`: psfz has no fractional natives yet), as the
  IR draft's §3 note says.
- A device file psfz cannot read as a heavy-hex lattice is `DEVICE_UNREADABLE` (check 4).
- Strict JSON (RFC 8259): a duplicate key, `+1`, `01`, NaN or Infinity is `SCHEMA_INVALID` at check 0. Python's own
  `json` accepts duplicate keys and NaN; gate 1 does not.

**Not done:** the 34 requests of N2 need a model (M4); gate 1's output is not yet the `GateResult` artifact of
ARCHITECTURE §3 in full (it carries the record's and the device file's SHA-256, the derived formulas and the built
counts).
