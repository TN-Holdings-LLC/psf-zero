# psf-intent IR v0: specification draft (2026-10-20)

**Status.** A draft for the owner, written at the workplace on 2026-10-20 at about 01:50 CEST. Nothing here is built.

- **Basis:** the design memo `docs/design/llm-surface-rust-skeleton.md` §3-4, and the owner's decision of 2026-10-19
  (three families: `tfim_chain`, `ring_classifier`, `rzz_layer`).
- **Facts taken from:** psfz's README and source (r4 placement, r5 templates) and the template check of 2026-10-19.
- **Open values** are marked `[ ]`. A proposal, where given, is marked as one.

## 1. Principles carried over from the memo

- **One record per request.** The model writes only this record; Rust checks it (gate 1) and builds from it.
- **Closed set of families.** Anything else is refused with a code. There is no fallback, no hand-off and no second
  try.
- **Every range check is a formula** on the record and, for scope, on the device file. No search, no score, no VF2.
- **The device enters only as a file reference** (with its SHA-256). The model never sees calibration data.
- **The checks run in a fixed order.** The first failing check gives the refusal code. Each code names the field and
  the formula that failed.

## 2. JSON Schema (draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "psf-intent/0",
  "title": "psf-intent IR v0",
  "type": "object",
  "additionalProperties": false,
  "required": ["ir", "family", "params", "output", "device"],
  "properties": {
    "ir": {"const": "psf-intent/0"},
    "family": {"enum": ["tfim_chain", "ring_classifier", "rzz_layer"]},
    "output": {"enum": ["template", "circuit"]},
    "device": {"type": "string", "pattern": "^file:[^\\s]{1,255}$"},
    "device_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "params": {"type": "object"}
  },
  "oneOf": [
    {"properties": {"family": {"const": "ring_classifier"}, "params": {"$ref": "#/$defs/ring_classifier"}}},
    {"properties": {"family": {"const": "tfim_chain"},      "params": {"$ref": "#/$defs/tfim_chain"}}},
    {"properties": {"family": {"const": "rzz_layer"},       "params": {"$ref": "#/$defs/rzz_layer"}}}
  ],
  "$defs": {
    "angle": {"type": "number", "minimum": -6.283185307179586, "maximum": 6.283185307179586},
    "ring_classifier": {
      "type": "object", "additionalProperties": false,
      "required": ["n", "layers", "encoding", "encoding_scale", "readout"],
      "properties": {
        "n":              {"type": "integer", "minimum": 12, "maximum": 64, "multipleOf": 2},
        "layers":         {"type": "integer", "minimum": 1, "maximum": 64},
        "encoding":       {"enum": ["ry"]},
        "encoding_scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 3.141592653589793},
        "readout":        {"type": "string", "pattern": "^z[0-9]{1,2}$"}
      }
    },
    "tfim_chain": {
      "type": "object", "additionalProperties": false,
      "required": ["n", "J", "h", "dt", "steps", "order"],
      "properties": {
        "n":     {"type": "integer", "minimum": 2, "maximum": 100},
        "J":     {"type": "number", "minimum": -10, "maximum": 10},
        "h":     {"type": "number", "minimum": -10, "maximum": 10},
        "dt":    {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "steps": {"type": "integer", "minimum": 1, "maximum": 64},
        "order": {"enum": [1, 2]},
        "slots": {"enum": ["field", "none"], "default": "field"}
      }
    },
    "rzz_layer": {
      "type": "object", "additionalProperties": false,
      "required": ["graph", "n", "angles"],
      "properties": {
        "graph":  {"enum": ["chain", "ring"]},
        "n":      {"type": "integer", "minimum": 2, "maximum": 100},
        "angles": {"oneOf": [
          {"const": "slots"},
          {"type": "array", "items": {"$ref": "#/$defs/angle"}, "minItems": 1, "maxItems": 100}
        ]}
      }
    }
  }
}
```

**Notes on the schema:**

- **The numeric limits are proposals** (`[ ]`, owner): n ≤ 64 for rings and ≤ 100 for chains, layers and steps
  ≤ 64, |J| and |h| ≤ 10, 0 < dt ≤ 1. The schema only bounds the record. The device and resource formulas of §3 decide
  the real limits.
- **`readout` uses the form z⟨k⟩;** k < n is checked by formula in §3 (a schema cannot compare two fields).
- **No field for gate lists, qubit indices or calibration values,** and no free-text field. With `additionalProperties:
  false`, a record cannot carry them.
- **Values of a ring classifier's trainable angles are not in the IR.** They come from TOQB as a values file at bind
  time, never from the model, so `ring_classifier` has `output: template` only (§3, R5).

## 3. Gate 1: checks by formula, in order

The notation, read from the device file:

- **λ_chain(D):** the length of the longest working row-and-bridge path; psfz r4's exact dynamic programme over
  rows, columns and bridges.
- **ring(D, n):** the first ring of length n, in psfz's fixed candidate order of hexagon-band boundaries, whose qubits
  and couplers all work; ⊥ if there is none. This is a fixed enumeration in a declared order, not a search by score.
- **failed(D):** psfz's existing rule, an element with error ≥ 0.5.

| # | check | formula (pass when true) | refusal code |
|---|---|---|---|
| 0 | parse | the record is valid JSON and validates against §2 | `SCHEMA_INVALID` (field path in the reason) |
| 1 | version | ir = "psf-intent/0" | `IR_VERSION` |
| 2 | finite numbers | every number is finite (not NaN or ±Inf) | `NUMBER_NOT_FINITE` |
| 3 | device file | the file is readable and parses as a psfz device; SHA-256 equals `device_sha256` when given | `DEVICE_UNREADABLE`, `DEVICE_HASH_MISMATCH` |
| 4 | device gate set | two-qubit gate = CZ when `output` = template (r5 refuses ECR) | `TEMPLATE_ECR_UNSUPPORTED` |
| R1 | ring classifier: readout | readout = z⟨k⟩ with 0 ≤ k < n | `READOUT_RANGE` |
| R2 | ring classifier: ring length | n even and n ≥ 12 (heavy-hex has no cycle shorter than 12) | `RING_TOO_SHORT`, `RING_ODD` |
| R3 | ring classifier: placement | ring(D, n) ≠ ⊥ | `NO_WORKING_PLACEMENT` |
| R4 | ring classifier: resources | CZ = n · L ≤ CZ_max; sx = n · (2L + 2) ≤ SX_max (both formulas match the template check of 2026-10-19 at n = 12, L = 8 / 16: 96 / 216 and 192 / 408) | `RESOURCE_LIMIT` |
| R5 | ring classifier: output | output = template | `OUTPUT_KIND` |
| T1 | TFIM: placement | n ≤ λ_chain(D) | `CHAIN_TOO_LONG` |
| T2 | TFIM: resources | rzz = (n − 1) · k · steps, with k rzz layers per step (k = 1 for first order; k for second order by the family's splitting, `[ ]`). Each rzz(θ_zz), where θ_zz is the family's rzz angle, for example 2 J dt (convention `[ ]`), costs 2 CZ in general, 1 CZ when θ_zz ≡ ±π/2, and none when θ_zz ≡ 0 or π (mod 2π) (psfz step 5). CZ ≤ CZ_max | `RESOURCE_LIMIT` |
| T3 | TFIM: twin bound (only if a twin run is requested) | max over cuts i of min(2^(i+1), 2^(n−i−1), 2^(k · steps)) ≤ χ_max (rank 2 per rzz block on each edge; psfz improvement proposal §3) | `TWIN_BOND_LIMIT` |
| Z1 | rzz layer: graph and length | chain: n ≥ 2; ring: n even, n ≥ 12 | `RING_TOO_SHORT`, `RING_ODD` |
| Z2 | rzz layer: angle count | array length = n − 1 (chain) or n (ring) | `PARAM_COUNT` |
| Z3 | rzz layer: placement | chain: n ≤ λ_chain(D); ring: ring(D, n) ≠ ⊥ | `CHAIN_TOO_LONG`, `NO_WORKING_PLACEMENT` |
| Z4 | rzz layer: native gate | the device file lists an rzz native (fractional gates) | `DEVICE_NO_RZZ` |
| Z5 | rzz layer: angle range | each angle, folded by closed-form identities into the device's allowed interval [ ] (from the device file), lands inside it. Folds: rzz(θ + 2π) = rzz(θ); rzz(−θ) = (X ⊗ I) rzz(θ) (X ⊗ I); rzz(θ ± π) = rzz(θ) (Z ⊗ Z) up to phase | `RZZ_ANGLE_RANGE` |
| Z6 | rzz layer: slots | angles = "slots" is refused while two-qubit angles cannot be template slots (r5) | `TEMPLATE_2Q_SLOT_UNSUPPORTED` |
| 9 | psfz build | psfz exits 0 for the record's circuit; exit 2 is a refusal with psfz's reason | `BUILD_REFUSED` |

**Notes:**

- **Limits** (`[ ]`, owner): CZ_max, SX_max and χ_max are proposed as device- or task-level constants written in a
  config file next to the schema.
- **Z4 and Z5 need what psfz does not have yet.** Its scope says "Not yet: fractional gates (rzz / rx natives)". Until
  then, every `rzz_layer` record ends at Z4 or at check 9, with a reason. Which interval the device allows for rzz is
  read from the device file; it is not assumed here (`[ ]` to confirm against IBM's description of fractional gates).
- **TFIM's slots.** `slots` = "field" makes each rx(h·dt) a slot; the two-qubit angle J·dt is fixed (r5).

## 4. Refusal codes (complete list for v0)

| code | stage | meaning |
|---|---|---|
| `SCHEMA_INVALID` | 0 | does not validate against the schema (unknown field, wrong type, missing field, enum, range) |
| `IR_VERSION` | 1 | unknown IR version |
| `NUMBER_NOT_FINITE` | 2 | NaN or infinity |
| `DEVICE_UNREADABLE` | 3 | the device file cannot be read or parsed |
| `DEVICE_HASH_MISMATCH` | 3 | the device file's SHA-256 differs from the record's |
| `TEMPLATE_ECR_UNSUPPORTED` | 4 | a template was asked for on an ECR device |
| `READOUT_RANGE` | R1 | readout qubit outside 0 … n − 1 |
| `RING_ODD` | R2, Z1 | ring with an odd number of qubits |
| `RING_TOO_SHORT` | R2, Z1 | ring with fewer than 12 qubits (no such cycle on heavy-hex) |
| `NO_WORKING_PLACEMENT` | R3, Z3 | no ring of that length with working elements in psfz's fixed candidate order |
| `CHAIN_TOO_LONG` | T1, Z3 | chain longer than the device's longest working row-and-bridge path |
| `RESOURCE_LIMIT` | R4, T2 | a counted resource exceeds its declared limit (the reason names which, with the formula's value) |
| `OUTPUT_KIND` | R5 | the output kind is not available for the family |
| `TWIN_BOND_LIMIT` | T3 | the formula bound on the twin's bond dimension exceeds χ_max |
| `PARAM_COUNT` | Z2 | the angle list has the wrong length for the graph |
| `DEVICE_NO_RZZ` | Z4 | the device offers no rzz native |
| `RZZ_ANGLE_RANGE` | Z5 | an angle cannot be folded into the device's allowed interval |
| `TEMPLATE_2Q_SLOT_UNSUPPORTED` | Z6 | two-qubit angles as slots (not in r5) |
| `BUILD_REFUSED` | 9 | psfz refused the built circuit (its reason attached) |

**Format of a refusal:** `{"refused": true, "code": "...", "check": "R3", "field": "params.n", "formula": "ring(D, 14) =
⊥", "reason": "..."}`. The reason is written by Rust, never by the model.

## 5. Open points for the owner

- The numeric limits in §2 and §3 (`[ ]`).
- **Whether `circuit` output for TFIM** (all angles bound to the record's J, h, dt) is part of v0, or only `template`.
- **rzz layer.** v0 can define it now, with every record refused at Z4 until psfz has fractional natives. The
  alternative is to defer the family to v1.
- **Phase P0's test set N2** (memo §9) can be written from §4 alone: at least one record per refusal code.
