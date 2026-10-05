# Workplace diagnosis (not pre-registered): qiskit-aer returns a wrong `save_expectation_value` when the operator acts on a subset of qubits and qubit truncation is on. Reproduced in Aer 0.16.4, 0.17.0, 0.17.1 and 0.17.2 (silent wrong value); in 0.15.1 the same case raises an error instead. No PSF-Zero harness uses this call (2026-10-05)

**Status: diagnosis of the discrepancy noticed during the DEPTH pilot (handoff 2026-10-05, section 4).**

- **Setting:** workplace sandbox. Python 3.11; Qiskit 2.5.2 with Aer 0.17.2 (installed), plus Aer 0.17.1, 0.17.0 and
  0.16.4 (Qiskit 2.5.2) and 0.15.1 (Qiskit 1.4.3) in throw-away venvs.

## 1. Minimal reproducer (`repro_aer_expval_truncation.py`)

```python
c = QuantumCircuit(3); c.x(2); c.cx(2, 1)                 # |110>: <Z_1> = -1
s = c.copy(); s.save_expectation_value(SparsePauliOp("Z"), [1], label="e")
AerSimulator().run(s).result().data()["e"]               # 1.0  (wrong)
AerSimulator(enable_truncation=False).run(s)...           # -1.0 (right)
```

| Aer | truncation on (default) | truncation off |
|---|---|---|
| 0.17.2, 0.17.1, 0.17.0, 0.16.4 | **+1.0 (wrong, silent)**; statevector and density_matrix alike | -1.0 |
| 0.15.1 (Qiskit 1.4.3) | error: `Invalid Pauli "0"` | -1.0 |

## 2. When it happens, and when it does not

**It happens when all of these hold:**

- the operator is given on a subset of the qubits (`save_expectation_value(op, qubits)` with fewer qubits than the
  circuit has);
- truncation is on;
- the active qubits are not 0..k-1;
- there are at least two active qubits, linked by a two-qubit gate.

**It does not happen:**

- with a full-width operator (`IZI` on [0, 1, 2]);
- with `enable_truncation=False`;
- with `save_density_matrix(qubits=...)`;
- with one active qubit;
- with active qubits {0, 1};
- with single-qubit gates only (x on 14, h on 13);
- in Aer's `EstimatorV2` and V1 `Estimator`, which pass full-width observables.

**Fusion is not involved** (`fusion_enable=False` gives the same wrong value).

**How it was first seen:** the DEPTH pilot's whole-device circuit (FakeAuckland, 27 qubits). Gate-by-gate reduction
brought it down to 9 gates on qubits 13, 14 and 16, and then to the 2-gate case above.

**Likely origin** (not verified in the C++ source): the truncation of `save_expval` ("Truncate save_expval",
qiskit-aer PR #2216). The operator's qubits do not appear to be remapped together with the circuit's when unused qubits
are dropped. Before that change (0.15.1), the same call failed loudly.

## 3. Impact on PSF-Zero

- **None on recorded results.** No script in the repository (`5cfa7c3`) calls `save_expectation_value`. All harnesses
  read `save_density_matrix(qubits=...)`, which is correct. The workplace pilot switched to `save_density_matrix`
  before any run (Addendum to come, DEPTH section 5).
- **For new code:** do not use `save_expectation_value` with a subset of qubits on Aer 0.16-0.17. Use a full-width
  operator, `enable_truncation=False` or `save_density_matrix`.

## 4. Upstream

- This looks like a qiskit-aer bug: a silent wrong result for documented usage.
- Whether and how to report it is the owner's decision. As with #17057, the owner posts in their own words.
- Material for a report:
  - the reproducer;
  - the version table;
  - the conditions in section 2;
  - the observation that 0.15.1 raised an error for the same case.
- **Before posting:** search the qiskit-aer issues for an existing report (only the PR title was found here).
