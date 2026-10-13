# Workplace review of benchmarks/ns_sim.py (request 5 of 2026-10-13), from the home bundle psf-zero_2026-10-13b (commit 16da01e, not yet on GitHub): no coding error found in readout, ALAP or the Kraus form; two modelling points that bias the prediction (2026-10-13)

**Status: independent reading by a session that did not write the file, plus numerical checks.** The module's own
tests pass here (6/6, release 2026-10-11.4, qiskit 2.5.2, aer 0.17.2). Times in CEST.

## 1. Checks made

| point | check | result |
|---|---|---|
| readout: bit order | `_measured_probs` on a random 4-qubit state with clbits mapped to qubits 2, 0, 3 and qubit 1 unmeasured, against a brute-force sum | equal to 3e-17 |
| readout: confusion | `_apply_readout` with different (P(1\|0), P(0\|1)) per clbit, against a brute-force product | equal to 3e-17 |
| one-qubit gate noise | relaxation for 32 ns followed by depolarizing with lambda = 2 (e - e_relax): average infidelity of the composed channel | 2.99955e-4 for e = 3e-4 |
| CZ noise | relaxation of both qubits for 68 ns followed by two-qubit depolarizing with lambda = (4/3)(e - e_relax) | 2.4984e-3 for e = 2.5e-3 |
| Kraus form | `Kraus(err.to_quantumchannel())` against the QuantumError's superoperator | identical (3 operators) |
| ALAP | read: backward pass from the ASAP makespan, each gate ends at the earliest start of its qubits' next gates | correct |
| XEB averaging | linear XEB is linear in the distribution, so the mean over trajectories equals the XEB of the averaged distribution | no bias |

## 2. Modelling points (not coding errors), in order of likely effect

1. **Idle relaxation uses the published T2, which is a Hahn-echo time** (IBM's QPU-information guide: "T2 is
   reported from a Hahn echo sequence"; see `heron-idle-literature-2026-10-13.md`). Without DD a waiting qubit is not
   echoed, so it dephases faster than `thermal_relaxation_error(T1, T2, t)` assumes. NS1 therefore likely
   underestimates the cost of waiting, which is the effect HW1 pointed to. Suggestion: a pre-registered sensitivity
   arm with a shorter dephasing time for idle windows only (for example T2/2), reported beside the main arm.
2. **ALAP is an assumption, not a documented property of the hardware.** Qiskit Runtime's DD options default to
   ALAP only when DD is enabled; with DD off the documentation found does not say how the instructions are timed. The
   difference matters most exactly where HW1 lost: with ASAP and a final barrier, early-finishing qubits wait in
   arbitrary states until the common measurement; with ALAP they do not wait at all. Suggestion: either submit
   explicitly scheduled circuits (delays inserted), so that the model and the device agree by construction, or
   pre-register both ASAP and ALAP predictions and say in advance which one is scored.
3. **Initialisation error is probably counted twice for measured qubits.** The model applies an X with the reported
   "Init error" before the circuit and then the readout confusion P(1|0), P(0|1). IBM's guide describes
   prob_meas1_prep0 as also including the discrimination error, i.e. it is measured from a nominal |0> preparation and
   already contains the initialisation error. On HW1's CSV the init error is about 0.002 per qubit against P(1|0) of
   about 0.01, so the double count is a few tenths of a per cent per measured qubit and up to a few per cent of F on
   20 qubits; it differs between arms only through which qubits they measure. Suggestion: drop the separate init
   channel for measured qubits, or subtract it from the confusion entries; record which.
4. **Smaller points.**
   - `x` gates take the `sx` error (the CSV's "Pauli-X error" column is not used); usually equal on Heron.
   - Barriers are dropped. Harmless for HW1's circuits (one final barrier before a common measurement), but
     intermediate barriers would not be respected by the schedule.
   - A qubit flipped by the init channel sits unrelaxed until its first gate (under ALAP that wait can be long);
     negligible at 0.2%.
   - Not modelled, as the docstring says: crosstalk and residual ZZ during simultaneous gates and idling, coherent
     errors, leakage, drift.

## 3. Bottom line

The implementation does what its docstring says. Before NS1's verdict (Addendum 462) is used to accept or reject
the Markovian model, points 1 and 2 should be addressed in NS2's design, because both act on idle time, the quantity
under test; point 3 is a small, fixable bias.
