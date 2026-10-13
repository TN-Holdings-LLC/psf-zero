# Workplace note, fixed before computing: the approximation rule for a14's estimate on circuits too wide for a state vector (request 1 of the home handover of 2026-10-13)

**Status: rule fixed and saved to the Project before any number is computed with it.** Times in CEST.

## The rule (R6/16)

a14's estimate needs, after each instruction, the reduced state of the instruction's qubits: two-qubit reduced
states for the gate terms of two-qubit gates, one-qubit reduced states for one-qubit gate terms and for the idle term.

For the reduced state of qubit set S (one or two qubits) right after instruction g:

1. **Exact** when every qubit in S has taken part in at most **K = 6** two-qubit gates up to and including g, **and**
   the backward light cone of g (all instructions that precede g and are connected to S through the circuit's wires,
   with g itself) touches at most **L = 16** qubits. The state is then computed from the light cone alone, which is
   exact (instructions outside the cone cannot change S's reduced state).
2. **Maximally mixed** otherwise: one qubit x = y = z = 0; two qubits, every two-qubit Pauli expectation 0.

Readout terms do not depend on the state and are unchanged. Instructions are taken as written (`rz` included).

**Why these values (chosen before computing).** L = 16: a 2^16 state vector per light cone is cheap. K = 6: about
three rounds of nearest-neighbour two-qubit layers, after which the qubits of QAOA- and Trotter-type circuits are
expected to be strongly entangled with the rest. Known weak point, stated in advance: circuits that stay close to a
product state at any depth (TFIM near |0...0>, small angles) are not mixed when deep, and the rule will overstate their
idle and gate terms.

## How it is checked

1. On every HW1 circuit with at most 22 active qubits: the rule against the exact estimate (a14's state-vector code),
   per circuit (gate term, idle term, total) and per PSF/QK3 pair (the predicted ratio and the winner).
2. qaoa_n20_p2 (25 active qubits in the PSF arm): computed with the rule, and, if memory and time allow, exactly with
   qiskit-aer's density-matrix saves on the compact circuit (checked first against a14's own code on the circuits
   of at most 22 qubits).
3. Then the seven HW1 pairs (tfim_n16 and GHZ left out as in Addendum 455): same winner and rank correlation, with the
   exact numbers where available and the rule's numbers beside them.

Nothing in the rule is changed after a number is seen; any later variant is reported as such.
