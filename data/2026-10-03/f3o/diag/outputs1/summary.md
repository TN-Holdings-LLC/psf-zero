# f3o_diag summary (exploratory, not a test)

## FakeAuckland (150 circuits; rows differing from HOLD2: {'C5': 0, 'C6': 0, 'L3T': 0}; noiseless max 1.2e-14)

| arm | infid / L3T | two-qubit | sx+x | depth | S_eff 2q | S_eff 1q | mean 2q error | same qubits as L3T | same qubits as C6 |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 1.111 | 60.0 | 99.3 | 91.9 | 0.4366 | 0.0263 | 0.00725 | 150 | 150 |
| C6 | 1.111 | 60.0 | 99.3 | 91.9 | 0.4366 | 0.0263 | 0.00725 | 150 | 150 |
| L3T | 1.000 | 60.0 | 81.9 | 73.1 | 0.4366 | 0.0233 | 0.00725 | 150 | 150 |
| L3onC6 | 1.001 | 60.0 | 82.0 | 73.2 | 0.4366 | 0.0234 | 0.00725 | 150 | 150 |
| RELonL3 | 1.114 | 60.0 | 99.3 | 91.9 | 0.4366 | 0.0272 | 0.00725 | 150 | 150 |

S_eff orders C6 against L3T as the measured infidelity does in 139 of 150 circuits.
C6 most used qubit sets: [2, 3, 5, 8, 11, 14] x150
L3T most used qubit sets: [2, 3, 5, 8, 11, 14] x150

## FakeHanoiV2 (150 circuits; rows differing from HOLD2: {'C5': 0, 'C6': 0, 'L3T': 0}; noiseless max 1.3e-14)

| arm | infid / L3T | two-qubit | sx+x | depth | S_eff 2q | S_eff 1q | mean 2q error | same qubits as L3T | same qubits as C6 |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 1.159 | 60.0 | 99.3 | 91.9 | 0.3361 | 0.0235 | 0.00558 | 150 | 150 |
| C6 | 1.158 | 60.0 | 99.3 | 91.9 | 0.3358 | 0.0237 | 0.00558 | 150 | 150 |
| L3T | 1.000 | 60.0 | 82.1 | 72.3 | 0.3308 | 0.0191 | 0.00550 | 150 | 150 |
| L3onC6 | 1.018 | 60.0 | 82.1 | 71.7 | 0.3308 | 0.0197 | 0.00550 | 150 | 150 |
| RELonL3 | 1.158 | 60.0 | 99.3 | 91.9 | 0.3358 | 0.0237 | 0.00558 | 150 | 150 |

S_eff orders C6 against L3T as the measured infidelity does in 144 of 150 circuits.
C6 most used qubit sets: [6, 7, 10, 12, 13, 14] x150
L3T most used qubit sets: [6, 7, 10, 12, 13, 14] x150

## FakeAlgiers (150 circuits; rows differing from HOLD2: {'C5': 0, 'C6': 0, 'L3T': 0}; noiseless max 1.2e-14)

| arm | infid / L3T | two-qubit | sx+x | depth | S_eff 2q | S_eff 1q | mean 2q error | same qubits as L3T | same qubits as C6 |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 1.223 | 60.0 | 99.3 | 91.9 | 0.4265 | 0.0418 | 0.00708 | 150 | 0 |
| C6 | 1.126 | 60.0 | 99.3 | 91.9 | 0.4067 | 0.0237 | 0.00676 | 0 | 150 |
| L3T | 1.000 | 60.0 | 81.8 | 73.2 | 0.4300 | 0.0359 | 0.00714 | 150 | 0 |
| L3onC6 | 1.046 | 60.0 | 82.0 | 73.3 | 0.4067 | 0.0205 | 0.00676 | 0 | 150 |
| RELonL3 | 1.226 | 60.0 | 99.3 | 91.9 | 0.4267 | 0.0430 | 0.00708 | 150 | 0 |

S_eff orders C6 against L3T as the measured infidelity does in 8 of 150 circuits.
C6 most used qubit sets: [1, 2, 3, 4, 5, 8] x150
L3T most used qubit sets: [8, 11, 12, 13, 14, 15] x150

## FakeGeneva (150 circuits; rows differing from HOLD2: {'C5': 0, 'C6': 0, 'L3T': 0}; noiseless max 1.1e-14)

| arm | infid / L3T | two-qubit | sx+x | depth | S_eff 2q | S_eff 1q | mean 2q error | same qubits as L3T | same qubits as C6 |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 1.091 | 60.0 | 99.3 | 91.9 | 0.3278 | 0.0152 | 0.00545 | 150 | 150 |
| C6 | 1.091 | 60.0 | 99.3 | 91.9 | 0.3278 | 0.0152 | 0.00545 | 150 | 150 |
| L3T | 1.000 | 60.0 | 82.2 | 73.2 | 0.3278 | 0.0137 | 0.00545 | 150 | 150 |
| L3onC6 | 1.001 | 60.0 | 82.1 | 73.2 | 0.3278 | 0.0137 | 0.00545 | 150 | 150 |
| RELonL3 | 1.092 | 60.0 | 99.3 | 91.9 | 0.3278 | 0.0157 | 0.00545 | 150 | 150 |

S_eff orders C6 against L3T as the measured infidelity does in 142 of 150 circuits.
C6 most used qubit sets: [1, 2, 3, 5, 8, 11] x150
L3T most used qubit sets: [1, 2, 3, 5, 8, 11] x150

## FakeTorino (150 circuits; rows differing from HOLD2: {'C5': 0, 'C6': 0, 'L3T': 0}; noiseless max 1.2e-14)

| arm | infid / L3T | two-qubit | sx+x | depth | S_eff 2q | S_eff 1q | mean 2q error | same qubits as L3T | same qubits as C6 |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 0.996 | 60.0 | 173.4 | 116.7 | 0.1695 | 0.0387 | 0.00282 | 150 | 150 |
| C6 | 0.996 | 60.0 | 173.4 | 116.7 | 0.1695 | 0.0387 | 0.00282 | 150 | 150 |
| L3T | 1.000 | 60.0 | 173.4 | 102.6 | 0.1695 | 0.0386 | 0.00282 | 150 | 150 |
| L3onC6 | 0.996 | 60.0 | 173.3 | 102.4 | 0.1695 | 0.0391 | 0.00282 | 150 | 150 |
| RELonL3 | 0.993 | 60.0 | 173.4 | 116.7 | 0.1695 | 0.0390 | 0.00282 | 150 | 150 |

S_eff orders C6 against L3T as the measured infidelity does in 108 of 150 circuits.
C6 most used qubit sets: [11, 12, 18, 31, 32, 33] x150
L3T most used qubit sets: [11, 12, 18, 31, 32, 33] x150

