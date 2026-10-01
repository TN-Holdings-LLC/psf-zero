#!/usr/bin/env bash
# bisect_17057.sh -- run the reproducer of Qiskit issue #17057 on several released Qiskit versions.
# Each version gets a throw-away venv under ~/qk17057_venvs that is deleted after its run.
#   bash bisect_17057.sh 2>&1 | tee ~/qk17057_bisect.txt
set -u
VERS="${VERS:-0.45.3 1.0.2 1.1.2 1.2.4 1.4.6 2.0.3 2.2.3 2.4.2 2.5.2}"
ROOT=~/qk17057_venvs
mkdir -p "$ROOT"
cat > "$ROOT/repro.py" <<'EOF'
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, average_gate_fidelity
try:
    from qiskit.synthesis import TwoQubitBasisDecomposer
except ImportError:
    from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
dec = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
row = []
for c in (1e-8, 1e-7, 2e-7, 1e-5):
    core = QuantumCircuit(2)
    core.rxx(-1.2, 0, 1)
    core.ryy(-0.6, 0, 1)
    core.rzz(-2 * c, 0, 1)
    u = Operator(core)
    out = dec(u.data)
    ncx = sum(1 for ins in out.data if ins.operation.name == "cx")
    row.append(f"c={c:.0e}: {ncx} cx, 1-F {1 - average_gate_fidelity(Operator(out), u):.3e}")
print(f"qiskit {qiskit.__version__:8s} | " + " | ".join(row), flush=True)
EOF
for v in $VERS; do
  d="$ROOT/v$v"
  rm -rf "$d"
  python3 -m venv "$d" >/dev/null 2>&1 || { echo "qiskit $v: venv failed"; continue; }
  if "$d/bin/pip" install -q "qiskit==$v" >/dev/null 2>"$d.piperr"; then
    "$d/bin/python" "$ROOT/repro.py" 2>&1 | tail -1
  else
    echo "qiskit $v: install failed ($(tail -1 "$d.piperr"))"
  fi
  rm -rf "$d"
done
echo "BISECT DONE"
