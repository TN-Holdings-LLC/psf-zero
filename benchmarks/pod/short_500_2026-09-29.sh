#!/usr/bin/env bash
# short_500_2026-09-29.sh -- the pre-registered 500-lap short version (home; no IBM).
# Part A of pl_heavyhex_100k.py (candidate stack, FakeAuckland, spare 0) for 500 laps,
# whole-circuit GPU checks at laps 1, 10, 100, 500. Target: tens of seconds.
#
# Run with the Python environment that has Qiskit 2.5.2, PennyLane 0.45.1,
# pennylane-lightning-gpu 0.45.0, qiskit-ibm-runtime 0.50.0, networkx and maturin ACTIVE:
#     REPO=<path to the psf-zero clone> bash short_500_2026-09-29.sh [<pod part A CSV>]
# The first run builds the candidate core into ~/core_cand_0929 and extracts the candidate
# layout into ~/layout_cand_0929 (the repository itself is not changed). Nothing is deleted.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$HOME/psf-zero}"
E100K=dad56b1b2f2351b413e6f2b096ee4f9b2e3ae6d8db405760a5fcb0ae523288cb
EGPU=a0081c2917591de57a2e3a9d4914441c47f030282c5128b3f18642b7dd8402ea
LIB_REL=bf3bf537df3eb3e81444d2d2ba6fe723770048f2c80c5c6bb9da48884148d234
LIB_CAND=5364630ec4e3648fe944b4103d76e8caa440e722c27ee6621b0ac51fdb26ad8f
LAYOUT_REL=a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875
LAYOUT_CAND=e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa
nsha() { python3 - "$1" <<'EOF'
import hashlib, sys
lines = [l.rstrip() for l in open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n").split("\n")]
while lines and lines[-1] == "":
    lines.pop()
print(hashlib.sha256("\n".join(lines).encode()).hexdigest())
EOF
}
[ "$(nsha "$B/pl_heavyhex_100k.py")" = "$E100K" ] || { echo "STOP: pl_heavyhex_100k.py is not the locked file"; exit 1; }
[ "$(nsha "$B/pl_heavyhex_gpu.py")" = "$EGPU" ] || { echo "STOP: pl_heavyhex_gpu.py is not the locked file"; exit 1; }
export PATH="$HOME/.cargo/bin:$PATH"

# candidate core (built once, from the release src/lib.rs + the patch, outside the repository)
if [ ! -d ~/core_cand_0929/psf_zero_core ]; then
    [ "$(nsha "$REPO/src/lib.rs")" = "$LIB_REL" ] || { echo "STOP: $REPO/src/lib.rs is not the release core bf3bf537"; exit 1; }
    d=~/core_build_0929_cand
    mkdir -p "$d/src"
    cp "$REPO/Cargo.toml" "$d/"
    cp "$REPO/src/lib.rs" "$d/src/lib.rs"
    printf '[build-system]\nrequires = ["maturin>=1.0"]\nbuild-backend = "maturin"\n[project]\nname = "psf_zero_core"\nversion = "0.1.0"\n[tool.maturin]\nfeatures = []\n' > "$d/pyproject.toml"
    (cd "$d" && git apply "$B/lib_rs_eigen_route_2026-09-29.patch")
    [ "$(nsha "$d/src/lib.rs")" = "$LIB_CAND" ] || { echo "STOP: patched lib.rs hash mismatch"; exit 1; }
    (cd "$d" && maturin build --release -o wheels 2>&1 | tail -2)
    pip install -q --no-deps --target ~/core_cand_0929 "$d"/wheels/psf_zero_core-*.whl
fi
# candidate layout (extracted once)
if [ ! -f ~/layout_cand_0929/psf_smart_layout.py ]; then
    [ "$(nsha "$REPO/benchmarks/psf_smart_layout.py")" = "$LAYOUT_REL" ] || { echo "STOP: $REPO/benchmarks/psf_smart_layout.py is not the release m1"; exit 1; }
    mkdir -p ~/layout_cand_0929/work/benchmarks
    cp "$REPO/benchmarks/psf_smart_layout.py" ~/layout_cand_0929/work/benchmarks/
    (cd ~/layout_cand_0929/work && git apply --include=benchmarks/psf_smart_layout.py "$B/psf_smart_layout_c1_2026-09-29.patch")
    cp ~/layout_cand_0929/work/benchmarks/psf_smart_layout.py ~/layout_cand_0929/psf_smart_layout.py
fi
[ "$(nsha ~/layout_cand_0929/psf_smart_layout.py)" = "$LAYOUT_CAND" ] || { echo "STOP: candidate layout hash mismatch"; exit 1; }

OUT="$B/out_short"
mkdir -p "$OUT"
cd "$OUT"
{
  echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  lscpu | grep 'Model name' || true
  python3 --version
  git -C "$REPO" log --oneline -1
} > env_short.txt
T0=$(date +%s)
PYTHONPATH=~/core_cand_0929 python3 -u "$B/pl_heavyhex_100k.py" run --part A --laps 500 --checkpoints 1,10,100,500 \
    --layout-dir ~/layout_cand_0929 --repo "$REPO" --tag short > pl_100k_A_short.txt 2>&1
echo "total wall including imports and setup of the run: $(( $(date +%s) - T0 )) s" | tee -a env_short.txt
if [ $# -ge 1 ]; then
    python3 -u "$B/pl_heavyhex_100k.py" score --short --tag short --pod-csv "$1" | tee score_short.txt
else
    python3 -u "$B/pl_heavyhex_100k.py" score --short --tag short | tee score_short.txt
fi
echo "SHORT DONE"
