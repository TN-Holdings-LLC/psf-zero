#!/usr/bin/env bash
# setup_gpu_2026-09-29.sh -- prepare a RunPod pod for pl_heavyhex_gpu.py.
# Idempotent: every step is skipped when its result is already there and correct.
# Nothing is deleted. New folders: ~/core_build_0929_{rel,cand}, ~/core_rel_0929,
# ~/core_cand_0929, ~/layout_cand_0929 (and ~/psf_env, ~/psf-zero, rustup if missing).
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
COMMIT=f4b4a6c
LIB_REL=bf3bf537df3eb3e81444d2d2ba6fe723770048f2c80c5c6bb9da48884148d234
LIB_CAND=5364630ec4e3648fe944b4103d76e8caa440e722c27ee6621b0ac51fdb26ad8f
LAYOUT_CAND=e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa

nsha() { python3 - "$1" <<'EOF'
import hashlib, sys
lines = [l.rstrip() for l in open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n").split("\n")]
while lines and lines[-1] == "":
    lines.pop()
print(hashlib.sha256("\n".join(lines).encode()).hexdigest())
EOF
}

echo "== 1. Python environment"
[ -d ~/psf_env ] || python3 -m venv ~/psf_env
# shellcheck disable=SC1090
source ~/psf_env/bin/activate
pip install -q "qiskit==2.5.2" "pennylane==0.45.1" "pennylane-lightning==0.45.0" \
    "pennylane-lightning-gpu==0.45.0" "qiskit-ibm-runtime==0.50.0" maturin networkx scipy
echo "== 2. Rust"
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
fi
rustc --version
echo "== 3. Repository at $COMMIT"
[ -d ~/psf-zero/.git ] || git clone -q https://github.com/TN-Holdings-LLC/psf-zero ~/psf-zero
git -C ~/psf-zero cat-file -e "$COMMIT" 2>/dev/null || git -C ~/psf-zero fetch -q origin
[ "$(nsha ~/psf-zero/src/lib.rs)" = "$LIB_REL" ] || git -C ~/psf-zero checkout -q "$COMMIT"
[ "$(nsha ~/psf-zero/src/lib.rs)" = "$LIB_REL" ] || { echo "STOP: ~/psf-zero/src/lib.rs is not the release core"; exit 1; }
git -C ~/psf-zero log --oneline -1

echo "== 4. Two cores, built the same way (release, candidate)"
build_core() {  # $1 = rel|cand, $2 = expected lib.rs hash, $3 = target dir
    local d=~/core_build_0929_$1
    if [ -d "$3/psf_zero_core" ]; then echo "   $3 exists, not rebuilt"; return; fi
    mkdir -p "$d/src"
    cp ~/psf-zero/Cargo.toml "$d/"
    cp ~/psf-zero/src/lib.rs "$d/src/lib.rs"
    printf '[build-system]\nrequires = ["maturin>=1.0"]\nbuild-backend = "maturin"\n[project]\nname = "psf_zero_core"\nversion = "0.1.0"\n[tool.maturin]\nfeatures = []\n' > "$d/pyproject.toml"
    if [ "$1" = cand ] && [ "$(nsha "$d/src/lib.rs")" != "$2" ]; then
        (cd "$d" && git apply "$B/lib_rs_eigen_route_2026-09-29.patch")
    fi
    [ "$(nsha "$d/src/lib.rs")" = "$2" ] || { echo "STOP: $d/src/lib.rs hash is not $2"; exit 1; }
    (cd "$d" && maturin build --release -o wheels 2>&1 | tail -2)
    pip install -q --no-deps --target "$3" "$d"/wheels/psf_zero_core-*.whl
}
build_core rel  "$LIB_REL"  ~/core_rel_0929
build_core cand "$LIB_CAND" ~/core_cand_0929

echo "== 5. Candidate layout module"
if [ ! -f ~/layout_cand_0929/psf_smart_layout.py ]; then
    mkdir -p ~/layout_cand_0929/work/benchmarks
    cp ~/psf-zero/benchmarks/psf_smart_layout.py ~/layout_cand_0929/work/benchmarks/
    (cd ~/layout_cand_0929/work && git apply --include=benchmarks/psf_smart_layout.py "$B/psf_smart_layout_c1_2026-09-29.patch")
    cp ~/layout_cand_0929/work/benchmarks/psf_smart_layout.py ~/layout_cand_0929/psf_smart_layout.py
fi
[ "$(nsha ~/layout_cand_0929/psf_smart_layout.py)" = "$LAYOUT_CAND" ] || { echo "STOP: candidate layout hash mismatch"; exit 1; }

echo "== 6. Checks"
PYTHONPATH=~/core_rel_0929  python3 -c "import psf_zero_core as c; print('release core  ', c.__file__, c.CORE_VERSION)"
PYTHONPATH=~/core_cand_0929 python3 -c "import psf_zero_core as c; print('candidate core', c.__file__, c.CORE_VERSION)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import pennylane as qml; d = qml.device('lightning.gpu', wires=2); print('lightning.gpu ok', d.name)"
echo "pl_heavyhex_gpu.py $(nsha "$B/pl_heavyhex_gpu.py")"
echo "SETUP DONE"
