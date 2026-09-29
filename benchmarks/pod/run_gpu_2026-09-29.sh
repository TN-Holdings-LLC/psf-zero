#!/usr/bin/env bash
# run_gpu_2026-09-29.sh -- the pre-registered run of pl_heavyhex_gpu.py on the pod.
# Run setup_gpu_2026-09-29.sh first. Outputs go to ~/pod_gpu_out_0929 and are zipped
# into this folder as pod_gpu_outputs_2026-09-29.zip for download.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
EXPECT=a0081c2917591de57a2e3a9d4914441c47f030282c5128b3f18642b7dd8402ea
# shellcheck disable=SC1090
source ~/psf_env/bin/activate
nsha() { python3 - "$1" <<'EOF'
import hashlib, sys
lines = [l.rstrip() for l in open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n").split("\n")]
while lines and lines[-1] == "":
    lines.pop()
print(hashlib.sha256("\n".join(lines).encode()).hexdigest())
EOF
}
GOT="$(nsha "$B/pl_heavyhex_gpu.py")"
[ "$GOT" = "$EXPECT" ] || { echo "STOP: pl_heavyhex_gpu.py hash $GOT is not the locked $EXPECT"; exit 1; }
OUT=~/pod_gpu_out_0929
mkdir -p "$OUT"
cd "$OUT"
{
  echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "pl_heavyhex_gpu.py $GOT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  lscpu | grep 'Model name' || true
  nproc
  python3 --version
  git -C ~/psf-zero log --oneline -1
  echo "release lib.rs   $(nsha ~/core_build_0929_rel/src/lib.rs)"
  echo "candidate lib.rs $(nsha ~/core_build_0929_cand/src/lib.rs)"
  echo "candidate layout $(nsha ~/layout_cand_0929/psf_smart_layout.py)"
  pip list 2>/dev/null | grep -i -E "^(qiskit|pennylane|numpy|networkx) " || true
} > pod_env_gpu.txt
cat pod_env_gpu.txt
S="$B/pl_heavyhex_gpu.py"
echo "== Q3";  PYTHONPATH=~/core_rel_0929  python3 -u "$S" run --arm Q3 > pl_gpu_Q3.txt 2>&1; tail -3 pl_gpu_Q3.txt
echo "== P";   PYTHONPATH=~/core_rel_0929  python3 -u "$S" run --arm P  > pl_gpu_P.txt  2>&1; tail -3 pl_gpu_P.txt
echo "== PN";  PYTHONPATH=~/core_cand_0929 python3 -u "$S" run --arm PN --layout-dir ~/layout_cand_0929 > pl_gpu_PN.txt 2>&1; tail -3 pl_gpu_PN.txt
python3 -u "$S" score > pl_gpu_score.txt 2>&1
cat pl_gpu_score.txt
echo "end $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> pod_env_gpu.txt
python3 - "$OUT" "$B/pod_gpu_outputs_2026-09-29.zip" <<'EOF'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(src)):
        z.write(os.path.join(src, f), f)
print("wrote", dst)
EOF
echo "RUN DONE"
