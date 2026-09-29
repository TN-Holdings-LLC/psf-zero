#!/usr/bin/env bash
# run_100k_2026-09-29.sh -- the pre-registered 100,000-lap run of pl_heavyhex_100k.py on the pod.
# Needs the setup of 2026-09-29 (setup_gpu_2026-09-29.sh): ~/psf_env, ~/psf-zero,
# ~/core_rel_0929, ~/core_cand_0929, ~/layout_cand_0929.
# Start it so that it survives a closed terminal:
#     nohup bash run_100k_2026-09-29.sh > run_100k_log.txt 2>&1 &
# Progress:  tail -n 2 ~/pod_100k_out_0929/pl_100k_*_2026-09-29.txt
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
E100K=dad56b1b2f2351b413e6f2b096ee4f9b2e3ae6d8db405760a5fcb0ae523288cb
EGPU=a0081c2917591de57a2e3a9d4914441c47f030282c5128b3f18642b7dd8402ea
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
[ "$(nsha "$B/pl_heavyhex_100k.py")" = "$E100K" ] || { echo "STOP: pl_heavyhex_100k.py is not the locked file"; exit 1; }
[ "$(nsha "$B/pl_heavyhex_gpu.py")" = "$EGPU" ] || { echo "STOP: pl_heavyhex_gpu.py is not the locked file"; exit 1; }
OUT=~/pod_100k_out_0929
mkdir -p "$OUT"
cd "$OUT"
{
  echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "pl_heavyhex_100k.py $E100K"
  echo "pl_heavyhex_gpu.py  $EGPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  lscpu | grep 'Model name' || true
  nproc
  python3 --version
  git -C ~/psf-zero log --oneline -1
  echo "release lib.rs   $(nsha ~/core_build_0929_rel/src/lib.rs)"
  echo "candidate lib.rs $(nsha ~/core_build_0929_cand/src/lib.rs)"
  echo "candidate layout $(nsha ~/layout_cand_0929/psf_smart_layout.py)"
} > pod_env_100k.txt
cat pod_env_100k.txt
S="$B/pl_heavyhex_100k.py"
PYTHONPATH=~/core_cand_0929 python3 -u "$S" run --part A --layout-dir ~/layout_cand_0929 > pl_100k_A_2026-09-29.txt 2>&1 &
PYTHONPATH=~/core_cand_0929 python3 -u "$S" run --part B --layout-dir ~/layout_cand_0929 > pl_100k_B_2026-09-29.txt 2>&1 &
PYTHONPATH=~/core_rel_0929  python3 -u "$S" run --part C > pl_100k_C_2026-09-29.txt 2>&1 &
wait
echo "end $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> pod_env_100k.txt
python3 -u "$S" score > pl_100k_score.txt 2>&1 || true
cat pl_100k_score.txt
python3 - "$OUT" "$B/pod_100k_outputs_2026-09-29.zip" <<'EOF'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(src)):
        z.write(os.path.join(src, f), f)
print("wrote", dst)
EOF
echo "RUN DONE"
