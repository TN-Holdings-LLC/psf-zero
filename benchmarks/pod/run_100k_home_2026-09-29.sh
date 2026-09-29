#!/usr/bin/env bash
# run_100k_home_2026-09-29.sh -- Part I of the pl-heavyhex-100k pre-registration (Addendum 258),
# at the originally registered 100,000 laps, on the home machine (pre-registered as Addendum 261).
# Parts A, B, C run in parallel as on the pod. No IBM account, no network access to IBM, no QPU.
#
#     nohup bash run_100k_home_2026-09-29.sh <kit dir> <repo> > run_100k_home_log.txt 2>&1 &
# <kit dir>: the folder with the locked pl_heavyhex_100k.py and pl_heavyhex_gpu.py
# (work_2026-09-29_home_short_kit); <repo>: the psf-zero clone (release src/lib.rs and
# benchmarks/psf_smart_layout.py). Needs ~/core_cand_0929 and ~/layout_cand_0929 as built by
# short_500_2026-09-29.sh, and the release core installed in the active environment.
# Progress: tail -n 2 ~/home_100k_out_0929/pl_100k_*_home.txt
set -euo pipefail
K="$(cd "$1" && pwd)"
REPO="$(cd "$2" && pwd)"
E100K=dad56b1b2f2351b413e6f2b096ee4f9b2e3ae6d8db405760a5fcb0ae523288cb
EGPU=a0081c2917591de57a2e3a9d4914441c47f030282c5128b3f18642b7dd8402ea
LIB_REL=bf3bf537df3eb3e81444d2d2ba6fe723770048f2c80c5c6bb9da48884148d234
LIB_CAND=5364630ec4e3648fe944b4103d76e8caa440e722c27ee6621b0ac51fdb26ad8f
LAYOUT_REL=a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875
LAYOUT_CAND=e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa
nsha() { python3 - "$1" <<'PY'
import hashlib, sys
lines = [l.rstrip() for l in open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n").split("\n")]
while lines and lines[-1] == "":
    lines.pop()
print(hashlib.sha256("\n".join(lines).encode()).hexdigest())
PY
}
[ "$(nsha "$K/pl_heavyhex_100k.py")" = "$E100K" ] || { echo "STOP: pl_heavyhex_100k.py is not the locked file"; exit 1; }
[ "$(nsha "$K/pl_heavyhex_gpu.py")" = "$EGPU" ] || { echo "STOP: pl_heavyhex_gpu.py is not the locked file"; exit 1; }
[ "$(nsha "$REPO/src/lib.rs")" = "$LIB_REL" ] || { echo "STOP: $REPO/src/lib.rs is not the release core"; exit 1; }
[ "$(nsha "$REPO/benchmarks/psf_smart_layout.py")" = "$LAYOUT_REL" ] || { echo "STOP: release layout expected in the repo"; exit 1; }
[ "$(nsha ~/core_build_0929_cand/src/lib.rs)" = "$LIB_CAND" ] || { echo "STOP: candidate core build is not 5364630e"; exit 1; }
[ "$(nsha ~/layout_cand_0929/psf_smart_layout.py)" = "$LAYOUT_CAND" ] || { echo "STOP: candidate layout is not e25952a3"; exit 1; }
REL_V=$(python3 -c "import psf_zero_core as c; print(getattr(c, 'CORE_VERSION', None))")
CAND_V=$(PYTHONPATH=~/core_cand_0929 python3 -c "import psf_zero_core as c; print(getattr(c, 'CORE_VERSION', None))")
[ "$REL_V" = "2026-09-28.1" ] || { echo "STOP: the active environment's core is $REL_V, not the release 2026-09-28.1"; exit 1; }
[ "$CAND_V" = "2026-09-29.1" ] || { echo "STOP: ~/core_cand_0929 core is $CAND_V, not 2026-09-29.1"; exit 1; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$USED" -lt 4000 ] || { echo "STOP: $USED MiB of GPU memory already in use (a vLLM server still running?)"; exit 1; }
OUT=~/home_100k_out_0929
mkdir -p "$OUT"
cd "$OUT"
{
  echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "run script $(nsha "$0" 2>/dev/null || echo '?')"
  echo "pl_heavyhex_100k.py $E100K"
  echo "pl_heavyhex_gpu.py  $EGPU"
  echo "release core $REL_V (repo lib.rs $LIB_REL) | candidate core $CAND_V (lib.rs $LIB_CAND)"
  echo "release layout $LAYOUT_REL | candidate layout $LAYOUT_CAND"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  lscpu | grep 'Model name' || true
  nproc
  python3 --version
  git -C "$REPO" log --oneline -1
} > home_env_100k.txt
cat home_env_100k.txt
S="$K/pl_heavyhex_100k.py"
PYTHONPATH=~/core_cand_0929 python3 -u "$S" run --part A --layout-dir ~/layout_cand_0929 --repo "$REPO" --tag home > pl_100k_A_home.txt 2>&1 &
PYTHONPATH=~/core_cand_0929 python3 -u "$S" run --part B --layout-dir ~/layout_cand_0929 --repo "$REPO" --tag home > pl_100k_B_home.txt 2>&1 &
python3 -u "$S" run --part C --repo "$REPO" --tag home > pl_100k_C_home.txt 2>&1 &
wait
echo "end $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> home_env_100k.txt
python3 -u "$S" score --tag home > pl_100k_score_home.txt 2>&1 || true
cat pl_100k_score_home.txt
echo "RUN DONE"
