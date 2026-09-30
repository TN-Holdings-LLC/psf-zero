#!/usr/bin/env bash
# run_v10_eval_r1_2026-09-30.sh -- revision 1 (before any run) of run_v10_eval_2026-09-30.sh: the five HELD-OUT tasks
# only, to fit in about one hour. gpt-oss-120b, two arms run at the same time against one vLLM server, 3 runs each
# (seeds 21-23, written to run1..run3):
#   v9  arm: e2e_vllm_psf_v10.py with 1 candidate and numeric feedback only (= the v9 harness of the INVEST test)
#   v10 arm: e2e_vllm_psf_v10.py --n-candidates 3 --verbal
# Both: high effort, 48,000-token replies, salvage at medium effort, best-circuit memory, no tool.
# Then: re-timing of fill27g9 (sequential), score_v10_eval_r1.py, one zip with MANIFEST.tsv. Nothing is deleted.
#   bash run_v10_eval_r1_2026-09-30.sh
#   SMOKE=1 bash run_v10_eval_r1_2026-09-30.sh   (disclosed smoke run, not scored: v10 arm, ghz5 + bell3 (tuned tasks only),
#                                              seed run 97, 2 rounds, into ~/v10eval_smoke_0930; no retime/score/zip)
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
M=openai/gpt-oss-120b
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
  OUT=~/v10eval_smoke_0930; ARMS="v10"; RUNS="1"; SEEDBASE=96; TASKS="ghz5 bell3"; ROUNDS=2
else
  OUT=~/v10eval_out_0930; ARMS="v9 v10"; RUNS="1 2 3"; SEEDBASE=20
  TASKS="w4 dicke42 ghz3i singlet3 fill27g9"; ROUNDS=6
fi
declare -A ARMFLAGS=([v9]="" [v10]="--n-candidates 3 --verbal")
export HF_HOME=/workspace/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
command -v ninja >/dev/null || { echo "STOP: ninja missing (pip install ninja into ~/vllm_env and link it)"; exit 1; }
mkdir -p "$OUT"
{
  echo "date_utc $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  lscpu | grep -E "Model name|^CPU\(s\)"
  "$HOME/vllm_env/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
  "$HOME/psf_env/bin/python" -c "import qiskit, pennylane; print('qiskit', qiskit.__version__, '| pennylane', pennylane.__version__)"
  sha256sum "$B/e2e_vllm_psf_v10.py" "$B/score_v10_eval_r1.py" "$B/run_v10_eval_r1_2026-09-30.sh"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"
if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then echo "STOP: a server is already on port 8000"; exit 1; fi
"$HOME/vllm_env/bin/vllm" serve "$M" --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.90 \
    --max-model-len 65536 > "$OUT/vllm_server.log" 2>&1 &
SP=$!
t0=$(date +%s)
for i in $(seq 1 240); do
  curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $SP 2>/dev/null || { echo "SERVER FAILED"; tail -n 30 "$OUT/vllm_server.log" | tee "$OUT/SERVER_FAILED.txt"; exit 1; }
  sleep 10
done
echo "server ready after $(( $(date +%s) - t0 )) s"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUT/gpu_mem.txt"
pids=()
for arm in $ARMS; do
  for r in $RUNS; do
    for t in $TASKS; do
      # shellcheck disable=SC2086
      ( PYTHONPATH=~/core_cand_0929 timeout 10800 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v10.py" \
          --model "$M" --run $((SEEDBASE + r)) --tasks $t --rounds $ROUNDS --max-tokens 48000 --reasoning-effort high \
          ${ARMFLAGS[$arm]} --layout-dir ~/layout_cand_0929 --repo ~/psf-zero --out "$OUT/$arm/run$r" 2>&1 \
          | grep -v -i "warn" > "$OUT/log_${arm}_run${r}_$t.txt" ) &
      pids+=($!)
    done
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "task-runs finished after $(( $(date +%s) - t0 )) s"
grep -h -- "-> " "$OUT"/log_*.txt | cut -c1-160
kill $SP; wait $SP 2>/dev/null
[ "$SMOKE" = 1 ] && { echo "SMOKE DONE"; exit 0; }
for task in fill27g9; do
  echo "== re-timing $task (sequential) $(date -u +%T)"
  PYTHONPATH=~/core_cand_0929 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v10.py" --retime "$OUT" --retime-task $task \
      --retime-reps 5 --layout-dir ~/layout_cand_0929 --repo ~/psf-zero 2>&1 | grep -v -i "warn" | tee "$OUT/retime_${task}_log.txt"
done
echo "== score"
"$HOME/psf_env/bin/python" "$B/score_v10_eval_r1.py" "$OUT" | tee "$OUT/score_log.txt"
"$HOME/psf_env/bin/python" - "$OUT" "$B/v10eval_outputs_0930.zip" <<'PY'
import hashlib, os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
def nsha(p):
    lines = [ln.rstrip() for ln in open(p, encoding="utf-8").read().replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()
rows = []
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for d, _, fs in sorted(os.walk(src)):
        for f in sorted(fs):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, src)
            try:
                h, kind = nsha(p), "text"
            except UnicodeDecodeError:
                h, kind = hashlib.sha256(open(p, "rb").read()).hexdigest(), "bin"
            rows.append(f"{rel}\t{os.path.getsize(p)}\t{h}\t{kind}")
            z.write(p, rel)
    z.writestr("MANIFEST.tsv", "\n".join(rows) + "\n")
print("wrote", dst, len(rows), "files")
PY
sha256sum "$B/v10eval_outputs_0930.zip"
echo "V10 EVAL DONE"
