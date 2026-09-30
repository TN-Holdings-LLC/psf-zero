#!/usr/bin/env bash
# run_invest_2026-09-30.sh -- pre-registered go/no-go run (vLLM x PSF-Zero) on one RunPod B200.
# Needs: setup_gpu_2026-09-29.sh done (~/psf_env, ~/psf-zero, ~/core_cand_0929, ~/layout_cand_0929),
# ~/vllm_env with vLLM, and the three models in /workspace/hf_cache. Nothing is deleted.
# Per model: start the vLLM server, run 5 tasks x 3 runs in parallel (15 processes), stop the server.
# Then: G3 re-timing pass (sequential), scoring, and one zip with MANIFEST.tsv.
#   bash run_invest_2026-09-30.sh            (all three models)
#   SMOKE=1 bash run_invest_2026-09-30.sh    (disclosed smoke run, not scored: 7B, ghz5 + fill27, run 9,
#                                            2 rounds, into ~/invest_smoke_0930; no retime, score or zip)
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
  OUT=~/invest_smoke_0930; MODELS=qwen7b; RUNS="9"; TASKS="ghz5 fill27"; ROUNDS=2
else
  OUT=~/invest_out_0930; MODELS="qwen7b qwen72b gptoss120b"; RUNS="1 2 3"; TASKS="ghz5 w3 bell3 qft3 fill27"; ROUNDS=6
fi
export HF_HOME=/workspace/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT"

declare -A HF=([qwen7b]=Qwen/Qwen2.5-7B-Instruct [qwen72b]=Qwen/Qwen2.5-72B-Instruct [gptoss120b]=openai/gpt-oss-120b)
declare -A EXTRA=([qwen7b]="--max-tokens 4000" [qwen72b]="--max-tokens 4000" [gptoss120b]="--max-tokens 16000 --reasoning-effort high")

{
  echo "date_utc $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  lscpu | grep -E "Model name|^CPU\(s\)"
  free -g | head -2
  "$HOME/vllm_env/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
  "$HOME/psf_env/bin/python" -c "import qiskit, pennylane; print('qiskit', qiskit.__version__, '| pennylane', pennylane.__version__)"
  sha256sum "$B/e2e_vllm_psf_v6.py" "$B/score_vllm_invest.py" "$B/run_invest_2026-09-30.sh"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

wait_ready() {  # up to 40 minutes
  for i in $(seq 1 240); do
    curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

for tag in $MODELS; do
  m="${HF[$tag]}"
  echo "== $tag ($m) $(date -u +%T)"
  mkdir -p "$OUT/$tag"
  if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "STOP: a server is already running on port 8000; stop it first"; exit 1
  fi
  "$HOME/vllm_env/bin/vllm" serve "$m" --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.90 \
      --max-model-len 32768 > "$OUT/$tag/vllm_server.log" 2>&1 &
  SP=$!
  t0=$(date +%s)
  if ! wait_ready $SP; then
    echo "SERVER FAILED for $tag (recorded; its 15 task-runs count as not solved)"
    tail -n 30 "$OUT/$tag/vllm_server.log" > "$OUT/$tag/SERVER_FAILED.txt"
    kill $SP 2>/dev/null; wait $SP 2>/dev/null
    continue
  fi
  echo "server ready after $(( $(date +%s) - t0 )) s"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUT/$tag/gpu_mem.txt"
  pids=()
  for r in $RUNS; do
    for t in $TASKS; do
      # shellcheck disable=SC2086
      ( PYTHONPATH=~/core_cand_0929 timeout 5400 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v6.py" \
          --model "$m" --run $r --tasks $t --rounds $ROUNDS ${EXTRA[$tag]} --layout-dir ~/layout_cand_0929 --repo ~/psf-zero \
          --out "$OUT/$tag/run$r" 2>&1 | grep -v -i "warn" > "$OUT/$tag/log_run${r}_$t.txt" ) &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "$tag task-runs finished after $(( $(date +%s) - t0 )) s"
  grep -h -- "-> " "$OUT/$tag"/log_run*_*.txt | cut -c1-200
  kill $SP; wait $SP 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 2000 ] && break
    [ "$i" = 30 ] && pkill -f "vllm serve"   # engine processes still holding the GPU after 150 s
    sleep 5
  done
done

[ "$SMOKE" = 1 ] && { echo "SMOKE DONE"; exit 0; }
echo "== G3 re-timing pass (sequential) $(date -u +%T)"
PYTHONPATH=~/core_cand_0929 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v6.py" --retime "$OUT" --retime-reps 5 \
    --layout-dir ~/layout_cand_0929 --repo ~/psf-zero 2>&1 | grep -v -i "warn" | tee "$OUT/retime_log.txt"
echo "== score"
"$HOME/psf_env/bin/python" "$B/score_vllm_invest.py" "$OUT" | tee "$OUT/score_log.txt"

"$HOME/psf_env/bin/python" - "$OUT" "$B/invest_outputs_0930.zip" <<'PY'
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
sha256sum "$B/invest_outputs_0930.zip"
echo "INVEST RUN DONE"
