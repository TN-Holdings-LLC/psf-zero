#!/usr/bin/env bash
# pilot_w3_2026-09-30.sh -- EXPLORATORY pilot (not pre-registered, not scored): does v7 (bigger reply budget,
# cp gate, short-reasoning prompt) let gpt-oss-120b answer the weak task w3 at all?
# Arms: reasoning_effort high and medium, 3 runs each, up to 6 rounds, 32,000-token replies. 6 task-runs
# in parallel against one server. Needs the setup of 2026-09-30 (psf_env, vllm_env with ninja, the model in
# /workspace/hf_cache). Nothing is deleted. Output: ~/pilot_0930 and pilot_outputs_0930.zip next to this file.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
OUT=~/pilot_0930
M=openai/gpt-oss-120b
export HF_HOME=/workspace/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
command -v ninja >/dev/null || { echo "STOP: ninja missing (pip install ninja into ~/vllm_env and link it)"; exit 1; }
mkdir -p "$OUT"
sha256sum "$B/e2e_vllm_psf_v7.py" "$B/pilot_w3_2026-09-30.sh" | tee "$OUT/hashes.txt"
if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then echo "STOP: a server is already on port 8000"; exit 1; fi
"$HOME/vllm_env/bin/vllm" serve "$M" --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.90 \
    --max-model-len 40960 > "$OUT/vllm_server.log" 2>&1 &
SP=$!
t0=$(date +%s)
for i in $(seq 1 240); do
  curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  kill -0 $SP 2>/dev/null || { echo "SERVER FAILED"; tail -n 20 "$OUT/vllm_server.log"; exit 1; }
  sleep 10
done
echo "server ready after $(( $(date +%s) - t0 )) s"
pids=()
for eff in high medium; do
  for r in 1 2 3; do
    ( PYTHONPATH=~/core_cand_0929 timeout 5400 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v7.py" \
        --model "$M" --run $r --tasks w3 --max-tokens 32000 --reasoning-effort $eff \
        --layout-dir ~/layout_cand_0929 --repo ~/psf-zero --out "$OUT/$eff/run$r" 2>&1 \
        | grep -v -i "warn" > "$OUT/log_${eff}_run$r.txt" ) &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "task-runs finished after $(( $(date +%s) - t0 )) s"
kill $SP; wait $SP 2>/dev/null
"$HOME/psf_env/bin/python" - "$OUT" <<'PY' | tee "$OUT/pilot_summary.txt"
import glob, json, os, sys
out = sys.argv[1]
for f in sorted(glob.glob(os.path.join(out, "*", "run*", "w3", "rounds.jsonl"))):
    arm = os.path.relpath(f, out).split(os.sep)
    print("==", arm[0], arm[1])
    for r in map(json.loads, open(f)):
        tok = (r.get("usage") or {}).get("completion_tokens")
        print(f"  round {r['round']}: finish {r.get('finish_reason')} | tokens {tok} | reasoning chars {r.get('reasoning_chars')} | "
              f"llm {r.get('llm_s')} s | " + (f"error {r['error'][:60]}" if "error" in r else
              f"F {r['fidelity_compiled']:.6f} 2q {r['routed_2q']}"))
PY
"$HOME/psf_env/bin/python" -c "
import os, shutil; shutil.make_archive('$B/pilot_outputs_0930', 'zip', os.path.expanduser('~/pilot_0930'))"
sha256sum "$B/pilot_outputs_0930.zip"
echo "PILOT DONE"
