#!/usr/bin/env bash
# pilot2_v8_2026-09-30.sh -- EXPLORATORY pilot 2 (not pre-registered, not scored): v8 = v7 + salvage request
# on a token-limit cut + best-circuit memory in the feedback. gpt-oss-120b, fresh seeds (runs 4-6).
# Arms: w3 high, w3 medium, qft3 high (the other task that hit the limit); 3 runs each, up to 6 rounds,
# 32,000-token replies, 4,000-token salvage at low effort. 9 task-runs in parallel against one server. Needs the setup of 2026-09-30 (psf_env, vllm_env with ninja, the model in
# /workspace/hf_cache). Nothing is deleted. Output: ~/pilot_0930 and pilot_outputs_0930.zip next to this file.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
OUT=~/pilot2_0930
M=openai/gpt-oss-120b
export HF_HOME=/workspace/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
command -v ninja >/dev/null || { echo "STOP: ninja missing (pip install ninja into ~/vllm_env and link it)"; exit 1; }
mkdir -p "$OUT"
sha256sum "$B/e2e_vllm_psf_v8.py" "$B/pilot2_v8_2026-09-30.sh" | tee "$OUT/hashes.txt"
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
for arm in w3:high w3:medium qft3:high; do
  t=${arm%%:*}; eff=${arm##*:}
  for r in 4 5 6; do
    ( PYTHONPATH=~/core_cand_0929 timeout 5400 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v8.py" \
        --model "$M" --run $r --tasks $t --max-tokens 32000 --reasoning-effort $eff \
        --layout-dir ~/layout_cand_0929 --repo ~/psf-zero --out "$OUT/${t}_$eff/run$r" 2>&1 \
        | grep -v -i "warn" > "$OUT/log_${t}_${eff}_run$r.txt" ) &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "task-runs finished after $(( $(date +%s) - t0 )) s"
kill $SP; wait $SP 2>/dev/null
"$HOME/psf_env/bin/python" - "$OUT" <<'PY' | tee "$OUT/pilot_summary.txt"
import glob, json, os, sys
out = sys.argv[1]
for f in sorted(glob.glob(os.path.join(out, "*", "run*", "*", "rounds.jsonl"))):
    arm = os.path.relpath(f, out).split(os.sep)
    print("==", arm[0], arm[1])
    for r in map(json.loads, open(f)):
        tok = (r.get("usage") or {}).get("completion_tokens")
        print(f"  round {r['round']}: finish {r.get('finish_reason')} | tokens {tok} | reasoning chars {r.get('reasoning_chars')} | "
              f"llm {r.get('llm_s')} s | " + ("salvaged | " if "salvage" in r else "") + (f"error {r['error'][:60]}" if "error" in r else
              f"F {r['fidelity_compiled']:.6f} 2q {r['routed_2q']}"))
PY
"$HOME/psf_env/bin/python" -c "
import os, shutil; shutil.make_archive('$B/pilot2_outputs_0930', 'zip', os.path.expanduser('~/pilot2_0930'))"
sha256sum "$B/pilot2_outputs_0930.zip"
echo "PILOT2 DONE"
