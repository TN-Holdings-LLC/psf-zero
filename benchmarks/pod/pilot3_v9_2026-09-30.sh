#!/usr/bin/env bash
# pilot3_v9_2026-09-30.sh -- EXPLORATORY pilot 3 (not pre-registered, not scored): v9 = v8 + salvage at medium
# effort (8,000 tokens) + 48,000-token replies + optional simulate() tool. gpt-oss-120b, fresh seeds (runs 7-9).
# Arms: w3 high without the tool (3 runs), w3 high with the tool (3 runs), and a guard on the four tasks NOT used
# for tuning so far plus qft3 -- ghz5, bell3, fill27, qft3 -- high with the tool (run 7 each). 10 task-runs in
# parallel. The server is started with tool calling on; if it fails to start that way, it is restarted
# without it (recorded in server_mode.txt) and the tool arms then run as v9 without tools. Needs the setup of 2026-09-30 (psf_env, vllm_env with ninja, the model in
# /workspace/hf_cache). Nothing is deleted. Output: ~/pilot_0930 and pilot_outputs_0930.zip next to this file.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
OUT=~/pilot3_0930
M=openai/gpt-oss-120b
export HF_HOME=/workspace/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
command -v ninja >/dev/null || { echo "STOP: ninja missing (pip install ninja into ~/vllm_env and link it)"; exit 1; }
mkdir -p "$OUT"
sha256sum "$B/e2e_vllm_psf_v9.py" "$B/pilot3_v9_2026-09-30.sh" | tee "$OUT/hashes.txt"
if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then echo "STOP: a server is already on port 8000"; exit 1; fi
start_server() {  # $1 = extra flags
  # shellcheck disable=SC2086
  "$HOME/vllm_env/bin/vllm" serve "$M" --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.90 \
      --max-model-len 65536 $1 > "$OUT/vllm_server.log" 2>&1 &
  SP=$!
  for i in $(seq 1 240); do
    curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
    kill -0 $SP 2>/dev/null || return 1
    sleep 10
  done
  return 1
}
t0=$(date +%s)
if start_server "--tool-call-parser openai --enable-auto-tool-choice"; then
  echo "tools on" | tee "$OUT/server_mode.txt"
else
  cp "$OUT/vllm_server.log" "$OUT/vllm_server_tools_failed.log"; kill $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "tools off (server did not start with tool calling; see vllm_server_tools_failed.log)" | tee "$OUT/server_mode.txt"
  start_server "" || { echo "SERVER FAILED"; tail -n 20 "$OUT/vllm_server.log"; exit 1; }
fi
echo "server ready after $(( $(date +%s) - t0 )) s"
pids=()
launch() {  # $1 task, $2 run, $3 arm name, $4 extra flags
  # shellcheck disable=SC2086
  ( PYTHONPATH=~/core_cand_0929 timeout 7200 "$HOME/psf_env/bin/python" -u "$B/e2e_vllm_psf_v9.py" \
      --model "$M" --run $2 --tasks $1 --max-tokens 48000 --reasoning-effort high $4 \
      --layout-dir ~/layout_cand_0929 --repo ~/psf-zero --out "$OUT/$3/run$2" 2>&1 \
      | grep -v -i "warn" > "$OUT/log_$3_$1_run$2.txt" ) &
  pids+=($!)
}
for r in 7 8 9; do
  launch w3 $r w3_notool ""
  launch w3 $r w3_tool "--tool-sim"
done
for t in qft3 ghz5 bell3 fill27; do
  launch $t 7 guard_tool "--tool-sim"
done
for p in "${pids[@]}"; do wait "$p"; done
echo "task-runs finished after $(( $(date +%s) - t0 )) s"
kill $SP; wait $SP 2>/dev/null
"$HOME/psf_env/bin/python" - "$OUT" <<'PY' | tee "$OUT/pilot_summary.txt"
import glob, json, os, sys
out = sys.argv[1]
for f in sorted(glob.glob(os.path.join(out, "*", "run*", "*", "rounds.jsonl"))):
    arm = os.path.relpath(f, out).split(os.sep)
    print("==", arm[0], arm[1], arm[2])
    for r in map(json.loads, open(f)):
        tok = (r.get("usage") or {}).get("completion_tokens")
        print(f"  round {r['round']}: finish {r.get('finish_reason')} | tokens {tok} | reasoning chars {r.get('reasoning_chars')} | "
              f"llm {r.get('llm_s')} s | " + ("salvaged | " if "salvage" in r else "") +
              (f"tool calls {sum(1 for x in r['tool_calls'] if 'arguments' in x)} | " if r.get("tool_calls") else "") + (f"error {r['error'][:60]}" if "error" in r else
              f"F {r['fidelity_compiled']:.6f} 2q {r['routed_2q']}"))
PY
"$HOME/psf_env/bin/python" -c "
import os, shutil; shutil.make_archive('$B/pilot3_outputs_0930', 'zip', os.path.expanduser('~/pilot3_0930'))"
sha256sum "$B/pilot3_outputs_0930.zip"
echo "PILOT3 DONE"
