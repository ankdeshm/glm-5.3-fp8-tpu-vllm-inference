#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?Usage: glm53_c2_2p1d_worker.sh <prefill|decode>}"
C2_RUN_TAG="${C2_RUN_TAG:-c2_run1}"

ulimit -c 0
sysctl -w vm.max_map_count=1048576 >/dev/null 2>&1 || true
find /dev/shm -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
mount -o remount,size=800G /dev/shm || true

RERUN_ROOT="/models/atlas-v7x-rerun-20261104"
SHM_ROOT="/dev/shm/atlas_glm53"
COORD_DIR="${RERUN_ROOT}/scratch/coord_${C2_RUN_TAG}"
mkdir -p "${SHM_ROOT}/home" "${SHM_ROOT}/cache" "${SHM_ROOT}/hf_cache" \
         "${SHM_ROOT}/jit_cache" "${SHM_ROOT}/profiles" "${SHM_ROOT}/logs" \
         "${SHM_ROOT}/results" "${SHM_ROOT}/tmp" "${COORD_DIR}" /data

export HOME="${SHM_ROOT}/home"
export XDG_CACHE_HOME="${SHM_ROOT}/cache"
export HF_HOME="${SHM_ROOT}/hf_cache"
export JAX_COMPILATION_CACHE_DIR="${SHM_ROOT}/jit_cache"
export TMPDIR="${SHM_ROOT}/tmp"

for _jc_tar in "${RERUN_ROOT}/scratch/jit_cache_prefill.tar" "${RERUN_ROOT}/scratch/jit_cache_decode.tar"; do
  if [ -s "${_jc_tar}" ]; then
    tar -xf "${_jc_tar}" -C "${SHM_ROOT}/jit_cache" 2>/dev/null || true
  fi
done

ln -sfn /models/models/zai-org/GLM-5.3 /weights
cp -f "${RERUN_ROOT}/inputs/requests_mocked.jsonl.xz" /data/requests_mocked.jsonl.xz
cp -f "${RERUN_ROOT}/inputs/requests_mocked.jsonl" /data/requests_mocked.jsonl

rm -rf "${SHM_ROOT}/sglang-jax-src" && mkdir -p "${SHM_ROOT}/sglang-jax-src"
tar -xzf "${RERUN_ROOT}/inputs/sglang-jax-6a43163.tar.gz" --strip-components=1 -C "${SHM_ROOT}/sglang-jax-src"
cp -r "${SHM_ROOT}/sglang-jax-src/python/sgl_jax/"* /opt/sglang-jax/python/sgl_jax/
cp -f "${SHM_ROOT}/sglang-jax-src/benchmark/glm53/bench_empirical_isl_sessions.py" "${SHM_ROOT}/bench_empirical_isl_sessions.py"

python3 - "${SHM_ROOT}/bench_empirical_isl_sessions.py" <<'PYEOF'
import sys
p = sys.argv[1]
txt = open(p, "r", encoding="utf-8").read()

old_trunks = "    if trunks_above_n16 > 0 and per_dp_counts[r] > 16:\n      cnt_r = max(len(ISL_BUCKETS), trunks_above_n16)"
new_trunks = (
    "    _fixed_t = int(__import__('os').environ.get('FIXED_TRUNKS_PER_DP', '0'))\n"
    "    if _fixed_t > 0:\n"
    "      cnt_r = max(len(ISL_BUCKETS), _fixed_t)\n"
    "    elif trunks_above_n16 > 0 and per_dp_counts[r] > 16:\n"
    "      cnt_r = max(len(ISL_BUCKETS), trunks_above_n16)"
)
assert old_trunks in txt, "old_trunks not found"
txt = txt.replace(old_trunks, new_trunks, 1)

old_abort = "  for u in [prefill_url, decode_url]:\n    if not u:\n      continue"
new_abort = (
    "  for u in [x.strip() for raw_u in [prefill_url, decode_url] for x in (raw_u or '').split(',') if x.strip()]:"
)
assert old_abort in txt, "old_abort not found"
txt = txt.replace(old_abort, new_abort, 1)

old_flush = '  for label, u in [("prefill", prefill_url), ("decode", decode_url), ("lb", base_url)]:\n    if not u:\n      continue'
new_flush = (
    '  _targets = []\n'
    '  for _lbl, _raw in [("prefill", prefill_url), ("decode", decode_url), ("lb", base_url)]:\n'
    '    for _u in [x.strip() for x in (_raw or "").split(",") if x.strip()]:\n'
    '      _targets.append((_lbl, _u))\n'
    '  for label, u in _targets:'
)
assert old_flush in txt, "old_flush not found"
txt = txt.replace(old_flush, new_flush, 1)

old_health = '  for label, url in [("prefill", prefill_url), ("decode", decode_url)]:\n    if not url:\n      continue'
new_health = (
    '  _h_targets = []\n'
    '  for _lbl, _raw in [("prefill", prefill_url), ("decode", decode_url)]:\n'
    '    for _u in [x.strip() for x in (_raw or "").split(",") if x.strip()]:\n'
    '      _h_targets.append((_lbl, _u))\n'
    '  for label, url in _h_targets:'
)
assert old_health in txt, "old_health not found"
txt = txt.replace(old_health, new_health, 1)

old_state = '  for label, url, state_str in [\n      ("prefill", prefill_url, prefill_state_json),\n      ("decode", decode_url, decode_state_json),\n  ]:\n    if not url:\n      continue'
new_state = (
    '  _s_targets = []\n'
    '  for _lbl, _raw, _st in [("prefill", prefill_url, prefill_state_json), ("decode", decode_url, decode_state_json)]:\n'
    '    for _u in [x.strip() for x in (_raw or "").split(",") if x.strip()]:\n'
    '      _s_targets.append((_lbl, _u, _st))\n'
    '  for label, url, state_str in _s_targets:'
)
assert old_state in txt, "old_state not found"
txt = txt.replace(old_state, new_state, 1)

old_tok = (
    '      "tokens": {\n'
    '          "mean_prompt_tokens": total_prompt / len(ok_results) if ok_results else 0.0,\n'
    '          "mean_cached_tokens": total_cached / len(ok_results) if ok_results else 0.0,\n'
    '          "mean_uncached_tokens": total_uncached / len(ok_results) if ok_results else 0.0,\n'
    '          "mean_completion_tokens": total_compl / len(ok_results) if ok_results else 0.0,\n'
    '          "observed_cache_hit_pct": cache_hit_pct,\n'
    '      },\n'
    '      "throughput": {\n'
    '          "window_output_tok_per_s": window_output_tokens / eff_window_s if eff_window_s > 0 else 0.0,'
)
new_tok = (
    '      "num_total_chips": int(__import__("os").environ.get("NUM_TOTAL_CHIPS", "24")),\n'
    '      "gate_evaluation": {\n'
    '          "p90_low_decode_tps": percentile(all_ok_decode_tps, 10),\n'
    '          "ttft_p95_s": percentile(all_ok_ttfts, 95),\n'
    '          "p90_low_gate_pass": bool(ok_results and percentile(all_ok_decode_tps, 10) >= 50.0),\n'
    '          "ttft_p95_gate_pass": bool(ok_results and percentile(all_ok_ttfts, 95) <= 3.0),\n'
    '          "gate_compliant": bool(ok_results and percentile(all_ok_decode_tps, 10) >= 50.0 and percentile(all_ok_ttfts, 95) <= 3.0),\n'
    '      },\n'
    '      "tags": ["GATE-COMPLIANT"] if (ok_results and percentile(all_ok_decode_tps, 10) >= 50.0 and percentile(all_ok_ttfts, 95) <= 3.0) else [],\n'
    '      "tokens": {\n'
    '          "mean_prompt_tokens": total_prompt / len(ok_results) if ok_results else 0.0,\n'
    '          "p50_prompt_tokens": statistics.median([float(r["prompt_tokens"] or 0) for r in ok_results]) if ok_results else 0.0,\n'
    '          "p95_prompt_tokens": percentile([float(r["prompt_tokens"] or 0) for r in ok_results], 95),\n'
    '          "mean_cached_tokens": total_cached / len(ok_results) if ok_results else 0.0,\n'
    '          "mean_uncached_tokens": total_uncached / len(ok_results) if ok_results else 0.0,\n'
    '          "mean_completion_tokens": total_compl / len(ok_results) if ok_results else 0.0,\n'
    '          "p50_completion_tokens": statistics.median([float(r["completion_tokens"] or 0) for r in ok_results]) if ok_results else 0.0,\n'
    '          "p95_completion_tokens": percentile([float(r["completion_tokens"] or 0) for r in ok_results], 95),\n'
    '          "observed_cache_hit_pct": cache_hit_pct,\n'
    '      },\n'
    '      "throughput": {\n'
    '          "window_output_tok_per_s": window_output_tokens / eff_window_s if eff_window_s > 0 else 0.0,\n'
    '          "window_output_tok_per_s_per_total_chip": (window_output_tokens / eff_window_s / int(__import__("os").environ.get("NUM_TOTAL_CHIPS", "24"))) if eff_window_s > 0 else 0.0,\n'
    '          "output_tok_per_s_per_total_chip": (total_compl / wall_clock / int(__import__("os").environ.get("NUM_TOTAL_CHIPS", "24"))) if wall_clock > 0 else 0.0,'
)
assert old_tok in txt, "old_tok not found"
txt = txt.replace(old_tok, new_tok, 1)

old_save = '      s["pass"] = "measured_210s_5_or_n_trunks_exact_temps"\n      all_summaries.append(s)'
new_save = (
    '      s["pass"] = f"measured_{int(args.duration_per_n)}s_5_or_n_trunks_exact_temps"\n'
    '      all_summaries.append(s)\n'
    '      _max_t = max(all_summaries, key=lambda x: x["throughput"]["window_output_tok_per_s"])\n'
    '      _g_ok = [x for x in all_summaries if x["gate_evaluation"]["gate_compliant"]]\n'
    '      _pk_g = max(_g_ok, key=lambda x: x["throughput"]["window_output_tok_per_s"]) if _g_ok else None\n'
    '      for _it in all_summaries:\n'
    '        _tg = []\n'
    '        if _it["gate_evaluation"]["gate_compliant"]: _tg.append("GATE-COMPLIANT")\n'
    '        if _pk_g is not None and _it is _pk_g: _tg.append("PEAK-GATE-COMPLIANT")\n'
    '        if _it is _max_t: _tg.append("MAX-THROUGHPUT")\n'
    '        _it["tags"] = _tg'
)
assert old_save in txt, "old_save not found"
txt = txt.replace(old_save, new_save, 1)

open(p, "w", encoding="utf-8").write(txt)
PYEOF

PHYS_ID=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number \
  || echo "${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX}}")
MY_IP=$(hostname -i | awk '{print $1}')
echo "${MY_IP}" > "${COORD_DIR}/${ROLE}_phys_${PHYS_ID}.ip"
echo "[C2-2P1D] role=${ROLE} job_idx=${JOB_COMPLETION_INDEX} phys_id=${PHYS_ID} ip=${MY_IP}"

# Wait until all 4 prefill nodes (0..3) and both decode nodes (0..1) publish their IPs
for r_check in prefill_phys_0 prefill_phys_1 prefill_phys_2 prefill_phys_3 decode_phys_0 decode_phys_1; do
  until [ -s "${COORD_DIR}/${r_check}.ip" ]; do
    sleep 2
  done
done

P0_IP0=$(cat "${COORD_DIR}/prefill_phys_0.ip")
P0_IP1=$(cat "${COORD_DIR}/prefill_phys_1.ip")
P1_IP0=$(cat "${COORD_DIR}/prefill_phys_2.ip")
P1_IP1=$(cat "${COORD_DIR}/prefill_phys_3.ip")
D0_IP0=$(cat "${COORD_DIR}/decode_phys_0.ip")
D0_IP1=$(cat "${COORD_DIR}/decode_phys_1.ip")
BOOTSTRAP_URL="http://${P0_IP0}:8998"
export DISAGG_LAUNCH_BOOTSTRAP=0

if [ "${ROLE}" = "prefill" ]; then
  if [ "${UNSET_VBAR:-0}" = "1" ]; then
    unset VBAR_CONTROL_SERVICE_URL
  fi
  export TPU_TOPOLOGY=2x2x2
  export TPU_HOST_BOUNDS=1,1,2
  export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_ACCELERATOR_TYPE=tpu7x-16
  export TPU_TOPOLOGY_WRAP=false,false,false
  export TPU_SKIP_MDS_QUERY=true

  if [ "${PHYS_ID}" -lt 2 ]; then
    SUB_TAG="P0"
    SUB_RANK="${PHYS_ID}"
    SUB_HEAD="${P0_IP0}"
    export TPU_WORKER_ID="${SUB_RANK}"
    export TPU_WORKER_HOSTNAMES="${P0_IP0},${P0_IP1}"
    export TPU_PROCESS_ADDRESSES="${P0_IP0}:8471,${P0_IP1}:8471"
  else
    SUB_TAG="P1"
    SUB_RANK=$((PHYS_ID - 2))
    SUB_HEAD="${P1_IP0}"
    export TPU_WORKER_ID="${SUB_RANK}"
    export TPU_WORKER_HOSTNAMES="${P1_IP0},${P1_IP1}"
    export TPU_PROCESS_ADDRESSES="${P1_IP0}:8471,${P1_IP1}:8471"
  fi

  if [ "${PHYS_ID}" = "0" ]; then
    nohup python3 -u -m sgl_jax.srt.disaggregation.run_bootstrap \
      --host 0.0.0.0 --port 8998 \
      > "${SHM_ROOT}/logs/c2_bootstrap.log" 2>&1 &
    echo $! > "${SHM_ROOT}/logs/c2_bootstrap.pid"

    nohup python3 -u -m sgl_jax.srt.disaggregation.launch_router \
      --pd-disaggregation --mini-lb \
      --prefill "http://${P0_IP0}:30271" 8998 \
      --prefill "http://${P1_IP0}:30271" 8998 \
      --prefill-bootstrap-host "${P0_IP0}" \
      --decode "http://${D0_IP0}:30271" \
      --host 0.0.0.0 --port 8000 \
      --request-timeout-secs 3600 \
      > "${SHM_ROOT}/logs/c2_mini_lb.log" 2>&1 &
    echo $! > "${SHM_ROOT}/logs/c2_mini_lb.pid"
  fi

  LOG_FILE="${SHM_ROOT}/logs/c2_${SUB_TAG}_rank${SUB_RANK}.log"
  python3 -u -m sgl_jax.launch_server \
    --model-path /weights \
    --tokenizer-path /weights \
    --trust-remote-code \
    --dist-init-addr "${SUB_HEAD}:5000" \
    --nnodes 2 \
    --node-rank "${SUB_RANK}" \
    --tp-size 16 \
    --dp-size 8 \
    --ep-size 16 \
    --moe-backend fused_v2 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --device tpu \
    --random-seed 3 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 256 \
    --page-size 128 \
    --chunked-prefill-size "${C2_CHUNKED_PREFILL_SIZE:-2048}" \
    --max-prefill-tokens "${C2_MAX_PREFILL_TOKENS:-16384}" \
    --mem-fraction-static 0.80 \
    --context-length 110000 \
    --attention-backend dsa_sparse \
    --hicache-storage none \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --enable-cache-report \
    --enable-metrics \
    --log-level-http warning \
    --speculative-algorithm NEXTN \
    --speculative-draft-model-path /weights \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --disaggregation-mode prefill \
    --disaggregation-host-ip "${MY_IP}" \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-bootstrap-url "${BOOTSTRAP_URL}" \
    --disaggregation-transfer-port 19000 \
    --disaggregation-side-channel-port 20000 \
    --disaggregation-max-inflight-transfers 32 \
    --precompile-bs-paddings 16 32 64 128 \
    --precompile-token-paddings ${C2_PRECOMPILE_TOKS:-256 512 1024 2048 4096 8192 16384} \
    --host 0.0.0.0 \
    --port 30271 \
    > "${LOG_FILE}" 2>&1 || true
else
  export DISAGG_LAUNCH_BOOTSTRAP=0
  export TPU_WORKER_ID="${PHYS_ID}"
  export TPU_WORKER_HOSTNAMES="${D0_IP0},${D0_IP1}"
  export TPU_PROCESS_ADDRESSES="${D0_IP0}:8471,${D0_IP1}:8471"
  LOG_FILE="${SHM_ROOT}/logs/c2_D0_rank${PHYS_ID}.log"
  python3 -u -m sgl_jax.launch_server \
    --model-path /weights \
    --tokenizer-path /weights \
    --trust-remote-code \
    --dist-init-addr "${D0_IP0}:5000" \
    --nnodes 2 \
    --node-rank "${PHYS_ID}" \
    --tp-size 16 \
    --dp-size 8 \
    --ep-size 16 \
    --moe-backend fused_v2 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --device tpu \
    --random-seed 3 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 256 \
    --page-size 128 \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 16384 \
    --mem-fraction-static 0.85 \
    --context-length 110000 \
    --attention-backend dsa_sparse \
    --enable-cache-report \
    --disable-radix-cache \
    --enable-metrics \
    --log-level-http warning \
    --speculative-algorithm NEXTN \
    --speculative-draft-model-path /weights \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --disaggregation-mode decode \
    --disaggregation-host-ip "${MY_IP}" \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-bootstrap-url "${BOOTSTRAP_URL}" \
    --disaggregation-transfer-port 19000 \
    --disaggregation-side-channel-port 20000 \
    --disaggregation-max-inflight-transfers 32 \
    --disaggregation-enable-overlap-schedule \
    --disaggregation-num-reserved-decode-tokens 512 \
    --precompile-bs-paddings 16 32 64 128 \
    --precompile-token-paddings 256 \
    --host 0.0.0.0 \
    --port 30271 \
    > "${LOG_FILE}" 2>&1 || true
fi

sleep 86400
