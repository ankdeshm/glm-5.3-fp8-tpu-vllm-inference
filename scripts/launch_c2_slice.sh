#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?Usage: launch_c2_slice.sh <prefill|decode> <TAG> <IP0> <IP1> <BOOTSTRAP_URL> [LAUNCH_BOOTSTRAP:0|1]}"
TAG="${2:?}"
IP0="${3:?}"
IP1="${4:?}"
BOOTSTRAP_URL="${5:?}"
LAUNCH_BS="${6:-0}"

SHM_ROOT="/dev/shm/atlas_glm53"
RERUN_ROOT="/models/atlas-v7x-rerun-20261104"

mkdir -p "${SHM_ROOT}/home" "${SHM_ROOT}/cache" "${SHM_ROOT}/hf_cache" \
         "${SHM_ROOT}/jit_cache" "${SHM_ROOT}/profiles" "${SHM_ROOT}/logs" \
         "${SHM_ROOT}/results" "${SHM_ROOT}/tmp" /data

export HOME="${SHM_ROOT}/home"
export XDG_CACHE_HOME="${SHM_ROOT}/cache"
export HF_HOME="${SHM_ROOT}/hf_cache"
export JAX_COMPILATION_CACHE_DIR="${SHM_ROOT}/jit_cache"
export TMPDIR="${SHM_ROOT}/tmp"

ln -sfn /models/models/zai-org/GLM-5.3 /weights
if [ ! -f /data/requests_mocked.jsonl.xz ]; then
  cp -f "${RERUN_ROOT}/inputs/requests_mocked.jsonl.xz" /data/requests_mocked.jsonl.xz
fi

# Ensure patched wrapper_tcp_v3.py and decode.py are installed and byte-compiled
cp -f "${RERUN_ROOT}/scripts/wrapper_tcp_v3.py" /opt/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py
grep -q "_pd_multisocket_patched = True" /opt/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py
rm -rf /opt/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/__pycache__
python3 -m py_compile /opt/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py
python3 "${RERUN_ROOT}/scripts/patch_decode_multi_prefill.py" /opt/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py

MY_IP=$(hostname -i | awk '{print $1}')
if [ "${MY_IP}" = "${IP0}" ]; then
  SUB_RANK=0
elif [ "${MY_IP}" = "${IP1}" ]; then
  SUB_RANK=1
else
  echo "ERROR: MY_IP=${MY_IP} matches neither IP0=${IP0} nor IP1=${IP1}" >&2
  exit 1
fi

rm -f /tmp/libtpu_lockfile || true
export DISAGG_LAUNCH_BOOTSTRAP=0

if [ "${SUB_POPULATE_SUBSLICE:-0}" = "1" ]; then
  if [ "${UNSET_VBAR:-0}" = "1" ]; then
    unset VBAR_CONTROL_SERVICE_URL || true
  fi
  export TPU_TOPOLOGY=2x2x2
  export TPU_HOST_BOUNDS=1,1,2
  export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_ACCELERATOR_TYPE=tpu7x-16
  export TPU_TOPOLOGY_WRAP=false,false,false
  export TPU_SKIP_MDS_QUERY=true
fi

export TPU_WORKER_ID="${SUB_RANK}"
export TPU_WORKER_HOSTNAMES="${IP0},${IP1}"
export TPU_PROCESS_ADDRESSES="${IP0}:8471,${IP1}:8471"

echo "[LAUNCH-C2] $(date -u +%FT%TZ) role=${ROLE} tag=${TAG} rank=${SUB_RANK} ip=${MY_IP} bs_url=${BOOTSTRAP_URL} launch_bs=${LAUNCH_BS}"

if [ "${LAUNCH_BS}" = "1" ] && [ "${SUB_RANK}" = "0" ]; then
  nohup python3 -u -m sgl_jax.srt.disaggregation.run_bootstrap \
    --host 0.0.0.0 --port 8998 \
    > "${SHM_ROOT}/logs/c2_${TAG}_bootstrap.log" 2>&1 &
  echo $! > "${SHM_ROOT}/logs/c2_${TAG}_bootstrap.pid"
  sleep 2
fi

LOG_FILE="${SHM_ROOT}/logs/c2_${TAG}_rank${SUB_RANK}.log"

if [ "${ROLE}" = "prefill" ]; then
  if [ ! -f "${SHM_ROOT}/.jit_prefill_loaded" ] && [ -f "${RERUN_ROOT}/scratch/jit_cache_prefill_tp16_dp8.tar" ]; then
    tar -xf "${RERUN_ROOT}/scratch/jit_cache_prefill_tp16_dp8.tar" -C "${SHM_ROOT}" && touch "${SHM_ROOT}/.jit_prefill_loaded" || true
  fi
  export SGL_ENABLE_RADIX_CACHE="1"
  export SGL_MLA_COOP_PREFIX="0"
  export SGL_NUM_PREFILL_DP_RANKS="8"

  exec python3 -u -m sgl_jax.launch_server \
    --model-path /weights \
    --tokenizer-path /weights \
    --trust-remote-code \
    --dist-init-addr "${IP0}:5000" \
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
    --precompile-token-paddings ${C2_PRECOMPILE_TOKS:-256 512 1024 2048} \
    --host 0.0.0.0 \
    --port 30271 \
    > "${LOG_FILE}" 2>&1
else
  if [ ! -f "${SHM_ROOT}/.jit_decode_loaded" ] && [ -f "${RERUN_ROOT}/scratch/jit_cache_decode_tp16_dp8.tar" ]; then
    tar -xf "${RERUN_ROOT}/scratch/jit_cache_decode_tp16_dp8.tar" -C "${SHM_ROOT}" && touch "${SHM_ROOT}/.jit_decode_loaded" || true
  fi
  unset SGL_ENABLE_RADIX_CACHE || true
  unset SGL_MLA_COOP_PREFIX || true
  unset SGL_NUM_PREFILL_DP_RANKS || true

  exec python3 -u -m sgl_jax.launch_server \
    --model-path /weights \
    --tokenizer-path /weights \
    --trust-remote-code \
    --dist-init-addr "${IP0}:5000" \
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
    > "${LOG_FILE}" 2>&1
fi
