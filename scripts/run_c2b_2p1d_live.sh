#!/usr/bin/env bash
set -euo pipefail

P0_IP0="${1:?Usage: run_c2b_2p1d_live.sh <P0_IP0> <P1_IP0> <D0_IP0> [DUR_S]}"
P1_IP0="${2:?}"
D0_IP0="${3:?}"
DUR_S="${4:-50}"

SHM_ROOT="/dev/shm/atlas_glm53"
SHM_OUT="${SHM_ROOT}/results"
GCS_OUT="/models/atlas-v7x-rerun-20261104/results"
mkdir -p "${SHM_OUT}" "${GCS_OUT}" "${SHM_ROOT}/logs"

cp -f /models/atlas-v7x-rerun-20261104/scripts/bench_empirical_isl_sessions.py "${SHM_ROOT}/bench_empirical_isl_sessions.py"
python3 /models/atlas-v7x-rerun-20261104/scripts/patch_mini_lb_rr_decode.py /opt/sglang-jax/python/sgl_jax/srt/disaggregation/mini_lb.py

# Kill any existing MiniLB or stale bench_empirical_isl_sessions on P0
python3 - <<'PYEOF'
import glob, os, signal
my_pid = os.getpid()
for p in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        pid = int(p.split("/")[2])
        if pid in (1, my_pid): continue
        cmd = open(p, "rb").read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        if "launch_router" in cmd or "bench_empirical_isl_sessions.py" in cmd:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
PYEOF

echo "[C2b-LIVE] $(date -u +%FT%TZ) Waiting for P0=${P0_IP0}, P1=${P1_IP0}, D0=${D0_IP0} /health == 200 ..."
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${P0_IP0}:30271/health" || true)" = "200" ] && \
      [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${P1_IP0}:30271/health" || true)" = "200" ] && \
      [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${D0_IP0}:30271/health" || true)" = "200" ]; do
  sleep 2
done

for u in "http://${P0_IP0}:30271" "http://${P1_IP0}:30271" "http://${D0_IP0}:30271"; do
  curl -s -X POST "${u}/abort_request" -H "Content-Type: application/json" -d '{"rid": "", "abort_all": true}' >/dev/null || true
done
sleep 2
for u in "http://${P0_IP0}:30271" "http://${P1_IP0}:30271" "http://${D0_IP0}:30271"; do
  curl -s -X POST "${u}/flush_cache" >/dev/null || true
done

# Note: omit --prefill-bootstrap-host so MiniLB injects each Prefill's own host:8998
# and D0 resolves peers from both P0:8998 and P1:8998 dynamically!
echo "[C2b-LIVE] $(date -u +%FT%TZ) Starting 2P1D MiniLB on :8000 ..."
nohup python3 -u -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill "http://${P0_IP0}:30271" 8998 \
  --prefill "http://${P1_IP0}:30271" 8998 \
  --decode "http://${D0_IP0}:30271" \
  --host 0.0.0.0 --port 8000 --request-timeout-secs 3600 \
  > "${SHM_ROOT}/logs/c2_2p1d_mini_lb.log" 2>&1 < /dev/null &
echo $! > "${SHM_ROOT}/logs/c2_2p1d_mini_lb.pid"

until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${P0_IP0}:8000/v1/models" || true)" = "200" ]; do
  sleep 1
done

echo "[C2b-LIVE] $(date -u +%FT%TZ) Verifying 4 smoke requests (testing both P0 and P1 -> D0)..."
for i in 1 2 3 4; do
  curl -sf --max-time 90 "http://${P0_IP0}:8000/v1/chat/completions" \
    -H "Content-Type: application/json" -H "X-SMG-Routing-Key: smoke2p-${i}" \
    -d '{"model":"/weights","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16,"temperature":0.0}' \
    > "/tmp/c2b_smoke_${i}.json"
  echo "[C2b-LIVE] Smoke ${i} OK: $(head -c 160 "/tmp/c2b_smoke_${i}.json")"
done

echo "[C2b-LIVE] $(date -u +%FT%TZ) Running C2b Mode (b): 95% cache hit (40 trunks = FIXED_TRUNKS_PER_DP=5, dur=${DUR_S}s)..."
NUM_TOTAL_CHIPS=24 FIXED_TRUNKS_PER_DP=5 python3 -u "${SHM_ROOT}/bench_empirical_isl_sessions.py" \
  --base-url "http://${P0_IP0}:8000" \
  --prefill-url "http://${P0_IP0}:30271,http://${P1_IP0}:30271" \
  --decode-url "http://${D0_IP0}:30271" \
  --dataset /data/requests_mocked.jsonl.xz \
  --dp-size 8 \
  --max-pinned-prefix-slots 16 \
  --trunks-above-n16 0 \
  --sessions "16,32,64,128" \
  --warmup-pass-s 0 \
  --duration-per-n "${DUR_S}" \
  --cache-hit-rate 0.95 \
  --max-tokens 0 \
  --no-ignore-eos \
  --flush-each-n \
  --warmup-concurrency 16 \
  --output-json "${GCS_OUT}/C2b_glm53_2p1d_95pct_40trunks.json"
cp -f "${GCS_OUT}/C2b_glm53_2p1d_95pct_40trunks.json" "${SHM_OUT}/C2b_glm53_2p1d_95pct_40trunks.json"

echo "[C2b-LIVE] $(date -u +%FT%TZ) Running C2b Mode (a): 80% cache hit (128 trunks = FIXED_TRUNKS_PER_DP=16, dur=${DUR_S}s)..."
NUM_TOTAL_CHIPS=24 FIXED_TRUNKS_PER_DP=16 python3 -u "${SHM_ROOT}/bench_empirical_isl_sessions.py" \
  --base-url "http://${P0_IP0}:8000" \
  --prefill-url "http://${P0_IP0}:30271,http://${P1_IP0}:30271" \
  --decode-url "http://${D0_IP0}:30271" \
  --dataset /data/requests_mocked.jsonl.xz \
  --dp-size 8 \
  --max-pinned-prefix-slots 16 \
  --trunks-above-n16 0 \
  --sessions "16,32,64,128" \
  --warmup-pass-s 0 \
  --duration-per-n "${DUR_S}" \
  --cache-hit-rate 0.80 \
  --max-tokens 0 \
  --no-ignore-eos \
  --flush-each-n \
  --warmup-concurrency 16 \
  --output-json "${GCS_OUT}/C2b_glm53_2p1d_80pct_128trunks.json"
cp -f "${GCS_OUT}/C2b_glm53_2p1d_80pct_128trunks.json" "${SHM_OUT}/C2b_glm53_2p1d_80pct_128trunks.json"

echo "[C2b-LIVE] $(date -u +%FT%TZ) ALL_C2B_2P1D_DONE"
