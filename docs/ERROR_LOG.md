# Troubleshooting & Error Log — Experimental & Multi-Slice Combinations

This document records experimental `GLM-5.3 FP8` (`743.91B` total / `40.83B` active) serving combinations on Cloud TPU that encountered startup deadlocks, colocation interference, or profiler/layout issues, along with root causes and recommended fixes.

---

## 1. Cross-Slice `24`-Chip Disaggregated `1P2D` (`8P + 16D`) Startup Deadlock

- **Attempted Combination:**
  - **Topology:** `1P2D` across `24` physical TPU v7x chips (`48` TensorCores): `1` Prefill slice (`2x2x2`, `8` chips on `v7x-8-pool-a`) + `2` Decode sub-slices (`2` × `2x2x2` = `16` chips on `v7x-16-pool`) running `SGLang-JAX` (`6a43163`) with `MiniLB` (`:8000`) and `BootstrapServer` (`:8998`).
- **Symptom / Where It Got Stuck:**
  - **Prefill Job (`c2-1p` on `v7x-8-pool-a`):** Both indexed pods stalled in `ErrImagePull` / `ImagePullBackOff` when pulling `<EXTERNAL_REGISTRY>/vllm-tpu:glm5_2_dsa`.
  - **Decode Job (`c2-2d` on `v7x-16-pool`):** All `4` indexed pods started immediately from node-local cached image layers (`imagePullPolicy: IfNotPresent`), published `decode_phys_{0..3}.ip` to GCS Fuse (`scratch/coord_c2_1p2d_v1/`), and hung indefinitely in the shell IP-discovery barrier (`until [ -s "${COORD_DIR}/${r_check}.ip" ]; do sleep 2; done`) waiting for `prefill_phys_0.ip` and `prefill_phys_1.ip`.
- **Root Cause:**
  1. **Short-Lived OAuth Token Expiry (`imagePullSecrets`):** The Kubernetes pull secret held a `1`-hour GCP OAuth2 access token (`oauth2accesstoken`) for a cross-project Artifact Registry. `v7x-16-pool` had cached the image (`sha256:06c0907d8261...`) during earlier standalone `16`-chip runs, whereas `v7x-8-pool-a` did not have the image cached and received `401 Unauthorized` once the OAuth token expired.
  2. **Unbounded Coordination Loop:** The worker entrypoint waited forever for peer IP files without a timeout (`COORD_TIMEOUT_SEC`) or Kubernetes `activeDeadlineSeconds`, keeping `16-pool` allocated while `pool-a` could not pull the container.
- **Recommended Fixes:**
  1. **Mirror Container Images to Same-Project Artifact Registry:** Push/mirror `glm5_2_dsa` into the cluster's own GCP project Artifact Registry so GKE node service accounts pull via native IAM (`roles/artifactregistry.reader`) with zero expiring `dockerconfigjson` secrets.
  2. **Add Barrier & Job Timeouts:** Enforce a `600 s` timeout on cross-slice IP discovery (`if [ $SECONDS -ge $DEADLINE ]; then exit 1; fi`) and set `spec.activeDeadlineSeconds` on multi-job P/D deployments so asymmetric failures release TPU slices automatically.

---

## 2. Long-Context (`20.4K` ISL) Colocated Prefill/Decode Throughput Collapse (`NEXTN`)

- **Attempted Combination:**
  - Single `16`-chip `v7x-16` (`2x2x4`, `DP=16, TP=2, EP=32`) `SGLang-JAX` instance serving `20.4K`-token prompts (`89.9%` RadixCache hit, `~2,050` uncached suffix tokens/req) colocated with `1,024`-token `NEXTN` (`4`-step) speculative decoding (`chunked_prefill_size=2048`).
- **Observed Degradation:**
  - Compared to pure decode (`ISL=44`), colocating `20.4K` prefill suffixes reduced steady output throughput by **`52%–66%`** (`3,534.1 -> 1,276.0 tok/s` at `n=64`; `5,071.0 -> 1,733.5 tok/s` at `n=128`) and pushed `10th`-percentile slow-tail stream speed (`P90_low`) down to `12.4–30.4 tok/s` across `n=8..128`.
- **Root Cause & Solution:**
  - Under `EP=32` across `4` hosts, interleaving `2,048`-token prefill chunks into decode steps stalls all `16` `DP` ranks on synchronized `AllToAll` MoE dispatch/combine barriers. Use **Prefill/Decode (`P/D`) Disaggregation** (`1P2D` or `2P1D` with project-local container registry + bounded bootstrap barrier above) for long-context (`>= 16K`) workloads requiring `P90_low >= 50 tok/s`.

---

## 3. `TPU v5p` Sparse-MLA `16`-Lane SparseCore Gather Failure Without TensorCore Layout Override

- **Attempted Combination:**
  - Running `GLM-5.3` on `TPU v5p-16` (`2x2x4`) with default `v7x` Sparse-MLA KV-cache layout (`SparseCore` layout + SparseCore Lightning-Indexer top-k).
- **Root Cause & Verified Fix:**
  - Default `v7x` Sparse-MLA gathers KV pages using `16`-lane SparseCore DMAs, whereas `TPU v5p` SparseCores have `8` lanes (`sc.num_lanes != 16`). Setting `TPU_SPARSE_MLA_NOPE_LAYOUT=tensorcore` and `TPU_SPARSE_MLA_ROPE_LAYOUT=tensorcore` (with automatic `jax.lax.top_k` TensorCore fallback in [`k8s/v5p-16-glm53.yaml`](../k8s/v5p-16-glm53.yaml)) resolves the layout mismatch cleanly.

---

## 4. `SGLang-JAX` `/stop_profile` HTTP 500 & Offline XPlane Parsing

- **Observed Issue:**
  - Calling `POST /stop_profile` on multi-host `SGLang-JAX` `2x2x4` returned `HTTP 500` even though `.xplane.pb` trace files were successfully flushed to `/dev/shm`, and the minimal serving container omitted `tensorboard_plugin_profile`.
- **Workaround:**
  - Treat `/stop_profile` HTTP 500 as non-fatal after verifying `.xplane.pb` file creation, sync `/dev/shm/.../profiles/` to GCS, and parse `.xplane.pb` in an analysis environment with `xprof` / `tensorboard-plugin-profile` installed.
