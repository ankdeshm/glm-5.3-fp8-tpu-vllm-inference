# `GLM-5.3` FP8 Serving on TPU `v7x` with `vllm-torchtpu`

This repo has the deployment manifests, benchmark harness and results for serving **[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)** (`744B` total parameters, `755.7 GB` FP8 checkpoint, `147` safetensors shards) on TPU with vLLM + `vllm-torchtpu`.

- **Primary target:** TPU v7x-16 (`2x2x4` topology, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e).
- **Secondary target:** TPU v5p-16, with the FP8 checkpoint served as `W8A16` (Section 2B).
- **What's covered:**
  - Benchmarks at `0%`, `80%` and `95%` prefix-cache hit.
  - Speculative decoding (`MTP=3` with `IndexShare=True`).
  - A parallelism sweep across `TP=1/2/4/8` with `EP=32`.

**Result Directory Guide (`Pass 1A` → `Pass 9`):** [`results/README.md`](./results/README.md)

---

## 1. Model & Serving Stack

- **Model:** [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (`GlmMoeDsaForCausalLM`).
  - 78 Transformer layers plus 1 native Multi-Token Prediction (MTP) layer.
  - 256 routed experts with top-8 routing.
  - DeepSeek Sparse Attention (`DSA`) with a Lightning Indexer.
- **Hardware:** 1× `v7x-16` slice.
  - `2x2x4` topology: 4 hosts × 4 chips = 16 chips / **32 TensorCores**.
  - `96 GiB` HBM3e per TensorCore, `3,072 GiB` in total.
  - Roofline model: [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py).
- **Serving image (base):** `vllm/vllm-tpu:nightly-20261029-75749f8-2fb320d`, which provides vLLM + `vllm_torchtpu` + TorchTPU.
  - Started with `--attention-backend=CUSTOM`.
  - The TPU kernels used here come from this base image: Pallas Sparse MLA, the SparseCore Top-K Indexer and SparseCore hierarchical EP.
  - Pin this exact nightly tag to reproduce the numbers below.
- **Benchmark image:** built from [`docker/Dockerfile`](./docker/Dockerfile) on top of the base image. It adds:
  - [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) at commit `f6c1f5b5`.
  - The Hugging Face → GCS stager.
  - The Run:AI model streamer, used for EP-filtered weight streaming from GCS.
  - The benchmark sweep script.

  It doesn't modify vLLM or the TPU kernels. The image path used by the jobs is set in the `image:` field of the manifests under [`k8s/`](./k8s/).

> **Note:** The Dockerfile installs `runai-model-streamer` with `|| true`, so the build succeeds even if that install fails. Check that it installed (`pip show runai-model-streamer`) before deploying configurations that stream weights from GCS.

---

## 2. Benchmark Results (`v7x-16`, `32` TensorCores)

Each row links to its Kubernetes manifest, the benchmark harness function (`#line`), and the `REPORT.md` / `summary.json` / `InferenceX` JSON outputs.

| Configuration & Pass | Parallelism (`32` TC) | Cluster FP8 MLA KV | Stream TPS (`TPOT`) | `c=8` Out TPS (`80%` / `95%` prefix hit) | `InferenceX` (`4096:1024`, `c=8`)* | Peak Slice Output TPS (concurrency, prefix hit)** | Script & K8s Manifest | Result Files |
|---|---|---:|---:|---:|---:|---:|---|---|
| **Intra-Chip High Throughput (`Pass 5 / 5B`)** | **`TP=2, DP=16, EP=32`** | **`16,596,992` (`16.60M`)** | `37.34` (`26.78 ms`) | `265.23` tok/s | `971.19` tot / `190.30` out (`1.07x`) | **`4,421.49` out tok/s** *(`c=512`, `80%`)* | [`sweep.py#L355`](./scripts/run_tpu_benchmark_sweep.py#L355-L370) · [`pass5`](./k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml) · [`pass5b`](./k8s/v7x-16-flex-pass5b-highconc.yaml) | [`5A MD`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-prefixcache__20261030T145741Z/REPORT.md) · [`5B MD`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-highconc-and-15k-prefixcache__20261030T152048Z/REPORT.md) · [`5B JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-highconc-and-15k-prefixcache__20261030T152048Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-prefixcache__20261030T145741Z/inferencex_in4096_out1024_c8.json) |
| **Intra-Host Low Latency + `MTP=3` (`Pass 8`)** | **`TP=8, DP=4, EP=32 + MTP=3`** | `~4.90M` tok | **`72.66` (`13.93 ms`, `95%`, `c=4`)** · `66.24` (`c=1`) | **`310.43`** / **`338.76`** | **`2,160.34` tot / `423.31` out (`2.39x`)** | **`1,019.42` out tok/s** *(`c=64`, `95%`)* | [`sweep.py#L364`](./scripts/run_tpu_benchmark_sweep.py#L364-L371) · [`pass8.yaml`](./k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml) | [`REPORT`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp8-dp4-ep32-mtp3-prefixcache__20261030T200521Z/REPORT.md) · [`JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp8-dp4-ep32-mtp3-prefixcache__20261030T200521Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp8-dp4-ep32-mtp3-prefixcache__20261030T200521Z/inferencex_in4096_out1024_c8.json) |
| **Interactive `MTP=3` (`Pass 2`)** | **`TP=4, DP=8, EP=32 + MTP=3`** | `9,035,776` (`9.04M`) | **`62.14` (`16.31 ms`, `80%`, `c=1`)** | `288.86` tok/s | **`1,978.57` tot / `387.69` out (`2.18x`)** | `683.03` out tok/s *(`c=64`, `0%`)* | [`sweep.py#L231`](./scripts/run_tpu_benchmark_sweep.py#L231-L295) · [`pass2.yaml`](./k8s/v7x-16-spot-pass2-glm53-mtp3.yaml) | [`REPORT`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache__20261030T124702Z/REPORT.md) · [`JSON`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache__20261030T124702Z/summary.json) · [`InfX`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache__20261030T124702Z/inferencex_in4096_out1024_c8.json) |
| **Intra-Chip + `MTP=3` (`Pass 7`)** | **`TP=2, DP=16, EP=32 + MTP=3`** | **`16,187,392` (`16.19M`)** | **`53.64` (`18.80 ms`, `80%`, `c=1`)** | `256.12` tok/s | **`1,651.46` tot / `323.59` out (`1.82x`)** | `733.12` out tok/s *(`c=64`, `0%`)* | [`sweep.py#L231`](./scripts/run_tpu_benchmark_sweep.py#L231-L295) · [`pass7.yaml`](./k8s/v7x-16-flex-pass7-glm53-tp2-dp16-mtp3.yaml) | [`REPORT`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache__20261030T184305Z/REPORT.md) · [`JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache__20261030T184305Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache__20261030T184305Z/inferencex_in4096_out1024_c8.json) |
| **Zero-TP `DP=32` + `MTP=3` (`Pass 9`)** | **`TP=1, DP=32, EP=32 + MTP=3`** | **`~30M+` tok** | `40.42` (`24.74 ms`) | `224.80` / `242.15` | `1,354.12` tot / `265.34` out (`1.50x`) | `1,218.60` out tok/s *(`c=128`, `95%`)* | [`sweep.py#L364`](./scripts/run_tpu_benchmark_sweep.py#L364-L371) · [`pass9.yaml`](./k8s/v7x-16-pass9-glm53-tp1-dp32-mtp3.yaml) | [`REPORT`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp1-dp32-ep32-mtp3-prefixcache__20261030T211945Z/REPORT.md) · [`JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp1-dp32-ep32-mtp3-prefixcache__20261030T211945Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp1-dp32-ep32-mtp3-prefixcache__20261030T211945Z/inferencex_in4096_out1024_c8.json) |
| **Baseline (`Pass 1A / 1B`)** | `TP=4, DP=8, EP=32` (no MTP) | `9,248,768` (`9.25M`) | `31.83` (`31.42 ms`) | `244.20` tok/s | `905.55` tot / `177.44` out *(1.00x base)* | `2,628.32` out tok/s *(`c=256`, `80%`)* | [`sweep.py#L346`](./scripts/run_tpu_benchmark_sweep.py#L346-L354) · [`pass1`](./k8s/v7x-16-spot-glm53.yaml) · [`pass1b`](./k8s/v7x-16-spot-pass1b-highconc.yaml) | [`1A MD`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-prefixcache__20261030T104128Z/REPORT.md) · [`1B MD`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-highconc-and-15k-prefixcache__20261030T113034Z/REPORT.md) · [`InfX`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-prefixcache__20261030T104128Z/inferencex_in4096_out1024_c8.json) |

> **\* `InferenceX` methodology:** Every data point was measured directly on the TPU `v7x-16` slice using [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) at commit `f6c1f5b5` (see [`docker/Dockerfile`](./docker/Dockerfile)). The driver is `benchmark_serving.py`, called from `maybe_run_inferencex()` in [`scripts/run_tpu_benchmark_sweep.py#L231`](./scripts/run_tpu_benchmark_sweep.py#L231-L295). Speedup ratios are relative to the `Pass 1A` baseline (`905.55` total / `177.44` output tok/s). This column is the only one where every row uses identical load (`4096` in / `1024` out, `c=8`), so it is the right column for comparing configurations.
>
> **\*\* Peak slice output TPS:** Each row reports its best measured value, at the concurrency and prefix-cache hit rate shown. These differ by row (`c=64` to `c=512`, `0%` to `95%`), so this column shows each configuration's ceiling and is not a like-for-like comparison.
>
> **Stream TPS:** For MTP rows, stream TPS is slightly higher than `1000 / TPOT` because of how accepted draft tokens are averaged per request. Single-stream conditions (concurrency and prefix hit) are shown where they differ from `c=1`.

### What Each Change Does (`Pass 1A` Baseline → `Pass 5B` / `Pass 8` / `Pass 9`)

1. **Intra-chip `TP=2, DP=16, EP=32` (`Pass 5`): best for maximum throughput.**
   - **What changes:** tensor parallelism stays within the 2 TensorCores of a single `v7x` chip ([`k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml`](./k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml)). This avoids inter-chip TP all-reduces and reduces MLA KV-cache replication.
   - **KV capacity:** `9.25M` → `16.60M` tokens (`1.79x`).
   - **Peak output throughput:** `2,628.32` → `4,421.49` output tok/s at `80%` prefix hit. The baseline peak was measured at `c=256` and Pass 5B at `c=512`, so part of the gain comes from the larger KV cache supporting higher concurrency.
   - **At fixed load:** the gain on `InferenceX` (`c=8`) is `1.07x`.
2. **`MTP=3` speculative decoding (`IndexShare=True`): best for per-request latency.**
   - **How to enable:** `--speculative-config='{"method":"mtp","num_speculative_tokens":3}'`. The DSA Lightning Indexer top-k indices are reused across draft positions (`index_share_for_mtp_iteration=True`).
   - **Same parallelism (`TP=4, DP=8`), MTP isolated:** Pass 2 against the Pass 1A baseline.
     - Single-stream: `31.83` → `62.14` tok/s, with TPOT `31.42 ms` → `16.31 ms`.
     - `InferenceX`: `2.18x`.
   - **Best latency, adding `TP=8` (`Pass 8`)** ([`pass8.yaml`](./k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml)):
     - Single-stream: `66.24` tok/s at `c=1`, and `72.66` tok/s (`13.93 ms` TPOT) at `c=4` with `95%` prefix hit.
     - `InferenceX`: `2.39x`, at `2,160.34` total / `423.31` output tok/s.
   - **Trade-off:** higher TP reduces cluster KV capacity (`~4.90M` tokens) and peak slice throughput.
3. **Zero-TP `TP=1, DP=32, EP=32 + MTP=3` (`Pass 9`): best for KV capacity.**
   - **Why it works:** about `97%` of `GLM-5.3` weights (`~730 GB`) are routed MoE experts, which are sharded by `EP=32` anyway. Dropping TP removes the TP all-reduces across all 78 layers ([`k8s/v7x-16-pass9-glm53-tp1-dp32-mtp3.yaml`](./k8s/v7x-16-pass9-glm53-tp1-dp32-mtp3.yaml)).
   - **Gain:** cluster KV-cache capacity grows to `~30M+` tokens.
   - **Trade-off:** single-stream speed (`40.42` tok/s) and `InferenceX` (`1.50x`) are lower than Pass 2 and Pass 8. Choose this configuration for long-context or high-concurrency workloads where KV capacity is the constraint.

---

## 2B. TPU `v5p-16` (16 chips): FP8 checkpoint served as `W8A16`

This section uses the same `zai-org/GLM-5.3` block-FP8 checkpoint and serving stack on a TPU `v5p-16` slice (`2x2x4` = 4 hosts × 4 chips, `95 GB` HBM2e per chip = `1.52 TB`). `v5p` has no FP8 matrix unit, so the FP8 weights stay FP8 in HBM and are up-converted to BF16 for each matmul (**`W8A16`**). The KV cache is FP8, which the DSA sparse-MLA kernel requires.

| Configuration | Parallelism (16 chips) | Stream TPS (`TPOT`), `c=1` | `c=64` Out TPS (`0%` / `80%` / `95%` prefix hit) | `InferenceX` (`4096:1024`, `c=8`) | Result Files |
|---|---|---:|---:|---:|---|
| **`W8A16`, custom sparse-MLA backend** | **`TP=4, DP=4, EP=16`** | `21.61` (`46.3 ms`) | **`545.15`** / **`655.26`** / **`743.43`** | `561.70` tot / `110.06` out tok/s (`62.4 ms` TPOT, `1.00x`) | [`REPORT`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout__20261103T020625Z/REPORT.md) · [`JSON`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout__20261103T020625Z/summary.json) · [`InfX`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout__20261103T020625Z/inferencex_in4096_out1024_c8.json) |
| **`W8A16` + `MTP=3` (`IndexShare=True`)** | **`TP=4, DP=4, EP=16 + MTP=3`** | **`32.12` (`32.1 ms`)** · **`35.41` (`29.7 ms`, `80%`)** | `392.87` / `385.32` / `453.72` | **`897.99` tot / `175.95` out tok/s (`34.5 ms` TPOT, `1.60x`)** | [`REPORT`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout-mtp3__20261103T034545Z/REPORT.md) · [`JSON`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout-mtp3__20261103T034545Z/summary.json) · [`InfX`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout-mtp3__20261103T034545Z/inferencex_in4096_out1024_c8.json) |

- **Non-MTP throughput vs concurrency** (`4096` in / `1024` out, `0%` prefix hit):

  | Concurrency | `c=1` | `c=4` | `c=8` | `c=16` | `c=32` | `c=64` |
  |---|---:|---:|---:|---:|---:|---:|
  | Output tok/s | `20.18` | `82.34` | `144.87` | `249.10` | `410.50` | `545.15` |

  - At `c=64` with `95%` prefix hit, output reaches `743.43` tok/s.
  - All requests succeeded.
  - Context sweep at `c=4`: `81.94` (`2K`), `76.36` (`8K`), `69.77` (`15K`) output tok/s.
- **`MTP=3` on `v5p-16`:** enabled with `--speculative-config='{"method":"mtp","num_speculative_tokens":3}'`.
  - **Single-stream decode:** `21.61` → `32.12` tok/s (`46.3 ms` → `32.1 ms` TPOT, `1.49x`) at `0%` prefix hit, and `35.41` tok/s (`29.7 ms` TPOT) at `80%` prefix hit.
  - **`InferenceX`** (`4096:1024`, `c=8`): **`1.60x`** higher throughput, at `897.99` total / `175.95` output tok/s and `34.5 ms` mean TPOT (vs `561.70` / `110.06` at `62.4 ms`).
  - **High concurrency (`c >= 16`):** non-MTP gets higher slice throughput (`545.15`–`743.43` output tok/s at `c=64`). `v5p` MXU compute saturates earlier when it has to verify 4 draft + target tokens per step.
- **`v5p`-specific changes:** all are in [`k8s/v5p-16-glm53.yaml`](./k8s/v5p-16-glm53.yaml), with no model weight changes.
  1. The sparse-MLA KV cache uses the TensorCore layout (`TPU_SPARSE_MLA_NOPE_LAYOUT=tensorcore`, `TPU_SPARSE_MLA_ROPE_LAYOUT=tensorcore`). The default SparseCore layout gathers with 16-lane DMAs, but the `v5p` SparseCore has 8 lanes.
  2. The Lightning-Indexer top-k falls back from the SparseCore kernel to exact TensorCore `jax.lax.top_k` when `sc.num_lanes != 16`, with the same padding semantics.
  3. `--max-num-batched-tokens=512`, `--gpu-memory-utilization=0.86`, `compile_sizes=[4,16,64,256,512]`.
- **Hardware generation difference vs `v7x-16`:**
  - **`v5p-16`:** BF16 MXU compute (`W8A16` weight upcast) on 16 chips (`1.52 TB` HBM2e).
  - **`v7x-16`:** native FP8 MXU compute on 16 chips / 32 TensorCores (`3.07 TB` HBM3e).

---

## 3. Repository Structure & Reproducing the Benchmarks

- [`docker/Dockerfile`](./docker/Dockerfile): benchmark image built on `vllm/vllm-tpu:nightly-20261029-75749f8-2fb320d`. It adds `SemiAnalysisAI/InferenceX@f6c1f5b5`, the GCS/Run:AI streaming dependencies and the scripts below.
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py): the benchmark driver. It covers OpenAI SSE streaming, `0%` / `80%` / `95%` prefix-cache runs, and the `InferenceX` runs.
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py): parallel Hugging Face → GCS checkpoint stager.
- [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py): memory and MXU roofline calculator for `GLM-5.3` on TPU.
- [`k8s/`](./k8s/): GKE LeaderWorkerSet and benchmark Job YAML manifests for `v7x-16`, `v5p-16` and `v6e-16/32`. The `v6e` manifests are included for reference only; this README reports no `v6e` results.
- [`results/`](./results/README.md): raw JSON and Markdown benchmark outputs from the `v7x-16` and `v5p-16` sweeps.

```bash
# 0. Build and push the benchmark image, then set the image path in the k8s manifests
docker build -t YOUR_REGISTRY/glm53-tpu-bench:TAG docker/
docker push YOUR_REGISTRY/glm53-tpu-bench:TAG

# 1. Stage zai-org/GLM-5.3 (755.7 GB FP8) to GCS
python3 scripts/stage_hf_to_gcs.py --repos zai-org/GLM-5.3 --bucket YOUR_GCS_BUCKET --prefix models

# 2. Deploy the benchmark driver ConfigMap
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py

# 3A. v7x-16 intra-chip TP=2, DP=16, EP=32 (max throughput: 4,421.49 out tok/s at c=512, 16.60M KV tokens)
kubectl apply -f k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml
kubectl apply -f k8s/v7x-16-flex-pass5b-highconc.yaml

# 3B. v7x-16 intra-host TP=8, DP=4, EP=32 + MTP=3 (min latency: 72.66 stream tok/s, 13.93 ms TPOT)
kubectl apply -f k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml

# 3C. v7x-16 interactive TP=4, DP=8, EP=32 + MTP=3 (62.14 stream tok/s, 9.04M KV tokens)
kubectl apply -f k8s/v7x-16-spot-pass2-glm53-mtp3.yaml

# 3D. v5p-16 W8A16 TP=4, DP=4, EP=16 (optional MTP=3 via EXTRA_VLLM_ARGS)
kubectl apply -f k8s/v5p-16-glm53.yaml
```






