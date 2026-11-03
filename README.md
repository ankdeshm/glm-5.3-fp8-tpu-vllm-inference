# `GLM-5.3` FP8 Serving on TPU `v7x` with `vllm-torchtpu`

Multi-host deployment manifests, benchmark harness (`0%`, `80%`, and `95%` prefix-cache hit), speculative decoding (`MTP=3` with `IndexShare=True`), and parallelism sweeps (`TP=1, DP=32, EP=32`, `TP=2, DP=16, EP=32`, `TP=4, DP=8, EP=32`, `TP=8, DP=4, EP=32`) for **[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)** (`744B` total parameters, `755.7 GB` FP8, `147` safetensors shards) on **TPU v7x (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** and **TPU v5p-16 (`W8A16`, section 2B)** using **`vllm 0.32.0` + `vllm-torchtpu` + `TorchTPU 0.2.0` GA**.

- **Result Directory Guide (`Pass 1A` → `Pass 9`):** [`results/README.md`](./results/README.md)

---

## 1. Model & Serving Stack

- **Model:** [`https://huggingface.co/zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (`GlmMoeDsaForCausalLM`, 78 Transformer layers + 1 native Multi-Token Prediction Eagle3 layer, 256 routed experts, top-8 active routing, DeepSeek Sparse Attention `DSA` with Lightning Indexer)
- **Hardware:** 1× `v7x-16` slice (`2x2x4` topology = 4 hosts × 4 chips/host = 16 chips / **32 TensorCores**, `96 GiB` HBM3e per TensorCore = `3,072 GiB` total HBM3e; roofline model in [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py))
- **Serving Engine:** `vllm 0.32.0` + `vllm_torchtpu` (`--attention-backend=CUSTOM`, Pallas Sparse MLA + SparseCore Top-K Indexer + SparseCore Hierarchical EP + EP-filtered Run:AI GCS streamer built via [`docker/Dockerfile`](./docker/Dockerfile))

---

## 2. Benchmark Results (`v7x-16`, `32` TensorCores)

Each row links to the Kubernetes manifest, benchmark harness function (`#line`), and `REPORT.md` / `summary.json` / `InferenceX` JSON outputs:

| Configuration & Pass | Parallelism (`32` TC) | Cluster FP8 MLA KV | Stream TPS (`TPOT`) | `c=8` (`80%`/`95%`) Out TPS | `InferenceX` (`4096:1024`, `c=8`)* | Slice Output TPS | Script & K8s Manifest | Result Files |
|---|---|---:|---:|---:|---:|---:|---|---|
| **Intra-Chip High Throughput (`Pass 5 / 5B`)** | **`TP=2, DP=16, EP=32`** | **`16,596,992` (`16.60M`)** | `37.34` (`26.78 ms`) | `265.23` tok/s | `971.19` tot / `190.30` out (`1.07x`) | **`4,421.49` out tok/s** *(`c=512`, `80%`)* | [`sweep.py#L355`](./scripts/run_tpu_benchmark_sweep.py#L355-L370) · [`pass5`](./k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml) · [`pass5b`](./k8s/v7x-16-flex-pass5b-highconc.yaml) | [`5A MD`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-prefixcache__20261030T145741Z/REPORT.md) · [`5B MD`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-highconc-and-15k-prefixcache__20261030T152048Z/REPORT.md) · [`5B JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-highconc-and-15k-prefixcache__20261030T152048Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-prefixcache__20261030T145741Z/inferencex_in4096_out1024_c8.json) |
| **Intra-Host Low Latency + `MTP=3` (`Pass 8`)** | **`TP=8, DP=4, EP=32 + MTP=3`** | `~4.90M` tok | **`72.66` (`13.93 ms`, `95%` `c=4`)** · `66.24` (`c=1`) | **`310.43`** / **`338.76`** | **`2,160.34` tot / `423.31` out (`2.39x`)** | **`1,019.42` out tok/s** *(`c=64`, `95%`)* | [`sweep.py#L364`](./scripts/run_tpu_benchmark_sweep.py#L364-L371) · [`pass8.yaml`](./k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml) | [`REPORT`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp8-dp4-ep32-mtp3-prefixcache__20261030T200521Z/REPORT.md) · [`JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp8-dp4-ep32-mtp3-prefixcache__20261030T200521Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp8-dp4-ep32-mtp3-prefixcache__20261030T200521Z/inferencex_in4096_out1024_c8.json) |
| **Interactive `MTP=3` (`Pass 2`)** | **`TP=4, DP=8, EP=32 + MTP=3`** | `9,035,776` (`9.04M`) | **`62.14` (`16.31 ms`, `80%` `c=1`)** | `288.86` tok/s | **`1,978.57` tot / `387.69` out (`2.18x`)** | `683.03` out tok/s *(`c=64`, `0%`)* | [`sweep.py#L231`](./scripts/run_tpu_benchmark_sweep.py#L231-L295) · [`pass2.yaml`](./k8s/v7x-16-spot-pass2-glm53-mtp3.yaml) | [`REPORT`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache__20261030T124702Z/REPORT.md) · [`JSON`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache__20261030T124702Z/summary.json) · [`InfX`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache__20261030T124702Z/inferencex_in4096_out1024_c8.json) |
| **Intra-Chip + `MTP=3` (`Pass 7`)** | **`TP=2, DP=16, EP=32 + MTP=3`** | **`16,187,392` (`16.19M`)** | **`53.64` (`18.80 ms`, `80%` `c=1`)** | `256.12` tok/s | **`1,651.46` tot / `323.59` out (`1.82x`)** | `733.12` out tok/s *(`c=64`, `0%`)* | [`sweep.py#L231`](./scripts/run_tpu_benchmark_sweep.py#L231-L295) · [`pass7.yaml`](./k8s/v7x-16-flex-pass7-glm53-tp2-dp16-mtp3.yaml) | [`REPORT`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache__20261030T184305Z/REPORT.md) · [`JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache__20261030T184305Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache__20261030T184305Z/inferencex_in4096_out1024_c8.json) |
| **Zero-TP `DP=32` + `MTP=3` (`Pass 9`)** | **`TP=1, DP=32, EP=32 + MTP=3`** | **`~30M+` tok** | `40.42` (`24.74 ms`) | `224.80` (`80%`) / `242.15` (`95%`) | `1,354.12` tot / `265.34` out (`1.50x`) | `1,218.60` out tok/s *(`c=128`, `95%`)* | [`sweep.py#L364`](./scripts/run_tpu_benchmark_sweep.py#L364-L371) · [`pass9.yaml`](./k8s/v7x-16-pass9-glm53-tp1-dp32-mtp3.yaml) | [`REPORT`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp1-dp32-ep32-mtp3-prefixcache__20261030T211945Z/REPORT.md) · [`JSON`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp1-dp32-ep32-mtp3-prefixcache__20261030T211945Z/summary.json) · [`InfX`](./results/v7x-16-flex-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp1-dp32-ep32-mtp3-prefixcache__20261030T211945Z/inferencex_in4096_out1024_c8.json) |
| **Baseline (`Pass 1A / 1B`)** | `TP=4, DP=8, EP=32` (No MTP) | `9,248,768` (`9.25M`) | `31.83` (`31.42 ms`) | `244.20` tok/s | `905.55` tot / `177.44` out *(1.00x base)* | `2,628.32` out tok/s *(`c=256`, `80%`)* | [`sweep.py#L346`](./scripts/run_tpu_benchmark_sweep.py#L346-L354) · [`pass1`](./k8s/v7x-16-spot-glm53.yaml) · [`pass1b`](./k8s/v7x-16-spot-pass1b-highconc.yaml) | [`1A MD`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-prefixcache__20261030T104128Z/REPORT.md) · [`1B MD`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-highconc-and-15k-prefixcache__20261030T113034Z/REPORT.md) · [`InfX`](./results/v7x-16-spot-2x2x4-32cores__zai-org__GLM-5.3__glm53-fp8-tp4-dp8-ep32-prefixcache__20261030T104128Z/inferencex_in4096_out1024_c8.json) |

> **\*`InferenceX` Measurement Methodology:** Every data point was measured directly on the TPU `v7x-16` slice using [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) (`f6c1f5b5`, see [`docker/Dockerfile`](./docker/Dockerfile)) via `benchmark_serving.py` (`maybe_run_inferencex()` in [`scripts/run_tpu_benchmark_sweep.py#L231`](./scripts/run_tpu_benchmark_sweep.py#L231-L295)). Speedup ratios (`2.39x`, `2.18x`, `1.82x`) are relative to the `Pass 1A` baseline (`905.55` tot / `177.44` out tok/s).

### Optimization Summary (`Pass 1A` Baseline to `Pass 5B` / `Pass 8` / `Pass 9`)
1. **Intra-Chip `TP=2, DP=16, EP=32` (`Pass 1A` to `Pass 5B`, `2,628.32` to `4,421.49` out tok/s):** Keeping Tensor Parallelism within the 2 TensorCores on a single `v7x` die ([`k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml`](./k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml)) avoids inter-chip TP all-reduces and halves MLA KV-cache replication, increasing KV capacity from `9.25M` to `16.60M` tokens and scaling output throughput from `2,628.32` to `4,421.49` output tok/s at `c=512` (`80%` prefix-cache hit).
2. **`MTP=3` Speculative Decoding (`IndexShare=True`, `31.42 ms` to `13.93 ms` TPOT):** Enabling `--speculative-config='{"method":"mtp","num_speculative_tokens":3}'` reuses DSA Lightning Indexer top-k indices across draft positions (`index_share_for_mtp_iteration=True`), improving stream throughput from `31.83` tok/s (`31.42 ms` TPOT) to `72.66` tok/s (`13.93 ms` TPOT) at `95%` prefix-cache hit (`c=4`) and `2,160.34` total / `423.31` output tok/s (`2.39x`) on `InferenceX` ([`Pass 8`](./k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml)).
3. **Zero-TP `TP=1, DP=32, EP=32 + MTP=3` (`Pass 9`):** Because `~97%` of `GLM-5.3` weights (`~730 GB`) reside in routed MoE experts sharded by `EP=32`, [`k8s/v7x-16-pass9-glm53-tp1-dp32-mtp3.yaml`](./k8s/v7x-16-pass9-glm53-tp1-dp32-mtp3.yaml) removes TP all-reduces across all 78 layers while expanding cluster KV-cache capacity to `~30M+` tokens.

---

## 2B. TPU `v5p-16` (16 chips): FP8 checkpoint served as `W8A16`

Same `zai-org/GLM-5.3` block-FP8 checkpoint and serving stack on a TPU `v5p-16` slice (`2x2x4` = 4 hosts x 4 chips, `95 GB` HBM per chip = `1.52 TB`). `v5p` has no FP8 matrix unit, so the FP8 weights stay FP8 in HBM and are up-converted to BF16 for each matmul (**`W8A16`**); the KV cache is FP8 (required by the DSA sparse-MLA kernel).

| Configuration | Parallelism (16 chips) | Stream TPS (`TPOT`), `c=1` | `c=64` Out TPS (`0%` / `80%` / `95%` prefix hit) | `InferenceX` (`4096:1024`, `c=8`) | Result Files |
|---|---|---:|---:|---:|---|
| **`W8A16`, custom sparse-MLA backend** | **`TP=4, DP=4, EP=16`** | `21.61` (`46.3 ms`) | **`545.15`** / **`655.26`** / **`743.43`** | `561.70` tot / `110.06` out tok/s (`62.4 ms` TPOT, `1.00x`) | [`REPORT`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout__20261103T020625Z/REPORT.md) · [`JSON`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout__20261103T020625Z/summary.json) · [`InfX`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout__20261103T020625Z/inferencex_in4096_out1024_c8.json) |
| **`W8A16` + `MTP=3` (`IndexShare=True`)** | **`TP=4, DP=4, EP=16 + MTP=3`** | **`32.12` (`32.1 ms`)** · **`35.41` (`29.7 ms`, `80%`)** | `392.87` / `385.32` / `453.72` | **`897.99` tot / `175.95` out tok/s (`34.5 ms` TPOT, `1.60x`)** | [`REPORT`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout-mtp3__20261103T034545Z/REPORT.md) · [`JSON`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout-mtp3__20261103T034545Z/summary.json) · [`InfX`](./results/v5p-16-spot-2x2x4-16chips__zai-org__GLM-5.3__glm53-w8a16-tp4-dp4-ep16-custom-fp8kv-tclayout-mtp3__20261103T034545Z/inferencex_in4096_out1024_c8.json) |

- **Non-MTP throughput vs concurrency (`4096` in / `1024` out, `0%` prefix hit):** `20.18` (`c=1`), `82.34` (`c=4`), `144.87` (`c=8`), `249.10` (`c=16`), `410.50` (`c=32`), `545.15` (`c=64`) output tok/s (`743.43` output tok/s at `c=64`, `95%` prefix hit); all requests succeeded. Context sweep at `c=4`: `81.94` (`2K`), `76.36` (`8K`), `69.77` (`15K`) output tok/s.
- **`MTP=3` interactive latency & `InferenceX` speedup on `v5p-16`:** Enabling `--speculative-config='{"method":"mtp","num_speculative_tokens":3}'` accelerates single-stream decode from `21.61` tok/s (`46.3 ms` TPOT) to `32.12` tok/s (`32.1 ms` TPOT, `1.49x`) at `0%` prefix hit and `35.41` tok/s (`29.7 ms` TPOT) at `80%` prefix hit, while lifting `InferenceX` (`4096:1024`, `c=8`) throughput by **`1.60x`** (`897.99` total / `175.95` output tok/s at `34.5 ms` mean TPOT vs `561.70` / `110.06` at `62.4 ms`). For high-concurrency batching (`c >= 16`), non-MTP achieves higher slice throughput (`545.15`–`743.43` out tok/s at `c=64`) because `v5p` MXU compute saturates earlier when verifying 4 draft+target tokens per step.
- **`v5p`-specific changes (all in [`k8s/v5p-16-glm53.yaml`](./k8s/v5p-16-glm53.yaml), no model weight changes):** (1) the sparse-MLA KV cache uses the TensorCore layout (`TPU_SPARSE_MLA_NOPE_LAYOUT=tensorcore`, `TPU_SPARSE_MLA_ROPE_LAYOUT=tensorcore`) because the default SparseCore layout gathers with 16-lane DMAs and the `v5p` SparseCore has 8 lanes; (2) the Lightning-Indexer top-k falls back from the SparseCore kernel to exact TensorCore `jax.lax.top_k` when `sc.num_lanes != 16` (same padding semantics); (3) `--max-num-batched-tokens=512`, `--gpu-memory-utilization=0.86`, `compile_sizes=[4,16,64,256,512]`.
- **Hardware generation difference vs `v7x-16`:** `v5p-16` executes BF16 MXU compute (`W8A16` weight upcast) across 16 TensorCores (`1.52 TB` HBM2e), whereas `v7x-16` executes native FP8 MXU compute across 32 TensorCores (`3.07 TB` HBM3e).

---

## 3. Repository Structure & Reproducing the Benchmarks

- [`docker/Dockerfile`](./docker/Dockerfile) — `vllm 0.32.0` + `TorchTPU 0.2.0` GA + `SemiAnalysisAI/InferenceX@f6c1f5b5`
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) — OpenAI SSE streaming + `0%/80%/95%` prefix-cache + `InferenceX` driver
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel Hugging Face -> GCS checkpoint stager
- [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py) — Memory & MXU roofline calculator for `GLM-5.3` on TPU
- [`k8s/`](./k8s/) — GKE LeaderWorkerSet & Benchmark Job YAML manifests (`v7x-16`, `v6e-16/32`, `v5p-16`)
- [`results/`](./results/README.md) — Raw JSON & Markdown benchmark outputs across `v7x-16` and `v5p-16` sweeps

```bash
# 1. Stage zai-org/GLM-5.3 (755.7 GB FP8) to GCS
python3 scripts/stage_hf_to_gcs.py --repos zai-org/GLM-5.3 --bucket YOUR_GCS_BUCKET --prefix models

# 2. Deploy Benchmark Driver ConfigMap
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py

# 3A. v7x-16 Intra-Chip TP=2, DP=16, EP=32 (Max Throughput: 4,421.49 out tok/s, 16.60M KV tokens)
kubectl apply -f k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml
kubectl apply -f k8s/v7x-16-flex-pass5b-highconc.yaml

# 3B. v7x-16 Intra-Host TP=8, DP=4, EP=32 + MTP=3 (Min Latency: 72.66 stream tok/s, 13.93 ms TPOT)
kubectl apply -f k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml

# 3C. v7x-16 Interactive TP=4, DP=8, EP=32 + MTP=3 (62.14 stream tok/s, 9.04M KV tokens)
kubectl apply -f k8s/v7x-16-spot-pass2-glm53-mtp3.yaml

# 3D. v5p-16 W8A16 TP=4, DP=4, EP=16 (+ optional MTP=3 via EXTRA_VLLM_ARGS)
kubectl apply -f k8s/v5p-16-glm53.yaml
```
