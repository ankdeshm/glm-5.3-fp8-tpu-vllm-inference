# `GLM-5.3` FP8 Serving on TPU `v7x` with `vllm-torchtpu`

Multi-host deployment manifests, benchmark harness (`0%`, `80%`, and `95%` prefix-cache hit), speculative decoding (`MTP=3` with `IndexShare=True`), and parallelism sweeps (`TP=1, DP=32, EP=32`, `TP=2, DP=16, EP=32`, `TP=4, DP=8, EP=32`, `TP=8, DP=4, EP=32`) for **[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)** (`744B` total parameters, `755.7 GB` FP8, `147` safetensors shards) on **TPU v7x (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** using **`vllm 0.32.0` + `vllm-torchtpu` + `TorchTPU 0.2.0` GA**.

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

## 3. Repository Structure & Reproducing the Benchmarks

- [`docker/Dockerfile`](./docker/Dockerfile) — `vllm 0.32.0` + `TorchTPU 0.2.0` GA + `SemiAnalysisAI/InferenceX@f6c1f5b5`
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) — OpenAI SSE streaming + `0%/80%/95%` prefix-cache + `InferenceX` driver
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel Hugging Face -> GCS checkpoint stager
- [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py) — Memory & MXU roofline calculator for `GLM-5.3` on TPU
- [`k8s/`](./k8s/) — GKE LeaderWorkerSet & Benchmark Job YAML manifests (`v7x-16`, `v6e-16/32`, `v5p-16`)
- [`results/`](./results/README.md) — Raw JSON & Markdown benchmark outputs across all 8 passes

```bash
# 1. Stage zai-org/GLM-5.3 (755.7 GB FP8) to GCS
python3 scripts/stage_hf_to_gcs.py --repos zai-org/GLM-5.3 --bucket YOUR_GCS_BUCKET --prefix models

# 2. Deploy Benchmark Driver ConfigMap
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py

# 3A. Intra-Chip TP=2, DP=16, EP=32 (Max Throughput: 4,421.49 out tok/s, 16.60M KV tokens)
kubectl apply -f k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml
kubectl apply -f k8s/v7x-16-flex-pass5b-highconc.yaml

# 3B. Intra-Host TP=8, DP=4, EP=32 + MTP=3 (Min Latency: 72.66 stream tok/s, 13.93 ms TPOT)
kubectl apply -f k8s/v7x-16-flex-pass8-glm53-tp8-dp4-mtp3.yaml

# 3C. Interactive TP=4, DP=8, EP=32 + MTP=3 (62.14 stream tok/s, 9.04M KV tokens)
kubectl apply -f k8s/v7x-16-spot-pass2-glm53-mtp3.yaml
```
