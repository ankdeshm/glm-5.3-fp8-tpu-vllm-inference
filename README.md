# `zai-org/GLM-5.3` (744B MoE FP8) Serving & Benchmarking on Cloud TPU `v7x` with `vllm-torchtpu`

Reproducible multi-host deployment manifests, benchmark harness, speculative decoding (`MTP=3` with `IndexShare=True`), and intra-chip parallelism (`TP=2, DP=16, EP=32`) benchmarks for **[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)** (`744B` total parameters, `755.7 GB` FP8, `147` safetensors shards) on **TPU v7x (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** using **`vllm 0.32.0` + `vllm-torchtpu` + `TorchTPU 0.2.0` GA**.

---

## 1. Model & Serving Stack

- **Model:** [`https://huggingface.co/zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (`GlmMoeDsaForCausalLM`, 78 Transformer layers + 1 native Multi-Token Prediction Eagle3 layer, 256 routed experts, top-8 active routing, DeepSeek Sparse Attention `DSA` with Lightning Indexer)
- **Hardware:** 1× `v7x-16` slice (`2x2x4` topology = 4 hosts × 4 chips/host = 16 chips / **32 TensorCores**, `96 GiB` HBM3e per TensorCore = `3,072 GiB` total HBM3e)
- **Serving Engine:** `vllm 0.32.0` + `vllm_torchtpu` (`--attention-backend=CUSTOM`, Pallas Sparse MLA + SparseCore Top-K Indexer + SparseCore Hierarchical EP + EP-filtered Run:AI GCS streamer)

---

## 2. Key Benchmark Highlights (`v7x-16`, `32` TensorCores)

| Configuration | Parallelism | Cluster FP8 MLA KV Capacity | `c=1` Stream TPS (`TPOT`) | `c=8` (`80%` Prefix Hit) Out TPS | `InferenceX` (`4096:1024`, `c=8`) | Peak Slice Output Throughput |
|---|---|---:|---:|---:|---:|---:|
| **Intra-Chip Throughput (`Pass 5/5B`)** | **`TP=2, DP=16, EP=32`** | **`16,596,992` (`16.60M` tok)** | `37.34` (`26.78 ms`) | `265.23` tok/s | `971.19` tot / `190.30` out tok/s | **`4,421.49` out tok/s** *(`c=512`, `80%` hit)* |
| **Interactive `MTP=3` (`Pass 2`)** | **`TP=4, DP=8, EP=32 + MTP=3`** | `9,035,776` (`9.04M` tok) | **`62.14` (`16.31 ms`, `1.96x`)** | **`288.86` tok/s** *(`49.1` tok/s/stream)* | **`1,978.57` tot / `387.69` out tok/s (`2.18x`)** | `683.03` out tok/s *(`c=64`, `0%` hit)* |
| **Standard Baseline (`Pass 1A/1B`)** | `TP=4, DP=8, EP=32` | `9,248,768` (`9.25M` tok) | `31.83` (`31.42 ms`) | `244.20` tok/s | `905.55` tot / `177.44` out tok/s | `2,628.32` out tok/s *(`c=256`, `80%` hit)* |

### Core Engineering Takeaways
1. **Intra-Chip `TP=2, DP=16, EP=32` (`+68.2%` Peak Throughput, `+79.5%` KV Cache):** Keeping Tensor Parallelism strictly within the 2 TensorCores on a single `v7x` die eliminates inter-chip TP all-reduces and halves MLA KV-cache replication, expanding KV capacity from `9.25M` -> **`16.60M` tokens** and scaling total output throughput to **`4,421.49` output tok/s** at `c=512` (`80%` prefix-cache hit) and **`2,645.53` output tok/s** at `0%` cache (`0` failed requests).
2. **`MTP=3` Speculative Decoding (`IndexShare=True`, `1.96x–2.18x` Interactive Speedup):** Activating `--speculative-config='{"method":"mtp","num_speculative_tokens":3}'` reuses DSA Lightning Indexer top-k indices across draft positions (`index_share_for_mtp_iteration=True`), cutting single-stream TPOT from `31.45 ms` -> **`16.31 ms` (`62.14` stream tok/s)** and boosting `InferenceX` (`4096:1024`, `c=8`) throughput **`2.18x`** to **`1,978.57` total tok/s** (`15.33 ms` median TPOT).

---

## 3. Repository Structure

```text
├── README.md
├── docker/
│   └── Dockerfile                              # vllm 0.32.0 + TorchTPU 0.2.0 GA benchmark image
├── scripts/
│   ├── run_tpu_benchmark_sweep.py              # OpenAI SSE streaming + prefix-cache + InferenceX driver
│   ├── stage_hf_to_gcs.py                      # Parallel Hugging Face -> GCS checkpoint stager
│   └── glm53_roofline.py                       # Memory & MXU roofline calculator for GLM-5.3 on TPU
├── k8s/                                        # GKE LeaderWorkerSet & Benchmark Job YAML manifests
└── results/                                    # Raw JSON & Markdown benchmark outputs across all passes
```

---

## 4. Reproducing the Benchmarks

```bash
# 1. Stage zai-org/GLM-5.3 (755.7 GB FP8) to GCS
python3 scripts/stage_hf_to_gcs.py --repo-id zai-org/GLM-5.3 --gcs-prefix gs://YOUR_GCS_BUCKET/models/zai-org/GLM-5.3

# 2. Deploy Benchmark Driver ConfigMap
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py

# 3A. Run Intra-Chip TP=2, DP=16, EP=32 (Max Throughput: 4,421.49 out tok/s, 16.60M KV tokens)
kubectl apply -f k8s/v7x-16-flex-pass5-glm53-tp2-dp16.yaml
kubectl apply -f k8s/v7x-16-flex-pass5b-highconc.yaml

# 3B. Run TP=4, DP=8, EP=32 + MTP=3 IndexShare=True (Min Latency: 62.14 stream tok/s, 16.31 ms TPOT)
kubectl apply -f k8s/v7x-16-spot-pass2-glm53-mtp3.yaml
```
