# `GLM-5.3` FP8 Serving on TPU `v7x` (`vllm-torchtpu` & `SGLang-JAX`)

This repo has the deployment manifests, benchmark harnesses, analytical roofline models, and hardware results for serving **[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)** (`743.91B` total parameters, `40.83B` active target parameters + `1.47B` `NEXTN` MTP draft parameters, `755.7 GB` FP8 checkpoint across `147` safetensors shards) on Google Cloud TPU across **`vllm-torchtpu`** and **`SGLang-JAX` (`6a43163`)**:

- **Primary target:** TPU `v7x-16` (`2x2x4` topology, 16 physical chips / 32 TensorCores, `3,072 GiB` HBM3e).
- **Secondary target:** TPU `v5p-16`, with the FP8 checkpoint served as `W8A16` (Section 2B).
- **What's covered:**
  - **`vllm-torchtpu` (`Pass 1A` → `Pass 9`, Section 2):** Parallelism sweep (`TP=1/2/4/8` with `EP=32`), `0%` / `80%` / `95%` prefix-cache hit, and `MTP=3` (`IndexShare=True`) up to **`4,421.49` out tok/s** (`TP=2, DP=16`) and **`72.66` stream tok/s** (`13.93 ms` TPOT, `TP=8, DP=4 + MTP=3`).
  - **`SGLang-JAX` (`6a43163`, `Run B1` & `Run B2`, Section 2C):** Verified `0.0%`-hit long-context prefill (`32K` → `131K` `ISL`, **`22,577.3`–`21,284.6` uncached tok/s** with `chunked_prefill_size=2048`, **`184,893` cached prompt tok/s** at `91.1%` hit), **`NEXTN` (`4`-step / `5`-draft)** standalone decode (**`5,071.0` out tok/s** max at `n=128`, **`83.61` median stream tok/s** at `n=16`, **`2,043.0` out tok/s** gate-compliant `P90_low = 54.85` at `n=32`), analytical `HBM3e` / `MXU` rooflines, and multi-slice `xPyD` (`1P2D` / `2P1D`) disaggregation patches.

**Result Directory Guide (`Pass 1A` → `Pass 9` & `SGLang-JAX B1/B2`):** [`results/README.md`](./results/README.md)

---

## 1. Model & Serving Stack

- **Model:** [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (`GlmMoeDsaForCausalLM`).
  - 78 Transformer layers (`3` dense + `75` MoE) plus 1 native Multi-Token Prediction (`NEXTN` / `MTP`) layer.
  - 256 routed experts with top-8 routing (`40.83B` active target parameters + `1.47B` active MTP draft parameters per token).
  - DeepSeek Sparse Attention (`DSA`, top-`2048`) with a Lightning Indexer.
- **Hardware:** 1× `v7x-16` slice.
  - `2x2x4` topology: 4 hosts × 4 chips = 16 chips / **32 TensorCores**.
  - `96 GiB` HBM3e per TensorCore (`192 GiB` / chip), `3,072 GiB` in total (`7,373 GB/s` HBM3e bandwidth and `4,614` FP8 TFLOP/s per physical chip).
  - Roofline models: [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py) & [`scripts/dsv41_glm53_roofline.py`](./scripts/dsv41_glm53_roofline.py).
- **Serving stacks evaluated:**
  1. **`vllm-torchtpu` (Section 2 & 2B):** `vllm/vllm-tpu:nightly-20261029-75749f8-2fb320d` (`--attention-backend=CUSTOM`, Pallas Sparse MLA, SparseCore Top-K Indexer, SparseCore hierarchical EP), benchmarked with [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) (`f6c1f5b5`) via [`docker/Dockerfile`](./docker/Dockerfile).
  2. **`SGLang-JAX` (Section 2C):** `sglang-jax` commit `6a43163` (`DP=16, TP=2, EP=32` across `16` physical chips, `DSA` Sparse MLA top-`2048`, `--speculative-algorithm NEXTN --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5`).

> **Note:** The Dockerfile installs `runai-model-streamer` with `|| true`, so the build succeeds even if that install fails. Check that it installed (`pip show runai-model-streamer`) before deploying configurations that stream weights from GCS.

---

## 2. Benchmark Results — `vllm-torchtpu` (`v7x-16`, `32` TensorCores)

Each row links to its Kubernetes manifest, the benchmark harness function (`#line`), and the `REPORT.md` / `summary.json` / `InferenceX` JSON outputs.

| Configuration & Pass | Parallelism (`32` TC) | Cluster FP8 MLA KV | Stream TPS (`TPOT`) | `c=8` Out TPS (`80%` / `95%` prefix hit) | `InferenceX` (`4096:1024`, `c=8`)* | Peak Slice Output TPS (concurrency, prefix hit)** | Script & K8s Manifest | Result Files |
|---|---|---|---|---|---|---|---|---|
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

## 2C. `SGLang-JAX` (`6a43163`) + `NEXTN` (`4`-Step / `5`-Draft) Long-Context Prefill & Decode on TPU `v7x-16`

Evaluated on `16` physical TPU `v7x` chips (`2x2x4`, `DP=16, TP=2, EP=32`, `--speculative-algorithm NEXTN --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5`, [`scripts/bench_glm_ds_standalone.py`](./scripts/bench_glm_ds_standalone.py) & [`scripts/dsv41_glm53_roofline.py`](./scripts/dsv41_glm53_roofline.py)):

### 2C.1 `Run B1` — Uncached Long-Context Prefill (`32K` → `131K`, `0.0%` Cache Hit Verified) & Chunk Size Ablation

| Target `ISL` | Concurrency `c` | **`cp=2048` Uncached `tok/s`** (**`tok/s / chip`**) | `cp=2048` `TTFT P50 / P90` (`s`) | `cp=8192` Uncached `tok/s` (`tok/s / chip`) | **`cp=2048` Gain vs `cp=8192`** | Result Files |
|---:|---:|---:|---:|---:|---:|---|
| `32,768` | `4` | `7,725.9` (`482.9`) | `16.81 / 17.76` | `5,256.3` (`328.5`) | **`+47.0%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`cp8192`](./results/sglang-jax-v7x-16-b1-b2/B1_cp8192_uncached.json) |
| `32,768` | `8` | `13,753.9` (`859.6`) | `18.49 / 19.42` | `8,946.0` (`559.1`) | **`+53.7%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`cp8192`](./results/sglang-jax-v7x-16-b1-b2/B1_cp8192_uncached.json) |
| **`32,768`** | **`16` (`MAX`)** | **`22,577.3` (`1,411.1`)** | **`22.98 / 23.92`** | `14,190.6` (`886.9`) | **`+59.1%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`roofline`](./results/sglang-jax-v7x-16-b1-b2/analytical_rooflines.json) |
| `100,000` | `8` | `13,177.1` (`823.6`) | `60.35 / 61.49` | `10,515.3` (`657.2`) | **`+25.3%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`cp8192`](./results/sglang-jax-v7x-16-b1-b2/B1_cp8192_uncached.json) |
| **`100,000`** | **`16` (`MAX`)** | **`21,494.4` (`1,343.4`)** | **`73.65 / 74.76`** | `16,081.9` (`1,005.1`) | **`+33.7%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`roofline`](./results/sglang-jax-v7x-16-b1-b2/analytical_rooflines.json) |
| `131,072` | `8` | `12,832.5` (`802.0`) | `81.20 / 82.46` | `10,281.6` (`642.6`) | **`+24.8%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`cp8192`](./results/sglang-jax-v7x-16-b1-b2/B1_cp8192_uncached.json) |
| **`131,072`** | **`16` (`MAX`)** | **`21,284.6` (`1,330.3`)** | **`98.15 / 98.97`** | `15,822.8` (`988.9`) | **`+34.5%`** | [`cp2048`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_uncached.json) · [`roofline`](./results/sglang-jax-v7x-16-b1-b2/analytical_rooflines.json) |

- **Flat `32K` → `131K` Scaling (`DSA` Top-`2048`):** Uncached prefill throughput drops by only **`5.7%`** across a `4×` context increase (`1,411.1` → `1,330.3 tok/s/chip`).
- **RadixCache Prefix Hit (`91.06%` hit on `22.4K` `ISL`, [`B1_cp2048_cached_lockstep_on.json`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_cached_lockstep_on.json)):** Reaches **`184,893` total prompt `tok/s`** (`11,556/chip`, `TTFT P50 = 3.89 s`) at `c=32`, and **`128 / 128` (`100%`)** cold prefix-miss barrier requests succeed with `lockstep_warmup=False` ([`B1_cp2048_barrier_lockstep_off.json`](./results/sglang-jax-v7x-16-b1-b2/B1_cp2048_barrier_lockstep_off.json)).

### 2C.2 `Run B2` — Standalone `NEXTN` Decode (`ISL = 44`) vs Colocated (`ISL = 20.4K`) on `16` Physical Chips (`OSL = 1,024`)

*(Per-chip columns divide by `16` physical `v7x` chips; **`P90_low`** is the 10th-percentile slowest-stream decode TPS.)*

| Scenario | `n` (`seqs/chip`) | Operating Point | Steady Out (`tok/s`) / Drain (`tok/s`) | **Out / Physical Chip (`tok/s`)** | Median Stream TPS | **`P90_low` TPS** | `[SPEC-STEP-TIMING]` `verify / draft_4 / total` (`ms` P50) | Sharded `HBM3e` BW Util | Result JSON |
|---|---:|---|---:|---:|---:|---:|---|---:|---|
| **Pure Decode (`ISL=44`)** | `8` (`0.5`) | Pass `P90_low>=50` | `555.1` / `558.6` | `34.7` | `80.66` | **`71.71`** | `56.36 / 6.92 (10.9%) / 63.69` | `12.93%` | [`B2_isl1.json`](./results/sglang-jax-v7x-16-b1-b2/B2_glm53_decode_isl1.json) |
| **Pure Decode (`ISL=44`)** | `16` (`1.0`) | Pass `P90_low>=50` | `1,155.0` / `1,140.7` | `72.2` | **`83.61`** | **`73.65`** | `53.62 / 6.86 (11.3%) / 60.83` | **`15.68%`** | [`B2_isl1.json`](./results/sglang-jax-v7x-16-b1-b2/B2_glm53_decode_isl1.json) |
| **Pure Decode (`ISL=44`)** | **`32` (`2.0`)** | **`GATE-COMPLIANT`** | **`2,043.0` / `1,995.3`** | **`127.7`** | **`73.12`** | **`54.85`** | `59.85 / 7.69 (11.3%) / 68.12` | `14.78%` | [`B2_isl1.json`](./results/sglang-jax-v7x-16-b1-b2/B2_glm53_decode_isl1.json) |
| **Pure Decode (`ISL=44`)** | `64` (`4.0`) | High-Throughput | `3,534.1` / `3,446.0` | `220.9` | `63.13` | `31.26` | `67.61 / 8.04 (10.6%) / 75.91` | `13.42%` | [`B2_isl1.json`](./results/sglang-jax-v7x-16-b1-b2/B2_glm53_decode_isl1.json) |
| **Pure Decode (`ISL=44`)** | **`128` (`8.0`)** | **`MAX-THROUGHPUT`** | **`5,071.0` / `4,702.4`** | **`316.9`** | `46.31` | `24.60` | `93.11 / 9.40 (9.2%) / 102.67` | `9.97%` | [`B2_isl1.json`](./results/sglang-jax-v7x-16-b1-b2/B2_glm53_decode_isl1.json) · [`step_summary`](./results/sglang-jax-v7x-16-b1-b2/B2_spec_step_and_xplane_summary.json) |
| **Colocated (`20.4K`, `89.9%` hit)** | `8` → `128` | Colocated ceiling | `379.7` → **`1,733.5`** | `23.7` → **`108.3`** | `57.21` → `16.16` | `30.37` → `12.36` | Interleaved `cp=2048` prefill + decode | — | [`B2_isl20k.json`](./results/sglang-jax-v7x-16-b1-b2/B2_glm53_decode_isl20k.json) |

---

## 3. Repository Structure & Reproducing the Benchmarks

- [`docker/Dockerfile`](./docker/Dockerfile): benchmark image built on `vllm/vllm-tpu:nightly-20261029-75749f8-2fb320d`. It adds `SemiAnalysisAI/InferenceX@f6c1f5b5`, the GCS/Run:AI streaming dependencies and the scripts below.
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) & [`scripts/bench_glm_ds_standalone.py`](./scripts/bench_glm_ds_standalone.py): OpenAI SSE streaming, `0%` / `80%` / `95%` prefix-cache, `InferenceX`, and `SGLang-JAX` `B1`/`B2` benchmark drivers.
- [`scripts/wrapper_tcp.py`](./scripts/wrapper_tcp.py), [`scripts/patch_decode_multi_prefill.py`](./scripts/patch_decode_multi_prefill.py), [`scripts/patch_mini_lb_rr_decode.py`](./scripts/patch_mini_lb_rr_decode.py), [`scripts/launch_c2_slice.sh`](./scripts/launch_c2_slice.sh), [`scripts/run_c2a_1p2d_live.sh`](./scripts/run_c2a_1p2d_live.sh), [`scripts/run_c2b_2p1d_live.sh`](./scripts/run_c2b_2p1d_live.sh): multi-slice `SGLang-JAX` `1P2D` / `2P1D` TCP KV-transfer patches and disaggregated launchers.
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py): parallel Hugging Face → GCS checkpoint stager.
- [`scripts/glm53_roofline.py`](./scripts/glm53_roofline.py) & [`scripts/dsv41_glm53_roofline.py`](./scripts/dsv41_glm53_roofline.py): memory and MXU roofline calculators for `GLM-5.3` on TPU.
- [`k8s/`](./k8s/): GKE LeaderWorkerSet and benchmark Job YAML manifests for `v7x-16`, `v5p-16` and `v6e-16/32`. The `v6e` manifests are included for reference only; this README reports no `v6e` results.
- [`results/`](./results/README.md): raw JSON and Markdown benchmark outputs from the `v7x-16` (`vllm-torchtpu` & `SGLang-JAX`) and `v5p-16` sweeps.

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

---

*Note: Additional multi-slice and long-context colocation combinations (such as cross-pool `24`-chip `1P2D` disaggregation) were also attempted and encountered startup/coordination issues; see [`docs/ERROR_LOG.md`](./docs/ERROR_LOG.md) for failure details and fixes.*

