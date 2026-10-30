# Cloud TPU vLLM Benchmark Report — `v7x-16-flex-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T184305Z`
- **TPU Slice / Config**: `v7x-16-flex-2x2x4-32cores` (`glm53-fp8-tp2-dp16-ep32-mtp3-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3`
- **Sanity Check**: `2.687s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_0pct_cache | 4096 | 1024 | 0% | 1 | **41.89** | **47.1** | 42.3 | 2456.8 | 2492.1 | 21.49 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 4 | **139.38** | **43.18** | 38.66 | 2756.3 | 3098.6 | 23.75 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 8 | **192.63** | **36.76** | 33.25 | 2818.9 | 3288.7 | 29.21 | 16/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 16 | **344.2** | **30.45** | 27.92 | 2576.0 | 2877.4 | 37.88 | 32/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 32 | **535.66** | **24.05** | 22.17 | 3260.6 | 4955.6 | 47.28 | 64/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 64 | **733.12** | **16.56** | 15.32 | 4405.0 | 8187.4 | 72.06 | 128/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 1 | **47.22** | **53.64** | 47.54 | 2449.4 | 2458.8 | 18.8 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 4 | **126.6** | **44.25** | 40.12 | 2295.4 | 2485.3 | 23.54 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 8 | **256.12** | **40.23** | 37.29 | 1933.1 | 2065.4 | 26.3 | 16/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 16 | **372.49** | **34.14** | 31.8 | 1949.9 | 2032.4 | 32.76 | 32/0 |
| context_sweep | 2048 | 1024 | 0% | 4 | **129.74** | **44.83** | 41.95 | 1529.7 | 1830.3 | 23.04 | 8/0 |
| context_sweep | 8192 | 1024 | 0% | 4 | **90.05** | **30.35** | 26.29 | 4904.2 | 5356.4 | 34.96 | 8/0 |
| context_sweep | 15000 | 1024 | 0% | 4 | **82.96** | **25.79** | 21.42 | 7946.2 | 8784.0 | 39.72 | 8/0 |
