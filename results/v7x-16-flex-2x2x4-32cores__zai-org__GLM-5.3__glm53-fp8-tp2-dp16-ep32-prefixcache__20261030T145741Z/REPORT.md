# Cloud TPU vLLM Benchmark Report — `v7x-16-flex-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T145741Z`
- **TPU Slice / Config**: `v7x-16-flex-2x2x4-32cores` (`glm53-fp8-tp2-dp16-ep32-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3`
- **Sanity Check**: `4.026s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_0pct_cache | 4096 | 1024 | 0% | 1 | **34.38** | **37.28** | 34.38 | 2339.8 | 2359.0 | 26.83 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 4 | **132.33** | **36.12** | 33.09 | 2624.0 | 2669.6 | 27.69 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 8 | **252.63** | **34.63** | 31.62 | 2840.5 | 3354.3 | 28.88 | 16/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 16 | **500.71** | **34.53** | 31.33 | 3059.2 | 3297.3 | 28.97 | 32/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 32 | **881.89** | **30.8** | 27.73 | 3713.9 | 5472.7 | 32.51 | 64/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 64 | **1446.98** | **25.58** | 22.82 | 4805.7 | 7642.2 | 39.31 | 128/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 1 | **34.46** | **37.34** | 34.46 | 2320.4 | 2324.4 | 26.78 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 4 | **132.63** | **35.95** | 33.19 | 2383.8 | 2974.9 | 27.82 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 8 | **265.23** | **35.05** | 33.19 | 1662.2 | 1994.1 | 28.53 | 16/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 16 | **533.13** | **35.51** | 33.34 | 1911.3 | 1994.9 | 28.16 | 32/0 |
| context_sweep | 2048 | 1024 | 0% | 4 | **137.41** | **36.03** | 34.38 | 1393.7 | 1671.5 | 27.76 | 8/0 |
| context_sweep | 8192 | 1024 | 0% | 4 | **121.99** | **35.69** | 30.53 | 4876.0 | 5349.8 | 28.02 | 8/0 |
| context_sweep | 15000 | 1024 | 0% | 4 | **109.28** | **35.05** | 27.36 | 8239.0 | 9027.2 | 28.54 | 8/0 |
