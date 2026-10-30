# Cloud TPU vLLM Benchmark Report — `v7x-16-spot-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T104128Z`
- **TPU Slice / Config**: `v7x-16-spot-2x2x4-32cores` (`glm53-fp8-tp4-dp8-ep32-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3`
- **Sanity Check**: `5.398s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_0pct_cache | 4096 | 1024 | 0% | 1 | **29.99** | **31.83** | 29.99 | 2007.7 | 2024.8 | 31.42 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 4 | **117.46** | **30.88** | 29.42 | 1718.6 | 2913.5 | 32.39 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 8 | **235.99** | **32.53** | 29.54 | 3206.7 | 3700.8 | 30.75 | 16/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 16 | **429.85** | **29.66** | 26.95 | 3488.7 | 5150.0 | 33.76 | 32/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 32 | **768.41** | **27.26** | 24.19 | 4876.1 | 7455.5 | 36.77 | 64/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 64 | **1252.61** | **22.67** | 19.93 | 6549.4 | 11474.9 | 44.46 | 128/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 1 | **30.11** | **31.8** | 30.11 | 1839.8 | 1981.5 | 31.45 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 4 | **117.72** | **30.96** | 29.44 | 1730.3 | 1995.0 | 32.31 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 8 | **244.2** | **32.67** | 30.56 | 2190.9 | 2570.8 | 30.62 | 16/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 16 | **468.62** | **31.56** | 29.33 | 2484.9 | 3086.2 | 31.7 | 32/0 |
| context_sweep | 2048 | 1024 | 0% | 4 | **118.77** | **30.81** | 29.71 | 1269.8 | 1647.6 | 32.45 | 8/0 |
| context_sweep | 8192 | 1024 | 0% | 4 | **107.84** | **30.28** | 26.98 | 4158.4 | 5095.4 | 33.04 | 8/0 |
| context_sweep | 15000 | 1024 | 0% | 4 | **98.85** | **28.62** | 24.81 | 5555.4 | 7459.9 | 35.0 | 8/0 |
