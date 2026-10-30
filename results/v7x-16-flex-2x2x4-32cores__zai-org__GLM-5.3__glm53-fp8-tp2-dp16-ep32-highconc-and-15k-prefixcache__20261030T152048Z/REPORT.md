# Cloud TPU vLLM Benchmark Report — `v7x-16-flex-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T152048Z`
- **TPU Slice / Config**: `v7x-16-flex-2x2x4-32cores` (`glm53-fp8-tp2-dp16-ep32-highconc-and-15k-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3`
- **Sanity Check**: `3.432s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cached_80pct_15k | 15000 | 512 | 80% | 1 | **32.25** | **37.2** | 32.25 | 2140.1 | 2130.1 | 26.88 | 8/0 |
| cached_80pct_15k | 15000 | 512 | 80% | 8 | **237.09** | **35.82** | 29.65 | 2999.2 | 3107.7 | 27.92 | 16/0 |
| cached_80pct_15k | 15000 | 512 | 80% | 16 | **454.59** | **34.15** | 28.47 | 3014.8 | 3128.4 | 29.33 | 32/0 |
| cached_80pct_15k | 15000 | 512 | 80% | 32 | **811.48** | **31.08** | 25.62 | 3599.8 | 5174.7 | 32.32 | 64/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 64 | **1517.04** | **26.62** | 23.88 | 2233.9 | 3628.4 | 37.71 | 128/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 128 | **2389.92** | **21.57** | 19.09 | 3378.4 | 7229.7 | 46.63 | 256/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 256 | **3270.44** | **14.97** | 13.03 | 5163.7 | 11788.6 | 67.39 | 512/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 512 | **4421.49** | **12.44** | 8.94 | 15139.5 | 25531.9 | 82.44 | 512/0 |
| high_concurrency_0pct_cache | 4096 | 512 | 0% | 128 | **1686.44** | **16.79** | 13.87 | 7136.8 | 14569.7 | 61.06 | 256/0 |
| high_concurrency_0pct_cache | 4096 | 512 | 0% | 256 | **2157.78** | **10.91** | 8.99 | 11505.8 | 27461.1 | 94.31 | 512/0 |
| high_concurrency_0pct_cache | 4096 | 512 | 0% | 512 | **2645.53** | **9.15** | 5.44 | 33620.2 | 59555.9 | 118.43 | 512/0 |
