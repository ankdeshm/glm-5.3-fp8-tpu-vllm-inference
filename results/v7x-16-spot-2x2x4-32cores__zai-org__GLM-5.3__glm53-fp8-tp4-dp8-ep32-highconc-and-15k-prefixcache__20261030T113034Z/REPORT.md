# Cloud TPU vLLM Benchmark Report — `v7x-16-spot-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T113034Z`
- **TPU Slice / Config**: `v7x-16-spot-2x2x4-32cores` (`glm53-fp8-tp4-dp8-ep32-highconc-and-15k-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3`
- **Sanity Check**: `4.2s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cached_80pct_15k | 15000 | 512 | 80% | 1 | **28.6** | **31.73** | 28.61 | 1794.9 | 1806.7 | 31.51 | 8/0 |
| cached_80pct_15k | 15000 | 512 | 80% | 4 | **102.78** | **30.64** | 25.72 | 3218.9 | 3528.1 | 32.66 | 8/0 |
| cached_80pct_15k | 15000 | 512 | 80% | 8 | **217.54** | **33.11** | 27.21 | 3378.2 | 3543.1 | 30.22 | 16/0 |
| cached_80pct_15k | 15000 | 512 | 80% | 16 | **398.66** | **30.7** | 25.06 | 3837.3 | 5264.5 | 32.63 | 32/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 32 | **824.69** | **28.98** | 25.86 | 2159.3 | 3294.9 | 34.57 | 64/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 64 | **1376.97** | **24.83** | 21.73 | 3007.3 | 5411.3 | 40.46 | 128/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 128 | **2044.37** | **18.77** | 16.26 | 4384.6 | 8794.9 | 53.67 | 256/0 |
| cached_80pct_4k_highconc | 4096 | 512 | 80% | 256 | **2628.32** | **12.58** | 10.66 | 8148.7 | 18940.4 | 80.71 | 512/0 |
| high_concurrency_0pct_cache | 4096 | 512 | 0% | 128 | **1286.91** | **13.25** | 10.62 | 10104.0 | 20054.5 | 78.9 | 256/0 |
| high_concurrency_0pct_cache | 4096 | 512 | 0% | 256 | **1621.22** | **8.76** | 6.74 | 18077.0 | 38927.5 | 120.71 | 512/0 |
| high_concurrency_0pct_cache | 4096 | 512 | 0% | 512 | **1894.93** | **6.97** | 3.83 | 50800.1 | 90484.5 | 162.18 | 512/0 |
