# Cloud TPU vLLM Benchmark Report — `v7x-16-spot-2x2x4-32cores`

- **Timestamp (UTC)**: `20261030T124702Z`
- **TPU Slice / Config**: `v7x-16-spot-2x2x4-32cores` (`glm53-fp8-tp4-dp8-ep32-mtp3-prefixcache`)
- **Served Model**: `zai-org/GLM-5.3`
- **Sanity Check**: `2.445s` — `{'role': 'assistant', 'content': None, 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'reasoning': 'The user wants an explanation of Mixture-of-Experts...`

## Sweep Results

| Suite | Input Toks | Output Toks | Prefix Hit | Concurrency | Total Out TPS | Mean Stream TPS | Mean E2E TPS | Mean TTFT (ms) | P90 TTFT (ms) | Mean TPOT (ms) | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_0pct_cache | 4096 | 1024 | 0% | 1 | **51.32** | **58.87** | 51.99 | 2287.1 | 2314.0 | 17.27 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 4 | **174.24** | **55.62** | 49.5 | 2171.3 | 3245.1 | 19.56 | 8/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 8 | **227.68** | **44.11** | 39.19 | 2466.5 | 2854.5 | 26.57 | 16/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 16 | **369.33** | **32.17** | 29.11 | 3103.0 | 4630.2 | 34.53 | 32/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 32 | **492.79** | **22.97** | 20.86 | 3709.2 | 6477.7 | 52.78 | 64/0 |
| concurrency_0pct_cache | 4096 | 1024 | 0% | 64 | **683.03** | **14.61** | 13.49 | 5403.8 | 11709.2 | 80.22 | 128/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 1 | **54.27** | **62.14** | 54.86 | 2183.6 | 2272.4 | 16.31 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 4 | **181.07** | **56.1** | 50.74 | 1927.7 | 2276.7 | 18.02 | 8/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 8 | **288.86** | **49.06** | 44.4 | 1956.0 | 2259.8 | 22.18 | 16/0 |
| cached_80pct_prefix | 4096 | 1024 | 80% | 16 | **445.48** | **36.19** | 33.35 | 2236.5 | 2955.5 | 30.4 | 32/0 |
| context_sweep | 2048 | 1024 | 0% | 4 | **160.14** | **51.15** | 47.86 | 1362.1 | 1709.9 | 19.95 | 8/0 |
| context_sweep | 8192 | 1024 | 0% | 4 | **108.49** | **36.6** | 32.19 | 3590.0 | 4079.2 | 29.15 | 8/0 |
| context_sweep | 15000 | 1024 | 0% | 4 | **100.07** | **30.28** | 25.95 | 5416.4 | 7121.0 | 34.05 | 8/0 |
