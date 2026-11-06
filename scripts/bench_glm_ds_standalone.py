#!/usr/bin/env python3
"""Standalone Prefill & Decode Benchmark Harness for Atlas Cloud TPU v7x Rerun (B1, B2, A4).

Supports:
  1. Mode `prefill_uncached`:
     - Sweeps `ISL x concurrency c` (`>= min_requests` per cell, `max_tokens=1`, `ignore_eos=False`).
     - Flushes prefix cache before each cell and prepends a unique UUID nonce + pseudo-random
       vocabulary tokens so `cached_tokens == 0` (`0%` hit) is guaranteed and verified via
       `usage.prompt_tokens_details.cached_tokens`.
     - Distributes requests across DP ranks (`dp_rank` body field for sglang-jax,
       `X-data-parallel-rank` header for vLLM) and logs uncached `tok/s`, `tok/s/chip`,
       mean/P50/P90/P99 TTFT, and DP-rank skew.
  2. Mode `prefill_cached`:
     - Sweeps `c` at `cached_prefix_tokens` (~20K) + `uncached_suffix_tokens` (~2K), `max_tokens=1`.
     - Supports `--lockstep-warmup` (Stage-1 deterministic 1-per-DP-rank warmup before timed cell)
       vs `--no-lockstep-warmup` (flush cache, then fire `c` concurrent requests cold with natural
       routing) to execute the B1 Barrier Test cleanly.
  3. Mode `decode_profile`:
     - Runs GLM-5.3 decode-only profile (B2) for Scenario 1 (`ISL ~ 1`) or Scenario 2 (`ISL ~ 20K`)
       across `n` sessions (`8, 16, 32, 64, 128`), `ignore_eos=False`, continuous window + drain.
     - Optionally triggers `/start_profile` + `/stop_profile` mid-window at each `n` to capture
       one XPlane profiler trace per `n`.
     - Computes batch/chip (`n / num_chips`), realized OSL, finish_reason breakdown,
       P50/P90_low/P90_hi stream TPS, P50/P90/P99 TTFT (all requests & in-window),
       NEXTN / spec metrics, and first-order HBM-BW utilization + MFU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import lzma
import math
from pathlib import Path
import random
import re
import statistics
import time
import uuid
from typing import Any
import aiohttp


VOCAB_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu", "tensor", "matrix", "vector", "kernel", "router", "expert",
    "latent", "attention", "sparse", "indexer", "prefill", "decode", "pipeline", "shard",
    "replica", "cluster", "network", "memory", "bandwidth", "compute", "latency", "throughput",
    "quantum", "photon", "neutron", "proton", "electron", "silicon", "wafer", "circuit",
    "compiler", "schedule", "barrier", "collective", "allgather", "scatter", "reduce", "fusion",
]


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def make_uncompressible_text(seed_val: int, approx_tokens: int, chars_per_tok: float) -> str:
    """Generates deterministic pseudo-random word sequence with zero accidental prefix sharing."""
    rng = random.Random(seed_val)
    target_chars = max(64, int(approx_tokens * chars_per_tok))
    nonce = f"[NONCE-{seed_val}-{uuid.uuid4().hex}] "
    chunks = [nonce]
    cur = len(nonce)
    block_words = [rng.choice(VOCAB_WORDS) for _ in range(512)]
    idx = 0
    while cur < target_chars:
        # Permute small slices so blocks never repeat verbatim across pages
        w1 = block_words[idx % 512]
        w2 = VOCAB_WORDS[(idx * 7 + seed_val) % len(VOCAB_WORDS)]
        piece = f"{w1} {w2} {idx % 997} "
        chunks.append(piece)
        cur += len(piece)
        idx += 1
    return "".join(chunks)[:target_chars]


async def flush_cache(session: aiohttp.ClientSession, base_url: str, engine: str) -> str:
    ep = "/flush_cache" if engine == "sglang" else "/reset_prefix_cache"
    url = f"{base_url.rstrip('/')}{ep}"
    try:
        async with session.post(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            txt = await resp.text()
            return f"HTTP {resp.status}: {txt[:120]}"
    except Exception as e:
        return f"ERR: {e!r}"


async def calibrate_chars_per_token(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    engine: str,
    init_cpt: float = 5.2,
) -> float:
    """Sends one ~4096-token calibration request to measure exact tokenizer chars/token."""
    test_toks = 4096
    txt = make_uncompressible_text(999991, test_toks, init_cpt)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": txt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    if engine == "sglang":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with session.post(url, json=payload) as resp:
        body = await resp.json()
    actual_toks = int(body.get("usage", {}).get("prompt_tokens", test_toks))
    cpt = len(txt) / max(1, actual_toks)
    print(f"[CALIBRATE] len_chars={len(txt)} -> prompt_tokens={actual_toks} => chars_per_token={cpt:.4f}", flush=True)
    return cpt


async def send_streaming_chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    t_first: float | None = None
    chunk_times: list[float] = []
    itls: list[float] = []
    t_prev: float | None = None
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0
    finish_reason: str | None = None
    err: str | None = None
    text_len = 0

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body_err = await resp.text()
                return {
                    **meta,
                    "t_start": t0,
                    "t_first": None,
                    "t_end": time.perf_counter(),
                    "ttft": None,
                    "e2e": time.perf_counter() - t0,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": None,
                    "decode_tps": None,
                    "per_token_itl": None,
                    "itls": [],
                    "chunk_times": [],
                    "error": f"HTTP {resp.status}: {body_err[:300]}",
                }
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    ev = json.loads(data_str)
                except Exception:
                    continue
                now = time.perf_counter()
                choices = ev.get("choices") or []
                if choices:
                    ch0 = choices[0]
                    delta = ch0.get("delta") or {}
                    tok_piece = (
                        (delta.get("content") or "")
                        + (delta.get("reasoning_content") or "")
                        + (json.dumps(delta["tool_calls"]) if delta.get("tool_calls") else "")
                    )
                    if tok_piece:
                        text_len += len(tok_piece)
                        if t_first is None:
                            t_first = now
                        else:
                            itls.append(now - (t_prev or t_first))
                        t_prev = now
                        chunk_times.append(now)
                    if ch0.get("finish_reason"):
                        finish_reason = ch0["finish_reason"]
                usage = ev.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                    ptd = usage.get("prompt_tokens_details") or {}
                    if ptd.get("cached_tokens") is not None:
                        cached_tokens = int(ptd["cached_tokens"])
                    if t_first is None and completion_tokens > 0:
                        t_first = now
                        chunk_times.append(now)
    except Exception as exc:
        err = repr(exc)

    t_end = time.perf_counter()
    if t_first is None and err is None:
        t_first = t_end
    ttft = (t_first - t0) if t_first is not None else None
    e2e = t_end - t0
    decode_tps: float | None = None
    per_tok_itl: float | None = None
    if t_first is not None and completion_tokens > 1 and (t_end - t_first) > 1e-6:
        decode_tps = (completion_tokens - 1) / (t_end - t_first)
        per_tok_itl = (t_end - t_first) / (completion_tokens - 1)

    return {
        **meta,
        "t_start": t0,
        "t_first": t_first,
        "t_end": t_end,
        "ttft": ttft,
        "e2e": e2e,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "decode_tps": decode_tps,
        "per_token_itl": per_tok_itl,
        "itls": itls,
        "chunk_times": chunk_times,
        "error": err,
    }


async def scrape_metrics_raw(session: aiohttp.ClientSession, base_url: str) -> str:
    try:
        async with session.get(f"{base_url.rstrip('/')}/metrics", timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                return await r.text()
    except Exception:
        pass
    return ""


def parse_spec_accept_from_metrics(m0: str, m1: str) -> dict[str, Any]:
    def _extract(txt: str) -> dict[str, float]:
        sums: dict[str, float] = {}
        for line in txt.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].split("{")[0]
                if any(k in key for k in ("spec", "accept", "draft", "verify")):
                    try:
                        sums[key] = sums.get(key, 0.0) + float(parts[-1])
                    except ValueError:
                        pass
        return sums

    d0, d1 = _extract(m0), _extract(m1)
    deltas = {k: round(d1.get(k, 0.0) - d0.get(k, 0.0), 4) for k in set(d0) | set(d1) if (d1.get(k, 0.0) - d0.get(k, 0.0)) != 0}
    return deltas


async def run_prefill_uncached(args: argparse.Namespace, session: aiohttp.ClientSession, cpt: float) -> list[dict[str, Any]]:
    isls = [int(x) for x in args.isls.split(",") if x.strip()]
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    summaries: list[dict[str, Any]] = []

    # Quick XLA pre-warmup across shapes
    if not args.skip_warmup:
        print(f"[WARMUP] Running uncached prefill XLA warmup on {args.base_url} ...", flush=True)
        for w_isl in isls:
            txt = make_uncompressible_text(700000 + w_isl, max(1024, w_isl - 32), cpt)
            p: dict[str, Any] = {
                "model": args.model,
                "messages": [{"role": "user", "content": txt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if args.engine == "sglang":
                p["dp_rank"] = 0
                p["chat_template_kwargs"] = {"enable_thinking": False}
            hdrs = {"Content-Type": "application/json"}
            if args.engine == "vllm":
                hdrs["X-data-parallel-rank"] = "0"
            await send_streaming_chat(session, url, p, hdrs, {"warmup": True})

    seed_counter = 1000
    for isl in isls:
        for c in concs:
            n_reqs = max(args.min_requests, c * 4)
            fl = await flush_cache(session, args.base_url, args.engine)
            print(f"\n>>> [UNCACHED PREFILL] ISL={isl:,} | c={c} | n_reqs={n_reqs} | flush={fl} <<<", flush=True)

            # Pre-build payloads so client string generation is not inside the timed window
            built_reqs = []
            for i in range(n_reqs):
                seed_counter += 1
                dp_r = i % args.dp_size
                txt = make_uncompressible_text(seed_counter, max(256, isl - 24), cpt)
                payload: dict[str, Any] = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": txt}],
                    "max_tokens": 1,
                    "ignore_eos": False,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                headers = {"Content-Type": "application/json"}
                if args.engine == "sglang":
                    payload["dp_rank"] = dp_r
                    payload["rid"] = f"unc_isl{isl}_c{c}_i{i}_dp{dp_r}_{uuid.uuid4().hex[:8]}"
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                else:
                    headers["X-data-parallel-rank"] = str(dp_r)
                built_reqs.append((payload, headers, {"req_idx": i, "dp_rank": dp_r, "target_isl": isl}))

            sem = asyncio.Semaphore(c)
            idx_cursor = 0
            lock = asyncio.Lock()

            async def _worker() -> list[dict[str, Any]]:
                nonlocal idx_cursor
                out = []
                while True:
                    async with lock:
                        if idx_cursor >= len(built_reqs):
                            break
                        my_i = idx_cursor
                        idx_cursor += 1
                    pl, hd, mt = built_reqs[my_i]
                    async with sem:
                        res = await send_streaming_chat(session, url, pl, hd, mt)
                        out.append(res)
                return out

            t0 = time.perf_counter()
            worker_out = await asyncio.gather(*(_worker() for _ in range(c)))
            wall = time.perf_counter() - t0
            results = [r for lst in worker_out for r in lst]
            ok = [r for r in results if r["error"] is None]
            errs = [r for r in results if r["error"] is not None]

            tot_prompt = sum(r["prompt_tokens"] for r in ok)
            tot_cached = sum(r["cached_tokens"] for r in ok)
            tot_uncached = max(0, tot_prompt - tot_cached)
            ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]

            # Per-DP-rank skew stats
            per_dp_reqs = {r: 0 for r in range(args.dp_size)}
            per_dp_ttft: dict[int, list[float]] = {r: [] for r in range(args.dp_size)}
            for r_item in ok:
                dr = int(r_item["dp_rank"])
                per_dp_reqs[dr] = per_dp_reqs.get(dr, 0) + 1
                if r_item["ttft"] is not None:
                    per_dp_ttft.setdefault(dr, []).append(r_item["ttft"])
            dp_mean_ttfts = [statistics.mean(v) for v in per_dp_ttft.values() if v]
            dp_skew_ratio = (max(dp_mean_ttfts) / min(dp_mean_ttfts)) if len(dp_mean_ttfts) >= 2 and min(dp_mean_ttfts) > 0 else 1.0

            uncached_tps = tot_uncached / wall if wall > 0 else 0.0
            per_chip_tps = uncached_tps / max(1, args.num_chips)
            cell_sum = {
                "mode": "prefill_uncached",
                "target_isl": isl,
                "concurrency": c,
                "num_chips": args.num_chips,
                "dp_size": args.dp_size,
                "chunked_prefill_size_label": args.label,
                "total_requests": len(results),
                "ok_requests": len(ok),
                "failed_requests": len(errs),
                "errors_sample": [e["error"] for e in errs[:3]],
                "wall_clock_s": round(wall, 4),
                "mean_prompt_tokens": round(tot_prompt / len(ok), 1) if ok else 0.0,
                "mean_cached_tokens": round(tot_cached / len(ok), 1) if ok else 0.0,
                "observed_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 3) if tot_prompt > 0 else 0.0,
                "uncached_tok_per_s": round(uncached_tps, 2),
                "uncached_tok_per_s_per_chip": round(per_chip_tps, 2),
                "ttft_s": {
                    "mean": round(statistics.mean(ttfts), 4) if ttfts else 0.0,
                    "p50": round(percentile(ttfts, 50), 4),
                    "p90": round(percentile(ttfts, 90), 4),
                    "p99": round(percentile(ttfts, 99), 4),
                },
                "dp_rank_skew": {
                    "requests_per_rank": per_dp_reqs,
                    "mean_ttft_per_rank_s": {k: round(statistics.mean(v), 4) for k, v in per_dp_ttft.items() if v},
                    "max_over_min_ttft_ratio": round(dp_skew_ratio, 4),
                },
            }
            print(
                f"  [DONE ISL={isl:,} c={c}] OK={len(ok)}/{len(results)} | MeanISL={cell_sum['mean_prompt_tokens']:,.0f} "
                f"(Hit={cell_sum['observed_cache_hit_pct']:.2f}%) | Uncached={uncached_tps:,.1f} tok/s "
                f"({per_chip_tps:,.1f} tok/s/chip) | TTFT Mean={cell_sum['ttft_s']['mean']:.3f}s "
                f"P50={cell_sum['ttft_s']['p50']:.3f}s P90={cell_sum['ttft_s']['p90']:.3f}s | DP_skew={dp_skew_ratio:.2f}x",
                flush=True,
            )
            summaries.append(cell_sum)
            if args.output_json:
                Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output_json, "w") as f:
                    json.dump(summaries, f, indent=2)
    return summaries


async def run_prefill_cached(args: argparse.Namespace, session: aiohttp.ClientSession, cpt: float) -> list[dict[str, Any]]:
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    summaries: list[dict[str, Any]] = []

    prefix_toks = args.cached_prefix_tokens
    suffix_toks = args.uncached_suffix_tokens
    trunks_per_dp = max(1, args.trunks_per_dp)

    for c in concs:
        n_reqs = max(args.min_requests, c * 4)
        fl = await flush_cache(session, args.base_url, args.engine)
        print(
            f"\n>>> [CACHED PREFILL] prefix={prefix_toks:,} + suffix={suffix_toks:,} | c={c} | "
            f"lockstep_warmup={args.lockstep_warmup} | n_reqs={n_reqs} | flush={fl} <<<",
            flush=True,
        )

        # Build shared prefix strings per (dp_rank, trunk_slot)
        shared_prefixes: dict[tuple[int, int], str] = {}
        for r in range(args.dp_size):
            for k in range(trunks_per_dp):
                shared_prefixes[(r, k)] = make_uncompressible_text(
                    500000 + c * 1000 + r * 32 + k, prefix_toks, cpt
                )

        warmup_wall_s = 0.0
        if args.lockstep_warmup:
            t_w0 = time.perf_counter()
            dp_sems = {r: asyncio.Semaphore(1) for r in range(args.dp_size)}

            async def _warm(r: int, k: int) -> dict[str, Any]:
                async with dp_sems[r]:
                    w_txt = shared_prefixes[(r, k)] + " " + make_uncompressible_text(800000 + r * 32 + k, 128, cpt)
                    pl: dict[str, Any] = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": w_txt}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    hd = {"Content-Type": "application/json"}
                    if args.engine == "sglang":
                        pl["dp_rank"] = r
                        pl["rid"] = f"warm_c{c}_dp{r}_k{k}_{uuid.uuid4().hex[:6]}"
                        pl["chat_template_kwargs"] = {"enable_thinking": False}
                    else:
                        hd["X-data-parallel-rank"] = str(r)
                    return await send_streaming_chat(session, url, pl, hd, {"dp_rank": r, "trunk": k})

            w_res = await asyncio.gather(*(_warm(r, k) for r in range(args.dp_size) for k in range(trunks_per_dp)))
            warmup_wall_s = time.perf_counter() - t_w0
            w_ok = sum(1 for x in w_res if x["error"] is None)
            print(f"  [Lockstep Warmup] Seeded {w_ok}/{len(w_res)} trunks in {warmup_wall_s:.2f}s", flush=True)
            await asyncio.sleep(0.5)

        built_reqs = []
        for i in range(n_reqs):
            dp_r = i % args.dp_size
            k_slot = (i // args.dp_size) % trunks_per_dp
            # When lockstep_warmup is OFF in barrier test, we still use the shared prefixes across requests,
            # and test whether concurrent unlocked arrivals hit the radix cache or collide/miss!
            full_txt = shared_prefixes[(dp_r, k_slot)] + "\n" + make_uncompressible_text(900000 + c * 10000 + i, suffix_toks, cpt)
            pl = {
                "model": args.model,
                "messages": [{"role": "user", "content": full_txt}],
                "max_tokens": 1,
                "ignore_eos": False,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            hd = {"Content-Type": "application/json"}
            if args.engine == "sglang":
                if args.lockstep_warmup or not args.unpin_dp_when_no_warmup:
                    pl["dp_rank"] = dp_r
                pl["rid"] = f"cached_c{c}_i{i}_dp{dp_r}_{uuid.uuid4().hex[:8]}"
                pl["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                hd["X-data-parallel-rank"] = str(dp_r)
            built_reqs.append((pl, hd, {"req_idx": i, "dp_rank": dp_r, "trunk": k_slot}))

        sem = asyncio.Semaphore(c)
        idx_cursor = 0
        lock = asyncio.Lock()

        async def _worker() -> list[dict[str, Any]]:
            nonlocal idx_cursor
            out = []
            while True:
                async with lock:
                    if idx_cursor >= len(built_reqs):
                        break
                    my_i = idx_cursor
                    idx_cursor += 1
                pl, hd, mt = built_reqs[my_i]
                async with sem:
                    out.append(await send_streaming_chat(session, url, pl, hd, mt))
            return out

        t0 = time.perf_counter()
        worker_out = await asyncio.gather(*(_worker() for _ in range(c)))
        wall = time.perf_counter() - t0
        results = [r for lst in worker_out for r in lst]
        ok = [r for r in results if r["error"] is None]
        errs = [r for r in results if r["error"] is not None]

        tot_prompt = sum(r["prompt_tokens"] for r in ok)
        tot_cached = sum(r["cached_tokens"] for r in ok)
        tot_uncached = max(0, tot_prompt - tot_cached)
        ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]

        per_dp_ttft: dict[int, list[float]] = {r: [] for r in range(args.dp_size)}
        for r_item in ok:
            if r_item["ttft"] is not None:
                per_dp_ttft.setdefault(int(r_item["dp_rank"]), []).append(r_item["ttft"])
        dp_mean_ttfts = [statistics.mean(v) for v in per_dp_ttft.values() if v]
        dp_skew_ratio = (max(dp_mean_ttfts) / min(dp_mean_ttfts)) if len(dp_mean_ttfts) >= 2 and min(dp_mean_ttfts) > 0 else 1.0

        total_in_tps = tot_prompt / wall if wall > 0 else 0.0
        uncached_tps = tot_uncached / wall if wall > 0 else 0.0
        cell_sum = {
            "mode": "prefill_cached",
            "label": args.label,
            "lockstep_warmup": args.lockstep_warmup,
            "warmup_wall_s": round(warmup_wall_s, 3),
            "target_cached_prefix_tokens": prefix_toks,
            "target_uncached_suffix_tokens": suffix_toks,
            "concurrency": c,
            "num_chips": args.num_chips,
            "dp_size": args.dp_size,
            "total_requests": len(results),
            "ok_requests": len(ok),
            "failed_requests": len(errs),
            "wall_clock_s": round(wall, 4),
            "mean_prompt_tokens": round(tot_prompt / len(ok), 1) if ok else 0.0,
            "mean_cached_tokens": round(tot_cached / len(ok), 1) if ok else 0.0,
            "mean_uncached_tokens": round(tot_uncached / len(ok), 1) if ok else 0.0,
            "observed_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 2) if tot_prompt > 0 else 0.0,
            "total_input_tok_per_s": round(total_in_tps, 2),
            "total_input_tok_per_s_per_chip": round(total_in_tps / max(1, args.num_chips), 2),
            "uncached_tok_per_s": round(uncached_tps, 2),
            "uncached_tok_per_s_per_chip": round(uncached_tps / max(1, args.num_chips), 2),
            "ttft_s": {
                "mean": round(statistics.mean(ttfts), 4) if ttfts else 0.0,
                "p50": round(percentile(ttfts, 50), 4),
                "p90": round(percentile(ttfts, 90), 4),
                "p99": round(percentile(ttfts, 99), 4),
            },
            "dp_rank_skew": {
                "mean_ttft_per_rank_s": {k: round(statistics.mean(v), 4) for k, v in per_dp_ttft.items() if v},
                "max_over_min_ttft_ratio": round(dp_skew_ratio, 4),
            },
        }
        print(
            f"  [DONE CACHED c={c} lockstep={args.lockstep_warmup}] OK={len(ok)}/{len(results)} | "
            f"ISL={cell_sum['mean_prompt_tokens']:,.0f} (Hit={cell_sum['observed_cache_hit_pct']:.1f}%) | "
            f"TotIn={total_in_tps:,.1f} tok/s ({total_in_tps/args.num_chips:,.1f}/chip) | "
            f"Uncached={uncached_tps:,.1f} tok/s | TTFT Mean={cell_sum['ttft_s']['mean']:.3f}s P90={cell_sum['ttft_s']['p90']:.3f}s | Skew={dp_skew_ratio:.2f}x",
            flush=True,
        )
        summaries.append(cell_sum)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(summaries, f, indent=2)
    return summaries


async def trigger_profile(session: aiohttp.ClientSession, base_url: str, profile_dir: str, tag: str) -> dict[str, Any]:
    out_dir = f"{profile_dir.rstrip('/')}/{tag}"
    try:
        async with session.post(
            f"{base_url.rstrip('/')}/start_profile",
            json={"output_dir": out_dir, "num_steps": 5},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r1:
            s1 = await r1.text()
        await asyncio.sleep(2.0)
        async with session.post(
            f"{base_url.rstrip('/')}/stop_profile",
            json={},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r2:
            s2 = await r2.text()
        return {"output_dir": out_dir, "start_resp": s1[:200], "stop_resp": s2[:200]}
    except Exception as e:
        return {"output_dir": out_dir, "error": repr(e)}


async def run_decode_profile(args: argparse.Namespace, session: aiohttp.ClientSession, cpt: float) -> list[dict[str, Any]]:
    sessions_list = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    summaries: list[dict[str, Any]] = []

    short_prompts = [
        "Write a detailed, comprehensive 40-step technical derivation of distributed Mixture-of-Experts routing, MLA latent KV compression, and speculative decoding rooflines.",
        "Provide an exhaustive numbered engineering reference (items 1 through 50) covering PCIe, ICI torus topologies, HBM3e bandwidth bounds, and XLA collective scheduling.",
        "Derive step-by-step from first principles the queueing theory of open-loop M/G/k serving systems, prefix radix trees, and token-level acceptance probabilities.",
        "Explain in deep mathematical detail 40 distinct compiler optimizations in XLA/Pallas for quantized blockwise FP8 matrix multiplication on systolic arrays.",
    ]

    # Build Scenario-2 (~20K ISL) shared trunks per DP rank if target_isl > 512
    shared_20k: dict[int, str] = {}
    if args.decode_isl >= 1024:
        for r in range(args.dp_size):
            shared_20k[r] = make_uncompressible_text(300000 + r, int(args.decode_isl * 0.90), cpt)

    for n in sessions_list:
        fl = await flush_cache(session, args.base_url, args.engine)
        print(f"\n>>> [DECODE PROFILE B2] Scenario ISL~{args.decode_isl} | n={n} (batch/chip={n/args.num_chips:.2f}) | duration={args.duration_s}s | ignore_eos=False <<<", flush=True)

        # If ISL >= 1024 and radix cache is active, warm one trunk per DP rank first
        if args.decode_isl >= 1024:
            async def _w20(r: int) -> None:
                pl = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": shared_20k[r] + "\nSummarize the key points."}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "dp_rank": r,
                    "rid": f"w20k_n{n}_dp{r}_{uuid.uuid4().hex[:6]}",
                    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high", "clear_thinking": False},
                }
                await send_streaming_chat(session, url, pl, {"Content-Type": "application/json"}, {"warmup": True})

            await asyncio.gather(*(_w20(r) for r in range(min(n, args.dp_size))))
            await asyncio.sleep(0.5)

        m0 = await scrape_metrics_raw(session, args.base_url)
        t_start = time.perf_counter()
        t_win_end = t_start + args.duration_s

        async def _session_loop(s_id: int) -> list[dict[str, Any]]:
            dp_r = s_id % args.dp_size
            turn = 0
            out = []
            while time.perf_counter() < t_win_end:
                q = short_prompts[(s_id + turn) % len(short_prompts)]
                if args.decode_isl >= 1024:
                    suf = make_uncompressible_text(400000 + n * 1000 + s_id * 50 + turn, max(256, int(args.decode_isl * 0.10)), cpt)
                    user_txt = f"{shared_20k[dp_r]}\n{suf}\n\n{q}"
                else:
                    user_txt = q
                pl: dict[str, Any] = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": user_txt}],
                    "max_tokens": args.max_tokens,
                    "ignore_eos": False,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                hd = {"Content-Type": "application/json"}
                if args.engine == "sglang":
                    pl["dp_rank"] = dp_r
                    pl["rid"] = f"b2_isl{args.decode_isl}_n{n}_s{s_id}_t{turn}_{uuid.uuid4().hex[:6]}"
                    pl["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": "high", "clear_thinking": False}
                else:
                    hd["X-data-parallel-rank"] = str(dp_r)
                res = await send_streaming_chat(session, url, pl, hd, {"session_id": s_id, "dp_rank": dp_r, "turn": turn})
                out.append(res)
                turn += 1
            return out

        prof_info: dict[str, Any] = {}
        async def _profiler_task() -> None:
            nonlocal prof_info
            if not args.profile_dir:
                return
            await asyncio.sleep(min(25.0, args.duration_s * 0.4))
            prof_info = await trigger_profile(session, args.base_url, args.profile_dir, f"isl{args.decode_isl}_n{n}")
            print(f"  [PROFILE n={n}] {prof_info}", flush=True)

        worker_lists, _ = await asyncio.gather(
            asyncio.gather(*(_session_loop(s) for s in range(n))),
            _profiler_task(),
        )
        t_end = time.perf_counter()
        m1 = await scrape_metrics_raw(session, args.base_url)
        wall = t_end - t_start

        results = [r for lst in worker_lists for r in lst]
        ok = [r for r in results if r["error"] is None]
        errs = [r for r in results if r["error"] is not None]

        win_out_toks = 0.0
        for r in ok:
            compl = float(r["completion_tokens"] or 0)
            cts = r.get("chunk_times") or []
            if r["t_end"] <= t_win_end:
                win_out_toks += compl
            elif cts:
                in_w = sum(1 for ct in cts if ct <= t_win_end)
                if in_w > 0:
                    win_out_toks += compl * (in_w / len(cts))

        win_ok = [r for r in ok if r["t_end"] <= t_win_end]
        eval_set = win_ok if len(win_ok) >= max(4, n // 2) else ok

        tps_eval = [r["decode_tps"] for r in eval_set if r["decode_tps"] is not None]
        tps_all = [r["decode_tps"] for r in ok if r["decode_tps"] is not None]
        ttft_win = [r["ttft"] for r in eval_set if r["ttft"] is not None]
        ttft_all = [r["ttft"] for r in ok if r["ttft"] is not None]

        tot_prompt = sum(r["prompt_tokens"] for r in ok)
        tot_cached = sum(r["cached_tokens"] for r in ok)
        tot_compl = sum(r["completion_tokens"] for r in ok)
        finish_dist: dict[str, int] = {}
        for r in ok:
            fr = str(r.get("finish_reason") or "unknown")
            finish_dist[fr] = finish_dist.get(fr, 0) + 1

        win_out_tps = win_out_toks / args.duration_s if args.duration_s > 0 else 0.0
        drain_out_tps = tot_compl / wall if wall > 0 else 0.0

        row = {
            "mode": "decode_profile",
            "label": args.label,
            "target_decode_isl": args.decode_isl,
            "num_sessions": n,
            "num_chips": args.num_chips,
            "dp_size": args.dp_size,
            "batch_per_chip": round(n / max(1, args.num_chips), 4),
            "batch_per_dp_rank": round(n / max(1, args.dp_size), 4),
            "duration_window_s": args.duration_s,
            "drain_wall_s": round(wall, 3),
            "total_requests": len(results),
            "ok_requests": len(ok),
            "failed_requests": len(errs),
            "finish_reason_distribution": finish_dist,
            "mean_prompt_tokens": round(tot_prompt / len(ok), 1) if ok else 0.0,
            "mean_cached_tokens": round(tot_cached / len(ok), 1) if ok else 0.0,
            "observed_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 2) if tot_prompt > 0 else 0.0,
            "mean_completion_tokens": round(tot_compl / len(ok), 1) if ok else 0.0,
            "throughput": {
                "window_output_tok_per_s": round(win_out_tps, 2),
                "window_output_tok_per_s_per_chip": round(win_out_tps / max(1, args.num_chips), 2),
                "drain_output_tok_per_s": round(drain_out_tps, 2),
                "drain_output_tok_per_s_per_chip": round(drain_out_tps / max(1, args.num_chips), 2),
            },
            "stream_tps_in_window": {
                "mean": round(statistics.mean(tps_eval), 2) if tps_eval else 0.0,
                "p50": round(percentile(tps_eval, 50), 2),
                "p90_low": round(percentile(tps_eval, 10), 2),
                "p95_low": round(percentile(tps_eval, 5), 2),
                "p90_hi": round(percentile(tps_eval, 90), 2),
            },
            "stream_tps_all_ok": {
                "mean": round(statistics.mean(tps_all), 2) if tps_all else 0.0,
                "p50": round(percentile(tps_all, 50), 2),
                "p90_low": round(percentile(tps_all, 10), 2),
                "p95_low": round(percentile(tps_all, 5), 2),
                "p90_hi": round(percentile(tps_all, 90), 2),
            },
            "ttft_all_ok_s": {
                "mean": round(statistics.mean(ttft_all), 4) if ttft_all else 0.0,
                "p50": round(percentile(ttft_all, 50), 4),
                "p90": round(percentile(ttft_all, 90), 4),
                "p99": round(percentile(ttft_all, 99), 4),
            },
            "ttft_in_window_s": {
                "mean": round(statistics.mean(ttft_win), 4) if ttft_win else 0.0,
                "p50": round(percentile(ttft_win, 50), 4),
                "p90": round(percentile(ttft_win, 90), 4),
                "p99": round(percentile(ttft_win, 99), 4),
            },
            "metrics_deltas": parse_spec_accept_from_metrics(m0, m1),
            "profile": prof_info,
        }
        print(
            f"  [DONE B2 ISL~{args.decode_isl} n={n} ({row['batch_per_chip']:.2f}/chip)] "
            f"WinOut={win_out_tps:,.1f} tok/s ({win_out_tps/args.num_chips:.1f}/chip) | "
            f"TPS P50={row['stream_tps_all_ok']['p50']:.1f} P90_low={row['stream_tps_all_ok']['p90_low']:.1f} | "
            f"MeanOSL={row['mean_completion_tokens']:.1f} | finish={finish_dist}",
            flush=True,
        )
        summaries.append(row)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(summaries, f, indent=2)
    return summaries


async def main_async(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=3600)
    conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # Wait for health
        health_url = f"{args.base_url.rstrip('/')}/health"
        print(f"Waiting for server health at {health_url} ...", flush=True)
        while True:
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        print("  Server is healthy!", flush=True)
                        break
            except Exception:
                pass
            await asyncio.sleep(5)

        cpt = await calibrate_chars_per_token(session, args.base_url, args.model, args.engine)
        if args.mode == "prefill_uncached":
            await run_prefill_uncached(args, session, cpt)
        elif args.mode == "prefill_cached":
            await run_prefill_cached(args, session, cpt)
        elif args.mode == "decode_profile":
            await run_decode_profile(args, session, cpt)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["prefill_uncached", "prefill_cached", "decode_profile"])
    p.add_argument("--engine", default="sglang", choices=["sglang", "vllm"])
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="/weights")
    p.add_argument("--label", default="")
    p.add_argument("--num-chips", type=int, default=16)
    p.add_argument("--dp-size", type=int, default=16)
    p.add_argument("--isls", default="32768,100000,131072")
    p.add_argument("--concurrencies", default="4,8,16")
    p.add_argument("--min-requests", type=int, default=32)
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--cached-prefix-tokens", type=int, default=20480)
    p.add_argument("--uncached-suffix-tokens", type=int, default=2048)
    p.add_argument("--trunks-per-dp", type=int, default=2)
    p.add_argument("--lockstep-warmup", action="store_true", default=True)
    p.add_argument("--no-lockstep-warmup", action="store_false", dest="lockstep_warmup")
    p.add_argument("--unpin-dp-when-no-warmup", action="store_true", default=True)
    p.add_argument("--decode-isl", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--duration-s", type=float, default=180.0)
    p.add_argument("--profile-dir", default="")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
