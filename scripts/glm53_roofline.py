#!/usr/bin/env python3
"""First-order roofline calculator for GLM 5.2 / 5.3 serving on Cloud TPU v7x, v6e, and v5p.

What it does
------------
For every (chip type x chip count x quantization) it prints:
  1. Weights: total size, whether they fit, and how much HBM is left for KV cache.
  2. Prefill (drives TTFT): ideal compute time per 1,000 uncached prompt tokens.
  3. Decode (drives tokens/s per user): ideal HBM-bound time per decode step,
     at 1 and 64 concurrent users, 8K-token context, no MTP.

"Ideal" means 100% of peak FLOPs or HBM bandwidth, weights spread evenly over
all chips, and no communication or software overhead. Real systems include
collective communication, kernel launch, and sharding overheads; use this script
to understand the first-order memory and compute bounds across slice sizes.

Model architecture constants come from zai-org/GLM-5.3 config.json.
Chip specifications come from public Google Cloud TPU documentation for v7x
(Ironwood), v6e (Trillium), and v5p.

Usage: python3 scripts/glm53_roofline.py
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# GLM 5.2 architecture (config.json)
# ----------------------------------------------------------------------------
H = 6144                   # hidden size
LAYERS = 78                # 3 dense + 75 MoE
DENSE_LAYERS = 3
MOE_LAYERS = LAYERS - DENSE_LAYERS
N_EXPERTS = 256            # routed experts per MoE layer
TOP_K = 8                  # routed experts used per token
MOE_INTER = 2048
DENSE_INTER = 12288
VOCAB = 154880
HEADS = 64
Q_LORA, KV_LORA = 2048, 512
QK_NOPE, QK_ROPE, V_HEAD = 192, 64, 256
IDX_HEADS, IDX_DIM = 32, 128
IDX_TOPK = 2048            # DSA: each query attends to the top 2,048 tokens
IDX_FULL_LAYERS = 21       # indexer runs on 21 of 78 layers (others reuse it)

attn_params = (H * Q_LORA + Q_LORA * HEADS * (QK_NOPE + QK_ROPE)
               + H * (KV_LORA + QK_ROPE) + KV_LORA * HEADS * (QK_NOPE + V_HEAD)
               + HEADS * V_HEAD * H
               + Q_LORA * IDX_HEADS * IDX_DIM + H * IDX_DIM + H * IDX_HEADS)
expert_params = 3 * H * MOE_INTER          # one routed or shared expert
dense_mlp_params = 3 * H * DENSE_INTER
router_params = N_EXPERTS * H
embed_params = VOCAB * H                    # embedding; LM head is the same size

P_ATTN = LAYERS * attn_params
P_ROUTED = MOE_LAYERS * N_EXPERTS * expert_params
P_SHARED = MOE_LAYERS * expert_params
P_DENSE = DENSE_LAYERS * dense_mlp_params
P_ROUTER = MOE_LAYERS * router_params
P_EMB = 2 * embed_params                    # embedding + LM head
P_TOTAL = P_ATTN + P_ROUTED + P_SHARED + P_DENSE + P_ROUTER + P_EMB

# Parameters each token actually uses (matmul FLOPs = 2 x this).
P_ACTIVE = (P_ATTN + MOE_LAYERS * TOP_K * expert_params + P_SHARED + P_DENSE
            + P_ROUTER + embed_params)

# KV cache per token, FP8: MLA latent (512 + 64) on every layer + indexer key
# (128) on the 21 indexer layers.
KV_BYTES_PER_TOKEN = LAYERS * (KV_LORA + QK_ROPE) + IDX_FULL_LAYERS * IDX_DIM

# Attention math per new token (absorbed MLA over the 2,048 selected tokens,
# plus the indexer scanning the whole context).
def attn_flops_per_token(context: int) -> float:
  selected = min(context, IDX_TOPK)
  core = LAYERS * HEADS * (2 * KV_LORA + QK_ROPE) * 2 * selected
  indexer = IDX_FULL_LAYERS * IDX_HEADS * IDX_DIM * 2 * context
  return core + indexer

# ----------------------------------------------------------------------------
# Public Cloud TPU specifications (per chip).
# Note: v5p executes W8A16 / FP8-stored weights on BF16 MXUs (0.459 PF/chip).
# ----------------------------------------------------------------------------
CHIPS = {
    #        BF16 PF  FP8 PF   FP4 PF   HBM GB  HBM TB/s
    "v7x":  (2.307,   4.614,   4.614,   192,    7.37),
    "v6e":  (0.918,   1.836,   1.836,   32,     1.60),
    "v5p":  (0.459,   0.459,   0.459,   95,     2.765),
}
SLICE_SIZES = (8, 16)

# ----------------------------------------------------------------------------
# Quantization schemes: bytes per parameter for (experts, attention+dense+router)
# and the precision the big matmuls run at. MXFP4 = 4-bit + one 8-bit scale per
# 32 values = 0.53125 bytes. Embedding / LM head stay BF16 in every scheme.
# ----------------------------------------------------------------------------
MXFP4 = 0.5 + 1 / 32
SCHEMES = {
    # name:                (expert B, other B, matmul precision)
    "BF16":                (2.0,   2.0,   "bf16"),
    "FP8 (official)":      (1.0,   1.0,   "fp8"),
    "W4A8 (FP4 experts)":  (MXFP4, 1.0,   "fp8"),
    "FP4 (W4A4)":          (MXFP4, MXFP4, "fp4"),
}

HBM_USABLE = 0.90          # vLLM-style memory utilization
CONTEXT = 8192             # tokens of context per user for KV / attention


def weight_bytes(expert_b: float, other_b: float) -> float:
  return ((P_ROUTED + P_SHARED) * expert_b
          + (P_ATTN + P_DENSE + P_ROUTER) * other_b + P_EMB * 2.0)


def decode_step_bytes(users: int, expert_b: float, other_b: float) -> float:
  """Bytes read from HBM in one decode step (1 new token per user)."""
  touched = 1 - (1 - TOP_K / N_EXPERTS) ** users     # share of experts touched
  experts = MOE_LAYERS * N_EXPERTS * touched * expert_params * expert_b
  shared = P_SHARED * expert_b
  rest = (P_ATTN + P_DENSE + P_ROUTER) * other_b + embed_params * 2.0
  kv = users * (LAYERS * min(CONTEXT, IDX_TOPK) * (KV_LORA + QK_ROPE)
                + IDX_FULL_LAYERS * CONTEXT * IDX_DIM)
  return experts + shared + rest + kv


def main() -> None:
  gb = 1e9
  print(f"GLM-5.3: {P_TOTAL/1e9:.1f}B params total, {P_ACTIVE/1e9:.1f}B active "
        f"per token, KV {KV_BYTES_PER_TOKEN/1e3:.1f} KB/token (FP8)\n")

  # 1) Weights and fit --------------------------------------------------------
  print("1) WEIGHTS AND FIT (HBM left for KV cache; 8K-token users that fit)")
  print(f"{'scheme':22} {'weights':>8} | " + " | ".join(
      f"{c}-{n:<2}".rjust(16) for c in CHIPS for n in SLICE_SIZES))
  for name, (eb, ob, _) in SCHEMES.items():
    w = weight_bytes(eb, ob)
    cells = []
    for chip, (_, _, _, hbm, _) in CHIPS.items():
      for n in SLICE_SIZES:
        free = n * hbm * gb * HBM_USABLE - w
        if free <= 0:
          cells.append("no fit".rjust(16))
        else:
          users = free / (KV_BYTES_PER_TOKEN * CONTEXT)
          cells.append(f"{free/gb:5.0f} GB/{users:5.0f}u".rjust(16))
    print(f"{name:22} {w/gb:6.0f}GB | " + " | ".join(cells))

  # 2) Prefill ----------------------------------------------------------------
  print("\n2) PREFILL (TTFT compute): ideal ms per 1,000 uncached tokens, 16 chips, "
        "8K context")
  print(f"{'scheme':22} {'attention math':>15} | " + " | ".join(
      c.rjust(8) for c in CHIPS))
  attn = attn_flops_per_token(CONTEXT)
  for name, (_, _, prec) in SCHEMES.items():
    for attn_prec in ("bf16", "fp8"):
      if name == "BF16" and attn_prec == "fp8":
        continue
      cells = []
      for chip, (bf16, fp8, fp4, _, _) in CHIPS.items():
        rate = {"bf16": bf16, "fp8": fp8, "fp4": fp4}
        t = 1000 * (2 * P_ACTIVE / (16 * rate[prec] * 1e15)
                    + attn / (16 * rate[attn_prec] * 1e15))
        cells.append(f"{t*1e3:6.2f}".rjust(8))
      print(f"{name:22} {attn_prec.upper():>15} | " + " | ".join(cells))

  # 3) Decode -----------------------------------------------------------------
  print("\n3) DECODE: ideal HBM-bound ms per step -> tokens/s per user "
        "(no MTP, 8K context)")
  for users in (1, 64):
    print(f"   {users} concurrent user(s)")
    print(f"{'scheme':22} | " + " | ".join(
        f"{c}-{n}".rjust(18) for c in CHIPS for n in SLICE_SIZES))
    for name, (eb, ob, _) in SCHEMES.items():
      w = weight_bytes(eb, ob)
      step_bytes = decode_step_bytes(users, eb, ob)
      cells = []
      for chip, (_, _, _, hbm, bw) in CHIPS.items():
        for n in SLICE_SIZES:
          if n * hbm * gb * HBM_USABLE - w <= 0:
            cells.append("no fit".rjust(18))
            continue
          t = step_bytes / (n * bw * 1e12)
          cells.append(f"{t*1e3:5.2f}ms {1/t:6.0f}t/s".rjust(18))
      print(f"{name:22} | " + " | ".join(cells))


if __name__ == "__main__":
  main()
