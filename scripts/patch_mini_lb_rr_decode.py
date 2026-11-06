#!/usr/bin/env python3
"""Patch sgl_jax.srt.disaggregation.mini_lb.select_pair to round-robin stateless Decode servers.

Keeps Prefill selection sticky by routing_key (preserving RadixCache trunk affinity across
P0 and P1 in 2P1D), while distributing Decode requests evenly round-robin across D0 and D1
in 1P2D regardless of CRC32 parity on small trunk counts.
"""
import py_compile
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/opt/sglang-jax/python/sgl_jax/srt/disaggregation/mini_lb.py"
txt = open(path, "r", encoding="utf-8").read()

if "_rr_decode_counter" in txt:
    print(f"Already patched: {path}")
    sys.exit(0)

old = """        if routing_key is not None and len(servers) > 0 and len(d_servers) > 0:
            h = zlib.crc32(str(routing_key).encode("utf-8"))
            pidx = h % len(servers)
            didx = h % len(d_servers)
            return (
                self.prefill_urls[pidx],
                self.prefill_bootstrap_ports[pidx],
                self.decode_urls[didx],
            )"""

new = """        if routing_key is not None and len(servers) > 0 and len(d_servers) > 0:
            h = zlib.crc32(str(routing_key).encode("utf-8"))
            pidx = h % len(servers)
            rr = getattr(self, "_rr_decode_counter", 0)
            self._rr_decode_counter = rr + 1
            didx = rr % len(d_servers)
            return (
                self.prefill_urls[pidx],
                self.prefill_bootstrap_ports[pidx],
                self.decode_urls[didx],
            )"""

assert old in txt, "Target block not found in mini_lb.py"
txt = txt.replace(old, new, 1)
open(path, "w", encoding="utf-8").write(txt)
py_compile.compile(path, doraise=True)
print(f"Successfully patched and compiled: {path}")
