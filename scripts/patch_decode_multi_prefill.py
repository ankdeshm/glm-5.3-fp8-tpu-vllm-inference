#!/usr/bin/env python3
"""Patch sglang-jax decode.py so multi-host Decode supports multiple Prefill instances (2P1D)."""

from __future__ import annotations

import pathlib
import py_compile
import shutil
import sys


def patch_decode_file(path: pathlib.Path) -> None:
    src = path.read_text(encoding="utf-8")
    if "_pick_prefill_peer_for_this_host(prefill_dp_rank, req)" in src:
        print(f"Already patched: {path}")
        return

    old_call = "p_info = self._pick_prefill_peer_for_this_host(prefill_dp_rank)"
    new_call = "p_info = self._pick_prefill_peer_for_this_host(prefill_dp_rank, req)"
    if old_call not in src:
        raise RuntimeError(f"Could not find call site in {path}")
    src = src.replace(old_call, new_call, 1)

    old_fn = '''    def _pick_prefill_peer_for_this_host(self: Scheduler, dp_rank: int) -> dict[str, object]:
        """Multi-host: find the P host whose jax_process_index matches ours.
        That host's local KV shard is exactly the slice this D host needs.
        Requires P/D to have the same nproc (same-TP constraint).
        """
        peers = getattr(self, "_disagg_prefill_peers", None)
        if peers is None:
            peers = self._disagg_prefill_peers = {}
        if dp_rank in peers:
            return peers[dp_rank]
        my_pidx = jax.process_index()
        my_nproc = jax.process_count()
        all_p = self.disagg_bootstrap_client.list_prefills()
        for p in all_p:
            if (
                int(p.get("jax_process_index", -1)) == my_pidx
                and int(p.get("system_dp_rank", 0)) == dp_rank
            ):
                if int(p.get("jax_process_count", 0)) != my_nproc:
                    raise RuntimeError(
                        f"P/D process_count mismatch: P={p.get('jax_process_count')} "
                        f"D={my_nproc}. Per-host shard transfer requires same nproc."
                    )
                peers[dp_rank] = p
                return p
        raise RuntimeError(
            f"no prefill host with jax_process_index={my_pidx}, dp_rank={dp_rank} "
            "registered (got "
            f"{[(p.get('host'), p.get('jax_process_index'), p.get('system_dp_rank')) for p in all_p]})"
        )'''

    new_fn = '''    def _pick_prefill_peer_for_this_host(
        self: Scheduler, dp_rank: int, req: Req | None = None
    ) -> dict[str, object]:
        """Multi-host: find the P host whose jax_process_index matches ours.
        Supports both single-Prefill (1P2D) and multi-Prefill (2P1D) by keying
        on (req.bootstrap_host, req.bootstrap_port, dp_rank) when present.
        """
        b_host = getattr(req, "bootstrap_host", None) if req is not None else None
        b_port = getattr(req, "bootstrap_port", None) if req is not None else None
        cache_key = (str(b_host or ""), int(b_port or 0), int(dp_rank))
        peers = getattr(self, "_disagg_prefill_peers", None)
        if peers is None:
            peers = self._disagg_prefill_peers = {}
        if cache_key in peers:
            return peers[cache_key]
        my_pidx = jax.process_index()
        my_nproc = jax.process_count()
        client = self.disagg_bootstrap_client
        if b_host and b_port:
            target_url = f"http://{b_host}:{int(b_port)}"
            extra_clients = getattr(self, "_disagg_extra_bootstrap_clients", None)
            if extra_clients is None:
                extra_clients = self._disagg_extra_bootstrap_clients = {}
            if target_url != getattr(client, "_base_url", None):
                if target_url not in extra_clients:
                    from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient
                    from sgl_jax.srt.disaggregation.pd_auth import resolve_secret

                    extra_clients[target_url] = BootstrapClient(
                        target_url,
                        timeout_s=self.server_args.disaggregation_bootstrap_timeout_seconds,
                        shared_secret=resolve_secret(self.server_args.disaggregation_shared_secret),
                    )
                client = extra_clients[target_url]
        all_p = client.list_prefills()
        candidates = [
            p
            for p in all_p
            if int(p.get("jax_process_index", -1)) == my_pidx
            and int(p.get("system_dp_rank", 0)) == dp_rank
        ]
        chosen = None
        if b_host and len(candidates) > 1:
            for p in candidates:
                if str(p.get("host", "")) == str(b_host):
                    chosen = p
                    break
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is not None:
            if int(chosen.get("jax_process_count", 0)) != my_nproc:
                raise RuntimeError(
                    f"P/D process_count mismatch: P={chosen.get('jax_process_count')} "
                    f"D={my_nproc}. Per-host shard transfer requires same nproc."
                )
            logger.info(
                "PD multi-host peer resolved: my_pidx=%d dp_rank=%d bootstrap=%s:%s -> peer=%s:%s (peer_pidx=%s)",
                my_pidx,
                dp_rank,
                b_host,
                b_port,
                chosen.get("host"),
                chosen.get("transfer_port"),
                chosen.get("jax_process_index"),
            )
            peers[cache_key] = chosen
            return chosen
        raise RuntimeError(
            f"no prefill host with jax_process_index={my_pidx}, dp_rank={dp_rank}, "
            f"bootstrap={b_host}:{b_port} registered (got "
            f"{[(p.get('host'), p.get('jax_process_index'), p.get('system_dp_rank')) for p in all_p]})"
        )'''

    if old_fn not in src:
        raise RuntimeError(f"Could not find _pick_prefill_peer_for_this_host body in {path}")
    src = src.replace(old_fn, new_fn, 1)
    path.write_text(src, encoding="utf-8")
    pycache = path.parent / "__pycache__"
    if pycache.exists():
        for pyc in pycache.glob("decode.*.pyc"):
            pyc.unlink(missing_ok=True)
    py_compile.compile(str(path), doraise=True)
    print(f"Successfully patched and byte-compiled: {path}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = pathlib.Path(arg)
        if p.exists():
            patch_decode_file(p)
