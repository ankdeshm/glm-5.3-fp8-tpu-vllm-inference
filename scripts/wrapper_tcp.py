"""Multi-channel TCP drop-in replacement for ``JaxTransferWrapper``.

Eliminates the JAX 0.11.1 ``jax.experimental.transfer`` multi-host PJRT
``TransferServer::Connect`` collective rendezvous deadlock when multiple
multi-host Decode instances (``1P2D``) or multiple multi-host Prefill
instances (``2P1D``) connect concurrently across independent 2x2x2 slices.

Each host process binds its own multi-threaded TCP server at ``host_ip:port``
and serves process-local device shards directly over persistent TCP sockets.
"""

from __future__ import annotations

import collections
import logging
import queue
import socket
import struct
import threading
import time
from typing import Any

import jax
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_WRAPPER: JaxTransferWrapper | None = None

_HDR_REQ = struct.Struct("!I")
_HDR_RESP = struct.Struct("!BI")  # status (0=OK, 1=ERR), num_leaves
_HDR_LEAF = struct.Struct("!I")   # num_shards
_HDR_SHARD = struct.Struct("!Q")  # nbytes


def _recv_exact(sock: socket.socket, n: int) -> memoryview:
    buf = bytearray(n)
    mv = memoryview(buf)
    pos = 0
    while pos < n:
        r = sock.recv_into(mv[pos:], n - pos)
        if r == 0:
            raise ConnectionError(f"EOF while reading {n} bytes (got {pos})")
        pos += r
    return mv


def _ordered_shards_for_leaf(leaf: Any) -> list[bytes]:
    if hasattr(leaf, "addressable_shards") and leaf.addressable_shards:
        dev_to_data = {s.device: s.data for s in leaf.addressable_shards}
        sharding = getattr(leaf, "sharding", None)
        if sharding is not None and hasattr(sharding, "mesh"):
            local_set = set(dev_to_data.keys())
            ordered_devs = [d for d in sharding.mesh.devices.flat if d in local_set]
            if len(ordered_devs) == len(dev_to_data):
                arrs = [dev_to_data[d] for d in ordered_devs]
            else:
                arrs = [dev_to_data[d] for d in jax.local_devices() if d in dev_to_data]
        else:
            arrs = [dev_to_data[d] for d in jax.local_devices() if d in dev_to_data]
        if not arrs:
            arrs = [s.data for s in leaf.addressable_shards]
        return [np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes() for a in arrs]
    return [np.ascontiguousarray(np.asarray(leaf)).view(np.uint8).tobytes()]


def _reconstruct_leaf_from_shards(shard_blobs: list[memoryview], spec_leaf: Any) -> jax.Array:
    sharding = getattr(spec_leaf, "sharding", None)
    if sharding is None:
        raise ValueError("JAX transfer requires sharding on every result spec")
    np_dt = np.dtype(spec_leaf.dtype)
    if hasattr(sharding, "mesh"):
        ldev = [d for d in sharding.mesh.devices.flat if d.process_index == jax.process_index()]
        if not ldev:
            ldev = list(sharding.mesh.devices.flat)
    else:
        ldev = sorted(
            [d for d in sharding.device_set if d.process_index == jax.process_index()],
            key=lambda d: d.id,
        )
    if len(shard_blobs) != len(ldev):
        raise ValueError(
            f"Shard count mismatch: received {len(shard_blobs)} shards for {len(ldev)} local devices"
        )
    # Compute per-shard shape from spec_leaf.shape and sharding
    shard_shape = sharding.shard_shape(spec_leaf.shape)
    dev_arrays = []
    for dev, blob in zip(ldev, shard_blobs):
        host_arr = np.frombuffer(blob, dtype=np.uint8).view(np_dt).reshape(shard_shape)
        dev_arrays.append(jax.device_put(host_arr, dev))
    return jax.make_array_from_single_device_arrays(spec_leaf.shape, sharding, dev_arrays)


class _TCPClientPool:
    def __init__(self, remote_addr: str) -> None:
        host, port_s = remote_addr.rsplit(":", 1)
        self._addr = (host, int(port_s))
        self._pool: queue.LifoQueue[socket.socket] = queue.LifoQueue()

    def _new_sock(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
        except OSError:
            pass
        s.settimeout(600.0)
        s.connect(self._addr)
        return s

    def pull(self, uuid: str, spec: Any) -> Any:
        uuid_b = uuid.encode("utf-8")
        req_pkt = _HDR_REQ.pack(len(uuid_b)) + uuid_b
        last_err: Exception | None = None
        for attempt in range(3):
            sock: socket.socket | None = None
            reused = False
            try:
                try:
                    sock = self._pool.get_nowait()
                    reused = True
                except queue.Empty:
                    sock = self._new_sock()
                sock.sendall(req_pkt)
                status, num_leaves = _HDR_RESP.unpack(_recv_exact(sock, _HDR_RESP.size))
                if status != 0:
                    err_mv = _recv_exact(sock, num_leaves)
                    raise RuntimeError(
                        f"Remote pull error from {self._addr} for {uuid!r}: "
                        f"{bytes(err_mv).decode('utf-8', 'replace')}"
                    )
                leaves_shards: list[list[memoryview]] = []
                for _ in range(num_leaves):
                    (num_shards,) = _HDR_LEAF.unpack(_recv_exact(sock, _HDR_LEAF.size))
                    s_list: list[memoryview] = []
                    for _ in range(num_shards):
                        (nbytes,) = _HDR_SHARD.unpack(_recv_exact(sock, _HDR_SHARD.size))
                        s_list.append(_recv_exact(sock, nbytes))
                    leaves_shards.append(s_list)
                self._pool.put(sock)
                sock = None

                spec_leaves, treedef = jax.tree.flatten(spec)
                if len(spec_leaves) != len(leaves_shards):
                    raise ValueError(
                        f"Leaf count mismatch for {uuid!r}: got {len(leaves_shards)}, expected {len(spec_leaves)}"
                    )
                out_leaves = [
                    _reconstruct_leaf_from_shards(blobs, sp_leaf)
                    for blobs, sp_leaf in zip(leaves_shards, spec_leaves)
                ]
                return jax.tree.unflatten(treedef, out_leaves)
            except Exception as e:
                last_err = e
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if not reused and attempt >= 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
        assert last_err is not None
        raise last_err


class JaxTransferWrapper:
    """One lazily started multi-threaded TCP transfer server per process."""

    # Prevent sgl_jax.srt.disaggregation.factory._create_jax_backend from
    # monkeypatching start() back to C++ jax.experimental.transfer.
    _pd_multisocket_patched = True
    _GRACE_TTL_S = 15.0
    _GRACE_MAX_ENTRIES = 32

    def __init__(self, host_ip: str, port: int, channel_number: int = 1) -> None:
        self._host_ip = host_ip
        self._port = port
        self._channel_number = channel_number
        self._init_lock = threading.Lock()
        self._server: socket.socket | None = None
        self._pending_lock = threading.Lock()
        self._cond = threading.Condition(self._pending_lock)
        # uuid -> list of leaves, where each leaf is list[bytes] (one per local shard)
        self._pending: dict[str, list[list[bytes]]] = {}
        self._released: collections.OrderedDict[str, tuple[float, list[list[bytes]]]] = (
            collections.OrderedDict()
        )
        self._links_lock = threading.Lock()
        self._links: dict[str, _TCPClientPool] = {}

    @property
    def host_ip(self) -> str:
        return self._host_ip

    @property
    def port(self) -> int:
        return self._port

    @property
    def channel_number(self) -> int:
        return self._channel_number

    @property
    def is_started(self) -> bool:
        return self._server is not None

    @property
    def server(self) -> Any:
        return self._server

    def _prune_released_locked(self, now: float) -> None:
        cutoff = now - self._GRACE_TTL_S
        while self._released:
            oldest_k, (ts, _) = next(iter(self._released.items()))
            if ts < cutoff or len(self._released) > self._GRACE_MAX_ENTRIES:
                self._released.pop(oldest_k, None)
            else:
                break

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            except OSError:
                pass
            while True:
                hdr_buf = bytearray(_HDR_REQ.size)
                mv = memoryview(hdr_buf)
                pos = 0
                while pos < _HDR_REQ.size:
                    r = conn.recv_into(mv[pos:], _HDR_REQ.size - pos)
                    if r == 0:
                        if pos == 0:
                            return
                        raise ConnectionError("Truncated request header")
                    pos += r
                (ulen,) = _HDR_REQ.unpack(mv)
                uuid = bytes(_recv_exact(conn, ulen)).decode("utf-8")

                deadline = time.monotonic() + 600.0
                entry: list[list[bytes]] | None = None
                with self._cond:
                    while True:
                        entry = self._pending.get(uuid)
                        if entry is None:
                            rel = self._released.get(uuid)
                            if rel is not None:
                                entry = rel[1]
                        if entry is not None:
                            break
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            break
                        self._cond.wait(timeout=min(1.0, rem))

                if entry is None:
                    msg = f"Timeout waiting for uuid={uuid}".encode("utf-8")
                    conn.sendall(_HDR_RESP.pack(1, len(msg)) + msg)
                    continue

                out_chunks: list[bytes] = [_HDR_RESP.pack(0, len(entry))]
                for leaf_shards in entry:
                    out_chunks.append(_HDR_LEAF.pack(len(leaf_shards)))
                    for sb in leaf_shards:
                        out_chunks.append(_HDR_SHARD.pack(len(sb)))
                        out_chunks.append(sb)
                for chunk in out_chunks:
                    conn.sendall(chunk)
        except (ConnectionError, BrokenPipeError, OSError):
            pass
        except Exception:
            logger.exception("JaxTransferWrapper TCP client handler error")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _accept_loop(self, srv: socket.socket) -> None:
        while True:
            try:
                conn, _ = srv.accept()
                t = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                t.start()
            except OSError:
                return
            except Exception:
                logger.exception("JaxTransferWrapper accept error")

    def start(self) -> Any:
        if self._server is not None:
            return self._server
        with self._init_lock:
            if self._server is not None:
                return self._server
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", self._port))
            srv.listen(1024)
            self._server = srv
            t = threading.Thread(
                target=self._accept_loop,
                args=(srv,),
                name=f"JaxTransferTCP-accept-{self._port}",
                daemon=True,
            )
            t.start()
            logger.info(
                "JaxTransferWrapper (TCP multi-peer) started at %s:%d (channels=%d, jax=%s)",
                self._host_ip,
                self._port,
                self._channel_number,
                jax.__version__,
            )
        return self._server

    def register_pull(self, uuid: str, data: Any) -> None:
        if self._server is None:
            raise RuntimeError("JaxTransferWrapper.start() must be called before register_pull()")
        leaves = jax.tree.leaves(data)
        for leaf in leaves:
            sharding = getattr(leaf, "sharding", None)
            if sharding is not None and hasattr(leaf, "is_fully_addressable") and not leaf.is_fully_addressable:
                raise ValueError(
                    f"register_pull received a non-fully-addressable array spanning "
                    f"{len(sharding.device_set)} devices"
                )
        serialized_leaves = [_ordered_shards_for_leaf(leaf) for leaf in leaves]
        total_bytes = sum(len(sb) for lf in serialized_leaves for sb in lf)
        now = time.monotonic()
        with self._cond:
            if uuid in self._pending:
                raise RuntimeError(
                    f"uuid {uuid!r} is already registered; release it before re-registering"
                )
            self._released.pop(uuid, None)
            self._prune_released_locked(now)
            self._pending[uuid] = serialized_leaves
            self._cond.notify_all()
        try:
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            PD_TRANSFER_BYTES_TOTAL.labels(direction="net", role="prefill").inc(int(total_bytes))
        except Exception:  # noqa: BLE001
            pass

    def pull(self, uuid: str, spec: Any, remote_addr: str | None = None) -> Any:
        for leaf in jax.tree.leaves(spec):
            if getattr(leaf, "sharding", None) is None:
                raise ValueError("JAX transfer requires sharding on every result spec")
        if self._server is None:
            raise RuntimeError("JaxTransferWrapper.start() must be called before pull()")
        if remote_addr is None:
            raise ValueError("remote_addr is required")
        return self._connect(remote_addr).pull(uuid, spec)

    def release(self, uuid: str) -> None:
        now = time.monotonic()
        with self._cond:
            val = self._pending.pop(uuid, None)
            if val is not None:
                self._released[uuid] = (now, val)
            self._prune_released_locked(now)

    def _connect(self, remote_addr: str) -> _TCPClientPool:
        with self._links_lock:
            link = self._links.get(remote_addr)
            if link is None:
                link = _TCPClientPool(remote_addr)
                self._links[remote_addr] = link
            return link


def get_or_create_wrapper(
    host_ip: str,
    port: int,
    channel_number: int = 1,
) -> JaxTransferWrapper:
    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        if _GLOBAL_WRAPPER is None:
            _GLOBAL_WRAPPER = JaxTransferWrapper(host_ip, port, channel_number)
            return _GLOBAL_WRAPPER
        existing = _GLOBAL_WRAPPER
        if (existing.host_ip, existing.port) != (host_ip, port):
            raise RuntimeError(
                f"JaxTransferWrapper cannot rebind from "
                f"{existing.host_ip}:{existing.port} to {host_ip}:{port}"
            )
        if existing.channel_number != channel_number:
            raise RuntimeError(
                f"JaxTransferWrapper already uses channel_number={existing.channel_number}"
            )
        return existing


def _reset_singleton_for_test() -> None:
    global _GLOBAL_WRAPPER
    with _GLOBAL_LOCK:
        _GLOBAL_WRAPPER = None
