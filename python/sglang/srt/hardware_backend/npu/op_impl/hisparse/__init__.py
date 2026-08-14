"""
HiSparse NPU graph-compatible kernels.

Provides functions backed by the compiled AscendC extension ``hisparse_lru.so``:
  - lru_update_npu: per-request top-k device-buffer lookup / LRU update / is_miss flags.
  - sieve_update_npu: same contract with SIEVE eviction (visited byte + hand
    pointer instead of an int32 recency row + per-miss ReduceMin scan).
  - scatter_from_host_npu: scatter missing KV rows from pinned host cache to device buffer.

``block_dim=0`` (default) auto-selects the number of AIV cores. Override with the
HISPARSE_BLOCK_DIM environment variable when tuning.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    from . import hisparse_lru as _hisparse_lru
    _HAS_KERNEL = True
except Exception:
    _hisparse_lru = None
    _HAS_KERNEL = False

_BLOCK_DIM_AUTO = None


def _resolve_block_dim(block_dim: int) -> int:
    """Resolve block_dim <= 0 to the auto-detected AIV core count (cached)."""
    global _BLOCK_DIM_AUTO
    if block_dim > 0:
        return block_dim
    if _BLOCK_DIM_AUTO is None:
        n = 0
        override = os.environ.get("HISPARSE_BLOCK_DIM", "")
        if override:
            n = int(override)
        else:
            try:
                props = torch.npu.get_device_properties(torch.npu.current_device())
                n = int(getattr(props, "multi_processor_count", 0) or 0)
            except Exception:
                n = 0
        if n <= 0:
            n = 48
        _BLOCK_DIM_AUTO = n
        logger.info("hisparse: auto block_dim = %d", n)
    return _BLOCK_DIM_AUTO


def _require_kernel() -> None:
    if not _HAS_KERNEL:
        raise RuntimeError(
            "hisparse_lru native extension is not available; build it on an NPU "
            "host with `python setup.py build_ext --inplace` in "
            "python/sglang/srt/hardware_backend/npu/op_impl/hisparse."
        )


def lru_update_npu(
    layer_id: int,
    topk_indices: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    device_buffer_recency: torch.Tensor,
    top_k_device_slots: torch.Tensor,
    is_miss: torch.Tensor,
    num_real_reqs: torch.Tensor,
    global_recency_counter: torch.Tensor,
    top_k: int,
    device_buffer_size: int,
    padded_buffer_size: int,
    max_num_reqs: int,
    block_dim: int = 0,
) -> None:
    """
    In-place graph-compatible LRU update and miss detection.

    All output tensors are device-side tensors that the caller pre-allocates and
    reuses across graph captures. ``is_miss`` is updated in-place by the kernel.
    ``layer_id`` is kept for call-site context; the per-layer buffer tensors are
    already sliced by the caller.

    ``seq_lens`` must be int32 (the kernel reads it as int32, matching the
    static buffers used under graph capture/replay).
    """
    if padded_buffer_size % 16 != 0:
        raise ValueError(
            "lru_update_npu: padded_buffer_size must be a multiple of 16 "
            "(64-byte rows required by the multi-block kernel)"
        )
    _require_kernel()
    _hisparse_lru.lru_update(
        topk_indices,
        req_pool_indices,
        seq_lens,
        prefill_len,
        device_buffer_tokens,
        device_buffer_recency,
        top_k_device_slots,
        is_miss,
        num_real_reqs,
        global_recency_counter,
        top_k,
        device_buffer_size,
        padded_buffer_size,
        max_num_reqs,
        _resolve_block_dim(block_dim),
    )


def sieve_update_npu(
    layer_id: int,
    topk_indices: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    device_buffer_visited: torch.Tensor,
    device_buffer_ht: torch.Tensor,
    sieve_hand: torch.Tensor,
    top_k_device_slots: torch.Tensor,
    is_miss: torch.Tensor,
    num_real_reqs: torch.Tensor,
    top_k: int,
    device_buffer_size: int,
    padded_buffer_size: int,
    max_num_reqs: int,
    block_dim: int = 0,
) -> None:
    """
    In-place graph-compatible SIEVE update and miss detection.

    Same contract as ``lru_update_npu``, but eviction state is a per-slot
    visited byte (``device_buffer_visited``, uint8, row stride a multiple of
    64 and >= padded_buffer_size) plus a persistent hand pointer per request
    (``sieve_hand``, int32, shape [pool_size, 16] so each request owns a full
    64-byte cache line).  Hits set the visited byte; misses sweep the hand to
    the first unvisited slot (amortized O(1), no full-buffer ReduceMin scan).

    Lookups use a persistent GM token-to-slot table
    (``device_buffer_ht``, int16, one row per pool slot with kT2sCap=65536
    entries).  When prefill_len < 65536 the table is a direct-index bitmap
    (t2s[token] = slot+1, single read, no hashing); otherwise the first
    ht_size entries are used as an open-addressing hash table.  Built by
    ``sieve_ht_init_npu`` and maintained in place.

    Output contract: every row of ``top_k_device_slots`` / ``is_miss`` in
    [0, max_num_reqs) is fully rewritten on each call.  Rows beyond
    ``num_real_reqs`` (graph-replay padding) get slot -1 / is_miss 0
    without touching per-request state, and invalid top-k positions get
    slot -1.  The sparse attention operator treats -1 as an invalid entry
    and skips it, so the caller needs no fill_/clamp_ passes.
    """
    if padded_buffer_size % 16 != 0:
        raise ValueError(
            "sieve_update_npu: padded_buffer_size must be a multiple of 16 "
            "(64-byte rows required by the multi-block kernel)"
        )
    if device_buffer_visited.dtype != torch.uint8:
        raise ValueError("sieve_update_npu: device_buffer_visited must be uint8")
    if device_buffer_visited.shape[1] < padded_buffer_size or (
        device_buffer_visited.shape[1] % 64 != 0
    ):
        raise ValueError(
            "sieve_update_npu: device_buffer_visited row stride must be >= "
            "padded_buffer_size and a multiple of 64"
        )
    if device_buffer_ht.dtype != torch.int16:
        raise ValueError("sieve_update_npu: device_buffer_ht must be int16")
    if device_buffer_ht.size(1) < 1024 or (
        device_buffer_ht.size(1) & (device_buffer_ht.size(1) - 1) != 0
    ):
        raise ValueError(
            "sieve_update_npu: device_buffer_ht row size must be a power of two >= 1024"
        )
    if device_buffer_ht.shape[0] != device_buffer_tokens.shape[0]:
        raise ValueError(
            "sieve_update_npu: device_buffer_ht must have one row per pool slot"
        )
    if sieve_hand.dtype != torch.int32 or sieve_hand.shape[1] != 16:
        raise ValueError(
            "sieve_update_npu: sieve_hand must be int32 with shape [pool_size, 16]"
        )
    _require_kernel()
    _hisparse_lru.sieve_update(
        topk_indices,
        req_pool_indices,
        seq_lens,
        prefill_len,
        device_buffer_tokens,
        device_buffer_visited,
        device_buffer_ht,
        sieve_hand,
        top_k_device_slots,
        is_miss,
        num_real_reqs,
        top_k,
        device_buffer_size,
        padded_buffer_size,
        max_num_reqs,
        _resolve_block_dim(block_dim),
    )


def sieve_ht_init_npu(
    device_buffer_tokens: torch.Tensor,
    device_buffer_ht: torch.Tensor,
    prefill_lens: torch.Tensor,
    req_start: int,
    num_reqs: int,
    device_buffer_size: int,
    padded_buffer_size: int,
    ht_size: int,
) -> None:
    """
    (Re)build the persistent SIEVE token-to-slot table for a request range.

    Clears ``device_buffer_ht[:, req_start:req_start + num_reqs, :]`` and
    re-inserts every in-use buffer slot from ``device_buffer_tokens``.
    ``prefill_lens`` selects the format per request: when
    ``prefill_lens[req_idx] < 65536`` the row is initialized as a direct-index
    bitmap (t2s[token] = slot+1); otherwise as a hash table in the first
    ``ht_size`` entries.

    Call once after allocating the tensors, and again whenever a request's
    buffer row is repurposed for a different sequence.  ``device_buffer_tokens``
    is [layer_num, pool_size, padded] int32; ``device_buffer_ht`` is
    [layer_num, pool_size, 65536] int16; ``prefill_lens`` is [pool_size]
    int32.
    """
    _require_kernel()
    _hisparse_lru.sieve_ht_init(
        device_buffer_tokens,
        device_buffer_ht,
        prefill_lens,
        req_start,
        num_reqs,
        device_buffer_size,
        padded_buffer_size,
        ht_size,
    )


def scatter_from_host_npu(
    host_kv_cache_ptr: int,
    topk_indices: torch.Tensor,
    top_k_device_slots: torch.Tensor,
    is_miss: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_host_pool: torch.Tensor,
    req_to_device_buffer: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    layer_id: int,
    host_entries: int,
    k_row_bytes: int,
    v_row_bytes: int,
    max_context_len: int,
    device_buffer_row_stride: int,
    padded_buffer_size: int,
    max_num_reqs: int,
    top_k: int,
    block_dim: int = 0,
) -> None:
    """
    Scatter missing KV rows from the pinned host cache into the device buffer.

    ``host_kv_cache_ptr`` is the raw base pointer of the interleaved pinned host
    cache (e.g. ``TieredHostMemoryPool.get_host_kv_data_ptr(0)``) where each
    token row is laid out as ``[K_data | V_data]`` contiguously; the kernel
    applies the per-layer offset itself. ``max_context_len`` is the row stride
    of ``req_to_host_pool`` and ``device_buffer_row_stride`` the row stride of
    ``req_to_device_buffer``. The kernel reads directly from host memory using
    these addresses.
    """
    if k_row_bytes <= 0 or v_row_bytes <= 0:
        raise ValueError("scatter_from_host_npu: k_row_bytes and v_row_bytes must be > 0")
    if k_row_bytes % 32 != 0 or v_row_bytes % 32 != 0:
        raise ValueError(
            "scatter_from_host_npu: k_row_bytes and v_row_bytes must be multiples of 32 bytes"
        )
    _require_kernel()
    _hisparse_lru.scatter_from_host(
        host_kv_cache_ptr,
        topk_indices,
        top_k_device_slots,
        is_miss,
        req_pool_indices,
        req_to_host_pool,
        req_to_device_buffer,
        device_k_buffer,
        device_v_buffer,
        layer_id,
        host_entries,
        k_row_bytes,
        v_row_bytes,
        max_context_len,
        device_buffer_row_stride,
        padded_buffer_size,
        max_num_reqs,
        top_k,
        _resolve_block_dim(block_dim),
    )


def scatter_from_host_group_npu(
    host_kv_cache_ptr: int,
    topk_indices: torch.Tensor,
    top_k_device_slots: torch.Tensor,
    is_miss: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_host_pool: torch.Tensor,
    req_to_device_buffer: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    anchor_layer_id: int,
    group_size: int,
    host_entries: int,
    k_row_bytes: int,
    v_row_bytes: int,
    max_context_len: int,
    device_buffer_row_stride: int,
    padded_buffer_size: int,
    max_num_reqs: int,
    top_k: int,
    block_dim: int = 0,
) -> None:
    """
    Group scatter for IndexShare models: scatter missing KV rows into the
    anchor layer's device buffer AND its trailing shared-index (skip) layers'
    buffers in one launch.

    ``device_k_buffer`` / ``device_v_buffer`` must be the FULL layer-stacked
    pools (shape ``[layer_num, ...]``); the kernel derives each group layer's
    base from ``anchor_layer_id + g``.  ``group_size = 1 + number of skip
    layers`` covered by this anchor.  Skip layers then reuse the anchor's
    ``top_k_device_slots`` directly without launching their own kernels.
    """
    if k_row_bytes <= 0 or v_row_bytes <= 0:
        raise ValueError(
            "scatter_from_host_group_npu: k_row_bytes and v_row_bytes must be > 0"
        )
    if k_row_bytes % 32 != 0 or v_row_bytes % 32 != 0:
        raise ValueError(
            "scatter_from_host_group_npu: k_row_bytes and v_row_bytes must be "
            "multiples of 32 bytes"
        )
    if device_k_buffer.dim() < 2 or device_k_buffer.size(0) != device_v_buffer.size(0):
        raise ValueError(
            "scatter_from_host_group_npu: device buffers must be the full "
            "layer-stacked pools with matching layer counts"
        )
    layer_num = device_k_buffer.size(0)
    if not (0 <= anchor_layer_id < layer_num):
        raise ValueError("scatter_from_host_group_npu: anchor_layer_id out of range")
    if not (1 <= group_size <= layer_num - anchor_layer_id):
        raise ValueError("scatter_from_host_group_npu: group exceeds the layer range")
    _require_kernel()
    _hisparse_lru.scatter_from_host_group(
        host_kv_cache_ptr,
        topk_indices,
        top_k_device_slots,
        is_miss,
        req_pool_indices,
        req_to_host_pool,
        req_to_device_buffer,
        device_k_buffer,
        device_v_buffer,
        anchor_layer_id,
        group_size,
        host_entries,
        k_row_bytes,
        v_row_bytes,
        max_context_len,
        device_buffer_row_stride,
        padded_buffer_size,
        max_num_reqs,
        top_k,
        _resolve_block_dim(block_dim),
    )


__all__ = [
    "lru_update_npu",
    "sieve_update_npu",
    "sieve_ht_init_npu",
    "scatter_from_host_npu",
    "scatter_from_host_group_npu",
]
