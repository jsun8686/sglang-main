"""
Performance benchmark for the HiSparse NPU kernels (lru_update / scatter_from_host).

Default scenario (production-like):
    num_reqs            = 128
    top_k               = 2048
    device_buffer_size  = 4096
    padded_buffer_size  = 4112  (64B rows, multiple of 16)
    max_decode_len      = 2048
    miss_ratio          = 0.1   (~26k miss rows per call)
    K row               = 512 * 2B = 1024B   (kv_lora_rank=512, fp16/bf16)
    V row               =  64 * 2B =  128B   (qk_rope_head_dim=64)
    seq_len = prefill   = 8192  (all prefill tokens; topk only hit/miss)

State is reset before every timed iteration so the hit-scan cost is stable
(buffer slot s holds token s, hit scan averages device_buffer_size/2 reads).

Layout A/B section (M2): measures the entry-major host pool layout against
the legacy layer-major layout at the kernel level, with a manually
constructed miss state (the scatter kernels read but never mutate
is_miss / top_k_device_slots, so identical state drives both layouts, timed
interleaved to cancel drift):
    1. group scatter   : entry vs layer, group_size sweep   (M2 main effect)
    2. legacy scatter  : single-layer entry vs layer        (control; both
                         layouts move one 1152B row, must be ~equal)
    3. backup          : backup_to_host kernel vs index_copy_ fallback
Each layout is additionally cross-verified against expected markers so the
two address formulas are regression-checked, not just timed.

Run on an NPU host:
    python perf_hisparse_lru_npu.py                     # legacy + A/B (8 layers)
    python perf_hisparse_lru_npu.py --skip-legacy       # A/B only
    python perf_hisparse_lru_npu.py --num-layers 78     # production scale
"""

import argparse
import os
import time

import torch

from sglang.srt.hardware_backend.npu.op_impl import hisparse as hisparse_pkg
from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
    _HAS_KERNEL,
    backup_to_host_npu,
    lru_update_npu,
    scatter_from_host_group_npu,
    scatter_from_host_npu,
    sieve_ht_init_npu,
    sieve_update_npu,
)
from sglang.srt.hardware_backend.npu.op_impl.hisparse.test_hisparse_lru_npu import (
    _T2S_CAP,
    _acl_free_host,
    _acl_malloc_host,
    _sieve_ht_size,
    _visited_stride,
)

# ----------------------------- benchmark config -----------------------------
PADDED_BUFFER_SIZE = 4112
MAX_DECODE_LEN = 2048
MAX_CONTEXT_LEN = 8192  # seq_len == prefill_len for every request

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
DTYPE = torch.float16
K_ROW_BYTES = KV_LORA_RANK * DTYPE.itemsize   # 1024
V_ROW_BYTES = QK_ROPE_HEAD_DIM * DTYPE.itemsize  # 128

# (num_reqs, top_k, device_buffer_size, miss_ratio)
SCENARIOS = (
    (1, 2048, 4096, 0.0),
    (1, 2048, 4096, 0.1),
    (4, 2048, 4096, 0.1),
    (8, 2048, 4096, 0.1),
    (16, 2048, 4096, 0.1),
    (128, 2048, 4096, 0.1),
    (256, 2048, 4096, 0.1),
)

BLOCK_DIMS = (0, 8, 16, 32, 48, 64)  # 0 = auto (AIV core count)
ALGOS = ("lru", "sieve")  # eviction algorithm for the update kernel
WARMUP = 1
ITERS = 5


def _make_state(device, num_reqs, top_k, device_buffer_size):
    device_rows = num_reqs * (PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
    topk_indices = torch.zeros((num_reqs, top_k), dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(num_reqs, dtype=torch.int64, device=device)
    seq_lens = torch.full((num_reqs,), MAX_CONTEXT_LEN, dtype=torch.int32, device=device)
    prefill_len = torch.full((num_reqs,), MAX_CONTEXT_LEN, dtype=torch.int32, device=device)
    device_buffer_tokens = torch.full(
        (num_reqs, PADDED_BUFFER_SIZE), -1, dtype=torch.int32, device=device
    )
    device_buffer_recency = torch.zeros(
        (num_reqs, PADDED_BUFFER_SIZE), dtype=torch.int32, device=device
    )
    device_buffer_visited = torch.zeros(
        (num_reqs, _visited_stride(PADDED_BUFFER_SIZE)), dtype=torch.uint8, device=device
    )
    device_buffer_ht = torch.zeros(
        (1, num_reqs, _T2S_CAP),
        dtype=torch.int16,
        device=device,
    )
    sieve_hand = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)
    top_k_device_slots = torch.full((num_reqs, top_k), -1, dtype=torch.int32, device=device)
    is_miss = torch.zeros((num_reqs, top_k), dtype=torch.int8, device=device)
    num_real_reqs = torch.tensor([num_reqs], dtype=torch.int32, device=device)
    global_recency_counter = torch.ones(1, dtype=torch.int32, device=device)

    req_to_host_pool = (
        torch.arange(MAX_CONTEXT_LEN, dtype=torch.int64, device=device)
        .unsqueeze(0)
        .repeat(num_reqs, 1)
        .contiguous()
    )
    req_to_device_buffer = (
        torch.arange(device_rows, dtype=torch.int64, device=device)
        .view(num_reqs, PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
        .contiguous()
    )
    device_k_buffer = torch.zeros((device_rows, KV_LORA_RANK), dtype=DTYPE, device=device)
    device_v_buffer = torch.zeros((device_rows, QK_ROPE_HEAD_DIM), dtype=DTYPE, device=device)

    return dict(
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_recency=device_buffer_recency,
        device_buffer_visited=device_buffer_visited,
        device_buffer_ht=device_buffer_ht,
        sieve_hand=sieve_hand,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        global_recency_counter=global_recency_counter,
        req_to_host_pool=req_to_host_pool,
        req_to_device_buffer=req_to_device_buffer,
        device_k_buffer=device_k_buffer,
        device_v_buffer=device_v_buffer,
    )


def _reset_state(state, init_tokens_row, device_buffer_size, num_reqs, top_k):
    """Restore the initial buffer contents so every timed call sees the same
    hit-scan cost: slot s holds token s for s < device_buffer_size."""
    state["device_buffer_tokens"].fill_(-1)
    state["device_buffer_tokens"][:, :device_buffer_size] = init_tokens_row
    state["device_buffer_recency"].zero_()
    state["device_buffer_visited"].zero_()
    state["sieve_hand"].zero_()
    state["top_k_device_slots"].fill_(-1)
    state["is_miss"].zero_()
    state["global_recency_counter"].fill_(1)
    # Rebuild the persistent SIEVE hash table from the restored tokens (not
    # timed; mirrors the alloc-time init in the coordinator).
    sieve_ht_init_npu(
        state["device_buffer_tokens"].unsqueeze(0),
        state["device_buffer_ht"],
        state["prefill_len"],
        0,
        num_reqs,
        device_buffer_size,
        PADDED_BUFFER_SIZE,
        _sieve_ht_size(PADDED_BUFFER_SIZE, top_k),
    )


def _gen_topk(device, num_reqs, top_k, device_buffer_size, miss_ratio):
    """~(1-miss_ratio) hits (tokens in [0, dbs)), ~miss_ratio misses."""
    hits = torch.randint(0, device_buffer_size, (num_reqs, top_k))
    misses = torch.randint(device_buffer_size, MAX_CONTEXT_LEN, (num_reqs, top_k))
    mask = torch.rand(num_reqs, top_k) < miss_ratio
    return torch.where(mask, misses, hits).to(torch.int32).to(device)


def _run_update(algo, state, block_dim, num_reqs, top_k, device_buffer_size):
    if algo == "sieve":
        sieve_update_npu(
            layer_id=0,
            topk_indices=state["topk_indices"],
            req_pool_indices=state["req_pool_indices"],
            seq_lens=state["seq_lens"],
            prefill_len=state["prefill_len"],
            device_buffer_tokens=state["device_buffer_tokens"],
            device_buffer_visited=state["device_buffer_visited"],
            device_buffer_ht=state["device_buffer_ht"][0],
            sieve_hand=state["sieve_hand"],
            top_k_device_slots=state["top_k_device_slots"],
            is_miss=state["is_miss"],
            num_real_reqs=state["num_real_reqs"],
            top_k=top_k,
            device_buffer_size=device_buffer_size,
            padded_buffer_size=PADDED_BUFFER_SIZE,
            max_num_reqs=num_reqs,
            block_dim=block_dim,
        )
    else:
        lru_update_npu(
            layer_id=0,
            topk_indices=state["topk_indices"],
            req_pool_indices=state["req_pool_indices"],
            seq_lens=state["seq_lens"],
            prefill_len=state["prefill_len"],
            device_buffer_tokens=state["device_buffer_tokens"],
            device_buffer_recency=state["device_buffer_recency"],
            top_k_device_slots=state["top_k_device_slots"],
            is_miss=state["is_miss"],
            num_real_reqs=state["num_real_reqs"],
            global_recency_counter=state["global_recency_counter"],
            top_k=top_k,
            device_buffer_size=device_buffer_size,
            padded_buffer_size=PADDED_BUFFER_SIZE,
            max_num_reqs=num_reqs,
            block_dim=block_dim,
        )


def _run_once(state, block_dim, host_kv_dev_ptr,
              num_reqs, top_k, device_buffer_size, algo="lru"):
    _run_update(algo, state, block_dim, num_reqs, top_k, device_buffer_size)
    scatter_from_host_npu(
        host_kv_cache_ptr=host_kv_dev_ptr,
        topk_indices=state["topk_indices"],
        top_k_device_slots=state["top_k_device_slots"],
        is_miss=state["is_miss"],
        req_pool_indices=state["req_pool_indices"],
        req_to_host_pool=state["req_to_host_pool"],
        req_to_device_buffer=state["req_to_device_buffer"],
        device_k_buffer=state["device_k_buffer"],
        device_v_buffer=state["device_v_buffer"],
        layer_id=0,
        host_entries=MAX_CONTEXT_LEN,
        k_row_bytes=K_ROW_BYTES,
        v_row_bytes=V_ROW_BYTES,
        max_context_len=MAX_CONTEXT_LEN,
        device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
        padded_buffer_size=PADDED_BUFFER_SIZE,
        max_num_reqs=num_reqs,
        top_k=top_k,
        block_dim=block_dim,
    )


def _run_scenario(device, num_reqs, top_k, device_buffer_size, miss_ratio,
                  host_kv_dev):
    device_rows = num_reqs * (PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
    print(f"scenario: reqs={num_reqs} top_k={top_k} dbs={device_buffer_size} "
          f"padded={PADDED_BUFFER_SIZE} miss_ratio={miss_ratio} | "
          f"device buffer {device_rows * (K_ROW_BYTES + V_ROW_BYTES) / 2**20:.0f} MiB")

    state = _make_state(device, num_reqs, top_k, device_buffer_size)
    init_tokens_row = torch.arange(device_buffer_size, dtype=torch.int32, device=device)

    for algo in ALGOS:
      for block_dim in BLOCK_DIMS:
        lru_ms, scatter_ms, miss_rows = [], [], 0

        for it in range(WARMUP + ITERS):
            _reset_state(state, init_tokens_row, device_buffer_size, num_reqs, top_k)
            state["topk_indices"].copy_(
                _gen_topk(device, num_reqs, top_k, device_buffer_size, miss_ratio))

            start_lru = torch.npu.Event(enable_timing=True)
            end_lru = torch.npu.Event(enable_timing=True)
            end_scatter = torch.npu.Event(enable_timing=True)

            start_lru.record()
            _run_update(algo, state, block_dim, num_reqs, top_k, device_buffer_size)
            end_lru.record()

            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev,
                topk_indices=state["topk_indices"],
                top_k_device_slots=state["top_k_device_slots"],
                is_miss=state["is_miss"],
                req_pool_indices=state["req_pool_indices"],
                req_to_host_pool=state["req_to_host_pool"],
                req_to_device_buffer=state["req_to_device_buffer"],
                device_k_buffer=state["device_k_buffer"],
                device_v_buffer=state["device_v_buffer"],
                layer_id=0,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=num_reqs,
                top_k=top_k,
                block_dim=block_dim,
            )
            torch.npu.synchronize()
            end_scatter.record()

            if it >= WARMUP:
                lru_ms.append(start_lru.elapsed_time(end_lru))
                scatter_ms.append(end_lru.elapsed_time(end_scatter))
                miss_rows = int(state["is_miss"].sum().item())

        avg_lru = sum(lru_ms) / len(lru_ms)
        avg_scatter = sum(scatter_ms) / len(scatter_ms)
        miss_bytes = miss_rows * (K_ROW_BYTES + V_ROW_BYTES)
        bw_gbps = miss_bytes / (avg_scatter / 1000.0) / 2**30 if avg_scatter > 0 else 0.0
        tag = "auto" if block_dim == 0 else f"{block_dim:4d}"
        print(f"algo={algo:5s} block_dim={tag} | upd {avg_lru:9.3f} ms "
              f"(min {min(lru_ms):9.3f}) | scatter {avg_scatter:8.3f} ms "
              f"(min {min(scatter_ms):8.3f}) | miss {miss_rows} rows "
              f"({miss_bytes / 2**20:.1f} MiB, {bw_gbps:6.2f} GiB/s) | "
              f"total {avg_lru + avg_scatter:9.3f} ms")

    # Correctness sanity: one extra run per algo verified against expected miss count.
    for algo in ALGOS:
        _reset_state(state, init_tokens_row, device_buffer_size, num_reqs, top_k)
        state["topk_indices"].copy_(
            _gen_topk(device, num_reqs, top_k, device_buffer_size, miss_ratio))
        _run_once(state, BLOCK_DIMS[-1], host_kv_dev,
                  num_reqs, top_k, device_buffer_size, algo=algo)
        torch.npu.synchronize()
        miss = int(state["is_miss"].sum().item())
        ratio = miss / (num_reqs * top_k)
        # The effective miss ratio is slightly higher than the mask ratio because
        # earlier misses in a call evict slots that later "hit" tokens would use.
        print(f"sanity[{algo}]: miss_rows={miss} ratio={ratio:.4f} (target ~{miss_ratio})")
        assert abs(ratio - miss_ratio) < 0.05, "miss ratio out of expected range"

    del state
    torch.npu.empty_cache()


def main():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available, skipping perf benchmark.")
        return
    if not _HAS_KERNEL:
        print("hisparse_lru native extension not available, skipping perf benchmark.")
        return

    parser = argparse.ArgumentParser(description="HiSparse NPU kernel benchmarks")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=int(os.environ.get("HISPARSE_PERF_NUM_LAYERS", 8)),
        help="Layer count for the layout A/B section (default 8; 78 = production "
        "scale. NOTE: 78 needs ~734 MiB mlock-able pinned host memory and "
        "~4.4 GiB device memory)",
    )
    parser.add_argument(
        "--skip-legacy", action="store_true", help="skip the original scenarios"
    )
    parser.add_argument(
        "--backup-tokens",
        type=int,
        default=BACKUP_TOKENS_DEFAULT,
        help="token count for the backup A/B (default 8192)",
    )
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = "npu:0"

    print(f"K row {K_ROW_BYTES}B, V row {V_ROW_BYTES}B; "
          f"auto block_dim = {hisparse_pkg._resolve_block_dim(0)}")

    if not args.skip_legacy:
        # Pinned host cache (interleaved K+V, 1152B/row for host_entries=8192).
        kv_row_bytes = K_ROW_BYTES + V_ROW_BYTES
        host_kv_ptr, host_kv_dev, host_kv_size = _acl_malloc_host(MAX_CONTEXT_LEN * kv_row_bytes)
        try:
            for num_reqs, top_k, device_buffer_size, miss_ratio in SCENARIOS:
                _run_scenario(device, num_reqs, top_k, device_buffer_size, miss_ratio,
                              host_kv_dev)
            print("perf benchmark done.")
        finally:
            _acl_free_host(host_kv_ptr, host_kv_dev, host_kv_size)
    else:
        print("skipping legacy scenarios (--skip-legacy)")

    _run_layout_ab_section(device, args.num_layers, args.backup_tokens)


# ------------------------ layout A/B benchmark (M2) ------------------------

LAYOUT_AB_REQS = 8
LAYOUT_AB_TOP_K = 2048
LAYOUT_AB_ANCHOR = 1          # production anchors sit at L1, L5, ...
LAYOUT_AB_BLOCK_DIMS = (0, 32, 64)
LAYOUT_AB_MISS_RATIOS = (0.0, 0.01, 0.1, 0.5, 1.0)
LAYOUT_AB_GROUP_SIZES = (1, 2, 4, 8)
LAYOUT_WARMUP = 1
LAYOUT_ITERS = 5
BACKUP_TOKENS_DEFAULT = 8192
BACKUP_ITERS = 3
# Observed production step time at 8-way concurrency (tok/s -> ms/step) used
# only for the e2e projection printed at the end of the section.
E2E_STEP_TOK_S = 141.0
E2E_ANCHOR_GROUPS = 20


def _make_ab_state(device, num_layers):
    """Manual miss state for the scatter A/B: no LRU/SIEVE kernel involved.
    The scatter kernels only READ is_miss / top_k_device_slots, so the same
    state can drive both layouts without any reset between iterations."""
    num_reqs, top_k = LAYOUT_AB_REQS, LAYOUT_AB_TOP_K
    device_rows = num_reqs * (PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
    gen = torch.Generator(device="cpu").manual_seed(20260817)

    topk_indices = torch.randint(
        0, MAX_CONTEXT_LEN, (num_reqs, top_k), dtype=torch.int32, generator=gen
    ).to(device)
    # Unique valid slots per request (top_k < padded_buffer_size).
    slots_row = (torch.arange(top_k, dtype=torch.int32) % PADDED_BUFFER_SIZE)
    top_k_device_slots = slots_row.unsqueeze(0).repeat(num_reqs, 1).contiguous().to(device)
    is_miss = torch.zeros((num_reqs, top_k), dtype=torch.int8, device=device)
    req_pool_indices = torch.arange(num_reqs, dtype=torch.int64, device=device)
    req_to_host_pool = (
        torch.arange(MAX_CONTEXT_LEN, dtype=torch.int64, device=device)
        .unsqueeze(0)
        .repeat(num_reqs, 1)
        .contiguous()
    )
    req_to_device_buffer = (
        torch.arange(device_rows, dtype=torch.int64, device=device)
        .view(num_reqs, PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
        .contiguous()
    )
    device_k_buffer = torch.zeros(
        (num_layers, device_rows, KV_LORA_RANK), dtype=DTYPE, device=device
    )
    device_v_buffer = torch.zeros(
        (num_layers, device_rows, QK_ROPE_HEAD_DIM), dtype=DTYPE, device=device
    )
    return dict(
        topk_indices=topk_indices,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        req_pool_indices=req_pool_indices,
        req_to_host_pool=req_to_host_pool,
        req_to_device_buffer=req_to_device_buffer,
        device_k_buffer=device_k_buffer,
        device_v_buffer=device_v_buffer,
    )


def _set_ab_miss(state, device, miss_ratio):
    gen = torch.Generator(device="cpu").manual_seed(97531)
    mask = torch.rand(LAYOUT_AB_REQS, LAYOUT_AB_TOP_K, generator=gen) < miss_ratio
    state["is_miss"].copy_(mask.to(torch.int8).to(device))
    return int(mask.sum().item())


def _host_pool_view(host_ptr, num_layers, entry_major):
    """fp16 torch view over the pinned pool in the requested layout."""
    import ctypes

    row_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    total = num_layers * MAX_CONTEXT_LEN * row_dim
    buf = (ctypes.c_uint8 * (total * DTYPE.itemsize)).from_address(host_ptr)
    flat = torch.frombuffer(buf, dtype=DTYPE, count=total)
    if entry_major:
        return flat.view(MAX_CONTEXT_LEN, num_layers, row_dim)
    return flat.view(num_layers, MAX_CONTEXT_LEN, row_dim)


def _fill_host_pool_markers(host_ptr, num_layers, entry_major):
    """Layout-aware marker fill for correctness checks.  Each K row encodes
    its host token as two exactly-representable fp16 ints (tok % 1024,
    tok // 1024); each V row carries its layer id.  The scatter kernels copy
    raw bytes, so both layouts reconstruct to identical device content."""
    view = _host_pool_view(host_ptr, num_layers, entry_major)
    tok = torch.arange(MAX_CONTEXT_LEN, dtype=torch.int32)
    k_view = view[..., :KV_LORA_RANK]
    v_view = view[..., KV_LORA_RANK:]
    for layer in range(num_layers):
        if entry_major:
            k_col, v_col = k_view[:, layer], v_view[:, layer]
        else:
            k_col, v_col = k_view[layer], v_view[layer]
        k_col[:, 0] = (tok % 1024).to(DTYPE)
        k_col[:, 1] = (tok // 1024).to(DTYPE)
        v_col[:, 0] = torch.full_like(v_col[:, 0], float(layer))


def _run_group_scatter(state, host_dev, entry_major, anchor, group_size, block_dim,
                       num_layers):
    scatter_from_host_group_npu(
        host_kv_cache_ptr=host_dev,
        topk_indices=state["topk_indices"],
        top_k_device_slots=state["top_k_device_slots"],
        is_miss=state["is_miss"],
        req_pool_indices=state["req_pool_indices"],
        req_to_host_pool=state["req_to_host_pool"],
        req_to_device_buffer=state["req_to_device_buffer"],
        device_k_buffer=state["device_k_buffer"],
        device_v_buffer=state["device_v_buffer"],
        anchor_layer_id=anchor,
        group_size=group_size,
        host_entries=MAX_CONTEXT_LEN,
        k_row_bytes=K_ROW_BYTES,
        v_row_bytes=V_ROW_BYTES,
        max_context_len=MAX_CONTEXT_LEN,
        device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
        padded_buffer_size=PADDED_BUFFER_SIZE,
        max_num_reqs=LAYOUT_AB_REQS,
        top_k=LAYOUT_AB_TOP_K,
        host_entry_major=entry_major,
        host_num_layers=num_layers,
        block_dim=block_dim,
    )


def _run_legacy_scatter(state, host_dev, entry_major, block_dim, num_layers):
    scatter_from_host_npu(
        host_kv_cache_ptr=host_dev,
        topk_indices=state["topk_indices"],
        top_k_device_slots=state["top_k_device_slots"],
        is_miss=state["is_miss"],
        req_pool_indices=state["req_pool_indices"],
        req_to_host_pool=state["req_to_host_pool"],
        req_to_device_buffer=state["req_to_device_buffer"],
        device_k_buffer=state["device_k_buffer"][0],
        device_v_buffer=state["device_v_buffer"][0],
        layer_id=0,
        host_entries=MAX_CONTEXT_LEN,
        k_row_bytes=K_ROW_BYTES,
        v_row_bytes=V_ROW_BYTES,
        max_context_len=MAX_CONTEXT_LEN,
        device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
        padded_buffer_size=PADDED_BUFFER_SIZE,
        max_num_reqs=LAYOUT_AB_REQS,
        top_k=LAYOUT_AB_TOP_K,
        host_entry_major=entry_major,
        host_num_layers=num_layers,
        block_dim=block_dim,
    )


def _verify_group_scatter(state, anchor, group_size):
    """Layout-agnostic check: every miss row must land in every group layer
    with the marker of its host token (both layouts must pass this)."""
    req_i, k_i = state["is_miss"].nonzero(as_tuple=True)
    if req_i.numel() == 0:
        return
    tok = state["topk_indices"][req_i, k_i].to(torch.int64)
    slot = state["top_k_device_slots"][req_i, k_i].to(torch.int64)
    row = state["req_to_device_buffer"][req_i, slot]
    for g in range(group_size):
        layer = anchor + g
        k0 = state["device_k_buffer"][layer][row, 0].to(torch.int64)
        k1 = state["device_k_buffer"][layer][row, 1].to(torch.int64)
        assert torch.equal(k1 * 1024 + k0, tok), (
            f"group scatter K marker mismatch at layer {layer}"
        )
        v0 = state["device_v_buffer"][layer][row, 0].to(torch.int64)
        assert torch.all(v0 == layer), (
            f"group scatter V marker mismatch at layer {layer}"
        )


def _time_interleaved(fn_a, fn_b):
    """Interleaved A/B event timing (cancels thermal/clock drift).  Returns
    (avg_ms_a, avg_ms_b, min_ms_a, min_ms_b)."""
    times = ([], [])
    torch.npu.synchronize()
    for it in range(LAYOUT_WARMUP + LAYOUT_ITERS):
        for which, fn in enumerate((fn_a, fn_b)):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.npu.synchronize()
            if it >= LAYOUT_WARMUP:
                times[which].append(start.elapsed_time(end))
    return (
        sum(times[0]) / len(times[0]),
        sum(times[1]) / len(times[1]),
        min(times[0]),
        min(times[1]),
    )


def _run_layout_ab_section(device, num_layers, backup_tokens):
    kv_row_bytes = K_ROW_BYTES + V_ROW_BYTES
    pool_bytes = num_layers * MAX_CONTEXT_LEN * kv_row_bytes
    device_rows = LAYOUT_AB_REQS * (PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
    print(
        f"\n=== layout A/B (M2): layers={num_layers} reqs={LAYOUT_AB_REQS} "
        f"top_k={LAYOUT_AB_TOP_K} anchor={LAYOUT_AB_ANCHOR} | "
        f"host pool {pool_bytes / 2**20:.0f} MiB, device pool "
        f"{num_layers * device_rows * kv_row_bytes / 2**20:.0f} MiB ==="
    )

    host_ptr, host_dev, host_size = _acl_malloc_host(pool_bytes)
    try:
        state = _make_ab_state(device, num_layers)
        group_sizes = [g for g in LAYOUT_AB_GROUP_SIZES
                       if g <= num_layers - LAYOUT_AB_ANCHOR]
        step_ms = 1000.0 * LAYOUT_AB_REQS / E2E_STEP_TOK_S
        best_gain = None  # (speedup, saved_us_per_anchor, tag)

        for miss_ratio in LAYOUT_AB_MISS_RATIOS:
            miss_rows = _set_ab_miss(state, device, miss_ratio)
            for group_size in group_sizes:
                # Correctness pass: fill markers per layout, scatter once,
                # verify - both address formulas must reconstruct the same
                # host tokens (regression check for the M2 entry-major math).
                for entry_major in (True, False):
                    _fill_host_pool_markers(host_ptr, num_layers, entry_major)
                    _run_group_scatter(state, host_dev, entry_major,
                                       LAYOUT_AB_ANCHOR, group_size, 0,
                                       num_layers)
                    torch.npu.synchronize()
                    _verify_group_scatter(state, LAYOUT_AB_ANCHOR, group_size)

                # Timing pass: moved byte volume is identical under either
                # interpretation, so fill once and interleave A/B timing.
                _fill_host_pool_markers(host_ptr, num_layers, True)
                for block_dim in LAYOUT_AB_BLOCK_DIMS:
                    tag_bd = "auto" if block_dim == 0 else f"{block_dim:4d}"
                    entry_ms, layer_ms, e_min, l_min = _time_interleaved(
                        lambda: _run_group_scatter(
                            state, host_dev, True, LAYOUT_AB_ANCHOR, group_size,
                            block_dim, num_layers),
                        lambda: _run_group_scatter(
                            state, host_dev, False, LAYOUT_AB_ANCHOR, group_size,
                            block_dim, num_layers),
                    )

                    moved = miss_rows * group_size * kv_row_bytes
                    e_bw = moved / (entry_ms / 1000.0) / 2**30 if entry_ms > 0 else 0.0
                    l_bw = moved / (layer_ms / 1000.0) / 2**30 if layer_ms > 0 else 0.0
                    speedup = layer_ms / entry_ms if entry_ms > 0 else 0.0
                    print(
                        f"miss_ratio={miss_ratio:4.2f} ({miss_rows:6d} rows) "
                        f"group={group_size} bd={tag_bd} | "
                        f"entry {entry_ms:8.3f} ms (min {e_min:8.3f}, {e_bw:6.2f} GiB/s) | "
                        f"layer {layer_ms:8.3f} ms (min {l_min:8.3f}, {l_bw:6.2f} GiB/s) | "
                        f"speedup {speedup:5.2f}x"
                    )
                    if miss_rows > 0 and group_size == 4 and block_dim == 0:
                        saved_us = max(0.0, layer_ms - entry_ms) * 1000.0
                        if best_gain is None or saved_us > best_gain[1]:
                            best_gain = (speedup, saved_us,
                                         f"miss={miss_ratio} group=4")

        # Control group: single-layer legacy scatter, entry vs layer.  Both
        # layouts move one 1152B row per miss, so a significant difference
        # here means the timing itself is polluted, not the layout.
        print("--- control: legacy single-layer scatter (entry vs layer) ---")
        for miss_ratio in LAYOUT_AB_MISS_RATIOS:
            if miss_ratio == 0.0:
                continue
            miss_rows = _set_ab_miss(state, device, miss_ratio)
            for entry_major in (True, False):
                _fill_host_pool_markers(host_ptr, num_layers, entry_major)
                _run_legacy_scatter(state, host_dev, entry_major, 0, num_layers)
                torch.npu.synchronize()
                _verify_group_scatter(state, 0, 1)  # layer 0 only
            entry_ms, layer_ms, e_min, l_min = _time_interleaved(
                lambda: _run_legacy_scatter(state, host_dev, True, 0, num_layers),
                lambda: _run_legacy_scatter(state, host_dev, False, 0, num_layers),
            )
            print(
                f"miss_ratio={miss_ratio:4.2f} ({miss_rows:6d} rows) legacy bd=auto | "
                f"entry {entry_ms:8.3f} ms (min {e_min:8.3f}) | "
                f"layer {layer_ms:8.3f} ms (min {l_min:8.3f}) | "
                f"delta {(layer_ms - entry_ms) * 1000:8.1f} us"
            )

        _run_backup_ab(device, state, host_ptr, host_dev, num_layers, backup_tokens)

        if best_gain is not None:
            speedup, saved_us, tag = best_gain
            total_us = saved_us * E2E_ANCHOR_GROUPS
            print(
                f"\nprojection (best group=4 case, {tag}): "
                f"{saved_us:.0f} us saved/anchor x {E2E_ANCHOR_GROUPS} anchors = "
                f"{total_us / 1000:.2f} ms per step vs observed step "
                f"{step_ms:.1f} ms -> up to {100.0 * total_us / 1000.0 / step_ms:.1f}% "
                f"e2e at this miss level"
            )
        del state
        torch.npu.empty_cache()
    finally:
        _acl_free_host(host_ptr, host_dev, host_size)


def _backup_fallback(host_view_entry, k_buffer, v_buffer, host_idx_cpu, dev_idx):
    """Mirror of TieredHostMemoryPool.backup_from_device_all_layer's
    index_copy_ fallback (entry-major host views)."""
    for layer in range(k_buffer.size(0)):
        dk = k_buffer[layer].view(-1, KV_LORA_RANK)[dev_idx]
        dv = v_buffer[layer].view(-1, QK_ROPE_HEAD_DIM)[dev_idx]
        host_view_entry[:, layer, :KV_LORA_RANK].index_copy_(0, host_idx_cpu, dk.cpu())
        host_view_entry[:, layer, KV_LORA_RANK:].index_copy_(0, host_idx_cpu, dv.cpu())


def _verify_backup(host_view_entry, host_idx_cpu, num_layers):
    """Only the backed-up entries are written; check those."""
    k_view = host_view_entry[host_idx_cpu, :, :KV_LORA_RANK]
    v_view = host_view_entry[host_idx_cpu, :, KV_LORA_RANK:]
    for layer in range(num_layers):
        assert torch.all(k_view[:, layer, 0] == float(layer) + 0.5), (
            f"backup K marker mismatch at layer {layer}"
        )
        assert torch.all(v_view[:, layer, 0] == float(layer) * 2.0 + 0.25), (
            f"backup V marker mismatch at layer {layer}"
        )


def _run_backup_ab(device, state, host_ptr, host_dev, num_layers, num_tokens):
    num_tokens = min(num_tokens, MAX_CONTEXT_LEN)
    device_rows = state["device_k_buffer"].size(1)
    print(
        f"--- backup A/B: {num_tokens} tokens x {num_layers} layers "
        f"(kernel vs index_copy_ fallback) ---"
    )

    # Distinct per-layer constants make any layer/token address error visible.
    for layer in range(num_layers):
        state["device_k_buffer"][layer].fill_(float(layer) + 0.5)
        state["device_v_buffer"][layer].fill_(float(layer) * 2.0 + 0.25)

    gen = torch.Generator(device="cpu").manual_seed(24680)
    host_idx = torch.randperm(MAX_CONTEXT_LEN, generator=gen)[:num_tokens]
    dev_idx = torch.randint(0, device_rows, (num_tokens,), generator=gen)
    host_idx_dev = host_idx.to(device=device, dtype=torch.int64)
    dev_idx_dev = dev_idx.to(device=device, dtype=torch.int64)
    host_idx_cpu = host_idx.to(torch.int64)

    host_view_entry = _host_pool_view(host_ptr, num_layers, True)

    # Kernel path (timed with events, warmup + iters).
    k_times = []
    torch.npu.synchronize()
    for it in range(LAYOUT_WARMUP + LAYOUT_ITERS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        backup_to_host_npu(
            device_k_buffer=state["device_k_buffer"],
            device_v_buffer=state["device_v_buffer"],
            host_kv_cache_ptr=host_dev,
            host_indices=host_idx_dev,
            device_indices=dev_idx_dev,
            host_entries=MAX_CONTEXT_LEN,
            host_num_layers=num_layers,
            k_row_bytes=K_ROW_BYTES,
            v_row_bytes=V_ROW_BYTES,
            num_tokens=num_tokens,
            block_dim=0,
        )
        end.record()
        torch.npu.synchronize()
        if it >= LAYOUT_WARMUP:
            k_times.append(start.elapsed_time(end))
    _verify_backup(host_view_entry, host_idx_cpu, num_layers)

    # Fallback path (host-driven; wall clock).
    f_times = []
    torch.npu.synchronize()
    for it in range(LAYOUT_WARMUP + BACKUP_ITERS):
        t0 = time.perf_counter()
        _backup_fallback(host_view_entry, state["device_k_buffer"],
                         state["device_v_buffer"], host_idx_cpu, dev_idx_dev)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        if it >= LAYOUT_WARMUP:
            f_times.append((t1 - t0) * 1000.0)
    _verify_backup(host_view_entry, host_idx_cpu, num_layers)

    moved = num_tokens * num_layers * (K_ROW_BYTES + V_ROW_BYTES)
    k_avg = sum(k_times) / len(k_times)
    f_avg = sum(f_times) / len(f_times)
    print(
        f"backup kernel   {k_avg:9.3f} ms (min {min(k_times):9.3f}, "
        f"{moved / (k_avg / 1000.0) / 2**30:6.2f} GiB/s) | "
        f"fallback {f_avg:9.3f} ms (min {min(f_times):9.3f}) | "
        f"speedup {f_avg / k_avg:6.1f}x | moved {moved / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    main()
