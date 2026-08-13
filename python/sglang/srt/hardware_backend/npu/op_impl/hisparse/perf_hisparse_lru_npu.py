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

Run on an NPU host:
    python perf_hisparse_lru_npu.py
"""

import torch

from sglang.srt.hardware_backend.npu.op_impl import hisparse as hisparse_pkg
from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
    _HAS_KERNEL,
    lru_update_npu,
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

    torch.npu.set_device(0)
    device = "npu:0"

    print(f"K row {K_ROW_BYTES}B, V row {V_ROW_BYTES}B; "
          f"auto block_dim = {hisparse_pkg._resolve_block_dim(0)}")

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


if __name__ == "__main__":
    main()
