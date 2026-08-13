"""
NPU standalone tests for the HiSparse graph-compatible kernels.

These tests exercise the compiled AscendC extension ``hisparse_lru.so`` with
real NPU tensors. They require an NPU device and CANN toolchain.

Run on an NPU host:
    python test_hisparse_lru_npu.py
"""

import ctypes
import ctypes.util
import random
from typing import Tuple

import torch

from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
    lru_update_npu,
    scatter_from_host_npu,
    sieve_ht_init_npu,
    sieve_update_npu,
)


def _acl_malloc_host(size: int) -> Tuple[int, int, int]:
    """Allocate pinned host memory and return (host_ptr, dev_ptr, size).

    Memory is allocated with posix_memalign, locked with mlock, and registered
    with aclrtHostRegister so that the NPU kernel can access it via dev_ptr.
    """
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.posix_memalign.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    libc.posix_memalign.restype = ctypes.c_int
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int

    host_ptr = ctypes.c_void_p()
    ret = libc.posix_memalign(ctypes.byref(host_ptr), 4096, size)
    if ret != 0:
        raise RuntimeError(f"posix_memalign failed: {ret}")

    ret = libc.mlock(host_ptr, size)
    if ret != 0:
        libc.free(host_ptr)
        raise RuntimeError(f"mlock failed: {ret}")

    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    acl.aclrtHostRegister.restype = ctypes.c_int

    dev_ptr = ctypes.c_void_p()
    ret = acl.aclrtHostRegister(host_ptr, size, 0, ctypes.byref(dev_ptr))
    if ret != 0:
        libc.munlock(host_ptr, size)
        libc.free(host_ptr)
        raise RuntimeError(f"aclrtHostRegister failed: {ret}")

    return host_ptr.value, dev_ptr.value, size


def _acl_free_host(host_ptr: int, dev_ptr: int, size: int) -> None:
    """Free pinned host memory allocated by posix_memalign + mlock + aclrtHostRegister."""
    del dev_ptr
    if host_ptr == 0:
        return
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    acl.aclrtHostUnregister.restype = ctypes.c_int
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munlock.restype = ctypes.c_int

    acl.aclrtHostUnregister(ctypes.c_void_p(host_ptr))
    libc.munlock(ctypes.c_void_p(host_ptr), size)
    libc.free(ctypes.c_void_p(host_ptr))


def _make_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len):
    device = "npu:0"
    topk_indices = torch.full((max_num_reqs, top_k), -1, dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(max_num_reqs, dtype=torch.int64, device=device)
    seq_lens = torch.full((max_num_reqs,), max_context_len, dtype=torch.int32, device=device)
    prefill_len = torch.full((max_num_reqs,), max_context_len, dtype=torch.int32, device=device)
    device_buffer_tokens = torch.full(
        (max_num_reqs, padded_buffer_size), -1, dtype=torch.int32, device=device
    )
    device_buffer_recency = torch.zeros(
        (max_num_reqs, padded_buffer_size), dtype=torch.int32, device=device
    )
    top_k_device_slots = torch.full((max_num_reqs, top_k), -1, dtype=torch.int32, device=device)
    is_miss = torch.zeros((max_num_reqs, top_k), dtype=torch.int8, device=device)
    num_real_reqs = torch.tensor([max_num_reqs], dtype=torch.int32, device=device)
    global_recency_counter = torch.ones(1, dtype=torch.int32, device=device)
    # Initialize slots: slot s holds token s.
    for s in range(device_buffer_size):
        device_buffer_tokens[:, s] = s
    return (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
        num_real_reqs, global_recency_counter,
    )


def test_all_hits():
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 2, 4, 8, 16
    max_context_len = 16
    (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
        num_real_reqs, global_recency_counter,
    ) = _make_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)

    topk_indices[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    topk_indices[1] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    lru_update_npu(
        layer_id=0,
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_recency=device_buffer_recency,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        global_recency_counter=global_recency_counter,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        padded_buffer_size=padded_buffer_size,
        max_num_reqs=max_num_reqs,
    )

    assert is_miss.sum().item() == 0
    assert top_k_device_slots[0].tolist() == [0, 1, 2, 3]
    assert top_k_device_slots[1].tolist() == [0, 1, 2, 3]
    print("test_all_hits passed")


def test_misses_evict_coldest():
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 1, 4, 4, 16
    max_context_len = 16
    (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
        num_real_reqs, global_recency_counter,
    ) = _make_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)

    # All tokens are misses. They should evict slots 0,1,2,3 in order.
    topk_indices[0] = torch.tensor([4, 5, 6, 7], dtype=torch.int32)

    lru_update_npu(
        layer_id=0,
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_recency=device_buffer_recency,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        global_recency_counter=global_recency_counter,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        padded_buffer_size=padded_buffer_size,
        max_num_reqs=max_num_reqs,
    )

    assert is_miss.sum().item() == 4
    assert top_k_device_slots[0].tolist() == [0, 1, 2, 3]
    assert is_miss[0].tolist() == [1, 1, 1, 1]
    print("test_misses_evict_coldest passed")


def test_decode_tokens_use_extension():
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 1, 4, 4, 16
    max_context_len = 16
    (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
        num_real_reqs, global_recency_counter,
    ) = _make_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)

    prefill_len[0] = 4
    seq_lens[0] = 6
    # tokens 0,1 are prefill hits; tokens 4,5 are decode tokens.
    topk_indices[0] = torch.tensor([0, 1, 4, 5], dtype=torch.int32)

    lru_update_npu(
        layer_id=0,
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_recency=device_buffer_recency,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        global_recency_counter=global_recency_counter,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        padded_buffer_size=padded_buffer_size,
        max_num_reqs=max_num_reqs,
    )

    assert is_miss.sum().item() == 0
    assert top_k_device_slots[0].tolist() == [0, 1, padded_buffer_size + 0, padded_buffer_size + 1]
    print("test_decode_tokens_use_extension passed")


def test_scatter_one_miss():
    """Minimal smoke test for hisparse_scatter_from_host kernel."""
    max_num_reqs, top_k = 1, 1
    device_buffer_size, padded_buffer_size = 4, 5
    max_context_len = 8
    max_decode_len = 4
    kv_lora_rank = 16
    qk_rope_head_dim = 16
    dtype = torch.int16
    es = dtype.itemsize
    k_row_bytes = kv_lora_rank * es
    v_row_bytes = qk_rope_head_dim * es
    device = "npu:0"

    torch.npu.set_device(0)

    # One host cache entry per token, filled with distinguishable values.
    # Use aclrtMallocHost + aclrtHostRegister so the memory is accessible from the NPU kernel.
    # Interleaved K/V layout: each row is [K(kv_lora_rank) | V(qk_rope_head_dim)].
    host_entries = max_context_len
    kv_row_bytes = k_row_bytes + v_row_bytes
    kv_total_bytes = host_entries * kv_row_bytes
    host_kv_cache_ptr, host_kv_dev_ptr, host_kv_size = _acl_malloc_host(kv_total_bytes)

    host_kv_cache = torch.frombuffer(
        (ctypes.c_uint16 * (host_entries * (kv_lora_rank + qk_rope_head_dim))).from_address(host_kv_cache_ptr),
        dtype=dtype,
        count=host_entries * (kv_lora_rank + qk_rope_head_dim),
    ).view(host_entries, kv_lora_rank + qk_rope_head_dim)
    host_k_cache = host_kv_cache[:, :kv_lora_rank]
    host_v_cache = host_kv_cache[:, kv_lora_rank:]
    for h in range(host_entries):
        host_k_cache[h].fill_(h + 1000)
        host_v_cache[h].fill_(h + 2000)

    # Device buffer one physical slot for the miss.
    device_k_buffer = torch.zeros((1, kv_lora_rank), dtype=dtype, device=device)
    device_v_buffer = torch.zeros((1, qk_rope_head_dim), dtype=dtype, device=device)

    # Tensors describing the miss.
    topk_indices = torch.tensor([[4]], dtype=torch.int32, device=device)
    top_k_device_slots = torch.tensor([[0]], dtype=torch.int32, device=device)
    is_miss = torch.tensor([[1]], dtype=torch.int8, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)
    req_to_host_pool = torch.full(
        (max_num_reqs, max_context_len), -1, dtype=torch.int64, device=device
    )
    req_to_host_pool[0, 4] = 4  # token 4 maps to host entry 4
    req_to_device_buffer = torch.full(
        (max_num_reqs, padded_buffer_size + max_decode_len), -1, dtype=torch.int64, device=device
    )
    req_to_device_buffer[0, 0] = 0  # device buffer slot 0 maps to physical loc 0

    try:
        scatter_from_host_npu(
            host_kv_cache_ptr=host_kv_dev_ptr,
            topk_indices=topk_indices,
            top_k_device_slots=top_k_device_slots,
            is_miss=is_miss,
            req_pool_indices=req_pool_indices,
            req_to_host_pool=req_to_host_pool,
            req_to_device_buffer=req_to_device_buffer,
            device_k_buffer=device_k_buffer,
            device_v_buffer=device_v_buffer,
            layer_id=0,
            host_entries=host_entries,
            k_row_bytes=k_row_bytes,
            v_row_bytes=v_row_bytes,
            max_context_len=max_context_len,
            device_buffer_row_stride=padded_buffer_size + max_decode_len,
            padded_buffer_size=padded_buffer_size,
            max_num_reqs=max_num_reqs,
            top_k=top_k,
        )

        expected_k = torch.full((kv_lora_rank,), 4 + 1000, dtype=dtype, device=device)
        expected_v = torch.full((qk_rope_head_dim,), 4 + 2000, dtype=dtype, device=device)
        assert torch.all(device_k_buffer[0] == expected_k), f"K mismatch: {device_k_buffer[0]} != {expected_k}"
        assert torch.all(device_v_buffer[0] == expected_v), f"V mismatch: {device_v_buffer[0]} != {expected_v}"
        print("test_scatter_one_miss passed")
    finally:
        _acl_free_host(host_kv_cache_ptr, host_kv_dev_ptr, host_kv_size)


def test_scatter_layer_offset_and_row_stride():
    """Regression test: scatter must apply the layer offset on top of the host
    base pointer exactly once, and must index req_to_device_buffer with its own
    row stride (padded_buffer_size + max_decode_len), not max_context_len.

    Uses layer_id=1 and non-contiguous req_pool_indices [1, 3] so both bugs
    would produce wrong addresses.
    """
    max_num_reqs, top_k = 4, 2
    padded_buffer_size = 5
    max_context_len = 8
    max_decode_len = 4
    device_row_stride = padded_buffer_size + max_decode_len  # 9 != max_context_len
    num_layers = 2
    host_entries = 16
    layer_id = 1
    kv_lora_rank = 16
    qk_rope_head_dim = 16
    dtype = torch.int16
    es = dtype.itemsize
    k_row_bytes = kv_lora_rank * es  # 32
    v_row_bytes = qk_rope_head_dim * es  # 32
    device = "npu:0"

    torch.npu.set_device(0)

    kv_row_bytes = k_row_bytes + v_row_bytes
    kv_total_bytes = num_layers * host_entries * kv_row_bytes
    host_kv_cache_ptr, host_kv_dev_ptr, host_kv_size = _acl_malloc_host(kv_total_bytes)

    host_kv_cache = torch.frombuffer(
        (ctypes.c_uint16 * (num_layers * host_entries * (kv_lora_rank + qk_rope_head_dim))).from_address(host_kv_cache_ptr),
        dtype=dtype,
        count=num_layers * host_entries * (kv_lora_rank + qk_rope_head_dim),
    ).view(num_layers, host_entries, kv_lora_rank + qk_rope_head_dim)
    host_k_cache = host_kv_cache[:, :, :kv_lora_rank]
    host_v_cache = host_kv_cache[:, :, kv_lora_rank:]
    for l in range(num_layers):
        for h in range(host_entries):
            host_k_cache[l, h].fill_(l * 10000 + h + 1000)
            host_v_cache[l, h].fill_(l * 10000 + h + 2000)

    device_rows = 400
    device_k_buffer = torch.zeros((device_rows, kv_lora_rank), dtype=dtype, device=device)
    device_v_buffer = torch.zeros((device_rows, qk_rope_head_dim), dtype=dtype, device=device)

    # Two active requests at req_pool_idx 1 and 3; one miss each.
    topk_indices = torch.tensor([[2, -1], [-1, 6]], dtype=torch.int32, device=device)
    top_k_device_slots = torch.tensor([[1, -1], [-1, 3]], dtype=torch.int32, device=device)
    is_miss = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8, device=device)
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int64, device=device)
    req_to_host_pool = torch.full(
        (max_num_reqs, max_context_len), -1, dtype=torch.int64, device=device
    )
    req_to_host_pool[1, 2] = 5
    req_to_host_pool[3, 6] = 7
    req_to_device_buffer = torch.full(
        (max_num_reqs, device_row_stride), -1, dtype=torch.int64, device=device
    )
    phys_a = 103  # req 1, buffer slot 1
    phys_b = 307  # req 3, buffer slot 3
    req_to_device_buffer[1, 1] = phys_a
    req_to_device_buffer[3, 3] = phys_b

    try:
        scatter_from_host_npu(
            host_kv_cache_ptr=host_kv_dev_ptr,
            topk_indices=topk_indices,
            top_k_device_slots=top_k_device_slots,
            is_miss=is_miss,
            req_pool_indices=req_pool_indices,
            req_to_host_pool=req_to_host_pool,
            req_to_device_buffer=req_to_device_buffer,
            device_k_buffer=device_k_buffer,
            device_v_buffer=device_v_buffer,
            layer_id=layer_id,
            host_entries=host_entries,
            k_row_bytes=k_row_bytes,
            v_row_bytes=v_row_bytes,
            max_context_len=max_context_len,
            device_buffer_row_stride=device_row_stride,
            padded_buffer_size=padded_buffer_size,
            max_num_reqs=max_num_reqs,
            top_k=top_k,
        )

        # Each miss copies BOTH the K row and the V row to the same device slot.
        expected_k_a = torch.full((kv_lora_rank,), 10000 + 5 + 1000, dtype=dtype, device=device)
        expected_v_a = torch.full((qk_rope_head_dim,), 10000 + 5 + 2000, dtype=dtype, device=device)
        expected_k_b = torch.full((kv_lora_rank,), 10000 + 7 + 1000, dtype=dtype, device=device)
        expected_v_b = torch.full((qk_rope_head_dim,), 10000 + 7 + 2000, dtype=dtype, device=device)
        assert torch.all(device_k_buffer[phys_a] == expected_k_a), \
            f"K mismatch: {device_k_buffer[phys_a]} != {expected_k_a}"
        assert torch.all(device_v_buffer[phys_a] == expected_v_a), \
            f"V mismatch: {device_v_buffer[phys_a]} != {expected_v_a}"
        assert torch.all(device_k_buffer[phys_b] == expected_k_b), \
            f"K mismatch: {device_k_buffer[phys_b]} != {expected_k_b}"
        assert torch.all(device_v_buffer[phys_b] == expected_v_b), \
            f"V mismatch: {device_v_buffer[phys_b]} != {expected_v_b}"
        # Nothing else may have been written.
        written_k = (device_k_buffer != 0).any(dim=1).nonzero().flatten().tolist()
        written_v = (device_v_buffer != 0).any(dim=1).nonzero().flatten().tolist()
        assert written_k == [phys_a, phys_b], f"unexpected K rows written: {written_k}"
        assert written_v == [phys_a, phys_b], f"unexpected V rows written: {written_v}"
        print("test_scatter_layer_offset_and_row_stride passed")
    finally:
        _acl_free_host(host_kv_cache_ptr, host_kv_dev_ptr, host_kv_size)


class _KernelModel:
    """CPU model that mirrors the AscendC kernel semantics exactly.

    One kernel call applies a single recency value C to the whole batch; the
    caller increments ``global_recency_counter`` after each call (same protocol
    as ``HiSparseCoordinator``).  Per request the kernel scans buffer slots
    linearly for a hit, and on a miss evicts the first slot with the minimum
    recency.
    """

    def __init__(self, device_buffer_size, padded_buffer_size):
        self.dbs = device_buffer_size
        self.pbs = padded_buffer_size
        self.tokens = {}
        self.recency = {}

    def init_req(self, req_idx, tokens_row, recency_row):
        self.tokens[req_idx] = list(tokens_row)
        self.recency[req_idx] = list(recency_row)

    def update(self, req_idx, topk_row, prefill, seq_len, C):
        tokens = self.tokens[req_idx]
        recency = self.recency[req_idx]
        slots, miss = [], []
        for token_pos in topk_row:
            if token_pos < 0 or token_pos >= seq_len:
                slots.append(-1)
                miss.append(0)
                continue
            if token_pos >= prefill:
                slots.append(self.pbs + (token_pos - prefill))
                miss.append(0)
                continue
            hit = -1
            for s in range(self.dbs):
                if tokens[s] == token_pos:
                    hit = s
                    recency[s] = C
                    break
            if hit >= 0:
                slots.append(hit)
                miss.append(0)
            else:
                coldest = 0
                min_r = recency[0]
                for s in range(1, self.dbs):
                    if recency[s] < min_r:
                        min_r = recency[s]
                        coldest = s
                slots.append(coldest)
                miss.append(1)
                tokens[coldest] = token_pos
                recency[coldest] = C
        return slots, miss


def _lru_call(tensors, block_dim, max_num_reqs, top_k, device_buffer_size, padded_buffer_size):
    (topk_indices, req_pool_indices, seq_lens, prefill_len,
     device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
     num_real_reqs, global_recency_counter) = tensors
    lru_update_npu(
        layer_id=0,
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_recency=device_buffer_recency,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        global_recency_counter=global_recency_counter,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        padded_buffer_size=padded_buffer_size,
        max_num_reqs=max_num_reqs,
        block_dim=block_dim,
    )


def test_multi_block_stress():
    """Stress multi-block distribution against an exact CPU model.

    Covers hits, misses (LRU eviction), decode-extension slots and invalid
    positions across many rounds, with all 8 AIV blocks actively processing
    disjoint contiguous request ranges.
    """
    for top_k, max_num_reqs in ((4, 128), (8, 64)):
        for block_dim in (1, 2, 8):
            device_buffer_size, padded_buffer_size = 8, 16
            max_context_len = 32
            rounds = 12
            rng = random.Random((top_k << 16) ^ (max_num_reqs << 4) ^ block_dim)
            (topk_indices, req_pool_indices, seq_lens, prefill_len,
             device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
             num_real_reqs, global_recency_counter) = _make_tensors(
                max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
            prefill_len.fill_(max_context_len - 4)

            model = _KernelModel(device_buffer_size, padded_buffer_size)
            for r in range(max_num_reqs):
                model.init_req(r, device_buffer_tokens[r].tolist(),
                               device_buffer_recency[r].tolist())

            tensors = (topk_indices, req_pool_indices, seq_lens, prefill_len,
                       device_buffer_tokens, device_buffer_recency, top_k_device_slots,
                       is_miss, num_real_reqs, global_recency_counter)
            tag = f"top_k={top_k} reqs={max_num_reqs} block_dim={block_dim}"
            for rd in range(rounds):
                # Top-k indices are unique per request in production (torch.topk);
                # sample without replacement to match that invariant.
                rows = [rng.sample(range(-2, max_context_len + 2), top_k)
                        for _ in range(max_num_reqs)]
                topk_indices.copy_(torch.tensor(rows, dtype=torch.int32))

                C = int(global_recency_counter.item())
                _lru_call(tensors, block_dim, max_num_reqs, top_k,
                          device_buffer_size, padded_buffer_size)

                for b in range(max_num_reqs):
                    req_idx = int(req_pool_indices[b].item())
                    exp_slots, exp_miss = model.update(
                        req_idx, rows[b],
                        int(prefill_len[req_idx].item()),
                        int(seq_lens[b].item()), C)
                    assert top_k_device_slots[b].tolist() == exp_slots, \
                        f"{tag} round {rd} bid {b} slots mismatch"
                    assert is_miss[b].tolist() == exp_miss, \
                        f"{tag} round {rd} bid {b} is_miss mismatch"
                    assert device_buffer_tokens[req_idx][:device_buffer_size].tolist() == \
                        model.tokens[req_idx][:device_buffer_size], \
                        f"{tag} round {rd} bid {b} buffer tokens mismatch"
                    assert device_buffer_recency[req_idx][:device_buffer_size].tolist() == \
                        model.recency[req_idx][:device_buffer_size], \
                        f"{tag} round {rd} bid {b} buffer recency mismatch"

                global_recency_counter.add_(1)
            print(f"  {tag} OK")
    print("test_multi_block_stress passed")


def test_multi_block_partial_active():
    """num_real < max_num_reqs with non-contiguous req_pool_indices.

    Verifies active requests spread over multiple blocks are processed exactly
    as the model predicts, inactive output rows stay untouched, and buffer rows
    of requests outside the active set are never written.
    """
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 128, 4, 8, 16
    max_context_len = 32
    block_dim = 8
    num_active = 40
    rounds = 6
    rng = random.Random(1234)
    (topk_indices, req_pool_indices, seq_lens, prefill_len,
     device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
     num_real_reqs, global_recency_counter) = _make_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    prefill_len.fill_(max_context_len - 4)

    active = torch.tensor([(i * 37) % max_num_reqs for i in range(num_active)],
                          dtype=torch.int64, device=req_pool_indices.device)
    req_pool_indices[:num_active] = active
    num_real_reqs.fill_(num_active)

    init_tokens = device_buffer_tokens.clone()
    init_recency = device_buffer_recency.clone()

    model = _KernelModel(device_buffer_size, padded_buffer_size)
    for i in range(num_active):
        r = int(active[i].item())
        model.init_req(r, device_buffer_tokens[r].tolist(),
                       device_buffer_recency[r].tolist())

    tensors = (topk_indices, req_pool_indices, seq_lens, prefill_len,
               device_buffer_tokens, device_buffer_recency, top_k_device_slots,
               is_miss, num_real_reqs, global_recency_counter)
    for rd in range(rounds):
        rows = [rng.sample(range(-2, max_context_len + 2), top_k)
                for _ in range(num_active)]
        topk_indices[:num_active].copy_(torch.tensor(rows, dtype=torch.int32))

        C = int(global_recency_counter.item())
        _lru_call(tensors, block_dim, max_num_reqs, top_k,
                  device_buffer_size, padded_buffer_size)

        for b in range(num_active):
            req_idx = int(active[b].item())
            exp_slots, exp_miss = model.update(
                req_idx, rows[b],
                int(prefill_len[req_idx].item()),
                int(seq_lens[b].item()), C)
            assert top_k_device_slots[b].tolist() == exp_slots, \
                f"round {rd} bid {b} slots mismatch"
            assert is_miss[b].tolist() == exp_miss, \
                f"round {rd} bid {b} is_miss mismatch"

        # Inactive output rows must stay at their initial values.
        assert top_k_device_slots[num_active:].tolist() == \
            [[-1] * top_k] * (max_num_reqs - num_active), \
            f"round {rd} inactive slots were modified"
        assert int(is_miss[num_active:].sum().item()) == 0, \
            f"round {rd} inactive is_miss were modified"

        # Buffer rows: active rows match the model, all others match init.
        exp_tokens = init_tokens.clone()
        exp_recency = init_recency.clone()
        for b in range(num_active):
            req_idx = int(active[b].item())
            exp_tokens[req_idx] = torch.tensor(model.tokens[req_idx], dtype=torch.int32)
            exp_recency[req_idx] = torch.tensor(model.recency[req_idx], dtype=torch.int32)
        assert torch.equal(device_buffer_tokens, exp_tokens), \
            f"round {rd} buffer tokens mismatch"
        assert torch.equal(device_buffer_recency, exp_recency), \
            f"round {rd} buffer recency mismatch"

        global_recency_counter.add_(1)
    print("test_multi_block_partial_active passed")


def test_long_context_hit_no_overflow():
    """Regression test: hit detection must not overflow for large token positions.

    With int32 arithmetic (token - token_pos)^2 overflows once the difference
    exceeds 46340; e.g. (70000 - 12345)^2 wraps negative, which used to turn a
    real hit into a miss.
    """
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 1, 2, 4, 16
    max_context_len = 100000
    (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
        num_real_reqs, global_recency_counter,
    ) = _make_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)

    device_buffer_tokens[0, :4] = torch.tensor(
        [50000, 70000, 99999, 12345], dtype=torch.int32
    )
    topk_indices[0] = torch.tensor([70000, 88888], dtype=torch.int32)

    lru_update_npu(
        layer_id=0,
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_recency=device_buffer_recency,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        global_recency_counter=global_recency_counter,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        padded_buffer_size=padded_buffer_size,
        max_num_reqs=max_num_reqs,
    )

    # 70000 must hit slot 1; 88888 is a miss and evicts the coldest slot (0).
    assert top_k_device_slots[0].tolist() == [1, 0]
    assert is_miss[0].tolist() == [0, 1]
    assert device_buffer_tokens[0, 0].item() == 88888
    print("test_long_context_hit_no_overflow passed")


def test_graph_capture_replay():
    """The lru kernel must re-execute when captured in an NPU graph and replayed.

    The kernels are launched via raw ACLRT_LAUNCH_KERNEL, bypassing torch_npu's
    op dispatch. This test captures lru_update in a torch.npu.graph (same
    mechanism as NPUGraphRunner), mutates the input, replays, and checks that
    the outputs reflect the NEW input. If replay does not re-run the kernel,
    the raw launch is not capture-compatible.
    """
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 2, 4, 8, 16
    max_context_len = 16
    (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_recency, top_k_device_slots, is_miss,
        num_real_reqs, global_recency_counter,
    ) = _make_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)

    input_a = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32)
    input_b = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]], dtype=torch.int32)
    topk_indices.copy_(input_a)

    def _run():
        lru_update_npu(
            layer_id=0,
            topk_indices=topk_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            prefill_len=prefill_len,
            device_buffer_tokens=device_buffer_tokens,
            device_buffer_recency=device_buffer_recency,
            top_k_device_slots=top_k_device_slots,
            is_miss=is_miss,
            num_real_reqs=num_real_reqs,
            global_recency_counter=global_recency_counter,
            top_k=top_k,
            device_buffer_size=device_buffer_size,
            padded_buffer_size=padded_buffer_size,
            max_num_reqs=max_num_reqs,
        )

    # Warmup on a side stream (standard capture prerequisite).
    side = torch.npu.Stream()
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        _run()
    torch.npu.current_stream().wait_stream(side)

    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            _run()
    except RuntimeError as e:
        raise AssertionError(
            f"NPU graph capture failed for the lru_update kernel: {e}"
        )

    # Mutate the input in-place, poison the outputs, then replay.
    topk_indices.copy_(input_b)
    top_k_device_slots.fill_(-1)
    is_miss.fill_(0)
    graph.replay()
    torch.npu.synchronize()

    # All topk entries are hits (slot s holds token s), so slots must equal
    # the new input and there must be no misses.
    assert top_k_device_slots.tolist() == input_b.tolist(), (
        "graph replay did not re-execute the lru kernel with the mutated input "
        "(raw ACLRT_LAUNCH_KERNEL launches are not captured by torch.npu.graph): "
        f"got {top_k_device_slots.tolist()}, want {input_b.tolist()}"
    )
    assert int(is_miss.sum().item()) == 0
    print("test_graph_capture_replay passed")


def _visited_stride(padded_buffer_size: int) -> int:
    return (padded_buffer_size + 63) // 64 * 64


_T2S_CAP = 65536


def _sieve_ht_size(padded_buffer_size: int, top_k: int) -> int:
    ht_size = 1024
    while ht_size < 2 * (padded_buffer_size + top_k):
        ht_size <<= 1
    return ht_size


def _make_sieve_tensors(max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len):
    device = "npu:0"
    topk_indices = torch.full((max_num_reqs, top_k), -1, dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(max_num_reqs, dtype=torch.int64, device=device)
    seq_lens = torch.full((max_num_reqs,), max_context_len, dtype=torch.int32, device=device)
    prefill_len = torch.full((max_num_reqs,), max_context_len, dtype=torch.int32, device=device)
    device_buffer_tokens = torch.full(
        (max_num_reqs, padded_buffer_size), -1, dtype=torch.int32, device=device
    )
    device_buffer_visited = torch.zeros(
        (max_num_reqs, _visited_stride(padded_buffer_size)), dtype=torch.uint8, device=device
    )
    device_buffer_ht = torch.zeros(
        (1, max_num_reqs, _T2S_CAP),
        dtype=torch.int16,
        device=device,
    )
    sieve_hand = torch.zeros((max_num_reqs, 16), dtype=torch.int32, device=device)
    top_k_device_slots = torch.full((max_num_reqs, top_k), -1, dtype=torch.int32, device=device)
    is_miss = torch.zeros((max_num_reqs, top_k), dtype=torch.int8, device=device)
    num_real_reqs = torch.tensor([max_num_reqs], dtype=torch.int32, device=device)
    # Initialize slots: slot s holds token s.
    for s in range(device_buffer_size):
        device_buffer_tokens[:, s] = s
    sieve_ht_init_npu(
        device_buffer_tokens.unsqueeze(0),
        device_buffer_ht,
        prefill_len,
        0,
        max_num_reqs,
        device_buffer_size,
        padded_buffer_size,
        _sieve_ht_size(padded_buffer_size, top_k),
    )
    return (
        topk_indices, req_pool_indices, seq_lens, prefill_len,
        device_buffer_tokens, device_buffer_visited, sieve_hand,
        top_k_device_slots, is_miss, num_real_reqs,
        device_buffer_ht[0],
    )


def _sieve_call(tensors, block_dim, max_num_reqs, top_k, device_buffer_size, padded_buffer_size):
    (topk_indices, req_pool_indices, seq_lens, prefill_len,
     device_buffer_tokens, device_buffer_visited, sieve_hand,
     top_k_device_slots, is_miss, num_real_reqs, device_buffer_ht) = tensors
    sieve_update_npu(
        layer_id=0,
        topk_indices=topk_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        device_buffer_tokens=device_buffer_tokens,
        device_buffer_visited=device_buffer_visited,
        device_buffer_ht=device_buffer_ht,
        sieve_hand=sieve_hand,
        top_k_device_slots=top_k_device_slots,
        is_miss=is_miss,
        num_real_reqs=num_real_reqs,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        padded_buffer_size=padded_buffer_size,
        max_num_reqs=max_num_reqs,
        block_dim=block_dim,
    )


def _check_ht_consistency(device_buffer_ht, device_buffer_tokens, req_idx, tag,
                          prefill=None):
    """The persistent token-to-slot table must match the in-use buffer slots.

    Bitmap mode (prefill < 65536): t2s[token] = slot+1 for each in-use slot,
    0 elsewhere.  Hash mode (prefill >= 65536): every nonzero entry e means
    "token device_buffer_tokens[e-1] lives in slot e-1"; every in-use slot
    must appear exactly once.
    """
    tokens_row = device_buffer_tokens[req_idx].tolist()
    expected = {}
    for s, t in enumerate(tokens_row):
        if t != -1:
            expected[t] = s + 1

    ht_row = device_buffer_ht[req_idx]

    if prefill is not None and prefill < _T2S_CAP:
        # ===== Bitmap direct-index validation =====
        for t, sp1 in expected.items():
            actual = ht_row[t].item()
            assert actual == sp1, (
                f"{tag} req {req_idx}: t2s[{t}] = {actual}, expected {sp1}"
            )
        nonzero = (ht_row[:max(tokens_row) + 1] != 0).sum().item()
        assert nonzero == len(expected), (
            f"{tag} req {req_idx}: {nonzero} nonzero t2s entries, "
            f"expected {len(expected)}"
        )
    else:
        # ===== Hash table validation =====
        ht_list = ht_row.tolist()
        seen = 0
        for e in ht_list:
            if e == 0:
                continue
            seen += 1
            token = tokens_row[e - 1]
            assert expected.get(token) == e, (
                f"{tag} req {req_idx}: ht entry {e} points at token {token} "
                f"but that token lives in slot {expected.get(token)}"
            )
        assert seen == len(expected), (
            f"{tag} req {req_idx}: ht has {seen} entries, expected {len(expected)}"
        )


class _SieveModel:
    """CPU model that mirrors the hisparse_sieve_update kernel semantics.

    Hits set the visited byte; a miss sweeps the hand forward (clearing
    visited bytes) and evicts the first unvisited slot.  The hand persists
    across calls, like the GM sieve_hand tensor.
    """

    def __init__(self, device_buffer_size, padded_buffer_size):
        self.dbs = device_buffer_size
        self.pbs = padded_buffer_size
        self.tokens = {}
        self.visited = {}
        self.hand = {}

    def init_req(self, req_idx, tokens_row, visited_row, hand):
        self.tokens[req_idx] = list(tokens_row)
        self.visited[req_idx] = list(visited_row)
        self.hand[req_idx] = hand

    def update(self, req_idx, topk_row, prefill, seq_len):
        tokens = self.tokens[req_idx]
        visited = self.visited[req_idx]
        hand = self.hand[req_idx]
        slots, miss = [], []
        for token_pos in topk_row:
            if token_pos < 0 or token_pos >= seq_len:
                slots.append(-1)
                miss.append(0)
                continue
            if token_pos >= prefill:
                slots.append(self.pbs + (token_pos - prefill))
                miss.append(0)
                continue
            hit = -1
            for s in range(self.dbs):
                if tokens[s] == token_pos:
                    hit = s
                    visited[s] = 1
                    break
            if hit >= 0:
                slots.append(hit)
                miss.append(0)
            else:
                h = hand
                while visited[h] != 0:
                    visited[h] = 0
                    h += 1
                    if h == self.dbs:
                        h = 0
                slots.append(h)
                miss.append(1)
                tokens[h] = token_pos
                visited[h] = 1
                hand = h + 1
                if hand == self.dbs:
                    hand = 0
        self.hand[req_idx] = hand
        return slots, miss


def test_sieve_all_hits():
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 2, 4, 8, 16
    max_context_len = 16
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices = tensors[0]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]

    topk_indices[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    topk_indices[1] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)

    assert is_miss.sum().item() == 0
    assert top_k_device_slots[0].tolist() == [0, 1, 2, 3]
    assert top_k_device_slots[1].tolist() == [0, 1, 2, 3]
    # Hits must have raised the visited bytes.
    assert tensors[5][0][:4].tolist() == [1, 1, 1, 1]
    print("test_sieve_all_hits passed")


def test_sieve_misses_evict_in_hand_order():
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 1, 4, 4, 16
    max_context_len = 16
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices = tensors[0]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]

    # All tokens are misses; with hand=0 and all visited=0 they must evict
    # slots 0,1,2,3 in order.
    topk_indices[0] = torch.tensor([4, 5, 6, 7], dtype=torch.int32)

    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)

    assert is_miss.sum().item() == 4
    assert top_k_device_slots[0].tolist() == [0, 1, 2, 3]
    assert is_miss[0].tolist() == [1, 1, 1, 1]
    assert tensors[6][0][0].item() == 0  # hand wrapped to 0
    print("test_sieve_misses_evict_in_hand_order passed")


def test_sieve_second_chance():
    """Visited slots must be skipped (and cleared) by the eviction hand."""
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 1, 1, 4, 16
    max_context_len = 16
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices = tensors[0]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]
    sieve_hand = tensors[6]

    # Round 1: hit tokens 0 and 1 -> visited[0] = visited[1] = 1.
    topk_indices[0] = torch.tensor([0], dtype=torch.int32)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)
    topk_indices[0] = torch.tensor([1], dtype=torch.int32)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)

    # Round 2: token 4 misses.  The hand starts at 0, clears visited slots
    # 0 and 1, and evicts slot 2 (the first unvisited slot).
    topk_indices[0] = torch.tensor([4], dtype=torch.int32)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)
    assert top_k_device_slots[0].tolist() == [2]
    assert is_miss[0].tolist() == [1]
    assert sieve_hand[0][0].item() == 3
    # Visited bytes: slot 0,1 cleared by the sweep; slot 2 set by the insert.
    assert tensors[5][0][:4].tolist() == [0, 0, 1, 0]

    # Round 3: token 5 misses; hand is 3, evicts slot 3 immediately.
    topk_indices[0] = torch.tensor([5], dtype=torch.int32)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)
    assert top_k_device_slots[0].tolist() == [3]
    assert sieve_hand[0][0].item() == 0  # wrapped
    print("test_sieve_second_chance passed")


def test_sieve_decode_tokens_use_extension():
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 1, 4, 4, 16
    max_context_len = 16
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices, _, seq_lens, prefill_len = tensors[:4]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]

    prefill_len[0] = 4
    seq_lens[0] = 6
    # tokens 0,1 are prefill hits; tokens 4,5 are decode tokens.
    topk_indices[0] = torch.tensor([0, 1, 4, 5], dtype=torch.int32)

    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)

    assert is_miss.sum().item() == 0
    assert top_k_device_slots[0].tolist() == [0, 1, padded_buffer_size + 0, padded_buffer_size + 1]
    print("test_sieve_decode_tokens_use_extension passed")


def test_sieve_multi_block_stress():
    """Stress the SIEVE kernel against the exact CPU model (same coverage as
    test_multi_block_stress: hits, misses, second-chance sweeps, decode
    extension, invalid positions, multi-block request distribution)."""
    for top_k, max_num_reqs in ((4, 128), (8, 64)):
        for block_dim in (1, 2, 8):
            device_buffer_size, padded_buffer_size = 8, 16
            max_context_len = 32
            rounds = 12
            rng = random.Random((top_k << 16) ^ (max_num_reqs << 4) ^ block_dim)
            tensors = _make_sieve_tensors(
                max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
            (topk_indices, req_pool_indices, seq_lens, prefill_len,
             device_buffer_tokens, device_buffer_visited, sieve_hand,
             top_k_device_slots, is_miss, num_real_reqs, device_buffer_ht) = tensors
            prefill_len.fill_(max_context_len - 4)

            model = _SieveModel(device_buffer_size, padded_buffer_size)
            for r in range(max_num_reqs):
                model.init_req(r, device_buffer_tokens[r].tolist(),
                               device_buffer_visited[r][:padded_buffer_size].tolist(),
                               int(sieve_hand[r][0].item()))

            tag = f"top_k={top_k} reqs={max_num_reqs} block_dim={block_dim}"
            for rd in range(rounds):
                # Top-k indices are unique per request in production (torch.topk);
                # sample without replacement to match that invariant.
                rows = [rng.sample(range(-2, max_context_len + 2), top_k)
                        for _ in range(max_num_reqs)]
                topk_indices.copy_(torch.tensor(rows, dtype=torch.int32))

                _sieve_call(tensors, block_dim, max_num_reqs, top_k,
                            device_buffer_size, padded_buffer_size)

                for b in range(max_num_reqs):
                    req_idx = int(req_pool_indices[b].item())
                    exp_slots, exp_miss = model.update(
                        req_idx, rows[b],
                        int(prefill_len[req_idx].item()),
                        int(seq_lens[b].item()))
                    assert top_k_device_slots[b].tolist() == exp_slots, \
                        f"{tag} round {rd} bid {b} slots mismatch"
                    assert is_miss[b].tolist() == exp_miss, \
                        f"{tag} round {rd} bid {b} is_miss mismatch"
                    assert device_buffer_tokens[req_idx][:device_buffer_size].tolist() == \
                        model.tokens[req_idx][:device_buffer_size], \
                        f"{tag} round {rd} bid {b} buffer tokens mismatch"
                    assert device_buffer_visited[req_idx][:device_buffer_size].tolist() == \
                        model.visited[req_idx][:device_buffer_size], \
                        f"{tag} round {rd} bid {b} visited mismatch"
                    assert int(sieve_hand[req_idx][0].item()) == model.hand[req_idx], \
                        f"{tag} round {rd} bid {b} hand mismatch"
                    _check_ht_consistency(
                        device_buffer_ht, device_buffer_tokens, req_idx,
                        f"{tag} round {rd}",
                        prefill=int(prefill_len[req_idx].item()))
            print(f"  {tag} OK")
    print("test_sieve_multi_block_stress passed")


def test_sieve_ht_churn():
    """Hammer the persistent hash table with misses so every slot is evicted
    and re-inserted many times (backward-shift deletion under wraparound)."""
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 4, 8, 8, 16
    max_context_len = 4096
    rounds = 200
    rng = random.Random(20260728)
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    (topk_indices, req_pool_indices, _, prefill_len,
     device_buffer_tokens, device_buffer_visited, sieve_hand,
     top_k_device_slots, is_miss, num_real_reqs, device_buffer_ht) = tensors

    model = _SieveModel(device_buffer_size, padded_buffer_size)
    for r in range(max_num_reqs):
        model.init_req(r, device_buffer_tokens[r].tolist(),
                       device_buffer_visited[r][:padded_buffer_size].tolist(),
                       int(sieve_hand[r][0].item()))

    for rd in range(rounds):
        # All-miss rows: every query evicts (hand wraps ~top_k/dbs times per round).
        rows = [rng.sample(range(max_context_len), top_k)
                for _ in range(max_num_reqs)]
        topk_indices.copy_(torch.tensor(rows, dtype=torch.int32))

        _sieve_call(tensors, 0, max_num_reqs, top_k,
                    device_buffer_size, padded_buffer_size)

        for b in range(max_num_reqs):
            req_idx = int(req_pool_indices[b].item())
            exp_slots, exp_miss = model.update(
                req_idx, rows[b],
                int(prefill_len[req_idx].item()),
                max_context_len)
            assert top_k_device_slots[b].tolist() == exp_slots, \
                f"round {rd} bid {b} slots mismatch"
            assert is_miss[b].tolist() == exp_miss, \
                f"round {rd} bid {b} is_miss mismatch"
            _check_ht_consistency(
                device_buffer_ht, device_buffer_tokens, req_idx, f"round {rd}",
                prefill=int(prefill_len[req_idx].item()))
    print("test_sieve_ht_churn passed")


def test_sieve_ht_reinit():
    """sieve_ht_init must rebuild the table after buffer rows are repurposed."""
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 2, 4, 8, 16
    # Tokens 100..107 below must be valid positions, i.e. < seq_len (and
    # < prefill so they take the hash-table path rather than the decode
    # extension): seq_lens/prefill_len are both filled with max_context_len.
    max_context_len = 256
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices, _, seq_lens, prefill_len, device_buffer_tokens = tensors[:5]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]
    device_buffer_ht = tensors[10]

    # Repurpose request 1's row for a brand-new sequence without going through
    # the kernel (radix-cache reuse): stale ht entries must be dropped.
    device_buffer_tokens[1].fill_(-1)
    for s in range(device_buffer_size):
        device_buffer_tokens[1, s] = 100 + s
    sieve_ht_init_npu(
        device_buffer_tokens.unsqueeze(0),
        device_buffer_ht.unsqueeze(0),
        prefill_len,
        1,
        1,
        device_buffer_size,
        padded_buffer_size,
        _sieve_ht_size(padded_buffer_size, top_k),
    )
    _check_ht_consistency(device_buffer_ht, device_buffer_tokens, 1, "reinit",
                          prefill=int(prefill_len[1].item()))

    # The rebuilt table must serve hits for the new tokens.
    topk_indices[1] = torch.tensor([100, 103, 105, 107], dtype=torch.int32)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)
    assert top_k_device_slots[1].tolist() == [0, 3, 5, 7]
    assert is_miss[1].tolist() == [0, 0, 0, 0]
    print("test_sieve_ht_reinit passed")


def test_sieve_partial_active_fills_padding():
    """Rows beyond num_real (graph-replay padding) must be rewritten with
    invalid outputs (slot -1 / is_miss 0) without touching per-request state,
    even when req_pool_indices there holds stale valid-looking values."""
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 2, 4, 8, 16
    max_context_len = 16
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices = tensors[0]
    num_real_reqs = tensors[9]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]

    # Warm both requests so request 1 has live state to protect.
    topk_indices[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    topk_indices[1] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)
    state_snapshot = [t.clone() for t in tensors[4:7]] + [tensors[10].clone()]

    # Shrink the batch: row 1 becomes padding.  Poison its outputs first.
    num_real_reqs.fill_(1)
    top_k_device_slots[1].fill_(-1)
    is_miss[1].fill_(1)
    _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)

    # Padding row rewritten with invalid outputs.
    assert top_k_device_slots[1].tolist() == [-1, -1, -1, -1]
    assert is_miss[1].tolist() == [0, 0, 0, 0]
    # Request 1's eviction state must be untouched by the padding pass.
    for snap, cur in zip(state_snapshot, list(tensors[4:7]) + [tensors[10]]):
        assert torch.equal(snap, cur), "padding pass modified per-request state"
    num_real_reqs.fill_(max_num_reqs)
    print("test_sieve_partial_active_fills_padding passed")


def test_sieve_graph_capture_replay():
    """The sieve kernel must re-execute when captured in an NPU graph and
    replayed (same protocol as test_graph_capture_replay)."""
    max_num_reqs, top_k, device_buffer_size, padded_buffer_size = 2, 4, 8, 16
    max_context_len = 16
    tensors = _make_sieve_tensors(
        max_num_reqs, top_k, device_buffer_size, padded_buffer_size, max_context_len)
    topk_indices = tensors[0]
    is_miss = tensors[8]
    top_k_device_slots = tensors[7]

    input_a = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32)
    input_b = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]], dtype=torch.int32)
    topk_indices.copy_(input_a)

    def _run():
        _sieve_call(tensors, 0, max_num_reqs, top_k, device_buffer_size, padded_buffer_size)

    # Warmup on a side stream (standard capture prerequisite).
    side = torch.npu.Stream()
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        _run()
    torch.npu.current_stream().wait_stream(side)

    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            _run()
    except RuntimeError as e:
        raise AssertionError(
            f"NPU graph capture failed for the sieve_update kernel: {e}"
        )

    # Mutate the input in-place, poison the outputs, then replay.
    topk_indices.copy_(input_b)
    top_k_device_slots.fill_(-1)
    is_miss.fill_(0)
    graph.replay()
    torch.npu.synchronize()

    # All topk entries are hits (slot s holds token s), so slots must equal
    # the new input and there must be no misses.
    assert top_k_device_slots.tolist() == input_b.tolist(), (
        "graph replay did not re-execute the sieve kernel with the mutated input: "
        f"got {top_k_device_slots.tolist()}, want {input_b.tolist()}"
    )
    assert int(is_miss.sum().item()) == 0
    print("test_sieve_graph_capture_replay passed")


def main():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available, skipping NPU tests.")
        return

    test_all_hits()
    test_misses_evict_coldest()
    test_decode_tokens_use_extension()
    test_scatter_one_miss()
    test_scatter_layer_offset_and_row_stride()
    test_multi_block_stress()
    test_multi_block_partial_active()
    test_long_context_hit_no_overflow()
    test_graph_capture_replay()
    test_sieve_all_hits()
    test_sieve_misses_evict_in_hand_order()
    test_sieve_second_chance()
    test_sieve_decode_tokens_use_extension()
    test_sieve_multi_block_stress()
    test_sieve_ht_churn()
    test_sieve_ht_reinit()
    test_sieve_partial_active_fills_padding()
    test_sieve_graph_capture_replay()
    print("All NPU tests passed.")


if __name__ == "__main__":
    main()
