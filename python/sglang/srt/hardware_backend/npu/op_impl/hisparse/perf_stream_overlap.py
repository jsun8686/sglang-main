#!/usr/bin/env python3
"""
perf_stream_overlap.py — NPU multi-stream pipeline overlap benchmark.

Three experiments to determine whether 910C (ascend910_9391) supports
hardware-level cross-stream parallelism (DMA vs compute overlap):

  Exp 0: Pure torch — torch.matmul (AIC/Cube) vs H2D copy (MTE2/DMA).
         No kernel build required.  Answers: does CANN overlap streams at all?

  Exp 1: scatter_from_host vs scatter_from_host (MTE vs MTE).
         Two independent state sets on separate streams.  Tests DMA-DMA overlap.

  Exp 2: sieve_update vs scatter_from_host (AIV+MTE vs MTE).
         Production workload: plan kernel (sieve) overlaps with IO kernel (scatter).

For each experiment we measure:
  T_serial   — A then B on default stream
  T_parallel — A on stream_a, B on stream_b, concurrent
  overlap    = 1 - T_parallel / T_serial   (>0 means parallelism is effective)

Usage on NPU host:
  # Exp 0 only (no kernel build needed):
  python perf_stream_overlap.py --exp 0

  # All experiments (requires hisparse_lru.so built):
  python perf_stream_overlap.py --exp all
"""

import argparse
import ctypes
import ctypes.util
import statistics
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Constants — match GLM-5.2 / production HisSparse config
# ---------------------------------------------------------------------------
PADDED_BUFFER_SIZE = 4112
MAX_DECODE_LEN = 2048
MAX_CONTEXT_LEN = 8192

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
DTYPE = torch.float16
K_ROW_BYTES = KV_LORA_RANK * DTYPE.itemsize    # 1024
V_ROW_BYTES = QK_ROPE_HEAD_DIM * DTYPE.itemsize  # 128

NUM_REQS = 128
TOP_K = 2048
DEVICE_BUFFER_SIZE = 4096
MISS_RATIO = 0.1

WARMUP = 3
ITERS = 10

# ---------------------------------------------------------------------------
# Pinned host memory helpers (from test_hisparse_lru_npu.py)
# ---------------------------------------------------------------------------

def _acl_malloc_host(size: int) -> Tuple[int, int, int]:
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


# ---------------------------------------------------------------------------
# SIEVE helpers (from test_hisparse_lru_npu.py)
# ---------------------------------------------------------------------------

def _visited_stride(padded_buffer_size: int) -> int:
    return (padded_buffer_size + 63) // 64 * 64


_T2S_CAP = 65536


def _sieve_ht_size(padded_buffer_size: int, top_k: int) -> int:
    ht_size = 1024
    while ht_size < 2 * (padded_buffer_size + top_k):
        ht_size <<= 1
    return ht_size


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _median(values: List[float]) -> float:
    return statistics.median(values)


def _fmt_ms(values: List[float]) -> str:
    if not values:
        return "  N/A"
    m = _median(values)
    lo = min(values)
    return f"med {m:8.3f} ms (min {lo:8.3f})"


# ---------------------------------------------------------------------------
# State factory for HiSparse kernels (adapted from perf_hisparse_lru_npu.py)
# ---------------------------------------------------------------------------

def _make_state(device: str, num_reqs: int, top_k: int,
                device_buffer_size: int) -> Dict[str, torch.Tensor]:
    device_rows = num_reqs * (PADDED_BUFFER_SIZE + MAX_DECODE_LEN)
    topk_indices = torch.zeros((num_reqs, top_k), dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(num_reqs, dtype=torch.int64, device=device)
    seq_lens = torch.full((num_reqs,), MAX_CONTEXT_LEN, dtype=torch.int32, device=device)
    prefill_len = torch.full((num_reqs,), MAX_CONTEXT_LEN, dtype=torch.int32, device=device)
    device_buffer_tokens = torch.full(
        (num_reqs, PADDED_BUFFER_SIZE), -1, dtype=torch.int32, device=device
    )
    device_buffer_visited = torch.zeros(
        (num_reqs, _visited_stride(PADDED_BUFFER_SIZE)), dtype=torch.uint8, device=device
    )
    device_buffer_ht = torch.zeros(
        (1, num_reqs, _T2S_CAP), dtype=torch.int16, device=device
    )
    sieve_hand = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)
    top_k_device_slots = torch.full((num_reqs, top_k), -1, dtype=torch.int32, device=device)
    is_miss = torch.zeros((num_reqs, top_k), dtype=torch.int8, device=device)
    num_real_reqs = torch.tensor([num_reqs], dtype=torch.int32, device=device)

    req_to_host_pool = (
        torch.arange(MAX_CONTEXT_LEN, dtype=torch.int64, device=device)
        .unsqueeze(0).repeat(num_reqs, 1).contiguous()
    )
    req_to_device_buffer = (
        torch.arange(device_rows, dtype=torch.int64, device=device)
        .view(num_reqs, PADDED_BUFFER_SIZE + MAX_DECODE_LEN).contiguous()
    )
    device_k_buffer = torch.zeros(
        (device_rows, KV_LORA_RANK), dtype=DTYPE, device=device
    )
    device_v_buffer = torch.zeros(
        (device_rows, QK_ROPE_HEAD_DIM), dtype=DTYPE, device=device
    )

    return dict(
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
        req_to_host_pool=req_to_host_pool,
        req_to_device_buffer=req_to_device_buffer,
        device_k_buffer=device_k_buffer,
        device_v_buffer=device_v_buffer,
    )


def _gen_topk(device: str, num_reqs: int, top_k: int,
              device_buffer_size: int, miss_ratio: float) -> torch.Tensor:
    hits = torch.randint(0, device_buffer_size, (num_reqs, top_k))
    misses = torch.randint(device_buffer_size, MAX_CONTEXT_LEN, (num_reqs, top_k))
    mask = torch.rand(num_reqs, top_k) < miss_ratio
    return torch.where(mask, misses, hits).to(torch.int32).to(device)


def _init_tokens(state: Dict[str, torch.Tensor], device_buffer_size: int) -> None:
    state["device_buffer_tokens"].fill_(-1)
    state["device_buffer_tokens"][:, :device_buffer_size] = \
        torch.arange(device_buffer_size, dtype=torch.int32, device=state["device_buffer_tokens"].device)


def _pre_populate_for_scatter(state: Dict[str, torch.Tensor],
                              num_reqs: int, top_k: int,
                              device_buffer_size: int) -> None:
    """Set is_miss=1, top_k_device_slots to valid slots, and topk_indices to
    random token positions so scatter reads diverse host addresses."""
    state["is_miss"].fill_(1)
    slots = torch.arange(top_k, dtype=torch.int32, device=state["top_k_device_slots"].device)
    slots = slots % device_buffer_size
    state["top_k_device_slots"][:, :] = slots.unsqueeze(0).expand(num_reqs, top_k)
    state["topk_indices"].copy_(
        _gen_topk(state["topk_indices"].device, num_reqs, top_k,
                  device_buffer_size, miss_ratio=0.0)
    )


# ---------------------------------------------------------------------------
# Generic timing harness
# ---------------------------------------------------------------------------

def measure_serial(
    fn_a: Callable[[], None],
    fn_b: Callable[[], None],
    warmup: int = WARMUP,
    iters: int = ITERS,
) -> Dict[str, List[float]]:
    """Run A then B on default stream; measure T_total, T_a, T_b."""
    total_ms, a_ms, b_ms = [], [], []
    for i in range(warmup + iters):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev2 = torch.npu.Event(enable_timing=True)

        ev0.record()
        fn_a()
        ev1.record()
        fn_b()
        ev2.record()
        torch.npu.synchronize()

        if i >= warmup:
            total_ms.append(ev0.elapsed_time(ev2))
            a_ms.append(ev0.elapsed_time(ev1))
            b_ms.append(ev1.elapsed_time(ev2))
    return {"total": total_ms, "a": a_ms, "b": b_ms}


def measure_parallel(
    fn_a: Callable[[], None],
    fn_b: Callable[[], None],
    warmup: int = WARMUP,
    iters: int = ITERS,
) -> Dict[str, List[float]]:
    """Run A on stream_a, B on stream_b concurrently; measure overlap."""
    stream_a = torch.npu.Stream()
    stream_b = torch.npu.Stream()

    total_ms, a_ms, b_ms = [], [], []
    for i in range(warmup + iters):
        ev0 = torch.npu.Event(enable_timing=True)
        ev_a = torch.npu.Event(enable_timing=True)
        ev_b = torch.npu.Event(enable_timing=True)
        ev_end = torch.npu.Event(enable_timing=True)

        ev0.record()

        stream_a.wait_stream(torch.npu.current_stream())
        stream_b.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(stream_a):
            fn_a()
            ev_a.record(stream_a)

        with torch.npu.stream(stream_b):
            fn_b()
            ev_b.record(stream_b)

        torch.npu.current_stream().wait_stream(stream_a)
        torch.npu.current_stream().wait_stream(stream_b)
        ev_end.record()
        torch.npu.synchronize()

        if i >= warmup:
            total_ms.append(ev0.elapsed_time(ev_end))
            a_ms.append(ev0.elapsed_time(ev_a))
            b_ms.append(ev0.elapsed_time(ev_b))
    return {"total": total_ms, "a": a_ms, "b": b_ms}


def report(label: str, serial: Dict, parallel: Dict) -> None:
    s_total = _median(serial["total"])
    p_total = _median(parallel["total"])
    overlap = 1.0 - p_total / s_total if s_total > 0 else 0.0

    s_a = _median(serial["a"])
    s_b = _median(serial["b"])
    p_a = _median(parallel["a"])
    p_b = _median(parallel["b"])
    slow_a = p_a / s_a if s_a > 0 else 0.0
    slow_b = p_b / s_b if s_b > 0 else 0.0

    print(f"  [{label}]")
    print(f"    serial   total {_fmt_ms(serial['total'])}")
    print(f"              A   {_fmt_ms(serial['a'])}")
    print(f"              B   {_fmt_ms(serial['b'])}")
    print(f"    parallel total {_fmt_ms(parallel['total'])}")
    print(f"              A   {_fmt_ms(parallel['a'])}  (slowdown x{slow_a:.2f})")
    print(f"              B   {_fmt_ms(parallel['b'])}  (slowdown x{slow_b:.2f})")
    print(f"    >>> overlap = {overlap * 100:.1f}%   "
          f"(parallel {'FASTER' if overlap > 0.05 else 'slower or same'} than serial)")
    print()


# ===========================================================================
# Experiment 0: Pure torch — matmul (AIC) vs H2D copy (MTE2)
# ===========================================================================

def run_experiment_0(device: str) -> None:
    print("=" * 72)
    print("Experiment 0: torch.matmul (AIC/Cube) vs H2D copy (MTE2/DMA)")
    print("  Purpose: Does CANN support cross-stream parallelism at all?")
    print("=" * 72)

    # Workload A: matmul on device (Cube pipeline)
    MATMUL_N = 8192
    mat_a = torch.randn(MATMUL_N, MATMUL_N, dtype=torch.float16, device=device)
    mat_b = torch.randn(MATMUL_N, MATMUL_N, dtype=torch.float16, device=device)
    mat_c = torch.empty_like(mat_a)

    MATMUL_REPEAT = 50

    def fn_matmul():
        for _ in range(MATMUL_REPEAT):
            torch.matmul(mat_a, mat_b, out=mat_c)

    # Workload B: H2D copy (DMA/MTE2 pipeline)
    COPY_BYTES = 256 * 1024 * 1024  # 256 MiB per copy
    COPY_ELEM = COPY_BYTES // 4     # float32
    COPY_REPEAT = 50

    host_buf = torch.randn(COPY_ELEM, dtype=torch.float32, pin_memory=True)
    dev_buf = torch.empty(COPY_ELEM, dtype=torch.float32, device=device)

    def fn_h2d():
        for _ in range(COPY_REPEAT):
            dev_buf.copy_(host_buf, non_blocking=True)

    print(f"\n  Workload A: {MATMUL_REPEAT}x matmul({MATMUL_N}x{MATMUL_N} fp16)")
    print(f"  Workload B: {COPY_REPEAT}x H2D copy {COPY_BYTES / 1024 / 1024:.0f} MiB\n")

    serial = measure_serial(fn_matmul, fn_h2d)
    parallel = measure_parallel(fn_matmul, fn_h2d)
    report("matmul vs H2D", serial, parallel)

    del mat_a, mat_b, mat_c, host_buf, dev_buf
    torch.npu.empty_cache()


# ===========================================================================
# Experiment 1: scatter_from_host vs scatter_from_host (MTE vs MTE)
# ===========================================================================

def run_experiment_1(
    device: str,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    host_kv_dev_ptr: int,
    bd_configs: List[Tuple[int, int]],
) -> None:
    print("=" * 72)
    print("Experiment 1: scatter_from_host vs scatter_from_host (MTE vs MTE)")
    print("  Purpose: Can two DMA-heavy kernels overlap on separate streams?")
    print("=" * 72)

    from sglang.srt.hardware_backend.npu.op_impl.hisparse import scatter_from_host_npu

    for bd_a, bd_b in bd_configs:
        _pre_populate_for_scatter(state_a, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)
        _pre_populate_for_scatter(state_b, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

        def fn_scatter_a(sa=state_a):
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=sa["topk_indices"],
                top_k_device_slots=sa["top_k_device_slots"],
                is_miss=sa["is_miss"],
                req_pool_indices=sa["req_pool_indices"],
                req_to_host_pool=sa["req_to_host_pool"],
                req_to_device_buffer=sa["req_to_device_buffer"],
                device_k_buffer=sa["device_k_buffer"],
                device_v_buffer=sa["device_v_buffer"],
                layer_id=0,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=bd_a,
            )

        def fn_scatter_b(sb=state_b):
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=sb["topk_indices"],
                top_k_device_slots=sb["top_k_device_slots"],
                is_miss=sb["is_miss"],
                req_pool_indices=sb["req_pool_indices"],
                req_to_host_pool=sb["req_to_host_pool"],
                req_to_device_buffer=sb["req_to_device_buffer"],
                device_k_buffer=sb["device_k_buffer"],
                device_v_buffer=sb["device_v_buffer"],
                layer_id=1,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=bd_b,
            )

        label = f"scatter(block_dim={bd_a}) vs scatter(block_dim={bd_b})"
        print(f"\n  {label}")
        serial = measure_serial(fn_scatter_a, fn_scatter_b)
        parallel = measure_parallel(fn_scatter_a, fn_scatter_b)
        report(label, serial, parallel)


# ===========================================================================
# Experiment 2: sieve_update vs scatter_from_host (AIV+MTE vs MTE)
# ===========================================================================

def run_experiment_2(
    device: str,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    host_kv_dev_ptr: int,
    bd_configs: List[Tuple[int, int]],
) -> None:
    print("=" * 72)
    print("Experiment 2: sieve_update vs scatter_from_host (AIV+MTE vs MTE)")
    print("  Purpose: Production plan-then-IO workload overlap.")
    print("=" * 72)

    from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
        sieve_update_npu,
        scatter_from_host_npu,
    )

    # Pre-populate topk for sieve (state_a)
    state_a["topk_indices"].copy_(
        _gen_topk(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE, MISS_RATIO)
    )
    _init_tokens(state_a, DEVICE_BUFFER_SIZE)
    # Pre-populate scatter inputs (state_b)
    _pre_populate_for_scatter(state_b, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

    for bd_a, bd_b in bd_configs:
        def fn_sieve(sa=state_a):
            sieve_update_npu(
                layer_id=0,
                topk_indices=sa["topk_indices"],
                req_pool_indices=sa["req_pool_indices"],
                seq_lens=sa["seq_lens"],
                prefill_len=sa["prefill_len"],
                device_buffer_tokens=sa["device_buffer_tokens"],
                device_buffer_visited=sa["device_buffer_visited"],
                device_buffer_ht=sa["device_buffer_ht"][0],
                sieve_hand=sa["sieve_hand"],
                top_k_device_slots=sa["top_k_device_slots"],
                is_miss=sa["is_miss"],
                num_real_reqs=sa["num_real_reqs"],
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                block_dim=bd_a,
            )

        def fn_scatter(sb=state_b):
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=sb["topk_indices"],
                top_k_device_slots=sb["top_k_device_slots"],
                is_miss=sb["is_miss"],
                req_pool_indices=sb["req_pool_indices"],
                req_to_host_pool=sb["req_to_host_pool"],
                req_to_device_buffer=sb["req_to_device_buffer"],
                device_k_buffer=sb["device_k_buffer"],
                device_v_buffer=sb["device_v_buffer"],
                layer_id=1,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=bd_b,
            )

        label = f"sieve(block_dim={bd_a}) vs scatter(block_dim={bd_b})"
        print(f"\n  {label}")
        serial = measure_serial(fn_sieve, fn_scatter)
        parallel = measure_parallel(fn_sieve, fn_scatter)
        report(label, serial, parallel)


# ===========================================================================
# Experiment 3: Core-partition sweep (bd_a + bd_b = 48)
# ===========================================================================

# Production core partitions to sweep.  Total = 48 (full chip).
# sieve is compute-light → give it few cores; scatter is DMA-heavy → give it
# the rest.  The sweep finds the sweet spot.
PARTITION_CONFIGS = [
    (1, 47),
    (2, 46),
    (4, 44),
    (8, 40),
    (16, 32),
    (24, 24),
    (32, 16),
    (40, 8),
]


def run_experiment_3(
    device: str,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    host_kv_dev_ptr: int,
) -> None:
    print("=" * 72)
    print("Experiment 3: Core-partition sweep (bd_a + bd_b = 48)")
    print("  Purpose: Find optimal core split for plan-then-IO overlap.")
    print("  Tests both scatter-vs-scatter and sieve-vs-scatter.")
    print("=" * 72)

    from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
        sieve_update_npu,
        scatter_from_host_npu,
    )

    # ---- Part A: scatter vs scatter (MTE vs MTE) partition sweep ----
    print("\n--- Part A: scatter vs scatter (bd_a + bd_b = 48) ---\n")
    _pre_populate_for_scatter(state_a, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)
    _pre_populate_for_scatter(state_b, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

    results_svs = []
    for bd_a, bd_b in PARTITION_CONFIGS:
        def fn_scatter_a(sa=state_a, _bd=bd_a):
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=sa["topk_indices"],
                top_k_device_slots=sa["top_k_device_slots"],
                is_miss=sa["is_miss"],
                req_pool_indices=sa["req_pool_indices"],
                req_to_host_pool=sa["req_to_host_pool"],
                req_to_device_buffer=sa["req_to_device_buffer"],
                device_k_buffer=sa["device_k_buffer"],
                device_v_buffer=sa["device_v_buffer"],
                layer_id=0,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=_bd,
            )

        def fn_scatter_b(sb=state_b, _bd=bd_b):
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=sb["topk_indices"],
                top_k_device_slots=sb["top_k_device_slots"],
                is_miss=sb["is_miss"],
                req_pool_indices=sb["req_pool_indices"],
                req_to_host_pool=sb["req_to_host_pool"],
                req_to_device_buffer=sb["req_to_device_buffer"],
                device_k_buffer=sb["device_k_buffer"],
                device_v_buffer=sb["device_v_buffer"],
                layer_id=1,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=_bd,
            )

        label = f"scatter({bd_a:2d}) vs scatter({bd_b:2d})"
        serial = measure_serial(fn_scatter_a, fn_scatter_b)
        parallel = measure_parallel(fn_scatter_a, fn_scatter_b)
        report(label, serial, parallel)

        s_total = _median(serial["total"])
        p_total = _median(parallel["total"])
        overlap = 1.0 - p_total / s_total if s_total > 0 else 0.0
        results_svs.append((bd_a, bd_b, s_total, p_total, overlap,
                            _median(serial["a"]), _median(serial["b"]),
                            _median(parallel["a"]), _median(parallel["b"])))

    _print_summary_table("scatter vs scatter", results_svs)

    # ---- Part B: sieve vs scatter partition sweep ----
    print("\n--- Part B: sieve vs scatter (bd_a + bd_b = 48) ---\n")
    # Re-init state_a for sieve
    state_a["topk_indices"].copy_(
        _gen_topk(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE, MISS_RATIO)
    )
    _init_tokens(state_a, DEVICE_BUFFER_SIZE)
    # Re-init state_b for scatter
    _pre_populate_for_scatter(state_b, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

    results_svs2 = []
    for bd_a, bd_b in PARTITION_CONFIGS:
        def fn_sieve(sa=state_a, _bd=bd_a):
            sieve_update_npu(
                layer_id=0,
                topk_indices=sa["topk_indices"],
                req_pool_indices=sa["req_pool_indices"],
                seq_lens=sa["seq_lens"],
                prefill_len=sa["prefill_len"],
                device_buffer_tokens=sa["device_buffer_tokens"],
                device_buffer_visited=sa["device_buffer_visited"],
                device_buffer_ht=sa["device_buffer_ht"][0],
                sieve_hand=sa["sieve_hand"],
                top_k_device_slots=sa["top_k_device_slots"],
                is_miss=sa["is_miss"],
                num_real_reqs=sa["num_real_reqs"],
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                block_dim=_bd,
            )

        def fn_scatter(sb=state_b, _bd=bd_b):
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=sb["topk_indices"],
                top_k_device_slots=sb["top_k_device_slots"],
                is_miss=sb["is_miss"],
                req_pool_indices=sb["req_pool_indices"],
                req_to_host_pool=sb["req_to_host_pool"],
                req_to_device_buffer=sb["req_to_device_buffer"],
                device_k_buffer=sb["device_k_buffer"],
                device_v_buffer=sb["device_v_buffer"],
                layer_id=1,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=_bd,
            )

        label = f"sieve({bd_a:2d}) vs scatter({bd_b:2d})"
        serial = measure_serial(fn_sieve, fn_scatter)
        parallel = measure_parallel(fn_sieve, fn_scatter)
        report(label, serial, parallel)

        s_total = _median(serial["total"])
        p_total = _median(parallel["total"])
        overlap = 1.0 - p_total / s_total if s_total > 0 else 0.0
        results_svs2.append((bd_a, bd_b, s_total, p_total, overlap,
                             _median(serial["a"]), _median(serial["b"]),
                             _median(parallel["a"]), _median(parallel["b"])))

    _print_summary_table("sieve vs scatter", results_svs2)


def _print_summary_table(title: str, results: List[Tuple]) -> None:
    """Print a compact summary table of partition sweep results."""
    print(f"\n  ┌─────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ Summary: {title:<66}│")
    print(f"  ├──────┬──────┬──────────┬──────────┬──────────┬──────────────────────┤")
    print(f"  │ bd_a │ bd_b │ serial   │ parallel │ overlap  │ A_par   B_par        │")
    print(f"  │      │      │ (ms)     │ (ms)     │ (%)      │ (ms)    (ms)        │")
    print(f"  ├──────┼──────┼──────────┼──────────┼──────────┼──────────────────────┤")
    for bd_a, bd_b, s, p, ov, sa, sb, pa, pb in results:
        tag = " <<<" if ov == max(r[4] for r in results) else ""
        print(f"  │ {bd_a:4d} │ {bd_b:4d} │ {s:8.3f}  │ {p:8.3f}  │ {ov*100:6.1f}%  │ "
              f"{pa:7.3f}  {pb:7.3f}     │{tag}")
    print(f"  └──────┴──────┴──────────┴──────────┴──────────┴──────────────────────┘")


# ===========================================================================
# Main
# ===========================================================================

def main():
    global WARMUP, ITERS

    parser = argparse.ArgumentParser(
        description="NPU multi-stream pipeline overlap benchmark"
    )
    parser.add_argument(
        "--exp", type=str, default="0",
        choices=["0", "1", "2", "3", "all"],
        help="Which experiment to run (default: 0)",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP,
        help=f"Warmup iterations (default: {WARMUP})",
    )
    parser.add_argument(
        "--iters", type=int, default=ITERS,
        help=f"Timed iterations (default: {ITERS})",
    )
    args = parser.parse_args()

    WARMUP = args.warmup
    ITERS = args.iters

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available — this benchmark requires an NPU device.")
        sys.exit(1)

    torch.npu.set_device(0)
    device = "npu:0"

    print(f"Device: {device}")
    print(f"SoC: ascend910_9391, CANN 9.0.0")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")
    print(f"Config: reqs={NUM_REQS}, top_k={TOP_K}, "
          f"buf_size={DEVICE_BUFFER_SIZE}, miss_ratio={MISS_RATIO}")
    print()

    run_exp0 = args.exp in ("0", "all")
    run_exp1 = args.exp in ("1", "all")
    run_exp2 = args.exp in ("2", "all")
    run_exp3 = args.exp in ("3", "all")

    # --- Experiment 0 ---
    if run_exp0:
        run_experiment_0(device)

    # --- Experiments 1, 2, 3 (require kernel build) ---
    if run_exp1 or run_exp2 or run_exp3:
        from sglang.srt.hardware_backend.npu.op_impl import hisparse as hisparse_pkg
        if not hisparse_pkg._HAS_KERNEL:
            print("\nERROR: hisparse_lru native extension not built.")
            print("Build it on the NPU host with:")
            print("  cd python/sglang/srt/hardware_backend/npu/op_impl/hisparse")
            print("  python setup.py build_ext --inplace")
            sys.exit(1)

        print(f"\nAuto block_dim = {hisparse_pkg._resolve_block_dim(0)}")

        # Allocate pinned host cache — must cover all layer_ids used by
        # scatter (layer 0 for state_a, layer 1 for state_b).
        # Kernel addresses: host_kv + (layer_id * host_entries + offset) * row_bytes
        kv_row_bytes = K_ROW_BYTES + V_ROW_BYTES
        HOST_LAYERS = 2  # layer_id 0 and 1
        host_kv_ptr, host_kv_dev_ptr, host_kv_size = \
            _acl_malloc_host(HOST_LAYERS * MAX_CONTEXT_LEN * kv_row_bytes)

        try:
            # Two independent state sets
            state_a = _make_state(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)
            state_b = _make_state(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

            if run_exp1:
                run_experiment_1(
                    device, state_a, state_b, host_kv_dev_ptr,
                    bd_configs=[(1, 1), (48, 48)],
                )

            if run_exp2:
                run_experiment_2(
                    device, state_a, state_b, host_kv_dev_ptr,
                    bd_configs=[(1, 1), (8, 4)],
                )

            if run_exp3:
                run_experiment_3(
                    device, state_a, state_b, host_kv_dev_ptr,
                )

            del state_a, state_b
            torch.npu.empty_cache()
        finally:
            _acl_free_host(host_kv_ptr, host_kv_dev_ptr, host_kv_size)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
