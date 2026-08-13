#!/usr/bin/env python3
"""
perf_stream_overlap.py — NPU multi-stream pipeline overlap benchmark.

Experiments to determine whether 910C (ascend910_9391) supports
hardware-level cross-stream parallelism (DMA vs compute overlap):

  Exp 0: Pure torch — torch.matmul (AIC/Cube) vs H2D copy (MTE2/DMA).
         No kernel build required.  Answers: does CANN overlap streams at all?

  Exp 1: scatter_from_host vs scatter_from_host (MTE vs MTE).
         Two independent state sets on separate streams.  Tests DMA-DMA overlap.

  Exp 2: sieve_update vs scatter_from_host (AIV+MTE vs MTE).
         Production workload: plan kernel (sieve) overlaps with IO kernel (scatter).

  Exp 3: Core-partition sweep (bd_a + bd_b = 48).
         Finds optimal core split for plan-then-IO overlap.

  Exp 4: Production pipeline simulation.
         Main stream: sieve+scatter(anchor) → attention (proxy) on main.
         Side stream: N× serial scatter (shared layers).
         Tests whether side-stream scatters are fully covered by main-stream
         attention computation.  Three attention proxy modes are compared:
           - matmul:        Cube only (AIC) — no DMA contention baseline
           - matmul_softmax: Cube + Vector — adds softmax pipeline
           - sdpa:          Cube + Vector + MTE — real attention DMA contention

  Exp 4b: Side-stream block_dim sweep.
          Fixed attention: sdpa(kv=2048, x4) = GLM-5.2 decode scenario.
          Phase 1 (anchor) always block_dim=48; sweep side-stream scatter
          block_dim to find optimal AIV core partition (910B = 24 AIC + 48 AIV).

Usage on NPU host:
  # Exp 0 only (no kernel build needed):
  python perf_stream_overlap.py --exp 0

  # Exp 4 production pipeline (default 3 shared layers):
  python perf_stream_overlap.py --exp 4

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
# Experiment 4: Production pipeline simulation
# ===========================================================================
#
# Three attention proxy modes to quantify DMA bandwidth contention:
#
#   "matmul":  pure torch.matmul — Cube only (AIC).
#              Baseline: no DMA contention from attention on scatter.
#
#   "matmul_softmax": manual QK^T → softmax → PV — Cube + Vector.
#              Adds Vector (softmax) but still light MTE.
#
#   "sdpa":    scaled_dot_product_attention — full Cube + Vector + MTE.
#              NPU maps this to fused flash attention, which reads KV from GM
#              (MTE2 heavy) → directly competes with scatter for DMA bandwidth.
#
# Comparing matmul vs sdpa overlap reveals how much DMA contention costs.

# Matmul sizes to sweep: (matrix_N, repeats)
MATMUL_CONFIGS = [
    (2048,  20),
    (4096,  20),
    (4096,  50),
    (8192,  50),
    (8192, 100),
]

# SDPA configs: (kv_len, repeats).  Decode pattern: Q=[1, heads, 128, d], K=V=[1, heads, kv_len, d]
# head_dim=128, num_heads=16, batch=128 (=NUM_REQS).
SDPA_CONFIGS = [
    (256,  4),
    (512,  4),
    (1024, 4),
    (2048, 4),
    (4096, 4),
]


def _calibrate_workload(
    device: str,
    fn_factory: Callable,
    configs: list,
) -> List[Tuple]:
    """Generic calibration: for each config, measure solo workload time."""
    results = []
    for cfg in configs:
        workload, cleanup = fn_factory(device, cfg)
        times = []
        for i in range(WARMUP + ITERS):
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            ev0.record()
            workload()
            ev1.record()
            torch.npu.synchronize()
            if i >= WARMUP:
                times.append(ev0.elapsed_time(ev1))
        ms = _median(times)
        results.append((*cfg, ms))
        cleanup()
    return results


def _make_matmul_factory():
    """Factory that creates a matmul workload from (N, repeats) config."""
    def factory(device, cfg):
        n, reps = cfg
        st = dict(
            a=torch.randn(n, n, dtype=torch.float16, device=device),
            b=torch.randn(n, n, dtype=torch.float16, device=device),
            c=torch.empty(n, n, dtype=torch.float16, device=device),
        )
        def workload():
            for _ in range(reps):
                torch.matmul(st["a"], st["b"], out=st["c"])
        def cleanup():
            st.clear()
            torch.npu.empty_cache()
        return workload, cleanup
    return factory


def _make_sdpa_factory():
    """Factory for scaled_dot_product_attention workload from (kv_len, reps) config."""
    def factory(device, cfg):
        kv_len, reps = cfg
        num_heads = 16
        head_dim = 128
        batch = NUM_REQS
        # Decode pattern: 1 query token, kv_len KV tokens
        # sdpa shape: [batch, heads, seq, head_dim]
        st = dict(
            q=torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device=device),
            k=torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float16, device=device),
            v=torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float16, device=device),
        )
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        def workload():
            for _ in range(reps):
                sdpa(st["q"], st["k"], st["v"])
        def cleanup():
            st.clear()
            torch.npu.empty_cache()
        return workload, cleanup
    return factory


def _make_matmul_softmax_factory():
    """Factory for manual matmul(Q,K^T)→softmax→matmul(P,V) workload."""
    def factory(device, cfg):
        kv_len, reps = cfg
        num_heads = 16
        head_dim = 128
        batch = NUM_REQS
        st = dict(
            q=torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device=device),
            k=torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float16, device=device),
            v=torch.randn(batch, num_heads, kv_len, head_dim, dtype=torch.float16, device=device),
        )
        scale = head_dim ** -0.5
        def workload():
            for _ in range(reps):
                scores = torch.matmul(st["q"], st["k"].transpose(-2, -1)) * scale
                attn = torch.softmax(scores, dim=-1)
                torch.matmul(attn, st["v"])
        def cleanup():
            st.clear()
            torch.npu.empty_cache()
        return workload, cleanup
    return factory


def run_experiment_4(
    device: str,
    state_anchor: Dict[str, torch.Tensor],
    shared_states: List[Dict[str, torch.Tensor]],
    host_kv_dev_ptr: int,
    num_shared: int,
    block_dim: int = 48,
) -> None:
    """
    Production pipeline simulation with realistic attention proxies.

      Phase 1 (serial on main):
        Main:  |== sieve(anchor) ==|== scatter(anchor) ==|
        Side:  (idle)

      Phase 2 (parallel, fork after Phase 1):
        Main:  |== attention ×N (proxy) ===================|
        Side:  |== scatter(sh1) ==|== scatter(sh2) ==|== ... ==|

    Runs three attention proxy modes and compares their overlap to quantify
    DMA bandwidth contention between attention and scatter.
    """
    print("=" * 72)
    print(f"Experiment 4: Production pipeline simulation")
    print(f"  1 anchor + {num_shared} shared layers (index_topk_freq={num_shared + 1})")
    print(f"  Phase 1: sieve+scatter(anchor) on main stream")
    print(f"  Phase 2: attention (proxy) on main  ||  {num_shared}× scatter on side")
    print(f"  Modes: matmul(Cube) | matmul+softmax(Cube+Vec) | sdpa(Cube+Vec+MTE)")
    print("=" * 72)

    from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
        sieve_update_npu,
        scatter_from_host_npu,
    )

    # --- Prepare anchor state for sieve ---
    state_anchor["topk_indices"].copy_(
        _gen_topk(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE, MISS_RATIO)
    )
    _init_tokens(state_anchor, DEVICE_BUFFER_SIZE)

    # --- Prepare shared states for scatter ---
    for i, ss in enumerate(shared_states[:num_shared]):
        _pre_populate_for_scatter(ss, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

    # ==================================================================
    # Helper: run anchor sieve+scatter on a given stream context
    # ==================================================================
    def _run_anchor_swap_in():
        sieve_update_npu(
            layer_id=0,
            topk_indices=state_anchor["topk_indices"],
            req_pool_indices=state_anchor["req_pool_indices"],
            seq_lens=state_anchor["seq_lens"],
            prefill_len=state_anchor["prefill_len"],
            device_buffer_tokens=state_anchor["device_buffer_tokens"],
            device_buffer_visited=state_anchor["device_buffer_visited"],
            device_buffer_ht=state_anchor["device_buffer_ht"][0],
            sieve_hand=state_anchor["sieve_hand"],
            top_k_device_slots=state_anchor["top_k_device_slots"],
            is_miss=state_anchor["is_miss"],
            num_real_reqs=state_anchor["num_real_reqs"],
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            padded_buffer_size=PADDED_BUFFER_SIZE,
            max_num_reqs=NUM_REQS,
            block_dim=block_dim,
        )
        scatter_from_host_npu(
            host_kv_cache_ptr=host_kv_dev_ptr,
            topk_indices=state_anchor["topk_indices"],
            top_k_device_slots=state_anchor["top_k_device_slots"],
            is_miss=state_anchor["is_miss"],
            req_pool_indices=state_anchor["req_pool_indices"],
            req_to_host_pool=state_anchor["req_to_host_pool"],
            req_to_device_buffer=state_anchor["req_to_device_buffer"],
            device_k_buffer=state_anchor["device_k_buffer"],
            device_v_buffer=state_anchor["device_v_buffer"],
            layer_id=0,
            host_entries=MAX_CONTEXT_LEN,
            k_row_bytes=K_ROW_BYTES,
            v_row_bytes=V_ROW_BYTES,
            max_context_len=MAX_CONTEXT_LEN,
            device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
            padded_buffer_size=PADDED_BUFFER_SIZE,
            max_num_reqs=NUM_REQS,
            top_k=TOP_K,
            block_dim=block_dim,
        )

    def _run_side_scatters():
        for idx in range(num_shared):
            ss = shared_states[idx]
            scatter_from_host_npu(
                host_kv_cache_ptr=host_kv_dev_ptr,
                topk_indices=ss["topk_indices"],
                top_k_device_slots=ss["top_k_device_slots"],
                is_miss=ss["is_miss"],
                req_pool_indices=ss["req_pool_indices"],
                req_to_host_pool=ss["req_to_host_pool"],
                req_to_device_buffer=ss["req_to_device_buffer"],
                device_k_buffer=ss["device_k_buffer"],
                device_v_buffer=ss["device_v_buffer"],
                layer_id=1 + idx,
                host_entries=MAX_CONTEXT_LEN,
                k_row_bytes=K_ROW_BYTES,
                v_row_bytes=V_ROW_BYTES,
                max_context_len=MAX_CONTEXT_LEN,
                device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                padded_buffer_size=PADDED_BUFFER_SIZE,
                max_num_reqs=NUM_REQS,
                top_k=TOP_K,
                block_dim=block_dim,
            )

    # ==================================================================
    # Step 1: Phase 1 + side timing
    # ==================================================================
    print("\n  --- Step 1: Phase timings (serial) ---\n")

    # T_phase1
    times_p1 = []
    for i in range(WARMUP + ITERS):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        _run_anchor_swap_in()
        ev1.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times_p1.append(ev0.elapsed_time(ev1))
    T_phase1 = _median(times_p1)

    # T_side
    times_side = []
    for i in range(WARMUP + ITERS):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        _run_side_scatters()
        ev1.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times_side.append(ev0.elapsed_time(ev1))
    T_side_solo = _median(times_side)

    print(f"    T_phase1 (sieve+scatter) = {T_phase1:8.3f} ms")
    print(f"    T_side ({num_shared}× scatter)   = {T_side_solo:8.3f} ms")

    # ==================================================================
    # Step 2: Calibrate all three attention proxy modes
    # ==================================================================
    print("\n  --- Step 2: Calibrate attention proxies ---\n")

    modes = [
        ("matmul",        MATMUL_CONFIGS,  _make_matmul_factory()),
        ("matmul_softmax", SDPA_CONFIGS,    _make_matmul_softmax_factory()),
        ("sdpa",          SDPA_CONFIGS,    _make_sdpa_factory()),
    ]

    calibrated = {}
    for mode_name, configs, factory in modes:
        print(f"    [{mode_name}]")
        cal = _calibrate_workload(device, factory, configs)
        for cfg_val in cal:
            if mode_name == "matmul":
                print(f"      matmul({cfg_val[0]}x{cfg_val[0]}) x{cfg_val[1]:3d}  = {cfg_val[2]:8.3f} ms")
            else:
                print(f"      kv_len={cfg_val[0]:4d} x{cfg_val[1]}  = {cfg_val[2]:8.3f} ms")
        calibrated[mode_name] = cal
        print()

    # ==================================================================
    # Step 3: Production timeline per mode
    # ==================================================================
    print("  --- Step 3: Production timeline (Phase 1 → fork → Phase 2) ---\n")

    side_stream = torch.npu.Stream()
    all_results = {}

    for mode_name, configs, factory in modes:
        print(f"  ====== Mode: {mode_name} ======\n")
        mode_results = []

        for cal_entry in calibrated[mode_name]:
            cfg = cal_entry[:-1]
            T_attn_solo = cal_entry[-1]
            workload, cleanup = factory(device, cfg)

            # Serial baseline
            def run_serial(_workload=workload):
                _run_anchor_swap_in()
                _workload()
                _run_side_scatters()

            # Production: Phase 1 → fork → Phase 2
            def run_production(_workload=workload):
                _run_anchor_swap_in()
                side_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(side_stream):
                    _run_side_scatters()
                _workload()
                torch.npu.current_stream().wait_stream(side_stream)

            times_serial = []
            for i in range(WARMUP + ITERS):
                ev0 = torch.npu.Event(enable_timing=True)
                ev1 = torch.npu.Event(enable_timing=True)
                ev0.record()
                run_serial()
                ev1.record()
                torch.npu.synchronize()
                if i >= WARMUP:
                    times_serial.append(ev0.elapsed_time(ev1))

            times_prod = []
            for i in range(WARMUP + ITERS):
                ev0 = torch.npu.Event(enable_timing=True)
                ev1 = torch.npu.Event(enable_timing=True)
                ev0.record()
                run_production()
                ev1.record()
                torch.npu.synchronize()
                if i >= WARMUP:
                    times_prod.append(ev0.elapsed_time(ev1))

            T_serial = _median(times_serial)
            T_prod = _median(times_prod)
            speedup = T_serial / T_prod if T_prod > 0 else 0

            if T_side_solo > 0:
                coverage = max(0.0, min(1.0, T_attn_solo / T_side_solo))
            else:
                coverage = 1.0

            # DMA contention factor: how much did T_prod exceed the ideal?
            # Ideal = T_phase1 + max(T_attn, T_side)
            T_ideal = T_phase1 + max(T_attn_solo, T_side_solo)
            contention = (T_prod - T_ideal) / T_ideal * 100 if T_ideal > 0 else 0

            if mode_name == "matmul":
                label = f"matmul({cfg[0]}x{cfg[0]})x{cfg[1]}"
            else:
                label = f"{mode_name}(kv={cfg[0]},x{cfg[1]})"
            print(f"    [{label}]  T_attn={T_attn_solo:.1f}ms")
            print(f"      serial  {_fmt_ms(times_serial)}")
            print(f"      prod    {_fmt_ms(times_prod)}")
            print(f"      speedup={speedup:.2f}x  coverage={coverage*100:.0f}%  "
                  f"contention={contention:+.1f}%")
            print()

            mode_results.append((cfg, T_attn_solo, T_serial, T_prod,
                                 speedup, coverage, contention))
            cleanup()

        all_results[mode_name] = mode_results

    # ==================================================================
    # Step 4: Cross-mode comparison table
    # ==================================================================
    print("\n  ╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║ Cross-mode comparison (1 anchor + {num_shared} shared, block_dim={block_dim})              ║")
    print(f"  ║ T_phase1={T_phase1:.1f}ms  T_side={T_side_solo:.1f}ms                                           ║")
    print(f"  ╠══════════════════════════════════╦══════════╦══════════╦══════════╦═══════════╣")
    print(f"  ║ attention proxy                  ║ T_attn   ║ T_prod   ║ speedup  ║ contention║")
    print(f"  ╠══════════════════════════════════╬══════════╬══════════╬══════════╬═══════════╣")

    for mode_name in ["matmul", "matmul_softmax", "sdpa"]:
        for cfg, t_attn, t_ser, t_prod, sp, cov, cont in all_results[mode_name]:
            if mode_name == "matmul":
                desc = f"matmul({cfg[0]}x{cfg[0]})x{cfg[1]}"
            else:
                desc = f"{mode_name}(kv={cfg[0]},x{cfg[1]})"
            print(f"  ║ {desc:<32} ║ {t_attn:8.1f} ║ {t_prod:8.1f} ║ {sp:7.2f}x ║ {cont:+9.1f}% ║")

    print(f"  ╚══════════════════════════════════╩══════════╩══════════╩══════════╩═══════════╝")
    print()
    print("  Legend:")
    print("    contention = (T_prod - T_ideal) / T_ideal * 100")
    print("    T_ideal = T_phase1 + max(T_attn, T_side)")
    print("    Higher contention → more DMA bandwidth conflict between attention and scatter")


# ===========================================================================
# Experiment 4b: Side-stream block_dim sweep
# ===========================================================================
#
# Fixed: sdpa(kv=2048, x4) as attention proxy (GLM-5.2 decode scenario).
# Phase 1 (sieve+scatter anchor) always uses block_dim=48 (fastest serial).
# Sweep: side-stream scatter block_dim to find optimal AIV core partition.
#
# Background: 910B has 24 AIC + 48 AIV.  sdpa uses ~48 AIV (CV split).
# scatter also uses AIV (DataCopy + scan).  When both want 48 AIV, they
# compete for physical cores → high contention (+49% in Exp 4).
# Reducing side-stream block_dim frees AIV cores for attention.

SIDE_BLOCK_DIMS = [48, 40, 32, 24, 16, 12, 8, 4]

# GLM-5.2 decode attention config: kv_len=2048, 4 layers
SDPA_KV_LEN = 2048
SDPA_REPS = 4


def run_experiment_4b(
    device: str,
    state_anchor: Dict[str, torch.Tensor],
    shared_states: List[Dict[str, torch.Tensor]],
    host_kv_dev_ptr: int,
    num_shared: int,
) -> None:
    """
    Side-stream block_dim sweep with fixed sdpa(kv=2048) attention proxy.

    Finds the optimal AIV core partition between main-stream attention
    and side-stream scatter to minimize total pipeline time.
    """
    print("=" * 72)
    print(f"Experiment 4b: Side-stream block_dim sweep")
    print(f"  Fixed attention: sdpa(kv={SDPA_KV_LEN}, x{SDPA_REPS}) = 4 decode layers")
    print(f"  Phase 1: sieve+scatter(anchor) block_dim=48 (serial, fastest)")
    print(f"  Sweep: side-stream {num_shared}× scatter block_dim in {SIDE_BLOCK_DIMS}")
    print(f"  Goal: minimize contention by freeing AIV cores for attention")
    print("=" * 72)

    from sglang.srt.hardware_backend.npu.op_impl.hisparse import (
        sieve_update_npu,
        scatter_from_host_npu,
    )
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    PHASE1_BD = 48

    # --- Prepare states ---
    state_anchor["topk_indices"].copy_(
        _gen_topk(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE, MISS_RATIO)
    )
    _init_tokens(state_anchor, DEVICE_BUFFER_SIZE)
    for ss in shared_states[:num_shared]:
        _pre_populate_for_scatter(ss, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)

    # --- Allocate attention tensors ---
    num_heads = 16
    head_dim = 128
    batch = NUM_REQS
    attn_st = dict(
        q=torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device=device),
        k=torch.randn(batch, num_heads, SDPA_KV_LEN, head_dim, dtype=torch.float16, device=device),
        v=torch.randn(batch, num_heads, SDPA_KV_LEN, head_dim, dtype=torch.float16, device=device),
    )

    def _run_attention():
        for _ in range(SDPA_REPS):
            sdpa(attn_st["q"], attn_st["k"], attn_st["v"])

    def _run_phase1():
        sieve_update_npu(
            layer_id=0,
            topk_indices=state_anchor["topk_indices"],
            req_pool_indices=state_anchor["req_pool_indices"],
            seq_lens=state_anchor["seq_lens"],
            prefill_len=state_anchor["prefill_len"],
            device_buffer_tokens=state_anchor["device_buffer_tokens"],
            device_buffer_visited=state_anchor["device_buffer_visited"],
            device_buffer_ht=state_anchor["device_buffer_ht"][0],
            sieve_hand=state_anchor["sieve_hand"],
            top_k_device_slots=state_anchor["top_k_device_slots"],
            is_miss=state_anchor["is_miss"],
            num_real_reqs=state_anchor["num_real_reqs"],
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            padded_buffer_size=PADDED_BUFFER_SIZE,
            max_num_reqs=NUM_REQS,
            block_dim=PHASE1_BD,
        )
        scatter_from_host_npu(
            host_kv_cache_ptr=host_kv_dev_ptr,
            topk_indices=state_anchor["topk_indices"],
            top_k_device_slots=state_anchor["top_k_device_slots"],
            is_miss=state_anchor["is_miss"],
            req_pool_indices=state_anchor["req_pool_indices"],
            req_to_host_pool=state_anchor["req_to_host_pool"],
            req_to_device_buffer=state_anchor["req_to_device_buffer"],
            device_k_buffer=state_anchor["device_k_buffer"],
            device_v_buffer=state_anchor["device_v_buffer"],
            layer_id=0,
            host_entries=MAX_CONTEXT_LEN,
            k_row_bytes=K_ROW_BYTES,
            v_row_bytes=V_ROW_BYTES,
            max_context_len=MAX_CONTEXT_LEN,
            device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
            padded_buffer_size=PADDED_BUFFER_SIZE,
            max_num_reqs=NUM_REQS,
            top_k=TOP_K,
            block_dim=PHASE1_BD,
        )

    def _make_side_scatter_fn(side_bd):
        def _run_side():
            for idx in range(num_shared):
                ss = shared_states[idx]
                scatter_from_host_npu(
                    host_kv_cache_ptr=host_kv_dev_ptr,
                    topk_indices=ss["topk_indices"],
                    top_k_device_slots=ss["top_k_device_slots"],
                    is_miss=ss["is_miss"],
                    req_pool_indices=ss["req_pool_indices"],
                    req_to_host_pool=ss["req_to_host_pool"],
                    req_to_device_buffer=ss["req_to_device_buffer"],
                    device_k_buffer=ss["device_k_buffer"],
                    device_v_buffer=ss["device_v_buffer"],
                    layer_id=1 + idx,
                    host_entries=MAX_CONTEXT_LEN,
                    k_row_bytes=K_ROW_BYTES,
                    v_row_bytes=V_ROW_BYTES,
                    max_context_len=MAX_CONTEXT_LEN,
                    device_buffer_row_stride=PADDED_BUFFER_SIZE + MAX_DECODE_LEN,
                    padded_buffer_size=PADDED_BUFFER_SIZE,
                    max_num_reqs=NUM_REQS,
                    top_k=TOP_K,
                    block_dim=side_bd,
                )
        return _run_side

    # ==================================================================
    # Step 1: Baseline timings (all serial, block_dim=48)
    # ==================================================================
    print("\n  --- Step 1: Baseline timings (serial, block_dim=48) ---\n")

    # T_attn_solo
    times_attn = []
    for i in range(WARMUP + ITERS):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        _run_attention()
        ev1.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times_attn.append(ev0.elapsed_time(ev1))
    T_attn_solo = _median(times_attn)

    # T_phase1
    times_p1 = []
    for i in range(WARMUP + ITERS):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        _run_phase1()
        ev1.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times_p1.append(ev0.elapsed_time(ev1))
    T_phase1 = _median(times_p1)

    # T_side at block_dim=48 (baseline)
    _run_side_48 = _make_side_scatter_fn(48)
    times_side = []
    for i in range(WARMUP + ITERS):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        _run_side_48()
        ev1.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times_side.append(ev0.elapsed_time(ev1))
    T_side_48 = _median(times_side)

    # Full serial baseline (phase1 + attn + side, all block_dim=48)
    times_full_serial = []
    for i in range(WARMUP + ITERS):
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        _run_phase1()
        _run_attention()
        _run_side_48()
        ev1.record()
        torch.npu.synchronize()
        if i >= WARMUP:
            times_full_serial.append(ev0.elapsed_time(ev1))
    T_full_serial = _median(times_full_serial)

    print(f"    T_attn(sdpa kv={SDPA_KV_LEN} x{SDPA_REPS}) = {T_attn_solo:8.3f} ms")
    print(f"    T_phase1 (sieve+scatter bd=48)       = {T_phase1:8.3f} ms")
    print(f"    T_side ({num_shared}× scatter bd=48)          = {T_side_48:8.3f} ms")
    print(f"    T_full_serial (all serial)           = {T_full_serial:8.3f} ms")

    # ==================================================================
    # Step 2: Sweep side-stream block_dim
    # ==================================================================
    print(f"\n  --- Step 2: Sweep side-stream block_dim ---\n")

    side_stream = torch.npu.Stream()
    sweep_results = []

    for side_bd in SIDE_BLOCK_DIMS:
        _run_side = _make_side_scatter_fn(side_bd)

        # Measure T_side at this block_dim (solo)
        times_side_bd = []
        for i in range(WARMUP + ITERS):
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            ev0.record()
            _run_side()
            ev1.record()
            torch.npu.synchronize()
            if i >= WARMUP:
                times_side_bd.append(ev0.elapsed_time(ev1))
        T_side_bd = _median(times_side_bd)

        # Production timeline: Phase 1 → fork → attention || side_scatter
        times_prod = []
        for i in range(WARMUP + ITERS):
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            ev0.record()
            _run_phase1()
            side_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(side_stream):
                _run_side()
            _run_attention()
            torch.npu.current_stream().wait_stream(side_stream)
            ev1.record()
            torch.npu.synchronize()
            if i >= WARMUP:
                times_prod.append(ev0.elapsed_time(ev1))

        T_prod = _median(times_prod)
        speedup = T_full_serial / T_prod if T_prod > 0 else 0
        T_ideal = T_phase1 + max(T_attn_solo, T_side_bd)
        contention = (T_prod - T_ideal) / T_ideal * 100 if T_ideal > 0 else 0

        # How much of T_side is hidden behind T_attn?
        coverage = max(0.0, min(1.0, T_attn_solo / T_side_bd)) if T_side_bd > 0 else 1.0

        print(f"  [side_bd={side_bd:2d}]  T_side={T_side_bd:6.2f}ms  "
              f"T_prod={T_prod:6.2f}ms  speedup={speedup:.2f}x  "
              f"contention={contention:+6.1f}%  coverage={coverage*100:3.0f}%")

        sweep_results.append((side_bd, T_side_bd, T_prod, speedup,
                              contention, coverage))

    # ==================================================================
    # Step 3: Summary table
    # ==================================================================
    best = max(sweep_results, key=lambda r: r[3])  # best speedup

    print(f"\n  ╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║ Experiment 4b: Side-stream block_dim sweep                                  ║")
    print(f"  ║ Fixed: sdpa(kv={SDPA_KV_LEN},x{SDPA_REPS})={T_attn_solo:.1f}ms  T_phase1={T_phase1:.1f}ms  T_serial={T_full_serial:.1f}ms            ║")
    print(f"  ╠══════════╦════════════╦══════════╦══════════╦══════════════╦═══════════════╣")
    print(f"  ║ side_bd  ║ T_side     ║ T_prod   ║ speedup  ║ contention   ║ coverage      ║")
    print(f"  ╠══════════╬════════════╬══════════╬══════════╬══════════════╬═══════════════╣")
    for side_bd, t_side, t_prod, sp, cont, cov in sweep_results:
        tag = " <<<" if side_bd == best[0] else ""
        print(f"  ║ {side_bd:8d} ║ {t_side:8.2f} ms ║ {t_prod:8.2f} ║ {sp:7.2f}x ║ {cont:+10.1f}%  ║ {cov*100:10.0f}%   ║{tag}")
    print(f"  ╚══════════╩════════════╩══════════╩══════════╩══════════════╩═══════════════╝")
    print()
    print(f"  Best: side_bd={best[0]} → speedup={best[3]:.2f}x "
          f"(T_prod={best[2]:.2f}ms vs serial={T_full_serial:.2f}ms)")
    print(f"  T_side grows from {sweep_results[0][1]:.2f}ms (bd=48) to "
          f"{sweep_results[-1][1]:.2f}ms (bd={sweep_results[-1][0]}) "
          f"→ trade-off: fewer cores = slower scatter but less contention")

    del attn_st
    torch.npu.empty_cache()


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
        choices=["0", "1", "2", "3", "4", "4b", "all"],
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
    parser.add_argument(
        "--num-shared", type=int, default=3,
        help="Number of shared layers per anchor group for Exp 4 (default: 3, "
             "matching GLM-5.2 index_topk_freq=4)",
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
    run_exp4 = args.exp in ("4", "all")
    run_exp4b = args.exp in ("4b", "all")
    num_shared = args.num_shared

    # --- Experiment 0 ---
    if run_exp0:
        run_experiment_0(device)

    # --- Experiments 1, 2, 3, 4, 4b (require kernel build) ---
    if run_exp1 or run_exp2 or run_exp3 or run_exp4 or run_exp4b:
        from sglang.srt.hardware_backend.npu.op_impl import hisparse as hisparse_pkg
        if not hisparse_pkg._HAS_KERNEL:
            print("\nERROR: hisparse_lru native extension not built.")
            print("Build it on the NPU host with:")
            print("  cd python/sglang/srt/hardware_backend/npu/op_impl/hisparse")
            print("  python setup.py build_ext --inplace")
            sys.exit(1)

        print(f"\nAuto block_dim = {hisparse_pkg._resolve_block_dim(0)}")

        # Allocate pinned host cache — must cover all layer_ids used by scatter.
        # Exp 1/2/3 use layer 0,1.  Exp 4 uses layer 0..num_shared.
        kv_row_bytes = K_ROW_BYTES + V_ROW_BYTES
        host_layers = max(2, 1 + num_shared)
        host_kv_ptr, host_kv_dev_ptr, host_kv_size = \
            _acl_malloc_host(host_layers * MAX_CONTEXT_LEN * kv_row_bytes)

        try:
            # Independent state sets
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
                    bd_configs=[(1, 1), (8, 4), (4, 44), (48, 48)],
                )

            if run_exp3:
                run_experiment_3(
                    device, state_a, state_b, host_kv_dev_ptr,
                )

            if run_exp4:
                # Need num_shared independent states for the side-stream scatters
                shared_states = [
                    _make_state(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)
                    for _ in range(num_shared)
                ]
                run_experiment_4(
                    device, state_a, shared_states, host_kv_dev_ptr,
                    num_shared=num_shared,
                )
                del shared_states

            if run_exp4b:
                shared_states = [
                    _make_state(device, NUM_REQS, TOP_K, DEVICE_BUFFER_SIZE)
                    for _ in range(num_shared)
                ]
                run_experiment_4b(
                    device, state_a, shared_states, host_kv_dev_ptr,
                    num_shared=num_shared,
                )
                del shared_states

            del state_a, state_b
            torch.npu.empty_cache()
        finally:
            _acl_free_host(host_kv_ptr, host_kv_dev_ptr, host_kv_size)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
