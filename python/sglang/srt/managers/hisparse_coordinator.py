# to be combined with the sparse coordinator class and sparse algorithm family

import logging
import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.kernels.ops.memory.allocator import hisparse_decode_slot_kernel
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip, is_npu

device_module = get_device_module()

_is_hip = is_hip()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def resolve_shared_index_layers(
    *,
    hf_text_config,
    pp_size: int,
    is_speculative: bool,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (PP, speculative decoding, kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    if pp_size != 1 or is_speculative:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism / speculative decoding; falling back to synchronous "
            "swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
        max_decode_len: int = 2048,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        self.max_decode_len = max_decode_len
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self.compress_ratio = getattr(
            self.token_to_kv_pool_allocator, "compress_ratio", 1
        )
        self.is_npu = _is_npu
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        if self.is_npu:
            self._init_npu(
                req_to_token_pool,
                host_to_device_ratio,
            )
        else:
            self._init_cuda(
                req_to_token_pool,
                host_to_device_ratio,
            )

        # NPU uses a different swap-in path; the CUDA plan-then-IO prefetch
        # stream machinery is not applicable.  NPU instead groups the
        # shared-index layers under their anchor: the anchor's scatter fills
        # the whole group's device buffers and skip layers become no-ops.
        if not self.is_npu:
            self._init_shared_index_prefetch(
                shared_index_layers=shared_index_layers,
                layer_num=self.mem_pool_device.layer_num,
                max_num_req_slots=req_to_token_pool.req_to_token.shape[0],
            )
        else:
            self._init_shared_index_npu(
                shared_index_layers=shared_index_layers,
                layer_num=self.mem_pool_device.layer_num,
            )

    def _init_cuda(
        self,
        req_to_token_pool: ReqToTokenPool,
        host_to_device_ratio: int,
    ) -> None:
        """CUDA/ROCm initialization path (original coordinator logic)."""
        device = self.device
        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

    def _init_npu(
        self,
        req_to_token_pool: ReqToTokenPool,
        host_to_device_ratio: int,
    ) -> None:
        """NPU initialization path using TieredHostMemoryPool + AscendC kernels."""
        from sglang.srt.mem_cache.tiered_host_memory_pool import TieredHostMemoryPool
        from sglang.srt.hardware_backend.npu.op_impl import hisparse as hisparse_lru

        device = self.device
        self.is_dsv4_hisparse = False
        self._lru_npu = hisparse_lru

        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()
        self.page_size = self.mem_pool_device.page_size

        if self.top_k >= self.device_buffer_size:
            raise ValueError(
                f"HiSparse requires top_k ({self.top_k}) < device_buffer_size "
                f"({self.device_buffer_size}); otherwise tokens swapped in during a "
                f"step can be evicted again within the same step"
            )

        # Padded to a multiple of 16 int32 (64 bytes) for cache-line-aligned
        # scalar GM stores in the hisparse kernels.
        self.padded_buffer_size = (
            (self.device_buffer_size + self.page_size + 15) // 16 * 16
        )

        self.mem_pool_host = TieredHostMemoryPool(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=host_to_device_ratio,
            shm_name=f"hisparse_npu_{os.getpid()}",
            override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            block_size=self.top_k,
            numa_node=self._resolve_numa_node(),
        )
        self.item_size_bytes = self.mem_pool_host.token_stride
        self.host_kv_cache_base_ptr = self.mem_pool_host.get_host_kv_data_ptr(0)

        # Size per-req state by the pool's actual row count, not `pool.size`:
        # ReqToTokenPool reserves a dummy row 0 and hands out slots
        # [1, size], so a legal req_pool_idx can reach `size`. Sizing by
        # pool.size would make slot `size` an out-of-bounds index here.
        max_num_reqs = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size + self.max_decode_len),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )
        self.req_prefill_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # CPU mirror of req_prefill_len: the scheduler-side decode
        # bookkeeping (alloc_decode_buffer_slot) reads prefill lengths from
        # CPU to avoid per-step device gathers and .item() syncs.  All
        # writes go through _set_req_prefill_len to keep the two in sync.
        self.req_prefill_len_cpu = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )
        # Cached CPU arange for decode page-slot column construction; sized
        # to the hisparse allocator page size on first use.
        self._page_arange_cpu = None
        self.req_decode_buffer_capacity = torch.zeros(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self.req_to_host_pool = torch.full(
            (max_num_reqs, max_context_len + self.top_k - 1),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_host_allocated_len = torch.zeros(
            max_num_reqs, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = None
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = None
        self._has_pending_backup = False

        self._eviction_algo = os.environ.get("HISPARSE_EVICTION", "sieve").lower()
        if self._eviction_algo not in ("lru", "sieve"):
            raise ValueError(
                f"HISPARSE_EVICTION must be 'lru' or 'sieve', got "
                f"{self._eviction_algo!r}"
            )

        layer_num = self.mem_pool_device.layer_num
        self.device_buffer_tokens = torch.full(
            (layer_num, max_num_reqs, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        if self._eviction_algo == "lru":
            self.device_buffer_recency = torch.zeros(
                (layer_num, max_num_reqs, self.padded_buffer_size),
                dtype=torch.int32,
                device=device,
            )
            self.device_buffer_visited = None
            self.device_buffer_ht = None
            self.sieve_hand = None
        else:
            self.device_buffer_recency = None
            self._visited_stride = (self.padded_buffer_size + 63) // 64 * 64
            self.device_buffer_visited = torch.zeros(
                (layer_num, max_num_reqs, self._visited_stride),
                dtype=torch.uint8,
                device=device,
            )
            ht_size = 1024
            while ht_size < 2 * (self.padded_buffer_size + self.top_k):
                ht_size <<= 1
            self._sieve_ht_size = ht_size
            self._t2s_cap = 65536
            self.device_buffer_ht = torch.zeros(
                (layer_num, max_num_reqs, self._t2s_cap),
                dtype=torch.int16,
                device=device,
            )
            self.sieve_hand = torch.zeros(
                (layer_num, max_num_reqs, 16),
                dtype=torch.int32,
                device=device,
            )

        self.top_k_device_slots = torch.full(
            (max_num_reqs, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.top_k_device_locs_buffer = self.top_k_device_slots
        self.raw_indices_buffer = torch.full(
            (max_num_reqs, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        self.global_recency_counter = torch.ones(1, dtype=torch.int32, device=device)
        self._skip_first_backup = [False] * max_num_reqs
        self.is_miss = torch.zeros(
            (max_num_reqs, self.top_k), dtype=torch.int8, device=device
        )

        # CUDA-graph compatibility aliases
        self.req_device_buffer_tokens = self.device_buffer_tokens
        self.req_device_buffer_token_locs = None
        self.lru_slots = None

        # Debug / crash-bisect switches
        self._debug_check = os.environ.get("HISPARSE_LRU_DEBUG", "0") == "1"
        self._bypass_swap = os.environ.get("HISPARSE_BYPASS_SWAP", "0") == "1"
        self._bypass_swap_1 = os.environ.get("HISPARSE_BYPASS_SWAP_1", "0") == "1"
        self._bypass_swap_2 = os.environ.get("HISPARSE_BYPASS_SWAP_2", "0") == "1"

        # M5: fused decode-slot bookkeeping — one triton launch replaces the
        # per-step host-side scatter/gather ops of the M4 torch path.
        # SGLANG_HISPARSE_FUSED_SLOT=0 falls back to the torch path.
        self._fused_slot = os.environ.get("SGLANG_HISPARSE_FUSED_SLOT", "1") == "1"

        # Benchmark stub quality: HISPARSE_BYPASS_SPREAD=1 replaces the
        # slot-0 fill of the bypass paths with a fixed pseudo-random spread
        # over [0, device_buffer_size) so the attention gather reads a
        # realistic row distribution (throughput ceiling measurement).
        # HISPARSE_BYPASS_SPREAD_VARY=1 rotates the pattern by one slot per
        # call via a device counter (also advances under NPU graph replay).
        # Built eagerly here: a lazy H2D build inside the first decode
        # forward would be captured by the NPU graph.
        self._bypass_spread = os.environ.get("HISPARSE_BYPASS_SPREAD", "0") == "1"
        self._bypass_spread_vary = (
            os.environ.get("HISPARSE_BYPASS_SPREAD_VARY", "0") == "1"
        )
        self._bypass_spread_slots = None
        self._bypass_spread_off = None
        if self._bypass_spread:
            self._init_bypass_spread_slots()
            self._bypass_spread_off = torch.zeros(1, dtype=torch.int32, device=device)
            logger.info(
                "HiSparse NPU: bypass spread stub enabled (vary=%s), slots "
                "spread over [0, %d)",
                self._bypass_spread_vary,
                self.device_buffer_size,
            )

        # Lightweight miss-rate statistics
        self._miss_stats_on = os.environ.get("HISPARSE_MISS_STATS", "0") == "1"
        self._miss_stats_interval = int(
            os.environ.get("HISPARSE_MISS_STATS_INTERVAL", "20")
        )
        self._layer_num = layer_num
        # Layers that actually run LRU/SIEVE + scatter at decode.  Defaults
        # cover the non-shared path (every layer); _init_shared_index_npu
        # overrides these when IndexShare grouping is active.
        self._swap_layers = list(range(layer_num))
        self._last_swap_layer = layer_num - 1
        self._swap_layer_count = layer_num
        self._stats_rank0 = torch.distributed.get_rank(self.tp_group) == 0
        if self._miss_stats_on:
            self._miss_rows_accum = torch.zeros(1, dtype=torch.int64, device=device)
            self._miss_per_req_accum = torch.zeros(
                max_num_reqs, dtype=torch.int64, device=device
            )
            self._stats_steps = 0
            self._timing_events = [
                (
                    device_module.Event(enable_timing=True),
                    device_module.Event(enable_timing=True),
                    device_module.Event(enable_timing=True),
                )
                for _ in range(self._miss_stats_interval)
            ]

        logger.info(
            "HiSparseCoordinator NPU init: eviction=%s, layers=%d, "
            "max_num_reqs=%d, padded_buffer_size=%d, top_k=%d, "
            "max_decode_len=%d",
            self._eviction_algo,
            layer_num,
            max_num_reqs,
            self.padded_buffer_size,
            self.top_k,
            self.max_decode_len,
        )

    def _resolve_numa_node(self) -> Optional[int]:
        """Pick the NUMA node for this rank's host pool."""
        policy = os.environ.get("SGLANG_HISPARSE_NUMA_POLICY", "bind").lower()
        if policy == "off":
            return None
        try:
            device_idx = (
                int(str(self.device).split(":")[1])
                if ":" in str(self.device)
                else torch.npu.current_device()
            )
        except Exception:
            device_idx = 0

        from sglang.srt.utils.numa_utils import (
            get_npu_numa_node,
            get_numa_node_count,
        )

        override = os.environ.get("SGLANG_HISPARSE_NUMA_NODE", "")
        if override:
            try:
                nodes = [int(x) for x in override.split(",")]
                if device_idx < len(nodes):
                    return nodes[device_idx]
            except ValueError:
                pass

        node = get_npu_numa_node(device_idx)
        if node is not None:
            return node
        count = get_numa_node_count()
        if count > 0:
            return device_idx % count
        return None

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        self._miss_src = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=self.device
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int32, device=self.device
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def _init_shared_index_npu(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
    ) -> None:
        """Set up IndexShare group tracking for the NPU swap-in path.

        Anchors run LRU/SIEVE once and a single group-scatter kernel fills
        their trailing shared-index (skip) layers' device buffers; skip
        layers return the anchor's slot table directly (zero kernel work).
        Correctness rests on the lockstep invariant: admission initializes
        and finish/abort clears every layer's buffer state identically, and
        only anchors ever mutate the shared slot table, so a group's layers
        always agree on slot contents.
        """
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            logger.warning(
                "HiSparse NPU shared-index grouping disabled: pattern length "
                "%d != KV pool layer_num %d; using per-layer swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if self.enable_prefetch:
            self._swap_layers = [
                i for i, shared in enumerate(self._is_shared_index_layer) if not shared
            ]
            self._last_swap_layer = max(self._swap_layers)
            self._swap_layer_count = len(self._swap_layers)
            logger.info(
                "HiSparse NPU: shared-index group swap-in enabled; %d anchor "
                "group(s), %d skip layer(s) of %d total (%d swap-in layers).",
                len(self._prefetch_groups),
                sum(self._is_shared_index_layer),
                layer_num,
                self._swap_layer_count,
            )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        self.write_staging_stream.synchronize()
        if self.decode_backup_stream is not None:
            self.decode_backup_stream.synchronize()
        if not self.is_npu and self.enable_prefetch:
            self.prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        if self.is_npu:
            host_capacity = self.mem_pool_host.host_entries
            host_tokens = host_capacity - self.mem_pool_host._free_tokens
        else:
            host_capacity = self.mem_pool_host.size
            host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def new_decode_buffer_tokens_required(self, requests: List[Req]) -> int:
        """Predict the hisparse device-pool tokens the next decode step claims.

        Mirrors the page-boundary predicate of
        _alloc_decode_buffer_slot_{fused,torch}: one page of device-buffer
        slots per request whose (kv_committed_len - prefill_len) % page_size
        == 0, with kv_committed_len read at check time (before
        prepare_for_decode's increment; the coordinator-side offset
        seq_lens - 1 - prefill_len equals this expression once seq_lens was
        bumped). The generic kv_committed_len % page_size criterion used by
        check_decode_mem diverges whenever prefill_len is not page-aligned —
        e.g. after a retraction resume — undercounting the demand so the
        scheduler crashes on pool exhaustion instead of retracting.
        """
        if self.max_decode_len <= 0:
            return 0
        page_size = self.token_to_kv_pool_allocator.page_size
        prefill_lens = self.req_prefill_len_cpu
        new_pages = sum(
            1
            for r in requests
            if (r.kv_committed_len - int(prefill_lens[r.req_pool_idx])) % page_size == 0
        )
        return new_pages * page_size

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True
        if self.is_npu:
            self._admit_request_into_staging_npu(req)
            return

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_npu:
            self._alloc_device_buffer_npu(req)
            return
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler.
            # Every rank must enter this collective on every call, including
            # ranks whose staging queue is empty (they contribute 0): staging
            # admission and event-completion timing differ per rank, so an
            # early return on an empty local queue splits ranks between
            # "inside this collective" and "moved on to the next one",
            # deadlocking TP until the watchdog fires (sglang#23288). MIN
            # keeps the pop set aligned: only the prefix whose staging
            # completed on ALL ranks pops.
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def _set_req_prefill_len(self, req_pool_idx: int, value: int) -> None:
        """Write the per-request prefill length to the device copy (consumed
        by the LRU/SIEVE kernels) and the CPU mirror (consumed by the
        scheduler-side decode bookkeeping) atomically from the caller's
        perspective."""
        self.req_prefill_len[req_pool_idx] = value
        self.req_prefill_len_cpu[req_pool_idx] = value

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        if self.is_npu:
            self.alloc_decode_buffer_slot(
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu,
            )
            return
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        self.write_staging_stream.synchronize()
        if self.is_npu:
            self._abort_staging_request_npu(req)
            return

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        if self.is_npu:
            self._request_finished_npu(req)
            return
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
    ) -> torch.Tensor:
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan into self._miss_{src,dst,count} for the skip layers to replay.
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        plan = (
            dict(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
            )
            if record_plan
            else {}
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
            **plan,
        )
        return top_k_indices

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        if self.is_npu:
            return self._swap_in_selected_pages_npu(
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )
        if not self.enable_prefetch:
            return self._run_swap_in_kernel(
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs

    # ------------------------------------------------------------------
    # NPU-specific methods (ported from sglang-0730-npu coordinator)
    # ------------------------------------------------------------------

    def _aligned_size(self, size: int) -> int:
        bs = self.mem_pool_host.block_size
        return ((size + bs - 1) // bs) * bs

    def _admit_request_into_staging_npu(self, req: Req) -> None:
        logical_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(
            logical_indices
        )

        prefill_len = len(device_indices)
        self._set_req_prefill_len(req.req_pool_idx, prefill_len)
        aligned_prefill_len = self._aligned_size(prefill_len)
        host_indices = self.mem_pool_host.alloc(aligned_prefill_len)
        if host_indices is None:
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {aligned_prefill_len} tokens"
            )
        self.req_to_host_pool[req.req_pool_idx, :aligned_prefill_len] = host_indices.to(
            self.device
        )
        self.req_host_allocated_len[req.req_pool_idx] = aligned_prefill_len

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices[:prefill_len], device_indices
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def _alloc_device_buffer_npu(self, req: Req) -> None:
        kv_len = req.extend_range.end
        allocated_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len
        ]
        page_size = self.mem_pool_device.page_size
        alloc_size = min(
            ((kv_len + page_size - 1) // page_size) * page_size,
            self.device_buffer_size,
        )
        if alloc_size == self.device_buffer_size:
            alloc_size = self.padded_buffer_size
        head_keep = tail_keep = 0
        if kv_len >= self.device_buffer_size:
            head_keep = (self.device_buffer_size // (2 * page_size)) * page_size
            tail_keep = self.device_buffer_size - head_keep
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            allocated_indices,
            alloc_size,
            head_keep,
            tail_keep,
        )
        if buffer_indices is None:
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size
        self.req_decode_buffer_capacity[req.req_pool_idx] = 0

        init_tokens = torch.full(
            (self.mem_pool_device.layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        valid = min(alloc_size, self.device_buffer_size)
        if tail_keep > 0:
            init_tokens[:, :head_keep] = torch.arange(
                head_keep, dtype=torch.int32, device=self.device
            )
            init_tokens[:, head_keep : head_keep + tail_keep] = torch.arange(
                kv_len - tail_keep,
                kv_len,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            init_tokens[:, :valid] = torch.arange(
                valid, dtype=torch.int32, device=self.device
            )
        self.device_buffer_tokens[:, req.req_pool_idx, :] = init_tokens
        if self._eviction_algo == "lru":
            self.device_buffer_recency[:, req.req_pool_idx, :] = 0
        else:
            self.device_buffer_visited[:, req.req_pool_idx, :] = 0
            self.sieve_hand[:, req.req_pool_idx, :] = 0
            self._lru_npu.sieve_ht_init_npu(
                self.device_buffer_tokens,
                self.device_buffer_ht,
                self.req_prefill_len,
                req.req_pool_idx,
                1,
                self.device_buffer_size,
                self.padded_buffer_size,
                self._sieve_ht_size,
            )

    def _free_device_buffer_npu(self, req_idx: int) -> None:
        current_cap = int(self.req_device_buffer_size[req_idx])
        decode_cap = int(self.req_decode_buffer_capacity[req_idx])
        parts = []
        if current_cap > 0:
            parts.append(self.req_to_device_buffer[req_idx, :current_cap])
        if decode_cap > 0:
            parts.append(
                self.req_to_device_buffer[
                    req_idx,
                    self.padded_buffer_size : self.padded_buffer_size + decode_cap,
                ]
            )
            self.req_decode_buffer_capacity[req_idx] = 0
        if parts:
            self.token_to_kv_pool_allocator.free_hisparse_indices(torch.cat(parts))

    def _abort_staging_request_npu(self, req: Req) -> None:
        self._free_device_buffer_npu(req.req_pool_idx)

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            allocated_locs.to(torch.int64)
        ] = 0

        host_allocated_len = int(self.req_host_allocated_len[req.req_pool_idx].item())
        host_indices = self.req_to_host_pool[req.req_pool_idx, :host_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_host_allocated_len[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self._set_req_prefill_len(req.req_pool_idx, 0)
        self.req_decode_buffer_capacity[req.req_pool_idx] = 0
        self.device_buffer_tokens[:, req.req_pool_idx, :] = -1
        if self._eviction_algo == "lru":
            self.device_buffer_recency[:, req.req_pool_idx, :] = 0
        req.hisparse_staging = False

    def _request_finished_npu(self, req: Req):
        self._free_device_buffer_npu(req.req_pool_idx)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv.kv_allocated_len
        ]
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            allocated_locs.to(torch.int64)
        ] = 0

        host_allocated_len = int(self.req_host_allocated_len[req.req_pool_idx].item())
        host_indices = self.req_to_host_pool[req.req_pool_idx, :host_allocated_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_host_allocated_len[req.req_pool_idx] = 0
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self._set_req_prefill_len(req.req_pool_idx, 0)
        self.req_decode_buffer_capacity[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.device_buffer_tokens[:, req.req_pool_idx, :] = -1
        if self._eviction_algo == "lru":
            self.device_buffer_recency[:, req.req_pool_idx, :] = 0
        self._skip_first_backup[req.req_pool_idx] = False

    def alloc_decode_buffer_slot(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Allocate decode-extension slots on-demand and map out_cache_loc."""
        bs = req_pool_indices.shape[0]
        if self.max_decode_len <= 0 or bs == 0:
            return

        if self._fused_slot and seq_lens_cpu.device.type == "cpu":
            self._alloc_decode_buffer_slot_fused(
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu,
            )
        else:
            # Fallback: env opt-out, or the *_cpu tensors unexpectedly live
            # on device (the fused path must not pay hidden .tolist() syncs).
            self._alloc_decode_buffer_slot_torch(
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu,
            )

    def _alloc_decode_buffer_slot_torch(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """M4 path: CPU bookkeeping + a handful of device scatter ops.

        All bookkeeping math (decode offsets, page-boundary detection,
        capacity alignment, scatter index construction) runs on CPU
        mirrors, so a decode step pays no device synchronization and no
        small-op launch storm here; only the scatter writes touch the
        device (one gather of the mapped slots plus, on page-boundary
        steps, one bulk page-slot write)."""
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        page_size = allocator.page_size
        decode_offsets = (
            seq_lens_cpu.to(torch.int64)
            - 1
            - self.req_prefill_len_cpu[req_pool_indices_cpu]
        )
        out_loc = (
            out_cache_loc
            if out_cache_loc.dtype == torch.int64
            else out_cache_loc.to(torch.int64)
        )

        if page_size == 1:
            new_slots = allocator.alloc(bs)
            if new_slots is None:
                raise RuntimeError("HiSparse decode buffer alloc returned None")
            col_indices = (self.padded_buffer_size + decode_offsets).to(self.device)
            self.req_to_device_buffer[req_pool_indices, col_indices] = new_slots
            self.req_decode_buffer_capacity[req_pool_indices] = (
                decode_offsets + 1
            ).to(self.device)
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                out_loc
            ] = new_slots.to(torch.int64)
            return

        needs_alloc = decode_offsets % page_size == 0
        num_new = int(needs_alloc.sum())
        if num_new > 0:
            total_slots = num_new * page_size
            page_slots = allocator.alloc(total_slots)
            if page_slots is None:
                raise RuntimeError(
                    f"HiSparse decode buffer alloc returned None (requested {total_slots})"
                )
            page_slots = page_slots.view(num_new, page_size)
            col_base = self.padded_buffer_size + decode_offsets[needs_alloc]
            page_arange = self._page_arange_cpu
            if page_arange is None or page_arange.numel() != page_size:
                page_arange = torch.arange(page_size, dtype=torch.int64)
                self._page_arange_cpu = page_arange
            cols_cpu = col_base.unsqueeze(1) + page_arange.unsqueeze(0)
            rows_cpu = req_pool_indices_cpu[needs_alloc].unsqueeze(1).expand(
                -1, page_size
            )
            self.req_to_device_buffer[
                rows_cpu.to(self.device), cols_cpu.to(self.device)
            ] = page_slots

        device_indices = self.req_to_device_buffer[
            req_pool_indices,
            (self.padded_buffer_size + decode_offsets).to(self.device),
        ]
        raw_cap = decode_offsets + 1
        page_aligned_cap = (raw_cap + page_size - 1) // page_size * page_size
        self.req_decode_buffer_capacity[req_pool_indices] = (
            page_aligned_cap.to(self.device)
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            out_loc
        ] = device_indices

    def _alloc_decode_buffer_slot_fused(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Fused path: python-side page-boundary bookkeeping (zero device
        sync, zero tensor temporaries) plus a single triton kernel that
        gathers the decode slot, publishes the full->hisparse mapping, and
        writes the aligned capacity for every request in one launch."""
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        page_size = allocator.page_size
        if page_size == 1:
            # Non-paged configs keep the torch path.
            self._alloc_decode_buffer_slot_torch(
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu,
            )
            return

        bs = req_pool_indices.shape[0]
        seq_list = seq_lens_cpu.tolist()
        prefill_list = self.req_prefill_len_cpu[req_pool_indices_cpu].tolist()
        req_ids = req_pool_indices_cpu.tolist()

        rows_idx = []
        cols_idx = []
        for i in range(bs):
            offset = int(seq_list[i]) - 1 - int(prefill_list[i])
            if offset % page_size == 0:
                base = self.padded_buffer_size + offset
                rows_idx.extend([req_ids[i]] * page_size)
                cols_idx.extend(range(base, base + page_size))

        if rows_idx:
            total_slots = len(rows_idx)
            page_slots = allocator.alloc(total_slots)
            if page_slots is None:
                raise RuntimeError(
                    f"HiSparse decode buffer alloc returned None (requested {total_slots})"
                )
            self.req_to_device_buffer[
                torch.tensor(rows_idx, dtype=torch.int64, device=self.device),
                torch.tensor(cols_idx, dtype=torch.int64, device=self.device),
            ] = page_slots

        hisparse_decode_slot_kernel[(bs,)](
            seq_lens,
            req_pool_indices,
            out_cache_loc,
            self.req_prefill_len,
            self.req_to_device_buffer,
            self.req_decode_buffer_capacity,
            self.mem_pool_device.full_to_hisparse_device_index_mapping,
            self.req_to_device_buffer.shape[1],
            self.padded_buffer_size,
            page_size,
        )

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        top_k_indices = self.req_to_device_buffer[req_pool_indices, : self.top_k].to(
            torch.int32
        )
        topk_col_indices = torch.arange(self.top_k, device=self.device).unsqueeze(0)
        mask = topk_col_indices >= seq_lens.unsqueeze(1)
        top_k_indices[mask] = -1
        return top_k_indices

    def _init_bypass_spread_slots(self) -> None:
        """Benchmark-only: build the fixed pseudo-random slot table used by
        the bypass spread stub.  Each request row gets its own distinct slot
        permutation (fixed CPU seed, reproducible); values stay within
        [0, device_buffer_size) so every downstream translation path is
        in-bounds."""
        dbs = self.device_buffer_size
        num_rows = self.req_to_device_buffer.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(20260817)
        rows = []
        for _ in range(num_rows):
            if self.top_k <= dbs:
                rows.append(
                    torch.randperm(dbs, generator=gen, dtype=torch.int32)[
                        : self.top_k
                    ]
                )
            else:
                rows.append(
                    torch.randint(
                        0, dbs, (self.top_k,), generator=gen, dtype=torch.int32
                    )
                )
        self._bypass_spread_slots = torch.stack(rows).contiguous().to(self.device)

    def _fill_bypass_slots(
        self,
        slots: torch.Tensor,
        top_k_tokens: torch.Tensor,
        num_real: int,
    ) -> None:
        """Overwrite the live ``top_k_device_slots`` rows with the spread
        stub, preserving the -1 padding semantics for invalid tokens.  With
        VARY enabled the pattern rotates by one slot per call (int32 device
        counter, so the rotation also advances under NPU graph replay)."""
        if self._bypass_spread_vary:
            self._bypass_spread_off.add_(1)
            slots.copy_(
                (self._bypass_spread_slots[:num_real] + self._bypass_spread_off)
                % self.device_buffer_size
            )
        else:
            slots.copy_(self._bypass_spread_slots[:num_real])
        slots.masked_fill_(top_k_tokens == -1, -1)

    def _swap_in_selected_pages_npu(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        if req_pool_indices.dtype != torch.int64:
            raise ValueError(
                f"req_pool_indices dtype {req_pool_indices.dtype} is not int64"
            )
        seq_lens = seq_lens.to(torch.int32)
        if top_k_result.dtype != torch.int32:
            raise ValueError(
                f"top_k_result dtype {top_k_result.dtype} is not int32"
            )

        top_k_result = top_k_result.reshape(top_k_result.shape[0], self.top_k)
        num_real = req_pool_indices.shape[0]

        if self._bypass_swap:
            slots = self.top_k_device_slots[:num_real]
            if self._bypass_spread:
                self._fill_bypass_slots(slots, top_k_result, num_real)
            else:
                slots.fill_(0)
                slots.masked_fill_(top_k_result == -1, -1)
            return slots.view(num_real, 1, -1)

        # IndexShare skip layer: reuse the anchor's swap-in results.  Same
        # top-k indices -> same LRU/SIEVE decisions -> same slot table, and
        # the anchor's group scatter already filled this layer's device
        # buffer (lockstep layout), so there is nothing left to do.
        if self.enable_prefetch and self._is_shared_index_layer[layer_id]:
            return self.top_k_device_slots[:num_real].view(num_real, 1, -1)

        self._load_cache_to_device_buffer_npu(
            top_k_tokens=top_k_result,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            layer_id=layer_id,
        )
        return self.top_k_device_slots[:num_real].view(num_real, 1, -1)

    def _load_cache_to_device_buffer_npu(
        self,
        top_k_tokens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        layer_id: int,
    ) -> None:
        max_num_reqs = req_pool_indices.shape[0]
        device = req_pool_indices.device

        if self._eviction_algo == "lru":
            self.is_miss.fill_(0)
            self.top_k_device_slots.fill_(-1)

        timed_layer = (
            self._miss_stats_on
            and layer_id
            == self._swap_layers[self._stats_steps % self._swap_layer_count]
        )
        if timed_layer:
            ev_lru_start, ev_lru_end, ev_scatter_end = self._timing_events[
                self._stats_steps % self._miss_stats_interval
            ]
            ev_lru_start.record()

        if self._eviction_algo == "sieve":
            self._lru_npu.sieve_update_npu(
                layer_id=layer_id,
                topk_indices=top_k_tokens,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                prefill_len=self.req_prefill_len,
                device_buffer_tokens=self.device_buffer_tokens[layer_id],
                device_buffer_visited=self.device_buffer_visited[layer_id],
                device_buffer_ht=self.device_buffer_ht[layer_id],
                sieve_hand=self.sieve_hand[layer_id],
                top_k_device_slots=self.top_k_device_slots,
                is_miss=self.is_miss,
                num_real_reqs=self.num_real_reqs,
                top_k=self.top_k,
                device_buffer_size=self.device_buffer_size,
                padded_buffer_size=self.padded_buffer_size,
                max_num_reqs=max_num_reqs,
            )
        else:
            self._lru_npu.lru_update_npu(
                layer_id=layer_id,
                topk_indices=top_k_tokens,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                prefill_len=self.req_prefill_len,
                device_buffer_tokens=self.device_buffer_tokens[layer_id],
                device_buffer_recency=self.device_buffer_recency[layer_id],
                top_k_device_slots=self.top_k_device_slots,
                is_miss=self.is_miss,
                num_real_reqs=self.num_real_reqs,
                global_recency_counter=self.global_recency_counter,
                top_k=self.top_k,
                device_buffer_size=self.device_buffer_size,
                padded_buffer_size=self.padded_buffer_size,
                max_num_reqs=max_num_reqs,
            )
        if timed_layer:
            ev_lru_end.record()

        if self._bypass_swap_1:
            num_real = req_pool_indices.shape[0]
            slots = self.top_k_device_slots[:num_real]
            if self._bypass_spread:
                self._fill_bypass_slots(slots, top_k_tokens, num_real)
            else:
                slots.fill_(0)
                slots.masked_fill_(top_k_tokens == -1, -1)
            return

        group = self._prefetch_groups.get(layer_id) if self.enable_prefetch else None
        if group:
            # IndexShare anchor: one group-scatter launch fills this layer's
            # and all its skip layers' device buffers (they share the slot
            # table produced by the LRU/SIEVE call above).
            self._lru_npu.scatter_from_host_group_npu(
                host_kv_cache_ptr=self.host_kv_cache_base_ptr,
                topk_indices=top_k_tokens,
                top_k_device_slots=self.top_k_device_slots,
                is_miss=self.is_miss,
                req_pool_indices=req_pool_indices,
                req_to_host_pool=self.req_to_host_pool,
                req_to_device_buffer=self.req_to_device_buffer,
                device_k_buffer=self.mem_pool_device.k_buffer,
                device_v_buffer=self.mem_pool_device.v_buffer,
                anchor_layer_id=layer_id,
                group_size=1 + len(group),
                host_entries=self.mem_pool_host.host_entries,
                k_row_bytes=self.mem_pool_device.kv_lora_rank * self.mem_pool_device.dtype.itemsize,
                v_row_bytes=self.mem_pool_device.qk_rope_head_dim * self.mem_pool_device.dtype.itemsize,
                max_context_len=self.req_to_host_pool.shape[1],
                device_buffer_row_stride=self.req_to_device_buffer.shape[1],
                padded_buffer_size=self.padded_buffer_size,
                max_num_reqs=max_num_reqs,
                top_k=self.top_k,
                host_entry_major=self.mem_pool_host.entry_major,
                host_num_layers=self.mem_pool_host.num_layers,
            )
        else:
            self._lru_npu.scatter_from_host_npu(
                host_kv_cache_ptr=self.host_kv_cache_base_ptr,
                topk_indices=top_k_tokens,
                top_k_device_slots=self.top_k_device_slots,
                is_miss=self.is_miss,
                req_pool_indices=req_pool_indices,
                req_to_host_pool=self.req_to_host_pool,
                req_to_device_buffer=self.req_to_device_buffer,
                device_k_buffer=self.mem_pool_device.k_buffer[layer_id],
                device_v_buffer=self.mem_pool_device.v_buffer[layer_id],
                layer_id=layer_id,
                host_entries=self.mem_pool_host.host_entries,
                k_row_bytes=self.mem_pool_device.kv_lora_rank * self.mem_pool_device.dtype.itemsize,
                v_row_bytes=self.mem_pool_device.qk_rope_head_dim * self.mem_pool_device.dtype.itemsize,
                max_context_len=self.req_to_host_pool.shape[1],
                device_buffer_row_stride=self.req_to_device_buffer.shape[1],
                padded_buffer_size=self.padded_buffer_size,
                max_num_reqs=max_num_reqs,
                top_k=self.top_k,
                host_entry_major=self.mem_pool_host.entry_major,
                host_num_layers=self.mem_pool_host.num_layers,
            )
        if timed_layer:
            ev_scatter_end.record()

        if self._eviction_algo == "lru":
            self.global_recency_counter.add_(1)

        if self._miss_stats_on:
            self._miss_rows_accum += self.is_miss[:max_num_reqs].sum()
            self._miss_per_req_accum[:max_num_reqs] += self.is_miss[
                :max_num_reqs
            ].sum(dim=1)
            if layer_id == self._last_swap_layer:
                self._stats_steps += 1
                if self._stats_steps % self._miss_stats_interval == 0:
                    if self._stats_rank0:
                        self._flush_miss_stats_npu(
                            max_num_reqs, layer_id, req_pool_indices
                        )
                    self._miss_rows_accum.zero_()
                    self._miss_per_req_accum.zero_()

    def _flush_miss_stats_npu(
        self, batch_size: int, layer_id: int, req_pool_indices: torch.Tensor
    ) -> None:
        interval = self._miss_stats_interval
        # Skip layers accumulate no stats (they early-return before the
        # LRU/scatter), so denominators count swap-in layers only.
        layers = self._swap_layer_count * interval
        total_miss = int(self._miss_rows_accum.item())
        denom = batch_size * self.top_k * layers
        miss_rate = total_miss / denom * 100 if denom > 0 else 0.0
        per_req_layer = (
            self._miss_per_req_accum[:batch_size].float() / layers
        ).cpu()
        row_bytes = (
            self.mem_pool_device.kv_lora_rank + self.mem_pool_device.qk_rope_head_dim
        ) * self.mem_pool_device.dtype.itemsize
        scatter_mb_per_step = total_miss * row_bytes / interval / 1e6
        page = self.mem_pool_device.page_size
        resident_pages = (
            (self.device_buffer_tokens[layer_id, req_pool_indices] >= 0)
            .sum(dim=1)
            .float()
            .cpu()
            / page
        )
        buffer_pages = self.padded_buffer_size // page

        last_slot = (self._stats_steps - 1) % interval
        self._timing_events[last_slot][2].synchronize()
        lru_ms_per_step = (
            sum(ev[0].elapsed_time(ev[1]) for ev in self._timing_events)
            / interval
            * self._swap_layer_count
        )
        scatter_ms_per_step = (
            sum(ev[1].elapsed_time(ev[2]) for ev in self._timing_events)
            / interval
            * self._swap_layer_count
        )
        scatter_bw = (
            scatter_mb_per_step / scatter_ms_per_step
            if scatter_ms_per_step > 0
            else 0.0
        )

        logger.info(
            "[HiSparseStats] step=%d reqs=%d miss_rate=%.1f%% "
            "miss_rows/req/layer avg=%.0f max=%.0f (top_k=%d) "
            "scatter=%.0fMB/step resident_pages avg=%.1f max=%.0f/%d "
            "lru=%.1fms/step scatter=%.1fms/step scatter_bw=%.1fGB/s",
            self._stats_steps,
            batch_size,
            miss_rate,
            per_req_layer.mean().item(),
            per_req_layer.max().item(),
            self.top_k,
            scatter_mb_per_step,
            resident_pages.mean().item(),
            resident_pages.max().item(),
            buffer_pages,
            lru_ms_per_step,
            scatter_ms_per_step,
            scatter_bw,
        )
