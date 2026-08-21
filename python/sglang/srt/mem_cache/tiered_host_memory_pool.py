"""
TieredHostMemoryPool: Manages a single shared-memory host pool for NPU HiSparse.

Architecture:
- Only one shm_type (SHM_NEAR) is allocated; far tier is disabled.
- Total capacity = device_size * host_to_device_ratio.
- Offset allocation uses unique pre-allocated slots to prevent collisions.
- No tier eviction / prefetch logic and no per-entry validity tracking.

Data Layout:
- Each token row stores K and V interleaved: [K(kv_lora_rank) | V(qk_rope_head_dim)].
- host_kv_tensor shape: (num_layers, host_entries, kv_lora_rank + qk_rope_head_dim)
- host_k_tensor = host_kv_tensor[:, :, :kv_lora_rank]  (non-contiguous view)
- host_v_tensor = host_kv_tensor[:, :, kv_lora_rank:]  (non-contiguous view)
- Interleaving allows the scatter kernel to DMA K+V in a single pipeline pass.
"""

import bisect
import ctypes
import contextlib
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch

import torch_npu

from sglang.srt.mem_cache.tiered_shm_mem_adapter import ShmMemAdapter, shm_type_t

logger = logging.getLogger(__name__)


class TieredHostMemoryPool:
    def __init__(
        self,
        device_pool,
        host_to_device_ratio: int,
        shm_name: str,
        override_kv_cache_dim: Optional[int] = None,
        block_size: int = 1,
        numa_node: Optional[int] = None,
    ):
        self.device_pool = device_pool
        self.num_layers = device_pool.layer_num
        self.shm_name = shm_name
        self.override_kv_cache_dim = override_kv_cache_dim
        self.block_size = block_size

        device_size = device_pool.size
        token_stride = self.get_token_stride_size()
        entry_stride = token_stride * self.num_layers

        self.k_size = device_pool.kv_lora_rank * device_pool.dtype.itemsize
        self.v_size = device_pool.qk_rope_head_dim * device_pool.dtype.itemsize
        self.token_stride = token_stride
        self.entry_stride = entry_stride

        logger.debug(
            "HiSparse: device_size %d token_stride %d entry_stride %d "
            "host_to_device_ratio %d block_size %d",
            device_size,
            token_stride,
            entry_stride,
            host_to_device_ratio,
            block_size,
        )

        host_entries = int(device_size * host_to_device_ratio) + 1
        host_entries = ((host_entries + block_size - 1) // block_size) * block_size
        total_size = host_entries * entry_stride

        self.adapter = ShmMemAdapter()
        # Only allocate SHM_NEAR; far tier size is 0.
        # When numa_node is given, hard-bind page allocations to that node so
        # the pool's pages (faulted in below) are physically placed there;
        # otherwise allocation follows the process default policy.
        if numa_node is not None:
            from sglang.srt.utils.numa_utils import (
                get_shm_numa_distribution,
                temp_membind,
            )

            bind_ctx = temp_membind(numa_node)
        else:
            bind_ctx = contextlib.nullcontext()
        with bind_ctx:
            ret = self.adapter.alloc(shm_name, [total_size, 0])
            if ret != 0:
                raise RuntimeError(f"shm_mem_alloc failed: {ret}")

            descs, _ = self.adapter.get(shm_name)
            desc = descs[shm_type_t.SHM_NEAR]
            host_ptr = self.adapter.mmap_region(desc, pin=True)
            if host_ptr is None:
                raise RuntimeError("Failed to mmap host shared memory region")
            self.host_ptr = int(host_ptr)
            self.host_dev_ptr = int(desc.dev_ptr)
            self._shm_dev_path = desc.dev_path.decode(errors="ignore").rstrip("\x00")

            if numa_node is not None:
                # Fault in every page while the bind policy is active so the
                # pages are allocated on the target NUMA node.
                ctypes.memset(self.host_ptr, 0, total_size)

        if numa_node is not None:
            dist = get_shm_numa_distribution(self._shm_dev_path)
            logger.info(
                "HiSparse: host pool (%d bytes) bound to numa node %d, "
                "page distribution %s",
                total_size,
                numa_node,
                dist,
            )

        logger.debug(
            "HiSparse: host_size %d host_ptr %d host_dev_ptr %d host_entries %d",
            total_size,
            self.host_ptr,
            self.host_dev_ptr,
            host_entries,
        )

        self.host_entries = host_entries

        self.logical_to_offset = [-1] * self.host_entries
        self.logical_to_pin_count = [0] * self.host_entries

        self.lock = threading.Lock()

        # Ordered free-list mode (SGLANG_HISPARSE_ORDERED_HOST_POOL, default
        # on): the pool is kept as a sorted, coalesced run list so that
        # allocations are as contiguous as possible. The legacy mode (flat
        # FIFO list) lets the offset order degrade arbitrarily under
        # mixed-length workloads, which scatters H2D DMA source regions.
        self._ordered = (
            os.environ.get("SGLANG_HISPARSE_ORDERED_HOST_POOL", "1") == "1"
        )

        self._init_offset_pool()
        self._init_host_tensors()
        self._init_host_cache_ptrs()

    def _init_host_cache_ptrs(self):
        """Expose NPU-accessible host cache base pointers for kernel direct access.

        The host pool uses an interleaved K/V layout: each token row is
        ``[K_data (kv_lora_rank) | V_data (qk_rope_head_dim)]`` contiguous in
        memory.  The scatter kernel reads K+V in a single DMA using this base
        pointer and the combined ``kv_row_bytes`` stride.

        The memory returned by ShmMemAdapter.mmap_region(pin=True) is registered with
        aclrtHostRegister; the kernel must use the returned device address, not the host
        virtual address, to avoid MTE out-of-range errors.
        """
        kv_row_bytes = (
            self.device_pool.kv_lora_rank + self.device_pool.qk_rope_head_dim
        ) * self.device_pool.dtype.itemsize
        self._host_kv_data_ptr = self.host_dev_ptr
        self._host_kv_layer_stride = self.host_entries * kv_row_bytes

    def get_host_kv_data_ptr(self, layer_id: int = 0) -> int:
        """Return the NPU-accessible host pointer for interleaved KV cache of ``layer_id``."""
        return self._host_kv_data_ptr + layer_id * self._host_kv_layer_stride

    def _init_offset_pool(self):
        if self._ordered:
            # Sorted list of (start_offset, length) runs, always coalesced.
            self._free_runs: List[Tuple[int, int]] = [(0, self.host_entries)]
            self._free_tokens = self.host_entries
        else:
            self.offset_pool: List[int] = list(range(self.host_entries))

    def get_fragmentation_stats(self) -> Dict[str, int]:
        """Snapshot of free-space fragmentation for diagnostics."""
        with self.lock:
            if self._ordered:
                return {
                    "avail": self._free_tokens,
                    "runs": len(self._free_runs),
                    "max_run": max((l for _, l in self._free_runs), default=0),
                }
            runs = 0
            max_run = 0
            cur = 0
            prev = -2
            for off in self.offset_pool:
                if off == prev + 1:
                    cur += 1
                else:
                    runs += 1
                    max_run = max(max_run, cur)
                    cur = 1
                prev = off
            max_run = max(max_run, cur)
            return {"avail": len(self.offset_pool), "runs": runs, "max_run": max_run}

    def _init_host_tensors(self):
        dtype = self.device_pool.dtype
        num_layers = self.num_layers
        num_tokens = self.host_entries
        kv_lora_rank = self.device_pool.kv_lora_rank
        qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        row_dim = kv_lora_rank + qk_rope_head_dim

        total_elem = num_layers * num_tokens * row_dim
        kv_buf = (ctypes.c_uint8 * (total_elem * dtype.itemsize)).from_address(self.host_ptr)
        self.host_kv_tensor = torch.frombuffer(
            kv_buf, dtype=dtype, count=total_elem
        ).view(num_layers, num_tokens, row_dim)
        # Non-contiguous views into the interleaved layout; PyTorch tensor
        # operations (copy_, index_copy_, advanced indexing) handle the
        # strides transparently.
        self.host_k_tensor = self.host_kv_tensor[:, :, :kv_lora_rank]
        self.host_v_tensor = self.host_kv_tensor[:, :, kv_lora_rank:]

    def get_token_stride_size(self) -> int:
        kv_cache_dim = self.override_kv_cache_dim or (
            self.device_pool.kv_lora_rank + self.device_pool.qk_rope_head_dim
        )
        dtype = self.device_pool.dtype
        if dtype in (torch.float16, torch.bfloat16):
            elem_bytes = 2
        elif dtype == torch.float32:
            elem_bytes = 4
        elif dtype == torch.float64:
            elem_bytes = 8
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        return kv_cache_dim * elem_bytes

    def alloc(self, size: int) -> torch.Tensor:
        with self.lock:
            if self._ordered:
                return self._alloc_ordered(size)
            if len(self.offset_pool) < size:
                raise RuntimeError(
                    f"TieredHostMemoryPool: allocation failed. "
                    f"Requested {size}, available {len(self.offset_pool)}"
                )

            offsets = self.offset_pool[:size]
            self.offset_pool = self.offset_pool[size:]

            for offset in offsets:
                self.logical_to_offset[offset] = offset

        return torch.tensor(offsets, dtype=torch.int64)

    def _alloc_ordered(self, size: int) -> torch.Tensor:
        """Lowest-address first-fit over the coalesced run list.

        Consumes whole small runs first (fragment compaction); the result is
        a single contiguous range whenever a run of >= size exists.
        Caller must hold self.lock.
        """
        if self._free_tokens < size:
            raise RuntimeError(
                f"TieredHostMemoryPool: allocation failed. "
                f"Requested {size}, available {self._free_tokens}"
            )
        remaining = size
        chunks = []
        i = 0
        while remaining > 0:
            start, length = self._free_runs[i]
            take = min(length, remaining)
            chunks.append((start, take))
            if take == length:
                self._free_runs.pop(i)
            else:
                self._free_runs[i] = (start + take, length - take)
            remaining -= take
        self._free_tokens -= size
        offsets = []
        for start, take in chunks:
            offsets.extend(range(start, start + take))
        for offset in offsets:
            self.logical_to_offset[offset] = offset
        if len(chunks) == 1:
            start, take = chunks[0]
            return torch.arange(start, start + take, dtype=torch.int64)
        return torch.tensor(offsets, dtype=torch.int64)

    def _insert_runs_ordered(self, offsets: List[int]) -> None:
        """Return freed offsets to the run list, coalescing with neighbours.
        Caller must hold self.lock.
        """
        if not offsets:
            return
        offsets.sort()
        runs = []
        start = prev = offsets[0]
        for off in offsets[1:]:
            if off == prev + 1:
                prev = off
            else:
                runs.append((start, prev - start + 1))
                start = prev = off
        runs.append((start, prev - start + 1))

        # Whatever the merge pattern is, the token delta equals the number of
        # freed offsets (merged runs were already counted when freed earlier).
        self._free_tokens += len(offsets)
        for start, length in runs:
            idx = bisect.bisect_left(self._free_runs, (start, 0))
            if idx > 0:
                ps, pl = self._free_runs[idx - 1]
                if ps + pl == start:
                    # Extend the left neighbour.
                    start = ps
                    length += pl
                    idx -= 1
                    self._free_runs.pop(idx)
            while idx < len(self._free_runs) and self._free_runs[idx][0] <= (
                start + length
            ):
                ns, nl = self._free_runs.pop(idx)
                length = ns + nl - start
            self._free_runs.insert(idx, (start, length))

    def free(self, indices: torch.Tensor) -> None:
        remaining_indices = indices.tolist()

        while remaining_indices:
            deferred_indices = []
            with self.lock:
                if self._ordered:
                    releasable = []
                    for logical_idx in remaining_indices:
                        if self.logical_to_pin_count[logical_idx] == 0:
                            offset = self.logical_to_offset[logical_idx]
                            if offset == -1:
                                continue
                            self.logical_to_offset[logical_idx] = -1
                            releasable.append(offset)
                        else:
                            deferred_indices.append(logical_idx)
                    self._insert_runs_ordered(releasable)
                else:
                    for logical_idx in remaining_indices:
                        if self.logical_to_pin_count[logical_idx] == 0:
                            offset = self.logical_to_offset[logical_idx]
                            if offset == -1:
                                continue
                            self.offset_pool.append(offset)
                            self.logical_to_offset[logical_idx] = -1
                        else:
                            deferred_indices.append(logical_idx)

            remaining_indices = deferred_indices
            if remaining_indices:
                time.sleep(0.001)

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ):
        num_tokens = host_indices.numel()
        batch_size = self.block_size

        for layer_id in range(device_pool.layer_num):
            kv_lora_rank = device_pool.kv_lora_rank
            qk_rope_head_dim = device_pool.qk_rope_head_dim

            for start in range(0, num_tokens, batch_size):
                end = min(start + batch_size, num_tokens)
                host_batch = host_indices[start:end]
                device_batch = device_indices[start:end]

                host_start = int(host_batch[0].item())
                host_end = int(host_batch[-1].item()) + 1

                device_k = (
                    device_pool.k_buffer[layer_id]
                    .view(-1, kv_lora_rank)[device_batch]
                    .contiguous()
                )
                device_v = (
                    device_pool.v_buffer[layer_id]
                    .view(-1, qk_rope_head_dim)[device_batch]
                    .contiguous()
                )

                host_k_block = self.host_k_tensor[layer_id][host_start:host_end]
                host_v_block = self.host_v_tensor[layer_id][host_start:host_end]
                host_k_block.copy_(device_k)
                host_v_block.copy_(device_v)

    def backup_from_device_all_layer_split(
            self,
            device_pool,
            host_indices: torch.Tensor,
            device_indices: torch.Tensor,
    ):
        for layer_id in range(device_pool.layer_num):
            kv_lora_rank = device_pool.kv_lora_rank
            qk_rope_head_dim = device_pool.qk_rope_head_dim

            device_k = device_pool.k_buffer[layer_id].view(-1, kv_lora_rank)[device_indices]
            device_v = device_pool.v_buffer[layer_id].view(-1, qk_rope_head_dim)[device_indices]

            self.host_k_tensor[layer_id].index_copy_(
                0, host_indices, device_k.cpu()
            )
            self.host_v_tensor[layer_id].index_copy_(
                0, host_indices, device_v.cpu()
            )

    def load_to_device_per_layer(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layer_id: int,
    ):
        kv_lora_rank = self.device_pool.kv_lora_rank
        qk_rope_head_dim = self.device_pool.qk_rope_head_dim

        host_k_block = self.host_k_tensor[layer_id][host_indices]
        host_v_block = self.host_v_tensor[layer_id][host_indices]

        device_k_block = host_k_block.to("npu")
        device_v_block = host_v_block.to("npu")
        self.device_pool.k_buffer[layer_id].view(-1, kv_lora_rank).index_copy_(
            0, device_indices, device_k_block
        )
        self.device_pool.v_buffer[layer_id].view(-1, qk_rope_head_dim).index_copy_(
            0, device_indices, device_v_block
        )

    def gather_kv_rows_to(
        self,
        host_indices: torch.Tensor,
        layer_id: int,
        out: torch.Tensor,
    ) -> None:
        out_ptr = int(out.data_ptr())
        row_size = self.token_stride
        host_indices_list = host_indices.tolist()
        # In the interleaved layout each token's K+V row is contiguous, so we
        # can copy the full row in a single memmove via host_kv_tensor.
        host_kv = self.host_kv_tensor[layer_id][host_indices_list]
        for i in range(host_indices.numel()):
            ctypes.memmove(out_ptr + i * row_size, int(host_kv[i].data_ptr()), row_size)

    def shutdown(self):
        self.adapter.free(self.shm_name)
