import triton
import triton.language as tl


# free_page_ptr aliases self.free_pages, which the paged allocator re-slices
# after every allocation (self.free_pages = self.free_pages[num_new_pages:]).
# Slicing only advances data_ptr() by num_new_pages * 8 bytes, so the pointer
# flips between 16-byte-aligned and unaligned across calls. Triton specializes
# on pointer alignment by default and bakes it into the cache key, compiling two
# kernel variants (one with tt.divisibility=16 on free_page_ptr, one without)
# so the second prefill on a fresh DCP server hits the alternate alignment and
# pays an extra ~100ms JIT for that kernel variant. do_not_specialize skips
# that specialization so only one kernel is ever compiled; the perf cost is
# negligible (this kernel runs in ~10us and only loads ~4KB through this ptr).
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling -- no constexpr dependency
    # on extend size and only one kernel compilation.
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


# Same free_page_ptr alignment rationale as alloc_extend_kernel above.
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)

# HiSparse decode bookkeeping fusion (NPU/CUDA shared, triton JIT — no AOT
# build step).  One program per request derives the decode-extension slot
# for the current step, publishes the full->hisparse-device mapping for the
# freshly written token, and updates the page-aligned buffer capacity:
#     slot    = req_to_device_buffer[req_idx, padded + (seq_len-1-prefill)]
#     mapping[out_cache_loc[pid]] = slot
#     capacity[req_idx] = ceil((offset+1)/page)*page
# This replaces per-step host-side index_put/index gathers (each paying
# ~200-400us of torch-npu dispatch overhead) with a single launch.  The
# page-boundary slot allocation itself stays on the host: it only triggers
# every page_size steps per request and needs a cross-request page count,
# so batching it in the kernel would cost more than it saves.
# dtype contract: seq_lens/prefill_len are int32 (ForwardBatch convention),
# everything else int64; offset math is widened to int64 for >32K contexts.
@triton.jit
def hisparse_decode_slot_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    out_cache_loc_ptr,
    prefill_len_ptr,
    req_to_device_buffer_ptr,
    capacity_ptr,
    mapping_ptr,
    stride_cols,
    padded_buffer_size,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)
    req_idx = tl.load(req_pool_indices_ptr + pid)
    offset = (
        tl.load(seq_lens_ptr + pid).to(tl.int64)
        - 1
        - tl.load(prefill_len_ptr + req_idx).to(tl.int64)
    )
    slot = tl.load(
        req_to_device_buffer_ptr + req_idx * stride_cols + padded_buffer_size + offset
    )
    tl.store(mapping_ptr + tl.load(out_cache_loc_ptr + pid), slot)
    tl.store(
        capacity_ptr + req_idx, ((offset + page_size) // page_size) * page_size
    )