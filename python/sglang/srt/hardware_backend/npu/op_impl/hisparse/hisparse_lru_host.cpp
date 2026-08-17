// =============================================================================
// hisparse_lru_host.cpp — pybind11 host wrapper for HiSparse NPU kernels.
// =============================================================================

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>

extern "C" void launch_hisparse_lru_update(
    uint32_t blockDim,
    void* stream,
    void* topk_indices,
    void* req_pool_indices,
    void* seq_lens,
    void* prefill_len,
    void* device_buffer_tokens,
    void* device_buffer_recency,
    void* top_k_device_slots,
    void* is_miss,
    void* num_real_reqs,
    void* global_recency_counter,
    int32_t top_k,
    int32_t device_buffer_size,
    int32_t padded_buffer_size,
    int32_t max_num_reqs,
    int32_t reqs_per_core,
    int32_t ht_size,
    int32_t pool_size);

extern "C" void launch_hisparse_sieve_update(
    uint32_t blockDim,
    void* stream,
    void* topk_indices,
    void* req_pool_indices,
    void* seq_lens,
    void* prefill_len,
    void* device_buffer_tokens,
    void* device_buffer_visited,
    void* device_buffer_ht,
    void* sieve_hand,
    void* top_k_device_slots,
    void* is_miss,
    void* num_real_reqs,
    int32_t top_k,
    int32_t device_buffer_size,
    int32_t padded_buffer_size,
    int32_t visited_stride,
    int32_t max_num_reqs,
    int32_t reqs_per_core,
    int32_t ht_size,
    int32_t pool_size,
    int32_t t2s_cap);

extern "C" void launch_hisparse_sieve_ht_init(
    uint32_t blockDim,
    void* stream,
    void* device_buffer_tokens,
    void* device_buffer_ht,
    void* prefill_lens,
    int32_t req_start,
    int32_t num_reqs,
    int32_t layer_num,
    int32_t pool_size,
    int32_t device_buffer_size,
    int32_t padded_buffer_size,
    int32_t ht_size,
    int32_t t2s_cap);

extern "C" void launch_hisparse_scatter_from_host(
    uint32_t blockDim,
    void* stream,
    void* host_kv_cache,
    void* topk_indices,
    void* top_k_device_slots,
    void* is_miss,
    void* req_pool_indices,
    void* req_to_host_pool,
    void* req_to_device_buffer,
    void* device_k_buffer,
    void* device_v_buffer,
    int32_t layer_id,
    int32_t host_entries,
    int32_t k_row_bytes,
    int32_t v_row_bytes,
    int32_t max_context_len,
    int32_t device_buffer_row_stride,
    int32_t padded_buffer_size,
    int32_t max_num_reqs,
    int32_t top_k,
    int32_t positions_per_core,
    int32_t host_pool_rows,
    int32_t device_pool_rows,
    int32_t host_entry_major,
    int32_t host_num_layers);

extern "C" void launch_hisparse_scatter_from_host_group(
    uint32_t blockDim,
    void* stream,
    void* host_kv_cache,
    void* topk_indices,
    void* top_k_device_slots,
    void* is_miss,
    void* req_pool_indices,
    void* req_to_host_pool,
    void* req_to_device_buffer,
    void* device_k_buffer,
    void* device_v_buffer,
    int32_t anchor_layer_id,
    int32_t group_size,
    int32_t host_entries,
    int32_t k_row_bytes,
    int32_t v_row_bytes,
    int32_t max_context_len,
    int32_t device_buffer_row_stride,
    int32_t padded_buffer_size,
    int32_t max_num_reqs,
    int32_t top_k,
    int32_t positions_per_core,
    int32_t host_pool_rows,
    int32_t device_layer_row_count,
    int32_t device_layer_num,
    int32_t host_entry_major,
    int32_t host_num_layers);

extern "C" void launch_hisparse_backup_to_host(
    uint32_t blockDim,
    void* stream,
    void* device_k_buffer,
    void* device_v_buffer,
    void* host_kv_cache,
    void* host_indices,
    void* device_indices,
    int32_t num_tokens,
    int32_t host_entries,
    int32_t host_num_layers,
    int32_t device_layer_row_count,
    int32_t device_layer_num,
    int32_t k_row_bytes,
    int32_t v_row_bytes,
    int32_t tokens_per_core);

namespace {

static inline void* get_npu_stream() {
    return c10_npu::getCurrentNPUStream().stream();
}

static void check_tensor(const torch::Tensor& t, const std::string& name) {
    if (!t.is_contiguous()) {
        throw std::runtime_error(name + " must be contiguous");
    }
}

}  // namespace

void lru_update(
    torch::Tensor topk_indices,
    torch::Tensor req_pool_indices,
    torch::Tensor seq_lens,
    torch::Tensor prefill_len,
    torch::Tensor device_buffer_tokens,
    torch::Tensor device_buffer_recency,
    torch::Tensor top_k_device_slots,
    torch::Tensor is_miss,
    torch::Tensor num_real_reqs,
    torch::Tensor global_recency_counter,
    int64_t top_k,
    int64_t device_buffer_size,
    int64_t padded_buffer_size,
    int64_t max_num_reqs,
    int64_t block_dim)
{
    check_tensor(topk_indices, "topk_indices");
    check_tensor(req_pool_indices, "req_pool_indices");
    check_tensor(seq_lens, "seq_lens");
    check_tensor(prefill_len, "prefill_len");
    check_tensor(device_buffer_tokens, "device_buffer_tokens");
    check_tensor(device_buffer_recency, "device_buffer_recency");
    check_tensor(top_k_device_slots, "top_k_device_slots");
    check_tensor(is_miss, "is_miss");
    check_tensor(num_real_reqs, "num_real_reqs");
    check_tensor(global_recency_counter, "global_recency_counter");

    if (block_dim <= 0) {
        throw std::runtime_error("lru_update: block_dim must be > 0");
    }

    // The kernel writes GM with scalar stores. Concurrent scalar stores from
    // different blocks to the same 64-byte cache line clobber each other, so
    // every block's contiguous output region must be 64B aligned:
    //   - device buffer rows: padded_buffer_size * 4 bytes must be 64B aligned.
    //   - per-block slot/miss region: reqs_per_core * top_k must be a multiple
    //     of 64 elements (which also makes reqs_per_core * top_k * 4 a multiple
    //     of 64 bytes).
    if (padded_buffer_size % 16 != 0) {
        throw std::runtime_error(
            "lru_update: padded_buffer_size must be a multiple of 16 (64-byte rows)");
    }

    // The UB hash table stores slot indices as uint16 (slot+1).
    if (padded_buffer_size > 65534) {
        throw std::runtime_error(
            "lru_update: padded_buffer_size must be <= 65534 (uint16 hash slots)");
    }

    // Hash table size: power of two with load factor <= 0.5 even when every
    // query is a miss (buffer slots + top_k insertions per request).
    int64_t ht_size = 1024;
    while (ht_size < 2 * (padded_buffer_size + top_k)) {
        ht_size <<= 1;
    }

    // UB budget check (192KB per AIV core, keep some headroom):
    // 4 * padded_buffer_size * 4B vector buffers + ht_size * 2B + dstBuf.
    int64_t ub_bytes = 4 * padded_buffer_size * 4 + ht_size * 2 + 8 * 4;
    if (ub_bytes > 176 * 1024) {
        throw std::runtime_error(
            "lru_update: UB usage " + std::to_string(ub_bytes) +
            " bytes exceeds the 176KB budget");
    }

    int64_t rpc_align = 64 / std::gcd(top_k, static_cast<int64_t>(64));
    int64_t reqs_per_core = (max_num_reqs + block_dim - 1) / block_dim;
    reqs_per_core = (reqs_per_core + rpc_align - 1) / rpc_align * rpc_align;
    if (reqs_per_core < 1) {
        reqs_per_core = 1;
    }

    launch_hisparse_lru_update(
        static_cast<uint32_t>(block_dim),
        get_npu_stream(),
        topk_indices.data_ptr(),
        req_pool_indices.data_ptr(),
        seq_lens.data_ptr(),
        prefill_len.data_ptr(),
        device_buffer_tokens.data_ptr(),
        device_buffer_recency.data_ptr(),
        top_k_device_slots.data_ptr(),
        is_miss.data_ptr(),
        num_real_reqs.data_ptr(),
        global_recency_counter.data_ptr(),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(device_buffer_size),
        static_cast<int32_t>(padded_buffer_size),
        static_cast<int32_t>(max_num_reqs),
        static_cast<int32_t>(reqs_per_core),
        static_cast<int32_t>(ht_size),
        static_cast<int32_t>(device_buffer_tokens.size(0)));
}

void sieve_update(
    torch::Tensor topk_indices,
    torch::Tensor req_pool_indices,
    torch::Tensor seq_lens,
    torch::Tensor prefill_len,
    torch::Tensor device_buffer_tokens,
    torch::Tensor device_buffer_visited,
    torch::Tensor device_buffer_ht,
    torch::Tensor sieve_hand,
    torch::Tensor top_k_device_slots,
    torch::Tensor is_miss,
    torch::Tensor num_real_reqs,
    int64_t top_k,
    int64_t device_buffer_size,
    int64_t padded_buffer_size,
    int64_t max_num_reqs,
    int64_t block_dim)
{
    check_tensor(topk_indices, "topk_indices");
    check_tensor(req_pool_indices, "req_pool_indices");
    check_tensor(seq_lens, "seq_lens");
    check_tensor(prefill_len, "prefill_len");
    check_tensor(device_buffer_tokens, "device_buffer_tokens");
    check_tensor(device_buffer_visited, "device_buffer_visited");
    check_tensor(device_buffer_ht, "device_buffer_ht");
    check_tensor(sieve_hand, "sieve_hand");
    check_tensor(top_k_device_slots, "top_k_device_slots");
    check_tensor(is_miss, "is_miss");
    check_tensor(num_real_reqs, "num_real_reqs");

    if (block_dim <= 0) {
        throw std::runtime_error("sieve_update: block_dim must be > 0");
    }

    // Same 64-byte cache-line rules as lru_update (see above).  Additionally
    // every per-request GM row written with scalar stores (visited bytes,
    // hash table, hand pointer) must start on its own 64-byte line.
    if (padded_buffer_size % 16 != 0) {
        throw std::runtime_error(
            "sieve_update: padded_buffer_size must be a multiple of 16 (64-byte rows)");
    }

    // The hash table stores slot indices as uint16 (slot+1).
    if (padded_buffer_size > 65534) {
        throw std::runtime_error(
            "sieve_update: padded_buffer_size must be <= 65534 (uint16 hash slots)");
    }

    // The visited bytemap is staged with DataCopy, so rows must be whole
    // 64-byte cache lines and cover the padded slot range.
    int64_t visited_stride = device_buffer_visited.size(1);
    if (visited_stride < padded_buffer_size || visited_stride % 64 != 0) {
        throw std::runtime_error(
            "sieve_update: device_buffer_visited row stride must be >= "
            "padded_buffer_size and a multiple of 64");
    }

    // The kernel addresses the hand pointer as sieve_hand[req_idx * 16] so
    // per-request scalar GM stores land on distinct 64-byte cache lines.
    if (sieve_hand.size(1) != 16) {
        throw std::runtime_error(
            "sieve_update: sieve_hand must have shape [pool_size, 16] "
            "(64-byte row per request)");
    }

    // Persistent GM token-to-slot table: one row per request, t2s_cap entries.
    // When prefill_len < t2s_cap the kernel uses direct-index bitmap mode;
    // otherwise it uses the first ht_size entries as a hash table.  The
    // capacity is driven by the Python side via tensor shape.
    int64_t t2s_cap = device_buffer_ht.size(1);
    if (device_buffer_ht.size(0) != device_buffer_tokens.size(0)) {
        throw std::runtime_error(
            "sieve_update: device_buffer_ht must have one row per pool slot");
    }
    if (t2s_cap < 1024 || (t2s_cap & (t2s_cap - 1)) != 0) {
        throw std::runtime_error(
            "sieve_update: device_buffer_ht row size must be a power of two >= 1024");
    }

    // Hash table size for the fallback path (power of two, load factor <= 0.5).
    int64_t ht_size = 1024;
    while (ht_size < 2 * (padded_buffer_size + top_k)) {
        ht_size <<= 1;
    }
    if (ht_size > t2s_cap) {
        throw std::runtime_error(
            "sieve_update: computed ht_size exceeds t2s_cap");
    }

    // UB budget check (192KB per AIV core, keep some headroom):
    // padded_buffer_size * 4B tokens + visited_stride * 1B visited
    // + t2s_cap * 2B t2s staging.
    int64_t ub_bytes = padded_buffer_size * 4 + visited_stride + t2s_cap * 2;
    if (ub_bytes > 176 * 1024) {
        throw std::runtime_error(
            "sieve_update: UB usage " + std::to_string(ub_bytes) +
            " bytes exceeds the 176KB budget");
    }

    int64_t rpc_align = 64 / std::gcd(top_k, static_cast<int64_t>(64));
    int64_t reqs_per_core = (max_num_reqs + block_dim - 1) / block_dim;
    reqs_per_core = (reqs_per_core + rpc_align - 1) / rpc_align * rpc_align;
    if (reqs_per_core < 1) {
        reqs_per_core = 1;
    }

    launch_hisparse_sieve_update(
        static_cast<uint32_t>(block_dim),
        get_npu_stream(),
        topk_indices.data_ptr(),
        req_pool_indices.data_ptr(),
        seq_lens.data_ptr(),
        prefill_len.data_ptr(),
        device_buffer_tokens.data_ptr(),
        device_buffer_visited.data_ptr(),
        device_buffer_ht.data_ptr(),
        sieve_hand.data_ptr(),
        top_k_device_slots.data_ptr(),
        is_miss.data_ptr(),
        num_real_reqs.data_ptr(),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(device_buffer_size),
        static_cast<int32_t>(padded_buffer_size),
        static_cast<int32_t>(visited_stride),
        static_cast<int32_t>(max_num_reqs),
        static_cast<int32_t>(reqs_per_core),
        static_cast<int32_t>(ht_size),
        static_cast<int32_t>(device_buffer_tokens.size(0)),
        static_cast<int32_t>(t2s_cap));
}

void sieve_ht_init(
    torch::Tensor device_buffer_tokens,
    torch::Tensor device_buffer_ht,
    torch::Tensor prefill_lens,
    int64_t req_start,
    int64_t num_reqs,
    int64_t device_buffer_size,
    int64_t padded_buffer_size,
    int64_t ht_size)
{
    check_tensor(device_buffer_tokens, "device_buffer_tokens");
    check_tensor(device_buffer_ht, "device_buffer_ht");
    check_tensor(prefill_lens, "prefill_lens");

    int64_t layer_num = device_buffer_tokens.size(0);
    int64_t pool_size = device_buffer_tokens.size(1);
    int64_t t2s_cap = device_buffer_ht.size(2);

    if (device_buffer_ht.size(0) != layer_num ||
        device_buffer_ht.size(1) != pool_size) {
        throw std::runtime_error(
            "sieve_ht_init: device_buffer_ht must have shape "
            "[layer_num, pool_size, t2s_cap]");
    }
    if (t2s_cap < 1024 || (t2s_cap & (t2s_cap - 1)) != 0) {
        throw std::runtime_error(
            "sieve_ht_init: device_buffer_ht row size must be a power of two >= 1024");
    }
    if (req_start < 0 || num_reqs < 1 || req_start + num_reqs > pool_size) {
        throw std::runtime_error(
            "sieve_ht_init: invalid [req_start, req_start + num_reqs) range");
    }
    if (padded_buffer_size % 16 != 0 || padded_buffer_size > 65534) {
        throw std::runtime_error(
            "sieve_ht_init: padded_buffer_size must be a multiple of 16 and "
            "<= 65534");
    }
    if (ht_size < 1024 || (ht_size & (ht_size - 1)) != 0) {
        throw std::runtime_error(
            "sieve_ht_init: ht_size must be a power of two >= 1024");
    }
    if (ht_size > t2s_cap) {
        throw std::runtime_error(
            "sieve_ht_init: ht_size must not exceed t2s_cap");
    }

    // UB budget check: one t2s_cap * 2B staging buffer per block.
    if (t2s_cap * 2 > 176 * 1024) {
        throw std::runtime_error(
            "sieve_ht_init: t2s_cap exceeds the 176KB UB budget");
    }

    int64_t block_dim = layer_num * num_reqs;

    launch_hisparse_sieve_ht_init(
        static_cast<uint32_t>(block_dim),
        get_npu_stream(),
        device_buffer_tokens.data_ptr(),
        device_buffer_ht.data_ptr(),
        prefill_lens.data_ptr(),
        static_cast<int32_t>(req_start),
        static_cast<int32_t>(num_reqs),
        static_cast<int32_t>(layer_num),
        static_cast<int32_t>(pool_size),
        static_cast<int32_t>(device_buffer_size),
        static_cast<int32_t>(padded_buffer_size),
        static_cast<int32_t>(ht_size),
        static_cast<int32_t>(t2s_cap));
}

void scatter_from_host(
    int64_t host_kv_cache_ptr,
    torch::Tensor topk_indices,
    torch::Tensor top_k_device_slots,
    torch::Tensor is_miss,
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_host_pool,
    torch::Tensor req_to_device_buffer,
    torch::Tensor device_k_buffer,
    torch::Tensor device_v_buffer,
    int64_t layer_id,
    int64_t host_entries,
    int64_t k_row_bytes,
    int64_t v_row_bytes,
    int64_t max_context_len,
    int64_t device_buffer_row_stride,
    int64_t padded_buffer_size,
    int64_t max_num_reqs,
    int64_t top_k,
    int64_t host_entry_major,
    int64_t host_num_layers,
    int64_t block_dim)
{
    check_tensor(topk_indices, "topk_indices");
    check_tensor(top_k_device_slots, "top_k_device_slots");
    check_tensor(is_miss, "is_miss");
    check_tensor(req_pool_indices, "req_pool_indices");
    check_tensor(req_to_host_pool, "req_to_host_pool");
    check_tensor(req_to_device_buffer, "req_to_device_buffer");
    check_tensor(device_k_buffer, "device_k_buffer");
    check_tensor(device_v_buffer, "device_v_buffer");

    if (block_dim <= 0) {
        throw std::runtime_error("scatter_from_host: block_dim must be > 0");
    }

    if (k_row_bytes <= 0 || v_row_bytes <= 0) {
        throw std::runtime_error("scatter_from_host: k_row_bytes and v_row_bytes must be > 0");
    }
    if (k_row_bytes % 32 != 0 || v_row_bytes % 32 != 0) {
        throw std::runtime_error(
            "scatter_from_host: k_row_bytes and v_row_bytes must be multiples of 32 bytes");
    }
    if (device_buffer_row_stride < padded_buffer_size) {
        throw std::runtime_error(
            "scatter_from_host: device_buffer_row_stride must be >= padded_buffer_size");
    }
    if (host_num_layers <= 0) {
        throw std::runtime_error("scatter_from_host: host_num_layers must be > 0");
    }

    int64_t total_positions = max_num_reqs * top_k;
    int64_t positions_per_core = (total_positions + block_dim - 1) / block_dim;
    if (positions_per_core < 1) {
        positions_per_core = 1;
    }

    launch_hisparse_scatter_from_host(
        static_cast<uint32_t>(block_dim),
        get_npu_stream(),
        reinterpret_cast<void*>(host_kv_cache_ptr),
        topk_indices.data_ptr(),
        top_k_device_slots.data_ptr(),
        is_miss.data_ptr(),
        req_pool_indices.data_ptr(),
        req_to_host_pool.data_ptr(),
        req_to_device_buffer.data_ptr(),
        device_k_buffer.data_ptr(),
        device_v_buffer.data_ptr(),
        static_cast<int32_t>(layer_id),
        static_cast<int32_t>(host_entries),
        static_cast<int32_t>(k_row_bytes),
        static_cast<int32_t>(v_row_bytes),
        static_cast<int32_t>(max_context_len),
        static_cast<int32_t>(device_buffer_row_stride),
        static_cast<int32_t>(padded_buffer_size),
        static_cast<int32_t>(max_num_reqs),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(positions_per_core),
        static_cast<int32_t>(req_to_host_pool.size(0)),
        static_cast<int32_t>(device_k_buffer.size(0)),
        static_cast<int32_t>(host_entry_major ? 1 : 0),
        static_cast<int32_t>(host_num_layers));
}

void scatter_from_host_group(
    int64_t host_kv_cache_ptr,
    torch::Tensor topk_indices,
    torch::Tensor top_k_device_slots,
    torch::Tensor is_miss,
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_host_pool,
    torch::Tensor req_to_device_buffer,
    torch::Tensor device_k_buffer,
    torch::Tensor device_v_buffer,
    int64_t anchor_layer_id,
    int64_t group_size,
    int64_t host_entries,
    int64_t k_row_bytes,
    int64_t v_row_bytes,
    int64_t max_context_len,
    int64_t device_buffer_row_stride,
    int64_t padded_buffer_size,
    int64_t max_num_reqs,
    int64_t top_k,
    int64_t host_entry_major,
    int64_t host_num_layers,
    int64_t block_dim)
{
    check_tensor(topk_indices, "topk_indices");
    check_tensor(top_k_device_slots, "top_k_device_slots");
    check_tensor(is_miss, "is_miss");
    check_tensor(req_pool_indices, "req_pool_indices");
    check_tensor(req_to_host_pool, "req_to_host_pool");
    check_tensor(req_to_device_buffer, "req_to_device_buffer");
    check_tensor(device_k_buffer, "device_k_buffer");
    check_tensor(device_v_buffer, "device_v_buffer");

    if (block_dim <= 0) {
        throw std::runtime_error("scatter_from_host_group: block_dim must be > 0");
    }

    if (k_row_bytes <= 0 || v_row_bytes <= 0) {
        throw std::runtime_error("scatter_from_host_group: k_row_bytes and v_row_bytes must be > 0");
    }
    if (k_row_bytes % 32 != 0 || v_row_bytes % 32 != 0) {
        throw std::runtime_error(
            "scatter_from_host_group: k_row_bytes and v_row_bytes must be multiples of 32 bytes");
    }
    if (device_buffer_row_stride < padded_buffer_size) {
        throw std::runtime_error(
            "scatter_from_host_group: device_buffer_row_stride must be >= padded_buffer_size");
    }
    if (host_num_layers <= 0) {
        throw std::runtime_error("scatter_from_host_group: host_num_layers must be > 0");
    }

    // device_k_buffer / device_v_buffer are the FULL layer-stacked pools.
    const int64_t layer_num = device_k_buffer.size(0);
    if (layer_num != device_v_buffer.size(0)) {
        throw std::runtime_error(
            "scatter_from_host_group: device_k_buffer and device_v_buffer layer counts differ");
    }
    if (anchor_layer_id < 0 || anchor_layer_id >= layer_num) {
        throw std::runtime_error("scatter_from_host_group: anchor_layer_id out of range");
    }
    if (group_size < 1 || anchor_layer_id + group_size > layer_num) {
        throw std::runtime_error("scatter_from_host_group: group exceeds the layer range");
    }
    if (host_entry_major && host_num_layers != layer_num) {
        throw std::runtime_error(
            "scatter_from_host_group: entry-major host pool layer count mismatch");
    }

    // Flattened token rows per layer (page dims collapse into the row space).
    const int64_t k_layer_elems = device_k_buffer.numel() / layer_num;
    const int64_t v_layer_elems = device_v_buffer.numel() / layer_num;
    const int64_t k_row_count = k_layer_elems / device_k_buffer.size(-1);
    const int64_t v_row_count = v_layer_elems / device_v_buffer.size(-1);
    if (k_row_count != v_row_count || k_row_count <= 0) {
        throw std::runtime_error(
            "scatter_from_host_group: per-layer row counts inconsistent or empty");
    }
    if (k_row_count > INT32_MAX) {
        throw std::runtime_error("scatter_from_host_group: per-layer row count exceeds int32");
    }

    int64_t total_positions = max_num_reqs * top_k;
    int64_t positions_per_core = (total_positions + block_dim - 1) / block_dim;
    if (positions_per_core < 1) {
        positions_per_core = 1;
    }

    launch_hisparse_scatter_from_host_group(
        static_cast<uint32_t>(block_dim),
        get_npu_stream(),
        reinterpret_cast<void*>(host_kv_cache_ptr),
        topk_indices.data_ptr(),
        top_k_device_slots.data_ptr(),
        is_miss.data_ptr(),
        req_pool_indices.data_ptr(),
        req_to_host_pool.data_ptr(),
        req_to_device_buffer.data_ptr(),
        device_k_buffer.data_ptr(),
        device_v_buffer.data_ptr(),
        static_cast<int32_t>(anchor_layer_id),
        static_cast<int32_t>(group_size),
        static_cast<int32_t>(host_entries),
        static_cast<int32_t>(k_row_bytes),
        static_cast<int32_t>(v_row_bytes),
        static_cast<int32_t>(max_context_len),
        static_cast<int32_t>(device_buffer_row_stride),
        static_cast<int32_t>(padded_buffer_size),
        static_cast<int32_t>(max_num_reqs),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(positions_per_core),
        static_cast<int32_t>(req_to_host_pool.size(0)),
        static_cast<int32_t>(k_row_count),
        static_cast<int32_t>(layer_num),
        static_cast<int32_t>(host_entry_major ? 1 : 0),
        static_cast<int32_t>(host_num_layers));
}

void backup_to_host(
    torch::Tensor device_k_buffer,
    torch::Tensor device_v_buffer,
    int64_t host_kv_cache_ptr,
    torch::Tensor host_indices,
    torch::Tensor device_indices,
    int64_t host_entries,
    int64_t host_num_layers,
    int64_t k_row_bytes,
    int64_t v_row_bytes,
    int64_t num_tokens,
    int64_t block_dim)
{
    check_tensor(device_k_buffer, "device_k_buffer");
    check_tensor(device_v_buffer, "device_v_buffer");
    check_tensor(host_indices, "host_indices");
    check_tensor(device_indices, "device_indices");

    if (block_dim <= 0) {
        throw std::runtime_error("backup_to_host: block_dim must be > 0");
    }
    if (k_row_bytes <= 0 || v_row_bytes <= 0) {
        throw std::runtime_error("backup_to_host: k_row_bytes and v_row_bytes must be > 0");
    }
    if (k_row_bytes % 32 != 0 || v_row_bytes % 32 != 0) {
        throw std::runtime_error(
            "backup_to_host: k_row_bytes and v_row_bytes must be multiples of 32 bytes");
    }
    if (num_tokens <= 0) {
        throw std::runtime_error("backup_to_host: num_tokens must be > 0");
    }
    if (host_indices.numel() != num_tokens || device_indices.numel() != num_tokens) {
        throw std::runtime_error("backup_to_host: index tensor lengths must equal num_tokens");
    }
    if (host_indices.scalar_type() != torch::kInt64 ||
        device_indices.scalar_type() != torch::kInt64) {
        throw std::runtime_error("backup_to_host: index tensors must be int64");
    }

    const int64_t layer_num = device_k_buffer.size(0);
    if (layer_num != device_v_buffer.size(0)) {
        throw std::runtime_error(
            "backup_to_host: device_k_buffer and device_v_buffer layer counts differ");
    }
    if (host_num_layers != layer_num) {
        throw std::runtime_error("backup_to_host: host pool layer count mismatch");
    }

    // Flattened token rows per layer (page dims collapse into the row space).
    const int64_t k_layer_elems = device_k_buffer.numel() / layer_num;
    const int64_t v_layer_elems = device_v_buffer.numel() / layer_num;
    const int64_t k_row_count = k_layer_elems / device_k_buffer.size(-1);
    const int64_t v_row_count = v_layer_elems / device_v_buffer.size(-1);
    if (k_row_count != v_row_count || k_row_count <= 0) {
        throw std::runtime_error(
            "backup_to_host: per-layer row counts inconsistent or empty");
    }
    if (k_row_count > INT32_MAX) {
        throw std::runtime_error("backup_to_host: per-layer row count exceeds int32");
    }

    int64_t tokens_per_core = (num_tokens + block_dim - 1) / block_dim;
    if (tokens_per_core < 1) {
        tokens_per_core = 1;
    }

    launch_hisparse_backup_to_host(
        static_cast<uint32_t>(block_dim),
        get_npu_stream(),
        device_k_buffer.data_ptr(),
        device_v_buffer.data_ptr(),
        reinterpret_cast<void*>(host_kv_cache_ptr),
        host_indices.data_ptr(),
        device_indices.data_ptr(),
        static_cast<int32_t>(num_tokens),
        static_cast<int32_t>(host_entries),
        static_cast<int32_t>(host_num_layers),
        static_cast<int32_t>(k_row_count),
        static_cast<int32_t>(layer_num),
        static_cast<int32_t>(k_row_bytes),
        static_cast<int32_t>(v_row_bytes),
        static_cast<int32_t>(tokens_per_core));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HiSparse NPU graph-compatible kernels";
    m.def("lru_update", &lru_update,
          "LRU update and miss detection for HiSparse device buffer");
    m.def("sieve_update", &sieve_update,
          "SIEVE update and miss detection for HiSparse device buffer");
    m.def("sieve_ht_init", &sieve_ht_init,
          "Build/rebuild the persistent SIEVE hash table for a request range");
    m.def("scatter_from_host", &scatter_from_host,
          "Scatter missing KV rows from host cache to device buffer");
    m.def("scatter_from_host_group", &scatter_from_host_group,
          "Scatter missing KV rows for an anchor layer and its shared-index group");
    m.def("backup_to_host", &backup_to_host,
          "Backup full-layer KV rows from device pool to entry-major host pool");
}