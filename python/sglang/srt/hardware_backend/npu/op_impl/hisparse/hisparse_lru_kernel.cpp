// =============================================================================
// hisparse_lru_kernel.cpp — AscendC kernels for HiSparse graph-mode support.
// =============================================================================

#include "kernel_operator.h"

using namespace AscendC;

// Token -> slot hash table in UB (open addressing, linear probing, uint16
// entries storing slot+1; 0 = empty, 0xFFFF = tombstone).  One table is built
// per request inside the kernel, giving O(1) hit detection and slot lookup.
namespace {

constexpr uint16_t kHtEmpty = 0;
constexpr uint16_t kHtTombstone = 0xFFFF;

// sieve_hand rows are strided by 16 int32 (64 bytes) per request so that
// scalar GM stores of the hand pointer from different blocks never share a
// cache line (concurrent scalar stores to the same line clobber each other).
constexpr int32_t kHandStride = 16;

// Capacity of the persistent token-to-slot table (one row per request in GM).
// When prefill_len < t2s_cap, the table is used as a direct-index bitmap
// (t2s[token] = slot+1, O(1) lookup/insert/erase, no hashing).  When
// prefill_len >= t2s_cap, the first ht_size entries are used as an
// open-addressing hash table (the original path).  Passed in at launch time
// from the host wrapper (which reads it from device_buffer_ht.size(-1)), so
// the Python side controls the capacity via tensor shape alone.

__aicore__ inline uint32_t ht_hash(int32_t token) {
    return (static_cast<uint32_t>(token) * 2654435761u) >> 16;
}

// Returns the slot holding `token`, or -1 when absent.
__aicore__ inline int32_t ht_find(LocalTensor<uint16_t>& ht, LocalTensor<int32_t>& tok,
                                  int32_t token, int32_t ht_mask) {
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & ht_mask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            return -1;
        }
        if (e != kHtTombstone && tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            return static_cast<int32_t>(e) - 1;
        }
        idx = (idx + 1) & ht_mask;
    }
}

// Inserts token -> slot.  A duplicate token keeps its first (lowest) slot,
// matching the first-hit semantics of the previous full-scan kernel.
__aicore__ inline void ht_insert(LocalTensor<uint16_t>& ht, LocalTensor<int32_t>& tok,
                                 int32_t token, int32_t slot, int32_t ht_mask) {
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & ht_mask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty || e == kHtTombstone) {
            ht.SetValue(idx, static_cast<uint16_t>(slot + 1));
            return;
        }
        if (tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            return;
        }
        idx = (idx + 1) & ht_mask;
    }
}

__aicore__ inline void ht_erase(LocalTensor<uint16_t>& ht, LocalTensor<int32_t>& tok,
                                int32_t token, int32_t ht_mask) {
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & ht_mask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            return;
        }
        if (e != kHtTombstone && tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            ht.SetValue(idx, kHtTombstone);
            return;
        }
        idx = (idx + 1) & ht_mask;
    }
}

// ---------------------------------------------------------------------------
// Backward-shift deletion (Knuth 6.4R) on a UB-staged table: removes `token`,
// then walks the probe cluster forward (scan pointer i, independent of the
// hole j) and pulls back into the hole any entry whose home slot is not in
// the circular interval (j, i], keeping probe chains intact without
// tombstones.  Used on the persistent GM hash table's UB staging copy, so no
// tombstone may ever be written.  No-op when the token is absent (defensive).
// ---------------------------------------------------------------------------
__aicore__ inline void ub_ht_erase_bs(LocalTensor<uint16_t>& ht, LocalTensor<int32_t>& tok,
                                      int32_t token, int32_t ht_mask) {
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & ht_mask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            return;
        }
        if (tok.GetValue(static_cast<int32_t>(e) - 1) == token) {
            break;
        }
        idx = (idx + 1) & ht_mask;
    }
    int32_t j = idx;
    int32_t i = idx;
    while (true) {
        i = (i + 1) & ht_mask;
        uint16_t e = ht.GetValue(i);
        if (e == kHtEmpty) {
            break;
        }
        int32_t home = static_cast<int32_t>(ht_hash(tok.GetValue(static_cast<int32_t>(e) - 1))) & ht_mask;
        bool move = (j < i) ? (home <= j || home > i)
                            : (home <= j && home > i);
        if (move) {
            ht.SetValue(j, e);
            j = i;
        }
    }
    ht.SetValue(j, kHtEmpty);
}

// UB-table insert used by the init kernel: same probing as ht_insert but
// duplicate compares read the GM token row.
__aicore__ inline void ub_ht_insert_gm_tok(LocalTensor<uint16_t>& ht, __gm__ int32_t* tok,
                                           int32_t token, int32_t slot, int32_t ht_mask) {
    int32_t idx = static_cast<int32_t>(ht_hash(token)) & ht_mask;
    while (true) {
        uint16_t e = ht.GetValue(idx);
        if (e == kHtEmpty) {
            ht.SetValue(idx, static_cast<uint16_t>(slot + 1));
            return;
        }
        if (tok[static_cast<int32_t>(e) - 1] == token) {
            return;
        }
        idx = (idx + 1) & ht_mask;
    }
}

}  // namespace

extern "C" __global__ __aicore__ void hisparse_lru_update(
    GM_ADDR topk_indices,
    GM_ADDR req_pool_indices,
    GM_ADDR seq_lens,
    GM_ADDR prefill_len,
    GM_ADDR device_buffer_tokens,
    GM_ADDR device_buffer_recency,
    GM_ADDR top_k_device_slots,
    GM_ADDR is_miss,
    GM_ADDR num_real_reqs,
    GM_ADDR global_recency_counter,
    int32_t top_k,
    int32_t device_buffer_size,
    int32_t padded_buffer_size,
    int32_t max_num_reqs,
    int32_t reqs_per_core,
    int32_t ht_size,
    int32_t pool_size)
{
    __gm__ int32_t* num_real_ptr = reinterpret_cast<__gm__ int32_t*>(num_real_reqs);
    int32_t num_real = num_real_ptr[0];
    // Defensive: num_real comes from GM and may be stale/inconsistent; never
    // process more rows than req_pool_indices actually holds.
    if (num_real > max_num_reqs) {
        num_real = max_num_reqs;
    }

    __gm__ int64_t* req_pool_ptr = reinterpret_cast<__gm__ int64_t*>(req_pool_indices);
    // seq_lens is int32 (matches both the eager ForwardBatch convention and,
    // critically, the int32 static buffers used by graph capture/replay).
    __gm__ int32_t* seq_len_ptr = reinterpret_cast<__gm__ int32_t*>(seq_lens);
    __gm__ int32_t* prefill_len_ptr = reinterpret_cast<__gm__ int32_t*>(prefill_len);
    __gm__ int32_t* topk_ptr = reinterpret_cast<__gm__ int32_t*>(topk_indices);
    __gm__ int32_t* buf_tokens_ptr = reinterpret_cast<__gm__ int32_t*>(device_buffer_tokens);
    __gm__ int32_t* buf_recency_ptr = reinterpret_cast<__gm__ int32_t*>(device_buffer_recency);
    __gm__ int32_t* out_slots_ptr = reinterpret_cast<__gm__ int32_t*>(top_k_device_slots);
    __gm__ int8_t* is_miss_ptr = reinterpret_cast<__gm__ int8_t*>(is_miss);
    __gm__ int32_t* recency_ptr = reinterpret_cast<__gm__ int32_t*>(global_recency_counter);
    int32_t recency = recency_ptr[0];

    int32_t block_idx = GetBlockIdx();
    int32_t start_req = block_idx * reqs_per_core;
    int32_t end_req = start_req + reqs_per_core;
    if (end_req > num_real) {
        end_req = num_real;
    }

    // Stage each request's buffer rows in UB.  Hit detection uses a UB hash
    // table (token -> slot) built once per request; the recency ReduceMin
    // eviction search only runs on misses.  Rows are written back once per
    // request via DataCopy.
    // (padded_buffer_size % 16 == 0 is enforced on the host, so the copies are
    // always 32B-aligned multiples.)
    TPipe pipe;
    TBuf<TPosition::VECCALC> tokBuf;
    TBuf<TPosition::VECCALC> recBuf;
    TBuf<TPosition::VECCALC> keyBuf;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> dstBuf;
    TBuf<TPosition::VECCALC> htBuf;
    pipe.InitBuffer(tokBuf, padded_buffer_size * sizeof(int32_t));
    pipe.InitBuffer(recBuf, padded_buffer_size * sizeof(int32_t));
    pipe.InitBuffer(keyBuf, padded_buffer_size * sizeof(float));
    pipe.InitBuffer(workBuf, padded_buffer_size * sizeof(float));
    pipe.InitBuffer(dstBuf, 8 * sizeof(float));
    pipe.InitBuffer(htBuf, ht_size * sizeof(uint16_t));
    LocalTensor<int32_t> tokLocal = tokBuf.Get<int32_t>();
    LocalTensor<int32_t> recLocal = recBuf.Get<int32_t>();
    LocalTensor<float> keyLocal = keyBuf.Get<float>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> dstLocal = dstBuf.Get<float>();
    LocalTensor<uint16_t> htLocal = htBuf.Get<uint16_t>();

    const int32_t kMaxInt = 0x7FFFFFFF;
    const int32_t ht_mask = ht_size - 1;

    for (int32_t bid = start_req; bid < end_req; ++bid) {
        int64_t req_idx = req_pool_ptr[bid];

        __gm__ int32_t* req_out_slots = out_slots_ptr + bid * top_k;
        __gm__ int8_t* req_is_miss = is_miss_ptr + bid * top_k;

        // Defensive: skip rows whose req_pool_idx is out of range instead of
        // addressing GM out of bounds.
        if (req_idx < 0 || req_idx >= pool_size) {
            for (int32_t k = 0; k < top_k; ++k) {
                req_out_slots[k] = -1;
                req_is_miss[k] = 0;
            }
            continue;
        }

        int64_t seq_len = seq_len_ptr[bid];
        int32_t prefill = prefill_len_ptr[req_idx];

        __gm__ int32_t* req_topk = topk_ptr + bid * top_k;
        __gm__ int32_t* req_buf_tokens = buf_tokens_ptr + req_idx * padded_buffer_size;
        __gm__ int32_t* req_buf_recency = buf_recency_ptr + req_idx * padded_buffer_size;

        GlobalTensor<int32_t> tokGm;
        GlobalTensor<int32_t> recGm;
        tokGm.SetGlobalBuffer(req_buf_tokens);
        recGm.SetGlobalBuffer(req_buf_recency);
        DataCopy(tokLocal, tokGm, padded_buffer_size);
        DataCopy(recLocal, recGm, padded_buffer_size);
        PipeBarrier<PIPE_ALL>();

        // Neutralize the padding region [device_buffer_size, padded_buffer_size):
        // tokens = -1 can never match a query (queries are >= 0 here), and
        // recency = INT32_MAX is never picked as the eviction victim.
        for (int32_t s = device_buffer_size; s < padded_buffer_size; ++s) {
            tokLocal.SetValue(s, -1);
            recLocal.SetValue(s, kMaxInt);
        }
        PipeBarrier<PIPE_ALL>();

        // Build the token -> slot hash table for this request.
        Duplicate<uint16_t>(htLocal, kHtEmpty, ht_size);
        PipeBarrier<PIPE_ALL>();
        for (int32_t s = 0; s < padded_buffer_size; ++s) {
            int32_t t = tokLocal.GetValue(s);
            if (t >= 0) {
                ht_insert(htLocal, tokLocal, t, s, ht_mask);
            }
        }

        for (int32_t k = 0; k < top_k; ++k) {
            int32_t token_pos = req_topk[k];

            if (token_pos < 0 || token_pos >= seq_len) {
                req_out_slots[k] = -1;
                req_is_miss[k] = 0;
                continue;
            }

            if (token_pos >= prefill) {
                int32_t decode_offset = token_pos - prefill;
                req_out_slots[k] = padded_buffer_size + decode_offset;
                req_is_miss[k] = 0;
                continue;
            }

            int32_t hit_slot = ht_find(htLocal, tokLocal, token_pos, ht_mask);
            if (hit_slot >= 0) {
                req_out_slots[k] = hit_slot;
                req_is_miss[k] = 0;
                recLocal.SetValue(hit_slot, recency);
            } else {
                // Eviction: first index of the minimum recency (calIndex
                // tie-break returns the first minimum, matching the CPU model).
                // Barrier: recLocal may have scalar writes from earlier hits.
                PipeBarrier<PIPE_ALL>();
                Cast(keyLocal, recLocal, RoundMode::CAST_NONE, padded_buffer_size);
                ReduceMin<float>(dstLocal, keyLocal, workLocal, padded_buffer_size, true);
                PipeBarrier<PIPE_ALL>();
                float cold_idx_raw = dstLocal.GetValue(1);
                uint32_t cold_idx = *reinterpret_cast<uint32_t*>(&cold_idx_raw);
                int32_t coldest_slot = static_cast<int32_t>(cold_idx);

                int32_t old_token = tokLocal.GetValue(coldest_slot);
                if (old_token >= 0) {
                    ht_erase(htLocal, tokLocal, old_token, ht_mask);
                }
                ht_insert(htLocal, tokLocal, token_pos, coldest_slot, ht_mask);

                req_out_slots[k] = coldest_slot;
                req_is_miss[k] = 1;
                tokLocal.SetValue(coldest_slot, token_pos);
                recLocal.SetValue(coldest_slot, recency);
            }
        }

        // Restore padding recency so the GM padding region keeps its initial
        // value (0) after the write-back.
        for (int32_t s = device_buffer_size; s < padded_buffer_size; ++s) {
            recLocal.SetValue(s, 0);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(tokGm, tokLocal, padded_buffer_size);
        DataCopy(recGm, recLocal, padded_buffer_size);
        PipeBarrier<PIPE_ALL>();
    }
}

// Builds/rebuilds the persistent GM token-to-slot table for a (layer, request)
// pair.  One block per (layer, request): the table is assembled in UB and
// written back with a single DataCopy.  Called once per request admission (or
// benchmark reset), never inside graph capture.
//
// When prefill_len < t2s_cap the table is a direct-index bitmap
// (t2s[token] = slot+1); otherwise the first ht_size entries form an
// open-addressing hash table (the original path).
extern "C" __global__ __aicore__ void hisparse_sieve_ht_init(
    GM_ADDR device_buffer_tokens,
    GM_ADDR device_buffer_ht,
    GM_ADDR prefill_lens,
    int32_t req_start,
    int32_t num_reqs,
    int32_t layer_num,
    int32_t pool_size,
    int32_t device_buffer_size,
    int32_t padded_buffer_size,
    int32_t ht_size,
    int32_t t2s_cap)
{
    int32_t b = GetBlockIdx();
    if (b >= layer_num * num_reqs) {
        return;
    }
    int32_t layer = b / num_reqs;
    int64_t req_idx = req_start + b % num_reqs;
    if (req_idx < 0 || req_idx >= pool_size) {
        return;
    }

    __gm__ int32_t* prefill_ptr = reinterpret_cast<__gm__ int32_t*>(prefill_lens);
    int32_t prefill = prefill_ptr[req_idx];

    __gm__ int32_t* tok_ptr = reinterpret_cast<__gm__ int32_t*>(device_buffer_tokens)
        + ((int64_t)layer * pool_size + req_idx) * padded_buffer_size;
    __gm__ uint16_t* ht_ptr = reinterpret_cast<__gm__ uint16_t*>(device_buffer_ht)
        + ((int64_t)layer * pool_size + req_idx) * t2s_cap;

    TPipe pipe;
    TBuf<TPosition::VECCALC> htBuf;
    pipe.InitBuffer(htBuf, t2s_cap * sizeof(uint16_t));
    LocalTensor<uint16_t> htLocal = htBuf.Get<uint16_t>();

    const int32_t ht_mask = ht_size - 1;
    GlobalTensor<uint16_t> htGm;
    htGm.SetGlobalBuffer(ht_ptr);

    if (prefill < t2s_cap) {
        // ===== Bitmap direct-index: t2s[token] = slot+1, 0 = empty =====
        Duplicate<uint16_t>(htLocal, kHtEmpty, t2s_cap);
        PipeBarrier<PIPE_ALL>();
        for (int32_t s = 0; s < device_buffer_size; ++s) {
            int32_t t = tok_ptr[s];
            if (t >= 0 && t < t2s_cap) {
                htLocal.SetValue(t, static_cast<uint16_t>(s + 1));
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(htGm, htLocal, t2s_cap);
    } else {
        // ===== Hash table: first ht_size entries, open addressing =====
        Duplicate<uint16_t>(htLocal, kHtEmpty, ht_size);
        PipeBarrier<PIPE_ALL>();
        for (int32_t s = 0; s < device_buffer_size; ++s) {
            int32_t t = tok_ptr[s];
            if (t >= 0) {
                ub_ht_insert_gm_tok(htLocal, tok_ptr, t, s, ht_mask);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(htGm, htLocal, ht_size);
    }
    PipeBarrier<PIPE_ALL>();
}

// SIEVE eviction variant of hisparse_lru_update.  Replaces the per-slot
// int32 recency row (and its per-miss full-buffer ReduceMin eviction scan)
// with a per-slot visited byte plus a persistent hand pointer per request:
// hits set visited=1; on a miss the hand sweeps forward, clearing visited
// bytes, and evicts the first unvisited slot.  Eviction work is amortized
// O(1) — the hand clears each slot at most once per revolution, so the total
// scan work per request per step is bounded by device_buffer_size + misses.
//
// Hit detection uses a persistent token -> slot table in GM (device_buffer_ht,
// one row per pool slot, t2s_cap entries).  When prefill_len < t2s_cap the
// table is a direct-index bitmap (t2s[token] = slot+1, single GetValue, no
// hashing); when prefill_len >= t2s_cap the first ht_size entries are used as
// an open-addressing hash table (backward-shift deletion, no tombstones).
// Built once per request admission by hisparse_sieve_ht_init and maintained
// incrementally by this kernel.  Per request the token row, visited row and
// t2s row are staged into UB, all probing/eviction runs on the UB copies,
// and the rows are written back once per request.  Requests with no miss
// skip the token/t2s write-back entirely.
//
// Output contract: every row of top_k_device_slots / is_miss in
// [0, max_num_reqs) is fully rewritten on every call, so the caller needs
// no fill_ pass for graph-replay padding rows.  Invalid top-k positions
// (negative or >= seq_len) and padded rows get slot -1; the sparse
// attention operator treats -1 as an invalid entry and skips it, so the
// outputs feed the attention kernel directly (no clamp_ pass).
extern "C" __global__ __aicore__ void hisparse_sieve_update(
    GM_ADDR topk_indices,
    GM_ADDR req_pool_indices,
    GM_ADDR seq_lens,
    GM_ADDR prefill_len,
    GM_ADDR device_buffer_tokens,
    GM_ADDR device_buffer_visited,
    GM_ADDR device_buffer_ht,
    GM_ADDR sieve_hand,
    GM_ADDR top_k_device_slots,
    GM_ADDR is_miss,
    GM_ADDR num_real_reqs,
    int32_t top_k,
    int32_t device_buffer_size,
    int32_t padded_buffer_size,
    int32_t visited_stride,
    int32_t max_num_reqs,
    int32_t reqs_per_core,
    int32_t ht_size,
    int32_t pool_size,
    int32_t t2s_cap)
{
    __gm__ int32_t* num_real_ptr = reinterpret_cast<__gm__ int32_t*>(num_real_reqs);
    int32_t num_real = num_real_ptr[0];
    // Defensive: num_real comes from GM and may be stale/inconsistent; never
    // process more rows than req_pool_indices actually holds.
    if (num_real > max_num_reqs) {
        num_real = max_num_reqs;
    }

    __gm__ int64_t* req_pool_ptr = reinterpret_cast<__gm__ int64_t*>(req_pool_indices);
    // seq_lens is int32 (matches both the eager ForwardBatch convention and,
    // critically, the int32 static buffers used by graph capture/replay).
    __gm__ int32_t* seq_len_ptr = reinterpret_cast<__gm__ int32_t*>(seq_lens);
    __gm__ int32_t* prefill_len_ptr = reinterpret_cast<__gm__ int32_t*>(prefill_len);
    __gm__ int32_t* topk_ptr = reinterpret_cast<__gm__ int32_t*>(topk_indices);
    __gm__ int32_t* buf_tokens_ptr = reinterpret_cast<__gm__ int32_t*>(device_buffer_tokens);
    __gm__ uint8_t* buf_visited_ptr = reinterpret_cast<__gm__ uint8_t*>(device_buffer_visited);
    __gm__ uint16_t* buf_ht_ptr = reinterpret_cast<__gm__ uint16_t*>(device_buffer_ht);
    __gm__ int32_t* hand_ptr = reinterpret_cast<__gm__ int32_t*>(sieve_hand);
    __gm__ int32_t* out_slots_ptr = reinterpret_cast<__gm__ int32_t*>(top_k_device_slots);
    __gm__ int8_t* is_miss_ptr = reinterpret_cast<__gm__ int8_t*>(is_miss);

    int32_t block_idx = GetBlockIdx();
    int32_t start_req = block_idx * reqs_per_core;
    int32_t end_req = start_req + reqs_per_core;
    if (end_req > max_num_reqs) {
        end_req = max_num_reqs;
    }

    // UB staging buffers (row sizes are 32B multiples — enforced on the
    // host — so the DataCopy bursts are always aligned).
    TPipe pipe;
    TBuf<TPosition::VECCALC> tokBuf;
    TBuf<TPosition::VECCALC> visBuf;
    TBuf<TPosition::VECCALC> htBuf;
    pipe.InitBuffer(tokBuf, padded_buffer_size * sizeof(int32_t));
    pipe.InitBuffer(visBuf, visited_stride * sizeof(uint8_t));
    pipe.InitBuffer(htBuf, t2s_cap * sizeof(uint16_t));
    LocalTensor<int32_t> tokLocal = tokBuf.Get<int32_t>();
    LocalTensor<uint8_t> visLocal = visBuf.Get<uint8_t>();
    LocalTensor<uint16_t> htLocal = htBuf.Get<uint16_t>();

    const int32_t ht_mask = ht_size - 1;

    for (int32_t bid = start_req; bid < end_req; ++bid) {
        __gm__ int32_t* req_out_slots = out_slots_ptr + bid * top_k;
        __gm__ int8_t* req_is_miss = is_miss_ptr + bid * top_k;

        // Padding rows beyond the real batch (graph replay): dummy outputs
        // only.  req_pool_indices/seq_lens content there is undefined and
        // must not be read.
        if (bid >= num_real) {
            for (int32_t k = 0; k < top_k; ++k) {
                req_out_slots[k] = -1;
                req_is_miss[k] = 0;
            }
            continue;
        }

        int64_t req_idx = req_pool_ptr[bid];

        // Defensive: skip rows whose req_pool_idx is out of range instead of
        // addressing GM out of bounds.
        if (req_idx < 0 || req_idx >= pool_size) {
            for (int32_t k = 0; k < top_k; ++k) {
                req_out_slots[k] = -1;
                req_is_miss[k] = 0;
            }
            continue;
        }

        int64_t seq_len = seq_len_ptr[bid];
        int32_t prefill = prefill_len_ptr[req_idx];

        __gm__ int32_t* req_topk = topk_ptr + bid * top_k;
        __gm__ int32_t* req_buf_tokens = buf_tokens_ptr + req_idx * padded_buffer_size;
        __gm__ uint8_t* req_buf_visited = buf_visited_ptr + req_idx * visited_stride;
        __gm__ uint16_t* req_ht = buf_ht_ptr + req_idx * t2s_cap;

        GlobalTensor<int32_t> tokGm;
        GlobalTensor<uint8_t> visGm;
        GlobalTensor<uint16_t> htGm;
        tokGm.SetGlobalBuffer(req_buf_tokens);
        visGm.SetGlobalBuffer(req_buf_visited);
        htGm.SetGlobalBuffer(req_ht);
        DataCopy(tokLocal, tokGm, padded_buffer_size);
        DataCopy(visLocal, visGm, visited_stride);

        // Load the t2s table: full t2s_cap for hash mode (hash entries spread
        // across [0, ht_size)), or only the live [0, bm_count) range for bitmap
        // mode (all tokens are < prefill, saving DMA bandwidth on short prefills).
        bool use_bitmap = (prefill < t2s_cap);
        int32_t bm_count = t2s_cap;
        if (use_bitmap) {
            bm_count = (prefill + 15) & ~15;
            if (bm_count > t2s_cap) {
                bm_count = t2s_cap;
            }
            if (bm_count < 16) {
                bm_count = 16;
            }
            DataCopy(htLocal, htGm, bm_count);
        } else {
            DataCopy(htLocal, htGm, ht_size);
        }
        PipeBarrier<PIPE_ALL>();

        int32_t hand = hand_ptr[req_idx * kHandStride];
        // Defensive: clamp a stale/corrupt hand into the valid range.
        if (hand < 0 || hand >= device_buffer_size) {
            hand = 0;
        }

        // Tracks whether any miss modified the token/hash rows; hit-only
        // requests skip their write-back.
        bool ht_dirty = false;

        for (int32_t k = 0; k < top_k; ++k) {
            int32_t token_pos = req_topk[k];

            if (token_pos < 0 || token_pos >= seq_len) {
                req_out_slots[k] = -1;
                req_is_miss[k] = 0;
                continue;
            }

            if (token_pos >= prefill) {
                int32_t decode_offset = token_pos - prefill;
                req_out_slots[k] = padded_buffer_size + decode_offset;
                req_is_miss[k] = 0;
                continue;
            }

            if (use_bitmap) {
                // ===== Bitmap direct-index: t2s[token_pos] = slot+1 =====
                uint16_t sp1 = htLocal.GetValue(token_pos);
                if (sp1 != kHtEmpty) {
                    int32_t hit_slot = static_cast<int32_t>(sp1) - 1;
                    req_out_slots[k] = hit_slot;
                    req_is_miss[k] = 0;
                    visLocal.SetValue(hit_slot, 1);
                } else {
                    // SIEVE eviction: sweep hand to first unvisited slot.
                    int32_t h = hand;
                    while (visLocal.GetValue(h) != 0) {
                        visLocal.SetValue(h, 0);
                        ++h;
                        if (h == device_buffer_size) {
                            h = 0;
                        }
                    }

                    int32_t old_token = tokLocal.GetValue(h);
                    if (old_token >= 0) {
                        htLocal.SetValue(old_token, kHtEmpty);  // O(1) erase
                    }
                    htLocal.SetValue(token_pos, static_cast<uint16_t>(h + 1));  // O(1) insert
                    ht_dirty = true;

                    req_out_slots[k] = h;
                    req_is_miss[k] = 1;
                    tokLocal.SetValue(h, token_pos);
                    visLocal.SetValue(h, 1);
                    hand = h + 1;
                    if (hand == device_buffer_size) {
                        hand = 0;
                    }
                }
            } else {
                // ===== Hash table path (prefill >= t2s_cap) =====
                int32_t hit_slot = ht_find(htLocal, tokLocal, token_pos, ht_mask);
                if (hit_slot >= 0) {
                    req_out_slots[k] = hit_slot;
                    req_is_miss[k] = 0;
                    visLocal.SetValue(hit_slot, 1);
                } else {
                    // SIEVE eviction: sweep hand to first unvisited slot.
                    int32_t h = hand;
                    while (visLocal.GetValue(h) != 0) {
                        visLocal.SetValue(h, 0);
                        ++h;
                        if (h == device_buffer_size) {
                            h = 0;
                        }
                    }

                    int32_t old_token = tokLocal.GetValue(h);
                    if (old_token >= 0) {
                        ub_ht_erase_bs(htLocal, tokLocal, old_token, ht_mask);
                    }
                    ht_insert(htLocal, tokLocal, token_pos, h, ht_mask);
                    ht_dirty = true;

                    req_out_slots[k] = h;
                    req_is_miss[k] = 1;
                    tokLocal.SetValue(h, token_pos);
                    visLocal.SetValue(h, 1);
                    hand = h + 1;
                    if (hand == device_buffer_size) {
                        hand = 0;
                    }
                }
            }
        }

        hand_ptr[req_idx * kHandStride] = hand;

        PipeBarrier<PIPE_ALL>();
        DataCopy(visGm, visLocal, visited_stride);
        if (ht_dirty) {
            DataCopy(tokGm, tokLocal, padded_buffer_size);
            if (use_bitmap) {
                DataCopy(htGm, htLocal, bm_count);
            } else {
                DataCopy(htGm, htLocal, ht_size);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void hisparse_scatter_from_host(
    GM_ADDR host_kv_cache,
    GM_ADDR topk_indices,
    GM_ADDR top_k_device_slots,
    GM_ADDR is_miss,
    GM_ADDR req_pool_indices,
    GM_ADDR req_to_host_pool,
    GM_ADDR req_to_device_buffer,
    GM_ADDR device_k_buffer,
    GM_ADDR device_v_buffer,
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
    int32_t device_pool_rows)
{
    int32_t total_positions = max_num_reqs * top_k;
    int32_t block_idx = GetBlockIdx();
    int32_t start_pos = block_idx * positions_per_core;
    int32_t end_pos = start_pos + positions_per_core;
    if (end_pos > total_positions) {
        end_pos = total_positions;
    }

    int32_t kv_row_bytes = k_row_bytes + v_row_bytes;

    TPipe pipe;
    // Quad-buffered queues provide deeper pipeline overlap: while miss N
    // is written back to device (MTE3), misses N+1..N+3 can already be
    // fetched from host (MTE1).  Each slot holds a full K+V row so a single
    // pipeline pass copies both.
    TQue<QuePosition::VECIN, 4> inQue;
    TQue<QuePosition::VECOUT, 4> outQue;
    pipe.InitBuffer(inQue, 4, kv_row_bytes);
    pipe.InitBuffer(outQue, 4, kv_row_bytes);

    __gm__ int32_t* topk_ptr = reinterpret_cast<__gm__ int32_t*>(topk_indices);
    __gm__ int32_t* slots_ptr = reinterpret_cast<__gm__ int32_t*>(top_k_device_slots);
    __gm__ int8_t* is_miss_ptr = reinterpret_cast<__gm__ int8_t*>(is_miss);
    __gm__ int64_t* req_pool_ptr = reinterpret_cast<__gm__ int64_t*>(req_pool_indices);
    __gm__ int64_t* req_to_host_pool_ptr = reinterpret_cast<__gm__ int64_t*>(req_to_host_pool);
    __gm__ int64_t* req_to_device_buffer_ptr = reinterpret_cast<__gm__ int64_t*>(req_to_device_buffer);
    __gm__ uint8_t* host_kv_cache_ptr = reinterpret_cast<__gm__ uint8_t*>(host_kv_cache);

    for (int32_t pos = start_pos; pos < end_pos; ++pos) {
        int32_t bid = pos / top_k;
        int32_t k = pos % top_k;
        if (is_miss_ptr[pos] == 0) {
            continue;
        }

        int64_t req_idx = req_pool_ptr[bid];
        int32_t token_pos = topk_ptr[pos];
        int32_t device_slot_in_buffer = slots_ptr[pos];

        // Defensive: never address GM/host memory out of bounds on
        // inconsistent state.
        if (req_idx < 0 || req_idx >= host_pool_rows) {
            continue;
        }
        if (token_pos < 0 || token_pos >= max_context_len) {
            continue;
        }
        if (device_slot_in_buffer < 0 || device_slot_in_buffer >= device_buffer_row_stride) {
            continue;
        }

        int64_t host_offset = req_to_host_pool_ptr[req_idx * max_context_len + token_pos];
        if (host_offset < 0 || host_offset >= host_entries) {
            continue;
        }
        int64_t device_offset = req_to_device_buffer_ptr[
            req_idx * (int64_t)device_buffer_row_stride + device_slot_in_buffer];
        if (device_offset < 0 || device_offset >= device_pool_rows) {
            continue;
        }

        __gm__ uint8_t* host_kv_row = host_kv_cache_ptr
            + ((int64_t)layer_id * host_entries + host_offset) * kv_row_bytes;

        __gm__ uint8_t* device_k_row = reinterpret_cast<__gm__ uint8_t*>(device_k_buffer)
            + device_offset * k_row_bytes;
        __gm__ uint8_t* device_v_row = reinterpret_cast<__gm__ uint8_t*>(device_v_buffer)
            + device_offset * v_row_bytes;

        {
            GlobalTensor<uint8_t> host_kv_gm, device_k_gm, device_v_gm;
            host_kv_gm.SetGlobalBuffer(host_kv_row);
            device_k_gm.SetGlobalBuffer(device_k_row);
            device_v_gm.SetGlobalBuffer(device_v_row);

            LocalTensor<uint8_t> local_kv = inQue.AllocTensor<uint8_t>();
            DataCopy(local_kv, host_kv_gm, kv_row_bytes);
            inQue.EnQue(local_kv);

            LocalTensor<uint8_t> local_kv_deq = inQue.DeQue<uint8_t>();
            LocalTensor<uint8_t> local_kv_out = outQue.AllocTensor<uint8_t>();
            DataCopy(local_kv_out, local_kv_deq, kv_row_bytes);
            outQue.EnQue(local_kv_out);
            inQue.FreeTensor(local_kv_deq);

            LocalTensor<uint8_t> local_kv_out_deq = outQue.DeQue<uint8_t>();
            DataCopy(device_k_gm, local_kv_out_deq, k_row_bytes);
            DataCopy(device_v_gm, local_kv_out_deq[k_row_bytes], v_row_bytes);
            outQue.FreeTensor(local_kv_out_deq);
        }
    }
}
