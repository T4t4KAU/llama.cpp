#include "fork-attn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#    include <mma.h>

namespace {

constexpr int plan_header_size = 8;
constexpr int plan_magic       = 0x4641544e;
constexpr int kv_per_split     = 64;
constexpr int max_splits       = 16;

template <typename T> __device__ __forceinline__ float fork_to_float(T value);

template <> __device__ __forceinline__ float fork_to_float(half value) {
    return __half2float(value);
}

template <> __device__ __forceinline__ float fork_to_float(nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float warp_sum(float value) {
#    pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

template <typename T, int D>
__global__ void fork_attn_partial_wmma(const char * __restrict__ q,
                                       const char * __restrict__ k,
                                       const char * __restrict__ v,
                                       const int32_t * __restrict__ plan,
                                       float * __restrict__ partial,
                                       float2 * __restrict__ meta,
                                       float  scale,
                                       int    n_queries,
                                       int    n_heads,
                                       int    n_kv_heads,
                                       int    n_kv,
                                       int    n_splits,
                                       size_t nbq1,
                                       size_t nbq2,
                                       size_t nbk1,
                                       size_t nbk2,
                                       size_t nbv1,
                                       size_t nbv2) {
    using namespace nvcuda;

    const int split       = blockIdx.x % n_splits;
    const int kv_head     = blockIdx.x / n_splits;
    const int gqa         = n_heads / n_kv_heads;
    const int n_rows      = n_queries * gqa;
    const int n_row_tiles = (n_rows + 15) / 16;
    const int n_col_tiles = D / 16;
    const int warp        = threadIdx.x / WARP_SIZE;
    const int lane        = threadIdx.x % WARP_SIZE;

    if (plan[0] != plan_magic || plan[2] == 0) {
        return;
    }

    const int   common_len    = plan[4];
    const int * common        = plan + plan_header_size;
    const int * private_lens  = common + n_kv;
    const int * private_cells = private_lens + n_queries;

    extern __shared__ unsigned char raw_smem[];
    T *                             sq        = (T *) raw_smem;
    T *                             sk        = sq + n_row_tiles * 16 * D;
    T *                             sv        = sk + 16 * D;
    T *                             sp        = sv + 16 * D;
    float *                         scores    = (float *) (sp + n_row_tiles * 16 * 16);
    float *                         sout      = scores + n_row_tiles * 16 * 16;
    float *                         row_max_s = sout + n_row_tiles * 16 * D;
    float *                         row_sum_s = row_max_s + n_rows;
    int *                           valid     = (int *) (row_sum_s + n_rows);

    for (int linear = threadIdx.x; linear < n_row_tiles * 16 * D; linear += blockDim.x) {
        const int row = linear / D;
        const int d   = linear % D;
        if (row < n_rows) {
            const int query  = row / gqa;
            const int q_head = kv_head * gqa + row % gqa;
            sq[linear]       = (T) (*(const float *) (q + query * nbq1 + q_head * nbq2 + d * sizeof(float)));
        } else {
            sq[linear] = (T) 0.0f;
        }
    }
    __syncthreads();

    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        row_max_s[row] = -FLT_MAX;
        row_sum_s[row] = 0.0f;
    }
    __syncthreads();

    const int common_split_len = split < common_len ? (common_len - split + n_splits - 1) / n_splits : 0;
    const int common_tiles     = (common_split_len + 15) / 16;

    for (int tile = 0; tile < common_tiles; ++tile) {
        for (int linear = threadIdx.x; linear < 16 * D; linear += blockDim.x) {
            const int t = linear / D;
            const int d = linear % D;
            const int p = split + (tile * 16 + t) * n_splits;
            if (p < common_len) {
                const int cell = common[p];
                sk[linear]     = *(const T *) (k + cell * nbk1 + kv_head * nbk2 + d * sizeof(T));
            } else {
                sk[linear] = (T) 0.0f;
            }
        }
        if (threadIdx.x < 16) {
            valid[threadIdx.x] = split + (tile * 16 + threadIdx.x) * n_splits < common_len;
        }
        __syncthreads();

        if (warp < n_row_tiles) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> aq;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> bk;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>           acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int d = 0; d < D; d += 16) {
                wmma::load_matrix_sync(aq, sq + warp * 16 * D + d, D);
                wmma::load_matrix_sync(bk, sk + d, D);
                wmma::mma_sync(acc, aq, bk, acc);
            }
            wmma::store_matrix_sync(scores + warp * 16 * 16, acc, 16, wmma::mem_row_major);
        }
        __syncthreads();

        if (lane == 0) {
            for (int row = warp; row < n_rows; row += blockDim.x / WARP_SIZE) {
                float row_max = row_max_s[row];
                float row_sum = row_sum_s[row];
                for (int t = 0; t < 16; ++t) {
                    if (!valid[t]) {
                        continue;
                    }
                    const float score    = scores[row * 16 + t] * scale;
                    const float next_max = fmaxf(row_max, score);
                    const float alpha    = row_sum == 0.0f ? 0.0f : expf(row_max - next_max);
                    row_sum              = row_sum * alpha + expf(score - next_max);
                    row_max              = next_max;
                }
                row_max_s[row] = row_max;
                row_sum_s[row] = row_sum;
            }
        }
        __syncthreads();
    }

    for (int row = warp; row < n_rows; row += blockDim.x / WARP_SIZE) {
        const int    query  = row / gqa;
        const int    q_head = kv_head * gqa + row % gqa;
        const char * q_row  = q + query * nbq1 + q_head * nbq2;
        for (int p = split; p < private_lens[query]; p += n_splits) {
            const int cell  = private_cells[query * n_kv + p];
            float     score = 0.0f;
#    pragma unroll
            for (int j = 0; j < D / WARP_SIZE; ++j) {
                const int d = lane + j * WARP_SIZE;
                score += *(const float *) (q_row + d * sizeof(float)) *
                         fork_to_float(*(const T *) (k + cell * nbk1 + kv_head * nbk2 + d * sizeof(T)));
            }
            score = warp_sum(score) * scale;
            if (lane == 0) {
                float       row_max  = row_max_s[row];
                float       row_sum  = row_sum_s[row];
                const float next_max = fmaxf(row_max, score);
                const float alpha    = row_sum == 0.0f ? 0.0f : expf(row_max - next_max);
                row_sum              = row_sum * alpha + expf(score - next_max);
                row_max              = next_max;
                row_max_s[row]       = row_max;
                row_sum_s[row]       = row_sum;
            }
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_acc;
    wmma::fill_fragment(out_acc, 0.0f);
    const bool output_warp     = warp < n_row_tiles * n_col_tiles;
    const int  output_row_tile = warp / n_col_tiles;
    const int  output_col_tile = warp % n_col_tiles;

    for (int tile = 0; tile < common_tiles; ++tile) {
        for (int linear = threadIdx.x; linear < 16 * D; linear += blockDim.x) {
            const int t = linear / D;
            const int d = linear % D;
            const int p = split + (tile * 16 + t) * n_splits;
            if (p < common_len) {
                const int cell = common[p];
                sk[linear]     = *(const T *) (k + cell * nbk1 + kv_head * nbk2 + d * sizeof(T));
                sv[linear]     = *(const T *) (v + cell * nbv1 + kv_head * nbv2 + d * sizeof(T));
            } else {
                sk[linear] = (T) 0.0f;
                sv[linear] = (T) 0.0f;
            }
        }
        if (threadIdx.x < 16) {
            valid[threadIdx.x] = split + (tile * 16 + threadIdx.x) * n_splits < common_len;
        }
        __syncthreads();

        if (warp < n_row_tiles) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> aq;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> bk;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>           acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int d = 0; d < D; d += 16) {
                wmma::load_matrix_sync(aq, sq + warp * 16 * D + d, D);
                wmma::load_matrix_sync(bk, sk + d, D);
                wmma::mma_sync(acc, aq, bk, acc);
            }
            wmma::store_matrix_sync(scores + warp * 16 * 16, acc, 16, wmma::mem_row_major);
        }
        __syncthreads();

        if (lane < 16) {
            for (int row = warp; row < n_rows; row += blockDim.x / WARP_SIZE) {
                sp[row * 16 + lane] =
                    valid[lane] ? (T) expf(scores[row * 16 + lane] * scale - row_max_s[row]) : (T) 0.0f;
            }
        }
        for (int linear = threadIdx.x + n_rows * 16; linear < n_row_tiles * 16 * 16; linear += blockDim.x) {
            sp[linear] = (T) 0.0f;
        }
        __syncthreads();

        if (output_warp) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> ap;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bv;
            wmma::load_matrix_sync(ap, sp + output_row_tile * 16 * 16, 16);
            wmma::load_matrix_sync(bv, sv + output_col_tile * 16, D);
            wmma::mma_sync(out_acc, ap, bv, out_acc);
        }
        __syncthreads();
    }

    if (output_warp) {
        wmma::store_matrix_sync(sout + output_row_tile * 16 * D + output_col_tile * 16, out_acc, D,
                                wmma::mem_row_major);
    }
    __syncthreads();

    for (int row_out = warp; row_out < n_rows; row_out += blockDim.x / WARP_SIZE) {
        const int    query  = row_out / gqa;
        const int    q_head = kv_head * gqa + row_out % gqa;
        const char * q_row  = q + query * nbq1 + q_head * nbq2;
        float        out[D / WARP_SIZE];
#    pragma unroll
        for (int j = 0; j < D / WARP_SIZE; ++j) {
            const int d = lane + j * WARP_SIZE;
            out[j]      = sout[row_out * D + d];
        }

        for (int p = split; p < private_lens[query]; p += n_splits) {
            const int cell  = private_cells[query * n_kv + p];
            float     score = 0.0f;
#    pragma unroll
            for (int j = 0; j < D / WARP_SIZE; ++j) {
                const int d = lane + j * WARP_SIZE;
                score += *(const float *) (q_row + d * sizeof(float)) *
                         fork_to_float(*(const T *) (k + cell * nbk1 + kv_head * nbk2 + d * sizeof(T)));
            }
            score              = warp_sum(score) * scale;
            const float weight = expf(__shfl_sync(0xffffffff, score, 0) - row_max_s[row_out]);
#    pragma unroll
            for (int j = 0; j < D / WARP_SIZE; ++j) {
                const int d = lane + j * WARP_SIZE;
                out[j] += weight * fork_to_float(*(const T *) (v + cell * nbv1 + kv_head * nbv2 + d * sizeof(T)));
            }
        }

        const size_t row = (size_t(split) * n_queries + query) * n_heads + q_head;
        if (lane == 0) {
            meta[row] = make_float2(row_max_s[row_out], row_sum_s[row_out]);
        }
#    pragma unroll
        for (int j = 0; j < D / WARP_SIZE; ++j) {
            partial[row * D + lane + j * WARP_SIZE] = out[j];
        }
    }
}

template <int D>
__global__ void fork_attn_merge(const float * __restrict__ partial,
                                const float2 * __restrict__ meta,
                                float * __restrict__ dst,
                                int n_queries,
                                int n_heads,
                                int n_splits) {
    const int row_out = blockIdx.x;
    const int query   = row_out % n_queries;
    const int q_head  = row_out / n_queries;

    __shared__ float global_max;
    __shared__ float global_sum;
    if (threadIdx.x == 0) {
        float row_max = -FLT_MAX;
        for (int split = 0; split < n_splits; ++split) {
            const size_t row = (size_t(split) * n_queries + query) * n_heads + q_head;
            row_max          = fmaxf(row_max, meta[row].x);
        }

        float row_sum = 0.0f;
        for (int split = 0; split < n_splits; ++split) {
            const size_t row = (size_t(split) * n_queries + query) * n_heads + q_head;
            if (meta[row].y > 0.0f) {
                row_sum += expf(meta[row].x - row_max) * meta[row].y;
            }
        }
        global_max = row_max;
        global_sum = row_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float value = 0.0f;
        for (int split = 0; split < n_splits; ++split) {
            const size_t row = (size_t(split) * n_queries + query) * n_heads + q_head;
            if (meta[row].y > 0.0f) {
                value += expf(meta[row].x - global_max) * partial[row * D + d];
            }
        }
        dst[d + D * (q_head + n_heads * query)] = value / global_sum;
    }
}

template <typename T, int D> void launch_fork_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * plan = dst->src[5];

    const int    n_queries      = q->ne[1];
    const int    n_heads        = q->ne[2];
    const int    n_kv_heads     = k->ne[2];
    const int    n_kv           = k->ne[1];
    const int    gqa            = n_heads / n_kv_heads;
    const int    n_splits       = std::min(max_splits, std::max(1, (n_kv + kv_per_split - 1) / kv_per_split));
    const size_t n_partial_rows = size_t(n_splits) * n_queries * n_heads;

    ggml_cuda_pool_alloc<float>  partial(ctx.pool(), n_partial_rows * D);
    ggml_cuda_pool_alloc<float2> meta(ctx.pool(), n_partial_rows);

    CUDA_CHECK(cudaMemsetAsync(meta.get(), 0, n_partial_rows * sizeof(float2), ctx.stream()));

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    const int    n_rows      = n_queries * gqa;
    const int    n_row_tiles = (n_rows + 15) / 16;
    const int    nthreads    = std::max(std::min(n_rows, 16), n_row_tiles * (D / 16)) * WARP_SIZE;
    const size_t smem_t      = (size_t(n_row_tiles) * 16 * D + 2 * 16 * D + size_t(n_row_tiles) * 16 * 16) * sizeof(T);
    const size_t smem_f = (size_t(n_row_tiles) * 16 * 16 + size_t(n_row_tiles) * 16 * D + 2 * n_rows) * sizeof(float);
    const size_t smem   = smem_t + smem_f + 16 * sizeof(int);
    fork_attn_partial_wmma<T, D><<<n_splits * n_kv_heads, nthreads, smem, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data, (const int32_t *) plan->data,
        partial.get(), meta.get(), scale, n_queries, n_heads, n_kv_heads, n_kv, n_splits, q->nb[1], q->nb[2], k->nb[1],
        k->nb[2], v->nb[1], v->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    fork_attn_merge<D><<<n_queries * n_heads, D, 0, ctx.stream()>>>(partial.get(), meta.get(), (float *) dst->data,
                                                                    n_queries, n_heads, n_splits);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

#endif

bool ggml_cuda_fork_attn_supported(int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!dst->src[5]) {
        return false;
    }

    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * plan = dst->src[5];
    const int           cc   = ggml_cuda_info().devices[device].cc;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && q->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) &&
           k->type == v->type && (q->ne[0] == 64 || q->ne[0] == 128) && k->ne[0] == q->ne[0] && v->ne[0] == q->ne[0] &&
           k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && q->ne[1] >= 2 && q->ne[1] <= 8 && q->ne[3] == 1 &&
           k->ne[3] == 1 && v->ne[3] == 1 && q->ne[2] % k->ne[2] == 0 && q->ne[1] * (q->ne[2] / k->ne[2]) <= 32 &&
           q->nb[0] == sizeof(float) && k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
           plan->type == GGML_TYPE_I32 && ggml_is_contiguous(plan) &&
           plan->ne[0] >= plan_header_size + k->ne[1] + q->ne[1] + q->ne[1] * k->ne[1];
#endif
}

void ggml_cuda_fork_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ForkAttention is available only on CUDA");
#else
    GGML_ASSERT(ggml_cuda_fork_attn_supported(ctx.device, dst));
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];

    if (k->type == GGML_TYPE_F16) {
        if (q->ne[0] == 64) {
            launch_fork_attn<half, 64>(ctx, dst);
        } else {
            launch_fork_attn<half, 128>(ctx, dst);
        }
    } else {
        if (q->ne[0] == 64) {
            launch_fork_attn<nv_bfloat16, 64>(ctx, dst);
        } else {
            launch_fork_attn<nv_bfloat16, 128>(ctx, dst);
        }
    }
#endif
}
