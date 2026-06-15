#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "weight_unpack.cuh"

namespace iml::cuda::gemm {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kActSubBlockSize = 16;

// ── Launch wrapper type (visible to MSVC) ───────────────────────────────

struct LaunchConfig {
    void (*fn)(
        const int8_t*, const int16_t*, const float*,
        const void*, const float*, const float*,
        const float*, const float*,
        __half*, int, int, int, int, int, WeightFmt, cudaStream_t
    );
    int bm, bn, bk;
};

LaunchConfig select_launch(int M, int N, int K);
void launch_balanced_64_64_128(
    const int8_t*, const int16_t*, const float*,
    const void*, const float*, const float*,
    const float*, const float*,
    __half*, int, int, int, int, int, WeightFmt, cudaStream_t);
void launch_tall_m_128_64_128(
    const int8_t*, const int16_t*, const float*,
    const void*, const float*, const float*,
    const float*, const float*,
    __half*, int, int, int, int, int, WeightFmt, cudaStream_t);
void launch_wide_n_64_128_128(
    const int8_t*, const int16_t*, const float*,
    const void*, const float*, const float*,
    const float*, const float*,
    __half*, int, int, int, int, int, WeightFmt, cudaStream_t);
void launch_large_128_128_64(
    const int8_t*, const int16_t*, const float*,
    const void*, const float*, const float*,
    const float*, const float*,
    __half*, int, int, int, int, int, WeightFmt, cudaStream_t);
void launch_small_m_32_64_128(
    const int8_t*, const int16_t*, const float*,
    const void*, const float*, const float*,
    const float*, const float*,
    __half*, int, int, int, int, int, WeightFmt, cudaStream_t);

// ── Fused quantized GEMM kernel (NVCC only) ─────────────────────────────
//
// Template parameters:
//   BM  - tile rows along M (output rows per block)
//   BN  - tile cols along N (output cols per block)
//   BK  - tile size along K (must divide super_block_size at runtime)
//
// Grid: (ceil(N / BN), ceil(M / BM))
// Block: kThreadsPerBlock threads
//
// Computes:
//   output[m][n] = sum_{sb} act_scale[m][sb] * (
//       sum_{sub in sb} dot(q_act[m][sub], q_weight[n][sub]) * sub_scale[n][sub]
//                       + bsums[m][sub] * sub_min[n][sub]
//   )
//
#ifdef __CUDACC__

template <int BM, int BN, int BK>
__launch_bounds__(kThreadsPerBlock, 2)
__global__ void fused_gemm_kernel(
    const int8_t*  __restrict__ q_act,
    const int16_t* __restrict__ bsums,
    const float*   __restrict__ act_scales,
    const void*    __restrict__ q_weight,
    const float*   __restrict__ sub_scales,
    const float*   __restrict__ sub_mins,
    const float*   __restrict__ super_scales,
    const float*   __restrict__ super_mins,
    __half*        __restrict__ output,
    int M, int N, int K,
    int super_block_size,
    int sub_block_size,
    WeightFmt fmt
) {
    int m_start = blockIdx.y * BM;
    int n_start = blockIdx.x * BN;
    int m_tile = min(BM, M - m_start);
    int n_tile = min(BN, N - n_start);
    if (m_tile <= 0 || n_tile <= 0) return;

    int subs_per_super = super_block_size / sub_block_size;
    int bk_subs = BK / sub_block_size;
    int total_super_blocks = K / super_block_size;
    int total_act_sub_blocks = K / kActSubBlockSize;
    int act_subs_per_wsub = sub_block_size / kActSubBlockSize;

    // ── Shared memory ──────────────────────────────────────────────────
    extern __shared__ int8_t shared[];
    int8_t* smem_act = shared;
    int8_t* smem_w   = smem_act + BM * BK;
    float*  smem_sub_scale = reinterpret_cast<float*>(smem_w + BK * BN);
    float*  smem_sub_min   = smem_sub_scale + BN * bk_subs;

    // ── Per-thread accumulators ────────────────────────────────────────
    int outputs_per_thread = (BM * BN + kThreadsPerBlock - 1) / kThreadsPerBlock;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < outputs_per_thread; i++) acc[i] = 0.0f;

    // ── Main K-loop: iterate over super-blocks ─────────────────────────
    for (int sb = 0; sb < total_super_blocks; sb++) {
        int k_sb_start = sb * super_block_size;

        // Accumulate per-super-block partial before applying act_scale
        float sb_partial[64];
        #pragma unroll
        for (int i = 0; i < outputs_per_thread; i++) sb_partial[i] = 0.0f;

        // ── Process super-block in BK-sized tiles ─────────────────────
        for (int kk = 0; kk < super_block_size; kk += BK) {
            int k_global = k_sb_start + kk;
            int current_bk = min(BK, super_block_size - kk);

            // ── Load activation tile into smem ────────────────────────
            int act_total = m_tile * current_bk;
            for (int idx = threadIdx.x; idx < act_total; idx += kThreadsPerBlock) {
                int mi = idx / current_bk;
                int ki = idx % current_bk;
                if (mi < m_tile && ki < current_bk) {
                    smem_act[mi * BK + ki] = q_act[(m_start + mi) * K + k_global + ki];
                }
            }

            // ── Load + unpack weight tile into smem ───────────────────
            // Each thread handles one or more (output_col, sub_block) pairs
            int total_unpack = bk_subs * n_tile;
            for (int idx = threadIdx.x; idx < total_unpack; idx += kThreadsPerBlock) {
                int n = idx / bk_subs;
                int s = idx % bk_subs;       // sub-block index within BK
                int sub_k_start = s * sub_block_size;
                int sub_in_super = (kk + sub_k_start) / sub_block_size;
                if (sub_in_super >= subs_per_super) continue;

                int8_t q[32];
                float scale_val, min_val;
                int super_idx = (n_start + n) * total_super_blocks + sb;
                unpack_by_fmt(fmt, q_weight, sub_scales, sub_mins,
                              super_scales, super_mins,
                              super_idx, sub_in_super, subs_per_super,
                              q, &scale_val, &min_val);

                int limit = min(sub_block_size, current_bk - sub_k_start);
                for (int i = 0; i < limit; i++) {
                    smem_w[(sub_k_start + i) * BN + n] = q[i];
                }
                smem_sub_scale[n * bk_subs + s] = scale_val;
                smem_sub_min[n * bk_subs + s] = min_val;
            }

            __syncthreads();

            // ── Combined: load act AND compute dot in one pass ─────────
            // Each thread reads smem_act + smem_w for its output positions
            for (int t = 0; t < outputs_per_thread; t++) {
                int flat = threadIdx.x + t * kThreadsPerBlock;
                int mi = flat / BN;
                int ni = flat % BN;
                if (mi >= m_tile || ni >= n_tile) continue;

                for (int s = 0; s < bk_subs; s++) {
                    int sub_k_start = s * sub_block_size;
                    int limit = min(sub_block_size, current_bk - sub_k_start);
                    if (limit <= 0) break;

                    int dot = 0;
                    for (int i = 0; i < limit; i++) {
                        int a = smem_act[mi * BK + sub_k_start + i];
                        int w = smem_w[(sub_k_start + i) * BN + ni];
                        dot += a * w;
                    }

                    int global_act_sub = (k_global + sub_k_start) / kActSubBlockSize;
                    float bsum = 0.0f;
                    for (int j = 0; j < act_subs_per_wsub; j++) {
                        bsum += (float)bsums[(m_start + mi) * total_act_sub_blocks + global_act_sub + j];
                    }
                    float ws = smem_sub_scale[ni * bk_subs + s];
                    float wm = smem_sub_min[ni * bk_subs + s];

                    sb_partial[t] += (float)dot * ws + bsum * wm;
                }
            }

            __syncthreads();
        }

        // ── Apply activation scale and accumulate to final output ──────
        for (int t = 0; t < outputs_per_thread; t++) {
            int flat = threadIdx.x + t * kThreadsPerBlock;
            int mi = flat / BN;
            int ni = flat % BN;
            if (mi >= m_tile || ni >= n_tile) continue;

            float ascale = act_scales[(m_start + mi) * total_super_blocks + sb];
            acc[t] += sb_partial[t] * ascale;
        }
    }

    // ── Write final output ─────────────────────────────────────────────
    for (int t = 0; t < outputs_per_thread; t++) {
        int flat = threadIdx.x + t * kThreadsPerBlock;
        int mi = flat / BN;
        int ni = flat % BN;
        if (mi >= m_tile || ni >= n_tile) continue;
        output[(m_start + mi) * N + n_start + ni] = __float2half(acc[t]);
    }
}

#endif // __CUDACC__

} // namespace iml::cuda::gemm
