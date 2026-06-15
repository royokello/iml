#include "gemm_fused.cuh"

namespace iml::cuda::gemm {

// ── Launch macro ────────────────────────────────────────────────────────
// Instantiates the kernel template and creates a launch wrapper.

#define DEFINE_LAUNCH(NAME, BM_VAL, BN_VAL, BK_VAL)                        \
void launch_##NAME(                                                         \
    const int8_t* q_act, const int16_t* bsums, const float* act_scales,     \
    const void* q_weight, const float* sub_scales, const float* sub_mins,   \
    const float* super_scales, const float* super_mins,                     \
    __half* output,                                                         \
    int M, int N, int K,                                                    \
    int super_block_size, int sub_block_size, WeightFmt fmt,                \
    cudaStream_t stream                                                     \
) {                                                                         \
    int bk_subs = BK_VAL / sub_block_size;                                  \
    int smem_bytes = BM_VAL * BK_VAL                                        \
                   + BK_VAL * BN_VAL                                        \
                   + BN_VAL * bk_subs * (int)sizeof(float) * 2;            \
    dim3 grid(                                                              \
        (N + BN_VAL - 1) / BN_VAL,                                         \
        (M + BM_VAL - 1) / BM_VAL                                           \
    );                                                                      \
    fused_gemm_kernel<BM_VAL, BN_VAL, BK_VAL>                               \
    <<<grid, kThreadsPerBlock, smem_bytes, stream>>>(                       \
        q_act, bsums, act_scales,                                           \
        q_weight, sub_scales, sub_mins, super_scales, super_mins,           \
        output, M, N, K,                                                    \
        super_block_size, sub_block_size, fmt                               \
    );                                                                      \
}

DEFINE_LAUNCH(balanced_64_64_128,   64,  64, 128)
DEFINE_LAUNCH(tall_m_128_64_128,   128,  64, 128)
DEFINE_LAUNCH(wide_n_64_128_128,    64, 128, 128)
DEFINE_LAUNCH(large_128_128_64,    128, 128,  64)
DEFINE_LAUNCH(small_m_32_64_128,    32,  64, 128)

#undef DEFINE_LAUNCH

// ── Launch dispatch ─────────────────────────────────────────────────────
// Selects the best tile config based on problem dimensions.
// This is the default heuristic; the autotuner overrides it.

LaunchConfig select_launch(int M, int N, int K) {
    // Tall matrices (M >> N): use tall_m config
    if (M > 4 * N && N <= 64) {
        return { launch_tall_m_128_64_128, 128, 64, 128 };
    }
    // Wide matrices (N >> M): use wide_n config
    if (N > 4 * M) {
        return { launch_wide_n_64_128_128, 64, 128, 128 };
    }
    // Small M: use small_m config (good occupancy)
    if (M <= 32) {
        return { launch_small_m_32_64_128, 32, 64, 128 };
    }
    // Large M and N: use large tile
    if (M >= 2048 && N >= 128) {
        return { launch_large_128_128_64, 128, 128, 64 };
    }
    // Default: balanced
    return { launch_balanced_64_64_128, 64, 64, 128 };
}

} // namespace iml::cuda::gemm
