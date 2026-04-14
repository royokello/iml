#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace iml::cuda::int4_dequant {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kGt1060_6gbSmCount = 10;

void init_byte_to_half2_lut(cudaStream_t stream = nullptr);

void launch_byte_lut_dequant(
    const uint8_t* packed,
    const __half* scales,
    __half* out,
    int64_t original_numel,
    int block_size,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::int4_dequant
