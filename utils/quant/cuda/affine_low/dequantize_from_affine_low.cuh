#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_affine_low {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kSubBlockSize = 16;
constexpr int kWeightBits = 2;
constexpr int kScaleBits = 4;
constexpr int kMinBits = 4;
constexpr int kPackedWordsPerWeightSubBlock = 1;

void launch_dequantize_from_affine_low(
    const int32_t* packed,
    const int32_t* sub_scales,
    const int32_t* sub_mins,
    const __half* super_scales,
    const __half* super_mins,
    __half* out,
    int64_t original_numel,
    int super_block_size,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::dequantize_from_affine_low
