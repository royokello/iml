#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_affine_high {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kSubBlockSize = 32;
constexpr int kBitsPerWeight = 5;
constexpr int kPackedWordsPerSubBlock = 5;
constexpr int kWeightsPerPackedSubBlock = 32;

void launch_dequantize_from_affine_high(
    const int32_t* packed,
    const int8_t* sub_scales,
    const int8_t* sub_mins,
    const __half* super_scales,
    const __half* super_mins,
    __half* out,
    int64_t original_numel,
    int super_block_size,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::dequantize_from_affine_high
