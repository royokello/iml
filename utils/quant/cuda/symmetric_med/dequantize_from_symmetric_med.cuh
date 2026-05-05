#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_symmetric_med {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kSubBlockSize = 16;
constexpr int kBitsPerWeight = 6;
constexpr int kPackedWordsPerSubBlock = 3;
constexpr int kWeightsPerPackedSubBlock = 16;

void launch_dequantize_from_symmetric_med(
    const int32_t* packed,
    const int8_t* sub_scales,
    const __half* super_scales,
    __half* out,
    int64_t original_numel,
    int super_block_size,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::dequantize_from_symmetric_med
