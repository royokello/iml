#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_intermediate {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;

void launch_dequantize_from_intermediate(
    const int8_t* qweight,
    const float* super_scales,
    __half* out,
    int64_t original_numel,
    int super_block_size,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::dequantize_from_intermediate
