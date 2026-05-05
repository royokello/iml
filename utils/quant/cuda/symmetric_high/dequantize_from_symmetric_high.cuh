#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_symmetric_high {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kWarpSize = 32;
constexpr int kBlockSize = 32;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kQuantBlockLanes = 8;
constexpr int kValuesPerLane = kBlockSize / kQuantBlockLanes;
constexpr int kQuantBlocksPerWarp = kWarpSize / kQuantBlockLanes;
constexpr int kQuantBlocksPerThreadBlock = kWarpsPerBlock * kQuantBlocksPerWarp;

void launch_dequantize_from_symmetric_high(
    const int8_t* qweight,
    const __half* scales,
    __half* out,
    int64_t original_numel,
    int row_size,
    int blocks_per_row,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::dequantize_from_symmetric_high
