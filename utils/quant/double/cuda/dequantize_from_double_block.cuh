#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_double_block {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kGt1060_6gbSmCount = 10;

constexpr int kSuperBlockSize = 256;
constexpr int kSubBlockSize = 32;
constexpr int kSubBlocksPerSuper = kSuperBlockSize / kSubBlockSize;

constexpr int kOutputsPerThread = 2;
constexpr int kThreadsPerSubBlock = kSubBlockSize / kOutputsPerThread;
constexpr int kOutputsPerBlock = kThreadsPerBlock * kOutputsPerThread;
constexpr int kSuperBlocksPerCta = kOutputsPerBlock / kSuperBlockSize;
constexpr int kScaleSlotsPerCta = kOutputsPerBlock / kSubBlockSize;

void init_byte_to_half2_lut(cudaStream_t stream = nullptr);

void launch_dequantize_from_double_block(
    const int8_t* packed,
    const int8_t* sub_scales,
    const __half* super_scales,
    __half* out,
    int64_t original_numel,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::dequantize_from_double_block
