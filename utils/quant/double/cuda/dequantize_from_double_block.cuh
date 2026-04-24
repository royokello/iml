#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace iml::cuda::dequantize_from_double_block {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kGt1060_6gbSmCount = 10;

constexpr int kSuperBlockSize = 128;
constexpr int kSubBlockSize = 16;
constexpr int kSubBlocksPerSuper = kSuperBlockSize / kSubBlockSize;

constexpr int kOutputsPerThread = 2;
constexpr int kThreadsPerSubBlock = kSubBlockSize / kOutputsPerThread;
constexpr int kOutputsPerBlock = kThreadsPerBlock * kOutputsPerThread;
constexpr int kSuperBlocksPerCta = kOutputsPerBlock / kSuperBlockSize;
constexpr int kScaleSlotsPerCta = kOutputsPerBlock / kSubBlockSize;

static_assert(kSuperBlockSize % kSubBlockSize == 0, "super block size must divide evenly into sub blocks");
static_assert(kSubBlockSize % kOutputsPerThread == 0, "sub block size must align with the vectorized decode width");
static_assert(kOutputsPerBlock % kSubBlockSize == 0, "CTA output span must cover whole sub blocks");

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
