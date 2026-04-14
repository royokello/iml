#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace iml::cuda::int8_dequant {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetMinBlocksPerSm = 8;
constexpr int kGt1060_6gbSmCount = 10;

// Launches the int8 -> fp16 blockwise dequant kernel.
// quantized: Device pointer to the contiguous int8 input values.
// scales: Device pointer to one fp16 scale per quantization block.
// out: Device pointer to the fp16 output buffer.
// original_numel: Number of scalar output values to reconstruct.
// block_size: Quantization block size; must be 32, 64, or 128.
// stream: CUDA stream used for the kernel launch.
void launch_int8_dequant(
    const int8_t* quantized,
    const __half* scales,
    __half* out,
    int64_t original_numel,
    int block_size,
    cudaStream_t stream = nullptr
);

}  // namespace iml::cuda::int8_dequant
