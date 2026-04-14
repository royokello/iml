#include "int8_dequant.cuh"

#include <cstdint>

namespace iml::cuda::int8_dequant {

namespace {

// GTX 1060 6GB is GP106 (cc 6.1): 10 SMs, 2048 resident threads/SM, 64 warps/SM.
// 256 threads/block gives 8 warps/block and 8 resident blocks/SM at full thread occupancy.
__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void int8_dequant_kernel(
    const int8_t* __restrict__ quantized,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int original_numel,
    int block_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= original_numel) {
        return;
    }

    const int block_index = index / block_size;
    const __half value = __int2half_rn(static_cast<int>(quantized[index]));
    out[index] = __hmul(value, scales[block_index]);
}

}  // namespace

void launch_int8_dequant(
    const int8_t* quantized,
    const __half* scales,
    __half* out,
    int64_t original_numel,
    int block_size,
    cudaStream_t stream
) {
    if (original_numel <= 0) {
        return;
    }

    const int numel = static_cast<int>(original_numel);
    const int blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;

    int8_dequant_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        quantized,
        scales,
        out,
        numel,
        block_size
    );
}

}  // namespace iml::cuda::int8_dequant
