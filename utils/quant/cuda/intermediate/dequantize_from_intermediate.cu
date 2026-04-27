#include "dequantize_from_intermediate.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_intermediate {

namespace {

__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_intermediate_kernel(
    const int8_t* __restrict__ qweight,
    const float* __restrict__ super_scales,
    __half* __restrict__ out,
    int original_numel,
    int super_block_size
) {
    const int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index >= original_numel) {
        return;
    }

    const int super_block = out_index / super_block_size;
    const float q = static_cast<float>(qweight[out_index]);
    out[out_index] = __float2half(q * super_scales[super_block]);
}

}  // namespace

void launch_dequantize_from_intermediate(
    const int8_t* qweight,
    const float* super_scales,
    __half* out,
    int64_t original_numel,
    int super_block_size,
    cudaStream_t stream
) {
    if (original_numel <= 0) {
        return;
    }

    const int numel = static_cast<int>(original_numel);
    const int blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;

    dequantize_from_intermediate_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        qweight,
        super_scales,
        out,
        numel,
        super_block_size
    );
}

}  // namespace iml::cuda::dequantize_from_intermediate
