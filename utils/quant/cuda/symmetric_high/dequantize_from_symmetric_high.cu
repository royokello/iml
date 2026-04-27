#include "dequantize_from_symmetric_high.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_symmetric_high {

namespace {

__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_symmetric_high_kernel(
    const int8_t* __restrict__ qweight,
    const int8_t* __restrict__ sub_scales,
    const __half* __restrict__ super_scales,
    __half* __restrict__ out,
    int original_numel,
    int super_block_size,
    int sub_blocks_per_super
) {
    const int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index >= original_numel) {
        return;
    }

    const int super_block = out_index / super_block_size;
    const int local_index = out_index - (super_block * super_block_size);
    const int sub_block = local_index / kSubBlockSize;
    const int scale_index = super_block * sub_blocks_per_super + sub_block;

    const float scale =
        static_cast<float>(sub_scales[scale_index]) * __half2float(super_scales[super_block]);
    const float q = static_cast<float>(qweight[out_index]);
    out[out_index] = __float2half(q * scale);
}

}  // namespace

void launch_dequantize_from_symmetric_high(
    const int8_t* qweight,
    const int8_t* sub_scales,
    const __half* super_scales,
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
    const int sub_blocks_per_super = super_block_size / kSubBlockSize;

    dequantize_from_symmetric_high_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        qweight,
        sub_scales,
        super_scales,
        out,
        numel,
        super_block_size,
        sub_blocks_per_super
    );
}

}  // namespace iml::cuda::dequantize_from_symmetric_high
