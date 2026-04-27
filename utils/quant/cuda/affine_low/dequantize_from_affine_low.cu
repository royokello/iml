#include "dequantize_from_affine_low.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_affine_low {

namespace {

__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_affine_low_kernel(
    const uint8_t* __restrict__ packed,
    const int8_t* __restrict__ sub_scales,
    const int8_t* __restrict__ sub_mins,
    const __half* __restrict__ super_scales,
    const __half* __restrict__ super_mins,
    __half* __restrict__ out,
    int original_numel,
    int super_block_size,
    int sub_blocks_per_super
) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_index = pair_index * kWeightsPerPackedByte;
    if (out_index >= original_numel) {
        return;
    }

    const int super_block = out_index / super_block_size;
    const int local_index = out_index - (super_block * super_block_size);
    const int sub_block = local_index / kSubBlockSize;
    const int scale_index = super_block * sub_blocks_per_super + sub_block;

    const float scale =
        static_cast<float>(sub_scales[scale_index]) * __half2float(super_scales[super_block]);
    const float min_value =
        static_cast<float>(sub_mins[scale_index]) * __half2float(super_mins[super_block]);

    const uint8_t byte = packed[pair_index];
    const float lo_q = static_cast<float>(byte & 0x0F);
    const float hi_q = static_cast<float>((byte >> 4) & 0x0F);

    const __half lo = __float2half((lo_q * scale) + min_value);

    if (out_index + 1 < original_numel) {
        const __half hi = __float2half((hi_q * scale) + min_value);
        reinterpret_cast<__half2*>(out)[pair_index] = __halves2half2(lo, hi);
    } else {
        out[out_index] = lo;
    }
}

}  // namespace

void launch_dequantize_from_affine_low(
    const uint8_t* packed,
    const int8_t* sub_scales,
    const int8_t* sub_mins,
    const __half* super_scales,
    const __half* super_mins,
    __half* out,
    int64_t original_numel,
    int super_block_size,
    cudaStream_t stream
) {
    if (original_numel <= 0) {
        return;
    }

    const int numel = static_cast<int>(original_numel);
    const int num_pairs = (numel + kWeightsPerPackedByte - 1) / kWeightsPerPackedByte;
    const int blocks = (num_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int sub_blocks_per_super = super_block_size / kSubBlockSize;

    dequantize_from_affine_low_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        packed,
        sub_scales,
        sub_mins,
        super_scales,
        super_mins,
        out,
        numel,
        super_block_size,
        sub_blocks_per_super
    );
}

}  // namespace iml::cuda::dequantize_from_affine_low
