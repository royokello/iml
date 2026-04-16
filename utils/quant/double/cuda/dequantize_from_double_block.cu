#include "dequantize_from_double_block.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_double_block {

namespace {

__constant__ __half2 kByteToHalf2Lut[256];

inline int8_t sign_extend_int4(uint8_t value) {
    return (value & 0x08) ? static_cast<int8_t>(value | 0xF0) : static_cast<int8_t>(value);
}

// GTX 1060 6GB is GP106 (cc 6.1): 10 SMs, 2048 resident threads/SM, 64 warps/SM.
// 256 threads/block gives 8 warps/block and 8 resident blocks/SM at full thread occupancy.
__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_double_block_kernel(
    const int8_t* __restrict__ packed,
    const int8_t* __restrict__ sub_scales,
    const __half* __restrict__ super_scales,
    __half* __restrict__ out,
    int original_numel
) {
    __shared__ __half cta_effective_scales[kScaleSlotsPerCta];

    const int num_super_blocks = (original_numel + kSuperBlockSize - 1) / kSuperBlockSize;
    const int base_super_block = blockIdx.x * kSuperBlocksPerCta;

    if (threadIdx.x < kScaleSlotsPerCta) {
        const int scale_slot = threadIdx.x;
        const int global_super_block = base_super_block + (scale_slot / kSubBlocksPerSuper);
        const int local_sub_block = scale_slot % kSubBlocksPerSuper;

        __half effective_scale = __float2half(0.0f);
        if (global_super_block < num_super_blocks) {
            const __half super_scale = super_scales[global_super_block];
            const int8_t sub_scale = sub_scales[global_super_block * kSubBlocksPerSuper + local_sub_block];
            effective_scale = __hmul(
                super_scale,
                __int2half_rn(static_cast<int>(sub_scale))
            );
        }
        cta_effective_scales[scale_slot] = effective_scale;
    }

    __syncthreads();

    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pairs = (original_numel + 1) >> 1;
    if (pair_index >= num_pairs) {
        return;
    }

    const int out_index = pair_index << 1;
    const uint8_t byte = static_cast<uint8_t>(packed[pair_index]);
    const __half2 decoded = kByteToHalf2Lut[byte];
    const __half scale = cta_effective_scales[threadIdx.x / kThreadsPerSubBlock];
    const __half2 scaled = __hmul2(decoded, __halves2half2(scale, scale));

    if (out_index + 1 < original_numel) {
        reinterpret_cast<__half2*>(out)[pair_index] = scaled;
    } else {
        out[out_index] = __hmul(__low2half(decoded), scale);
    }
}

}  // namespace

void init_byte_to_half2_lut(cudaStream_t stream) {
    __half2 host_lut[256];

    for (int byte = 0; byte < 256; ++byte) {
        const uint8_t lo_nibble = static_cast<uint8_t>(byte & 0x0F);
        const uint8_t hi_nibble = static_cast<uint8_t>((byte >> 4) & 0x0F);

        const int lo = static_cast<int>(sign_extend_int4(lo_nibble));
        const int hi = static_cast<int>(sign_extend_int4(hi_nibble));

        host_lut[byte] = __halves2half2(__int2half_rn(lo), __int2half_rn(hi));
    }

    cudaMemcpyToSymbolAsync(
        kByteToHalf2Lut,
        host_lut,
        sizeof(host_lut),
        0,
        cudaMemcpyHostToDevice,
        stream
    );
}

void launch_dequantize_from_double_block(
    const int8_t* packed,
    const int8_t* sub_scales,
    const __half* super_scales,
    __half* out,
    int64_t original_numel,
    cudaStream_t stream
) {
    if (original_numel <= 0) {
        return;
    }

    const int numel = static_cast<int>(original_numel);
    const int num_pairs = (numel + 1) >> 1;
    const int blocks = (num_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;

    dequantize_from_double_block_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        packed,
        sub_scales,
        super_scales,
        out,
        numel
    );
}

}  // namespace iml::cuda::dequantize_from_double_block
