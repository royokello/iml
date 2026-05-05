#include "dequantize_from_symmetric_med.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_symmetric_med {

namespace {

__device__ __forceinline__ int sign_extend_int6(uint32_t code) {
    return static_cast<int>(code & 0x1FU) - static_cast<int>(code & 0x20U);
}

__device__ __forceinline__ int load_signed_int6(
    const uint32_t* words,
    int value_index
) {
    const int bit_offset = value_index * kBitsPerWeight;
    const int word_offset = bit_offset >> 5;
    const int word_shift = bit_offset & 31;

    uint32_t code = words[word_offset] >> word_shift;
    if (word_shift > 26) {
        code |= words[word_offset + 1] << (32 - word_shift);
    }

    return sign_extend_int6(code & 0x3FU);
}

template<int kSubBlocksPerSuper>
__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_symmetric_med_kernel(
    const int32_t* __restrict__ packed,
    const int8_t* __restrict__ sub_scales,
    const __half* __restrict__ super_scales,
    __half* __restrict__ out,
    int original_numel
) {
    const int sub_block_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_out_index = sub_block_index * kSubBlockSize;
    if (base_out_index >= original_numel) {
        return;
    }

    const int super_block = sub_block_index / kSubBlocksPerSuper;
    const int sub_block = sub_block_index - (super_block * kSubBlocksPerSuper);
    const int scale_index = super_block * kSubBlocksPerSuper + sub_block;
    const int packed_word_index = scale_index * kPackedWordsPerSubBlock;

    uint32_t words[kPackedWordsPerSubBlock];
    words[0] = static_cast<uint32_t>(packed[packed_word_index]);
    words[1] = static_cast<uint32_t>(packed[packed_word_index + 1]);
    words[2] = static_cast<uint32_t>(packed[packed_word_index + 2]);

    const float scale =
        static_cast<float>(sub_scales[scale_index]) * __half2float(super_scales[super_block]);

    #pragma unroll
    for (int value_index = 0; value_index < kSubBlockSize; value_index += 2) {
        const int out_index = base_out_index + value_index;
        if (out_index >= original_numel) {
            break;
        }

        const float lo_q = static_cast<float>(load_signed_int6(words, value_index));
        const __half lo = __float2half(lo_q * scale);

        if (out_index + 1 < original_numel) {
            const float hi_q = static_cast<float>(load_signed_int6(words, value_index + 1));
            const __half hi = __float2half(hi_q * scale);
            reinterpret_cast<__half2*>(out)[out_index >> 1] = __halves2half2(lo, hi);
        } else {
            out[out_index] = lo;
        }
    }
}

}  // namespace

void launch_dequantize_from_symmetric_med(
    const int32_t* packed,
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

    constexpr int kSubBlocksPerSuper128 = 128 / kSubBlockSize;
    constexpr int kSubBlocksPerSuper256 = 256 / kSubBlockSize;
    const int numel = static_cast<int>(original_numel);
    const int num_sub_blocks = (numel + kSubBlockSize - 1) / kSubBlockSize;
    const int blocks = (num_sub_blocks + kThreadsPerBlock - 1) / kThreadsPerBlock;

    if (super_block_size == 128) {
        dequantize_from_symmetric_med_kernel<kSubBlocksPerSuper128>
            <<<blocks, kThreadsPerBlock, 0, stream>>>(
                packed,
                sub_scales,
                super_scales,
                out,
                numel
            );
    } else if (super_block_size == 256) {
        dequantize_from_symmetric_med_kernel<kSubBlocksPerSuper256>
            <<<blocks, kThreadsPerBlock, 0, stream>>>(
                packed,
                sub_scales,
                super_scales,
                out,
                numel
            );
    }
}

}  // namespace iml::cuda::dequantize_from_symmetric_med
