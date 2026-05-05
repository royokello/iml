#include "dequantize_from_symmetric_low.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_symmetric_low {

namespace {

__device__ __forceinline__ int sign_extend_int3(uint32_t code) {
    return static_cast<int>(code & 0x3U) - static_cast<int>(code & 0x4U);
}

__device__ __forceinline__ int load_signed_int3(
    const uint32_t* words,
    int value_index
) {
    const int bit_offset = value_index * kBitsPerWeight;
    const int word_offset = bit_offset >> 5;
    const int word_shift = bit_offset & 31;

    uint32_t code = words[word_offset] >> word_shift;
    if (word_shift + kBitsPerWeight > 32) {
        code |= words[word_offset + 1] << (32 - word_shift);
    }

    return sign_extend_int3(code & 0x7U);
}

__device__ __forceinline__ uint32_t load_unsigned_int6(
    const int32_t* words,
    int value_index
) {
    const int bit_offset = value_index * kBitsPerSubScale;
    const int word_offset = bit_offset >> 5;
    const int word_shift = bit_offset & 31;

    uint32_t code = static_cast<uint32_t>(words[word_offset]) >> word_shift;
    if (word_shift + kBitsPerSubScale > 32) {
        code |= static_cast<uint32_t>(words[word_offset + 1]) << (32 - word_shift);
    }

    return code & 0x3FU;
}

template<int kSubBlocksPerSuper, int kPackedScaleWordsPerSuper>
__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_symmetric_low_kernel(
    const int32_t* __restrict__ packed,
    const int32_t* __restrict__ packed_sub_scales,
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
    const int packed_scale_word_index = super_block * kPackedScaleWordsPerSuper;

    uint32_t words[kPackedWordsPerSubBlock];
    words[0] = static_cast<uint32_t>(packed[packed_word_index]);
    words[1] = static_cast<uint32_t>(packed[packed_word_index + 1]);

    const uint32_t sub_scale = load_unsigned_int6(
        packed_sub_scales + packed_scale_word_index,
        sub_block
    );
    const float scale =
        static_cast<float>(sub_scale) * __half2float(super_scales[super_block]);

    #pragma unroll
    for (int value_index = 0; value_index < kSubBlockSize; value_index += 2) {
        const int out_index = base_out_index + value_index;
        if (out_index >= original_numel) {
            break;
        }

        const float lo_q = static_cast<float>(load_signed_int3(words, value_index));
        const __half lo = __float2half(lo_q * scale);

        if (out_index + 1 < original_numel) {
            const float hi_q = static_cast<float>(load_signed_int3(words, value_index + 1));
            const __half hi = __float2half(hi_q * scale);
            reinterpret_cast<__half2*>(out)[out_index >> 1] = __halves2half2(lo, hi);
        } else {
            out[out_index] = lo;
        }
    }
}

}  // namespace

void launch_dequantize_from_symmetric_low(
    const int32_t* packed,
    const int32_t* packed_sub_scales,
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
    constexpr int kPackedScaleWordsPerSuper128 =
        (kSubBlocksPerSuper128 * kBitsPerSubScale + 31) / 32;
    constexpr int kPackedScaleWordsPerSuper256 =
        (kSubBlocksPerSuper256 * kBitsPerSubScale + 31) / 32;
    const int numel = static_cast<int>(original_numel);
    const int num_sub_blocks = (numel + kSubBlockSize - 1) / kSubBlockSize;
    const int blocks = (num_sub_blocks + kThreadsPerBlock - 1) / kThreadsPerBlock;

    if (super_block_size == 128) {
        dequantize_from_symmetric_low_kernel<
            kSubBlocksPerSuper128,
            kPackedScaleWordsPerSuper128
        ><<<blocks, kThreadsPerBlock, 0, stream>>>(
            packed,
            packed_sub_scales,
            super_scales,
            out,
            numel
        );
    } else if (super_block_size == 256) {
        dequantize_from_symmetric_low_kernel<
            kSubBlocksPerSuper256,
            kPackedScaleWordsPerSuper256
        ><<<blocks, kThreadsPerBlock, 0, stream>>>(
            packed,
            packed_sub_scales,
            super_scales,
            out,
            numel
        );
    }
}

}  // namespace iml::cuda::dequantize_from_symmetric_low
