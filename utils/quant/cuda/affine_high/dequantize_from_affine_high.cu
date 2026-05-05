#include "dequantize_from_affine_high.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_affine_high {

namespace {

inline int packed_words_for_values(
    int value_count,
    int bits
) {
    return (value_count * bits + 31) / 32;
}

__device__ __forceinline__ int load_unsigned_bits(
    const uint32_t* words,
    int value_index,
    int bits
) {
    const int bit_offset = value_index * bits;
    const int word_offset = bit_offset >> 5;
    const int word_shift = bit_offset & 31;
    const uint32_t mask = (1U << bits) - 1U;

    uint32_t code = words[word_offset] >> word_shift;
    if (word_shift + bits > 32) {
        code |= words[word_offset + 1] << (32 - word_shift);
    }

    return static_cast<int>(code & mask);
}

__device__ __forceinline__ int load_signed_bits(
    const uint32_t* words,
    int value_index,
    int bits
) {
    const int code = load_unsigned_bits(words, value_index, bits);
    const int sign_bit = 1 << (bits - 1);
    return (code ^ sign_bit) - sign_bit;
}

__device__ __forceinline__ int load_uint5(
    const uint32_t* words,
    int value_index
) {
    const int bit_offset = value_index * kBitsPerWeight;
    const int word_offset = bit_offset >> 5;
    const int word_shift = bit_offset & 31;

    uint32_t code = words[word_offset] >> word_shift;
    if (word_shift > 27) {
        code |= words[word_offset + 1] << (32 - word_shift);
    }

    return static_cast<int>(code & 0x1FU);
}

__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_affine_high_kernel(
    const int32_t* __restrict__ packed,
    const int32_t* __restrict__ sub_scales,
    const int32_t* __restrict__ sub_mins,
    const __half* __restrict__ super_scales,
    const __half* __restrict__ super_mins,
    __half* __restrict__ out,
    int original_numel,
    int super_block_size,
    int sub_blocks_per_super,
    int metadata_words_per_super
) {
    const int sub_block_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_out_index = sub_block_index * kSubBlockSize;
    if (base_out_index >= original_numel) {
        return;
    }

    const int super_block = base_out_index / super_block_size;
    const int local_index = base_out_index - (super_block * super_block_size);
    const int sub_block = local_index / kSubBlockSize;
    const int meta_word_index = super_block * metadata_words_per_super;
    const int packed_word_index =
        (super_block * sub_blocks_per_super + sub_block) * kPackedWordsPerSubBlock;

    uint32_t words[kPackedWordsPerSubBlock];
    words[0] = static_cast<uint32_t>(packed[packed_word_index]);
    words[1] = static_cast<uint32_t>(packed[packed_word_index + 1]);
    words[2] = static_cast<uint32_t>(packed[packed_word_index + 2]);
    words[3] = static_cast<uint32_t>(packed[packed_word_index + 3]);
    words[4] = static_cast<uint32_t>(packed[packed_word_index + 4]);

    const int scale_code = load_unsigned_bits(
        reinterpret_cast<const uint32_t*>(sub_scales + meta_word_index),
        sub_block,
        kScaleBits
    );
    const int min_code = load_signed_bits(
        reinterpret_cast<const uint32_t*>(sub_mins + meta_word_index),
        sub_block,
        kMinBits
    );

    const float scale = static_cast<float>(scale_code) * __half2float(super_scales[super_block]);
    const float min_value = static_cast<float>(min_code) * __half2float(super_mins[super_block]);

    #pragma unroll
    for (int value_index = 0; value_index < kSubBlockSize; value_index += 2) {
        const int out_index = base_out_index + value_index;
        if (out_index >= original_numel) {
            break;
        }

        const float lo_q = static_cast<float>(load_uint5(words, value_index));
        const __half lo = __float2half((lo_q * scale) + min_value);

        if (out_index + 1 < original_numel) {
            const float hi_q = static_cast<float>(load_uint5(words, value_index + 1));
            const __half hi = __float2half((hi_q * scale) + min_value);
            reinterpret_cast<__half2*>(out)[out_index >> 1] = __halves2half2(lo, hi);
        } else {
            out[out_index] = lo;
        }
    }
}

}  // namespace

void launch_dequantize_from_affine_high(
    const int32_t* packed,
    const int32_t* sub_scales,
    const int32_t* sub_mins,
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
    const int num_sub_blocks = (numel + kSubBlockSize - 1) / kSubBlockSize;
    const int blocks = (num_sub_blocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int sub_blocks_per_super = super_block_size / kSubBlockSize;
    const int metadata_words_per_super = packed_words_for_values(sub_blocks_per_super, kScaleBits);

    dequantize_from_affine_high_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        packed,
        sub_scales,
        sub_mins,
        super_scales,
        super_mins,
        out,
        numel,
        super_block_size,
        sub_blocks_per_super,
        metadata_words_per_super
    );
}

}  // namespace iml::cuda::dequantize_from_affine_high
