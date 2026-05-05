#include "dequantize_from_affine_low.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_affine_low {

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

__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_affine_low_kernel(
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
    const int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index >= original_numel) {
        return;
    }

    const int super_block = out_index / super_block_size;
    const int local_index = out_index - (super_block * super_block_size);
    const int sub_block = local_index / kSubBlockSize;
    const int value_index = local_index - (sub_block * kSubBlockSize);
    const int meta_word_index = super_block * metadata_words_per_super;

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

    const int packed_index =
        (super_block * sub_blocks_per_super + sub_block) * kPackedWordsPerWeightSubBlock;
    const int32_t word = packed[packed_index];
    const int shift = value_index * kWeightBits;
    const float q = static_cast<float>((word >> shift) & 0x03);

    out[out_index] = __float2half((q * scale) + min_value);
}

}  // namespace

void launch_dequantize_from_affine_low(
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
    const int blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int sub_blocks_per_super = super_block_size / kSubBlockSize;
    const int metadata_words_per_super = packed_words_for_values(sub_blocks_per_super, kScaleBits);

    dequantize_from_affine_low_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
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

}  // namespace iml::cuda::dequantize_from_affine_low
