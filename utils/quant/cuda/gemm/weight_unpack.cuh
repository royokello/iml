#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace iml::cuda::gemm {

// ── Format enum (visible to host compiler) ──────────────────────────────

enum class WeightFmt : int {
    SymHigh, SymMed, SymLow,
    AffHigh, AffMed, AffLow
};

// ── Device-only unpack helpers ──────────────────────────────────────────
#ifdef __CUDACC__

__device__ __forceinline__ int load_unsigned_bits(
    const uint32_t* words, int value_index, int bits
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
    const uint32_t* words, int value_index, int bits
) {
    const int code = load_unsigned_bits(words, value_index, bits);
    const int sign_bit = 1 << (bits - 1);
    return (code ^ sign_bit) - sign_bit;
}

__device__ __forceinline__ int sign_extend_int6(uint32_t code) {
    return static_cast<int>(code & 0x1FU) - static_cast<int>(code & 0x20U);
}

__device__ __forceinline__ int sign_extend_int3(uint32_t code) {
    return static_cast<int>(code & 0x3U) - static_cast<int>(code & 0x4U);
}

// ── sym-high ── int8 weights, 32-element sub-block, no hierarchy ─────────

__device__ void unpack_sym_high(
    const void* q_weight, const void* sub_scales, const void* sub_mins,
    const void* super_scales, const void* super_mins,
    int super_block_idx, int sub_idx, int subs_per_super,
    int8_t* q_out, float* scale_out, float* min_out
) {
    int block_idx = super_block_idx * subs_per_super + sub_idx;
    const int8_t* w = static_cast<const int8_t*>(q_weight);
    const __half* s = static_cast<const __half*>(sub_scales);
    for (int i = 0; i < 32; i++) {
        q_out[i] = w[block_idx * 32 + i];
    }
    *scale_out = __half2float(s[block_idx]);
    *min_out = 0.0f;
}

// ── sym-med ── 6-bit signed, 16-weight sub-block, 3 int32 words/sub-block

__device__ void unpack_sym_med(
    const void* q_weight, const void* sub_scales, const void* sub_mins,
    const void* super_scales, const void* super_mins,
    int super_block_idx, int sub_idx, int subs_per_super,
    int8_t* q_out, float* scale_out, float* min_out
) {
    constexpr int kSub = 16;
    constexpr int kWords = 3;
    const int32_t* w = static_cast<const int32_t*>(q_weight);
    const int8_t* ss = static_cast<const int8_t*>(sub_scales);
    const __half* sps = static_cast<const __half*>(super_scales);

    int sb_global = super_block_idx * subs_per_super + sub_idx;
    int base = sb_global * kWords;
    uint32_t w0 = static_cast<uint32_t>(w[base]);
    uint32_t w1 = static_cast<uint32_t>(w[base + 1]);
    uint32_t w2 = static_cast<uint32_t>(w[base + 2]);

    for (int i = 0; i < kSub; i++) {
        int off = i * 6;
        int wo = off >> 5;
        int sh = off & 31;
        uint32_t code = (wo == 0 ? w0 : (wo == 1 ? w1 : w2)) >> sh;
        if (sh > 26) {
            code |= (wo == 0 ? w1 : w2) << (32 - sh);
        }
        q_out[i] = static_cast<int8_t>(sign_extend_int6(code & 0x3FU));
    }

    *scale_out = static_cast<float>(ss[sb_global])
               * __half2float(sps[super_block_idx]);
    *min_out = 0.0f;
}

// ── sym-low ── 3-bit signed, 16-weight sub-block, 2 int32 words/sub-block

__device__ void unpack_sym_low(
    const void* q_weight, const void* sub_scales, const void* sub_mins,
    const void* super_scales, const void* super_mins,
    int super_block_idx, int sub_idx, int subs_per_super,
    int8_t* q_out, float* scale_out, float* min_out
) {
    constexpr int kSub = 16;
    constexpr int kWords = 2;
    const int32_t* w = static_cast<const int32_t*>(q_weight);
    const int32_t* pss = static_cast<const int32_t*>(sub_scales);
    const __half* sps = static_cast<const __half*>(super_scales);

    int sb_global = super_block_idx * subs_per_super + sub_idx;
    int base = sb_global * kWords;
    uint32_t w0 = static_cast<uint32_t>(w[base]);
    uint32_t w1 = static_cast<uint32_t>(w[base + 1]);

    for (int i = 0; i < kSub; i++) {
        int off = i * 3;
        int wo = off >> 5;
        int sh = off & 31;
        uint32_t code = (wo == 0 ? w0 : w1) >> sh;
        if (sh + 3 > 32) {
            code |= (wo == 0 ? w1 : w0) << (32 - sh);
        }
        q_out[i] = static_cast<int8_t>(sign_extend_int3(code & 0x7U));
    }

    int meta_words = (subs_per_super * 6 + 31) / 32;
    uint32_t sc = load_unsigned_bits(
        reinterpret_cast<const uint32_t*>(pss + super_block_idx * meta_words),
        sub_idx, 6);
    *scale_out = static_cast<float>(sc) * __half2float(sps[super_block_idx]);
    *min_out = 0.0f;
}

// ── aff-high ── 5-bit unsigned, 32-weight sub-block, 5 int32 words

__device__ void unpack_aff_high(
    const void* q_weight, const void* sub_scales, const void* sub_mins,
    const void* super_scales, const void* super_mins,
    int super_block_idx, int sub_idx, int subs_per_super,
    int8_t* q_out, float* scale_out, float* min_out
) {
    constexpr int kSub = 32;
    constexpr int kBits = 5;
    constexpr int kWords = 5;
    const int32_t* w = static_cast<const int32_t*>(q_weight);
    const int32_t* ss = static_cast<const int32_t*>(sub_scales);
    const int32_t* sm = static_cast<const int32_t*>(sub_mins);
    const __half* sps = static_cast<const __half*>(super_scales);
    const __half* spm = static_cast<const __half*>(super_mins);

    int sb_global = super_block_idx * subs_per_super + sub_idx;
    int base = sb_global * kWords;

    for (int i = 0; i < kSub; i++) {
        int off = i * kBits;
        int wo = off >> 5;
        int sh = off & 31;
        uint32_t word = static_cast<uint32_t>(w[base + wo]);
        uint32_t code = word >> sh;
        if (sh + kBits > 32) {
            code |= static_cast<uint32_t>(w[base + wo + 1]) << (32 - sh);
        }
        q_out[i] = static_cast<int8_t>(code & 0x1FU);
    }

    int meta_words = (subs_per_super * 6 + 31) / 32;
    int sc = load_unsigned_bits(
        reinterpret_cast<const uint32_t*>(ss + super_block_idx * meta_words),
        sub_idx, 6);
    int mn = load_signed_bits(
        reinterpret_cast<const uint32_t*>(sm + super_block_idx * meta_words),
        sub_idx, 6);
    *scale_out = static_cast<float>(sc) * __half2float(sps[super_block_idx]);
    *min_out = static_cast<float>(mn) * __half2float(spm[super_block_idx]);
}

// ── aff-med ── 4-bit unsigned, 32-weight sub-block, 4 int32 words

__device__ void unpack_aff_med(
    const void* q_weight, const void* sub_scales, const void* sub_mins,
    const void* super_scales, const void* super_mins,
    int super_block_idx, int sub_idx, int subs_per_super,
    int8_t* q_out, float* scale_out, float* min_out
) {
    constexpr int kSub = 32;
    constexpr int kBits = 4;
    constexpr int kWords = 4;
    const int32_t* w = static_cast<const int32_t*>(q_weight);
    const int32_t* ss = static_cast<const int32_t*>(sub_scales);
    const int32_t* sm = static_cast<const int32_t*>(sub_mins);
    const __half* sps = static_cast<const __half*>(super_scales);
    const __half* spm = static_cast<const __half*>(super_mins);

    int sb_global = super_block_idx * subs_per_super + sub_idx;
    int base = sb_global * kWords;

    for (int i = 0; i < kSub; i++) {
        int off = i * kBits;
        int wo = off >> 5;
        int sh = off & 31;
        uint32_t word = static_cast<uint32_t>(w[base + wo]);
        uint32_t code = word >> sh;
        if (sh + kBits > 32) {
            code |= static_cast<uint32_t>(w[base + wo + 1]) << (32 - sh);
        }
        q_out[i] = static_cast<int8_t>(code & 0xFU);
    }

    int meta_words = (subs_per_super * 6 + 31) / 32;
    int sc = load_unsigned_bits(
        reinterpret_cast<const uint32_t*>(ss + super_block_idx * meta_words),
        sub_idx, 6);
    int mn = load_signed_bits(
        reinterpret_cast<const uint32_t*>(sm + super_block_idx * meta_words),
        sub_idx, 6);
    *scale_out = static_cast<float>(sc) * __half2float(sps[super_block_idx]);
    *min_out = static_cast<float>(mn) * __half2float(spm[super_block_idx]);
}

// ── aff-low ── 2-bit unsigned, 16-weight sub-block, 1 int32 word

__device__ void unpack_aff_low(
    const void* q_weight, const void* sub_scales, const void* sub_mins,
    const void* super_scales, const void* super_mins,
    int super_block_idx, int sub_idx, int subs_per_super,
    int8_t* q_out, float* scale_out, float* min_out
) {
    constexpr int kSub = 16;
    constexpr int kBits = 2;
    const int32_t* w = static_cast<const int32_t*>(q_weight);
    const int32_t* ss = static_cast<const int32_t*>(sub_scales);
    const int32_t* sm = static_cast<const int32_t*>(sub_mins);
    const __half* sps = static_cast<const __half*>(super_scales);
    const __half* spm = static_cast<const __half*>(super_mins);

    int sb_global = super_block_idx * subs_per_super + sub_idx;
    int32_t word = w[sb_global];

    for (int i = 0; i < kSub; i++) {
        q_out[i] = static_cast<int8_t>((word >> (i * kBits)) & 0x03);
    }

    int meta_words = (subs_per_super * 4 + 31) / 32;
    int sc = load_unsigned_bits(
        reinterpret_cast<const uint32_t*>(ss + super_block_idx * meta_words),
        sub_idx, 4);
    int mn = load_signed_bits(
        reinterpret_cast<const uint32_t*>(sm + super_block_idx * meta_words),
        sub_idx, 4);
    *scale_out = static_cast<float>(sc) * __half2float(sps[super_block_idx]);
    *min_out = static_cast<float>(mn) * __half2float(spm[super_block_idx]);
}

// ── Tag dispatch ────────────────────────────────────────────────────────

__device__ __forceinline__ void unpack_by_fmt(
    WeightFmt fmt,
    const void* qw, const void* ss, const void* sm,
    const void* sps, const void* spm,
    int sb_idx, int sub, int subs_per,
    int8_t* q_out, float* s_out, float* m_out
) {
    switch (fmt) {
        case WeightFmt::SymHigh:
            unpack_sym_high(qw, ss, sm, sps, spm, sb_idx, sub, subs_per, q_out, s_out, m_out); break;
        case WeightFmt::SymMed:
            unpack_sym_med(qw, ss, sm, sps, spm, sb_idx, sub, subs_per, q_out, s_out, m_out); break;
        case WeightFmt::SymLow:
            unpack_sym_low(qw, ss, sm, sps, spm, sb_idx, sub, subs_per, q_out, s_out, m_out); break;
        case WeightFmt::AffHigh:
            unpack_aff_high(qw, ss, sm, sps, spm, sb_idx, sub, subs_per, q_out, s_out, m_out); break;
        case WeightFmt::AffMed:
            unpack_aff_med(qw, ss, sm, sps, spm, sb_idx, sub, subs_per, q_out, s_out, m_out); break;
        case WeightFmt::AffLow:
            unpack_aff_low(qw, ss, sm, sps, spm, sb_idx, sub, subs_per, q_out, s_out, m_out); break;
    }
}

#endif // __CUDACC__

} // namespace iml::cuda::gemm
