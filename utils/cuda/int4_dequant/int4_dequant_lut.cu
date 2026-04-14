#include "int4_dequant_lut.cuh"

#include <cstdint>

namespace iml::cuda::int4_dequant {

namespace {

__constant__ __half2 kByteToHalf2Lut[256];

inline int8_t sign_extend_int4(uint8_t value) {
    return (value & 0x08) ? static_cast<int8_t>(value | 0xF0) : static_cast<int8_t>(value);
}

// GTX 1060 6GB is GP106 (cc 6.1): 10 SMs, 2048 resident threads/SM, 64 warps/SM.
// 256 threads/block gives 8 warps/block and 8 resident blocks/SM at full thread occupancy.
__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void byte_lut_dequant_kernel(
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int original_numel,
    int block_size
) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pairs = (original_numel + 1) >> 1;
    if (pair_index >= num_pairs) {
        return;
    }

    const int out_index = pair_index << 1;
    const int block_index = out_index / block_size;
    const uint8_t byte = packed[pair_index];
    const __half2 decoded = kByteToHalf2Lut[byte];
    const __half scale = scales[block_index];
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

void launch_byte_lut_dequant(
    const uint8_t* packed,
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
    const int num_pairs = (numel + 1) >> 1;
    const int blocks = (num_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;

    byte_lut_dequant_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        packed,
        scales,
        out,
        numel,
        block_size
    );
}

}  // namespace iml::cuda::int4_dequant
