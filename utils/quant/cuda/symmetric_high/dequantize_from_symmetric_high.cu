#include "dequantize_from_symmetric_high.cuh"

#include <cstdint>

namespace iml::cuda::dequantize_from_symmetric_high {

namespace {

__device__ __forceinline__ void store_scaled_pair(
    const int8_t* __restrict__ qweight,
    __half* __restrict__ out,
    int qweight_index,
    int out_index,
    int column,
    int row_size,
    int original_numel,
    float scale
) {
    if (column >= row_size || out_index >= original_numel) {
        return;
    }

    const __half lo = __float2half(static_cast<float>(qweight[qweight_index]) * scale);
    if (column + 1 < row_size && out_index + 1 < original_numel) {
        const __half hi = __float2half(static_cast<float>(qweight[qweight_index + 1]) * scale);
        if ((out_index & 1) == 0) {
            reinterpret_cast<__half2*>(out)[out_index >> 1] = __halves2half2(lo, hi);
        } else {
            out[out_index] = lo;
            out[out_index + 1] = hi;
        }
    } else {
        out[out_index] = lo;
    }
}

__launch_bounds__(kThreadsPerBlock, kTargetMinBlocksPerSm)
__global__ void dequantize_from_symmetric_high_kernel(
    const int8_t* __restrict__ qweight,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int num_blocks,
    int original_numel,
    int row_size,
    int blocks_per_row
) {
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int quant_block_in_warp = lane / kQuantBlockLanes;
    const int lane_in_quant_block = lane & (kQuantBlockLanes - 1);
    const int block_index =
        blockIdx.x * kQuantBlocksPerThreadBlock +
        warp_in_block * kQuantBlocksPerWarp +
        quant_block_in_warp;
    if (block_index >= num_blocks) {
        return;
    }

    const int row = block_index / blocks_per_row;
    const int block_in_row = block_index - (row * blocks_per_row);
    const int source_lane = quant_block_in_warp * kQuantBlockLanes;
    const unsigned int mask = ((1U << kQuantBlockLanes) - 1U) << source_lane;
    float scale = lane_in_quant_block == 0 ? __half2float(scales[block_index]) : 0.0f;
    scale = __shfl_sync(mask, scale, source_lane);

    const int column = block_in_row * kBlockSize + lane_in_quant_block * kValuesPerLane;
    const int out_index = row * row_size + column;
    const int qweight_index = block_index * kBlockSize + lane_in_quant_block * kValuesPerLane;
    store_scaled_pair(
        qweight,
        out,
        qweight_index,
        out_index,
        column,
        row_size,
        original_numel,
        scale
    );
    store_scaled_pair(
        qweight,
        out,
        qweight_index + 2,
        out_index + 2,
        column + 2,
        row_size,
        original_numel,
        scale
    );
}

}  // namespace

void launch_dequantize_from_symmetric_high(
    const int8_t* qweight,
    const __half* scales,
    __half* out,
    int64_t original_numel,
    int row_size,
    int blocks_per_row,
    cudaStream_t stream
) {
    if (original_numel <= 0) {
        return;
    }

    const int numel = static_cast<int>(original_numel);
    const int row_count = (numel + row_size - 1) / row_size;
    const int num_quant_blocks = row_count * blocks_per_row;
    const int blocks =
        (num_quant_blocks + kQuantBlocksPerThreadBlock - 1) / kQuantBlocksPerThreadBlock;

    dequantize_from_symmetric_high_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        qweight,
        scales,
        out,
        num_quant_blocks,
        numel,
        row_size,
        blocks_per_row
    );
}

}  // namespace iml::cuda::dequantize_from_symmetric_high
