#include <cstdint>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dequantize_from_affine_high.cuh"

namespace {

int64_t packed_words_for_values(int64_t value_count, int64_t bits) {
    return (value_count * bits + 31) / 32;
}

void validate_inputs(
    const torch::Tensor& packed,
    const torch::Tensor& sub_scales,
    const torch::Tensor& sub_mins,
    const torch::Tensor& super_scales,
    const torch::Tensor& super_mins,
    const torch::Tensor& out,
    int64_t original_numel,
    int64_t super_block_size
) {
    if (!packed.is_cuda()) {
        throw std::invalid_argument("packed must be a CUDA tensor");
    }
    if (!sub_scales.is_cuda()) {
        throw std::invalid_argument("sub_scales must be a CUDA tensor");
    }
    if (!sub_mins.is_cuda()) {
        throw std::invalid_argument("sub_mins must be a CUDA tensor");
    }
    if (!super_scales.is_cuda()) {
        throw std::invalid_argument("super_scales must be a CUDA tensor");
    }
    if (!super_mins.is_cuda()) {
        throw std::invalid_argument("super_mins must be a CUDA tensor");
    }
    if (!out.is_cuda()) {
        throw std::invalid_argument("out must be a CUDA tensor");
    }

    if (packed.scalar_type() != torch::kInt32) {
        throw std::invalid_argument("packed must have dtype torch.int32");
    }
    if (sub_scales.scalar_type() != torch::kInt32) {
        throw std::invalid_argument("sub_scales must have dtype torch.int32");
    }
    if (sub_mins.scalar_type() != torch::kInt32) {
        throw std::invalid_argument("sub_mins must have dtype torch.int32");
    }
    if (super_scales.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("super_scales must have dtype torch.float16");
    }
    if (super_mins.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("super_mins must have dtype torch.float16");
    }
    if (out.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("out must have dtype torch.float16");
    }

    if (!packed.is_contiguous()) {
        throw std::invalid_argument("packed must be contiguous");
    }
    if (!sub_scales.is_contiguous()) {
        throw std::invalid_argument("sub_scales must be contiguous");
    }
    if (!sub_mins.is_contiguous()) {
        throw std::invalid_argument("sub_mins must be contiguous");
    }
    if (!super_scales.is_contiguous()) {
        throw std::invalid_argument("super_scales must be contiguous");
    }
    if (!super_mins.is_contiguous()) {
        throw std::invalid_argument("super_mins must be contiguous");
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("out must be contiguous");
    }

    const int device = packed.get_device();
    if (
        sub_scales.get_device() != device ||
        sub_mins.get_device() != device ||
        super_scales.get_device() != device ||
        super_mins.get_device() != device ||
        out.get_device() != device
    ) {
        throw std::invalid_argument("all tensors must be on the same CUDA device");
    }

    if (original_numel < 0) {
        throw std::invalid_argument("original_numel must be non-negative");
    }
    if (super_block_size <= 0) {
        throw std::invalid_argument("super_block_size must be positive");
    }
    if (super_block_size % iml::cuda::dequantize_from_affine_high::kSubBlockSize != 0) {
        throw std::invalid_argument("super_block_size must be divisible by 32");
    }
    if (out.numel() < original_numel) {
        throw std::invalid_argument("out is smaller than original_numel");
    }

    if (packed.numel() % iml::cuda::dequantize_from_affine_high::kPackedWordsPerSubBlock != 0) {
        throw std::invalid_argument("packed must contain 5 int32 words per sub-block");
    }

    const int64_t num_super_blocks =
        (original_numel + super_block_size - 1) / super_block_size;
    const int64_t sub_blocks_per_super =
        super_block_size / iml::cuda::dequantize_from_affine_high::kSubBlockSize;
    const int64_t metadata_words_per_super = packed_words_for_values(
        sub_blocks_per_super,
        iml::cuda::dequantize_from_affine_high::kScaleBits
    );
    const int64_t expected_sub_metadata = num_super_blocks * metadata_words_per_super;
    const int64_t expected_packed_words =
        num_super_blocks *
        sub_blocks_per_super *
        iml::cuda::dequantize_from_affine_high::kPackedWordsPerSubBlock;

    if (packed.numel() < expected_packed_words) {
        throw std::invalid_argument("packed does not contain enough words");
    }
    if (sub_scales.numel() < expected_sub_metadata) {
        throw std::invalid_argument("sub_scales does not contain enough values");
    }
    if (sub_mins.numel() < expected_sub_metadata) {
        throw std::invalid_argument("sub_mins does not contain enough values");
    }
    if (super_scales.numel() < num_super_blocks) {
        throw std::invalid_argument("super_scales does not contain enough values");
    }
    if (super_mins.numel() < num_super_blocks) {
        throw std::invalid_argument("super_mins does not contain enough values");
    }
}

void dequantize_from_affine_high_fp16(
    torch::Tensor packed,
    torch::Tensor sub_scales,
    torch::Tensor sub_mins,
    torch::Tensor super_scales,
    torch::Tensor super_mins,
    torch::Tensor out,
    int64_t original_numel,
    int64_t super_block_size
) {
    validate_inputs(
        packed,
        sub_scales,
        sub_mins,
        super_scales,
        super_mins,
        out,
        original_numel,
        super_block_size
    );

    const int device = packed.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();

    iml::cuda::dequantize_from_affine_high::launch_dequantize_from_affine_high(
        packed.data_ptr<int32_t>(),
        sub_scales.data_ptr<int32_t>(),
        sub_mins.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(super_scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(super_mins.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        original_numel,
        static_cast<int>(super_block_size),
        stream
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dequantize_from_affine_high_fp16",
        &dequantize_from_affine_high_fp16,
        "Dequantize packed affine uint5 values into fp16"
    );
}
