#include <cstdint>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dequantize_from_intermediate.cuh"

namespace {

void validate_inputs(
    const torch::Tensor& qweight,
    const torch::Tensor& super_scales,
    const torch::Tensor& out,
    int64_t original_numel,
    int64_t super_block_size
) {
    if (!qweight.is_cuda()) {
        throw std::invalid_argument("qweight must be a CUDA tensor");
    }
    if (!super_scales.is_cuda()) {
        throw std::invalid_argument("super_scales must be a CUDA tensor");
    }
    if (!out.is_cuda()) {
        throw std::invalid_argument("out must be a CUDA tensor");
    }

    if (qweight.scalar_type() != torch::kInt8) {
        throw std::invalid_argument("qweight must have dtype torch.int8");
    }
    if (super_scales.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("super_scales must have dtype torch.float32");
    }
    if (out.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("out must have dtype torch.float16");
    }

    if (!qweight.is_contiguous()) {
        throw std::invalid_argument("qweight must be contiguous");
    }
    if (!super_scales.is_contiguous()) {
        throw std::invalid_argument("super_scales must be contiguous");
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("out must be contiguous");
    }

    const int device = qweight.get_device();
    if (super_scales.get_device() != device || out.get_device() != device) {
        throw std::invalid_argument("all tensors must be on the same CUDA device");
    }

    if (original_numel < 0) {
        throw std::invalid_argument("original_numel must be non-negative");
    }
    if (super_block_size <= 0) {
        throw std::invalid_argument("super_block_size must be positive");
    }
    if (out.numel() < original_numel) {
        throw std::invalid_argument("out is smaller than original_numel");
    }
    if (qweight.numel() < original_numel) {
        throw std::invalid_argument("qweight does not contain enough values");
    }

    const int64_t num_super_blocks =
        (original_numel + super_block_size - 1) / super_block_size;
    if (super_scales.numel() < num_super_blocks) {
        throw std::invalid_argument("super_scales does not contain enough values");
    }
}

void dequantize_from_intermediate_fp16(
    torch::Tensor qweight,
    torch::Tensor super_scales,
    torch::Tensor out,
    int64_t original_numel,
    int64_t super_block_size
) {
    validate_inputs(qweight, super_scales, out, original_numel, super_block_size);

    const int device = qweight.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();

    iml::cuda::dequantize_from_intermediate::launch_dequantize_from_intermediate(
        qweight.data_ptr<int8_t>(),
        super_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        original_numel,
        static_cast<int>(super_block_size),
        stream
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dequantize_from_intermediate_fp16",
        &dequantize_from_intermediate_fp16,
        "Dequantize Q8 intermediate int8 block values into fp16"
    );
}
