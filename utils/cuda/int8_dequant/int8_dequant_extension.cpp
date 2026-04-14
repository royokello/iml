#include <cstdint>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "int8_dequant.cuh"

namespace {

void validate_dequant_inputs(
    const torch::Tensor& quantized,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t original_numel,
    int64_t block_size
) {
    if (!quantized.is_cuda()) {
        throw std::invalid_argument("quantized must be a CUDA tensor");
    }
    if (!scales.is_cuda()) {
        throw std::invalid_argument("scales must be a CUDA tensor");
    }
    if (!out.is_cuda()) {
        throw std::invalid_argument("out must be a CUDA tensor");
    }
    if (quantized.scalar_type() != torch::kInt8) {
        throw std::invalid_argument("quantized must have dtype torch.int8");
    }
    if (scales.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("scales must have dtype torch.float16");
    }
    if (out.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("out must have dtype torch.float16");
    }
    if (!quantized.is_contiguous()) {
        throw std::invalid_argument("quantized must be contiguous");
    }
    if (!scales.is_contiguous()) {
        throw std::invalid_argument("scales must be contiguous");
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("out must be contiguous");
    }
    if (block_size != 32 && block_size != 64 && block_size != 128) {
        throw std::invalid_argument("block_size must be 32, 64, or 128");
    }
    if (original_numel < 0) {
        throw std::invalid_argument("original_numel must be non-negative");
    }
    if (quantized.numel() < original_numel) {
        throw std::invalid_argument("quantized is smaller than original_numel");
    }
    if (out.numel() < original_numel) {
        throw std::invalid_argument("out is smaller than original_numel");
    }

    const int64_t num_blocks = (original_numel + block_size - 1) / block_size;
    if (scales.numel() < num_blocks) {
        throw std::invalid_argument("scales does not contain enough block scales");
    }
}

void dequantize_int8_fp16(
    torch::Tensor quantized,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t original_numel,
    int64_t block_size
) {
    validate_dequant_inputs(quantized, scales, out, original_numel, block_size);

    const auto stream = at::cuda::getCurrentCUDAStream(quantized.get_device()).stream();
    iml::cuda::int8_dequant::launch_int8_dequant(
        quantized.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        original_numel,
        static_cast<int>(block_size),
        stream
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dequantize_int8_fp16",
        &dequantize_int8_fp16,
        "Dequantize int8 values into fp16 with fp16 block scales"
    );
}
