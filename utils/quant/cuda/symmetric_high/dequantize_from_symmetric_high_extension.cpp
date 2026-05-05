#include <cstdint>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dequantize_from_symmetric_high.cuh"

namespace {

void validate_inputs(
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t original_numel,
    int64_t row_size,
    int64_t blocks_per_row
) {
    if (!qweight.is_cuda()) {
        throw std::invalid_argument("qweight must be a CUDA tensor");
    }
    if (!scales.is_cuda()) {
        throw std::invalid_argument("scales must be a CUDA tensor");
    }
    if (!out.is_cuda()) {
        throw std::invalid_argument("out must be a CUDA tensor");
    }

    if (qweight.scalar_type() != torch::kInt8) {
        throw std::invalid_argument("qweight must have dtype torch.int8");
    }
    if (scales.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("scales must have dtype torch.float16");
    }
    if (out.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("out must have dtype torch.float16");
    }

    if (!qweight.is_contiguous()) {
        throw std::invalid_argument("qweight must be contiguous");
    }
    if (!scales.is_contiguous()) {
        throw std::invalid_argument("scales must be contiguous");
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("out must be contiguous");
    }

    const int device = qweight.get_device();
    if (scales.get_device() != device || out.get_device() != device) {
        throw std::invalid_argument("all tensors must be on the same CUDA device");
    }

    if (original_numel < 0) {
        throw std::invalid_argument("original_numel must be non-negative");
    }
    if (row_size < 0) {
        throw std::invalid_argument("row_size must be non-negative");
    }
    if (blocks_per_row < 0) {
        throw std::invalid_argument("blocks_per_row must be non-negative");
    }
    if (out.numel() < original_numel) {
        throw std::invalid_argument("out is smaller than original_numel");
    }
    if (original_numel > 0 && row_size == 0) {
        throw std::invalid_argument("row_size must be positive when original_numel is non-zero");
    }
    if (original_numel > 0 && blocks_per_row == 0) {
        throw std::invalid_argument("blocks_per_row must be positive when original_numel is non-zero");
    }

    const int64_t row_count = row_size == 0 ? 0 : (original_numel + row_size - 1) / row_size;
    const int64_t expected_blocks = row_count * blocks_per_row;
    const int64_t expected_qweight_values =
        expected_blocks * iml::cuda::dequantize_from_symmetric_high::kBlockSize;

    if (qweight.numel() < expected_qweight_values) {
        throw std::invalid_argument("qweight does not contain enough values");
    }
    if (scales.numel() < expected_blocks) {
        throw std::invalid_argument("scales does not contain enough values");
    }
}

void dequantize_from_symmetric_high_fp16(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t original_numel,
    int64_t row_size,
    int64_t blocks_per_row
) {
    validate_inputs(qweight, scales, out, original_numel, row_size, blocks_per_row);

    const int device = qweight.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();

    iml::cuda::dequantize_from_symmetric_high::launch_dequantize_from_symmetric_high(
        qweight.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        original_numel,
        static_cast<int>(row_size),
        static_cast<int>(blocks_per_row),
        stream
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dequantize_from_symmetric_high_fp16",
        &dequantize_from_symmetric_high_fp16,
        "Dequantize symmetric int8 block values into fp16"
    );
}
