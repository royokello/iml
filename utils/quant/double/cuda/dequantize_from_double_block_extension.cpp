#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dequantize_from_double_block.cuh"

namespace {

struct DeviceLutInitState {
    bool initialized = false;
    cudaEvent_t ready_event = nullptr;
};

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result)
        );
    }
}

void ensure_byte_lut_initialized(int device, cudaStream_t stream) {
    static std::mutex lut_mutex;
    static std::vector<DeviceLutInitState> states;

    cudaEvent_t ready_event = nullptr;
    bool needs_wait = false;

    {
        std::lock_guard<std::mutex> lock(lut_mutex);
        if (device >= static_cast<int>(states.size())) {
            states.resize(device + 1);
        }

        DeviceLutInitState& state = states[device];
        if (!state.initialized) {
            iml::cuda::dequantize_from_double_block::init_byte_to_half2_lut(stream);
            if (state.ready_event == nullptr) {
                check_cuda(
                    cudaEventCreateWithFlags(&state.ready_event, cudaEventDisableTiming),
                    "cudaEventCreateWithFlags"
                );
            }
            check_cuda(cudaEventRecord(state.ready_event, stream), "cudaEventRecord");
            state.initialized = true;
        } else {
            ready_event = state.ready_event;
            needs_wait = true;
        }
    }

    if (needs_wait) {
        check_cuda(cudaStreamWaitEvent(stream, ready_event, 0), "cudaStreamWaitEvent");
    }
}

void validate_inputs(
    const torch::Tensor& packed,
    const torch::Tensor& sub_scales,
    const torch::Tensor& super_scales,
    const torch::Tensor& out,
    int64_t original_numel
) {
    if (!packed.is_cuda()) {
        throw std::invalid_argument("packed must be a CUDA tensor");
    }
    if (!sub_scales.is_cuda()) {
        throw std::invalid_argument("sub_scales must be a CUDA tensor");
    }
    if (!super_scales.is_cuda()) {
        throw std::invalid_argument("super_scales must be a CUDA tensor");
    }
    if (!out.is_cuda()) {
        throw std::invalid_argument("out must be a CUDA tensor");
    }

    if (packed.scalar_type() != torch::kInt8) {
        throw std::invalid_argument("packed must have dtype torch.int8");
    }
    if (sub_scales.scalar_type() != torch::kInt8) {
        throw std::invalid_argument("sub_scales must have dtype torch.int8");
    }
    if (super_scales.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("super_scales must have dtype torch.float16");
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
    if (!super_scales.is_contiguous()) {
        throw std::invalid_argument("super_scales must be contiguous");
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("out must be contiguous");
    }

    const int device = packed.get_device();
    if (sub_scales.get_device() != device || super_scales.get_device() != device || out.get_device() != device) {
        throw std::invalid_argument("all tensors must be on the same CUDA device");
    }

    if (original_numel < 0) {
        throw std::invalid_argument("original_numel must be non-negative");
    }
    if (out.numel() < original_numel) {
        throw std::invalid_argument("out is smaller than original_numel");
    }

    const int64_t num_pairs = (original_numel + 1) / 2;
    if (packed.numel() < num_pairs) {
        throw std::invalid_argument("packed does not contain enough bytes");
    }

    const int64_t num_super_blocks =
        (original_numel + iml::cuda::dequantize_from_double_block::kSuperBlockSize - 1) /
        iml::cuda::dequantize_from_double_block::kSuperBlockSize;
    if (super_scales.numel() < num_super_blocks) {
        throw std::invalid_argument("super_scales does not contain enough values");
    }
    if (sub_scales.numel() < num_super_blocks * iml::cuda::dequantize_from_double_block::kSubBlocksPerSuper) {
        throw std::invalid_argument("sub_scales does not contain enough values");
    }
}

void dequantize_from_double_block_fp16(
    torch::Tensor packed,
    torch::Tensor sub_scales,
    torch::Tensor super_scales,
    torch::Tensor out,
    int64_t original_numel
) {
    validate_inputs(packed, sub_scales, super_scales, out, original_numel);

    const int device = packed.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();
    ensure_byte_lut_initialized(device, stream);

    iml::cuda::dequantize_from_double_block::launch_dequantize_from_double_block(
        packed.data_ptr<int8_t>(),
        sub_scales.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(super_scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        original_numel,
        stream
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dequantize_from_double_block_fp16",
        &dequantize_from_double_block_fp16,
        "Dequantize packed double-block int4 values into fp16"
    );
}
