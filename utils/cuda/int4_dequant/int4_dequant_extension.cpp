#include <cstdint>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "int4_dequant_lut.cuh"

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
            iml::cuda::int4_dequant::init_byte_to_half2_lut(stream);
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

void validate_dequant_inputs(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t original_numel,
    int64_t block_size
) {
    if (!packed.is_cuda()) {
        throw std::invalid_argument("packed must be a CUDA tensor");
    }
    if (!scales.is_cuda()) {
        throw std::invalid_argument("scales must be a CUDA tensor");
    }
    if (!out.is_cuda()) {
        throw std::invalid_argument("out must be a CUDA tensor");
    }
    if (packed.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("packed must have dtype torch.uint8");
    }
    if (scales.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("scales must have dtype torch.float16");
    }
    if (out.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("out must have dtype torch.float16");
    }
    if (!packed.is_contiguous()) {
        throw std::invalid_argument("packed must be contiguous");
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
    if (out.numel() < original_numel) {
        throw std::invalid_argument("out is smaller than original_numel");
    }

    const int64_t num_blocks = (original_numel + block_size - 1) / block_size;
    if (scales.numel() < num_blocks) {
        throw std::invalid_argument("scales does not contain enough block scales");
    }
}

void dequantize_int4_fp16(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t original_numel,
    int64_t block_size
) {
    validate_dequant_inputs(packed, scales, out, original_numel, block_size);

    const int device = packed.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();
    ensure_byte_lut_initialized(device, stream);
    iml::cuda::int4_dequant::launch_byte_lut_dequant(
        packed.data_ptr<uint8_t>(),
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
        "dequantize_int4_fp16",
        &dequantize_int4_fp16,
        "Dequantize packed signed int4 values into fp16 with fp16 block scales"
    );
}
