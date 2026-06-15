#include <cstdint>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gemm_fused.cuh"

using namespace iml::cuda::gemm;

// ── Format string → enum ────────────────────────────────────────────────

static WeightFmt parse_format(const std::string& fmt) {
    if (fmt == "sym-high") return WeightFmt::SymHigh;
    if (fmt == "sym-med")  return WeightFmt::SymMed;
    if (fmt == "sym-low")  return WeightFmt::SymLow;
    if (fmt == "aff-high") return WeightFmt::AffHigh;
    if (fmt == "aff-med")  return WeightFmt::AffMed;
    if (fmt == "aff-low")  return WeightFmt::AffLow;
    throw std::invalid_argument("Unknown weight format: " + fmt);
}

static int sub_block_size_for(WeightFmt fmt) {
    switch (fmt) {
        case WeightFmt::SymHigh: return 32;
        case WeightFmt::SymMed:  return 16;
        case WeightFmt::SymLow:  return 16;
        case WeightFmt::AffHigh: return 32;
        case WeightFmt::AffMed:  return 32;
        case WeightFmt::AffLow:  return 16;
    }
    return 16;
}

// ── Validation ──────────────────────────────────────────────────────────

static void validate(
    const torch::Tensor& q_act,
    const torch::Tensor& bsums,
    const torch::Tensor& act_scales,
    const torch::Tensor& output,
    int64_t M, int64_t N, int64_t K
) {
    auto check = [](bool cond, const char* msg) {
        if (!cond) throw std::invalid_argument(msg);
    };

    check(q_act.is_cuda(), "q_act must be CUDA");
    check(bsums.is_cuda(), "bsums must be CUDA");
    check(act_scales.is_cuda(), "act_scales must be CUDA");
    check(output.is_cuda(), "output must be CUDA");

    check(q_act.scalar_type() == torch::kInt8, "q_act must be int8");
    check(bsums.scalar_type() == torch::kInt16, "bsums must be int16");
    check(act_scales.scalar_type() == torch::kFloat32, "act_scales must be float32");
    check(output.scalar_type() == torch::kFloat16, "output must be float16");

    check(q_act.is_contiguous(), "q_act must be contiguous");
    check(bsums.is_contiguous(), "bsums must be contiguous");
    check(act_scales.is_contiguous(), "act_scales must be contiguous");
    check(output.is_contiguous(), "output must be contiguous");

    int dev = q_act.get_device();
    check(bsums.get_device() == dev, "all tensors must be on same device");
    check(act_scales.get_device() == dev, "all tensors must be on same device");
    check(output.get_device() == dev, "all tensors must be on same device");

    check(M > 0 && N > 0 && K > 0, "dimensions must be positive");
}

// ── Main entry point ────────────────────────────────────────────────────

void fused_gemm_fp16(
    torch::Tensor q_act,
    torch::Tensor bsums,
    torch::Tensor act_scales,
    torch::Tensor q_weight,
    std::optional<torch::Tensor> sub_scales,
    std::optional<torch::Tensor> sub_mins,
    std::optional<torch::Tensor> super_scales,
    std::optional<torch::Tensor> super_mins,
    torch::Tensor output,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t super_block_size,
    std::string weight_format,
    int64_t force_bm,
    int64_t force_bn,
    int64_t force_bk
) {
    validate(q_act, bsums, act_scales, output, M, N, K);

    auto fmt = parse_format(weight_format);
    int sub_sz = sub_block_size_for(fmt);
    int total_super = static_cast<int>(K) / static_cast<int>(super_block_size);
    constexpr int kActSub = 16;
    int total_act_subs = static_cast<int>(K) / kActSub;

    // Validate bsums/act_scales sizes
    if (bsums.numel() < M * total_act_subs) {
        throw std::invalid_argument("bsums too small");
    }
    if (act_scales.numel() < M * total_super) {
        throw std::invalid_argument("act_scales too small");
    }

    const int device = q_act.get_device();
    const auto stream = at::cuda::getCurrentCUDAStream(device).stream();

    // Select launch config
    LaunchConfig lc;
    if (force_bm > 0 && force_bn > 0 && force_bk > 0) {
        // Forced config (used by autotuner)
        if (force_bm == 64 && force_bn == 64 && force_bk == 128)
            lc = { launch_balanced_64_64_128, 64, 64, 128 };
        else if (force_bm == 128 && force_bn == 64 && force_bk == 128)
            lc = { launch_tall_m_128_64_128, 128, 64, 128 };
        else if (force_bm == 64 && force_bn == 128 && force_bk == 128)
            lc = { launch_wide_n_64_128_128, 64, 128, 128 };
        else if (force_bm == 128 && force_bn == 128 && force_bk == 64)
            lc = { launch_large_128_128_64, 128, 128, 64 };
        else if (force_bm == 32 && force_bn == 64 && force_bk == 128)
            lc = { launch_small_m_32_64_128, 32, 64, 128 };
        else
            throw std::invalid_argument("Unknown tile config");
    } else {
        lc = select_launch(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    }

    lc.fn(
        q_act.data_ptr<int8_t>(),
        bsums.data_ptr<int16_t>(),
        act_scales.data_ptr<float>(),
        q_weight.data_ptr(),
        sub_scales.has_value() ? reinterpret_cast<const float*>(sub_scales->data_ptr()) : nullptr,
        sub_mins.has_value() ? reinterpret_cast<const float*>(sub_mins->data_ptr()) : nullptr,
        super_scales.has_value() ? reinterpret_cast<const float*>(super_scales->data_ptr()) : nullptr,
        super_mins.has_value() ? reinterpret_cast<const float*>(super_mins->data_ptr()) : nullptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(super_block_size), sub_sz, fmt,
        stream
    );
}

// ── Pybind11 module ─────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_fp16", &fused_gemm_fp16,
          "Fused int8 activation x quantized weight GEMM with bsums correction");
}
