// int8_conv2d_dp4a_forward.cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

__global__ void int8_conv2d_dp4a_kernel(
    const int8_t* __restrict__ x,      // [N, C_in, H_in, W_in]
    const int8_t* __restrict__ w,      // [C_out, C_in, K_h, K_w]
    const __half* __restrict__ bias,   // [C_out] or nullptr
    __half scale_x_h,                  // scalar
    __half scale_w_h,                  // scalar
    __half* __restrict__ y,            // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool has_bias
) {
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z; // flatten (N, C_out)
    int n = nc / C_out;
    int c_out = nc % C_out;

    if (n >= N || c_out >= C_out || out_h >= H_out || out_w >= W_out)
        return;

    // pointers for convenience
    // x: [N, C_in, H_in, W_in]
    // w: [C_out, C_in, K_h, K_w]
    // y: [N, C_out, H_out, W_out]
    int x_stride_n = C_in * H_in * W_in;
    int x_stride_c = H_in * W_in;
    int x_stride_h = W_in;
    int x_stride_w = 1;

    int w_stride_co = C_in * K_h * K_w;
    int w_stride_ci = K_h * K_w;
    int w_stride_kh = K_w;
    int w_stride_kw = 1;

    int y_stride_n = C_out * H_out * W_out;
    int y_stride_c = H_out * W_out;
    int y_stride_h = W_out;
    int y_stride_w = 1;

    // precompute scales as float
    float scale_x = __half2float(scale_x_h);
    float scale_w = __half2float(scale_w_h);
    float eff_scale = scale_x * scale_w;  // single per-tensor scale for conv output (before bias)

    // total inner length for dot product
    int K_total = C_in * K_h * K_w;
    int K4 = K_total / 4;
    int K_tail = K_total % 4;

    int acc = 0;  // INT32 accumulator for dp4a

    // iterate over K dimension in groups of 4
    for (int i4 = 0; i4 < K4; ++i4) {
        int8_t x_vals[4];
        int8_t w_vals[4];

        // fill 4 lanes
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            int k = i4 * 4 + lane;

            int ci = k / (K_h * K_w);
            int rem = k % (K_h * K_w);
            int kh = rem / K_w;
            int kw = rem % K_w;

            int in_h = out_h * stride_h - pad_h + kh * dilation_h;
            int in_w = out_w * stride_w - pad_w + kw * dilation_w;

            int8_t x_q = 0;
            if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                int x_idx = n * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w;
                x_q = x[x_idx];
            }

            int w_idx = c_out * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw;
            int8_t w_q = w[w_idx];

            x_vals[lane] = x_q;
            w_vals[lane] = w_q;
        }

        // pack 4 int8 into one int32 each
        int a_packed =
            (int)(uint8_t)x_vals[0] |
            ((int)(uint8_t)x_vals[1] << 8) |
            ((int)(uint8_t)x_vals[2] << 16) |
            ((int)(uint8_t)x_vals[3] << 24);

        int b_packed =
            (int)(uint8_t)w_vals[0] |
            ((int)(uint8_t)w_vals[1] << 8) |
            ((int)(uint8_t)w_vals[2] << 16) |
            ((int)(uint8_t)w_vals[3] << 24);

        // dp4a: signed 8-bit dot product + accumulate: acc += dot(a, b)
        acc = __dp4a(a_packed, b_packed, acc);
    }

    // tail elements (if K_total not multiple of 4)
    for (int k = K4 * 4; k < K_total; ++k) {
        int ci = k / (K_h * K_w);
        int rem = k % (K_h * K_w);
        int kh = rem / K_w;
        int kw = rem % K_w;

        int in_h = out_h * stride_h - pad_h + kh * dilation_h;
        int in_w = out_w * stride_w - pad_w + kw * dilation_w;

        int8_t x_q = 0;
        if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
            int x_idx = n * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w;
            x_q = x[x_idx];
        }

        int w_idx = c_out * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw;
        int8_t w_q = w[w_idx];

        acc += (int)x_q * (int)w_q;
    }

    // convert INT32 accumulator to float, apply FP16 scales and bias
    float out_f = (float)acc * eff_scale;

    if (has_bias && bias != nullptr) {
        float b_f = __half2float(bias[c_out]);
        out_f += b_f;
    }

    __half out_h = __float2half(out_f);

    int y_idx = n * y_stride_n + c_out * y_stride_c + out_h * y_stride_h + out_w * y_stride_w;
    y[y_idx] = out_h;
}

// C++ launcher
torch::Tensor int8_conv2d_dp4a_forward(
    torch::Tensor x_q,      // int8 [N, C_in, H_in, W_in]
    torch::Tensor w_q,      // int8 [C_out, C_in, K_h, K_w]
    torch::Tensor scale_x,  // scalar fp16
    torch::Tensor scale_w,  // scalar fp16
    torch::Tensor bias,     // fp16 [C_out] or empty
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    CHECK_INPUT(x_q);
    CHECK_INPUT(w_q);
    CHECK_INPUT(scale_x);
    CHECK_INPUT(scale_w);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }

    TORCH_CHECK(x_q.dtype() == torch::kChar, "x_q must be int8");
    TORCH_CHECK(w_q.dtype() == torch::kChar, "w_q must be int8");
    TORCH_CHECK(scale_x.dtype() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(scale_w.dtype() == torch::kHalf, "scale_w must be fp16");
    TORCH_CHECK(groups == 1, "groups != 1 not implemented in this kernel");

    auto x = x_q.contiguous();
    auto w = w_q.contiguous();

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int C_out = w.size(0);
    int K_h = w.size(2);
    int K_w = w.size(3);

    // derive output size
    int H_out = (H_in + 2 * pad_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    // create output tensor: fp16
    auto y = torch::empty({N, C_out, H_out, W_out}, x.options().dtype(torch::kHalf));

    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = w.data_ptr<int8_t>();
    __half* y_ptr = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    const __half* bias_ptr = nullptr;
    bool has_bias = false;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.dtype() == torch::kHalf, "bias must be fp16");
        TORCH_CHECK(bias.size(0) == C_out, "bias size must match C_out");
        bias_ptr = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
        has_bias = true;
    }

    __half scale_x_h = *reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>());
    __half scale_w_h = *reinterpret_cast<const __half*>(scale_w.data_ptr<at::Half>());

    // launch config
    dim3 block(16, 16, 1);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C_out
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int8_conv2d_dp4a_kernel<<<grid, block, 0, stream>>>(
        x_ptr,
        w_ptr,
        bias_ptr,
        scale_x_h,
        scale_w_h,
        y_ptr,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "int8_conv2d_dp4a_forward",
        &int8_conv2d_dp4a_forward,
        "INT8 Conv2d with DP4A (int8 x int8 -> int32, fp16 scale & bias)"
    );
}
// int8_conv2d_dp4a_forward.cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

__global__ void int8_conv2d_dp4a_kernel(
    const int8_t* __restrict__ x,      // [N, C_in, H_in, W_in]
    const int8_t* __restrict__ w,      // [C_out, C_in, K_h, K_w]
    const __half* __restrict__ bias,   // [C_out] or nullptr
    __half scale_x_h,                  // scalar
    __half scale_w_h,                  // scalar
    __half* __restrict__ y,            // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool has_bias
) {
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z; // flatten (N, C_out)
    int n = nc / C_out;
    int c_out = nc % C_out;

    if (n >= N || c_out >= C_out || out_h >= H_out || out_w >= W_out)
        return;

    // pointers for convenience
    // x: [N, C_in, H_in, W_in]
    // w: [C_out, C_in, K_h, K_w]
    // y: [N, C_out, H_out, W_out]
    int x_stride_n = C_in * H_in * W_in;
    int x_stride_c = H_in * W_in;
    int x_stride_h = W_in;
    int x_stride_w = 1;

    int w_stride_co = C_in * K_h * K_w;
    int w_stride_ci = K_h * K_w;
    int w_stride_kh = K_w;
    int w_stride_kw = 1;

    int y_stride_n = C_out * H_out * W_out;
    int y_stride_c = H_out * W_out;
    int y_stride_h = W_out;
    int y_stride_w = 1;

    // precompute scales as float
    float scale_x = __half2float(scale_x_h);
    float scale_w = __half2float(scale_w_h);
    float eff_scale = scale_x * scale_w;  // single per-tensor scale for conv output (before bias)

    // total inner length for dot product
    int K_total = C_in * K_h * K_w;
    int K4 = K_total / 4;
    int K_tail = K_total % 4;

    int acc = 0;  // INT32 accumulator for dp4a

    // iterate over K dimension in groups of 4
    for (int i4 = 0; i4 < K4; ++i4) {
        int8_t x_vals[4];
        int8_t w_vals[4];

        // fill 4 lanes
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            int k = i4 * 4 + lane;

            int ci = k / (K_h * K_w);
            int rem = k % (K_h * K_w);
            int kh = rem / K_w;
            int kw = rem % K_w;

            int in_h = out_h * stride_h - pad_h + kh * dilation_h;
            int in_w = out_w * stride_w - pad_w + kw * dilation_w;

            int8_t x_q = 0;
            if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                int x_idx = n * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w;
                x_q = x[x_idx];
            }

            int w_idx = c_out * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw;
            int8_t w_q = w[w_idx];

            x_vals[lane] = x_q;
            w_vals[lane] = w_q;
        }

        // pack 4 int8 into one int32 each
        int a_packed =
            (int)(uint8_t)x_vals[0] |
            ((int)(uint8_t)x_vals[1] << 8) |
            ((int)(uint8_t)x_vals[2] << 16) |
            ((int)(uint8_t)x_vals[3] << 24);

        int b_packed =
            (int)(uint8_t)w_vals[0] |
            ((int)(uint8_t)w_vals[1] << 8) |
            ((int)(uint8_t)w_vals[2] << 16) |
            ((int)(uint8_t)w_vals[3] << 24);

        // dp4a: signed 8-bit dot product + accumulate: acc += dot(a, b)
        acc = __dp4a(a_packed, b_packed, acc);
    }

    // tail elements (if K_total not multiple of 4)
    for (int k = K4 * 4; k < K_total; ++k) {
        int ci = k / (K_h * K_w);
        int rem = k % (K_h * K_w);
        int kh = rem / K_w;
        int kw = rem % K_w;

        int in_h = out_h * stride_h - pad_h + kh * dilation_h;
        int in_w = out_w * stride_w - pad_w + kw * dilation_w;

        int8_t x_q = 0;
        if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
            int x_idx = n * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w;
            x_q = x[x_idx];
        }

        int w_idx = c_out * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw;
        int8_t w_q = w[w_idx];

        acc += (int)x_q * (int)w_q;
    }

    // convert INT32 accumulator to float, apply FP16 scales and bias
    float out_f = (float)acc * eff_scale;

    if (has_bias && bias != nullptr) {
        float b_f = __half2float(bias[c_out]);
        out_f += b_f;
    }

    __half out_h = __float2half(out_f);

    int y_idx = n * y_stride_n + c_out * y_stride_c + out_h * y_stride_h + out_w * y_stride_w;
    y[y_idx] = out_h;
}

// C++ launcher
torch::Tensor int8_conv2d_dp4a_forward(
    torch::Tensor x_q,      // int8 [N, C_in, H_in, W_in]
    torch::Tensor w_q,      // int8 [C_out, C_in, K_h, K_w]
    torch::Tensor scale_x,  // scalar fp16
    torch::Tensor scale_w,  // scalar fp16
    torch::Tensor bias,     // fp16 [C_out] or empty
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    CHECK_INPUT(x_q);
    CHECK_INPUT(w_q);
    CHECK_INPUT(scale_x);
    CHECK_INPUT(scale_w);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }

    TORCH_CHECK(x_q.dtype() == torch::kChar, "x_q must be int8");
    TORCH_CHECK(w_q.dtype() == torch::kChar, "w_q must be int8");
    TORCH_CHECK(scale_x.dtype() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(scale_w.dtype() == torch::kHalf, "scale_w must be fp16");
    TORCH_CHECK(groups == 1, "groups != 1 not implemented in this kernel");

    auto x = x_q.contiguous();
    auto w = w_q.contiguous();

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int C_out = w.size(0);
    int K_h = w.size(2);
    int K_w = w.size(3);

    // derive output size
    int H_out = (H_in + 2 * pad_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    // create output tensor: fp16
    auto y = torch::empty({N, C_out, H_out, W_out}, x.options().dtype(torch::kHalf));

    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = w.data_ptr<int8_t>();
    __half* y_ptr = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    const __half* bias_ptr = nullptr;
    bool has_bias = false;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.dtype() == torch::kHalf, "bias must be fp16");
        TORCH_CHECK(bias.size(0) == C_out, "bias size must match C_out");
        bias_ptr = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
        has_bias = true;
    }

    __half scale_x_h = *reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>());
    __half scale_w_h = *reinterpret_cast<const __half*>(scale_w.data_ptr<at::Half>());

    // launch config
    dim3 block(16, 16, 1);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C_out
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int8_conv2d_dp4a_kernel<<<grid, block, 0, stream>>>(
        x_ptr,
        w_ptr,
        bias_ptr,
        scale_x_h,
        scale_w_h,
        y_ptr,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "int8_conv2d_dp4a_forward",
        &int8_conv2d_dp4a_forward,
        "INT8 Conv2d with DP4A (int8 x int8 -> int32, fp16 scale & bias)"
    );
}
