#include <torch/extension.h>

torch::Tensor int8_conv2d_1x1_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor bias,
    torch::Tensor scale,
    bool apply_scale,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);

torch::Tensor int8_conv2d_3x3_im2col_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor bias,
    torch::Tensor scale,
    bool apply_scale,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "int8_conv2d_1x1",
      &int8_conv2d_1x1_cuda,
      "INT8 Conv2d 1x1 implicit GEMM");
  m.def(
      "int8_conv2d_3x3_im2col",
      &int8_conv2d_3x3_im2col_cuda,
      "INT8 Conv2d 3x3 im2col + GEMM");
}
