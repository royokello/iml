#include <torch/extension.h>

torch::Tensor int8_linear_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor bias,
    double scale_product,
    bool apply_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_linear", &int8_linear_cuda, "INT8 Linear GEMM");
}
