#include <torch/extension.h>

// Forward declaration of the CUDA implementation
torch::Tensor conv_1x1_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor scale,
    torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "conv_1x1",
      &conv_1x1_cuda,
      "INT8 1x1 conv, batch=1, GEMM + (S * Z + B) epilogue");
}
