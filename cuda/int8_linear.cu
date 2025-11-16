#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <cublasLt.h>
#include <vector>

#define CHECK_INPUT(x)                                            \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");          \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define TORCH_CUBLAS_CHECK(expr)                                      \
  do {                                                                \
    cublasStatus_t _status = (expr);                                  \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                     \
        "cuBLASLt error at ", __FILE__, ":", __LINE__,                \
        " status=", static_cast<int>(_status));                       \
  } while (0)

namespace {

torch::Tensor run_int8_gemm(
    const torch::Tensor& w_mat,
    const torch::Tensor& x_mat,
    int64_t rows,
    int64_t K_total) {
  auto out_mat = torch::empty(
      {w_mat.size(0), rows},
      w_mat.options().dtype(torch::kInt32));

  auto stream = at::cuda::getCurrentCUDAStream();

  cublasLtHandle_t lt_handle;
  TORCH_CUBLAS_CHECK(cublasLtCreate(&lt_handle));

  cublasLtMatmulDesc_t matmul_desc;
  TORCH_CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  TORCH_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &transa,
      sizeof(transa)));
  TORCH_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transb,
      sizeof(transb)));

  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &a_desc,
      CUDA_R_8I,
      w_mat.size(0),
      static_cast<int>(K_total),
      static_cast<int>(K_total)));
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &b_desc,
      CUDA_R_8I,
      static_cast<int>(K_total),
      static_cast<int>(rows),
      static_cast<int>(rows)));
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &c_desc,
      CUDA_R_32I,
      w_mat.size(0),
      static_cast<int>(rows),
      static_cast<int>(rows)));

  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

  cublasLtMatmulPreference_t preference;
  TORCH_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  size_t max_workspace = 4 * 1024 * 1024;
  TORCH_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace,
      sizeof(max_workspace)));

  cublasLtMatmulHeuristicResult_t heuristic;
  int returned_results = 0;
  TORCH_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt_handle,
      matmul_desc,
      a_desc,
      b_desc,
      c_desc,
      c_desc,
      preference,
      1,
      &heuristic,
      &returned_results));
  TORCH_CHECK(returned_results > 0, "No cuBLASLt heuristic for int8 matmul");

  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  size_t workspace_size = heuristic.workspaceSize;
  if (workspace_size > 0) {
    workspace = torch::empty(
        static_cast<int64_t>(workspace_size),
        w_mat.options().dtype(torch::kByte));
    workspace_ptr = workspace.data_ptr<uint8_t>();
  }

  int32_t alpha = 1;
  int32_t beta = 0;

  TORCH_CUBLAS_CHECK(cublasLtMatmul(
      lt_handle,
      matmul_desc,
      &alpha,
      w_mat.data_ptr<int8_t>(),
      a_desc,
      x_mat.data_ptr<int8_t>(),
      b_desc,
      &beta,
      out_mat.data_ptr<int32_t>(),
      c_desc,
      out_mat.data_ptr<int32_t>(),
      c_desc,
      &heuristic.algo,
      workspace_ptr,
      workspace_size,
      stream));

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(matmul_desc);
  cublasLtDestroy(lt_handle);

  return out_mat;
}

__global__ void scale_acc_linear_kernel(
    float* __restrict__ out,
    const int32_t* __restrict__ acc,
    const float* __restrict__ bias,
    const float* __restrict__ scale_vec,
    bool apply_scale,
    int64_t rows,
    int C_out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * static_cast<int64_t>(C_out);
  if (idx >= total) {
    return;
  }
  int row = static_cast<int>(idx / C_out);
  int col = static_cast<int>(idx % C_out);

  int64_t acc_idx = static_cast<int64_t>(col) * rows + row;
  float val = static_cast<float>(acc[acc_idx]);
  float scale = apply_scale ? scale_vec[col] : 1.0f;
  val *= scale;
  if (bias != nullptr) {
    val += bias[col];
  }
  out[idx] = val;
}

} // namespace

torch::Tensor int8_linear_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor bias,
    torch::Tensor scale,
    bool apply_scale) {
  CHECK_INPUT(x_q);
  CHECK_INPUT(w_q);
  CHECK_INPUT(bias);
  CHECK_INPUT(scale);
  TORCH_CHECK(x_q.dtype() == torch::kChar, "x_q must be int8");
  TORCH_CHECK(w_q.dtype() == torch::kChar, "w_q must be int8");
  TORCH_CHECK(bias.dtype() == torch::kHalf, "bias must be fp16");
  TORCH_CHECK(w_q.dim() == 2, "weight must be [C_out, C_in]");

  auto x = x_q.contiguous();
  std::vector<int64_t> sizes;
  sizes.reserve(x.dim());
  for (int64_t i = 0; i < x.dim(); ++i) {
    sizes.push_back(x.size(i));
  }
  TORCH_CHECK(
      x.size(-1) == w_q.size(1),
      "Input in_features must match weight inner dim");

  int64_t rows = x.numel() / x.size(-1);
  int64_t in_features = x.size(-1);
  int C_out = static_cast<int>(w_q.size(0));
  TORCH_CHECK(
      scale.dim() == 1 && scale.size(0) == C_out,
      "scale must be 1D with length matching the output dimension");

  auto x_mat = x.view({rows, in_features}).transpose(0, 1).contiguous();
  auto w_mat = w_q.contiguous();

  auto out_mat = run_int8_gemm(w_mat, x_mat, rows, in_features);

  auto out = torch::empty({rows, C_out}, x.options().dtype(torch::kFloat));
  auto bias_fp32 = bias.to(torch::kFloat);
  auto scale_fp32 = scale.to(torch::kFloat).contiguous();

  int threads = 256;
  int64_t total = rows * static_cast<int64_t>(C_out);
  int blocks = static_cast<int>((total + threads - 1) / threads);

  scale_acc_linear_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(),
      out_mat.data_ptr<int32_t>(),
      bias_fp32.data_ptr<float>(),
      scale_fp32.data_ptr<float>(),
      apply_scale,
      rows,
      C_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  sizes.back() = C_out;
  return out.view(sizes);
}
