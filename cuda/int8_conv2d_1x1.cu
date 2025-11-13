#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

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

__global__ void scale_acc_kernel(
    float* __restrict__ out,
    const int32_t* __restrict__ acc,
    float eff_scale,
    bool apply_scale,
    const float* __restrict__ bias,
    int64_t total,
    int N,
    int C_out,
    int H,
    int W) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int n = (idx / (C_out * H * W)) % N;
  int co = (idx / (H * W)) % C_out;
  int hw = idx % (H * W);
  int h = hw / W;
  int w_idx = hw % W;

  int64_t acc_offset = ((int64_t)co * N * H * W) +
                       (int64_t)n * H * W +
                       (int64_t)h * W +
                       w_idx;

  float val = static_cast<float>(acc[acc_offset]);
  if (apply_scale) {
    val *= eff_scale;
    if (bias != nullptr) {
      val += bias[co];
    }
  }
  out[idx] = val;
}

torch::Tensor run_int8_gemm(
    const torch::Tensor& w_mat,
    const torch::Tensor& x_mat,
    int C_out,
    int64_t n_cols,
    int64_t K_total) {
  auto out_mat = torch::empty(
      {C_out, n_cols},
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
      &a_desc, CUDA_R_8I, C_out, static_cast<int>(K_total), static_cast<int>(K_total)));
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &b_desc, CUDA_R_8I, static_cast<int>(K_total), static_cast<int>(n_cols), static_cast<int>(n_cols)));
  TORCH_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &c_desc, CUDA_R_32I, C_out, static_cast<int>(n_cols), static_cast<int>(n_cols)));

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

} // namespace

torch::Tensor int8_conv2d_1x1_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor bias,
    double scale_product,
    bool apply_scale,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups) {
  CHECK_INPUT(x_q);
  CHECK_INPUT(w_q);
  TORCH_CHECK(x_q.dtype() == torch::kChar, "x_q must be int8");
  TORCH_CHECK(w_q.dtype() == torch::kChar, "w_q must be int8");
  TORCH_CHECK(groups == 1, "groups != 1 not implemented");
  CHECK_INPUT(bias);
  TORCH_CHECK(bias.dtype() == torch::kHalf, "bias must be fp16");
  TORCH_CHECK(bias.size(0) == w_q.size(0), "bias size must match C_out");
  auto bias_fp32 = bias.to(torch::kFloat);

  TORCH_CHECK(stride_h == 1 && stride_w == 1, "1x1 path requires stride=1");
  TORCH_CHECK(pad_h == 0 && pad_w == 0, "1x1 path requires pad=0");
  TORCH_CHECK(dilation_h == 1 && dilation_w == 1, "1x1 path requires dilation=1");

  auto x = x_q.contiguous();
  auto w = w_q.contiguous();

  int N = x.size(0);
  int C_in = x.size(1);
  int H = x.size(2);
  int W = x.size(3);
  int C_out = w.size(0);

  TORCH_CHECK(w.size(2) == 1 && w.size(3) == 1, "Weights must be 1x1");

  int H_out = H;
  int W_out = W;

  int64_t K_total = C_in;
  int64_t n_cols = static_cast<int64_t>(N) * H_out * W_out;

  auto x_mat = x.view({N, C_in, H_out, W_out})
                  .permute({1, 0, 2, 3})
                  .contiguous()
                  .view({K_total, n_cols});

  auto w_mat = w.view({C_out, static_cast<int>(K_total)}).contiguous();

  auto out_mat = run_int8_gemm(w_mat, x_mat, C_out, n_cols, K_total);

  auto out = torch::empty({N, C_out, H_out, W_out}, x.options().dtype(torch::kFloat));
  float eff_scale = static_cast<float>(scale_product);
  int64_t total = out.numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  const float* bias_ptr = bias_fp32.data_ptr<float>();

  scale_acc_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(),
      out_mat.data_ptr<int32_t>(),
      eff_scale,
      apply_scale,
      bias_ptr,
      total,
      N,
      C_out,
      H_out,
      W_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}
