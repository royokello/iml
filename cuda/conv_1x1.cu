#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <cublas_v2.h>

#define CHECK_INPUT(x)                                            \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");          \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define TORCH_CUBLAS_CHECK(expr)                                      \
  do {                                                                \
    cublasStatus_t _status = (expr);                                  \
    TORCH_CHECK(                                                      \
        _status == CUBLAS_STATUS_SUCCESS,                             \
        "cuBLAS error at ", __FILE__, ":", __LINE__,                  \
        " code: ", static_cast<int>(_status));                        \
  } while (0)

// Epilogue: Y = S * Z + B
// Z: [H*W, C_out] int32 (flattened)
// S: [C_out] float (per-output-channel scale)
// B: [C_out] float (per-output-channel bias)
// Y: [H*W, C_out] float
__global__ void scale_bias_epilogue_kernel(
    const int32_t* __restrict__ Z,
    const float* __restrict__ S,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int num_spatial,   // H * W
    int C_out) {       // number of output channels
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_spatial * C_out;
  if (idx >= total) {
    return;
  }

  // We interpret Z as [num_spatial, C_out] in row-major order.
  // Channel index is the fast dimension.
  int c = idx % C_out;

  float scale = S ? S[c] : 1.0f;
  float bias  = B ? B[c] : 0.0f;

  int32_t z_val = Z[idx];
  Y[idx] = static_cast<float>(z_val) * scale + bias;
}

// X_q : [C_in, H, W]      int8
// W_q : [C_out, C_in]     int8
// scale : [C_out]         float (combined Sx * Sw[c])
// bias  : [C_out]         float
// Returns: [C_out, H, W]  float
torch::Tensor conv_1x1_cuda(
    torch::Tensor x_q,
    torch::Tensor w_q,
    torch::Tensor scale,
    torch::Tensor bias) {
  CHECK_INPUT(x_q);
  CHECK_INPUT(w_q);
  CHECK_INPUT(scale);
  CHECK_INPUT(bias);

  TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8");
  TORCH_CHECK(w_q.scalar_type() == at::kChar, "w_q must be int8");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
  TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");

  TORCH_CHECK(x_q.dim() == 3, "x_q must be [C_in, H, W] (batch=1)");
  TORCH_CHECK(w_q.dim() == 2, "w_q must be [C_out, C_in]");

  const int64_t C_in  = x_q.size(0);
  const int64_t H     = x_q.size(1);
  const int64_t W     = x_q.size(2);
  const int64_t C_out = w_q.size(0);

  TORCH_CHECK(
      w_q.size(1) == C_in,
      "w_q shape mismatch: expected [C_out, C_in] with C_in=",
      C_in, ", got C_in=", w_q.size(1));
  TORCH_CHECK(
      scale.numel() == C_out,
      "scale must have size C_out, got ", scale.numel(), " vs ", C_out);
  TORCH_CHECK(
      bias.numel() == C_out,
      "bias must have size C_out, got ", bias.numel(), " vs ", C_out);

  // Flatten spatial dims: X_mat = [H*W, C_in] (int8)
  // Start from [C_in, H, W], permute to [H, W, C_in], then view.
  const int64_t num_spatial = H * W;
  auto x_perm = x_q.permute({1, 2, 0}).contiguous();  // [H, W, C_in]
  auto x_mat  = x_perm.view({num_spatial, C_in});     // [H*W, C_in]

  // Allocate Z_mat in int32: [H*W, C_out]
  auto options_i32 = x_q.options().dtype(at::kInt);
  auto Z_mat = torch::empty({num_spatial, C_out}, options_i32);

  // cuBLAS uses column-major convention. We use the standard trick:
  //
  // Let A_row = X_mat  [M, K]  = [H*W, C_in]  row-major
  //     B_row = W_mat  [K, N]  = [C_in, C_out] row-major
  // We want: C_row = A_row * B_row  [M, N] = [H*W, C_out].
  //
  // In column-major:
  //   A_col = A_row^T → [K, M]
  //   B_col = B_row^T → [N, K]
  //   C_col = C_row^T → [N, M]
  //
  // Then C_col = B_col * A_col gives the same memory layout as C_row.
  //
  // So we call:
  //   cublasGemmEx(handle, N, M, K, B_col, A_col, C_col)
  //
  // and interpret the resulting [N, M] column-major buffer as
  // [M, N] row-major.

  const int M = static_cast<int>(num_spatial);
  const int K = static_cast<int>(C_in);
  const int N = static_cast<int>(C_out);

  const int8_t* A_data = x_mat.data_ptr<int8_t>();     // A_row
  const int8_t* B_data = w_q.data_ptr<int8_t>();       // W_row, but we use as B_col = W_row^T
  int32_t*      C_data = Z_mat.data_ptr<int32_t>();    // Z_mat buffer

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CUBLAS_CHECK(cublasSetStream(handle, stream));

  // alpha, beta for int32 compute
  int32_t alpha = 1;
  int32_t beta  = 0;

  // Matrix dims in column-major:
  // B_col: [N, K] with leading dim N
  // A_col: [K, M] with leading dim K
  // C_col: [N, M] with leading dim N
  int lda = N;  // leading dimension of B_col
  int ldb = K;  // leading dimension of A_col
  int ldc = N;  // leading dimension of C_col

  TORCH_CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N,       // op(B_col)
      CUBLAS_OP_N,       // op(A_col)
      N,                 // m: rows of C_col
      M,                 // n: cols of C_col
      K,                 // k
      &alpha,
      B_data, CUDA_R_8I, lda,   // B_col
      A_data, CUDA_R_8I, ldb,   // A_col
      &beta,
      C_data, CUDA_R_32I, ldc,  // C_col (Z_mat buffer)
      CUDA_R_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Now Z_mat is effectively [H*W, C_out] in row-major layout
  // (via the transpose trick above).

  // Allocate output in float32: [H*W, C_out]
  auto Y_mat = torch::empty({num_spatial, C_out},
                            x_q.options().dtype(at::kFloat));

  const int total = static_cast<int>(num_spatial * C_out);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  scale_bias_epilogue_kernel<<<blocks, threads, 0, stream>>>(
      Z_mat.data_ptr<int32_t>(),
      scale.data_ptr<float>(),
      bias.data_ptr<float>(),
      Y_mat.data_ptr<float>(),
      static_cast<int>(num_spatial),
      static_cast<int>(C_out));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Reshape back to [C_out, H, W]
  auto Y_hw_c = Y_mat.view({H, W, C_out});      // [H, W, C_out]
  auto Y = Y_hw_c.permute({2, 0, 1}).contiguous();  // [C_out, H, W]

  return Y;
}
