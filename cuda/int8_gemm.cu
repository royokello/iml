#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ------------- plain-C DP4A GEMM kernel (row-major) ------------- */

template<int TILE_M = 16, int TILE_N = 16>
__global__ void int8_gemm_kernel(const int8_t* __restrict__ A,
                                 const int8_t* __restrict__ B,
                                 int32_t*       __restrict__ C,
                                 int M, int N, int K)
{
    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;
    if (row >= M || col >= N)  return;

    int32_t sum = 0;
    for (int k = 0; k < K; ++k)
        sum += static_cast<int32_t>(A[row * K + k]) *
               static_cast<int32_t>(B[k * N + col]);

    C[row * N + col] = sum;
}

/* ------------- PyTorch binding ------------- */

torch::Tensor int8_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim()==2 && B.dim()==2, "A and B must be 2-D");
    TORCH_CHECK(A.size(1)==B.size(0), "K mismatch");
    TORCH_CHECK(A.dtype()==torch::kInt8 && B.dtype()==torch::kInt8, "must be int8");

    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);
    int8_gemm_kernel<<<grid, block>>>(A.data_ptr<int8_t>(),
                                      B.data_ptr<int8_t>(),
                                      C.data_ptr<int32_t>(),
                                      M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm", &int8_gemm, "INT8 GEMM (DP4A, SM6.1)");
}
