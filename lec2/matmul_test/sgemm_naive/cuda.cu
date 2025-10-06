#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B,
                            float beta, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

torch::Tensor sgemm_naive_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                  float alpha, float beta) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    TORCH_CHECK(B.size(0) == K, "A and B incompatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C has wrong shape");

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                       B.data_ptr<float>(), beta, C.data_ptr<float>());

    cudaDeviceSynchronize(); // 可选，用于调试；profiler 中可移除以减少开销
    return C;
}
