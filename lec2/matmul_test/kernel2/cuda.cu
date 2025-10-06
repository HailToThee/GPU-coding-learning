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

template <const int BLOCKSIZE>
__global__ void kernel2(int M, int N, int K, float alpha, const float *A, const float* B,
                    float beta, float* C){
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    if(x < M && y < M){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
torch::Tensor kernel2_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                  float alpha, float beta){
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    TORCH_CHECK(B.size(0) == K, "A and B incompatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C has wrong shape");
    const int BLOCKSIZE = 32;
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    kernel2<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                       B.data_ptr<float>(), beta, C.data_ptr<float>());
    return C;
}
