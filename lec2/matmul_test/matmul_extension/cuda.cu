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

template<const int BM, const int BN, const int BK, const int TM>
__global__ void kernel(int M, int N, int K, float alpha, const float* A, const float* B,
                       float beta, float* C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    const float* A_tile = A + cRow*K;
    const float* B_tile = B + cCol;
    // Note: C_tile is not used in write-back; we compute global address directly
    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;
    float tmp[TM] = {0.0f};
    for (int idx = 0; idx < K; idx += BK) {
        // Load A to shared memory
        if (innerRowA < BM && innerColA < BK && (idx + innerColA) < K) {
            As[innerRowA * BK + innerColA] = A_tile[innerRowA * K + innerColA];
        } else {
            As[innerRowA * BK + innerColA] = 0.0f;
        }

        // Load B to shared memory
        if (innerRowB < BK && innerColB < BN && (idx + innerRowB) < K) {
            Bs[innerRowB * BN + innerColB] = B_tile[innerRowB * N + innerColB];
        } else {
            Bs[innerRowB * BN + innerColB] = 0.0f;
        }

        __syncthreads();

        // Compute block multiply
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int t = 0; t < TM; ++t) {
                const int row = innerRowA;
                const int col = t * blockDim.x + threadIdx.x;
                if (row < BM && col < BN) {
                    tmp[t] += As[row * BK + k] * Bs[k * BN + col % BN];
                }
            }
        }

        __syncthreads();
        A_tile += BK;
        B_tile += BK * N;
    }

    // Write back
    #pragma unroll
    for (int t = 0; t < TM; ++t) {
        const int row = innerRowA;
        const int col = t * blockDim.x + threadIdx.x;
        if (row < BM && col < BN) {
            int global_row = cRow * BM + row;
            int global_col = cCol * BN + col;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = alpha * tmp[t] + beta * C[global_row * N + global_col];
            }
        }
    }
}
torch::Tensor kernel_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                             float alpha, float beta) {
    CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(C);
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    const int BM = 32;
    const int BN = 32;
    const int BK = 8;
    const int TM = 1;
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim(BM * BK, 1);
    kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                       B.data_ptr<float>(), beta, C.data_ptr<float>());
    return C; 
}
