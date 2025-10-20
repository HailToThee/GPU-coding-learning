import os
import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile

os.environ['TORCH_EXTENSIONS_DIR'] = './matmul_test'
cuda_begin = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
"""

cuda_kernel = r"""
template<const int BM, const int BN, const int BK, const int TM>
__global__ void kernel_4(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    const uint threadCol = threadIdx.x % BN; // BN instead of TN
    const uint threadRow = threadIdx.x / BN;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    const float* A_tile = A + cRow*BM*K;
    const float* B_tile = B + cCol*BN;
    float* C_tile = C + cRow*BM*N + cCol*BN;

    // assert(BM*BK == blockDim.x); // blockDim.x = BM * BK, already in blockDim
    // assert(BN*BK == blockDim.x); // Not needed

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    float tmp[TM] = {0.0f};

    for (int idx = 0; idx < K; idx += BK) {
        // Load A to shared memory
        if (cRow * BM + innerRowA < M && idx + innerColA < K) {
            As[innerRowA * BK + innerColA] = A_tile[innerRowA * K + idx + innerColA];
        } else {
            As[innerRowA * BK + innerColA] = 0.0f;
        }
        // Load B to shared memory
        if (idx + innerRowB < K && cCol * BN + innerColB < N) {
            Bs[innerRowB * BN + innerColB] = B_tile[(idx + innerRowB) * N + innerColB];
        } else {
            Bs[innerRowB * BN + innerColB] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float tmpB = Bs[k * BN + threadCol];
            for (int i = 0; i < TM; ++i) {
                tmp[i] += As[(threadRow * TM + i) * BK + k] * tmpB;
            }
        }
        __syncthreads();
    }
    // Write back to C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        if (cRow * BM + threadRow * TM + i < M && cCol * BN + threadCol < N) {
            C_tile[(threadRow * TM + i) * N + threadCol] =
                alpha * tmp[i] + beta * C_tile[(threadRow * TM + i) * N + threadCol];
        }
    }
}

torch::Tensor kernel4_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                              float alpha, float beta) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "A and B incompatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C has wrong shape");
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 4;
    assert(BM % TM == 0);
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim(BM * BK, 1);
    kernel_4<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                       B.data_ptr<float>(), beta, C.data_ptr<float>());
    return C;
}
"""
cpp_source = r"""
#include <torch/extension.h>
torch::Tensor kernel4_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                  float alpha, float beta);
"""
kernel4_module = load_inline(
    name="kernel4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_begin+cuda_kernel,
    functions=['kernel4_wrapper'],
    verbose=True,
)

kernel4 = kernel4_module.kernel4_wrapper
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
    torch.profiler.tensorboard_trace_handler("./tmp/kernel4")(prof)

# Test parameters
M, N, K = 1024, 1024, 1024
alpha = 1.0
beta = 0.0

A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)
C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

# Profile the kernel
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=trace_handler,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True
) as prof:
    for step in range(5):
        if step == 0:
            _ = kernel4(A, B, C.clone(), alpha, beta)
        else:
            C_out = kernel4(A, B, C.clone(), alpha, beta)
        prof.step()

print("Profiler logs saved to: ./tmp/kernel4")