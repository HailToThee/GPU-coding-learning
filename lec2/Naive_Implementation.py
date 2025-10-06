import os
import torch
from torch.utils.cpp_extension import load_inline
import torch.profiler as profiler

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

# Naive GEMM CUDA kernel
cuda_naive_kernel = r"""
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
"""

# Host wrapper function
cuda_wrapper = r"""
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
"""

cuda_source = cuda_begin + cuda_naive_kernel + cuda_wrapper

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor sgemm_naive_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                  float alpha, float beta);
"""

sgemm_module = load_inline(
    name='sgemm_naive',
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source,
    functions=['sgemm_naive_wrapper'],
    verbose=True
)

sgemm_naive = sgemm_module.sgemm_naive_wrapper

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
    torch.profiler.tensorboard_trace_handler(log_dir)(prof)
log_dir = "./tmp/naive_gemm"

os.makedirs("./tmp/naive_gemm", exist_ok=True)

M, N, K = 1024, 1024, 1024
alpha = 1.0
beta = 0.0

A = torch.randn(M, K, device='cuda', dtype=torch.float32)
B = torch.randn(K, N, device='cuda', dtype=torch.float32)
C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

# 使用 profiler
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
            _ = sgemm_naive(A, B, C.clone(), alpha, beta)
        else:
            C_out = sgemm_naive(A, B, C.clone(), alpha, beta)
        prof.step()  

        

print(f"Profiler logs saved to: {log_dir}")