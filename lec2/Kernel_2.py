import torch
import os
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

## threadId = threadIdx.x + blockDim.x * threadIdx.y + 
#             blockDim.x * blockDim.y * threadIdx.z

cuda_kernel = r"""
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
"""
cuda_source = cuda_begin+cuda_kernel
cpp_source = r"""
#include <torch/extension.h>
torch::Tensor kernel2_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                  float alpha, float beta);
"""

kernel2_module = load_inline(
    name="kernel2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['kernel2_wrapper'],
    verbose=True
)

kernel2 = kernel2_module.kernel2_wrapper

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
    torch.profiler.tensorboard_trace_handler(log_dir)(prof)
log_dir = "./tmp/kernel2"

os.makedirs("./tmp/kernel2", exist_ok=True)

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
            _ = kernel2(A, B, C.clone(), alpha, beta)
        else:
            C_out = kernel2(A, B, C.clone(), alpha, beta)
        prof.step()  

        

print(f"Profiler logs saved to: {log_dir}")