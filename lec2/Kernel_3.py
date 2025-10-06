import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile
import os

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
template<const int BLOCKSIZE>
__global__ void kernel_3(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // Global memory pointers for current tile
    const float* A_tile = A + cRow * BLOCKSIZE * K + threadRow * K;
    const float* B_tile = B + threadCol + threadRow * N;  // 注意：B 是 K x N，按行主序
    float* C_tile = C + cRow * BLOCKSIZE * N + cCol * BLOCKSIZE + threadRow * N + threadCol;

    float tmp = 0.0;

    for (int idx = 0; idx < K; idx += BLOCKSIZE) {
        // Load A and B into shared memory
        if (cRow * BLOCKSIZE + threadRow < M && idx + threadCol < K) {
            As[threadRow][threadCol] = A_tile[threadCol];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }

        if (idx + threadRow < K && cCol * BLOCKSIZE + threadCol < N) {
            Bs[threadRow][threadCol] = B[idx + threadRow * N + threadCol];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();

        // Advance pointers
        A_tile += BLOCKSIZE;
        B_tile += BLOCKSIZE * N;
    }

    // Write back to global memory
    if (cRow * BLOCKSIZE + threadRow < M && cCol * BLOCKSIZE + threadCol < N) {
        *C_tile = alpha * tmp + beta * (*C_tile);
    }
}

torch::Tensor kernel3_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                              float alpha, float beta) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    TORCH_CHECK(B.size(0) == K, "A and B incompatible for matrix multiplication");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C has wrong shape");

    const int BLOCKSIZE = 32;
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);  // 1D thread block

    kernel_3<BLOCKSIZE><<<gridDim, blockDim>>>(
        M, N, K, alpha,
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        beta,
        C.data_ptr<float>()
    );
    return C;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor kernel3_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                              float alpha, float beta);
"""

# Load the CUDA kernel
kernel3_module = load_inline(
    name="kernel3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_begin + cuda_kernel,
    functions=['kernel3_wrapper'],
    verbose=True
)

kernel3 = kernel3_module.kernel3_wrapper

# Profiler setup
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
    torch.profiler.tensorboard_trace_handler("./tmp/kernel3")(prof)

os.makedirs("./tmp/kernel3", exist_ok=True)

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
            _ = kernel3(A, B, C.clone(), alpha, beta)
        else:
            C_out = kernel3(A, B, C.clone(), alpha, beta)
        prof.step()

print("Profiler logs saved to: ./tmp/kernel3")