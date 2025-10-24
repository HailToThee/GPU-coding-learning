import torch
from torch.utils.cpp_extension import load_inline
from torch.profiler import profile
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'
os.environ['TORCH_EXTENSIONS_DIR'] = './matmul_test'

cuda_begin = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
'''

cuda_kernel = r'''
/*
The cuda kernel is defined within a block. Manually, we define block size and grid size.
So when we defined the blocksize within each grid, we can get the row and column of block we are operating on.
So cRow and cCol are defined as blockIdx.x and blockIdx.y respectively.
Writing a kernel, the first step is to seperate the Block and Grid. 
Once we know our Block and grid, we can locate the pointer to A, B, C matrix.

each block is responsible for computing a BM x BN sub-matrix of C.
inside each block, load A:BM*BK and B:BK*BN into shared memory in loops.
each thread computes a TM x 1 sub-matrix of C. using register to store.
write back to global memory after all K is done.
*/

template<const int BM, const int BN, const int BK, const int TM>
__global__ void kernel(int M, int N, int K, float alpha, const float* A, const float* B,
                       float beta, float* C) {
    
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    const float* A_tile = A + cRow*K*BM;
    const float* B_tile = B + cCol*BN;
    float* C_tile = C + cRow*BM*N + cCol*BN;

    const uint threadRow = threadIdx.x / BK;    // ThreadRow and ThreadCol is the target position of each element
    const uint threadCol = threadIdx.x % BK;

    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;

    float tmp[TM] = {0.0f};
    for (int idx = 0; idx < K; idx += BK) {
        // Load A to shared memory
        if (innerRowA < BM && innerColA < BK && (idx + innerColA) < K) {
            As[innerRowA * BK + innerColA] = A_tile[innerRowA * K + innerColA + idx];
        } else {
            As[innerRowA * BK + innerColA] = 0.0f;
        }

        // Load B to shared memory
        if (innerRowB < BK && innerColB < BN && (idx + innerRowB) < K) {
            Bs[innerRowB * BN + innerColB] = B_tile[(idx + innerRowB) * N + innerColB];
        } else {
            Bs[innerRowB * BN + innerColB] = 0.0f;
        }

        __syncthreads();

        // Compute block multiply
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            float tmpB = Bs[k * BN + threadCol];
            for (int t = 0; t < TM; ++t) {
                tmp[t] += As[(threadRow * TM + t) * BK + k] * tmpB;
            }
        }

        __syncthreads();
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        if (cRow * BM + threadRow * TM + i < M && cCol * BN + threadCol < N) {
            C_tile[(threadRow * TM + i) * N + threadCol] =
                alpha * tmp[i] + beta * C_tile[(threadRow * TM + i) * N + threadCol];
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
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim(BM * BK, 1);
    kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<float>(),
                                       B.data_ptr<float>(), beta, C.data_ptr<float>());
    return C; 
}
'''
cpp_source = r'''
#include <torch/extension.h>
torch::Tensor kernel_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                             float alpha, float beta);
'''

matmul_cuda = load_inline(
    name='kernel',
    cpp_sources=cpp_source,
    cuda_sources=cuda_begin+cuda_kernel,
    functions=['kernel_wrapper'],
    with_cuda=True,
    extra_cuda_cflags=['--use_fast_math']
)
kernel_wrapper = matmul_cuda.kernel_wrapper
def check():
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    C = torch.randn(M, N, device='cuda')
    C_ref = C.clone()

    alpha = 1.0
    beta = 0.0

    with profile(record_shapes=True, profile_memory=True, with_stack=True) as prof:
        # ❌ 错误：kernel_wrapper(M, N, K, alpha, A, B, beta, C)
        # ✅ 正确：
        kernel_wrapper(A, B, C, alpha, beta)  # ← 只传这 5 个参数

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    C_ref = alpha * torch.matmul(A, B) + beta * C_ref
    if torch.allclose(C, C_ref, atol=1e-5):
        print("✅ Result is correct!")
    else:
        print("❌ Result is incorrect!")
        print("Max diff:", (C - C_ref).abs().max().item())

check()