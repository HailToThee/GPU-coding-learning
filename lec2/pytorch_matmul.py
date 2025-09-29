import os
import torch
from torch.utils.cpp_extension import load_inline

os.environ["TORCH_EXTENSIONS_DIR"] = "./ext_build"
# ===== CUDA Kernel 源码 =====

cuda_begin = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
"""

# naive kernel
cuda_naive = r"""
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0;
        for (int i = 0; i < K; i++) {
            val += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(32, 32);
    dim3 blocks((N + 31) / 32, (M + 31) / 32);
    matmul_naive_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

# tiled kernel (原版)
cuda_tiled = r"""
#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            val += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cuda_tiled_transpose = r"""
#define TILE_SIZE 32

__global__ void matmul_tiled_transpose_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE]; // 这里我们存的是转置的

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            // 注意这里交换了 threadIdx.x / threadIdx.y 来存转置
            tile_B[threadIdx.x][threadIdx.y] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            // 注意访问方式也变了
            val += tile_A[threadIdx.y][i] * tile_B[threadIdx.x][i];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled_transpose(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_transpose_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cpp_source = """
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled_transpose(torch::Tensor A, torch::Tensor B);
"""

# ===== Build extension =====
module = load_inline(
    name="matmul_bench",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_begin + cuda_naive + cuda_tiled + cuda_tiled_transpose],
    functions=["matmul_naive", "matmul_tiled", "matmul_tiled_transpose"],
    verbose=True,
)

# ===== Benchmark =====
def benchmark(func, A, B, repeat=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(3):
        func(A, B)

    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start.record()
        func(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)

# ===== Test =====
if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024  # 大矩阵
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    t_naive = benchmark(module.matmul_naive, A, B)
    t_tiled = benchmark(module.matmul_tiled, A, B)
    t_tiled_T = benchmark(module.matmul_tiled_transpose, A, B)
    t_torch = benchmark(torch.matmul, A, B)

    print("\n===== Benchmark Results (ms) =====")
    print(f"Naive CUDA kernel        : {t_naive:.3f} ms")
    print(f"Tiled CUDA kernel        : {t_tiled:.3f} ms")
    print(f"Tiled CUDA kernel (B^T)  : {t_tiled_T:.3f} ms")
    print(f"torch.matmul (cuBLAS)    : {t_torch:.3f} ms")
