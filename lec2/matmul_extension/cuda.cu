#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void matmul_kernel(float* m, float* n, float* out, int h, int w, int k){
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < h && c < w) {
        float value = 0;
        for (int i = 0; i < k; i++) {
            value += m[r * k + i] * n[i * w + c];
        }
        out[r * w + c] = value;
    }
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n){
    CHECK_INPUT(m);
    CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch");
    auto output = torch::zeros({h, w}, m.options());
    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    return output;
}
