#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void square_matrix_kernel1(const float* input, float* output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

torch::Tensor square_matrix_1(torch::Tensor matrix){
    const auto n = matrix.numel();
    auto result = torch::empty_like(matrix);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    square_matrix_kernel1<<<blocks, threads>>>(matrix.data_ptr<float>(), result.data_ptr<float>(), n);
    return result;
}
