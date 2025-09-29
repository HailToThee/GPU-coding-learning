#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        result[row * width + col] = matrix[row * width + col] * matrix[row * width + col];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix){
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);
    auto result = torch::empty_like(matrix);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    square_matrix_kernel<<<grid, block>>>(matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
}
