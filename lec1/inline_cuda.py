import os
os.environ['TORCH_EXTENSIONS_DIR'] = './tmp'
if not os.path.exists('./tmp'):
    os.makedirs('./tmp')
    
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = '''
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
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

square_matrix_extension = load_inline(
    name="square_matrix_extension",
    cpp_sources=[cpp_source],
    cuda_sources=cuda_source,
    functions=["square_matrix"],
    with_cuda=True,
    extra_cflags=['-O2'],
    verbose=True
)
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))
