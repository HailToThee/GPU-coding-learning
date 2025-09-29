import os
os.environ['TORCH_EXTENSIONS_DIR'] = './tmp'
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = '''
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
'''

cpp_source = '''torch::Tensor square_matrix_1(torch::Tensor matrix);'''
square_matrix_extension = load_inline(
    name="square_matrix_extension_1",
    cpp_sources=[cpp_source],
    cuda_sources=cuda_source,
    functions=["square_matrix_1"],
    with_cuda=True,
    extra_cflags=['-O2'],
    verbose=True
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix_1(a))
