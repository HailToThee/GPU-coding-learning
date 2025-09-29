#include <torch/extension.h>

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled_transpose(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul_naive", torch::wrap_pybind_function(matmul_naive), "matmul_naive");
m.def("matmul_tiled", torch::wrap_pybind_function(matmul_tiled), "matmul_tiled");
m.def("matmul_tiled_transpose", torch::wrap_pybind_function(matmul_tiled_transpose), "matmul_tiled_transpose");
}