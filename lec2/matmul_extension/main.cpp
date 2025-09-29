#include <torch/extension.h>
torch::Tensor matmul(torch::Tensor m, torch::Tensor n);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul", torch::wrap_pybind_function(matmul), "matmul");
}