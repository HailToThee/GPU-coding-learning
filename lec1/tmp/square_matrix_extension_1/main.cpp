#include <torch/extension.h>
torch::Tensor square_matrix_1(torch::Tensor matrix);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("square_matrix_1", torch::wrap_pybind_function(square_matrix_1), "square_matrix_1");
}