#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor kernel_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                             float alpha, float beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("kernel_wrapper", torch::wrap_pybind_function(kernel_wrapper), "kernel_wrapper");
}