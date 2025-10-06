#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor kernel3_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                              float alpha, float beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("kernel3_wrapper", torch::wrap_pybind_function(kernel3_wrapper), "kernel3_wrapper");
}