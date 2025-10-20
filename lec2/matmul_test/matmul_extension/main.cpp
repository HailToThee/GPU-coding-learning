#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("kernel_wrapper", torch::wrap_pybind_function(kernel_wrapper), "kernel_wrapper");
}