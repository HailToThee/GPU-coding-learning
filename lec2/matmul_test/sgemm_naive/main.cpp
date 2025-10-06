#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor sgemm_naive_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                                  float alpha, float beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("sgemm_naive_wrapper", torch::wrap_pybind_function(sgemm_naive_wrapper), "sgemm_naive_wrapper");
}