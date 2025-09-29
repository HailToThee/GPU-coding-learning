#include <torch/extension.h>

std::string hello_world() {
    return "Hello World!";
}
//这是 pybind11 提供的宏，用于定义一个 Python 模块（即一个 .so 动态库），使其可以被 Python import;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //can be called by TORCH_EXTENSION_NAME.hello_world();
    //Should use torch::wrap_pybind_function if is tensor wrap;
    //the last param is docstring;
m.def("hello_world", torch::wrap_pybind_function(hello_world), "hello_world");
}