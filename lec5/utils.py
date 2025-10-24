import torch
cuda_begin = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
'''
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False, name=None):
    if name is None:
        name = funcs[0]
    return load_inline(cuda_sources=[cuda_src],
                        cpp_sources=[cpp_src],
                        functions=funcs,
                        extra_cuda_cflags=["--O2"] if opt else [],
                        verbose=verbose,
                        name=name)


