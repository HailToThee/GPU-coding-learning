import sys
import os
import torch
import torch.profiler as profiler
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import matmul_extension.matmul_extension as ext0 # type: ignore

def matmul_python(a, b):
    c = torch.zeros(a.shape[0], b.shape[1], device=a.device)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
    return c

def matmul_cuda(a, b):
    return ext0.matmul(a, b)


x = torch.randn(5000, 784).cuda()
weights = torch.randn(784, 10).cuda()



torch.matmul(x, weights)
# matmul_python(x, weights)
matmul_cuda(x, weights)

print("==============")
print("Start Profile...")
print("==============")

# 纯 Python 的 Matmul
# with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof1:
#     with profiler.record_function("Python_Matmul"):
#         matmul_python(x, weights)

# print("\n--- Python Matmul  ---")
# print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# PyTorch 内置的 Matmul
with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof2:
    with profiler.record_function("PyTorch_Matmul"):
        y = torch.matmul(x, weights)

print("\n--- PyTorch Matmul  ---")
print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 自定义的 CUDA Matmul
with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof3:
    with profiler.record_function("Custom_CUDA_Matmul"):
        y = matmul_cuda(x, weights)

print("\n--- CUDA Matmul ---")
print(prof3.key_averages().table(sort_by="cuda_time_total", row_limit=10))

