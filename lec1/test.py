# import os
# test_dir = './testdir'
# if not os.path.exists(test_dir):
#     os.makedirs(test_dir)
# os.environ['TORCH_EXTENSIONS_DIR'] = test_dir

import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))

tmp_dir_path = os.path.join(current_dir, 'tmp')
sys.path.append(tmp_dir_path)

import square_matrix_extension.square_matrix_extension as ext0  # type: ignore
import square_matrix_extension_1.square_matrix_extension_1 as ext1  #type: ignore


def square_2(a):
    return ext0.square_matrix(a) 

def square_3(a):
    return ext1.square_matrix_1(a) 

def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()


time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

with torch.profiler.profile() as prof:
    torch.square(b)
    
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("./testdir/test_trace_1.json")

print("=============")
print("Profiling square_matrix")
print("=============")

with torch.profiler.profile() as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("./testdir/test_trace_2.json")

print("=============")
print("Profiling square matrix 2")
print("=============")

with torch.profiler.profile() as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("./testdir/test_trace_3.json")
