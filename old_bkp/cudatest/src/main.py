import torch
import time
N = 4096
FLOPS =  N*N*2*N
GFLOPS = FLOPS*1.0e-9

a = torch.randn(size=(N, N))
b = torch.randn(size=(N, N))
a, b = a.to(device="cuda:0"), a.to(device="cuda:0")

st = time.monotonic_ns()
c = a@b;
ed = time.monotonic_ns()

elapsed_ns = ed - st
elapsed_s = elapsed_ns * 1e-9;

print( f"elapsed : {elapsed_s} s\n");
print( f"gflops/s : {GFLOPS / elapsed_s}\n");