import time
import torch
import torch_npu

sizes=[128,512,2048]
for N in sizes:
    a_cpu=torch.randn(N,N,dtype=torch.float32)
    b_cpu=torch.randn(N,N,dtype=torch.float32)

    t0=time.time()
    c_cpu=torch.matmul(a_cpu,b_cpu)
    time_cpu=time.time()-t0

    a_npu=a_cpu.npu()
    b_npu=b_cpu.npu()
    torch.npu.synchronize()
    t0=time.time()
    x_npu=torch.matmul(a_npu,b_npu)
    torch.npu.synchronize()
    time_npu=time.time()-t0

    speedup = time_cpu / time_npu
    print(f"{N:>8} | {time_npu*1000:>10.2f}ms | {time_npu*1000:>10.2f}ms | {speedup:>7.1f}x")


