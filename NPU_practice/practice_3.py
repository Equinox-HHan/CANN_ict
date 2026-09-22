import torch
import torch_npu
import time

batch_sizes = [8, 32, 128]
M, K, N = 256, 256, 256
reps = 20

print(f"{'B':>6} | {'bmm 耗时':>12} | {'循环 耗时':>12} | {'加速比':>8} | {'结果一致':>8}")
print("-" * 65)

for B in batch_sizes:
    # Step 1: 创建 (B, M, K) 和 (B, K, N) 随机矩阵，搬到 NPU
    a = torch.randn(B, M, K, dtype=torch.float32).npu()
    b = torch.randn(B, K, N, dtype=torch.float32).npu()

    # Step 2: Warmup
    _ = torch.bmm(a, b)
    torch.npu.synchronize()

    # Step 3: torch.bmm 计时
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(reps):
        c_bmm = torch.bmm(a, b)
    torch.npu.synchronize()
    t_bmm = (time.time() - t0) / reps * 1000

    # Step 4: 循环 torch.matmul 计时
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(reps):
        c_loop = torch.stack([torch.matmul(a[i], b[i]) for i in range(B)])
    torch.npu.synchronize()
    t_loop = (time.time() - t0) / reps * 1000

    # Step 5: 验证结果一致
    consistent = torch.allclose(c_bmm, c_loop, atol=1e-1)

    # Step 6: 打印结果
    speedup = t_loop / t_bmm
    print(f"{B:>6} | {t_bmm:>10.2f}ms | {t_loop:>10.2f}ms | {speedup:>7.1f}x | {'✓' if consistent else '✗':>8}")
