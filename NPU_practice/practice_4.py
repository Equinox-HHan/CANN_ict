import torch, torch_npu, torch.nn.functional as F
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, time

# Step 1: 加载图片并转为张量 [1, 3, H, W]
img = Image.open('./images/lena.jpg').convert('RGB')
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(f"图片张量形状: {img_tensor.shape}")

# 显示输入图片
plt.figure(figsize=(6, 5))
plt.imshow(img)
plt.title('Input: Lena (512x512)')
plt.axis('off')
plt.show()

# Step 2: 构造 5×5 高斯卷积核
size, sigma = 5, 1.0
coords = torch.arange(size, dtype=torch.float32) - size // 2  # [-2, -1, 0, 1, 2]
g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))            # 一维高斯
kernel2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)                 # 外积得到二维高斯
kernel2d = kernel2d / kernel2d.sum()                           # 归一化（权重之和=1）
weight = kernel2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1) # [3, 1, 5, 5]

# Step 3: CPU 卷积计时
start = time.time()
blur_cpu = F.conv2d(img_tensor, weight, padding=2, groups=3)
cpu_time = time.time() - start

# Step 4: NPU 卷积计时
img_npu = img_tensor.npu()
weight_npu = weight.npu()
torch.npu.synchronize()
start = time.time()
blur_npu = F.conv2d(img_npu, weight_npu, padding=2, groups=3)
torch.npu.synchronize()
npu_time = time.time() - start
result = blur_npu.cpu()

# Step 5: 打印耗时与加速比
print(f"CPU 卷积耗时: {cpu_time*1000:.2f} ms")
print(f"NPU 卷积耗时: {npu_time*1000:.2f} ms")
print(f"加速比: {cpu_time/npu_time:.1f}x")

# Step 6: 显示输出图片
blur_img = Image.fromarray((result.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
plt.figure(figsize=(6, 5))
plt.imshow(blur_img)
plt.title(f'Gaussian Blur (NPU, {npu_time*1000:.1f} ms)')
plt.axis('off')
plt.show()
