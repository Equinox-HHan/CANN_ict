import torch

# 1. 验证张量创建与补全
x = torch.randn(3, 3)
print("本地张量测试成功:\n", x)

# 2. 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前本地测试计算设备: {device}")