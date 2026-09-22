import torch
import torch_npu

x=torch.full((1000,),2.0,dtype=torch.float32)
y=torch.full((1000,),3.0,dtype=torch.float32)

x_npu=x.npu()
y_npu=y.npu()

z_npu=x_npu*x_npu+y_npu
z=z_npu.cpu()

expected=torch.full((1000,),7.0,dtype=torch.float32)
print(f"验证结果：{'通过'if torch.allclose(expected,z) else '不通过'}")
