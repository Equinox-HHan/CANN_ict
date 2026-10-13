//pybind是一个将cpp代码与python解释器集成的库，从而实现“让python解释器能看懂cpp代码”
//使用Pybind提供的API将算子aclnn接口与Python解释器进行绑定
//只是将aclnn接口封装成Python函数，方便在Python中调用

#include<torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
//其中pytorch_npu_helper.hpp内定义了宏EXEC_NPU_CMD用于执行aclnn二段式接口

//算子调用函数编写
at::Tensor npu_add_custom_template(const at::Tensor &x, const at::Tensor &y) {
    at::Tensor z = at::empty_like(x);
    EXEC_NPU_CMD(aclnnAddCustomTemplate, x, y, z);
    return z;
}

/*
绑定算子调用函数
借助pybind11的PYBIND11_MODULE宏，将C++实现的npu_add_custom_template函数，
封装成可供Python导入和调用的扩展模块接口
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("npu_add_custom_template", &npu_add_custom_template, "torch add");
}