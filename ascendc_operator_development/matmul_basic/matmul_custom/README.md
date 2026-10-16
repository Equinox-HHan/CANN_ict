# 基于高阶API的矩阵乘算子开发

## 算子分析

### 1. 明确算子的数学表达式和计算逻辑
    
    核心计算公式：C=A*B+Bias
- CopyIn阶段对应`SetTensorA,SetTensorB,SetBias`接口
- Compute阶段对应`Iterate`接口
- CopyOut阶段对应`GetTensorC`接口

### 2. 明确输入输出

    number dtype shape format

### 3. 确定和函数名称和参数

## 创建算子工程

## 算子开发
AscendC提供了高阶API，封装了数据的切分搬运和计算逻辑

在Host侧调用API获取参数并传递到Kernel侧进行计算

### 1. 结构体定义：

### 2. 矩阵乘Tiling算法实现
- Tiling函数实现
动态计算矩阵乘法所需的切分参数并将参数填充到预定义Tiling数据结构
- 算子原型定义
对算子做出描述并且关联Tiling等实现函数

### 3. 算子核函数的开发
1. 准备工作
2. 创建和函数类
3. 进行核函数定义
4. Init函数的实现
5. Process函数实现
6. 重新写入核函数 