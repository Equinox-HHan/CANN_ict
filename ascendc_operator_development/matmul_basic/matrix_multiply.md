# 矩阵乘法、高维空间与神经网络偏置：学习笔记

## 一、矩阵乘法的计算规则

设 $A\in\mathbb R^{m\times n}$、$B\in\mathbb R^{n\times p}$，则 $C=AB\in\mathbb R^{m\times p}$，其中

$$
c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
$$

即 **左矩阵第 $i$ 行与右矩阵第 $j$ 列做点积**。只有左矩阵的列数等于右矩阵的行数，乘法才有定义。

例如：

$$
\begin{aligned}
A&=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix},\\
AB&=\begin{bmatrix}1\cdot5+2\cdot7&1\cdot6+2\cdot8\\3\cdot5+4\cdot7&3\cdot6+4\cdot8\end{bmatrix}
=\begin{bmatrix}19&22\\43&50\end{bmatrix}.
\end{aligned}
$$

矩阵乘法不是对应元素逐项相乘。

## 二、代数本质：列向量的线性组合

若矩阵 $A=[\boldsymbol a_1\ \boldsymbol a_2\ \cdots\ \boldsymbol a_n]$，输入向量 $\boldsymbol x=(x_1,\ldots,x_n)^T$，则

$$
\boxed{A\boldsymbol x=x_1\boldsymbol a_1+\cdots+x_n\boldsymbol a_n.}
$$

例如：

$$
\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}3\\2\end{bmatrix}
=3\begin{bmatrix}2\\1\end{bmatrix}+2\begin{bmatrix}1\\3\end{bmatrix}
=\begin{bmatrix}8\\9\end{bmatrix}.
$$

对矩阵乘矩阵，若 $B=[\boldsymbol b_1\ \cdots\ \boldsymbol b_p]$，则

$$
\boxed{AB=[A\boldsymbol b_1\ \cdots\ A\boldsymbol b_p].}
$$

即 $A$ 分别作用在 $B$ 的每一个列向量上。

## 三、几何本质：矩阵代表线性变换

### 1. 二维空间

二维标准基向量为

$$
\boldsymbol e_1=\begin{bmatrix}1\\0\end{bmatrix},\qquad
\boldsymbol e_2=\begin{bmatrix}0\\1\end{bmatrix}.
$$

任意向量 $\boldsymbol v=x\boldsymbol e_1+y\boldsymbol e_2$。由线性性可得

$$
A\boldsymbol v=xA\boldsymbol e_1+yA\boldsymbol e_2.
$$

**矩阵的第 $j$ 列就是标准基向量 $\boldsymbol e_j$ 经过该变换后的结果。** 因而只要知道基向量如何变化，就知道所有向量如何变化。

例如

$$
A=\begin{bmatrix}2&0\\0&1\end{bmatrix},\qquad
A\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}2x\\y\end{bmatrix}
$$

表示沿 $x$ 轴拉伸两倍，而 $y$ 方向保持不变。一般线性变换还可以旋转、切变、反射、投影或降维，但都保持原点不动。

### 2. 三维空间

三维空间有三个标准基向量：

$$
\boldsymbol e_1=(1,0,0)^T,\quad
\boldsymbol e_2=(0,1,0)^T,\quad
\boldsymbol e_3=(0,0,1)^T.
$$

任何 $\boldsymbol v=(x,y,z)^T$ 都满足

$$
A\boldsymbol v=xA\boldsymbol e_1+yA\boldsymbol e_2+zA\boldsymbol e_3.
$$

例如

$$
A=\begin{bmatrix}2&0&0\\0&1&0\\0&0&3\end{bmatrix}
$$

将 $(x,y,z)^T$ 映射为 $(2x,y,3z)^T$，相当于分别沿三个坐标轴进行不同程度的拉伸。

### 3. 更高维空间

在 $n$ 维空间中，任意向量可写为

$$
\boldsymbol x=\sum_{j=1}^n x_j\boldsymbol e_j,
\qquad A\boldsymbol x=\sum_{j=1}^n x_jA\boldsymbol e_j.
$$

维度增加并不改变原理：只是基向量数量增加。$m\times n$ 矩阵表示从 $\mathbb R^n$ 到 $\mathbb R^m$ 的线性映射，输入和输出的维度可以不同。

例如

$$
\begin{bmatrix}1&0&2\\0&1&1\end{bmatrix}
\begin{bmatrix}3\\4\\5\end{bmatrix}
=\begin{bmatrix}13\\9\end{bmatrix}.
$$

这是三维到二维的映射，可能丢失信息。

## 四、为什么矩阵乘法是“行乘列”？

两个线性变换先后作用在同一向量上，若先执行 $B$、后执行 $A$，则

$$
\boldsymbol z=A(B\boldsymbol x)=(AB)\boldsymbol x.
$$

写成分量：

$$
y_k=\sum_j b_{kj}x_j,
\qquad
z_i=\sum_k a_{ik}y_k
=\sum_j\left(\sum_k a_{ik}b_{kj}\right)x_j.
$$

为了用单一矩阵 $C$ 表示整个复合变换，必须定义

$$
c_{ij}=\sum_k a_{ik}b_{kj}.
$$

因此，“行乘列”规则来自**线性变换复合**的需要，并非任意规定。

### 变换顺序不可随意交换

设

$$
A=\begin{bmatrix}2&0\\0&1\end{bmatrix}
\quad\text{（横向拉伸两倍）},\qquad
B=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
\quad\text{（逆时针旋转 }90^\circ\text{）}.
$$

则

$$
AB=\begin{bmatrix}0&-2\\1&0\end{bmatrix},\qquad
BA=\begin{bmatrix}0&-1\\2&0\end{bmatrix}.
$$

对 $\boldsymbol v=(1,1)^T$，有

$$
AB\boldsymbol v=(-2,1)^T,\qquad
BA\boldsymbol v=(-1,2)^T.
$$

所以一般 $AB\ne BA$：**先旋转后拉伸，与先拉伸后旋转，结果通常不同。**

## 五、深度学习中的权重矩阵

神经网络的全连接层通常写作

$$
\boldsymbol z=W\boldsymbol x+\boldsymbol b.
$$

其中 $\boldsymbol x\in\mathbb R^n$ 是输入特征，$W\in\mathbb R^{m\times n}$ 是权重矩阵，$\boldsymbol z\in\mathbb R^m$ 是输出的预激活向量，$\boldsymbol b\in\mathbb R^m$ 是偏置向量。

权重矩阵可以把 $n$ 个输入特征组合成 $m$ 个新特征。它对应高维特征空间中的线性变换，并不要求输入输出维度相等。

## 六、偏置向量的本质：平移

### 1. 为什么需要偏置？

纯线性变换始终满足

$$
W\boldsymbol 0=\boldsymbol 0.
$$

因此无论如何调整权重，原点都不能移动。加入偏置后

$$
\boxed{\boldsymbol z=W\boldsymbol x+\boldsymbol b,}
$$

可先进行线性变换，再将所有输出统一平移 $\boldsymbol b$。这称为**仿射变换**。

例如

$$
W=\begin{bmatrix}2&0\\0&1\end{bmatrix},\quad
\boldsymbol b=\begin{bmatrix}1\\2\end{bmatrix},\quad
\boldsymbol x=\begin{bmatrix}3\\1\end{bmatrix},
$$

则

$$
W\boldsymbol x+\boldsymbol b
=\begin{bmatrix}6\\1\end{bmatrix}+\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}7\\3\end{bmatrix}.
$$

几何上先横向拉伸，再向右平移 1、向上平移 2。**偏置改变位置，而不改变线性部分造成的形状和面积变化。**

### 2. 偏置与激活阈值

神经元通常还会经过激活函数，例如

$$
h=\operatorname{ReLU}(wx+b)=\max(0,wx+b).
$$

若 $w=2$、$b=-4$，则 $h=\max(0,2x-4)$，在 $x>2$ 时开始输出正值；若将偏置改为 $-2$，则在 $x>1$ 时开始输出正值。

因此偏置还可以调整神经元的激活阈值。

### 3. 偏置本身不会引入非线性

$W\boldsymbol x+\boldsymbol b$ 仍是仿射映射。神经网络一般再应用非线性激活函数

$$
\boxed{\boldsymbol h=\sigma(W\boldsymbol x+\boldsymbol b)}
$$

来增强表达能力。

## 七、用齐次坐标把偏置并入矩阵

通过在输入末尾增加一个常数 $1$，可以将仿射变换写成矩阵乘法。

例如

$$
\begin{bmatrix}2&0&1\\0&3&2\end{bmatrix}
\begin{bmatrix}x\\y\\1\end{bmatrix}
=\begin{bmatrix}2x+1\\3y+2\end{bmatrix}.
$$

若同时保留扩展后的输出维度，则

$$
\begin{bmatrix}2&0&1\\0&3&2\\0&0&1\end{bmatrix}
\begin{bmatrix}x\\y\\1\end{bmatrix}
=\begin{bmatrix}2x+1\\3y+2\\1\end{bmatrix}.
$$

这是**齐次坐标**的基本思想，也广泛用于计算机图形学中的平移、旋转、缩放等操作的统一表示。

## 八、为什么神经网络需要非线性激活函数？

假设连续两层都没有非线性激活：

$$
\boldsymbol h=W_1\boldsymbol x+\boldsymbol b_1,
\qquad
\boldsymbol y=W_2\boldsymbol h+\boldsymbol b_2.
$$

代入可得

$$
\begin{aligned}
\boldsymbol y
&=W_2(W_1\boldsymbol x+\boldsymbol b_1)+\boldsymbol b_2\\
&=(W_2W_1)\boldsymbol x+(W_2\boldsymbol b_1+\boldsymbol b_2).
\end{aligned}
$$

令 $W'=W_2W_1$、$\boldsymbol b'=W_2\boldsymbol b_1+\boldsymbol b_2$，则

$$
\boldsymbol y=W'\boldsymbol x+\boldsymbol b'.
$$

也就是说，**任意多层纯仿射变换的复合仍是一个仿射变换**。若希望神经网络表示复杂的非线性关系，通常需要在层间引入 ReLU 等非线性激活函数。

## 九、知识结构总览

| 层次 | 数学表达 | 直观理解 |
|---|---|---|
| 矩阵乘法 | $c_{ij}=\sum_k a_{ik}b_{kj}$ | 行与列做点积 |
| 矩阵乘向量 | $A\boldsymbol x=\sum_jx_j\boldsymbol a_j$ | 对矩阵列向量做线性组合 |
| 空间变换 | $A\boldsymbol e_j=\boldsymbol a_j$ | 基向量的去向决定整个变换 |
| 变换复合 | $(AB)\boldsymbol x=A(B\boldsymbol x)$ | 右侧矩阵先作用 |
| 高维映射 | $A:\mathbb R^n\to\mathbb R^m$ | 特征空间的维度可以改变 |
| 权重 | $W\boldsymbol x$ | 线性组合和特征映射 |
| 偏置 | $W\boldsymbol x+\boldsymbol b$ | 在线性变换之后平移 |
| 激活函数 | $\sigma(W\boldsymbol x+\boldsymbol b)$ | 引入非线性表达能力 |
| 齐次坐标 | $\tilde W[\boldsymbol x;1]$ | 将平移并入扩展矩阵乘法 |

**一句话总结：矩阵通过控制基向量的去向来实现线性空间变换；矩阵乘法表示这些变换的复合；深度学习中的权重负责特征的线性变换，偏置负责平移，激活函数负责引入非线性。**

## 十、练习题

**练习 1：矩阵乘法与变换顺序**

设 $A=\begin{bmatrix}2&0\\0&1\end{bmatrix}$、$B=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$。计算 $AB$ 与 $BA$，解释几何差别。

**练习 2：观察列向量**

设 $A=\begin{bmatrix}1&2\\0&1\end{bmatrix}$。求 $A\boldsymbol e_1$ 和 $A\boldsymbol e_2$，并判断单位正方形会变成什么形状。

**练习 3：神经网络层**

设

$$
\boldsymbol y=\operatorname{ReLU}\left(\begin{bmatrix}1&2\\-1&1\end{bmatrix}\boldsymbol x+\begin{bmatrix}-3\\1\end{bmatrix}\right).
$$

1. 当 $\boldsymbol x=(1,1)^T$ 时，计算输出。
2. 移除偏置后，输出如何变化？
3. 移除 ReLU 后，映射属于哪类变换？
4. 为什么加入 ReLU 后，整体通常不再是仿射变换？
