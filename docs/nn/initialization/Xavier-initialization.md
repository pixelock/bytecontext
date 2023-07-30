**Paper**

[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

# 原理

## 稳定初始阶段的前向传播

设计各种参数初始化方法的目的, 在于**尽量让输入输出具有同样的均值和方差, 从而能保证初始阶段模型的稳定性**. 输出的方差过大/过小, 可能会导致模型发生爆炸/消失.

我们用**二阶中心矩**来衡量输出相对于输入的变化, 可以看作是方差的近似.

对于输入节点数为$$m$$, 输出节点数为$$n$$的无激活函数的全连接层, 有:

$$
y_j = b_j + \sum_i x_i w_{i,j}
$$

我们通常用全零初始化偏置项$$b_j$$, 并且将$$w_{i,j}$$的均值$$\mathbb{E}[w_{i,j}]$$也设为0, 计算二阶矩:

$$
\begin{aligned} 
\mathbb{E}[y_j^2] =&\, \mathbb{E}\left[\left(\sum_i x_i w_{i,j}\right)^2\right]=\mathbb{E}\left[\left(\sum_{i_1} x_{i_1} w_{i_1,j}\right)\left(\sum_{i_2} x_{i_2} w_{i_2,j}\right)\right]\\ 
=&\,\mathbb{E}\left[\sum_{i_1, i_2} (x_{i_1}x_{i_2}) (w_{i_1,j} w_{i_2,j})\right] = \sum_{i_1, i_2} \mathbb{E}[x_{i_1}x_{i_2}] \mathbb{E}[w_{i_1,j} w_{i_2,j}] 
\end{aligned}
$$

其中$$w_{i_1,j},w_{i_2,j}$$是独立同分布的, 所以当$$i_1\neq i_2$$时, $$\mathbb{E}[w_{i_1,j}w_{i_2,j}]=\mathbb{E}[w_{i_1,j}]\mathbb{E}[w_{i_2,j}]=0$$, 因此只需要考虑$$i_1=i_2=i$$的情况, 假设输入的二阶矩为1:

$$
\mathbb{E}[y_j^2] = \sum_{i} \mathbb{E}[x_i^2] \mathbb{E}[w_{i,j}^2]= m\mathbb{E}[w_{i,j}^2]
$$

因此, 要使得$$\mathbb{E}[y_j^2]$$为1, 则需要$$\mathbb{E}[w_{i,j}^2]=1/m$$.

所以我们需要的初始化为均值为0, 方差为$$\frac{1}{m}$$的随机分布. 至此, **前向传播**的输入输出分布可以做到具有相同的方差.

## 稳定反向传播

同理, 我们也可以通过控制参数初始化的方差, 来稳定训练初始阶段反向传播中的梯度.

一样先不考虑激活函数, 对于输入节点数为$$m$$, 输出节点数为$$n$$的无激活函数的全连接层, 有:

$$
y_j = b_j + \sum_i x_i w_{i,j}
$$

我们要求loss对$$x_i$$的梯度, 第$$i$$个输入节点为所有的$$n$$个输出节点都做出了贡献, 对应的梯度为$$w_{i,j}$$. 因此有:

$$
\frac{\partial l}{\partial x_i} = \sum_j \frac{\partial l}{\partial y_j}w_{i,j}
$$

通过与上面相似的推理过程, 可以得到稳定这层梯度传播中的输入和输出方差, 需要$$\mathbb{E}[w_{i,j}^2]=1/n$$.

## Xavier初始化方法

Xavier初始化综合了前向和反向传播的需求, 最终选择了$$\mathbb{E}[w_{i,j}^2]=2/(m + n)$$作为参数初始化的方差.

因此Xavier正态分布初始化为:

$$
\mathcal{N}(0, \frac{2}{m + n})
$$

Xavier均匀分布初始化为:

$$U[-\frac{\sqrt 6}{\sqrt{m + n}}, \frac{\sqrt 6}{\sqrt{m + n}}]$$

# 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- [浅谈Transformer的初始化、参数化与标准化](https://kexue.fm/archives/8620)
- [从几何视角来理解模型参数的初始化策略](https://kexue.fm/archives/7180)
