# 原理

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

所以我们需要的初始化为均值为0, 方差为$$\frac{1}{m}$$的随机分布.

以上是无激活函数的情况, 考虑到激活函数, 需要具体情形具体分析. 比如比如激活函数是relu的话, 我们可以假设大致有一半的$$y_j$$被置零了, 于是二阶矩的估计结果是原来的一半, 即:

$$
\mathbb{E}[y_j^2] = \frac{m}{2}\mathbb{E}[w_{i,j}^2]
$$

从而使得二阶矩不变的初始化方差为$$\frac{2}{m}$$. 这就是专门针对relu网络的**He初始化**.

# 参考资料

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
- [浅谈Transformer的初始化、参数化与标准化](https://kexue.fm/archives/8620)
