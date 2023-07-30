# 为什么需要Normalization

在机器学习领域, 有个很重要的假设, independent and identically distributed(简称**i.i.d.**), 即**独立同分布**. 独立同分布假设**训练数据**和**测试数据**数据是满足相同的分布的. 这是使用训练集训练得到的模型能够在测试集上取得好的效果的基础. 满足立同分布的数据能够简化模型训练, 提升模型学习能力, 基本已是共识.

但对于数据$$(X,Y)$$, $$X$$的分布一直在变化, 而不再满足独立同分布的约束, 这种问题就被称为**covariate shift**. 这样学习到的模型在面对不断变化的输入时, 得到的预测结果会很差.

这种问题不止产生在训练和测试阶段分布不一致时, 在训练阶段, 对于由很多层组成的深度学习模型, 每层的参数变化都会该层的输入分布一直变化, 每一层都面临着covariate shift问题. 由于这种问题发生在网络内部的隐层上, 因此被称为**Internal Covariate Shift**问题.

每个隐层的输入数据的分布一直变化, 经过层层叠加, 导致高层的输入分布变化非常剧烈, 迫使高层需要不断地适应底层的参数更新.

ICS导致**每个神经元**的输入不再是独立同分布的, 造成:

- 高层参数需要不断适应新的输入数据分布, 降低训练速度
- 低层输入的变化可能趋向于变大或者变小, 导致高层落入饱和区, 使得学习过早停止, 甚至没有收敛
- 每层的更新都会影响到其它层, 因此每层的参数更新策略需要尽可能的谨慎

简单来说, 会导致网络收敛速度慢, 甚至难以收敛. 而且得到的模型能力也受到影响.

Normalization就是要解决ICS的问题, 使得**每个隐层神经元节点的激活输入分布相对固定下来**.

# Normalization的本质思想

## 启发

Normalization的思路源自图像预处理中常用的操作, **白化**(Whiten). 所谓白化, 即将输入数据处理成均值为0, 方差为单位方差, 经过白化处理后的网络一般收敛更快.

以这种思想为引导, 能不能对每一层的输入都做白化, 从而消除偏移, 加速收敛.

## 表现

以一个神经元为例, 它输入向量为$$\mathbf{x}=\left(x_{1}, x_{2}, \cdots, x_{d}\right)$$, 经过该层的参数作用后, 即$$z=f(\mathbf{x})$$, 得到非线性变换激活函数的输入值, 这就是我们所说的输入值. 由于训练过程中参数的不断变化, 以及网络的加深, $$z$$的分布逐渐发生偏移或者变动. Normalization的思路就是在将输入传入到激活函数之前, 先对其做**平移和伸缩变换**, 将$$z$$的分布规范化为具有固定的均值和方差, 一般会使之趋近于标准正态分布.

将一层神经元的输入向量记为$$\mathbf{z}$$, 则Normalization的思路可以表示为, 其中的$$\mu$$, $$\sigma$$, $$\mathbf{g}$$, $$\mathbf{b}$$都是向量:

$$h=f\left(\mathbf{g} \cdot \frac{\mathbf{x}-\mu}{\sigma}+\mathbf{b}\right)$$

前面说过每个神经元的输入分布变化, 会导致训练速度缓慢. 这一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近, 导致后向传播时低层神经网络的梯度消失. 而Normalization将输入的分布拉回到接近正态分布, 使得激活函数的输入分布在梯度较大的部分, 避免了梯度消失, 从而加快了收敛. 这也是**Normalization能使训练加快的根本原因**.

## 缩放与还原

上面的式子中, 有好理解的平移参数$$\mu$$和缩放参数$$\sigma$$. 但还有着**再平移参数**$$\mathbf{b}$$和**再缩放参数**$$\mathbf{g}$$, 使得最终分布的均值和方差由这两个参数决定.

这是因为, 如果把整个神经网络的每一层, 层中的每一点的输入都统一成**全局一致**的确定范围, 无论前面的层如何学习, 到了下一层, 其输入都会是一致的, 即前面层的学习毫无意义. 甚至对于`sigmoid`这种在0附近是线性函数的激活函数, 网络的每一层都退化层线性计算, 则网络的深度也就没了意义, 整个网络退化层单一层的线性函数模型.

因此, 对于每一层, 层中的每个神经元, 都会由**再平移参数**和**再缩放参数**调整最终的偏移缩放量. 这个参数是在训练过程中学习到的, 使得每个神经元都能够学习到独特的知识, 表现出不同的形式, 保护了**非线性**能力, 同时又保证了其输入不会随着输入的变化而发生大的变化.

更详细的说明参考[详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)中的`Normalization 的通用框架与基本思想`章节部分.

# Normalization的数学性质

各种形式的Normalization都具有**伸缩不变性**(scale invariance), 这种优良的性质, 可以有效的消除梯度消失和梯度爆炸的风险, 提高反向传播的效率, 且在使用更高学习率时仍然可以稳定训练, 不损害模型表现, 进一步提升了训练的收敛速度.

伸缩不变性体现在**权重伸缩不变性**(weight scale invariance)和**数据伸缩不变性**(data scale invariance)两个方面.

## 权重伸缩不变性

权重伸缩不变性指的是当权重$$\mathbf{W}$$按照常量$$\lambda$$进行伸缩时, 即$$\mathbf{W}^{\prime}=\lambda\mathbf{W}$$, **得到的规范化后的值保持不变**:

$$
\begin{aligned}
\operatorname{Norm}\left(\mathbf{W}^{\prime} \mathbf{x}\right) &= \mathbf{g} \cdot \frac{\mathbf{W}^{\prime} \mathbf{x}-\mu^{\prime}}{\sigma^{\prime}}+\mathbf{b} \\
&= \mathbf{g} \cdot \frac{\lambda \mathbf{W} \mathbf{x}-\lambda \mu}{\lambda \sigma}+\mathbf{b} \\
&= \mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}-\mu}{\sigma}+\mathbf{b} \\
&= \operatorname{Norm}(\mathbf{W} \mathbf{x})
\end{aligned}
$$

而对于反向传播, 有:

$$
\begin{aligned}
\frac{\operatorname{Norm}\left(\mathbf{W}^{\prime} \mathbf{x}\right)}{\partial \mathbf{x}} &= \mathbf{g} \cdot \frac{\lambda \mathbf{W}}{\lambda \sigma} \\
&= \frac{\operatorname{Norm}(\mathbf{W} \mathbf{x})}{\partial \mathbf{x}}
\end{aligned}
$$

而这就是反向传播的连乘项中的权值部分, 因此过小或者过大的参数对反向传播没有负面影响, 从而避免了**因权重过大或过小导致的梯度消失和梯度爆炸的问题**, 梯度可以保持稳定, 从而**加速了网络的训练收敛过程**.

另外考虑对权重的梯度, 有:

$$
\begin{aligned}
\frac{\operatorname{Norm}\left(\mathbf{W}^{\prime} \mathbf{x}\right)}{\partial \mathbf{W}^{\prime}} &= \mathbf{g} \cdot \frac{\mathbf{x}}{\lambda \sigma} = \frac{1}{\lambda} (\mathbf{g} \cdot \frac{\mathbf{x}}{\sigma}) \\
&=\frac{1}{\lambda} \cdot \frac{\operatorname{Norm}(\mathbf{W} \mathbf{x})}{\partial \mathbf{W}}
\end{aligned}
$$

在对权重求梯度时, 权重数值越大, 即$$\lambda$$越大, 对应的梯度就会被缩小, 反之亦然. 这样参数的变化会比较稳定, 相当于**参数正则化**的效果, 避免了训练过程中参数的大幅震荡. 被限制大小的梯度, 也支持训练**使用更高的学习率**, 从而进一步加快收敛速度.

## 数据伸缩不变性

数据伸缩不变性指的是当数据$$\mathbf{x}$$按照常量$$\lambda$$进行缩放时, 即$$\mathbf{x}^{\prime} = \lambda \mathbf{x}$$, 得到的规范后的值保持不变. 此时仍然有:

$$
\begin{aligned}
\operatorname{Norm}\left(\mathbf{W} \mathbf{x}^{\prime} \right) &= \mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}^{\prime} - \mu^{\prime}}{\sigma^{\prime}}+\mathbf{b} \\
&= \mathbf{g} \cdot \frac{\lambda \mathbf{W} \mathbf{x}-\lambda \mu}{\lambda \sigma}+\mathbf{b} \\
&= \mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}-\mu}{\sigma}+\mathbf{b} \\
&= \operatorname{Norm}(\mathbf{W} \mathbf{x})
\end{aligned}
$$

考虑此时对权重的梯度:

$$
\begin{aligned}
\frac{\operatorname{Norm}\left(\mathbf{W} \mathbf{x}^{\prime} \right)}{\partial \mathbf{W}} &= \mathbf{g} \cdot \frac{\lambda\mathbf{x}}{\lambda \sigma} = \mathbf{g} \cdot \frac{\mathbf{x}}{\sigma} \\
&=\frac{\operatorname{Norm}(\mathbf{W} \mathbf{x})}{\partial \mathbf{W}}
\end{aligned}
$$

即每一层输入$$\mathbf{x}$$大小的变化, 不再会影响到该层权重参数的更新, 也会使得训练过程更具有鲁棒性, **简化了学习率的选择**.

本节内容参考: [详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)

# Normalization原理再探讨

## 平滑了损失函数平面

以上在说Normalization有效的原因在于能够缓解**Internal Covariate Shift**问题, 但[How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)这篇论文否定了Normalization在于解决ICS问题, 而是因为起到了**平滑损失平面**的作用, 平滑的损失平面加快了收敛速度.

![](/resources/images/nn/norm-1.jpg)

作者证明了在经过Batch Normalization处理后, 损失函数满足Lipschitz连续, 即**损失函数的梯度小于一个常量**, 因此网络的损失平面不会震荡的过于严重. 而且损失函数的梯度也满足Lipschitz连续, 也称为$$\beta$$平滑, 即斜率的斜率也不会超过一个常量.

作者认为当着两个常量的值均比较小的时候, 损失平面就可以看做是平滑的. BN收敛快的原因是由于**BN产生了更光滑的损失平面**.

## 将优化分解成了优化方向和长度两个任务

在[Exponential convergence rates for Batch Normalization: The power of length-direction decoupling in non-convex optimization](https://arxiv.org/abs/1805.10694)论文中, 提出了Batch Normalization实质上是将参数优化分解成了两个任务:

- 选择参数迭代的方向
- 确定长度

可以解耦层与层之间的依赖, 使得模型更易于优化.

# 常见的Normalization

- [Batch Normalization](/docs/nn/normalization/Batch-Normalization.md)
- [Layer Normalization](/docs/nn/normalization/Layer-Normalization.md)
- Instance Normalization
- Group Normalization

# Normalization作用总结

## 防止梯度消失和梯度爆炸

具体原理参考[梯度消失与梯度爆炸](/神经网络/梯度/梯度消失与梯度爆炸.md), 原因总结如下:

- 统计的滑动标准差$$\sigma_l$$在反向传播时调节不同参数$$w_l$$大小带来的影响, 避免了梯度消失和梯度爆炸
- 在进入激活函数前, 将输入分布稳定, 使得输入落在激活函数的敏感区域, 在反向传播时能够提供较大的梯度, 避免了梯度消失

## 提高训练时模型对于不同超参的鲁棒性

使用Normalization, 可以提升模型对学习率大小, 参数初始化方法这些超参数的鲁棒性, 使得这些模型训练过程对这些超参数不这么敏感. 训练时可以使用较大的学习率加快收敛速度, 且不会影响模型的最终效果.

因为Normalization具有**权重伸缩不变性**, 在反向传播时, 对于大小不同的权重, 能够做到链式跨层传播时梯度不受权重值大小影响, 且对于值较大的权重求导时, 能够限制其梯度, 权值越大, 限制越强. 因此使用高学习率也能够正常训练, 大大降低了学习率, 参数初始化方法的选择难度. 详情参考上面的**权重伸缩不变性**章节.

## 带来一定的正则效果

Normalization 的正则效果, 来自于对权重求梯度时, 得到的梯度大小与权重值大小成反比, 从而对权重值较大的参数的更新带来了稳定的效果, 相当于实现了参数正则化的效果. 参考上面的**权重伸缩不变性**一节.

另外, Batch Normalization 还从另一个角度, 带来了一定的正则化效果.

Batch Normalization在计算每一层输出的均值和方差时, 使用的时mini-batch的结果进行滑动平均得到, 这与总体样本的估计有一定的偏差. 将这样得到的统计数值作用与每个样本, 相当于对样本**引入了随机噪声**, 也可以理解为对样本做了**隐式的数据增强**, 因此具有正则化的效果, 提升了模型的泛化性.

## 加快收敛速度

Normalization在上述优点帮助下, 以及解决了ICS问题, 稳定了每一层的输出分布, 即使batch之间的分布差异较大, 训练时高层也能保持输入稳定, 不会再随着低层输出的变化而被迫调整, 近似于实现了层与层之间的训练学习解耦, 给训练带来了稳定性, 从而加快了整体的训练收敛速度. 另外Normalization将输出拉回至0附近, 避免了高层输出落入饱和区, 使得整体学习过早停止.
