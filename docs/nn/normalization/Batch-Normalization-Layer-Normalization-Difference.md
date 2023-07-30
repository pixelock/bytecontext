# Transformer 结构中为什么使用 Layer Normalization

这个问题可以进一步扩展至为什么NLP任务使用的模型结构, 都在使用LN, 而非BN?

当前在CV领域更多的在使用BN结构, 而在NLP领域使用的模型基本都用LN结构, 且BN结构在NLP中表现很差. 可以从这个角度分析它们之间的差异性, 而今探讨LN在NLP中的有效性的源头.

## 数据特点

首先在**数据层面**上. NLP使用的训练数据样本之间的差异性很大, 导致单个batch的统计值在整个训练过程中有很大的方差. 如下图所示, 统计的是训练过程中每个batch得到的统计量, 与统计量的滑动平均值之间的距离. 其中蓝色的是CV任务, 橙色是NLP任务. 可以看到NLP任务中单个batch的统计量在整个训练过程中波动非常大, 还有很多距离很远的离群点.

![](/resources/images/nn/layer-batch-diff-1.png)

而在推理阶段, 我们使用的统计值, 是滑动平均后得到的统计值, 而batch内统计量极大的波动性, 会导致**推理阶段使用的统计量与训练使用数据统计量之间存在巨大的差异**, 这种**训练阶段和推理阶段不一致**的问题, 会严重影响BN的性能, 进而降低了模型整体的性能.

以上观点来自与论文[PowerNorm: Rethinking Batch Normalization in Transformers](https://arxiv.org/abs/2003.07845). 论文还从反向传播的角度进行了分析. 将统计量$$\mu$$和$$\sigma$$对**损失函数对本层输入的梯度值**的贡献, 分别记为$$g_{\mu}$$和$$g_{\sigma^2}$$, 并且也绘制出了这两个梯度贡献项的**模长**在训练过程中的分布:

![](/resources/images/nn/layer-batch-diff-2.jpg)

可以看到统计量的梯度贡献项也是在大幅波动, 且存在很多离群点, 这也不利于模型的稳定训练, 可能造成梯度的消失或爆炸, 进一步导致过拟合甚至无法收敛.

## LN 更适合序列任务

对于 NLP 这类序列任务, 其输入的形式为`(batch_size, sequence_length, hidden_size)`的形式. BN 和 LN 在这种形式的数据上执行Normalization时, 差异体现在:

**BN** 是对 batch 维度去做归一化, 对不同样本的同一特征进行归一化操作. 而在 NLP 任务中, 对于`hidden_size`中的每一维, 需要对batch内所有样本, 以及每个样本所有token对应的位置做统计. 这样存在多个问题:

- 样本之间差异性很大, 这在上面一节中有论述, 得到的统计量在batch之间会有很大的方差
- 一个样本序列中, 不同位置的token包含的信息具有很大的差异性, 而这个差异性正是我们需要的, 使用归一化会对损失每个位置信息(token本身的信息, position信息), 影响模型表现
- `[PAD]`的存在导致在batch维度计算统计量时, 引入无效信息影响统计值, 而 pad 在不同样本内又是有很大差异的, 这进一步加剧了统计量的波动性

因此 BN 在 NLP 任务中有着天然的缺陷, 无论在任何包含 batch 维度的空间内进行统计, 都会造成我们本身所需要的差异信息被抹平, 有效信息量减少, 更多的噪音被引入, 模型的表现自然会下降.

**LN** 与 batch 无关, 是对一个样本的`hidden_size`维度进行归一化, 对单个样本的不同特征进行归一化操作. 在 NLP 任务中计算统计量是每个位置的token单独计算的:

```python
mean = K.mean(outputs, axis=-1, keepdims=True)
variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
```

token之间相互不干扰, 不会造成信息损失. 只是考虑token表征向量内部各个维度之间是有关联的, 通过归一化降低方法, 稳定训练过程.

## LN 与 Embedding

LN 是对每一个 token embedding 单独作用的, 将每个 embedding vector 中的参数分布稳定在均值为0, 方差为1. 而这就相当与 LN 将 embedding vector 约束为**以原点为中心, 半径为1的超球面上**, 形成了越远离这个超球面越稀疏的球体空间.

这种特性给 token embedding vector 带来了优秀的性质, 例如相似性分析, 相关性分析等, 为语义表达的分析带来便利.
