# Attention 结构

## Multi-head Attention 的时间复杂度

Bert中的self-attention使用的是**Multi-Head Attention**.

首先明确符号:

- 序列的长度记为: $$n$$
- 隐向量长度 hidden size 记为: $$h$$
- heads 的数量记为: $$m$$

整个Multi-Head Attention的计算过程分为以下几步.

**1. 生成Q, K, V**

Self Attention的输入`input_embed`为一个$$(n, h)$$大小的矩阵, 在计算token之间的attention之前, 每个head都要经过Linear层分别转换成Q, K, V, 变换成Q, K, V的Linear参数不共享, 且head之间的参数不共享. 每个head中的Q, K, V对应的Linear层参数大小为 $$(h, \frac{h}{m})$$. 在`huggingface`的`transformers`包中, 是使用一个大的Linear层(对应的参数大小为$$(h,3h)$$)进行转换, 在通过`split`, `reshape`方法, 等价模拟实现拆分成每个head的Q, K, V, 再进行下一步的attention计算.

因此这一步为$$(n, h)$$大小的输入矩阵与$$(h, 3h)$$大小的Linear层参数矩阵相乘, 对应的时间复杂度为$$O(nh^2)$$.

这一步可以用下面这张图表示. 图中$$n = 4$$, $$h = 8$$, $$m = 2$$.

![](/resources/images/nlp/multi-attention-1.png)

对应的代码为:

```python
# Attention heads [n, b, h] --> [n, b, (3 * m * hn)], b for batch_size, hn = h / n
mixed_x_layer = self.query_key_value(hidden_states)
(query_layer, key_layer, value_layer) = mixed_x_layer.split(
    [
        self.num_attention_heads * self.hidden_size_per_attention_head,
        self.num_attention_heads * self.hidden_size_per_attention_head,
        self.num_attention_heads * self.hidden_size_per_attention_head,
    ],
    dim=-1,
)
query_layer = query_layer.view(
    query_layer.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
)  # (n, b, m, hn)
key_layer = key_layer.view(
    key_layer.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
)  # (n, b, m, hn)
value_layer = value_layer.view(
    value_layer.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
)  # (n, b, m, hn)
```

**2. 计算相关性矩阵**

普通的Attention中, Q与K两个矩阵相乘, 得到相关性矩阵. 两者的大小都是$$(n, h)$$, 矩阵相乘得到$$(n, n)$$大小的相关性矩阵, 对应的时间复杂度为$$O(n^2h)$$.

但Multi-Head Attention要先把Q, K, V分割为$$m$$个head, 每个head内部分别计算得到相关性矩阵$, 不同head之间互不干扰. 因此考虑上head, Q和K矩阵转变为$$(n, m, \frac{h}{m})$$大小的tensor, 点积得到$$(n, n, m)$$大小的相关性矩阵, 因此这一步的时间复杂度为$$O(mn^2\frac{h}{m})=O(n^2h)$$.

因此可以看到`Multi-head attention`与普通的`Attention`的在计算相关性矩阵时, 对应的时间复杂度是一样的.

**3. Softmax计算**

对得到的按head划分的, 大小为$$(n, n, m)$$的相关性矩阵计算softmax, 得到大小仍为$$(n, n, m)$$, 这一步的时间复杂度为$$O(n^2m)$$.

**4. 计算加权和**

将$$(n, n, m)$$大小的权值矩阵与$$(n, m, \frac{h}{m})$$大小的O点乘, 得到$$O(n, m, \frac{h}{m})$$大小的输入. 这一步沿着$$n$$所在的维度进行点乘, 所用的时间复杂度为$$O(mn^2\frac{h}{m})=O(n^2h)$$.

再经过reshape变成最后的输出, 大小仍然为$$(n, h)$$.

上面2~4步可以用下面的图表示.

![](/resources/images/nlp/multi-attention-2.png)

对应的公式为:

$$
\text{head}_{i}(Q_i, K_i, V_i) = \text{softmax}(\frac{Q_{i}K_{i}^{T}}{\sqrt{h}})V_{i}
$$

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(head_{1}, \dots, head_{m})W_{O}
$$

**总结**

- 生成Q, K, V的时间复杂度为$$O(nh^2)$$
- 计算相似度矩阵的时间复杂度为$$O(n^2h)$$
- Softamax计算的时间复杂度为$$O(n^2m)$$
- 计算加权和的时间复杂度为$$O(n^2h)$$

整体的时间复杂度为$$O(n^2h+nh^2)$$, 即序列长度与hidden size更大的一个占主导.

---

## Multi-head Attention中多头的作用

**原论文中的说法**

Attention is all you need论文中说, 将模型分为多个头, 每个头都是一个子空间, 相当于每个头去关注不同角度的信息, 然后进行融合, 类似于CNN中的每个卷积核捕获不同的纹理信息, 或者理解为是一种ensemble思想.

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

头和头之间也没有相互之间的制约, 每个空间都是完全独立的, 理论上可以做到子空间关注点的差异化.

在[A Multiscale Visualization of Attention in the Transformer Model](https://arxiv.org/abs/1906.05714)这篇论文中, 可视化了层中各个头的Attention情况, 如下图:

![](/resources/images/nlp/multi-attention-3.png)

可以看到:

- 同一层中多数头的关注模型近似, 如第3层中的1, 2, 3, 4头
- 每一层总有几个头与其他头区别较大, 如第2层中的0头

是可以印证不同子空间关注的角度不同, 只是有些头之间的差异程度, 没有我们想象中的这么大.

**空间差异探讨**

差异是如何产生的. 如果所有头的参数初始化为一模一样的, 由于输入相同, 则输出也是相同的, 因此在梯度更新的时候, 更新的情况也是一样的. 这样无论更新的步数多少, 单层中所有头的参数会一直保持相同, 头和头之间没有差别, 相当于就是一个头.

**可以说头和头之间的差别, 起始来源是初始化的差异性**.

**使用多头可能的原因: 信息冗余**

以BERT base为例, 最大长度512, 此时$$Q, K \in \mathbb{R}^{512 \times 768}$$, 如果直接将两个向量点积, 得到一个`(512, 512)`大小的Attention矩阵, 去计算其中一个位置的数值时, 需要两个长度为768的向量. 实际上可能不需要向量的全部参数都参与计算, 就能得到高质量的结果了. 这里会产生计算的冗余.

而且依据上面的图, **每个token通常只是注意到有限的若干个token**, 说明得到的Attention矩阵是很稀疏的. 这可能是因为使用了**Softmax函数**对点积结果进行转换的原因, 会将元素间的差距拉大, 比较小的点积值就被转化为接近0的结果. 而这种稀疏性意味着`(512, 512)`大小的Attention矩阵本身也是冗余的, 里面没有包含这么多信息, 是可以由**两个低维矩阵的乘积**得到, 相当于可以**低秩分解**.

从SVD的角度看, 这种稀疏的Attention矩阵的奇异值, 有很多是接近于0的. BERT base中有12个头, 每个头的head size为64, 相当于假设了每个`(512, 512)`大小的Attention矩阵, 其明显不等于0的奇异值不超过64个, 因此可以用$$Q, K \in \mathbb{R}^{512 \times 64}$$通过点乘来拟合Attention矩阵.

从这个角度来看, 每个头都可以拟合出单个注意力, 多个头都拟合相当于一种model ensemble, 可以提升整体的效果.

**总结**

多头可以带来好处:

- 多个子空间从各个角度获取信息, 且每个子空间获取的信息都不差(信息冗余部分), 进行融合, 提升效果
- 计算点积时, 相比于`hidden_size * hidden_size`的计算量, `head_size * head_size`会小很多

这种**split, transform, merge**的模型设计思想,都能够带来多路信息融合的效果.

**参考资料**

- [为什么Transformer 需要进行 Multi-head Attention？ - 香侬科技的回答](https://www.zhihu.com/question/341222779/answer/814111138)
- [BERT中，multi-head 768*64*12与直接使用768*768矩阵统一计算，有什么区别？ - 苏剑林的回答](https://www.zhihu.com/question/446385446/answer/1752279087)

---

## qkv为什么要乘上不同的参数矩阵

### 避免过分关注自身

Self-Attention的核心是用文本中的其它词来增强目标词的语义表示, 从而更好的利用上下文的信息.

在self-attention中, sequence中的每个token都会和sequence中的每个token做点积去计算相似度, 也包括这个词本身. 而对于self-attention, 它的q, k, v来自同一向量, 如果不乘各自的参数, 即使用各自的`Dense`作用, 则q, k, v是完全一样的.

在这种情况下, 如果每个token对应的向量, 如果模长相近, $$q_i$$与$$k_i$$两个向量的点积, 即同一个token之间的点积, 得到的值是最大的, 因为两个向量的方向一致, 没有夹角.

然后对得到的点积矩阵求softmax, 相同位置的最终得到的值会是很大的(可以看到attention矩阵对角线的值很大), 然后使用这个值对所有token的向量进行加权平均, 那么这个token本身占的比重将会是最大的, 而且往往远超其他位置的token. 这就使得其他token的比重很小, 无法有效利用上下文信息来增强当前词的语义表示.

而乘上各自不同的矩阵之后, 同一个token对应q, k, v将会不一样, 在很大程度上能够**缓解**上面的影响(不是完全解决, 一般情况下在经过attention转换后, 还是同一个token成分最大).

### 更多的信息容量, 带来更强的表征能力

如果只看Query和Key, 这两个共享参数矩阵的情况下, 就会导致同一个token对应的query和key向量是完全相同的, 进而得到的attention score矩阵是一个**对称矩阵**, 即如果token A最关注token B, 那么token B必然也最关注token A, 这样的attention score矩阵中包含的信息相比之下缩减了很多, 导致模型的表征性能肯定会下降.

---

## qk相乘得到attention矩阵后, 为什么要进行scale

对于Attention矩阵中的每个值, 都是由某个位置的$$\mathbf{q}$$向量和对应位置的$$\mathbf{k}$$向量点积得到. 假设$$\mathbf{q}$$向量和$$\mathbf{k}$$向量的各分量都是相互独立的随机变量, 且均值是0, 方差是1, 那么点积$$\mathbf{q} \cdot \mathbf{k}$$的均值为0, 方差为$$d_k$$, 其中$$d_k$$是向量的长度.

推导过程为: 对于$$\forall i=1, \cdots, d_{k}$$, $$q_i$$和$$k_i$$是独立的随机变量, 记这两个随机变量为: $$X=q_i$$, $$Y=k_i$$. 则有$$E(X)=E(Y)=0$$, $$D(X)=D(Y)=1$$.

则:

$$
E(XY)=E(X)E(Y) = 0
$$

$$
\begin{aligned}
D(X Y) &=E\left(X^{2} \cdot Y^{2}\right)-[E(X Y)]^{2} \\
&=E\left(X^{2}\right) E\left(Y^{2}\right)-[E(X) E(Y)]^{2} \\
&=E\left(X^{2}-0^{2}\right) E\left(Y^{2}-0^{2}\right)-[E(X) E(Y)]^{2} \\
&=E\left(X^{2}-[E(X)]^{2}\right) E\left(Y^{2}-[E(Y)]^{2}\right)-[E(X) E(Y)]^{2} \\
&=D(X) D(Y)-[E(X) E(Y)]^{2} \\
&=1 \times 1-(0 \times 0)^{2} \\
&=1
\end{aligned}
$$

那么遍历$$\forall i=1, \cdots, d_{k}$$, $$q_i$$和$$k_i$$的乘积都符合均值为0, 方差为1, 而且不同$$i$$之间是相互独立的, 把$$q_i \cdot k_i$$这个随机变量记为$$Z_i$$. 则有:

$$E\left(\sum_{i} Z_{i}\right)=\sum_{i} E\left(Z_{i}\right)$$

$$D\left(\sum_{i} Z_{i}\right)=\sum_{i} D\left(Z_{i}\right)$$

因此$$\mathbf{q} \cdot \mathbf{k}$$点积的分布符合: $$E(\mathbf{q} \cdot \mathbf{k})=0$$, $$D(\mathbf{q} \cdot \mathbf{k})=d_k$$. 即维度越高, 点积结果分布的方差越大.

Attention矩阵的基础是对所有的$$\mathbf{q}$$和$$\mathbf{k}$$向量两两点积计算. 而方差越大, 点积得到的结果值较大的概率也会越大, 即矩阵中每个位置取值越大, 则Attention矩阵中行向量的模长也会越大.

**向量模长与Softmax梯度**

对于一个输入向量$$\mathbf{x} \in \mathbb{R}^{d}$$, softmax函数将其归一化到一个分布$$\hat{\mathbf{y}} \in \mathbb{R}^{d}$$下. softmax计算过程中, 使用**自然底数$$e$$将输入原始的差距先拉大**, 然后再归一化为一个分布.

假设输入$$\mathbf{x}$$向量中最大的元素对应的下标为$$k$$, 经过softmax转换后对应的$$\hat{y}_k$$对应的概率也是最大的. 但**如果输入向量的模长增加**, 即输入向量中各个元素等比例扩大, 在每个输入元素都很大的情况下, **这时的$$\hat{y}_k$$会非常接近1**.

这时由于softmax的特性决定的, 对于向量$$\mathbf{x}=[a, a, 2 a]^{\top}$$, softmax得到的$$\hat{y}_3$$值随自变量$$a$$的变化如下图所示, 横坐标是$$a$$的值, 纵坐标是$$\hat{y}_3$$的值:

![](/resources/images/nlp/multi-attention-4.jpg)

可以看到, 输入向量的模长对softmax得到的分布影响非常大. 在模长较长时, softmax归一化的结果, 几乎将所有概率都分配给输入最大值的位置, 其余位置的结果基本为0.

这会导致一个问题: **反向传播时, softmax的梯度几乎为0, 即发生了梯度消失, 造成参数更新困难**. softmax函数的导数, 以及输出接近one-hot向量时导数的结果参考[transformer中的attention为什么scaled? - TniL的回答](https://www.zhihu.com/question/339723385/answer/782509914).

**解决方法**

因此在Transformer结构中, 在向量两两点积获得点积结果矩阵后, 需要**先将点积结果除以$$\sqrt{d_k}$$, 然后再对行向量进行softmax**, 以避免梯度消失. 因为对点积结果除以$$\sqrt{d_k}$$, **会使得结果的分布方差回归到1**:

$$
D\left(\frac{q \cdot k}{\sqrt{d}_{k}}\right)=\frac{d_{k}}{\left(\sqrt{d}_{k}\right)^{2}}=1
$$

这里的$$d_k$$指的是进行attention向量的长度, 对于multi-head attention, 每个head中单独进行attention, 所以这里的$$d_k$$为`hidden size / num heads`. 消除了模长过大的问题, 进而消除了梯度消失的情况.

**一句话总结**

如果计算softmax的元素方差太大, 将会导致softmax结果稀疏, 进而导致梯度稀疏. 通过scaled除以系数$$\sqrt{d_k}$$, 将softmax函数的输入变量的方差重新拉回到1, 解决梯度消失的情况.

**参考资料**

- [transformer中的attention为什么scaled? - TniL的回答](https://www.zhihu.com/question/339723385/answer/782509914)
- [transformer中的attention为什么scaled? - 小莲子的回答](https://www.zhihu.com/question/339723385/answer/811341890)

---

# Normalization

## Transformer为什么使用LN而不是BN

- 数据层面上, NLP任务使用的训练数据样本之间的差异性很大, 导致batch内的统计值在整个训练过程中有很大的方差, 影响训练的稳定性; 推理阶段使用的统计量与训练使用数据统计量之间存在巨大的差异, 这种训练阶段和推理阶段不一致的问题, 会严重影响BN的性能, 进而降低了模型整体的性能
- BN是在batch内对每一个特征值单独地做归一化, 需要使用到batch内所有样本序列以及每个样本序列的所有token做归一化, **这种归一化会对损失每个token信息**, 而这些差异信息正是任务需要的
- LN是对每个样本序列, 以及序列中的每个token, 独立地, 相互之间不影响地做normalization, 每个token内的表征不会被其他token, 其他样本所影响, 不会产生信息的损失
- LN相比BN不需要存储均值和方差的动量参数, 节省显存

以上是总结, 详细原因查看: [Batch Normalization 与 Layer Normalization 的差异](/docs/nn/normalization/Batch-Normalization-Layer-Normalization-Difference.md).

## LN在Transformer中起到什么作用

- 反向传播时缓解了梯度消失和梯度爆炸, 这是通过以下机制产生的作用
  - 统计的滑动标准差$$\sigma_l$$在反向传播时调节不同参数$$w_l$$大小带来的影响
  - 在进入激活函数前, 将输入分布稳定, 使得输入落在激活函数的敏感区域, 在反向传播时能够提供较大的梯度, 避免了梯度消失
- 有一定的正则效果, 训练过程更平稳. 对权重求梯度时, 得到的梯度大小与权重值大小成反比, 从而对权重值较大的参数的更新带来了稳定的效果
- 可以使用更大的学习率, 加速了训练. 解决了ICS(**Internal Covariate Shift**)问题, 实现了层与层之间的训练学习解耦, 高层输入也能保持稳定, 不会再随着低层输出的变化而被迫调整
  - 层与层之间的解耦带来了一个额外的好处: 稳定了前向传播的数值, 每个模块(层)的输出行为具有一致性(方差一致). 我们可以在最后一层接一个Dense来分类, 也可以取第6层接一个Dense来分类

---

# Residual 残差结构

![](/resources/images/nlp/multi-attention-7.png)

Transformer Block中的残差在self-attention和feed-forward子结构中都有使用.

## Transform中残差结构的作用

一句话总结:

**因为残差结构是可以同时稳定前向传播和反向传播, 防止梯度消失和梯度爆炸; 并且可以缩放参数梯度以解决参数更新量过大的问题, 避免进入不大好的局部最优点, 能帮助我们训练更深层的模型.**

下面从稳定前向传播, 反向传播, 以及防止增量爆炸(更新量过大)三个角度分析.

**稳定前向传播**

不失一般性, 假设输入输出维度相等, 我们考虑:

$$
\boldsymbol{y} = \boldsymbol{x} + \varepsilon \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\theta})
$$

很显然, 只要$$\varepsilon$$足够小, 那么前向传播必然是稳定的.

**稳定反向传播**

$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = \boldsymbol{I} + \varepsilon\frac{\partial \boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{x}}
$$

也可以看出, $$\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}$$作为反向传播链的一项, 只要$$\varepsilon$$足够小, 反向传播也是稳定的.

**防止增量爆炸**

增量指的是在一个step中, 模型参数的更新量. 如果解决了梯度消失/爆炸问题, 即认为每个参数的梯度是$$\mathscr{O}(1)$$常数量级, 则模型每一步的更新量可以由以下得到.

假设损失函数为$$\mathcal{L}(\boldsymbol{\theta})$$, $$\boldsymbol{\theta}$$是模型参数参数, 当参数由$$\boldsymbol{\theta}$$变为$$\boldsymbol{\theta}+\Delta\boldsymbol{\theta}$$时, 有:

$$
\Delta\mathcal{L} = \mathcal{L}(\boldsymbol{\theta}+\Delta\boldsymbol{\theta}) - \mathcal{L}(\boldsymbol{\theta}) \approx \langle\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}),\Delta\boldsymbol{\theta}\rangle
$$

$$\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$$是对参数的梯度, $$\Delta\boldsymbol{\theta}$$是参数的变化量. 以SGD优化器为例, 对应的参数的变化量为$$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$$, 所以有$$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2$$. 而我们假设反向传播是稳定的, 即梯度是常数$$\mathscr{O}(1)$$量级, 所以有$$\Delta\mathcal{L}=\mathscr{O}(\eta NK)$$. 因此**模型每一步的更新量是正比于模型深度$$N$$的**. 如果模型越深, 那么更新量就越大, 这意味着初始阶段模型越容易进入不大好的局部最优点, 然后训练停滞甚至崩溃, 这就是“增量爆炸”问题.

残差结构可以解决这个问题. 参数的梯度为:

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{\theta}} = \varepsilon\frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}\frac{\partial \boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{\theta}}
$$

说明我们可以通过控制$$\varepsilon$$来实现层数相关的梯度缩放, 以应对参数的更新量与层数相关的问题. 比如要想梯度缩放到$$1/\sqrt{N}$$, 那么让$$\varepsilon=1/\sqrt{N}$$即可.

**参考**

- [为什么需要残差？一个来自DeepNet的视角](https://kexue.fm/archives/8994)

---

# Optimizer 优化器

## Adam优化器的作用

对于Adam来说, 由于包含了动量和二阶矩校正, 所以近似来看它的更新量大致上为:

$$
\Delta \theta = -\eta\frac{\mathbb{E}_t[g_t]}{\sqrt{\mathbb{E}_t[g_t^2]}}
$$

分子分母是都是同量纲的, 因此分式结果其实就是$$\mathscr{O}(1)$$的量级, 即参数的更新量就是$$\mathscr{O}(1)$$量级的. 也就是说, 理论上只要梯度的绝对值大于随机误差, 那么对应的参数都会有常数量级的更新量.

与SGD相比, SGD的更新量是正比于梯度的, 只要梯度小, 更新量也会很小, 如果梯度过小, 那么参数几乎会没被更新.

---

# Initialization 初始化

## BERT的使用了怎样的初始化方法, 有什么作用

使用**均值为0, 标准差为0.02的, 上下限宽度为2倍标准差的截断正态分布**进行初始化, 是小于Xavier初始化的标准. 这会使得输出整体偏小, 从而使得残差中的直路权重就越接近于1, 那么模型初始阶段就越接近一个恒等函数, 就越不容易梯度消失. 稳定初始训练阶段.

---

# Gradient 梯度

## Transformer中有哪些要素可以防止梯度消失

1. 残差结构. 稳定了反向传播路径.

2. LN结构, 作用原因参考[Normalization](/docs/nn/normalization/norm-summary.md):
  i. 统计的滑动标准差$$\sigma_l$$在反向传播时调节不同参数$$w_l$$大小带来的影响, 避免了梯度消失和梯度爆炸
  ii. 在进入激活函数前, 将输入分布稳定, 使得输入落在激活函数的敏感区域, 在反向传播时能够提供较大的梯度, 避免了梯度消失

3. Adam优化器. 通过一阶动量和二阶动量的调整, 保证参数的更新量是$$\mathscr{O}(1)$$常数量级的. 即使梯度过小的情况下, 只要梯度的绝对值大于随机误差, 参数也会有常数量的更新. 即**即使梯度消失了也能有常量级的更新**

4. 初始化. 使用**均值为0, 标准差为0.02的, 上下限宽度为2倍标准差的截断正态分布**进行初始化, 是小于Xavier初始化的标准. 这会使得输出整体偏小, 从而使得残差中的直路权重就越接近于1, 那么模型初始阶段就越接近一个恒等函数, 就越不容易梯度消失

---

# Training Process 训练过程

## Warmup为什么是Transformer训练的关键步骤

Warmup是Transformer训练的关键步骤, 没有它可能不收敛, 或者收敛到比较糟糕的位置. 原因为:

Adam解决的是梯度消失带来的参数更新量过小问题, 即不管梯度消失与否, 更新量都不会过小. 但对于Post Norm结构的模型来说, 梯度消失依然存在, 只不过它的意义变了. 根据泰勒展开式:

$$
f(x+\Delta x) \approx f(x) + \langle\nabla_x f(x), \Delta x\rangle
$$

增量$$f(x+\Delta x) - f(x)$$是正比于梯度的. 因此, 梯度衡量了输出对输入的依赖程度, 如果梯度消失, 那么意味着模型的输出对输入的依赖变弱了.

Warmup是在训练开始阶段, 将学习率从0缓增到指定大小, 而不是一开始从指定大小训练. 如果不进行Wamrup, 模型一开始就快速地学习, 由于梯度消失, 导致模型越靠后的层越敏感, 也就是越靠后的层学习得越快, 然后后面的层是以前面的层的输出为输入的, 前面的层根本就没学好, 所以后面的层虽然学得快, 但却是建立在糟糕的输入基础上的.

很快地, 后面的层以糟糕的输入为基础到达了一个糟糕的局部最优点. 此时它的学习开始放缓(因为已经到达了它认为的最优点附近), 同时反向传播给前面层的梯度信号进一步变弱, 这就导致了前面的层的梯度变得不准. 但我们说过, Adam的更新量是常数量级的, 梯度不准, 但更新量依然是数量级, 意味着可能就是一个常数量级的随机噪声了, 于是学习方向开始不合理, 前面的输出开始崩盘, 导致后面的层也一并崩盘.

所以, 如果Post Norm结构的模型不进行Wamrup, 我们能观察到的现象往往是: loss快速收敛到一个常数附近, 然后再训练一段时间, loss开始发散, 直至NAN. 如果进行Wamrup, 那么留给模型足够多的时间进行*预热*, 在这个过程中, 主要是抑制了后面的层的学习速度, 并且给了前面的层更多的优化时间, 以促进每个层的同步优化.

这里的讨论前提是梯度消失, 如果是Pre Norm之类的结果, 没有明显的梯度消失现象, 那么不加Warmup往往也可以成功训练.

**参考**:

- [Warmup是怎样起作用的？](https://kexue.fm/archives/8747/comment-page-1#comments)

---

# BERT

## 为什么BERT在输入前会加一个[CLS]标志

在Google放出的[BERT源码](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L423)中, 对`[CLS]`做出的解释为:

> The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.

用来作为对整句作为表征, 用来做下游的分类任务.

为什么这个token可以作为证据的表征呢? `[CLS]`相对与其他tokens, 本身不包含任何语义信息, 在经过多层attention layer加权融合各tokens的信息时, 相对于其他token, 能更公平地融合其他token的信息, 从而更好的表示整句话的语义.

### 辅助定位作用

在[相对位置编码Transformer的一个理论缺陷与对策](https://kexue.fm/archives/9105)这篇blog中, 作者提出了使用相对位置编码的一个缺陷.

相对位置编码是**作用在softmax之前的attention矩阵上的**. 如果说一个模型是理想的, 我们希望它能够是*万能拟合器*. 但是在探针任务中, 相对位置编码却无法解决. **探针任务**指的是, 对于一个有识别位置能力的模型，应该有能力准确实现如下映射:

$$
\begin{array}{lc}
\text{Input:} & [0, 0, \cdots, 0, 0] \\
& \downarrow\\
\text{Output:} & [1, 2, \cdots, n-1, n]
\end{array}
$$

也就是说, 输入相同的n个token, 能够输出每个token的位置编号. 如果有能力做到这一点, 说明识别位置是模型自身具备的能力, 跟外部输入无关. 不难发现, 绝对位置由于是直接施加在输入上的, 所以即使输入的token一样, 对应的embedding也会融合了绝对位置信息, 很容易能够完成探针测试.

但对于相对位置, 输入中没有位置信息, 所有token的输入$$x_i$$是完全相同的. 对Transformer模型来说, Token之间的交互的唯一来源是Self Attention的$$\boldsymbol{A}\boldsymbol{V}$$这一步, 或者说是$$\boldsymbol{o}_i = \sum\limits_j a_{i,j}\boldsymbol{v}_j$$. 相同的输入意味着每个$$v_j$$都是相同的, 所以有:

$$
\boldsymbol{o}_i = \sum_j a_{i,j}\boldsymbol{v}_j = \sum_j a_{i,j}\boldsymbol{v} = \left(\sum_j a_{i,j}\right)\boldsymbol{v} = \boldsymbol{v}
$$

这意味这每个token的输出$$\boldsymbol{o}_i$$也是相同的, 且不管经过多少层都是这个结果. 换句话说, 模型的每个位置自始至终都输出相同的结果, 所以模型根本不可能输出各不相同的$$[1, 2, \cdots, n-1, n]$$.

这个问题的具体原因, 是因为每个$$\boldsymbol{v}_j$$是相同的, 所以softmax的性质$$\sum\limits_j a_{i,j}=1$$导致探针任务无法完成. 但我们的输入是全部为0的序列$$[0, 0, \cdots, 0, 0]$$, 如果按BERT的形式`[CLS] SENT [SEP]`输入呢?

这会使得$$\boldsymbol{v}_j$$不再全部相同, 因为对应的输入为$$[\text{[CLS]}, 0, 0, \cdots, 0, 0, \text{[SEP]}]$$. 作者实验后, 发现是可以完成探针实验的.

这种现象可以总结为:

**使用相对位置编码的情况下, 特殊Token `[CLS]`, `[SEP]`还有辅助定位的作用**.

### 虚拟Token

在[Attention机制竟有bug，Softmax是罪魁祸首，影响所有Transformer](https://mp.weixin.qq.com/s/cSwWapqFhxu9zafzPUeVEw)一文中, 指出了attention结构使用softmax会带来一些问题.

Attention的计算公式为:

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^{T}}{\sqrt{d}})V
$$

多头注意力每层并行执行多次上述过程. 使用 softmax 的问题在于, 对于每个token的输出, 它强制将其他token的信息融合进去, 即使没有信息可以添加到输出向量中. 这还是因为softmax的性质, 其输出之和一定为1导致的.

![](/resources/images/nlp/multi-attention-5.png)

如果当前token在模型中试图避免融合其他位置的token信息时, 其query与其他位置的key点积值会是一个非常小的值, 趋近于负无穷. 当输入到softmax的向量所有位置的值都趋近于负无穷时, 经过softmax后, 每个位置的输出为:

![](/resources/images/nlp/multi-attention-6.png)

因此, 当前位置还是会融合其他位置的tokens的信息.

为什么会有token在输出时避免融合其他位置的信息呢? 例如在LLM中, 研究员发现97%的异常值(数值过大发生上溢, 导致输出为nan, 使得模型发生数值错误, 训练终止)发生在非语义token(逗号, 空格等). 这些异常值就是融合了各个token的信息, 加权sum导致越界. 如果在这些位置可以输出趋近于0的权重, 相当于舍弃了对其他向量信息的融合, 就不会产生这种异常了.

文章中提出对softmax函数进行改进来解决这个问题. 但实际上, BERT的`[CLS]`, `[SEP]`这些特殊的token, 可以等价于一个**虚拟token**, 虚拟token对应的key和value为全零的向量. 模型可以通过训练, 给这些虚拟token分配一个较大值的点击, 在softmax之后, 就会重点关注这个虚拟token的输入, 对应的时全为0的value向量, 从而也能解决上面的问题.

By the way, 在torch的`MultiheadAttention`类中, 初始化附带了`add_zero_attn`参数, 默认为False. 如果设置为True, 将会在query和key序列中拼接一个值全部为0的向量, 起到虚拟token的作用, 可以作为当前token提供一个不去融合输入token信息的*逃出口(escape hatch)*, 避免出现异常值的问题.

这种作用可以总结为:

**`[CLS]`, `[SEP]` 这类特殊Token, 可以作为虚拟Token, 为某个token选择不去融合任何token信息的行为提供一个逃出口(escape hatch), 避免运算过程中因融合产生异常值. 特别是在在量化场景中, 由于低精度带来的数值上限低, 加权容易导致上溢, 进而导致模型崩溃, 可以有效缓解上面的问题.**

## 为什么NSP, MLM要多加一层Dense

越靠近输出层的, 越是依赖任务的(Task-Specified), 多接一个Dense层, 希望这个Dense层是MLM-Specified的, 然后下游任务微调的时候就不是MLM-Specified的, 所以把它去掉.

导致这个问题的原因是. 在[BERT中的初始化](/docs/nn/initialization/BERT中的初始化.md)中提到的, 跟BERT用了0.02的标准差做初始化, 这个初始化是偏小的. 最后一层的输出在进入到softmax激活之前, 还要乘上Embedding预测概率分布(这里的Embedding用的是tied-embedding, 即一开始输入的Embedding).

在乘上数值偏小的embedding之后, 得到的logits此时在每个可能的token位置都是比较小的, 所以经过softmax之后与label的差距很大, 于是模型就想着要把softmax后的数值放大. 现在模型有两个选择:

- 放大Embedding层的数值. 但是Embedding层的更新是稀疏的, 优先的batch内能见到的token有限, 放大上一个batch出现token的embedding在下一个batch内的效果有限
- 放大输入. BERT编码器最后一层是LN, LN最后有个初始化为1的gamma参数, 直接将那个参数放大就好

模型优化使用的是梯度下降, 它会选择最快的路径, 显然是第二个选择更快, 所以模型会优先走第二条路, 这就导致了一个现象, 最后一个LN层的gamma值会偏大. 如果预测MLM概率分布之前不加一个Dense+LN, 那么BERT编码器的最后一层的LN的gamma值会偏大. 而多加了一个Dense+LN后, 偏大的gamma就转移到了新增的LN上去了, **而编码器的每一层则保持了一致性**.

中文BERT-base中各个LN中的gamma参数的均值分别为:

```
Embedding-Norm/gamma:0 mean: 0.88696283
Transformer-0-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.82883006
Transformer-0-FeedForward-Norm/gamma:0 mean: 0.9062193
Transformer-1-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.8794548
Transformer-1-FeedForward-Norm/gamma:0 mean: 0.95188266
Transformer-2-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.88048005
Transformer-2-FeedForward-Norm/gamma:0 mean: 0.91880983
Transformer-3-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.8744504
Transformer-3-FeedForward-Norm/gamma:0 mean: 0.94945145
Transformer-4-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.8591768
Transformer-4-FeedForward-Norm/gamma:0 mean: 0.9897261
Transformer-5-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.8541221
Transformer-5-FeedForward-Norm/gamma:0 mean: 0.9948392
Transformer-6-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.86367863
Transformer-6-FeedForward-Norm/gamma:0 mean: 0.96400976
Transformer-7-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.83335274
Transformer-7-FeedForward-Norm/gamma:0 mean: 0.9587536
Transformer-8-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.80936116
Transformer-8-FeedForward-Norm/gamma:0 mean: 0.9643423
Transformer-9-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.81411123
Transformer-9-FeedForward-Norm/gamma:0 mean: 0.96500415
Transformer-10-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.8703678
Transformer-10-FeedForward-Norm/gamma:0 mean: 0.9663982
Transformer-11-MultiHeadSelfAttention-Norm/gamma:0 mean: 0.868598
Transformer-11-FeedForward-Norm/gamma:0 mean: 0.81089574
MLM-Norm/gamma:0 mean: 2.5846736
```

数值印证了上面的推理.

## BERT-base的参数量

BERT参数量来自4部分:

- Embedding: 由3种类型的embedding组成, 分别为: token embedding, position embedding, segment embedding
- Multi-head Attention: QKV分别有对应的权重矩阵, 最后输出O也有对应的权重矩阵
- Feed-forward: 两个全连接层
- Layer Normalization: 在三个地方用到了LN, 分别在Embedding层之后, Multi-head Attention以及Feed-forward之后. LN之中有两个参数, 分别为gamma和beta

需要注意的是输出层, 最后一层的输出还要与Token Embedding层相乘才是每个token的logit. 这里的Token Embedding与最前面的Embedding层中的Token Embedding是共享的(tied embedding技术), 因此不会引入额外的参数.

### Embedding

**Token Embedding**

大小为(词表大小, 隐向量长度), BERT chinese对应的词表大小为21128. 因此参数量为`21128 * 768`

**Position Embedding**

输入长度最长为512. 参数量为`512 * 768`

**Segment Embedding**

使用`0, 1`来区分前后句. 参数量为`2 * 768`

Embedding层总参数量为: 16,621,056

### Multi-head Attention

每个Head互不影响地进行计算, 首先通过的Linear层用来做QKV参数转换到各自的低维空间中, 对应的空间维度大小为`768 / 12 = 64`, 对应的参数量为: `768 * (768 / 12) * 3 * 12`, 分别对应的是`hidden size * head size(hidden size / head num) * 3 * head num`.

这是Linear中的`weight`参数. 另外还有`bias`参数, 对应的数量为`(768 / 12) * 3 * 12`.

然后是输出部分. 由每个head的输出拼接之后, 再经过一层Linear进行转换, Linear层的输入输出维度相同. weight参数量大小为`768 * 768`, bias参数量大小为`768`.

每一层Multi-head Attention对应的参数量为: 2,362,368

### Feed-forward

Feed-forward由两个Linear层组成, 两层中间的维度intermediate size为3072, 因此weight参数量为`768 * 3072 + 3072 * 768`, bias的参数量为`3072 + 768`.

每一层Feed-forward对应的参数量为: 4,722,432

### Layer Normalization

在三个地方用到了LN, 分别在Embedding层之后, Multi-head Attention以及Feed-forward之后. 这三个地方处理的向量的长度都是768, 因此gamma和beta合在一起的参数量为:

- Embedding层之后: `768 * 2`
- Multi-head Attention之后: `768 * 2`
- Feed-forward之后: `768 * 2`

### 总结

Multi-head Attention, Feed-forward, Layer Normalization这三部分我们得到了每个Transformer层中的参数的数量. BERT-base中共有12层. BERT-base chinese最终的参数为:

- Embedding: `(21128 + 512 + 2) * 768 + 768 * 2 = 16,622,592`
- 12层Transformer Block: `(2,362,368 + 4,722,432 + 768 * 2 + 768 * 2) * 12 = 85,054,464`

总参数量为: `16,622,592 + 85,054,464 = 101,677,056`

Huggingface提供的[bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)模型大小为`412 MB`, 以float32数据类型每个浮点数暂用4个字节来算, 101,677,056个浮点参数对应的字节为$$101,677,056 * 4 = 406,708,224 \approx 406\text{MB}$$. 与模型文件大小接近. 这是正常的, 词表大小为30522的[bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main)参数量为108808704, 理论模型大小为`435MB`, 实际模型大小为`440 MB`.
