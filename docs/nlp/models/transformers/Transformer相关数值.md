首先明确符号:

- 序列的长度记为: $$s$$
- 隐向量长度 hidden size 记为: $$h$$
- heads 的数量记为: $$a$$
- 每个 head 内的隐向量长度: $$d = \frac{h}{a}$$
- 词表大小为: $$V$$
- transformer 层数为: $$l$$
- batch size: $$b$$

---

# 参数量

每层 transformer 包含两部分:

- Self-attention 以及相应的 layer normalization
- Feed-forward 以及相应的 layer normalization

## Self-attention

Self-attention模块参数包含 $$Q$$, $$K$$, $$V$$ 的权重矩阵 $$W_{Q}$$, $$W_{K}$$, $$W_{V}$$, 输出 $$W_{O}$$ 以及相应的偏置 Bias. 4个权重矩阵和4个偏置的总参数量为 $$4h^2 + 4h$$.

## Feed-forward

Feed-forward模块由2个线性层组成. 一般地(例如BERT中), 第一个线性层先将维度从 $$h$$ 映射到 $$4h$$ 了, 第二个线性层再将维度从 $$4h$$ 映射到 $$h$$. 因此第一个线性层权重矩阵形状为 $$[h, 4h]$$, 偏置为 $$4h$$; 第二个线性层权重矩阵形状为 $$[4h, h]$$, 偏置为 $$h$$. 两个线性层的总参数量为 $$8h^2 + 5h$$.

## LayerNorm

Self-attention 和 Feed-forward 各有一个 LayerNorm, 每个包含2个可训练参数: 缩放参数$$\gamma$$和平移参数$$\beta$$, 形状都是$$h$$. 两个LayerNorm对应的参数量为 $$4h$$.

## 每层参数量

因此每个transformer层的参数总量为 $$12h^2 + 13h$$.

## 模型总参数量

$$l$$ 层transformer的参数量为 $$l(12h^2 + 13h)$$.

词嵌入矩阵的参数量为 $$Vh$$. 可训练式的位置编码, 相比于词嵌入矩阵的参数量可忽略.

因此模型的总参数量为$$l(12h^2 + 13h) + Vh$$. 当模型规模较大时, 模型的参数量近似为$$12lh^2$$.

---

# 计算量 / Flops

FLOPs(floating point operations), 表示浮点数运算次数, 衡量了计算量的大小.

对于 $$A \in R^{1 \times n}, B \in R^{n \times 1}$$, 计算 $$AB$$ 需要进行 $$n$$ 次乘法运算和 $$n$$ 次加法运算, 共 $$2n$$ 次浮点运算, 即 $$2n$$ FLOPs. 对于 $$A \in R^{m \times n}, B \in R^{n \times k}$$, 计算 $$AB$$ 对应的计算量为 $$2mnk$$ FLOPs.

## Input

输入数据大小为 $$[b, s]$$, 经过 token embedding 层 $$[V, h]$$ 的映射转换为 $$[b, s, h]$$. 这一步的计算过程为: $$[b, s, V] \times[V, h] \rightarrow[b, s, h]$$, 对应的计算量为 $$2bshV$$

## Transformer Layer

### Self-attention

**第一步**

$$Q$$, $$K$$, $$V$$ 由输入 $$X$$ 映射得到. 对应的计算过程为 $$[b, s, h] \times[h, h] \rightarrow [b, s, h]$$, 计算量为 $$3 * 2 * bsh^2 = 6bsh^2$$

**第二步**

计算得到 attention score 矩阵. 这一步由 $$QK^{T}$$ 得到, 对应的计算过程为 $$[b, a, s, d] \times [b, a, d, s] \rightarrow [b, a, s, s]$$, 对应的计算量为 $$2 * ba * d * s^2 = 2 * b * a * \frac{h}{a} * s^2 = 2bhs^2$$.

**第三步**

计算 $$V$$ 在 attention score 矩阵上的加权, 对应的计算过程为 $$[b, a, s, s] \times [b, a, s, d] \rightarrow [b, a, s, d]$$, 相应的计算量为 $$2 * ba * s^2d = 2bhs^2$$.

**第四步**

对 attention 结果进行一次线性层的映射, 对应的计算过程为 $$[b, s, h] \times [h, h] \rightarrow [b, s, h]$$, 计算量为$$2bsh^2$$

**Self-attention 计算量总结**

$$8bsh^2 + 4bs^2h$$

### Feed-forward

Feed-forward 的计算公式如下 $$X=f_{\text {gelu }}\left(X_{\text {out }} W_1\right) W_2+X_{\text {out }}$$. 计算过程分为两个线性层.

第一个线性层, 对应的张量乘法为 $$[b, s, h] \times [h, 4h] \rightarrow [b, s, 4h]$$, 相应的计算量为 $$8bsh^2$$.

第二个线性层, 对应的张量乘法为 $$[b, s, 4h] \times [4h, h] \rightarrow [b, s, h]$$, 相应的计算量为 $$8bsh^2$$.

所以 Feed-forward 层的总计算量为 $$16bsh^2$$.

### Transformer Layer

每个 Transformer Layer 的计算量为 $$8bsh^2 + 4bs^2h + 16bsh^2 = 24bsh^2 + 4bs^2h$$.

## Output

经过所有的 Transformer Layer, 将转换为对词表中每个 token 的 logits. 对应的计算过程为 $$[b, s, h] \times [h, V] \rightarrow [b, s, V]$$, 相应的计算量为 $$2bshV$$.

## 总FLOPs

对于一个$$l$$层的Transformer模型, 一个 batch 的计算总量为:

$$l \times (24bsh^2 + 4bs^2h) + 2bshV$$

## 每个token + 每个模型参数 的平均计算量

模型总参数量为 $$l(12h^2 + 13h) + Vh$$, 一个 batch 内的 tokens 数量为 $$bs$$, 对应的计算总量为 $$l \times (24bsh^2 + 4bs^2h) + 2bshV$$. 对于规模比较大的模型, 来自词典的量可以忽略, 且一般有 $$h \gg s$$, 因此可以近似地认为每个token, 在每个参数上需要进行的浮点数平均计算量为:

$$
24lbsh^2 / (bs \times 12lh^2) = 2
$$

即平均下来, 每个token, 在每个参数上需要进行2次浮点数计算.

---

# 训练时间估计

训练transformer模型的时间由以下因素决定:

- 计算量, 进一步地由**模型参数量**和**训练总tokens**决定
- 计算效率, 主要取决于**GPU利用率**, **GPU峰值FLOPs**

上面得到对于每个 token, 每个参数平均需要计算2次浮点运算. 这是在前向传播过程中运算量. 在一个batch完整的训练过程中, 总共有以下几步:

- 前向传播
- 反向传播, 这里需要计算梯度, 还需要更新模型参数以及优化器梯度相关的计算, 相比于前向传递的计算量约为2倍
- 激活重计算, gradient checkpoints, 相当于重新进行了一次前向传播计算

因此一次完整训练过程的计算系数为 $$1 + 2 + 1 = 4$$, 由于每一次平均需要2次浮点运算, 因此完整过程需要8次浮点运算. 至此, 模型的训练时间约为:

$$
\text{训练时间} \approx \frac{8 \times \text{tokens数量} \times \text{\#params}}{\text{GPU数} \times {GPU峰值FLOPs} \times \text{GPU利用率}}
$$

一般来讲, GPU利用率一般在**0.3∼0.55**之间.

---

# 显存占用

## 训练阶段

训练阶段占用的显存分为两类. **模型状态**和**剩余状态**.

### 模型状态

![](/resources/images/llm/mixed-2.png)

假设模型的参数量为$$\Phi$$, 参考上图, 模型训练中的状态内存占用分为:

- 模型参数备份, Adam优化器中的 `momentum` 和 `variance`. 这三部分都使用 fp32 类型存储, 对应的字节占用大小为 $$4\Phi + 4\Phi + 4\Phi = 12\Phi$$
- 模型参数. 这部分使用 fp16, 对应字节大小为 $$2\Phi$$
- 模型梯度. 这部分使用 fp16, 对应字节大小为 $$2\Phi$$

总计$$16\Phi$$.

上面我们分析了 $$l$$ 层 transformer 模型的参数量为 $$l(12h^2 + 13h)$$, 因此模型状态占用的大小约为 $$16l(12h^2 + 13h)$$.

### 剩余状态

除了模型状态之外的显存占用, 包括**激活值**(activation), 各种临时缓冲区(buffer)和无法使用的显存碎片(fragmentation).

这部分主要就是前向传播过程中计算得到的**中间激活值**了, 需要保存中间激活以便在后向传递计算梯度时使用. 在混合精度训练的过程中, 中间激活值是以 fp16 或 bf16 的格式存储, 占用两个字节, 只有 **dropout 使用的 mask 矩阵**例外, 它只会占用一个字节.

我们主要分析 Transformer 层中间激活占用的显存大小.

#### Self-attention的中间激活

1. $$Q$$, $$K$$, $$V$$ 需要保存它们共同的输入 $$X$$, 这是来自上一层的中间激活, $$X$$ 的形状为 $$[b, s, h]$$, 元素个数为 $$bsh$$, 占用的显存大小为 $$2bsh$$ 字节.
2. $$QK^{T}$$ 矩阵乘法, 需要保存 $$Q$$, $$K$$ 中间激活, 两个张量的形状都是 $$[b, s, h]$$, 占用的显存大小为 $$4bsh$$ 字节.
3. 对于 $$\text{attention score} = \text{softmax}(QK^{T}/\sqrt{d_k})$$, 需要保存中间激活 $$QK^{T}$$, 对应的形状为 $$[b, a, s, s]$$, 显存占用大小为 $$2bs^2a$$ 字节.
4. 计算完 `softmax` 之后, 会进行一次 `dropout`, 需要保存一个 **mask 矩阵**, 形状与 attention score 矩阵相同为 $$[b, a, s, s]$$, 由于 mask 矩阵只占用一个字节, 因此显存占用大小为 $$bs^2a$$ 字节.
5. 使用 attention score 对 $$V$$ 进行加权求和, 即 $$\text{attention score} \cdot V$$, 需要将两部分都作为中间状态保存, $$\text{attention score}$$ 占用的显存大小为 $$2bs^2a$$, $$V$$ 占用的大小为与 $$Q$$ / $$K$$ 相同, 为 $$2bsh$$. 总计为 $$2bs^2a + 2bsh$$ 字节.
6. 对上一步求得的结果做一次输出线性映射, 保存映射前的张量 $$[b, s, h]$$, 对应的显存占用为 $$2bsh$$ 字节.
7. 最后还要进行一次 dropout 操作, 需要保存一个 mask 矩阵, 这部分占用 $$bsh$$ 字节.

因此 Self-attention 占用的中间激活总大小为 $$5bs^2a + 11bsh$$ 个字节.

#### Feed-forward的中间激活

$$
X=f_{\text {gelu }}\left(X_{\text {out }} W_1\right) W_2+X_{\text {out }}
$$

1. 第一个线性层需要保存其输入, 形状为 $$[b, s, h]$$, 占用显存为 $$2bsh$$ 字节.
2. 激活函数需要保存其输入, 形状为 $$[b, s, 4h]$$, 占用显存为 $$8bsh$$ 字节.
3. 第二个线性层需要保存其输入, 形状为 $$[b, s, 4h]$$, 占用显存为 $$8bsh$$ 字节.
4. 最后有一个 dropout 操作, 需要保存一个 mask 矩阵, 占用的显存为 $$bsh$$ 字节.

因此 Feed-forward 中间激活占用的总大小为 $$19bsh$$ 个字节.

#### Layer normalization的中间激活

Self-attention 和 Feed-forward 各自还对应一个 Layer Norm 层, 每个都需要保存其输入作为中间激活, 因此每个占用显存 $$2bsh$$ 字节, 两个 Layer Norm 占用 $$4bsh$$ 个字节.

#### 小结

综上, 每个 transformer 层需要保存的中间激活占用显存大小为 $$5bs^2a + 34bsh$$. 除了 transformer 层还有 embedding 层和最后的输出层, 在模型规模较大时, 这部分可以忽略. 因此对于 $$l$$ 层的 transformer 模型, 中间激活占用的显存大小近似为:

$$
l(5bs^2a + 34bsh)
$$

#### 中间激活与模型参数的显存占用对比

在一次训练迭代中, 模型参数或梯度占用显存大小只与模型参数量和参数数据类型有关, 与输入数据的大小是没有关系的. 优化器状态占用的显存大小与优化器类型有关, 与模型参数量有关, 但与输入数据的大小无关.

而中间激活值与输入数据的大小(批次大小 $$b$$ 和序列长度 $$s$$)是成正比的.

以**ChatGLM-6b**模型为例, [模型规模](https://huggingface.co/THUDM/chatglm-6b/blob/main/config.json)为:
201,326,592
```
h = 4096
a = 32
l = 28
s(max) = 2048
```

由此可以计算模型参数占用显存为:

$$
2l(12h^2 + 13h) = 2 * 32 * (12 * 4096^2 + 13 * 4096) = 11,277,271,040\text{Bytes} = 10.5\text{GB}
$$

计算中间激活占用的显存, 我们以输入序列长度为最大长度为例 `s = 4096`.

**`b = 1`**

$$
l(5bs^2a + 34bsh) = 26,776,436,736\text{Bytes} = 24.9\text{GB}
$$

中间激活占用的显存大小远大于模型参数, 而且随着批次大小的增大, 中间激活占用的显存远远超过了模型参数显存. 通常会采用**激活重计算技术**(gradient checkpoints)来减少中间激活, 代价是增加了一次额外前向计算的时间, 本质上是时间换空间.

## 推理阶段

在神经网络的推理阶段, 没有优化器状态和梯度, 也不需要保存中间激活. 因此模型状态占用的显存仅剩模型参数. 假设模型的参数量为$$\Phi$$, 使用 fp32 进行推理模型参数占用的显存为 $$4\Phi$$, 使用 fp16 占用的显存为 $$2\Phi$$.

### KV Cache

使用KV cache来加速推理过程. 一个典型的大模型生成式推断包含了两个阶段:

- **预填充阶段**: 输入一个prompt序列, 为每个transformer层生成 key cache和value cache, 即KV cache
- **解码阶段**: 使用并更新KV cache, 一个接一个地生成词, 当前生成的词依赖于之前已经生成的词, 并更新KV cache

假设输入序列长度为 $$s$$, 输出序列长度为 $$n$$, 以 fp16 来保存KV cache, 那么 KV cache 的峰值显存占用大小为 $$b(s+n)*l*2*2 = 4blh(s+n)$$. 第一个2表示 K/V Cache, 第二个2表示每个 fp16 占用2个字节.

还是以ChatGLM-6b为例, 假设`b = 1, s = 2048, n = 2048`, 则峰值显存占用为:

$$
4blh(s+n) = 4 * 1 * 28 * 4096 * (2048 + 2048) = 1,879,048,192\text{Bytes} = 1.75\text{GB}
$$

---

# 参考资料

- [Transformer参数量、计算量、显存占用分析](https://mp.weixin.qq.com/s/4_6J7-NZML5pTGTSH1-KMg)
- [DeepSpeed之ZeRO系列：将显存优化进行到底](https://zhuanlan.zhihu.com/p/513571706)
