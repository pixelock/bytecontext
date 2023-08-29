# GLM

ChatGLM [v1](https://github.com/THUDM/ChatGLM-6B) [v2](https://github.com/THUDM/ChatGLM2-6B) 是基于 [GLM](https://arxiv.org/abs/2103.10360) 与训练模型进一步训练得到的产物. 因此在了解 ChatGLM 之前, 我们要对基座模型 GLM 了解清楚.

## Motivation

现有的与训练语言模型可以分为三类:

- **自编码模型 Autoencoding**: BERT, RoBERTa, ALBERT
- **自回归模型 Autoregressive**: GPT, GPT-2, GPT-3
- **编码器-解码器模型 Encoder-Decoder**: MASS, BART, PALM, T5

实际中的 NLP 任务也可以分为三类:

- **自然语言理解 NLU**: 情感分类, 抽取式问答, 自然语言推理等
- **条件生成 Seq2seq**: 摘要, 生成式问答, 机器翻译
- **无条件生成**: 语言建模

而上面三种预训练框架没有一种, 可以在所有任务上都取得最佳性能:

- 自编码模型: 擅长自然语言理解任务
  - 无法直接应用在条件生成或者无条件生成任务中, 而是需要进一步的 adaption 适配任务
- 自回归模型: 最擅长的是无条件生成任务
  - 由于没有双向 Attention 去建模 context, 所以在自然语言理解和条件生成中, 会表现的差一些. 而且这种输入完整 context 的推理方式与训练过程也不一致
- 编码器-解码器模型: 最擅长条件生成, 结构天然适合

因此, 不同的 NLP 任务适配不同的模型框架. 但随着预训练语言模型规模的快速扩增, 以及针对不同下游任务使用不同的模型开发将会变得代价高昂. 作者们希望用一个统一的新的预训练框架来适配不同类型的下游任务.

## 对标

现有的统一不同任务的预训练模型有:

- T5. 本身是编码器-解码器模型框架, 统一解决自然语言理解和条件生成任务. 但实验表明**需要更多的参数来达到和自编码模型相同的性能**.
- UniLM. 基于**掩码语言模型 Masked Language Modeling** 实现, 用不同的 attention masking 进行了针对自然语言理解, 无条件生成, 条件生成任务的预训练. 但 MLM 不能完全捕捉被掩盖字符之间的关系, 因此不适合文本生成任务

## GLM 的训练目标

### 自回归填空

GLM 通过 **自回归填空 Autoregressive blank infilling** 的任务目标来训练. 通过移除文本片段(下图中的绿色部分), 然后用**自回归**的方式来生成它们.

![](/resources/images/llm/glm-1.png)

给定一段输入文本 $$\boldsymbol{x}=\left[x_1, \cdots, x_n\right]$$, 从中采样多个文本片段 $$\{s_1, \cdots, s_m\}$$, 每个文本片段用一个 `[MASK]` token 来代替, 这样就构成了一段损坏后的文本 $$\boldsymbol{x}_{\text {corrupt }}$$.

这样就有 $$m$$ 个文本片段, 然后随机打乱这 $$m$$ 个文本片段的顺序, 可能会将在原文本中最后的文本片段排列到第一位, 而原位置靠前的片段排序靠后的情况, 将有 $$m$$ 个文本片段的所有可能的排列集合记为 $$Z_m$$. 预训练的目标为:

$$
\max _\theta \mathbb{E}_{\mathbf{z} \sim Z_m}\left[\sum_{i=1}^m \log p_\theta\left(\boldsymbol{s}_{\boldsymbol{z}_i} \mid \boldsymbol{x}_{\text {corrupt }}, \boldsymbol{s}_{\mathbf{z}_{<i}}\right)\right]
$$

可以看出打乱文本片段的目的, 是为了避免单纯地使用前面的片段生成后面的片段这种情况, 从而更好地完整捕捉不同片段之间的依赖关系.

在**一个片段内**, GLM 用从左到右的顺序按照自回归的方式, 预测缺失的 token. 生成片段 $$s_i$$ 的概率可以分解为:

$$
p_\theta\left(\boldsymbol{s}_i \mid \boldsymbol{x}_{\text {corrupt }}, \boldsymbol{s}_{\mathbf{z}_{<i}}\right)=\prod_{j=1}^{l_i} p\left(s_{i, j} \mid \boldsymbol{x}_{\text {corrupt }}, \boldsymbol{s}_{\mathbf{z}_{<i}}, \boldsymbol{s}_{i,<j}\right)
$$

上面是自回归填空的原理. 在具体实现上, 输入格式上, 输入序列 $$\boldsymbol{x}$$ 可以分为两部分, 如下图. A 部分是损坏后的文本 $$\boldsymbol{x}_{\text {corrupt }}$$, B 部分是被掩盖的文本片段.

在注意力机制中, A 部分的所有字符之间可以相互注意到, 但是不能注意到 B 部分的任何字符; B 部分的字符可以注意到 A 部分的所有字符, 以及 B 部分中自身之前的字符, 但是不能注意到 B 部分中自身之后的字符.

![](/resources/images/llm/glm-2.png)

### 三种基于自回归填空的预训练目标

通过改变**填空片段的数量**, 来产生针对上面三种不同任务的预训练目标.

**自然语言理解**

例如文本分类任务. 一般是要一个标签单词, 或者一些短片段. 为了让这种任务在下游中有更好的表现, 构建了一个 **token-level objective** 训练任务目标.

方式是从一个均值为 3 的泊松分布中, 采样片段的长度, 一个个片段采集, 直到原始文本中 15% 的 tokens 被覆盖, 然后将这些文本片段随机打乱后进行预测. 实际中, 15% 的填空比例, 对于下游自然语言理解任务的良好表现非常重要.

**无条件生成**

针对无条件文本生成, 设计了 **document-level objective** 训练目标. 从原始文本中使用均匀分布采样一个片段, 这个片段占原始文本长度的 50% 到 100%, 预测这个片段之后的内容.

**条件生成**

设计了 **sentence-level objective** 训练目标. 限制了每个被掩盖的文本片段必须是完整的句子, 采样多个句子, 直到覆盖了原始文本中 15% 的字符. 这个预训练目标针对的是输出目标为句子或者段落的下游有条件文本生成的任务.

**三种目标使用的 mask token 的区别**

上面三种目标实际中使用三种不同的 mask token:

- 遮掩单词: `[MASK]`
- 遮掩句子: `[sMASK]`
- 遮掩文档: `[gMASK]`

预训练的过程中需要用到三种目标的不同组合进行混合, 实际中的实现方法就是用不同的 mask token 替代原本的 token. 其中 `[gMASK]` 代表的是遮掩从当前位置到文档结束位置, 使用这**一个** `[gMASK]` 进行遮掩, 然后生成的是随机长度的文本(当前位置到文本结束的长度不定, 且样本之间的差别很大), 所以 GLM 对生成不定长的文档效果会比较优秀.

而像UniLM 这类自编码框架下的模型, 在预训练过程中, 要生成的每个位置都要用一个 `[MASK]` 来占位, 但推理过程中, 如果任务中生成的长度不定, 就会出现推理与训练阶段不一致的问题, 从而**削弱了模型对被 masked 的跨度和它们的上下文之间的依赖关系进行建模的能力**.

## 模型结构特点

### 二维位置编码

自回归填空的挑战之一, 是如何编码位置信息. B 部分的字符, 它同时具有两部分的位置信息:

- Position 1: token对应的是在哪个待填充文本片段
- Position 2: token在这个待填充片段中的位置

![](/resources/images/llm/glm-3.png)

GLM 使用了二维位置编码, 即**每个 token 都使用两个 position id 进行编码**. 第一个 position id 表示它在损坏后文本中的位置. 如果是 A 部分的 token, 对应的就是它自身的位置; 如果是 B 部分的 token, 对应的是它所对应的 mask token 的位置.

第二个 position id 表示它在填空内部的相对位置. 如果是 A 部分的 token, 对应的就是 0; 如果是 B 部分的 token, 对应的就是它在填空片段内部, 相对于片段起始的位置.

两个 position id, 通过两个不同的 embedding table, 映射成两个不同的位置向量, 然后加到输入的词向量上.

### RoPE

RoPE 是一种旋转式编码, 本质是**用绝对位置编码实现相对位置编码**. 从空间上理解, 通过乘性计算作用到每个 token 向量上, 相当于对每个向量进行了旋转, 旋转后向量相乘(attention 计算时)就会得到两个向量的夹角, 这个夹角代表 token 之间的相对位置关系.

### LayerNorm

Post-LN 在前向传播的链路上存在一个无法绕过的 LN 结构, 会导致训练过程容易发散. 为了训练的稳定性, 大模型普遍采用了 Pre-LN 结构. Pre-LN 结构在前向传播中, 有一条直通的恒等路径, LN 是加在 Attention 和 FFN 结构之前的..

但是在 **大规模 / 多模态** 混合精度训练(FP16)中, Pre-LN 也会存在不稳定的现象. 而 Pre-LN 的变体, Sandwich-LN 可以缓解这种现象, 做法是在 Attention 和 FFN 之后再加一个 LN, 进一步地缓解数值的溢出现象.

![](/resources/images/llm/glm-6.png)

GLM 采用的方法是 **DeepNet**, 本质上还是一种 Post-LN, 通过**更改初始化**以及**调整残差系数**, 稳定传播的稳定性. 实验证明可以做到千层 Post-LN 结构的稳定训练.

$$
\text{DeepNorm}(x) = \text{LayerNorm}(\alpha x + g(x)), \alpha > 1
$$

DeepNet 的详细原理可以参考: [为什么需要残差？一个来自DeepNet的视角](https://kexue.fm/archives/8994).

## 微调

### 分类任务

给定一个标注样本 `(x, y)`, 将输入文本 `x` 通过模板转化为有一个 `[MASK]` 字符的填空题问题 `c(x)`, 标签 `y` 也映射到了填空问题的答案 `v(y)`. 然后模型预测不同答案的概率, 概率对应了预测不同类别的概率.

![](/resources/images/llm/glm-4.png)

### 文本生成

针对文本生成的下游任务微调, 直接将 GLM 作为一个自回归模型应用. 给定的上下文构成了输入的部分, 即上文中的 A 部分. 在结尾附上一个 `[MASK]` 字符, 模型用自回归的方式生成了 B 部分的文本.

![](/resources/images/llm/glm-5.png)

## 训练设置

### 训练语料

与 BERT 相同, 使用了 BookCorpus 和英文的 Wikipedia. 并使用了 BERT 的 wordpiece 分词器.

### 训练情况

在 64 张 V100 GPU 上训练了 200k 步, batch size 为 1024, 最大序列长度为 1024.

### 训练产出

1. 使用 token-level objective 目标, 训练得到了 $$\text{GLM}_{\text{Base}}$$ 和 $$\text{GLM}_{\text{Large}}$$. 用来对标 BERT 的 Base 和 Large 模型, 分别有 110M 和 340M 的参数量.

2. 用 token-level objective 分别与 document-level objective 或者 sentence-level objective 目标混合, 训练了两个 Large 规模的模型: $$\text{GLM}_{\text{Doc}}$$ 和 $$\text{GLM}_{\text{Sent}}$$

3. 用 token-level objective 与 document-level objective 目标混合, 训练了两个更大的 GLM 模型, 分别为 410M (30 layers, 1024 hidden size), 515M (30 layers, 1152 hidden size, 18 heads), 分别称为 $$\text{GLM}_{\text{410M}}$$ 和 $$\text{GLM}_{\text{515M}}$$

# 与 T5 的对比

T5 中也提出了一个与 GLM 相似的文本填空目标, 来训练一个编码器-解码器框架的 Transformer 模型. GLM 与 T5 相比, 主要的区别有:

- GLM 使用了一个单一的编码器 Transformer 模型来同时学习单向和双向的注意力机制
- T5 的编码器和解码器使用了独立的位置编码, 并依赖多个 **sentinel token**(哨兵 token) 来区别不同的填空片段. 而 GLM 中使用二维位置编码来表示填空中的位置信息
  - 下游任务往往只有一个填空片段, 这种方法避免了模型拟合区分不同的 **sentinel token**
- 当有多个填空片段时, T5 是按照固定的从左到右的顺序来预测. 而 GLM 会随机打乱片段的顺序, 来完整地捕捉片段之间的依赖关系

# Scaling Up GLM

进一步训练了中文和英文两个版本的具有百亿参数规模的 GLM (48 layers, 4096 hidden size, 64 heads). 英文预训练数据为 Pile, 去掉了其中的代码预料等非自然语言的内容, 共计约 800GB 的文本数据. 中文使用了 WudaoCorpus 数据集, 共计约 1.2TB 的中文互联网文本数据.

训练规模为 48 个 DGX-A100 节点 (384 张 A100 GPU) 上进行训练. 一次完整训练大约花费 20 天.

---

# 参考资料

- [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)
- [自然语言大模型 ：GLM 通用语言模型的训练与微调](https://www.bilibili.com/video/BV1M84y1y7yu/?spm_id_from=333.337.search-card.all.click&vd_source=0e2d47c0c67fb509b32ba3bfc5b73819)
