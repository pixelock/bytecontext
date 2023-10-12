# ABSA 任务介绍

早期的情感分析是判断整段文本的情感倾向, 但是很多实际应用需要更细粒度的情感分析. 比如一句话中描述了多个主体, 而不同的主体具有不同的情感. 比如 `这家餐厅披萨很好吃但是服务太差了` 这句话, 通过细粒度的分析可以提取出: `(披萨, +), (服务, -)`.

Aspect Based Sentiment Analysis(ABSA) 细粒度情感分析, 可以拆分成两个个小任务, 分别是:

- 属性识别 Aspect Extraction
- 属性粒度情感分类 Aspect-level Sentiment Analysis

# 属性识别

属性识别常见的有两类任务:

- Aspect Category Detection, ACD, 属性类型检测
  - 任务类型: 多标签分类
  - **预定义一些属性**, 与舆情文本进行分类, 一条舆情可能属于多个属性, 也可能不属于任意一个属性
- Aspect Term/Target Extraction, ATE, 属性词提取
  - 任务类型: 序列标注
  - 指直接从原文本中提取与属性关联的单词或词组

# 属性粒度情感识别

- Aspect Category Sentiment Classification, ACSA, 属性类型极性检测
  - 任务类型: 文本分类
  - 对上个任务得到的实体类型情感极性做预测

## 模型

### AEN-BERT

Paper: [Attentional Encoder Network for Targeted Sentiment Classiﬁcation](https://arxiv.org/pdf/1902.09314.pdf)

Code: [Github](https://github.com/songyouwei/ABSA-PyTorch)

![](/resources/images/tasks/bert-aen.png)

#### 模型结构

**Embedding Layer**

使用 BERT 做这一层, 产生每个 token 的 embedding.

输入分为 context 和 target 两部分, context 是文本, target 是识别出的实体, 也是要判断其情感类别的实体. 这两部分分别用以下的方法组织输入:

- context: `[CLS] + context + [SEP]`
- target: `[CLS] + target + [SEP]`

这里的 target 就是每个实体本身, 详情可见数据集 [acl-14-short-data](https://github.com/songyouwei/ABSA-PyTorch/tree/master/datasets/acl-14-short-data).

**Attention Encoder Layer**

1. **Intra-MHA**

Intra-MHA 就是 self attention, 对 context 建模, context 的 token 之间交互, 得到每个 token 的交互表征.

2. **Inter-MHA**

将 target 中的 token 与 context 中的 token 通过 cross multi-head attention 交互, 得到融合上下文信息的 target 每个 token 的新的表征.

论文中这部分使用的 attention 中计算 softmax 之前的 score, 并不是通过 q, k 的点积得到, 而是 q, k 拼接后的再通过 Linear 层映射.

![](/resources/images/tasks/bert-aen-2.png)

![](/resources/images/tasks/bert-aen-3.png)

3. **PCT**

Point-wise Convolution Transformation, 逐点卷积变换. 使用卷积对上面两种 attention 的输出再做一次映射. 公式如下, 由于使用的卷积核大小为 1, 相当于对每个 token 进行两次单独的线性映射变换, 相当于是过了一次 FFN 结构.

![](/resources/images/tasks/bert-aen-4.png)

**Target-specific Attention Layer**

通过另外一个 cross multi-head attention, 再次以 target 为输入, context 为 cross hidden states, 将两者做交互, 输出 target 融合 context 信息的表征.

**Output Layer**

将 context, target, 以及 Target-specific Attention Layer 这三部分输出的 tokens 表征, 通过 avg pooling 的方法, 得到序列表征, 然后拼接在一起, 通过 Linear 层, 得到最终的交叉熵损失.

#### 正则化和模型训练

由于中性情感是非常模糊的情感状态, 因此标签为中性的训练样本是不可靠的. 刨除掉中性标签, 只剩下 positive 和 negative 标签, 一般是要做二分类.

论文中使用的是**标签平滑正则化(LSR)**, 可以通过防止网络在训练过程中为每个训练样本分配全部概率来减少过拟合. 公式如下:

![](/resources/images/tasks/bert-aen-5.png)

`q(k|x)` 代表样本的 ground-truth 标签分布(值为 1 或 0), 将 ground-truth 替换为上式, 即加入了标签分布的先验概率 `u(k)` 作为平滑. 并且定义标签先验概率 `u(k)` 与模型预测结果 $$p_{\theta}$$ 之间的 KL 散度作为一个损失项:

![](/resources/images/tasks/bert-aen-6.png)

最终要优化的损失函数, 包括 KL 散度和 L2 正则项的限制, 为:

![](/resources/images/tasks/bert-aen-7.webp)

### RoBERTa + MLP

在 [Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa](https://arxiv.org/abs/2104.04986) ([Github](https://github.com/ROGERDJQ/RoBERTaABSA/tree/main)) 这篇论文中, 对比了当时热门的方案: 引入句法信息和图神经网络到类 BERT 模型中, 完成细粒度情感分析任务, 研究不同的依存树/图结构在ABSA模型中有怎样的效果.

实验结果表明, 不同的树结构对任务的影响**有限**, 一些任务微调后的预训练模型中诱导的依存树, 可能比语言学家定义的句法依存树更适应任务本身.

最重要的是, 实验验证**直接基于预训练模型进行微调**, 就可以达到 SOTA 的效果. 原因是可以利用到预训练模型中隐式蕴含的依存树, 这是在漫长的训练过程中, 基于海量样本学习到的.

下图是当时 ABSA 任务排行榜上的成绩, 可以看到 RoBERTa + MLP 的性能结果在不包含额外训练数据的模型中取得第一的成绩.

![](/resources/images/tasks/absa-1.png)

即使在今天(2023年10月11日), 排行榜上仍然排在第六, 且与第三的差距也不大.

![](/resources/images/tasks/absa-2.png)

查看代码, 使用的就是 RoBERTa + 两层 MLP 的结构.

```python
if model_type == "roberta":
    embed = RobertaWordPieceEncoder(model_dir_or_name=model_name, requires_grad=True)
elif model_type == "bert":
    embed = BertWordPieceEncoder(model_dir_or_name=model_name, requires_grad=True)
elif model_type == "xlnet":
    embed = XLNetModel.from_pretrained(pretrained_model_name_or_path=model_name)
elif model_type == "xlmroberta":
    embed = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path=model_name)


class AspectModel(nn.Module):
    def __init__(self, embed, dropout, num_classes, pool="max"):
        super().__init__()
        assert pool in ("max", "mean")
        self.embed = embed
        self.embed_dropout = nn.Dropout(dropout)
        if hasattr(embed, "embedding_dim"):
            embed_size = embed.embedding_dim
        else:
            embed_size = embed.config.hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size, num_classes),
        )
        self.pool = pool


model = AspectModel(
    embed,
    dropout=args.dropout,
    num_classes=len(data_bundle.get_vocab("target")) - 1,
    pool=args.pool,
)
```

使用的输入格式为 `[CLS] s [SEP] t [SEP]`, s 代表 context, t 代表 target, 即实体本身文本.

在 [Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification](https://arxiv.org/abs/1908.11860) 这篇论文中使用 BERT 使用相同的方式实现. 这篇论文中还提到使用领域数据进行 continue pretraining, 会得到更好的效果. (肯定的)

### Constructing Auxiliary Sentence

Paper: [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf)

Code: [Github](https://github.com/HSLCY/ABSA-BERT-pair)

这篇论文的思路是为**属性集合中的每个属性**, 结合句子中出现的每个 target, 构成若干个 (target, aspect) pair. 例如当前的属性有 4 种: `{general, price, transitlocation, safety}`, 下面的句子中包含两个 target, 就可以构建出 8 个 pairs:

![](/resources/images/tasks/absa-3.png)

然后对这些 target-aspect pair 构建辅助的问句, 引导模型根据属性进行回答, 将任务转换为**阅读理解**任务. 例如这里的标签为 `{positive, negative, none}`, 对上面的 `(LOCATION1, safety)` 这一对, 可以构建 `what do you think of the safety of $LOCATION1 ?`. 把这个句子称为 Auxiliary Sentence.

然后根据 Auxiliary Sentence 的引导, 通过 **Fine-tune** 的方式训练模型. 样本的构建方式为 `[CLS] raw sentence [SEP] auxiliary sentence [SEP]`, 使用最后一层的 `[CLS]` 对应的向量作为 pooling vector, 然后接一个 Linear, 得到 3 个类别对应的 logits, 再使用 cross-entropy 得到 loss.

还可以更近一步, 在 (target, aspect) 二元组的基础上, 增加标签类别, 扩建为 (target, aspect, label) 三元组, 相应的 auxiliary sentence 变为:

```
the polarity of the aspect safety of location - 1 is positive
the polarity of the aspect safety of location - 1 is negative
the polarity of the aspect safety of location - 1 is none
...
```

共 `num target * num aspect * num label` 条样本, 每条样本是一个二分类. 实验结果表明这种方式效果更好.

#### 优缺点

优点是:

1. 比较灵活, 属性可以无限的扩充, 不管是新增或者减少属性, 不需要对模型结构进行改动
2. 准确率高

缺点是:

1. 效率低, 如果属性很多, 再加上句子中 target 数量很多, 要对所有的组合过一遍模型, 资源消耗大

### Multi-label 多标签

转化为一个多任务联合训练. 每个属性就是一个任务, 多分类任务. 在训练时, 绝大多数句子只包含部分属性, 因此还需要为每个任务增加一个 `None` 类别, 代表该属性在句子中没有出现过. 然后将每个属性任务的 loss 汇总起来得到总的 loss, 进行联合训练.

对于属性较多的情况, 常常会出现标注全部属性的成本很多, 难度很大(要求标注人员对一条文本进行全属性标注, 需要标注人员考虑很多方便, 容易造成遗忘, 影响数据质量). 改为按属性标注. 引入 label mask 机制, 将一条样本中没有标注的属性任务 mask 掉, 不参与到总 loss 的汇总, 如下图:

![](/resources/images/tasks/absa-4.png)

这样做有以下好处:

1. 将属性识别和属性情感分类两个任务综合为一个任务(推理阶段, 如果属性的预测值为 `None`, 代表没有这个属性)
2. 灵活应对属性的变化. 如果业务域中的属性变化较快, 经常增删属性, 使用这种方法只需增加一个任务, 对应增加任务的 output layer(通常就是一个 Linear 层), 然后只对这一个任务进行单独训练即可(例如 freeze 住 base encoder 的参数)
3. 降低了标注成本, 提升了对噪音文本的容忍. 如果一个句子中属性较多, 出现了漏标的情况, 在其他模型的训练中会带来噪音. 或者只标注某个属性的标签(以降低标注成本), 结合 label mask 机制, 这些样本都可以正常使用
4. 训练和推理时, 句子只编码一次, 效率高

### Prompt

Code: [Github](https://github.com/jzm-chairman/sohu2022-nlp-rank1)

以 BERT 或 XLNet 这种 Masked Language Model 的性质, 以 `[SEP]` 符号为界, 第一段作为文本输入, 第二段按顺序输入所有实体, 实体之间使用 `[MASK]` 进行间隔. 每个实体后面紧跟的 `[MASK]` 作为输出位置, 通过 BERT Encoder 得到 `[MASK]` 位置的输出, 接一个 MLP, 得到最后的分类结果.

![](/resources/images/tasks/absa-7.webp)

第二段实体部分可以认为是 prompt, 通过 prompt 引导的方式, 对 mask 的位置输出.

### InstructABSA

Paper: [InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis](https://arxiv.org/pdf/2302.08624v5.pdf)

Code: [Github](https://github.com/kevinscaria/instructabsa)

通过 Instruction tuning 的思路, 组织训练语料对 LLM 进行 SFT.

对于属性粒度的情感识别任务, 按照下面的方式构建 instruction, 包含以下几部分:

- Definition 任务定义, 对任务进行描述
- Few shot samples, 每种分类给至少一个例子
- Input prompt, 待识别的文本, 拼接要识别的属性

经过 LLM, 通过生成的方式生成 output label.

![](/resources/images/tasks/absa-5.png)

一个具体的例子:

![](/resources/images/tasks/absa-6.png)

# 参考资料

- [Aspect-Based Sentiment Analysis (ABSA) on SemEval 2014 Task 4 Sub Task 2](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval)
- [细粒度情感分析：还在用各种花式GNN？或许只用RoBERTa就够了](https://zhuanlan.zhihu.com/p/366133681)
- [BERT在情感分析ATSC子任务的应用](https://zhuanlan.zhihu.com/p/250958929)
- [【情感分析】华为云细粒度文本情感分析及应用](https://blog.csdn.net/qq_27590277/article/details/114465660)
- [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch#bert-for-sentence-pair-classification-bert_spcpy)
- [2022搜狐校园算法大赛 NLP赛道第一名方案分享](https://zhuanlan.zhihu.com/p/533808475)
