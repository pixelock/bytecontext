# 为什么LLM大多使用Decoder-only架构

## 总结

Decoder-only架构, 并行处理更简单, efficient attention的加持, 使其在工程实现和训练效率上明显优于Encoder-Decoder架构.

在理论上是因为Encoder的双向注意力会存在低秩问题, 这可能会削弱模型表达能力, 就生成任务而言, 引入双向注意力并无实质好处.

所以, 在同等参数量, 同等推理成本下, Decoder-only架构就是最优选择了.

## 数学理论

在Encoder-Decoder架构中, 处理输入的Encoder部分使用的注意力是**双向的**, 输出部分的Decoder是**单向的**. Decoder-only使用相同的方法处理输入和输出, 都是**单向的**.

在模型参数量为同等规模的情况下, 通过UniLM(输入双向, 输出单向)与GPT(输入输出都为单向, 与UniLM一样都只有一侧)的对比, 可以得到:

> **输入部分的注意力改为双向不会带来收益, Encoder-Decoder架构的优势很可能只是源于参数翻倍**

数学上的原因为:

Attention矩阵一般是一个$$n\times d$$的矩阵和一个$$d\times n$$的矩阵相乘后, 再进行`softmax`, 由于$$n\gg d$$, **可以将Attention矩阵视为由低秩分解而来**. 因此得到的Attention矩阵是**低秩**的. **低秩问题表达能力的下降**.

而Decoder-only架构中, Attention矩阵是一个**下三角阵**, 而三角阵的行列式等于它对角线元素之积, 由于softmax的存在, 对角线必然都是正数, 所以它的行列式必然是正数, 即**Decoder-only架构的Attention矩阵一定是满秩的**. **满秩意味着理论上有更强的表达能力**.

Decoder-only架构的Attention矩阵在理论上具有更强的表达能力, 改为双向注意力反而会变得不足.

## 工程原因

模型规模大到一定程度, 工程的可行性, **稳定性**就成为了关键问题. 大模型训练一般都要使用pipeline parallelism, 使用megatron-lm框架训练, 可以看到Encoder-Decoder架构(如T5)是不支持pipeline parallelism.

再加上各种**flash/memory efficient attention**(如FlashAttention), T5这种使用relative positional encoding就没办法得到优化, 训练的效率会降低.

总之对于Encoder-Decoder架构, 并行训练工程难度, 以及各种efficient优化不适配, 导致训练效率低与Decoder-only架构.

## 实验表现

[What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832)

Decoder-only架构在没有任何tuning数据的情况下, zero-shot表现最好, 而encoder-decoder则需要在一定量的标注数据上做multitask finetuning才能激发最佳性能.

[UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)

发现模型在scale up之后, 尤其是**加上instruction tuning**之后, 其他架构能做的decoder-only模型也都能做了(各种NLU任务), 同时还有更高的上限和多样性.

# 参考资料

- [为什么现在的LLM都是Decoder-only的架构？](https://kexue.fm/archives/9529)
- [为什么现在的LLM都是Decoder only的架构？ - 王睿的回答](https://www.zhihu.com/question/588325646/answer/2929047413)
