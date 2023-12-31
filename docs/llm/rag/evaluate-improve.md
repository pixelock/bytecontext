# RAG 系统的表现如何衡量

RAG 系统产生的回答质量, 取决于两个方面:

- 召回内容是否包含解答 query 的全部信息
- LLM 根据正确的召回文档, 生成答案的能力

因此, 要改进 RAG 系统的性能, 需要精准定位在实际应用中的问题. 如果只根据最终的回答内容, 很难判断 RAG 系统中的那个环节出现了问题, 就无从谈起从哪部分进行优化.

我们从两个维度获取每个样本的结果:

- 检索是否正确: 是否根据 query 找到了合适的文档进行解答
- 生成的答案是否正确: LLM 根据输入的文档中的信息, 最终产生的答案是否是正确的

每个维度都分别有正确和错误两种情况, 将两个维度的结果组合, 可以得到下面这张包含 4 个维度的图:

![](/resources/images/llm/rag-eval-1.png)

每个象限对应这不同的情况, 也代表着不同的系统优化方向. 分别来看:

> 第一象限代表检索结果和最终答案都正确的情况, 这正是我们所期待的.

> 第二象限代表最终答案正确, 但检索结果却不正确的情况. 要分情况来看:
>  - 召回的文档虽然与标注的结果不同, 但也包含了解答问题需要的信息. 这时需要检查整个文档知识库中是否有大面积文档知识重复的情况, 这代表着知识库可能出现低质量的情况, 影响整体的召回质量
> - 解答当前 query 并不需要借助更多文档中的知识进行补充, LLM 中已经包含的知识就足以解决这个问题. 可以将这部分作为召回错误的情况来分析
> - 当前的答案可能是**部分正确**, 即只回答了 query 中的一部分意图, 其他部分或缺失或错误. 这时问题的原因可能在于虽然召回定位到了文档, 但文档切分得到的 chunk 粒度可能有问题, 可能是 chunk 太小, 导致相关的内容被切分成多块, 进而无法完整召回的情况. 因此需要调整分块的策略

> 第三象限代表检索结果错误且答案错误. 没有向 LLM 提供正确的知识, 就不能期望能给出正确的答案. 因此这部分的问题在于检索召回模块的问题, 需要优化检索方法

> 第四象限代表检索结果正确, 但答案错误. 这种情况可能是由于:
> - LLM 模型本身的问题, 产生了幻觉
> - prompt 设计的问题, 导致 LLM 无法理解问题, 需要优化 prompt template

# 评价方法 Metrics

从上面的 RAG 系统表现中, 可以总结出, 我们需要评价这些方面.

## 检索模块 Retrieval

### Context Precision

Context Precision 衡量的是 `ground-truth relevant items` 是否在召回的 contexts 里面排名在靠前的位置. 需要用 question 和 召回的 contexts 进行衡量.

### Context Recall

Context Recall

# 参考资料

- [Best practices for your ChatGPT on your data solution](https://medium.com/@imicknl/how-to-improve-your-chatgpt-on-your-data-solution-d1e842d87404)
