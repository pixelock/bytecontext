# Chain-of-Note(CoN)

## RAG 的问题

1. 首先, 无法保证信息检索系统总能产生最相关或最可信的信息. 检索到不相关数据可能导致误导性的响应, 并可能导致模型忽略其固有知识, 即使它具有充分的信息来解决用户查询的问题

2. 目前的 LLMs 在处理事实型问题时, 常会出现幻觉

## 回答模块的期望

智能系统应该能够确定其是否具有足够的知识(无论是内在的还是检索的)来提供准确的答案, 如果知识不足, 系统应该在无法确定答案时回复`未知`.

## CoN 的思路

改进 RAG 系统的鲁棒性, 主要关注两个关键方面:

1. **抗噪声能力**: 识别和忽略不相关检索文档中噪声信息的能力, 同时适当利用其内在知识

2. **未知鲁棒性**: RAG 应承认其局限性的能力, 即当给定其没有相应知识来回答的查询时, 以`未知`回复

CON 的核心是**对检索文档生成一系列阅读笔记**, 全面评估每一条召回的文档内容与 query 的相关性, 还可以确定其中最关键和可靠的信息. 这个过程中, 有效地过滤掉不相关或可信度较低的内容, 生成更准确, 更相关的上下文回.

此外, 在检索文档没有提供任何相关信息的情况下, CON 可以引导模型承认其局限性，并回复`未知`, 增强模型的可靠性.

下图阐述了 CON 的工作过程. 对于召回的文档, LLM 的回答中, 首先依次总结了每个文档在讲什么内容, 以及文档与 query 是否有关联. 最后才会对 query 进行回答.

![](/resources/images/llm/con-1.png)

整个过程由一次对话实现: 在召回得到 relevant documents 之后, 按照下面的 Prompt 构建输入, 由大模型回答:

```
Task Description:
1. Read the given question and five Wikipedia passages to gather relevant information.
2. Write reading notes summarizing the key points from these passages.
3. Discuss the relevance of the given question and Wikipedia passages.
4. If some passages are relevant to the given question, provide a brief answer based on the passages.
5. If no passage is relevant, direcly provide answer without considering the passages.
```

# 参考资料

Paper: [Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models](https://arxiv.org/abs/2311.09210)

[Chain-of-Note：增强检索增强语言模型的鲁棒性](https://zhuanlan.zhihu.com/p/667469305)
