# Automix 要解决的问题

Automix 构建了一种**模型切换**(model switching)的工作流, 来提升系统回答的性能.

对于 LLM, 参数量的大小决定了模型的能力, 使用参数量更大的模型, 虽然得到的结果质量会更好, 但需要消耗更多的资源来做推理部署, 且生成相同数量的 tokens, 消耗的时间更长. 整体相应时间长, 即影响了用户的体验, 也会导致 QPS 上限不高.

而低参数量的 LLM 的能力有限, 存在一些 query 规模小的 LLM 不能正确回答, 而规模大的 LLM 可以正确回答.

Automix 对计算成本和性能的做了权衡, 同时使用两种不同规模的 LLM 参与到回答的系统中来.

# Automix 的做法

Automix 将一步问答转换成了三步.

1. 首先将用户的 query, 结合召回的相关文档, 使用小的 LLM 进行回答, 得到模型生成的答案

2. 答案不会直接返回, 而是进行**自我验证**(self-verification). 将 query, context document, answer 组成 prompt, 丢给大的 LLM, 评价当前生成的回答是否可以解答用户的问题
  2.1. 通过 prompt 控制模型输出的规模, 例如只输出`是`或`否`即可. 由于生成输出的规模很小, 因此高参数量的 LLM 的输出也是很快的, 对于系统的算力消耗可控, 可以支持高并发
  2.2. 为了提高验证的准确性, 还可以结合少量样本做 ICL

3. 如果第二步验证的结果为可以解答, 则可以将小模型生成的结果返回; 如果返回的结果为答案质量差, 再调用大模型对这个问题进行回答

总结一下, Automix 最终有两路工作流:

- 小模型先根据 context documents 生成答案
- 大模型验证上面生成的答案质量没有问题, 则返回答案, 一次问答中调用了一次小模型, 一次大模型, 大模型的输出很短, 资源消耗可控

或者:

- 小模型先根据 context documents 生成答案
- 大模型验证上面生成的答案质量有问题
- 大模型根据 context documents 生成答案, 并作为最终答案返回. 整个过程调用了一次小模型, 两次大模型, 其中一次输出短, 消耗可控, 另一次正常输出, 可能会输出很长的答案, 造成较多的资源消耗

但整体来说, 大部分用户的 query 使用小模型已经可以得到很好的回答, 因此整个系统中只有少部分情况会路由到大模型进行输出, 从系统的整体消耗上, 也做到了可控. 论文中使用 `LLAMA2-13/70B` 大小两个模型进行实验, 得到的结论是**将单位成本的增量效益提高了高达 89%**.

这样可以做到: **在资源消耗相对于只使用小模型扩增不大的前提下, 将系统的回答质量提升到与使用大模型近似的水平.**

过程可以用下图阐释.

![](/resources/images/llm/automix-1.png)

# 一些细节

## 自我验证使用的 Prompt

下面是结合了 ICL 的自我验证步骤使用的 prompt, 由 context, question, ai answer 三部分组成. 传给 LLM 进行判断, 输出限制在`正确`和`错误`的范围内.

```
"""Context: The manuscript, discovered in 1980 in a dusty attic, turned out to be a lost work of Shakespeare.

Question: Whose lost work was discovered in a dusty attic in 1980?

AI Generated Answer: Shakespeare

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context specifically mentions that a lost work of Shakespeare was discovered in 1980 in a dusty attic.

Verification Decision: The AI generated answer is Correct.

---

Context: The celestial event, known as the Pink Moon, is unique to the month of April and has cultural significance in many indigenous tribes.

Question: In which month does the celestial event, the Pink Moon, occur?

AI Generated Answer: July

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context clearly states that the Pink Moon is unique to the month of April.

Verification Decision: The AI generated answer is Incorrect.

---

Context: The Mona Lisa, housed in the Louvre Museum, is believed to be a portrait of Lisa Gherardini, painted by Leonardo da Vinci in the early 16th century.

Question: Who is believed to have painted the Mona Lisa in the early 16th century?

AI Generated Answer: Vincent van Gogh

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context specifies that the Mona Lisa was painted by Leonardo da Vinci in the early 16th century.

Verification Decision: The AI generated answer is Incorrect.

---

Context: The planet Kepler-442b, located 1,100 light-years away, is one of the most Earth-like planets ever discovered, having a similar size and orbiting within its star's habitable zone.

Question: How far away is the planet Kepler-442b?

AI Generated Answer: 1,100 light-years

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context states that Kepler-442b is located 1,100 light-years away.

Verification Decision: The AI generated answer is Correct.

---

Context: {context}

Question: {question}

AI Generated Answer: {generated_answer}

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation:"""
```

# 参考资料

Paper: [AutoMix: Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963)

Code: [Github](https://github.com/automix-llm/automix)

[【LLM研究解读】AutoMix：自动混合语言模型(AutoMix: Automatically Mixing Language Models)](https://zhuanlan.zhihu.com/p/662527085)
