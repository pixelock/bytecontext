# MLM 任务

以一定的掩码率在模型的输入句子中随机选取部分 token, 将它们掩盖(mask)掉, 然后让模型经过训练而最终学会预测出这些被掩盖住的 token.

在原始的 BERT 中, 每个 token 都有一定概率被 mask 掉, 且它们之间是否被 mask 是相互独立的, 每个句子被 mask 掉的字词数量不固定.

# 原始的 MASK 方案

原始的 BERT 中, 对于每个样本句子, 随机地选择 15% 的 token 比例进行 mask 操作. 在这 15% 需要被 mask 的 token 中, 又按比例随机分成三部分, 进行不同的操作:

- 80% 的概率, token 将会被 `[MASK]` 这个 token 替换掉
- 10% 的概率, token 将会被替换为一个随机的其他 token
- 10$ 的概率, token 会保持不变

## 为什么要设计三种 mask 的替换方案

这是因为**在 pre-training 阶段和 fine-tuning 阶段存在 gap**. 如果全部替换为 `[MASK]` 的话, 而这个 `[MASK]` token 只会在 pre-training 阶段出现, 而在 fine-tuning 阶段不会出现, 这样的 gap 会影响 fine-tuning 阶段下游任务的表现.

**为了缓解 pre-training 阶段和 fine-tuning 阶段的 gap**, 对 MLM 任务的 mask 方案进行了改进, 增加了另外两种 mask 方案. 三种方案不同比例对下游任务的 fine-tuning 结果的影响, 在论文中的实现结果如下:

![](/resources/images/nlp/mlm-1.png)

总的来说, 采用三种 mask 方案的优点有:

- 在 pre-training 阶段和 fine-tuning 阶段存在 gap, 设计的目的是为了缓解这种 gap
- 在 transformer encoder 中, 不使用 `[MASK]` token 的位置, 模型不知道哪个 token 将要被预测, 也不知道哪个 token 被随机替换为其他 token, 因此 encoder 需要学习每个输入上下文中所有的 token 表示
- 随机替换再预测的任务难度更高, 但实际上只有 `15% * 10% = 1.5%`` 的 token 会被替换, 因此替换的 token 数量并不大, 对模型的语言理解能力影响很小

**举例说明**

> 原始句子为: `my dog is hairy`

假设随机 mask 的过程中, 第 4 个 token `hairy` 被选中, 则对 `hairy` 的处理有 3 中可能的情况:

> 80%的情况下将 `hairy` 替换为 `[MASK]`, 即 my dog is hairy -> my dog is [MASK]

> 10%的情况下将 `hairy` 替换为一个随机词, 例如 my dog is hairy -> my dog is apple

> 10%的情况下，保持 `hairy` 不变, 即 my dog is hairy -> my dog is hairy
