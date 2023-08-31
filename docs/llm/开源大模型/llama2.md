Llama2 是一系列预训练的大型语言模型, Llama2-Chat 是在 Llama2 的基础上进行微调, 得到的对话大语言模型.

# Llama2 基座模型

相比于 Llama, Llama2 的优化点在与:

- 训练数据增加 40%
- 上下文窗口扩展到 4k
- Attention 使用了 GQA(Grouped-query attention)

Context length: 4k的上下文, 对于chat, summarization, understanding longer documents 等任务有较好效果.

# Llama2-chat

Llama2-Chat 的训练过程:

从Llama 2预训练开始. 然后, 通过 SFT, 得到了 Llama 2-Chat 的初始版本. 随后使用 RLHF 对模型进行迭代优化. 是标准的三段式方法.

## SFT

SFT数据集的质量很重要, **万级别的高质量效果就很好**. 没有使用公开的几百万指令数据集, 而是找供应商精标了27540条(人工撰写 prompt 和 answer), **发现效果比几百万公开的要好**.

SFT 的超参数:

- batch size 64
- cosine 学习率调度 + 初始学习率 2 × 10−5
- 权重衰减（weight decay） of 0.1
- 序列长度（sequence length） 4096

**训练细节**

- concatenate 所有 prompts 和 answers ，保证 sequence length 是4096
- 在 prompt 和 answer 间加入 special token
- 计算 loss 的时候 mask 掉 user prompt ，只对 answer tokens 进行反向传播
- fine-tune model 2 epochs

## RLHF

参考: [一文读懂Llama 2](https://zhuanlan.zhihu.com/p/653303123)
