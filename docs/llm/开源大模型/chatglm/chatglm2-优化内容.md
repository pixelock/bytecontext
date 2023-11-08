# 优化点

相比 ChatGLM, [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)做了如下的改进

- 更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练。对于更长的上下文，我们发布了 ChatGLM2-6B-32K 模型。LongBench 的测评结果表明，在等量级的开源模型中，ChatGLM2-6B-32K 有着较为明显的竞争优势

- 更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K

## ChatGLM2 与 ChatGLM 的区别

- Tokenizer 上使用的都是 SentencePiece, 但经过了重新训练, 相同的输入划分结果不同, vocab_size 大幅缩小
- 使用的 special tokens 不同
- 样本拼接 special tokens 的方法不同
- ChatGLM2 中的 position id 不再使用二维位置编码, 而是使用如同 Llama 的从0开始顺序+1的方式
  - 配合将 `[gMASK]` 放在序列的起始位置, 相当于只使用了上一版生成部分顺序递增那部分的 position id, 以适配训练方式产生的模型
- Attention 使用 Multi-Query Attention 技术(实际用的 Group-Query Attention)
- ChatGLM2 中, 最后的输出层没有与 token embedding 共享权重, 这点与 ChatGLM 是不同的
- 在计算RoPE时, query 和 key 只有一半的参数融合了位置信息, 另一半没有融合
- 使用 RMSNorm 代替 LayerNorm
- Attention 计算使用了 FlashAttention 实现.  具体来说使用的算子是 `torch.nn.functional.scaled_dot_product_attention()`
- FFN中使用了 SwiGLU 代替了 ChatGLM 中的 GELU 激活函数
