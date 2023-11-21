# vLLM

[Github](https://github.com/vllm-project/vllm)

[Blog](https://blog.vllm.ai/2023/06/20/vllm.html)

vLLM 主要用于快速 LLM 推理和服务, 其核心是 **PagedAttention** 和 **Continuous batching**. 在无需任何模型架构修改的情况下, 可以做到比 HuggingFace Transformers 高达 24 倍的 Throughput.

vLLM 拥有诸多优点.

**快速性**

- 最好的服务吞吐性能(Throughput)
- 使用 PagedAttention 优化 KV cache 内存管理
- 动态 batching
- 优化的 CUDA kernels

**易用性**

- 与 HuggingFace 模型无缝集成
- 流输出
- 支持各种 decoder 算法, 包括 beam search, 并行采样等
- 支持 Tensor parallel 的分布式推理
- 具备与 OpenAI 形式的 API 接口形式

## vLLM 原理

### 原理总结

vLLM 没有对计算的速度进行优化, 因而对于一条序列的生成, 它与其他的框架在速度上没有优势. 但通过 PagedAttention 优化显存的使用率, 进而扩大了 batch size, 再通过 Continuous batching 进一步扩大了 GPU 利用率的优势, 最终使得推理系统的整体**吞吐量**显著变大, 特别适合在实际应用中落地.

### PagedAttention

LLM 服务的性能受到内存瓶颈的影响. 所有输入到 LLM 的 tokens 会产生注意力的 key 和 value 张量作为 KV cache, 保存在 GPU 中, 用来生成下一个 token. 

系统中 KV 缓存的大小是动态变化的, 这取决于文本序列的长度. 在 LLaMA-13B 中, 单个文本序列(`batch_size = 1`)最多需要 1.7GB 显存. 在 `batch_size > 1` 的情景中, 由于不同的文本序列生成的长度差别很大, 对应的 KV cache 差别也很大, 有效管理 KV cache 挑战较大. 这会导致显存**碎片化和过度保留**, 浪费了 60% ~ 80% 的显存.

为了解决这个问题, vLLM 引入了 PagedAttention. 这是一种受操作系统中虚拟内存和分页经典思想启发的注意力算法.

与传统的注意力算法不同, PagedAttention 允许在非连续的内存空间中存储连续的 key 和 value. 具体来说, PagedAttention 将每个文本序列的 KV cache 划分为块, 每个块包含**固定数量** token 的键和值. 在注意力计算时, PagedAttention 内核可以有效地识别和获取这些块, 从而在保障速度的同时, 减少了显存的浪费.

如下图, 现在我们有 `Alan Turing is a computer scientist and mathematician renowned for`, 要预测下一个 token. 对于 `for` 这个 token 的 query vector, 要与 key cache 中的每一个 token 计算得到 attention 结果. 此时每个分块的大小为 4 个 tokens, 因此当前的 10 个 token 被分成了 3 个 block, 分散在显存的非连续空间中.

![](/resources/images/llm/vllm-1.gif)

因为块在内存中不需要连续, 因而可以用一种更加灵活的方式管理 key 和 value. **文本序列的连续逻辑块通过块表映射到非连续物理块中, 物理块在生成新 token 时按需分配**.

还是以 `Alan Turing is a computer scientist` 为例, 现在有 6 个 tokens, 并且有 `Block 0 ~ Block 7` 共 8 个物理 blocks. 此时前 4 个 tokens `Alan Turing is a` 作为第一个逻辑块, 被分配到 `Block 7` 物理块中, 并在 `Block table` 记录逻辑块对应的物理块映射以及该物理块已经占用了多少位置.

接下来 `computer scientist` 这两个 tokens 作为第二个逻辑块, 被分配到 `Block 1` 物理块中. 接下来, 生成的下一个 token `and`, 由于 `Block 1` 还没有填满, 因此也被写到 `Block 1` 物理块中. 接下来生成的 token `mathematician` 也写入到 `Block 1` 物理块中.

然后生成的第 9 个 token `renowned`, 由于 `Block 1` 物理块已经写满, 需要开辟新的物理块. 新开辟的 `Block 3` 物理块记录与映射表 `Block table` 中, 并将新生成的 token 对应的 KV cache 写入到对应的物理块中.

![](/resources/images/llm/vllm-2.gif)

通过上面的示例, 我们可以看到, 在 PagedAttention 中, **内存浪费只会发生在序列的最后一个块中**. 这使得在实践中可以实现接近最佳的内存使用, 可以实现仅个位数的浪费.

这种内存效率的提升被证明非常有用, 允许系统将更多文本序列进行批处理, 即有效地扩大了 batch_size 的大小, 提高 GPU 使用率, 显著提升吞吐量.

PagedAttention 还有另一个关键优势: 高效的内存共享. 这在并行采样中十分高效. 并行采样可以用下图示例:

![](/resources/images/llm/vllm-3.gif)

PagedAttention 通过其块表格来共享内存. 不同的文本序列在生成 token 时, 如果共享相同的 prompt 前缀, 那么可以通过将它们的逻辑块映射到同一个物理块的方式来共享块, 而不需要在物理显存中显式地保存几份前缀 KV cache. 为了确保安全共享, PagedAttention 会对物理块的引用计数进行跟踪, 并实现写时复制(Copy-on-Write, 即只复制最后一个未填满的 block).

还是以 `Alan Turing is a computer scientist` 为例. 前 4 个 tokens `Alan Turing is a` 作为第一个逻辑块已经填满某个物理块, 这部分不再进行物理复制, 而是每个文本序列在生成时, 通过指针将逻辑块映射到相同的物理块上. 接下来 `computer scientist` 这两个 tokens 作为第二个逻辑块, 还没有填满. 在 Copy-on-Write 机制下, 进行复制, 然后分别由不同的序列逻辑块使用, 继续生成每个序列后续的 tokens.

![](/resources/images/llm/vllm-4.gif)

PageAttention 的内存共享大大减少了复杂采样算法的内存开销, 使得并行采样, **beam search** 的内存使用量降低了 55%, 这可以转化为高达 2.2 倍的吞吐量提升.

### Continuous batching

目前绝大部分优化是模型量化和自定义 CUDA 优化, 但对于目前的 LLM 来说, 迭代生成 tokens 的过程中, **内存IO** 带来的限制比算力的限制要更明显. 一个好的 ststem-level batching 优化方法, 能够带来至少 10 倍的性能提升.

LLM 推理面临是的 memory-IO bound 问题, 而不是算力限制的问题. 从显存中读取 1MB 的数据到 GPU 的计算核心中消耗的时间, 比计算核心对这 1MB 数据执行在 LLM 上的计算更耗时. 这意味着 LLM 推理的吞吐量, 很大程度上取决于当前的高带宽显存能够支持多大的 batch_size.

对于没有任何优化的静态 batching, 如果 batch 内所有的序列推理同时结束(同时达到 max token 或者同时输出 stop token), 此时 GPU算力 是完美使用的, 但这种情况概率很低. 更普遍的情况是, 当 max token 越大时, batch 内不同序列推理结束的位置差异也就越大, 早结束的序列仍然占用着显存, 对应的计算时无意义的. 此时 GPU 的利用率很低.

![](/resources/images/llm/cb-1.png)

Continuous batching 是一种**动态 batching** 技术. 当 batch 内的某一个序列输出 stop token 后, 说明这个序列已经输出了完整的结果, 此时会将一个新的序列插入到这个位置, 对新的序列继续进行生成. 就这样交替插入新序列代替已经完成的序列, 直到所有序列都已经完成生成. 这种机制使得 batch 内对于每一个序列的每一个 token 的计算都是有意义的, 从而大幅增加了 GPU 的利用效率.

![](/resources/images/llm/cb-2.png)

# 参考资料

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [LLM推理框架2：vLLM源码学习](https://zhuanlan.zhihu.com/p/643336063)
- [从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)
- [LLM 高速推理框架 vLLM 源代码分析 / vLLM Source Code Analysis](https://zhuanlan.zhihu.com/p/641999400)
