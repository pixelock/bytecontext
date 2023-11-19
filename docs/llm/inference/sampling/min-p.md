# Top-p 采样的问题

![](/resources/images/llm/minp-1.png)

推理采样经常使用 `top-p` 的采样方法. `top-p` 采样的思路是, 在采样每个 token 时, 只从**累积概率**超过阈值 `p` 的概率排名靠前的 token 中进行抽样, 剩余的长尾 tokens 被舍弃.

这种思路, 关注了概率分布最核心的部分, 忽略了尾部部分, 因此也被称为核采样(nucleus sampling). 这样既可以避免采样到一些不合适 / 不相关的 token, 也保留了一些多样性.

但如果阈值 `p` 设置的比较大, 其实还是有很多低概率的 tokens 会进到抽样的候选池, 有概率被随机到. 低概率的长尾 tokens 相对来说相关性不高, 只要有一个 token 出问题, 后面就有概率偏离原始的意思.

# Min-p 的原理

`min-p` 做的事情很简单. 设置了一个最低值阈值, 只有概率超过这个值的 tokens 才会被考虑.

注意, 这个值代表的不是一个固定值, 它**会根据本次采样中最高概率的 token 对应的概率而变化**. 例如设置 `p` 为 `0.1`, 不是只保留概率值超过 `0.1` 的 token, 而是**允许概率至少是最大概率 token 对应的概率的 `1 / 10`**

![](/resources/images/llm/minp-2.png)

# Min-p 的优点

`min-p` 有助于以 `top-p` 通常无法实现的方式实现更多样化的选择, 从而不会限制创造力. 由于它截断的阈值参考了 top 概率, 因此真实的截断阈值不会过小, 将采样范围控制在 $$[p \times r_{top_{1}}, r_{top_{1}}]$$ 范围内, 范围内的最低概率最多就是一个量级(阈值设置为 `0.1`)左右的缩小, 从概率上来看, 仍然是比较合理的 token. 但 `top-p` 截断后, 对应的最小概率, 和本次采样中的最大概率, 可能就会相差很多个量级, 包含了大量的`无意义`的 tokens, 最终导致结果容易发生很大的偏移.

# 应用

在 [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3841) 框架中, 已经得到实现.

# 参考资料

- [放弃top p/k、温度采样，全面拥抱min p](https://mp.weixin.qq.com/s/ILwPmtLGSGDlKH6KbdTSaw)
- [Min P sampler implementation [alternative to Top P/Top K]](https://github.com/ggerganov/llama.cpp/pull/3841)
- [大模型文本生成——解码策略](https://zhuanlan.zhihu.com/p/647813179)
