# 整体指标

## pass@k

**pass@k** 指标的意义是: 对同一个问题(prompt), 让 LLM 生成 `k` 个答案, 只要有一个是正确的, 就认为模型通过了测试.

对同一个问题, 让模型直接生成 `k` 个答案, 去里面找有没有正确的. 但当 `k` 越来越大时, 模型更有可能产出正确的答案. 因此研究时, 也要考虑 `k`对评估指标的影响.

因此具体的做法是:

让模型一次生成 `n` 个答案, 从这 `n` 个答案中抽取 `k` 个, 对这 `k` 个答案进行评价. 具体的指标是

$$
\text{pass@k} = E_{\text{problems}}[1 - \frac{C_{n-c}^{k}}{C_{n}^{k}}]
$$

解释:

- $$n$$: 对同一个问题, 让模型产生n个答案
- $$k$$: 从这 $$n$$ 个答案中, 随机抽出 $$k$$ 个, $$k \le n$$
- $$c$$: 模型产生的这 $$n$$ 个答案中, 正确的答案有 $$c$$ 个
- $$\frac{C_{n-c}^{k}}{C_{n}^{k}}$$: 从模型产生的 $$n$$ 个答案里抽取 $$k$$ 个, 这 $$k$$ 个答案全部错误的概率
- $$\text{pass@k}$$: 从模型的答案中随机抽取 $$k$$ 个后, 能从这 $$k$$ 个里得到正确答案的概率

用 `numpy` 实现为:

```python
def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

**参考资料**

[ChatGPT技术解析系列之：赋予GPT写代码能力的Codex](https://zhuanlan.zhihu.com/p/611313567)
