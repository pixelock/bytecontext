BERT中的参数存在于Embedding, Attention和Feed-forward中的Linear模块, 以及Layer Norm中的参数.

BERT中各种参数的初始化方法为:

```python
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # initializer_range = 0.02
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

normal_中的关键逻辑为:

```python
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
```

除了bias以及layer norm, 其他参数的初始化均使用均值为0, 标准差为0.02的, 上下限宽度为2倍标准差的截断正态分布进行采样初始化. 由于是截断正态分布, 所以实际标准差会更小, 实际标准差大约是$$0.02/1.1368472\approx 0.0176$$.

我们知道初始化的目标, 在于**尽量让输入输出具有同样的均值和方差, 从而能保证初始阶段模型的稳定性**. 对于Xavier初始化, 一个$$n\times n$$的矩阵应该用$$1/n$$的方差初始化, 而BERT base的$$n$$为768, 算出来的标准差是$$1/\sqrt{768}\approx 0.0361$$, 这就意味着0.02这个初始化标准差是明显偏小的, 大约只有常见初始化标准差的一半.

# 初始标准差为什么是0.02？

为什么BERT要用偏小的标准差初始化呢.

偏小的标准差会导致函数的输出整体偏小, 从而使得**Post Norm** $$x_{t+1} = \text{Norm}(x_t + F_t(x_t))$$ 设计在初始化阶段更接近于恒等函数, 从而更利于优化. 具体来说, 如果$$x$$的方差是1, $$F(x)$$的方差是$$\sigma^2$$, 那么在初始化阶段, Norm操作就相当于除以$$\sqrt{1+\sigma^2}$$, 如果$$\sigma$$比较小, 那么**残差中的直路权重就越接近于1, 那么模型初始阶段就越接近一个恒等函数, 就越不容易梯度消失**.

那能不能设置得更小甚至全零, 一般来说初始化过小会丧失多样性, 缩小了模型的试错空间, 会带来负面效果. 综合来看, 缩小到标准的1/2, 是一个比较靠谱的选择了.

# 参考资料

- [初始标准差为什么是0.02？](https://kexue.fm/archives/8747/comment-page-1#%E5%88%9D%E5%A7%8B%E6%A0%87%E5%87%86%E5%B7%AE%E4%B8%BA%E4%BB%80%E4%B9%88%E6%98%AF0.02%EF%BC%9F)
