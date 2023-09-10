# torch.nn.functional.scaled_dot_product_attention

在 Pytorch2 提供了 `scaled_dot_product_attention` 算子, 来提升 attention 的计算速度. 在这个算子中:

- 首先计算 query 和 key 的 scaled dot product attentoin. 计算可以传入 attention mask
- 将 softmax 之后的 attention score 与 value 进行矩阵相乘, 加权汇总 value, 得到最后输出

等价于以下的 Pytorch 实现:

```python
# Efficient implementation equivalent to the following:
attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
attn_weight = torch.dropout(attn_weight, dropout_p)
return attn_weight @ V
```

## 实现方法

在这个算子中, 支持了三种 scaled dot product attention 实现:

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Memory-Efficient Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- 使用 C++ 实现, 实现的就是上面 Pytorch 等价代码中的逻辑

算子会**自动地**根据输入, 以及设备情况选择以上三种实现中, 最高效的一种. 如果想人工做到更精细力度的控制, 即关闭掉以上三种某一种的实现, 可以通过下面的 context manager 实现:

- `torch.backends.cuda.sdp_kernel()`: A context manager used to enable/disable any of the implementations.
- `torch.backends.cuda.enable_flash_sdp()`: Enables or Disables FlashAttention.
- `torch.backends.cuda.enable_mem_efficient_sdp()`: Enables or Disables Memory-Efficient Attention.
- `torch.backends.cuda.enable_math_sdp()`: Enables or Disables the PyTorch C++ implementation.

## 模型应用

- 在 ChatGLM2 中使用了这个算子来实现高效地计算.

---

# 参考资料

- [TORCH.NN.FUNCTIONAL.SCALED_DOT_PRODUCT_ATTENTION](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)