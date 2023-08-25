# 获取 Tensor 维度

使用 `ndim` 属性或者 `dim()` 方法获取是几维张量, 两者是等价的.

```python
import torch

tensor = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6]]
)

print(tensor.ndim)
# 2
print(tensor.dim())
# 2
```

# 获取 Tensor 形状

## 获取完整形状

可以使用 `torch.Tensor.shape` 属性, 或者 `torch.Tensor.size()` 方法获取 Tensor 的形状. 两种方法返回的结果是等价的, 都是 `torch.Size` 类型.

```python
import torch

tensor = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6]]
)

print(f'tensor.shape: {tensor.shape}, type of tensor.shape: {type(tensor.shape)}')
# tensor.shape: torch.Size([2, 3]), type of tensor.shape: <class 'torch.Size'>
print(f'tensor.size(): {tensor.size()}, type of tensor.size(): {type(tensor.size())}')
# tensor.size(): torch.Size([2, 3]), type of tensor.size(): <class 'torch.Size'>
```

## 获取指定维度的大小

如果要获取张亮第 `n` 维的大小, 可以对上面获取的 `torch.Size` 类型结果直接切片, 返回数值为 `int` 类型. 或者是向 `size()` 方法传递维度 index. 这些方法都是等价的.

```python
import torch

tensor = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6]]
)

n = 1
print(f'size of {n} dim: {tensor.shape[1]}, type: {type(tensor.shape[1])}')
# size of 1 dim: 3, type: <class 'int'>
print(f'size of {n} dim: {tensor.size()[1]}, type: {type(tensor.size()[1])}')
# size of 1 dim: 3, type: <class 'int'>
print(f'size of {n} dim: {tensor.size(1)}, type: {type(tensor.size(1))}')
# size of 1 dim: 3, type: <class 'int'>
```
