​expand 和 repeat 函数是 pytorch 中常用于进行张量数据复制和维度扩展的函数.

# ​expand

- 传入张量每个维度扩张后的新维度, 例如张量有 4 个维度, `expand()` 函数传入的是 4 个参数, 分别代表每个维度扩展后新的维度值.
- 传入的参数数量大于张量维度, `expand()` 函数可以将原始张量**升维**. 在原始张量的维度之前, 增加更多的维度.

**只能对维度值为 1 的维度进行扩展**, 无需扩展的维度, 对应位置可写上**原始维度**大小或直接写作 **`-1`**.

**扩展的 Tensor 不会分配新的内存**, 只是原来的基础上创建新的视图并返回, **返回的张量内存是不连续的**. 如果希望张量内存连续, 可以调用 `contiguous()` 函数.

```python
import torch

a = torch.tensor([[1], [0], [2]])   # a -> torch.Size([3, 1])
b2 = a.expand(-1, 2)                 # 保持第一个维度，第二个维度只有一个元素，可扩展
'''
b2 -> torch.Size([3, 2])
b2为  tensor([[1, 1],
             [0, 0],
             [2, 2]])
'''

x = torch.rand((2, 1, 3, 1))
x_expand = x.expand(2, 3, 3, 2)
x_expand_1 = x.expand(-1, -1, -1, 4)

print(x.shape)
print(x_expand.shape)
print(x_expand_1.shape)

# torch.Size([2, 1, 3, 1])
# torch.Size([2, 3, 3, 2])
# torch.Size([2, 1, 3, 4])
```

升维的例子, 将原始张量复制多份.

```python
import torch
 
a = torch.tensor([1, 0, 2])     # a -> torch.Size([3])
b1 = a.expand(2, -1)            # 第一个维度为升维，第二个维度保持原样
'''
b1为 -> torch.Size([3, 2])
tensor([[1, 0, 2],
        [1, 0, 2]])
'''
```

只能拓展单数维(singleton dimensions), 即张量在某个维度上的 size 为 1. 非单数维上进行扩展, 会报错.

```python
a = torch.Tensor([[1, 2, 3], [4, 5, 6]])  # a -> torch.Size([2, 3])
b4 = a.expand(4, 6)  # 最高几个维度的参数必须和原始shape保持一致，否则报错
'''
RuntimeError: The expanded size of the tensor (6) must match 
the existing size (3) at non-singleton dimension 1.
'''
```

# repeat

和 `expand()` 作用类似, 均是将 tensor 扩展到新的形状. 区别在于

- 输入的参数格式类似, 但每个值, 代表在这个维度上要扩展多少倍. 因此**不允许使用维度 -1**, 如果这个维度保持不变, 输入 1.
- 与 `expand()` 相比, **可以扩展非单数维**.

`tensor.repeat(*sizes)`

参数 `*sizes` 指定了原始张量在各维度上复制的次数, 整个原始张量作为一个整体进行复制, **repeat 函数会真正的复制数据并存放于内存中**, 开辟了新的内存空间, 返回的张量在内存中是连续的.

与 `expand()` 相同的是, 当传入到 `repeat()` 函数中的参数数量大于张量维度, 同样可以将原始张量**升维**. 在原始张量的维度之前, 增加更多的维度.

```python
import torch
a = torch.Tensor([[1,2,3]])
'''
tensor(
	[[1.,2.,3.]]
)
'''
 
aa = a.repeat(4, 3) # 维度不变，在各个维度上进行数据复制
'''
tensor(
	[[1.,2.,3.,1.,2.,3.,1.,2.,3.],
	 [1.,2.,3.,1.,2.,3.,1.,2.,3.],
	 [1.,2.,3.,1.,2.,3.,1.,2.,3.],
	 [1.,2.,3.,1.,2.,3.,1.,2.,3.]]
)
'''


a = torch.tensor([1, 0, 2])
b = a.repeat(3,2)  # 在轴0上复制3份，在轴1上复制2份
# b为 tensor([[1, 0, 2, 1, 0, 2],
#        [1, 0, 2, 1, 0, 2],
#        [1, 0, 2, 1, 0, 2]])


a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
'''
tensor(
	[[1.,2.,3.],
	 [4.,5.,6.]]
)
'''
aaa = a.repeat(1,2,3) # 可以在tensor的低维增加更多维度，并在各维度上复制数据
'''
tensor(
	[[[1.,2.,3.,1.,2.,3.,1.,2.,3.],
	  [4.,5.,6.,4.,5.,6.,4.,5.,6.],
	  [1.,2.,3.,1.,2.,3.,1.,2.,3.],
	  [4.,5.,6.,4.,5.,6.,4.,5.,6.]]]
)
'''
```

# repeat_intertile

```python
torch.repeat_interleave(input, repeats, dim=None)
```

参数 `input` 为原始张量, `repeats` 为指定轴上的复制次数, 而 `dim` 为复制的操作轴, 若取值为 `None` 则默认将所有元素进行复制, 并会返回一个flatten之后一维张量.

与 `repeat()` 将整个原始张量作为整体不同, `repeat_interleave()` 操作是逐元素的.

```python
a = torch.tensor([[1], [0], [2]])
b = torch.repeat_interleave(a, repeats=3)   # 结果flatten
# b为tensor([1, 1, 1, 0, 0, 0, 2, 2, 2])
 
c = torch.repeat_interleave(a, repeats=3, dim=1)  # 沿着axis=1逐元素复制
#　ｃ为tensor([[1, 1, 1],
#        [0, 0, 0],
#        [2, 2, 2]])
```
