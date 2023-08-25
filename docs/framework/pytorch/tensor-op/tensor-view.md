# 基础知识

## PyTorch 中 Tensor 存储方式

Tensor 采用**头信息区**(Tensor)和**存储区**(Storage)分开存储的方式, Tensor 变量的元数据及其存储的数据是分为两个区域分别存储的, 如下图所示. 比如我们定义 Tensor A 和 Tensor B 两个张量, 其中 Tensor B 是 Tensor A 通过形状转变得来的, 它们各自的形状(size), 步长(stride), 存储偏移(storageOffset)等元信息存储在各自的头信息区, 而 A, B 共享的数据位于同一存储区. 张量形状的改变, 只是重新计算了元信息. 这是一种**浅拷贝**.

![](/resources/images/framework/pytorch/view-1.png)

```python
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)
# `t` 和 `b` 共享底层数据
>>> t.storage().data_ptr() == b.storage().data_ptr()
True
# 对 view 之后的张量做修改也会影响到原来的张量
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)
```

## Tensor 的步长(Stride)属性

Tensor 的步长可以通过下图理解:

![](/resources/images/framework/pytorch/view-2.png)

可以将 Stride 总结为, **在遍历某一维的时, 该维度索引加1对应到在一维数据上的移动步长**. 在上面形状为 `(3, 3)` 的二维张量中, 对于第2维, 元素 `1` 的下一个数据是 `2`, 对应的步长为 1; 对于第1维, 元素 `1` 对应的下一个数据是 `4`, 在碾平的数据结构上跨度为3, 因此对应的步长为 3.

为什么要计算碾平后一维数组的偏移量呢? 因为数据在内存中, 是通过一维数组的形式存储, 通过偏移量获取数据. 而步长是计算偏移量的一个核心参数.

下图详细阐明了一个形状为 `(2, 3, 5)` 三维张量中, 以 `(0, 0, 2)` 为起点, 每一维索引 +1 的时候, 对应的内存中一维数组的偏移量.

![](/resources/images/framework/pytorch/view-3.png)

这里每一维的偏移量为 `[3 * 5, 5, 1] = [15, 5, 1]`.

再以一个形状为 `[2, 3, 4, 5]` 的四维张量为例, 计算每个维度的步长如下图:

![](/resources/images/framework/pytorch/view-4.png)

对应的代码为:

```python
arr = torch.rand(2, 3, 4, 5)  
stride = [1] # 初始化第一个元素  
# 从后往前遍历迭代生成 stride  
for i in range(len(arr.shape)-2, -1, -1):  
    stride.insert(0, stride[0] * arr.shape[i+1])  
print(stride)       # [60, 20, 5, 1]  
print(arr.stride()) # (60, 20, 5, 1)
```

# View Op 详解

`view()` 方法本质上是对 Tensor 中的元素进行**重新排列**, 修改了原来的排列方式, 并不会对数据进行显式地拷贝, 或者改变存储区(Storage)中元素数据的排列方式.

我们看到的 Tensor, 相当于是对存储区中的数据按照某种排列方式组成了一种**视图**. 通过 `view()` 方法, 可以通过**改变信息区(Tensor)**的方法, 改变数据的排列方式. `view()` 方法改变排列方式是通过**改变 stride 属性**来实现的. 看一个例子:

```python
import torch

A = torch.arange(6)
"""
tensor([0, 1, 2, 3, 4, 5])
"""
B = A.view(2,3)
"""
tensor([[0, 1, 2],
        [3, 4, 5]])
"""

# 查看 A 和 B 的存储区的内存地址
print(A.storage().data_ptr())
# 1881582170752
print(B.storage().data_ptr())
# 1881582170752

# 查看 A 和 B 的 stride 属性
print(A.stride())
# (1,)
print(B.stride())
# (3, 1)
```

## `view()` 与连续性

`view()` 只可以对**满足连续性条件的 Tensor 进行操作**. 张量的连续性条件指的是:

$$
\text{stride}[i] = \text{stride}[i + 1] \times \text{size}[i + 1]
$$

怎么理解这种连续性? 以张量的最后一个维度为例, 张量中某一个数字和它紧邻的数字, 在存储区中也是紧邻的, 即 stride 为 1, 这种情况下我们可以说是连续的. 对于倒数第二个维度, 将最后一个维度中的数字打包成一个 chunk 后, chunk 和 chunk 之间也是连续的.

新创建的 Tensor 是符合连续性条件的. 一般来说最后一个维度的 stride 为 1, 越靠前的维度对应的 stride 相对更大. 但是当对一个 Tensor 进行转置操作之后, 例如:

- `Tensor.t()`
- `Tensor.transpose()`
- `Tensor.permute()`

这些操作都会对 stride 属性进行修改, 而不是直接改动内存中数据的顺序来实现. 因此, 调整后的 stride 会失去连续性这个性质. 以 `Tensor.permute()` 为例.

```python
import torch
a = torch.arange(9).reshape(3, 3)  # 初始化张量a
print('struct of a:\n', a)
"""
struct of a:
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
"""

print('size   of a:', a.size())    # 查看a的shape
"""
size   of a: torch.Size([3, 3])
"""

print('stride of a:', a.stride())  # 查看a的stride
"""
stride of a: (3, 1)   # 满足连续性条件
"""
```

进行转置后:

```python
b = a.permute(1, 0)  # 对a进行转置
print('struct of b:\n', b)
"""
struct of b:
tensor([[0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]])
"""

print('size   of b:', b.size())    # 查看b的shape
"""
size   of b: torch.Size([3, 3])
"""

print('stride of b:', b.stride())  # 查看b的stride
"""
stride of b: (1, 3)   # 此时不满足连续性条件
"""
```

查看转置前后的存储区是否一致:

```python
print('ptr of storage of a: ', a.storage().data_ptr())  # 查看a的storage区的地址
# 2767173747136
print('ptr of storage of b: ', b.storage().data_ptr())  # 查看b的storage区的地址
# 2767173747136
```

整个过程可以用下图来阐释. 可以看出来, `permute()` 在 stride 列表上就是将对调维度的 stride 也进行对调.

![](/resources/images/framework/pytorch/view-5.png)

转置后的 stride 列表明显不符合连续性条件. 这时候再进行 `view()`, 就会报错:

```python
b = a.permute(1, 0)  # 转置
print(b.view(9))
"""
Traceback (most recent call last):
  File "xxx.py", line 23, in <module>
    print(b.view(9))
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
"""
```

这种情况下, 如果要使用 `view()` 方法, 需要使用 `contiguous()` 方法, 开辟一个新的存储区, 新的存储区中数字的存放顺序使得 Tensor 重新满足连续条件. 再使用`view()` 方法:

```python
import torch
a = torch.arange(9).reshape(3, 3)      # 初始化张量a
print('storage of a:\n', a.storage())  # 查看a的stride
# 0 1 2 3 4 5 6 7 8

b = a.permute(1, 0).contiguous()       # 转置,并转换为符合连续性条件的tensor
print('storage of b:\n', b.storage())  # 查看b的存储空间
# 0 3 6 1 4 7 2 5 8

print('size    of b:', b.size())       # 查看b的shape
print('stride  of b:', b.stride())     # 查看b的stride
print('viewd      b:\n', b.view(9))    # 对b进行view操作，并打印结果
"""
size    of b: torch.Size([3, 3])
stride  of b: (3, 1)
viewd      b: tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
"""

print('ptr of a:\n', a.storage().data_ptr())  # 查看a的存储空间地址
print('ptr of b:\n', b.storage().data_ptr())  # 查看b的存储空间地址
"""
ptr of a:
 1842671472000
ptr of b:
 1842671472128
"""
```

上面的结果可以看出, 张量 a 和 b 分别对应一个存储区, 且张量 b 可以使用 `view()` 方法改变张量的形状了.

## `view()` 与 `reshape()` 的区别

`view()` 和 `reshape()` 在表现上是一致的. 但 `reshape()` 可以不用关心张量是否满足连续性, 对于任意张量都可以调用, 相当于:

- 满足连续性的张量: `tensor.reshape() = tensor.view()`
- 不满足连续性的张量: `tensor.reshape() = tensor.contiguous().view()`

---

# 参考资料

- [一文读懂 Pytorch 中的 Tensor View 机制](https://zhuanlan.zhihu.com/p/464384583)
- [PyTorch：view() 与 reshape() 区别详解](https://www.iotword.com/2336.html)
- [Pytorch——Tensor的储存机制以及view()、reshape()、reszie_()三者的关系和区别](https://www.cnblogs.com/CircleWang/p/15658951.html)
- [TENSOR VIEWS](https://pytorch.org/docs/stable/tensor_view.html#tensor-views)
