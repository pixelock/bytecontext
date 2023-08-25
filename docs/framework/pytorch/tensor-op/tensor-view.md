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



---

# 参考资料

- [一文读懂 Pytorch 中的 Tensor View 机制](https://zhuanlan.zhihu.com/p/464384583)
- [PyTorch：view() 与 reshape() 区别详解](https://www.iotword.com/2336.html)
- [TENSOR VIEWS](https://pytorch.org/docs/stable/tensor_view.html#tensor-views)
