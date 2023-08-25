# InfoNCE Loss

一种应用于对比自监督学习的损失函数. 对比学习的任务设计为:

一条样本包含一个 query, 以及一系列的 keys. 假设 query 经过 Encoder 后的特征为 $$q$$, keys 编码后的特征为 $$k_0, k_1, \cdots$$. 这些 keys 中, 只有**一个**为正样本, 即跟 query 是匹配的, 记为 key positive, 对应编码后的表征为 $$k_{+}$$; 其余的 keys 为负样本, 跟 query 不匹配.

在对比学习的过程中, 通过损失函数指导模型进行学习. 损失函数要求: 当 query 与唯一的正样本 key positive 相似, 跟其他的 key negatives 都不相似的时候, 损失函数的值应该比较小. 反之, positive 与 query 不相似, 或者任何一个 negative 与 query 相似了, loss 就应该比较大.

InfoNCE Loss 的公式如下:

$$
L_{q} = -\log\frac{\exp(\boldsymbol{q} \cdot \boldsymbol{k_{+}} / \tau)}{\sum_{i=0}^{k}\exp(\boldsymbol{q} \cdot \boldsymbol{k}_i / \tau)}
$$

上式中的 $$\tau$$ 是一个温度超参数. 如果忽略 $$\tau$$, 在形式上其实就是 $$k + 1$$ 多分类任务对应的 Cross entropy loss, 把 query 在 $$k + 1$$ 个 keys 进行匹配分类.

## 温度系数的作用

上式中 $$\boldsymbol{q} \cdot \boldsymbol{k}$$ 相当于是 `logits`, 温度系数的作用是**控制 `logits` 的分布形状**. 当温度系数 $$\tau$$ 值变大, `logits` 的分布变得**更平滑**, 每个样本(负样本)的贡献会比较平均, 相当于更平等地对待所有负样本; 反之 $$\tau$$ 值越小, 分布会变得更集中, 如果某个负样本的 `logits` 比较大(对应于 hard negative sample), 会明显增加 loss 值, 相当于迫使模型关注更难的负样本, 而忽略其他简单的负样本. 但如果这些 hard negative samples 其实是潜在的正样本, 则会造成模型收敛或者泛化的问题.

总之, $$\tau$$ 的作用是控制模型对负样本的区分度.
