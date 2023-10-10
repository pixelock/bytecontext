# MMoE 理解

## Gate 的本质

每个 task 有各自的 gate 网络, 学习不同 expert 的权重, 本质上是学不同 task 的样本权重. 每个 task 通过 gate 得到各个 expert 的权重, 在梯度回传时, 不同 expert 的参数更新计算, 直接和 gate 的权重相关, 相当于在 expert 层对不同 task 的样本采用了不同的权重.

## Expert 子空间

Expert 对输入的 embedding 表征进行多层 MLP 操作, 得到高阶特征表达, 实现特征的提取, 不同 expert 对应不同特征子空间.

Embedding 表征的不同维度代表不同的特征, 例如在推荐场景, embedding 的不同位置会代表 item 的标题, 封面, 具体内容, 背景音乐, 人物颜值等, 不同的 expert 学到这些维度的特征子空间时, **实际是在不同的维度进行特征提取**, 这就是 Expert 的含义和目的.

实际模型学习过程中, 这些维度的含义并不像人为定义地这样清晰, 模型学的是隐式的一些空间维度, 这些维度的差异性, 可以通过对 expert 的信息表征进行可视化后对比.

模型结构的设计中, 和这个出发点类似于 attention 中的 multi-head, 希望可以通过不同的 head 提取到不同子空间的兴趣, 从而得到更充分的兴趣表征.

## 效果来源

相比 share-bottom 的结构相比, MMoE模型除了结构上的改进, expert 结构还增加了模型容量.

溯源模型带来的效果提升时, 首先将 expert 结构的参数与 share-bottom encoder 的参数量对齐, 然后将 embedding 分别直接输入到 share-bottom encoder 和 encoder 中, 再接多个 task 学习.

MMoE模型的能力上限没有提高, 理想情况下, 当 share-bottom 结构学得足够好, 达到了模型的上限, 这时候使用同等模型容量的 MMoE 模型, 不会带来收益.

MMoE之所以能带来效果的提升, 是因为模型在实际上很难达到理论的上限, 只是不断地接近上限. 通过更合理的结构设计, 降低了模型的学习难度, 使它相比于 share-bottom 这种结构, 能更容易接近效果上限.

# MMoE 问题

## 1. 不同任务的拉扯

**负迁移/跷跷板现象/拉扯**

多任务学习时, 存在不同任务的拉扯, 具体在网络学习过程中体现是对 shared bottom 的更新不一致, 对某个参数而言, 一个 task 的更新梯度是正向的, 另一个 task 的更新是负向的.

实践验证: 可以通过统计不同 task 对同一个参数更新的分布来验证.

或者通过简单地去掉某些 task, 观察剩下 task 的离线效果是否有提升. 如果有提升, 则验证了 share-bottom 结构在同时学习这些具有拉扯的任务时, 存在task之间的负向迁移作用, 使整体效果受影响.

## 2. Expert 学习不均衡

MMoE 中每个 task 的输入由各自的 gate 网络自适应生成 expert 的权重, 再对 expert 表征进行加权组合得到.

这种方式导致模型在训练过程中出现权重在 expert 上分布极度不均衡的问题, task 在不同 expert 的权重差异极大: 某些 expert 权重极高, 接近于 1, 而其它 expert 权重极低, 接近 0, 出现了**坍缩现象**, 只有极少数 expert 被激活.

当某个 expert 权重为 1 时, 模型则退化成 share bottom + MLP 的结构, expert的设计失效.

而在训练过程中, 权重高的 expert 重要性高, 学习速度快, 权重低的 expert 学习速度慢, 不同 expert 学习速度不均衡, 导致模型受学习速度快的 expert 主导, 学习慢的 expert 作用有限, 从而影响模型的表征能力, 造成了坍缩的现象.

强者越强, 弱者越弱.

## 3. Gate 收敛的稳定性

MMoE 每个 task 对 expert 的权重由各自对应的 gate 生成. 在训练过程中, gate 权重在 expert 上的分布容易出现不稳定的情况, 变化剧烈.

对于某个 task, 训练过程中 gate 权重分布的剧烈变化, 导致 task 对 expert 的学习不稳定, 网络收敛难度大, 甚至难以收敛好.

# MMoE 改进

## Expert 改进

当 task 的共性较强而 expert 网络较简单时, expert 容易先学习到共性信息, 而差异信息难被表征. 当 expert 被共性信息表征主导后, 因为它包含了各个 task 的共性的信息, 所有 task 的学习对它的依赖性高, 加快了它的学习速度, 逐渐导致 gate 在学习权重时, 则容易出现权重坍缩在共性信息表征强的 expert 上.

提高 expert 提取信息的能力, 是 expert 改进的主要方向. 可以从以下的方面进行:

- 在 expert 和 embedding 层中间引入了一层 share 层, share 层先基于 emebdding 学习低阶的共性信息的表征, expert 再基于低阶share 表征开始差异化学习
  - 引入 share 层还有一个好处是降低参数量. 如果 embedding 的维度较高(例如由多种输入的 embedding 表征拼接得到一个大的 embedding), 引入 share 之后, 和 expert 层的链接维度可以降低, 从而降低了 expert 和 gate 层的参数量
- 使用复杂的 Expert 结构. 比如加深层数, 使用比 MLP 复杂的结构
- ([PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236) 的思路)使用 task public expert 和 task specified expert, 以**减轻任务之间拉扯的现象**. 拉扯的原因是 task 之间存在不可调和的冲突, 通过设置 task specified expert, 让不同 task 的独立部分有独立学习的空间(参数), 减少了其它 task 的噪声, 可以缓解拉扯现象

## Gate 改进

- gate 的输出采用 Dropout, **缓解Expert 学习不均衡的问题**
- 在训练的过程中, 适时地停止 gate 的更新
- 调小 gate 的初始化方差, 相当于对 gate 输出的方差进行约束
- 为 gate 中的每一层增加 LN, 约束每个 gate 输出的变化, 减缓 gate 在训练过程中不稳定的问题
- 在所有的 gate 进行 softmax 之前, 增加一个大于 1 的超参 `scale`. 调整输出分布, 结合 softmax 激活, 拉小权重之间的差距

## Loss 加权

联合训练(Joint training)将每个任务的 loss 加权, 合并成一个最终的 loss. 加权的系数会影响最终模型的表现.

不同类型任务(文本分类, NER)的损失往往会有量级的差别, 需要通过权重进行均衡. 调整方法有以下几种.

1. 通过固定的权重, 将所有的 loss 数值大小统一在一个数量级上
2. UWL(Uncertainty to Weigh Losses). 为每个任务分配一个可训练的权重系数, 在训练中调整
   - 实际应用时, 可以将下面公式中的可训练参数取平方使用, 避免参数变为负值(负值的意义是任务效果越差, 最终的 loss 越小, 从而会导致模型没有意义)

```python
loss_final = loss1 * tf.exp(-loss1_var) + loss2 * tf.exp(-loss2_var) + loss1_var + loss2_var
```

![](/resources/images/nn/mmoe-1.png)

# 参考

- [多目标 | 模型结构: MMoE实际应用，改进必不可少](https://zhuanlan.zhihu.com/p/615021892)
- [多目标学习在推荐系统的应用(MMOE/ESMM/PLE)](https://zhuanlan.zhihu.com/p/291406172)
- [MMOE expert权重极化问题如何解决？](https://www.zhihu.com/question/472621284)
