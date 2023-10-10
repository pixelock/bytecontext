# 单标签到多标签

多分类问题指的就是单标签分类问题, 即从 n 个候选类别中选 1 个目标类别. 假设各个类的得分分别为 $$s_1,s_2,\dots,s_n$$, 目标类为 $$t\in\{1,2,\dots,n\}$$, 对应的交叉熵损失函数为:

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}= - s_t + \log \sum\limits_{i=1}^n e^{s_i}
$$

这个loss的优化方向是让目标类的得分 $$s_t$$ 变为 $$s_1,s_2,\dots,s_n$$ 中的最大值.

多标签分类问题, 即从 n 个候选类别中选 k 个目标类别. 这种情况下我们一种朴素的做法是用 sigmoid 激活, 然后变成 n 个二分类问题, 用二分类的交叉熵之和作为 loss.

显然, 当 $$n \gg k$$ 时, 这种做法会面临着**严重的类别不均衡问题**(任何一个二分类问题, 多数情况下标签为 0), 这时候需要一些平衡策略, 比如手动调整正负样本的权重, focal loss 等. 训练完成之后, 还需要根据验证集来进一步确定最优的阈值.

这种做法的 n 选 k 要比 n 选 1 多做很多工作. 但对比来说, n 选 1 这种情况下, 如果使用二分类, 这时的类别是最不平衡的, 反而是最难的. n 选 1 容易的地方, 就在于单标签使用了 softmax + 交叉熵 的形式, 它不会存在类别不平衡的问题. 而多标签分类中的 sigmoid + 交叉熵(二分类的交叉熵) 就存在不平衡的问题.

理想的解决办法, 就是将 softmax + 交叉熵 推广到多标签分类上去.

# 解决方案

## 统一个 loss 形式

换一种形式看单标签分类的交叉熵:

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}=-\log \frac{1}{\sum\limits_{i=1}^n e^{s_i-s_t}}=\log \sum\limits_{i=1}^n e^{s_i-s_t}=\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)
$$

这就是 **logsumexp** 的形式, 而我们知道 **logsumexp 实际上就是 max 的光滑近似**. 因此, 进一步地有:

$$
\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)\approx \max\begin{pmatrix}0 \\ s_1 - s_t \\ \vdots \\ s_{t-1} - s_t \\ s_{t+1} - s_t \\ \vdots \\ s_n - s_t\end{pmatrix}
$$

0 代表是 $$s_t - s_t$$.

这个 loss 的特点是, 所有的非目标类得分 $$\{s_1,\cdots,s_{t-1},s_{t+1},\cdots,s_n\}$$ 跟目标类得分 $$\{s_t\}$$ 两两作差比较, 它们的差的最大值都要尽可能小于零, 所以实现了**目标类得分都大于每个非目标类的得分**的效果.

所以, 假如是有多个目标类的多标签分类场景, 我们也希望**每个目标类得分都不小于每个非目标类的得分**, 多标签分类的损失函数也就可以顺理成章地拓展为:

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)
$$

其中 $$\Omega_{pos},\Omega_{neg}$$ 分别是样本的正负类别集合, 这个 loss 的形式很容易理解, 就是我们希望 $$s_i < s_j$$, 就往 log 里边加入 $$e^{s_i-s_j}$$ 这么一项. 如果补上缩放因子 $$\gamma$$ 和间隔 m, 就得到了Circle Loss论文里边的统一形式:

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{\gamma(s_i-s_j + m)}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{\gamma (s_i + m)}\sum\limits_{j\in\Omega_{pos}} e^{-\gamma s_j}\right)
$$

关于 logsumexp 和 max 关系的推到, 参考:

- [函数光滑化杂谈：不可导函数的可导逼近](https://kexue.fm/archives/6620)
- [寻求一个光滑的最大值函数](https://kexue.fm/archives/3290)

## 用于多标签分类

缩放因子 $$\gamma$$ 和间隔 m, 一般都是度量学习中才会考虑的, 这里不考虑. 如果 n 选 k 中的 k 是固定的, 那么直接使用上面的式子作为 loss 就行了, 然后预测时候直接输出得分最大的 k 个类别.

对于 k 不固定的多标签分类来说, 在预测时, 我们就需要一个阈值来确定输出哪些类. 为此, 在训练过程中, 我们同样引入一个额外的 0 类, 希望目标类的分数都大于 0 类的分数 $$s_0$$, 非目标类的分数都小于 $$s_0$$.

而前面已经已经提到过, 希望 $$s_i < s_j$$, 就往 log 里边加入 $$e^{s_i-s_j}$$ 这么一项, 所以为了能自动地确定阈值, 在训练时损失函数改造成:

$$
\begin{aligned} 
&\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}+\sum\limits_{i\in\Omega_{neg}} e^{s_i-s_0}+\sum\limits_{j\in\Omega_{pos}} e^{s_0-s_j}\right)\\ 
=&\log \left(e^{s_0} + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(e^{-s_0} + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\\ 
\end{aligned}
$$

看等式的左端, 这样定义的 loss 希望满足以下三项:

- $$s_i \lt s_j$$: 每个目标类得分都不小于每个非目标类的得分
- $$s_i \lt s_0$$: 0 类得分不小于每个非目标类的得分
- $$s_0 \lt s_i$$: 每个目标类得分不小于0 类的得分

如果固定阈值为 0, 上式就简化为:

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(1 + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)
$$

这便是最终的 softmax + 交叉熵 在多标签分类任务中的自然简明的推广对应的 loss 了. 在推理时, 输出分数大于 0 的类别对应的标签.

**它没有类别不均衡现象**, 因为它不是将多标签分类变成多个二分类问题, 而是变成目标类别得分与非目标类别得分的两两比较, 并且借助于 logsumexp 的良好性质, 自动平衡了每一项的权重.

# 参考

- [ZLPR: A Novel Loss for Multi-label Classification](https://arxiv.org/abs/2208.02955)
- [将“softmax+交叉熵”推广到多标签分类问题](https://kexue.fm/archives/7359)
