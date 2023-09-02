# RoPE

**总结版本**

通过**绝对位置编码的形式实现相对位置编码**.

以**乘性**(矩阵乘法)的形式作用在 Attention 计算中的 Query, Key 上, 注入绝对位置信息, 这种变换实际上对应着向量的旋转. 从空间上理解, 通过乘性计算作用到每个 token 向量上, 相当于对每个向量进行了旋转, 旋转后向量相乘(attention 计算时)就会得到两个向量的夹角, 这个夹角代表 token 之间的相对位置关系, 结果就是 attention 融合了相对位置信息.

最终的效果可以视为**乘性相对位置编码**的变体.

## 基本思路

RoPE(Rotary Position Embedding, 旋转式位置编码) 的出发点是通过绝对位置编码的方式实现相对位置编码.

**绝对位置编码形式**指的是在 query 和 key 上添加绝对位置信息, 然后 query, key 带着位置信息进行 attention 的计算:

$$
\tilde{\boldsymbol{q}}_m = \boldsymbol{f}(\boldsymbol{q}, m), \quad\tilde{\boldsymbol{k}}_n = \boldsymbol{f}(\boldsymbol{k}, n)
$$

这里的 $$m$$ 和 $$n$$ 表示 $$\boldsymbol{q}$$ 和 $$\boldsymbol{k}$$ 分别对应的位置索引. 注入位置信息的计算方法为 $$\boldsymbol{f}(\cdot, m),\boldsymbol{f}(\cdot, n)$$, 经过该操作后, $$\tilde{\boldsymbol{q}}_m,\tilde{\boldsymbol{k}}_n$$ 分别带有了位置 $$m$$, $$n$$ 的绝对位置信息.

**相对位置编码**指的是, 在內积计算 attention score 时, 希望內积的结果带有两者的相对位置信息, 即满足:

$$
\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n)
$$

## 求解过程

首先假设一些初始条件. 对于初始位置(索引0), 合理地设定: $$\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$$, $$\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$$.

在**复数空间**求解. 在二维情形中, $$\text{Re}[]$$ 代表负数的实部, 在复数中, 有內积运算 $$\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]$$.

因此有:

$$
\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = \text{Re}[\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n)] = g(\boldsymbol{q},\boldsymbol{k},m-n)
$$

简单起见, 这里再次假设存在复数 $$\boldsymbol{g}(\boldsymbol{q},\boldsymbol{k},m-n)$$, 使得 $$\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n) = \boldsymbol{g}(\boldsymbol{q},\boldsymbol{k},m-n)$$. 然后用复数的指数形式, 将实部虚部拆解表示:

$$
\begin{aligned} 
\boldsymbol{f}(\boldsymbol{q}, m) =&\, R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} \\ 
\boldsymbol{f}(\boldsymbol{k}, n) =&\, R_f (\boldsymbol{k}, n)e^{\text{i}\Theta_f(\boldsymbol{k}, n)} \\ 
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n)e^{\text{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)} \\ 
\end{aligned}
$$

实部虚部分别相等, 得到方程组:

$$
\begin{aligned} 
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n) \\ 
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, n) =&\, \Theta_g (\boldsymbol{q}, \boldsymbol{k}, m-n) 
\end{aligned}
$$

对于第一个方程 $$R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, n) = R_g (\boldsymbol{q}, \boldsymbol{k}, m-n)$$, 将 $$m = n$$ 带入, 又根据初始条件的假设 $$\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$$, $$\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$$, 得到:

$$
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, m) = R_g (\boldsymbol{q}, \boldsymbol{k}, 0) = R_f (\boldsymbol{q}, 0) R_f (\boldsymbol{k}, 0) = \Vert \boldsymbol{q}\Vert \Vert \boldsymbol{k}\Vert
$$

所以这里根据形式, 简单地设 $$R_f (\boldsymbol{q}, m)=\Vert \boldsymbol{q}\Vert, R_f (\boldsymbol{k}, m)=\Vert \boldsymbol{k}\Vert$$, 即实部部分不依赖于绝对位置索引 $$m$$.

对于第二个方程, 同样带入 $$m = n$$, 得到:

$$
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, m) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f (\boldsymbol{q}, 0) - \Theta_f (\boldsymbol{k}, 0) =  \Theta (\boldsymbol{q}) - \Theta (\boldsymbol{k})
$$

$$\Theta (\boldsymbol{q}),\Theta (\boldsymbol{k})$$ 是向量 $$\boldsymbol{q},\boldsymbol{k}$$ 本身的**幅角**. 最后一个等号同样来源于起始条件 $$\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$$, $$\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$$. 上式可以进一步推导出:

$$
\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q}) = \Theta_f (\boldsymbol{k}, m) - \Theta (\boldsymbol{k})
$$

所以 $$\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q})$$ 应该是一个只与索引 $$m$$ 相关, 与输入向量无关的函数, 记为 $$\varphi(m)$$, 因此有 $$\Theta_f (\boldsymbol{q}, m) = \Theta (\boldsymbol{q}) + \varphi(m)$$. 带入 $$n=m-1$$, 整理得到:

$$
\varphi(m) - \varphi(m-1) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 1) + \Theta (\boldsymbol{k}) - \Theta (\boldsymbol{q})
$$

等号右边的值在 $$\boldsymbol{q}$$, $$\boldsymbol{k}$$ 确定的情况下是恒定的, 因此 $$\{\varphi(m)\}$$ 是一个**等差序列**. 假设右端的值为 $$\theta$$, 则 $$\varphi(m)=m\theta$$.

至此, 我们可以得到**二维情况**下用复数表示的 RoPE:

$$
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} 
= \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}
$$

根据复数乘法的几何意义, 该变换实际上对应着**向量的旋转**, 因此将这种编码方式称为**旋转式位置编码**. 二维情况下写成矩阵的形式:

$$
\boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}
$$

由于内积满足**线性叠加性**, 因此对于任意偶数维度向量的 RoPE, 都可以表示为二维的拼接:

$$
\scriptsize{\underbrace{\begin{pmatrix} 
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ 
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}
$$

也就是说, 给位置为 $$m$$ 的向量 $$\boldsymbol{q}$$ 乘上矩阵 $$\boldsymbol{\mathcal{R}}_m$$, 位置为 $$n$$ 的向量 $$\boldsymbol{k}$$ 乘上矩阵 $$\boldsymbol{\mathcal{R}}_n$$ 之后, 用变换后的 query 和 key 进行 Attention 计算, 得到的结果中就包含相对位置信息了. 因为以下的恒等式成立:

$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
$$

$$\boldsymbol{\mathcal{R}}_m$$ 是一个正交矩阵, 它不会改变向量的模长. 因此通常来说它不会改变原模型的稳定性.

由于 $$\boldsymbol{\mathcal{R}}_m$$ 的稀疏性, 直接用矩阵乘法来实现会很浪费算力, 推荐通过下述方式来实现RoPE:

$$
\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix}
$$

其中 $$\otimes$$ 是逐位对应相乘, 可以看到, RoPE可以视为是**乘性位置编码**的变体.

# 实现

## ChatGLM

需要特别注意的是二维 position id 的融合方式, 两种不同的 position id 分别与 query 和 key 向量中不同的部分进行融合.

在代码中, `RotaryEmbedding` 负责预先计算 sin 和 cos: `rotate_half` 负责上式第二项中, 互换输入向量的上下部分并取负的操作; `apply_rotary_pos_emb_index` 则是对输入的 query 和 key 注入RoPE.

```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        # 预先计算好上面的theta
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        # learnable的效果并没有更好，通常learnable为False
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision
​
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass
​
    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # 这里使用了爱因斯坦求和约定，该操作就是t和self.inv_freq的外积
            # freqs中保存了所有的m\theta。e.g. 第一行是0\theta、第二行是1\theta
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # 根据上面的公式，每个\theta都需要两份，所以这里将两个freqs拼接起来
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
​
            # [seq_length, 1 (b * np), hn]
            # 计算cos和sin
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            # 缓存结果，方便重复利用
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
​
    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)
​
​
def rotate_half(x):
    # x1是x的前半部分，x2是x的后半部分
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    # 前后互换，且后半部分取负
    return torch.cat((-x2, x1), dim=x1.ndim - 1)
​
@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # cos, sin: (seq_len, 1, hidden_size)
    # q, k: (seq_len, batch_size, hidden_size)
    # position_id: (seq_len, batch_size)
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    # cos, sin: (seq_len, batch_size, 1, hidden_size)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k
```

在 Self Attention 中的用法是:

```python
self.rotary_emb = RotaryEmbedding(
    self.hidden_size // (self.num_attention_heads * 2) if position_encoding_2d else self.hidden_size // self.num_attention_heads,
    base=10000,
    precision=torch.half,
    learnable=False,
)

# 计算cos和sin值
cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
# 输入的 position_ids 为二维位置, 按第二维拆开
position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
    position_ids[:, 1, :].transpose(0, 1).contiguous()
# 将两种位置编码输入到不同的query和key上
q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
```

---

# 参考资料

- [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)
