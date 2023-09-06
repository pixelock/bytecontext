LLaMA 是一系列从 7B 到 65B 参数的**基础语言模型**, 而非 Chat 类模型. 下面来探究下 Llama 是怎么训练出来的.

# 预训练数据

LLaMa 预训练数据大约包含 1.4T tokens, 对于绝大部分的训练数据在训练期间模型只见到过1次, Wikipedia 和 Books 这两个数据集见过2次. 所有使用的数据集如下:

- English CommonCrawl(67%): 在行级别进行数据去重; 使用 fastText 线性分类器进行语言识别, 以删除非英语页面; 使用 n-gram 语言模型过滤低质量内容; 训练了一个线性模型, 用于将页面分类为 Wikipedia 中的引用页面与随机抽样页面, 并丢弃未被分类为引用的页面
- C4(15%): 依赖于标点符号的存在或网页中的词语和句子数量等启发式方法, 进行质量过滤
- Github(4.5%): 使用 Google BigQuery 上可用的公共 GitHub 数据集; 使用基于行长度或字母数字字符比例的启发式方法过滤低质量文件; 使用正则表达式删除了诸如头文件之类的样板文件; 使用完全匹配的方法, 对生成的数据集进行了文件级别的去重
- Wikipedia(4.5%): 涵盖20种语言, 处理数据以去除超链接, 评论和其他格式样板
- Gutenberg and Books3(4.5%): 添加了两个书的数据集, 分别是 Gutenberg 以及 ThePile 中的 Book3 部分. 处理数据时执行重复数据删除, 删除内容重叠超过 90% 的书籍
- ArXiv(2.5%): 处理了arXiv Latex文件, 以添加科学数据到数据集中
- Stack Exchange(2%): 添加了 Stack Exchange, 这是一个涵盖各种领域的高质量问题和答案网站, 范围从计算机科学到化学; 从文本中删除 HTML 标签并按分数对答案进行排序

# Tokenizer

LLaMa Tokenizer 使用字节对编码(BPE)算法对数据进行分词, 使用 SentencePiece 的实现. 值得注意的是, 作者将所有数字分割成单个数字.

```python
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
```

# 模型结构

## Pre-LN + RMSNorm

为了提高训练稳定性, Llama 使用了 Pre-LN. Pre-LN 结构在前向传播中, 有一条直通的恒等路径, LN 是加在 Attention 和 FFN 结构之前的.

LLaMa 使用了 **RMSNorm** 归一化函数.

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

RMSNorm 与 Layer Normalization 相比, 去掉了 shift 相关的参数:

$$
\bar{x}_i=\frac{a_i}{\operatorname{RMS}(\mathbf{x})} g_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}
$$

## SwiGLU 激活函数

LLaMa 使用 SwiGLU 激活函数替换 GeLU 以提高性能. 它结合了 SWISH 和 GLU 两种者的特点, SwiGLU 主要是为了提升 Transformer 中的 FFN 层的实现.

$$
\operatorname{SwiGLU}(\boldsymbol{x}, \boldsymbol{W}, \boldsymbol{V}, b, c, \beta)=\operatorname{Swish}_\beta(\boldsymbol{x} \boldsymbol{W}+b) \otimes(\boldsymbol{x} \boldsymbol{V}+c)
$$

## RoPE

LLaMa 使用了旋转位置编码(RoPE), 使用绝对位置编码的性质, 带来相对位置编码的做作用. 可以提升模型的外推性.

## Attention 提效

LLaMa 采用了高效的 causal multi-head attention, 不存储注意力权重, 且不计算 mask 掉的 query 和 key 的值, 降低显存消耗.

# 训练过程

LaMa 使用了 AdamW 优化器进行训练，超参数为：β1 = 0.9，β2 = 0.95. 使用 cosine 学习率衰减策略, 2000 步的 warm-up, 最终学习率等于最大学习率的 10%. 使用 **0.1 的权重衰减**和 1.0 的梯度裁剪.

不同参数规模的对应的模型结构如下:

| 参数 | 维度(hidden size) | head个数 | layer层数 | 学习率 | batch size | token数 |
| --- | --- | --- | --- | --- | --- | --- |
| 6.7B | 4096 | 32 | 32 | 3.0e−4 | 4M | 1.0T |
| 13.0B | 5120 | 40 | 40 | 3.0e−4 | 4M | 1.0T |
| 32.5B | 6656 | 52 | 60 | 1.5e−4 | 4M | 1.4T |
| 65.2B | 8192 | 64 | 80 | 1.5e−4 | 4M | 1.4T |

在训练一个包含 65B 参数的模型时, LLaMa 的代码在具有 80GB 内存的 2048 个 A100 GPU 上每秒处理约 380 个 token, 这意味着在包含 1.4万亿 token 的数据集上进行训练大约需要21天.
