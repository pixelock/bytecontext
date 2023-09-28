# LlamaTokenizer

## special tokens

LLaMA 中有三个 special token, 分别是:

- `<s>`: 代表 text 的起始, 对应 ID 为 1
- `</s>`: 代表 text 的结束, 对应 ID 为 2
- `<unk>`: 代表 unknown 等, 对应 ID 为 0

LLaMA 使用 `<unk>` 作为 pad token.

## tokenize 过程

### `batch_encode_plus()`

首先调用 `tokenizer.batch_encode_plus()` 方法, 对 batch 样本进行 tokenize 处理. 方法为以下几步:

- 对 batch 内的每条样本, 分别调用 `tokenize()` 以及 `convert_tokens_to_ids()` 得到每一条样本的 input ids
- 调用 `tokenizer._batch_prepare_for_model()` 方法将 batch 内的样本 inputs ids 进行规整

```python
def get_input_ids(text):
    if isinstance(text, str):
        tokens = self.tokenize(text, **kwargs)
        return self.convert_tokens_to_ids(tokens)
    elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
        if is_split_into_words:
            tokens = list(
                itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
            )
            return self.convert_tokens_to_ids(tokens)
        else:
            return self.convert_tokens_to_ids(text)
    elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
        return text
    else:
        raise ValueError(
            "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
        )

input_ids = []
for ids_or_pair_ids in batch_text_or_text_pairs:
    if not isinstance(ids_or_pair_ids, (list, tuple)):
        ids, pair_ids = ids_or_pair_ids, None
    elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
        ids, pair_ids = ids_or_pair_ids, None
    else:
        ids, pair_ids = ids_or_pair_ids

    first_ids = get_input_ids(ids)
    second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
    input_ids.append((first_ids, second_ids))

batch_outputs = self._batch_prepare_for_model(
    input_ids,
    add_special_tokens=add_special_tokens,
    padding_strategy=padding_strategy,
    truncation_strategy=truncation_strategy,
    max_length=max_length,
    stride=stride,
    pad_to_multiple_of=pad_to_multiple_of,
    return_attention_mask=return_attention_mask,
    return_token_type_ids=return_token_type_ids,
    return_overflowing_tokens=return_overflowing_tokens,
    return_special_tokens_mask=return_special_tokens_mask,
    return_length=return_length,
    return_tensors=return_tensors,
    verbose=verbose,
)
```

#### `tokenize()`

在 `tokenize()` 中调用 `_tokenize()` 实现具体逻辑. `LlamaTokenizer._tokenize()` 的代码如下:

```python
def _tokenize(self, text, **kwargs):
    """
    Returns a tokenized string.

    We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
    SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
    `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
    `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
    `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
    """
    tokens = self.sp_model.encode(text, out_type=str)
    if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
        return tokens

    # 1. Encode string + prefix ex: "<unk> Hey"
    tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
    # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
    return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens
```

过程拆解如下.

首先, 使用 `spm.SentencePieceProcessor` 对输入 text 进行 tokenize encode. 分别以中英文作为输入为例, encode 结果为:

```python
tokens = self.sp_model.encode(text, out_type=str)
tokens
```

```python
# 见到你很高兴
['▁', '<0xE8>', '<0xA7>', '<0x81>', '到', '你', '<0xE5>', '<0xBE>', '<0x88>', '高', '兴']
# Nice to meet you.
['▁Nice', '▁to', '▁meet', '▁you', '.']
```

可以看到:

- `SentencePieceProcessor` 会在句子的开始添加, 以及空白替换为特殊符号 `SPIECE_UNDERLINE = "▁"`
- 中文的一个字符, 可能会被拆解成 3 个 tokens, 说明原始的 tokenizer 对中文的编码效率较低

#### `convert_tokens_to_ids()`

在得到 tokens 之后, 调用 `convert_tokens_to_ids()`, 将其转化为对应的 id.

最后会调用到 `tokenizer._convert_token_to_id()` 方法. LlamaTokenizer 中的实现如下:

```python
def _convert_token_to_id(self, token):
    """Converts a token (str) in an id using the vocab."""
    return self.sp_model.piece_to_id(token)
```

上面两个对应的结果为:

```python
# 见到你很高兴
# ['▁', '<0xE8>', '<0xA7>', '<0x81>', '到', '你', '<0xE5>', '<0xBE>', '<0x88>', '高', '兴']
[29871, 235, 170, 132, 30780, 30919, 232, 193, 139, 30528, 31914]

# Nice to meet you.
# ['▁Nice', '▁to', '▁meet', '▁you', '.']
[20103, 304, 5870, 366, 29889]
```

### `_batch_prepare_for_model()`

在这个函数中, 将上面得到的 batch 内的每条样本对应的 input ids 组装成模型接受的 batch 格式.

方法为:

- 首先循环调用 `tokenizer.prepare_for_model()` 方法, 对 batch 内的每条样本进行处理
- 然后调用 `tokenizer.pad()` 获得 token 粒度的 attention_mask

注意**在 tokenize 这一步不进行 padding 操作**. 原因是 tokenizer 只是对数据集中的样本批量处理的 map function, padding 操作是在训练使用时, 由 `DataCollatorForSeq2Seq` 动态进行的.

对应的代码为:

```python
def _batch_prepare_for_model(
    self,
    batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> BatchEncoding:
    """
    Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
    adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
    manages a moving window (with user defined stride) for overflowing tokens

    Args:
        batch_ids_pairs: list of tokenized input ids or input ids pairs
    """

    batch_outputs = {}
    for first_ids, second_ids in batch_ids_pairs:
        outputs = self.prepare_for_model(
            first_ids,
            second_ids,
            add_special_tokens=add_special_tokens,
            padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=None,  # we pad in batch afterward
            return_attention_mask=False,  # we pad in batch afterward
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=None,  # We convert the whole batch to tensors at the end
            prepend_batch_axis=False,
            verbose=verbose,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    batch_outputs = self.pad(
        batch_outputs,
        padding=padding_strategy.value,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
    )

    batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

    return batch_outputs
```

#### `prepare_for_model()`

在 `prepare_for_model()` 方法中, 对每条样本进行处理. 主要是使用 `build_inputs_with_special_tokens()` 方法, 对输入拼接特殊字符, 符合模型需要的输入格式.

按以下的形式拼接特殊字符:

- 如果样本中只有一条 text, 则在句子的开头拼接一个起始特殊符号: `<s>` + `token_ids_0`
- 如果样本中包含两条 text, 则在两个 text 之间再拼接一个起始特殊符号: `<s>` + `token_ids_0``<s>` + `token_ids_1`

```python
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    bos_token_id = [self.bos_token_id] if self.add_bos_token else []
    eos_token_id = [self.eos_token_id] if self.add_eos_token else []

    output = bos_token_id + token_ids_0 + eos_token_id

    if token_ids_1 is not None:
        output = output + bos_token_id + token_ids_1 + eos_token_id

    return output
```

上面两个对应的结果为:

```python
# 见到你很高兴
# [29871, 235, 170, 132, 30780, 30919, 232, 193, 139, 30528, 31914]
[1, 29871, 235, 170, 132, 30780, 30919, 232, 193, 139, 30528, 31914]

# Nice to meet you.
# [20103, 304, 5870, 366, 29889]
[1, 20103, 304, 5870, 366, 29889]
```

#### `pad()`

`pad()` 函数实际上没有对 batch 进行 padding 操作, 而是对每条样本, 分别生成对应的 attention_mask.

```python
# Initialize attention mask if not present.
if return_attention_mask and "attention_mask" not in encoded_inputs:
    encoded_inputs["attention_mask"] = [1] * len(required_input)
```

# LlamaModel

先看 LLaMA 的组件.

## LlamaRMSNorm

RMSNorm 与 Layer Normalization 相比, 去掉了 shift 相关的参数:

$$
\bar{x}_i=\frac{a_i}{\operatorname{RMS}(\mathbf{x})} g_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}
$$

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

## RoPE

RoPE 的原理参考: [RoPE](/docs/nlp/models/transformers/position-encoding/rope.md).

LLaMA 使用 RoPE 位置编码. 代码中提供了三种实现:

- `LlamaRotaryEmbedding`: 标准的 RoPE 实现
- `LlamaLinearScalingRotaryEmbedding`: 线性内插版本, 提升模型支持的 context length. 对应 Paper: [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
- `LlamaDynamicNTKScalingRotaryEmbedding`: 高频外推, 低频内插. [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

### LlamaRotaryEmbedding

根据 max sequence length 生成对应长度的 sin cos 基数 cache 张量, 并通过 `register_buffer` 存储. 当输入长度超过预先设置的 max sequence length, 根据输入长度再次生成对应长度的 sin cos cache 并存储.

```python
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
```

### LlamaLinearScalingRotaryEmbedding

线性内插方法扩大模型支持的 context length, 需要配合少量样本的微调工作.

实现方式上, 首先指定扩大的比例 `scaling_factor`, 然后通过缩放, 将超过 `max_position_embeddings` 线性的缩放, 将 `[0, max_position_embeddings * scaling_factor]` 范围缩小到 `[0, max_position_embeddings]`.

```python
self.max_seq_len_cached = seq_len
t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
t = t / self.scaling_factor
```

整体代码如下:

```python
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
```

### LlamaDynamicNTKScalingRotaryEmbedding

[NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) 的原理是**高频外推, 低频内插**, 具体可以参考: [RoPE是一种β进制编码](https://kexue.fm/archives/9675#%E8%BF%BD%E6%A0%B9%E6%BA%AF%E6%BA%90).

高低频指的是每个位置的旋转向量依赖的角度向量 $$\theta$$ (代码中对应的 `inv_freq`), 角度大的代表低频, 角度小代表高频. 对高低频进行不同的调节.

```python
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
```

### apply on Q and K

上面得到的 cos 和 sin 的 cache, 前向传播将 cos 和 sin 应用在 query 和 key 上的代码如下:

逻辑如下:

- 根据 KV cache 的长度, 截取需要的 sin 和 cos cache, 得到 `cos` 和 `sin` 变量
- 在 `apply_rotary_pos_emb` 函数中注入 RoPE 位置信息

```python
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## LlamaMLP

即 Feed-forward 层实现. LLaMA 使用了 **SwiGLU** 激活函数:

由于使用到 gate 机制, 所以 `LlamaMLP` 中使用了 3 个 Linear 层.

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 第一个映射层使用的 gate 参数
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 第一个映射层的参数
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 第二个映射层的参数
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            # 以下的计算, 等价于
            # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
```

## LlamaAttention

LLaMA 使用的是 Multi-headed attention. 需要定义 q, k, v, o 对应的参数.

前向传播的逻辑如下:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # hidden_states: (bsz, q_len, hidden_size)
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        # 这里是 tensor parallel 的逻辑, 将 tensor 切分成 pretraining_tp 份, 然后并行计算
        # 但这里的并行计算并没有分配到不同的卡上...
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        # query_states: (bsz, q_len, num_heads * head_dim)
        query_states = self.q_proj(hidden_states)
        # key_states: (bsz, q_len, num_key_value_heads * head_dim)
        key_states = self.k_proj(hidden_states)
        # value_states: (bsz, q_len, num_key_value_heads * head_dim)
        value_states = self.v_proj(hidden_states)

    # query_states: (bsz, num_heads, q_len, head_dim)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states: (bsz, num_key_value_heads, q_len, head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states: (bsz, num_key_value_heads, q_len, head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 应用 RoPE
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # 更新 kv cache
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # 使用 Grouped Query Attention, KV 对应的 heads 数量为 num_key_value_heads, Q 对应的 heads 数量为 num_heads
    # num_key_value_heads < num_heads
    # 每一组 KV heads group 对应的 Q heads 为 num_key_value_groups
    # num_key_value_groups = num_heads // num_key_value_heads
    # 计算 attention 之前, 需要将 KV cache 重复 num_key_value_groups 次, 保持 Q 和 KV 的对齐
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    # repeat 之后的 KV cache 的形状为
    # key_states: (bsz, num_heads, q_len, head_dim)
    # value_states: (bsz, num_heads, q_len, head_dim)

    # matmul 计算 attention, 并归一化
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    # 应用 attention mask
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # 为了保持精确性, 计算 softmax attention score 时, 需要 upcast attention to fp32, 以保持高精度
    # attn_weights: (bsz, num_heads, q_len, q_len)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # 结合 V, 计算 attention 的输出
    # attn_output: (bsz, num_heads, q_len, head_dim)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # attn_output: (bsz, q_len, num_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output: (bsz, q_len, hidden_size)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        # tensor parallel
        # 对 attention 结果 tensor 分片, 分别与对应的 output weight 计算得到最后的输出结果
        # sum 聚合各个 tensor split 的结果
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        # attn_output: (bsz, q_len, hidden_size)
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

## LlamaDecoderLayer

定义了 LLaMA 的一层, 由 LlamaMLP, LlamaAttention 以及两个 LlamaRMSNorm 构成.

LLaMA 使用的是 **Pre-Norm**, 因此 LlamaDecoderLayer 输入和输出之间保持了一个通路. Norm 在进入到 MLP 和 Attention 之前进行, 残差的加和计算在 MLP 和 Attention 计算完之后立即进行.

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
```

## LlamaModel

### 定义

`LlamaModel` 的定义为 `Embedding` 层 + num_hidden_layers * `LlamaDecoderLayer` 层 + 最后输出的 `LlamaRMSNorm` 层. 即在输出之前, 还需要进行进行一次额外的 Norm 平滑.

```python
def __init__(self, config: LlamaConfig):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
    self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()
```

### 构建 attention mask

创建 Causal attention mask 由 `LlamaModel._prepare_decoder_attention_mask()` 方法完成.

attention_mask 的构建分为两步.

#### `_make_causal_mask()`

第一步是构建下三角形式的 causal mask, 通过 `_make_causal_mask()` 函数实现.

如果是 decoding 阶段(推理过程中逐个生成 token), 还有 KV cache 以 `past_key_values_length` 参数输入, 这部分是每个 token 都能看到的. 因此还需要在下三角的 mask 之前, 拼接上大小为 `(tgt_len, past_key_values_length)` 的全通矩阵, 最终形成大小为 `(tgt_len, tgt_len + past_key_values_length)` 的 mask 矩阵, 并调整到 `(bsz, 1, tgt_len, tgt_len + past_key_values_length)` 符合模型计算 attention 的格式输出.

其中 `0` 代表没有被 mask, 被 mask 的位置为一个超大值.

```python
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
```

#### `_expand_mask()`

这部分 mask 是为了屏蔽输入中的 pad token, 避免这些 token 影响到 attention 的计算. 这在所有 Attention 结构的模型中都会使用, 而非 Causal LM 特有.

输入的 mask 形状为 `(bsz, src_len)`, 其中为 1 的位置代表有效 token, 无效 token 对应的值为 0.

将输入 mask 拓展为 `(bsz, 1, tgt_len, src_len)`.

- 在 prefill 阶段, `tgt_len = src_len`
- 在 decoding 阶段, `tgt_len = 1, src_len = past_key_values_length + tgt_len`

同样地, `0` 代表没有被 mask, 被 mask 的位置为一个超大值.

```python
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
```

#### merge mask

两种 mask 融合.

```python
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask
```

### 前向传播

前向传播中, 如果没有传入 `position_ids`, 会动态地构建得到 `position_ids`. 构建的方法为生成一个从 `past_key_values_length` 到 `past_key_values_length + seq_length` 逐个递增的 id 序列. 这是考虑到在 decoding 阶段的输入长度 `seq_length` 只有 1, position id 的生成需要考虑 KV cache 的长度, 以还原当前 token 在序列中真正的位置.

另外经过所有的 decoder layer 之后, 还需要通过一个额外的 RMSNorm, 得到最后的输出.

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            # 更新 KV cache
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # 额外的 Norm
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
```

# LlamaForCausalLM

用于做生成的模型. 在 `LlamaModel` 的基础上, 增加了 `lm_head` 结构, 将 `LlamaModel` 输出的 `last_hidden_state` 映射为每个 token 上的概率, 以供抽样生成后续的文本.

`lm_head` 是一个形状为 `(hidden_size, vocab_size)` 的 Linear. LLaMA 中, lm_head 是一个独立层, **没有与 Embedding 共享参数**.

```python
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            # tensor parallel 将 lm_head 分片, 与完整的 hidden_states 计算, 最后拼接在一起, 得到所有 token 的 logits
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            # CrossEntropyLoss
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```

## `prepare_inputs_for_generation()`

`prepare_inputs_for_generation()` 是为生成阶段准备模型的输入.

1. 对于 decoding 阶段, 只取最后一个 token. 前的 tokens 的 KV 表征已经存储在 KV Cache 中(past_key_values), 没有必要重复计算.

```python
if past_key_values:
    # decoding 阶段, 只取最后一个 token
    # 之前的 tokens 的 KV 表征已经存储在 KV Cache 中(past_key_values), 没有必要重复计算
    input_ids = input_ids[:, -1:]
```

2. 如果输入中没有 position_ids, 在这里生成. 由于 Causal LM 采用的都是 `left padding` 方式, 所有如果有 padding, padding token 汇聚在左侧, 对应的 `attention_mask` 参数的左边为 0, 右边为 1. 用 `cumsum() - 1` 函数可以得到有效 token 对应的 position id. 对于 padding token, 对应的 `position_ids` 统一为 `1`. 对于 decoding 阶段, 由于输入的 `input_ids` 截断为最后一个 token, 所以这里的 `position_ids` 也截断为最后一个位置.

```python
position_ids = kwargs.get("position_ids", None)
if attention_mask is not None and position_ids is None:
    # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:
        position_ids = position_ids[:, -1].unsqueeze(-1)
```

完整的逻辑如下:

```python
def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        # decoding 阶段, 只取最后一个 token
        # 之前的 tokens 的 KV 表征已经存储在 KV Cache 中(past_key_values), 没有必要重复计算
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
```

# LlamaForSequenceClassification

用 LLaMA 做分类推断任务, 我们就不在需要 `lm_head` 层, 而是使用一个形状为 `(hidden_size, num_labels)` 的 Linear 层作为输出层替代.

在输出上, 取 `LlamaModel` 最后一层的输出进行输出层转换, 得到的结果 logits 形状为 `(bsz, seq_length, num_labels)`.

然后取最后一个有效 token 对应的 logits, 作为对整句的表征, 对应的代码为:

```python
if input_ids is not None:
    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
        logits.device
    )
else:
    sequence_lengths = -1

pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
```

然后使用这个 `pooled_logits` 计算 loss.

```python
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```
