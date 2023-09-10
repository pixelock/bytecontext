# 优化点

相比 ChatGLM, [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)做了如下的改进

- 更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练。对于更长的上下文，我们发布了 ChatGLM2-6B-32K 模型。LongBench 的测评结果表明，在等量级的开源模型中，ChatGLM2-6B-32K 有着较为明显的竞争优势

- 更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K

## ChatGLM2 与 ChatGLM 的区别

- Tokenizer 上使用的都是 SentencePiece, 但经过了重新训练, 相同的输入划分结果不同, vocab_size 大幅缩小
- 使用的 special tokens 不同
- 样本拼接 special tokens 的方法不同
- ChatGLM2 中的 position id 不再使用二维位置编码, 而是使用如同 Llama 的从0开始顺序+1的方式
  - 配合将 `[gMASK]` 放在序列的起始位置, 相当于只使用了上一版生成部分顺序递增那部分的 position id, 以适配训练方式产生的模型
- Attention 使用 Multi-Query Attention 技术(实际用的 Group-Query Attention)
- ChatGLM2 中, 最后的输出层没有与 token embedding 共享权重, 这点与 ChatGLM 是不同的
- 在计算RoPE时, query 和 key 只有一半的参数融合了位置信息, 另一半没有融合
- 使用 RMSNorm 代替 LayerNorm
- Attention 计算使用了 FlashAttention 实现.  具体来说使用的算子是 `torch.nn.functional.scaled_dot_product_attention()`
- FFN中使用了 SwiGLU 代替了 ChatGLM 中的 GELU 激活函数

# 代码解析

## Tokenizer

### 将文本 tokenize

ChatGLM2 使用 `ChatGLMTokenizer` 类对输入进行 tokenize. 与 ChatGLM 相同, 在 `self._tokenize()` 中, 首先对当前 prompt 进行 preprocess, 然后再使用 `sp_tokenizer` 进行 tokenize:

```python
tokenized_text.extend(self._tokenize(token))

# self.tokenizer = SPTokenizer(vocab_file)
def _tokenize(self, text, **kwargs):
    return self.tokenizer.tokenize(text)

# self.sp_model = SentencePieceProcessor(model_file=model_path)
# SPTokenizer 中的 tokenize 方法调用 SentencePiece 对文本进行 tokenize
def tokenize(self, s: str):
    return self.sp_model.EncodeAsPieces(s)
```

与 ChatGLM 相比, ChatGLM2 虽然也是用 `SentencePiece`, 但使用的模型不同. 例如对 `你和ChatGPT相比怎么样` 这句话进行 tokenizer, 得到的结果分别为:

```
ChatGLM:  ['▁你', '和', 'Chat', 'GPT', '相比', '怎么样']
ChatGLM2: ['▁你', '和', 'C', 'hat', 'G', 'PT', '相比', '怎么样']
```

### 得到 token ids

调用 `ChatGLMTokenizer._convert_token_to_id()` 方法, 将每个 token 转化成 id.

ChatGLM2 使用以下几种 special token:

- `[MASK]`: 64789
- `[gMASK]`: 64790
- `[sMASK]`: 64791
- `sop`: 64792
- `eop`: 64793

```python
# self.tokenizer = SPTokenizer(vocab_file)
def _convert_token_to_id(self, token):
    """ Converts a token (str) in an id using the vocab. """
    return self.tokenizer.convert_token_to_id(token)

def convert_token_to_id(self, token):
    """ Converts a token (str) in an id using the vocab. """
    if token in self.special_tokens:
        return self.special_tokens[token]
    return self.sp_model.PieceToId(token)
```

### 添加 special token

将 inputs_ids 输入到模型之前, 在 `tokenizer.prepare_for_model()` 方法中, 还需要对 `token_ids` 序列添加 special token:

对于一个 batch 内的输入, 循环地对 batch 内的每一条文本, 做添加.

首先调用 `num_special_tokens_to_add` 确定这个 `token_ids` 在添加完 special token 后, 长度是多少.

- 如果输入只有一句, 会在句子的前面添加 `[gMASK]` 和 `sop`, 因此序列最终的长度为 `token_ids` + 2
- 如果输入有两句, 除了在句子的前面添加 `[gMASK]` 和 `sop`, 还要在句子的最后添加 `<eos>`, 因此序列最终的长度为 `token_ids` + 3

```python
total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

def num_special_tokens_to_add(self, pair: bool = False) -> int:
    token_ids_0 = []
    token_ids_1 = []
    return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
) -> List[int]:
    prefix_tokens = self.get_prefix_tokens()
    token_ids_0 = prefix_tokens + token_ids_0
    if token_ids_1 is not None:
        token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
    return token_ids_0

def get_prefix_tokens(self):
    prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
    return prefix_tokens
```

再调用 `build_inputs_with_special_tokens()` 方法, 为 `token_ids` 在添加 special token.

```python
sequence = self.build_inputs_with_special_tokens(ids, pair_ids)

def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
) -> List[int]:
    prefix_tokens = self.get_prefix_tokens()
    token_ids_0 = prefix_tokens + token_ids_0
    if token_ids_1 is not None:
        token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
    return token_ids_0
```

### 生成 attention_mask 和 position_ids

这里的 attention_mask 是一维的 mask, 即代表 `input_ids` 中每个 token 是否是 pad token.

而 position_ids 与 **ChatGLM 完全不同**, 不再使用二维位置编码, 而是使用一维位置编码, 与 Casual LM 中的 position_ids 的形式一致, 即`list(range(seq_length))`.

```python
batch_outputs = self.pad(
    batch_outputs,
    padding=padding_strategy.value,
    max_length=max_length,
    pad_to_multiple_of=pad_to_multiple_of,
    return_attention_mask=return_attention_mask,
)
```

也是对 batch 内的样本一条条进行的:

```python
batch_outputs = {}
for i in range(batch_size):
    inputs = {k: v[i] for k, v in encoded_inputs.items()}
    outputs = self._pad(
        inputs,
        max_length=max_length,
        padding_strategy=padding_strategy,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
    )

    for key, value in outputs.items():
        if key not in batch_outputs:
            batch_outputs[key] = []
        batch_outputs[key].append(value)
```

ChatGLM2 也是在**左侧**进行 padding

```python
def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
) -> dict:
    # Load from model defaults
    assert self.padding_side == "left"

    required_input = encoded_inputs[self.model_input_names[0]]
    seq_length = len(required_input)

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)

    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

    # Initialize attention mask if not present.
    if "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * seq_length

    if "position_ids" not in encoded_inputs:
        encoded_inputs["position_ids"] = list(range(seq_length))

    if needs_to_be_padded:
        difference = max_length - len(required_input)

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
        if "position_ids" in encoded_inputs:
            encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
        encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

    return encoded_inputs
```

## ChatGLMForConditionalGeneration

`ChatGLMForConditionalGeneration` 是用来做**条件生成**的模型, 即给定一段输入, 根据输入做继续的生成. 提供了 `chat()` 和 `stream_chat()` 两种方法进行生成.

### `build_inputs()`

使用 `chat()` 进行生成, 首先会调用 `build_inputs()` 方法, 将输入包装成特定的形式.

```python
def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
    prompt = tokenizer.build_prompt(query, history=history)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(self.device)
    return inputs
```

其中 `tokenizer.build_prompt()` 的逻辑如下:

```python
def build_prompt(self, query, history=None):
    if history is None:
        history = []
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    return prompt
```

即:

- 使用 `"[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)` 这种一问一答的形式, 将当前的 query 包装起来引导生成
- 对于历史多伦对话的情况, 按照每一轮生成一问一答的形式拼接起来, 并拼接到当前 query 之前

最后得到包含 `input_ids`, `attention_mask`, `position_ids` 的输入 `inputs`.

### `build_stream_inputs()`

使用 `stream_chat()` 进行生成, 在生成第一个 token 之后(之前所有 token 的表征都已经转换成 KV cache, 即 `past_key_values`), 会调用 `build_stream_inputs()` 方法, 将输入包装成特定的形式. 组织的逻辑同上

```python
def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
    if history:
        prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = input_ids[1:]
        inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
    else:
        prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(self.device)
    return inputs
```

最后得到包含 `input_ids`, `attention_mask`, `position_ids` 的输入 `inputs`.

### `prepare_inputs_for_generation()`

在 batch 进入到模型进行推理之前, 还需要进行一些额外的处理, 即 `prepare_inputs_for_generation()` 方法, 其逻辑如下:

- 输入中如果没有 `position_ids`, 则调用 `get_position_ids()` 方法生成
- 如果不是生成第一个 token, 则说明已经进行过推理了, 也就是除了最后一个字之外, 其他所有 token 的表征已经确定(CLM 模型特点决定), 且每个 token 的表征也已经转换成 KV cache, 保存在 `past_key_values` 中了
  - 所以本次推理的输入, 只需要输入最后一个 token(也就是上次推理生成的 token)
  - position_ids 也只截取最后一位 token 对应的位置即可

```python
def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_first_forward: bool = True,
        **kwargs
) -> dict:
    # only last token for input_ids if past is not None
    if position_ids is None:
        position_ids = self.get_position_ids(input_ids, device=input_ids.device)
    if not is_first_forward:
        position_ids = position_ids[..., -1:]
        input_ids = input_ids[:, -1:]
    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "return_last_logit": True
    }
```

### `forward()`

模型推理逻辑.

```python
def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 得到模型的输出
    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        # 如果是生成第一个 token, past_key_values 的值为 None
        # 否则为每一层, 每一个 token 对应的表征
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    # 取最后一层的输出序列, 形状为 (seq_length, batch_size, hidden_size)
    hidden_states = transformer_outputs[0]
    if return_last_logit:
        # 新生成的 token 对应的 hidden_states, 形状为 (1, batch_size, hidden_size)
        hidden_states = hidden_states[-1:]
    # 得到 tokens 的 logits, 形状为:
    # 训练: (seq_length, batch_size, vocab_size)
    # 推理: (1, batch_size, vocab_size)
    lm_logits = self.transformer.output_layer(hidden_states)
    # (seq_length, batch_size, vocab_size) / (1, batch_size, vocab_size)
    lm_logits = lm_logits.transpose(0, 1).contiguous()

    loss = None
    # 如果有传标签, 也就是在进行训练
    if labels is not None:
        lm_logits = lm_logits.to(torch.float32)

        # teacher force 训练时, logits 与 labels 错位对齐
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # Flatten the tokens
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        lm_logits = lm_logits.to(hidden_states.dtype)
        loss = loss.to(hidden_states.dtype)

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
```

### `chat()` 与 `stream_chat()`

`chat()` 与 `stream_chat()` 可以类比 ChatGLM 中对应的逻辑, 参考: [ChatGLM 推理过程](/docs/llm/开源大模型/chatglm/chatglm推理过程.md).

## ChatGLMModel

`ChatGLMModel` 定义了模型结构. 这里定义了三部分:

- RotaryEmbedding: RoPE 位置编码
- GLMTransformer: Transformer 主体结构
- output linear: 大小为 `(hidden_size, padded_vocab_size)` 的输出层

如果使用 P-tuning 进行微调, 这里还定义了 `PrefixEncoder`.

还有一个重要的点: **ChatGLM2 中, 最后的输出层没有与 token embedding 共享权重, 这点与 ChatGLM 是不同的**.

```python
class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
                                              dtype=config.torch_dtype)
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)
```

正向传播的过程如下:

```python
def forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    # 将 input_ids 转换为 embedding
    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    # 如果使用 P-tuning 进行微调, 获取 trainable prompt parameters
    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                dtype=inputs_embeds.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                        attention_mask], dim=-1)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

    # 根据序列长度, 获取到相应的 position embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # 核心推理
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
```

### GLMTransformer

Transformer 计算的主体定义在 `GLMTransformer`, 它包含以下的结构:

- 多层 `GLMBlock`
- 最后输出再使用 RMSNorm 进行一次归一化

```python
class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False
```

正向传播过程如下.

```python
def forward(
        self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
):
    if not kv_caches:
        kv_caches = [None for _ in range(self.num_layers)]
    presents = () if use_cache else None
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
        if self.gradient_checkpointing and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(
                layer,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_caches[index],
                use_cache
            )
        else:
            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches[index],
                use_cache=use_cache
            )
        # hidden_states: (seq_length, batch_size, hidden_size)
        # kv_cache: 包含两个分别代表 K, V 的形状为 (seq_length, batch_size, multi_query_group_num, hidden_size) 张量
        hidden_states, kv_cache = layer_ret
        if use_cache:
            presents = presents + (kv_cache,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # Final layer norm.
    if self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states, presents, all_hidden_states, all_self_attentions
```

#### GLMBlock

接下来是每一层的定义. LN 使用的是 RMSNorm. 模型中除了 LN, 分为两部分:

- SelfAttention
- MLP, 即 FFN

ChatGLM2 依然使用的是 DeepNorm, 本质上还是一种 Post-LN, 通过**更改初始化**以及**调整残差系数**, 稳定传播的稳定性. 实验证明可以做到千层 Post-LN 结构的稳定训练.

$$
\text{DeepNorm}(x) = \text{LayerNorm}(\alpha x + g(x)), \alpha > 1
$$

在每一层的输出为 `output = residual + output`, 在下一层的开始, 使用 `layernorm_output = self.input_layernorm(hidden_states)` 对上一层的 残差 + FFN 输出进行 DeepNorm. 同理, 每一层中也会对 残差 + Attention输出 进行 DeepNorm.

```python
class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP
        self.mlp = MLP(config, device=device)
    
    def forward(
        self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache
```

##### RMSNorm

RMSNorm 与 Layer Normalization 相比, 去掉了 shift 相关的参数:

$$
\bar{x}_i=\frac{a_i}{\operatorname{RMS}(\mathbf{x})} g_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}
$$


```python
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)
```

##### SelfAttention

ChatGLM2 使用的 **Multi-Query Attention**, 实际上用的是 Group-Query Attention. 需要注意几个大小参数:

- projection_size: 输入的 hidden_size, 6B为4096
- hidden_size_per_attention_head: projection_size // num_attention_heads, 6B为128
- num_attention_heads_per_partition: num_attention_heads, 6B为32
- num_multi_query_groups_per_partition: multi_query_group_num, 6B为2
- qkv_hidden_size: self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num, 6B为4608
  - Q, K, V 堆叠在一起的向量大小
  - Query 还是需要分在每个 head 中计算, 因此这一部分的总维度为 `projection_size`
  - Key 和 Value 在所有的 head 中使用相同的表征(Multi-Query Attention), 或者在同一 group 内使用相同的表征(Group-Query Attention). ChatGLM2 使用的是 Group-Query Attention, 所以这部分的维度是 `2 * hidden_size_per_attention_head * multi_query_group_num`. 乘以2是因为 K, V 都需要转化

```python
class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # b: batch_size
        # sq: sequence_length
        # h: hidden_size
        # np: num_attention_heads_per_partition
        # hn: hidden_size_per_attention_head
        # gn: num_multi_query_groups_per_partition

        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads
        # multi-attention: [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # multi-query: [sq, b, h] --> [sq, b, np * hn + 2 * hn]
        # group-query: [sq, b, h] --> [sq, b, np * hn + 2 * gn * hn]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            # 使用 group-query
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,  # Query size
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,  # Key size
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,  # Value size
                ],
                dim=-1,
            )
            # [sq, b, np * hn] -> [sq, b, np, hn]
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            # [sq, b, gn * hn] -> [sq, b, gn, hn]
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            # [sq, b, gn * hn] -> [sq, b, gn, hn]
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            # 使用 multi-attention
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # 拼接 KV Cache
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)  # [sq, b, gn, 1, hn]
            # 将所有 heads 按 group num 平均分配, 每个组与对应的 head 进行 attention 计算
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )  # [sq, b, gn, np / gn, hn]
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )  # [sq, b, np, hn]
            value_layer = value_layer.unsqueeze(-2)  # [sq, b, gn, 1, hn]
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )  # [sq, b, gn, np / gn, hn]
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )  # [sq, b, np, hn]

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache
```

##### CoreAttention

ChatGLM2 中的 attention 计算, 以及加权汇总 values 得到 scaled_dot_product_attention, 是通过 `torch.nn.functional.scaled_dot_product_attention()` 算子实现的. 算子的详细内容参考: [torch.nn.functional.scaled_dot_product_attention](/docs/framework/pytorch/operator/transformer/scaled_dot_product_attention).

```python
class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # q, k, v: [sq, b, np, hn]
        pytorch_major_version = int(torch.__version__.split('.')[0])
        if pytorch_major_version >= 2:
            # q, k, v: [b, np, sq, hn]
            # 在 Pytorch2 中, 使用 torch.nn.functional.scaled_dot_product_attention 进行 attention 以及之后的 value 加权计算
            # 以得到更快的速度
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            # [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            # [sq, b, h]
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
```

##### MLP

ChatGLM2 的 FFN 将输入映射到 4h 维度, 然后再还原. 使用的激活函数为 SWIGLU.

$$
\text{MLP}(X) = \text{SWIGLU}(XW_1)W_2
$$

SWIGLU 的实现如下:

```python
def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)  # 将tensor按dim分割成chunk_num个tensor块
    return F.silu(x[0]) * x[1]
```

```python
class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output
```

### RotaryEmbedding

ChatGLM2 不再使用 ChatGLM 中的二维位置编码, 而是使用一维位置编码, 编码的形式就是 Causal LM 中使用的从零递增的 position ids. 实现上不再需要 `position_id` 参数作为输入, 而是在生成 cos 和 sin 的 cache 张量时已经生成了每个 index 对应的中间值, 然后使用`apply_rotary_pos_emb` 直接将 `cache[:seq_length]` 应用到每个 token 上.

RoPE 从定义到应用在 query 和 key 上, 有以下的几个步骤.

首先定义 `RotaryEmbedding`.

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
```

在 `ChatGLMModel` 定义模型时, 使用如下的参数初始化 `RotaryEmbedding`. 对应的 position embedding 大小为**每个 head 中的 hidden size 的一半**. 在 6B 中的大小为 64.

```python
# hidden_size: 4096
# num_attention_heads: 32
# rotary_dim: 128
rotary_dim = (
    config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
)
# rotary embedding dim: 64
self.rotary_pos_emb = RotaryEmbedding(
    rotary_dim // 2,
    original_impl=config.original_rope,
    device=device,
    dtype=config.torch_dtype
)
```

然后根据 max sequence length(6B为8192), 生成计算 position embedding 所需的值. 实现如下(其中的数值以 6B 模型为例).

首先, 计算单个向量中, 每个位置对应的基数, 即公式中的每个 $$\theta_i$$. 将所有的 $$\theta_i$$ 按顺序拼接成向量 $$\Theta$$. 这里的 $$d=64$$, 是 `hidden_size_per_attention_head` 的一半. 而元素两两一组进行旋转, 因此基数的大小又要再除以2, 因此这里的 $$\Theta$$ 的长度为 32.

$$\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$$

对于 max sequence length, 每个位置的旋转角度, 由对应的下标和共同的基数 $$\Theta$$ 相乘决定. 因此将下标也以向量表示, 并将下标向量与基数向量进行外积, 得到每个位置对应的旋转角度向量拼接而成的矩阵 `idx_theta`.

最后计算每个角度的 `cos` 和 `sin` 值, 并拼接在一起, 得到 `cache`, 形状大小为 `(max_seq_len, hidden_size_per_attention_head / 4, 2)`. 最后一维的 2, 代表是一个是 cos 值, 一个是 sin 值.

```python
def forward_impl(
        self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
):
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
    return cache
```

这样就得到了每个位置对应的 position embedding, 形状大小为 `(max_seq_len, hidden_size_per_attention_head / 4, 2)`. 只不过使用的方法不能像 token embedding 一样直接 lookup, 而是需要借助下面的 `apply_rotary_pos_emb()` 方法, 对 query 和 key 代表的矩阵施加.

`apply_rotary_pos_emb` 的逻辑为:

`x` 为 query 或 key 对应的张量, 大小为 `(sq, b, np, hn)`. `rope_cache` 对应的 `(sq, hn / 4, 2)`. 然后将 `x` 按最后一维从中间划分为两部分, 每部分的大小为 `hn / 2`, 上下两部分分别记为 `x` 和 `x_pass`.

将 `x` 的形状为调整为 `(sq, b, np, hn / 4, 2)`, 目的是最后一维两个元素为1组, 分别与之前得到的 cos 和 sin 元素相乘, 融合位置信息, 完成 RoPE 的计算, 得到的结果为 `x_out2`, 然后通过 `flatten` 将形状恢复为 `x` 原来的大小, 即 `(sq, b, np, hn / 2)`.

最后将**融合了位置信息的上半部分 `x` 与没有融合位置信息的 `x_pass`** 重新拼接在一起, 返回.

也就是说, **query 和 key 只有一半的参数融合了位置信息, 另一半没有融合**. 至于为什么这么做, 作者们没有给出原因, 可以参考:

- [Github Issue](https://github.com/lucidrains/x-transformers/issues/40)

大体意思是: 只旋转前半部分可以带来微小的性能提升, 并且实验验证不会有性能的损失

```python
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)
```

## ChatGLMPreTrainedModel

`ChatGLMPreTrainedModel` 提供了获取 attention mask 和 position ids.

### get attention mask

```python
def get_masks(self, input_ids, past_key_values, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    # 根据输入创建一个下三角矩阵, Causal LM 对应的 MASK 矩阵
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    past_length = 0
    if past_key_values:
        past_length = past_key_values[0][0].shape[0]
    if past_length:
        # 有 KV Cache, 则 cache 代表的输入中, 所有 tokens 之间是可以相互看到的
        full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                    device=input_ids.device), full_attention_mask), dim=-1)
    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    if not past_length and padding_mask is not None:
        full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask
```

### get position ids

不再使用二维位置编码. 使用普通的从零递增的位置编码.

```python
def get_position_ids(self, input_ids, device):
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids
```
