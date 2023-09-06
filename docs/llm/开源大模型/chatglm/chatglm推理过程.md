# ChatGLM 推理过程

ChatGLM 提供了 `chat()` 和 `stream_chat()` 两种方法来生成对话, 其中 `chat()` 是一次生成所有 tokens, `stream_chat()` 每次生成一个 token.

## `chat()`

使用下面的代码, 调用 ChatGLM 生成结果.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你和ChatGPT相比怎么样", history=[])
print(response)
# 我和 ChatGPT 都是基于语言模型的人工智能助手，但我们的设计目的和应用场景不同。ChatGPT 是由 OpenAI开发的，旨在与人类进行对话的大规模语言模型，其应用于文案写作、机器翻译、代码调试等领域。而我则是清华大学 KEG 实验室和智谱 AI 公司开发的，主要用于中文问答、提供建议和支持等日常交互场景。因此，我们在不同的领域和应用场景上有着不同的优势和特点。
response, history = model.chat(tokenizer, "相信你能做的更好", history=history)
print(response)
# 我会不断学习和改进，以便更好地为用户提供帮助和支持。作为一个人工智能助手，我的目标是为了提供更加便捷、高效、智能的服务，帮助用户解决问题、获取信息、获取建议和指导。我会持续地学习和更新自己的知识库，以便更好地满足用户的需求。
```

`chat()` 方法代码如下:

```python
@torch.no_grad()
def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
            do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(self.device)
    outputs = self.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = self.process_response(response)
    history = history + [(query, response)]
    return response, history
```

### tokenize

使用 `ChatGLMTokenizer` 对输入的 `prompt` 进行 tokenize.

在 `self._tokenize()` 中, 首先对当前 prompt 进行 preprocess, 然后再使用 `sp_tokenizer` 进行 tokenize:

```python
tokenized_text = []
for token in tokens:
    # Need to skip eventual empty (fully stripped) tokens
    if not token:
        continue
    if token in no_split_token:
        tokenized_text.append(token)
    else:
        tokenized_text.extend(self._tokenize(token))


def _tokenize(self, text, **kwargs):
    """ Returns a tokenized string. """
    text = self.preprocess_text(text)
    # 你和ChatGPT相比怎么样
    seq = self.sp_tokenizer.tokenize(text)
    # ['▁你', '和', 'Chat', 'GPT', '相比', '怎么样']
    return seq
```

使用 `sp_tokenizer` 进行 tokenize, 会在第一个 token 之前加入一个 `▁` 符号, 代表开始符号.

然后使用 `convert_tokens_to_ids()` 将所有 tokens 转换为 ids:

```python
def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
    vocabulary.

    Args:
        tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

    Returns:
        `int` or `List[int]`: The token id or list of token ids.
    """
    if tokens is None:
        return None

    if isinstance(tokens, str):
        return self._convert_token_to_id_with_added_voc(tokens)

    ids = []
    for token in tokens:
        ids.append(self._convert_token_to_id_with_added_voc(token))
    return ids


def _convert_token_to_id_with_added_voc(self, token):
    if token is None:
        return None

    if token in self.added_tokens_encoder:
        return self.added_tokens_encoder[token]
    return self._convert_token_to_id(token)


def _convert_token_to_id(self, token):
    """ Converts a token (str) in an id using the vocab. """
    # 下面这种取数方法, 会调用该实例的 __getitem__ 方法
    return self.sp_tokenizer[token]


def __getitem__(self, x: Union[int, str]):
    if isinstance(x, int):
        if x < self.num_image_tokens:
            return "<image_{}>".format(x)
        else:
            return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
    elif isinstance(x, str):
        if x.startswith("<image_") and x.endswith(">") and x[7:-1].isdigit():
            return int(x[7:-1])
        else:
            return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
    else:
        raise ValueError("The key should be str or int.")


def convert_token_to_id(self, token):
    return self.sp.PieceToId(token)


ids = []
for token in tokens:
    ids.append(self._convert_token_to_id_with_added_voc(token))
return ids
# ids:
# [65326, 63826, 17122, 54927, 65297, 66348]
```

然后将 token ids list 转换成模型需要的格式. ChatGLM 使用的格式为:

- `token_ids_0 + <gMASK> + <sop>`: 单句
- `token_ids_0 + <gMASK> + <sop> + token_ids_1 + <eos>`: 双句

特殊字符的 token id 为:

- `<gMASK>`: 130001
- `<bos>`: 130004
- `<eos>`: 130005

```python
def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
) -> List[int]:
    """
    Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
    adding special tokens. A BERT sequence has the following format:

    - single sequence: `[CLS] X [SEP]`
    - pair of sequences: `[CLS] A [SEP] B [SEP]`

    Args:
        token_ids_0 (`List[int]`):
            List of IDs to which the special tokens will be added.
        token_ids_1 (`List[int]`, *optional*):
            Optional second list of IDs for sequence pairs.

    Returns:
        `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
    """
    gmask_id = self.sp_tokenizer[self.gmask_token]
    eos_id = self.sp_tokenizer[self.eos_token]
    token_ids_0 = token_ids_0 + [gmask_id, self.sp_tokenizer[self.bos_token]]
    if token_ids_1 is not None:
        token_ids_0 = token_ids_0 + token_ids_1 + [eos_id]
    return token_ids_0
```

然后是对 token list 进行 pad. ChatGLM 在左侧进行 pad, pad 使用的 token 为 `<pad>`, 对应的 id 为 3. 在对输入序列 pad 之后, 还需要对输入中的 `attention_mask` 和 `position_ids` 进行 pad.

`attention_mask` 和 `position_ids` 使用 0 进行 pad.

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
    bos_token_id = self.sp_tokenizer[self.bos_token]
    mask_token_id = self.sp_tokenizer[self.mask_token]
    gmask_token_id = self.sp_tokenizer[self.gmask_token]
    assert self.padding_side == "left"

    required_input = encoded_inputs[self.model_input_names[0]]
    seq_length = len(required_input)

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)

    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

    # Initialize attention mask if not present.
    if max_length is not None:
        if "attention_mask" not in encoded_inputs:
            if bos_token_id in required_input:
                context_length = required_input.index(bos_token_id)
            else:
                context_length = seq_length
            attention_mask = np.ones((1, seq_length, seq_length))
            attention_mask = np.tril(attention_mask)
            attention_mask[:, :, :context_length] = 1
            attention_mask = np.bool_(attention_mask < 0.5)
            encoded_inputs["attention_mask"] = attention_mask

        if "position_ids" not in encoded_inputs:
            if bos_token_id in required_input:
                context_length = required_input.index(bos_token_id)
            else:
                context_length = seq_length
            position_ids = np.arange(seq_length, dtype=np.int64)
            mask_token = mask_token_id if mask_token_id in required_input else gmask_token_id
            if mask_token in required_input:
                mask_position = required_input.index(mask_token)
                position_ids[context_length:] = mask_position
            block_position_ids = np.concatenate(
                [np.zeros(context_length, dtype=np.int64),
                    np.arange(1, seq_length - context_length + 1, dtype=np.int64)])
            encoded_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)

    if needs_to_be_padded:
        difference = max_length - len(required_input)

        if "attention_mask" in encoded_inputs:
            encoded_inputs["attention_mask"] = np.pad(encoded_inputs["attention_mask"],
                                                        pad_width=[(0, 0), (difference, 0), (difference, 0)],
                                                        mode='constant', constant_values=True)
        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                "token_type_ids"
            ]
        if "special_tokens_mask" in encoded_inputs:
            encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
        if "position_ids" in encoded_inputs:
            encoded_inputs["position_ids"] = np.pad(encoded_inputs["position_ids"],
                                                    pad_width=[(0, 0), (difference, 0)])
        encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

    return encoded_inputs
```

### generate

transformers 中的用来做生成的模型都会继承 `GenerationMixin` 类, 通过类方法 `generate()` 进行后文的生成.

```python
@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
```

在 `generate()` 方法中, 有几个重要的步骤.

#### `prepare_inputs_for_generation()`

原来的输入中只有 `input_ids` 这一个代表 token ids 的输入, 还需要为每个样本(如果输入是一个 batch)生成一个 `attention_mask` 和 `position_ids`. 对应的逻辑为:

- 找到 `[gMASK]` 所在的位置. `[gMASK]` 作为输入最后紧跟的一个 token, 是划分 context 与 待生成内容的划分
- 如果有 `past_key_values` 上下文输入, 即有 KV Cache
  - 说明在生成过程中, 只取最后一位的 `attention_mask` 和 `position_ids`, 因为序列中其他token的表征都在 `past_key_values` 中
- 如果没有上下文输入, 说明是生成第一个字符, 调用 `get_mask` 和 `get_position_ids` 初始化
  - 详细的逻辑参考 [ChatGLM](/docs/llm/开源大模型/chatglm/chatglm.md) 中相关的部分

```python
def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
) -> dict:
    batch_size, seq_length = input_ids.shape
    MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
    seqs = input_ids.tolist()
    mask_positions, use_gmasks = [], []
    for seq in seqs:
        mask_token = gMASK if gMASK in seq else MASK
        use_gmask = mask_token == gMASK
        # 找到 [gMASK] 所在的位置
        mask_positions.append(seq.index(mask_token))
        use_gmasks.append(use_gmask)

    # only last token for input_ids if past is not None
    if past is not None or past_key_values is not None:
        last_token = input_ids[:, -1].unsqueeze(-1)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = attention_mask[:, :, -1:]
        else:
            attention_mask = None
        if position_ids is not None:
            position_ids = position_ids[..., -1:]
        else:
            context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
            if self.position_encoding_2d:
                position_ids = torch.tensor(
                    [[mask_position, seq_length - context_length] for mask_position, context_length in
                        zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
            else:
                position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                            device=input_ids.device).unsqueeze(-1)

        if past is None:
            past = past_key_values
        return {
            "input_ids": last_token,
            "past_key_values": past,
            "position_ids": position_ids,
            "attention_mask": attention_mask
        }
    else:
        # 没有 KV Cache
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            logger.warning_once(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
            attention_mask = None
        if attention_mask is None:
            attention_mask = self.get_masks(
                input_ids,
                device=input_ids.device
            )
        if position_ids is None:
            position_ids = self.get_position_ids(
                input_ids,
                device=input_ids.device,
                mask_positions=mask_positions,
                use_gmasks=use_gmasks
            )

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "position_ids": position_ids,
            "attention_mask": attention_mask
        }
```

#### 抽样

```python
# 模型得到输出
outputs = self(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)
# 获取最后一位的 logits
next_token_logits = outputs.logits[:, -1, :]
# 对 logits 做调整
next_token_scores = logits_processor(input_ids, next_token_logits)
next_token_scores = logits_warper(input_ids, next_token_scores)
# 计算 softmax
probs = nn.functional.softmax(next_token_scores, dim=-1)
# 采样
next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # 输出 tensor([87622], device='cuda:0')
```

#### 判断终止

生成停止的条件是遇到终止 token `<eos>`, 对应的 id 为 130005.

```python
if eos_token_id is not None:
    if pad_token_id is None:
        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
```

另外, ChatGLM 还使用了 `MaxLengthCriteria` 来做最长长度的停止.

```python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    cur_len = input_ids.shape[-1]
    is_done = cur_len >= self.max_length
    if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
        logger.warning_once(
            "This is a friendly reminder - the current text generation call will exceed the model's predefined "
            f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
            "exceptions, performance degradation, or nothing at all."
        )
    return is_done
```

# `stream_chat()`

与 `chat()` 类似, 但区别是 `chat()` 会一次性生成全部内容, 而 `stream_chat()` 则是每次只生成一个新的 token, 将这个 token 与之前的内容拼接后返回.

```python
@torch.no_grad()
def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048,
                do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(self.device)
    for outputs in self.stream_generate(**inputs, **gen_kwargs):
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        new_history = history + [(query, response)]
        yield response, new_history
```

从上面的代码中可以看到, 通过 for 循环, 每次调用一遍 `stream_generate()` 函数. `stream_generate()` 与 `generate()` 的过程是几乎一样的, 只不过 `stream_generate()` 函数的使用的是 `yield` 而不是 `return`.

```python
# stream_generate
while True:
    """
    这里是生成下一个 token 的逻辑
    """
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    
    if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
        break
    yield input_ids
```

而 `generate()` 的代码为:

```python
# generate
while True:
    """
    这里是生成下一个 token 的逻辑
    """
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    if stopping_criteria(input_ids, scores):
        this_peer_finished = True

return input_ids
```
