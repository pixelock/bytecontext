# LlamaTokenizer

## special tokens

LLaMA 中有三个 special token, 分别是:

- `<s>`: 代表 text 的起始, 对应 ID 为 1
- `</s>`: 代表 text 的结束, 对应 ID 为 2
- `<unk>`: 代表 unknown 等, 对应 ID 为 0

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

### `tokenize()`

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

### `convert_tokens_to_ids()`

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
