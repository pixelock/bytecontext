解析的代码来自 [Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese), 是一个基于 LLaMA 的开源中文预训练大模型. 这仍然是一个**预训练模型**, 即没有经过 SFT 和 RLHF 训练 Chat 模型. 目标是对 Llama2 模型进行中文能力的持续迭代升级.

在 Llama2 的基础上, 采用大规模的中文数据进行持续预训练, 包含百科、书籍、博客、新闻、公告、小说、金融数据、法律数据、医疗数据、代码数据、专业论文数据、中文自然语言处理竞赛数据集等. 同时对庞大的数据进行了过滤、打分、去重，筛选出超过1T token的高质量中文数据, 持续不断加入训练迭代中.

为了提高中文文本处理的效率, 针对 Llama2 模型的词表进行了深度优化:

- 首先, 基于数百G的中文文本, 在该模型词表的基础上扩展词库至65000个单词(原来为32000个token)
- 经过测试, 改进使得中文编码/解码速度提高了约350％
- 此外, 我们还扩大了中文字符集的覆盖范围, 包括所有emoji符号, 这使得生成带有表情符号的文章更加高效

# 预训练

预训练的代码入口为仓库中的 [train/pretrain/pretrain.sh](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/pretrain/pretrain.sh), 对应的实现代码为 [train/pretrain/pretrain_clm.py](https://github.com/FlagAlpha/Llama2-Chinese).

## 读取数据

入口的 `main()` 函数中, 首先读取了数据, 并做了一些处理.

```python
data_files = {}
dataset_args = {}
if data_args.train_files is not None:
    # 这里是获取训练集对应的所有文件地址
    if type(data_args.train_files) == str:
        print(pd.read_csv(data_args.train_files))
        data_files["train"] = pd.read_csv(data_args.train_files)['file_name'].to_list()
        print('训练文件总个数', len(data_files["train"]))
    else:
        print(pd.read_csv(data_args.train_files[0]))
        data_files["train"] = pd.read_csv(data_args.train_files[0])['file_name'].to_list()
        print('训练文件总个数', len(data_files["train"]))
if data_args.validation_files is not None:
    # 如果有指定验证集的文件, 也进行记录
    data_files["validation"] = data_args.validation_files

# 这里的 `extension` 参数代表的是文件的类型, 因为下面使用 `load_dataset()` 读取文件时, 需要指明文件类型
extension = (
    data_files["train"][0].split(".")[-1]
    if data_files["train"] is not None
    else data_args.validation_files.split(".")[-1]
)
if extension == "txt":
    extension = "text"
    dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

# 读取数据集
raw_datasets = load_dataset(
    extension,
    data_files=data_files,  # 可能只读取训练集, 如果有指定验证集, 则验证集也读取
    streaming=data_args.streaming,  # 如果文件太大, 需要使用流模式
    cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'),
    use_auth_token=True if model_args.use_auth_token else None,
    **dataset_args,
)

if data_args.streaming:
    # 如果使用流模式, 需要先进行 shuffle, 这里的 buffer size 为 1000000, 也就是先读取百万条数据, 再打乱
    raw_datasets = raw_datasets.shuffle(seed=training_args.seed, buffer_size=1000000)

# 如果没有指定验证集文件, 则从训练集中划分出一部分, 作为验证集. 剩余部分作为真正的训练集
if "validation" not in raw_datasets.keys():
    raw_datasets["validation"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train[:{data_args.validation_split_percentage}%]",
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        **dataset_args,
    )
    raw_datasets["train"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train[{data_args.validation_split_percentage}%:]",
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        **dataset_args,
    )
```

## 初始化

初始化 config, tokenizer 以及 model.

**config**

```python
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")
```

**tokenizer**

```python
print(training_args.local_rank, 'start load tokenizer')
tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )
```

**model**

```python
print(training_args.local_rank, 'end load tokenizer')
print(training_args.local_rank, 'start load model')
if model_args.model_name_or_path:
    # 这里是读取指定位置的预训练模型
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
else:
    # 这里是从 config 初始化模型, 模型参数随机初始化, 如果从零开始进行预训练, 需要使用这里的初始化方法
    model = AutoModelForCausalLM.from_config(config)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")
print(training_args.local_rank, 'end load model')
```

对 model 的 embedding 层进行调整:

```python
# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
```

## 输入处理

### 拼接特殊符号

通过以下的方法, 为数据集中的每个文本拼接特殊符号, 然后进行 tokenize, 并调整为模型需要的格式:

```python
def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(['<s>' + item + '</s>' for item in examples[text_column_name]])
    return output

with training_args.main_process_first(desc="dataset map tokenization"):
    if not data_args.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            batch_size=60000,
        )
```

Llama2 对文本的处理方式, 是在文本的前后分别拼接上起始和终止的特殊符号 `<s>` 和 `</s>`, 然后进行 tokenize.

### 确定输入序列的最大长度

通过 `--block_size` 参数指定序列的最大长度. 对于 Llama2, 最大长度为 4096.

```python
if data_args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
            " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--block_size xxx`."
        )
        block_size = 1024
else:
    if data_args.block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(data_args.block_size, tokenizer.model_max_length)
```

### 拼接样本, 直到填满 block_size

拼接的方法为:

- 首先传入一个大批量的句子集合, 代码中使用的 batch_size 为 40000(streaming) / 60000(non-streaming)
  - 这里每个句子的格式为 `<s> 句子内容 </s>` 对应 tokenize 后的 token ids, 拼接了起始和结束的特殊字符
- 将 batch 内的所有句子前后拼接, 得到的是所有 token ids 的列表
- 判断总长度是否超过 block_size(一般都是超过的)
  - 没有超过 block_size, 这不进行处理
- 超过了 block_size, 则对 token ids 的列表进行截断, 截断到 **`(total_length // block_size) * block_size`**
  - 相当于是创建了 `(total_length // block_size)` 个长度为 `block_size` 的 token 序列
- 对拼接在一起得到的 token 序列进行切分, 按长度 `block_size` 进行切分, 得到一个大小为 `(total_length // block_size), block_size` 的二维矩阵
- 这个二维矩阵, 就是模型输入中的 `input_ids`.

这样重新组织得到的数据集, 每个样本的长度都为 `block_size`, 其中没有 pad token, 避免了算力的浪费.

```python
# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
    logger.info("group texts input examples length%d after_group size%d" % (
    len(examples['input_ids']), len(result["input_ids"])))
    result["labels"] = result["input_ids"].copy()
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

with training_args.main_process_first(desc="grouping texts together"):
    if not data_args.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
            batch_size=40000,
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=60000,
        )
```

## 初始化 Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=IterableWrapper(train_dataset) if training_args.do_train else None,
    eval_dataset=IterableWrapper(eval_dataset) if training_args.do_eval else None,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    # callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
)
```

## 执行训练

```python
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(training_args.local_rank, 'start train')
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
```

## 评价指标

使用**准确率**指标评价模型训练效果. **按 token 的粒度比较**, 即对验证数据集的 batch 内每个每个样本的每个 token 比较是否准确.

```python
from sklearn.metrics import accuracy_score

def _compute(self, predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        )
    }
```

```python
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)
```

# SFT

指令微调分为对全量参数进行微调, 以及使用 LoRA 进行微调两种方案, 对应的代码入口分别为仓库中的:

- 全量: [train/sft/finetune.sh](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune.sh)
- LoRA: [train/sft/finetune_lora.sh](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_lora.sh)

对应的实现代码分别为:

- 全量: [train/sft/finetune_clm.py](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_clm.py)
- LoRA: [train/sft/finetune_clm_lora.py](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_clm_lora.py)

与预训练的过程相比, 区别在于训练样本的组织方式. 在预训练过程中, 会将样本拼接, 保证 batch 内每个样本都达到最长长度而且没有 padding, 充分保证训练的效率.

微调的过程则不需要再拼接, 数据集中的每条样本单独处理, 作为一条训练样本即可.

如果输入样本只有一个字段, 对应的是**无条件生成**学习, 使用如下的处理方式:

```python
def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer([ item for item in examples[text_column_name]],truncation=True,max_length=data_args.block_size,padding=False,return_tensors=None)
        output['labels'] = output['input_ids'].copy()
    return output
```

如果输入样本有两个字段, 即一个 `input` 字段, 一个 `target` 字段, 对应的是**有条件生成**学习, 使用另一种处理函数:

```python
def generate_and_tokenize_prompt(data_point):
    input_text = data_point[input_column_name]
    target_text = data_point[target_column_name]
    full_prompt = input_text+target_text
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = input_text
        tokenized_user_prompt = tokenize(user_prompt)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ] 
    return tokenized_full_prompt
```
