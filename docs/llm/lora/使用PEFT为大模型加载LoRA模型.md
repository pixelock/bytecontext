# 加载

## 加载一个 LoRA 模型

通过 `peft.PeftModel.from_pretrained` 为大模型加载一个 LoRA **适配器**(Adapter), 我们把大模型称为 `base_model`. 在 `from_pretrained` 方法中, 可以通过 `adapter_name` 来为当前的适配器指定一个名称.

```python
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# 加载 base_model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    use_auth_token=True
)

# 为 base_model 加载 LoRA Adapter
model = PeftModel.from_pretrained(model, "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA", adapter_name="chinese_chat")
```

## 加载多个 LoRA 模型, 并切换

`peft` package 支持在一个 `base_model` 下挂在多个不同的适配器, 从而支持在线动态地切换适配器.

通过 `load_adapter()` 方法, 在通过 `PeftModel.from_pretrained()` 方式获得的基础上, 加载另一个适配器.

```python
model = PeftModel.from_pretrained(model, "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA", adapter_name="chinese_chat")
# 查看已经加载的适配器
model.peft_config
"""
{
    'chinese_chat': LoraConfig(peft_type='LORA', auto_mapping=None, base_model_name_or_path='meta-llama/Llama-2-7b-chat-hf', revision=None, task_type='CAUSAL_LM', inference_mode=True, r=8, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'], lora_alpha=32, lora_dropout=0.05, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None)
}
"""
# 查看当前激活的适配器
model.active_adapter
"""
'chinese_chat'
"""

model.load_adapter('PulsarAI/llama-2-alpacagpt4-1000step', adapter_name='english_alpaca')
# 查看已经加载的适配器
model.peft_config
"""
{
    'chinese_chat': LoraConfig(peft_type='LORA', auto_mapping=None, base_model_name_or_path='meta-llama/Llama-2-7b-chat-hf', revision=None, task_type='CAUSAL_LM', inference_mode=True, r=8, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'], lora_alpha=32, lora_dropout=0.05, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None),
    'english_alpaca': LoraConfig(peft_type='LORA', auto_mapping=None, base_model_name_or_path='decapoda-research/llama-7b-hf', revision=None, task_type='CAUSAL_LM', inference_mode=True, r=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_alpha=16, lora_dropout=0.05, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None)
}
"""
# 查看当前激活的适配器
model.active_adapter
"""
'chinese_chat'
"""
```

通过 `load_adapter()` 加载进来的新适配器模型还没有被激活, 如果要使用新模型, 还需要调用 `set_adapter(adapter_name)` 方法, 根据适配器名称, 选择要激活的适配器.

```python
model.set_adapter('english_alpaca')
# 查看当前激活的适配器
model.active_adapter
"""
'english_alpaca'
"""
```

# 禁用

可以通过上下文管理器 `with model.disable_adapter()` 禁用已经加载的适配器, 只使用原始的 `base_model`.

# 融合

我们在以下场景中会有将 `base_model` 与 `adapter` 融合的需求:

- `base_model` 与 `adapter` 参数权重分开保存, 虽有灵活性, 但在一些垂直领域使用固定的训练好的 LoRA 适配器即可, 不需要灵活性. 这样, 权重分开保存, 每次启动在进行融合的范式就显得很麻烦
- 同时加载多个 LoRA 适配器, 这些适配器之间有**串联关系**, 即第二个 LoRA model 是在第一个 LoRA model 的基础上, 训练得到的
  - 比如以 LLaMA 为基座的大模型在中文上表现不佳, 需要 continue pre-training, 以提升预训练的大模型在中文上的表现. 然后在此基础上, 进一步使用 instruction dataset 进行 SFT, 提升对指令的相应能力. 这套范式是开源社区将大模型汉化的常用方法
  - 使用上面介绍的 `load_adapter` 和 `set_adapter` 配合, 是可以加载多个大模型, 但同时只能激活一个

使用 `merge_and_unload()` 方法, 可以将 `adapter` 中 LoRA 的两个低秩 `lora_A`, `lora_B` 通过矩阵乘法还原成原始模型中对应参数的形状, 然后与原始模型的参数加载一起(`original + lora_A @ lora_B`), 完成 `base_model` 与 `adapter` 的融合.

融合得到的模型结构与 `base_model` 完全一致, 相比之下只是参数得到了更新. 将融合后的参数保存到本地, 推理使用可以直接加载, 避免分别加载再融合; 或者可以继续加载新的 adapter.

```python
model = PeftModel.from_pretrained(model, "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA", adapter_name="chinese_chat")
print(type(model))
# 输出: <class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
# 是 base_model 的类
model = model.merge_and_unload()
model.save_pretrained(output_path)
```

保存后的文件包含:

![](/resources/images/llm/lora-merge.png)

与基座模型的文件一致. 再看看 `config.json` 中关于模型的定义:

```json
{
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.34.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}
```

与基座模型的配置文件完全相同, 包括模型名称(`_name_or_path`)以及模型种类(`model_type`)等都没有变. 使用 `AutoModelForCausalLM` 仍然会加载为基座模型的类. 

# 参考

- [【peft】huggingface大模型加载多个LoRA并随时切换](https://blog.csdn.net/liuqixuan1994/article/details/130664198)
- [LLM - LoRA 模型合并与保存](https://bitddd.blog.csdn.net/article/details/132065177)
- [LLaMA2 手动模型合并与转换](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/manual_conversion_zh)
- [LLaMA 手动模型合并与转换](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2)
