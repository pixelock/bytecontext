# 实验环境

- GPU: RTX3090, 24G

# 实验记录

实验在 [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) 上进行, 7B 模型大小为 13.5GB 左右.

## SFT with LoRA

### fp16 混合精度训练

以下的测试, 是截取了 [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) 数据集中, `instruction + input + output` 三个字段的字符长度 `len()` 超过 2048 的样本, 共 274 条, 进行的实验.

由于实验的最长训练长度为 2048, 因此只会出现截断, 不会出现 padding 的情况, 保证实验过程中的每一条都是满长度的非 padding token, 保证了显存和时效具有实践性参考.

| batch size | accumulation | max seq length | #samples | GPU memory cost | cost time | cost time per sample |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 512 | 274 | 15.81GB | 115.5s | 0.4215s |
| 8 | 1 | 512 | 274 | 17.51GB | 111.1s | s |
| 16 | 1 | 512 | 274 | 20.90GB | 107.3s | s |
| 16 | 8 | 512 | 274 | 20.90GB | 102.4s | s |
| 28 | 1 | 512 | 274 | 24.10GB | 104.3s | 0.3806s |

### int8 量化训练

以下的测试, 是截取了 [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) 数据集中, `instruction + input + output` 三个字段的字符长度 `len()` 超过 2048 的样本, 共 274 条, 进行的实验.

由于实验的最长训练长度为 2048, 因此只会出现截断, 不会出现 padding 的情况, 保证实验过程中的每一条都是满长度的非 padding token, 保证了显存和时效具有实践性参考.

| batch size | accumulation | max seq length | #samples | GPU memory cost | cost time | cost time per sample |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 512 | 48.8k | 11.1GB | 5h35min | 0.4118s |
| 8 | 1 | 512 | 48.8k | 12.6GB | 4h57min | 0.3652s |
| 8 | 16 | 512 | 48.8k | 13.1GB | 5h2min | 0.3713s |
| 16 | 1 | 512 | 48.8k | 15.46GB | 4h45min | 0.3504s |
| 16 | 8 | 512 | 48.8k | 15.46GB | 4h50min | 0.3566s |
| 32 | 1 | 512 | 48.8k | 23.52GB | 4h38min | 0.3418s |
| 32 | 4 | 512 | 48.8k | 23.52GB | 4h39min | 0.3430s |

### int4 量化训练

以下的测试, 是截取了 [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) 数据集中, `instruction + input + output` 三个字段的字符长度 `len()` 超过 2048 的样本, 共 274 条, 进行的实验.

由于实验的最长训练长度为 2048, 因此只会出现截断, 不会出现 padding 的情况, 保证实验过程中的每一条都是满长度的非 padding token, 保证了显存和时效具有实践性参考.

| batch size | accumulation | max seq length | #samples | GPU memory cost | cost time | cost time per sample |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 512 | 274 | 7.63GB | s | s |
| 8 | 1 | 512 | 274 | 8.43GB | s | s |
| 16 | 1 | 512 | 274 | 11.77GB | s | s |
| 32 | 1 | 512 | 274 | 19.47GB | 103.9s | s |
| 44 | 1 | 512 | 274 | 23.46GB | 103.8s | s |

### 长度限制

以下的测试, 是截取了 [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) 数据集中, `instruction + input + output` 三个字段的字符长度 `len()` 超过 2048 的样本, 共 274 条, 进行的实验.

由于实验的最长训练长度为 2048, 因此只会出现截断, 不会出现 padding 的情况, 保证实验过程中的每一条都是满长度的非 padding token, 保证了显存和时效具有实践性参考.

| batch size | accumulation | max seq length | quant | #samples | GPU memory cost | cost time | cost time per sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 64 | fp16 | 274 | 14.33GB | 19.0s | s |
| 128 | 1 | 64 | fp16 | 274 | 20.41GB | 13.2s | s |
| 192 | 1 | 64 | fp16 | 274 | 23.53GB | 16.1s | s |
| 4 | 1 | 128 | fp16 | 274 | 14.54GB | 34.9s | 0.1274s |
| 64 | 1 | 128 | fp16 | 274 | 21.05GB | 25.8s | 0.0942s |
| 96 | 1 | 128 | fp16 | 274 | 23.45GB | 26.9s | 0.0982s |
| 4 | 1 | 256 | fp16 | 274 | 15.00GB | 60.2s | 0.2197s |
| 32 | 1 | 256 | fp16 | 274 | 20.87GB | 51.9s | 0.1894s |
| 48 | 1 | 256 | fp16 | 274 | 24.56GB | 54.0s | 0.1971s |
| 4 | 1 | 512 | fp16 | 274 | 15.81GB | 115.5s | 0.4215s |
| 28 | 1 | 512 | fp16 | 274 | 24.10GB | 104.3s | 0.3806s |
| 4 | 1 | 1024 | fp16 | 274 | 18.13GB | 234.9s | 0.8573s |
| 8 | 1 | 1024 | fp16 | 274 | 22.17GB | 229.5s | 0.8376s |
| 10 | 1 | 1024 | fp16 | 274 | 24.18GB | 233.0s | 0.8504s |
| 2 | 1 | 2048 | fp16 | 274 | 20.65GB | 347.1s | 1.2667s |
| 3 | 1 | 2048 | fp16 | 274 | 23.92GB | 383.1s | 1.3982s |
| 4 | 1 | 2048 | fp16 | 274 | OOM | - | - |
