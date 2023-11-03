# 要解决的问题

解决[Huggingface](https://huggingface.co/models)平台上开源模型如何下载到本地的问题。

Huggingface平台上的开源模型如下图，包含模型文件、配置文件、词典、tokenziner模型或文件等，以git的方式管理。以一个中文版的RoBERTa模型为例：

![](/resources/images/framework/huggingface/QQ20230412-224243.png)

在使用`transformers`时，通过`from_pretrained`方法读取调用huggingface上的模型时，会先下载这些文件到本地。但这个过程是隐式的。Huggingface会提供可以直接使用模型的python代码：

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
```

执行过程如下：

```python
>>> from transformers import AutoTokenizer, AutoModelForMaskedLM
>>> 
>>> tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
Downloading (…)okenizer_config.json: 100%|██████████████████████████████████████████████████████████| 19.0/19.0 [00:00<00:00, 36.8kB/s]
Downloading (…)lve/main/config.json: 100%|████████████████████████████████████████████████████████████| 689/689 [00:00<00:00, 1.28MB/s]
Downloading (…)solve/main/vocab.txt: 100%|███████████████████████████████████████████████████████████| 110k/110k [00:00<00:00, 308kB/s]
Downloading (…)/main/tokenizer.json: 100%|███████████████████████████████████████████████████████████| 269k/269k [00:01<00:00, 214kB/s]
Downloading (…)in/added_tokens.json: 100%|██████████████████████████████████████████████████████████| 2.00/2.00 [00:00<00:00, 3.73kB/s]
Downloading (…)cial_tokens_map.json: 100%|█████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 205kB/s]
>>> 
>>> model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
Downloading pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████| 412M/412M [00:54<00:00, 7.52MB/s]
Some weights of the model checkpoint at hfl/chinese-roberta-wwm-ext were not used when initializing BertForMaskedLM: ['cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```

可以看到文件被下载且加载，但这些文件被下载到哪个位置了呢？

默认地，这些模型文件会被放置到`~/.cache/huggingface/hub`目录下，且其中的命名方式是很奇怪的。

![](/resources/images/framework/huggingface/QQ20230412-231312.png)

这种方式可能会存在多个问题：

- 以特殊的方式（与git仓库不同）组织文件，在跨用户、跨服务器使用时无法方便地迁移
- 存储在用户根目录`~`之下，可能会存在占用系统盘的问题。例如系统盘和平时开发使用的或存储使用的不是一个硬盘，可能会造成不知情的情况下系统盘堆满，影响服务器正常功能，影响同一服务器下其他用户的使用

怎么优雅地下载和组织Huggingface开源模型的存储，有以下几种方法。

# 解决方案

## `from_pretrained`方法指定存储路径

使用`from_pretrained`方法加载模型，除了给`pretrained_model_name_or_path`参数指定模型名称或者位置，可以使用`cache_dir`参数来指定存放下载的文件目录。

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir='/home/garnet/develop/models/RoBERTa')
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir='/home/garnet/develop/models/RoBERTa')
```

![](/resources/images/framework/huggingface/QQ20230412-232722.png)

执行完可以看到，与不指定`cache_dir`参数的区别就是在于将模型文件下载到了指定的位置，方便我们自己管理。而模型文件的组织形式是没有变化的。

## git lfs 克隆模型仓库

上面的模型文件组织形式只能被`transformers`包使用，且不方便跨用户使用或者在设备之间迁移使用。而平台提供的模型实际上是存储在git仓库中的，那么自然就会想到能不能像使用github这种托管平台一样，将模型通过git指令clone到本地？

答案自然是可以的。Huggingface平台对每个模型也都提供了git指令。继续以上面的中文RoBERTa模型为例，平台提供的clone指令为：

```bash
git lfs install
git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext
```

由于模型文件一般都比较大，对于这种大文件的同步需要借助`git-lfs`工具，安装过程参考`https://git-lfs.com`。以Ubuntu为例，安装过程为：

**第一步**，添加packagecloud仓库，packagecloud提供脚本来自动执行在系统上配置包存储库、导入签名密钥等过程。

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```

**第二步**，安装 Git LFS：

```bash
sudo apt-get install git-lfs
```

安装完成！

但这种方法存在一个问题：即会下载仓库中的所有文件。而很多模型仓库中会存在多个版本的模型文件，比如我们举例的中文RoBERTa模型中，存在着`flax_model.msgpack`、`tf_model.h5`和`pytorch_model.bin`三个不同网络框架使用的模型文件，但我们使用pytorch模型其实只需要`pytorch_model.bin`这一种模型文件就足够了，下载所有版本的模型文件会大大延长下载的时间和占用的空间。而使用`from_pretrained`方法`transformers`包会根据框架自动筛选出我们需要的模型文件，避免了这个问题。

另外注意，下载过程中没有进度条，要下载的模型文件大，且通过git服务器的下载速度慢，因此会出现类似与卡住的现象，是正常情况，需要耐心等待。

## 通过 huggingface_hub 下载

安装`huggingface_hub`包。

```bash
pip install huggingface_hub
```

huggingface_hub提供了很多种模型下载的方案，详细参考文档[Download files from the Hub](https://huggingface.co/docs/huggingface_hub/v0.13.4/guides/download). 其中的[snapshot_download()](https://huggingface.co/docs/huggingface_hub/v0.13.4/en/package_reference/file_download#huggingface_hub.snapshot_download)方法可以下载整个模型git仓库，且可以指定要排除的文件或文件类型。

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="hfl/chinese-roberta-wwm-ext",
    local_dir='/home/garnet/develop/models/RoBERTa/chinese-roberta-wwm-ext',
    local_dir_use_symlinks=False,
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
)
```

下载过程如下：

```python
Downloading (…)44/added_tokens.json: 100%|██████████████████████████████████████████████████████████| 2.00/2.00 [00:00<00:00, 3.59kB/s]
Downloading (…)okenizer_config.json: 100%|██████████████████████████████████████████████████████████| 19.0/19.0 [00:00<00:00, 28.4kB/s]
Downloading (…)bf00bfdb44/README.md: 100%|████████████████████████████████████████████████████████| 2.07k/2.07k [00:00<00:00, 4.01MB/s]
Downloading (…)cial_tokens_map.json: 100%|█████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 211kB/s]
Downloading (…)00bfdb44/config.json: 100%|████████████████████████████████████████████████████████████| 689/689 [00:00<00:00, 1.25MB/s]
Downloading (…)fdb44/.gitattributes: 100%|█████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 667kB/s]
Downloading (…)fdb44/tokenizer.json: 100%|███████████████████████████████████████████████████████████| 269k/269k [00:00<00:00, 640kB/s]
Downloading (…)bf00bfdb44/vocab.txt: 100%|███████████████████████████████████████████████████████████| 110k/110k [00:00<00:00, 297kB/s]
Downloading pytorch_model.bin: 100%|█████████████████████████████████████████████████████████████████| 412M/412M [13:55<00:00, 492kB/s]
Fetching 9 files: 100%|██████████████████████████████████████████████████████████████████████████████████| 9/9 [13:57<00:00, 93.11s/it]
'/home/garnet/develop/models/RoBERTa/chinese-roberta-wwm-ext'
```

看一眼模型长什么样子。

![](/resources/images/framework/huggingface/QQ20230413-092354.png)

可以看到所有模型相关的文件，包括大的模型参数文件都下载到指定的路径中了。
