Hugging Face 推出的 [Datasets](https://huggingface.co/docs/datasets/main/en/index) 能够快速地获取, 加载, 处理多样的数据, 方便模型训练的 pipeline 的构建. 对于文本, 图像, 音频多种内容形态, 以及大模型训练所需要的海量数据的读取处理, 都可以提供简单, 高效且低资源消耗的方法.

Datasets 底层使用 **Apache Arrow** 的格式组织数据, 并且对大规模数据集使用 **zero-copy reads** 避免全部加载到内存中造成OOM, 同时对数据处理又有着很高的效率. 同时也提供了很方便的API帮助我们加载各种样式的本地和线上数据.

# Install

安装 Datasets 包的方法:

```bash
pip install datasets

# 要处理图片数据集, 使用下面的指令安装
pip install datasets[vision]

# 处理音频数据集
pip install datasets[audio]
```

# 内容

Datasets 相关的知识包含如下.

## 读写数据

数据集可能存放在各种地方: 本地磁盘, Github Repository, 内存等等. 读取数据有以下方法:


- [使用 Datasets 读取本地数据](/docs/framework/datasets/读取本地数据.md)
- [使用 Datasets 读取线上数据](/docs/framework/datasets/读取线上数据.md)
- [流式读取](/docs/framework/datasets/流式读取.md)
- [保存与读取](/docs/framework/datasets/保存与读取.md)

## 处理数据

- [对数据集进行重新布置](/docs/framework/datasets/对数据集进行重新布置.md)

## 底层原理

- [Datasets 底层原理](/docs/framework/datasets/底层原理.md)
