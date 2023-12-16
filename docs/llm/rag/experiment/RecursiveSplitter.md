# 实验设置

基于 langchain 的 `RecursiveCharacterTextSplitter` 进行实验. 对于中文语料, 使用的递归 `separators` 为 `['\n\n', '\n', '。', '！', '？', '；', '，']`.

使用的 embedding 模型为 [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5), 模型涉及以下几种:

- [SUSTech/SUS-Chat-34B](https://huggingface.co/SUSTech/SUS-Chat-34B)
- [lmsys/vicuna-13b-v1.5-16k](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k)
- [gpt-3.5-turbo-16k](https://platform.openai.com/docs/models)
- [ERNIE-Bot-4](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu)

实验变量包含:

- chunk_size
- chunk_overlap
- 检索方法, 包括向量召回(VectorStore)和关键词召回(BM25)

# 实验结果

| 检索方法 | chunk_size | chunk_overlap | 实验结果 |
| -------- | ---------- | ------------- | -------- |
| vector | 200 | 100 | 0.6786 |
| vector | 200 | 200 | 0.6964 |