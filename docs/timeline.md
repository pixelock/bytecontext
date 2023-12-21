# 2023

## 12 December

### 19

- [VLLM 推理速度实验](/docs/llm/inference/framework/vllm/speed-experiment.md): vLLM 细致的性能测试

## 11 November

### 22

**新增**

- [OpenAI API](/docs/llm/servering/openai-api.md): OpenAI Chat Completion API 的使用方法和出入参详细说明

### 21

**新增**

- [vLLM](/docs/llm/inference/framework/vllm/vllm.md): vLLM 框架能够实现推理高吞吐量的原理

### 20

**新增**

- [Automix](/docs/llm/rag/flow/automix.md): 介绍了一种 RAG 系统工作流, 将不同参数大小的 LLM 结合, 在资源消耗相对于只使用小模型扩增不大的前提下, 将系统的回答质量提升到与使用大模型近似的水平
- [Chain-of-Note](/docs/llm/rag/answering/chain-of-note.md): 介绍了一个 prompt, 在 RAG 系统回答时使用. 通过 prompt 引导 LLM 依次输出对于每条 relevant document 的总结以及指明与 query 是否相关, 最后再输出 query 的回答. 通过这种引导的方式, 提升 RAG 系统回答的鲁棒性

### 19

**新增**

- [Min-p 采样](/docs/llm/inference/sampling/min-p.md): 介绍了一种新的采样方法, 有效地在保证多样性的基础上, 提升采样的质量
- [RAG 系统评估优化策略](/docs/llm/rag/evaluate-improve.md): 如何对 RAG 进行系统评估, 并针对性地对系统不同的模块进行优化
