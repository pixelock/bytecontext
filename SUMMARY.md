# SUMMARY

- [Welcome 欢迎](README.md)

## 技术

- [八股]()
  - [Transformer 细节八股](/docs/nlp/models/transformers/Transformer中的细节.md)
  - [T5 八股](/docs/nlp/models/transformers/t5/T5.md)
  - [开源LLM八股](/docs/llm/开源LLM总结.md)
  - [大模型优化八股](/docs/llm/大模型优化方法概览.md)
- [Neural Network 神经网络](/docs/nn)
  - [Initialization 初始化](/docs/nn/initialization/initialization-summary.md)
    - [Xavier Initialization](/docs/nn/initialization/Xavier-initialization.md)
    - [He Initialization](/docs/nn/initialization/He-initialization.md)
    - [BERT中的初始化](/docs/nn/initialization/BERT中的初始化.md)
  - [Gradient 梯度](/docs/nn/gradient)
    - [梯度消失与梯度爆炸](/docs/nn/gradient/梯度消失与梯度爆炸.md)
  - [Optimizer 优化器](/docs/nn/optimizer)
    - [Adam](/docs/nn/optimizer/adam.md)
  - [Normalization](/docs/nn/normalization/norm-summary.md)
    - [Batch Normalization](/docs/nn/normalization/Batch-Normalization.md)
    - [Layer Normalization](/docs/nn/normalization/Layer-Normalization.md)
    - [Batch Normalization 与 Layer Normalization 的差异](/docs/nn/normalization/Batch-Normalization-Layer-Normalization-Difference.md)
  - [Loss 损失函数](/docs/nn/loss)
    - [ZLPR Loss](/docs/nn/loss/zlpr.md)
  - [Multi task 多任务学习](/docs/nn/multi-task)
    - [MMoE 的问题和解决方法](/docs/nn/multi-task/mmoe的问题和解决方法.md)
  - [Training Trick 训练技巧](/docs/nn/training)
    - [Dataset 数据集准备技巧](/docs/nn/training/dataset)
      - [缓解标注数据的噪声问题](/docs/nn/training/dataset/缓解标注数据的噪声问题.md)
    - [Adversarial Training 对抗训练](/docs/nn/training/adversarial)
      - [NLP 中的对抗训练](/docs/nn/training/adversarial/nlp-adversarial.md)
- [NLP 自然语言处理](/docs/nlp)
  - [Models 模型结构](/docs/nlp/models)
    - [Transformers](/docs/nlp/models/transformers)
      - [Transformer中的细节](/docs/nlp/models/transformers/Transformer中的细节.md)
      - [Transformer相关数值](/docs/nlp/models/transformers/Transformer相关数值.md)
      - [BERT](/docs/nlp/models/transformers/bert)
        - [MLM 任务](/docs/nlp/models/transformers/bert/mlm-task.md)
      - [T5](/docs/nlp/models/transformers/t5/T5.md)
      - [Position Encoding 位置编码](/docs/nlp/models/transformers/position-encoding)
        - [RoPE](/docs/nlp/models/transformers/position-encoding/rope.md)
  - [NER 实体识别](/docs/nlp/ner)
    - [边界错分解决方案](/docs/nlp/ner/边界错分解决方案.md)
  - [Text2Vec 文本向量表征](/docs/nlp/text2vec)
  - [Contrastive Learning 对比学习](/docs/nlp/contrastive)
    - [Loss 对比学习损失函数](/docs/nlp/contrastive/loss)
      - [Triplet Loss](/docs/nlp/contrastive/loss/triplet-loss.md)
      - [InfoNCE Loss](/docs/nlp/contrastive/loss/infonce.md)
      - [Center Loss](/docs/nlp/contrastive/loss/center-loss.md)
      - [ArcFace, CosFace, SphereFace](/docs/nlp/contrastive/loss/ArcFace-CosFace-and-SphereFace.md)
- [LLM 大语言模型](/docs/llm)
  - [开源LLM总结](/docs/llm/开源LLM总结.md)
  - [大模型优化方法概览](/docs/llm/大模型优化方法概览.md)
  - [底层原理](/docs/llm/theory)
    - [为什么LLM大多使用Decoder-only架构](/docs/llm/theory/为什么LLM大多使用Decoder-only架构.md)
    - [LLM 每个训练阶段的作用](/docs/llm/theory/LLM每个训练阶段的作用.md)
  - [开源大模型](/docs/llm/开源大模型/开源大模型对比.md)
    - [ChatGLM](/docs/llm/开源大模型/chatglm)
      - [GLM](/docs/llm/开源大模型/chatglm/glm.md)
      - [ChatGLM](/docs/llm/开源大模型/chatglm/chatglm.md)
      - [ChatGLM 推理过程](/docs/llm/开源大模型/chatglm/chatglm推理过程.md)
      - [ChatGLM2](/docs/llm/开源大模型/chatglm/chatglm2.md)
      - [ChatGLM2 优化内容](/docs/llm/开源大模型/chatglm/chatglm2-优化内容.md)
    - [LLama](/docs/llm/开源大模型/llama)
      - [LLaMA](/docs/llm/开源大模型/llama/llama.md)
      - [LLaMA2 升级内容](/docs/llm/开源大模型/llama/llama2.md)
      - [LLaMA2 源码解析](/docs/llm/开源大模型/llama/llama2-源码解析.md)
      - [LLaMA2 预训练及SFT代码解析](/docs/llm/开源大模型/llama/llama2-预训练及SFT代码解析.md)
    - [Baichuan](/docs/llm/开源大模型/baichuan)
      - [Baichuan 技术方案](/docs/llm/开源大模型/baichuan/baichuan-技术方案.md)
      - [Baichuan2 技术方案](/docs/llm/开源大模型/baichuan/baichuan2-技术方案.md)
  - [Parallel 并行计算](/docs/llm/parallel)
    - [数据并行](/docs/llm/parallel/data-parallel/数据并行.md)
  - [Quantization 量化](/docs/llm/quantization)
    - [浮点数的存储方式](/docs/llm/quantization/浮点数的存储方式.md)
    - [混合精度训练](/docs/llm/quantization/混合精度训练.md)
    - [INT8](/docs/llm/quantization/int8.md)
  - [Training 训练](/docs/llm/training/tricks-summary.md)
    - [7B-LLM 训练实验记录](/docs/llm/training/7B-LLM-训练实验记录.md)
  - [Inference 推理](/docs/llm/inference)
    - [Framework 推理框架](/docs/llm/inference/framework)
      - [vLLM](/docs/llm/inference/framework/vllm/vllm.md)
        - [VLLM 推理速度实验](/docs/llm/inference/framework/vllm/speed-experiment.md)
        - [vLLM 的坑](/docs/llm/inference/framework/vllm/error-case.md)
    - [Sampling 采样](/docs/llm/inference/sampling)
      - [Min-p 采样](/docs/llm/inference/sampling/min-p.md)
  - [Servering 服务](/docs/llm/servering)
    - [OpenAI API](/docs/llm/servering/openai-api.md)
  - [LoRA](/docs/llm/lora)
    - [使用 PEFT 为大模型加载 LoRA 模型](/docs/llm/lora/使用PEFT为大模型加载LoRA模型.md)
  - [ICL](/docs/llm/icl)
    - [ICL 底层原理](/docs/llm/icl/theory)
      - [样本 Label 到 Target 的信息流动](/docs/llm/icl/theory/样本Label到Target的信息流动.md)
  - [RAG 检索增强生成](/docs/llm/rag/summary.md)
    - [RAG 系统评估优化策略](/docs/llm/rag/evaluate-improve.md)
      - [评价指标](/docs/llm/rag/evaluate-index.md)
    - [文本切分策略](/docs/llm/rag/split)
      - [策略切分](/docs/llm/rag/split/策略切分.md)
    - [检索召回](/docs/llm/rag/recall/summary.md)
      - [关键词检索](/docs/llm/rag/recall/keyword.md)
    - [答案生成](/docs/llm/rag/answering)
      - [Chain-of-Note](/docs/llm/rag/answering/chain-of-note.md)
    - [流程优化](/docs/llm/rag/flow)
      - [Automix](/docs/llm/rag/flow/automix.md)
    - [实验结果](/docs/llm/rag/experiment)
      - [Recursive Splitter 的实验结果](/docs/llm/rag/experiment/RecursiveSplitter.md)
- [Framework 模型框架](/docs/framework)
  - [Pytorch](/docs/framework/pytorch)
    - [Tensor Operations](/docs/framework/pytorch/tensor-op/summary.md)
      - [Basic 基础](/docs/framework/pytorch/tensor-op/basic)
        - [获取 Tensor 形状](/docs/framework/pytorch/tensor-op/get-shape.md)
      - [Shape mutating 形状变化](/docs/framework/pytorch/tensor-op/shape-mutating)
        - [Tenosr View 机制](/docs/framework/pytorch/tensor-op/tensor-view.md)
        - [expand 和 repeat](/docs/framework/pytorch/tensor-op/expand-repeat.md)
    - [Operator 算子](/docs/framework/pytorch/operator)
      - [一维卷积](/docs/framework/pytorch/operator/一维卷积.md)
      - [Transformer相关算子](/docs/framework/pytorch/operator/transformer)
        - [torch.nn.functional.scaled_dot_product_attention](/docs/framework/pytorch/operator/transformer/scaled_dot_product_attention.md)
    - [Pytorch中的广播机制](/docs/framework/pytorch/Pytorch中的广播机制.md)
  - [Huggingface](/docs/framework/huggingface)
    - [下载huggingface中的模型](/docs/framework/huggingface/下载huggingface中的模型.md)
  - [Huggingface datasets](/docs/framework/huggingface/datasets/datasets简介.md)
    - [使用 Datasets 读取本地数据](/docs/framework/huggingface/datasets/读取本地数据.md)
    - [使用 Datasets 读取本地数据](/docs/framework/huggingface/datasets/读取本地数据.md)
    - [使用 Datasets 读取线上数据](/docs/framework/huggingface/datasets/读取线上数据.md)
    - [流式读取](/docs/framework/huggingface/datasets/流式读取.md)
    - [保存与读取](/docs/framework/huggingface/datasets/保存与读取.md)
    - [对数据集进行重新布置](/docs/framework/huggingface/datasets/对数据集进行重新布置.md)
    - [Datasets 底层原理](/docs/framework/huggingface/datasets/底层原理.md)
- [Stack 技术栈](/docs/stack)
  - [Python](/docs/stack/python)
    - [Fire](/docs/stack/python/fire/Python命令行工具库.md)
    - [pipenv](/docs/stack/python/pipenv/什么是pipenv.md)
      - [使用pipenv对项目运行环境进行管理](/docs/stack/python/pipenv/使用pipenv对项目运行环境进行管理.md)
  - [Linux](/docs/stack/linux)
    - [WSL2](/docs/stack/linux/wsl/wsl2环境配置.md)
    - [tmux](/docs/stack/linux/tmux/tmux.md)
    - [Ubuntu](/docs/stack/linux/ubuntu)
      - [Ubuntu功能文件夹默认目录设置](/docs/stack/linux/ubuntu/Ubuntu功能文件夹默认目录设置.md)
  - [Hive](/docs/stack/hive)
    - [Hive语句执行逻辑](/docs/stack/hive/Hive语句执行逻辑.md)
    - [随机采样](/docs/stack/hive/随机采样.md)
  - [Gitbook](/docs/stack/gitbook)
    - [Gitbook 中的配置项](/docs/stack/gitbook/gitbook中的配置项.md)
- [Solution 解决方案](/docs/solutions/summary.md)
  - [QA System 问答系统](/docs/solutions/qa)
    - [基于 LLM 解决专业领域问答 - 经验分享](/docs/solutions/qa/基于LLM解决专业领域问答.md)
  - [ABSA 细粒度情感分析](/docs/solutions/absa/summary.md)
- [Problems 编程题](/docs/problems/算法题知识归类.md)
  - [解法归类](/docs/problems/解法归类)
    - [背包问题](/docs/problems/解法归类/背包问题.md)
    - [二分总结](/docs/problems/解法归类/二分总结.md)
  - [动态规划](/docs/problems/动态规划)
    - [[10][困难][动态规划] 正则表达式匹配](/docs/problems/字符串/10-正则表达式匹配.md)
    - [[279][中等][动态规划][BFS] 完全平方数](/docs/problems/动态规划/279-完全平方数.md)
    - [[322][中等][动态规划][DFS] 零钱兑换](/docs/problems/动态规划/322-零钱兑换.md)
    - [[343][中等][动态规划] 整数拆分](/docs/problems/动态规划/343-整数拆分.md)
    - [[416][中等][动态规划] 分割等和子集](/docs/problems/动态规划/416-分割等和子集.md)
    - [[474][中等][动态规划] 一和零](/docs/problems/动态规划/474-一和零.md)
    - [[494][中等][动态规划] 目标和](/docs/problems/动态规划/494-目标和.md)
    - [[518][中等][动态规划] 零钱兑换 II](/docs/problems/动态规划/518-零钱兑换-II.md)
    - [[983][中等][动态规划] 最低票价](/docs/problems/动态规划/983-最低票价.md)
    - [[1049][困难][动态规划] 最后一块石头的重量 II](/docs/problems/动态规划/1049-最后一块石头的重量-II.md)
    - [[面试题 08.11][中等][动态规划] 硬币](/docs/problems/动态规划/08.11-硬币.md)
  - [滑动窗口](/docs/problems/滑动窗口)
    - [[3][中等][滑动窗口] 无重复字符的最长子串](/docs/problems/滑动窗口/3-无重复字符的最长子串.md)
    - [[76][困难][滑动窗口] 最小覆盖子串](/docs/problems/滑动窗口/76-最小覆盖子串.md)
    - [[239][困难][队列][辅助结构] 滑动窗口最大值](/docs/problems/队列/239-滑动窗口最大值.md)
    - [[438][中等][滑动窗口] 找到字符串中所有字母异位词](/docs/problems/滑动窗口/438-找到字符串中所有字母异位词.md)
    - [[567][中等][滑动窗口] 字符串的排列](/docs/problems/滑动窗口/567-字符串的排列.md)
  - [字符串](/docs/problems/字符串)
    - [[5][中等][动态规划] 最长回文子串](/docs/problems/字符串/5-最长回文子串.md)
    - [[10][困难][动态规划] 正则表达式匹配](/docs/problems/字符串/10-正则表达式匹配.md)
    - [[28][简单] 找出字符串中第一个匹配项的下标](/docs/problems/字符串/28-找出字符串中第一个匹配项的下标.md)
    - [[44][困难][动态规划] 通配符匹配](/docs/problems/字符串/44-通配符匹配.md)
    - [[72][困难][动态规划] 编辑距离](/docs/problems/字符串/72-编辑距离.md)
    - [[115][困难][动态规划] 不同的子序列](/docs/problems/字符串/115-不同的子序列.md)
    - [[214][困难] 最短回文串](/docs/problems/字符串/214-最短回文串.md)
    - [[459][简单] 重复的子字符串](/docs/problems/字符串/459-重复的子字符串.md)
    - [[712][中等][动态规划] 两个字符串的最小ASCII删除和](/docs/problems/字符串/712-两个字符串的最小ASCII删除和.md)
    - [[796][简单] 旋转字符串](/docs/problems/字符串/796-旋转字符串.md)
    - [[1143][中等][动态规划] 最长公共子序列](/docs/problems/字符串/1143-最长公共子序列.md)
    - [[1316][困难] 不同的循环子字符串](/docs/problems/字符串/1316-不同的循环子字符串.md)
    - [[1392][困难] 最长快乐前缀](/docs/problems/字符串/1392-最长快乐前缀.md)
  - [堆](/docs/problems/堆/堆数据结构.md)
    - [[264][中等][堆] 丑数 II](/docs/problems/堆/264-丑数-II.md)
    - [[313][中等][堆] 超级丑数](/docs/problems/堆/313-超级丑数.md)
    - [[1046][简单][堆] 最后一块石头的重量](/docs/problems/堆/1046-最后一块石头的重量.md)
  - [栈](/docs/problems/栈)
    - [[42][困难][栈][动态规划] 接雨水](/docs/problems/栈/42-接雨水.md)
    - [[84][困难][栈] 柱状图中最大的矩形](/docs/problems/栈/84-柱状图中最大的矩形.md)
    - [[155][简单][栈][滑动窗口] 最小栈](/docs/problems/栈/155-最小栈.md)
    - [[232][简单][栈][队列] 用栈实现队列](/docs/problems/栈/232-用栈实现队列.md)
  - [队列](/docs/problems/队列)
    - [[225][简单][栈][队列] 用队列实现栈](/docs/problems/队列/225-用队列实现栈.md)
  - [数组](/docs/problems/数组)
    - [[1][简单][哈希] 两数之和](/docs/problems/数组/1-两数之和.md)
    - [[4][困难][二分][双指针] 寻找两个正序数组的中位数](/docs/problems/数组/4-寻找两个正序数组的中位数.md)
    - [[34][中等][二分] 在排序数组中查找元素的第一个和最后一个位置](/docs/problems/数组/34-在排序数组中查找元素的第一个和最后一个位置.md)
    - [[35][简单][二分] 搜索插入位置](/docs/problems/数组/35-搜索插入位置.md)
    - [[53][中等][动态规划] 最大子序和](/docs/problems/数组/53-最大子序和.md)
    - [[153][中等][二分] 寻找旋转排序数组中的最小值](/docs/problems/数组/153-寻找旋转排序数组中的最小值.md)
    - [[154][困难][二分] 寻找旋转排序数组中的最小值 II](/docs/problems/数组/154-寻找旋转排序数组中的最小值-II.md)
    - [[167][简单][双指针][二分] 两数之和 II - 输入有序数组](/docs/problems/数组/167-两数之和-II-输入有序数组.md)
    - [[215][中等][堆] 数组中的第K个最大元素](/docs/problems/数组/215-数组中的第K个最大元素.md)
    - [[287][中等][双指针][二分] 寻找重复数](/docs/problems/数组/287-寻找重复数.md)
    - [[300][中等][动态规划][贪心] 最长上升子序列](/docs/problems/数组/300-最长上升子序列.md)
    - [[354][困难][动态规划][贪心] 俄罗斯套娃信封问题](/docs/problems/数组/354-俄罗斯套娃信封问题.md)
    - [[378][中等][归并][二分] 有序矩阵中第K小的元素](/docs/problems/数组/378-有序矩阵中第K小的元素.md)
    - [[435][中等][动态规划][贪心] 无重叠区间](/docs/problems/数组/435-无重叠区间.md)
    - [[452][中等][动态规划][贪心] 用最少数量的箭引爆气球](/docs/problems/数组/452-用最少数量的箭引爆气球.md)
    - [[456][中等][栈] 132模式](/docs/problems/数组/456-132模式.md)
    - [[480][困难][堆] 滑动窗口中位数](/docs/problems/数组/480-滑动窗口中位数.md)
    - [[491][中等][DFS] 递增子序列](/docs/problems/数组/491-递增子序列.md)
    - [[646][中等][动态规划][贪心] 最长数对链](/docs/problems/数组/646-最长数对链.md)
    - [[673][中等][动态规划][贪心] 最长递增子序列的个数](/docs/problems/数组/673-最长递增子序列的个数.md)
    - [[674][简单][动态规划] 最长连续递增序列](/docs/problems/数组/674-最长连续递增序列.md)
    - [[704][中等][二分] 二分查找](/docs/problems/数组/704-二分查找.md)
    - [[718][中等][动态规划][滑动窗口] 最长重复子数组](/docs/problems/数组/718-最长重复子数组.md)
    - [[873][中等][动态规划] 最长的斐波那契子序列的长度](/docs/problems/数组/873-最长的斐波那契子序列的长度.md)
    - [[1035][中等][动态规划] 不相交的线](/docs/problems/数组/1035-不相交的线.md)
  - [树](/docs/problems/树)
    - [[94][中等] 二叉树的中序遍历](/docs/problems/树/94-二叉树的中序遍历.md)
    - [[102][中等] 二叉树的层序遍历](/docs/problems/树/102-二叉树的层序遍历.md)
    - [[108][简单][DFS] 将有序数组转换为二叉搜索树](/docs/problems/树/108-将有序数组转换为二叉搜索树.md)
    - [[109][中等][DFS][双指针] 有序链表转换二叉搜索树](/docs/problems/树/109-有序链表转换二叉搜索树.md)
    - [[114][中等][DFS] 二叉树展开为链表](/docs/problems/树/114-二叉树展开为链表.md)
    - [[144][中等] 二叉树的前序遍历](/docs/problems/树/144-二叉树的前序遍历.md)
    - [[145][困难] 二叉树的后序遍历](/docs/problems/树/145-二叉树的后序遍历.md)
    - [[173][中等] 二叉搜索树迭代器](/docs/problems/树/173-二叉搜索树迭代器.md)
    - [[297][困难][BFS] 二叉树的序列化与反序列化](/docs/problems/树/297-二叉树的序列化与反序列化.md)
    - [[面试题 04.06][中等][DFS] 后继者](/docs/problems/树/04.06-后继者.md)
    - [[剑指Offer-33][中等][分治] 二叉搜索树的后序遍历序列](/docs/problems/树/剑指Offer-33-二叉搜索树的后序遍历序列.md)
    - [[剑指Offer-36][中等] 二叉搜索树与双向链表](/docs/problems/树/剑指Offer-36-二叉搜索树与双向链表.md)
    - [[剑指Offer-54][简单] 二叉搜索树的第k大节点](/docs/problems/树/剑指Offer-54-二叉搜索树的第k大节点.md)
  - [链表](/docs/problems/链表)
    - [[23][困难][堆] 合并K个排序链表](/docs/problems/链表/23-合并K个排序链表.md)
    - [[142][中等][双指针] 环形链表 II](/docs/problems/链表/142-环形链表-II.md)
  - [数学](/docs/problems/数学)
    - [[223][中等] 矩形面积](/docs/problems/数学/223-矩形面积.md)
    - [[939][中等][哈希] 最小面积矩形](/docs/problems/数学/939-最小面积矩形.md)
  - [数据流](/docs/problems/数据流)
    - [[295][困难][二分][堆] 数据流的中位数](/docs/problems/数据流/295-数据流的中位数.md)
