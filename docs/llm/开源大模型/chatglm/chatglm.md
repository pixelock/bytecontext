# 从 GLM 到 ChatGLM

需要解决的任务, 以自然语言的指令, 即 Prompt 写出. 而语言模型很多时候对 Prompt 都非常敏感. Chat 类模型主要是为了降低 prompt 的难度诞生, 但 prompt 本身仍然可以有较大的影响. 询问同一个模型, 使用不同的问法, 往往会得到不同的答案.

GPT, GLM 这种**基座语言模型**, 输入一个问题, 续写生成后面文本, 一般来说基座模型的行为是对这个问题进行续写, 或者补充一些信息. 但我们期待的是对问题进行回答. 下图中我们对模型提问 `Explain the moon landing to a 6 year old in a few sentences.`, 我们期待地是对问题进行回答, 但 GLM(下左) 和 GPT-3(下右) 的回答是对问题进行了改写续写.

![](/resources/images/llm/glm-7.png)

训练 Chat 类模型的秘诀, 是通过 RLHF 训练, 让模型能更好的理解人类的意图, 并且将输入更好地与人类偏好对齐.

产生这种现象的原因是:

基座模型是在大规模**互联网语料**上进行训练的. 而互联网语料, 它并不是高质量分布的语料, 并且噪音很大, 很多内容并非符合人类偏好(e.g. 广告), 也不是回答问题的形式. 通过人类反馈, 引入人类的知识, 将模型的输出分布, 调整到人类偏好的高质量数据分布上.

OpenAI 在 2020 年发布 [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) 中, 在文本摘要的任务上, 通过收集人类的反馈数据(偏好排序, 收集了 6.4k 对), 训练了一个奖励模型用户模拟人类反馈, 然后使用训练得到的奖励模型提供反馈, 通过强化学习来微调语言模型, 提升模型生成质量.

通过奖励模型, 有限地避免了收集人类的反馈是有限且昂贵的问题.

# ChatGLM-6B 模型结构代码解析

从最外层的 `ChatGLMModel` 开始, 一层层向内剖析.

## ChatGLMModel

```python
class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init

        # 保存各类参数
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.params_dtype = torch.half
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.position_encoding_2d = config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection

        # 初始化 word embedding 层
        self.word_embeddings = init_method(
            torch.nn.Embedding,
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            dtype=self.params_dtype
        )
        self.gradient_checkpointing = False
​
        def get_layer(layer_id):
            return GLMBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=LayerNorm,
                use_bias=True,
                params_dtype=self.params_dtype,
                position_encoding_2d=self.position_encoding_2d,
                empty_init=empty_init
            )

        # 堆叠GLMBlock
        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(self.num_layers)]
        )
​
        # 最后的Layer Norm层
        self.final_layernorm = LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)
​
    def get_input_embeddings(self):
        return self.word_embeddings
​
    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(CHATGLM_6B_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        ### (开始)一些输入输出和参数设置, 可以忽略
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
​
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
​
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        ### (结束)一些输入输出和参数设置, 可以忽略
        
        # embedding 层, 获得 word embeddings 表示
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
​
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

            # 获得注意力mask，该功能继承自ChatGLMPreTrainedModel
            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device
                )
                
            if position_ids is None:
                MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
                seqs = input_ids.tolist()  # GPU to CPU

                # 记录input_ids中是否使用了mask以及mask的位置
                # mask_positions记录每个样本中mask的位置
                # use_gmasks记录是否使用了gMask
                # 这是训练阶段使用的逻辑
                mask_positions, use_gmasks = [], []
                for seq in seqs:
                    mask_token = gMASK if gMASK in seq else MASK  # 训练样本中, 只会包含 MASK / gMASK 中的一种
                    use_gmask = mask_token == gMASK
                    mask_positions.append(seq.index(mask_token))
                    use_gmasks.append(use_gmask)

                 # 获得position_ids，该功能继承自ChatGLMPreTrainedModel
                position_ids = self.get_position_ids(
                    input_ids,
                    mask_positions=mask_positions,
                    device=input_ids.device,
                    use_gmasks=use_gmasks
                )
​
        hidden_states = inputs_embeds.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if attention_mask is None:
            attention_mask = torch.zeros(1, 1, device=input_ids.device).bool()
        else:
            attention_mask = attention_mask.to(hidden_states.device)  # attention_mask 由外部传入, 在 dataset 的输入中拼装好
            
        # 模型的前向传播
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values[i]  # 这一层的 KV cache
​
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    torch.tensor(i),
                    layer_past,
                    use_cache,
                    output_attentions
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    layer_id=torch.tensor(i),
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
​
            hidden_states = layer_ret[0]
​
            if use_cache:
                presents = presents + (layer_ret[1],)
​
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_ret[2 if use_cache else 1],)
​
        # 最终的Layer Norm
        hidden_states = self.final_layernorm(hidden_states)
​
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
​
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
​
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
```

## ChatGLMPreTrainedModel

ChatGLMPreTrainedModel 中定义了 attention_mask 和 position_id 的生成方法.

### Attention Mask

由于 ChatGLM-6B 使用的是 prefix-LM 形式, 对于输入的前缀使用双向注意力; 对于后续的生成部分则是Causal Mask.

![](/resources/images/llm/glm-2.png)

```python
def get_masks(self, input_ids, device):
    batch_size, seq_length = input_ids.shape

    # context_lengths记录了batch中每个样本的真实长度
    context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]

    # 生成causal mask，即下三角以及对角线为1，上三角为0
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()

    # 将前缀部分的注意力改为双向
    # 对 batch 内每个样本单独处理
    for i, context_length in enumerate(context_lengths):
        attention_mask[i, :, :context_length] = 1
    attention_mask.unsqueeze_(1)  # 增加 head 维度 (batch_size, num_heads, seq_length, seq_length)
    attention_mask = (attention_mask < 0.5).bool()
        
    return attention_mask
```

### Position ID

GLM 使用的是二维位置编码. 代码中, `position_ids` 是 Position 1; `block_position_ids` 是 Position 2.

在实际操作中, 每个样本中只能有一个 MASK token.

![](/resources/images/llm/glm-3.png)

```python
def get_position_ids(self, input_ids, mask_positions, device, use_gmasks=None):
    """
    input_ids: [batch_size, seq_length]
    mask_positions: [batch_size]，由于GLM系列中会使用[Mask]或[gMask]标志，mask_positions就是指这些标注的具体位置
    """
    batch_size, seq_length = input_ids.shape
    if use_gmasks is None:
        use_gmasks = [False] * batch_size

    # context_lengths：未被padding前，batch中各个样本的真实长度
    context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]  # GPU to CPU

    # 2维位置编码
    if self.position_encoding_2d:
        """
        Position 1
        """
        # 先不区分前缀和生成部分, 规划出 Position 1 的基本格式, 对于每个样本, 对应的值为:
        # [0,1,2,...,seq_length-1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

        # 将前缀输入后的所有位置的postion id都设置为[Mask]或者[gMask]的位置id
        # 每个样本中只能有一个 MASK token
        # 对 batch 内每个样本单独操作
        for i, context_length in enumerate(context_lengths):
            position_ids[i, context_length:] = mask_positions[i]

        """
        Position 2
        """
        # 输入中的前缀部分的位置编码全部设置为0，待生成的位置添加顺序的位置id
        # 例如：[0,0,0,0,1,2,3,4,5]
        block_position_ids = [torch.cat((
            torch.zeros(context_length, dtype=torch.long, device=device),
            torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
        )) for context_length in context_lengths]
        block_position_ids = torch.stack(block_position_ids, dim=0)

        # 将postion_ids和block_position_ids堆叠在一起，用于后续的参数传入；
        # 在注意力层中，还有将这个position_ids拆分为两部分
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)  # (batch_size, 2, seq_length)
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        for i, context_length in enumerate(context_lengths):
            if not use_gmasks[i]:
                position_ids[i, context_length:] = mask_positions[i]
​
    return position_ids
```

## GLMBlock

GLMBlock的基本结构为: LN, Self Attention(残差链接), LN, GLU(残差链接). 这里的 GLU 就是 Transformer 中的 FFN, 换了个叫法.

![](/resources/images/llm/glm-8.png)

```python
class GLMBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            layernorm_epsilon,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            layernorm=LayerNorm,
            use_bias=True,
            params_dtype=torch.float,
            num_layers=28,
            position_encoding_2d=True,
            empty_init=True
    ):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.
​
        self.layer_id = layer_id
​
        # LayerNorm层
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        # 是否使用2维位置编码
        self.position_encoding_2d = position_encoding_2d
        # 自注意力层
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
            params_dtype=params_dtype,
            position_encoding_2d=self.position_encoding_2d,
            empty_init=empty_init
        )
​
        # Post Layer Norm层
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        self.num_layers = num_layers
​
        # GLU层
        self.mlp = GLU(
            hidden_size,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            params_dtype=params_dtype,
            empty_init=empty_init
        )
​
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids,
            attention_mask: torch.Tensor,
            layer_id,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """
        # 这里进来的输入 `hidden_states` 对应的形状为 [seq_len, batch, hidden_size]
​
        # 对输入进行Layer Norm
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)
        # 自注意力
        attention_outputs = self.attention(
            attention_input,
            position_ids,
            attention_mask=attention_mask,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]

        alpha = (2 * self.num_layers) ** 0.5  # 使用了 DeepNorm, 残差链接时, 需要对原始输入先进行缩放, 缩放系数 alpha 这里由模型总层数决定
        hidden_states = attention_input * alpha + attention_output  # 自注意力的输出和输入残差连接
    
        # Layer Norm
        mlp_input = self.post_attention_layernorm(hidden_states)
        # 全连接层投影
        mlp_output = self.mlp(mlp_input)
        # MLP层的输出和输入残差连接
        output = mlp_input * alpha + mlp_output  # GLU 层的输出和输入残差连接, 输入用同样的 alpha 系数缩放
        
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
​
        return outputs  # hidden_states, present, attentions
```

### SelfAttention

在计算 attention score 时需要使用二维位置编码对 Query 和 Key 张量进行处理. 然后在计算內积时需要处理 MASK.

ChatGLM-6B 中 `SelfAttention` 中主要完成: 为query和key注入RoPE位置信息, 然后调用attention_fn实现注意力机制.

需要特别注意的是二维 position id 的融合方式, 两种不同的 position id 分别与 query 和 key 向量中不同的部分进行融合.

```python
class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=torch.float, position_encoding_2d=True, empty_init=True):
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        super(SelfAttention, self).__init__()
​
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        # position_encoding_2d：是否使用2维的位置编码
        self.position_encoding_2d = position_encoding_2d
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2) if position_encoding_2d else self.hidden_size // self.num_attention_heads,
            base=10000,
            precision=torch.half,
            learnable=False,
        )
​
        self.scale_mask_softmax = None
​
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
​
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
​
        # query、key、value的投影层
        self.query_key_value = init_method(
            torch.nn.Linear,
            hidden_size,
            3 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
​        
        # output 的投影层
        self.dense = init_method(
            torch.nn.Linear,
            self.inner_hidden_size,
            hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
​
    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        attention_scores.masked_fill_(attention_mask, -10000.0)  # 赋一个很小的值, 计算 softmax 之后对应位置的值为0
        return attention_scores
​
    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                    contiguous_split_chunks=False):
        """沿最后一个维度切分tensor
        参数:
            tensor: 输入tensor；
            num_partitions: 切分tensor的数量；
            contiguous_split_chunks: 若为True,切分的块在内存中连续；
        """
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # torch.split并不会默认创建连续的tensor
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)
​
        return tensor_list
​
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids,
            attention_mask: torch.Tensor,
            layer_id,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """
        # 一次性得到投影的Q、K、V，减少执行矩阵乘法的次数
        # [seq_len, batch, 3 * hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)
        
        # 拆分出多头
        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)
        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        # 经过下面一步, query_layer、key_layer、value_layer已经是拆分出多头的Q、K、V
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)
​
        if self.position_encoding_2d:
            ## 这里将query和key都拆分为两份, 一半用来融合 Position 1 信息, 另一半用来融合 Position 2 信息, 然后再拼接在一起
            # 拆分
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            # 计算cos和sin值
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            # 输入的 position_ids 为二维位置, 按第二维拆开
            position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                position_ids[:, 1, :].transpose(0, 1).contiguous()
            # 将两种位置编码输入到不同的query和key上
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            # 拼接注入不同位置信息的query和key，这样query和key中包含了两种位置信息
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:
            # 普通的RoPE
            position_ids = position_ids.transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)
​
        # [seq_len, batch, hidden_size]
        context_layer, present, attention_probs = attention_fn(
            self=self,
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
            hidden_size_per_partition=self.hidden_size_per_partition,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache
        )
​
        output = self.dense(context_layer)
​
        outputs = (output, present)
​
        if output_attentions:
            outputs += (attention_probs,)
​
        return outputs  # output, present, attention_probs
```

#### attention_fn

```python
def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        hidden_size_per_partition,
        layer_id,
        layer_past=None,
        scaling_attention_score=True,
        use_cache=False,
):
    # (推理场景)中, 将传递来的key和value合并至当前的Q和K上, 即合并 KV cache
    if layer_past is not None:
        past_key, past_value = layer_past[0], layer_past[1]
        key_layer = torch.cat((past_key, key_layer), dim=0)  # seq_len 在第一个维度, 所以是序列维度合并
        value_layer = torch.cat((past_value, value_layer), dim=0)
​
    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    seq_len, b, nh, hidden_size = key_layer.shape
​
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None
        
    # 对query层进行scaling
    query_key_layer_scaling_coeff = float(layer_id + 1)
    if scaling_attention_score:
        # 特别的地方是, 每一层对 attention score 进行 scale 的大小不同, 越高的层缩小的越厉害
        query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)
​
    # 注意力分数的输出形状: [batch_size, num_heads, seq_length, seq_length]
    output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))
    
    # 形状重塑：[seq_length, batch_size, num_heads, head_dim] ->
    # [seq_length, batch_size*num_heads, head_dim]
    query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
    key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
​
    matmul_result = torch.zeros(
        1, 1, 1,
        dtype=query_layer.dtype,
        device=query_layer.device,
    )
    
    # 计算非规范化的注意力分数，matmul_result形状为[batch_size*num_head, seq_length,seq_length]
    matmul_result = torch.baddbmm(
        matmul_result,
        query_layer.transpose(0, 1),  # [b * np, sq, hn]
        key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0,
        alpha=1.0,
    )
​
    # 重塑形状为:
    # [batch_size*num_head,seq_length,seq_length]
    # [batch_size,num_head,seq_length,seq_length]
    attention_scores = matmul_result.view(*output_size)
    
    # 对注意分数进行缩放和规范化
    if self.scale_mask_softmax:
        self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
    else:
        # 对注意力分数进行mask
        if not (attention_mask == 0).all():
            attention_scores.masked_fill_(attention_mask, -10000.0)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff
​
        attention_probs = F.softmax(attention_scores, dim=-1)
​
        attention_probs = attention_probs.type(dtype)
​
    ### 使用注意力分数对value进行加权求和
    # (batch_size, num_attention_heads, seq_length, hidden_size_per_attention_head)
    output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
    # 重塑value的形状
    # (seq_length, batch_size * num_attention_heads, hidden_size_per_attention_head)
    value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
    # 重塑注意力分数的形状
    # (batch_size * num_head, seq_length, seq_length)
    attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
    # 注意力分数乘以value，得到最终的输出context
    context_layer = torch.bmm(
        attention_probs,  # (batch_size * num_attention_heads, seq_length, seq_length)
        value_layer.transpose(0, 1)  # (batch_size * num_attention_heads, seq_length, hidden_size_per_attention_head)
    )  # (batch_size * num_attention_heads, seq_length, hidden_size_per_attention_head)
    # # (batch_size, num_attention_heads, seq_length, hidden_size_per_attention_head)
    context_layer = context_layer.view(*output_size)
    # (seq_length, batch_size, num_attention_heads, hidden_size_per_attention_head)
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (hidden_size_per_partition,)
    # (seq_length, batch_size, hidden_size)
    context_layer = context_layer.view(*new_context_layer_shape)
​
    outputs = (context_layer, present, attention_probs)
​
    return outputs
```

计算 attention_scores 使用了 `torch.baddbmm()`. 注意力分数加权汇总 values 时, 使用了 `torch.bmm()` 方法.

其中 [**TORCH.BADDBMM**](https://pytorch.org/docs/stable/generated/torch.baddbmm.html) 计算如下:

> `torch.baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) → Tensor`

$$
\text{out}_i = \beta \text{input}_i + \alpha (\text{batch1}_i \text{@} \text{batch2}_i)
$$

代码中的使用方法如下:

```python
matmul_result = torch.baddbmm(
    matmul_result,
    query_layer.transpose(0, 1),  # [b * np, sq, hn]
    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    beta=0.0,
    alpha=1.0,
)
```

[**TORCH.BMM**](https://pytorch.org/docs/stable/generated/torch.bmm.html?highlight=torch+bmm#torch.bmm) 的意义是提供 batch 批量计算的矩阵乘法计算, 计算如下:

> `torch.bmm(input, mat2, *, out=None) → Tensor`

$$
\text{out}_i = \text{input}_i \text{@} \text{mat2}_i
$$

#### RoPE 位置编码

在用法上, RoPE的目标是构建一个位置相关的投影矩阵, 使得:

$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
$$

$$\boldsymbol{q}$$ 和 $$\boldsymbol{k}$$ 分别对应注意力机制中的query和key向量, m和n代表两个位置, $$\boldsymbol{R}_{i}$$ 表示位置i处的投影矩阵, 以位置 $$m$$ 为例, 对应的矩阵为:

$$
R_{\theta,m}^{d} = 
\begin{pmatrix} 
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ 
\end{pmatrix}
$$

其中 $$d$$ 是query和key的维度, $$\theta$$ 是一个超参数, 通常 $$\theta$$ 会设置为:

$$
\theta=\left\{\theta_i=10000^{\frac{-2(i-1)}{d}}, i \in\left[1,2, \ldots, \frac{d}{2}\right]\right\}
$$

由于矩阵 $$\boldsymbol{R}$$ 非常稀疏, 为了提供运算速度, 作者也给出了向量逐位对应相乘实现方式:

$$
\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix}
$$

ChatGLM-6B实现采用了PaLM的实现方式, 区别于上面连续的两个数为一组(如 $$q_0$$, $$q_1$$), 这里以跨度为 $$\frac{d}{2}$$ 的两个数(如 $$q_0$$, $$q_{d/2}$$)为一组:

$$
\left[\begin{array}{c}
q_0 \\
\vdots \\
q_{d / 2-1} \\
q_{d / 2} \\
\vdots \\
q_{d-1}
\end{array}\right] \otimes\left[\begin{array}{c}
\cos m \theta_0 \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_0 \\
\vdots \\
\cos m \theta_{d / 2-1}
\end{array}\right]+\left[\begin{array}{c}
-q_{d / 2} \\
\vdots \\
-q_{d-1} \\
q_0 \\
\vdots \\
q_{d / 2-1}
\end{array}\right] \otimes\left[\begin{array}{c}
\sin m \theta_0 \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_0 \\
\vdots \\
\sin m \theta_{d / 2-1}
\end{array}\right]
$$

所以实际操作的时候, 将query或者key从中间分为两组, 然后分别与 cos 和 sin 值相乘后汇总. 这种方法该仍然满足对称性:

$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
$$

在代码中, `RotaryEmbedding` 负责预先计算 sin 和 cos: `rotate_half` 负责上式第二项中, 互换输入向量的上下部分并取负的操作; `apply_rotary_pos_emb_index` 则是对输入的 query 和 key 注入RoPE.

```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        # 预先计算好上面的theta
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        # learnable的效果并没有更好，通常learnable为False
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision
​
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass
​
    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # 这里使用了爱因斯坦求和约定，该操作就是t和self.inv_freq的外积
            # freqs中保存了所有的m\theta。e.g. 第一行是0\theta、第二行是1\theta
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # 根据上面的公式，每个\theta都需要两份，所以这里将两个freqs拼接起来
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
​
            # [seq_length, 1 (b * np), hn]
            # 计算cos和sin
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            # 缓存结果，方便重复利用
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
​
    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)
​
​
def rotate_half(x):
    # x1是x的前半部分，x2是x的后半部分
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    # 前后互换，且后半部分取负
    return torch.cat((-x2, x1), dim=x1.ndim - 1)
​
@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # cos, sin: (seq_len, 1, hidden_size)
    # q, k: (seq_len, batch_size, hidden_size)
    # position_id: (seq_len, batch_size)
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    # cos, sin: (seq_len, batch_size, 1, hidden_size)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k
```

在 Self Attention 中的用法是:

```python
self.rotary_emb = RotaryEmbedding(
    self.hidden_size // (self.num_attention_heads * 2) if position_encoding_2d else self.hidden_size // self.num_attention_heads,
    base=10000,
    precision=torch.half,
    learnable=False,
)

# 计算cos和sin值
cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
# 输入的 position_ids 为二维位置, 按第二维拆开
position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
    position_ids[:, 1, :].transpose(0, 1).contiguous()
# 将两种位置编码输入到不同的query和key上
q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
```

### GLU

虽然在实现代码中命名为GLU, 但这里实现的还是两层 MLP, 使用的激活函数为 GELU:

$$
\text{GLU}(X) = \text{GELU}(XW_1)W_2
$$

```python
class GLU(torch.nn.Module):
    def __init__(self, hidden_size, inner_hidden_size=None,
                 layer_id=None, bias=True, activation_func=gelu, params_dtype=torch.float, empty_init=True):
        super(GLU, self).__init__()
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        self.layer_id = layer_id
        self.activation_func = activation_func
​
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = init_method(
            torch.nn.Linear,
            self.hidden_size,
            self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = init_method(
            torch.nn.Linear,
            self.inner_hidden_size,
            self.hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
​
    def forward(self, hidden_states):
        """
        hidden_states: [seq_len, batch, hidden_size]
        """
​
        # [seq_len, batch, inner_hidden_size]
        # 投影
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # 激活
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # 投影
        output = self.dense_4h_to_h(intermediate_parallel)
​
        return output
```

#### GELU

ChatGLM-6B使用的激活函数为GELU, 其可以近似实现为:

$$
\text{GELU} = 0.5x(1 + \tanh(\sqrt{\frac{2}{\pi}}(x + 0.044715x^{3})))
$$

```python
@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
​
def gelu(x):
    return gelu_impl(x)
```
