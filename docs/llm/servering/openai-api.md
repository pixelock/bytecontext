# OpenAI API 格式

将 LLM 生成服务, 或者 LLM 网关服务的接口出入参格式与 OpenAI 相关的 API 保持一致, 有一个突出的优点: **可以轻松接入到任何基于 ChatGPT / GPT4 的应用中**. 接下来我们根据 [OpenAI API Document](https://platform.openai.com/docs/api-reference/chat) 来梳理下问答接口的使用方法和格式.

## OpenAI API 使用方法

使用 OpenAI 的模型完成回答, 调用的是 **Chat Completions API**, Chat Completions API 接收一系列的 `Messages` 作为输入, 并将模型生成的 `Message` 作为输出. 虽然 chat 的这种格式是为了多轮对话设计的, 但对于一问一答这种形式也可以简单适用.

Chat Completions API 分为 `Chat Completion` 和 `Chat Completion Stream` 两种形式:

- `Chat Completions`: 模型生成完整的回复后, 一次性返回
- `Chat Completions Stream`: 模型每生成一个 token, 就打包成 chunk 返回. 实际中为了更好的用户体验, 多使用这种形式

### 非流式 Chat Completion

一个简单的多轮对话调用 Chat Completions API 的例子如下:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
```

在 Chat Completions API 的使用中, `model` 和 `messages` 是两个必传的参数. 返回的内容如下:

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        "role": "assistant"
      }
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
```

使用下面的代码提取出模型生成的回复文本:

```python
response['choices'][0]['message']['content']
```

### 流式 Chat Completion Stream

希望 API 流式返回的调用方式, 与非流式的区别, 只在于入参 `stream` 上, 其余无论是使用的接口, 还是其他参数都是一致的.

```python
response = openai.ChatCompletion.create(
    engine="gpt-3.5-turbo",
    messages=[
        {'role': 'user', 'content': "Count to 10. E.g. 1, 2, 3, 4, ..."},
    ],
    stream=True,
)
```

返回的内容被称为一个 chunk, 完整的返回需要一系列的 chunks 组成.

```json
{
  "choices": [
    {
      "delta": {
        "role": "assistant",
        "content": "\n\n"
    },
      "finish_reason": null,
      "index": 0
    }
  ],
  "created": 1680710012,
  "id": "chatcmpl-71zjwZk3EB5fLgVup9S1BPZo73JUk",
  "model": "gpt-35-turbo",
  "object": "chat.completion.chunk",
  "usage": null
}
```

可以看到返回的内容还是在 `choices` 参数中, 由于每次只返回一个 token, 也可以看做一个 `delta`. 区分当前 chunk 是否是最后一个 chunk 是通过 `finish_reason` 是否为空判断的. 当 `finish_reason` 参数不为空时, 收到本次问答的最后一个 token.

## Chat Completion API 参数说明

### Request

#### messages

array, 必传.

代表着对话的一系列 Messages. `Message` 是 OpenAI 定义的装载对话信息的类. 基础的 `Message` 类包含以下的属性:

- content: 文本内容
- role: 这条 message 的角色. 定义了 5 中角色:
  - system
  - user: 用户的输入
  - assistant: 模型生成的返回
  - tool: 调用外部 Tools 返回的内容
  - function: 调用函数需要的入参, 由模型生成, 以 JSON 的格式组织

对于 assistant, tool, function 这三种 message 各自还有独特的参数. 在使用 Function tools 时有用处, 这里不再说明

#### model

string, 必传.

指明使用的模型名称. 如最常用的 `gpt-3.5-turbo` 或支持长上下文的 `gpt-3.5-turbo-16k`.

#### frequency_penalty

float | null, 默认值为 0.

值域为 `[-2.0, 2.0]`. 正值代表着在新生成 tokens 采样时进行惩罚, 惩罚的依据是目前为止所有生成的 tokens 的频率, 降低相同的内容反复出现的可能性.

#### logit_bias

map | null, 默认值为 null.

调整指定 tokens 的采样概率. map 的 key 为对应 token 的 id, value 为一个 `[-100, 100]` 范围内的值作为 bias, 这个 bias 将直接加和到 token id 对应的 logits 上, 用来影响采样的概率.

设置的值在 `[-1, 1]` 之间, 就可以有效地影响采样了. 设置 `-100` 或 `100` 这样的极端值, 可以完全避免或指定生成对应的 token.

#### max_tokens

int | null, 默认值为 `inf`.

在 chat completion 中允许生成 tokens 数量的上限. 注意这里控制的只是生成的 tokens 数量, 而**不是**输入 + 输出的总 tokens 数量.

但输入 + 输出的总 tokens 数量应当符合模型的 **context length** 限制. 这在每个模型中往往是不一样的.

#### n

int | null, 默认值为 `1`.

**How many chat completion choices to generate for each input message**.

#### presence_penalty

float | null, 默认值为 0.

值域为 `[-2.0, 2.0]`. 根据 token 是否在当前的上下文中出现过, 正数将提升没有出现过的 token 的采样概率.

#### seed

int | null, 默认值为 0.

系统会尽量保证在相同的 `seed` 设定下, 生成的返回时可以重复的.

#### stop

string | array | null, 默认值为 null.

最多设置 4 个 token, 如果模型生成到指定的 token, 就会提前结束生成, 并返回.

#### stream

bool | null, 默认值为 false.

是否采用流式传输.

流式传输模式下, 最后收到的一个 chunk 中, 对应的内容为 `data: [DONE]`.

#### temperature

float | null, 默认值为 1.

采样温度, 值域为 `[0, 2]`. 更高的值代表着模型生成越随机. 如果设置像 0.2 这种较小的值, 相同的输入生成的结果越固定.

下面还会介绍 `top_p` 参数, 也是用来控制采样的参数. 但 OpenAI 不建议两个参数同时使用.

#### top_p

float | null, 默认值为 1.

控制 `nucleus sampling` 的参数. 模型只考虑从累加概率在 `top_p` 范围内的 tokens, 长尾部分被忽略.

0.1 means only the tokens comprising the top 10% probability mass are considered.

#### tools

array.

列举了模型可以调用的 Tools. 这里的 Tools 指的是 functions, 因此这个参数是用来提供模型可以调用的函数, 以及这些函数需要的入参名称信息, 以 JSON 的格式传入.

一个完整的 tools 参数可以设置为:

```json
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                },
            },
            "required": ["location"],
        },
    },
}
```

### Response

#### chat completion object

非流式返回的 Response 结构. 一个例子:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0613",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

##### id

string.

代表这次 chat completion 的唯一 id.

##### choices

array.

返回的是一个列表, 代表着当前输入的多种可能的回复. 列表的大小由入参中的 `n` 参数决定. 一般 `n` 设置为 `1`, 此时对应的列表中回复的数量为 1.

一条回复的格式为:

- finish_reason: string. 模型停止继续生成的原因, 有以下几种:
  - stop: 模型自然输出到结束符号, 或命中到 `stop` 入参中指定的作为停止使用的 token
  - length: 触发了最大生成 tokens 数量的限制, 由 `max_tokens` 入参决定
  - content_filter: 触发过滤机制
  - tool_calls: 模型调用 function
- index: int. choices 列表中的 index
- message: object. 返回内容的主体, 有以下几个属性
  - content: string | null. 返回消息的文本内容
  - role: string. 这条消息的角色
  - tool_calls: array of object. 调用了哪些 functions
    - id: function tool 的 id
    - type: 目前只有 `function` 这一种 tool
    - function: object. 模型调用的 function
      - arguments: string. 函数入参, JSON 格式
      - name: string. 函数名称

##### created

int.

这条 chat completion 创建的时间(Unix timestamp).

##### model

string.

模型名称. 与入参一致.

##### system_fingerprint

string.

代表了系统后台的配置.

##### object

string.

值一直是 `chat.completion`.

##### usage

object.

本次 completion 的一些统计值. 包含以下属性:

- prompt_tokens: 输入的 prompt 对应的 tokens 数量
- completion_tokens: 生成文本对应的 tokens 数量
- total_tokens: 上面两者的总和

#### chat completion chunk object

流式返回的 Response 结构. 与非流式的主要区别在于 `choices` 参数中的内容结构不同. 一个返回参数的例子, 每一行代表了一次返回的 chunk:

```json
{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

....

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":" today"},"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"?"},"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
```

##### id

string.

代表这次 chat completion 的唯一 id. chunks 之间使用相同的 id.

##### choices

array.

流式与非流式返回的主要区别就在这里. 非流式返回的是 `index`, `finish_reason` 和 `message` 三部分. 在流式中, 代表完整返回内容的 `message` 字段被代表逐个 token 的 `delta` 替换. 下面是 `delta` 参数的结构.

**delta**

object.

- content: string | null. chunk 中的文本内容, 通常是一个 token 代表的文本
- role: string. 这条消息的角色
- tool_calls. 同非流式调用

##### created

int.

这条 chat completion 创建的时间(Unix timestamp). 不同的 chunks 之间的值是一样的.

##### model

string.

模型名称. 与入参一致.

##### system_fingerprint

string.

代表了系统后台的配置.

##### object

string.

值一直是 `chat.completion.chunk`.

# 参考资料

- [Chat - OpenAI API Document](https://platform.openai.com/docs/api-reference/chat)
- [Text generation models - OpenAI API Document](https://platform.openai.com/docs/guides/text-generation/chat-completions-api)
- [LLaMA-Factory 中的抽象实现](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/api/app.py)
