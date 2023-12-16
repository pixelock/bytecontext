# OpenAI Chat Style API 中的坑

## 调用时报错: pip install fschat

### 问题描述

vllm serving 可以正常启动, 但是在调用时会报错:

```
ModuleNotFoundError: fastchat is not installed. Please install fastchat to use the chat completion and conversation APIs: `$ pip install fschat`
```

在安装 `fschat` 后仍然报同样的错误.

### 问题原因

报错由下面的[代码](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py#L74)产生:

```python
if not _fastchat_available:
    raise ModuleNotFoundError(
        "fastchat is not installed. Please install fastchat to use "
        "the chat completion and conversation APIs: `$ pip install fschat`"
    )
```

触发是因为:

```python
try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False
```

定位到是因为代码没有兼容 `fastchat` 的最新版本导致的. 因此需要将 `fschat` 回退到合适的老版本.

### 解决方法

[Github Issues](https://github.com/vllm-project/vllm/issues/855)

使用低版本的 `fschat`:

```bash
pip install fschat==0.2.23
```
