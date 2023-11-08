# Python 中的 fire 是干什么的

`fire`是 Google 开源的一个可从任何 Python 代码自动生成命令行接口（CLI）的库。不需要做任何额外的工作，只需要从主模块中调用`fire.Fire()`，它会自动将你的代码转化为CLI指令。

- Python Fire 是一种在 Python 中创建 CLI 的简单方法
- Python Fire 使 Bash 和 Python 之间的转换更为容易

## 与`argparse`的区别

众所周知，`argparse`包也可以完成为执行的脚本在命令行中指定参数的功能，那么`fire`相对于`argparse`的区别以及优势是什么呢？

例如我们有一个最简单的打印文本的函数：

```python
def print_anything(text):
    print(text)
```

如果我们希望在命令行中传参，借助`argparse`，实现如下：

```python
import argparse

def print_anything(text):
    print(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', default='', type=str, help='input any text')
    args = parser.parse_args()
    print_anything(args.text)
```

使用`argparse`，我们想传什么样的函数，就需要显式定义参数，如果参数量很多，定义参数的过程就非常繁琐。

## 使用`fire`实现

安装 `fire` 模块。

```bash
pip install fire
```

直接在主模块中调用`fire.Fire()`。

```python
import fire

def print_anything(text):
    print(text)

if __name__ == '__main__':
    fire.Fire(print_anything)
```

使用下面的指令

```bash
python demo.py hello
# 或
python demo.py --text hello
```

都能够得到打印`hello`的结果

# `fire` 的灵活性

## 单个函数

上面演示的是单个函数的使用方法。我们可以使用`--help`指令，查看详细的情况：

```bash
python demo.py --help
```

得到如下的结果:

```bash
NAME
    demo.py

SYNOPSIS
    demo.py TEXT

POSITIONAL ARGUMENTS
    TEXT

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

可以在 `SYNOPSIS` 提要中看到 `text` 是必传参数。

接下来我们定义一个入参更复杂一些的参数，并指明参数类型，默认值和参数说明等。

```python
import fire

def add_mul(first: int, second:int, cal_type: str = 'add') -> int:
    """add or multiply two numbers.

    :param first: first number
    :param second: second number
    :param cal_type: calculation type, must be either of `add` or `mul`
    :return: calculation result
    """
    assert cal_type in ('add', 'mul')

    if cal_type == 'add':
        return first + second
    else:
        return first * second

if __name__ == '__main__':
    fire.Fire(add_mul)
```

使用`--help`指令查看详细的情况：

```bash
NAME
    demo.py - add or multiply two numbers.

SYNOPSIS
    demo.py FIRST SECOND <flags>

DESCRIPTION
    add or multiply two numbers.

POSITIONAL ARGUMENTS
    FIRST
        Type: int
        first number
    SECOND
        Type: int
        second number

FLAGS
    -c, --cal_type=CAL_TYPE
        Type: str
        Default: 'add'
        calculation type, either be one of `add` or `mul`

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

可以看到 `first` 和 `second` 是两个必传的参数，这两个参数的类型和解释在 `POSITIONAL ARGUMENTS` 中有说明。 带有默认值的 `cal_type` 为非必传参数，在 `FLAGS` 中有类型，默认值，参数解释等说明，以及 `fire` 为这个参数自动生成了 `-c` 和 `--cal_type` 两个 CLI 中的指令。

## 多个函数

定义多个函数：

```python
import fire


def add(first_number, second_number):
    return first_number + second_number


def subtract(first_number, second_number):
    return first_number - second_number


def multiply(first_number, second_number):
    return first_number * second_number


def divide(first_number, second_number):
    return first_number / second_number


if __name__ == '__main__':
    fire.Fire()
```

`fire.Fire()` 中也可以不传函数，我们使用 `--help` 指令来看看怎么使用，以下是输出的结果。

```bash
NAME
    demo.py

SYNOPSIS
    demo.py GROUP | COMMAND

GROUPS
    GROUP is one of the following:

     fire
       The Python Fire module.

COMMANDS
    COMMAND is one of the following:

     add

     subtract

     multiply

     divide
```

可以看到，代码中的函数转换成了指令，四个函数在 `COMMANDS` 中罗列。再看看使用方法，比如说使用乘法。

```bash
python demo.py multiply 3 4
# 或
python demo.py multiply --first_number=3 --second_number=4
# 或
python demo.py multiply --first_number 3 --second_number 4

# Output:
# 12
```

我们为其中的一个函数 `add` 增加内容，如下：

```python
def add(a: int, b: int = 0):
    """add two numbers
    
    :param a: first number
    :param b: second number
    :return: 
    """
    return a + b
```

使用如下的指令可以**查看每个函数/指令**的详细情况：

```bash
python demo.py add --help
```

结果如下：

```bash
NAME
    demo.py add - add two numbers

SYNOPSIS
    demo.py add A <flags>

DESCRIPTION
    add two numbers

POSITIONAL ARGUMENTS
    A
        Type: int
        first number

FLAGS
    -b, --b=B
        Type: int
        Default: 0
        second number

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

与上面单个函数一节中的机制相同。

## 类支持 / 函数的链式调用

`fire.Fire()` 也可以传入类：

```python
import fire


class Calculator(object):
    def __init__(self, init_number):
        self.init_number = init_number
        self.result = self.init_number

    def __str__(self):
        return str(self.result)

    def add(self, number):
        self.result = self.result + number
        return self

    def subtract(self, number):
        self.result = self.result - number
        return self

    def multiply(self, number):
        self.result = self.result * number
        return self

    def divide(self, number):
        self.result = self.result / number
        return self

    def to_integer(self):
        self.result = int(self.result)
        return self


if __name__ == '__main__':
    fire.Fire(Calculator)
```

使用 `--help` 查看情况：

```bash
NAME
    demo.py

SYNOPSIS
    demo.py --init_number=INIT_NUMBER

ARGUMENTS
    INIT_NUMBER
```

这里只看到了初始化需要的参数，这是因为 `Calculator` 类的初始化方法中包含一个必传的参数，使用下面的指令进行初始化，再使用 `--help` 方法查看：

```bash
python demo.py --init_number 0 --help
```

```bash
SYNOPSIS
    demo.py --init_number 0 COMMAND | VALUE

COMMANDS
    COMMAND is one of the following:

     add

     divide

     multiply

     subtract

     to_integer

VALUES
    VALUE is one of the following:

     init_number

     result
```

这样可以看到类的方法（`COMMANDS`）以及属性（`VALUES`）。这里要介绍 `fire` 提供的命令行中的链式调用方法，即使用命令行一次调用多个方法，例如我们用数值 `1` 初始化，再加上 `2`，再乘以 `3`，最后得到的结果应为 `9`，来验证一下：

```bash
python demo.py  --init_number 1 - add 2 - multiply 3
# 或
python demo.py  --init_number 1 add 2 multiply 3
```

链式方法的使用方法很简单，在使用参数初始化类之后，直接在后面跟要执行的函数名称，紧跟以该函数需要的参数。执行多个函数在命令堆叠函数即可。