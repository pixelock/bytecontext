Tmux 是一个**终端复用器**(terminal multiplexer), 是常用的开发工具.

# Tmux 是什么

## 会话与进程

命令行的典型使用方式是, 打开一个终端窗口(terminal window), 在里面输入命令. 用户与计算机的这种临时的交互, 称为一次**会话**(session).

这种方式下, 会话的一个重要特点是, 窗口与其中启动的进程是连在一起的. 打开窗口, 会话开始; 关闭窗口, 会话结束, 会话内部的进程也会随之终止, 不管有没有运行完.

为了解决这个问题, **会话与窗口可以解绑**. 窗口关闭时, 会话并不终止, 而是继续运行, 等到以后需要的时候, **让会话绑定其他窗口**.

## Tmux 的作用

**Tmux 就是会话与窗口的解绑工具, 将它们彻底分离.**

- 它可以让新窗口接入已经存在的会话
- 它允许每个会话有多个连接窗口, 因此可以多人实时共享会话
- 它允许在单个窗口中, 同时访问多个会话, 这对于同时运行多个命令行程序很有用
- 它还支持窗口任意的垂直和水平拆分

# 用法

## 安装

```bash
# Ubuntu 或 Debian
$ sudo apt-get install tmux

# CentOS 或 Fedora
$ sudo yum install tmux

# Mac
$ brew install tmux
```

## 启动和退出

安装完成后, 键入 `tmux` 命令, 就进入了 tmux 窗口.

```bash
tmux
```

上面命令会启动 tmux 窗口, 底部有一个状态栏, 状态栏的左侧是窗口信息(编号和名称), 右侧是系统信息.

![](/resources/images/stack/linux/tmux-1.png)

按下 `Ctrl+d` 或者显式输入 `exit` 命令, 就可以退出 tmux 窗口.

```bash
exit
```

# 会话管理

## 新建会话

第一个启动的 tmux 窗口, 编号是 `0`, 之后每个窗口的编号递增 `1`. 这些窗口对应的会话, 就是 `0` 号会话, `1` 号会话.

使用编号区分会话, 不太直观, 更好的方法是为会话起名.

```bash
tmux new -s <session-name>
```

上面命令新建一个指定名称的会话.

## 分离会话

在 Tmux 窗口中, 输入 `tmux detach` 命令, 就会将当前会话与窗口分离.

```python
tmux detach
```

上面命令执行后, 就会退出当前 Tmux 窗口, **但是会话和里面的进程仍然在后台运行**.

## 查看会话

`tmux ls` 命令可以查看当前所有的 Tmux 会话.

```bash
tmux ls
# or
tmux list-session
```

## 接入会话

`tmux attach` 命令用于重新接入某个已存在的会话.

```bash
# 使用会话编号
tmux attach -t 0

# 使用会话名称
tmux attach -t <session-name>
```

## 杀死会话

`tmux kill-session` 命令用于杀死某个会话.

```bash
# 使用会话编号
$ tmux kill-session -t 0

# 使用会话名称
$ tmux kill-session -t <session-name>
```

## 切换会话

在 tmux 窗口中使用 `tmux switch` 命令用于切换会话.

```bash
# 使用会话编号
$ tmux switch -t 0

# 使用会话名称
$ tmux switch -t <session-name>
```

## 重命名会话

`tmux rename-session` 命令用于重命名会话.

```bash
tmux rename-session -t 0 <new-name>
```

上面命令将0号会话重命名.

# 最简操作流程

以下是 Tmux 的最简操作流程:

- 新建会话tmux new -s my_session
- 在 Tmux 窗口运行所需的程序
- 关闭窗口
- 下次使用时, 重新连接到会话 tmux attach-session -t my_session

# 窗格操作

Tmux 可以将窗口分成多个窗格, 每个窗格运行不同的命令, 以下命令都是在 Tmux 窗口中执行.

## 划分窗格

`tmux split-window` 命令用来划分窗格.

```bash
# 划分上下两个窗格
$ tmux split-window

# 划分左右两个窗格
$ tmux split-window -h
```

## 移动光标

`tmux select-pane` 命令用来移动光标位置.

```bash
# 光标切换到上方窗格
$ tmux select-pane -U

# 光标切换到下方窗格
$ tmux select-pane -D

# 光标切换到左边窗格
$ tmux select-pane -L

# 光标切换到右边窗格
$ tmux select-pane -R
```

## 交换窗格位置

`tmux swap-pane` 命令用来交换窗格位置.

```bash
# 当前窗格上移
$ tmux swap-pane -U

# 当前窗格下移
$ tmux swap-pane -D
```

## 窗格快捷键

```
Ctrl+b %：划分左右两个窗格。
Ctrl+b "：划分上下两个窗格。
Ctrl+b <arrow key>：光标切换到其他窗格。<arrow key>是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键↓。
Ctrl+b ;：光标切换到上一个窗格。
Ctrl+b o：光标切换到下一个窗格。
Ctrl+b {：当前窗格与上一个窗格交换位置。
Ctrl+b }：当前窗格与下一个窗格交换位置。
Ctrl+b Ctrl+o：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。
Ctrl+b Alt+o：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。
Ctrl+b x：关闭当前窗格。
Ctrl+b !：将当前窗格拆分为一个独立窗口。
Ctrl+b z：当前窗格全屏显示，再使用一次会变回原来大小。
Ctrl+b Ctrl+<arrow key>：按箭头方向调整窗格大小。
Ctrl+b q：显示窗格编号。
```

# 窗口管理

除了将一个窗口划分成多个窗格, Tmux 也允许新建多个窗口.

## 新建窗口

`tmux new-window` 命令用来创建新窗口.

```bash
tmux new-window

# 新建一个指定名称的窗口
tmux new-window -n <window-name>
```

## 切换窗口

`tmux select-window` 命令用来切换窗口.

```bash
# 切换到指定编号的窗口
$ tmux select-window -t <window-number>

# 切换到指定名称的窗口
$ tmux select-window -t <window-name>
```

## 重命名窗口

`tmux rename-window` 命令用于为当前窗口起名, 或重命名.

```bash
tmux rename-window <new-name>
```

# 参考资料

- [Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)
