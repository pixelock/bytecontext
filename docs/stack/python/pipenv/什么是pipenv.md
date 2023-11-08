# `pipenv`的优势

`pipenv`结合了`Pipfile`, `pip`, `virtualenv`等项目环境管理工具, 目的是**有效管理Python多个环境, 各种包**.

> Pipfile是社区拟定的依赖管理文件, 用于替代过于简陋的`requirements.txt`文件. `Pipfile`文件是`TOML`格式而不是`requirements.txt`这样的纯文本

## 对比`virtualenv`的优势

使用`virtualenv`来创建并管理虚拟环境. 通过`pip freeze`生成`requirements.txt`文件, 然后通过`pip install -r requirements.txt`进行项目模块的管理与安装. `virtualenv`的一些问题:

- 需要手动执行命令来更新`requirements.txt`文件: 如果环境中的包有更新, 需要手动再次执行`pip freeze`来生成环境文件, 但`pipenv`带来了自动化的管理
- `virtualenv`能够很好地完成每个虚拟环境的搭建和`python`版本的管理. 但跨平台的使用不太一致, 特别是在处理**包之间的依赖**总存在问题.
- 且`virtualenv`使用`pip`管理包, 但简单的`requirements.txt`文件无法记录包之间的依赖关系, 这会偶尔导致一些问题, 例如:
  - 在使用pip安装包时, 会发现在安装的时候会安装其它的依赖包, 但当我们用pip移除一个包时, 却只移除了指定的包
- `pipenv`使用`Pipfile`和`Pipfile.lock`, 前者存放环境信息, 后者存放将包的依赖关系, 查看依赖关系十分方便. `pipenv`对包的管理更近一步, 甚至可以通过指令`pipenv graph`查看包之间的依赖图
