# 使用`pipenv`对项目运行环境进行管理

## 创建虚拟环境

切换到项目的根目录, 使用命令创建一个新的虚拟环境:

```sh
pipenv --python PYTHON_VERSION/PYTHON_PATH
```

这里的`PYTHON_VERSION`或`PYTHON_PATH`代表的是`python`的版本号以及本地的python环境位置. 不可以为空. 例如使用`python 3.9`环境:

```sh
pipenv --python 3.9
```

运行成功的结果为:

```
$> pipenv --python 3.9
Creating a virtualenv for this project...
Pipfile: E:\Code\chatgpt-demo\Pipfile
Using D:/Environment/Anaconda3/python.exe (3.9.7) to create virtualenv...
[    ] Creating virtual environment...created virtual environment CPython3.9.7.final.0-64 in 4531ms
  creator CPython3Windows(dest=C:\Users\GARNET\.virtualenvs\chatgpt-demo-HkymoQL8, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=C:\Users\GARNET\AppData\Local\pypa\virtualenv)
    added seed packages: pip==23.0, setuptools==67.1.0, wheel==0.38.4
  activators BashActivator,BatchActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

Successfully created virtual environment!
Virtualenv location: C:\Users\ABC\.virtualenvs\chatgpt-demo-HkymoQL8
Creating a Pipfile for this project...
```

命令执行后, 有两个变化:

- 项目根目录下, 会创建一个`Pipfile`文件, 记录了初始化的虚拟环境信息
- 虚拟环境没有被创建到项目目录下, 而是创建到了`C:\Users\ABC\.virtualenvs\chatgpt-demo-HkymoQL8`. 这是与`virtualenv`有区别的地方. 虚拟环境存放目录:
  - Windows: `%homepath%\.virtualenvs\<projectname>-<Random Code>`
  - Linux/MacOS: `~/.local/share/virtualenvs/<projectname>-<Random Code>`

---

更常用的创建虚拟环境的指令为:

```sh
pipenv install [OPTIONS]
```

使用这个指令创建虚拟, 如果该项目根目录中没有`Pipfile`文件, 则会自动创建`Pipfile`和`Pipfile.lock`两个文件. 如果该工程目录中有`Pipfile`, 则将安装`Pipfile`列出的相应依赖包, 安装完成后生成`Pipfile.lock`.

`OPTIONS`可以使用如下的参数, 来控制创建虚拟环境的行为:

- `--python <VERSION>`: `VERSION`为`python`版本. 指定该虚拟环境使用的`python`版本
- `--pypi-mirror`: 指定安装源
- `--site-packages` / `--no-site-packages`: 是否使用`python`本地环境中的`site-packages`包

---

另外, 如果指定的`python`版本, 本地没有安装, 则会报错:

```
$> pipenv --python 3.8
Warning: Python 3.8 was not found on your system...
Neither 'pyenv' nor 'asdf' could be found to install Python.
You can specify specific versions of Python with:
$ pipenv --python path\to\python
```

如果安装了`pyenv`或`asdf`工具, 则会自动安装对应版本的`python`到本地.

## 查看虚拟环境

可以使用如下的命令查看虚拟环境的情况:

```sh
# 查看工程根目录信息
pipenv --where

# 查看当前虚拟环境的信息
pipenv --venv

# 查看python解释器的信息
pipenv --py

# 查看环境变量选项
pipenv --envs
```

## 激活虚拟环境

类似于`virtualenv`中的`source activate`. 使用以下命令激活虚拟环境:

```sh
pipenv shell
```

```
E:\Code\chatgpt-demo>pipenv shell
Launching subshell in virtual environment...
Microsoft Windows [版本 10.0.19044.2604]
(c) Microsoft Corporation。保留所有权利。
```

注意, 在Windows激活虚拟环境时, **可能在命令行的前面不能显示虚拟环境的名称**, 但实际已经成功激活环境了, 不影响正常使用.

## 退出虚拟环境

```sh
exit
```

## 删除虚拟环境

在项目根目录中输入:

```sh
pipenv --rm
```

## 环境配置

### 包的安装

使用`pipenv`安装`python`的三方包使用的也是`install`指令:

```sh
pipenv install [OPTIONS] [PACKAGES]
# Installs provided packages and adds them to Pipfile, or (if no packages are given), installs all packages from Pipfile.
```

使用`pipenv`安装`python`的三方包, 与使用`pip`安装, 区别在于除了将包安装在虚拟环境的`site-packages`目录中, 还会将安装的包会将相关信息写入`Pipfile`和`Pipfile.lock`.

例如需要安装`numpy`包, 使用:

```sh
pipenv install numpy
```

事实上, 对一个新项目来说, 不必手动使用`pipenv install`来创建虚拟环境, 当使用`pipenv install <package_name>`直接安装依赖包时, 如果当前目录没有创建过虚拟环境, `pipenv`会自动创建一个.

而使用`pipenv`安装包之后, 会自动帮你管理依赖. 在使用`pipenv install`和`pipenv uninstall`命令安装和卸载包时会自动更新`Pipfile`和`Pipfile.lock`.

---

有时开发环境和生产环境的需求的包会有差异, 例如开发时需要用到测试逻辑的包`pytest`. 在使用`pip`时, 可以创建一个`requirements.txt`用在生产环境, 一个`requirements-dev.txt`用在开发环境.

`pipenv`提供了工具, 方便区分开发和生产的区别. 只需要在安装时, 在包名的后面添加一个`--dev`参数, 所安装的包就会自动被分类为开发依赖, 写入`Pipfile`的`dev-packages`一节中:

```sh
pipenv install pytest --dev
```

## 在部署环境安装依赖

在新环境部署项目环境, clone下来的项目代码中, 已经包含了配置环境需要的`Pipfile`和`Pipfile.lock`, 只需要在根目录下运行:

```sh
pipenv install
```

---

如果要在新环境中部署开发环境, 也只需要在`pipenv install`命令后添加`--dev`参数, 即可安装开发依赖:

```sh
pipenv install --dev
```
