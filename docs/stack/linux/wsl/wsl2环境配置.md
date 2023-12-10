WSL2(Windows Subsystem for Linux 2), 是微软开发的适用于多种Linux发行版的子系统, 允许开发人员直接在 Windows 上运行 GNU/Linux 环境. WSL作为WIndows原生的服务, 其可以更容易地直接访问GPU并利用cuda进行深度学习算法的开发.

这里记录在 Windows 11 系统中配置 WSL2 环境的过程.

# 环境检查

## 开启硬件虚拟化

在**任务管理器 - 性能**中检查硬件虚拟化是否打开:

![](/resources/images/stack/linux/wsl-1.png)

如果未开启硬件虚拟化, 则需要重启电脑并进入BIOS, 并找到相应的虚拟化功能开启即可.

## 检查是否开启所需的 Windows 功能

在搜索栏中输入 `启用或关闭Windows功能`, 开启以下选项:

- Hyper-V
- 虚拟机平台
- 适用于 Linux 的 Windows 子系统

![](/resources/images/stack/linux/wsl-2.png)

# 安装

## 下载分发版本并安装

通过 `wsl --list --online` 即可显示所有可安装的分发. 执行 `wsl.exe --install <分发名>` 完成下载和安装.

## 坑

安装的过程中可能会遇到下面的错误:

![](/resources/images/stack/linux/wsl-3.png)

解决方法是更新 WSL 2 Linux 内核, 以便在 Windows 操作系统中运行 WSL. 详情参考: [步骤 4 - 下载 Linux 内核更新包](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package).

下载最新的 WSL2 Linux 内核包: [适用于 x64 计算机的 WSL2 Linux 内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi), 并安装.

## 升级WSL到WSL2

关于WSL2和WSL1的区别, 参考微软的官方文档[适用于 Linux 的 Windows 子系统文档 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/).

WSL2需要手动升级, 以管理员身份打开powershell, 运行:

```bash
wsl --update
wsl --set-default-version 2  # 默认使用 WSL2
```

# 配置 CUDA 环境

以安装 [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) 版本为例, 其他版本 CUDA 可以在 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 中找到.

在页面中找到对应的安装指令, 这里的CUDA Toolkit是针对WSL的版本，规避掉了CUDA driver的安装.

![](/resources/images/stack/linux/wsl-4.png)

这里找到的指令为:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu1804-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

安装完成后, 将以下信息加入到环境变量中:

```bash
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PAT
```

此时就可以通过 `nvcc -V` 指令看到 cuda 相关的信息了:

![](/resources/images/stack/linux/wsl-5.png)

# 在 Visual Studio Code 中打开 WSL 项目

在 WSL 的项目根目录中, 通过 `code .` 指令启动 `VS Code Server`, 然后在 Windows 中会打开对应的 `VS Code Client`:

![](/resources/images/stack/linux/wsl-6.gif)

# 参考资料

- [WSL2 安装和基本环境配置流程](https://zhuanlan.zhihu.com/p/652537694)
- [无痛安装：Win10在WSL2里安装CUDA12.2+Pytorch GPU版本](https://zhuanlan.zhihu.com/p/648330821)
- [如何使用 WSL 在 Windows 上安装 Linux](https://learn.microsoft.com/zh-cn/windows/wsl/install)
- [开始通过适用于 Linux 的 Windows 子系统使用 Visual Studio Code](https://learn.microsoft.com/zh-cn/windows/wsl/tutorials/wsl-vscode#update-your-linux-distribution)
- [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
