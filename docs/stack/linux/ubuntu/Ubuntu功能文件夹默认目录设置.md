# 问题描述

Ubuntu安装后，在用户默认路径下`~`，存在着多个文件夹，包括桌面，下载，文档，图片等系统默认的存储某类文件或支持某类功能的目录。这些目录直接暴露在用户默认路径`~`下，整个用户目录显得很乱。特别是如果安装的是中文版本的Ubuntu，这些目录的名称还是中文，如果要使用这些目录，比如打开下载目录，就需要输入中文，非常的不方便。

如果你试过将这些支持系统功能的目录通过`mv`指令系统到某个位置或者改换名称，在系统重启后就会发现那些`mv`指令更改过的文件夹又都回来了，这说明有着某些配置文件管理控制着这些目录。

# 修改配置文件

这个配置文件位于`~/.config/user-dirs.dirs`。打开后可以看到默认的配置：

```ini
XDG_DESKTOP_DIR="/media/username/Desktop"  
XDG_DOWNLOAD_DIR="/media/username/Download"
XDG_TEMPLATES_DIR="/media/username/Templates"
XDG_PUBLICSHARE_DIR="/media/username/Public"
XDG_DOCUMENTS_DIR="/media/username/Documents"
XDG_MUSIC_DIR="/media/username/Music"
XDG_PICTURES_DIR="/media/username/Pictures"  
XDG_VIDEOS_DIR="/media/username/Videos"
```

可以看到这里有8个系统默认目录，根据名称可以对应到桌面，下载，模板，公用，文档，音乐，图片，视频等目录。修改配置，并使其生效，需要以下几部。

- 修改配置文件`~/.config/user-dirs.dirs`，将每个目录修改成自己想要放置的位置
- 对配置文件中指定的位置，手动创建对应的目录文件。这是因为重启后系统在找不到这些文件夹时，并不会自动创建这些文件夹，而是会自动修改配置文件`~/.config/user-dirs.dirs`到用户默认路径`~`
- 文件夹创建完毕后，重启系统
