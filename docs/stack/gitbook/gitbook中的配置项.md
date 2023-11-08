# Gitbook 中的配置项

## 定义 Gitbook 的解析方法

在根目录中的 `.gitbook.yaml` 文件, 将定义 Gitbook 解析 Git 仓库的行为. 一般此文件中的内容如下

```yaml
root: ./

structure:  
    readme: README.md  
    summary: SUMMARY.md

redirects:  
    previous/page: new-folder/page.md
```

### root

`root` 一项定义了查找文件存储位置的起点, 默认是仓库的根目录. 下面这段配置代码, 将查找起点定义为根目录下的 `docs` 目录:

```yaml
root: ./docs/
```

特别需要注意的是, **这里的定义的查找起点, 只作用于本配置文件, 即 `.gitbook.yaml` 文件**, 它将影响该配置文件下面其他项, 如后面设置的 `readme` 项和 `summary` 项文件, 都是关于 `root` 的相对位置. 但超出此文件, 对于其他配置文件中使用的路径, 就没有作用了.

### ​structure

`structure` 中有两项配置项:

- `readme`: 文档首页对应的文件, 默认的文件名是 `README.md`
- `summary`: 文档目录结构的维护文件, 默认的文件名是 `SUMMARY.md`

这两项配置的路径, 是相对于 `root` 项的路径. 例如 `root` 项是 `./conf`, `readme` 项是 `./product/README.md`, 则文档首页文件对应的路径是 `./conf/product/README.md`.

`readme` 项的作用很直观, 这里说明下 `summary` 项的形式.

#### summary

`summary` 项对应的也是一个 `Markdown` 文件(`.md`). 内容应当符合如下的形式.

```markdown
# Summary

## Use headings to create page groups like this one

- [First page's title](page1/README.md)    
    - [Some child page](page1/page1-1.md)    
    - [Some other child page](part1/page1-2.md)

- [Second page's title](page2/README.md)    
    - [Some child page](page2/page2-1.md)    
    - [Some other child page](part2/page2-2.md)    

## A second-page group

- [Yet another page](another-page.md)
```

`Gitbook` 首先会在根据 `summary` 配置项寻找对应的目录文件, 如果在指定的位置找不到, 则会在仓库的根目录中寻找. 如果这两个位置都不存在 `SUMMARY.md` 文件, `Gitbook` 会根据文件结构, 推测出目录结构, 并且生成一个 `SUMMARY.md` 文件.

### redirects

你可以直接指定 URL 与文件的映射关系. 这里的 URL 指的是 Gitbook 发布之后的地址后缀. 例如想指定 `/help` 到一个具体的文件, 可以设置:

```yaml
redirects:  
    help: ./support.md
```

可以看到, 在指定 URL 时, 不可以包含 `/`; 另外这里的文件路径, 也是相对于 `root` 的相对路径.

# 参考资料

- [Content Configuration](https://docs.gitbook.com/integrations/git-sync/content-configuration)
