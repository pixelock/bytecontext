# NLP 中的对抗训练

## 参考

- [对抗训练浅谈：意义、方法和思考（附Keras实现）](https://kexue.fm/archives/7234)
- [关于 Adversarial Training 在 NLP 领域的一些思考](https://zhuanlan.zhihu.com/p/31920187)
- [NLP 中的对抗训练（附 PyTorch 实现）](https://wmathor.com/index.php/archives/1537/)
- [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
- [一文搞懂NLP中的对抗训练FGSM/FGM/PGD/FreeAT/YOPO/FreeLB/SMART](https://zhuanlan.zhihu.com/p/103593948)

## 为什么有效

### Grammatical Role 相近的词 Embedding 相近

有些词与比如 good 和 bad, 其在语句中 Grammatical Role 是相近的, 这是由于它们周围一并出现的词语是相近的, 经常在相同的句式中修饰同样的内容, 但表达相反的方式.

这就导致这两次词的 Word Embedding 是非常相近的. 下图中的 `baseline` 和 `random` 列, good 和 bad 出现在了彼此的邻近词中. 使用对抗学习之后, `Adversarial` 一列中, 这种现象没有再出现.

![](/resources/images/nn/adversarial-1.png)

可以推测, 在 Word Embedding 上添加扰动(Perturbation), 会导致原来的 good 变成 bad, 导致分类错误, 从而计算的 Adversarial Loss 很大, 通过训练拉大了 embedding 之间的差距, 提升了模型的整体表现.

**一句话总结**

这些含义不同而语言结构角色类似的词能够通过这种 Adversarial Training 的方法而被分离开, 从而提升了 Word Embedding 的质量, 帮助模型取得了非常好的表现.

### 梯度惩罚

对输入样本施加一个 $$\epsilon \nabla_x L (x,y;\theta)$$ 的对抗扰动, 等价于往 loss 里面加入了下面这项**梯度惩罚**:

$$
\frac{1}{2}\epsilon ||\nabla_x L(x,y;\theta)||^2
$$

在 FGM 中使用的惩罚项为 $$\Delta x = \epsilon \frac {\nabla_x L (x,y;\theta)}{||\nabla_x L (x,y;\theta||}$$, 对应的梯度惩罚就是 $$\epsilon ||\nabla_x L (x,y;\theta)||$$.

## 方法

- FGSM
- FGM
- PGD
- FreeAT
- YOPO
- FreeLB
- SMART

### FGM

实现如下:

```python
# 如果模型中有多个 Embedding, 使用 `attack_multi_emd` 方法
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1e-6, emd_name='bert.embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                self.backup[name] = param.data.clone()
                # print(param.grad)
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emd_name='bert.embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def attack_multi_emd(self, epsilon=1e-6, emd_names=['bert.embeddings.']):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                is_update = False
                for emd_name in emd_names:
                    if emd_name in name:
                        is_update = True
                        break
                if is_update:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore_multi_emd(self, emd_names=['bert.embeddings.']):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                is_update = False
                for emd_name in emd_names:
                    if emd_name in name:
                        is_update = True
                        break
                if is_update:
                    assert name in self.backup
                    param.data = self.backup[name]
        self.backup = {}
```

训练中的使用方法:

```python
# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
  # 正常训练
  loss = model(batch_input, batch_label)
  loss.backward() # 反向传播，得到正常的grad
  # 对抗训练
  fgm.attack() # embedding被修改了
  # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
  loss_sum = model(batch_input, batch_label)
  loss_sum.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
  fgm.restore() # 恢复Embedding的参数
  # 梯度下降，更新参数
  optimizer.step()
  optimizer.zero_grad()
```

## PGD

```python
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
```

使用方法:

```python
pgd = PGD(model)
K = 3
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    pgd.backup_grad() # 保存正常的grad
    # 对抗训练
    for t in range(K):
        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K-1:
            optimizer.zero_grad()
        else:
            pgd.restore_grad() # 恢复正常的grad
        loss_sum = model(batch_input, batch_label)
        loss_sum.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    optimizer.zero_grad()
```
