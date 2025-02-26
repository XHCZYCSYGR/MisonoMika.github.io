# 论文相关 (https://arxiv.org/abs/2111.06377)
## Mask Loss
损失函数为MSE，计算时只计算Masked部分的loss，可见部分不进行计算。
Masked Patch计算均值和标准差进行规一化mask，模型表现会更好。

## Mask实现
无需稀疏化操作

1. 生成token（含有position embedding）
2. Shuffle token list
3. 按照mask比例（75%），取前25%，后面舍弃（encoder输入）
4. 还原token list（unshuffle）（decoder输入）

## 相关实验
对于Figure 5

![Image](https://github.com/user-attachments/assets/ade514f4-2107-455a-8cd1-607b2fa1f770)

实验操作为：训练完成后将encoder的输出作为一个classification（MLP）的输入（作为一个分类模型并进行训练），纵坐标为分类的准确率。
其中fine-tuning表示训练过程中对encoder和MLP的参数都进行调整；linear probing表示固定encoder，只调整MLP的参数。