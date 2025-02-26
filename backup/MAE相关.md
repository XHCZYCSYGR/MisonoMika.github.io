# 论文相关
## Mask Loss
损失函数为MSE，计算时只计算Masked部分的loss，可见部分不进行计算。
Masked Patch计算均值和标准差进行规一化mask，模型表现会更好。

## Mask实现
无需稀疏化操作

1. 生成token（含有position embedding）
2. Shuffle token list
3. 按照mask比例（75%），取前25%，后面舍弃（encoder输入）
4. 还原token list（unshuffle）（decoder输入）