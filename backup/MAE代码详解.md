# models_mae.py

## MaskedAutoencoderViT类
```python
def __init__(self, img_size=224, patch_size=16, in_chans=3,        # 输入数据大小（224*224，16*16，RGB）
                 embed_dim=1024, depth=24, num_heads=16,          # ViT参数，此处默认使用ViT-Large，可视算力情况修改
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,   # maedecoder参数
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)   
# transformer线性层的宽度 ，层归一化，计算loss前是否对像素值进行归一化

```

![Image](https://github.com/user-attachments/assets/c391b794-1eb3-4bb5-b02f-344129f62231)