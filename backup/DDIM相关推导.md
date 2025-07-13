# DDIM 相关
## Background 
基于DDPM的工作，目前已经得到了表现优良的扩散模型。但是，由于DDPM在推导过程中利用了正向加噪过程具有Markov性质这一条件，其在逆向推理（Denoising）过程中只能“逐步去噪”，在step数较多时运算效率极其低下。为了解决DDPM的这一“痛点”，DDIM在DDPM的基础上构建了一个"non-Markov"的"Diffusion Model"。
个人理解上，DDIM实际上是借用DDPM的“骨架”，修改优化其反向去噪的部分，使模型效率提高。另外需要注意一点的是，DDIM修改后的"non-Markov"模型可以直接套用DDPM的预训练模型，下面会证明修改后两个模型的损失函数 $J_{\sigma}$ （DDIM）和 $L_{\gamma}$ （DDPM）是等价的。

先看完EDM，后续会研究DDIM，SM，FM等相关模型