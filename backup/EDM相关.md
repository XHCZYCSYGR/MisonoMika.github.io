本篇主要内容来自于EDM论文内容，其对扩散模型统一化的表示有助于后续对于一大类基于扩散模型的其他模型（SMLD，FM/RF等）进行研究。
# 通用加噪公式
P8,P9。求 $$ p(x_t | x_0) $$ 的均值和方差
# 扩散模型的通用形式探索
P10，P11
# 模型预测内容（确定性采样）
P12
# 通用模型框架的进一步探索
在上一节中，我们得到扩散模型可以使用一个去噪模型 $D_{\theta}(x,\sigma)$ 表示，但是，实验证明直接训练一个去噪模型效果往往难以达到理想的效果。因此，还需要对 $D_{\theta}(x,\sigma)$ 进行拆解与深入分析。
## 思路来源
由 $D_{\theta}(x,\sigma)$ 的定义不难发现， $D_{\theta}(x,\sigma)$ 定义的目标实际上就是原图，下面直接给出不同扩散模型对应的 $D_{\theta}(x,\sigma)$ 的表达式：

$$
D_{\theta}(x,\sigma(t))  \approx \frac{x_t-\sqrt{1-\bar{\alpha}}\varepsilon}{\sqrt{\bar{\alpha_t}}} =\frac{1}{\sqrt{\bar{\alpha}}}x_t-\frac{\sqrt{1-\bar{\alpha}}}{\sqrt{\bar{\alpha_t}}}\varepsilon_{\theta}(x_t;t)  \ \ \ ,\ \ \  DDPM    
$$

$$
D_{\theta}(x,\sigma(t)) \approx \textbf{x}+\sigma^2s_{\theta}(x_t,t) \ \ \ ,\ \ \ SMLD
$$

$$
D_{\theta}(x,\sigma(t)) \approx x_t+tv_{\theta}(x_t,t) \ \ \ ,\ \ \ FL/RF
$$

