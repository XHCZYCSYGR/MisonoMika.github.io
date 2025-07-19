本篇主要内容来自于EDM论文内容，其对扩散模型统一化的表示有助于后续对于一大类基于扩散模型的其他模型（SMLD，FM/RF等）进行研究。
# 通用加噪公式
P8,P9。求 $p(x_t | x_0)$ 的均值和方差
# 扩散模型的通用形式探索
P10，P11
# 模型预测内容（确定性采样）
P12
# 通用框架的进一步探索
在上一节中，我们得到扩散模型可以使用一个去噪模型 $D_{\theta}(x,\sigma)$ 表示，但是，实验证明直接训练一个去噪模型效果往往难以达到理想的效果。因此，还需要对 $D_{\theta}(x,\sigma)$ 进行拆解与深入分析。
## 通用模型框架
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

观察上面三个 $D_{\theta}(x,\sigma(t))$不难发现，其都可以写成如下所示的统一形式：

$$
D_{\theta}(x,\sigma) = C_{skip}(\sigma)\textbf{x}+ C_{out}(\sigma)F_{\theta}(C_{in}(\sigma)\textbf{x};C_{noise}(\sigma))
$$

其中 $F_{\theta}$是模型真正需要训练的部分。另外需要说明：
（1） 这里的 $\sigma$实际上是 $\sigma(t)$，而实际上 $\sigma(t)$是“简单”且可逆的函数，所以 $\sigma \Leftrightarrow t$后续也不对 $\sigma$和 $t$进行区分。
（2） EDM要求模型 $F_{\theta}$的输入和输出符合一定的标准，所以 $C_{in}$和 $C_{noise}$主要是对输入 $\textbf{x}$进行标准化，而 $C_{skip}$和 $C_{out}$主要是使输出满足规定。

## 通用训练框架

首先定义噪声图像 $x=y+n$其中 $y$是原始图像， $n$是添加的噪声，则根据MSE损失可得 $D_{\theta}(x,\sigma)$的损失函数如下：

$$
\begin{aligned}
\mathbb{L} &=    E_{x,y \sim p_{data},\sigma \sim p_{train}}  \left[  \lambda(\sigma)  || D_{\theta}(x,\sigma)-y  ||^2_2 \right]\\
&= E_{n \sim N(0,\sigma^2I),y \sim p_{data},\sigma \sim p_{train}}  \left[  \lambda(\sigma)  || D_{\theta}(y+n,\sigma)-y  ||^2_2 \right]\\
&= E_{n,\sigma,y}  \left[  \lambda(\sigma)  || C_{skip}(\sigma)(\textbf{y+n})+ C_{out}(\sigma)F_{\theta}(C_{in}(\sigma)(\textbf{y+n});C_{noise}(\sigma))  ||^2_2 \right]\\
&=   E_{n,\sigma,y}  \left[  \lambda(\sigma) C_{out}^2(\sigma) || F_{\theta}(C_{in}(\sigma)(\textbf{y+n});C_{noise}(\sigma))-\frac{1}{C_{out}(\sigma)} \left(y-C_{skip}(\sigma)(\textbf{y+n}) \right)  ||^2_2 \right]\\
&= E_{n,\sigma,y}  \left[  w(\sigma)  || F_{\theta}(C_{in}(\sigma)(\textbf{y+n});C_{noise}(\sigma))- F_{target}(n,\sigma,y)  ||^2_2 \right]
\end{aligned}\\
$$

其中 $w(\sigma)$用来平衡模型对于不同 $\sigma$的关注度， $w(\sigma)=\lambda(\sigma) C_{out}^2(\sigma)$， $F_{target}(n,\sigma,y)=\frac{1}{C_{out}(\sigma)} \left(y-C_{skip}(\sigma)(\textbf{y+n}) \right)$表示模型的目标。
## 超参数设定
为了保证训练过程的稳定，所以需要对上述损失函数中的超参数 $C_{skip},C_{in},C_{out},\lambda(\sigma)$做出约束。
### 对神经网络输入的要求
 $F_{\theta}$的输主要由 $C_{in}$控制，为了避免模型输入值域跨度过大，所以要求模型输入方差为1，也即：

$$
\begin{aligned}
Var_{y,n} \left[  C_{in}(\sigma)(y+n)  \right]=1\\
C_{in}^2(\sigma)Var_{y,n}[y+n]= 1\\
C_{in}^2(\sigma) \left(  \sigma_{data}^2 + \sigma^2 \right)= 1\\
C_{in}(\sigma)= 1 / \sqrt{  \sigma_{data}^2 + \sigma^2 }
\end{aligned}
$$ 
### 对训练目标的要求
为了保证梯度值的稳定，要求训练目标 $F_{target}(n,\sigma,y)$的方差为1，也即：

$$
\begin{align}
Var_{y,n} \left[  \frac{1}{C_{out}(\sigma)} \left(y-C_{skip}(\sigma)(\textbf{y+n}) \right)   \right] = 1\\
\frac{1}{C_{out}^2(\sigma)}Var_{y,n} \left[  (1-C_{skip}(\sigma))y-C_{skip}(\sigma)n \right] = 1\\
C_{out}^2(\sigma) =  (1-C_{skip}(\sigma))^2\sigma_{data}^2-C_{skip}^2(\sigma)\sigma^2
\end{align}
$$

从上述 $C_{out}^2(\sigma)$的表达式可以发现， $C_{out}^2(\sigma)$d的大小和 $C_{skip}(\sigma)$有关，所以我们可以在求得 $C_{skip}(\sigma)$的同时限制（最小化） $C_{out}^2(\sigma)$的范围。直接对 $C_{skip}(\sigma)$求导得：

$$
\begin{aligned}
\frac{\mathrm{d} C_{out}^2(\sigma) }{\mathrm{d} C_{skip}(\sigma)} = 0\\
2\sigma_{data}(C_{skip}(sigma)-1)+2C_{skip}(\sigma)\sigma^2 = 0\\
C_{skip}(\sigma) = \frac{\sigma_{data}^2}{\sigma^2+\sigma_{data}^2}
\end{aligned}
$$

讲求得的 $C_{skip}(\sigma)$带入 $C_{out}^2(\sigma)$中得：

$$
\begin{aligned}
C_{out}^2(\sigma)=\left[  1-  \frac{\sigma_{data}^2}{\sigma^2+\sigma_{data}^2} \right]^2\sigma_{data}^2+
\left[  \frac{\sigma_{data}^2}{\sigma^2+\sigma_{data}^2} \right]^2\sigma^2\\
C_{out}(\sigma)=\frac{\sigma  \cdot \sigma{data}}{\sqrt{\sigma^2+\sigma_{data}^2}}
\end{aligned}
$$

### 平等关注所有样本
前面说过， $w(\sigma)$用来平衡模型对于不同 $\sigma$的关注度，对所有样本一视同仁也即：

$$
\begin{aligned}
w(\sigma)=1\\
\lambda(\sigma) C_{out}^2(\sigma)=1\\
\lambda(\sigma)  = \frac{1}{C_{out}^2(\sigma)}\\
\lambda(\sigma) = \frac{\sigma^2+\sigma_{data}^2}{(\sigma  \cdot \sigma{data})^2}
\end{aligned}
$$

### 整合
在神经网络初始值 $F_{\theta}(\cdot)=0$的前提下，将上述推导的超参数带入损失函数，得：


$$
\begin{aligned}
E_{n,\sigma,y}  \left[  \lambda(\sigma) C_{out}^2(\sigma) || F_{\theta}(C_{in}(\sigma)(\textbf{y+n});C_{noise}(\sigma))-\frac{1}{C_{out}(\sigma)} \left(y-C_{skip}(\sigma)(\textbf{y+n}) \right)  ||^2_2 \right]\\ 
=E_{\sigma,y} \left[ ||  \frac{\sigma^2+\sigma_{data}^2}{(\sigma  \cdot \sigma{data})^2} (\frac{\sigma_{data}^2}{\sigma^2+\sigma_{data}^2}(y+n)-y)  ||_2^2  \right] \\ 
=E_{\sigma,y} \left[ ||  \frac{\sigma^2+\sigma_{data}^2}{(\sigma  \cdot \sigma{data})^2} (\frac{\sigma_{data}^2}{\sigma^2+\sigma_{data}^2}(y+n)-y)  ||_2^2  \right] \\ 
dsad
\end{aligned}
$$








