# 基本概念
## GAN的任务
     顾名思义，我们的目标是生成和输入X类似或具有相同特征的图像、文本等。
## ~~那我问你~~  GAN（Generative Adversarial Networks）与一般的神经网络有什么不同？
   一般的神经网络是“单一”的输入，即tensor[X]，但是GAN的输入是X+Z（类似通信原理的AWGN信道，只不过Z不一定是高斯分布），Z是已知简单分布的采样值（同样转为tensor），不一定和X同维，可以与X做不同的运算，且每次都会更新。然后整体作为输入。我们的目标是让GAN不仅注意到X，会受到Z的影响。
## 如何理解对抗？
     Generator（G）与Discriminator（D）的对抗可以通俗的理解为：捕食者与被捕食者。对抗的过程大致如下：
- G.v1（初始化参数，随机的）对采样得到的Z进行学习，得到输出Y.v1
- 更新D.v1中的参数，使D.v1可以分辨Y.v1与输入X的差别。对D的理解：D的功能可以看作对Y.v1和X进行分类（或打分），D的输出可以是分类的Label（分数）
- 固定D.v1，更新G.v1的参数（Z也会变），使更新后G.v2的输出Y.v2可以“骗过”D.v1，即经D.v1判决或评分后，Y.v2和X没有区别
- 固定G.v2，更新D.v1的参数，使更新后D.v2可以分辨Y.v2和X的区别
- 循环上述步骤得到具有生成的G.vxxx

# GAN的理论支持
## GAN遇到的问题
我们要优化的目标函数：
    
$$
G^* = arg \  \underset{G}{min} \ Div(P_G \  , \  P_{data}) 
$$

其中， $P_G$表示经过Generator输出的Y的分布， $P_{data}$表示X的原始分布，Div是divergence的缩写。也就是说，我们的目标是找到一个最佳的G，使 $P_G$和 $P_{data}$ 的概率分布最为接近。
一般情况下， $Div(P_G \  , \  P_{data})$ 可以是KL散度或JS散度，但是，对于未知的 $P_{data}$，若按照定义式计算，可行性极差（学习途中偶遇超级积分，拼尽全力无法战胜）。
GAN的解决方案：**Sampling is GOOD enough**

## 为什么能够通过Sampling解决问题？
首先，我们回看Discriminator的功能：

$$
D^* = arg \  \underset{D}{max} \  V(D \   , \   G)
$$

其中

$$
V(D \  , \  G) = E_{y \sim P_{data}}[logD(y)] + E_{y \sim P_G}[log(1-D(y))]
$$

从上面的公式不难看出，D的目标是使 $y \sim P_{data}$得到的分数比较高，而让 $y \sim P_G$的分数尽可能低，最后使得D的分辨力足够强大。
也就是说，对D的训练可以看作对一个二元分类问题进行训练。
另外地，经过数学推导，发现 $\underset{D}{max} \  V(D \   , \   G)$和JS散度有关。而在上面的目标函数中， $Div(P_G \  , \  P_{data})$可以是JS散度，所以，目标函数可更新为

$$
G^* = arg \  \underset{G}{min} \ \underset{D}{max} \  V(D \   , \   G) 
$$

补充说明：实际上我们也可以使用其他的 $Div(P_G \  , \  P_{data})$，在使用不同散度时应当使用的公式在有关f-GAN的文章中亦有记载，网址[https://arxiv.org/abs/1606.00709](url)

由此，我们便拥有计算 $Div(P_G \  , \  P_{data})$的方法。

但是，即使我们可以计算 $Div(P_G \  , \  P_{data})$，GAN还是很难train动下面列举两个最常见的原因。

1. 一般情况下， $P_{data}$和 $P_G$在高维空间中的测度较小。以二维平面举例， $P_G$和 $P_{data}$的分布可能就是两条线（平面上大部分点都不属于我们想要的概率分布），这会导致两个分布的重合部分极小（几乎可以忽略不计），最终使D没有改善或修改。
2. 即使两者重合部分可以避免上面的情况，我们也有可能因为sampling过程中的样本数目过少，导致取样点无法充分表示 $P_{data}$和 $P_{G}$的分布。

在上面两种原因的作用下，以使用JS散度为例，可能的实验结果表示为D分辨的正确率为100%，且loss为 $log2$。也就是说，通过loss和D的输出，我们无法判断模型是否真正学到东西，只能通过输出G的图片人眼观察是否有生成效果。（难用o(≧口≦)o）  

## 解决JS散度带来的问题——————Wasserstein distance
Wasserstein distance（瓦瑟斯坦距离），也称为Earth Mover's Distance (EMD)。通俗地来讲，原始分布记为Q，G输出的分布记为P，EMD是将P分布调整为Q所需“距离”的度量。
由此，即使JS散度相同，我们也能通过EMD的数值来了解P和Q是否在接近。换一个角度说，EMD的值可以反应G是否真正在学习。
但是，对于相同的P和Q，调整方案有很多种（对应不同的EMD），我们取最小的作为EMD值，数学公式表达如下：

$$
\underset{D \in 1-Lipschitz}{max}  E_{y \sim P_{data}} [D(y)] - E_{y \sim P_G}[D(y)] 
$$

其中 $D \in 1-Lipschitz$表示D是足够平缓的（smooth enough），由此保证，当 $P_{data}$和 $P_{G}$分布不重叠时，D不会为相应部分分配  $\pm \infty$ ，如下图所示。

![Image](https://github.com/user-attachments/assets/5e56f2ff-2e13-4ae5-9317-a3e242202304)














