[Variational Diffusion Models](https://arxiv.org/pdf/2107.00630)

### 摘要
将扩散过程的schedule使用signal-to-noise ratio进行表示，并直接使用VLB即$\log p(\boldsymbol{x})$作为损失函数进行优化。同时结合了bits-back coding实现了图像无损压缩。 ^30abaa

### 扩散过程
#### 前向过程
与DDPM介绍类似，通过添加高斯噪声使得原始图像$\boldsymbol{x}_{0}$变为含噪声图像$\boldsymbol{x}_{t}$：
$$
q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})=\mathcal{N}(\alpha_{t}\boldsymbol{x}_{0},\sigma_{t^2}\boldsymbol{I})
$$
进一步定义信噪比（Signal-to-Noise Ratio, SNR）为
$$
\text{SNR}=\frac{\alpha_{t}^2}{\sigma_{t}^2}
$$
满足对任意$t>s$，有$\text{SNR}(t)<\text{SNR}(s)$。后续着重分析[[SDE_Diffusion#^7a2a2d|VP-SDE]]的情况，即$\alpha_{t}^2+\sigma_{t}^2=1$。

#### Noise Schedule
基于SNR的定义，使用可学习的schedule
$$
\begin{align}
\sigma_{t}^2&=\text{sigmoid}(\gamma_{n}(t))\\
\alpha_{t}^2&=\text{sigmoid}(-\gamma_{t}(t)) \\
\text{SNR(t)}&=\exp(-\gamma_{n}(t))
\end{align}
$$