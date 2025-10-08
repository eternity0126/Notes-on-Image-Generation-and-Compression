[DIFFUSION POSTERIOR SAMPLING FOR GENERAL NOISY INVERSE PROBLEMS](https://arxiv.org/pdf/2209.14687)


论文针对noisy inverse problems提出了一种新的扩散后验采样方式。
### 逆问题的定义
定义逆问题（inverse problem）为已知前向操作$\mathcal{A}$和噪声$n$从观测量$y$中复原$x$。其中前向操作为

$$
\boldsymbol{y}=\mathcal{A}(\boldsymbol{x})+\boldsymbol{n},\quad \boldsymbol{y},\boldsymbol{n}\in\mathbb{R}^n,\boldsymbol{x}_0\in\mathbb{R}^d
$$

其中 $x$为原始信息， $y$为观测到的信息。
当 $\mathcal{A}(x)\triangleq \boldsymbol{A}\boldsymbol{x}$时为linear inverse problems，包含图像修复、去除高斯噪声、超分辨率、去除运动模糊等任务。
nonlinear inverse problems包含相位恢复、去除非线性噪声等任务。
### Score-based Diffusion Models
基于SDE的扩散定义为

$$
d\boldsymbol{x}=-\frac{\beta(t)}{2}\boldsymbol{x}dt+\sqrt{\beta(t)}d\boldsymbol{w}
$$

如果想从 $p(\boldsymbol{x}|\boldsymbol{y})$中采样，则可以使用 $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{y}|\boldsymbol{x})$实现。
SDE的反向过程：

$$
d\boldsymbol{x}=\left[-\frac{\beta(t)}{2}\boldsymbol{x}-\beta(t)\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right]dt+\sqrt{\beta(t)}d\overline{\boldsymbol{w}}
$$

在有条件限制的情况下，使用扩散模型预测先验概率时，需要将上述式子中的$p_t(\mathbf{x}_t)$替换为$p_t(\mathbf{x}_t|\mathbf{y})$：

$$
d\mathbf{x}=\left[-\frac{\beta(t)}{2}\mathbf{x}-\beta(t)(\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)+\nabla_{\mathbf{x}_t}\log p_t(\mathbf{y}|\mathbf{x}_t))\right]dt+\sqrt{\beta(t)}d\overline{\mathbf{w}}
$$

其中分数函数$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$可以使用预训练好的模型预测，即$s_{\theta^*}$。

$$
\begin{aligned}
p(\mathbf{y}|\mathbf{x}_t)
&=\int p(\mathbf{y}|\mathbf{x}_0,\mathbf{x}_t)p(\mathbf{x}_0|\mathbf{x}_t)d\mathbf{x}_0\\
&=\int p(\mathbf{y}|\mathbf{x}_0)p(\mathbf{x}_0|\mathbf{x}_t)d\mathbf{x}_0\\
&=\mathbb{E}_{\mathbf{x}_0\sim p(\mathbf{x}_0|\mathbf{x}_t)}p(\mathbf{y}|\mathbf{x}_0)
\end{aligned}
$$
使用$p(\mathbf{y}|\hat{\mathbf{x}}_0)$近似$p(\mathbf{y}|\mathbf{x}_t)$，其中

$$
\hat{\mathbf{x}}_0:=\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]=\mathbb{E}_{\mathbf{x}_0\sim p(\mathbf{x}_0|\mathbf{x}_t)}[\mathbf{x}_0]
$$


此时可以使用Jensen不等式估计误差上界：

$$
\mathcal{J}(f,\mathbf{x}\sim p(\mathbf{x}))=\mathbb{E}[f(\mathbf{x})]-f(\mathbb{E}[\mathbf{x}])
$$

针对 $\boldsymbol{n}\sim\mathcal{N}(0,\sigma^2\boldsymbol{I})$的情况，使用DPS进行后验近似，得到的Jensen误差满足

$$
\mathcal{J}\leq \frac{d}{\sqrt{2\pi\sigma^2}}e^{-1/2\sigma^2}||\nabla_{\boldsymbol{x}}\mathcal{A}(\boldsymbol{x})||m_1
$$
，其中 $||\nabla_{\boldsymbol{x}}\mathcal{A}(\boldsymbol{x})||:=\max_{\boldsymbol{x}}||\nabla_{\boldsymbol{x}}\mathcal{A}(\boldsymbol{x})||$， $m_1:=\int||\boldsymbol{x}_0-\hat{\boldsymbol{x}}_0||p(\boldsymbol{x}_0|\boldsymbol{x}_t)d\boldsymbol{x}_0$。
从该公式得到的推论为，当 $||\nabla_{\boldsymbol{x}}\mathcal{A}(\boldsymbol{x})||$与 $m_1$都是有界的时候，Jensen误差随着 $\sigma\rightarrow 0$时逐渐趋于0，即近似误差随着测量噪声的增长而降低。

对应的采样方式为
```pseudo
    \begin{algorithm}
    \caption{DPS-Gaussian}
    \begin{algorithmic}
	\Require    $N,\boldsymbol{y},\{\zeta_i\}_{i=1}^N,\{\tilde{\sigma}_i\}_{i=1}^N$
    $\boldsymbol{x}_N\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$
    \For{$i=N-1$ \TO 0}
    \State $\hat{\boldsymbol{s}}\leftarrow \boldsymbol{s}_{\theta}(\boldsymbol{x}_i,i)$
    \State $\hat{\boldsymbol{x}}_0\leftarrow \frac{1}{\sqrt{\overline{\alpha}}_i}(\boldsymbol{x}_i+(1-\overline{\alpha}_i)\hat{\boldsymbol{s}})$
    
    \State $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$
    
    \State $\boldsymbol{x}_{i-1}'\leftarrow \frac{\sqrt{\alpha_i}(1-\overline{\alpha}_{i-1})}{1-\overline{\alpha}_i}\boldsymbol{x}_i+\frac{\sqrt{\overline{\alpha}_{i-1}}\beta_i}{1-\overline{\alpha}_i}\hat{\boldsymbol{x}}_0+\tilde{\sigma}_i\boldsymbol{z}$
    
    \STATE $\boldsymbol{x}_{i-1}\leftarrow \boldsymbol{x}_{i-1}'-\zeta_i\nabla_{\boldsymbol{x}_i}||\boldsymbol{y}-\mathcal{A}(\hat{\boldsymbol{x}}_0)||_2^2$
    \EndFor
    \Return $\hat{\boldsymbol{x}}_0$
	\end{algorithmic}
    \end{algorithm}
```
