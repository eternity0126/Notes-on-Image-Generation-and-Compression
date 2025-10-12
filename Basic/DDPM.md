[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)
### 摘要
本论文提出了使用扩散模型进行图像生成任务。论文在之前的DDPM的理论上对扩散过程加上了部分限制，并构建了简化后的变分下界损失函数，通过使用简化的损失函数让模型预测噪声。

### 背景知识
DDPM可以视为HVAE的一种特例，其编码过程为基于马尔可夫链的固定过程，可视为向数据逐步添加高斯噪声：
$$
q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0}):=\prod_{t=1}^Tq(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1}),\quad q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1}):=\mathcal{N}(\boldsymbol{x}_{t};\sqrt{ 1-\beta_{t} }\boldsymbol{x}_{t-1},\beta_{t}\boldsymbol{I})
$$
其优化目标为常见的变分下界：
$$
\mathbb{E}[-\log p_{\theta}(\boldsymbol{x}_{0})]\leq \mathbb{E}_{q}\left[-\log \frac{p_{\theta}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}\right]=\mathbb{E}_{q}\left[-\log p(\boldsymbol{x}_{T})-\sum_{t\geq 1}\log \frac{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})}\right]=:L
$$
损失函数可以进一步简化为
$$
L:=\mathbb{E}_{q}\left[\underbrace{D_{\text{KL}}(q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0}) \Vert p(\boldsymbol{x}_{T}))}_{L_{T}}
+\sum_{t>1}\underbrace{D_{\text{KL}}(q(\boldsymbol{x}_{t-1}\vert \boldsymbol{x}_{t},\boldsymbol{x}_{0})\Vert p_{\theta}(\boldsymbol{x}_{t-1}\vert \boldsymbol{x}_{t}))}_{L_{t-1}}\underbrace{-\log p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})}_{L_{0}}\right]
$$

^2936e5

前向扩散过程支持任意时间步的采样，使用$\alpha_{t}:=1-\beta_{t}$以及$\overline{\alpha}_{t}:=\prod_{s=1}^t\alpha_{s}$，有
$$
q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})=\mathcal{N}(\boldsymbol{x}_{t};\sqrt{ \overline{\alpha}_{t}}\boldsymbol{x}_{0},(1-\overline{\alpha}_{t})\boldsymbol{I})
$$

^34fb76

从而可以推导出后验分布：
$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})=\mathcal{N}\left(\boldsymbol{x}_{t-1};\frac{\sqrt{ \overline{\alpha}_{t-1} }\beta_{t}}{1-\overline{\alpha}_{t}}\boldsymbol{x}_{0}+\frac{\sqrt{ \alpha_{t} }(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}\boldsymbol{x}_{t},\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_{t}\right)
$$

### Diffusion models and denoising autoencoders
本论文在上述定义下做出了以下限制：
- 将原本可参数化的$\beta_{t}$设为固定值（范围在$0.0001 \sim 0.02$），此时$L_{T}$为常数，可忽略。
- 将$p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})=\mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{t},t),\boldsymbol{\Sigma}_{\theta}(\boldsymbol{x}_{t},t))$的方差限制为常数$\boldsymbol{\Sigma}_{\theta}(\boldsymbol{x}_{t},t)=\sigma_{t}^2 \boldsymbol{I}$

在上述限制下，[[DDPM#^2936e5|DDPM损失函数]]中的$L_{t-1}$可简化为
$$L_{t-1}=\mathbb{E}_{q}\left[\frac{1}{2\sigma^2}\Vert \tilde{\boldsymbol{\mu}}_{t}(\boldsymbol{x}_{t},\boldsymbol{x}_{0})-\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{t},t)\Vert^2\right]+C
$$

^8932d1

最直观的方式是使用一个模型预测$\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{t},t)$，但是可以借助[[DDPM#^34fb76|前向扩散过程的公式]]，使用
$$
\boldsymbol{x}_{0}=\frac{1}{\sqrt{ \overline{\alpha}_{t} }}(\boldsymbol{x}_{t}(\boldsymbol{x}_{0},\boldsymbol{\epsilon})-\sqrt{ 1-\overline{\alpha}_{t} }\boldsymbol{\epsilon})
$$
替换[[DDPM#^8932d1|式子]]中的$\boldsymbol{x}_{0}$，从而有
$$
\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{t},t)=\tilde{\boldsymbol{\mu}}_{t}\left(\boldsymbol{x}_{t}, \frac{1}{\sqrt{ \overline{\alpha}_{t} }}(\boldsymbol{x}_{t}-\sqrt{ 1-\overline{\alpha}_{t} }\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_{t}))\right)=\frac{1}{\sqrt{ \alpha_{t} }}\left(\boldsymbol{x}_{t}-\frac{\beta_{t}}{\sqrt{ 1-\overline{\alpha}_{t} }}\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_{t},t)\right)
$$
将预测$\boldsymbol{\mu}_{\theta}(\boldsymbol{x}_{t},t)$转变为了从$\boldsymbol{x}_{t}$中预测噪声$\boldsymbol{\epsilon}$。进而进一步将$L_{t-1}$简化为
$$
L_{t-1}-C=\mathbb{E}_{\boldsymbol{x}_{0},\boldsymbol{\epsilon}}\left[\frac{\beta_{t}^2}{2\sigma_{t}^2\alpha_{t}(1-\overline{\alpha}_{t})}\Vert \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\theta}(\sqrt{ \overline{\alpha}_{t} }\boldsymbol{x}_{0}+\sqrt{ 1-\overline{\alpha}_{t} }\boldsymbol{\epsilon},t)\Vert ^2\right]
$$
忽略加权系数，最终训练使用的损失函数为
$$
L_{\text{simple}}(\theta):=\mathbb{E}_{t,\boldsymbol{x}_{0},\boldsymbol{\epsilon}}\left[\Vert\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\theta}(\sqrt{ \overline{\alpha}_{t} }\boldsymbol{x}_{0}+\sqrt{ 1-\overline{\alpha}_{t} }\boldsymbol{\epsilon},t) \Vert^2 \right]
$$
在更大时间步的情况下，原始的$L_{t-1}$的加权系数会更小，忽略加权系数的作用则能使模型更好地关注大时间步情况下的噪声预测。论文也通过实验分析出预测噪声$\boldsymbol{\epsilon}$相比预测$\tilde{\boldsymbol{\mu}}_{t}$在使用不加权损失函数$L_{\text{simple}}(\theta)$会有更好的生成效果。

### 模型训练
训练时每次随机采样时间步$t$与高斯噪声$\boldsymbol{\epsilon}$，构建$\boldsymbol{x}_{t}$，将$\boldsymbol{x}_{t}$与$t$作为模型输入，让模型预测添加的噪声$\boldsymbol{\epsilon}_{\theta}$。
```pseudo
    \begin{algorithm}
    \caption{Training}
    \begin{algorithmic}
	\Repeat
    \State $\boldsymbol{x}_0\sim q(\boldsymbol{x}_0)$
    \State $t\sim \text{Uniform}(\{1,\cdots, T\})$
    \State $\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$
    \State Take gradient descent step on 
    \State $\quad \nabla_{\theta}\Vert\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\theta}(\sqrt{ \overline{\alpha}_{t} }\boldsymbol{x}_{0}+\sqrt{ 1-\overline{\alpha}_{t} }\boldsymbol{\epsilon},t) \Vert^2$
    \Until{converged }
	\end{algorithmic}
    \end{algorithm}
```
采样时初始化随机高斯噪声$\boldsymbol{x}_{T}$，让模型预测噪声逐步去噪。
```pseudo
    \begin{algorithm}
    \caption{Sampling}
    \begin{algorithmic}
	\State $\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$
	\For{$t=T,\cdots,1$}
		\State $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ if t> 1, else $\boldsymbol{z}=\boldsymbol{0}$
		\State $\boldsymbol{x}_{t-1}=\frac{1}{\sqrt{ \alpha_{t} }}\left(\boldsymbol{x}_{t}-\frac{\beta_{t}}{\sqrt{ 1-\overline{\alpha}_{t} }}\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_{t},t)\right)+\sigma_t \boldsymbol{z}$
    \EndFor
    \Return $\boldsymbol{x}_0$
	\end{algorithmic}
    \end{algorithm}
```
本论文设置的超参数为$\beta_{1}=10^{-4}$，$\beta_{T}=0.02$，采样时间步为$T=1000$。