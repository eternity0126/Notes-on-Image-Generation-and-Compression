[DENOISING DIFFUSION IMPLICIT MODELS](https://arxiv.org/pdf/2010.02502)

### 摘要
DDPM依赖于马尔可夫假设，导致采样速度很慢。DDIM通过将DDPM的逆向去噪SDE变为常微分方程ODE，实现**采样加速**，**且不需要重新训练DDPM模型**。

### 背景
#### DDPM
DDPM的前向加噪过程服从马尔可夫链，是一个正态转移概率，形式为![[DDPM#^0ce7ed]]
根据马尔可夫性质，跳步加噪的解析解为![[DDPM#^34fb76]]
需要注意的点是，DDPM最终走向的分布$p_{\text{prior}}\sim\mathcal{N}(0,\boldsymbol{I})$。理论上这需要无穷多步，这就导致步数$T$足够大时，$\alpha_{T}$只能无限接近于0，加噪路径的终点是无法到达的。

### 方法
#### 去噪公式
[[DDPM#^cc26e0|DDPM反向去噪]]的公式中利用率Markov性质，将$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1},\boldsymbol{x}_{0})$替换为了$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})$。DDIM不使用Markov性质，则需要满足特定要求：
1. 观察DDPM的损失函数![[DDPM#^a4f030]]只使用了$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})$这个边缘概率分布。DDIM为了兼容DDPM模型，不能因为跳步而改变$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})$。
2. 为了便于采样，$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})$最好为高斯分布，从而通过重参数化进行$\boldsymbol{x}_{t-1}$的采样。
给出假设
$$
q_{\sigma_{t}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})\sim\mathcal{N}(\mu(\boldsymbol{x}_{0}, \boldsymbol{x}_{t}),\sigma_{t}^2\boldsymbol{I})
$$
其中$\mu(\boldsymbol{x}_{0},\boldsymbol{x}_{t})=k\boldsymbol{x}_{0}+m\boldsymbol{x}_{t}$，方差$\sigma_{t}^2>0$为任意常数。由此可以求解出$\boldsymbol{x}_{t-1}$。同时根据第一个要求，也可以根据$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})$推导出$\boldsymbol{x}_{t-1}$，则存在等式
$$
\sqrt{ \alpha_{t-1} }\boldsymbol{x}_{0}+\sqrt{ 1-\alpha_{t-1} }\epsilon_{t-1}=x_{t-1}=k\boldsymbol{x}_{0}+m\boldsymbol{x}_{t}+\sigma_{t}^2\epsilon_{t}'=k\boldsymbol{x}_{0}+m(\sqrt{ \alpha_{t} }\boldsymbol{x}_{0}+\sqrt{ 1-\alpha_{t} }\epsilon_{t}'')+\sigma_{t}^2\epsilon_{t}'
$$
上述式子，左边服从$\mathcal{N}(\sqrt{ \alpha_{t-1} }\boldsymbol{x}_{0},(1-\alpha_{t-1})\boldsymbol{I})$，右边服从$\mathcal{N}((k+m\sqrt{ \alpha_{t} })\boldsymbol{x}_{0},(m^2(1-\alpha_{t})+\sigma_{t}^2)\boldsymbol{I})$
可以联立等式
$$
\begin{aligned}
k+m\sqrt{ \alpha_{t} }&=\sqrt{ \alpha_{t-1} }\\
m^2(1-\alpha_{t})+\sigma_{t}^2&=1-\alpha_{t-1}
\end{aligned}
$$
最终解得
$$
q_{\sigma_{t}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})=\mathcal{N}\left(\sqrt{ \alpha_{t-1} }\boldsymbol{x}_{0}+\sqrt{ 1-\alpha_{t-1}- \sigma_{t}^2}\cdot\frac{\boldsymbol{x}_{t}-\sqrt{ \alpha_{t} }\boldsymbol{x}_{0}}{\sqrt{ 1-\alpha_{t} }},\sigma_{t}^2\boldsymbol{I}\right)
$$
需要注意的是，这里的$t$和$t-1$并不代表相邻的时间步，因为没有用到马尔可夫性质，可以理解为第$t$个与第$t-1$个时间步，如第800步与第700步。
使用DDPM模型预测噪声$\epsilon_{\theta}(\boldsymbol{x}_{t})$，通过$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})$预测$\hat{\boldsymbol{x}}_{0}$，代入上式可得
$$
\boldsymbol{x}_{t-1}=\sqrt{ \alpha_{t-1} }\left(\frac{\boldsymbol{x}_{t}-\sqrt{ 1-\alpha_{t} }\epsilon_{\theta}(\boldsymbol{x}_{t})}{\sqrt{ \alpha_{t} }}\right)+\sqrt{ 1-\alpha_{t-1}-\sigma_{t}^2 }\cdot\epsilon_{\theta}(\boldsymbol{x}_{t})+\sigma_{t}\epsilon_{t}
$$
当$\sigma_{t}=0$时，会消除采样的随机性。

#### 损失函数的等价性
DDIM对应了一个新的加噪过程，$q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1},\boldsymbol{x}_{0})$不能通过Markov性质进行化简。


#### 改进点
DDIM相比DDPM加噪过程和损失函数完全一致，唯一的改进点是去噪过程。
令$\sigma_{t}=0$，可以得出
$$
\begin{aligned}
&\boldsymbol{x}_{t-1}=\sqrt{ \alpha_{t-1} }\left(\frac{\boldsymbol{x}_{t}-\sqrt{ 1-\alpha_{t} }\epsilon_{\theta}(\boldsymbol{x}_{t})}{\sqrt{ \alpha_{t} }}\right)+\sqrt{ 1-\alpha_{t-1}-\sigma_{t}^2 }\cdot\epsilon_{\theta}(\boldsymbol{x}_{t})+\sigma_{t}\epsilon_{t}\\
&\Rightarrow \frac{\boldsymbol{x}_{t}}{\sqrt{ \alpha_{t} }}-\frac{\boldsymbol{x}_{t-1}}{\sqrt{ \alpha_{t-1} }}=\left( \sqrt{ \frac{1-\alpha_{t}}{\alpha_{t}} }-\sqrt{ \frac{1-\alpha_{t-1}}{\alpha_{t-1}} } \right)\epsilon_{\theta}(\boldsymbol{x}_{t})\\
& \Rightarrow \frac{\boldsymbol{x}_{t}}{\sqrt{ \alpha_{t} }}-\frac{\boldsymbol{x}_{t-\Delta t}}{\sqrt{ \alpha_{t-\Delta t} }}=\left( \sqrt{ \frac{1-\alpha_{t}}{\alpha_{t}} }-\sqrt{ \frac{1-\alpha_{t-\Delta t}}{\alpha_{t-\Delta t}} } \right)\epsilon_{\theta}(\boldsymbol{x}_{t})
\end{aligned}
$$
令$\frac{\boldsymbol{x}_{t}}{\sqrt{ \alpha_{t} }}=\bar{\boldsymbol{x}}_{t}$，$\sqrt{ \frac{1-\alpha_{t}}{\alpha_{t}} }=\sigma_{t}$，两边取极限可得
$$
d \bar{\boldsymbol{x}}_{t}=\epsilon_{\theta}\left(\frac{\bar{\boldsymbol{x}}_{t}(t)}{\sqrt{ \sigma^2+1 }}\right)d\sigma
$$
**可以发现DDIM逆向去噪是一个常微分方程ODE，对应着一致性模型中的概率流ODE。令SDE迭代的扰动项系数为0，可以将SDE变为一个概率流ODE。由于ODE不存在扰动，其数值求解更简单，求解速度快于SDE，且步数可以迈得更大。**