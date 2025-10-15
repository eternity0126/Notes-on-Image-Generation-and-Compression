[SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456)

### 摘要
通过使用随机微分方程将NCSN（VE-SDE）和DDPM（VP-SDE）统一起来。 ^83fdd9

### Score-based Diffusion Models
使用随机微分方程（SDE，Stochastic Differential Equation)表述扩散过程。其中前向过程为
$$
d\boldsymbol{x}=\boldsymbol{f}(\boldsymbol{x},t)dt+g(t)d\boldsymbol{w}
$$
使用$\Delta t$表示，可以写成
$$
\boldsymbol{x}_{t+\Delta t}-\boldsymbol{x}_{t}=\boldsymbol{f}(\boldsymbol{x},t)\Delta t+g(t)\sqrt{ \Delta t }\epsilon
$$
反向过程为
$$
d\boldsymbol{x}=[\boldsymbol{f}(\boldsymbol{x},t)-g^2(t) \nabla a_ {\boldsymbol{x}}\log p_ {t}(\boldsymbol{x})]dt+g(t)d\overline{\boldsymbol{w}}
$$
针对Variance-Preserving SDE，也即DDPM，其前向过程为
$$
d\boldsymbol{x}=-\frac{\beta(t)}{2}\boldsymbol{x}dt+\sqrt{\beta(t)}d\boldsymbol{w}
$$

^7a2a2d

反向过程为

$$
d\boldsymbol{x}=\left[-\frac{\beta(t)}{2}\boldsymbol{x}-\beta(t)\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right]dt+\sqrt{\beta(t)}d\overline{\boldsymbol{w}}
$$

^f8b96d

