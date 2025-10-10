[SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456)


对于VP-SDE也即DDPM，有

$$
d\mathbf{x}=-\frac{1}{2}\beta(t)\mathbf{x}dt+\sqrt{\beta(t)}d\mathbf{w}
$$
$$
d\mathbf{x}=\left[-\frac{\beta(t)}{2}\mathbf{x}-\beta(t)\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)\right]dt+\sqrt{\beta(t)}d\overline{\mathbf{w}}
$$

### Score-based Diffusion Models
使用随机微分方程（SDE，Stochastic Differential Equation)表述扩散过程。其中前向过程为
$$
d\boldsymbol{x}=\boldsymbol{f}(\boldsymbol{x},t)dt+g(t)d\boldsymbol{w}
$$
反向过程为
$$
d\boldsymbol{x}=[\boldsymbol{f}(\boldsymbol{x},t)-g^2(t) \nabla a_ {\boldsymbol{x}}\log p_ {t}(\boldsymbol{x})]dt+g(t)d\overline{\boldsymbol{w}}
$$
针对Variance-Preserving SDE，也即DDPM，其前向过程为
$$
d\boldsymbol{x}=-\frac{\beta(t)}{2}\boldsymbol{x}dt+\sqrt{\beta(t)}d\boldsymbol{w}
$$

反向过程为

$$
d\boldsymbol{x}=\left[-\frac{\beta(t)}{2}\boldsymbol{x}-\beta(t)\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right]dt+\sqrt{\beta(t)}d\overline{\boldsymbol{w}}
$$

^f8b96d

