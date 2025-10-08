[SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456)

SDE为
前向过程为
$$
d\mathbf{x}=\mathbf{f}(\mathbf{x},t)dt+g(t)d\mathbf{w}
$$
反向过程为
$$
d\mathbf{x}=[\mathbf{f}(\mathbf{x,t})-g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})]dt+g(t)d\overline{\mathbf{w}}
$$


对于VP-SDE，有
$$
d\mathbf{x}=-\frac{1}{2}\beta(t)\mathbf{x}dt+\sqrt{\beta(t)}d\mathbf{w}
$$
$$
d\mathbf{x}=\left[-\frac{\beta(t)}{2}\mathbf{x}-\beta(t)\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)\right]dt+\sqrt{\beta(t)}d\overline{\mathbf{w}}
$$