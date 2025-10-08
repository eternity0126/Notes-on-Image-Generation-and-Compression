DIFFUSION POSTERIOR SAMPLING FOR GENERAL NOISY INVERSE PROBLEMS


DPS可以实现引导扩散模型的生成方向？
定义逆问题（inverse problem）为已知前向操作$\mathcal{A}$和噪声$n$从观测量$y$中复原$x$。其中前向操作为
$$
y=\mathcal{A}(x)+n
$$
如果想从$p(x|y)$中采样，则可以使用$\nabla_{x}\log p(y|x)$实现。
SDE的反向过程：
$$
d\mathbf{x}=\left[-\frac{\beta(t)}{2}\mathbf{x}-\beta(t)\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)\right]dt+\sqrt{\beta(t)}d\overline{\mathbf{w}}
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
\mathcal{J}(f,\mathbf)
$$