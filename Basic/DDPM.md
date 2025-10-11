[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)


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
前向扩散过程支持任意时间步的采样，使用$\alpha_{t}:=1-\beta_{t}$以及$\overline{\alpha}_{t}:=\prod_{s=1}^t\alpha_{s}$，有
$$
q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})=\mathcal{N}(\boldsymbol{x}_{t};\sqrt{ \overline{\alpha}_{t}}\boldsymbol{x}_{0},(1-\overline{\alpha}_{t})\boldsymbol{I})
$$从而可以推导出后验分布：
$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})=\mathcal{N}\left(\boldsymbol{x}_{t-1};\frac{\sqrt{ \overline{\alpha}_{t-1} }\beta_{t}}{1-\overline{\alpha}_{t}}\boldsymbol{x}_{0}+\frac{\sqrt{ \alpha_{t} }(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}\boldsymbol{x}_{t},\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_{t}\right)
$$

### Diffusion models and denoising autoencoders
本论文在上述定义下做出了以下限制：
- 将原本可参数化的$\beta_{t}$设为固定值（范围在$0.001 \sim 0.02$），此时$L_{T}$为常数，可忽略。


$$
\boldsymbol{x}_{i-1}\leftarrow \frac{\sqrt{ \overline{\alpha}_{t-1} }\beta_{t}}{1-\overline{\alpha}_{t}}\hat{\boldsymbol{x}}_{0}+\frac{\sqrt{ \alpha_{t} }(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}\boldsymbol{x}_{t}\underbrace{-\zeta_{t}\nabla_{\boldsymbol{x}_{t}}\Vert \boldsymbol{y}-\mathcal{A}(\hat{\boldsymbol{x}}_{0})\Vert_{2}^2}_{\text{DPS}}+\tilde{\sigma}\underbrace {\boldsymbol{z}}_{\text{DiffC编码的部分}}
$$