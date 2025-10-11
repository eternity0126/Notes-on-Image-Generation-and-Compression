Paper1:[LOSSY COMPRESSION WITH GAUSSIAN DIFFUSION](https://arxiv.org/pdf/2206.08889)
Paper2:[LOSSY COMPRESSION WITH PRETRAINED DIFFUSION MODELS](https://arxiv.org/pdf/2501.09815)

### 摘要
使用扩散过程实现结合随机编码实现了图像的有损压缩。

### Reverse Channel Coding
发送采样$x\sim q(x)$，接收方共享一个已知的分布$p(x)$（如高斯分布），则传输$x$的最小代价为$D_ {\text{KL}}(q \Vert p)$。
目前RCC最好的实现为PFR算法，编码代价最大为
$$
D_{\text{KL}}(q\Vert p)+\log (D_{\text{KL}}(q\Vert p))+5
$$
存在的问题是PFR算法的复杂度为$D_{\text{KL}}(q\Vert p)$的指数级。
### DiffC
算法的步骤为
1. 发送方使用RCC传输一个部分加噪的样本$x_t$。
2. 接收方使用RCC重建$x_t$，之后使用扩散模型继续去噪$x_t \rightarrow x_0$。

在该方案下，传输$\mathbf{x}_t$的最小代价为
$$
\sum_{i=T}^{t+1} D_ {\text{KL}}[q(x_{i-1}|x_i)\Vert p_{\theta}(x_{i-1}\Vert x_i)
$$

Paper2的workaround：
1. RCC加速
2. 使用贪婪算法选取最优的编码时间步序列
3. 每个时间步的KL散度必须发送方和接收方都已知，论文避免了传输side information，而是根据小规模数据集测试的结果设置了硬编码的KL散度，对实际的RD性能影响较小。