# 有关图像生成和图像压缩的论文阅读笔记
笔记使用[Obsidian](https://obsidian.md/)创建，使用的插件包含Git、PseudoCode。建议使用Obsidian查看。

本仓库的笔记主要记录了图像生成领域以及基于生成模型实现的图像压缩的相关论文。


2025.10.08：目前主要先了解Diffusion部分的进展。

### 图像生成
#### 扩散模型基础理论
- [DDPM](Basic/DDPM.md)：继GAN之后的图像生成的基本范式。
- [VDM](Basic/VDM.md)：相比于DDPM，引入了连续时间下的扩散过程，并通过直接优化ELBO对模型进行训练。
- [SDE diffusion](Basic/SDE_Diffusion.md)：通过使用随机微分方程将NCSN（VE-SDE）和DDPM（VP-SDE）统一起来。
- [Flow matching](Basic/Flow_Matching.md)

#### 推理采样
- [DPS](Sampling/DPS.md)：使用 $p(\boldsymbol{y}|\hat{\boldsymbol{x}} _0)$近似 $p(\boldsymbol{y}|\boldsymbol{x} _t)$优化后验采样。

#### 生成控制
- [Classifier guidance](Conditional_Diffusion/Guided_Diffusion.md)
- [Classifier-free guidance](Classifier_Free_Guidance.md)
- ControlNet
- IP-Adapter

### 图像压缩
- [VDM](Basic/VDM.md)：由于直接优化对数似然 $\log p(\boldsymbol{x})$，可以结合算数编码或者ANS实现实现针对图像的无损压缩。
- [DiffC](Compression/DiffC.md)：使用扩散模型结合随机编码实现了图像有损压缩。2025年ICLR论文的实现提升了编解码器效率，验证了预训练的扩散模型可以实现zero-shot的图像有损压缩。