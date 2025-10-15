# DeepLearning

2025-10-11  学习1-pytorch入门到张量的基本运算-张量的形状操作
2025-10-12  学习1-pytorch入门结束，开始2-神经网络到LeNet实现MNIST识别
2025-10-13  学习2-神经网络，完成AlexNet和ResNet，开始3-Transformer到attention
2025-10-14  重新学习3-Transformer，完成encoder学习
2025-10-15  完成3-Transformer，形成一版完整的Transformer，完成4-transformer-translation，将自己写的transformer用于英译中任务，完成5-ViT，实现ViT模型

2025-10-15总结

这段时间系统学习了深度学习的核心内容，从PyTorch基础 → CNN → Transformer → ViT，几乎涵盖了现代深度学习的核心脉络

首先，从 PyTorch 基础入手，掌握了张量运算、自动求导机制，并用 PyTorch 实现了线性函数的训练，熟悉了模型训练的完整流程，尤其是掌握了代码实现的方法。

随后入门神经网络，实现了多层感知机（MLP），接着进一步学习经典卷积网络 LeNet，并在 MNIST 上完成训练，训练完成后尝试通过加入 BatchNorm 和 Dropout 提升效果。LeNet完成之后进一步学习了 AlexNet 与 ResNet，理解了深层卷积网络的结构特点与残差连接在缓解梯度消失中的作用。

接着进入 Transformer，学习掌握了 Self-Attention、多头注意力、位置编码等关键机制以及 transformer 的 encoder-decoder 结构，反复打磨，代码实现了transformer，并在此基础上将自己实现的 Transformer 应用于英译中任务。

最后学习了 Vision Transformer（ViT）的原理，理解了将图像分块后转化为序列、利用 Transformer 进行全局建模的思路，并完成了代码实现。

其实从大一就开始接触深度学习，但是当时积累还太少，没有真正掌握神经网络的代码实现方式，当时应该还执着于用基础python写出来，遇到了很多问题。后来上过一些人工智能课，对卷积神经网络等的原理进行了学习，但是由于利用GPT，实验作业往往变成了复制粘贴，没有自己去敲代码实现，因而一直对深度学习有种神秘的感觉。

非常庆幸现在能有时间去认真的重新学一次，去自己敲一遍代码，现在发现其实在 pytorch 的辅助下，模型网络实现起来还是非常简单的，函数封装的很好，用起来就没有什么细节上的复杂问题。看起来代码量大，其实理解了每一部分的意义，发现代码能跟原理对应的上，也就觉得掌握了。

接下来准备对 CLIP 进行学习，然后对 swing transformer 等transformer的变体也很好奇，想去了解。