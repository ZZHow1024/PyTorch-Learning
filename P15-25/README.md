# PyTorch深度学习入门笔记P15-25

@ZZHow(ZZHow1024)

参考课程：

【**PyTorch深度学习快速入门教程【小土堆】**】

[https://www.bilibili.com/video/BV1hE411t7RN]

# P15. 神经网络的基本骨架-nn.Module的使用

- 前向传播：input → forward → output
- 案例演示：[**p15_nn_module.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p15_nn_module.py)

# P16. 土堆说卷积操作

- 概念：卷积是一种数学运算，广泛应用于信号处理、图像处理和深度学习（尤其是卷积神经网络CNN）。其核心思想是：**用一个小的“滑动窗口”（卷积核/滤波器）在输入数据上滑动，计算局部区域的加权和**，从而提取局部特征。
- 核心组件
    - **输入（Input）**
        - **形式**：通常是多维数组（如图像是2D矩阵，视频是3D张量）。
    - **卷积核（Kernel/Filter）**
        - **作用**：提取输入的局部特征（如边缘、颜色变化）。
    - **卷积操作（Convolution Operation）**
        - **步骤**：
            1. **滑动窗口**：卷积核在输入上按步长（stride）滑动（如步长=1时逐像素移动）。
            2. **局部相乘求和**：卷积核与当前覆盖的输入区域逐元素相乘后求和，得到输出的一个值。
            3. **填充（Padding）**（可选）：在输入边缘填充0，控制输出尺寸。
    - **输出（Feature Map）**
        - **作用**：卷积操作的结果，表示输入数据在特定特征上的响应强度。
- `torch.nn.functional.conv2d`
    - 作用：实现 **2D 卷积操作** 的底层函数（与 `nn.Conv2d` 模块功能相同，但更灵活）。
    - 常用参数
        - **`input` (必需)**：输入张量，格式为 **4D 张量**，表示一批图像数据。
        - **`weight` (必需)**：卷积核权重，格式为 **4D 张量**。
        - **`bias` (默认值: `None`)**：偏置项，格式为 **1D 张量**。
        - **`stride` (默认值: `1`)**：控制卷积核滑动的步长（步幅）。
        - **`padding` (默认值: `0`)**：在输入张量的边缘填充 `0`，控制输出尺寸。
        - **`dilation` (默认值: `1`)**：控制卷积核的**空洞率**（dilated convolution），扩大感受野。
        - **`groups` (默认值: `1`)**：控制**分组卷积**的分组数，用于减少参数量或实现特定结构（如深度可分离卷积）。
- 案例演示：[**p16_nn_conv.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p16_nn_conv.py)

# P17. 神经网络-卷积层

- `torch.nn.conv2d`
    - 作用：用于实现 **2D 卷积** 的标准模块（类），它封装了 `torch.nn.functional.conv2d` 的功能，并自动管理权重和偏置参数。
    - 常用参数
        - **`in_channels` (必需)**：输入数据的通道数。
        - **`out_channels` (必需)**：卷积核的数量（即输出通道数），决定了输出特征图的深度。
        - **`kernel_size` (必需)**：卷积核的大小（高度和宽度）。
        - **`stride` (默认值: `1`)**：控制卷积核滑动的步长（步幅）。
        - **`padding` (默认值: `0`)**：在输入张量的边缘填充 `0`，控制输出尺寸。
        - **`padding_mode` (默认值: `'zeros'`)**：指定填充模式（仅当 `padding>0` 时生效）。
        - **`dilation` (默认值: `1`)**：控制卷积核的**空洞率**（dilated convolution），扩大感受野。
        - **`groups` (默认值: `1`)**：控制**分组卷积**的分组数，用于减少参数量或实现特定结构（如深度可分离卷积）。
        - **`bias` (默认值: `True`)**：是否在卷积后添加偏置项。
- 案例演示：[**p17_nn_conv2d.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p17_nn_conv2d.py)

# P18. 神经网络-最大池化的使用

- `torch.nn.MaxPool2d`
    - 作用：对输入特征图进行下采样（降维），保留局部区域的最大值以提取显著特征并减少计算量。
    - 常用参数
        - **`kernel_size` (必需)**：池化窗口的大小（高度和宽度）。
        - **`stride` (默认值: `None`，即等于 `kernel_size`)**：控制池化窗口滑动的步长（步幅）。
        - **`padding` (默认值: `0`)**：在输入特征图的边缘填充 `0`，控制输出尺寸。
        - **`dilation` (默认值: `1`)**：控制池化窗口的**空洞率**（dilated pooling），扩大感受野。
        - **`return_indices` (默认值: `False`)**：是否返回最大值在输入特征图中的**索引位置**。
        - **`ceil_mode` (默认值: `False`)**：控制输出尺寸的计算方式。
- 案例演示：[**p18_nn_maxpool.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p18_nn_maxpool.py)

# P19. 神经网络-非线性激活

- `torch.nn.ReLU`
    - 作用：将所有负值置为 `0`，正值保持不变。
    - 常用参数
        - **`inplace` (默认值: `False`)**：是否进行**原地操作**（in-place operation），即直接修改输入张量的值，而非创建新的张量。
- `torch.nn.Sigmoid`的常用参数
    - 作用：将输入 *x* 映射到 `(0, 1)` 区间，输出值可解释为概率。
- 案例演示：[**p19_nn_relu.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p19_nn_relu.py)

# P20. 神经网络-线性层及其他层介绍

- 总结对比表
    
    
    | 层类型 | 核心功能 | 典型应用场景 |
    | --- | --- | --- |
    | Normalization | 数据归一化 | 加速训练，稳定梯度 |
    | Recurrent | 序列建模（时序依赖） | NLP、时间序列预测 |
    | Transformer | 自注意力机制 | 机器翻译、视觉Transformer |
    | Linear | 线性变换 | 分类器、特征映射 |
    | Dropout | 正则化（防过拟合） | 所有深度学习模型 |
    | Sparse | 稀疏数据高效处理 | 推荐系统、图神经网络 |
- 线性层
    - `torch.flatten`
        - **功能**：将输入张量的指定维度范围（从 `start_dim` 到 `end_dim`）合并为一个维度，生成一个新的展平后的张量。
        - **常用参数**
            - `input`（必须）：输入的任意维度张量（如 1D、2D、3D、4D 等）。
            - `start_dim`（默认为 0）：**开始展平的维度索引**（从该维度起，后续维度会被合并）。
            - `end_dim`（默认为 -1）：**结束展平的维度索引**（到该维度止，所有中间维度会被合并）。负数表示从后往前计数（如 `-1` 表示最后一个维度）。
    - `torch.nn.Linear`
        - **功能**：对输入张量的最后一个维度执行线性变换。
        - 常用参数
            - **`in_features` (必需)**：输入张量最后一个维度的大小（即输入特征的维度）。
            - **`out_features` (必需)**：输出张量最后一个维度的大小（即输出特征的维度）。
            - **`bias` (默认值: `True`)**：是否为线性变换添加偏置项 *b*。
- 案例演示：[**p20_nn_linear.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p20_nn_linear.py)

# P21. 神经网络-搭建小实战和Sequential的使用

- CIFAR10 模型结构
    
    ![CIFAR10模型结构](https://www.notion.so/image/attachment%3A2035678e-d864-43a3-b685-03356580c941%3ACIFAR10%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png?table=block&id=243e64bd-e40f-80cd-9b26-c0f721b36562&t=243e64bd-e40f-80cd-9b26-c0f721b36562)
    
    CIFAR10模型结构
    
- `torch.nn.Sequential`的**功能**：将多个神经网络模块（如卷积层、全连接层、激活函数等）按**定义的顺序**组合成一个整体模型，数据会**依次通过每个模块**，无需手动编写前向传播代码。
- 案例演示：[**p21_nn_seq.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p21_nn_seq.py)

# P22. 损失函数与反向传播

- `torch.nn.L1Loss`
    - 功能：计算预测值与目标值之间 **绝对差值的平均值（或总和）**。
    - 常用参数
        - **`reduction='mean'`（默认）**：指定如何对每个样本的损失进行汇总，有三个可选值：`'none'`、`'mean'`、`'sum'`。
- `torch.nn.CrossEntropyLoss`
    - 功能：**交叉熵损失函数**，**主要用于多分类任务（multi-class classification）**，比如图像分类、文本分类等场景。
    - 常用参数
        - `weight`**（默认值：None）**：为每个类别指定一个权重，用于处理类别不平衡问题。形状为 `[num_classes]`。
        - `ignore_index`**（默认值：-100）**：指定一个目标类别索引，该类别的损失将被忽略（常用于分割任务中的背景类）
        - `reduction`**（默认值：'mean’）**：指定如何汇总损失：`'none'`、`'mean'`（默认）、`'sum'`。
    - 输入要求
        - **输入（logits）**：模型的原始输出，形状为 `[batch_size, num_classes]`，是 **未经过 softmax 的分数（logits）**。
        - **目标（target）**：每个样本的真实类别索引，是一个 **整数张量**，形状为 `[batch_size]`，其中的值范围是 `[0, num_classes - 1]`。
- 反向传播`result_loss.backward()`
    - 调用 `result_loss.backward()`，**自动计算所有可学习参数的梯度**（即 ∂loss/∂weight, ∂loss/∂bias 等）。
- 案例演示：[**p22_nn_loss.py](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p22_nn_loss.py) 和 [p22_nn_loss_network.py](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p22_nn_loss_network.py)**

# P23. 优化器

- 什么是优化器（Optimizer）
    - 作用：**根据梯度信息，智能地调整模型参数，让损失函数下降得更快、更稳定。**
    - 在训练神经网络的过程中：
        1. 我们通过**前向传播**计算预测值；
        2. 通过**损失函数**计算预测值与真实值之间的误差；
        3. 通过**反向传播**计算损失对模型参数的梯度；
        4. 最后，**优化器** 根据这些梯度来更新模型的参数，使损失逐步减小。
- **`torch.optim` 中常见的优化器**
    
    
    | 优化器类名 | 简介 |
    | --- | --- |
    | `torch.optim.SGD` | 随机梯度下降（Stochastic Gradient Descent），最基础的优化器 |
    | `torch.optim.Adam` | 自适应矩估计优化器，结合了动量和自适应学习率，非常流行 |
    | `torch.optim.AdamW` | Adam 的改进版本，解决了权重衰减实现上的问题 |
    | `torch.optim.RMSprop` | 均方根传播优化器，适合非平稳目标，常用于 RNN |
    | `torch.optim.Adagrad` | 自适应学习率优化器，适合稀疏数据 |
    | `torch.optim.Adadelta` | Adagrad 的改进版，不需要手动设置初始学习率 |
    | `torch.optim.LBFGS` | 二阶优化方法，适合小批量问题，但计算开销较大 |
- 共同参数
    
    
    | 参数名 | 类型 | 含义 |
    | --- | --- | --- |
    | `params` | Iterable | 需要优化的模型参数（通常传入 `model.parameters()`） |
    | `lr` (learning rate) | float | **学习率**，控制每次参数更新的步长，是最重要的超参数之一 |
    | `weight_decay` | float | **权重衰减（L2 正则化）**，用于防止过拟合，默认值通常为 0 |
    | `momentum` | float (部分优化器支持) | 动量因子，用于加速 SGD 在相关方向上的收敛，减少震荡（如 SGD、RMSprop） |
    | `dampening` | float (部分优化器支持) | 动量的抑制因子，通常与 momentum 配合使用（如 SGD） |
    | `nesterov` | bool (部分优化器支持) | 是否使用 Nesterov 动量（如 SGD），可以让动量“预见未来”，效果更好 |
    | `eps` | float (部分优化器支持) | 数值稳定性常数，防止除零错误（如 Adam、RMSprop） |
    | `amsgrad` | bool (Adam 相关优化器支持) | 是否使用 AMSGrad 变体，对学习率进行更严格的约束（Adam/AdamW） |
- 使用流程
    
    ```python
    # 训练循环
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    
            # 反向传播
            optimizer.zero_grad()  # 清空之前的梯度
            loss.backward()        # 计算新的梯度
    
            # 参数更新
            optimizer.step()       # 更新模型参数
    ```
    
- 案例演示：[**p23_nn_optim.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p23_nn_optim.py)

# P24. 现有网络模型的使用及修改

- `torchvision.models`核心功能
    - **图像分类模型**（最常用）：如 ResNet、VGG、AlexNet、EfficientNet、MobileNet 等。
    - **目标检测/实例分割模型**：如 Faster R-CNN、Mask R-CNN、RetinaNet 等。
    - **语义分割模型**：如 FCN、DeepLabV3、U-Net 等。
    - **其他视觉任务模型**：如生成对抗网络（GAN）、光流估计等。
- `pretrained` 参数详解
    - **`pretrained=True`**：加载官方提供的预训练权重（模型参数初始化为已训练好的值），适合直接用于推理或迁移学习。
    - **`pretrained=False`（默认值）**：模型参数初始化为随机值，需从头开始训练（适合自定义任务且无预训练需求时）。
- 修改现有网络模型
    
    ```python
    # 添加 module
    vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))
    
    # 修改 module
    vgg16.classifier[6] = nn.Linear(1000, 10)
    
    ```
    
- 案例演示：[**p24_model_pretrained.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p24_model_pretrained.py)

# P25. 网络模型的保存与读取

- 保存方式 1（模型结构 + 模型参数）
    - 保存
        
        ```python
        torch.save(model, 'model_name.pth')
        ```
        
    - 读取
        
        ```python
        model = torch.load('model_name.pth', weights_only=False)
        ```
        
- 保存方式 2（模型参数）
    - 保存
        
        ```python
        torch.save(model.state_dict(), 'model_name.pth')
        ```
        
    - 读取
        
        ```python
        model.load_state_dict(torch.load('model_name.pth'))
        ```
        
- 案例演示：[**p25_model_load.py](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p25_model_load.py) 和 [p25_model_save.py](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P15-25/p25_model_save.py)**