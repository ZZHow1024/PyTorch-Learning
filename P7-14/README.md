# PyTorch深度学习入门笔记P7-14

@ZZHow(ZZHow1024)

参考课程：

【**PyTorch深度学习快速入门教程【小土堆】**】

[https://www.bilibili.com/video/BV1hE411t7RN]

# P7. TensorBoard的使用（一）

- TensorBoard 的安装与导入
    
    `pip install -i tensorboard`
    
    ```python
    from torch.utils.tensorboard import SummaryWriter
    ```
    
- add_scalar() 的使用
    
    ```python
    writer = SummaryWriter('logs')
    
    # y = x
    for i in range(100):
        writer.add_scalar("y = x", i, i)
    
    writer.close()
    ```
    
- TensorBoard 的启动
    
    `tensorboard --logdir=logs --port=6006`
    
- 案例演示：[**p7_tensorboard_1.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p7_tensorboard_1.py)

# P8. TensorBoard的使用（二）

- add_image() 的使用（常用来观察训练结果）
    
    ```python
    writer = SummaryWriter('logs')
    
    image_path = 'xxx'
    image_pil = Image.open(image_path)
    image_array = np.array(image_pil)
    
    writer.add_image('image', image_array, 1, dataformats='HWC')
    
    writer.close()
    ```
    
- 案例演示：[**p8_tensorboard_2.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p8_tensorboard_2.py)

# P9. Transforms的使用（一）

- transforms 该如何使用
    
    ```python
    tensor_transform = transforms.ToTensor()
    tensor_image = tensor_transform(image)
    ```
    
- 案例演示：[**p9_transforms_1.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p9_transforms_1.py)

# P10. Transforms的使用（二）

- 为什么需要 Tensor 数据类型
    - Tensor（张量）是PyTorch、TensorFlow等深度学习框架中的核心数据结构，本质上是**多维数组**（类似于NumPy的`ndarray`），但具备更强大的功能，专为高效计算和硬件加速设计。
    - Tensor 是深度学习的基石，它统一了多维数据的表示、支持硬件加速和自动微分，是构建高效、灵活神经网络的必备工具。
- 案例演示：[**p10_transforms_2.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p10_transforms_2.py)

# P11. 常见的Transforms（一）

- ToTensor 的使用
    
    ```python
    transforms_tensor = transforms.ToTensor()
    image_tensor = transforms_tensor(image)
    ```
    
- Normalize 的使用
    
    ```python
    transforms_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    image_norm = transforms_normalize(image_tensor)
    ```
    
- 案例演示：[**p11_transforms_use_1.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p11_transforms_use_1.py)

# P12. 常见的Transforms（二）

- Resize 的使用
    
    ```python
    transforms_resize = transforms.Resize((512, 512))
    image_resize = transforms_resize(image)
    image_resize = transforms_tensor(image_resize)
    ```
    
- Compose 的使用
    
    ```python
    transforms_resize_2 = transforms.Resize(512)
    transforms_compose = transforms.Compose([transforms_resize_2, transforms_tensor])
    image_resize_2 = transforms_compose(image)
    ```
    
- RandomCrop 的使用
    
    ```python
    transforms_random = transforms.RandomCrop(256)
    transforms_compose_2 = transforms.Compose([transforms_random, transforms_tensor])
    for i in range(10):
        image_crop = transforms_compose_2(image)
        writer.add_image('RandomCrop', image_crop, i)
    ```
    
- 总结
    - 关注输入和输出类型
    - 多看官方文档
    - 关注方法需要什么参数
- 不知道返回值的时候
    - `print`
    - `print(type())`
    - debug
- 案例演示：[**p12_transforms_use_2.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p12_transforms_use_2.py)

# P13. torchvision中的数据集使用

- 官方数据集
    
    [Datasets — Torchvision 0.22 documentation](https://docs.pytorch.org/vision/stable/datasets.html)
    
- 使用
    
    ```python
    # 训练数据集
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
    # 测试数据集
    test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
    ```
    
- 数据集的常用参数
    - **`root` (必需参数)**：指定数据集存储的根目录路径。
    - **`train` (默认值: `True`)**：指定加载训练集还是测试集。
    - **`transform` (默认值: `None`)**：定义对图像数据的预处理操作（如缩放、归一化、数据增强）。
    - **`target_transform` (默认值: `None`)**：定义对标签（target）的预处理操作（如标签映射、编码转换）。
    - **`download` (默认值: `False`)**：是否自动下载数据集到 `root` 目录。
- 案例演示：[**p13_datasets.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p13_datasets.py)

# P14. DataLoader的使用

- 使用
    
    ```python
    test_data = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                             download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    ```
    
- DataLoader 的常用参数
    - **`dataset` (必需参数)**：指定要加载的数据集对象，必须是继承自 `torch.utils.data.Dataset` 的实例。
    - **`batch_size` (默认值: `1`)**：每个批次（batch）加载的样本数量。
    - **`shuffle` (默认值: `False`)**：是否在每个 epoch 开始时打乱数据顺序。
    - **`num_workers` (默认值: `0`)**：用于数据加载的子进程数量。
    - **`drop_last` (默认值: `False`)**：是否丢弃最后一个不完整的批次（当样本总数不能被 `batch_size` 整除时）。
    - **`pin_memory` (默认值: `False`)**：是否将数据加载到 CUDA 的固定内存（pinned memory）中。
    - **`timeout` (默认值: `0`)**：设置从子进程获取数据的超时时间（秒）。
    - **`sampler` 和 `batch_sampler`**：自定义数据采样策略（替代默认的随机打乱或顺序采样）。
    - **`collate_fn` (默认值: `None`)**：自定义如何将多个样本合并成一个批次（batch）。
- 案例演示：[**p14_dataloader.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P7-14/p14_dataloader.py)