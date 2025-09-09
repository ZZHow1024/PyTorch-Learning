# PyTorch深度学习入门笔记P26-32

@ZZHow(ZZHow1024)

参考课程：

【**PyTorch深度学习快速入门教程【小土堆】**】

[https://www.bilibili.com/video/BV1hE411t7RN]

# P26. 完整的模型训练套路（一）

- 训练部分
    - model.py
        
        ```python
        import torch
        from torch import nn
        from torch.nn import Sequential
        
        # 搭建神经网络
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.model = Sequential(
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(in_features=64 * 4 * 4, out_features=64),
                    nn.Linear(in_features=64, out_features=10),
                )
        
            def forward(self, x):
                x = self.model(x)
                return x
        
        # 测试神经网络模型结构的正确性
        if __name__ == '__main__':
            model = MyModel()
            input = torch.ones([64, 3, 32, 32])
            output = model(input)
            print(output.shape)
        ```
        
    - train.py
        
        ```python
        import torch
        import torchvision.datasets
        from model import MyModel
        
        # 准备数据集
        train_data = torchvision.datasets.CIFAR10('dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                                  download=True)
        test_data = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                                 download=True)
        
        # 获取数据集的长度
        train_data_size = len(train_data)
        test_data_size = len(test_data)
        print(f'训练数据集的长度为：{train_data_size}')
        print(f'测试数据集的长度为：{test_data_size}')
        
        # 使用 Dataloader 加载数据集
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)
        
        # 创建网络模型
        model = MyModel()
        
        # 损失函数
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # 优化器
        learning_rate = 1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # 设置训练网络的参数
        total_train_step = 0  # 训练次数
        total_test_step = 0  # 测试次数
        epoch = 10  # 训练轮次
        
        for i in range(epoch):
            print(f'---第 {i + 1} 轮训练开始---')
        
            # 训练步骤开始
            for data in train_dataloader:
                images, targets = data
                outputs = model(images)
                loss = loss_fn(outputs, targets)
        
                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                total_train_step += 1
        
                print(f'训练次数：{total_train_step}，Loss：{loss.item()}')
        
        ```
        

# P27. 完整的模型训练套路（二）

- 测试（验证）部分
    
    ```python
    # 测试步骤开始
    total_test_loss = 0  # 总测试 Loss
    total_accuracy = 0  # 总正确率
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    print(f'测试集上的总 Loss：{total_test_loss}')
    print(f'测试集上的总 正确率：{total_accuracy / test_data_size}')
    
    torch.save(model.state_dict(), os.path.join('model', f'model_{i}.pth'))
    print(f'模型已保存，文件名：model_{i}.pth')
    
    total_test_step += 1
    ```
    

# P28. 完整的模型训练套路（三）

- 训练步骤开始时
    
    ```python
    model.train()
    ```
    
- 测试步骤开始时
    
    ```python
    model.eval()
    ```
    
- 案例演示：[**model.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P26-32/model.py) 和 [**train.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P26-32/train.py)

# P29. 利用GPU训练（一）

- 方式一
    - 在网络模型、数据（输入，标注）和损失函数后加上 `.cuda()`
    
    ```python
    # 创建网络模型
    model = MyModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
    
    # 数据（输入，标注）
    images, targets = data
    if torch.cuda.is_available():
        images = images.cuda()
        targets = targets.cuda()
    ```
    
- 案例演示：[**train_gpu_1.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P26-32/train_gpu_1.py)

# P30. 利用GPU训练（二）

- 方式二
    - 在网络模型、数据（输入，标注）和损失函数后通过 `.to(device)` 转移到对应设备
    
    ```python
    # 训练设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    print(f'训练设备：{device}')
    
    # 创建网络模型
    model = MyModel()
    model.to(device)
    
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    
    # 数据（输入，标注）
    images, targets = data
    images = images.to(device)
    targets = targets.to(device)
    ```
    
- 案例演示：[**train_gpu_2.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P26-32/train_gpu_2.py)

# P31. 完整的模型验证套路

- test.py
    
    ```python
    import os
    
    import torch
    import torchvision
    from PIL import Image
    from model import MyModel
    
    # 测试图片名称
    image_name = 'dog.png'
    # 测试模型名称
    model_name = 'model_29.pth'
    
    # 测试设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    print(f'测试设备：{device}')
    
    # 测试图片路径
    image_path = os.path.join('images', image_name)
    image = Image.open(image_path)
    image = image.convert('RGB')
    print(image)
    
    # 图片预处理
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    image = transform(image)
    image = torch.reshape(image, (1, 3, 32, 32))
    print(image.shape)
    
    # 加载模型
    model = MyModel()
    model.load_state_dict(torch.load(os.path.join('model', model_name), map_location=torch.device(device)))
    
    # 开始测试
    model.eval()
    with torch.no_grad():
        output = model(image)
    print(output)
    print(output.argmax(1))
    
    ```
    
- **注意：若训练模型的设备与当前加载加载模型的设备不一致时，需要在 `torch.load()` 时指定 `map_location=torch.device(device)`。**
- 案例演示：[**test.py**](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P26-32/test.py)