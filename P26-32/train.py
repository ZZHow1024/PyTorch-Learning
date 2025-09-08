import os.path

import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

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

# TensorBoard
writer = SummaryWriter('logs')

for i in range(epoch):
    print(f'---第 {i + 1} 轮训练开始---')

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        images, targets = data
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_train_step % 100 == 0:
            writer.add_scalar('train_loss', loss.item(), total_train_step)
            print(f'训练次数：{total_train_step}，Loss：{loss.item()}')

        total_train_step += 1

    # 测试步骤开始
    model.eval()
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
