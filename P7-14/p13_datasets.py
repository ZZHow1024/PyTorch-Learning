import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

# 训练数据集
train_set = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# 测试数据集
test_set = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

# 读取测试数据集中的前 10 个 image 和 target
writer = SummaryWriter('logs')

for i in range(10):
    image, target = test_set[i]
    writer.add_image('test_set', image, i)

writer.close()
