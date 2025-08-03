import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第 1 个 image 和 target
image, target = test_data[0]
print(image.shape)
print(target)

# 遍历 test_loader 中的 images 和 targets
writer = SummaryWriter('logs')

for epoch in range(2):
    step: int = 0
    for data in test_loader:
        images, targets = data
        writer.add_images(f'epoch{epoch}', images, step)
        step += 1

writer.close()
