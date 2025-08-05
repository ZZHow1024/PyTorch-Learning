import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class MyModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(x)
        return x


input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

model1 = MyModel1()
output = model1(input)
print('input = ', input)
print('output = ', output)

dataset = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)


class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(x)
        return x


model2 = MyModel2()

writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    images, targets = data
    writer.add_images('input', images, step)
    output = model2(images)
    writer.add_images('output', output, step)
    step += 1

writer.close()
