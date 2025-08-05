import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        output = self.conv1(input)
        return output


model = MyModel()
print(model)

writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    images, labels = data
    output = model(images)
    # torch.Size([64, 3, 32, 32])
    writer.add_images('input', images, step)

    # torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, step)

    step += 1

writer.close()
