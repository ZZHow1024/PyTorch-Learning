import torch
import torchvision.datasets
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

model = MyModel()
output = model(input)
print(output)

dataset = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    images, targets = data
    writer.add_images('input', images, step)
    output = model(images)
    writer.add_images('output', output, step)
    step += 1

writer.close()
