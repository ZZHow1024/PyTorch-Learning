import torch
import torchvision.datasets
from torch import nn

dataset = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        output = self.linear1(x)
        return output


model = MyModel()

for data in dataloader:
    images, labels = data
    print(images.shape)
    output = torch.flatten(images)
    print(output.shape)
    output = model(output)
    print(output.shape)
