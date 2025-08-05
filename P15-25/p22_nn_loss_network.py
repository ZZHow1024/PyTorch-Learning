import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential

datasets = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = torch.utils.data.DataLoader(datasets, batch_size=1)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
model = MyModel()
for data in dataloader:
    image, target = data
    output = model(image)
    print(output)
    print(target)
    result_loss = loss(output, target)
    result_loss.backward()
    print('ok')
