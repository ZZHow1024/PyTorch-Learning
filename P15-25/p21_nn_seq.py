import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter


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


model = MyModel()

print(model)
input = torch.ones((64, 3, 32, 32))
print(input.shape)
output = model(input)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(model, input)
writer.close()
