import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x + 1
        return output


model = MyModel()
x = torch.tensor([1.0])
output = model(x)
print(output)
