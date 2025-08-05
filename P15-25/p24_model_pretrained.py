import torch
import torchvision.datasets
from torch import nn

# train_data = torchvision.datasets.ImageNet('dataset', train=True, transform=torchvision.transforms.ToTensor(),
#                                            download=True)
# dataloader = torch.utils.data.DataLoader(train_data, batch_size=1)

vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)

print(vgg16_true)
print(vgg16_false)

# 添加 module
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 修改 module
vgg16_false.classifier[6] = nn.Linear(1000, 10)
print(vgg16_false)
