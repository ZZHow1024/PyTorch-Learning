import torch
import torchvision

# 保存方式 1（模型结构 + 模型参数）
model = torch.load('vgg16_method1.pth', weights_only=False)
print(model)

# 保存方式 2（模型参数）
model = torchvision.models.vgg16(pretrained=False)
model.load_state_dict(torch.load('vgg16_method2.pth'))
print(model)
