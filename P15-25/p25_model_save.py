import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)

# 保存方式 1（模型结构 + 模型参数）
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式 2（模型参数）
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')
