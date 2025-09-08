import os

import torch
import torchvision
from PIL import Image
from model import MyModel

# 测试图片名称
image_name = 'dog.png'
# 测试模型名称
model_name = 'model_29.pth'

# 测试设备
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
print(f'测试设备：{device}')

# 测试图片路径
image_path = os.path.join('images', image_name)
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

# 图片预处理
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)

# 加载模型
model = MyModel()
model.load_state_dict(torch.load(os.path.join('model', model_name), map_location=torch.device(device)))

# 开始测试
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
predicted_index = output.argmax(1).item()

cifar10_classes = {
    0: {'en': 'airplane', 'zh': '飞机'},
    1: {'en': 'automobile', 'zh': '汽车'},
    2: {'en': 'bird', 'zh': '鸟'},
    3: {'en': 'cat', 'zh': '猫'},
    4: {'en': 'deer', 'zh': '鹿'},
    5: {'en': 'dog', 'zh': '狗'},
    6: {'en': 'frog', 'zh': '青蛙'},
    7: {'en': 'horse', 'zh': '马'},
    8: {'en': 'ship', 'zh': '船'},
    9: {'en': 'truck', 'zh': '卡车'}
}

print(f"预测类别索引: {predicted_index}")
category_info = cifar10_classes.get(predicted_index, {'en': 'unknown', 'zh': '未知'})
print(f"英文类别: {category_info['en']}")
print(f"中文类别: {category_info['zh']}")
