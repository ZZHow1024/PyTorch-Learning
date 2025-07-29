from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
image = Image.open('../P15-20/dataset/train/ants/0013035.jpg')

# ToTensor 的使用
transforms_tensor = transforms.ToTensor()
image_tensor = transforms_tensor(image)
writer.add_image('ToTensor', image_tensor)

# Normalize 的使用
print(image_tensor[0][0][0])
transforms_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
image_norm = transforms_normalize(image_tensor)
print(image_norm[0][0][0])
writer.add_image('Normalize', image_norm)

writer.close()
