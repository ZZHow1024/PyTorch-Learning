from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
image = Image.open('dataset/train/ants/0013035.jpg')
transforms_tensor = transforms.ToTensor()

# Resize 的使用
print(image.size)
transforms_resize = transforms.Resize((512, 512))
image_resize = transforms_resize(image)
print(image_resize.size)
image_resize = transforms_tensor(image_resize)
writer.add_image('Resize', image_resize)

# Compose 的使用
transforms_resize_2 = transforms.Resize(512)
transforms_compose = transforms.Compose([transforms_resize_2, transforms_tensor])
image_resize_2 = transforms_compose(image)
writer.add_image('Compose', image_resize_2)

# RandomCrop 的使用
transforms_random = transforms.RandomCrop(256)
transforms_compose_2 = transforms.Compose([transforms_random, transforms_tensor])
for i in range(10):
    image_crop = transforms_compose_2(image)
    writer.add_image('RandomCrop', image_crop, i)

writer.close()
