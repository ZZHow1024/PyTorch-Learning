from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path = '../P15-20/dataset/train/ants/0013035.jpg'
image = Image.open(image_path)

tensor_transforms = transforms.ToTensor()
tensor_image = tensor_transforms(image)

writer = SummaryWriter('logs')

writer.add_image('tensor_image', tensor_image)

writer.close()
