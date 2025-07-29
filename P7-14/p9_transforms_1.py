from PIL import Image
from torchvision import transforms

image_path = '../P15-20/dataset/train/ants/0013035.jpg'
image = Image.open(image_path)

tensor_transforms = transforms.ToTensor()
tensor_image = tensor_transforms(image)

print(type(tensor_image))
