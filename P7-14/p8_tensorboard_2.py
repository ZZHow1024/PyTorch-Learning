import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')

# Step1 图片
image_path = '../P15-20/dataset/train/ants/0013035.jpg'
image_pil = Image.open(image_path)
image_array = np.array(image_pil)

writer.add_image('image', image_array, 1, dataformats='HWC')

# Step2 图片
image_path = '../P15-20/dataset/train/bees/16838648_415acd9e3f.jpg'
image_pil = Image.open(image_path)
image_array = np.array(image_pil)

writer.add_image('image', image_array, 2, dataformats='HWC')

print(type(image_array))
print(image_array.shape)

writer.close()
