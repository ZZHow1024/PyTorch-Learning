import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir: str = root_dir
        self.label_dir: str = label_dir
        self.path: str = os.path.join(self.root_dir, self.label_dir)
        self.image_path_list = os.listdir(str(self.path))

    def __getitem__(self, index):
        image_name: str = self.image_path_list[index]
        image_item_path: str = os.path.join(self.path, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir

        return image, label

    def __len__(self):
        return len(self.image_path_list)


root_path = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'

# 蚂蚁数据集
ants_dataset = MyData(root_dir=root_path, label_dir=ants_label_dir)

# 蜜蜂数据集
bees_dataset = MyData(root_dir=root_path, label_dir=bees_label_dir)

# 获取蚂蚁数据集中的第 1 个图片
image, label = ants_dataset[0]
image.show()
print('label = ' + label)

# 获取蜜蜂数据集中的第 1 个图片
image, label = bees_dataset[0]
image.show()
print('label = ' + label)

# 数据集合并
train_dataset = ants_dataset + bees_dataset
print('蚂蚁数据集长度：' + str(len(ants_dataset)))
print('蜜蜂数据集长度：' + str(len(bees_dataset)))
print('训练数据集长度：' + str(len(train_dataset)))
