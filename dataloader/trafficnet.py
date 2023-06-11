import torch
import torch.utils.data as data

from PIL import Image
import os

class trafficnet_dataset(data.Dataset):
    def __init__(self, datapath, train = True, train_trainsform = None, target_transform = None):
        self.datapath           = datapath
        self.transforms         = train_trainsform
        self.target_transform   = target_transform
        self.train              = train

        image_path = os.path.join(self.datapath + "/train") if train else os.path.join(self.datapath + "/test")

        self.images     = []
        self.targets    = []

        for i in os.listdir(os.path.join(image_path, "dense_traffic")):
            self.images.append(os.path.join(image_path, "dense_traffic", i))
            self.targets.append(0)
        
        for i in os.listdir(os.path.join(image_path, "sparse_traffic")):
            self.images.append(os.path.join(image_path, "sparse_traffic", i))
            self.targets.append(1)

    
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        target = self.targets[index]
        if self.transforms is not None:
            image = self.transforms(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

