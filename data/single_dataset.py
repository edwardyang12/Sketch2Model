import os 
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import torch
from PIL import Image, ImageFile
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, imagepath, size = 256):
        self.image = pd.read_csv(imagepath)
        self.size = size

    def data_aug(self):
        transform_list = []
        transform_list += [Transforms.ToTensor(),]
        transform_list += [
            Transforms.ColorJitter(brightness=.5, contrast=.3, saturation=.3),
            Transforms.RandomHorizontalFlip(),
            Transforms.Normalize(
                mean=np.array([0.5, 0.5, 0.5]),
                std=np.array([0.5, 0.5, 0.5]),
            ),
        ]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def load_image(self, filename, index):
        name = filename.values[index][0]
        temp = Image.open(name).convert('RGB')
        temp = temp.resize((self.size,self.size), resample=Image.Resampling.BICUBIC)
        return temp

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image = self.load_image(self.image, index)
        transform = self.data_aug()
        image = transform(image)
        return image

if __name__ == "__main__":
    image_path = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/airplane.csv"

    dataset = CustomDataset(image_path)
    batch_size = 20
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, pin_memory=True, drop_last=True)
    print(len(dataloader)*batch_size)
    for i, data in tqdm(enumerate(dataloader)):
        print(data[0])