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
    def __init__(self, sketchpath, imagepath, size = 256):
        self.sketch = pd.read_csv(sketchpath)
        self.image = pd.read_csv(imagepath)
        self.size = size
        self.classes = set()
        for index, row in self.image.iterrows():
            self.classes.add(row['class'])
        self.classes = list(self.classes)

    def data_aug(self, sketch=False):
        transform_list = []
        transform_list += [Transforms.ToTensor(),]
        if not sketch:
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

    def load_image(self, filename, index, sketch=False):
        class_name = filename.values[index][1]
        name = filename.values[index][0]
        temp = Image.open(name).convert('RGB')
        temp = temp.resize((self.size,self.size), resample=Image.Resampling.BICUBIC)
        return temp, class_name, self.classes.index(class_name)

    def __len__(self):
        return len(self.sketch)

    def __getitem__(self, index):
        transform = self.data_aug(sketch=True)
        sketch, class_name, class_id = self.load_image(self.sketch, index)
        sketch = transform(sketch)

        temp = random.randrange(0,len(self.image))
        image, rName, rid = self.load_image(self.image, temp)
        transform = self.data_aug()
        image = transform(image)
        return sketch, class_id, image, rid, class_name, rName

if __name__ == "__main__":
    sketch_path = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/overlap_sketch.csv"
    image_path = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/combined_csv.csv"

    dataset = CustomDataset(image_path, sketch_path)
    batch_size = 20
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, pin_memory=True, drop_last=True)
    print(len(dataloader)*batch_size)
    for i, data in tqdm(enumerate(dataloader)):
        print(data[0])
        print(i*batch_size)