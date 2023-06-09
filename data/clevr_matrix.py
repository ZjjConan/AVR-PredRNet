import os
import random
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset


sub_folders = {'0': "problem1", 
               '1': "problem2", 
               '2': "problem3"}

class CLEVR_MATRIX(Dataset):
    def __init__(
        self, dataset_dir, data_split=None, image_size=80, 
        transform=None, subset=None
    ):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform
        self.permute = (data_split == "train")

        if subset == 'None':
            subsets = os.listdir(self.dataset_dir)
        else:
            subsets = [sub_folders[subset]]

        self.file_names = []
        for i in subsets:
            file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, i, "*_" + self.data_split + "_*.npz"))]
            self.file_names += [os.path.join(i, f) for f in file_names]


    def __len__(self):
        return len(self.file_names)

    def get_data(self, idx):
        data_file = self.file_names[idx]

        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)
        # print(data["image"].size())
        # image = np.transpose(data["image"], axes=(0,3,1,2)).reshape(16, 3, 240, 320)
        image = data["image"]
        if self.image_size != 240 or self.image_size != 320:
            resize_image = np.zeros((16, self.image_size, self.image_size, 3))
            for idx in range(0, 16):
                resize_image[idx] = cv2.resize(
                    image[idx], (self.image_size, self.image_size), 
                    interpolation = cv2.INTER_NEAREST
                )
        else:
            resize_image = image
        resize_image = np.transpose(resize_image, axes=(0,3,1,2))
        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_file = self.get_data(idx)

        # Get additional data
        target = data["target"]
        # print(target)
        meta_target = torch.tensor(0)
        structure = torch.tensor(0)
        structure_encoded = torch.tensor(0)
        del data

        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)         
            image = self.transform(image)


        if self.permute:
            new_target = random.choice(range(8))
            if new_target != target:
                image[[8 + new_target, 8 + target]] = image[[8 + target, 8 + new_target]]
                target = new_target

        target = torch.tensor(target, dtype=torch.long)

        return image, target, meta_target, structure_encoded, data_file

    