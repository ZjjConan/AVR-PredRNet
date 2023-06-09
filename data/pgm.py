import os
import glob
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class PGM(Dataset):

    def __init__(
        self, dataset_dir, data_split=None, image_size=80, 
        transform=None, subset=None
    ):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform

        self.file_names = [
            os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, "*_" + self.data_split + "_*.npz"))
        ]
        self.file_names.sort()

        # Sanity
        assert subset != 'train' or len(self.file_names) == 1200000, f'Train length = {len(self.file_names)}'
        assert subset != 'val' or len(self.file_names) == 20000, f'Validation length = {len(self.file_names)}'
        assert subset != 'test' or len(self.file_names) == 200000, f'Test length = {len(self.file_names)}'


    def __len__(self):
        return len(self.file_names)

    def _get_data(self, idx):
        data_file = self.file_names[idx]
 
        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        image = data["image"].reshape(16, 160, 160)
        if self.image_size != 160:
            resize_image = np.zeros((16, self.image_size, self.image_size))
            for idx in range(0, 16):
                resize_image[idx] = cv2.resize(
                    image[idx], (self.image_size, self.image_size),
                    interpolation = cv2.INTER_NEAREST
                )
        else:
            resize_image = image

        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_file = self._get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure_encoded = data["relation_structure_encoded"]
        del data

        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)       
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)

        return image, target, meta_target, structure_encoded, data_file
