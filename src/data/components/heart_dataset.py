import os
import random
from collections import Counter, defaultdict
from typing import Optional, Dict

import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt


class HeartDataset(Dataset):
    def __init__(self,
                 label_path: str,
                 img_dir: str,
                 img_alias: str,  # tr(training), ts(test), val(validation)
                 transform=None,
                 ):
        self.label_path = label_path
        self.img_dir = img_dir
        self.img_dir_for_set = os.path.join(self.img_dir, img_alias)
        self.transform = transform
        self.img_alias = img_alias
        self.dataset = self._read_dataset()

    def _read_dataset(self) -> Dict:
        dataset = []
        files = os.listdir(self.img_dir_for_set)
        for file in files:
            _, id, _, instance = file.split('.')[0].split('_')
            img_path = os.path.join(self.img_dir_for_set, file)
            gold_path = os.path.join(self.label_path, f'gold_{id}_Image_{instance}.png')
            dataset.append((img_path, gold_path, f"{id}00{instance}"))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, gold_path, id = self.dataset[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(gold_path, cv2.IMREAD_UNCHANGED, )

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {'image': image, 'mask': mask, 'img_id': int(id)}


if __name__ == '__main__':
    label_path = "/home/gsoykan/Desktop/ku/spring24_comp548_medical_imaging/hw3/unet_segmentation/data/heart/golds"
    img_dir = "/home/gsoykan/Desktop/ku/spring24_comp548_medical_imaging/hw3/unet_segmentation/data/heart/images"

    transformations = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])  # training için rotate, scale, pad from sides to not distort the image...

    dataset = HeartDataset(label_path,
                           img_dir,
                           transform=transformations,
                           img_alias='tr', )
    print(dataset)

    for i in tqdm(range(len(dataset))):
        instance = dataset[i]
        print(instance)

        image, mask = instance['image'], instance['mask']

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display the image tensor
        ax[0].imshow(image.squeeze(), cmap='gray')
        ax[0].set_title('Image')

        # Display the mask tensor
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')

        plt.show()
