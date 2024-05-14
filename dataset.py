import os
import cv2
import torch
from torch.utils.data import Dataset


class GrapeDataset(Dataset):
    def __init__(self, img_dir, ann_df, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(os.listdir(img_dir))
        self.img_labels = ann_df

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        return image, label
