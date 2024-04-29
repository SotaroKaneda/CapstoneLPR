import cv2
import numpy as np
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    def __init__(self, images_filepaths, transforms=None):
        self.images_filepaths = images_filepaths
        self.transforms = transforms
        # self.labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.labels = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"     # no O

    def __len__(self):
        return len(self.images_filepaths)
    
    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        label = self.labels.index(image_filepath.split("\\")[-1].split(".")[0][0])
        image_name = image_filepath.split("\\")[-1]

        return image, label, image_name