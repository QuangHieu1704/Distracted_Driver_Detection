import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader


class DDDDataset(Dataset):
    def __init__(self, data_path, transform = None):
        self.data_path = data_path
        self.classes = os.listdir(self.data_path)
        self.images_path = []
        self.transform = transform

        for cls in self.classes:
            for img_name in os.listdir(os.path.join(self.data_path, cls)):
                img_path = os.path.join(self.data_path, cls, img_name)
                img_path = img_path.replace("\\", "/")
                self.images_path.append(img_path)
        self.idx_to_cls = {i: j for i, j in enumerate(self.classes)}
        self.cls_to_idx = {value: key for key, value in self.idx_to_cls.items()}

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.images_path[idx].split("/")[-2]
        label = self.cls_to_idx[label]
        # label = np.eye(10, dtype='uint8')[label]

        return img, label


if __name__ == "__main__":
    print("=== Test Dataset and DataLoader ==")

    trainDDD = DDDDataset(data_path="Distracted_Driver_Detection/Dataset/train")
    print(trainDDD[0])