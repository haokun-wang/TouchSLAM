import os
import cv2
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

def sort_rank(elem):
    return int(elem[:-4])

class MyDataset(Dataset):
    def __init__(self, img_path, label_path):
        self._X = []
        self._Y = []

        imgs = os.listdir(img_path)
        deps = os.listdir(label_path)

        for img, dep in zip(imgs, deps):
            img_data = cv2.imread(img_path+img)
            dep_data = cv2.imread(label_path+dep)

            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            dep_data = dep_data[:,:,0]
            
            self._X.append(img_data[..., None])
            self._Y.append(dep_data[..., None])
        
        print(np.array(self._X).shape)
        print(np.array(self._Y).shape)
        self._X = torch.tensor(self._X)
        self._Y = torch.tensor(self._Y)

    def __getitem__(self, data_index):
        input_tensor = torch.tensor(self._X[data_index])
        output_tensor = torch.tensor(self._Y[data_index])
        return input_tensor, output_tensor

    def __len__(self):
        return len(self._X)
