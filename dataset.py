import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import PIL.Image as Image
import torch
import torch.utils.data as thd
import math
import cv2
import albumentations as albu
import random
import re


def get_augmentation(img_size):  # robią augmentacje, sobie wybierasz jak chcesz przerabiać obrazy, do treningu i walidacji
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Resize(height=img_size, width=img_size),
        albu.Normalize(mean=[0], std=[1]),
    ]
    return albu.Compose(train_transform)



class PortraitsDataset(thd.Dataset):
    def __init__(
        self,
        data_type: str = "train",
        image_size = 256
    ):
        self.data_type = data_type
        self.images = []
        self.img_size = image_size 

        print("Loading {} data".format(data_type))

        for img in os.listdir(img_path):  #  img path to ścieżka względna/bezwzgledna do obrazków - tych dobrych obrazków
            # sprawdzasz czy obraz jest ok i ładujesz ścieżkę do niego do obiektu klasy - self.images
            # 1. czytasz cv2.imread ten obraz (ścieżka to będzie img_path + "/" + img w stringu) cv2 instalujesz pip install python-opencv albo moze opencv-python
            self.images.append(images)

        # tu jest podział na zbiory treningowy i walidacyjny do zbioru testowego możesz zrobić podfolder
        # na train uczysz model, na valid testujesz model co epoke i patrzysz czy sie nie przeucza i czy sie uczy. na test testujesz juz wyuczony model
        # epoka to jeden cykl treningowy, czyli dane przechodzą przez model jeden raz i nadpisywane sa wagi polaczen w sieci 
        if data_type == "train" or data_type == "valid":
            random.Random(42).shuffle(self.images)
            if data_type == "valid":
                self.images = self.images[: int(len(self.images) / 10)]  # 0-10%
            elif data_type == "train":
                self.images = self.images[int(len(self.images) / 10) :]  # 10-100%
        else:
            pass

        print("Images and labels loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # tutaj ładujesz pojedycza instancje obrazu i etykiety
        label = cv2.imread(str(self.images[idx]), 0)
        #  img size zmien uzywajac cv2 
        label = torch.from_numpy(label).unsqueeze(0)  # zamieniasz na tenor, moze sie nie zgadzac wymiar, wiec sprawdz sobie print(label.shape) - powinno byc np 3, 256, 256
        # ładujesz img w taki sam sposób ale używasz np biblioteki pillow albo cv2 do tego zeby dodac szum, rozmycie, jakies inne gowna
        # print(label.shape) 
        # print(ig.shape)
        return img.float(), label.float()

# gowno = PortraitsDataset("train")
# print(gowno)