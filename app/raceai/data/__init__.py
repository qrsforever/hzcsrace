#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ( # noqa
        Resize,
        Compose,
        ToTensor,
        Normalize,
        RandomOrder,
        ColorJitter,
        RandomRotation,
        RandomGrayscale,
        RandomResizedCrop,
        RandomVerticalFlip,
        RandomHorizontalFlip)


class RaceDataset(ABC, Dataset):

    @abstractmethod
    def data_reader(self, **kwargs):
        """
        (M)
        """


class PredictImageDataset(RaceDataset):
    def __init__(self, img_path, input_size, mean, std):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.images = self.data_reader(img_path)
        self.itrans = Compose([
                Resize(input_size),
                ToTensor(),
                Normalize(mean=mean, std=std)])

    def data_reader(self, path):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_list = []
        for filename in os.listdir(path):
            if not filename.lower().endswith(extensions):
                continue
            image_list.append(f'{path}/{filename}')
        return image_list

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        return self.itrans(img)

    def __len__(self):
        return len(self.images)
