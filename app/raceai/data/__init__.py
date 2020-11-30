#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2 # noqa
import json

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

from raceai.utils.misc import race_load_class


class RaceDataset(ABC, Dataset):

    @abstractmethod
    def data_reader(self, **kwargs):
        """
        (M)
        """


class PredictDirectoryImageRaw(RaceDataset):
    def __init__(self, img_path, cfg):
        self.images = self.data_reader(img_path)

    def data_reader(self, path):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_list = []
        for filename in os.listdir(path):
            if not filename.lower().endswith(extensions):
                continue
            image_list.append(f'{path}/{filename}')
        return image_list

    def __getitem__(self, index):
        return cv2.imread(self.images[index])

    def __len__(self):
        return len(self.images)


class PredictSingleImageRaw(PredictDirectoryImageRaw):
    def data_reader(self, path):
        if isinstance(path, (list, tuple)):
            return path
        return [path]


class PredictDirectoryImageDataset(RaceDataset):
    def __init__(self, img_path, cfg):
        if isinstance(cfg.input_size, int):
            input_size = (cfg.input_size, cfg.input_size)
        self.images = self.data_reader(img_path)
        self.itrans = Compose([
                Resize(input_size),
                ToTensor(),
                Normalize(mean=cfg.mean, std=cfg.std)])

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


class PredictSingleImageDataset(PredictDirectoryImageDataset):
    def data_reader(self, path):
        return [path]


class JsonFileDataset(RaceDataset):
    def __init__(self, path, cfg):
        input_size = cfg.input_size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.images, self.labels = self.data_reader(path)
        augtrans = []
        if "imgaugs" in cfg:
            for it in cfg.imgaugs:
                augtrans.append(race_load_class(it.class_name)(**it.params))
        self.augtrans = RandomOrder(augtrans)
        self.imgtrans = Compose([
                Resize(input_size),
                ToTensor(),
                Normalize(mean=cfg.mean, std=cfg.std)])

    def data_reader(self, path):
        image_list = []
        label_list = []
        root = os.path.dirname(path)
        with open(path, 'r') as f:
            items = json.load(f)
            for item in items:
                image_list.append(os.path.join(root, item['image_path']))
                label_list.append(item['label'])
        return image_list, label_list

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.augtrans:
            img = self.augtrans(img)
        return self.imgtrans(img), self.labels[index], self.images[index]

    def __len__(self):
        return len(self.images)
