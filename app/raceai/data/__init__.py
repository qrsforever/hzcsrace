#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import cv2
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
from raceai.utils.misc import race_data


class RaceDataset(ABC, Dataset):
    def __init__(self, cfg):
        if 'data_prefix' in cfg:
            self.data_prefix = cfg.data_prefix
        else:
            self.data_prefix = ''

    def data_path(self, p):
        if self.data_prefix:
            path = os.path.join(self.data_prefix, p)
        else:
            path = p
        return race_data(path)

    @abstractmethod
    def data_reader(self, **kwargs):
        """
        (M)
        """


class RawRaceDataset(RaceDataset):
    def __init__(self, source, cfg):
        super().__init__(cfg)
        self.images, self.sources = self.data_reader(source)

    def data_reader(self, sources):
        if isinstance(sources, str):
            return [self.data_path(sources)], [sources]
        elif isinstance(sources, (tuple, list)):
            images = []
            datass = []
            for item in sources:
                if isinstance(item, str):
                    images.append(self.data_path(item))
                    datass.append(item)
                else:
                    assert RuntimeError(type(item))
            return images, datass
        assert RuntimeError(type(sources))

    def __getitem__(self, index):
        return self.images[index], self.sources[index]

    def __len__(self):
        return len(self.images)


class ClsRaceDataset(RaceDataset):
    def __init__(self, source, cfg):
        super().__init__(cfg)
        input_size = cfg.input_size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.images, self.labels = self.data_reader(source)
        augtrans = []
        if "imgaugs" in cfg:
            for it in cfg.imgaugs:
                augtrans.append(race_load_class(it.class_name)(**it.params))
        self.augtrans = RandomOrder(augtrans)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        if 'mean' in cfg:
            mean = cfg.mean
        if 'std' in cfg:
            std = cfg.std
        self.imgtrans = Compose([
                Resize(input_size),
                ToTensor(),
                Normalize(mean=mean, std=std)])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.augtrans(img)
        return self.imgtrans(img), self.labels[index], self.images[index]

    def __len__(self):
        return len(self.images)


class PredictDirectoryImageDataset(ClsRaceDataset):
    def data_reader(self, path):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_list = []
        for filename in os.listdir(path):
            if not filename.lower().endswith(extensions):
                continue
            image_list.append(f'{path}/{filename}')
        return image_list, [-1 for _ in image_list]


class PredictListImageDataset(ClsRaceDataset):
    def data_reader(self, sources):
        if isinstance(sources, str):
            return [self.data_path(sources)], ['-1']
        images = []
        labels = []
        if isinstance(sources, (list, tuple)):
            for item in sources:
                if isinstance(item, str):
                    images.append(self.data_path(item))
                    labels.append('-1')
                elif isinstance(item, dict):
                    if 'image_path' not in item:
                        raise ValueError('not found image_path')
                    images.append(self.data_path(item['image_path']))
                    if 'image_id' in item:
                        labels.append(item['image_id'])
                    else:
                        labels.append('-1')
                else:
                    assert RuntimeError(type(item))
        else:
            assert RuntimeError(type(sources))
        return images, labels


class JsonFileDataset(ClsRaceDataset):
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


class PredictListImageRaw(RawRaceDataset):
    def data_reader(self, sources):
        if isinstance(sources, str):
            return [self.data_path(sources)], ['-1']
        images = []
        labels = []
        if isinstance(sources, (list, tuple)):
            for item in sources:
                if isinstance(item, str):
                    images.append(self.data_path(item))
                    labels.append('-1')
                elif isinstance(item, dict):
                    if 'image_path' not in item:
                        raise ValueError('not found image_path')
                    images.append(self.data_path(item['image_path']))
                    if 'image_id' in item:
                        labels.append(item['image_id'])
                    else:
                        labels.append('-1')
                else:
                    assert RuntimeError(type(item))
        else:
            assert RuntimeError(type(sources))
        return images, labels


# class PredictDirectoryImageRaw(RaceDataset):
#     def __init__(self, img_path, cfg):
#         self.images = self.data_reader(img_path)
#
#     def data_reader(self, path):
#         extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
#         image_list = []
#         for filename in os.listdir(path):
#             if not filename.lower().endswith(extensions):
#                 continue
#             image_list.append(f'{path}/{filename}')
#         return image_list
#
#     def __getitem__(self, index):
#         return cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED), self.images[index]
#
#     def __len__(self):
#         return len(self.images)
#
#
# class PredictSingleImageRaw(PredictDirectoryImageRaw):
#     def data_reader(self, path):
#         if isinstance(path, (list, tuple)):
#             return path
#         return [path]

class VideoFramesDataset(Dataset):
    def __init__(self, video_path, cfg):
        if isinstance(video_path, list):
            video_path = video_path[0]
        if 'data_prefix' in cfg:
            video_path = f'{cfg.data_prefix}/{video_path}'
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), f'open video error: {video_path}'
        self.retrieve_count = cfg.retrieve_count
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.retrieve_index = range(0, self.nframes - 1, int(self.nframes / self.retrieve_count))

    def __getitem__(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.retrieve_index[index])
        retval, img = self.cap.read()
        if not retval or index == (self.retrieve_count - 1):
            self.cap.release()
        return img, self.retrieve_index[index]

    def __len__(self):
        return self.retrieve_count
