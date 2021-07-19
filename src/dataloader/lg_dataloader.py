from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations.augmentations import transforms as pixel
from albumentations.augmentations.geometric import transforms as geometric
import cv2
import albumentations as A
import os
import numpy as np
from einops import rearrange
from tqdm import tqdm


def get_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def lg_preprocessing(path):
    test_meta_data = pd.read_csv(f'{path}/test.csv')
    test_path = f'{path}/test_input_img'

    test_xs = []
    for id, file_test, _ in tqdm(test_meta_data.iloc):
        test_x = get_image(f'{test_path}/{file_test}')
        test_xs.append(test_x)
    np.save(f'{path}/test_save.npy', test_xs)


class TrainingDataset(Dataset):
    def __init__(self, data_path, size, data_num, valid_items):
        input_files = pd.read_csv(f'{data_path}/train.csv')['input_img'].to_list()
        target_files = pd.read_csv(f'{data_path}/train.csv')['label_img'].to_list()

        input_image = [f'{data_path}/train_input_img/{file_name}' for file_name in input_files]
        target_image = [f'{data_path}/train_label_img/{file_name}' for file_name in target_files]

        valid_input_image, valid_target_image = valid_items

        self.input_image = sorted(list(set(input_image) - set(valid_input_image)))
        self.target_image = sorted(list(set(target_image) - set(valid_target_image)))

        assert len(input_image) == len(self.input_image) + len(valid_input_image)
        assert len(target_image) == len(self.target_image) + len(valid_target_image)

        self.total_length = len(self.input_image)
        self.cat_transform = A.Compose([
            A.augmentations.crops.transforms.RandomCrop(*size),
        ])
        self.num_iter = int(self.total_length * data_num)

    def __len__(self):
        return self.num_iter

    def __getitem__(self, batch_idx):
        batch_idx = batch_idx % len(self.input_image)
        input_image = cv2.imread(self.input_image[batch_idx])
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        label_image = cv2.imread(self.target_image[batch_idx])
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

        common = np.concatenate([input_image, label_image], -1)
        common = self.cat_transform(image=common)['image']

        input_image, label_image = common[..., :3], common[..., 3:]

        input_image = (input_image / 255.0).astype(np.float32)
        label_image = (label_image / 255.0).astype(np.float32)
        return np.transpose(input_image, (2, 0, 1)), np.transpose(label_image, (2, 0, 1))


class ValidDataset(Dataset):
    def __init__(self, data_path, seed, ratio: float):
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'{data_path} is not a directory')

        input_files = pd.read_csv(f'{data_path}/train.csv')['input_img'].to_list()
        target_files = pd.read_csv(f'{data_path}/train.csv')['label_img'].to_list()

        splits = train_test_split(input_files,
                                  target_files,
                                  test_size=ratio,
                                  random_state=seed,
                                  shuffle=True)
        train_input_files, valid_input_files, train_target_files, valid_target_files = splits

        self.input_image = [f'{data_path}/train_input_img/{file_name}' for file_name in valid_input_files]
        self.target_image = [f'{data_path}/train_label_img/{file_name}' for file_name in valid_target_files]
        self.total_length = len(self.input_image)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        input_image = get_image(self.input_image[idx])
        label_image = get_image(self.target_image[idx])
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        return input_image.transpose((2, 0, 1)), label_image.transpose((2, 0, 1))

    def get_items(self):
        return self.input_image, self.target_image


class Datasets(object):

    def __init__(self, batch_size, num_workers,
                 data_path, size, seed, data_num, ratio: float):
        valid_set = ValidDataset(data_path, seed, ratio)
        train_set = TrainingDataset(data_path, size, data_num, valid_set.get_items())

        self.train_loader = DataLoader(train_set, batch_size, True, num_workers=num_workers)
        self.valid_loader = DataLoader(valid_set, 1, False, num_workers=0)

    def get_dataloaders(self):
        return self.train_loader, self.valid_loader
