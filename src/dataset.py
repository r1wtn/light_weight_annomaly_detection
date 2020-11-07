import torch.utils.data as data
from glob import glob
import os
from random import choice
import cv2
import numpy as np


class LWADDataset(data.Dataset):
    def __init__(self, data_path, coco_path, input_resolution=512, mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]):
        super(LWADDataset, self).__init__()
        self.data = glob(os.path.join(data_path, "*"))
        self.coco = glob(os.path.join(coco_path, "*"))
        self.input_resolution = input_resolution
        self.mean = mean
        self.std = std

    def convert_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(
            image, (self.input_resolution, self.input_resolution))  # 512x512

        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.mean).astype(np.float32)) / \
            np.array(self.std).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target_path = self.data[index]
        target_image = self.convert_image(target_path)

        positive_image_path = choice(self.data)
        positive_image = self.convert_image(positive_image_path)

        negative_image_path = choice(self.coco)
        negative_image = self.convert_image(negative_image_path)

        return target_image, positive_image, negative_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path')
    parser.add_argument('--coco_path')
    args = parser.parse_args()

    data_path = args.data_path
    coco_path = args.coco_path

    dataset = LWADDataset(data_path, coco_path)

    print(dataset[40])
