import torch.utils.data as data
from glob import glob
import os
from random import choice, sample
import cv2
import numpy as np


class LWADDataset(data.Dataset):
    def __init__(self, target_path, coco_path, input_resolution=128, mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]):
        super(LWADDataset, self).__init__()
        self.target = glob(os.path.join(target_path, "*"))
        self.all_coco = glob(os.path.join(coco_path, "*"))
        print(len(self.all_coco))
        self.coco = sample(self.all_coco, len(self.target))
        self.data = self.target + self.coco
        target_labels = [1 for i in range(len(self.target))]
        coco_labels = [0 for i in range(len(self.coco))]
        self.labels = target_labels + coco_labels

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
        data_path = self.data[index]
        label = self.labels[index]
        image = self.convert_image(data_path)
        return image, label
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path')
    parser.add_argument('--coco_path')
    args = parser.parse_args()

    data_path = args.data_path
    coco_path = args.coco_path

    dataset = LWADDataset(data_path, coco_path)

    print(len(dataset))
    print(dataset[19])
