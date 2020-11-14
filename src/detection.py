import numpy as np
import torch
import cv2
from models import *
import argparse
import json
from torch.utils.data.sampler import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_file', type=str)
parser.add_argument('--feature_file', type=str)
parser.add_argument('--image_path_list', type=str)
args = parser.parse_args()
ckpt_file = args.ckpt_file
feature_file = args.feature_file
image_path_list = args.image_path_list.split(",")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_resolution = (128, 128)
mean = 0.5
std = 0.5

mean_embedding = np.loadtxt(feature_file)

model = MobileNetFeatures()
checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

for i, image_path in enumerate(image_path_list):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_resolution)
    input_img = image.astype(np.float32) / 255.0
    input_img = (input_img - np.array(mean).astype(np.float32)
                    ) / np.array(std).astype(np.float32)
    input_img = input_img.transpose(2, 0, 1)
    input_img = torch.Tensor(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_img).cpu().numpy().reshape(-1)

    cos_sim = np.dot(mean_embedding, embedding) / (np.linalg.norm(mean_embedding, ord=2) * np.linalg.norm(embedding, ord=2))

    print(cos_sim)
