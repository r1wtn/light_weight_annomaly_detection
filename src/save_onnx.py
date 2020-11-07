import argparse
import torch
from torch.autograd import Variable
from torchvision import models
from PIL import Image
from models import *
import json
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model_file', type=str)
args = parser.parse_args()
model_file = args.model_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LWADModel()
img = torch.rand(1, 3, 512, 512).to(device)

if model_file is not None:
    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
model.to(device)
model.eval()

input_names = ["input_1"]
output_names = ["output_1"]

if model_file is not None:
    onnx_file = model_file + ".onnx"
else:
    onnx_file = "debug.onnx"
print(onnx_file)
torch.onnx.export(
    model, img, onnx_file, verbose=True,
    input_names=input_names, output_names=output_names,
    dynamic_axes={'input_1': {0: 'batch_size'}, 'output_1': {0: 'batch_size'}}
)
