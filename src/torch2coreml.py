import argparse
import torch
from torch.autograd import Variable
from torchvision import models
from PIL import Image
from models import *
import json
import cv2
import numpy as np
import coremltools as ct
from torchvision.models import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model_file', type=str)
args = parser.parse_args()
model_file = args.model_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetFeatures()
img = torch.rand(1, 3, 128, 128).to(device)

if model_file is not None:
    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
model.to(device)
model.eval()

traced_model = torch.jit.trace(model, img)
model = ct.convert(
    traced_model,
    # name "input_1" is used in 'quickstart'
    inputs=[ct.ImageType(name="input_1", shape=img.shape)],
    # # provide only if step 4 was performed
    # classifier_config=ct.ClassifierConfig(class_labels)
)
