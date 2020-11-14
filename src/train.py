import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from dataset import LWADDataset
from models import *
import argparse
from tqdm import tqdm
import codecs
import os
from termcolor import colored
from pytorch_metric_learning import losses
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_name')
parser.add_argument('--train_data_path')
parser.add_argument('--train_coco_path')
parser.add_argument('--valid_data_path')
parser.add_argument('--valid_coco_path')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch_size', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_interval', type=int, default=10)
args = parser.parse_args()

experiment_name = args.experiment_name
train_data_path = args.train_data_path
train_coco_path = args.train_coco_path
valid_data_path = args.valid_data_path
valid_coco_path = args.valid_coco_path
batch_size = int(args.batch_size)
epoch_size = int(args.epoch_size)
lr = args.lr
save_interval = args.save_interval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU mode")
else:
    print("CPU mode")

if not os.path.exists("../model_files/"):
    os.makedirs("../model_files/")

if not os.path.exists("../saved_features/"):
    os.makedirs("../saved_features/")

if not os.path.exists("../logs/"):
    os.makedirs("../logs/")

model = MobileNetFeatures()
model.to(device=device)
model.train()

# define optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
print(batch_size)
# define data loader
train_dataset = LWADDataset(train_data_path, train_coco_path, input_resolution=128)
valid_dataset = LWADDataset(valid_data_path, valid_coco_path, input_resolution=128)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    sampler=None)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    sampler=None)

# define loss
arcface_loss = losses.ArcFaceLoss(2, 512, margin=28.6, scale=64)
loss_optimizer = torch.optim.SGD(arcface_loss.parameters(), lr=0.01)


for epoch_id in range(epoch_size):
    for iter_id, batch in tqdm(enumerate(train_loader)):
        images = batch[0].to(device)
        labels = batch[1].to(device)

        embeddings = model(images)

        loss = arcface_loss(embeddings, labels)
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_optimizer.step()

    if save_interval > 0 and epoch_id % save_interval == 0:
        model.eval()
        positive_dist = []
        negative_dist = []
        for batch in valid_loader:
            images = batch[0].to(device)
            labels = batch[1].numpy().tolist()
            labels = [bool(i) for i in labels]
            with torch.no_grad():
                embeddings = model(images).cpu().numpy()
            
            positive_embeddings = embeddings[labels]
            negative_embeddings = embeddings[[not i for i in labels]]

            mean_embedding = np.mean(positive_embeddings, axis=0)
            for pe in positive_embeddings:
                cos_sim = np.dot(mean_embedding, pe) / (np.linalg.norm(mean_embedding, ord=2) * np.linalg.norm(pe, ord=2))
                positive_dist.append(cos_sim)
            for ne in negative_embeddings:
                cos_sim = np.dot(mean_embedding, ne) / (np.linalg.norm(mean_embedding, ord=2) * np.linalg.norm(ne, ord=2))
                negative_dist.append(cos_sim)
        mean_positive_dist = sum(positive_dist) / len(positive_dist)
        mean_negative_dist = sum(negative_dist) / len(negative_dist)

        print(f"epoch{epoch_id}: {mean_positive_dist} {mean_negative_dist}")
        print(f"epoch{epoch_id}: {mean_positive_dist} {mean_negative_dist}", file=codecs.open(f'../logs/{experiment_name}.txt', 'a', 'utf-8')
        )
        model.train()

        model_save_path = f"../model_files/{experiment_name}_{epoch_id:04d}.pth"
        features_save_path = f"../saved_features/{experiment_name}_{epoch_id:04d}.txt"
        save_model(model_save_path, epoch_id, model, optimizer)
        np.savetxt(features_save_path, mean_embedding, delimiter=",")
    
    if epoch_id == epoch_size - 1:
        model_save_path = f"../model_files/{experiment_name}_last.pth"
        features_save_path = f"../saved_features/{experiment_name}_last.txt"
        save_model(model_save_path, epoch_id, model, optimizer)
        np.savetxt(features_save_path, mean_embedding, delimiter=",")

