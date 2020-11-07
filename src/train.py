import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from dataset import LWADDataset
from model import *
import argparse
from tqdm import tqdm
import os
from termcolor import colored

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_data_path')
parser.add_argument('--train_coco_path')
parser.add_argument('--valid_data_path')
parser.add_argument('--valid_coco_path')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch_size', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--val_interval', type=int, default=2)
parser.add_argument('--save_interval', type=int, default=2)
args = parser.parse_args()

train_data_path = args.train_data_path
train_coco_path = args.train_coco_path
valid_data_path = args.valid_data_path
valid_coco_path = args.valid_coco_path
batch_size = int(args.batch_size)
epoch_size = int(args.epoch_size)
lr = args.lr
val_interval = args.val_interval
save_interval = args.save_interval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU mode")
else:
    print("CPU mode")

if not os.path.exists("../model_files/"):
    os.makedirs("../model_files/")

if not os.path.exists("../logs/"):
    os.makedirs("../logs/")

model = LWADModel()
model.to(device=device)
model.train()

# define optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
print(batch_size)
# define data loader
train_dataset = LWADDataset(train_data_path, train_coco_path)
valid_dataset = LWADDataset(valid_data_path, valid_coco_path)
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
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    sampler=None)

# define loss
triplet_loss = nn.TripletMarginLoss(margin=3.0, p=2)

start_epoch = 0

for epoch_id in range(epoch_size):
    for iter_id, batch in tqdm(enumerate(train_loader)):
        target_img = batch[0].to(device=device)
        positive_img = batch[1].to(device=device)
        negative_img = batch[2].to(device=device)

        target_feature = model(target_img)
        positive_feature = model(positive_img)
        negative_feature = model(negative_img)

        loss = triplet_loss(target_feature, positive_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch_id % val_interval == 0:
        model.eval()
        dist_pos_sum = 0.0
        dist_neg_sum = 0.0
        count = 0
        for batch in valid_loader:
            target_img = batch[0].to(device=device)
            positive_img = batch[1].to(device=device)
            negative_img = batch[2].to(device=device)
            with torch.no_grad():
                target_feature = model(target_img)
                positive_feature = model(positive_img)
                negative_feature = model(negative_img)
            dist_pos = torch.sqrt(nn.MSELoss()(target_feature, positive_feature))
            dist_neg = torch.sqrt(nn.MSELoss()(target_feature, negative_feature))
            dist_pos_sum += dist_pos
            dist_neg_sum += dist_neg
            count += 1
        dist_pos_mean = dist_pos_sum / count
        dist_neg_mean = dist_neg_sum / count

        print(f"Positive: {dist_pos_mean}")
        print(f"Negative: {dist_neg_mean}")
        model.train()

    if save_interval > 0 and epoch_id % save_interval == 0:
        save_path = f"../model_files/lwad_{epoch_id:03d}.pth"
        save_model(save_path, epoch_id, model, optimizer)
