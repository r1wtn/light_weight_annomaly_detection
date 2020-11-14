import torch
from torch import nn, rand
import torchvision.models as models
from torchvision.models import *

def save_model(path, epoch_id, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch_id,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def load_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model


class MobileNetFeatures(nn.Module):
    def __init__(self):
        super(MobileNetFeatures, self).__init__()
        self.head = MobileNetV2().features
        self.pool = nn.AvgPool2d(4, 4)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(1280, 512)

    def forward(self, x):
        x = self.head(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MobileNetFeatures()
    img = rand((1, 3, 128, 128), requires_grad=True)
    out = model(img)
    print(out)
