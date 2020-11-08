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


class FReLU(nn.Module):
    def __init__(self, in_c, k=3, s=1, p=1):
        super().__init__()
        self.f_cond = nn.Conv2d(in_c, in_c, kernel_size=k,
                                stride=s, padding=p, groups=in_c)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        tx = self.bn(self.f_cond(x))
        out = torch.max(x, tx)
        return out


class Conv(nn.Module):
    def __init__(self, in_C, out_C, kernel_size=3, stride=1, bn=True):
        super(Conv, self).__init__()
        self.in_C = in_C
        self.conv = nn.Conv2d(in_C, out_C, kernel_size,
                              stride, padding=(kernel_size-1)//2, bias=True)
        self.bn = None
        self.activation = FReLU(out_C)
        if bn:
            self.bn = nn.BatchNorm2d(out_C)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x



class LWADModel(nn.Module):
    def __init__(self):
        super(LWADModel, self).__init__()
        self.conv0 = Conv(3, 16)
        self.pool0 = nn.AvgPool2d(4, 4)
        self.conv1 = Conv(16, 32)
        self.pool1 = nn.AvgPool2d(4, 4)
        self.conv2 = Conv(32, 64)
        self.pool2 = nn.AvgPool2d(4, 4)
        self.conv3 = Conv(64, 128)
        self.pool3 = nn.AvgPool2d(4, 4)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.sigmoid(x)
        return x


class MobileNetFeatures(nn.Module):
    def __init__(self):
        super(MobileNetFeatures, self).__init__()
        self.head = MobileNetV2().features
        self.pool = nn.AvgPool2d(4, 4)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.head(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.sigmoid(x)
        return x




if __name__ == "__main__":
    model = MobileNetFeatures()
    # model = LWADModel()
    img = rand((1, 3, 512, 512), requires_grad=True)
    out = model(img)
    print(out.shape)
