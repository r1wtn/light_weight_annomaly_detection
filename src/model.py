import torch
from torch import nn, rand


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


class LWADModel(nn.Module):
    def __init__(self):
        super(LWADModel, self).__init__()

    def forward(self, x):
        return


if __name__ == "__main__":
    return
