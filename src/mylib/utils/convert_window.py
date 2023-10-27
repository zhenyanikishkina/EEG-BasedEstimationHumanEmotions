import torch
from models.constants import *

def convert_to_windows(data):
    windows = []
    w_size = WINDOW
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)
