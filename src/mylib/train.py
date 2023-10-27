import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from models.constants import *
from utils.convert_window import convert_to_windows
from models.tranad import TranAD
from utils.choose_threshold import get_t

def train_epoch(epoch_step, dataloader, model, optimizer, scheduler, device):
    model.train()
    l = nn.MSELoss(reduction = 'none')
    l1s = []

    # epoch_step should be >= 1
    for d, _ in dataloader:
        optimizer.zero_grad()

        d = d.to(device)
        local_bs = d.shape[0]
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, DIMS)
        z = model(window, elem)
        l1 = (1 / epoch_step) * l(z[0], elem) + (1 - 1/epoch_step) * l(z[1], elem)
        z = z[1]
        l1s.append(torch.mean(l1).item())
        loss = torch.mean(l1)

        loss.backward(retain_graph=True)
        optimizer.step()

    scheduler.step()
    print(f'Train loss: {np.mean(l1s)}')

def val_epoch(epoch_step, dataloader, model, device, marker='Val L1', flag=True):
    model.eval()
    l = nn.MSELoss(reduction = 'none')

    with torch.no_grad():
        ls = []
        l1 = []
        z1 = []
        for d, _ in dataloader:
            d = d.to(device)
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, -1, DIMS)
            z = model(window, elem)
            z = z[1]
            cur_l = l(z, elem)
            ls.append(torch.mean(cur_l).item())
            l1.append(cur_l[0])
            z1.append(z[0])

        if flag:
            print(f'{marker} loss: {np.mean(ls)}')
        return torch.concat(l1, dim=0), torch.concat(z1, dim=0)


if __name__ == "__main__":
    mn = torch.full((DIMS,), 1e9)
    mx = torch.full((DIMS,), -1e9)
    r_path = PATH + 'train'
    paths = sorted(os.listdir(r_path))

    for name in paths[:8]:
        cur_tensor = torch.DoubleTensor(np.load(os.path.join(r_path, name)).T)
        cur_tensor[(cur_tensor > torch.quantile(cur_tensor, 1 - QUANTILE, axis=0)) | (cur_tensor < torch.quantile(cur_tensor, QUANTILE, axis=0))] = 0
        mn = torch.minimum(mn, torch.min(cur_tensor, dim=0)[0])
        mx = torch.maximum(mx, torch.max(cur_tensor, dim=0)[0])

    trainD = []
    for name in paths[:8]:
        cur_tensor = torch.DoubleTensor(np.load(os.path.join(r_path, name)).T)
        cur_tensor[(cur_tensor > torch.quantile(cur_tensor, 1 - QUANTILE, axis=0)) | (cur_tensor < torch.quantile(cur_tensor, QUANTILE, axis=0))] = 0
        cur_tensor = (cur_tensor - mn) / (mx - mn)
        trainD.append(convert_to_windows(cur_tensor))
    trainD = torch.concat(trainD, dim=0)
    print('trainD size:', trainD.shape)

    valD = []
    for name in paths[8:]:
        cur_tensor = torch.DoubleTensor(np.load(os.path.join(r_path, name)).T)
        cur_tensor[(cur_tensor > torch.quantile(cur_tensor, 1 - QUANTILE, axis=0)) | (cur_tensor < torch.quantile(cur_tensor, QUANTILE, axis=0))] = 0
        cur_tensor = (cur_tensor - mn) / (mx - mn)
        valD.append(convert_to_windows(cur_tensor))
    valD = torch.concat(valD, dim=0)
    print('valD size:', valD.shape)

    r_path = PATH + 'val/0'
    paths = sorted(os.listdir(r_path))

    test0 = []
    for name in paths:
        cur_tensor = torch.DoubleTensor(np.load(os.path.join(r_path, name)).T)
        cur_tensor[(cur_tensor > torch.quantile(cur_tensor, 1 - QUANTILE, axis=0)) | (cur_tensor < torch.quantile(cur_tensor, QUANTILE, axis=0))] = 0
        cur_tensor = (cur_tensor - mn) / (mx - mn)
        test0.append(convert_to_windows(cur_tensor))
    test0 = torch.concat(test0, dim=0)
    print('test0 size:', test0.shape)

    r_path = PATH + 'val/1'
    paths = sorted(os.listdir(r_path))

    test1 = []
    for name in paths:
        cur_tensor = torch.DoubleTensor(np.load(os.path.join(r_path, name)).T)
        cur_tensor[(cur_tensor > torch.quantile(cur_tensor, 1 - QUANTILE, axis=0)) | (cur_tensor < torch.quantile(cur_tensor, QUANTILE, axis=0))] = 0
        cur_tensor = (cur_tensor - mn) / (mx - mn)
        test1.append(convert_to_windows(cur_tensor))
    test1 = torch.concat(test1, dim=0)
    print('test1 size:', test1.shape)

    train_dataset = torch.utils.data.TensorDataset(trainD, trainD)
    train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(valD, valD)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    test0_dataset = torch.utils.data.TensorDataset(test0, test0)
    test0_dataloader = torch.utils.data.DataLoader(test0_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    test1_dataset = torch.utils.data.TensorDataset(test1, test1)
    test1_dataloader = torch.utils.data.DataLoader(test1_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = TranAD(DIMS).double().to(device)
    optimizer = torch.optim.AdamW(model.parameters() , lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    for epoch in range(1, 11):
        print(f'Epoch {epoch}')
        train_epoch(epoch, train_dataloader, model, optimizer, scheduler, device)
        val_epoch(epoch, train_dataloader, model, device, 'Train L1')
        val_epoch(epoch, val_dataloader, model, device)
        print()

    loss, _ = val_epoch(-1, test1_dataloader, model, device, flag=False)
    lossT, _ = val_epoch(-1, train_dataloader, model, device, flag=False)
    loss = np.array(loss.cpu())
    lossT = np.array(lossT.cpu())

    ths = np.zeros(DIMS)
    for i in range(loss.shape[1]):
        lt, l = lossT[:, i], loss[:, i]
        cur_t = get_t(lt, l)
        ths[i] = cur_t

    print('Thresholds:', ths)
