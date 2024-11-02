
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SRResNet
from test import test
from train import train
from data import SRImageDataset

### define global variables
scale_factor = 4
crop_out = 8
crop_size = 64
epochs = 2000 ### 200,000 iterations 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_img_path = '/mnt/home/20160788/data/DIV2K_train_LR_bicubic/X4'
train_lbl_path = '/mnt/home/20160788/data/DIV2K_train_HR'

valid_img_path = '/mnt/home/20160788/data/DIV2K_valid_LR_bicubic/X4'
valid_lbl_path = '/mnt/home/20160788/data/DIV2K_valid_HR'

set5_img_path = '/mnt/home/20160788/data/Set5_LR'
set5_lbl_path = '/mnt/home/20160788/data/Set5_HR'
set14_img_path = '/mnt/home/20160788/data/Set14_LR'
set14_lbl_path = '/mnt/home/20160788/data/Set14_HR'
urban100_img_path = '/mnt/home/20160788/data/Urban100_LR'
urban100_lbl_path = '/mnt/home/20160788/data/Urban100_HR'

last_pnt_path = '../l1_clamp_last.pt'
check_pnt_path = '../l1_clamp_best.pt'
log_path = '../logdir_l1_clamp'

if not os.path.exists(log_path):
    os.makedirs(log_path)

resume = (len(sys.argv) > 1 and (sys.argv[1] == '-r' or sys.argv[1] == '-R' or sys.argv[1] == '-resume'))

### define data loaders
train_dataset = SRImageDataset(train_img_path, train_lbl_path, scale_factor=scale_factor, crop_size=crop_size)
train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=16)

valid_dataset = SRImageDataset(valid_img_path, valid_lbl_path, scale_factor=scale_factor, crop_size=crop_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=16)

set5_dataset = SRImageDataset(set5_img_path, set5_lbl_path, scale_factor=scale_factor)
set14_dataset = SRImageDataset(set14_img_path, set14_lbl_path, scale_factor=scale_factor, ignore_list=[2,9])
urban100_dataset = SRImageDataset(urban100_img_path, urban100_lbl_path, scale_factor=scale_factor)
set5_dataloader = DataLoader(set5_dataset, batch_size=1, num_workers=32)
set14_dataloader = DataLoader(set14_dataset, batch_size=1, num_workers=32)
urban100_dataloader = DataLoader(urban100_dataset, batch_size=1, num_workers=32)

### define network
net = SRResNet(scale_factor).to(device)

### define train variables
writer = SummaryWriter(log_path)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)
scheduler = MultiStepLR(optimizer, milestones=[epochs*0.7, epochs*0.9], gamma=0.5)

### make args
args = {
    'net': net,
    'scale_factor': scale_factor,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'criterion': criterion,
    'device': device,
    'crop_out': crop_out,
    'epochs': epochs,
    'train_dataloader': train_dataloader,
    'valid_dataloader': valid_dataloader,
    'check_pnt_path': check_pnt_path,
    'last_pnt_path': last_pnt_path,
    'writer': writer
}

### do training
train(args, resume)

### do test
args['test_dataloader'] = set5_dataloader
test(args, 'Set5')

args['test_dataloader'] = set14_dataloader
test(args, 'Set14')

args['test_dataloader'] = urban100_dataloader
test(args, 'Urban100')
