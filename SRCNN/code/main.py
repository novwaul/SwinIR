
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SRCNN
from test import TestNet
from train import TrainNet
from data import SRImageDataset


### define global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scale_factor = 2
input_size = 64
train_img_path = 'DIV2K_train_LR_bicubic/X2'
train_lbl_path = 'DIV2K_train_HR'
valid_img_path = 'DIV2K_valid_LR_bicubic/X2'
valid_lbl_path = 'DIV2K_valid_HR'
check_pnt_path = 'rgb_best'
log_path = 'logdir_rgb'

### define data loaders
train_dataset = SRImageDataset(train_img_path, train_lbl_path, scale_factor=scale_factor, input_crop_size=input_size)
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=32)
valid_dataset = SRImageDataset(valid_img_path, valid_lbl_path, scale_factor=scale_factor, input_crop_size=input_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, num_workers=32)
### define network

channels = [3, 64, 32, 3]
kernel_sizes = [9, 5, 5]
net = SRCNN(channels, kernel_sizes).to(device)

### define train variables
preprocess = bicubic = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
writer = SummaryWriter(log_path)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = None

### do training
t = TrainNet(train_dataloader, valid_dataloader, device, net, bicubic, preprocess, criterion, optimizer, scheduler, writer)
t.train(100, check_pnt_path)
### do test
set5_dataset = SRImageDataset('Set5_LO', 'Set5_HI', scale_factor=scale_factor, input_crop_size=0)
set14_dataset = SRImageDataset('Set14_LO', 'Set14_HI', scale_factor=scale_factor, input_crop_size=0, ignore_list=[2,9])
urban100_dataset = SRImageDataset('Urban100_LO', 'Urban100_HI', scale_factor=scale_factor, input_crop_size=0)
set5_dataloader = DataLoader(set5_dataset, batch_size=1, num_workers=32)
set14_dataloader = DataLoader(set14_dataset, batch_size=1, num_workers=32)
urban100_dataloader = DataLoader(urban100_dataset, batch_size=1, num_workers=32)

p = TestNet(check_pnt_path, device, net, bicubic, preprocess, criterion, writer)
p.test(set5_dataloader, 'Set5')
p.test(set14_dataloader, 'Set14')
p.test(urban100_dataloader, 'Urban100')
