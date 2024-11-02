
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import DualRegressionNet
from test import test
from train import train
from data import TrainDataset, TestDataset, HrOnlyTrainDataset

### define global variables
crop_out = 8
epochs = 273 ### around 300,000 iterations 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

div2k_img_path = '/mnt/home/20160788/data/DIV2K_train_LR_bicubic/X4'
div2k_lbl_2x_path = '/mnt/home/20160788/srcnn/DIV2K_train_LR_bicubic/X2'
div2k_lbl_4x_path = '/mnt/home/20160788/data/DIV2K_train_HR'

voc_img_path = '/mnt/home/20160788/data/VOC/VOC2012/JPEGImages'

valid_img_path = '/mnt/home/20160788/data/DIV2K_valid_LR_bicubic/X4'
valid_lbl_2x_path = '/mnt/home/20160788/srcnn/DIV2K_valid_LR_bicubic/X2'
valid_lbl_4x_path = '/mnt/home/20160788/data/DIV2K_valid_HR'

set5_img_path = '/mnt/home/20160788/data/Set5_LR'
set5_lbl_path = '/mnt/home/20160788/data/Set5_HR'
set14_img_path = '/mnt/home/20160788/data/Set14_LR'
set14_lbl_path = '/mnt/home/20160788/data/Set14_HR'
urban100_img_path = '/mnt/home/20160788/data/Urban100_LR'
urban100_lbl_path = '/mnt/home/20160788/data/Urban100_HR'

last_pnt_path = '../last-voc.pt'
old_pnt_path = '../old-voc.pt'
check_pnt_path = '../best-voc.pt'
log_path = '../logdir-voc'

if not os.path.exists(log_path):
    os.makedirs(log_path)

resume = (len(sys.argv) > 1 and (sys.argv[1] == '-r' or sys.argv[1] == '-R' or sys.argv[1] == '-resume'))
check_param = (len(sys.argv) > 1 and (sys.argv[1] == '-p' or sys.argv[1] == '-P' or sys.argv[1] == '-param'))

### define data loaders
div2k_dataset = TrainDataset(div2k_img_path, div2k_lbl_2x_path, div2k_lbl_4x_path, do_rand=True)
div2k_dataloader = DataLoader(div2k_dataset, batch_size=16, num_workers=16)

voc_dataset = HrOnlyTrainDataset(voc_img_path, do_rand=True)
voc_dataloader = DataLoader(voc_dataset, batch_size=16, num_workers=16)

valid_dataset = TrainDataset(valid_img_path, valid_lbl_2x_path, valid_lbl_4x_path, do_rand=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, num_workers=16)

set5_dataset = TestDataset(set5_img_path, set5_lbl_path)
set14_dataset = TestDataset(set14_img_path, set14_lbl_path, ignore_list=[2,9])
urban100_dataset = TestDataset(urban100_img_path, urban100_lbl_path)
set5_dataloader = DataLoader(set5_dataset, batch_size=1)
set14_dataloader = DataLoader(set14_dataset, batch_size=1)
urban100_dataloader = DataLoader(urban100_dataset, batch_size=1)

### define network
net = DualRegressionNet().to(device)

if check_param:
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    quit()

### define train variables
bicubic = nn.Upsample(scale_factor=4, mode='bicubic').to(device)
writer = SummaryWriter(log_path)

criterion = nn.L1Loss()

optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.99))
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

### make args
args = {
    'net': net,
    'bicubic': bicubic,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'criterion': criterion,
    'device': device,
    'crop_out': crop_out,
    'epochs': epochs,
    'train_dataloaders': [div2k_dataloader, voc_dataloader],
    'valid_dataloader': valid_dataloader,
    'check_pnt_path': check_pnt_path,
    'last_pnt_path': last_pnt_path,
    'old_pnt_path': old_pnt_path,
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
