
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SRResNet, DisNet
from test import test
train = __import__('train-voc').train
from data import SRImageDataset, SRHrImageOnlyDataset
from utils import ContentLoss

### define global variables
scale_factor = 4
crop_out = 8
crop_size = 24
epochs = 182+91 ### around 300,000 iterations 

device = 'cuda:0'

div2k_img_path = '/mnt/home/20160788/data/DIV2K_train_LR_bicubic/X4'
div2k_lbl_path = '/mnt/home/20160788/data/DIV2K_train_HR'

voc_path = '/mnt/home/20160788/data/VOC/VOC2012/JPEGImages'

valid_img_path = '/mnt/home/20160788/data/DIV2K_valid_LR_bicubic/X4'
valid_lbl_path = '/mnt/home/20160788/data/DIV2K_valid_HR'

set5_img_path = '/mnt/home/20160788/data/Set5_LR'
set5_lbl_path = '/mnt/home/20160788/data/Set5_HR'
set14_img_path = '/mnt/home/20160788/data/Set14_LR'
set14_lbl_path = '/mnt/home/20160788/data/Set14_HR'
urban100_img_path = '/mnt/home/20160788/data/Urban100_LR'
urban100_lbl_path = '/mnt/home/20160788/data/Urban100_HR'

resnet_path = '/mnt/home/20160788/srresnet/l1_clamp_best.pt'

resume = (len(sys.argv) > 1 and (sys.argv[1] == '-r' or sys.argv[1] == '-R' or sys.argv[1] == '-resume')) or (len(sys.argv) > 2 and (sys.argv[2] == '-r' or sys.argv[2] == '-R' or sys.argv[2] == '-resume'))
check_param = (len(sys.argv) > 1 and (sys.argv[1] == '-p' or sys.argv[1] == '-P' or sys.argv[1] == '-param'))

old_pnt_path = '../old-voc.pt'
last_pnt_path = '../last-voc.pt'
check_pnt_path = '../best-voc.pt'
log_path = '../logdir-voc'

if not os.path.exists(log_path):
    os.makedirs(log_path)

### define data loaders
div2k_dataset = SRImageDataset(div2k_img_path, div2k_lbl_path, scale_factor=scale_factor, crop_size=crop_size, do_rand=True)
div2k_dataloader = DataLoader(div2k_dataset, batch_size=16, num_workers=16)

voc_dataset = SRHrImageOnlyDataset(voc_path, scale_factor=scale_factor, crop_size=crop_size, do_rand=True)
voc_dataloader = DataLoader(voc_dataset, batch_size=16, num_workers=16)

valid_dataset = SRImageDataset(valid_img_path, valid_lbl_path, scale_factor=scale_factor, crop_size=crop_size, do_rand=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, num_workers=16)

set5_dataset = SRImageDataset(set5_img_path, set5_lbl_path, scale_factor=scale_factor)
set14_dataset = SRImageDataset(set14_img_path, set14_lbl_path, scale_factor=scale_factor, ignore_list=[2,9])
urban100_dataset = SRImageDataset(urban100_img_path, urban100_lbl_path, scale_factor=scale_factor)
set5_dataloader = DataLoader(set5_dataset, batch_size=1)
set14_dataloader = DataLoader(set14_dataset, batch_size=1)
urban100_dataloader = DataLoader(urban100_dataset, batch_size=1)

### define network
generator = SRResNet(scale_factor).to(device)
disciminator = DisNet(crop_size, scale_factor).to(device)

if check_param:
    print(f'G: { sum(p.numel() for p in generator.parameters() if p.requires_grad) }')
    print(f'D: { sum(p.numel() for p in disciminator.parameters() if p.requires_grad) }')
    quit()

### define train variables
bicubic = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False).to(device)
writer = SummaryWriter(log_path)

adversarial_loss = nn.BCEWithLogitsLoss()
content_loss = ContentLoss().to(device)
pixel_loss = nn.MSELoss()

d_optimizer = optim.Adam(disciminator.parameters(), lr=1e-4)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
d_scheduler = MultiStepLR(d_optimizer, milestones=[epochs*0.5], gamma=0.1)
g_scheduler = MultiStepLR(g_optimizer, milestones=[epochs*0.5], gamma=0.1)

### make args
args = {
    'generator': generator,
    'discriminator': disciminator,
    'bicubic': bicubic,
    'd_optimizer': d_optimizer,
    'g_optimizer': g_optimizer,
    'd_scheduler': d_scheduler,
    'g_scheduler': g_scheduler,
    'adversarial_loss': adversarial_loss,
    'content_loss': content_loss,
    'pixel_loss': pixel_loss,
    'device': device,
    'crop_out': crop_out,
    'epochs': epochs,
    'train_dataloaders': [div2k_dataloader, voc_dataloader],
    'valid_dataloader': valid_dataloader,
    'check_pnt_path': check_pnt_path,
    'last_pnt_path': last_pnt_path,
    'old_pnt_path': old_pnt_path,
    'resnet_path': resnet_path,
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
