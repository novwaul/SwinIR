import numpy as np
import torch
import torch.nn as nn
import torchvision.models as model
from torchvision.transforms import Normalize
from skimage.metrics import structural_similarity as ssim

def calc_psnr(img_np, lbl_np, crop_out):
    diff = img_np - lbl_np
    mse = np.mean(diff[:,crop_out:-crop_out,crop_out:-crop_out]**2)
    return -10*np.log10(mse + 1e-10)

def calc_ssim(img_np, lbl_np, crop_out):
    img_crop = img_np[:, crop_out:-crop_out, crop_out:-crop_out]
    lbl_crop = lbl_np[:, crop_out:-crop_out, crop_out:-crop_out]
    return ssim(img_crop, lbl_crop, channel_axis=0)

def cvrt_rgb_to_y(img_np):
   return (16.0 + 65.481*img_np[:,0,:,:] + 128.553*img_np[:,1,:,:] + 24.966*img_np[:,2,:,:]) / 255.0

def norm(img_tensor):
    return (img_tensor - 0.5) * 2.0

def denorm(img_tensor):
    return img_tensor / 2.0 + 0.5

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_features = model.vgg19(pretrained=True).features[:36]
        vgg_features.eval()
        for param in vgg_features.parameters():
            param.requires_grad = False
        self.inner = vgg_features
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _vgg_norm(self, img):
        return (img - self.mean) / self.std


    def forward(self, img, lbl):
        img = denorm(img).clamp(min=0.0, max=1.0)
        lbl = denorm(lbl).clamp(min=0.0, max=1.0)

        diff = self.inner(self._vgg_norm(img)) - self.inner(self._vgg_norm(lbl))
        return (diff**2).mean()

    def to(self, device):
        self.inner = self.inner.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self