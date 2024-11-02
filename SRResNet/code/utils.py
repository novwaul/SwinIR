import numpy as np
import torch.nn as nn
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

def bicubic(img_tensor, scale_factor):
    return nn.functional.interpolate(img_tensor, scale_factor=scale_factor, mode='bicubic').clamp(min=0.0, max=1.0)