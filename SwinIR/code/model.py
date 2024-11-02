import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwinIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallow_feature_extractor = ShallowFE()
        self.deep_feature_extractor = DeepFE()
        self.hq_image_reconstructor = HQImgRecon()
    
    def _padWithReflect(self, img):
        *_, H, W = img.shape
        padding_bottom = 8-H%8
        padding_right = 8-W%8
        return F.pad(img, (0, padding_right, 0, padding_bottom), mode='reflect')
    
    def _crop(self, img, shape):
        *_, H, W = shape
        return img[:,:,:(4*H),:(4*W)]
        
    def forward(self, img):
        shape = img.shape
        img = self._padWithReflect(img)
        lf = self.shallow_feature_extractor(img)
        hf = self.deep_feature_extractor(lf)
        z = hf+lf
        out = self.hq_image_reconstructor(z)
        return self._crop(out, shape)

class ShallowFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Conv2d(3, 180, 3, padding=1)
    
    def forward(self, img):
        return self.inner(img)

class DeepFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Sequential(
            *[RSTB() for _ in range(6)],
            nn.Conv2d(180, 180, 3, padding=1),
        )

    def forward(self, img):
        return self.inner(img)

class HQImgRecon(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Conv2d(180, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def forward(self, img):
        return self.inner(img)

class RSTB(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Sequential(*[STL(i) for i in range(6)])
        self.conv = nn.Conv2d(180, 180, 3, padding=1)
    
    def forward(self, img):
        raw = img
        cvrt_img = raw.permute(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
        cvrt_img = self.inner(cvrt_img)
        raw = cvrt_img.permute(0, 3, 1, 2)# (B,H,W,C) -> (B,C,H,W)
        out = self.conv(raw)
        return img + out

class STL(nn.Module):
    def __init__(self, order):
        super().__init__()
        cycShft = (order%2 != 0)
        self.inner1 = nn.Sequential(
            nn.LayerNorm(180),
            MSA(cycShft)
        )
        self.inner2 = nn.Sequential(
            nn.LayerNorm(180),
            MLP()
        )

    def forward(self, cvrt_img):
        x = cvrt_img
        x = self.inner1(x)
        z = x + cvrt_img

        r = z
        z = self.inner2(z)
        out = z + r

        return out

class MSA(nn.Module):
    def __init__(self, cycShft):
        super().__init__()
        self.cyc_shft_wndw_partition = CycShftWndwPartition(8, cycShft)
        self.self_attention = SelfAttention()
        self.un_cyc_shft_wndw_partition = UnCycShftWndwPartition(8, cycShft)
    
    def forward(self, cvrt_img):
        windows, mask, shape = self.cyc_shft_wndw_partition(cvrt_img)
        windows = self.self_attention(windows, mask)
        new_cvrt_img = self.un_cyc_shft_wndw_partition(windows, shape)
        return new_cvrt_img #(B, H, W, C)


class CycShftWndwPartition(nn.Module):
    def __init__(self, window_size, cycShft):
        super().__init__()
        self.wsize = window_size
        self.cycShft = cycShft
        self.h_slices = [slice(0,-window_size), slice(-window_size,-window_size//2), slice(-window_size//2,None)]
        self.w_slices = [slice(0,-window_size), slice(-window_size,-window_size//2), slice(-window_size//2,None)]
    
    def _mask(self, H, W):
        att_partition = torch.zeros((1,H,W))
        attention_idx = 0
        for h_slice in self.h_slices:
            for w_slice in self.w_slices:
                att_partition[:,h_slice,w_slice] = attention_idx
                attention_idx += 1
        att_partition = att_partition.view(1, H//self.wsize, self.wsize, W//self.wsize, self.wsize)
        att_partition = att_partition.transpose(2, 3).reshape(-1, self.wsize*self.wsize)
        mask = att_partition.unsqueeze(1) - att_partition.unsqueeze(2) #(i,j): 0 if "i" is in same window with "j"
        mask = mask.masked_fill(mask==0, 0.0)
        mask = mask.masked_fill(mask!=0, -100.0)
        return mask # (H/w*W/w, N, N)

    def forward(self, cvrt_img):
        B, H, W, C = cvrt_img.shape
        if self.cycShft:
            x = torch.roll(cvrt_img, shifts=(-8//2,-8//2), dims=(1,2))
            mask = self._mask(H,W).to(x.device)
        else:
            x = cvrt_img
            mask = torch.zeros((H*W//(self.wsize*self.wsize),self.wsize*self.wsize,self.wsize*self.wsize)).to(x.device)
        x = x.view(B, H//self.wsize, self.wsize, W//self.wsize, self.wsize, C)
        windows = x.transpose(2, 3).reshape(-1, self.wsize*self.wsize, C) #(B=B*H/w*W/w, N=w*w, C)

        return windows, mask, (B, H, W, C)

class UnCycShftWndwPartition(nn.Module):
    def __init__(self, window_size, cycShft):
        super().__init__()
        self.wsize = window_size
        self.cycShft = cycShft
    
    def forward(self, windows, shape):
        B, H, W, C = shape
        x = windows.view(B, H//self.wsize, W//self.wsize, self.wsize, self.wsize, C)
        x = x.transpose(2, 3).reshape(B, H, W, C)
        if self.cycShft:
            cvrt_img = torch.roll(x, shifts=(8//2,8//2), dims=(1,2))
        else:
            cvrt_img = x
        return cvrt_img

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(180, 3*180, bias=False)
        self.biasMatrix = nn.Parameter(torch.zeros((2*8-1)**2, 6))
        self.relativeIndex = self._getRelativeIndex()
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(180, 180)

    def _getRelativeIndex(self):
        h_cord = torch.arange(8)
        w_cord = torch.arange(8)
        h_grid, w_grid = torch.meshgrid([h_cord, w_cord], indexing="ij") # (8,8), (8,8)
        x = torch.stack((h_grid, w_grid)) # (2,8,8)
        x = torch.flatten(x, 1) # (2,64)
        x = x.unsqueeze(dim=2) - x.unsqueeze(dim=1) #(2,64,64), (i,j): distance from i to j
        x[0,:,:] += (8-1)
        x[0,:,:] *= (2*8 - 1)
        x[1,:,:] += (8-1)
        relative_index_matrix = x[0,:,:] + x[1,:,:] # (64,64)
        return relative_index_matrix.reshape(-1)

    def forward(self, windows, mask):
        B, N, C = windows.shape
        WNum, *_ = mask.shape #(windownum, N, N)

        qkv = self.qkv(windows).view(B, N, 3, 6, C//6).permute(2,0,3,1,4) #(3,B,headnum,N,dimension)
        q,k,v = qkv[0], qkv[1], qkv[2]

        x = torch.matmul(q, k.transpose(-2,-1)) / ((C//6)**0.5) #(B,headnum,N,N)
        relative_pos_bias = self.biasMatrix[self.relativeIndex].view((8*8),(8*8),6).permute(2,0,1) #(headnum,64,64)
        x = x+relative_pos_bias.unsqueeze(dim=0) #(B,headnum,N=w*w=64,N)
        x = x.view(B//WNum, WNum, 6, N, N).transpose(1, 2) + mask.view(1, 1, WNum, N, N)
        x = x.transpose(1,2).reshape(-1, 6, N, N)
        attention = self.softmax(x)
        self_attention = torch.matmul(attention, v) #(B,headnum,N,dimension)
        z = self_attention.transpose(1,2).reshape(B, N, C)
        new_windows = self.proj(z)
        return new_windows #(B, w*w, C)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Linear(180,2*180),
            nn.GELU(),
            nn.Linear(2*180,180)
        )
    
    def forward(self, cvrt_img):
        return self.inner(cvrt_img)
