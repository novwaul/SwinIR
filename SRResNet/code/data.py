from os import listdir
from os.path import join
from torch.utils.data import Dataset
import random
import cv2
import torchvision.transforms as transforms

def getRandomCropArgs(img_shape, crop_size):
    _, ih, iw = img_shape
    if ih >= crop_size and iw >= crop_size:
        ix = random.randrange(0, iw-crop_size+1)
        iy = random.randrange(0, ih-crop_size+1)
    else:
        raise Exception('image size is smaller than crop size')
    return ix, iy, crop_size, crop_size

def crop(img_tensor, ix, iy, tx, ty):
    return img_tensor[:, iy:iy+ty, ix:ix+tx]

class SRImageDataset(Dataset):
    def __init__(self, img_path, lbl_path, scale_factor, crop_size=None, ignore_list=None):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_names = sorted([name for name in listdir(lbl_path)])
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        
        if ignore_list:
            for i, idx in enumerate(sorted(ignore_list)):
                idx = idx-i
                del self.img_names[idx]
                del self.lbl_names[idx]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = cv2.imread(join(self.img_path, self.img_names[idx]))
        lbl = cv2.imread(join(self.lbl_path, self.lbl_names[idx]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)

        img_tensor = transforms.ToTensor()(img)
        lbl_tensor = transforms.ToTensor()(lbl)
        
        if self.crop_size:
            args = getRandomCropArgs(img_tensor.shape, self.crop_size)
            img_tensor = crop(img_tensor, *args)
            lbl_tensor = crop(lbl_tensor, *[self.scale_factor*p for p in args])
        return img_tensor, lbl_tensor