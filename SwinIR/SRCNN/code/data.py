from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SRImageDataset(Dataset):
    def __init__(self, img_path, lbl_path, scale_factor, input_crop_size, ignore_list=None):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_names = sorted([name for name in listdir(lbl_path)])
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.scale_factor = scale_factor
        self.input_crop_size = input_crop_size
        self.ignore_list = ignore_list
    
    def __len__(self):
        offset = 0
        if self.ignore_list:
            offset = len(self.ignore_list)
        return len(self.img_names) - offset

    def __getitem__(self, idx):
        if self.ignore_list:
            while idx in self.ignore_list:
                idx += 1
        img = Image.open(join(self.img_path, self.img_names[idx]))
        lbl = Image.open(join(self.lbl_path, self.lbl_names[idx]))

        img_tensor = transforms.ToTensor()(img)
        lbl_tensor = transforms.ToTensor()(lbl)
        
        if self.input_crop_size > 0:
            params = transforms.RandomCrop(self.input_crop_size).get_params(img_tensor, (self.input_crop_size, self.input_crop_size))
            img_tensor_crop = transforms.functional.crop(img_tensor, *params)
            lbl_tensor_crop = transforms.functional.crop(lbl_tensor, *[self.scale_factor*p for p in params])
        else:
            img_tensor_crop = img_tensor
            lbl_tensor_crop = lbl_tensor
        
        return img_tensor_crop, lbl_tensor_crop