from os import listdir
from os.path import join
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms

class HrOnlyTrainDataset(Dataset):
    def __init__(self, img_path, do_rand):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.img_path = img_path
        self.scale_factor = 4
        self.crop_size = 32
        self.do_rand = do_rand
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img = Image.open(join(self.img_path, self.img_names[idx]))

        c4x = self.scale_factor*self.crop_size
        if self.do_rand:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            params = transforms.RandomCrop(c4x).get_params(img, (c4x, c4x))
            img = transforms.functional.crop(img, *params)
        else:
            img = transforms.CenterCrop(c)(img)

        img_tensor = transforms.ToTensor()(img.resize((self.crop_size, self.crop_size), Image.BICUBIC))
        lbl_2x_tensor = transforms.ToTensor()(img.resize((c4x//2, c4x//2), Image.BICUBIC))
        lbl_4x_tensor = transforms.ToTensor()(img)
        
        return img_tensor, lbl_2x_tensor, lbl_4x_tensor


class TrainDataset(Dataset):
    def __init__(self, img_path, lbl_2x_path, lbl_4x_path, do_rand):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_2x_names = sorted([name for name in listdir(lbl_2x_path)])
        self.lbl_4x_names = sorted([name for name in listdir(lbl_4x_path)])
        self.img_path = img_path
        self.lbl_2x_path = lbl_2x_path
        self.lbl_4x_path = lbl_4x_path
        self.scale_factor = 4
        self.crop_size = 32
        self.do_rand = do_rand
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(join(self.img_path, self.img_names[idx]))
        lbl_2x = Image.open(join(self.lbl_2x_path, self.lbl_2x_names[idx]))
        lbl_4x = Image.open(join(self.lbl_4x_path, self.lbl_4x_names[idx]))

        if self.do_rand:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                lbl_2x = lbl_2x.transpose(Image.FLIP_LEFT_RIGHT)
                lbl_4x = lbl_4x.transpose(Image.FLIP_LEFT_RIGHT)
                    
            params = transforms.RandomCrop(self.crop_size).get_params(img, (self.crop_size, self.crop_size))
            img = transforms.functional.crop(img, *params)
            lbl_2x = transforms.functional.crop(lbl_2x, *[(self.scale_factor*p)//2 for p in params])
            lbl_4x = transforms.functional.crop(lbl_4x, *[self.scale_factor*p for p in params])
        else:
            img = transforms.CenterCrop(self.crop_size)(img)
            lbl_2x = transforms.CenterCrop(self.crop_size*(self.scale_factor//2))(lbl_2x)
            lbl_4x = transforms.CenterCrop(self.crop_size*self.scale_factor)(lbl_4x)
            
        img_tensor = transforms.ToTensor()(img)
        lbl_2x_tensor = transforms.ToTensor()(lbl_2x)
        lbl_4x_tensor = transforms.ToTensor()(lbl_4x)

        return img_tensor, lbl_2x_tensor, lbl_4x_tensor

class TestDataset(Dataset):
    def __init__(self, img_path, lbl_path, ignore_list=None):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_names = sorted([name for name in listdir(lbl_path)])
        self.img_path = img_path
        self.lbl_path = lbl_path
        
        if ignore_list:
            for i, idx in enumerate(sorted(ignore_list)):
                idx = idx-i
                del self.img_names[idx]
                del self.lbl_names[idx]
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(join(self.img_path, self.img_names[idx]))
        lbl = Image.open(join(self.lbl_path, self.lbl_names[idx]))

        img_tensor = transforms.ToTensor()(img)
        lbl_tensor = transforms.ToTensor()(lbl)

        return img_tensor, lbl_tensor
