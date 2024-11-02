from os import listdir
from os.path import join
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size=None, ignore_list=None, do_rand=False):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_names = sorted([name for name in listdir(lbl_path)])
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.scale_factor = 4
        self.crop_size = crop_size
        self.do_rand = do_rand
        
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

        if self.crop_size:
            if self.do_rand:
                # random horizontal flip
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
                
                # random rotation
                angle = 90 * random.randint(0, 3)
                img = img.rotate(angle, expand=True)
                lbl = lbl.rotate(angle, expand=True)

                params = transforms.RandomCrop(self.crop_size).get_params(img, (self.crop_size, self.crop_size))
                img = transforms.functional.crop(img, *params)
                lbl = transforms.functional.crop(lbl, *[self.scale_factor*p for p in params])

            else:
                img = transforms.CenterCrop(self.crop_size)(img)
                lbl = transforms.CenterCrop(self.scale_factor*self.crop_size)(lbl)

        img_tensor = transforms.ToTensor()(img)
        lbl_tensor = transforms.ToTensor()(lbl)

        return img_tensor, lbl_tensor

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
