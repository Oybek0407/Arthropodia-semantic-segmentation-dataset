import torch, os, cv2, gdown, random, numpy as np, pandas as pd, pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import  ToTensorV2
from glob import glob
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root,  transformatios =None):
        super(). __init__()
        self.transformatios = transformatios
    
        
        # path of images and labels
        
        self.im_path =sorted(glob(f"{root}/images/*.jpg")) 
        self.labels_path = sorted(glob(f"{root}/labels/*.png")) 
        
    def __len__(self): return len(self.im_path) # get a lenth of images data


    def __getitem__(self, idx):
        im = cv2.cvtColor(cv2.imread(self.im_path[idx]), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(self.labels_path[idx]), cv2.COLOR_BGR2GRAY) 
       
        if self.transformatios is not None: 
            transformer = self.transformatios(image = im, mask =gt)
            im = transformer["image"]
            gt = transformer["mask"]
          

        return im, (gt > 127).long().clone().detach()

# ds = CustomDataset(root = root, transformatios= trsform)
            
def get_dl(root, bs, split = [0.8,0.1,0.1], transformatios =None):
    ds = CustomDataset(root = root, transformatios= transformatios)
    ds_len = len(ds); tr_len =int(split[0]*ds_len); val_len =int(split[1]*ds_len); ts_len = ds_len-(tr_len+val_len)
    tr_ds, val_ds, ts_ds = random_split(ds,[tr_len, val_len, ts_len]) 
    
    # print(len(tr_ds)); print(len(val_ds));print(len(ts_ds))
    
    tr_dl = DataLoader(tr_ds, batch_size =bs, shuffle =True, num_workers =0)
    val_dl = DataLoader(val_ds, batch_size =bs, shuffle =True, num_workers =0)
    ts_dl = DataLoader(ts_ds, batch_size =1, shuffle =False, num_workers =0)
    return tr_dl, val_dl, ts_dl