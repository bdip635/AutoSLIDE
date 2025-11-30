import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import nibabel as nib
from utils import (generate_ex_list, gen_mask, correct_dims, resize_img,
                   center_crop, find_and_crop_lesions, random_crop)
from random import random, randint
from scipy import ndimage
import pandas as pd
from collections import defaultdict
from statistics import median

def normalize_img(x):
    mx, mn = x.max(), x.min()
    if mx==mn:
        x = np.zeros_like(x)
    else:
        x = (x-mn)/(mx-mn)
    return x

class MRIDataset(Dataset):
    def __init__(self, csv_path, patch_shape, sampling_mode="random", deterministic=False):
        assert sampling_mode in ['resize', 'center', 'center_val', 'random']
        # self.main_dir = main_dir
        self.df = pd.read_csv(csv_path)
        # self.inputs, self.labels = generate_ex_list(self.main_dir)
        self.size = patch_shape
        self.sampling_mode = sampling_mode
        self.deterministic = deterministic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # self.current_item_path = self.inputs[idx]
        # input_img = correct_dims(nib.load(self.inputs[idx]).get_data())
        # label_img = gen_mask(self.labels[idx])
        
        image_id = self.df.iloc[idx]['ID']
        image_path = self.df.iloc[idx]['Image']
        mask_path = self.df.iloc[idx]['Mask']
        
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        if image.shape!=mask.shape:
            zoom = np.array(np.shape(image)) / np.shape(mask)
            mask = ndimage.zoom(mask, zoom)
        assert image.shape==mask.shape
        
        # Resize to input image and label to size (self.size x self.size x self.size)
        if self.sampling_mode == "resize":
            ex, label = resize_img(image, mask, self.size)

        # Constant center-crop sample of size (self.size x self.size x self.size)
        elif self.sampling_mode == 'center':
            ex, label = center_crop(image, mask, self.size)

        # Find centers of lesion masks and crop image to include them
        # to measure consistent validation performance with small crops
        elif self.sampling_mode == "center_val":
            ex, label = find_and_crop_lesions(image, mask, self.size, self.deterministic)

        # Randomly crop sample of size (self.size x self.size x self.size)
        elif self.sampling_mode == "random":
            ex, label = random_crop(image, mask, self.size)

        else:
            print("Invalid sampling mode.")
            exit()
        
        ex = normalize_img(ex)
        label = np.array([(label > 127).astype(int)]).squeeze()
        # (experimental) APPLY RANDOM FLIPPING ALONG EACH AXIS
        if not self.deterministic:
            for i in range(3):
                if random() > 0.5:
                    ex = np.flip(ex, i)
                    label = np.flip(label, i)
        inputs = torch.unsqueeze(torch.from_numpy(ex.copy()).type(torch.FloatTensor), 0)
        labels = torch.unsqueeze(torch.from_numpy(label.copy()).type(torch.FloatTensor), 0)
        # return image_id, inputs, labels
        return inputs, labels


    def _load_full_label(self, label_paths):
        """
        Loading full-size label for evaluation.
        """
        label_full = gen_mask(label_paths)
        label_full = np.array([(label_full > 0).astype(int)]).squeeze()
        return torch.Tensor(label_full)

    def _project_full_label(self, input_path, preds):
        """
        Projecting a predicted subvolume to full-size volume for evaluation.
        Only supposed to work with center sampling_mode.
        """
        size = nib.load(input_path).get_data().shape
        preds_full = np.zeros(size)
        coords = [0]*3
        for i in range(3):
            coords[i] = int((size[i]-self.size[i])//2)
        x, y, z = coords
        preds_full[x:x+self.size[0], y:y+self.size[1], z:z+self.size[2]] = preds
        return torch.Tensor(preds_full)


class MI_Dataset(Dataset):
  def __init__(self, csv_path, patch_shape, sampling_mode= 'random', deterministic= False):
    # super().__init__()
    assert sampling_mode in ['resize', 'center', 'center_val', 'random']
    self.df= pd.read_csv(csv_path)
    self.patch_shape= patch_shape
    self.sampling_mode = sampling_mode
    self.deterministic = deterministic
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    image_id = self.df.iloc[idx]['ID']
    image_path= self.df.iloc[idx]['Image']
    mask_path= self.df.iloc[idx]['Mask']
    parce_path= self.df.iloc[idx]['MRI2']

    image= nib.load(image_path).get_fdata()
    mask= nib.load(mask_path).get_fdata()
    parce= nib.load(parce_path).get_fdata()
    
    if image.shape!=mask.shape:
        zoom = np.array(np.shape(image)) / np.shape(mask)
        mask = ndimage.zoom(mask, zoom)
    assert image.shape==mask.shape
    assert mask.shape==parce.shape

    size= self.patch_shape
    if self.sampling_mode == 'resize':
      zoom = np.array(size)/np.shape(image)
      image= ndimage.zoom(image, zoom)
      mask= ndimage.zoom(mask, zoom)
      parce= ndimage.zoom(parce, zoom)
    
    elif self.sampling_mode == 'center':
      coords=[int((image.shape[i]-size[i])//2) for i in range(3)]
      x, y, z = coords
      image = image[x:x+size[0], y:y+size[1], z:z+size[2]]
      mask = mask[x:x+size[0], y:y+size[1], z:z+size[2]]
      parce= parce[x:x+size[0], y:y+size[1], z:z+size[2]]
    
    elif self.sampling_mode == 'center_val':
      nonzeros = mask.nonzero()
      d = [0]*3
      if not self.deterministic:
        for i in range(3):
            d[i] = randint(-size[i]//4, size[i]//4)
      coords = [max(min(int(median(nonzeros[i])) - (size[i] // 2) + d[i], image.shape[i] - size[i] - 1), 0) for i in range(3)]
      x, y, z = coords
      image = image[x:x+size[0], y:y+size[1], z:z+size[2]]
      mask = mask[x:x+size[0], y:y+size[1], z:z+size[2]]
      parce = parce[x:x+size[0], y:y+size[1], z:z+size[2]]

    elif self.sampling_mode == 'random':
      remove_background = False
      non_zero_percentage = 0
      while non_zero_percentage < 0.7:
        """draw x,y,z coords
        """
        coords = [0]*3
        for i in range(3):
            if image.shape[i]!=size[i]:
                coords[i] = np.random.choice(image.shape[i]-size[i])
        x, y, z = coords
        image = image[x:x+size[0], y:y+size[1], z:z+size[2]]
        non_zero_percentage = np.count_nonzero(image) / float(size[0]*size[1]*size[2])
        if not remove_background:
            break
        if non_zero_percentage < 0.7:
            del image
      mask= mask[x:x+size[0], y:y+size[1], z:z+size[2]]
      parce= parce[x:x+size[0], y:y+size[1], z:z+size[2]]
    
    else:
        print('Invalid sampling mode.')
        exit()
    image = normalize_img(image)
    mask = np.array([(mask > 127).astype(int)]).squeeze()
    parce = normalize_img(parce) ##Check
    image= torch.unsqueeze(torch.from_numpy(image.copy()).type(torch.FloatTensor), 0)
    mask= torch.unsqueeze(torch.from_numpy(mask.copy()).type(torch.FloatTensor), 0)
    parce= torch.unsqueeze(torch.from_numpy(parce.copy()).type(torch.FloatTensor), 0)
    data = torch.cat((image, parce), dim=0)

    return data, mask