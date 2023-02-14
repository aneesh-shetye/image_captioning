'''
be careful while defining the transforms
'''
import pandas as pd 
import cv2

import numpy as np
from skimage import draw 

import torch
from torchvision import transforms as T

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import CLIPTokenizer

class Flickr8kDataset(Dataset): 

    def __init__(self, 
            root, 
            resolution,
            transform):  

        super().__init__()        
        self.root = root
        self.df = pd.read_csv(f'{self.root}/captions.txt')
        self.df = self.df.set_index('image').T.to_dict('list')
        self.T = transform
        self.res = resolution
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    
    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index: int): 

        img_id = list(self.df)[index]
        img = cv2.imread(f'{self.root}/Images/{img_id}')
        print(f'{self.root}/Images/{img_id}')
        img = torch.tensor(img)
        img = img.permute(-1, 0, 1)
        # print(f'img.shape: {img.shape}')
        caption = list(self.df.values())[index] 
        resize=T.Resize(self.res) 
        img = resize(img)        
        # print(f'image.shape after resize:{img.shape}')

        for t in self.T: 
            # print(img.shape)
            img = t(img)                    

        return {'img': img, 'caption': caption}

class MyCollate: 
    
    def __init__(self, 
    tokenizer): 

        super().__init__()
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id

    def __call__(self, batch): 
 
        imgs = [item['img'] for item in batch] 
        imgs = torch.cat(imgs, dim=0)

        captions = []
        for item in batch:
            captions.append(self.tokenizer(item['caption'][0].lower(), padding=True, 
                    max_length=512, return_tensors='pt', truncation=True)['input_ids'].T)
        captions = pad_sequence(captions, batch_first=True)


        return imgs, captions
         
def build(root, 
         resolution, 
         transform: list): 
    
    dataset = Flickr8kDataset(root=root, resolution=resolution, transform=transform) 
    return dataset 



