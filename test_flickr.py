

#region: IMPORTING LIBRARIES
import numpy as np
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim 
import torch.nn.functional as F 
from torchvision import transforms as T
from transformers import AutoTokenizer, BertModel, CLIPFeatureExtractor, CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor

from datasets.flickr8k import  MyCollate
from datasets.flickr8k import build

from model.model import SlotVQA

import wandb 
#endregion

#region: SETTING ENV VARIABLES
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#endregion

#region: SETTING SEED: 
MANUAL_SEED = 4444

random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True
#endregion

#region: ARGS
parser = argparse.ArgumentParser(description='slot_vqa')

#region: system config: 
parser.add_argument('--workers', default=5, type=int, metavar='N', 
                    help='number of data loader workers') 
parser.add_argument('--print_freq', default=5, type=int, metavar='PF', 
                    help='write in the stats file and print after PF steps') 
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='CD', 
                    help='path to directory in which checkpoint and stats are saved') 
parser.add_argument('--vg_img_path',default='/home/ladybug/Documents/fyp/Images/', 
                help='path to image directory')
parser.add_argument('--gqa_ann_path',default='/home/aneesh/Datasets/gqa_ann/OpenSource/', 
                help='path to annotations')
parser.add_argument('--gqa_split_type',default='balanced', 
        help='GQA split eg: balanced , all')

#other args: 
parser.add_argument('--load',default=True, 
        help='Load pretrained model')

# Segmentation #############################
# No idea what this sands for 
############################################
parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
parser.add_argument('--masks',action='store_true') 
#endregion

#region: training hyperparameters: 
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--imset', default='train', type=str, metavar='IS',
                    help='train, val or test set')
parser.add_argument('--batch_size', default=2, type=int, metavar='n',
                    help='mini-batch size')
parser.add_argument('--learning_rate', default=1e-5, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--dropout', default=0.01, type=float, metavar='d',
                    help='dropout for training translation transformer')
##################################################
parser.add_argument('--weight_decay', default=0.5, type=float, metavar='W',
                    help='weight decay')
##################################################
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--clip', default=100, type=float, metavar='GC',
                    help='Gradient Clipping')
parser.add_argument('--betas', default=(0.9, 0.98), type=tuple, metavar='B',
                    help='betas for Adam Optimizer')
parser.add_argument('--eps', default=1e-9, type=float, metavar='E',
                    help='eps for Adam optimizer')
parser.add_argument('--loss_fn', default='cross_entropy', type=str, metavar='LF',
                    help='loss function for translation')
##################################################################
parser.add_argument('--lambd', default=0.5, type=float, metavar='L',
                    help='lambd of loss function')
##################################################################
parser.add_argument('--optimizer', default='adam', type=str, metavar='OP',
                    help='selecting optimizer')
#endregion

#region: slot-attention hyperparameters: 
parser.add_argument('--simg', default=15, type=int, metavar='SI',
                    help='number of slots for image modality')
parser.add_argument('--itersimg', default=5, type=int, metavar='II',
                    help='numer of iterations for slot attention on images')
parser.add_argument('--slotdimimg', default=768, type=int, metavar='SDI',
                    help='dimension of slots for images')
parser.add_argument('--stext', default=15, type=int, metavar='ST',
                    help='number of slots for text modality')
parser.add_argument('--iterstext', default=4, type=int, metavar='IT',
                    help='number of iterations for slot attention on text')
parser.add_argument('--slotdimtext', default=512, type=int, metavar='IT',
                    help='number of iterations for slot attention on text')
#endregion

#region: transformer encoder hyperparameters: 
parser.add_argument('--nhead', default=8, type=int, metavar='NH',
                    help='number of heads in transformer')
parser.add_argument('--tdim', default=512, type=int, metavar='D',
                    help='dimension of transformer')
parser.add_argument('--nlayers', default=3, type=int, metavar='NL',
                    help='number of layers in transformer')
#endregion

#region: tokenizer
parser.add_argument('--text_encoder_type', default='openai/clip-vit-base-patch32', type=str, metavar='T',
                    help='text encoder')
#endregion


args = parser.parse_args()
#endregion

#region: HELPER FUNCTIONS: 
class Clip_feat_extractor(nn.Module): 
    def __init__(self, processor): 
        super().__init__()
        self.processor = processor

    def __call__(self, img): 
        return self.processor(img)['pixel_values'][0]

class Transf_CLIProcess(nn.Module): 
    def __init__(self, processor): 
        super().__init__()
        self.processor = processor
    
    def __call__(self, image): 
        return self.processor(images=image, return_tensors='pt')['pixel_values'] 

def tensor2img(img): 
    #inp.shape = 3, H, W

    test_img = img.permute(1,2,0)
    test_img = test_img.cpu().detach().numpy()
    test_img = 255*test_img
    test_img = test_img.astype(np.uint8)
    return test_img
#endregion

#region: LOSS FUNCTION DEFINATIONS: 
########################################
## changed outer torch.sum to torch.mean
########################################
def min_l2_loss(inp, trg): 
    return torch.mean(torch.min(torch.mean((inp-trg.unsqueeze(1))*(inp-trg.unsqueeze(1)), axis=(2,3,4)), axis=1).values)

def l2_loss(inp, trg): 
    return torch.mean((inp-trg)*(inp-trg))

#endregion

processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
dataset = build(root='/home/ladybug/Documents/fyp/dataset/', 
    #########################################333
    ## testing code with sett ='test'
    ########################################
resolution=[224, 224], 
transform= [Transf_CLIProcess(processor)])


loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, num_workers=args.workers,
    pin_memory=True,  collate_fn = MyCollate(tokenizer=dataset.tokenizer))


print("###########################")
for n, i in enumerate(loader): 

    j, k = i
    print(type(j),type(k))


