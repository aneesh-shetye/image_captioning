import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, CLIPVisionModel, CLIPTextModel 

from .img_encoder import SlotImage
from .text_encoder import SlotText 
from .decoder import SlotDecoder

import matplotlib.pyplot as plt 

"""
NOTE: Be careful about the resolution that is being passed
resolution should be equal to transform applied 
"""

class SlotVQA(nn.Module): 

    def __init__(self, 
            clip_vision_model, 
            mbert, 
            mbert_out_size: int =512, 
            img_enc_out_size: int=768, 
            resolution: tuple =(224, 224), 
            slots_img: int =5, 
            iters_img: int =5, 
            slot_dim_img: int =64, 
            num_head: int =6, 
            transf_dim: int =256, 
            transf_num_layers: int =3, 
            num_layers_dec: int = 3, 
            output_vocab_size: int = 1024):  

        super().__init__()
        self.clip_vision_model = clip_vision_model
        self.img_enc_out_size = img_enc_out_size
 
        self.mbert = mbert 
        self.mbert_out_size = mbert_out_size

        self.res = resolution
        self.slots_img = slots_img
        self.iters_img = iters_img
        self.slot_dim_img = slot_dim_img
        self.slots_text = slots_text
        self.iters_text = iters_text
        self.slot_dim_text = slot_dim_text
        self.transf_dim = transf_dim
        self.output_vocab_size = output_vocab_size

        self.img_enc = SlotImage(self.clip_vision_model, 
                resolution=self.res, 
                mbert_out_size=self.img_enc_out_size,num_slots=self.slots_img, num_iter=self.iters_img, 
                slot_dim=self.slot_dim_img)            

        self.img2transf= nn.Linear(self.slot_dim_img,self.transf_dim)  

        transf_layer = nn.TransformerEncoderLayer(d_model=transf_dim, nhead=num_head, batch_first=True)
        self.transf_enc = nn.TransformerEncoder(transf_layer, num_layers=transf_num_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model=transf_dim, nhead=num_head, batch_first=True)
        self.transf_dec = nn.TransformerDecoder(decoder_layer=dec_layer,num_layers=num_layers_dec) 

        self.out2vocab = nn.Linear(self.transf_dec_size, self.output_vocab_size)


    def forward(self, 
            img: torch.tensor, 
            text: torch.tensor,
            caption: torch.tensor) 


        self.device = next(self.learnable_cls.parameters()).device

        img_att, img_emb, img_slots = self.img_enc(inp=img)
        #img_slots.shape = batch, num_slots, slot_dim 

        #projecting img slots to textual embedding space
        img_slots = self.img2transf(img_slots) 

        #making target mask
        mask = torch.ones(caption.shape)
        mask = torch.triu(mask, diagonal=1)*(-1e10).to(device)  

        out = self.transf_dec(caption, memory=img_slots, tgt_mask=mask) 
        out_token = self.out2vocab(out) 

        return out, out_token

        

 
