'''
Module which takes new (spatial) layers and performs attention to weight over all previous spatial layers.  

Needs to know beforehand number of channels in each layer.  

-For key and query, do mean-pooling over spatial dimensions and then do an FC-layer.  

-For value, do a conv-layer.  When using on the current time-step, resize values to the current size, then reshape.  

(new_layer) --> (attentive_input).  Keeps and updates a key_lst and value_lst internally.  

-Should attention be same or different object?  Probably easiest to make it the same object.  

-Need to figure out how to resize the value layers

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
#from attention import ScaledDotProductAttention
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
#from sparse_attn import SparseAttention

import torch
import torch.nn as nn
import numpy

class SparseAttention(nn.Module):
    def __init__(self, top_k = 3):
        super(SparseAttention,self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):

        attn_plot = []
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            return attn_s
        else:
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            delta = delta.reshape((delta.shape[0],1))


        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        return attn_w_normalize

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout=0.1,att_sparsity=None):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        print('att sparsity?', att_sparsity)
        if att_sparsity is None:
            self.use_sparse = False
        else:
            self.use_sparse = True
            self.sa = SparseAttention(top_k=att_sparsity)

        self.dropout = nn.Dropout(dropout)

        self.sa = SparseAttention()

    def forward(self, q, k, v, mask=None):

        # bs x pos x key .. bs x key x pos

        # bs x pos x pos .. bs x pos x key

        attn = torch.matmul(q / self.temperature, k.permute(0,2,1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        if self.use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb*ins, outs))
            sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb,ins,outs))
            attn = sparse_attn*1.0

        output = torch.matmul(attn, v)

        return output, attn

class AttentiveDensenet(nn.Module):
    def __init__(self, layer_channels, key_size, val_size, n_heads, att_sparsity=3,attn_dropout=0.0,concentration=0.5):
        super(AttentiveDensenet, self).__init__()

        print('concentration?',  concentration)

        self.layer_channels = layer_channels
        self.key_size = key_size
        self.val_size = val_size
        self.n_heads = n_heads

        self.key_layers = []
        self.query_layers = []
        self.val_layers = []

        self.out_layers = []

        self.gammas = []

        for ch in layer_channels:
            self.key_layers.append(nn.Conv2d(ch, key_size*n_heads,1,stride=1,padding=0))
            self.query_layers.append(nn.Conv2d(ch, key_size*n_heads,1,stride=1,padding=0))
            self.val_layers.append(nn.Conv2d(ch, val_size*n_heads,1,stride=1,padding=0))
            out_layer = nn.Sequential(nn.Conv2d(val_size*n_heads,val_size*n_heads,1,stride=1,padding=0), nn.BatchNorm2d(val_size*n_heads,momentum=0.99, eps=1e-3), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(val_size*n_heads, ch,1,stride=1,padding=0))
            #out_layer = nn.Conv2d(val_size*n_heads, ch, 1)
            self.out_layers.append(out_layer)
            self.gammas.append(nn.Parameter(torch.tensor(0.0)))
        
        self.query_layers = nn.ModuleList(self.query_layers)
        self.key_layers = nn.ModuleList(self.key_layers)
        self.val_layers = nn.ModuleList(self.val_layers)
        self.out_layers = nn.ModuleList(self.out_layers)
        self.gammas = nn.ParameterList(self.gammas)

        self.layer_index = None

        self.attn = ScaledDotProductAttention(np.power(key_size, concentration),att_sparsity=att_sparsity,dropout=attn_dropout)

    def reset(self):
        self.key_lst = []
        self.val_lst = []
        self.layer_index = 0

    def forward(self, x, read, write):
    
        sz_b, ch, h, w = x.shape

        if write:
            h_key = self.key_layers[self.layer_index](x) #bs x n_heads*key_size x h x w
            val = self.val_layers[self.layer_index](x)
            self.key_lst.append(h_key)
            self.val_lst.append(val)
        else:
            self.key_lst.append(None)
            self.val_lst.append(None)

        if not read:
            self.layer_index += 1
            return x

        h_query = self.query_layers[self.layer_index](x)
        query = h_query.reshape((sz_b,self.n_heads,self.key_size,h,w)).permute(0,3,4,1,2).reshape((sz_b*h*w*self.n_heads, 1, self.key_size)) #sz_b, h, w, n_heads key_size

        #print('key shape', key.shape)
        #print('query shape', query.shape)
        
        vals_reshaped = []
        keys_reshaped = []


        for ind in range(len(self.val_lst)):

            val = self.val_lst[ind]
            key = self.key_lst[ind]


            if val is None:
                assert key is None
                continue

            val_resized = F.interpolate(val, (h,w), mode='nearest')
            key_resized = F.interpolate(key, (h,w), mode='nearest')

            #if h > val.shape[2]:
            #    #upsample
            #    val_resized = F.interpolate(val, (h,w), mode='nearest')
            #    key_resized = F.interpolate(key, (h,w), mode='nearest')
            #else:
                #downsample
            #    maxpool = nn.AvgPool2d((val.shape[2]//h, val.shape[2]//h))
            #    val_resized = maxpool(val)
            #    key_resized = maxpool(key)

            val_resized = val_resized.reshape((sz_b, self.n_heads,self.val_size,h,w)).permute(0,3,4,1,2).reshape((sz_b*h*w*self.n_heads, 1, self.val_size))#.repeat(1,self.n_heads,1,1)
            vals_reshaped.append(val_resized)
            key_old = key_resized.reshape((sz_b,self.n_heads,self.key_size,h,w)).permute(0,3,4,1,2).reshape((sz_b*h*w*self.n_heads, 1, self.key_size))
            keys_reshaped.append(key_old)

        vals_tensor = torch.cat(vals_reshaped, dim = 1)
        keys_tensor = torch.cat(keys_reshaped, dim = 1)

        #print('query shape', query.shape)
        #print('vals tensor shape', vals_tensor.shape)
        #print('keys tensor shape', keys_tensor.shape)

        keys_tensor = torch.cat([keys_tensor, torch.zeros_like(keys_tensor[:,0:1])], dim=1)
        vals_tensor = torch.cat([vals_tensor, torch.zeros_like(vals_tensor[:,0:1])], dim=1)

        att_out,iatt = self.attn(query, keys_tensor, vals_tensor)

        att_out = att_out.reshape((sz_b, h, w, self.n_heads, self.val_size)).permute(0,3,4,1,2).reshape((sz_b, self.n_heads*self.val_size, h, w))

        #print('att out shape', att_out.shape)
        att_out = self.out_layers[self.layer_index](att_out)

        att_out = x + att_out * self.gammas[self.layer_index]

        self.layer_index += 1

        return att_out

if __name__ == "__main__":
    channels = [3,64,64,128,256]

    ad = AttentiveDensenet(channels, 32, 32, 4)

    ad.reset()

    for ch in channels: 

        if ch == 3:
            h = 32
        elif ch == 64:
            h = 16
        else:
            h = 4

        layerx = torch.randn(64, ch, h, h)

        inp_out = ad(layerx)









