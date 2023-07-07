import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear


from model.TPED import *

class SAVER(nn.Module):
    def __init__(self,**kwargs):
        super(SAVER, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.hidden_dim = kwargs['hidden']
        self.n_layers = kwargs['depths']
        self.n_heads = kwargs['heads']
        self.pf_dim = kwargs['posfeed']
        self.dropout_ratio = kwargs['dropout']
        self.device = kwargs['device']
        self.batch = kwargs['batch']
        
        self.encoder = Encoder(self.fea, self.hidden_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout_ratio, self.device)
 
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(96*self.hidden_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, self.cla),
                
            )

        
    def forward(self, data):
        
        if self.fea > 5:
            x, pos, batch, tq = data.lwf, data.pos, data.batch, data.tq
        else:
            x, pos, batch, tq = data.x, data.pos, data.batch, data.tq

        fea = torch.cat([x,pos],dim=1)

        fea = torch.reshape(fea,(int(x.shape[0]/96),96,4))

        out = self.encoder(fea)
        out = torch.reshape(out, (int(x.shape[0]/96),96*self.hidden_dim))
        out = self.mlp(out)


            
        return out
