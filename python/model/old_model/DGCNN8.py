import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear
from model.PointConv import PointConvNet
from model.PointConv import PointConvNet2
from model.PoolingNet import PoolingNet
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_min
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool


# def MLP(channels, batch_norm=True):
#     return nn.Sequential(*[
#         nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
#         for i in range(1, len(channels))
#     ])


class DGCNN8(nn.Module):
    def __init__(self,**kwargs):
        super(DGCNN8, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        
        
        self.conv1 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear(4 * 2, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.LeakyReLU(),
            ), 5, 'add')
        self.conv2 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear(64 * 2, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.LeakyReLU(),
             ), 5, 'add')
        
        
        self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(64*4, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 512),
            )

 
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(512*4, 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, self.cla),
                
            )
        

        
    def forward(self, data):
        
       
        x, pos, batch = data.x, data.pos, data.batch

        xx = torch.cat([x,pos],dim=1)
        
        x1 = self.conv1(xx, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv2(x2, batch)
        x4 = self.conv2(x3, batch)
        
        out0 = self.lin1(torch.cat([x1, x2,x3,x4], dim=1))
        a, _ = scatter_max(out0, batch, dim=0)
        b, _ = scatter_min(out0, batch, dim=0)
        c = scatter_sum(out0, batch, dim=0)
        d = scatter_mean(out0, batch, dim=0)

        x5 = torch.cat((a,b,c,d),dim=1)
        
        

        out = self.mlp(x5)

        return out