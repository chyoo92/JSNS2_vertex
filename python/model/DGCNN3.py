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
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


# def MLP(channels, batch_norm=True):
#     return nn.Sequential(*[
#         nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
#         for i in range(1, len(channels))
#     ])


class DGCNN3(nn.Module):
    def __init__(self,**kwargs):
        super(DGCNN3, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        
        
        self.conv1 = DynamicEdgeConv(MLP([2 * 4, 64, 64, 64]), 5, 'add')
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), 5, 'add')
        self.lin1 = Linear(128 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, self.cla], dropout=0.5)
        

        
    def forward(self, data):
        
        x, pos, batch = data.x, data.pos, data.batch

        xx = torch.cat([x,pos],dim=1)
        
        x1 = self.conv1(xx, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        

        return out
