import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch

from model.PointConv import PointConvNet
from model.PointConv import PointConvNet2
from model.PoolingNet import PoolingNet
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


class GNN55layer(nn.Module):
    def __init__(self,**kwargs):
        super(GNN55layer, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
      
#         self.conv1 = GCNConv(self.fea, 32)
#         self.conv2 = GCNConv(32, 64)
    
 
        self.fc = nn.Sequential(
            nn.Linear( 10, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear( 256, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear( 512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear( 512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear( 256,  self.cla),
        )
        
    def forward(self, data):

        out = self.fc(data.x.reshape(-1,10))
        return out
