import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch

from model.PointConv import PointConvNet
from model.PointConv import PointConvNet2
from model.PoolingNet import PoolingNet
def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class GNN4layer(nn.Module):
    def __init__(self,**kwargs):
        super(GNN4layer, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        
      
        self.conv1 = PointConvNet(MLP([self.fea+3, 32, 64]))
        self.conv2 = PointConvNet2(MLP([64+3, 64, 128]))
        self.conv3 = PointConvNet2(MLP([128+3, 128, 128]))
        self.conv4 = PointConvNet2(MLP([128+3, 128, 128]))
        self.pool = PoolingNet(MLP([128+3, 128]))

        self.fc = nn.Sequential(
#             nn.Linear( 1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
#             nn.Linear( 256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear( 128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear( 64,   self.cla),
        )
        
    def forward(self, data):
#         x, pos, batch, edge_index = data.x, data.pos, data.batch, data.edge_index

        x, pos, batch, edge_index = self.conv1(data)

        x, pos, batch, edge_index = self.conv2(x, data)
        x, pos, batch, edge_index = self.conv3(x, data)
        x, pos, batch, edge_index = self.conv4(x, data)
        x, pos, batch = self.pool(x, pos, batch)
        out = self.fc(x)
        return out
