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


class GCN2layer(nn.Module):
    def __init__(self,**kwargs):
        super(GCN2layer, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
      
        self.conv1 = GCNConv(self.fea, 64)
        self.conv2 = GCNConv(64, 128)
#         self.conv3 = GCNConv(128, 256)
#         self.conv4 = GCNConv(256, 512)
    

        self.fc = nn.Sequential(
#             nn.Linear( 512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
#             nn.Linear( 256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear( 128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear( 64,  self.cla),
        )

        
    def forward(self, data):
        
        edge_index = PyG.knn_graph(data.pos, 95, data.batch, loop=False, flow='source_to_target')
        x = self.conv1(data.x, edge_index)
        x = self.conv2(x, edge_index)
#         x = self.conv3(x, edge_index)
#         x = self.conv4(x, edge_index)
        x = scatter_mean(x, data.batch, dim=0)
        out = self.fc(x)
        return out
