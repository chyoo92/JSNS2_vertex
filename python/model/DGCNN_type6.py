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
from torch_geometric.nn import GCNConv,knn_graph
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.utils.homophily import homophily

class DGCNN_type6(nn.Module):
    def __init__(self,**kwargs):
        super(DGCNN_type6, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.k = kwargs['edge']
        self.aggr = kwargs['aggr']
        self.depths = kwargs['depths']
        self.global_pool =kwargs['pool']

        self.conv1 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear((self.fea + 3) * 2, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.LeakyReLU(),
            ), self.k, self.aggr)

        self.conv2 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear(32 * 2, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.LeakyReLU(),
            ), self.k, self.aggr)

        self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(32*(self.depths) + (self.fea + 3), 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 128),
            )


        self.mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 3),
            )

        self.mlp2 = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 1),
            )
        
    def forward(self, data):
        
        
        x, pos, batch, tq = data.x, data.pos, data.batch, data.tq
        
        if self.fea == 1:
            xx = torch.cat([pos,x],dim=1)
        elif self.fea == 2:
            xx = torch.cat([tq,x,pos],dim=1)

        x1 = self.conv1(xx, batch)
        comb_fea = torch.cat([xx, x1],dim=1)
        
        for i in range(self.depths):
            
            if i == 0:
                continue
            elif i == 1:
                globals()['x%s' % (i+1)] = self.conv2(x1, batch)
                comb_fea = torch.cat([comb_fea,globals()['x%s' % (i+1)]],dim=1)
            else:
                globals()['x%s' % (i+1)] = self.conv2(globals()['x%s' % (i)], batch)
                comb_fea = torch.cat([comb_fea,globals()['x%s' % (i+1)]],dim=1)

        if self.global_pool == 0:
            out = global_max_pool(self.lin1(comb_fea), batch)
        elif self.global_pool == 1:
            out = global_mean_pool(self.lin1(comb_fea), batch)
        elif self.global_pool == 2:
            out = global_add_pool(self.lin1(comb_fea), batch)


        out1 = self.mlp(out)
        out2 = self.mlp2(out)
        out = torch.cat([out1,out2],dim=1)
        return out