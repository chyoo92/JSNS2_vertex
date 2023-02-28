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
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool
from torch_geometric.utils.homophily import homophily

class DGCNN_type4(nn.Module):
    def __init__(self,**kwargs):
        super(DGCNN_type4, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.k = kwargs['edge']
        self.aggr = kwargs['aggr']
        self.depths = kwargs['depths']
        self.global_pool =kwargs['pool']

        self.conv11 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear((self.fea + 2) * 2, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.LeakyReLU(),
            ), self.k, self.aggr)
        self.conv12 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear((self.fea + 1) * 2, 32),
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
                torch.nn.Linear(32*(self.depths) + 32*(self.depths) + (self.fea + 3 + self.fea), 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 256),
            )


        self.mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, self.cla),
                
            )
        
    def forward(self, data):
        
        
        x, pos, batch, tq = data.x, data.pos, data.batch, data.tq
    
        if self.fea == 1:
            xx1 = torch.cat([pos[:,:2],x],dim=1)
            xx2 = torch.cat([pos[:,2].reshape(-1,1),x],dim=1)
        elif self.fea == 2:
            xx = torch.cat([tq,x,pos],dim=1)

        x11 = self.conv11(xx1, batch)
        x21 = self.conv12(xx2, batch)

        comb_fea1 = torch.cat([xx1, x11],dim=1)
        comb_fea2 = torch.cat([xx2, x21],dim=1)

        for i in range(self.depths):
            
            if i == 0:
                continue
            elif i == 1:
                globals()['x1%s' % (i+1)] = self.conv2(x11, batch)
                comb_fea1 = torch.cat([comb_fea1,globals()['x1%s' % (i+1)]],dim=1)
                
                globals()['x2%s' % (i+1)] = self.conv2(x21, batch)
                comb_fea2 = torch.cat([comb_fea2,globals()['x2%s' % (i+1)]],dim=1)


            else:
                globals()['x1%s' % (i+1)] = self.conv2(globals()['x1%s' % (i)], batch)
                comb_fea1 = torch.cat([comb_fea1,globals()['x1%s' % (i+1)]],dim=1)

                globals()['x2%s' % (i+1)] = self.conv2(globals()['x2%s' % (i)], batch)
                comb_fea2 = torch.cat([comb_fea2,globals()['x2%s' % (i+1)]],dim=1)
        comb_fea = torch.cat([comb_fea1, comb_fea2],dim=1)
        if self.global_pool == 0:
            out = global_max_pool(self.lin1(comb_fea), batch)
        elif self.global_pool == 1:
            out = global_mean_pool(self.lin1(comb_fea), batch)


        out = self.mlp(out)

        return out
