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
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool, global_add_pool, EdgeConv
from torch_geometric.utils.homophily import homophily

class EDCN_type4(nn.Module):
    def __init__(self,**kwargs):
        super(EDCN_type4, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.k = kwargs['edge']
        self.aggr = kwargs['aggr']
        self.depths = kwargs['depths']
        self.global_pool =kwargs['pool']

        self.conv1 = EdgeConv(torch.nn.Sequential(
                torch.nn.Linear(2*(self.fea + 3), 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
            ), self.aggr)

        self.conv2 = EdgeConv(torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.LeakyReLU(),
            ), self.aggr)



        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(96*(32*(self.depths) + (self.fea + 3)), 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2),
                
            )
        self.mlp2 = torch.nn.Sequential(
                torch.nn.Linear(96*(32*(self.depths) + (self.fea + 3)), 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                
            )
        
    def forward(self, data):
        
        
        x, pos, batch, tq = data.x, data.pos, data.batch, data.tq

        edge_index = PyG.knn_graph(pos, self.k, batch=batch, loop=True, flow='source_to_target')
        
        if self.fea == 1:
            xx = torch.cat([x,pos],dim=1)
        elif self.fea == 2:
            xx = torch.cat([tq,x,pos],dim=1)

        x1 = self.conv1(xx, edge_index)
        comb_fea = torch.cat([xx, x1],dim=1)
        
        for i in range(self.depths):
            
            if i == 0:
                continue
            elif i == 1:
                globals()['x%s' % (i+1)] = self.conv2(x1, edge_index)
                comb_fea = torch.cat([comb_fea,globals()['x%s' % (i+1)]],dim=1)
            else:
                globals()['x%s' % (i+1)] = self.conv2(globals()['x%s' % (i)], edge_index)
                comb_fea = torch.cat([comb_fea,globals()['x%s' % (i+1)]],dim=1)

        out = comb_fea.reshape(batch[-1]+1,comb_fea.shape[1]*96)




        if self.cla == 3:
            out1 = self.mlp(out)
            out2 = self.mlp2(out)
            out = torch.cat([out1,out2],dim=1)
        elif self.cla == 4:
            out1 = self.mlp(out)
            out2 = self.mlp2(out)
            out3 = self.mlp2(out)
            out = torch.cat([out1,out2,out3],dim=1)
 
        return out
