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

class EDCN_type_wf2(nn.Module):
    def __init__(self,**kwargs):
        super(EDCN_type_wf2, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.k = kwargs['edge']
        self.aggr = kwargs['aggr']
        self.depths = kwargs['depths']
        self.global_pool =kwargs['pool']

        self.conv1 = EdgeConv(torch.nn.Sequential(
                torch.nn.Linear(2*(self.fea + 3), 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            ), self.aggr)

        self.conv2 = EdgeConv(torch.nn.Sequential(
                torch.nn.Linear(64*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            ), self.aggr)
        self.conv3 = EdgeConv(torch.nn.Sequential(
                torch.nn.Linear(64*2, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
            ), self.aggr)

        self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(64*3+256, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
            )


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512, 256),    
            torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.cla),
                
            )

        
    def forward(self, data):
        
        if self.fea > 5:
            x, pos, batch, tq = data.lwf, data.pos, data.batch, data.tq
        else:
            x, pos, batch, tq = data.x, data.pos, data.batch, data.tq

        edge_index = PyG.knn_graph(pos, self.k, batch=batch, loop=True, flow='source_to_target')
        
        xx = torch.cat([x,pos],dim=1)

        x1 = self.conv1(xx, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv2(x2, edge_index)
        x4 = self.conv3(x3, edge_index)
        comb_fea = torch.cat([x1,x2,x3,x4],dim=1)

        


        if self.global_pool == 0:
            out = global_max_pool(self.lin1(comb_fea), batch)
        elif self.global_pool == 1:
            out = global_mean_pool(self.lin1(comb_fea), batch)
        elif self.global_pool == 2:
            out = global_add_pool(self.lin1(comb_fea), batch)



        out = self.mlp(out)


            
        return out
