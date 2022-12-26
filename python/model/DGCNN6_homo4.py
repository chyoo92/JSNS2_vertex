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
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum


# def MLP(channels, batch_norm=True):
#     return nn.Sequential(*[
#         nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
#         for i in range(1, len(channels))
#     ])


class DGCNN6_homo4(nn.Module):
    def __init__(self,**kwargs):
        super(DGCNN6_homo4, self).__init__()
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        
        
        self.conv1 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear(5 * 2, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.LeakyReLU(),
            ), 15, 'add')
        self.conv2 = DynamicEdgeConv(torch.nn.Sequential(
                torch.nn.Linear(64 * 2, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
             ), 15, 'add')
        self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(64*3 + 5, 512),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 256),
            )

 
        self.mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256+5, 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, self.cla),
                
            )


        
    def forward(self, data):
        
       
        x, pos, batch, tq = data.x, data.pos, data.batch, data.tq

        xx = torch.cat([tq,x,pos],dim=1)
        
        edge_index = knn_graph(xx,k = 95,batch = batch)
        htq = homophily(edge_index, xx[:, 0], batch).reshape(-1, 1)
        hq = homophily(edge_index, xx[:, 1], batch).reshape(-1, 1)
        hx = homophily(edge_index, xx[:, 2], batch).reshape(-1, 1)
        hy = homophily(edge_index, xx[:, 3], batch).reshape(-1, 1) 
        hz = homophily(edge_index, xx[:, 4], batch).reshape(-1, 1) 

        edge_index = []
        
        
        
        
        
        x1 = self.conv1(xx, batch)
        x2= self.conv2(x1, batch)
        x3 = self.conv2(x2, batch)
        out = self.lin1(torch.cat([xx, x1, x2, x3], dim=1))
        
        # Aggregation across nodes
#         a, _ = scatter_max(out, batch, dim=0)
#         b, _ = scatter_min(out, batch, dim=0)
#         c = scatter_sum(out, batch, dim=0)
        d = scatter_mean(out, batch, dim=0)        
#         out = torch.cat((a,b,c,d,
#                          htq.reshape(-1,1),
#                          hq.reshape(-1,1),
#                          hx.reshape(-1,1),
#                          hy.reshape(-1,1),
#                          hz.reshape(-1,1),
                         
#                         ),dim=1,)
             
        out = torch.cat((d,
                         htq.reshape(-1,1),
                         hq.reshape(-1,1),
                         hx.reshape(-1,1),
                         hy.reshape(-1,1),
                         hz.reshape(-1,1),
                         
                        ),dim=1,)
  
        out = self.mlp(out)

        return out
