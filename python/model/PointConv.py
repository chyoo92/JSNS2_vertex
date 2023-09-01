import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch

class PointConvNet(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet, self).__init__()
        self.conv = PyG.PointConv(net)
        
    def forward(self, data, batch=None):
        x, pos, batch = data.x, data.pos, data.batch
   
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 5, batch, loop=False, flow='source_to_target')
        
        
        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
    
class PointConvNet2(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet2, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, x, data, batch=None):
        x = x
        pos, batch = data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 5, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index

class PointConvNet3(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet3, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, data, batch=None):
        x, pos, batch = data.x, data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 40, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
    
class PointConvNet33(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet33, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, x, data, batch=None):
        x = x
        pos, batch = data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 40, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
    
    
class PointConvNet4(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet4, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, data, batch=None):
        x, pos, batch = data.x, data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 60, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
class PointConvNet44(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet44, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, x, data, batch=None):
        x = x
        pos, batch = data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 60, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
class PointConvNet5(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet5, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, data, batch=None):
        x, pos, batch = data.x, data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 95, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index
class PointConvNet55(nn.Module):
    def __init__(self, net, **kwargs):
        super(PointConvNet55, self).__init__()
        
        self.conv = PyG.PointConv(net)
      
    def forward(self, x, data, batch=None):
        x = x
        pos, batch = data.pos, data.batch
        
#         edge_index = PyG.radius_graph(pos, 5, batch, loop=False, flow='source_to_target')
        edge_index = PyG.knn_graph(pos, 95, batch, loop=False, flow='source_to_target')

        x = self.conv(x, pos, edge_index)
        return x, pos, batch, edge_index