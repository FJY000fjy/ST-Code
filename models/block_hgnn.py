import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph,HypergraphConv 
from torch_cluster import knn_graph 



class DistanceWeightedHypergraphLayer(nn.Module):
    def __init__(self,in_channels,out_channels,k=8,hgat_dropout_rate = None):
        super().__init__()
        self.k=k
        self.sigma = nn.Parameter(torch.tensor([1.0]))
        self.hyper_conv = HypergraphConv(in_channels,out_channels,dropout = hgat_dropout_rate)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self,x,coords,batch):
        edge_index = knn_graph(coords,k=self.k,batch=batch,loop=True)
        src_coords = coords[edge_index[0]]
        dst_coords = coords[edge_index[1]]

        dist_sq = torch.sum((src_coords-dst_coords)**2,dim=-1)
        hyperedge_weight = torch.exp(-dist_sq / (2.0*self.sigma ** 2 + 1e-6))
        out = self.hyper_conv(x,edge_index,hyperedge_weight=hyperedge_weight)
        out = self.norm(out)
        out = self.act(out)
        return out 



