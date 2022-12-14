"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gcn_net import GCNNet
from nets.SAN import SAN
from nets.SAN_NodeLPE import SAN_NodeLPE
from nets.SAN_EdgeLPE import SAN_EdgeLPE

def NodeLPE(net_params):
    return SAN_NodeLPE(net_params)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return SAN(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def gnn_model(LPE, net_params):
    model = {
        'edge': EdgeLPE,
        'node': NodeLPE,
        'none': NoLPE,
        'gcn': GCN
    }
        
    return model[LPE](net_params)