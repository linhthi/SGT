import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl.function as fn
from layers.gcn_layer import GCNLayer


class GCNNet(nn.Module):
    """ """

    def __init__(self, net_params):
        super(GCNNet, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_layers = net_params['n_layers']
        self.layers = []
        self.layers.append(GCNLayer(in_dim, hidden_dim))
        for i in range(n_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, out_dim))

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        for i in range(1, len(self.layers)):
            x = F.relu(self.layers[i](g, x))
        return x

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)
        return loss
