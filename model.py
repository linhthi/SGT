import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import time
import os


class StructureEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, dropout, edge_dim, edge_hidden_dim):
        super(StructureEncoding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim

        self.W = nn.Linear(input_dim, hidden_dim * n_heads, bias=False)
        self.W_edge = nn.Linear(edge_dim, edge_hidden_dim, bias=False)
        self.W_edge_att = nn.Linear(edge_hidden_dim, n_heads, bias=False)
        self.W_att = nn.Linear(hidden_dim, n_heads, bias=False)
        self.W_out = nn.Linear(hidden_dim * n_heads, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_attr, edge_index):
        # x: [N, input_dim]
        # edge_attr: [E, edge_dim]
        # edge_index: [2, E]
        N = x.size(0)
        E = edge_index.size(1)

        # [N, hidden_dim * n_heads]
        x = self.W(x)
        # [N, n_heads, hidden_dim]
        x = x.view(N, self.n_heads, self.hidden_dim)
        # [N, n_heads, 1]
        x = self.W_att(x)
        # [N, n_heads, hidden_dim]
        x = x * x.new_ones(1, self.n_heads, self.hidden_dim)

        # [E, edge_hidden_dim]
        edge_attr = self.W_edge(edge_attr)
        # [E, n_heads]
        edge_attr = self.W_edge_att(edge_attr)
        # [E, n_heads, 1]
        edge_attr = edge_attr.unsqueeze(2)

        # [E, n_heads, hidden_dim]
        edge_attr = edge_attr * x.new_ones(E, self.n_heads, self.hidden_dim)

        # [E, n_heads, hidden_dim]
        x_src = x[edge_index[0]]
        # [E, n_heads, hidden_dim]
        x_dst = x[edge_index[1]]

        # [E, n_heads]
        alpha = (x_src * x_dst).sum(dim=-1) + edge_attr.sum(dim=-1)
        # [E, n_heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # [E, n_heads]
        alpha = F.softmax(alpha, dim=0)

        # [E, n_heads, hidden_dim]
        x_dst = x_dst * alpha.unsqueeze(2)

        # [N, n_heads, hidden_dim]
        x = x.new_zeros(N, self.n_heads, self.hidden_dim)
        # [N, n_heads, hidden_dim]
        x = x.scatter_add(0, edge_index[1].unsqueeze(0).expand(self.n_heads, -1), x_dst)

        # [N, hidden_dim * n_heads]
        x = x.view(N, -1)
        # [N, hidden_dim]
        x = self.W_out(x)
        # [N, hidden_dim]
        x = F.relu(x)
        # [N, hidden_dim]
        x = self.dropout(x)

        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout, edge_dim, edge_hidden_dim):
        super(GraphTransformerLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim

        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, edge_hidden_dim, bias=False)
        self.W_r = nn.Linear(edge_hidden_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.fc_dropout = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inputs, adj):
        # inputs: [N, input_dim]
        # adj: [N, N, edge_dim]

        N = inputs.shape[0]

        # compute query, key, value
        Q = self.W_q(inputs) # [N, output_dim]
        K = self.W_k(inputs) # [N, output_dim]
        V = self.W_v(inputs) # [N, output_dim]

        # split heads
        Q = Q.view(N, self.num_heads, self.output_dim // self.num_heads) # [N, num_heads, output_dim // num_heads]
        K = K.view(N, self.num_heads, self.output_dim // self.num_heads) # [N, num_heads, output_dim // num_heads]
        V = V.view(N, self.num_heads, self.output_dim // self.num_heads) # [N, num_heads, output_dim // num_heads]

        Q = Q.permute(1, 0, 2) # [num_heads, N, output_dim // num_heads]
        K = K.permute(1, 0, 2) # [num_heads, N, output_dim // num_heads]
        V = V.permute(1, 0, 2) # [num_heads, N, output_dim // num_heads]


class GraphTransformerEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, dropout):
        super(GraphTransformerEncoding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GraphTransformerLayer(input_dim, hidden_dim, n_heads, dropout))
        for i in range(n_layers - 2):
            self.layers.append(GraphTransformerLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout))
        self.layers.append(GraphTransformerLayer(hidden_dim * n_heads, output_dim, n_heads, dropout))

    def forward(self, x, adj):
        for i in range(self.n_layers):
            x = self.layers[i](x, adj)
        return x