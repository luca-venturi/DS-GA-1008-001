import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dataGenerator import dataGenerator

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    N = W.size()[-2]
    W = W.split(1, 3) # W is a list of J tensors of size (bs, N, N, 1)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # matrix multiplication (J*N,N) x (N,num_features): output has size (bs, J*N, num_features)
    output = output.split(N, 1) # output is a list of J tensors of size (bs, N, num_features)
    output = torch.cat(output, 2)
    # output has size (bs, N, J*num_features)
    return output

class Gconv(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv, self).__init__()
        self.num_inputs = J*feature_maps[0] # size of the input
        self.num_outputs = feature_maps[1] # size of the output
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # x has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous() # makes sure that x is stored in a contiguous chunk of memory
        x = x.view(-1, self.num_inputs)
        x1 = F.relu(self.fc1(x)) # x_1 has size (bs*N, num_outputs // 2)
        x2 = self.fc2(x) # x_2 has size (bs*N, num_outputs // 2)
        x = torch.cat((x1, x2), 1) # x has size (bs*N, num_outputs)
        x = self.bn(x)
        x = x.view(*x_size[:-1], self.num_outputs) # x has size (bs, N, num_outputs)
        return W, x
      
class Gconv_last(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv_last, self).__init__()
        self.num_inputs = J*feature_maps[0] # size of the input
        self.num_outputs = feature_maps[1] # size of the output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs) # the only difference is that there is no activations layer

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1) # x has size (bs*N, num_inputs)
        x = self.fc(x) # x has size (bs*N, num_outputs)
        x = x.view(*x_size[:-1], self.num_outputs) # x has size (bs, N, num_outputs)
        x = x.squeeze(2)
        # x = x.permute(0, 2, 1) # x has size (bs, num_outputs=2, N)
        return W, x

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_first = [1, num_features]
        self.featuremap = [num_features, num_features]
        self.featuremap_last = [num_features, 1]
        self.layer0 = Gconv(self.featuremap_first, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last(self.featuremap_last, J)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out[1]
      
def predict_clique(y):
    return torch.ge(y, 0.).type(dtype)

# Loss function
    
base_loss = nn.BCEWithLogitsLoss()

def compute_loss(pred, labels):
    pred = pred.view(-1)
    labels = labels.view(-1)
    return base_loss(pred, labels)

# Optimizer

def get_optimizer(model):
    optimizer = optim.Adam(model.type(dtype).parameters(), lr=1e-3)
    return optimizer
    
# Test

if __name__ == '__main__':
    generator = dataGenerator()
    generator.NUM_SAMPLES_train = 100
    generator.NUM_SAMPLES_test = 100
    generator.N = 50
    J = 5
    generator.J = J-2
    generator.create_train_dataset()
    num_features = 10
    num_layers = 5
    gnn = GNN(num_features, num_layers, J).type(dtype)
    G, c = generator.sample_batch(32)
    out = gnn(G)
    print(out.size(), c.size())
    print(predict_clique(out))
    print(compute_loss(out,c))
