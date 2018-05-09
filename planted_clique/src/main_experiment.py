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
from model import *
from logger import *
from trainer import train, test

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

logger = make_logger()
generator = dataGenerator()
generator.NUM_SAMPLES_train = 20000
generator.N = 50
J = 6
generator.J = J-2
num_features = 20
num_layers = 20
args = {'edge density' : 0.5, 'planted clique size' : 6}
logger.args = args
generator.edge_density = args['edge density']
generator.clique_size = args['planted clique size']
generator.create_train_dataset()
gnn = GNN(num_features, num_layers, J).type(dtype)
train(gnn, generator, logger, iterations=30000, batch_size=32)

densities = [0.1, 0.2, 0.3, 0.4, 0.5]
clique_sizes = [6, 10, 15, 20]
test_results = {}
for d in densities:
    for cs in clique_sizes:
        args = {'edge density' : d, 'planted clique size' : cs}
        logger.args = args
        generator.edge_density = args['edge density']
        generator.clique_size = args['planted clique size']
        generator.create_test_dataset()
        print('Test dataset created')
        test_results[d, cs, 'loss'], test_results[d, cs, 'accuracy'] = test(gnn, generator, logger)
        
# plot test loss

plt.figure(0)
plt.clf()
for cs in clique_sizes: 
    plt.semilogy(iters, test_results[:, cs, 'loss'], 'b', label='C={}'.format(cs))
plt.xlabel('Edge density')
plt.ylabel('Cross Entropy Loss')
plt.title('Test Loss')
plt.legend()
path = 'plots/test_loss.png') 
plt.savefig(path)

# plot accuracy loss

plt.figure(1)
plt.clf()
for cs in clique_sizes: 
    plt.plot(iters, test_results[:, cs, 'accuracy'], 'b', label='C={}'.format(cs))
plt.xlabel('Edge density')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
path = 'plots/test_accuracy.png') 
plt.savefig(path)
