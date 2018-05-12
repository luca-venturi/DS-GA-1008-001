import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import argparse

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataGenerator import dataGenerator
from model import *
from logger import *
from trainer import train, test

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

logger = make_logger()

clique_sizes = np.arange(5, 11) # 33
test_results = {}
for C in clique_sizes:
    generator = dataGenerator()
    generator.NUM_SAMPLES_train = 100# 10000
    generator.N = 30# 200
    J = 6
    generator.J = J-2
    num_features = 8 # must be even!
    num_layers = 6
    args = {'N' : generator.N, 'edge density' : 0.5, 'planted clique size' : C}
    logger.args = args
    generator.edge_density = args['edge density']
    generator.clique_size = args['planted clique size']
    generator.create_train_dataset()
    print('Dataset created')
    gnn = GNN(num_features, num_layers, J).type(dtype)
    train(gnn, generator, logger, iterations=1000, batch_size=32) # 10000
    
    generator.NUM_SAMPLES_test = 100
    generator.create_test_dataset()
    print('Test dataset created')
    test_results[C, 'loss'], test_results[C, 'accuracy'], _, test_results[C, 'mismatch'] = test(gnn, generator, logger)
    test_results[C, 'mismatch'] /= float(C)
        
# plot test loss

plt.figure(0)
plt.clf()
plt.semilogy(clique_sizes, [test_results[cs, 'loss'] for cs in clique_sizes], 'b')
plt.xlabel('Edge density')
plt.ylabel('Cross Entropy Loss')
plt.title('Test Loss: N={}'.format(logger.args['N']))
plt.legend()
path = 'plots/test_loss_N={}_trained_with_C'.format(logger.args['N'])
plt.savefig(path)

# plot test accuracy

plt.figure(1)
plt.clf()
plt.plot(clique_sizes, [test_results[cs, 'accuracy'] for cs in clique_sizes], 'b')
plt.xlabel('Edge density')
plt.ylabel('Accuracy')
plt.title('Test Accuracy: N={}'.format(logger.args['N']))
plt.legend()
path = 'plots/test_accuracy_N={}_trained_with_C'.format(logger.args['N'])
plt.savefig(path)

# plot test mismatch

plt.figure(2)
plt.clf()
plt.plot(clique_sizes, [test_results[cs, 'mismatch'] for d in clique_sizes], 'b')
plt.xlabel('Edge density')
plt.ylabel('Mismatch')
plt.title('Test Mismatch: N={}'.format(logger.args['N']))
plt.legend()
path = 'plots/test_mismatch_N={}_trained_with_C'.format(logger.args['N'])
plt.savefig(path)

# print

for cs in clique_sizes:
    print('CS = {} -> mismatch = {}').format(cs, test_results[cs, 'mismatch'])
