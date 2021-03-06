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

parser = argparse.ArgumentParser()
parser.add_argument('--clique_size', nargs='?', const=1, type=int, default=int(32))
parser_args = parser.parse_args()

logger = make_logger()
generator = dataGenerator()
generator.NUM_SAMPLES_train = 10000
generator.N = 200
J = 6
generator.J = J-2
num_features = 8 # must be even!
num_layers = 6
C = parser_args.clique_size
args = {'N' : generator.N, 'edge density' : 0.5, 'planted clique size' : C}
logger.args = args
generator.edge_density = args['edge density']
generator.clique_size = args['planted clique size']
generator.create_train_dataset()
print('Dataset created')
gnn = GNN(num_features, num_layers, J).type(dtype)
train(gnn, generator, logger, iterations=10000, batch_size=32)#10000

generator.NUM_SAMPLES_test = 100
densities = np.arange(0.,0.6,step=0.1)
clique_sizes = [C]
colors = {5:'b', 10:'k', 20:'r', 32:'g'}
test_results = {}
for d in densities:
    for cs in clique_sizes:
        args = {'N' : generator.N, 'edge density' : d, 'planted clique size' : cs}
        logger.args = args
        generator.edge_density = args['edge density']
        generator.clique_size = args['planted clique size']
        generator.create_test_dataset()
        print('Test dataset created')
        test_results[d, cs, 'loss'], test_results[d, cs, 'accuracy'], test_results[d, cs, 'exact accuracy'], test_results[d, cs, 'mismatch'] = test(gnn, generator, logger)
        
# plot test loss

plt.figure(0)
plt.clf()
for cs in clique_sizes: 
    plt.semilogy(densities, [test_results[d, cs, 'loss'] for d in densities], 'b', label='C={}'.format(cs), color=colors[cs])
plt.xlabel('Edge density')
plt.ylabel('Cross Entropy Loss')
plt.title('Test Loss: N={}'.format(logger.args['N']))
plt.legend()
path = 'plots/test_loss_N={}_C={}'.format(logger.args['N'], C)
plt.savefig(path)

# plot test accuracy

plt.figure(1)
plt.clf()
for cs in clique_sizes: 
    plt.plot(densities, [test_results[d, cs, 'accuracy'] for d in densities], 'b', label='C={}'.format(cs), color=colors[cs])
plt.xlabel('Edge density')
plt.ylabel('Accuracy')
plt.title('Test Accuracy: N={}'.format(logger.args['N']))
plt.legend()
path = 'plots/test_accuracy_N={}_C={}'.format(logger.args['N'], C)
plt.savefig(path)

# plot test mismatch

plt.figure(1)
plt.clf()
for cs in clique_sizes: 
    plt.plot(densities, [test_results[d, cs, 'mismatch'] for d in densities], 'b', label='C={}'.format(cs), color=colors[cs])
plt.xlabel('Edge density')
plt.ylabel('Mismatch')
plt.title('Test Mismatch: N={}'.format(logger.args['N']))
plt.legend()
path = 'plots/test_mismatch_N={}_C={}'.format(logger.args['N'], C)
plt.savefig(path)
