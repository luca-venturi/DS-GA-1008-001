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
generator.NUM_SAMPLES_train = 10#000
generator.N = 50#0
J = 6
generator.J = J-2
num_features = 8 # must be even!
num_layers = 6
args = {'edge density' : 0.5, 'planted clique size' : False}
logger.args = args
generator.edge_density = args['edge density']
generator.clique_size = args['planted clique size']
generator.create_train_dataset()
print('Dataset created')
gnn = GNN(num_features, num_layers, J).type(dtype)
train(gnn, generator, logger, iterations=300, batch_size=32)#600

def test(model, generator, logger):
    # model should be a gnn
    # generator is the data_generator
    G, labels = generator.sample_batch(generator.NUM_SAMPLES_test, is_training=False)
    pred = model(G)
    logger.loss_test = []
    logger.accuracy_test = []
    for i in range(generator.NUM_SAMPLES_test):
        loss = compute_loss(pred[i,:], labels[i,:])
        logger.add_test_loss(loss)
    logger.add_test_accuracy(pred, labels)
        
    print('Clique Size: {}, Density: {:.2}, Test Loss: {:.4}'.format(logger.args['planted clique size'], logger.args['edge density'],
        np.mean(logger.loss_test)))
    print('Clique Size: {}, Density: {:.2}, Test Accuracy: {:.4}'.format(logger.args['planted clique size'], logger.args['edge density'],
        np.mean(logger.accuracy_test)))
    return np.mean(logger.loss_test), np.mean(logger.accuracy_test)

generator.NUM_SAMPLES_test = 100
densities = np.arange(0.,1.,step=0.1)
clique_sizes = [6, 10, 15, 20]
colors = {6:'b', 10:'k', 15:'r', 20:'g'}
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
    plt.semilogy(densities, [test_results[d, cs, 'loss'] for d in densities], 'b', label='C={}'.format(cs), color=colors[cs])
plt.xlabel('Edge density')
plt.ylabel('Cross Entropy Loss')
plt.title('Test Loss')
plt.legend()
path = 'plots/test_loss' 
plt.savefig(path)

# plot accuracy loss

plt.figure(1)
plt.clf()
for cs in clique_sizes: 
    plt.plot(densities, [test_results[d, cs, 'accuracy'] for d in densities], 'b', label='C={}'.format(cs), color=colors[cs])
plt.xlabel('Edge density')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
path = 'plots/test_accuracy' 
plt.savefig(path)
