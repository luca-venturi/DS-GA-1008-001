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

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

def train(model, generator, logger, iterations=60000, batch_size=32, clip_grad_norm=40.0, print_freq=100):
    # model should be a gnn
    # generator is the data_generator
    optimizer = get_optimizer(model)
    for iter_count in range(iterations):
        G, labels = generator.sample_batch(batch_size)
        pred = model(G)
        loss = compute_loss(pred, labels)
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy(pred, labels)
        if iter_count % print_freq == 0:
            print('Iter: {}, Loss: {:.4}'.format(iter_count,loss.data[0]))
    logger.plot_train_accuracy()
    logger.plot_train_loss()
    logger.plot_train_exact_accuracy()
    logger.plot_train_mismatch()
    print('Optimization finished.')
    
def test(model, generator, logger):
    # model should be a gnn
    # generator is the data_generator
    G, labels = generator.sample_batch(generator.NUM_SAMPLES_test, is_training=False)
    pred = model(G)
    logger.loss_test = []
    logger.accuracy_test = []
    logger.exact_accuracy_test = []
    logger.mismatch_test = []
    for i in range(generator.NUM_SAMPLES_test):
        loss = compute_loss(pred[i,:], labels[i,:])
        logger.add_test_loss(loss)
    logger.add_test_accuracy(pred, labels)
    logger.add_test_exact_accuracy(pred, labels)
    logger.add_test_mismatch(pred, labels)
        
    print('Clique Size: {}, Density: {:.2}, Test Loss: {:.4}'.format(logger.args['planted clique size'], logger.args['edge density'],
        np.mean(logger.loss_test)))
    print('Clique Size: {}, Density: {:.2}, Test Accuracy: {:.4}'.format(logger.args['planted clique size'], logger.args['edge density'],
        np.mean(logger.accuracy_test)))
    print('Clique Size: {}, Density: {:.2}, Test Exact Accuracy: {:.4}'.format(logger.args['planted clique size'], logger.args['edge density'],
        np.mean(logger.exact_accuracy_test)))
    print('Clique Size: {}, Density: {:.2}, Test Mismatch: {:.4}'.format(logger.args['planted clique size'], logger.args['edge density'],
        np.mean(logger.mismatch_test)))    
    return np.mean(logger.loss_test), np.mean(logger.accuracy_test), np.mean(logger.exact_accuracy_test), np.mean(logger.mismatch_test)
    
# Test

if __name__=='__main__':
    args = {'edge density' : 0.4, 'planted clique size' : 10}
    logger = make_logger()
    logger.args = args  
    generator = dataGenerator() 
    generator.edge_density = args['edge density']
    generator.clique_size = args['planted clique size']
    generator.NUM_SAMPLES_train = 1000
    generator.N = 50
    J = 5
    generator.J = J-2
    generator.create_train_dataset()
    print('Dataset created')
    num_features = 10
    num_layers = 5
    gnn = GNN(num_features, num_layers, J).type(dtype)
    train(gnn, generator, logger, iterations=2000, batch_size=32)
