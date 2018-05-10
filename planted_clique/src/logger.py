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

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

def compute_recovery_rate(pred, labels):
    pred = predict_clique(pred)
    error = 1 - torch.eq(pred, labels).type(dtype)
    frob_norm = error.mean(1)
    accuracy = 1 - frob_norm
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()[0]

class make_logger(object):
    def __init__(self):
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.args = {}

    def add_train_loss(self, loss):
        self.loss_train.append(loss.data.cpu().numpy())

    def add_test_loss(self, loss):
        self.loss_test.append(loss.data.cpu().numpy())

    def add_train_accuracy(self, pred, labels):
        accuracy = compute_recovery_rate(pred, labels)
        self.accuracy_train.append(accuracy)

    def add_test_accuracy(self, pred, labels):
        accuracy = compute_recovery_rate(pred, labels)
        self.accuracy_test.append(accuracy)

    def plot_train_loss(self):
        plt.clf()
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss: p={}, C={}'.format(self.args['edge density'], self.args['planted clique size']))
        path = 'plots/training_loss_p={}_C={}.png'.format(self.args['edge density'], self.args['planted clique size']) 
        plt.savefig(path)

    def plot_train_accuracy(self):
        plt.clf()
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy: p={}, C={}'.format(self.args['edge density'], self.args['planted clique size']))
        path = 'plots/training_accuracy_p={}_C={}.png'.format(self.args['edge density'], self.args['planted clique size']) 
        plt.savefig(path)
