import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
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
    
def compute_exact_recovery_rate(pred, labels):
    pred = predict_clique(pred)
    error = 1 - torch.eq(pred, labels).type(dtype)
    frob_norm = torch.max(error, 1)[0]
    accuracy = 1 - frob_norm
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()[0]
    
def compute_mismatch(pred, labels):
    pred = predict_clique(pred)
    error = 1 - torch.eq(pred, labels).type(dtype)
    frob_norm = torch.sum(error, 1)
    accuracy = frob_norm.mean(0).squeeze()
    return accuracy.data.cpu().numpy()[0]

class make_logger(object):
    def __init__(self):
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.exact_accuracy_train = []
        self.exact_accuracy_test = []
        self.mismatch_train = []
        self.mismatch_test = []
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
        
    def add_train_exact_accuracy(self, pred, labels):
        accuracy = compute_exact_recovery_rate(pred, labels)
        self.exact_accuracy_train.append(accuracy)

    def add_test_exact_accuracy(self, pred, labels):
        accuracy = compute_exact_recovery_rate(pred, labels)
        self.exact_accuracy_test.append(accuracy)
        
    def add_train_mismatch(self, pred, labels):
        accuracy = compute_mismatch(pred, labels)
        self.mismatch_train.append(accuracy)

    def add_test_mismatch(self, pred, labels):
        accuracy = compute_mismatch(pred, labels)
        self.mismatch_test.append(accuracy)

    def plot_train_loss(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss: N={}, p={}, C={}'.format(self.args['N'], self.args['edge density'], self.args['planted clique size']))
        path = 'plots/training_loss_N={}_p={}_C={}.png'.format(self.args['N'], self.args['edge density'], self.args['planted clique size']) 
        plt.savefig(path)

    def plot_train_accuracy(self):
        plt.figure(1)
        plt.clf()
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy: N={}, p={}, C={}'.format(self.args['N'], self.args['edge density'], self.args['planted clique size']))
        path = 'plots/training_accuracy_N={}_p={}_C={}.png'.format(self.args['N'], self.args['edge density'], self.args['planted clique size'])
        plt.savefig(path)
        
    def plot_train_exact_accuracy(self):
        plt.figure(1)
        plt.clf()
        iters = range(len(self.exact_accuracy_train))
        plt.plot(iters, self.exact_accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Exact Accuracy: N={}, p={}, C={}'.format(self.args['N'], self.args['edge density'], self.args['planted clique size']))
        path = 'plots/training_exact_accuracy_N={}_p={}_C={}.png'.format(self.args['N'], self.args['edge density'], self.args['planted clique size'])
        plt.savefig(path)
        
    def plot_train_mismatch(self):
        plt.figure(1)
        plt.clf()
        iters = range(len(self.mismatch_train))
        plt.plot(iters, self.mismatch_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Mismatch')
        plt.title('Training Mismatch: N={}, p={}, C={}'.format(self.args['N'], self.args['edge density'], self.args['planted clique size']))
        path = 'plots/training_mismatch_N={}_p={}_C={}.png'.format(self.args['N'], self.args['edge density'], self.args['planted clique size'])
        plt.savefig(path)
