import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os.path

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

class dataGenerator:
    def __init__(self):
        self.NUM_SAMPLES_train = int(10e6)
        self.NUM_SAMPLES_test = int(10e4)
        self.data_train = []
        self.data_test = []
        self.J = 3
        self.N = 50
        self.edge_density = 0.5
        self.clique_size = 10
        
    def ErdosRenyi(self, p, N):
        G = nx.erdos_renyi_graph(N, p)
        W = nx.adjacency_matrix(G).todense().astype(float)
        W = np.array(W)
        return W
      
    def plantedClique(self, p, N, C):
        W = self.ErdosRenyi(p, N)
        if C == False:
            C = np.random.randint(3,N)
            clique = np.random.choice(N, C, replace=False)
        else:
            clique = np.random.choice(N, C, replace=False)
        for i in clique:
            for j in clique:
                if j != i:
                    W[i,j] = 1
        clique_labeling = np.zeros((N), dtype=np.int)
        for i in clique:
            clique_labeling[i] = 1
        return W, clique_labeling
      
    def find_max_clique_size(self, W):  
        G = nx.from_numpy_matrix(W)
        cliques = list(nx.find_cliques(G))
        if isinstance(cliques[0], int):
            return len(cliques)
        else:
            for i in range(len(cliques)):
                cliques[i] = len(cliques[i])
            return max(cliques)
      
    def average_clique(self, sample):
        sample_size = sample.shape[0]
        cliques = []
        for i in range(sample_size):
            cliques.append(self.find_max_clique_size(sample[i, :, :, 1].data.cpu().numpy()))
        return np.mean(cliques)
        
    def compute_operators(self, W):
        N = W.shape[0]
        # OP = operators: {Id, W, W^2, ..., W^{J-1}, D, U}
        deg = W.sum(1)
        D = np.diag(deg)
        W_pow = W.copy()
        OP = np.zeros([N, N, self.J + 2])
        OP[:, :, 0] = np.eye(N)
        for j in range(self.J):
            OP[:, :, j + 1] = W_pow.copy()
            #W_pow = np.minimum(np.dot(W_pow, W_pow), np.ones(W_pow.shape))
            W_pow = np.minimum(np.dot(W_pow, W), np.ones(W_pow.shape))
        OP[:, :, self.J] = D
        OP[:, :, self.J + 1] = np.ones((N, N)) * (1.0 / float(N))
        x = np.reshape(deg, (N, 1))
        return OP, x
        
    def compute_sample(self):
        sample = {}
        W, c = self.plantedClique(self.edge_density, self.N, self.clique_size)
        sample['OP'], sample['x'] = self.compute_operators(W)
        sample['c'] = c
        return sample
    
    def create_train_dataset(self):
        self.data_train = []
        file_path = 'data/train_data_N={}_Ntrain={}'.format(self.N, self.NUM_SAMPLES_train)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as train_data_file:
                self.data_train = pickle.load(train_data_file)
        else:
            for i in range(self.NUM_SAMPLES_train):
                sample = self.compute_sample()
                self.data_train.append(sample)
                if i % 250 == 0:
                    print('train samples number {} created'.format(i))
            with open(file_path, 'wb') as train_data_file:
                pickle.dump(self.data_train, train_data_file)
            
    def create_test_dataset(self):
        self.data_test = []
        for _ in range(self.NUM_SAMPLES_test):
            sample = self.compute_sample()
            self.data_test.append(sample)
            
    def sample_batch(self, BATCH_SIZE, is_training=True, cuda=True, volatile=False):
        if is_training:
            data = self.data_train
        else:
            data = self.data_test
        OP_size = data[0]['OP'].shape
        x_size = data[0]['x'].shape
        c_size = data[0]['c'].shape
        
        OP = torch.zeros(OP_size).expand(BATCH_SIZE, *OP_size)
        x = torch.zeros(x_size).expand(BATCH_SIZE, *x_size)
        c = torch.zeros(c_size).expand(BATCH_SIZE, *c_size)
        
        for i in range(BATCH_SIZE):
            if is_training:
                ind = np.random.randint(0, len(data))
            else:
                ind = i
            OP[i] = torch.from_numpy(data[ind]['OP'])
            x[i] = torch.from_numpy(data[ind]['x'])
            c[i] = torch.from_numpy(data[ind]['c'])
            
        OP = Variable(OP, volatile=volatile)
        x = Variable(x, volatile=volatile)
        c = Variable(c, volatile=volatile)
        
        if cuda:
            return [OP.cuda(), x.cuda()], c.cuda()
        else:
            return [OP, x, c]

# Test

if __name__ == '__main__':
    generator = dataGenerator()
    generator.NUM_SAMPLES_train = 100
    generator.NUM_SAMPLES_test = 100
    generator.N = 50
    J = 5
    generator.J = J-2
    generator.create_train_dataset()
    generator.create_test_dataset()
    G, c = generator.sample_batch(32)
    print(G[0].shape,x.shape, c.shape)
    print(G[0][0, :, :, 1])
    print(G[1][0, :, 0], c[0, :])
    print(sum(c[0,:].data.cpu().numpy()))
    print(generator.average_clique(G[0]))
