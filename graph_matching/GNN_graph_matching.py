# Initialization

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import networkx
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor
dtype_l = torch.cuda.LongTensor

# Data generator

class dataGenerator:
    def __init__(self):
        self.NUM_SAMPLES_train = int(10e6)
        self.NUM_SAMPLES_test = int(10e4)
        self.data_train = []
        self.data_test = []
        self.J = 3
        self.N = 50
        self.edge_density = 0.2
        self.noise = 0.03
        
    def ErdosRenyi(self, p, N):
        G = networkx.erdos_renyi_graph(N, p)
        W = networkx.adjacency_matrix(G).todense().astype(float)
        W = np.array(W)
        return W
      
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
            W_pow = np.minimum(np.dot(W_pow, W_pow), np.ones(W_pow.shape))
        OP[:, :, self.J] = D
        OP[:, :, self.J + 1] = np.ones((N, N)) * (1.0 / float(N))
        x = np.reshape(deg, (N, 1))
        return OP, x
        
    def compute_sample(self):
        sample = {}
        W = self.ErdosRenyi(self.edge_density,self.N)
        # noise model from [arxiv 1602.04181], eq. (3.10)
        pe1 = self.noise
        pe2 = (self.edge_density * self.noise) / (1.0 - self.edge_density)
        noise1 = self.ErdosRenyi(pe1, self.N)
        noise2 = self.ErdosRenyi(pe2, self.N)
        noisey_W = W * (1 - noise1) + (1 - W) * noise2
        sample['OP'], sample['x'] = self.compute_operators(W)
        sample['noisey_OP'], sample['noisey_x'] = self.compute_operators(noisey_W)
        return sample
    
    def create_train_dataset(self):
        self.data_train = []
        for _ in range(self.NUM_SAMPLES_train):
            sample = self.compute_sample()
            self.data_train.append(sample)
            
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
        
        OP = torch.zeros(OP_size).expand(BATCH_SIZE, *OP_size)
        x = torch.zeros(x_size).expand(BATCH_SIZE, *x_size)
        noisey_OP = torch.zeros(OP_size).expand(BATCH_SIZE, *OP_size)
        noisey_x = torch.zeros(x_size).expand(BATCH_SIZE, *x_size)
        
        for i in range(BATCH_SIZE):
            if is_training:
                ind = np.random.randint(0, len(data))
            else:
                ind = i
            OP[i] = torch.from_numpy(data[ind]['OP'])
            x[i] = torch.from_numpy(data[ind]['x'])
            noisey_OP[i] = torch.from_numpy(data[ind]['noisey_OP'])
            noisey_x[i] = torch.from_numpy(data[ind]['noisey_x'])
            
        OP = Variable(OP, volatile=volatile)
        x = Variable(x, volatile=volatile)
        noisey_OP = Variable(noisey_OP, volatile=volatile)
        noisey_x = Variable(noisey_x, volatile=volatile)
        
        if cuda:
            return [OP.cuda(), x.cuda()], [noisey_OP.cuda(), noisey_x.cuda()]
        else:
            return [OP, x], [noisey_OP, noisey_x]
          
# Test
'''
generator = dataGenerator()
generator.NUM_SAMPLES_train = 100
generator.NUM_SAMPLES_test = 100
generator.N = 50
J = 5
generator.J = J-2
generator.create_train_dataset()
generator.create_test_dataset()
G1, G2 = generator.sample_batch(32)
print(G1[0].size())
print(G1[1][0].data.cpu().numpy())
G1 = G1[0][0, :, :, 1]
G2 = G2[0][0, :, :, 1]
print(G1, G1.size())
print(G2, G2.size())
'''
# Test: OK

# GNN Model

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
        return W, x

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_first = [1, num_features]
        self.featuremap = [num_features, num_features]
        self.layer0 = Gconv(self.featuremap_first, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last(self.featuremap, J)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out[1]
      
class Siamese_GNN(nn.Module):
    def __init__(self, num_features, num_layers, J):
        super(Siamese_GNN, self).__init__()
        self.gnn = GNN(num_features, num_layers, J)
        
    def forward(self, G1, G2):
        emb1 = self.gnn(G1)
        emb2 = self.gnn(G2)
        # emb_ is a tensor of size (bs, N, num_features)
        out = torch.bmm(emb1, emb2.permute(0, 2, 1))
        # out is a tensor of size (bs, N, N)
        return out
      
# Test
'''
num_features = 10
num_layers = 5
siamese_gnn = Siamese_GNN(num_features, num_layers, J).type(dtype)
G1, G2 = generator.sample_batch(32)
out = siamese_gnn(G1, G2)
print(out.size())
'''
# Test: OK

# Loss function

base_loss = nn.CrossEntropyLoss()

def compute_loss(pred, labels):
    # pred has size (bs, N, N)
    # labels has size (bs,N)
    pred = pred.view(-1, pred.size()[-1])
    labels = labels.view(-1)
    return base_loss(pred, labels)

### Optimizer

def get_optimizer(model):
    optimizer = optim.Adam(model.type(dtype).parameters(), lr=1e-3)
    return optimizer

### Logger

def compute_recovery_rate(pred, labels):
    pred = pred.max(2)[1] # argmax
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
        self.loss_test.append(loss)

    def add_train_accuracy(self, pred, labels):
        accuracy = compute_recovery_rate(pred, labels)
        self.accuracy_train.append(accuracy)

    def add_test_accuracy(self, pred, labels):
        accuracy = compute_recovery_rate(pred, labels)
        self.accuracy_test.append(accuracy)

    def plot_train_loss(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss: p={}, p_e={}'.format(self.args['edge density'], self.args['noise']))

    def plot_train_accuracy(self):
        plt.figure(1)
        plt.clf()
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy: p={}, p_e={}'.format(self.args['edge density'], self.args['noise']))

# Trainer

def train(model, generator, logger, iterations=60000, batch_size=32, clip_grad_norm=40.0, print_freq=100):
    # model should be a siamese_gnn
    # generator is the data_generator
    labels = Variable(torch.arange(0, generator.N).unsqueeze(0).expand(batch_size,generator.N)).type(dtype_l)
    # labels: [1,...,N] -> [[1,...N]] -> [[1,...N],...[1,...N]] of shape [batch_size,N]
    # the labels are these since the embedding the GNN should reproduce are identities
    optimizer = get_optimizer(model)
    for iter_count in range(iterations):
        input = generator.sample_batch(batch_size)
        pred = model(*input)
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
    print('Optimization finished.')
    
# Test
'''
args = {'edge density' : 0.2, 'noise' : 0.03}
logger = make_logger()
logger.args = args
generator = dataGenerator()
generator.edge_density = args['edge density']
generator.noise = args['noise']
generator.NUM_SAMPLES_train = 1000
generator.N = 50
J = 5
generator.J = J-2
generator.create_train_dataset()
print('Dataset created')
num_features = 10
num_layers = 5
siamese_gnn = Siamese_GNN(num_features, num_layers, J).type(dtype)
train(siamese_gnn, generator, logger, iterations=2000, batch_size=32)
'''
# Test: OK

# Main

if __name__ == '__main__':
    args = {'edge density' : 0.2, 'noise' : 0.03}
    logger = make_logger()
    logger.args = args
    generator = dataGenerator()
    generator.edge_density = args['edge density']
    generator.noise = args['noise']
    generator.NUM_SAMPLES_train = 20000
    generator.N = 50
    J = 6
    generator.J = J-2
    generator.create_train_dataset()
    print('Dataset created')
    num_features = 20
    num_layers = 20
    siamese_gnn = Siamese_GNN(num_features, num_layers, J).type(dtype)
    train(siamese_gnn, generator, logger, iterations=60000, batch_size=32)
