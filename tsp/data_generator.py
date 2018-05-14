import numpy as np
import math
import os
from LKH.tsp_solver import TSPSolver
import torch
from torch.autograd import Variable

# Checking if GPU is available
cuda = False
if torch.cuda.is_available() :
    GPU_capab = torch.cuda.get_device_capability(torch.cuda.current_device()) 
    if GPU_capab[0] > 5:
        cuda = True
if cuda:
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

def l2_dist(x,y):
    return math.ceil(np.sqrt(np.square(x-y).sum()))

class Generator(object):
    """ Generate and manages datasets for TSP problem """
    def __init__(self, args):
        super().__init__()
        self.C = 10e6
        self.LKHsolver = TSPSolver(args.path_tsp, args.N)
        # Setting parameters
        self.path_dataset = args.path_dataset
        self.num_examples_train = args.num_examples_train
        self.num_examples_test = args.num_examples_test
        self.dual = args.dual
        self.N = args.N
        self.J = args.J
        self.sym = 1 - args.directed
        self.data_train = []
        self.data_test = []
        # Make directory for data
        try:
            os.stat(self.path_dataset)
        except:
            os.makedirs(self.path_dataset)
    ###########################################################################
    ############################ DATASET ASSEMBLING ###########################
    ###########################################################################
    def load_dataset(self, load = True):
        # Create orload training set
        filename = 'TSP{}train_dual_{}.np'.format(self.N, self.dual)
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path) and load:
            print('Reading training dataset at {}'.format(path))
            self.data_train = np.load(open(path, 'rb'))
        else:
            print('Creating training dataset...')
            self.create_dataset_train()
            print('Saving training datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_train)
        # Create or load test set
        filename = 'TSP{}test_dual_{}.np'.format(self.N, self.dual)
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path) and load:
            print('Reading testing dataset at {}'.format(path))
            self.data_test = np.load(open(path, 'rb'))
        else:
            print('Creating testing dataset.')
            self.create_dataset_test()
            print('Saving testing datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_test)
    def create_dataset_train(self):
        for i in range(self.num_examples_train):
            example = self.compute_example()
            self.data_train.append(example)
            if i % 100 == 0:
                print('{} training example of length {} computed.'
                                            .format(i+1, self.N))
    def create_dataset_test(self):
        for i in range(self.num_examples_test):
            example = self.compute_example()
            self.data_test.append(example)
            if i % 100 == 0:
                print('Test example {} of length {} computed.'
                                            .format(i+1, self.N))                                
    def recreate_dataset(self):
        self.load_dataset(load = False)
    ###########################################################################
    ############################# EXAMPLE GENERATION ##########################
    ###########################################################################                        
    def compute_example(self):
        """ 
            Generates an example. 
                NON-DUAL mode: K_N = N points are generated in the unit square 
                               for vertices and an adjaceny matrix W is given
                               inversely proportional to the pairwise distance 
                               of the cities. The graph signal x is given by
                               the city coordinates and the degree of the 
                               cities and so signal_dim = 3.
                              
                DUAL mode: The dual graph of the complete graph on N vertices
                           is used. This has K_N = N(N-1)/2 vertices and W
                           is its adjacency matrix. The graph signal x is given
                           by the pairwise weights between the cities that
                           correspond to a particular vertex in the dual graph
                           and so signal_dim = 1.
                           
            In the directed case the cities have an elevation coordinate and
            climbing hills is more costly. This is only done in the non-dual
            setting.
            
            An example is a dictionary with the following entries:
            
                    'Graph ops'   : K_N x K_N mx-s [I,W,...,W^{2^{J-1}}, D, U]
                    'Cities'      : N city coordinates
                    'Signal'      : (N x signal_dim) Graph signal
                    'TSP cycle'   : TSP cycle given by LKH
                    'Total weight': Total weight of the cycle
                    'WTSP'        : 0-1 Adjacency matrix of TSP cycle
                    'Labels'      : Adjacency list of TSP cycle 
        """
        cities = self.city_coord_generator()
        # Generating adjacency matrix
        if self.dual:
            W = self.dual_adj()
        else:
            W = self.adj_from_coord(cities)
        WW, d = self.compute_operators(W)
        # Generating graph signal
        if self.dual:
            x = self.create_dual_embeddings(cities)
        else:
            x = np.concatenate([d, cities], axis=1)
        # Computing LKH TSP solution
        # Creates files for LKH solver
        self.LKHsolver.setup_problem(cities) 
        # Runs the LKH code to generate GT cycle
        self.LKHsolver.solve()
        TSP_cycle, length = self.LKHsolver.extract_path()
        # Save results
        example = {}
        example['Graph ops'] = WW
        example['Cities'] = cities
        example['Signal'] = x
        example['TSP cycle'] = TSP_cycle
        if self.sym:
            example['Total weight'] = np.sqrt(2)*self.N - length
        else:
            example['Total weight'] = (np.sqrt(3) + 2)*self.N - length
        example['WTSP'] = self.perm_to_adj(TSP_cycle)
        example['Labels'] = self.perm_to_labels(TSP_cycle)
        return example
    def city_coord_generator(self):
        """ Generates N points in the unit square or cube """
        if self.sym:
            cities = np.random.uniform(0, 1, [self.N, 2])
        else:
            cities = np.random.uniform(0, 1, [self.N, 3])
        return cities
    def dual_adj(self):
        """ 
            Returns the hard adjacency matrix of the dual graph of a complete
            graph on N vertices with values 0-1.
        """
        E =  int(self.N*(self.N-1)/2)
        Edges = []
        W = np.zeros((E, E))
        for i in range(0, self.N-1):
            for j in range(i+1,self.N):
                Edges.append([i, j])
        assert len(Edges) == E
        Edges = np.array(Edges)
        for i in range(E):
            W[i] = ((Edges[i,0] == Edges) + (Edges[i,1] == Edges)).sum(1)
        # zero diagonal
        for i in range(E):
                W[i, i] = 0
        return W
    def adj_from_coord(self, cities):
        """ 
            Constructs the adjacency matrix from city coordinates
            based on pairwise distance
        """
        N = cities.shape[0]
        W = np.zeros((N,N))
        for i in range(0, N-1):
            for j in range(i+1,N):
                city1 = cities[i]*self.C
                city2 = cities[j]*self.C
                dist = l2_dist(city1, city2)/float(self.C)
                if self.sym:
                    W[i,j] = np.sqrt(2) - float(dist) # Max dist - dist
                    W[j,i] = W[i,j]
                else:
                    height_diff = (city2[2] - city1[2]) / float(self.C)
                    W[i,j] = np.sqrt(3) + 2 - (float(dist) + height_diff)
                    W[j,i] = W[i,j] + 2 * height_diff
        return W     
    def compute_operators(self, W):
        """ Returns: 
                WW: NxN matrices {Id, W, W^2, ..., W^{2^{J-1}}, D, U}, 
                 d: Nx1 array of degrees for each vertex containing 
                    the degrees of the vertices
        """
        N = W.shape[0]
        d = W.sum(1)
        D = np.diag(d) # Degree operator
        QQ = W.copy()
        WW = np.zeros([N, N, self.J + 2])
        WW[:, :, 0] = np.eye(N)
        for j in range(self.J):
            WW[:, :, j + 1] = QQ.copy()
            QQ = np.dot(QQ, QQ)
            QQ /= QQ.max() # Renormalization to keep spectral radius bounded
            QQ *= np.sqrt(2) # Make largest element of QQ be sqrt(2)
        WW[:, :, self.J] = D
        WW[:, :, self.J + 1] = np.ones((N, N)) * 1.0 / float(N)
        WW = np.reshape(WW, [N, N, self.J + 2])
        d = np.reshape(d, [N, 1])
        return WW, d
    def create_dual_embeddings(self, cities):
        """ Creates array of pairwise weight between cities """
        def l2_dist(x, y):
            return math.ceil(np.sqrt(np.square(x - y).sum()))
        x = []
        for i in range(0, self.N-1):
            for j in range(i+1, self.N):
                city1 = cities[i]*self.C
                city2 = cities[j]*self.C
                dist = l2_dist(city1, city2)/float(self.C)
                dist = np.sqrt(2) - float(dist)
                x.append(dist)
        x = np.reshape(np.array(x), [-1,1])
        return x
    def perm_to_adj(self, perm):
        """ Builds adjacency matrix of TSP cycle """
        W = np.zeros((self.N, self.N))
        perm = list(perm[1:]) + [perm[0]]
        W[perm[0], perm[1]] = 1
        W[perm[0], perm[self.N-1]] = 1
        if self.sym:        
            W[perm[self.N-1], perm[self.N-2]] = 1
            W[perm[self.N-1], perm[0]] = 1
            for i in range(1, self.N-1):
                W[perm[i], perm[i-1]] = 1
                W[perm[i], perm[i+1]] = 1
        else:
            W[perm[self.N-1], perm[0]] = 1
            for i in range(self.N-1):
                W[perm[i],perm[i+1]] = 1
        return W #(N x N)
    def perm_to_labels(self, perm):
        """ 0-1 adjacency list corresponding to cycle """
        if self.sym:
            labels = np.zeros((self.N, 2))
            labels[perm[0], 0] = perm[1]
            labels[perm[0], 1] = perm[self.N-1]
            labels[perm[self.N-1], 0] = perm[0]
            labels[perm[self.N-1], 1] = perm[self.N-2]
            for i in range(1, self.N-1):
                labels[perm[i], 0] = perm[i+1]
                labels[perm[i], 1] = perm[i-1]
        else:
            labels = np.zeros(self.N)
            labels[perm[self.N-1]] = perm[0]
            for i in range(self.N-1):
                labels[perm[i]] = perm[i+1]
        return labels
    ###########################################################################
    ########################## SCRIPT TO GET MINIBATCH ########################
    ###########################################################################
    def sample_batch(self, batch_size, training=True, 
                     it=0, cuda=False, volatile=False):
        """ Creates a minibatch of batch_size examples """
        graph_ops_size = self.data_train[0]['Graph ops'].shape
        signal_size = self.data_train[0]['Signal'].shape
        # Initialize batch elements
        Graph_ops = torch.zeros(batch_size, *graph_ops_size)
        Signals = torch.zeros(batch_size, *signal_size)
        WTSP = torch.zeros(batch_size, self.N, self.N)
        if self.sym:
            Labels = torch.zeros(batch_size, self.N, 2)
            Cities = torch.zeros((batch_size, self.N, 2))
        else:
            Labels = torch.zeros(batch_size, self.N)
            Cities = torch.zeros((batch_size, self.N, 3))
        TSP_cycles = torch.zeros((batch_size, self.N))        
        TSP_cycle_weights = np.zeros(batch_size)        
        # Fill batch elements 
        if training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        for b in range(batch_size):
            if training:
                ind = np.random.randint(0, len(dataset))
            else:
                ind = it * batch_size + b
            graph_ops = torch.from_numpy(dataset[ind]['Graph ops'])
            signal = torch.from_numpy(dataset[ind]['Signal'])
            Graph_ops[b], Signals[b] = graph_ops, signal
            WTSP[b] = torch.from_numpy(dataset[ind]['WTSP'])
            Labels[b] = torch.from_numpy(dataset[ind]['Labels'])
            Cities[b] = torch.from_numpy(dataset[ind]['Cities'])
            TSP_cycles[b] = torch.from_numpy(dataset[ind]['TSP cycle'])
            TSP_cycle_weights[b] = dataset[ind]['Total weight']
        # Wrap as variables
        Graph_ops = Variable(Graph_ops, volatile=volatile)
        Signals = Variable(Signals, volatile=volatile)
        WTSP = Variable(WTSP, volatile=volatile)
        Labels = Variable(Labels, volatile=volatile)
        # Assembling batch and putting it on GPU    
        if cuda:
            batch = [[Graph_ops.cuda(), Signals.cuda()], 
                    [WTSP.cuda(), Labels.cuda()],
                    Cities.cuda(), TSP_cycles, TSP_cycle_weights.cuda()]
        else:
            batch = [[Graph_ops, Signals], [WTSP, Labels], 
                     Cities, TSP_cycles, TSP_cycle_weights]
        if self.dual:
            W = Variable(self.create_adj(batch[2]))
        else:
            W = Variable(batch[0][0][:, :, :, 1])
        # Preparing return variables
        Cities = batch[2]
        Graphs = [Cities, W]
        GNN_inputs = batch[0] # Graph_ops and signals
        Targets = batch[1][0].type(dtype_l), batch[1][1].type(dtype_l) # WTSP and Labels
        TSP_Cycles = [batch[3], batch[4]] #TSP cycles and their weights
        return Graphs, GNN_inputs, Targets, TSP_Cycles
    def create_adj(self, cities):
        """ Creates adjancency matrix out of cities in a batch"""
        cities = cities.cpu().numpy()
        N = cities.shape[1]
        batch_size = cities.shape[0]
        W = np.zeros((batch_size, N, N))
        def l2_dist(x, y):
            return math.ceil(np.sqrt(np.square(x - y).sum()))
        for b in range(batch_size):
            for i in range(0, N - 1):
                for j in range(i + 1, N):
                    city1 = cities[b, i]*self.C
                    city2 = cities[b, j]*self.C
                    dist = l2_dist(city1, city2)/float(self.C)
                    W[b, i, j] = np.sqrt(2) - float(dist)
                    W[b, j, i] = W[b, i, j]
        W = torch.from_numpy(W).type(dtype)
        return W