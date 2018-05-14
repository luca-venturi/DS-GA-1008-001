import argparse
import time
import numpy as np

import torch
import torch.nn as nn

from logger import Logger
from data_generator import Generator
from model import GraphNetwork
###############################################################################
############################### SETTINGS ######################################
###############################################################################
description = """The main method for the GNN approach to the
                 travelling salesman problem. Missing command 
                 line arguments are treated as one"""
parser = argparse.ArgumentParser(description = description)
parser.add_argument('--dual', action='store_true',
                    help = """Dual method is used for generation""")
parser.add_argument('--directed', action='store_true',
                    help = """The generated graphs are directed""")
parser.add_argument('--load', action='store_true',
                    help = "Use this flag to load a saved model")
parser.add_argument('--num_examples_train', 
                    nargs='?', const=1, type=int, default=int(20000),
                    help = "Number of training examples (Default: 20000)")
parser.add_argument('--num_examples_test', 
                    nargs='?', const=1, type=int, default=int(1000),
                    help = "Number of test examples (Default: 1000)")
parser.add_argument('--iterations', 
                    nargs='?', const=1, type=int, default=int(20000),
                    help = "Number of iterations (Default:20000)")
parser.add_argument('--batch_size', 
                    nargs='?', const=1, type=int, default=32,
                    help = "Batch size (Default: 32)")
parser.add_argument('--beam_size', 
                    nargs='?', const=1, type=int, default=10)
parser.add_argument('--mode', 
                    nargs='?', const=1, type=str, default='train',
                    help = "train/test")
parser.add_argument('--path_dataset', 
                    nargs='?', const=1, type=str, default='./Dataset/',
                    help = "Path to dataset directory")
parser.add_argument('--path_load', 
                    nargs='?', const=1, type=str, default='./',
                    help = "Path to load model from")
parser.add_argument('--path_logger', 
                    nargs='?', const=1, type=str, default='./logs/',
                    help = "Directory for output")
parser.add_argument('--path_tsp', 
                    nargs='?', const=1, type=str, default='./LKH/',
                    help = "Path for TSP directory")
parser.add_argument('--print_freq', 
                    nargs='?', const=1, type=int, default=100,
                    help = """Frequency of printing to the screen 
                              during training (Default: 100)""")
parser.add_argument('--test_freq', 
                    nargs='?', const=1, type=int, default=500,
                    help = "Frequency of testing during training (Default: 500)")
parser.add_argument('--save_freq', 
                    nargs='?', const=1, type=int, default=2000,
                    help = "Frequency of saving during training (Default: 2000)")
parser.add_argument('--clip_grad_norm', 
                    nargs='?', const=1, type=float, default=100.0,
                    help = "Threshold for gradient clipping (Default: 100)")
parser.add_argument('--num_features', 
                    nargs='?', const=1, type=int, default=20,
                    help = "Number of hidden features (Default: 20)")
parser.add_argument('--num_layers', 
                    nargs='?', const=1, type=int, default=20,
                    help = "Number of layers (Default: 20)")
parser.add_argument('--J', 
                    nargs='?', const=1, type=int, default=4,
                    help = """Higest power of the adjacency matrix 
                              considered is 2 ** J""")
parser.add_argument('--N', 
                    nargs='?', const=1, type=int, default=20,
                    help = """Number of cities (Default: 20)""")
                    
args = parser.parse_args()
batch_size = args.batch_size

# Checking if GPU is available
cuda = False
if torch.cuda.is_available():
    GPU_capab = torch.cuda.get_device_capability(torch.cuda.current_device()) 
    if GPU_capab[0] > 5:
        cuda = True
if cuda:
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

batch_size = args.batch_size

template_train1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template_train2 = ('{:<10} {:<10} {:<10.5} {:<10.5} {:<10.5} {:<10}'
                   '{:<10.3} \n')
template_test1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'
template_test2 = '{:<10} {:<10} {:<10.5} {:<10.5} {:<10.5} {:<10} {:<10.5}'
info_train = ['TRAIN', 'iteration', 'loss', 'accuracy', 'path_weight', 'dual',
              'elapsed(s)']
info_test = ['TEST', 'iteration', 'loss', 'accuracy', 'path_weight',
             'beam_size', 'elapsed(s)']

def train(gnn, logger, gen):
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=1e-3)
    for it in range(args.iterations):
        start = time.time()
        Graphs, GNN_inputs, Targets, TSP_cycles = gen.sample_batch(batch_size, 
                                                                     cuda=cuda)
        preds = gnn(GNN_inputs) #(BS x N x N)
        loss = gnn.compute_loss(preds, Targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy(preds, Targets, Graphs)
        elapsed = time.time() - start
        if it % logger.args['print_freq'] == 0:
                logger.plot_train_logs()
                loss = loss.data.cpu().numpy()
                out = ['---', str(it), str(loss), str(logger.accuracy_train[-1]),
                       str(logger.ppw_train[-1]), str(args.dual), str(elapsed)]
                print(template_train1.format(*info_train))
                print(template_train2.format(*out))
        if it % logger.args['test_freq'] == 0:
            test(gnn, logger, gen)
            logger.plot_test_logs()
        if it % logger.args['save_freq'] == logger.args['save_freq'] - 1:
            logger.save_model(gnn)
    print('Optimization finished.')
    
def test(gnn, logger, gen):
    iterations_test = int(gen.num_examples_test / batch_size)
    for it in range(iterations_test):
        start = time.time()
        Graphs, GNN_inputs, Targets, TSP_cycles = gen.sample_batch(batch_size, 
                                                                   training=False, 
                                                                   it=it, 
                                                                   cuda=cuda)
        preds = gnn(GNN_inputs)
        loss = gnn.compute_loss(preds, Targets)
        last = (it == iterations_test-1)
        logger.add_test_accuracy(preds, Graphs, Targets, TSP_cycles,
                                 last=last, beam_size=args.beam_size)
        logger.add_test_loss(loss, last=last)
        elapsed = time.time() - start
        if not last and it % 100 == 0:
            loss = loss.data.cpu().numpy()
            out = ['---', str(it), str(loss), str(logger.accuracy_test_aux[-1]), 
                   str(logger.path_weight_test_aux[-1]), str(args.beam_size), 
                   str(elapsed)]
            print(template_test1.format(*info_test))
            print(template_test2.format(*out))
    print('TEST PATH WEIGHT: {} | TEST ACCURACY {}\n'
          .format(logger.path_weight_test[-1], logger.accuracy_test[-1]))
          
    oracle = logger.path_weight_test_oracle[-1]
    path_weight = logger.path_weight_test[-1]
    return path_weight / oracle 

if __name__ == '__main__':
    assert args.dual + args.directed < 2, """Dual method is not implemented 
                                             in the directed case"""
    Js = [2,4,6]
    performances = []
    for J in Js:
        # Logger setup
        logger = Logger(args.path_logger)
        logger.write_settings(args)
        # Generator setup
        gen = Generator(args)
        gen.J = J
        # Loading or generating datasets
        gen.load_dataset(load = False)
        # Initializing the model
        # gnn = GraphNetwork(args.num_features, args.num_layers, 
        #                   args.N, args.J, dual=args.dual, sym = gen.sym)
        gnn = GraphNetwork(args.num_features, args.num_layers, 
                       args.N, J, dual=args.dual, sym = gen.sym)
        if cuda:
            gnn.cuda()
        #if args.load:
        #    gnn = logger.load_model(args.path_load)
        logger = Logger(args.path_logger)
        logger.write_settings(args)
            
        train(gnn, logger, gen)
        approx_ratio = test(gnn, logger, gen)
        performances.append(approx_ratio)
        
    result = np.array(list(zip(Js, performances)))
    np.save(open("./result.np", 'wb'), performances)
        
    #if args.mode == 'train':
    #    train(gnn, logger, gen)
    #elif args.mode == 'test':
    #    test(gnn, logger, gen)