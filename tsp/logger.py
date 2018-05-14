import os
import torch
import utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})

class Logger(object):
    def __init__(self, path_logger):
        directory = os.path.join(path_logger, 'plots')
        self.path = path_logger
        self.path_dir = directory
        try:
            os.stat(directory)
        except:
            os.makedirs(directory)
        self.loss_train = []
        self.loss_test = []
        self.loss_test_aux = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.accuracy_test_aux = []
        self.ppw_train = []
        self.path_weight_test = []
        self.path_weight_test_oracle = []
        self.path_weight_test_aux = []
        self.path_weight_test_aux_oracle = []
        self.path_examples = None
        self.args = None
        self.example_counter = 0
    def write_settings(self, args):
        self.args = {}
        # write info
        path = os.path.join(self.path, 
                            'experiment{}.txt'.format(self.example_counter))
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')
                self.args[str(arg)] = getattr(args, arg)
                
    def load_model(self, parameters_path):
        path = os.path.join(parameters_path, 'parameters/gnn.pt')
        if os.path.exists(path):
            print('GNN successfully loaded from {}'.format(path))
            return torch.load(path)
        else:
            raise ValueError('Parameter path {} does not exist.'
                                                            .format(path))
    def save_model(self, model):
        save_dir = os.path.join(self.path, 'parameters/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        path = os.path.join(save_dir, 'gnn.pt')
        torch.save(model, path)
        print('Model Saved.')
    def add_train_loss(self, loss):
        self.loss_train.append(loss.data.cpu().numpy())
    def add_train_accuracy(self, preds, Targets, Graphs):
        Labels = Targets[1]
        W = Graphs[1]
        if len(Labels.size()) == 3:
            sym = True
        else:
            sym = False
        accuracy = utils.compute_accuracy(preds, Labels)
        pred_path_weight = utils.mean_pred_path_weight(preds, W, sym)
        self.accuracy_train.append(accuracy)
        self.ppw_train.append(pred_path_weight)
    def add_test_loss(self, loss, last=False):
        self.loss_test_aux.append(loss.data.cpu().numpy())
        if last:
            loss_test = np.array(self.loss_test_aux).mean()
            self.loss_test.append(loss_test)
            self.loss_test_aux = []
    def add_test_accuracy(self, preds, Graphs, Targets, 
                          TSP_cycles, last=False, beam_size=2):
        Cities, W = Graphs
        Labels = Targets[1]        
        Cycles, oracle_path_weights = TSP_cycles
        accuracy = utils.compute_accuracy(preds, Labels)
        # Constructing predicted TSP cycles
        path_weights, Paths = utils.beamsearch_hamcycle(preds.data, W.data,
                                                 beam_size=beam_size)
        self.accuracy_test_aux.append(accuracy)
        self.path_weight_test_aux.append(np.array(path_weights.cpu().numpy()).mean())
        self.path_weight_test_aux_oracle.append(np.array(oracle_path_weights).mean())
        if last:
            accuracy_test = np.array(self.accuracy_test_aux).mean()
            self.accuracy_test.append(accuracy_test)
            self.accuracy_test_aux = []
            path_weight_test = np.array(self.path_weight_test_aux).mean()
            self.path_weight_test.append(path_weight_test)
            self.path_weight_test_aux = []
            path_weight_test_oracle = np.array(self.path_weight_test_aux_oracle).mean()
            self.path_weight_test_oracle.append(path_weight_test_oracle)
            self.path_weight_test_aux_oracle = []
            self.plot_example(Paths, path_weights, oracle_path_weights, Cycles, Cities)
            self.example_counter += 1
    def plot_train_logs(self):
        plt.figure(0, figsize=(20, 20))
        plt.clf()
        # plot loss
        plt.subplot(3, 1, 1)
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss')
        # plot accuracy
        plt.subplot(3, 1, 2)
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        # plot predicted path weight
        plt.subplot(3, 1, 3)
        iters = range(len(self.ppw_train))
        plt.plot(iters, self.ppw_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Average Mean Path Weight')
        plt.title('Average Mean Path Weight')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
        path = os.path.join(self.path_dir, 'training{}.png') 
        plt.savefig(path)
    def plot_test_logs(self):
        plt.figure(1, figsize=(20, 20))
        plt.clf()
        # plot loss
        plt.subplot(3, 1, 1)
        test_freq = self.args['test_freq']
        iters = test_freq * np.arange(len(self.loss_test))
        plt.semilogy(iters, self.loss_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Testing Loss')
        # plot accuracy
        plt.subplot(3, 1, 2)
        iters = test_freq * np.arange(len(self.accuracy_test))
        plt.plot(iters, self.accuracy_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy')
        # plot costs
        plt.subplot(3, 1, 3)
        beam_size = self.args['beam_size']
        iters = range(len(self.path_weight_test))
        plt.plot(iters, self.path_weight_test, 'b')
        print('COST ORACLE', self.path_weight_test_oracle[-1])
        plt.plot(iters, self.path_weight_test_oracle, 'r')
        plt.xlabel('iterations')
        plt.ylabel('Mean path weight')
        plt.title('Mean path weight testing with beam_size : {}'.format(beam_size))
        plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2.0)
        path = os.path.join(self.path_dir, 
                            'testing{}.png'.format(self. example_counter)) 
        plt.savefig(path)
    def plot_example(self, Paths, costs, oracle_costs, Perms,
                     Cities, num_plots=1):
        num_plots = min(num_plots, Paths.size(0))
        num_plots = 1
        for fig in range(num_plots):
            cost = costs[fig]
            oracle_cost = oracle_costs[fig]
            predicted_path = Paths[fig].cpu().numpy()
            oracle_path = Perms[fig].cpu().numpy()
            cities = Cities[fig].cpu().numpy()
            oracle_path = oracle_path.astype(int)
            # print('predicted path: ', predicted_path)
            # print('oracle path: ', oracle_path)
            oracle_cities = cities[oracle_path]
            predicted_cities = cities[predicted_path]
            oracle_cities = (np.concatenate((oracle_cities, np.expand_dims(
                             oracle_cities[0], axis=0)), axis=0))
            predicted_cities = (np.concatenate((predicted_cities, np.
                                expand_dims(predicted_cities[0], axis=0)),
                                axis=0))
            plt.figure(2, figsize=(12, 12))
            plt.clf()
            plt.scatter(cities[:, 0], cities[:, 1], c='b')
            plt.plot(oracle_cities[:, 0], oracle_cities[:, 1], c='r')
            plt.title('Target: {0:.3f}'
                      .format(20*np.sqrt(2)-oracle_cost), fontsize=30)
            path = os.path.join(self.path_dir, 'ground_tsp{}{}.eps'.format(fig, self.example_counter))
            plt.savefig(path, format='eps')

            plt.figure(2, figsize=(12, 12))
            plt.clf()
            plt.scatter(cities[:, 0], cities[:, 1], c='b')
            plt.plot(predicted_cities[:, 0], predicted_cities[:, 1], c='g')
            plt.title('Predicted: {0:.3f}'
                      .format(20*np.sqrt(2) - cost), fontsize=100)
            path = os.path.join(self.path_dir, 'pred_tsp{}{}.eps'.format(fig, self.example_counter))
            plt.savefig(path, format='eps')
