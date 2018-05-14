import numpy as np
import math
import os
from subprocess import Popen, PIPE, STDOUT
import platform
import matplotlib.pyplot as plt

def l2_dist(x,y):
    return math.ceil(np.sqrt(np.square(x-y).sum()))
class TSPSolver(object):
    """ API to use the LKH heuristics to solve TSP. """
    def __init__(self, path_tsp, id):
        self.C = 10e4 # Big constant to turn coordinates to integers
        self.path_solver = path_tsp # Path for LKH executable
        self.path_datatsp = path_tsp + 'DATA/' # Input/Output for LKH
        self.id = id
        self.cities = None
        try:
            os.stat(self.path_datatsp)
        except:
            os.makedirs(self.path_datatsp)
        if platform.platform() == 'Windows':
            print("TSP solver does not work on Windows.")
            raise SystemExit
        # EXPORTING LKH_PATH (Won't run on windows!)
        cmd = "export LKH_PATH='{}'".format(self.path_solver)
        Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, 
              stderr=STDOUT, close_fds=True)
        cmd = "export PATH=$PATH:$LKH_PATH"
        Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, 
              stderr=STDOUT, close_fds=True)
        self.path_par = os.path.join(self.path_datatsp, 
                                     'pr{}.par'.format(self.id))
        self.path_tsp = os.path.join(self.path_datatsp, 
                                     'pr{}.tsp'.format(self.id))
        self.path_res = os.path.join(self.path_datatsp, 
                                     'res{}.tsp'.format(self.id))
    ############################################################################
    ############################## SETUP AND SOLVER ############################
    ############################################################################
    def setup_problem(self, cities):
        """ 
            Creates problem file for LKH solver. Writes the parameter file
            to prid.par, the problem file to prid.tsp. The LKH output file
            is set to be resid.tsp.
        """
        assert cities.shape[1] >= 2, "Cities need to have >2 coordinates."
        assert cities.shape[1] <= 3, "Too many coordinates for cities."
        sym = False
        if cities.shape[1] == 2:
            sym = True
        # Create header templates
        header_parameters = ('PROBLEM_FILE = {}\nMOVE_TYPE = {:d}\n'
                             'PATCHING_C = {:d}\nPATCHING_A = {:d}\n'
                                  'RUNS = {:d}\nTOUR_FILE = {}\n')
        header_euc2dtsp = ('NAME : {}\nCOMMENT : {}\nTYPE : TSP \n'
                           'DIMENSION : {}\nEDGE_WEIGHT_TYPE : {}\n')
        header_euc3dtsp = ('NAME : {}\nCOMMENT : {}\nTYPE : ATSP \n'
                           'DIMENSION : {}\nEDGE_WEIGHT_TYPE : {}\n'
                           'EDGE_WEIGHT_FORMAT : FULL_MATRIX\n')
        # Create parameter file
        with open(self.path_par, 'w+') as file:
            """ Write problem parameters into a .par file """
            MOVE_TYPE = 5
            PATCHING_C = 3
            PATCHING_A = 2
            RUNS = 10
            TOUR_FILE = self.path_res
            HEADER = [self.path_tsp,
                      MOVE_TYPE,
                      PATCHING_C,
                      PATCHING_A,
                      RUNS,
                      TOUR_FILE]
            HEADER = header_parameters.format(*HEADER)
            file.write(HEADER)
        # Create problem file
        with open(self.path_tsp, 'w+') as file:
            """ 
                Write parameters and city coordinates/asymmetric distances
                into a .tsp file. 
            """
            # Writing the parameters
            if sym:             
                HEADER = ['pr{}.tsp'.format(self.id),
                          '{}-city problem'.format(cities.shape[0]),
                          cities.shape[0], 'CEIL_2D']                                 
                HEADER = header_euc2dtsp.format(*HEADER)
            else:
                HEADER = ['pr{}.tsp'.format(self.id),
                          '{}-city problem'.format(cities.shape[0]),
                          cities.shape[0], 'EXPLICIT']       
                HEADER = header_euc3dtsp.format(*HEADER)
            file.write(HEADER)
            # Writing the data
            cities_int = (cities * self.C).astype(int)
            if sym:
                file.write('NODE_COORD_SECTION \n')
                for i in range(cities.shape[0]):
                    node = ('{:<2} {:<8e} {:<8e} \n'
                            .format(i + 1, *list(cities_int[i])))
                    file.write(node)
            else:
                file.write('EDGE_WEIGHT_SECTION \n')
                for i in range(cities.shape[0]):
                    row = []
                    for j in range(cities.shape[0]):
                        distxy = int(l2_dist(cities_int[i],cities_int[j]))
                        asym_dist = int(distxy+cities_int[j,2]-cities_int[i,2])
                        row.append(asym_dist)
                    row = ' '.join('{:<8d}'.format(edge) for edge in row)
                    file.write(row + '\n')
            file.write('EOF \n')
            self.cities = cities
    def solve(self):
        """ 
            Uses LKH solver using problem file written by save_solverformat(). 
            It reads from prid.par and writes to resid.tsp.
        """
        path_exec = os.path.join(self.path_solver, 'LKH')
        path_example = os.path.join(self.path_datatsp,
                                    'pr{}.par'.format(self.id))
        cmd = "{} {}".format(path_exec, path_example)
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                  close_fds=True)
        return p.stdout.read()
    ############################################################################
    ############################## DIAGNOSTIC METHODS ##########################
    ############################################################################
    def parameters(self):
        """ Prints the content of the LKH solver's parameters. """
        if os.path.exists(self.path_par):
            with open(self.path_par, 'rb') as file:
                for line in file.readlines():
                    print(line[:-1])
        else:
            print("ERROR: Parameter file does not exist.")
    def problem(self):
        """ Prints the contents of the problem file. """
        if os.path.exists(self.path_tsp):
            with open(self.path_tsp, 'rb') as file:
                for line in file.readlines()[:-1]:
                    print(line[:-1])
        else:
            print("ERROR: Problem file does not exist.")
    def dist_matrix(self):
        """ Returns the distance matrix of the problem """
        if self.cities is None:
            print("ERROR: Problem file does not exist.")
            return None
        N = self.cities.shape[0]
        distance_matrix = np.zeros((N,N))
        cities_int = (self.cities * self.C).astype(int)
        for i in range(N):
                for j in range(i+1, N):
                    city_i = cities_int[i]
                    city_j = cities_int[j]
                    distance_matrix[i,j] = l2_dist(city_i, city_j)
                    distance_matrix[j,i] = distance_matrix[i,j]
        if self.cities.shape[1] == 3:
            elev_matrix = np.zeros((N,N))
            for i in range(N):
                for j in range(i+1, N):
                    city_i = cities_int[i]
                    city_j = cities_int[j]
                    elev_matrix[i,j] = city_i[2] - city_j[2]
                    elev_matrix[j,i] = -elev_matrix[i,j]
            distance_matrix += elev_matrix
        return distance_matrix.astype(int).transpose()
    def extract_path(self):
        """ 
            Extracts the TSP solution from the solution file.
            Must have ha file 'resid.tsp' to extract from.
            RETURNS:
                TSP_cycle: The solution cycle as a permutation of vertices
                length_tour: The length of the TSP cycle.
        """
        TSP_cycle = []
        length_tour = 0
        if not os.path.exists(self.path_res):
            print("Solution file not found.")
            return None
        with open(self.path_res, 'rb') as file:
            content = file.readlines()
            tour = content[1]
            length = int(content[4][11:-1])
            length_tour = int(tour[19:-1])
            for node in range(length):
                TSP_cycle.append(int(content[node + 6][:-1]))
        TSP_cycle = np.array(TSP_cycle) - 1
        return TSP_cycle, float(length_tour) / self.C
    def plot_solution(self, save = None):
        """ Plots the TSP solution """
        if not os.path.exists(self.path_res):
            print("Solution file not found.")
        else:
            path, length = self.extract_path()
            cities = self.cities[:,:2]
            path_cities = cities[path]
            path_cities = (np.concatenate((path_cities, 
                           np.expand_dims(path_cities[0], axis=0)), axis=0))
            plt.figure(2, figsize=(12, 12))
            plt.clf()
            plt.scatter(cities[:, 0], cities[:, 1], c='b', s = 100)
            plt.plot(path_cities[:, 0], path_cities[:, 1], c='r')
            plt.title('Total length: {0:.3f}'.format(length), fontsize=20)
            plt.show()
            if save is not None:
                plt.savefig(save, format='eps')
