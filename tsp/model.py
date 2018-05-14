import torch
import torch.nn as nn
import torch.nn.functional as F
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
###############################################################################
########################## AUXILIARY FUNCTIONS ################################
###############################################################################
def gmul(input):
    """ Returns tensor of shape (BS x N x J * num_features), where for each
        example in the batch, the concatenated array
            [X, WX, W^2X,..., DX, UX)]  of size N x (J + 2) * dim_input
        is computed and returned"""
    WW, X = input
    # X is a tensor of size (BS, N, dim_input)
    # WW is a tensor of size (BS, N, N, J + 2)
    WW_size = WW.size()
    N = WW_size[-2]
    # Creates a tuple (BS x N x N x 1, ... , BS x N x N x 1) of length J + 2
    WW = WW.split(1, 3) 
    WW = torch.cat(WW, 1).squeeze(3) # W is now of size (BS, (J+2)*N, N)
    # Compute {X, WX, W^2X,...,W^{J-1}X, DX, UX}
    output = torch.bmm(WW, X) # output has size (BS, (J + 2)*N, dim_input)
    # Creates a tuple (BS x N x num_features, ... , BS x N x dim_input)
    output = output.split(N, 1) 
    output = torch.cat(output, 2) # output has size (BS x N x (J + 2)*dim_input)
    return output
def normalize_embeddings(emb):
    """ L2 normalizes the feature dimension. Emb is (BS x N x num_features) """
    norm = torch.mul(emb, emb).sum(2).unsqueeze(2).sqrt().expand_as(emb)
    return emb.div(norm)    
###############################################################################
############################# NETWORK LAYERS ##################################
###############################################################################        
class GraphNetwork(nn.Module):
    def __init__(self,num_features,num_layers,N,J,dual=False, sym = True):
        """ 
            num_layers: Number of GNN layers
            num_features: Dimension of intermediate GNN layers
        """
        super(GraphNetwork, self).__init__()
        self.N = N
        self.dual = dual
        self.sym = sym
        # Dimension of input graph signal
        if self.dual:
            dim_input = 1
            self.linear_dual = nn.Linear(num_features, 1)
        elif sym:
            dim_input = 3
        else:
            dim_input = 4
        self.gnn = GNN_layers(num_features, num_layers, J, dim_input=dim_input)
        self.CEL = nn.CrossEntropyLoss()
    def compute_loss(self, preds, Targets):
        """ Computes cross entropy loss """
        loss = 0.0
        preds = preds.view(-1, preds.size()[-1]) #(BS*N x N)
        Labels = Targets[1]
        if not self.sym:
            Labels = Labels.unsqueeze(2)
        for i in range(Labels.size()[-1]):
            lab = Labels[:, :, i].contiguous().view(-1) #(BS*N x 1)
            loss += self.CEL(preds, lab)
        return loss
    def forward(self, GNN_input):
        """ Takes a minibatch of inputs of the form GNN_input = [WW, X] 
            such that
                WW  : (BS x K_N x K_N x J+2) of Graph Operators
                            {I, W, ... , W ** (2 ** (J-1)), D, U} 
                 X  : (BS x K_N x dim_input) Graph Signal
            
             OUTPUT : (BS x N x N) prob of being on the cycle for each edge.
        """
        out = self.gnn(GNN_input) # (BS x K_N x num_features)
        if self.dual:
            out_size = out.size()
            # Fully connected layer to reduce to one output feature
            out = out.view(-1, out_size[-1]) # (BS*N(N-1)/2 x num_features)
            fc_out = self.linear_dual(out) # (BS*N(N-1)/2 x 1)
            fc_out = fc_out.view(*out_size[:2],1) # (BS x N(N-1)/2 x 1)
            
            # Preparing the output: A number on every edge
            batch_size = fc_out.size()[0]
            out = Variable(torch.zeros(batch_size, self.N, self.N)).type(dtype)
            for b in range(batch_size):
                count = 0
                for i in range(0, self.N-1):
                    for j in range(i+1, self.N):
                        out[b, i, j] = fc_out[b, count]
                        out[b, j, i] = out[b,i,j]
                        count += 1
        else:
            out = normalize_embeddings(out) #L2 normalize the feature dimension
            out = torch.bmm(out, out.permute(0,2,1)) # (BS x N x N)
            diag = Variable(torch.eye(self.N).unsqueeze(0).expand_as(out))
            diag = (-1000 * diag.type(dtype))
            out = out + diag
        return out # (BS x N x N)
class GNN_layers(nn.Module):
    def __init__(self, num_features, num_layers, J, dim_input=1):
        super(GNN_layers, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [dim_input, num_features]
        self.featuremap_mi = [num_features, num_features]
        self.featuremap_end = [num_features, num_features]
        self.layer0 = Gconv(self.featuremap_in, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last(self.featuremap_end, J)

    def forward(self, GNN_input):
        """ Takes a minibatch of inputs of the form GNN_input = [WW, X] 
            such that
                WW  : (BS x K_N x K_N x J+2) of Graph Operators
                            {I, W, ... , W ** (2 ** (J-1)), D, U} 
                 X  : (BS x K_N x dim_input) Graph Signal
                 
            The architecture is given by
            
           X0      X1       ...       Xnum_layers-1          Xnum_layer -- out
            \     /  \     /   \     /             \        / 
             GConv    GConv     GConv              GConvLast
            /     \  /     \   /     \             /        \
          WW       WW       ...       WW-----------          WW (discarded)
           
        """
        cur = self.layer0(GNN_input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out[1] # (BS x K_N x num_features)

    
class Gconv(nn.Module):
    """ Implements the GNN layer given by (3) in the paper """
    def __init__(self, feature_maps, J):
        super(Gconv, self).__init__()
        self.num_inputs = (J + 2) * feature_maps[0] 
        self.num_outputs = feature_maps[1] 
        # Note that the total output dimension is num_outputs        
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, (self.num_outputs + 1) // 2)
        self.bn = nn.BatchNorm1d(self.num_outputs)
        self.bn_instance = nn.InstanceNorm1d(self.num_outputs)
        
    def forward(self, GNN_input):
        """ Takes a minibatch of inputs of the form GNN_input = [WW, X] 
            such that
                WW  : (BS x K_N x K_N x J+2) of Graph Operators
                            {I, W, ... , W ** (2 ** (J-1)), D, U} 
                 X  : (BS x K_N x dim_input) Graph Signal
            
            The architecture is given by
            
           X          FC -- ReLU---
            \        /            |
             GraphMul       Concatenate -- BatchNorm1D -- InstNorm1D -- out2
            /        \            |
          WW          FC ----------
           |
           ------------------------------------------------------------ out1
        """
        WW = GNN_input[0]
        WWX = gmul(GNN_input).contiguous() # (BS x N x num_inputs)
        WWX_size = WWX.size()
        WWX = WWX.view(-1, self.num_inputs) # (BS * N, num_inputs)
        WWX1 = F.relu(self.fc1(WWX)) 
        WWX2 = self.fc2(WWX)
        out = torch.cat((WWX1, WWX2), 1) # (BS * N x num_outputs)
        out = self.bn(out)
        out = out.view(*WWX_size[:-1], self.num_outputs) 
        out = self.bn_instance(out) #(BS x N x num_outputs)
        return WW, out
        
class Gconv_last(nn.Module):
    """ Similar to previous but without relu and normalizations. """
    
    def __init__(self, feature_maps, J):
        super(Gconv_last, self).__init__()
        self.num_inputs = (J+2)*feature_maps[0]
        self.num_outputs = feature_maps[1]
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, GNN_input):
        WW = GNN_input[0]
        WWX = gmul(GNN_input).contiguous() # (BS x N x num_inputs)
        WWX_size = WWX.size()
        WWX = WWX.view(WWX_size[0]*WWX_size[1], -1) # (BS*N x num_inputs)
        out = self.fc(WWX) # (BS*N x num_outputs)
        out = out.view(*WWX_size[:-1], self.num_outputs)
        return WW, out