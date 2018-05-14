import torch
from beam_search import BeamSearch
import torch.nn.functional as F

# Checking if GPU is available
cuda = False
if torch.cuda.is_available() :
    GPU_capab = torch.cuda.get_device_capability(torch.cuda.current_device()) 
    if GPU_capab[0] > 5:
        cuda = True
if cuda:
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

def compute_accuracy(preds, Labels):
    if len(Labels.size()) == 3:
        sym = True
    else:
        sym = False
        Labels = Labels.unsqueeze(2)
    # Highest scoring vertices
    preds = torch.topk(preds, 1 + int(sym), dim=2)[1] 
    # Sort ascending
    p = torch.sort(preds, 2)[0] 
    l = torch.sort(Labels, 2)[0]
    # Error wherever there is an imperfect match at a vertex
    error = 1 - torch.eq(p,l).min(2)[0].type(dtype) # (BS x N)
    average_error_rate = error.mean(1)
    accuracy = 1 - average_error_rate
    # Average over minibatch
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()
def mean_pred_path_weight(preds, W, sym):
    # cost estimator for training time
    preds = F.softmax(preds, dim = 2) # (BS x N x N)
    mean_row_weight = torch.mul(preds, W).mean(2) # (BS x N)
    mean_path_weight= mean_row_weight.sum(1) # (BS)
    mean_path_weight = mean_path_weight.mean(0).squeeze().data.cpu().numpy()
    if sym:
        mean_path_weight /= 2 # Every edge weight is counted twice
    return mean_path_weight    
def beamsearch_hamcycle(preds, W, beam_size=2):
    N = W.size(-1)
    batch_size = W.size(0)
    BS = BeamSearch(beam_size, batch_size, N)
    trans_probs = preds.gather(1, BS.get_current_state())
    for step in range(N-1):
        BS.advance(trans_probs, step + 1)
        trans_probs = preds.gather(1, BS.get_current_state())
    ends = torch.zeros(batch_size, 1).type(dtype_l)
    # extract paths
    Paths = BS.get_hyp(ends)
    # Compute cost of path
    Costs = compute_cost_path(Paths, W)
    return Costs, Paths
def compute_cost_path(Paths, W):
    # Paths is a list of length N+1
    batch_size = W.size(0)
    N = W.size(-1)
    Costs = torch.zeros(batch_size)
    for b in range(batch_size):
        path = Paths[b].squeeze(0)
        Wb = W[b].squeeze(0)
        cost = 0.0
        for node in range(N-1):
            start = path[node]
            end = path[node + 1]
            cost += Wb[start, end]
        cost += Wb[end, 0]
        Costs[b] = cost
    return Costs