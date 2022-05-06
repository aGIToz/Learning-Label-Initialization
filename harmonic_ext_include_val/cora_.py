"""
Amitoz AZAD 2022-05-06 14:36
"""


import random
import argparse

import torch
import torch_geometric
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import is_undirected, to_undirected
from torchdiffeq import odeint_adjoint as odeint
from torch.nn import ReLU, Sigmoid, Dropout, Tanh
from torch_geometric.nn.dense.linear import Linear

from new_utils import *


# add more parser if possible.
parser = argparse.ArgumentParser()
parser.add_argument("--rs", type=int, default=70, help="random_seed")
parser.add_argument("--alpha1", type=float, default=0.974861496100242, help="alpha in dropout")
parser.add_argument("--alpha2", type=float, default=0.171243641522226, help="alpha in dropout")
parser.add_argument("--lr", type=float, default=0.00638084516757432)
parser.add_argument("--rtol", type=float, default=0.00264571895257663, help="rtol in diffeq")
parser.add_argument("--wd", type=float, default=0.00198638962964938)
parser.add_argument("--itr", type=int, default=30)
parser.add_argument("--time", type=float, default=5.0)
parser.add_argument("--lamb", type=float, default=1.0)
args = parser.parse_args()


# seed everything
random.seed(args.rs)
np.random.seed(args.rs)
torch.manual_seed(args.rs)
torch.cuda.manual_seed(args.rs)
dev = "cuda:0"


#get graph
graph = torch_geometric.datasets.Planetoid(root='../', name='Cora')[0]
graph.edge_index = graph.edge_index[torch.LongTensor([1,0])]
if not is_undirected(graph.edge_index):
    graph.edge_index = to_undirected(graph.edge_index)
graph.perm = (graph.edge_index[0]*len(graph.x) + graph.edge_index[1]).argsort()
graph.edge_index, _  = add_self_loops(graph.edge_index, num_nodes=len(graph.x))
graph = graph.to(dev)


# PDE 
class Laplacian(torch.nn.Module):
    def __init__(self, mask, lamb):
        super(Laplacian, self).__init__()
        self.run_pde = None
        self.mask = mask
        self.distances = Linear(1, graph.edge_index.shape[1]-len(graph.x), bias=False,weight_initializer='kaiming_uniform').to(dev) 
        self.rel = Sigmoid()
        self.lamb = lamb

    def forward(self, t, y):
        distances = self.distances.weight
        dist_fixed =  torch.ones(len(graph.x),1,requires_grad=False).to("cuda:0")
        distancesT = distances[graph.perm]
        distancesF = (distancesT + distances)/2
        distancesF = self.rel(distancesF)
        distancesF = torch.cat([distancesF, dist_fixed],dim=0)

        deg = scatter(distancesF,graph.edge_index[1], dim=0, dim_size=y.shape[1],reduce="add")
        laplacian = (scatter(distancesF.view(-1,)*((y[:,graph.edge_index[0]]/torch.sqrt(deg.T[:,graph.edge_index[1]] * deg.T[:,graph.edge_index[0]]))-(y[:,graph.edge_index[1]]/deg.T[:,graph.edge_index[1]])), graph.edge_index[1],dim=1, dim_size=y.shape[1],reduce="add"))
 
        if not self.run_pde:
            return self.lamb * laplacian
        else:
            return self.lamb * laplacian * self.mask

class LaplacianBlock(torch.nn.Module):
    def __init__(self, odefunc,t, **kwargs):
        super(LaplacianBlock, self).__init__()
        self.odefunc = odefunc
        self.t = t
        self.rtol = kwargs["rtol"]

    def run_pde(self):
        self.odefunc.run_pde = True

    def donot_run_pde(self):
        self.odefunc.run_pde = False

    def forward(self, x):
        z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10).to(dev)
        return  z[1]

class Net(torch.nn.Module):
    def __init__(self, mask, front_initial, time, **kwargs):
        super(Net, self).__init__()
        self.run_pde = False
        self.fzero = None
        self.m1 = Linear(kwargs["inp"],512,weight_initializer='kaiming_uniform')
        self.m2 = Linear(512,128,weight_initializer='kaiming_uniform')
        self.m3 = Linear(128,kwargs["out"], weight_initializer='kaiming_uniform')
        self.dropout1 = torch.nn.Dropout(kwargs["alpha1"])
        self.dropout2 = torch.nn.Dropout(kwargs["alpha2"])
        self.rel = ReLU()
        self.front_initial = front_initial
        self.mask = mask
        self.sig = Sigmoid()
        self.laplaceblock = LaplacianBlock(Laplacian(mask, kwargs["lamb"]), t=time, **dict(rtol=kwargs["rtol"]))

    def forward(self,x):
        x = self.m1(x)
        x = self.dropout1(x)
        x = self.rel(x)
        x = self.m2(x)
        x = self.dropout2(x)
        x = self.rel(x)
        x = self.m3(x)
        if not self.run_pde:
            self.laplaceblock.donot_run_pde()
            z = self.laplaceblock(x.T)
            return x, z
        else:
            self.fzero = (x.T * self.mask + self.front_initial).T
            self.laplaceblock.run_pde()
            z = self.laplaceblock(self.fzero.T)
            return x, z


t = (torch.linspace(0,args.time,2)).to(dev)
front_initial = get_front(graph.y, graph.train_mask) + get_front(graph.y, graph.val_mask)
maskF = torch.where(torch.sum(front_initial.T,dim=1)==1, False, True)
model = Net(maskF, front_initial, t, **dict(lamb=args.lamb, inp=graph.x.shape[1],out=(graph.y.max()+1).item(),rtol=args.rtol, alpha1=args.alpha1, alpha2=args.alpha2)).to(dev)
model.load_state_dict(torch.load("cora.pt"))


@torch.no_grad()
def test(itr=0):
    model.eval()
    model.run_pde = True
    out_front, out = model(graph.x)

    accs = []
    a = torch.argmax(out.T, dim=1)
    nmask_train = (a[graph.train_mask] == graph.y[graph.train_mask] )
    train_acc = (torch.sum(nmask_train))/len(graph.y[graph.train_mask])
    accs.append(train_acc)

    nmask_val = (a[graph.val_mask] == graph.y[graph.val_mask] )
    val_acc = (torch.sum(nmask_val))/len(graph.y[graph.val_mask])
    accs.append(val_acc)

    nmask_test = (a[graph.test_mask] == graph.y[graph.test_mask] )
    test_acc = (torch.sum(nmask_test))/len(graph.y[graph.test_mask])
    accs.append(test_acc)
    return accs


# Evaluate
train_acc, val_acc, tmp_test_acc = test()
log = 'Train: {:.4f}, Val: {:.4f}, Test: {:.4f}' 
print(log.format(train_acc.item(), val_acc.item(), tmp_test_acc.item()))
print(tmp_test_acc.item(), file=open("cora.txt","a"))
