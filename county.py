"""
Azad Amitoz 2021-11-16 15:24
- like main2.py but for pubmed
"""


import random
import argparse

import wandb
import torch
import torch_geometric
from torch_scatter import scatter
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import Data, DataLoader
from torchdiffeq import odeint_adjoint as odeint
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid,  BatchNorm1d as BN, Dropout, Tanh, Softplus
from torch_geometric.nn.dense.linear import Linear

#from utilities import *
from new_utils import *


# add more parser if possible.
parser = argparse.ArgumentParser()
parser.add_argument("--rs", type=int, default=0, help="random_seed")
parser.add_argument("--alpha1", type=float, default=0.0804269831670015, help="alpha in dropout")
parser.add_argument("--alpha2", type=float, default=0.400899124203731, help="alpha in dropout")
parser.add_argument("--lr", type=float, default=0.0268750312205146)
parser.add_argument("--rtol", type=float, default=0.00666082730145346, help="rtol in diffeq")
parser.add_argument("--wd", type=float, default=0.00525291464868861)
parser.add_argument("--masks", type=int, nargs="+", default=None)
parser.add_argument("--dataset", type=str, default="")
#parser.add_argument("--itr", type=int, default=100)
parser.add_argument("--itr", type=int, default=150)
#parser.add_argument("--time", type=float, default=5.1)
#parser.add_argument("--time", type=float, default=1.7) 82.96
#parser.add_argument("--time", type=float, default=1.9) 83.6
#parser.add_argument("--time", type=float, default=2.1) 83.44
#parser.add_argument("--time", type=float, default=2.3)  83.76
#parser.add_argument("--time", type=float, default=2.5) 83.76 
#parser.add_argument("--time", type=float, default=2.7) 84.24
parser.add_argument("--time", type=float, default=2.9) #84.57
#parser.add_argument("--time", type=float, default=3.1)  83.92
#parser.add_argument("--time", type=float, default=3.3) 82.96
#parser.add_argument("--time", type=float, default=3.5) 81.99
#parser.add_argument("--time", type=float, default=3.7)
parser.add_argument("--lamb", type=float, default=1.0)
parser.add_argument("--is_sweep", type=bool, default=False)
args = parser.parse_args()


# seed everything
random.seed(args.rs)
np.random.seed(args.rs)
torch.manual_seed(args.rs)
torch.cuda.manual_seed(args.rs)
torch.autograd.set_detect_anomaly(True)
dev = "cuda:0"


#get graph
#graph = torch_geometric.datasets.LastFMAsia(root='./')[0]
graph = torch.load("../../us_county/graph_us_county.pt")
graph.edge_index = graph.edge_index[torch.LongTensor([1,0])]#Check this thing?
graph.seed_mask, graph.train_mask, graph.val_mask, graph.test_mask = gen_mask_p(graph.y, 0, 40, 40, (graph.y.max()+1).item())
#graph.x = torch.where(graph.x !=0, 1, 0).type(torch.float32)

print(graph.x)
print(graph.x.shape)

# graph.seed_mask, graph.train_mask, graph.val_mask, graph.test_mask = gen_mask(graph.y, args.masks[0], 
#                 args.masks[1], args.masks[2], (graph.y.max()+1).item())

if not is_undirected(graph.edge_index):
    graph.edge_index = to_undirected(graph.edge_index)
    print("INFO:-> making symmetric.")

graph.perm = (graph.edge_index[0]*len(graph.x) + graph.edge_index[1]).argsort()
print("Perm vector")

graph.edge_index, _  = add_self_loops(graph.edge_index, num_nodes=len(graph.x))
#graph.self_loops_added = True
print("Adding self loops.")

graph = graph.to(dev)


# wandb
if args.is_sweep:
    print(f"The sweep is {args.is_sweep}")
    wandb.init(config=args)
else:
    wandb.init(dir="/tmp/wandb", config=args)
    print(f"The sweep is {args.is_sweep}")


# class pde():
class Laplacian(torch.nn.Module):
    """
    Evolves the signal according to the laplacian.
    """
    def __init__(self, mask, lamb):
        super(Laplacian, self).__init__()
        self.run_pde = None
        self.mask = mask
        #self.distances = Lin(1, graph.edge_index.shape[1]-len(graph.x), bias=False).to(dev) 
        #self.distances = Linear(1, graph.edge_index.shape[1]-len(graph.x), bias=False,weight_initializer='glorot').to(dev) 
        self.distances = Linear(1, graph.edge_index.shape[1]-len(graph.x), bias=False,weight_initializer='kaiming_uniform').to(dev) 
        #self.distances = None
        self.rel = Sigmoid() #try different !
        self.lamb = lamb

    def forward(self, t, y):
        distances = self.distances.weight
        dist_fixed =  torch.ones(len(graph.x),1,requires_grad=False).to("cuda:0")
        # if you want the wgts to be symmetric
        distancesT = distances[graph.perm]
        distancesF = (distancesT + distances)/2
        distancesF = self.rel(distancesF)
        #distancesF = torch.exp(-(distancesF ** 2)) #try it
        distancesF = torch.cat([distancesF, dist_fixed],dim=0)

        deg = scatter(distancesF,graph.edge_index[1], dim=0, dim_size=y.shape[1],reduce="add")
        #laplacian = (scatter(distancesF.view(-1,)*((y[:,graph.edge_index[0]]) - 
        #         (y[:,graph.edge_index[1]])), graph.edge_index[1],dim=1, dim_size=y.shape[1],reduce="add"))/deg.T
        laplacian = (scatter(distancesF.view(-1,)*((y[:,graph.edge_index[0]]/torch.sqrt(deg.T[:,graph.edge_index[1]] * deg.T[:,graph.edge_index[0]]))-
                (y[:,graph.edge_index[1]]/deg.T[:,graph.edge_index[1]])), graph.edge_index[1],dim=1, dim_size=y.shape[1],reduce="add"))
 
        if not self.run_pde:
            return self.lamb * laplacian
        else:
            #return self.lamb * laplacian
            return self.lamb * laplacian * self.mask

#
## class level_set():
class LaplacianBlock(torch.nn.Module):
    """
    Evolves the level_set on graphs.
    """

    def __init__(self, odefunc,t, **kwargs):
        super(LaplacianBlock, self).__init__()
        self.odefunc = odefunc
        self.t = t
        self.rtol = kwargs["rtol"]

    def set_distances(self, distances):
        self.odefunc.distances = distances

    def run_pde(self):
        self.odefunc.run_pde = True

    def donot_run_pde(self):
        self.odefunc.run_pde = False

    def forward(self, x):
        z = odeint(self.odefunc,x,self.t,rtol=self.rtol, atol=self.rtol/10).to(dev)
        #z = odeint(self.odefunc,x,self.t,method="rk4",options={"step_size":0.1}).to(dev) # try 
        return  z[1]


# define GNN, NN model.
class Net(torch.nn.Module):

    def __init__(self, mask, front_initial, time, **kwargs):
        super(Net, self).__init__()
        self.run_pde = False
        self.fzero = None
        #self.m1 = Lin(kwargs["inp"],256)
        #self.m1 = Lin(kwargs["inp"],512)
        #self.m1 = Linear(kwargs["inp"],512,weight_initializer='glorot')
        self.m1 = Linear(kwargs["inp"],16,weight_initializer='kaiming_uniform')
        #self.m1 = Linear(kwargs["inp"],16,weight_initializer='glorot')
        #self.m2 = Lin(512,128)#64
        #self.m2 = Linear(512,128,weight_initializer='glorot')#64
        #self.m2 = Linear(128,16,weight_initializer='kaiming_uniform')#64
        #self.m3 = Lin(128,kwargs["out"])
        #self.m3 = Linear(128,kwargs["out"], weight_initializer='glorot')
        self.m2 = Linear(16,kwargs["out"], weight_initializer='kaiming_uniform')
        #self.m2 = Linear(16,kwargs["out"], weight_initializer='glorot')
        self.dropout1 = torch.nn.Dropout(kwargs["alpha1"])
        #self.dropout2 = torch.nn.Dropout(kwargs["alpha2"])
        self.rel = ReLU()
        self.front_initial = front_initial
        self.mask = mask
        #self.softmax = torch.nn.Softmax(dim=1)
        self.sig = Sigmoid()
        self.laplaceblock = LaplacianBlock(Laplacian(mask, kwargs["lamb"]), t=time, **dict(rtol=kwargs["rtol"]))

    def forward(self,x):
        x = self.m1(x)
        x = self.dropout1(x)
        x = self.rel(x)
        x = self.m2(x)
        #x = self.dropout2(x)
        #x = self.rel(x)
        #x = self.m3(x)
        if not self.run_pde:
            self.laplaceblock.donot_run_pde()
            z = self.laplaceblock(x.T)
            return z
        else:
            #self.fzero = (x.T * self.mask + torch.where(self.front_initial == 1, torch.max(x), torch.min(x))).T
            self.fzero = (x.T * self.mask + self.front_initial).T
            #self.fzero = x
            self.laplaceblock.run_pde()
            z = self.laplaceblock(self.fzero.T)
            return z


t = (torch.linspace(0,args.time,2)).to(dev)


#front_initial = None
front_initial = get_front(graph.y, graph.train_mask)
print(f"The front_initial is {front_initial} and shape is {front_initial.shape}")
#maskF = None
maskF = torch.where(torch.sum(front_initial.T,dim=1)==1, False, True)
print(f"MaskF is {maskF} and shape is {maskF.shape}")


model = Net(maskF, front_initial, t, **dict(lamb=args.lamb, inp=graph.x.shape[1],out=(graph.y.max()+1).item(),rtol=args.rtol, alpha1=args.alpha1, alpha2=args.alpha2)).to(dev)
#opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) #try
opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd) #try
Loss = torch.nn.CrossEntropyLoss() # try


"""
- why I don't have a test method?
- recall how the gcn code has a test function after every epoch?
- why no such thing in my code. 
- See in the early stopping do they look at the loss or accuracy.
- chekc His code to see how it is done.
"""
def train(itr):
    model.train()
    model.run_pde = False
    out = model(graph.x) # Adaptive Runge-kutta here
    loss =  Loss(out.T[graph.train_mask], graph.y[graph.train_mask]) 
    print(loss.data)
    wandb.log({"Iteration":itr, "CrossEntropyLoss":loss.item()})

    opt.zero_grad()
    loss.backward()
    opt.step()


@torch.no_grad()
def test(itr): # Double check this code.
    model.eval()
    model.run_pde = True
    out = model(graph.x) # Adaptive Runge-kutta here

    accs = []
    a = torch.argmax(out.T, dim=1)
    nmask_train = (a[graph.train_mask] == graph.y[graph.train_mask] )
    train_acc = (torch.sum(nmask_train))/len(graph.y[graph.train_mask])
    accs.append(train_acc)
    wandb.log({"Iteration":itr, "TrainAccuracy":train_acc.item()})

    nmask_val = (a[graph.val_mask] == graph.y[graph.val_mask] )
    val_acc = (torch.sum(nmask_val))/len(graph.y[graph.val_mask])
    accs.append(val_acc)
    wandb.log({"Iteration":itr, "ValAccuracy":val_acc.item()})

    nmask_test = (a[graph.test_mask] == graph.y[graph.test_mask] )
    test_acc = (torch.sum(nmask_test))/len(graph.y[graph.test_mask])
    accs.append(test_acc)
    wandb.log({"Iteration":itr, "TestAccuracy":test_acc.item()})

    loss =  Loss(out.T[graph.val_mask], graph.y[graph.val_mask]) 
    accs.append(loss.item())
    return accs


early_stopping_counter = 0
best_val_acc = test_acc = 0
best_val_loss = np.inf
for itr in range(1, args.itr):
    train(itr)
    train_acc, val_acc, tmp_test_acc, val_loss = test(itr)
    #if (val_acc > best_val_acc) and (val_loss < best_val_loss):
    if (val_acc > best_val_acc):
    #if (val_loss < best_val_loss):
        best_val_acc = val_acc
        best_val_loss = val_loss
        test_acc = tmp_test_acc
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    log = 'itr: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}' 
    print(log.format(itr, train_acc.item(), best_val_acc.item(), test_acc.item()))
    wandb.log({"Iteration":itr, "BestValAccuracy":best_val_acc.item(), "BestTestAccuracy":test_acc.item()})
    if early_stopping_counter > 1000:
        break


wandb.finish()
