# Classification of points (images) lying on one side of the curve - the decision boundary:
#                     
#                                                                                  
#                  (1,3)                (3,3)  
#                       o-----------------o 
#                      /                   \      class 0     
#                     /                     \       
#                    /                       \                 
#                   /                         \                      
# -----------------o  (-1,1)            (5,1)  o-------------------------------------------
#                 
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 
import pdb
print("torch version = "+torch.__version__)

num_of_inputs = 2
# -------------------------
#    ANALYTICAL METHOD
# -------------------------
class Net_ANALY(nn.Module):
    def __init__(self):
        super(Net_ANALY, self).__init__()
        # an affine operation: y = Wx + b
        self.IW = nn.Linear(num_of_inputs,....)  
        self.IW.weight.data = torch.tensor([[.....], [.....]],requires_grad=False)
        self.IW.bias.data = torch.tensor([[....]],requires_grad=False)
        self.LW21 = ....
        self.LW21.weight.data = ....
        self.LW21.bias.data = .....
        #self.LW32 = ....
        #self.LW32.weight.data = ....
        #self.LW32.bias.data = .....


    def forward(self, x):
        #h1 = torch.greater(self.IW(x),0)
        #y = torch.greater(self.LW(h1),0)
        a1 = ..... # weighted sum of inputs
        h1 = torch.greater(a1,0).float()
        a2 = ....
        y = .....
        return y

net_anal = Net_ANALY()
print("ANALYTICAL METHOD:")
print(net_anal)

params = list(net_anal.parameters())
print("ANALY: number of weight parameter tensors = " + str(len(params)))

for i in range(len(params)):
    print(params[i].size())  

# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(-4, 8, 120), np.linspace(-4,8, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Pt = torch.tensor(P).float()
Y = net_anal(Pt).numpy()
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid 
plt.contourf(X1, X2, Z, levels=[0.5, 1.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('analytical method')
plt.show()

# -------------------------
#    LEARNING METHOD
# -------------------------

points = ..........          # maximum 30 points
class_labels = .......
nneu = [4,1]       # numbers of neurons in subsequent layers
num_of_epochs = 5000
num_to_show = 150
learning_rate = 0.05
class Net_LEARN(nn.Module):
    def __init__(self):
        super(Net_LEARN, self).__init__()
        self.IW = nn.Linear(num_of_inputs,nneu[0])
        .....
        
    def forward(self, x):
        .......
        return x

net_learned = Net_LEARN()
print("LEARNING METHOD:")
print(net_learned)

params = list(net_learned.parameters())
print("LEARN: number of weight parameter tensors = " + str(len(params)))

for i in range(len(params)):
    print(params[i].size())  # conv1's .weight

# ...................  can copy from       classification_torch.py

# drawing points:
ind0 = np.where(class_labels == 0)       # indexes of examples which belong to class 0
ind1 = np.where(class_labels == 1)
line1 = plt.plot(np.transpose(points[ind0[0],0]), np.transpose(points[ind0[0],1]), 'ro', label = 'class 0')
line2 = plt.plot(np.transpose(points[ind1[0],0]), np.transpose(points[ind1[0],1]), 'bs', label = 'class 1')
plt.title("points in 2d plane - decision boundary") 
plt.xlabel("x1") 
plt.ylabel("x2") 
plt.legend()

# drawing decision boundary:
X1, X2 = np.meshgrid(np.linspace(0, 8, 120), np.linspace(0,8, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Pt = torch.tensor(P).float()
Y = net_learned(Pt).detach().numpy()
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid 
plt.contourf(X1, X2, Z, levels=[0.5, 0.75, 1.0, 2.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('learning method')
plt.show()

