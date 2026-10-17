# Klasyfikacja punktów należących do prostokąta metodą analityczną i metodą uczenia
# w pytorch-u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import pdb
print("torch version = "+torch.__version__)

num_of_inputs = 2      
nneu = [6,3,1]           # numbers of neurons in subsequent layers
num_of_epochs = 1000
num_to_show = 150
learning_rate = 0.05
use_optimizer = True

# training data:
#points = np.array([[1,5],[3,5],[1,3],[2,3],[3,1],[5,3],[7,1],[3,3],[2,1],[5,5]], dtype=float)
#class_labels = np.transpose(np.array([[1,1,1,1,0,0,0,0,1,1]], dtype = float))
points = np.array([[0,6],[0,7],[2,2],[4,3],[4,4],[9,6],[9,7],[1,4],[2,5],[5,1],[6,7],[7,4],[7,6],[8,3]], dtype=float)
class_labels = np.transpose(np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,1]], dtype = int))

num_of_examples, num_of_features = points.shape
range_min = np.amin(points,axis=0)       # for plotting 
range_max = np.amax(points,axis=0)
print ("number of examples = "+str(num_of_examples)+ ", number of features = "+str(num_of_features) )

ind0 = np.where(class_labels == 0)       # indexes of examples which belong to class 0
ind1 = np.where(class_labels == 1)

line1 = plt.plot(np.transpose(points[ind0[0],0]), np.transpose(points[ind0[0],1]), 'ro', label = 'class 0')
line2 = plt.plot(np.transpose(points[ind1[0],0]), np.transpose(points[ind1[0],1]), 'bs', label = 'class 1')
plt.title("points in 2d plane (close to continue ...)") 
plt.xlabel("x1") 
plt.ylabel("x2") 
plt.legend()
plt.show()

class Net_LEARN(nn.Module):
    def __init__(self):
        super(Net_LEARN, self).__init__()
        self.IW = nn.Linear(num_of_inputs,nneu[0])
        self.IW.requires_grad_(True)
        self.LW21 = nn.Linear(nneu[0],nneu[1])
        self.LW21.requires_grad_()
        if  len(nneu) == 3:
            self.LW32 = nn.Linear(nneu[1],nneu[2])
            self.LW32.requires_grad_()
        
    def forward(self, x):
        x = F.tanh(self.IW(x))
        x = self.LW21(x) 
        if len(nneu) == 3:
            x = self.LW32(x) 
        return x

net_learned = Net_LEARN()
print("LEARNING METHOD:")
print(net_learned)

params = list(net_learned.parameters())
print("LEARN: number of weight parameter tensors = " + str(len(params)))

for i in range(len(params)):
    print(params[i].size())  # conv1's .weight

criterion = nn.MSELoss()
criterion.requires_grad_()

#optimizer = optim.SGD(net_learned.parameters(), lr=learning_rate, weight_decay=0.01)
#optimizer = optim.RMSprop(net_learned.parameters(), lr=learning_rate, momentum=0)
optimizer = optim.Adam(net_learned.parameters(),lr=learning_rate,weight_decay=0)
#optimizer = optim.Adagrad(net_learned.parameters(),lr=learning_rate)

optimizer.zero_grad()   # zero the gradient buffers

#net_learned.zero_grad()     # zeroes the gradient buffers of all parameters

for epoch in range(num_of_epochs):
    output = net_learned(torch.tensor(points).float())
    #print(output)
    loss = criterion(output, torch.tensor(class_labels).float())
    accuracy = sum((output > 0.5).numpy() == class_labels)/len(class_labels)
    
    #print("IW before grad calc.:")
    #print(net_learned.IW.weight.grad)

    loss.backward()

    #print("IW after grad calc.:")
    #print(net_learned.IW.weight.grad)

    if use_optimizer:
        optimizer.step()    # Does the update
        optimizer.zero_grad()   # zero the gradient buffers
    else:
        for param in net_learned.parameters():
            #print("param size = "+str(param.size()))
            #print("param.grad = " + str(param.grad))
            param.data -=  param.grad*learning_rate
        net_learned.zero_grad()

    if epoch % num_to_show == 0:
        print("epoch "+str(epoch) + ", loss = "+str(float(loss))+" accuracy = "+str(accuracy))
    

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
X1, X2 = np.meshgrid(np.linspace(0, 12, 120), np.linspace(0,10, 120))  # grid of points in 2D plane
P = np.stack((X1.flatten(),X2.flatten()), axis=1)                    # points formated for ANN input
Pt = torch.tensor(P).float()
Y = net_learned(Pt).detach().numpy()
Z = np.reshape(Y,X1.shape)                                           # reshaping to shape of grid 
plt.contourf(X1, X2, Z, levels=[0.5, 0.75, 1.0, 2.0])                           # curve for level=0.5 - a decision boundary, shaded class 1 area
plt.title('learning method')
plt.show()

