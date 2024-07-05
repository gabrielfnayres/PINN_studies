

'''

* Importing libs to load the datasets

'''

import scipy.io

from scipy.interpolate import griddata
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataCollect import  getData

torch.manual_seed(42)

'''

* Now let's collect the data 

* the variables x and t correspond to the equations mentioned
* MSE = MSEu + MSEf

'''
X_star, u_star, lb, ub = getData()

u_star.flatten()[:, None]

'''

* Now let's make the Neural Network

'''



class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

'''

* Now let's implement the Physics Informed part

'''

class PINN():
    '''
    * Initializing all variables and optimizer
    '''
    def __init__(self, X, u, lb, ub, physics):
        self.lb = torch.tensor(lb).float()
        self.ub = torch.tensor(ub).float()
        self.physics = physics

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float()
        self.u = torch.tensor(u).float()

        self.network = Network()

        self.optmizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    '''
    * Concatenate the x and t tensors and than pass it through the network, returning the output of operations
    '''
    def makeNetwork(self, x, t):
        X = torch.cat([x, t], 1)
        return self.network(X)
    
    '''
    * Calculate the u_x, u_t and u_xx to act as a regularizer in the loss function
    * Returning the Burger's equation
    '''

    def residual(self, x, t):
        u = self.makeNetwork(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t + (u*u_x) - ((0.01/np.pi)*u_xx) 


    '''
    * Implementing the training routine
    '''

    def train(self, epochs):
        lossTracker = []
        self.network.train()
        
        for i in range(epochs):
            
            u_pred = self.makeNetwork(self.x, self.t)
            residual_pred = self.residual(self.x, self.t)

            loss_function = torch.mean((self.u - u_pred)**2)
            if self.physics == True:
                loss_function += torch.mean(residual_pred**2)
            lossTracker.append(loss_function.item())

            self.optmizer.zero_grad() # reset the gradient of all optimized tensor
            loss_function.backward()
            self.optmizer.step()

        return lossTracker
            
    '''
    * Implementing predict function
    '''

    def predict(self):
        self.network.eval()
        u = self.makeNetwork(self.x, self.t)
        res = self.residual(self.x, self.t)
        return u.detach().numpy(), res.detach().numpy()
    

idx = np.random.choice(X_star.shape[0], 2000, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star.flatten()[:, None][idx,:]

model = PINN(X_u_train, u_train, lb[0], ub[0], True) # Keep False for Vanilla
pinn = model.train(1000)



predi = model.predict()

import plotly.graph_objects as go

epochs = list(range(len(pinn)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=pinn, mode='lines', name='Physics Informed Neural Network'))
fig.update_layout(
    title='Loss vs. Epochs',
    xaxis=dict(title='Epochs'),
    yaxis=dict(title='Loss'),
    legend=dict(x=0.7, y=1.0),
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode='x unified' 
)
fig.show()
