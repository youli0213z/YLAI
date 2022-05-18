import torch
import numpy as np
import matplotlib.pyplot as plt

def Sigmoid_(x, device='cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x, device=device)
    return 1 / (1 + torch.exp(-1 * x))

def show_Sigmoid():
    x = torch.tensor(np.linspace(-10, 10, 1000))
    y = 1 / (1 + torch.exp(-1 * x))
    plt.figure()
    plt.plot(x, y, 'r.', markersize=2)
    plt.xticks((-10, 10))
    plt.title('sigmoid')
    plt.show()

def Tanh_(x,device = 'cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x, device=device)
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def show_Tanh():
    x = torch.tensor(np.linspace(-10, 10, 1000))
    y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    plt.figure()
    plt.plot(x, y, 'r.', markersize=3)
    plt.xticks((-10, 10))
    plt.title('Tanh')
    plt.show()

def Relu_(x,device = 'cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x,device = device)
    a = torch.zeros_like(x)
    return torch.max(x,a)

def LeakyRelu_(x,alpha = 0.01,device = 'cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x,device = device)
    if torch.is_tensor(alpha) == False:
        alpha = torch.tensor(alpha, device=device)
    a = torch.ones_like(x) * alpha * (-1)

    return torch.max(x,a)

def show_Relu():
    x = np.linspace(-10, 10, 1000)
    y = Relu_(x)
    plt.figure()
    plt.plot(x, y, 'r.', markersize=2)
    plt.xticks((-10, 10))
    plt.title('Relu')
    plt.show()

def show_LeakyRelu():
    x = np.linspace(-10, 10, 1000)
    y = LeakyRelu_(x,1)
    plt.figure()
    plt.plot(x, y, 'r.', markersize=2)
    plt.xticks((-10, 10))
    plt.title('LeakyRelu')
    plt.show()

def Softmax_(x,device = 'cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x,device = device)
    return torch.exp(x) / torch.sum(torch.exp(x),axis = 1,keepdim=True)